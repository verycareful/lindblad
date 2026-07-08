// R.1.13.1 test patch — density-matrix simulator rework.
// Covers audit F-2 (apply_kraus out-of-place, no per-Kraus restore copy),
// F-13 (one DM buffer reused across shots), F-14 (folded-phase / popcount
// expectation_value_sparse), and the structured-op DM methods apply_permutation
// / apply_mcp_phase (F-7/F-9). Each is checked against an independent
// full-matrix or statevector reference.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"

#include <cmath>
#include <vector>

using namespace lindblad;

namespace {

std::vector<Complex128> prep_amps(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    return sim.run(qc, 0, 0).final_state.amplitudes();
}

// rho = |psi><psi| as a flat dim*dim row-major matrix.
std::vector<Complex128> outer(const std::vector<Complex128>& amp) {
    const size_t d = amp.size();
    std::vector<Complex128> r(d * d);
    for (size_t a = 0; a < d; ++a)
        for (size_t b = 0; b < d; ++b)
            r[a * d + b] = amp[a] * amp[b].conj();
    return r;
}

std::vector<Complex128> matmul(const std::vector<Complex128>& A,
                               const std::vector<Complex128>& B, size_t d) {
    std::vector<Complex128> C(d * d, Complex128(0, 0));
    for (size_t i = 0; i < d; ++i)
        for (size_t k = 0; k < d; ++k) {
            const Complex128 aik = A[i * d + k];
            for (size_t j = 0; j < d; ++j)
                C[i * d + j] += aik * B[k * d + j];
        }
    return C;
}

std::vector<Complex128> dagger(const std::vector<Complex128>& M, size_t d) {
    std::vector<Complex128> R(d * d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j)
            R[j * d + i] = M[i * d + j].conj();
    return R;
}

// Embed a single-qubit 2x2 op (row-major u[row*2+col]) on qubit q into an
// n-qubit register (LSB convention: qubit q is bit q of the basis index).
std::vector<Complex128> embed1(const std::vector<Complex128>& u, int q, int n) {
    const size_t dim = size_t(1) << n;
    const size_t qbit = size_t(1) << q;
    std::vector<Complex128> M(dim * dim, Complex128(0, 0));
    for (size_t a = 0; a < dim; ++a)
        for (size_t b = 0; b < dim; ++b) {
            if ((a & ~qbit) != (b & ~qbit)) continue;  // all other bits equal
            const int ai = (a & qbit) ? 1 : 0;
            const int bi = (b & qbit) ? 1 : 0;
            M[a * dim + b] = u[ai * 2 + bi];
        }
    return M;
}

void expect_mat_close(const std::vector<Complex128>& got,
                      const std::vector<Complex128>& want, double tol = 1e-9) {
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i].real, want[i].real, tol) << "re @ " << i;
        EXPECT_NEAR(got[i].imag, want[i].imag, tol) << "im @ " << i;
    }
}

} // namespace

// F-2: apply_kraus (out-of-place) must equal the textbook sum_k K_k rho K_k†,
// with the single-qubit channel correctly embedded on qubit q of a 2-qubit DM.
TEST(R1131Dm, ApplyKrausMatchesEmbeddedReference) {
    QuantumCircuit qc(2);
    qc.h(0).ry(0.6, 1).cx(0, 1).t(0);
    const auto amp = prep_amps(qc);
    const size_t dim = amp.size();
    const auto rho0 = outer(amp);

    for (const KrausChannel& ch :
         {NoiseChannels::amplitude_damping(0.3),
          NoiseChannels::depolarizing(0.2, 1),
          NoiseChannels::phase_damping(0.15)}) {
        DensityMatrix dm = DensityMatrix::from_statevector(
            [&] { Statevector sv(2); sv.set_amplitudes(amp); return sv; }());
        dm.apply_kraus(ch.operators, {0});

        // Reference: sum_k (K_k⊗I) rho (K_k⊗I)†  on qubit 0.
        std::vector<Complex128> ref(dim * dim, Complex128(0, 0));
        for (const auto& K : ch.operators) {
            const auto Kf = embed1(K, 0, 2);
            auto tmp = matmul(Kf, rho0, dim);
            tmp = matmul(tmp, dagger(Kf, dim), dim);
            for (size_t i = 0; i < ref.size(); ++i) ref[i] += tmp[i];
        }

        std::vector<Complex128> got(dm.data.begin(), dm.data.end());
        expect_mat_close(got, ref);
    }
}

// F-13: reusing one DM buffer across shots must not let state bleed between
// shots. This fully-deterministic feedforward circuit must give the SAME key on
// every shot; buffer corruption would drift later shots.
TEST(R1131Dm, PerShotBufferReuseIsClean) {
    QuantumCircuit qc(2, 2);
    qc.x(0);                                              // q0 = 1
    qc.measure(0, 0);                                     // c0 = 1 (always)
    qc.add_if(0, 1, Instruction::GateType::X, {1});       // feedforward: q1 = 1
    qc.measure(1, 1);                                     // c1 = 1 (always)

    DensityMatrixSimulator sim;
    NoiseModel ideal;
    auto res = sim.run(qc, ideal, /*shots=*/500, /*seed=*/123);

    int total = 0;
    for (const auto& [k, v] : res.counts) {
        EXPECT_EQ(k, "11") << "unexpected outcome from a deterministic circuit";
        total += v;
    }
    EXPECT_EQ(total, 500);
}

// F-14: folded-phase / popcount expectation must match the statevector Pauli
// expectation (independent code path), including Y terms (the i^#Y fold).
TEST(R1131Dm, ExpectationValueSparseMatchesStatevector) {
    QuantumCircuit qc(3);
    qc.h(0).ry(0.7, 1).rx(0.5, 2).cx(0, 1).t(2).cx(1, 2);
    const auto amp = prep_amps(qc);
    Statevector sv(3);
    sv.set_amplitudes(amp);

    auto H = SparsePauliOp::from_list({
        {"XYZ", Complex128(0.5, 0.0)},
        {"YIY", Complex128(0.3, 0.0)},
        {"ZZI", Complex128(0.2, 0.0)},
        {"IZX", Complex128(-0.4, 0.0)},
        {"YYY", Complex128(0.15, 0.0)},
    });

    const double ref = H.expectation_value(sv);
    DensityMatrix rho = DensityMatrix::from_statevector(sv);
    const double got = rho.expectation_value_sparse(H);
    EXPECT_NEAR(got, ref, 1e-9);
}

// F-7/F-9: DensityMatrix::apply_permutation (row/col relabel).
TEST(R1131Dm, ApplyPermutationRelabelsRowsAndCols) {
    QuantumCircuit qc(2);
    qc.h(0).ry(0.9, 1).cx(0, 1);
    const auto amp = prep_amps(qc);
    const size_t dim = amp.size();
    const auto rho0 = outer(amp);

    Statevector sv(2);
    sv.set_amplitudes(amp);
    DensityMatrix dm = DensityMatrix::from_statevector(sv);

    const std::vector<int> full_perm{2, 0, 3, 1};   // bijection of [0,4)
    dm.apply_permutation(full_perm);

    // Reference: new(perm[a], perm[b]) = old(a, b).
    std::vector<Complex128> ref(dim * dim, Complex128(0, 0));
    for (size_t a = 0; a < dim; ++a)
        for (size_t b = 0; b < dim; ++b)
            ref[static_cast<size_t>(full_perm[a]) * dim +
                static_cast<size_t>(full_perm[b])] = rho0[a * dim + b];

    std::vector<Complex128> got(dm.data.begin(), dm.data.end());
    expect_mat_close(got, ref);
}

// F-7/F-9: DensityMatrix::apply_mcp_phase (diagonal phase conjugation).
TEST(R1131Dm, ApplyMcpPhaseConjugates) {
    QuantumCircuit qc(2);
    qc.h(0).h(1).t(0);
    const auto amp = prep_amps(qc);
    const size_t dim = amp.size();
    const auto rho0 = outer(amp);

    Statevector sv(2);
    sv.set_amplitudes(amp);
    DensityMatrix dm = DensityMatrix::from_statevector(sv);

    const size_t mask = 0b11;      // both qubits = 1
    const double lambda = 0.8;
    dm.apply_mcp_phase(mask, lambda);

    // Reference: d_a = exp(i*lambda) iff (a&mask)==mask; new_{ab} = d_a rho_{ab} conj(d_b).
    std::vector<Complex128> ref(dim * dim);
    auto d = [&](size_t a) {
        return ((a & mask) == mask) ? Complex128::exp_i(lambda) : Complex128(1, 0);
    };
    for (size_t a = 0; a < dim; ++a)
        for (size_t b = 0; b < dim; ++b)
            ref[a * dim + b] = d(a) * rho0[a * dim + b] * d(b).conj();

    std::vector<Complex128> got(dm.data.begin(), dm.data.end());
    expect_mat_close(got, ref);
}
