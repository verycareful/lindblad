// R.1.13.1 test patch — shared low-level kernels.
// Covers audit F-11 (single shared sv_collapse_qubit for MEASURE and RESET),
// F-20 (apply_rccx single 3-level-stride pass), and F-8 (apply_unitary
// work-shape dispatch: par-rows for few large blocks, par-groups otherwise).

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/noise.hpp"

#include <cmath>
#include <vector>

using namespace lindblad;

namespace {

std::vector<Complex128> run_sv(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    return sim.run(qc, 0, 0).final_state.amplitudes();
}

// A deterministic, all-distinct, non-symmetric amplitude vector for nq qubits.
// Not normalised: these kernels are exact amplitude permutations and phase
// multiplies, so normalisation is irrelevant and distinct values expose any
// index or stride mistake. Handing one over therefore opts out of the
// normalization check, the same reason the apply_unitary calls below do.
std::vector<Complex128> distinct_state(int nq) {
    const size_t dim = size_t(1) << nq;
    std::vector<Complex128> v(dim);
    for (size_t k = 0; k < dim; ++k)
        v[k] = Complex128(0.3 + 0.1 * static_cast<double>(k),
                          -0.2 + 0.07 * static_cast<double>(k));
    return v;
}

size_t scatter(size_t sub, const std::vector<int>& targets) {
    size_t r = 0;
    for (size_t i = 0; i < targets.size(); ++i)
        if (sub & (size_t(1) << i)) r |= (size_t(1) << targets[i]);
    return r;
}

// Independent reference for apply_unitary: out[idx] = sum_c U[r,c] in[rest|c],
// where r/c index the target subspace (LSB = targets[0]).
std::vector<Complex128> ref_unitary(const std::vector<Complex128>& in,
                                    const std::vector<int>& targets,
                                    const std::vector<Complex128>& U) {
    const int k = static_cast<int>(targets.size());
    const size_t sd = size_t(1) << k;
    size_t tmask = 0;
    for (int t : targets) tmask |= (size_t(1) << t);
    std::vector<Complex128> out(in.size(), Complex128(0, 0));
    for (size_t idx = 0; idx < in.size(); ++idx) {
        const size_t rest = idx & ~tmask;
        size_t r = 0;
        for (int i = 0; i < k; ++i)
            if (idx & (size_t(1) << targets[i])) r |= (size_t(1) << i);
        Complex128 sum(0, 0);
        for (size_t c = 0; c < sd; ++c)
            sum += U[r * sd + c] * in[rest | scatter(c, targets)];
        out[idx] = sum;
    }
    return out;
}

// A dense k-qubit matrix with all-distinct entries (need not be unitary: the
// reference applies the identical linear map, so any index/stride bug shows).
std::vector<Complex128> distinct_matrix(int k) {
    const size_t sd = size_t(1) << k;
    std::vector<Complex128> U(sd * sd);
    for (size_t r = 0; r < sd; ++r)
        for (size_t c = 0; c < sd; ++c)
            U[r * sd + c] = Complex128(1.0 + static_cast<double>(r * sd + c),
                                       0.5 - 0.1 * static_cast<double>(c));
    return U;
}

void expect_amps_close(const std::vector<Complex128>& a,
                       const std::vector<Complex128>& b, double tol = 1e-9) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, tol) << "re @ " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, tol) << "im @ " << i;
    }
}

} // namespace

// F-20: the single-pass RCCX kernel (statevector) must match the DM's
// independent dense RCCX matrix on a non-symmetric input.
TEST(R1131Kernels, RccxSinglePassMatchesDenseMatrix) {
    QuantumCircuit qc(3);
    qc.h(0).h(1).ry(0.5, 2).t(0).rccx(0, 1, 2);

    const auto sv = run_sv(qc);

    DensityMatrixSimulator dm_sim;
    NoiseModel ideal;
    auto dm_res = dm_sim.run(qc, ideal, 0, 0);   // keep Result alive
    const DensityMatrix& rho = dm_res.final_state;
    for (size_t i = 0; i < sv.size(); ++i)
        for (size_t j = 0; j < sv.size(); ++j) {
            const Complex128 e = sv[i] * sv[j].conj();
            EXPECT_NEAR(rho(i, j).real, e.real, 1e-9);
            EXPECT_NEAR(rho(i, j).imag, e.imag, 1e-9);
        }
}

// F-11: RESET collapses a deterministic |1> to |0>.
TEST(R1131Kernels, ResetDeterministicOneToZero) {
    QuantumCircuit qc(2);
    qc.x(0).reset(0);
    const auto amp = run_sv(qc);
    EXPECT_NEAR(amp[0].real, 1.0, 1e-9);   // |00>
    EXPECT_NEAR(amp[0].imag, 0.0, 1e-9);
    for (size_t k = 1; k < amp.size(); ++k) {
        EXPECT_NEAR(amp[k].real, 0.0, 1e-9);
        EXPECT_NEAR(amp[k].imag, 0.0, 1e-9);
    }
}

// F-11: RESET on an entangled qubit zeros that qubit (no amplitude with the bit
// set) and preserves the norm, for any collapse branch.
TEST(R1131Kernels, ResetZerosBitAndNormalises) {
    for (uint64_t seed : {1ULL, 7ULL, 42ULL, 100ULL}) {
        QuantumCircuit qc(2);
        qc.h(0).cx(0, 1).reset(0);
        StatevectorSimulator sim;
        auto res = sim.run(qc, 0, seed);
        const auto amp = res.final_state.amplitudes();
        double norm = 0.0;
        for (size_t k = 0; k < amp.size(); ++k) {
            if (k & 1u) {                       // qubit 0 set -> must be zero
                EXPECT_NEAR(amp[k].real, 0.0, 1e-9);
                EXPECT_NEAR(amp[k].imag, 0.0, 1e-9);
            }
            norm += amp[k].norm_sq();
        }
        EXPECT_NEAR(norm, 1.0, 1e-9);
    }
}

// F-11: the MEASURE collapse path samples a fair coin for H|0>.
TEST(R1131Kernels, MeasureSamplesFairCoin) {
    QuantumCircuit qc(1, 1);
    qc.h(0).measure(0, 0);
    StatevectorSimulator sim;
    auto res = sim.run(qc, /*shots=*/4000, /*seed=*/99);
    int total = res.counts["0"] + res.counts["1"];
    EXPECT_EQ(total, 4000);
    EXPECT_GT(res.counts["0"], 1600);   // ~2000 each
    EXPECT_GT(res.counts["1"], 1600);
}

// F-8: apply_unitary with MANY groups (small k, large register) -> par-groups.
TEST(R1131Kernels, ApplyUnitaryManyGroups) {
    const int nq = 5;
    const std::vector<int> targets{2};       // k=1 -> 2^4 groups
    const auto U = distinct_matrix(1);
    const auto in = distinct_state(nq);

    Statevector sv(nq);
    sv.set_amplitudes(in, {Validation::Ignore});
    // distinct_matrix is deliberately not unitary, so the unitarity check is
    // opted out of here. What this test measures is index and stride
    // arithmetic, which a real unitary's symmetries would help conceal.
    gates::apply_unitary(sv, targets, U, {Validation::Ignore});
    expect_amps_close(sv.amplitudes(), ref_unitary(in, targets, U));
}

// F-8: apply_unitary with FEW groups (large k) -> par-rows branch.
TEST(R1131Kernels, ApplyUnitaryFewGroupsLargeBlock) {
    const int nq = 4;
    const std::vector<int> targets{0, 1, 3};  // k=3 -> only 2 groups, block=8
    const auto U = distinct_matrix(3);
    const auto in = distinct_state(nq);

    Statevector sv(nq);
    sv.set_amplitudes(in, {Validation::Ignore});
    // Not unitary by construction; see the note in ApplyUnitaryManyGroups.
    gates::apply_unitary(sv, targets, U, {Validation::Ignore});
    expect_amps_close(sv.amplitudes(), ref_unitary(in, targets, U));
}
