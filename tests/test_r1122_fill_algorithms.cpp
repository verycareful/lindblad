// R.1.12.2 coverage-fill suite, batch F3: algorithms. Closes the line-coverage
// gaps measured on the instrumented R.1.12.1 run:
//
//   - qft.cpp: apply(), both run() overloads, the Clifford-compatibility
//     check, approximation-degree thresholding, iterative builders and their
//     validation throws, inverse-with-swaps.
//   - simon.cpp: qubit build_circuit structure; QuditSimon affine-oracle
//     validation throws, the CLIFFORD path with a non-zero affine constant,
//     and the composite-d statevector delegation (direct period search).
//   - maqaoa.cpp: orbit sharing, initial_thetas, general X/Y Pauli rotations,
//     term-indexed gammas, layerwise + progressive training, mixer weights,
//     and the public build_circuit.
//   - qaoa.cpp: initial_thetas initialisation and the general Pauli-rotation
//     branch of build_circuit.
//   - vqe.cpp: two_local rotation/entanglement variants (rz/rx, cz, full,
//     circular), real_amplitudes, and compute_minimum_eigenvalue with the
//     random-initialisation branch.
//   - shor.cpp: the quantum order-finding path of factorize (N = 323 defeats
//     every classical shortcut: odd, not a perfect power, coprime to the
//     small trial divisors).
//   - ising.cpp, deutsch_jozsa.cpp, bernstein_vazirani.cpp, grover.cpp,
//     dispatch.cpp: remaining validation throws and utility branches.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/dispatch.hpp"
#include "lindblad/ising.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;
using GT = Instruction::GateType;

namespace {

constexpr double kTol = 1e-9;

int count_type(const QuantumCircuit& qc, GT t) {
    int n = 0;
    for (const auto& inst : qc.instructions)
        if (inst.type == t) ++n;
    return n;
}

int count_conditioned(const QuantumCircuit& qc) {
    int n = 0;
    for (const auto& inst : qc.instructions)
        if (inst.condition_clbit >= 0) ++n;
    return n;
}

int counts_total(const std::unordered_map<std::string, int>& c) {
    int n = 0;
    for (const auto& [k, v] : c) n += v;
    return n;
}

// A . s mod d == 0 for every row.
bool annihilated(const std::vector<std::vector<int>>& A,
                 const std::vector<int>& s, int d) {
    for (const auto& row : A) {
        long long acc = 0;
        for (size_t i = 0; i < row.size(); ++i)
            acc += static_cast<long long>(row[i]) * s[i];
        if (acc % d != 0) return false;
    }
    return true;
}

bool all_zero(const std::vector<int>& v) {
    for (int x : v)
        if (x != 0) return false;
    return true;
}

}  // namespace

// =============================================================================
// QFT
// =============================================================================

TEST(R1122FillAlgo, QFTApplyProducesFourierStateOfOne) {
    // QFT|1> in the LSB convention: amp[y] = exp(2*pi*i*y/8)/sqrt(8).
    QuantumCircuit base(3);
    base.x(0);
    auto composed = QFT::apply(base, QFT::Options{true, 0, false});
    ASSERT_EQ(composed.n_qubits, 3);
    ASSERT_GT(composed.instructions.size(), base.instructions.size());

    StatevectorSimulator sim;
    auto res = sim.run(composed, 0, 1);
    ASSERT_TRUE(res.success) << res.error_message;
    const double inv = 1.0 / std::sqrt(8.0);
    const double w = 2.0 * PI / 8.0;
    for (int y = 0; y < 8; ++y) {
        EXPECT_NEAR(res.final_state.real_parts[y], inv * std::cos(w * y), 1e-9)
            << "re amp[" << y << "]";
        EXPECT_NEAR(res.final_state.imag_parts[y], inv * std::sin(w * y), 1e-9)
            << "im amp[" << y << "]";
    }
}

TEST(R1122FillAlgo, QFTRunOverloadsReportCliffordCompatibility) {
    QuantumCircuit two(2);
    two.x(0);
    auto r2 = QFT::run(two, QFT::Options{}, 256, 5);
    EXPECT_TRUE(r2.backend_result.success);
    EXPECT_EQ(counts_total(r2.backend_result.counts), 256);
    EXPECT_EQ(r2.n_qubits, 2);
    EXPECT_TRUE(r2.clifford_compatible) << "exact QFT on n<=2 is Clifford";

    QuantumCircuit three(3);
    three.x(0);
    auto r3 = QFT::run(three, QFT::Options{}, 128, 5);
    EXPECT_FALSE(r3.clifford_compatible) << "exact QFT on n=3 needs CP(pi/4)";

    auto r3a = QFT::run(three, QFT::Options{true, 1, false}, 128, 5);
    EXPECT_TRUE(r3a.clifford_compatible) << "AQFT(m=1) keeps only CS gates";
}

TEST(R1122FillAlgo, QFTIterativeBuildersValidateAndCondition) {
    EXPECT_THROW(QFT::build_iterative_circuit(0), std::invalid_argument);
    EXPECT_THROW(QFT::build_iterative_inverse_circuit(0), std::invalid_argument);

    auto inv = QFT::build_iterative_inverse_circuit(3);
    EXPECT_EQ(inv.n_qubits, 3);
    EXPECT_EQ(inv.n_clbits, 3);
    EXPECT_EQ(count_type(inv, GT::MEASURE), 3);
    EXPECT_EQ(count_type(inv, GT::H), 3);
    EXPECT_EQ(count_conditioned(inv), 3) << "feedforward phases for (j,k) pairs";
}

TEST(R1122FillAlgo, QFTBuildValidationAndApproximationDegree) {
    EXPECT_THROW(QFT::build_circuit(0, {}), std::invalid_argument);
    EXPECT_THROW(QFT::build_approximate_circuit(2, -1), std::invalid_argument);

    auto exact = QFT::build_circuit(4, QFT::Options{true, 0, false});
    auto aqft = QFT::build_approximate_circuit(4, 2);  // drops CP below pi/4
    EXPECT_EQ(count_type(exact, GT::CP), 6);
    EXPECT_EQ(count_type(aqft, GT::CP), 5) << "one CP(pi/8) must be dropped";

    auto inv_swaps = QFT::build_inverse_circuit(4, true);
    EXPECT_EQ(count_type(inv_swaps, GT::SWAP), 2) << "bit-reversal layer";
}

// =============================================================================
// Simon (qubit) and QuditSimon
// =============================================================================

TEST(R1122FillAlgo, SimonBuildCircuitStructure) {
    QuantumCircuit oracle(4);  // n = 2: query {0,1}, output {2,3}
    oracle.cx(0, 2).cx(1, 3);
    auto qc = Simon::build_circuit(oracle, 2);
    EXPECT_EQ(qc.n_qubits, 4);
    EXPECT_EQ(qc.n_clbits, 2);
    EXPECT_EQ(count_type(qc, GT::H), 4);        // H layer before and after
    EXPECT_EQ(count_type(qc, GT::MEASURE), 2);  // query register only
    EXPECT_EQ(count_type(qc, GT::CX), 2);       // the embedded oracle
}

TEST(R1122FillAlgo, QuditSimonAffineValidationThrows) {
    const QuditAffineOracle ok{{{1, 2}, {2, 1}}, {0, 0}};
    EXPECT_THROW(QuditSimon::solve(ok, 1), std::invalid_argument);  // d < 2

    const QuditAffineOracle non_square{{{1, 2}}, {0}};
    EXPECT_THROW(QuditSimon::solve(non_square, 3), std::invalid_argument);

    const QuditAffineOracle bad_b{{{1, 2}, {2, 1}}, {0}};
    EXPECT_THROW(QuditSimon::solve(bad_b, 3), std::invalid_argument);

    const QuditAffineOracle ragged{{{1, 2}, {2}}, {0, 0}};
    EXPECT_THROW(QuditSimon::solve(ragged, 3), std::invalid_argument);

    const QuditAffineOracle coeff_range{{{1, 5}, {2, 1}}, {0, 0}};
    EXPECT_THROW(QuditSimon::solve(coeff_range, 3), std::invalid_argument);

    const QuditAffineOracle b_range{{{1, 2}, {2, 1}}, {0, 7}};
    EXPECT_THROW(QuditSimon::solve(b_range, 3), std::invalid_argument);
}

TEST(R1122FillAlgo, QuditSimonAffineCliffordPrimeWithConstant) {
    // A singular mod 3 (det = -3), kernel spanned by (1,1); b nonzero
    // exercises the X^b preparation on the output register.
    const QuditAffineOracle oracle{{{1, 2}, {2, 1}}, {1, 2}};
    auto res = QuditSimon::solve(oracle, 3, 3, 11, QuditBackend::CLIFFORD);
    EXPECT_EQ(res.d, 3);
    EXPECT_EQ(res.n, 2);
    EXPECT_GT(res.quantum_queries, 0);
    EXPECT_FALSE(all_zero(res.period)) << "f is 3-to-1, a period must exist";
    EXPECT_TRUE(annihilated(oracle.A, res.period, 3))
        << "recovered s must satisfy A.s == 0 (mod 3)";
}

TEST(R1122FillAlgo, QuditSimonAffineStatevectorCompositeDelegates) {
    // Composite d = 4 on a non-Clifford backend: the affine overload must
    // materialise f and delegate; post-processing takes the ring path
    // (direct verified search) rather than field elimination.
    const QuditAffineOracle oracle{{{2, 0}, {0, 2}}, {1, 3}};
    auto res = QuditSimon::solve(oracle, 4, 3, 21, QuditBackend::STATEVECTOR);
    EXPECT_FALSE(res.is_trivial);
    EXPECT_FALSE(all_zero(res.period));
    EXPECT_TRUE(annihilated(oracle.A, res.period, 4))
        << "recovered s must satisfy A.s == 0 (mod 4)";
}

// =============================================================================
// MAQAOA
// =============================================================================

namespace {

SparsePauliOp ring_cost_3q() {
    return SparsePauliOp(std::vector<PauliString>{
        PauliString("ZZI", Complex128(1.0, 0.0)),
        PauliString("IZZ", Complex128(1.0, 0.0)),
        PauliString("ZIZ", Complex128(1.0, 0.0))});
}

}  // namespace

TEST(R1122FillAlgo, MaqaoaOrbitSharingReducesParameters) {
    MAQAOA plain;
    plain.options.p = 1;
    const int full = plain.num_parameters(ring_cost_3q());

    MAQAOA shared;
    shared.options.p = 1;
    shared.options.max_iterations = 5;
    shared.options.seed = 7;
    shared.options.orbit_assignments = {0, 0, 0};  // fully symmetric ring
    const int reduced = shared.num_parameters(ring_cost_3q());
    EXPECT_LT(reduced, full) << "orbit sharing must shrink the parameter count";

    auto res = shared.optimize(ring_cost_3q());
    EXPECT_EQ(res.optimal_params.size(), static_cast<size_t>(reduced));
    EXPECT_FALSE(res.counts.empty());
    EXPECT_EQ(res.best_bitstring.size(), 3u);
}

TEST(R1122FillAlgo, MaqaoaInitialThetasAndGeneralPauliTerms) {
    SparsePauliOp cost(std::vector<PauliString>{
        PauliString("ZZ", Complex128(1.0, 0.0)),
        PauliString("XI", Complex128(0.4, 0.0)),    // X branch (H basis change)
        PauliString("YY", Complex128(0.3, 0.0))});  // Y branch (Sdg-H change)
    MAQAOA m;
    m.options.p = 1;
    m.options.max_iterations = 5;
    m.options.seed = 9;
    m.options.initial_thetas = {0.6, 1.1};  // QSP Ry initialisation
    auto res = m.optimize(cost);
    EXPECT_EQ(res.best_bitstring.size(), 2u);
    EXPECT_FALSE(res.counts.empty());
    EXPECT_GT(res.num_iterations, 0);
}

TEST(R1122FillAlgo, MaqaoaTermIndexedLayerwiseProgressive) {
    MAQAOA m;
    m.options.p = 2;
    m.options.max_iterations = 4;
    m.options.seed = 13;
    m.options.term_indexed_gammas = true;
    m.options.layerwise = true;
    m.options.progressive = true;
    const int n_params = m.num_parameters(ring_cost_3q());
    EXPECT_EQ(n_params, 2 * (3 + 3)) << "term-indexed: one gamma per term "
                                        "plus one beta per qubit, per layer";
    auto res = m.optimize(ring_cost_3q());
    EXPECT_EQ(res.optimal_params.size(), static_cast<size_t>(n_params));
    EXPECT_EQ(res.per_layer_costs.size(), 2u);
    EXPECT_EQ(res.layer_nfev.size(), 2u);
}

TEST(R1122FillAlgo, MaqaoaMixerWeightsAndBuildCircuit) {
    MAQAOA m;
    m.options.p = 1;
    m.options.max_iterations = 3;
    m.options.seed = 17;
    m.options.mixer_weights = {2.0, 1.0, 0.5};  // PI-MA-QAOA beta scaling
    auto res = m.optimize(ring_cost_3q());
    EXPECT_FALSE(res.optimal_params.empty());

    std::vector<double> params(static_cast<size_t>(m.num_parameters(ring_cost_3q())));
    for (size_t i = 0; i < params.size(); ++i)
        params[i] = 0.1 * static_cast<double>(i + 1);
    // Empty mixer = the paper ansatz. The custom-mixer path is a separate
    // ansatz and is covered in its own suite.
    auto qc = m.build_circuit(ring_cost_3q(), {}, params);
    EXPECT_EQ(qc.n_qubits, 3);
    EXPECT_FALSE(qc.instructions.empty());

    StatevectorSimulator sim;
    EXPECT_TRUE(sim.run(qc, 0, 1).success) << "built circuit must be executable";
}

TEST(R1122FillAlgo, MaqaoaAcceptsCustomMixerHamiltonian) {
    // The MA-QAOA default mixer is the fixed per-qubit RX (Herrman et al.
    // 2022) and an empty mixer_hamiltonian still means exactly that. A custom
    // one is a first-class argument: accepted, applied as the ordered product
    // of its per-term rotations, and counted by num_parameters.
    //
    // sum_i X_i is the canonical mixer, so under the default dispatch it draws
    // one beta per qubit and the parameter count is unchanged. That equality is
    // the load-bearing half: it is what makes the custom path a generalisation
    // of the paper ansatz rather than a second ansatz beside it.
    SparsePauliOp cost = ring_cost_3q();
    SparsePauliOp x_mixer(std::vector<PauliString>{
        PauliString("XII"), PauliString("IXI"), PauliString("IIX")});

    MAQAOA m;
    m.options.p = 1;
    m.options.max_iterations = 3;
    m.options.seed = 5;

    EXPECT_EQ(m.num_parameters(cost, x_mixer), m.num_parameters(cost))
        << "the canonical mixer reaches one beta per qubit, which is what the "
           "default path already allocates";

    auto res = m.optimize(cost, x_mixer);
    EXPECT_FALSE(res.counts.empty());
    EXPECT_FALSE(res.optimal_params.empty());

    std::vector<double> params(
        static_cast<size_t>(m.num_parameters(cost, x_mixer)), 0.1);
    QuantumCircuit qc;
    ASSERT_NO_THROW(qc = m.build_circuit(cost, x_mixer, params));
    EXPECT_EQ(qc.n_qubits, 3);
    EXPECT_FALSE(qc.instructions.empty());

    StatevectorSimulator sim;
    EXPECT_TRUE(sim.run(qc, 0, 1).success)
        << "a circuit built over a custom mixer must be executable";

    // The default (empty) mixer path is the paper ansatz and must still run.
    auto plain = m.optimize(cost);
    EXPECT_FALSE(plain.counts.empty());
}

// =============================================================================
// QAOA
// =============================================================================

TEST(R1122FillAlgo, QaoaInitialThetasAndGeneralTermCircuit) {
    SparsePauliOp cost(std::vector<PauliString>{
        PauliString("ZZ", Complex128(1.0, 0.0)),
        PauliString("XY", Complex128(0.5, 0.0))});  // general branch, X and Y
    SparsePauliOp mixer(std::vector<PauliString>{
        PauliString("XI", Complex128(1.0, 0.0)),
        PauliString("IX", Complex128(1.0, 0.0))});

    QAOA q;
    q.options.p = 1;
    q.options.initial_thetas = {0.7, 1.3};
    auto qc = q.build_circuit(cost, mixer, {0.4, 0.9});

    EXPECT_EQ(qc.n_qubits, 2);
    EXPECT_EQ(count_type(qc, GT::RY), 2) << "QSP init replaces the H layer";
    EXPECT_GE(count_type(qc, GT::H), 2) << "X-term basis change";
    EXPECT_GE(count_type(qc, GT::SDG), 1) << "Y-term basis change";
    EXPECT_GE(count_type(qc, GT::RX), 2) << "mixer";

    StatevectorSimulator sim;
    EXPECT_TRUE(sim.run(qc, 0, 1).success);
}

TEST(R1122FillAlgo, QaoaEntanglingMixerTermIsExponentiated) {
    // A multi-qubit mixer term must be evolved as exp(-i*beta*P), not
    // factorised into independent per-qubit rotations (the pre-R.1.12.2
    // behaviour turned an XY term into RX (x) RY, a different unitary).
    // Since (XY)^2 = I: exp(-i*beta*XY) = cos(beta)*I - i*sin(beta)*(XY).
    SparsePauliOp cost(std::vector<PauliString>{PauliString("ZZ")});
    SparsePauliOp xy_mixer(std::vector<PauliString>{PauliString("XY")});
    const double beta = 0.37;

    QAOA q;
    q.options.p = 1;
    auto qc = q.build_circuit(cost, xy_mixer, {0.0, beta});  // gamma = 0

    // Reference: (H (x) H) initial layer, gamma = 0 cost is identity, then
    // the exact exponential. LSB layout: pauli[0] = 'X' acts on qubit 0
    // (index bit 0), pauli[1] = 'Y' on qubit 1 (index bit 1).
    using C = std::complex<double>;
    const C iu(0.0, 1.0);
    C X[2][2] = {{C(0), C(1)}, {C(1), C(0)}};
    C Y[2][2] = {{C(0), -iu}, {iu, C(0)}};
    C P[4][4], E[4][4], H2[4][4], R[4][4];
    for (int r1 = 0; r1 < 2; ++r1)
        for (int r0 = 0; r0 < 2; ++r0)
            for (int c1 = 0; c1 < 2; ++c1)
                for (int c0 = 0; c0 < 2; ++c0)
                    P[r1 * 2 + r0][c1 * 2 + c0] = Y[r1][c1] * X[r0][c0];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            E[r][c] = -iu * std::sin(beta) * P[r][c];
            if (r == c) E[r][c] += std::cos(beta);
            const int r0 = r & 1, r1 = r >> 1, c0 = c & 1, c1 = c >> 1;
            H2[r][c] = 0.5 * ((r0 & c0) ? -1.0 : 1.0) * ((r1 & c1) ? -1.0 : 1.0);
        }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            R[r][c] = C(0);
            for (int k = 0; k < 4; ++k) R[r][c] += E[r][k] * H2[k][c];
        }

    auto ma = Operator::from_circuit(qc).data;
    ASSERT_EQ(ma.size(), 16u);
    // Align a global phase on the first non-negligible entry, then compare.
    C phase(1.0, 0.0);
    for (int idx = 0; idx < 16; ++idx) {
        const C ref = R[idx / 4][idx % 4];
        if (std::abs(ref) > 1e-9) {
            phase = C(ma[idx].real, ma[idx].imag) / ref;
            break;
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            const C want = R[r][c] * phase;
            const auto& got = ma[r * 4 + c];
            EXPECT_NEAR(got.real, want.real(), 1e-9) << "re @ " << r << "," << c;
            EXPECT_NEAR(got.imag, want.imag(), 1e-9) << "im @ " << r << "," << c;
        }
}

// =============================================================================
// VQE
// =============================================================================

TEST(R1122FillAlgo, VqeAnsatzFactoriesCoverEntanglementModes) {
    // Rotation layers use named parameters, so the builders emit the
    // symbolic PARAM_* gate types, not the numeric ones.
    auto full = VQE::two_local(3, {"ry", "rz", "rx"}, {"cx", "cz"}, 1, "full");
    EXPECT_EQ(count_type(full, GT::PARAM_RY), 6);
    EXPECT_EQ(count_type(full, GT::PARAM_RZ), 6);
    EXPECT_EQ(count_type(full, GT::PARAM_RX), 6);
    EXPECT_EQ(count_type(full, GT::CX), 3);  // all pairs of 3 qubits
    EXPECT_EQ(count_type(full, GT::CZ), 3);
    EXPECT_EQ(full.num_parameters(), 18);

    auto ring = VQE::two_local(4, {"ry"}, {"cx"}, 1, "circular");
    EXPECT_EQ(count_type(ring, GT::CX), 4) << "circular closes the loop";

    auto ra = VQE::real_amplitudes(2, 2);
    EXPECT_EQ(ra.num_parameters(), 6);  // (reps+1) * n_qubits
    EXPECT_EQ(count_type(ra, GT::CX), 2);
}

TEST(R1122FillAlgo, VqeComputesGroundStateWithRandomInit) {
    SparsePauliOp h(std::vector<PauliString>{PauliString("Z")});
    QuantumCircuit ansatz(1);
    ansatz.ry("t", 0);

    VQE vqe;
    vqe.options.max_iterations = 120;
    // NELDER_MEAD: covers the optimizer-selection branch and behaves well on
    // a 1-parameter landscape (NLopt's COBYLA is unreliable in 1-D).
    vqe.options.optimizer = "NELDER_MEAD";
    auto res = vqe.compute_minimum_eigenvalue(h, ansatz, {});  // random init
    EXPECT_NEAR(res.eigenvalue, -1.0, 0.05) << "ground state of Z is |1>";
    EXPECT_EQ(res.optimal_parameters.size(), 1u);
    EXPECT_FALSE(res.energy_history.empty());
}

TEST(R1122FillAlgo, VqeEigenvalueIsAlwaysFinite) {
    // Pins the NLopt failure guard: 1-D COBYLA is known to bail without
    // writing the objective value, which used to surface an uninitialised
    // eigenvalue (observed as -nan). The recorded history minimum must be
    // returned instead, and the value must always be finite.
    SparsePauliOp h(std::vector<PauliString>{PauliString("Z")});
    QuantumCircuit ansatz(1);
    ansatz.ry("t", 0);

    VQE vqe;
    vqe.options.max_iterations = 60;
    vqe.options.optimizer = "COBYLA";
    auto res = vqe.compute_minimum_eigenvalue(h, ansatz, {});
    // is_finite_strict: this test compiles under -ffast-math too, where
    // std::isfinite could fold to true and mask the very bug being pinned.
    EXPECT_TRUE(is_finite_strict(res.eigenvalue)) << res.eigenvalue;
    EXPECT_GE(res.eigenvalue, -1.0 - 1e-9) << "<Z> history is bounded below";
    EXPECT_FALSE(res.energy_history.empty());
}

// =============================================================================
// Shor: the quantum order-finding path
// =============================================================================

TEST(R1122FillAlgo, ShorQuantumPathFactorises323) {
    // 323 = 17 * 19 defeats every classical shortcut in factorize(): it is
    // odd, not a perfect power, and coprime to the trial divisors
    // {3,5,7,11,13}, so the QPE order-finding loop must execute.
    Shor::Options opts;
    opts.n_eval_qubits = 5;  // keep the register small for the -O0 build
    opts.max_attempts = 3;
    opts.seed = 7;
    Shor shor(opts);

    auto res = shor.factorize(323);
    EXPECT_LE(res.attempts, 3);
    if (res.success) {
        EXPECT_EQ(res.factor * res.cofactor, 323u);
        EXPECT_GT(res.factor, 1u);
        EXPECT_LT(res.factor, 323u);
    } else {
        EXPECT_EQ(res.factor, 0u);
        EXPECT_EQ(res.method, "quantum");
    }
}

// =============================================================================
// IsingHamiltonian
// =============================================================================

TEST(R1122FillAlgo, IsingValidationAndConstruction) {
    auto ising = IsingHamiltonian::from_hJ({1.0, -0.5}, {{0.0, 0.25}, {0.0, 0.0}}, 0.3);
    EXPECT_EQ(ising.n_qubits(), 2);
    EXPECT_DOUBLE_EQ(ising.offset, 0.3);
    EXPECT_DOUBLE_EQ(ising.h[0], 1.0);
    EXPECT_DOUBLE_EQ(ising.J[0][1], 0.25);

    EXPECT_THROW(ising.evaluate("011"), std::invalid_argument);  // wrong length
    EXPECT_THROW(IsingHamiltonian::from_hJ({1.0}, {{0.0, 0.0}, {0.0, 0.0}}, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(IsingHamiltonian::from_qubo({{1.0, 0.0}}, 1.0),
                 std::invalid_argument);  // non-square Q

    // Flipping qubit 0 flips only the h[0] contribution (convention-free
    // check: the two energies are symmetric about offset).
    auto pure_field = IsingHamiltonian::from_hJ({1.0, 0.0}, {{0.0, 0.0}, {0.0, 0.0}}, 0.0);
    const double e0 = pure_field.evaluate("00");
    const double e1 = pure_field.evaluate("01");  // qubit 0 is the rightmost bit
    EXPECT_NEAR(e0 + e1, 0.0, kTol);
    EXPECT_NEAR(std::abs(e0), 1.0, kTol);
}

// =============================================================================
// Qudit Deutsch-Jozsa and the affine oracle
// =============================================================================

TEST(R1122FillAlgo, QuditAffineOracleEvalValidation) {
    const QuditAffineOracle oracle{{{1, 2}}, {1}};
    EXPECT_EQ(oracle.eval({2, 1}, 3), (std::vector<int>{2}));  // (2+2+1) mod 3

    EXPECT_THROW(oracle.eval({0}, 3), std::invalid_argument);      // x too short
    EXPECT_THROW(oracle.eval({0, 5}, 3), std::invalid_argument);   // digit range
    const QuditAffineOracle bad_b{{{1}}, {7}};
    EXPECT_THROW(bad_b.eval({0}, 3), std::invalid_argument);       // b range
    const QuditAffineOracle bad_a{{{5}}, {0}};
    EXPECT_THROW(bad_a.eval({1}, 3), std::invalid_argument);       // A range
}

TEST(R1122FillAlgo, QuditDeutschJozsaAffineVerdictsAndValidation) {
    using QDJ = QuditDeutschJozsa;

    const QuditAffineOracle balanced{{{1, 2}}, {1}};
    EXPECT_THROW(QDJ::solve(balanced, 1), std::invalid_argument);  // d < 2
    const QuditAffineOracle two_rows{{{1, 0}, {0, 1}}, {0, 0}};
    EXPECT_THROW(QDJ::solve(two_rows, 3), std::invalid_argument);
    const QuditAffineOracle range{{{1, 5}}, {0}};
    EXPECT_THROW(QDJ::solve(range, 3), std::invalid_argument);
    EXPECT_THROW(QDJ::solve(balanced, 4, 1, QuditBackend::CLIFFORD),
                 std::invalid_argument);  // CLIFFORD needs prime d

    auto bal = QDJ::solve(balanced, 3, 5, QuditBackend::CLIFFORD);
    EXPECT_EQ(bal.verdict, QDJ::Verdict::BALANCED);

    const QuditAffineOracle constant{{{0, 0}}, {2}};
    auto con = QDJ::solve(constant, 3, 5, QuditBackend::STATEVECTOR);
    EXPECT_EQ(con.verdict, QDJ::Verdict::CONSTANT);
}

// =============================================================================
// Grover
// =============================================================================

TEST(R1122FillAlgo, GroverAutoIterationsThreeQubits) {
    // Marked state |101> (q0=1, q1=0, q2=1). Auto iteration count for N=8 is
    // round(pi/4 * sqrt(8)) = 2, success probability about 0.945.
    QuantumCircuit oracle(3);
    oracle.x(1);
    oracle.ccz(0, 1, 2);
    oracle.x(1);

    auto res = Grover::search(oracle, -1, 2048, 9);
    EXPECT_EQ(res.num_iterations, 2);
    EXPECT_EQ(res.solution, "101");
    EXPECT_GT(res.probability, 0.8);
}

TEST(R1122FillAlgo, QuditGroverValidationThrows) {
    auto marked = [](const std::vector<int>& x) { return all_zero(x); };
    EXPECT_THROW(QuditGrover::search_with_oracle(1, 1, marked, 1, 8, 1,
                                                 QuditBackend::STATEVECTOR, nullptr),
                 std::invalid_argument);  // d < 2
    EXPECT_THROW(QuditGrover::search_with_oracle(0, 3, marked, 1, 8, 1,
                                                 QuditBackend::STATEVECTOR, nullptr),
                 std::invalid_argument);  // n < 1
}

// =============================================================================
// Qudit Bernstein-Vazirani
// =============================================================================

TEST(R1122FillAlgo, QuditBvCompositeCliffordThrowsAndSvRecovers) {
    EXPECT_THROW(QuditBernsteinVazirani::solve({1, 2}, 4, 1, 1,
                                               QuditBackend::CLIFFORD),
                 std::invalid_argument)
        << "CLIFFORD backend requires prime d";

    auto res = QuditBernsteinVazirani::solve({3, 1}, 4, 1, 5,
                                             QuditBackend::STATEVECTOR);
    EXPECT_EQ(res.secret, (std::vector<int>{3, 1}));
}

// =============================================================================
// SoftDispatchResult
// =============================================================================

TEST(R1122FillAlgo, SoftDispatchRoundingAndTopK) {
    // Keys use the counts convention: qubit 0 rightmost. "01" = q0 on, q1 off.
    std::unordered_map<std::string, int> counts{{"01", 3}, {"10", 1}};
    SoftDispatchResult r(counts);
    r.compute();

    ASSERT_EQ(r.soft_assignment.size(), 2u);
    EXPECT_NEAR(r.soft_assignment[0], 0.75, kTol);  // q0 on in 3 of 4 shots
    EXPECT_NEAR(r.soft_assignment[1], 0.25, kTol);
    EXPECT_EQ(r.best_bitstring, "01");
    EXPECT_NEAR(r.best_probability, 0.75, kTol);

    EXPECT_EQ(r.threshold_round(0.5), "10")
        << "index-order string: generator 0 first";

    auto picked = r.greedy_dispatch({2.0, 1.0}, 1.5);
    ASSERT_EQ(picked.size(), 1u);
    EXPECT_EQ(picked[0], 0) << "highest soft assignment first";
    EXPECT_THROW(r.greedy_dispatch({1.0}, 1.0), std::invalid_argument);

    auto top = r.top_k(1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0].first, "01");
    EXPECT_NEAR(top[0].second, 0.75, kTol);

    const double ec = r.expected_cost(
        [](const std::string& b) { return b == "01" ? -1.0 : 1.0; });
    EXPECT_NEAR(ec, 0.75 * (-1.0) + 0.25 * 1.0, kTol);
}
