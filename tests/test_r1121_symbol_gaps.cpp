// R.1.12.1 coverage gap-closure: public symbols that the static symbol matrix
// reported as referenced by no suite. Plan DoD #1: zero unreferenced public
// symbols. Each test exercises one previously-untested symbol against a robust
// invariant (unitarity, trace/norm preservation, round-trip, range, equality).
// Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/dispatch.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/ising.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"
#include "lindblad/qudit/qudit_simulator.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

QuditStatevector entangled_qudit(int n, int d) {
    QuditStatevector sv(n, d);
    sv.apply_1qudit(0, qudit_gates::qft_matrix(d));
    for (int q = 0; q + 1 < n; ++q)
        sv.apply_2qudit(q, q + 1, qudit_gates::cadd_matrix(d, 1));
    return sv;
}

void expect_unitary_dd(const std::vector<Complex128>& M, int dim, double tol = 1e-9) {
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            Complex128 acc(0, 0);
            for (int k = 0; k < dim; ++k) acc += M[k * dim + i].conj() * M[k * dim + j];
            EXPECT_NEAR(acc.real, (i == j) ? 1.0 : 0.0, tol);
            EXPECT_NEAR(acc.imag, 0.0, tol);
        }
}

}  // namespace

// =============================================================================
// qudit_gates::controlled_power_matrix
// =============================================================================

TEST(R1121SymbolGaps, ControlledPowerMatrix) {
    const int d = 3;
    auto U = qudit_gates::shift_matrix(d, 1);  // unitary

    auto M0 = qudit_gates::controlled_power_matrix(d, U, 0);  // U^0 = I -> identity
    for (int i = 0; i < d * d; ++i)
        for (int j = 0; j < d * d; ++j)
            EXPECT_NEAR(M0[i * d * d + j].real, (i == j) ? 1.0 : 0.0, 1e-12);

    auto M1 = qudit_gates::controlled_power_matrix(d, U, 1);
    expect_unitary_dd(M1, d * d);  // controlled-power of a unitary is unitary
}

// =============================================================================
// SoftDispatchResult::greedy_dispatch / expected_cost
// =============================================================================

TEST(R1121SymbolGaps, SoftDispatchGreedyAndExpectedCost) {
    std::unordered_map<std::string, int> counts = {{"11", 50}, {"10", 30}, {"00", 20}};
    SoftDispatchResult sd(counts);
    sd.compute();
    // soft[0] (qubit 0, rightmost) set only in "11" -> 0.5; soft[1] in "11","10" -> 0.8.
    ASSERT_EQ(sd.soft_assignment.size(), 2u);

    // Generator 1 has the higher fractional commitment; capacity 20 alone meets demand.
    auto sel = sd.greedy_dispatch({10.0, 20.0}, 15.0);
    ASSERT_FALSE(sel.empty());
    EXPECT_EQ(sel[0], 1) << "highest soft-assignment generator selected first";

    double ec = sd.expected_cost(
        [](const std::string& b) { return double(std::count(b.begin(), b.end(), '1')); });
    // 0.5*2 + 0.3*1 + 0.2*0 = 1.3
    EXPECT_NEAR(ec, 1.3, 1e-9);
}

// =============================================================================
// QuditNoiseModel::add_amplitude_damping / add_phase_damping / add_lindblad_op
// =============================================================================

TEST(R1121SymbolGaps, QuditNoiseModelAdders) {
    QuditNoiseModel m;
    m.add_amplitude_damping(0, 3, 0.2);
    m.add_phase_damping(1, 3, 0.3);
    auto L = QuditNoiseModel::amplitude_damping_lindblad(3, 0.5).L;
    m.add_lindblad_op(0, L, 0.4);
    EXPECT_EQ(m.per_qudit.count(0), 1u);
    EXPECT_EQ(m.per_qudit.count(1), 1u);
    EXPECT_FALSE(m.per_qudit.at(0).lindblad.empty());

    // Kraus part is trace preserving.
    QuditDensityMatrix rho(2, 3);
    rho.apply_1qudit(0, qudit_gates::qft_matrix(3));
    rho.apply_noise(m);  // dt = 0 -> Kraus channels only
    EXPECT_NEAR(rho.trace(), 1.0, 1e-6);
}

// =============================================================================
// QuditDensityMatrix::apply_kraus_2qudit / symmetrize
// =============================================================================

TEST(R1121SymbolGaps, QuditDmKraus2QuditAndSymmetrize) {
    QuditDensityMatrix rho(2, 3);
    rho.apply_kraus_2qudit(0, 1, {qudit_gates::cadd_matrix(3, 1)});  // unitary Kraus
    EXPECT_NEAR(rho.trace(), 1.0, 1e-9);

    QuditDensityMatrix r2(1, 3);
    r2.rho[0 * 3 + 1] = Complex128(0.5, 0.3);  // break Hermiticity
    r2.symmetrize();
    EXPECT_NEAR(r2.rho[0 * 3 + 1].real, r2.rho[1 * 3 + 0].real, 1e-12);
    EXPECT_NEAR(r2.rho[0 * 3 + 1].imag, -r2.rho[1 * 3 + 0].imag, 1e-12);
}

// =============================================================================
// MPSState::apply_two_qubit_gate (direct) — symmetric CZ avoids operand-order doubt
// =============================================================================

TEST(R1121SymbolGaps, MpsStateApplyTwoQubitGate) {
    const std::array<Complex128, 4> H = {Complex128(INV_SQRT2, 0), Complex128(INV_SQRT2, 0),
                                         Complex128(INV_SQRT2, 0), Complex128(-INV_SQRT2, 0)};
    std::array<Complex128, 16> CZ{};
    for (int i = 0; i < 4; ++i) CZ[i * 4 + i] = Complex128(1, 0);
    CZ[3 * 4 + 3] = Complex128(-1, 0);

    MPSState mps(2);
    mps.apply_single_qubit_gate(H, 0);
    mps.apply_single_qubit_gate(H, 1);
    mps.apply_two_qubit_gate(CZ, 0, 1);
    auto got = mps.to_statevector();

    Statevector ref(2);
    gates::apply_h(ref, 0);
    gates::apply_h(ref, 1);
    gates::apply_cz(ref, 0, 1);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(got.amplitude(i).real, ref.amplitude(i).real, 1e-7) << i;
        EXPECT_NEAR(got.amplitude(i).imag, ref.amplitude(i).imag, 1e-7) << i;
    }
}

// =============================================================================
// QuditCliffordSimulator::apply_Z / measure_qudit / row_multiply
// =============================================================================

TEST(R1121SymbolGaps, QuditCliffordApplyZMeasureQuditRowMultiply) {
    QuditCliffordSimulator s(1, 3);
    s.apply_Z(0, 1);                       // Z on |0> is a no-op in the Z basis
    EXPECT_EQ(s.measure_qudit(0, 1), 0);   // deterministic 0

    QuditCliffordSimulator s2(2, 3);
    s2.row_multiply(0, 2);                 // exercise the Heisenberg product
    auto m = s2.measure(4);
    for (int v : m) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 3);
    }
}

// =============================================================================
// QuditMPS::left_canonicalize / right_canonicalize
// =============================================================================

TEST(R1121SymbolGaps, QuditMpsCanonicalizePreservesState) {
    auto sv = entangled_qudit(3, 3);
    QuditMPS mps(sv);
    mps.left_canonicalize();
    EXPECT_NEAR(mps.norm_sq(), 1.0, 1e-7);
    mps.right_canonicalize();
    auto out = mps.to_statevector();
    for (size_t i = 0; i < sv.dim; ++i) {
        // canonicalisation is gauge-only: probabilities are invariant.
        double pa = sv.amplitudes[i].real * sv.amplitudes[i].real +
                    sv.amplitudes[i].imag * sv.amplitudes[i].imag;
        double pb = out.amplitudes[i].real * out.amplitudes[i].real +
                    out.amplitudes[i].imag * out.amplitudes[i].imag;
        EXPECT_NEAR(pa, pb, 1e-6) << i;
    }
}

// =============================================================================
// MPSSiteTensor (qudit) — construct / at / reshape round-trip
// =============================================================================

TEST(R1121SymbolGaps, MpsSiteTensorReshapeRoundTrip) {
    MPSSiteTensor t(3, 2, 4);  // d=3, chiL=2, chiR=4
    t.at(1, 0, 2) = Complex128(0.5, -0.3);
    EXPECT_NEAR(t.at(1, 0, 2).real, 0.5, 1e-12);

    auto M = t.as_left_matrix();  // (d*chiL, chiR) = (6, 4)
    EXPECT_EQ(M.rows(), 6);
    EXPECT_EQ(M.cols(), 4);
    auto t2 = MPSSiteTensor::from_left_matrix(M, 3, 2);
    EXPECT_NEAR(t2.at(1, 0, 2).real, 0.5, 1e-12);
    EXPECT_NEAR(t2.at(1, 0, 2).imag, -0.3, 1e-12);

    auto R = t.as_right_matrix();  // (chiL, d*chiR) = (2, 12)
    EXPECT_EQ(R.rows(), 2);
    EXPECT_EQ(R.cols(), 12);

    // from_right_matrix(M, d, chi_R): M is (chiL, d*chiR); round-trips a value.
    MPSSiteTensor u(3, 2, 4);
    u.at(2, 1, 0) = Complex128(0.4, 0.2);
    auto u2 = MPSSiteTensor::from_right_matrix(u.as_right_matrix(), 3, 4);
    EXPECT_NEAR(u2.at(2, 1, 0).real, 0.4, 1e-12);
    EXPECT_NEAR(u2.at(2, 1, 0).imag, 0.2, 1e-12);
}

// =============================================================================
// StatevectorSimulator::apply_instruction
// =============================================================================

TEST(R1121SymbolGaps, StatevectorApplyInstruction) {
    StatevectorSimulator sim;
    Statevector sv(1);
    Instruction inst;
    inst.type = Instruction::GateType::H;
    inst.qubits = {0};
    sim.apply_instruction(sv, inst);  // |0> -> |+>
    EXPECT_NEAR(sv.probability(0), 0.5, 1e-9);
    EXPECT_NEAR(sv.probability(1), 0.5, 1e-9);
}

// =============================================================================
// MPSTensor (qubit) — construct / index
// =============================================================================

TEST(R1121SymbolGaps, MpsTensorIndexing) {
    MPSTensor t(2, 3);  // bond_left=2, bond_right=3
    EXPECT_EQ(t.bond_left, 2);
    EXPECT_EQ(t.bond_right, 3);
    t(1, 0, 2) = Complex128(0.7, 0.1);
    EXPECT_NEAR(t(1, 0, 2).real, 0.7, 1e-12);
    EXPECT_NEAR(t(1, 0, 2).imag, 0.1, 1e-12);
}

// =============================================================================
// QuditSimulator::run / QuditGateOp
// =============================================================================

TEST(R1121SymbolGaps, QuditSimulatorRun) {
    QuditStatevector sv(2, 3);
    QuditGateOp g;
    g.type = QuditGateOp::Type::SINGLE;
    g.q0 = 0;
    g.matrix = qudit_gates::shift_matrix(3, 1);  // |0> -> |1>
    auto res = QuditSimulator::run(sv, {g}, 1);
    ASSERT_EQ(res.outcome.size(), 2u);
    EXPECT_EQ(res.outcome[0], 1);
    EXPECT_EQ(res.outcome[1], 0);
}

// =============================================================================
// QAOA — build_circuit + optimize (the base class is never instantiated elsewhere)
// =============================================================================

TEST(R1121SymbolGaps, QaoaBuildsAndOptimises) {
    using namespace lindblad::algorithms;
    // Ferromagnetic ZZ cost on 2 qubits: optima are "00" and "11".
    auto cost = IsingHamiltonian::from_hJ({0.0, 0.0}, {{0.0, 1.0}, {0.0, 0.0}}, 0.0)
                    .to_sparse_pauli_op();
    QAOA qaoa;
    qaoa.options.p = 1;
    qaoa.options.seed = 1;
    qaoa.options.max_iterations = 60;

    auto circ = qaoa.build_circuit(cost, SparsePauliOp{}, {0.4, 0.7});
    EXPECT_EQ(circ.n_qubits, 2);
    EXPECT_GT(circ.size(), 0);

    auto res = qaoa.optimize(cost);
    EXPECT_EQ(res.best_bitstring.size(), 2u);
}

// =============================================================================
// Plain data structs referenced nowhere else
// =============================================================================

TEST(R1121SymbolGaps, DataStructsConstruct) {
    DAGEdge e{1, 2, 0, /*is_classical=*/true};
    EXPECT_EQ(e.src_node, 1);
    EXPECT_EQ(e.dst_node, 2);
    EXPECT_TRUE(e.is_classical);

    QuditKrausChannel k;
    k.ops.push_back(qudit_gates::shift_matrix(3, 1));
    EXPECT_EQ(k.ops.size(), 1u);

    QuditQuditNoise qn;
    qn.kraus = k;
    EXPECT_EQ(qn.kraus.ops.size(), 1u);
    EXPECT_TRUE(qn.lindblad.empty());
}
