// Audit verification suite (2026-06-11, audited at R.1.11.2).
//
// One test per unexplained correctness finding of the full performance and
// correctness audit. Finding IDs (C1..C17) match the audit report and GitHub
// issues #9..#26.
//
// Each test asserts the CORRECT behaviour, so it FAILED while the
// corresponding bug was open and became a permanent regression test once the
// fix landed in R.1.12.0. This mirrors the convention of
// test_bug_regression.cpp.
//
// R.1.12.1: C4a, C16, and C17 were rewritten to the conventions frozen in
// R.1.12.0 (docs/Architecture.md "Conventions"). Their original audit
// versions pinned the superseded pre-freeze expectations (MSB-first Pauli
// docs, Qiskit ECR order, unconstrained-when-empty CouplingMap) and were the
// three deliberate reds of the R.1.12.0 baseline. The whole suite must stay
// green from R.1.12.1 onward.
//
// Per the audit methodology, every expected value below was independently
// verified by replicating the exact C++ construction in NumPy and comparing
// against the mathematically correct result.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/algorithms.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/transpiler.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

static constexpr double kTol = 1e-9;

// =============================================================================
// C1 - DM RZX must match the statevector RZX (Z on first arg, X on second).
// The DM/MPS analytic matrix currently implements RX(theta) (x) I instead.
// From |00>, RZX(theta, 0, 1) must put sin^2(theta/2) on |q1=1, q0=0> (idx 2)
// and nothing on |q0=1, q1=0> (idx 1).
// =============================================================================
TEST(AuditR1112, C1_DM_RZX_MatchesStatevector) {
    const double theta = 0.7;
    QuantumCircuit qc(2);
    qc.rzx(theta, 0, 1);

    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(qc, 0, 1);
    ASSERT_TRUE(sv_res.success);

    DensityMatrixSimulator dm_sim;
    auto dm_res = dm_sim.run(qc, NoiseModel{}, 0, 1);
    ASSERT_TRUE(dm_res.success);
    auto dm_probs = dm_res.final_state.probabilities();

    const double s2 = std::sin(theta / 2.0) * std::sin(theta / 2.0);

    // Statevector implementation is the verified reference.
    EXPECT_NEAR(sv_res.final_state.probability(2), s2, kTol);
    EXPECT_NEAR(sv_res.final_state.probability(1), 0.0, kTol);

    // DM must agree with it.
    EXPECT_NEAR(dm_probs[2], s2, 1e-7)
        << "DM RZX does not excite the X qubit (acts as a local RX instead)";
    EXPECT_NEAR(dm_probs[1], 0.0, 1e-7)
        << "DM RZX excites the Z qubit (wrong coupling axis)";
}

// =============================================================================
// C2a - DM RXX must preserve the trace. The analytic 4x4 is missing the
// diagonal cos entries on the middle rows, so states with support there
// lose probability mass.
// =============================================================================
TEST(AuditR1112, C2a_DM_RXX_PreservesTrace) {
    QuantumCircuit qc(2);
    qc.x(0);             // populate a middle row of the pair subspace
    qc.rxx(0.7, 0, 1);

    DensityMatrixSimulator dm_sim;
    auto res = dm_sim.run(qc, NoiseModel{}, 0, 1);
    ASSERT_TRUE(res.success);
    EXPECT_NEAR(res.final_state.trace(), 1.0, 1e-9)
        << "DM RXX lost trace: matrix is non-unitary (missing diagonal terms)";
}

// =============================================================================
// C2b - MPS RYY must preserve the state norm (same missing-diagonal defect
// in the MPS gate4x4 builder; MPS RXX is correct).
// =============================================================================
TEST(AuditR1112, C2b_MPS_RYY_PreservesNorm) {
    QuantumCircuit qc(2);
    qc.x(0);
    qc.ryy(0.7, 0, 1);

    MPSSimulator mps_sim;
    auto res = mps_sim.run(qc, 16, 0, 1);
    auto sv = res.final_state.to_statevector();
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-9)
        << "MPS RYY lost norm: matrix is non-unitary (missing diagonal terms)";
}

// =============================================================================
// C3 - Grover at 4 qubits. The diffusion MCX matrix targets qubit 0 while the
// H sandwich is applied to qubit 3, breaking amplitude amplification for
// every circuit with >= 4 qubits. With a correct diffusion the success
// probability for a single marked item at n=4 after auto iterations is ~0.96.
// =============================================================================
TEST(AuditR1112, C3_GroverFourQubits) {
    const int n = 4;
    const size_t dim = 1ULL << n;
    const size_t target = 5;  // |0101> : q0=1, q2=1

    // Phase oracle as a diagonal UNITARY: -1 on the marked state.
    std::vector<Complex128> oracle_mat(dim * dim, Complex128(0.0, 0.0));
    for (size_t i = 0; i < dim; ++i)
        oracle_mat[i * dim + i] = Complex128(i == target ? -1.0 : 1.0, 0.0);

    QuantumCircuit oracle(n);
    std::vector<int> all_qubits;
    for (int q = 0; q < n; ++q) all_qubits.push_back(q);
    oracle.unitary(oracle_mat, all_qubits);

    auto result = algorithms::Grover::search(oracle, -1, 2048, 42);

    // Bitstring convention: qubit 0 is the rightmost character. K=5 -> "0101".
    EXPECT_EQ(result.solution, "0101")
        << "Grover did not find the marked item at n=4";
    EXPECT_GT(result.probability, 0.5)
        << "Grover success probability collapsed at n=4 (broken diffusion)";
}

// =============================================================================
// C4a - Pauli string convention, frozen in R.1.12 (docs/Architecture.md
// "Conventions"): strings are LSB-first, pauli[q] acts on qubit q. For the
// state X(0)|00> the observable "ZI" (Z on QUBIT 0) is therefore -1, and the
// exact statevector path and the DM path must agree on it.
// (Rewritten in R.1.12.1: the original audit version pinned the superseded
// MSB-first documentation convention and expected +1.)
// =============================================================================
TEST(AuditR1112, C4a_PauliConvention_SVMatchesDM) {
    QuantumCircuit qc(2);
    qc.x(0);
    SparsePauliOp zi = SparsePauliOp::from_list({{"ZI", Complex128(1.0, 0.0)}});

    StatevectorSimulator sv_sim;
    const double sv_val = sv_sim.eval_expectation(qc, zi);

    DensityMatrixSimulator dm_sim;
    auto dm_res = dm_sim.run(qc, NoiseModel{}, 0, 1);
    ASSERT_TRUE(dm_res.success);
    const double dm_val = dm_res.final_state.expectation_value_sparse(zi);

    EXPECT_NEAR(sv_val, -1.0, kTol)
        << "frozen LSB-first convention: 'ZI' is Z on qubit 0, so X(0)|00> "
        << "must give -1 on the exact statevector path";
    EXPECT_NEAR(dm_val, -1.0, kTol)
        << "DM expectation path must follow the same frozen LSB-first "
        << "convention";
    EXPECT_NEAR(sv_val, dm_val, kTol)
        << "SV and DM expectation paths disagree on the same operator";
}

// =============================================================================
// C4b - The Estimator must give the same answer in exact mode (shots=0) and
// sampling mode (shots>0). Currently the two modes interpret the Pauli string
// with opposite qubit order: -1 vs +1 for "ZI" on X(0)|00>.
// =============================================================================
TEST(AuditR1112, C4b_PauliConvention_EstimatorModesAgree) {
    QuantumCircuit qc(2);
    qc.x(0);
    SparsePauliOp zi = SparsePauliOp::from_list({{"ZI", Complex128(1.0, 0.0)}});

    Estimator exact_est;
    exact_est.options.shots = 0;
    const double exact = exact_est.run_single(qc, zi);

    Estimator sampled_est;
    sampled_est.options.shots = 4096;
    sampled_est.options.seed = 7;
    const double sampled = sampled_est.run_single(qc, zi);

    // The state is a computational basis state: sampling has zero variance,
    // so the two modes must agree tightly.
    EXPECT_NEAR(exact, sampled, 0.05)
        << "Estimator exact (" << exact << ") and sampling (" << sampled
        << ") paths use opposite Pauli-string conventions";
}

// =============================================================================
// C5a - control() of a gate without a dedicated controlled mapping (SX) must
// not silently degrade to the identity. With ctrl=1, controlled-SX leaves
// P(target=1) = 0.5; the current generic path emits an exact identity.
// =============================================================================
TEST(AuditR1112, C5a_ControlOfSXIsNotIdentity) {
    QuantumCircuit base(1);
    base.sx(0);
    QuantumCircuit ctrl = base.control(1);  // qubit 0 = control, qubit 1 = target
    ASSERT_EQ(ctrl.n_qubits, 2);

    QuantumCircuit qc(2);
    qc.x(0);  // set control to |1>
    for (const auto& inst : ctrl.instructions) qc.instructions.push_back(inst);

    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 1);
    ASSERT_TRUE(res.success);

    // Correct controlled-SX on |ctrl=1, tgt=0>: half the population moves to tgt=1.
    EXPECT_NEAR(res.final_state.probability(1), 0.5, kTol)
        << "controlled-SX acted as identity (generic control() path emits I)";
    EXPECT_NEAR(res.final_state.probability(3), 0.5, kTol);
}

// =============================================================================
// C5b - control() of a UNITARY instruction places the gate block on the wrong
// qubits: it currently applies U to the CONTROL conditioned on the TARGET.
// control(1) of UNITARY(X) must act exactly like CX(ctrl=0, tgt=1).
// =============================================================================
TEST(AuditR1112, C5b_ControlOfUnitaryTargetsTarget) {
    std::vector<Complex128> x_mat = {
        Complex128(0, 0), Complex128(1, 0),
        Complex128(1, 0), Complex128(0, 0)
    };
    QuantumCircuit base(1);
    base.unitary(x_mat, {0});
    QuantumCircuit ctrl = base.control(1);

    QuantumCircuit qc(2);
    qc.x(0);  // ctrl = |1>, tgt = |0>
    for (const auto& inst : ctrl.instructions) qc.instructions.push_back(inst);

    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 1);
    ASSERT_TRUE(res.success);

    // Expected: |q0=1, q1=1> = index 3 with probability 1.
    EXPECT_NEAR(res.final_state.probability(3), 1.0, kTol)
        << "controlled-UNITARY(X) did not flip the target "
        << "(block-diagonal layout conditions on the wrong qubits)";
}

// =============================================================================
// C6 - A UNITARY instruction must mean the same operation on the statevector
// and density-matrix simulators. apply_unitary maps qubits[0] to the LSB of
// the matrix index; DensityMatrix::apply_gate maps qubits[0] to the MSB and
// is missing the convention bridge that the MPS simulator has.
// =============================================================================
TEST(AuditR1112, C6_UnitarySVvsDMAgree) {
    // CX-shaped matrix in the apply_unitary convention: control = qubits[0].
    std::vector<Complex128> cx_mat(16, Complex128(0.0, 0.0));
    cx_mat[0 * 4 + 0] = Complex128(1, 0);
    cx_mat[1 * 4 + 3] = Complex128(1, 0);
    cx_mat[2 * 4 + 2] = Complex128(1, 0);
    cx_mat[3 * 4 + 1] = Complex128(1, 0);

    QuantumCircuit qc(2);
    qc.x(0);
    qc.unitary(cx_mat, {0, 1});

    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(qc, 0, 1);
    ASSERT_TRUE(sv_res.success);

    DensityMatrixSimulator dm_sim;
    auto dm_res = dm_sim.run(qc, NoiseModel{}, 0, 1);
    ASSERT_TRUE(dm_res.success);
    auto dm_probs = dm_res.final_state.probabilities();

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(sv_res.final_state.probability(i), dm_probs[i], 1e-7)
            << "SV and DM disagree on UNITARY qubit order at index " << i;
    }
}

// =============================================================================
// C7 - thermal_relaxation must dephase coherences as exp(-t/T2). In the pure
// dephasing regime (T1 -> inf, t = T2) the |+> coherence must decay by e^-1.
// The current channel decays it by e^-0.5 (half rate).
// =============================================================================
TEST(AuditR1112, C7_ThermalRelaxationT2Rate) {
    const double T1 = 1e9, T2 = 1.0, t = 1.0;
    auto ch = NoiseChannels::thermal_relaxation(T1, T2, t);
    ASSERT_TRUE(ch.is_valid(1e-8));  // sanity: trace preserving either way

    // Build rho = |+><+| and apply the channel.
    DensityMatrix dm(1);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    std::vector<Complex128> h_mat = {
        Complex128(inv_sqrt2, 0), Complex128(inv_sqrt2, 0),
        Complex128(inv_sqrt2, 0), Complex128(-inv_sqrt2, 0)
    };
    dm.apply_gate(h_mat, {0});
    dm.apply_kraus(ch.operators, {0});

    // rho_01 must be exp(-t/T2)/2 = e^-1/2 (up to the negligible T1 term).
    const double expected = std::exp(-t / T2) / 2.0;
    EXPECT_NEAR(dm.data[1].real, expected, 5e-3)
        << "coherence decayed to " << dm.data[1].real * 2.0
        << " instead of e^-(t/T2) = " << std::exp(-t / T2)
        << " (pure dephasing rate is half of what T2 specifies)";
}

// =============================================================================
// C8 - MPS sampling above the statevector crossover (n > 18) must use the
// project bitstring convention (qubit 0 = rightmost character).
// measure_sequential currently writes qubit q at position q (reversed).
// =============================================================================
TEST(AuditR1112, C8_MPSSequentialSamplingBitOrder) {
    const int n = 19;  // > MPS_SV_CROSSOVER, product state so chi = 1
    QuantumCircuit qc(n);
    qc.x(0);

    MPSSimulator sim;
    auto res = sim.run(qc, 4, 32, 42);

    std::string expected(n, '0');
    expected[n - 1] = '1';  // qubit 0 rightmost

    ASSERT_FALSE(res.counts.empty());
    for (const auto& [bits, cnt] : res.counts) {
        EXPECT_EQ(bits, expected)
            << "MPS sequential sampling returned reversed bitstring " << bits;
    }
}

// =============================================================================
// C9 - After a mid-circuit measurement the MPS state must stay normalised.
// The collapse renormalises by the LOCAL tensor norm, which is only correct
// in canonical form; for this circuit the post-measure global norm is ~0.707.
// =============================================================================
TEST(AuditR1112, C9_MPSMidCircuitMeasureKeepsNorm) {
    QuantumCircuit qc(3, 1);
    qc.h(0);
    qc.cx(0, 1);
    qc.cx(1, 2);
    qc.ry(0.9, 1);
    qc.measure(1, 0);  // mid-circuit collapse on a non-canonical site

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 1, 42);

    auto sv = res.final_state.to_statevector();
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-6)
        << "post-measurement MPS norm^2 = " << sv.norm_sq()
        << " (local-Frobenius renormalisation is wrong off canonical form)";
}

// =============================================================================
// C10 - shots=0 execution must honour classical conditions. Measuring q0 of
// |00> always stores 0, so a gate conditioned on clbit==1 must NOT fire.
// simulate_circuit currently applies conditional gates unconditionally.
// =============================================================================
TEST(AuditR1112, C10_ShotsZeroHonoursConditions) {
    QuantumCircuit qc(2, 1);
    qc.measure(0, 0);                       // deterministically writes 0
    qc.add_if(0, 1, GT::X, {1});            // must not fire

    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 42);
    ASSERT_TRUE(res.success);

    EXPECT_NEAR(res.final_state.probability(0), 1.0, kTol)
        << "conditional X fired although its clbit condition (==1) is false";
}

// =============================================================================
// C11a - Optimize1qGates must not merge a classically conditioned gate into an
// unconditional rotation. After transpile(level 1) the conditional P must
// still carry its condition.
// =============================================================================
TEST(AuditR1112, C11a_Optimize1qKeepsConditions) {
    QuantumCircuit qc(1, 1);
    qc.h(0);
    qc.measure(0, 0);
    qc.p_if(PI, 0, 0, 1);

    auto out = transpile(qc, CouplingMap(), {}, 1);

    bool has_conditional = false;
    for (const auto& inst : out.instructions) {
        if (inst.condition_clbit >= 0) { has_conditional = true; break; }
    }
    EXPECT_TRUE(has_conditional)
        << "transpile(level 1) dropped the classical condition from p_if";
}

// =============================================================================
// C11b - CXCancellation must not cancel a conditional CX against an
// unconditional CX on the same qubits.
// =============================================================================
TEST(AuditR1112, C11b_CXCancellationRespectsConditions) {
    QuantumCircuit qc(2, 1);
    qc.cx(0, 1);
    qc.add_if(0, 1, GT::CX, {0, 1});  // conditional CX, same qubits

    auto out = transpile(qc, CouplingMap(), {}, 1);

    int cx_count = 0;
    for (const auto& inst : out.instructions)
        if (inst.type == GT::CX) ++cx_count;

    EXPECT_EQ(cx_count, 2)
        << "a conditional CX was cancelled against an unconditional CX";
}

// =============================================================================
// C12 - The DAG must order a conditional gate after the measurement writing
// its condition bit, even when they share no qubit wire. The dependency must
// exist as an edge (successors of the measure node include the conditional).
// =============================================================================
TEST(AuditR1112, C12_DAGOrdersConditionalAfterMeasure) {
    QuantumCircuit qc(2, 1);
    qc.measure(0, 0);
    qc.p_if(0.5, 1, 0, 1);  // acts on qubit 1: only the clbit links them

    auto dag = DAGCircuit::from_circuit(qc);

    int measure_id = -1, cond_id = -1;
    for (const auto& node : dag.nodes) {
        if (node.type != DAGNode::Type::OP) continue;
        if (node.op.type == GT::MEASURE) measure_id = node.node_id;
        if (node.op.condition_clbit >= 0) cond_id = node.node_id;
    }
    ASSERT_GE(measure_id, 0);
    ASSERT_GE(cond_id, 0);

    bool ordered = false;
    for (int succ : dag.successors(measure_id)) {
        if (succ == cond_id) { ordered = true; break; }
    }
    EXPECT_TRUE(ordered)
        << "no DAG dependency from the measurement to the gate conditioned on "
        << "its clbit: topological order may legally reorder feedforward";
}

// =============================================================================
// C13 - ConsolidateBlocks (level 3) must preserve gates it cannot decompose.
// instruction_to_4x4 currently maps RZX (and ECR/CU/UNITARY) to identity
// inside multi-gate blocks, silently deleting their effect.
// =============================================================================
TEST(AuditR1112, C13_ConsolidateBlocksPreservesRZX) {
    QuantumCircuit qc(2);
    qc.rzx(0.4, 0, 1);
    qc.rzx(0.3, 0, 1);  // two-gate block on the same pair triggers KAK

    StatevectorSimulator sim;
    auto ref = sim.run(qc, 0, 1);
    ASSERT_TRUE(ref.success);

    auto out = transpile(qc, CouplingMap(), {}, 3);
    auto opt = sim.run(out, 0, 1);
    ASSERT_TRUE(opt.success);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(ref.final_state.probability(i),
                    opt.final_state.probability(i), 1e-6)
            << "level-3 transpile changed the circuit semantics at index " << i;
    }
}

// =============================================================================
// C14 - QASM 2 import must not silently drop whole-register measurements.
// =============================================================================
TEST(AuditR1112, C14_QASM2RegisterMeasure) {
    const std::string qasm =
        "OPENQASM 2.0;\n"
        "include \"qelib1.inc\";\n"
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "measure q -> c;\n";

    auto qc = QuantumCircuit::from_qasm2(qasm);
    auto ops = qc.count_ops();
    const int measures = ops.count("measure") ? ops.at("measure") : 0;

    EXPECT_EQ(measures, 2)
        << "whole-register 'measure q -> c;' was silently dropped on import";
}

// =============================================================================
// C15 - depolarizing(p, 3) must not return an empty (state-annihilating)
// channel. Either a valid 3-qubit channel or a thrown exception is correct;
// an empty operator list is not.
// =============================================================================
TEST(AuditR1112, C15_DepolarizingThreeQubitNotEmpty) {
    bool threw = false;
    KrausChannel ch;
    try {
        ch = NoiseChannels::depolarizing(0.1, 3);
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) {
        EXPECT_FALSE(ch.operators.empty())
            << "depolarizing(p, 3) returned an empty channel: applying it "
            << "maps every density matrix to zero";
        if (!ch.operators.empty()) {
            ch.n_qubits = 3;
            EXPECT_TRUE(ch.is_valid(1e-8));
        }
    }
    SUCCEED();
}

// =============================================================================
// C16 - ECR argument convention, frozen in R.1.12 (docs/api/gates.md):
// lindblad keeps its own order as a documented deliberate deviation,
// lindblad ecr(a, b) == Qiskit ecr(b, a). Pinned from both directions:
//   x(0); ecr(0,1): |q0=1, q1=0> -> (|00> + i|q1=1>)/sqrt(2)
//                   P(idx0) = P(idx2) = 0.5, P(idx1) = P(idx3) = 0
//   x(1); ecr(0,1): the |q1=1> input maps ENTIRELY into the q0=1 subspace,
//                   which is exactly Qiskit's ecr(1, 0) behaviour.
// (Rewritten in R.1.12.1: the original audit version asserted Qiskit's
// ecr(0,1) semantics; the maintainer decision was keep-and-document.)
// =============================================================================
TEST(AuditR1112, C16_ECRMatchesDocumentedConvention) {
    Statevector sv(2);
    gates::apply_x(sv, 0);        // |q0=1, q1=0>
    gates::apply_ecr(sv, 0, 1);

    EXPECT_NEAR(sv.probability(0), 0.5, kTol);
    EXPECT_NEAR(sv.probability(1), 0.0, kTol);
    EXPECT_NEAR(sv.probability(2), 0.5, kTol);
    EXPECT_NEAR(sv.probability(3), 0.0, kTol);

    Statevector sv2(2);
    gates::apply_x(sv2, 1);       // |q0=0, q1=1>
    gates::apply_ecr(sv2, 0, 1);

    const double p_q0_high = sv2.probability(1) + sv2.probability(3);
    EXPECT_NEAR(p_q0_high, 1.0, kTol)
        << "ecr(0,1) on |q1=1> must populate only the q0=1 subspace "
        << "(lindblad ecr(a,b) == Qiskit ecr(b,a), docs/api/gates.md)";
}

// =============================================================================
// C17 - CouplingMap semantics, frozen in R.1.12 (docs/api/transpiler.md):
// the edge list is LITERAL. An edgeless CouplingMap(n) declares n qubits
// where no pair may interact, so routing a 2-qubit gate against it must
// THROW (never silently truncate the circuit, which was the original bug).
// Unconstrained transpilation is expressed by CouplingMap() (n = 0) and must
// preserve circuit semantics.
// (Rewritten in R.1.12.1: the original audit version pinned the pre-freeze
// "unconstrained when empty" reading and expected the gates to survive.)
// =============================================================================
TEST(AuditR1112, C17_EdgelessCouplingMapIsLiteral) {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);

    // Literal no-edge graph: routing must refuse loudly.
    EXPECT_THROW(transpile(qc, CouplingMap(2), {}, 1), std::runtime_error)
        << "routing a 2-qubit gate against an edgeless CouplingMap(n) must "
        << "throw instead of silently dropping gates";

    // Unconstrained (no map declared): semantics preserved.
    auto out = transpile(qc, CouplingMap(), {}, 1);

    StatevectorSimulator sim;
    auto ref = sim.run(qc, 0, 1);
    auto opt = sim.run(out, 0, 1);
    ASSERT_TRUE(ref.success);
    ASSERT_TRUE(opt.success);

    // The Bell state has P(00) = P(11) = 0.5.
    EXPECT_NEAR(opt.final_state.probability(3),
                ref.final_state.probability(3), 1e-6)
        << "unconstrained transpile (CouplingMap()) changed the Bell statistics";
}
