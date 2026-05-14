// R.1.5.1 test suite — feedforward infrastructure + semi-classical (Griffiths-Niu) QFT
//
// Covers:
//   - p_if / add_if circuit construction API
//   - Classically-conditioned gate execution on all 4 simulators (SV, DM, Clifford, MPS)
//   - Clifford P-gate angle mapping and is_clifford() recognition
//   - build_iterative_circuit / build_iterative_inverse_circuit structure
//   - run_iterative correctness on all 4 simulators, including feedforward-exercises tests

#include <gtest/gtest.h>
#include "lindblad/circuit.hpp"
#include "lindblad/algorithms.hpp"
#include "lindblad/backends/local_backend.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <cmath>
#include <string>
#include <unordered_map>

using namespace lindblad;
using namespace lindblad::algorithms;
using GT = Instruction::GateType;

static constexpr double kPi = M_PI;

// =============================================================================
// Helpers
// =============================================================================

static backends::LocalBackend make_backend(backends::LocalBackend::SimType sim) {
    backends::LocalBackend::Config cfg;
    cfg.simulator = sim;
    return backends::LocalBackend(cfg);
}

static int total_shots(const std::unordered_map<std::string, int>& counts) {
    int n = 0;
    for (const auto& [k, v] : counts) n += v;
    return n;
}

// Run a simple 2-qubit conditional-X circuit on the given backend.
// qubit 0: X applied iff prep_x. MEASURE(0,0). add_if(c[0]==clval → X on qubit 1). MEASURE(1,1).
// Bitstring format: bits[0]=c[1], bits[1]=c[0]  (clbit 0 = rightmost, LSB).
static std::unordered_map<std::string, int>
run_conditional_x(backends::LocalBackend::SimType sim,
                  bool prep_x, int clval,
                  int shots = 200, uint64_t seed = 42)
{
    QuantumCircuit qc(2, 2);
    if (prep_x) qc.x(0);
    qc.measure(0, 0);
    qc.add_if(0, clval, GT::X, {1});
    qc.measure(1, 1);
    auto backend = make_backend(sim);
    return backend.run(qc, shots, seed).counts;
}

// =============================================================================
// Suite 1 — CircuitFeedforwardAPI: p_if / add_if construction
// =============================================================================

TEST(CircuitFeedforwardAPI, PIfGateTypeIsP) {
    QuantumCircuit qc(1, 1);
    qc.p_if(kPi / 2.0, 0, 0, 1);
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::P);
}

TEST(CircuitFeedforwardAPI, PIfStoresAngle) {
    QuantumCircuit qc(1, 1);
    qc.p_if(kPi / 4.0, 0, 0, 1);
    ASSERT_EQ(qc.instructions[0].params.size(), 1u);
    EXPECT_NEAR(qc.instructions[0].params[0], kPi / 4.0, 1e-12);
}

TEST(CircuitFeedforwardAPI, PIfStoresQubit) {
    QuantumCircuit qc(2, 2);
    qc.p_if(kPi, 1, 0, 1);
    ASSERT_EQ(qc.instructions[0].qubits.size(), 1u);
    EXPECT_EQ(qc.instructions[0].qubits[0], 1);
}

TEST(CircuitFeedforwardAPI, PIfStoresConditionClbit) {
    QuantumCircuit qc(2, 3);
    qc.p_if(kPi, 0, 2, 1);
    EXPECT_EQ(qc.instructions[0].condition_clbit, 2);
}

TEST(CircuitFeedforwardAPI, PIfDefaultClvalIsOne) {
    QuantumCircuit qc(1, 1);
    qc.p_if(kPi, 0, 0);  // clval omitted
    EXPECT_EQ(qc.instructions[0].condition_value, 1);
}

TEST(CircuitFeedforwardAPI, PIfCustomClvalZero) {
    QuantumCircuit qc(1, 1);
    qc.p_if(kPi, 0, 0, 0);
    EXPECT_EQ(qc.instructions[0].condition_value, 0);
}

TEST(CircuitFeedforwardAPI, AddIfStoresGateType) {
    QuantumCircuit qc(2, 1);
    qc.add_if(0, 1, GT::X, {1});
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::X);
}

TEST(CircuitFeedforwardAPI, AddIfStoresQubits) {
    QuantumCircuit qc(3, 1);
    qc.add_if(0, 1, GT::CX, {0, 2});
    ASSERT_EQ(qc.instructions[0].qubits.size(), 2u);
    EXPECT_EQ(qc.instructions[0].qubits[0], 0);
    EXPECT_EQ(qc.instructions[0].qubits[1], 2);
}

TEST(CircuitFeedforwardAPI, AddIfStoresConditionClbit) {
    QuantumCircuit qc(2, 3);
    qc.add_if(2, 1, GT::H, {1});
    EXPECT_EQ(qc.instructions[0].condition_clbit, 2);
}

TEST(CircuitFeedforwardAPI, AddIfStoresConditionValue) {
    QuantumCircuit qc(1, 1);
    qc.add_if(0, 0, GT::X, {0});
    EXPECT_EQ(qc.instructions[0].condition_value, 0);
}

TEST(CircuitFeedforwardAPI, AddIfStoresParams) {
    QuantumCircuit qc(1, 1);
    qc.add_if(0, 1, GT::RZ, {0}, {kPi / 3.0});
    ASSERT_EQ(qc.instructions[0].params.size(), 1u);
    EXPECT_NEAR(qc.instructions[0].params[0], kPi / 3.0, 1e-12);
}

TEST(CircuitFeedforwardAPI, PIfAppendsDoesNotClearPriorInstructions) {
    QuantumCircuit qc(1, 1);
    qc.h(0);
    qc.p_if(kPi, 0, 0, 1);
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[1].type, GT::P);
}

TEST(CircuitFeedforwardAPI, PIfFluentChain) {
    QuantumCircuit qc(2, 2);
    qc.h(0).p_if(kPi, 1, 0, 1).measure(1, 1);
    EXPECT_EQ(qc.instructions.size(), 3u);
}

TEST(CircuitFeedforwardAPI, PIfInvalidQubitThrows) {
    QuantumCircuit qc(1, 1);
    EXPECT_THROW(qc.p_if(kPi, 5, 0, 1), std::exception);
}

TEST(CircuitFeedforwardAPI, PIfInvalidClbitThrows) {
    QuantumCircuit qc(1, 1);
    EXPECT_THROW(qc.p_if(kPi, 0, 5, 1), std::exception);
}

TEST(CircuitFeedforwardAPI, AddIfInvalidClbitThrows) {
    QuantumCircuit qc(2, 1);
    EXPECT_THROW(qc.add_if(5, 1, GT::X, {0}), std::exception);
}

TEST(CircuitFeedforwardAPI, AddIfInvalidQubitThrows) {
    QuantumCircuit qc(2, 1);
    EXPECT_THROW(qc.add_if(0, 1, GT::X, {5}), std::exception);
}

// =============================================================================
// Suite 2 — FeedforwardStatevector
// =============================================================================

// Bitstring layout (n_clbits=2): bits[0]=c[1], bits[1]=c[0].
// So "XY" means c[1]=X, c[0]=Y.

TEST(FeedforwardStatevector, ConditionMet_XApplied) {
    // X on qubit 0 → c[0]=1 → add_if(clval=1) fires → qubit 1 = |1⟩ → c[1]=1 → "11"
    auto counts = run_conditional_x(backends::LocalBackend::SimType::STATEVECTOR,
                                    true, 1, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("11"), 200);
}

TEST(FeedforwardStatevector, ConditionNotMet_XSkipped) {
    // qubit 0 = |0⟩ → c[0]=0 → add_if(clval=1) skipped → qubit 1 = |0⟩ → "00"
    auto counts = run_conditional_x(backends::LocalBackend::SimType::STATEVECTOR,
                                    false, 1, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 200);
}

TEST(FeedforwardStatevector, ClvalZero_FiresOnZeroMeasurement) {
    // qubit 0 = |0⟩ → c[0]=0 == clval=0 → add_if fires → qubit 1 = |1⟩ → "10"
    auto counts = run_conditional_x(backends::LocalBackend::SimType::STATEVECTOR,
                                    false, 0, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("10"), 200);
}

TEST(FeedforwardStatevector, ClvalZero_BlockedOnOneMeasurement) {
    // qubit 0 = |1⟩ → c[0]=1 ≠ clval=0 → add_if blocked → qubit 1 = |0⟩ → "01"
    auto counts = run_conditional_x(backends::LocalBackend::SimType::STATEVECTOR,
                                    true, 0, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("01"), 200);
}

TEST(FeedforwardStatevector, TwoIndependentBits) {
    // c[0] from qubit 0 (|1⟩ → 1) triggers X on qubit 2.
    // c[1] from qubit 1 (|0⟩ → 0) does NOT trigger X on qubit 3.
    // n=4 qubits, n_clbits=4. Bitstring: bits[3]=c[0], bits[2]=c[1], bits[1]=c[2], bits[0]=c[3].
    // c[0]=1, c[1]=0, c[2]=1 (X applied), c[3]=0 (X skipped) → "0101"
    QuantumCircuit qc(4, 4);
    qc.x(0);
    qc.measure(0, 0);
    qc.measure(1, 1);
    qc.add_if(0, 1, GT::X, {2});
    qc.add_if(1, 1, GT::X, {3});
    qc.measure(2, 2);
    qc.measure(3, 3);

    auto backend = make_backend(backends::LocalBackend::SimType::STATEVECTOR);
    auto result  = backend.run(qc, 200, 42);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("0101"), 200);
}

TEST(FeedforwardStatevector, NonConditionalGatesAfterPIfStillApply) {
    // qubit 0 = |0⟩ → c[0]=0 → p_if(clval=1) skipped → X(0) unconditional applies.
    // Second MEASURE gets c[1]=1. Bitstring: bits[0]=c[1]=1, bits[1]=c[0]=0 → "10"
    QuantumCircuit qc(1, 2);
    qc.measure(0, 0);
    qc.p_if(kPi / 2.0, 0, 0, 1);  // skipped (c[0]=0 ≠ 1)
    qc.x(0);                        // always applied
    qc.measure(0, 1);

    auto backend = make_backend(backends::LocalBackend::SimType::STATEVECTOR);
    auto result  = backend.run(qc, 200, 7);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("10"), 200);
}

TEST(FeedforwardStatevector, TotalShotsPreserved) {
    QuantumCircuit qc(1, 1);
    qc.measure(0, 0);
    auto backend = make_backend(backends::LocalBackend::SimType::STATEVECTOR);
    auto result  = backend.run(qc, 333, 1);
    EXPECT_EQ(total_shots(result.counts), 333);
}

// =============================================================================
// Suite 3 — FeedforwardDensityMatrix
// =============================================================================

TEST(FeedforwardDensityMatrix, ConditionMet_XApplied) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::DENSITY_MATRIX,
                                    true, 1, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("11"), 200);
}

TEST(FeedforwardDensityMatrix, ConditionNotMet_XSkipped) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::DENSITY_MATRIX,
                                    false, 1, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 200);
}

TEST(FeedforwardDensityMatrix, ClvalZero_FiresOnZero) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::DENSITY_MATRIX,
                                    false, 0, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("10"), 200);
}

TEST(FeedforwardDensityMatrix, ClvalZero_BlockedOnOne) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::DENSITY_MATRIX,
                                    true, 0, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("01"), 200);
}

TEST(FeedforwardDensityMatrix, NoFeedforward_BellStateCorrect) {
    // Without feedforward, DM sim must still give correct Bell-state distribution.
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).measure_all();
    auto dm = make_backend(backends::LocalBackend::SimType::DENSITY_MATRIX);
    auto sv = make_backend(backends::LocalBackend::SimType::STATEVECTOR);
    auto r_dm = dm.run(qc, 2000, 99);
    auto r_sv = sv.run(qc, 2000, 99);
    // Both must produce only "00" and "11".
    EXPECT_GT(r_dm.counts.count("00"), 0u);
    EXPECT_GT(r_dm.counts.count("11"), 0u);
    EXPECT_EQ(r_dm.counts.count("01") + r_dm.counts.count("10"), 0u);
    // Same structure as SV.
    EXPECT_EQ(r_sv.counts.count("01") + r_sv.counts.count("10"), 0u);
}

TEST(FeedforwardDensityMatrix, MidCircuitCollapseAfterMeasure) {
    // qubit 0 = |1⟩. First MEASURE collapses DM; second must agree.
    // p_if(0, ...) = P(0)=identity, fires but has no effect on state.
    // Its presence (condition_clbit >= 0) activates per-shot feedforward mode,
    // which is the only DM path that tracks clreg across mid-circuit MEASUREs.
    QuantumCircuit qc(1, 2);
    qc.x(0);
    qc.measure(0, 0);
    qc.p_if(0.0, 0, 0, 1);  // P(0)=I, c[0]=1 → fires but no-op; enables feedforward mode
    qc.measure(0, 1);
    auto backend = make_backend(backends::LocalBackend::SimType::DENSITY_MATRIX);
    auto result  = backend.run(qc, 200, 3);
    // c[0]=1, c[1]=1 → bits[0]=c[1]=1, bits[1]=c[0]=1 → "11"
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("11"), 200);
}

TEST(FeedforwardDensityMatrix, TotalShotsPreserved) {
    QuantumCircuit qc(1, 1);
    qc.measure(0, 0);
    auto backend = make_backend(backends::LocalBackend::SimType::DENSITY_MATRIX);
    auto result  = backend.run(qc, 257, 1);
    EXPECT_EQ(total_shots(result.counts), 257);
}

// =============================================================================
// Suite 4 — FeedforwardClifford
// =============================================================================

TEST(FeedforwardClifford, ConditionMet_XApplied) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::CLIFFORD,
                                    true, 1, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("11"), 200);
}

TEST(FeedforwardClifford, ConditionNotMet_XSkipped) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::CLIFFORD,
                                    false, 1, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 200);
}

TEST(FeedforwardClifford, PGateAngle0_IsIdentity) {
    // P(0) = I. Applying to |1⟩ must still give |1⟩ on measurement.
    QuantumCircuit qc(1, 1);
    qc.x(0).p(0.0, 0).measure(0, 0);
    auto backend = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto result  = backend.run(qc, 100, 1);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("1"), 100);
}

TEST(FeedforwardClifford, PGateAnglePiOver2_MatchesSVSim) {
    // P(π/2) = S. H·S·H|0⟩ = H·S·H|0⟩: final state must match SV.
    QuantumCircuit qc(1, 1);
    qc.h(0).p(kPi / 2.0, 0).h(0).measure(0, 0);
    auto sv  = make_backend(backends::LocalBackend::SimType::STATEVECTOR);
    auto clf = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto r_sv  = sv.run(qc, 200, 5);
    auto r_clf = clf.run(qc, 200, 5);
    // H·S·H|0⟩ = H·(|0⟩+i|1⟩)/√2 = (1+i)/2|0⟩+(1-i)/2|1⟩ → 50/50.
    // Both simulators must agree that both outcomes appear.
    EXPECT_GT(r_clf.counts.count("0"), 0u);
    EXPECT_GT(r_clf.counts.count("1"), 0u);
    EXPECT_EQ(r_clf.counts.size(), r_sv.counts.size());
}

TEST(FeedforwardClifford, PGateAnglePi_IsZ) {
    // P(π) = Z. Z|1⟩ = -|1⟩ (global phase). Measurement still gives 1.
    QuantumCircuit qc(1, 1);
    qc.x(0).p(kPi, 0).measure(0, 0);
    auto backend = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto result  = backend.run(qc, 100, 1);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("1"), 100);
}

TEST(FeedforwardClifford, PGateAngle3PiOver2_IsSDG) {
    // P(3π/2) = SDG. SDG|1⟩ = -i|1⟩ → measurement gives 1.
    QuantumCircuit qc(1, 1);
    qc.x(0).p(3.0 * kPi / 2.0, 0).measure(0, 0);
    auto backend = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto result  = backend.run(qc, 100, 1);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("1"), 100);
}

TEST(FeedforwardClifford, PGateAngle0_OnPlus_StillFiftyFifty) {
    // P(0) = identity. |+⟩ measured in Z-basis: 50/50.
    QuantumCircuit qc(1, 1);
    qc.h(0).p(0.0, 0).measure(0, 0);
    auto backend = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto result  = backend.run(qc, 400, 7);
    EXPECT_GT(result.counts.count("0"), 0u);
    EXPECT_GT(result.counts.count("1"), 0u);
}

TEST(FeedforwardClifford, NonCliffordPAngle_LocalBackendReturnsFailure) {
    // P(π/4) is not a Clifford gate. LocalBackend catches the throw → success=false.
    QuantumCircuit qc(1, 1);
    qc.p(kPi / 4.0, 0).measure(0, 0);
    auto backend = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto result  = backend.run(qc, 10, 1);
    EXPECT_FALSE(result.success);
}

TEST(FeedforwardClifford, NonCliffordPAngle_DirectSimThrows) {
    // Direct CliffordSimulator::run() must throw for P(π/4).
    QuantumCircuit qc(1, 1);
    qc.p(kPi / 4.0, 0).measure(0, 0);
    CliffordSimulator sim;
    EXPECT_THROW(sim.run(qc, 10, 1), std::exception);
}

TEST(FeedforwardClifford, IsCliffordFalseForPiOver4) {
    QuantumCircuit qc(1, 1);
    qc.p(kPi / 4.0, 0).measure(0, 0);
    EXPECT_FALSE(CliffordSimulator::is_clifford(qc));
}

TEST(FeedforwardClifford, IsCliffordTrueForP0) {
    QuantumCircuit qc(1, 1);
    qc.p(0.0, 0).measure(0, 0);
    EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
}

TEST(FeedforwardClifford, IsCliffordTrueForPHalfPi) {
    QuantumCircuit qc(1, 1);
    qc.p(kPi / 2.0, 0).measure(0, 0);
    EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
}

TEST(FeedforwardClifford, IsCliffordTrueForPPi) {
    QuantumCircuit qc(1, 1);
    qc.p(kPi, 0).measure(0, 0);
    EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
}

TEST(FeedforwardClifford, IsCliffordTrueForP3HalfPi) {
    QuantumCircuit qc(1, 1);
    qc.p(3.0 * kPi / 2.0, 0).measure(0, 0);
    EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
}

TEST(FeedforwardClifford, ConditionalPIfOnCliffordAngle_Executes) {
    // p_if(π/2, qubit 1, c[0]=1). qubit 0 = |1⟩ → c[0]=1 → S on qubit 1 (|0⟩ stays |0⟩).
    // Bitstring: c[0]=1, c[1]=0 → "01"
    QuantumCircuit qc(2, 2);
    qc.x(0).measure(0, 0);
    qc.p_if(kPi / 2.0, 1, 0, 1);
    qc.measure(1, 1);
    auto backend = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto result  = backend.run(qc, 100, 1);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("01"), 100);
}

// =============================================================================
// Suite 5 — FeedforwardMPS
// =============================================================================

TEST(FeedforwardMPS, ConditionMet_XApplied) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::MPS,
                                    true, 1, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("11"), 200);
}

TEST(FeedforwardMPS, ConditionNotMet_XSkipped) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::MPS,
                                    false, 1, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("00"), 200);
}

TEST(FeedforwardMPS, ClvalZero_FiresOnZero) {
    auto counts = run_conditional_x(backends::LocalBackend::SimType::MPS,
                                    false, 0, 200, 42);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at("10"), 200);
}

// =============================================================================
// Suite 6 — BuildIterativeCircuit: structure
// =============================================================================

TEST(BuildIterativeCircuit, N1_QubitsAndClbits) {
    auto qc = QFT::build_iterative_circuit(1);
    EXPECT_EQ(qc.n_qubits, 1);
    EXPECT_EQ(qc.n_clbits, 1);
}

TEST(BuildIterativeCircuit, N2_QubitsAndClbits) {
    auto qc = QFT::build_iterative_circuit(2);
    EXPECT_EQ(qc.n_qubits, 2);
    EXPECT_EQ(qc.n_clbits, 2);
}

TEST(BuildIterativeCircuit, N4_QubitsAndClbitsMatch) {
    auto qc = QFT::build_iterative_circuit(4);
    EXPECT_EQ(qc.n_qubits, 4);
    EXPECT_EQ(qc.n_clbits, 4);
}

TEST(BuildIterativeCircuit, N1_InstructionCount) {
    // j=0: 0 p_if + H + MEASURE = 2
    EXPECT_EQ(QFT::build_iterative_circuit(1).instructions.size(), 2u);
}

TEST(BuildIterativeCircuit, N2_InstructionCount) {
    // j=1: 0+H+M=2; j=0: 1+H+M=3 → 5
    EXPECT_EQ(QFT::build_iterative_circuit(2).instructions.size(), 5u);
}

TEST(BuildIterativeCircuit, N3_InstructionCount) {
    // j=2: 2; j=1: 3; j=0: 4 → 9
    EXPECT_EQ(QFT::build_iterative_circuit(3).instructions.size(), 9u);
}

TEST(BuildIterativeCircuit, N4_InstructionCount) {
    // j=3: 2; j=2: 3; j=1: 4; j=0: 5 → 14
    EXPECT_EQ(QFT::build_iterative_circuit(4).instructions.size(), 14u);
}

TEST(BuildIterativeCircuit, N1_SequenceHthenMeasure) {
    auto qc = QFT::build_iterative_circuit(1);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[1].type, GT::MEASURE);
}

TEST(BuildIterativeCircuit, N2_FirstInstrHOnQubitNMinus1) {
    // j = n-1 = 1 processed first → H(1)
    auto qc = QFT::build_iterative_circuit(2);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[0].qubits[0], 1);
}

TEST(BuildIterativeCircuit, N2_MeasureQubit1ToClbit1) {
    auto qc = QFT::build_iterative_circuit(2);
    EXPECT_EQ(qc.instructions[1].type, GT::MEASURE);
    EXPECT_EQ(qc.instructions[1].qubits[0], 1);
    EXPECT_EQ(qc.instructions[1].clbits[0], 1);
}

TEST(BuildIterativeCircuit, N2_PIfOnQubit0CondOnClbit1_AnglePiOver2) {
    // j=0, k=1: p_if(π/2, qubit=0, clbit=1, clval=1)
    auto qc = QFT::build_iterative_circuit(2);
    const auto& inst = qc.instructions[2];
    EXPECT_EQ(inst.type, GT::P);
    EXPECT_EQ(inst.qubits[0], 0);
    EXPECT_EQ(inst.condition_clbit, 1);
    EXPECT_EQ(inst.condition_value, 1);
    EXPECT_NEAR(inst.params[0], kPi / 2.0, 1e-10);
}

TEST(BuildIterativeCircuit, N2_LastTwoHAndMeasureOnQubit0) {
    auto qc = QFT::build_iterative_circuit(2);
    EXPECT_EQ(qc.instructions[3].type, GT::H);
    EXPECT_EQ(qc.instructions[3].qubits[0], 0);
    EXPECT_EQ(qc.instructions[4].type, GT::MEASURE);
    EXPECT_EQ(qc.instructions[4].qubits[0], 0);
    EXPECT_EQ(qc.instructions[4].clbits[0], 0);
}

TEST(BuildIterativeCircuit, N3_PIfForJ1CondOnClbit2_AnglePiOver2) {
    // j=1, k=2: angle=π/2^(2-1)=π/2, qubit=1, clbit=2
    auto qc = QFT::build_iterative_circuit(3);
    const auto& inst = qc.instructions[2];
    EXPECT_EQ(inst.type, GT::P);
    EXPECT_EQ(inst.qubits[0], 1);
    EXPECT_EQ(inst.condition_clbit, 2);
    EXPECT_NEAR(inst.params[0], kPi / 2.0, 1e-10);
}

TEST(BuildIterativeCircuit, N3_PIfForJ0_K2_AnglePiOver4) {
    // j=0, k=2 (first inner k for j=0): angle=π/2^2=π/4
    auto qc = QFT::build_iterative_circuit(3);
    // Layout: [0]=H(2),[1]=M(2,2),[2]=p_if(j=1,k=2),[3]=H(1),[4]=M(1,1),
    //         [5]=p_if(j=0,k=2),[6]=p_if(j=0,k=1),[7]=H(0),[8]=M(0,0)
    const auto& inst = qc.instructions[5];
    EXPECT_EQ(inst.type, GT::P);
    EXPECT_EQ(inst.qubits[0], 0);
    EXPECT_EQ(inst.condition_clbit, 2);
    EXPECT_NEAR(inst.params[0], kPi / 4.0, 1e-10);
}

TEST(BuildIterativeCircuit, N3_PIfForJ0_K1_AnglePiOver2) {
    // j=0, k=1: angle=π/2^1=π/2
    auto qc = QFT::build_iterative_circuit(3);
    const auto& inst = qc.instructions[6];
    EXPECT_EQ(inst.type, GT::P);
    EXPECT_EQ(inst.qubits[0], 0);
    EXPECT_EQ(inst.condition_clbit, 1);
    EXPECT_NEAR(inst.params[0], kPi / 2.0, 1e-10);
}

TEST(BuildIterativeCircuit, AllMeasurementsMapQubitToSameClbit) {
    // MEASURE(j, j) for all j.
    auto qc = QFT::build_iterative_circuit(4);
    for (const auto& inst : qc.instructions) {
        if (inst.type == GT::MEASURE) {
            EXPECT_EQ(inst.qubits[0], inst.clbits[0])
                << "MEASURE qubit " << inst.qubits[0]
                << " maps to clbit " << inst.clbits[0];
        }
    }
}

TEST(BuildIterativeCircuit, AllPIfClvalIsOne) {
    // Every p_if in the forward circuit has condition_value == 1.
    auto qc = QFT::build_iterative_circuit(4);
    for (const auto& inst : qc.instructions) {
        if (inst.condition_clbit >= 0) {
            EXPECT_EQ(inst.condition_value, 1);
        }
    }
}

TEST(BuildIterativeCircuit, InvalidNThrows) {
    EXPECT_THROW(QFT::build_iterative_circuit(0), std::exception);
    EXPECT_THROW(QFT::build_iterative_circuit(-1), std::exception);
}

// =============================================================================
// Suite 7 — BuildIterativeInverseCircuit: structure
// =============================================================================

TEST(BuildIterativeInverseCircuit, N1_QubitsAndClbits) {
    auto qc = QFT::build_iterative_inverse_circuit(1);
    EXPECT_EQ(qc.n_qubits, 1);
    EXPECT_EQ(qc.n_clbits, 1);
}

TEST(BuildIterativeInverseCircuit, N2_QubitsAndClbits) {
    auto qc = QFT::build_iterative_inverse_circuit(2);
    EXPECT_EQ(qc.n_qubits, 2);
    EXPECT_EQ(qc.n_clbits, 2);
}

TEST(BuildIterativeInverseCircuit, N1_InstructionCount) {
    EXPECT_EQ(QFT::build_iterative_inverse_circuit(1).instructions.size(), 2u);
}

TEST(BuildIterativeInverseCircuit, N2_InstructionCount) {
    // j=0: 2; j=1: 3 → 5 (same total as forward for n=2)
    EXPECT_EQ(QFT::build_iterative_inverse_circuit(2).instructions.size(), 5u);
}

TEST(BuildIterativeInverseCircuit, N3_InstructionCount) {
    EXPECT_EQ(QFT::build_iterative_inverse_circuit(3).instructions.size(), 9u);
}

TEST(BuildIterativeInverseCircuit, N1_SequenceHthenMeasure) {
    auto qc = QFT::build_iterative_inverse_circuit(1);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[1].type, GT::MEASURE);
}

TEST(BuildIterativeInverseCircuit, N2_FirstInstrHOnQubit0) {
    // j=0 processed first (ascending order, opposite of forward)
    auto qc = QFT::build_iterative_inverse_circuit(2);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[0].qubits[0], 0);
}

TEST(BuildIterativeInverseCircuit, N2_PIfHasNegativeAngle) {
    // j=1, k=0: angle = -π/2^(1-0) = -π/2
    auto qc = QFT::build_iterative_inverse_circuit(2);
    const auto& inst = qc.instructions[2];
    EXPECT_EQ(inst.type, GT::P);
    EXPECT_NEAR(inst.params[0], -kPi / 2.0, 1e-10);
}

TEST(BuildIterativeInverseCircuit, N2_PIfConditionClbitIsZero) {
    // j=1, k=0: condition on clbit 0
    auto qc = QFT::build_iterative_inverse_circuit(2);
    const auto& inst = qc.instructions[2];
    EXPECT_EQ(inst.condition_clbit, 0);
    EXPECT_EQ(inst.condition_value, 1);
}

TEST(BuildIterativeInverseCircuit, N3_PIfForJ2_K0_AngleNegPiOver4) {
    // j=2, k=0: angle = -π/2^(2-0) = -π/4
    auto qc = QFT::build_iterative_inverse_circuit(3);
    // Layout: [0]=H(0),[1]=M(0,0), [2]=p_if(j=1,k=0), [3]=H(1),[4]=M(1,1),
    //         [5]=p_if(j=2,k=0), [6]=p_if(j=2,k=1), [7]=H(2),[8]=M(2,2)
    const auto& inst = qc.instructions[5];
    EXPECT_EQ(inst.type, GT::P);
    EXPECT_EQ(inst.qubits[0], 2);
    EXPECT_EQ(inst.condition_clbit, 0);
    EXPECT_NEAR(inst.params[0], -kPi / 4.0, 1e-10);
}

TEST(BuildIterativeInverseCircuit, N3_PIfForJ2_K1_AngleNegPiOver2) {
    // j=2, k=1: angle = -π/2^(2-1) = -π/2
    auto qc = QFT::build_iterative_inverse_circuit(3);
    const auto& inst = qc.instructions[6];
    EXPECT_NEAR(inst.params[0], -kPi / 2.0, 1e-10);
}

TEST(BuildIterativeInverseCircuit, ForwardVsInverse_AngleSign) {
    // For n=2: forward[2].params[0] == +π/2, inverse[2].params[0] == -π/2.
    auto fwd = QFT::build_iterative_circuit(2);
    auto inv = QFT::build_iterative_inverse_circuit(2);
    EXPECT_NEAR(fwd.instructions[2].params[0],  kPi / 2.0, 1e-10);
    EXPECT_NEAR(inv.instructions[2].params[0], -kPi / 2.0, 1e-10);
}

TEST(BuildIterativeInverseCircuit, SameInstructionCountAsForward) {
    for (int n = 1; n <= 5; ++n) {
        EXPECT_EQ(QFT::build_iterative_circuit(n).instructions.size(),
                  QFT::build_iterative_inverse_circuit(n).instructions.size())
            << "Mismatch at n=" << n;
    }
}

TEST(BuildIterativeInverseCircuit, AllMeasurementsMapQubitToSameClbit) {
    auto qc = QFT::build_iterative_inverse_circuit(4);
    for (const auto& inst : qc.instructions) {
        if (inst.type == GT::MEASURE) {
            EXPECT_EQ(inst.qubits[0], inst.clbits[0]);
        }
    }
}

TEST(BuildIterativeInverseCircuit, InvalidNThrows) {
    EXPECT_THROW(QFT::build_iterative_inverse_circuit(0), std::exception);
    EXPECT_THROW(QFT::build_iterative_inverse_circuit(-1), std::exception);
}

// =============================================================================
// Suite 8 — RunIterativeCorrectness
// =============================================================================

TEST(RunIterativeCorrectness, ZeroShotsThrows_DefaultOverload) {
    QuantumCircuit input(1);
    EXPECT_THROW(QFT::run_iterative(input, 0), std::invalid_argument);
}

TEST(RunIterativeCorrectness, ZeroShotsThrows_BackendOverload) {
    QuantumCircuit input(1);
    auto backend = make_backend(backends::LocalBackend::SimType::STATEVECTOR);
    EXPECT_THROW(QFT::run_iterative(input, backend, 0), std::invalid_argument);
}

TEST(RunIterativeCorrectness, N1_PlusInput_AlwaysZero) {
    // |+⟩ = H|0⟩. Iterative QFT: H·H|0⟩ = |0⟩ → always measures 0.
    QuantumCircuit input(1);
    input.h(0);
    auto result = QFT::run_iterative(input, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("0"), 200);
}

TEST(RunIterativeCorrectness, N1_MinusInput_AlwaysOne) {
    // |−⟩ = H·X|0⟩. Iterative QFT: H·H·X|0⟩ = X|0⟩ = |1⟩ → always 1.
    QuantumCircuit input(1);
    input.x(0).h(0);
    auto result = QFT::run_iterative(input, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("1"), 200);
}

TEST(RunIterativeCorrectness, N1_DefaultAndBackendOverloadAgree) {
    // Both overloads on the same input must return identical counts.
    QuantumCircuit input(1);
    input.h(0);
    auto r1 = QFT::run_iterative(input, 100, 7);
    auto backend = make_backend(backends::LocalBackend::SimType::STATEVECTOR);
    auto r2 = QFT::run_iterative(input, backend, 100, 7);
    EXPECT_EQ(r1.backend_result.counts, r2.backend_result.counts);
}

TEST(RunIterativeCorrectness, ResultNQubitsCorrect) {
    QuantumCircuit input(3);
    auto result = QFT::run_iterative(input, 50, 1);
    EXPECT_EQ(result.n_qubits, 3);
}

TEST(RunIterativeCorrectness, CliffordCompatibleAlwaysFalse) {
    // run_iterative hardcodes clifford_compatible = false (conservative; n>2 uses non-Clifford P).
    for (int n = 1; n <= 4; ++n) {
        QuantumCircuit input(n);
        auto result = QFT::run_iterative(input, 20, 1);
        EXPECT_FALSE(result.clifford_compatible) << "Expected false for n=" << n;
    }
}

TEST(RunIterativeCorrectness, ResultSuccessTrue) {
    QuantumCircuit input(2);
    auto result = QFT::run_iterative(input, 100, 1);
    EXPECT_TRUE(result.backend_result.success);
}

TEST(RunIterativeCorrectness, TotalShotsPreserved) {
    QuantumCircuit input(2);
    auto result = QFT::run_iterative(input, 300, 42);
    EXPECT_EQ(total_shots(result.backend_result.counts), 300);
}

TEST(RunIterativeCorrectness, N2_UniformInput_AllFourBitstrings) {
    // |00⟩ input → QFT → uniform superposition over 4 outputs.
    QuantumCircuit input(2);
    auto result = QFT::run_iterative(input, 4000, 42);
    const auto& counts = result.backend_result.counts;
    ASSERT_EQ(counts.size(), 4u) << "Expected all 4 bitstrings";
    for (const auto& [bs, cnt] : counts) {
        EXPECT_GT(cnt, 700) << bs << " appeared only " << cnt << " times";
        EXPECT_LT(cnt, 1300) << bs << " appeared " << cnt << " times";
    }
}

TEST(RunIterativeCorrectness, N2_PlusPlusInput_AlwaysZero) {
    // |+⟩|+⟩ input: after H(qubit1) → |0⟩₁ (always), p_if NOT fired,
    // after H(qubit0) → |0⟩₀ (always). Output "00" every shot.
    QuantumCircuit input(2);
    input.h(0).h(1);
    auto result = QFT::run_iterative(input, 200, 42);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("00"), 200);
}

TEST(RunIterativeCorrectness, N2_FeedforwardPGateExercised) {
    // Input: qubit 0 = |+i⟩ = S·H|0⟩, qubit 1 = |−⟩ = H·X|0⟩.
    // Iterative QFT step j=1: H|−⟩=|1⟩ → MEASURE → c[1]=1 always.
    // p_if fires: P(π/2)|+i⟩=(|0⟩+i·i|1⟩)/√2=(|0⟩-|1⟩)/√2=|−⟩.
    // H|−⟩=|1⟩ → MEASURE → c[0]=1 always.
    // Bitstring: c[1]=1, c[0]=1 → "11" always.
    // If p_if were skipped, H|+i⟩ is 50/50 — the determinism proves feedforward fired.
    QuantumCircuit input(2);
    input.h(0).s(0);    // qubit 0 = |+i⟩
    input.x(1).h(1);    // qubit 1 = |−⟩

    auto result = QFT::run_iterative(input, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("11"), 200);
}

// --- Backend coverage ---

TEST(RunIterativeCorrectness, DM_N1_PlusInput_AlwaysZero) {
    QuantumCircuit input(1);
    input.h(0);
    auto backend = make_backend(backends::LocalBackend::SimType::DENSITY_MATRIX);
    auto result  = QFT::run_iterative(input, backend, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("0"), 200);
}

TEST(RunIterativeCorrectness, DM_N1_MinusInput_AlwaysOne) {
    QuantumCircuit input(1);
    input.x(0).h(0);
    auto backend = make_backend(backends::LocalBackend::SimType::DENSITY_MATRIX);
    auto result  = QFT::run_iterative(input, backend, 200, 42);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("1"), 200);
}

TEST(RunIterativeCorrectness, DM_N2_FeedforwardPGateExercised) {
    // Same |+i⟩|−⟩ input as the SV test — must give "11" deterministically on DM backend.
    QuantumCircuit input(2);
    input.h(0).s(0);
    input.x(1).h(1);
    auto backend = make_backend(backends::LocalBackend::SimType::DENSITY_MATRIX);
    auto result  = QFT::run_iterative(input, backend, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("11"), 200);
}

TEST(RunIterativeCorrectness, Clifford_N1_PlusInput_AlwaysZero) {
    // n=1 iterative QFT uses only H — Clifford. |+⟩ → always "0".
    QuantumCircuit input(1);
    input.h(0);
    auto backend = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto result  = QFT::run_iterative(input, backend, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("0"), 200);
}

TEST(RunIterativeCorrectness, Clifford_N2_PlusPlusInput_AlwaysZero) {
    // n=2 iterative QFT uses only P(π/2)=S — Clifford angle.
    // |+⟩|+⟩ input → always "00" (same analysis as SV test).
    QuantumCircuit input(2);
    input.h(0).h(1);
    auto backend = make_backend(backends::LocalBackend::SimType::CLIFFORD);
    auto result  = QFT::run_iterative(input, backend, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("00"), 200);
}

TEST(RunIterativeCorrectness, MPS_N1_PlusInput_AlwaysZero) {
    QuantumCircuit input(1);
    input.h(0);
    auto backend = make_backend(backends::LocalBackend::SimType::MPS);
    auto result  = QFT::run_iterative(input, backend, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("0"), 200);
}

TEST(RunIterativeCorrectness, MPS_N2_FeedforwardPGateExercised) {
    QuantumCircuit input(2);
    input.h(0).s(0);
    input.x(1).h(1);
    auto backend = make_backend(backends::LocalBackend::SimType::MPS);
    auto result  = QFT::run_iterative(input, backend, 200, 42);
    ASSERT_TRUE(result.backend_result.success);
    ASSERT_EQ(result.backend_result.counts.size(), 1u);
    EXPECT_EQ(result.backend_result.counts.at("11"), 200);
}

// =============================================================================
// Suite 9 — RunIterativeInverse: correctness
// =============================================================================

// Helper: run the composed (input_state + iterative_inverse_circuit) on given backend.
static backends::BackendResult run_iterative_inverse(
    const QuantumCircuit& input,
    backends::LocalBackend::SimType sim,
    int shots = 200, uint64_t seed = 42)
{
    int n = input.n_qubits;
    QuantumCircuit composed(n, n);
    for (const auto& inst : input.instructions)
        composed.instructions.push_back(inst);
    auto inv = QFT::build_iterative_inverse_circuit(n);
    for (const auto& inst : inv.instructions)
        composed.instructions.push_back(inst);
    auto backend = make_backend(sim);
    return backend.run(composed, shots, seed);
}

TEST(RunIterativeInverse, N1_PlusInput_AlwaysZero_SV) {
    // IQFT for n=1 is H+MEASURE (same as forward). |+⟩ → "0" always.
    QuantumCircuit input(1);
    input.h(0);
    auto result = run_iterative_inverse(input, backends::LocalBackend::SimType::STATEVECTOR);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("0"), 200);
}

TEST(RunIterativeInverse, N1_MinusInput_AlwaysOne_SV) {
    QuantumCircuit input(1);
    input.x(0).h(0);
    auto result = run_iterative_inverse(input, backends::LocalBackend::SimType::STATEVECTOR);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("1"), 200);
}

TEST(RunIterativeInverse, N2_InverseFeedforwardExercised_SV) {
    // Input: qubit 0 = |−⟩ = H·X|0⟩, qubit 1 = |+i⟩ = S·H|0⟩.
    // Inverse QFT j=0: H|−⟩=|1⟩ → MEASURE → c[0]=1 always → p_if fires.
    // P(-π/2)|+i⟩ = P(-π/2)(|0⟩+i|1⟩)/√2 = (|0⟩+i·e^{-iπ/2}|1⟩)/√2
    //             = (|0⟩+i·(-i)|1⟩)/√2 = (|0⟩+|1⟩)/√2 = |+⟩.
    // H|+⟩=|0⟩ → MEASURE → c[1]=0 always.
    // Bitstring: c[0]=1, c[1]=0 → bits[1]=c[0]=1, bits[0]=c[1]=0 → "01".
    QuantumCircuit input(2);
    input.x(0).h(0);    // qubit 0 = |−⟩
    input.h(1).s(1);    // qubit 1 = S·H|0⟩ = |+i⟩

    auto result = run_iterative_inverse(input, backends::LocalBackend::SimType::STATEVECTOR);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("01"), 200);
}

TEST(RunIterativeInverse, N2_InverseFeedforwardExercised_DM) {
    // Same input as SV test — must give "01" on DM backend.
    QuantumCircuit input(2);
    input.x(0).h(0);
    input.h(1).s(1);
    auto result = run_iterative_inverse(input, backends::LocalBackend::SimType::DENSITY_MATRIX);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("01"), 200);
}

TEST(RunIterativeInverse, N2_InverseFeedforwardExercised_MPS) {
    QuantumCircuit input(2);
    input.x(0).h(0);
    input.h(1).s(1);
    auto result = run_iterative_inverse(input, backends::LocalBackend::SimType::MPS);
    ASSERT_EQ(result.counts.size(), 1u);
    EXPECT_EQ(result.counts.at("01"), 200);
}

TEST(RunIterativeInverse, TotalShotsPreserved) {
    QuantumCircuit input(2);
    auto result = run_iterative_inverse(input, backends::LocalBackend::SimType::STATEVECTOR,
                                        150, 1);
    EXPECT_EQ(total_shots(result.counts), 150);
}
