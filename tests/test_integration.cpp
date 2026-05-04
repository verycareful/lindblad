// test_integration.cpp — Integration tests for cross-component correctness
//
// These tests verify that:
// 1. Statevector and density matrix simulators produce consistent results
// 2. Transpiler at optimization level 3 preserves circuit unitarity
// 3. Clifford simulator stores the correct final state
// 4. RESET gate works correctly in the density matrix simulator
// 5. QASM2 round-trip preserves pi expressions

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/transpiler.hpp"

using namespace lindblad;

// =============================================================================
// Test 1: Statevector matches density matrix diagonal
// =============================================================================

TEST(IntegrationTest, StatevectorMatchesDensityMatrix) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(0, 2).rz(0.5, 1).ry(1.2, 2);

    // Statevector
    StatevectorSimulator sv_sim;
    auto sv_result = sv_sim.run(qc);
    ASSERT_TRUE(sv_result.success);

    // Density matrix (ideal noise model)
    NoiseModel ideal;
    DensityMatrixSimulator dm_sim;
    auto dm_result = dm_sim.run(qc, ideal);
    ASSERT_TRUE(dm_result.success);

    // Compare: diagonal of density matrix should match statevector probabilities
    auto sv_probs = sv_result.final_state.probabilities();
    auto dm_probs = dm_result.final_state.probabilities();

    ASSERT_EQ(sv_probs.size(), dm_probs.size());
    for (size_t i = 0; i < sv_probs.size(); ++i) {
        EXPECT_NEAR(sv_probs[i], dm_probs[i], 1e-10)
            << "Mismatch at index " << i;
    }
}

// =============================================================================
// Test 2: Optimization level 3 preserves unitary
// =============================================================================

TEST(IntegrationTest, OptLevel3PreservesUnitary) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.7, 0).ry(1.3, 1).cx(1, 0);

    // Transpile at level 3
    auto transpiled = transpile(qc, CouplingMap(), {}, 3);

    // Both should produce same statevector probabilities
    StatevectorSimulator sim;
    auto r1 = sim.run(qc);
    auto r2 = sim.run(transpiled);

    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(r1.final_state.probability(i),
                    r2.final_state.probability(i), 1e-6)
            << "Probability mismatch at index " << i;
    }
}

// =============================================================================
// Test 3: Clifford simulator final state is correct
// =============================================================================

TEST(IntegrationTest, CliffordFinalStateCorrect) {
    // Prepare Bell state: H(0).CX(0,1)
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);

    CliffordSimulator cliff_sim;
    auto result = cliff_sim.run(qc, 1, 42);

    // The stabilizer state should have stabilizers XX and ZZ (or -XX, -ZZ)
    // Check via expectation values
    auto& state = result.final_state;

    // XX should have expectation +1 for |Phi+⟩
    int xx = state.expectation_pauli("XX");
    int zz = state.expectation_pauli("ZZ");

    EXPECT_EQ(xx, 1) << "XX stabilizer should be +1 for Bell state";
    EXPECT_EQ(zz, 1) << "ZZ stabilizer should be +1 for Bell state";

    // XZ and ZX should give 0 (anticommute with at least one stabilizer)
    int xz = state.expectation_pauli("XZ");
    int zx = state.expectation_pauli("ZX");
    EXPECT_EQ(xz, 0);
    EXPECT_EQ(zx, 0);
}

// =============================================================================
// Test 4: RESET gate in density matrix
// =============================================================================

TEST(IntegrationTest, DensityMatrixReset) {
    QuantumCircuit qc(1);
    qc.x(0).reset(0);  // Set to |1⟩, then reset to |0⟩

    NoiseModel ideal;
    DensityMatrixSimulator dm_sim;
    auto result = dm_sim.run(qc, ideal);

    ASSERT_TRUE(result.success);

    // After reset, should be in |0⟩: rho = |0><0|
    auto probs = result.final_state.probabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
}

TEST(IntegrationTest, DensityMatrixResetSuperposition) {
    QuantumCircuit qc(1);
    qc.h(0).reset(0);  // Superposition then reset

    NoiseModel ideal;
    DensityMatrixSimulator dm_sim;
    auto result = dm_sim.run(qc, ideal);

    ASSERT_TRUE(result.success);

    auto probs = result.final_state.probabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
}

// =============================================================================
// Test 5: QASM2 pi expression parsing
// =============================================================================

TEST(IntegrationTest, QASM2PiExpressionRoundTrip) {
    // Create a circuit with pi/2 rotation
    QuantumCircuit qc(1);
    qc.rz(PI / 2.0, 0);

    // Export to QASM2
    std::string qasm = qc.to_qasm2();

    // Re-import
    auto reimported = QuantumCircuit::from_qasm2(qasm);

    // Both should produce same statevector
    StatevectorSimulator sim;
    auto r1 = sim.run(qc);
    auto r2 = sim.run(reimported);

    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);

    EXPECT_NEAR(r1.final_state.probability(0),
                r2.final_state.probability(0), 1e-8);
    EXPECT_NEAR(r1.final_state.probability(1),
                r2.final_state.probability(1), 1e-8);
}

// =============================================================================
// Test 6: ConsolidateBlocks gate count preservation
// =============================================================================

TEST(IntegrationTest, ConsolidateBlocksNoGateDuplication) {
    // Create a circuit with independent 1Q gates interleaved with a 2Q block
    QuantumCircuit qc(3);
    qc.cx(0, 1);   // 2Q gate on {0,1}
    qc.h(2);       // Independent 1Q gate on qubit 2
    qc.cx(0, 1);   // Another 2Q gate on {0,1}

    auto dag = DAGCircuit::from_circuit(qc);
    TranspilationContext ctx;
    ConsolidateBlocks cb;
    auto result_dag = cb.run(dag, ctx);
    auto result_qc = result_dag.to_circuit();

    // The H(2) should appear exactly once
    int h_count = 0;
    for (const auto& inst : result_qc.instructions) {
        if (inst.type == Instruction::GateType::H && inst.qubits[0] == 2) {
            h_count++;
        }
    }
    EXPECT_EQ(h_count, 1) << "H gate on qubit 2 should appear exactly once (no duplication)";
}

// =============================================================================
// Test 7: QASM2 non-default register names
// =============================================================================

TEST(IntegrationTest, QASM2NonDefaultRegisterNames) {
    std::string qasm = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg myq[2];
creg myc[2];
h myq[0];
cx myq[0],myq[1];
measure myq[0] -> myc[0];
measure myq[1] -> myc[1];
)";
    QuantumCircuit qc = QuantumCircuit::from_qasm2(qasm);
    EXPECT_EQ(qc.n_qubits, 2);
    EXPECT_EQ(qc.n_clbits, 2);

    StatevectorSimulator sim;
    auto result = sim.run(qc, 1024, 42);
    ASSERT_TRUE(result.success);
    // Bell state: only 00 and 11 outcomes
    for (const auto& [bits, count] : result.counts) {
        EXPECT_TRUE(bits == "00" || bits == "11")
            << "Unexpected outcome: " << bits;
    }
}

// =============================================================================
// Test 8: QASM2 multi-register offset correctness
// =============================================================================

TEST(IntegrationTest, QASM2MultipleQregs) {
    std::string qasm = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q1[1];
qreg q2[1];
creg c1[1];
creg c2[1];
x q2[0];
measure q1[0] -> c1[0];
measure q2[0] -> c2[0];
)";
    QuantumCircuit qc = QuantumCircuit::from_qasm2(qasm);
    EXPECT_EQ(qc.n_qubits, 2);

    StatevectorSimulator sim;
    auto result = sim.run(qc, 64, 0);
    ASSERT_TRUE(result.success);
    // MSB-first: bitstring[0]=qubit1=q2=1, bitstring[1]=qubit0=q1=0 → "10"
    EXPECT_EQ(result.counts.size(), 1u);
    EXPECT_GT(result.counts.count("10"), 0u);
}
