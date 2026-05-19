// test_qasm_parser.cpp — Direct tests for QASM 2.0 parser

#include <gtest/gtest.h>
#include "lindblad/circuit.hpp"

#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace lindblad;

// =============================================================================
// Register declarations
// =============================================================================

TEST(QASM2ParserTest, SingleQreg) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg q[3];\n"
    );
    EXPECT_EQ(qc.n_qubits, 3);
    EXPECT_EQ(qc.instructions.size(), 0u);
}

TEST(QASM2ParserTest, QregAndCreg) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "creg c[2];\n"
    );
    EXPECT_EQ(qc.n_qubits, 2);
    EXPECT_EQ(qc.n_clbits, 2);
}

TEST(QASM2ParserTest, MultipleQregs) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg a[2];\n"
        "qreg b[3];\n"
    );
    EXPECT_EQ(qc.n_qubits, 5);
}

// =============================================================================
// Standard gates
// =============================================================================

TEST(QASM2ParserTest, HadamardGate) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "h q[0];\n"
    );
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, Instruction::GateType::H);
    EXPECT_EQ(qc.instructions[0].qubits[0], 0);
}

TEST(QASM2ParserTest, CNOTGate) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "cx q[0],q[1];\n"
    );
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, Instruction::GateType::CX);
    EXPECT_EQ(qc.instructions[0].qubits[0], 0);
    EXPECT_EQ(qc.instructions[0].qubits[1], 1);
}

TEST(QASM2ParserTest, ParameterizedGate) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "rz(1.57079632679) q[0];\n"
    );
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, Instruction::GateType::RZ);
    EXPECT_EQ(qc.instructions[0].qubits[0], 0);
    EXPECT_NEAR(qc.instructions[0].params[0], M_PI/2, 1e-5);
}

// =============================================================================
// Math expressions
// =============================================================================

TEST(QASM2ParserTest, MathExpressionPi) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "rx(pi/2) q[0];\n"
    );
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, Instruction::GateType::RX);
    EXPECT_NEAR(qc.instructions[0].params[0], M_PI/2, 1e-10);
}

TEST(QASM2ParserTest, MathExpressionNegative) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "ry(-pi) q[0];\n"
    );
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, Instruction::GateType::RY);
    EXPECT_NEAR(qc.instructions[0].params[0], -M_PI, 1e-10);
}

// =============================================================================
// Measurement
// =============================================================================

TEST(QASM2ParserTest, MeasureInstruction) {
    auto qc = QuantumCircuit::from_qasm2(
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "creg c[2];\n"
        "measure q[1] -> c[0];\n"
    );
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, Instruction::GateType::MEASURE);
    EXPECT_EQ(qc.instructions[0].qubits[0], 1);
    EXPECT_EQ(qc.instructions[0].clbits[0], 0);
}
