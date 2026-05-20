#include <gtest/gtest.h>
#include "lindblad/circuit.hpp"

using namespace lindblad;

TEST(CircuitTest, BasicConstruction) {
    QuantumCircuit qc(3, 3);
    EXPECT_EQ(qc.n_qubits, 3);
    EXPECT_EQ(qc.n_clbits, 3);
    EXPECT_EQ(qc.size(), 0);
}

TEST(CircuitTest, FluentAPI) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).measure_all();
    EXPECT_EQ(qc.size(), 4);  // h + cx + 2 measures
}

TEST(CircuitTest, Depth) {
    QuantumCircuit qc(3);
    qc.h(0).h(1).h(2);  // depth 1 (all parallel)
    qc.cx(0, 1);          // depth 2
    qc.cx(1, 2);          // depth 3
    EXPECT_EQ(qc.depth(), 3);
}

TEST(CircuitTest, CountOps) {
    QuantumCircuit qc(2);
    qc.h(0).h(1).cx(0, 1).h(0);
    auto counts = qc.count_ops();
    EXPECT_EQ(counts["h"], 3);
    EXPECT_EQ(counts["cx"], 1);
}

TEST(CircuitTest, QASM2Export) {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    std::string qasm = qc.to_qasm2();
    EXPECT_NE(qasm.find("OPENQASM 2.0"), std::string::npos);
    EXPECT_NE(qasm.find("qreg q[2]"), std::string::npos);
    EXPECT_NE(qasm.find("h q[0]"), std::string::npos);
    EXPECT_NE(qasm.find("cx q[0], q[1]"), std::string::npos);
}

TEST(CircuitTest, Inverse) {
    QuantumCircuit qc(1);
    qc.s(0).t(0);
    auto inv = qc.inverse();
    EXPECT_EQ(inv.size(), 2);
    auto ops = inv.count_ops();
    EXPECT_EQ(ops["tdg"], 1);
    EXPECT_EQ(ops["sdg"], 1);
}

TEST(CircuitTest, Compose) {
    QuantumCircuit qc1(2);
    qc1.h(0);
    QuantumCircuit qc2(2);
    qc2.cx(0, 1);
    auto composed = qc1.compose(qc2);
    EXPECT_EQ(composed.size(), 2);
}

TEST(CircuitTest, ParameterBinding) {
    QuantumCircuit qc(1);
    qc.rx("theta", 0);
    EXPECT_EQ(qc.num_parameters(), 1);

    auto bound = qc.assign_parameters({{"theta", 1.57}});
    EXPECT_EQ(bound.num_parameters(), 0);
    EXPECT_EQ(bound.size(), 1);
}

TEST(CircuitTest, ASCIIDiagram) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    std::string ascii = qc.draw();
    EXPECT_FALSE(ascii.empty());
}

TEST(CircuitTest, DrawSmoke) {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    EXPECT_FALSE(qc.draw(DrawMode::ASCII).empty());
    EXPECT_FALSE(qc.draw(DrawMode::SVG  ).empty());
    EXPECT_FALSE(qc.draw(DrawMode::LATEX).empty());
    EXPECT_FALSE(qc.draw(DrawMode::HTML ).empty());
}

TEST(CircuitTest, InvalidQubit) {
    QuantumCircuit qc(2);
    EXPECT_THROW(qc.h(5), std::out_of_range);
    EXPECT_THROW(qc.cx(0, 5), std::out_of_range);
}
