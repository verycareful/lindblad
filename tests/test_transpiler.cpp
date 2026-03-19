#include <gtest/gtest.h>
#include "qpp/circuit.hpp"
#include "qpp/transpiler.hpp"

using namespace qpp;

TEST(TranspilerTest, CouplingMapLinear) {
    auto cm = CouplingMap::linear(5);
    EXPECT_EQ(cm.n_physical_qubits, 5);
    EXPECT_TRUE(cm.is_connected(0, 1));
    EXPECT_TRUE(cm.is_connected(3, 4));
    EXPECT_FALSE(cm.is_connected(0, 4));
}

TEST(TranspilerTest, ShortestPath) {
    auto cm = CouplingMap::linear(5);
    auto path = cm.shortest_path(0, 4);
    EXPECT_EQ(path.size(), 5u);
    EXPECT_EQ(path[0], 0);
    EXPECT_EQ(path[4], 4);
}

TEST(TranspilerTest, DistanceMatrix) {
    auto cm = CouplingMap::linear(4);
    auto dist = cm.distance_matrix();
    EXPECT_EQ(dist[0][3], 3);
    EXPECT_EQ(dist[1][2], 1);
}

TEST(TranspilerTest, TranspileBasic) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(0, 2);

    auto cm = CouplingMap::linear(3);
    auto transpiled = transpile(qc, cm, {}, 1);
    EXPECT_GE(transpiled.size(), qc.size());
}

TEST(TranspilerTest, OptimizeCXCancellation) {
    QuantumCircuit qc(2);
    qc.cx(0, 1).cx(0, 1);  // Should cancel

    auto transpiled = transpile(qc, CouplingMap(), {}, 1);
    EXPECT_EQ(transpiled.size(), 0);
}
