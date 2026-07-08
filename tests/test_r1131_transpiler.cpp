// R.1.13.1 test patch — transpiler performance refactors.
// Covers audit F-25 (CouplingMap::distance_matrix via BFS-per-node) and
// F-15/F-16 (SabreLayout adjacency lists + SabreSwap physical-neighbour lists).
// The refactors changed internals only: distances must stay correct and routed
// circuits must remain hardware-legal (every 2-qubit gate on a coupled edge).

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/transpiler.hpp"

#include <cstdlib>
#include <vector>

using namespace lindblad;

namespace {

// Every 2-qubit gate (including inserted SWAPs) must act on a coupled pair.
void expect_hardware_legal(const QuantumCircuit& qc, const CouplingMap& cm) {
    for (const auto& inst : qc.instructions) {
        using GT = Instruction::GateType;
        if (inst.type == GT::BARRIER || inst.type == GT::MEASURE ||
            inst.type == GT::RESET)
            continue;
        if (inst.qubits.size() == 2) {
            EXPECT_TRUE(cm.is_connected(inst.qubits[0], inst.qubits[1]))
                << "2-qubit gate on uncoupled pair (" << inst.qubits[0] << ","
                << inst.qubits[1] << ")";
        }
    }
}

} // namespace

// F-25: BFS distance matrix matches known linear-chain distances and is a valid
// metric (zero diagonal, symmetric).
TEST(R1131Transpiler, DistanceMatrixLinearKnownValues) {
    auto cm = CouplingMap::linear(5);
    auto d = cm.distance_matrix();
    ASSERT_EQ(d.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(d[i][i], 0);
        for (int j = 0; j < 5; ++j) {
            EXPECT_EQ(d[i][j], std::abs(i - j));   // chain distance
            EXPECT_EQ(d[i][j], d[j][i]);           // symmetric
        }
    }
}

// F-25: all-to-all is distance 1 off-diagonal.
TEST(R1131Transpiler, DistanceMatrixAllToAll) {
    auto cm = CouplingMap::all_to_all(6);
    auto d = cm.distance_matrix();
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            EXPECT_EQ(d[i][j], (i == j) ? 0 : 1);
}

// F-25: the BFS matrix agrees with shortest_path length on non-trivial
// topologies (independent traversal cross-check).
TEST(R1131Transpiler, DistanceMatrixAgreesWithShortestPath) {
    for (const CouplingMap& cm :
         {CouplingMap::linear(6), CouplingMap::grid(3, 3),
          CouplingMap::all_to_all(5)}) {
        auto d = cm.distance_matrix();
        const int n = cm.n_physical_qubits;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                auto path = cm.shortest_path(i, j);
                ASSERT_FALSE(path.empty()) << "no path " << i << "->" << j;
                EXPECT_EQ(d[i][j], static_cast<int>(path.size()) - 1)
                    << "distance/path mismatch " << i << "->" << j;
            }
    }
}

// F-15/F-16: routing a non-adjacent interaction onto a line yields a
// hardware-legal circuit (SWAP-routed), and adds instructions.
TEST(R1131Transpiler, SabreRoutesLinearChainLegally) {
    QuantumCircuit qc(5);
    qc.h(0).cx(0, 4).cx(1, 3).cx(0, 2);   // several non-adjacent interactions

    auto cm = CouplingMap::linear(5);
    auto routed = transpile(qc, cm, {}, 1);

    expect_hardware_legal(routed, cm);
    EXPECT_GE(routed.size(), qc.size());  // routing inserts SWAPs
}

// F-15/F-16: routing onto a 2D grid is also legal.
TEST(R1131Transpiler, SabreRoutesGridLegally) {
    QuantumCircuit qc(6);
    qc.h(0).cx(0, 5).cx(1, 4).cx(2, 3);

    auto cm = CouplingMap::grid(2, 3);
    auto routed = transpile(qc, cm, {}, 1);
    expect_hardware_legal(routed, cm);
}

// A circuit already respecting the coupling map stays legal after transpile.
TEST(R1131Transpiler, AlreadyLegalCircuitStaysLegal) {
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 1).cx(1, 2).cx(2, 3);   // all nearest-neighbour on a line
    auto cm = CouplingMap::linear(4);
    auto routed = transpile(qc, cm, {}, 1);
    expect_hardware_legal(routed, cm);
}
