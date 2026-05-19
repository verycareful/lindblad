// test_dag.cpp — Direct tests for DAGCircuit

#include <gtest/gtest.h>
#include "lindblad/dag.hpp"
#include "lindblad/circuit.hpp"

#include <algorithm>
#include <unordered_map>

using namespace lindblad;

// =============================================================================
// from_circuit / to_circuit roundtrip
// =============================================================================

TEST(DAGCircuitTest, RoundtripPreservesGateTypes) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).rz(0.5, 1).cx(1, 2);
    auto dag = DAGCircuit::from_circuit(qc);
    auto qc2 = dag.to_circuit();
    ASSERT_EQ(qc2.instructions.size(), qc.instructions.size());
    for (size_t i = 0; i < qc.instructions.size(); ++i)
        EXPECT_EQ(qc2.instructions[i].type, qc.instructions[i].type)
            << "Gate type mismatch at index " << i;
}

TEST(DAGCircuitTest, RoundtripPreservesQubits) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 2).rz(1.0, 1);
    auto dag = DAGCircuit::from_circuit(qc);
    auto qc2 = dag.to_circuit();
    ASSERT_EQ(qc2.instructions.size(), qc.instructions.size());
    EXPECT_EQ(qc2.n_qubits, qc.n_qubits);
}

// =============================================================================
// topological_sort
// =============================================================================

TEST(DAGCircuitTest, TopologicalSortRespectsDependencies) {
    // H(0) → CX(0,1) → CX(1,2): every edge must point forward in the ordering
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);
    auto dag = DAGCircuit::from_circuit(qc);
    auto order = dag.topological_sort();
    EXPECT_EQ(order.size(), dag.nodes.size());

    std::unordered_map<int, size_t> pos;
    for (size_t i = 0; i < order.size(); ++i) pos[order[i]] = i;

    for (const auto& e : dag.edges)
        EXPECT_LT(pos[e.src_node], pos[e.dst_node])
            << "Edge (" << e.src_node << " → " << e.dst_node << ") violates order";
}

TEST(DAGCircuitTest, TopologicalSortContainsAllNodes) {
    QuantumCircuit qc(2);
    qc.h(0).h(1).cx(0, 1);
    auto dag = DAGCircuit::from_circuit(qc);
    auto order = dag.topological_sort();
    EXPECT_EQ(order.size(), dag.nodes.size());
}

// =============================================================================
// front_layer
// =============================================================================

TEST(DAGCircuitTest, FrontLayerParallelH) {
    // H on 3 independent qubits — all 3 OP nodes have no OP predecessors
    QuantumCircuit qc(3);
    qc.h(0).h(1).h(2);
    auto dag = DAGCircuit::from_circuit(qc);
    auto front = dag.front_layer();
    EXPECT_EQ(front.size(), 3u);
}

TEST(DAGCircuitTest, FrontLayerSerialChain) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    auto dag = DAGCircuit::from_circuit(qc);
    auto front = dag.front_layer();
    EXPECT_EQ(front.size(), 1u);
    // Confirm the single node is an OP (not IN/OUT)
    for (const auto& n : dag.nodes) {
        if (n.node_id == front[0]) {
            EXPECT_EQ(n.type, DAGNode::Type::OP);
            break;
        }
    }
}

// =============================================================================
// successors / predecessors
// =============================================================================

TEST(DAGCircuitTest, SuccessorsAndPredecessors) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    auto dag = DAGCircuit::from_circuit(qc);

    int h_id = -1, cx_id = -1;
    for (const auto& n : dag.nodes) {
        if (n.type != DAGNode::Type::OP) continue;
        if (n.op.type == Instruction::GateType::H)  h_id  = n.node_id;
        if (n.op.type == Instruction::GateType::CX) cx_id = n.node_id;
    }
    ASSERT_GE(h_id,  0) << "H gate node not found";
    ASSERT_GE(cx_id, 0) << "CX gate node not found";

    auto succ_h  = dag.successors(h_id);
    auto pred_cx = dag.predecessors(cx_id);

    EXPECT_NE(std::find(succ_h.begin(),  succ_h.end(),  cx_id), succ_h.end())
        << "CX should be a successor of H";
    EXPECT_NE(std::find(pred_cx.begin(), pred_cx.end(), h_id),  pred_cx.end())
        << "H should be a predecessor of CX";
}

// =============================================================================
// depth
// =============================================================================

TEST(DAGCircuitTest, DepthLinearChain) {
    // H → CX(0,1) → CX(1,2): depth 3 (each depends on the previous)
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_EQ(dag.depth(), 3);
}

TEST(DAGCircuitTest, DepthParallelGates) {
    // H(0), H(1), H(2) all independent: depth 1
    QuantumCircuit qc(3);
    qc.h(0).h(1).h(2);
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_EQ(dag.depth(), 1);
}

// =============================================================================
// two_qubit_ops
// =============================================================================

TEST(DAGCircuitTest, TwoQubitOpsCount) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2).rz(0.5, 0);
    auto dag = DAGCircuit::from_circuit(qc);
    auto ops = dag.two_qubit_ops();
    EXPECT_EQ(ops.size(), 2u);
}

TEST(DAGCircuitTest, TwoQubitOpsQubitPairs) {
    QuantumCircuit qc(3);
    qc.cx(0, 1).cx(2, 0);
    auto dag = DAGCircuit::from_circuit(qc);
    auto ops = dag.two_qubit_ops();
    ASSERT_EQ(ops.size(), 2u);

    bool found_01 = false, found_20 = false;
    for (const auto& [q0, q1] : ops) {
        if ((q0 == 0 && q1 == 1) || (q0 == 1 && q1 == 0)) found_01 = true;
        if ((q0 == 2 && q1 == 0) || (q0 == 0 && q1 == 2)) found_20 = true;
    }
    EXPECT_TRUE(found_01);
    EXPECT_TRUE(found_20);
}

// =============================================================================
// num_op_nodes
// =============================================================================

TEST(DAGCircuitTest, NumOpNodes) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.3, 1);
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_EQ(dag.num_op_nodes(), 3);
}
