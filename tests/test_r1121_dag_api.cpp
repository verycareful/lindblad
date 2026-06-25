// R.1.12.1 total-coverage suite, Batch 1: lindblad/dag.hpp (DAGCircuit).
// Plan: docs (R.1.12.1 coverage plan), section "Batch 1: foundations".
//
// Covers DAGCircuit end to end: from_circuit node/edge wiring, to_circuit
// determinism and dependency preservation, classical read/write edges,
// successors/predecessors deduplication, front_layer, depth (vs
// QuantumCircuit::depth), two_qubit_ops, num_op_nodes, and the
// substitute_node / remove_node mutators with full-matrix circuit
// equivalence. Test-only release content (.1 slot).

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/operators.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

// node_id of the `which`-th (0-based) OP node whose gate_name() == gate.
int find_op(const DAGCircuit& dag, const std::string& gate, int which = 0) {
    int seen = 0;
    for (const auto& n : dag.nodes) {
        if (n.type == DAGNode::Type::OP && n.op.gate_name() == gate) {
            if (seen++ == which) return n.node_id;
        }
    }
    return -1;
}

int count_type(const DAGCircuit& dag, DAGNode::Type t) {
    int c = 0;
    for (const auto& n : dag.nodes) if (n.type == t) ++c;
    return c;
}

bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

// Full 2^n x 2^n matrix equivalence of two unitary (measurement-free) circuits.
void expect_equivalent(const QuantumCircuit& a, const QuantumCircuit& b,
                       double tol = 1e-9) {
    ASSERT_EQ(a.n_qubits, b.n_qubits);
    auto ma = Operator::from_circuit(a).data;
    auto mb = Operator::from_circuit(b).data;
    ASSERT_EQ(ma.size(), mb.size());
    for (size_t i = 0; i < ma.size(); ++i) {
        EXPECT_NEAR(ma[i].real, mb[i].real, tol) << "entry " << i;
        EXPECT_NEAR(ma[i].imag, mb[i].imag, tol) << "entry " << i;
    }
}

// Gate-name sequence of a circuit's instructions, in order.
std::vector<std::string> op_names(const QuantumCircuit& qc) {
    std::vector<std::string> out;
    for (const auto& inst : qc.instructions) out.push_back(inst.gate_name());
    return out;
}

}  // namespace

// =============================================================================
// from_circuit — node/edge counts and IN/OUT wiring
// =============================================================================

TEST(R1121Dag, FromCircuitNodeAndEdgeCounts) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    auto dag = DAGCircuit::from_circuit(qc);

    // 2 IN (qubits) + 2 OP + 2 OUT (qubits); no clbits.
    EXPECT_EQ(dag.nodes.size(), 6u);
    EXPECT_EQ(count_type(dag, DAGNode::Type::IN), 2);
    EXPECT_EQ(count_type(dag, DAGNode::Type::OUT), 2);
    EXPECT_EQ(dag.num_op_nodes(), 2);

    // Edges: IN_q0->h, h->cx, IN_q1->cx, cx->OUT_q0, cx->OUT_q1.
    EXPECT_EQ(dag.edges.size(), 5u);
    EXPECT_EQ(dag.n_qubits, 2);
    EXPECT_EQ(dag.n_clbits, 0);
}

TEST(R1121Dag, FromCircuitCreatesClbitInOutNodes) {
    QuantumCircuit qc(1, 1);
    qc.h(0).measure(0, 0);
    auto dag = DAGCircuit::from_circuit(qc);
    // 1 IN_q + 1 IN_c + 2 OP + 1 OUT_q + 1 OUT_c = 6
    EXPECT_EQ(dag.nodes.size(), 6u);
    EXPECT_EQ(count_type(dag, DAGNode::Type::IN), 2);
    EXPECT_EQ(count_type(dag, DAGNode::Type::OUT), 2);
    EXPECT_EQ(dag.num_op_nodes(), 2);
}

// =============================================================================
// to_circuit — determinism and dependency preservation
// =============================================================================

TEST(R1121Dag, ToCircuitPreservesLinearChain) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.3, 1).x(0);
    auto dag = DAGCircuit::from_circuit(qc);
    auto rebuilt = dag.to_circuit();
    EXPECT_EQ(rebuilt.n_qubits, 2);
    EXPECT_EQ(op_names(rebuilt), op_names(qc));
    expect_equivalent(rebuilt, qc);
}

TEST(R1121Dag, ToCircuitIsDeterministicAcrossRepeats) {
    QuantumCircuit qc(3);
    qc.h(0).h(1).h(2).cx(0, 1).cx(1, 2).rz(0.7, 0).cz(0, 2);
    auto dag = DAGCircuit::from_circuit(qc);
    auto reference = op_names(dag.to_circuit());
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(op_names(dag.to_circuit()), reference)
            << "to_circuit must be deterministic on repeat " << i;
    }
}

// =============================================================================
// Classical dependency edges (read-after-write, write-after-read)
// =============================================================================

TEST(R1121Dag, ConditionalGateDependsOnMeasureWriter) {
    // measure writes c0; the conditioned X reads c0 -> read-after-write edge,
    // even though they share no qubit wire.
    QuantumCircuit qc(2, 1);
    qc.measure(0, 0);
    qc.add_if(0, 1, GT::X, {1});
    auto dag = DAGCircuit::from_circuit(qc);

    int measure_id = find_op(dag, "measure");
    int x_id = find_op(dag, "x");
    ASSERT_GE(measure_id, 0);
    ASSERT_GE(x_id, 0);
    EXPECT_TRUE(contains(dag.predecessors(x_id), measure_id))
        << "conditioned gate must depend on the writer of its condition bit";
}

TEST(R1121Dag, RewriteOfReadClbitDependsOnReader) {
    // measure(0)->c0; conditioned X reads c0; measure(2)->c0 rewrites c0.
    // The rewrite must come after the reader (write-after-read edge), with no
    // shared qubit wire forcing the order.
    QuantumCircuit qc(3, 1);
    qc.measure(0, 0);
    qc.add_if(0, 1, GT::X, {1});
    qc.measure(2, 0);
    auto dag = DAGCircuit::from_circuit(qc);

    int x_id = find_op(dag, "x");
    int remeasure_id = find_op(dag, "measure", 1);  // second measure
    ASSERT_GE(x_id, 0);
    ASSERT_GE(remeasure_id, 0);
    EXPECT_TRUE(contains(dag.predecessors(remeasure_id), x_id))
        << "rewriting a read clbit must depend on the prior reader";
}

TEST(R1121Dag, FeedforwardChainOrdersDeterministically) {
    // measure(0)->c0, X(1) if c0, measure(1)->c1, Z(0) if c1.
    QuantumCircuit qc(2, 2);
    qc.measure(0, 0);
    qc.add_if(0, 1, GT::X, {1});
    qc.measure(1, 1);
    qc.add_if(1, 1, GT::Z, {0});
    auto dag = DAGCircuit::from_circuit(qc);
    auto names = op_names(dag.to_circuit());
    ASSERT_EQ(names.size(), 4u);
    // measure(q0) before X(q1) before measure(q1) before Z(q0).
    auto pos = [&](const std::string& g, int which) {
        int seen = 0;
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == g && seen++ == which) return static_cast<int>(i);
        return -1;
    };
    EXPECT_LT(pos("measure", 0), pos("x", 0));
    EXPECT_LT(pos("x", 0), pos("measure", 1));
    EXPECT_LT(pos("measure", 1), pos("z", 0));
}

// =============================================================================
// successors / predecessors — deduplication across shared wires
// =============================================================================

TEST(R1121Dag, SuccessorsAndPredecessorsDeduplicateMultiWire) {
    // cx then cz on the same pair: the two gates share BOTH wires, so there is
    // one edge per wire but successors/predecessors must report a single node.
    QuantumCircuit qc(2);
    qc.cx(0, 1).cz(0, 1);
    auto dag = DAGCircuit::from_circuit(qc);
    int cx_id = find_op(dag, "cx");
    int cz_id = find_op(dag, "cz");
    ASSERT_GE(cx_id, 0);
    ASSERT_GE(cz_id, 0);

    auto succ = dag.successors(cx_id);
    EXPECT_EQ(succ.size(), 1u) << "deduplicated across the two shared wires";
    EXPECT_TRUE(contains(succ, cz_id));

    auto pred = dag.predecessors(cz_id);
    EXPECT_EQ(pred.size(), 1u);
    EXPECT_TRUE(contains(pred, cx_id));
}

TEST(R1121Dag, SuccessorsOfMissingNodeIsEmpty) {
    QuantumCircuit qc(1);
    qc.h(0);
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_TRUE(dag.successors(99999).empty());
    EXPECT_TRUE(dag.predecessors(99999).empty());
}

// =============================================================================
// front_layer
// =============================================================================

TEST(R1121Dag, FrontLayerIsOpsWithNoOpPredecessors) {
    QuantumCircuit qc(2);
    qc.h(0).x(1).cx(0, 1);
    auto dag = DAGCircuit::from_circuit(qc);
    auto front = dag.front_layer();
    ASSERT_EQ(front.size(), 2u);
    EXPECT_TRUE(contains(front, find_op(dag, "h")));
    EXPECT_TRUE(contains(front, find_op(dag, "x")));
    EXPECT_FALSE(contains(front, find_op(dag, "cx")))
        << "cx has OP predecessors and cannot be in the front layer";
}

// =============================================================================
// depth (vs QuantumCircuit::depth on a barrier-free circuit)
// =============================================================================

TEST(R1121Dag, DepthMatchesCircuitDepth) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);  // longest OP chain length 3
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_EQ(dag.depth(), 3);
    EXPECT_EQ(dag.depth(), qc.depth());
}

TEST(R1121Dag, DepthOfParallelLayerIsOne) {
    QuantumCircuit qc(3);
    qc.h(0).h(1).h(2);  // all parallel
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_EQ(dag.depth(), 1);
}

// =============================================================================
// two_qubit_ops / num_op_nodes
// =============================================================================

TEST(R1121Dag, TwoQubitOpsListsOnlyTwoQubitGates) {
    QuantumCircuit qc(3);
    qc.cx(0, 1).ccx(0, 1, 2).cz(1, 2).h(0);
    auto dag = DAGCircuit::from_circuit(qc);
    auto pairs = dag.two_qubit_ops();
    ASSERT_EQ(pairs.size(), 2u);  // cx and cz only; ccx is 3-qubit, h is 1-qubit
    EXPECT_EQ(pairs[0], std::make_pair(0, 1));
    EXPECT_EQ(pairs[1], std::make_pair(1, 2));
    EXPECT_EQ(dag.num_op_nodes(), 4);
}

// =============================================================================
// substitute_node — 1q-for-1q, 1q-for-many, 2q, empty; equivalence preserved
// =============================================================================

TEST(R1121Dag, SubstituteNodeMissingIdThrows) {
    QuantumCircuit qc(1);
    qc.h(0);
    auto dag = DAGCircuit::from_circuit(qc);
    DAGCircuit empty(1, 0);
    EXPECT_THROW(dag.substitute_node(123456, empty), std::out_of_range);
}

TEST(R1121Dag, SubstituteSingleForSingle) {
    QuantumCircuit qc(1);
    qc.h(0);
    auto dag = DAGCircuit::from_circuit(qc);

    QuantumCircuit repl_circ(1);
    repl_circ.rz(0.5, 0);
    auto repl = DAGCircuit::from_circuit(repl_circ);

    dag.substitute_node(find_op(dag, "h"), repl);
    auto out = dag.to_circuit();
    EXPECT_EQ(op_names(out), (std::vector<std::string>{"rz"}));
    expect_equivalent(out, repl_circ);
}

TEST(R1121Dag, SubstituteSingleForManyOnSameWire) {
    QuantumCircuit qc(1);
    qc.x(0);
    auto dag = DAGCircuit::from_circuit(qc);

    QuantumCircuit repl_circ(1);
    repl_circ.rz(0.3, 0).rx(0.4, 0).rz(-0.2, 0);
    auto repl = DAGCircuit::from_circuit(repl_circ);

    dag.substitute_node(find_op(dag, "x"), repl);
    auto out = dag.to_circuit();
    EXPECT_EQ(op_names(out), (std::vector<std::string>{"rz", "rx", "rz"}));
    expect_equivalent(out, repl_circ);
}

TEST(R1121Dag, SubstituteTwoQubitSpanningBothWires) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    auto dag = DAGCircuit::from_circuit(qc);

    QuantumCircuit repl_circ(2);
    repl_circ.swap(0, 1);
    auto repl = DAGCircuit::from_circuit(repl_circ);

    dag.substitute_node(find_op(dag, "cx"), repl);
    auto out = dag.to_circuit();
    EXPECT_EQ(op_names(out), (std::vector<std::string>{"swap"}));
    expect_equivalent(out, repl_circ);
}

TEST(R1121Dag, SubstituteWithEmptyReplacementReconnects) {
    QuantumCircuit qc(1);
    qc.h(0).x(0);
    auto dag = DAGCircuit::from_circuit(qc);

    DAGCircuit empty(1, 0);
    dag.substitute_node(find_op(dag, "x"), empty);
    auto out = dag.to_circuit();
    EXPECT_EQ(op_names(out), (std::vector<std::string>{"h"}));

    QuantumCircuit expected(1);
    expected.h(0);
    expect_equivalent(out, expected);
}

TEST(R1121Dag, SubstituteInteriorNodePreservesNeighbours) {
    // h ; [x -> y] ; z : substituting the middle node keeps the chain order.
    QuantumCircuit qc(1);
    qc.h(0).x(0).z(0);
    auto dag = DAGCircuit::from_circuit(qc);

    QuantumCircuit repl_circ(1);
    repl_circ.y(0);
    auto repl = DAGCircuit::from_circuit(repl_circ);

    dag.substitute_node(find_op(dag, "x"), repl);
    auto out = dag.to_circuit();
    EXPECT_EQ(op_names(out), (std::vector<std::string>{"h", "y", "z"}));

    QuantumCircuit expected(1);
    expected.h(0).y(0).z(0);
    expect_equivalent(out, expected);
}

// =============================================================================
// remove_node — reconnect predecessor to successor per wire
// =============================================================================

TEST(R1121Dag, RemoveInteriorNodeReconnectsWire) {
    QuantumCircuit qc(1);
    qc.h(0).x(0).z(0);
    auto dag = DAGCircuit::from_circuit(qc);

    dag.remove_node(find_op(dag, "x"));
    auto out = dag.to_circuit();
    EXPECT_EQ(op_names(out), (std::vector<std::string>{"h", "z"}));

    QuantumCircuit expected(1);
    expected.h(0).z(0);
    expect_equivalent(out, expected);
}

TEST(R1121Dag, RemoveTwoQubitNodeReconnectsBothWires) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).x(1);
    auto dag = DAGCircuit::from_circuit(qc);

    dag.remove_node(find_op(dag, "cx"));
    auto out = dag.to_circuit();
    // cx gone; h on q0 and x on q1 remain, both wires reconnected to OUT.
    EXPECT_EQ(dag.num_op_nodes(), 2);
    EXPECT_EQ(op_names(out), (std::vector<std::string>{"h", "x"}));

    QuantumCircuit expected(2);
    expected.h(0).x(1);
    expect_equivalent(out, expected);
}
