// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.1 test wave - one definition of a scheduling layer, and resolving a
// DAG node id.
//
// 1.1.29.0 exposed the ASAP timing rule as asap_schedule_times(), made
// ASAPSchedule write it and made Anchor::every_layer() read its boundaries
// from it, so "layer" means one thing in the library. The rule is pinned here
// on hand-worked circuits, the pass is checked to stamp exactly what the
// function returns, and every_layer is checked to fire where the executed
// prefix of the circuit completes some layer and nowhere else: never inside an
// interleaved layer, always after the last instruction.
//
// The timing rule tracks QUBIT wires only. A classical condition does not delay
// the instruction it guards, so a gate conditioned on a measurement can be
// scheduled in, or before, the cycle of the measurement that sets its bit;
// ALAP inherits the same model, QuantumCircuit::depth() counts the same way,
// and DAGCircuit::depth(), which follows the DAG's classical edges, does not.
// That is the current modelling choice rather than a defect, and it is
// asserted here as it stands so that changing it is a visible decision. The
// tests carrying it say so in their names.
//
// DAGCircuit::node() is new public surface: every traversal hands out ids,
// and this is how a caller turns one into the node it names.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/transpiler.hpp"

#include <algorithm>
#include <climits>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

using v11261::plan_with;
using v11261::recorder;
using v11261::RecorderPtr;
using v11261::run_sv;

namespace {

std::vector<int> every_layer_indices(const QuantumCircuit& qc) {
    RecorderPtr rec = recorder();
    run_sv(qc, plan_with(Anchor::every_layer(), rec));
    return rec->indices();
}

// The instruction stamps a pass left on its output circuit.
std::vector<int> stamps(const QuantumCircuit& qc) {
    std::vector<int> t;
    for (const Instruction& inst : qc.instructions) t.push_back(inst.schedule_time);
    return t;
}

QuantumCircuit scheduled(const QuantumCircuit& qc, const TranspilationPass& pass) {
    return pass.run(DAGCircuit::from_circuit(qc), TranspilationContext{}).to_circuit();
}

// Random circuits over one- and two-qubit gates, barriers on random subsets
// and on the whole register, and measurements.
QuantumCircuit random_circuit(std::uint64_t seed, int n = 5, int length = 40) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> kind(0, 9), qubit(0, n - 1);
    QuantumCircuit qc(n, n);
    for (int i = 0; i < length; ++i) {
        const int k = kind(rng);
        const int a = qubit(rng);
        int b = qubit(rng);
        if (b == a) b = (a + 1) % n;
        if (k <= 3) qc.h(a);
        else if (k <= 6) qc.cx(a, b);
        else if (k == 7) qc.barrier({a, b});
        else if (k == 8) qc.barrier();
        else qc.measure(a, a);
    }
    return qc;
}

// ---- Circuits worked by hand ----------------------------------------------

// h(0) h(0) measure(0 -> 0) then x(1) conditioned on clbit 0.
QuantumCircuit feedforward_after_depth() {
    QuantumCircuit qc(2, 1);
    qc.h(0);
    qc.h(0);
    qc.measure(0, 0);
    qc.add_if(0, 1, GT::X, {1});
    return qc;
}

}  // namespace

// =============================================================================
// The timing rule
// =============================================================================

TEST(V11291Scheduling, AGateStartsWhenEveryOperandWireIsFree) {
    // Qubit 3 is free from cycle 0, so cx(2, 3) starts there although it
    // follows cx(0, 1) in the list; x(0) waits for cx(0, 1), and cx(1, 2) for
    // both cx gates.
    QuantumCircuit qc(4);
    qc.h(0).h(1).cx(0, 1).cx(2, 3).x(0).cx(1, 2);
    EXPECT_EQ(asap_schedule_times(qc), (std::vector<int>{0, 0, 1, 0, 2, 2}));
}

TEST(V11291Scheduling, ABarrierSynchronisesItsWiresWithoutOccupyingThem) {
    // The barrier takes the cycle its latest wire becomes free and holds every
    // wire it names to that cycle, but the gate after it starts in that same
    // cycle rather than one later.
    QuantumCircuit qc(3);
    qc.h(0).h(0).barrier({0, 1}).x(1).x(2);
    EXPECT_EQ(asap_schedule_times(qc), (std::vector<int>{0, 1, 2, 2, 0}));
}

TEST(V11291Scheduling, ABarrierDoesNotHoldWiresItDoesNotName) {
    QuantumCircuit qc(3);
    qc.h(0).h(0).h(0).barrier({0, 1}).h(2).h(1);
    EXPECT_EQ(asap_schedule_times(qc), (std::vector<int>{0, 1, 2, 3, 0, 3}));
}

TEST(V11291Scheduling, AFullBarrierHoldsTheWholeRegister) {
    QuantumCircuit qc(3);
    qc.h(0).h(0).barrier().h(2);
    EXPECT_EQ(asap_schedule_times(qc), (std::vector<int>{0, 1, 2, 2}));
}

TEST(V11291Scheduling, AMeasurementOccupiesItsQubitForOneCycle) {
    QuantumCircuit qc(2, 2);
    qc.h(0);
    qc.measure(0, 0);
    qc.h(0);
    qc.measure(1, 1);
    EXPECT_EQ(asap_schedule_times(qc), (std::vector<int>{0, 1, 2, 0}));
}

TEST(V11291Scheduling, AnEmptyCircuitHasNoTimes) {
    EXPECT_TRUE(asap_schedule_times(QuantumCircuit(3)).empty());
}

TEST(V11291Scheduling, TheAsapPassStampsExactlyTheRule) {
    const ASAPSchedule pass;
    for (std::uint64_t seed = 1; seed <= 25; ++seed) {
        SCOPED_TRACE(seed);
        const QuantumCircuit out = scheduled(random_circuit(seed), pass);
        EXPECT_EQ(stamps(out), asap_schedule_times(out));
    }
}

// ---- The classical-wire modelling choice, as it stands --------------------

TEST(V11291Scheduling, ClassicalWiresAreNotTrackedByAsap) {
    // The conditioned X on qubit 1 needs the measurement at cycle 2, and is
    // placed at cycle 0 because only qubit wires are tracked.
    EXPECT_EQ(asap_schedule_times(feedforward_after_depth()), (std::vector<int>{0, 1, 2, 0}));
    EXPECT_EQ(stamps(scheduled(feedforward_after_depth(), ASAPSchedule{})),
              (std::vector<int>{0, 1, 2, 0}));
}

TEST(V11291Scheduling, ClassicalWiresAreNotTrackedByAlap) {
    // Two cycles of slack on qubit 0 let ALAP push the measurement to the end,
    // two cycles after the gate it conditions.
    QuantumCircuit qc(2, 1);
    qc.measure(0, 0);
    qc.add_if(0, 1, GT::X, {1});
    qc.h(1);
    qc.h(1);
    EXPECT_EQ(stamps(scheduled(qc, ALAPSchedule{})), (std::vector<int>{2, 0, 1, 2}));
}

TEST(V11291Scheduling, ClassicalWiresAreNotCountedByCircuitDepthButAreByDagDepth) {
    const QuantumCircuit qc = feedforward_after_depth();
    EXPECT_EQ(qc.depth(), 3);
    EXPECT_EQ(DAGCircuit::from_circuit(qc).depth(), 4);
}

// =============================================================================
// Anchor::every_layer
// =============================================================================

TEST(V11291Scheduling, EveryLayerFiresAtEachCompletedLayerOfABarrierCircuit) {
    // Times 0 0 1 1 2: the barrier lands in cycle 1 with the CX after it, so
    // layer 0 ends after both H gates, layer 1 after the CX, layer 2 at the end.
    QuantumCircuit qc(2);
    qc.h(0).h(1).barrier().cx(0, 1).x(0);
    ASSERT_EQ(asap_schedule_times(qc), (std::vector<int>{0, 0, 1, 1, 2}));
    EXPECT_EQ(every_layer_indices(qc), (std::vector<int>{1, 3, 4}));
}

TEST(V11291Scheduling, EveryLayerFiresAfterEachInstructionOfASerialChain) {
    QuantumCircuit qc(1);
    qc.h(0).x(0).y(0).z(0);
    EXPECT_EQ(every_layer_indices(qc), (std::vector<int>{0, 1, 2, 3}));
}

TEST(V11291Scheduling, EveryLayerSkipsInterleavedLayersButNeverTheEnd) {
    // Times 0 1 0: h(2) belongs to layer 0 but runs after x(0) from layer 1,
    // so no executed prefix is exactly layer 0 and only the end is a boundary.
    QuantumCircuit qc(3);
    qc.cx(0, 1).x(0).h(2);
    ASSERT_EQ(asap_schedule_times(qc), (std::vector<int>{0, 1, 0}));
    EXPECT_EQ(every_layer_indices(qc), (std::vector<int>{2}));
}

TEST(V11291Scheduling, EveryLayerAgreesWithTheRuleOnRandomCircuits) {
    // A boundary is an index at which the executed prefix is exactly the set of
    // instructions scheduled at or before some cycle. Computed here from the
    // times alone, by comparing sets rather than counts.
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        SCOPED_TRACE(seed);
        QuantumCircuit qc = random_circuit(seed, 4, 25);
        // every_layer runs on the statevector without measuring mid-circuit.
        std::vector<Instruction> kept;
        for (const Instruction& inst : qc.instructions)
            if (inst.type != GT::MEASURE) kept.push_back(inst);
        qc.instructions = kept;
        if (qc.instructions.empty()) continue;

        const std::vector<int> t = asap_schedule_times(qc);
        std::vector<int> expected;
        for (std::size_t i = 0; i < t.size(); ++i) {
            int T = 0;
            for (std::size_t j = 0; j <= i; ++j) T = std::max(T, t[j]);
            std::set<std::size_t> upto, prefix;
            for (std::size_t j = 0; j < t.size(); ++j)
                if (t[j] <= T) upto.insert(j);
            for (std::size_t j = 0; j <= i; ++j) prefix.insert(j);
            if (upto == prefix) expected.push_back(static_cast<int>(i));
        }
        EXPECT_EQ(every_layer_indices(qc), expected);
        EXPECT_EQ(expected.back(), static_cast<int>(qc.instructions.size()) - 1);
    }
}

TEST(V11291Scheduling, EveryLayerFollowsTheQubitOnlyRuleAcrossAMeasurement) {
    // The classical-wire choice seen from the observer: times 0 1 2 0 put the
    // conditioned X in layer 0, so no prefix completes a layer before the end.
    EXPECT_EQ(every_layer_indices(feedforward_after_depth()), (std::vector<int>{3}));
}

// =============================================================================
// DAGCircuit::node
// =============================================================================

TEST(V11291DagNode, EveryNodeResolvesToItself) {
    const DAGCircuit dag = DAGCircuit::from_circuit(random_circuit(7));
    ASSERT_FALSE(dag.nodes.empty());
    for (const DAGNode& n : dag.nodes) {
        const DAGNode* found = dag.node(n.node_id);
        ASSERT_NE(found, nullptr) << "id " << n.node_id;
        EXPECT_EQ(found, &n);
    }
}

TEST(V11291DagNode, EveryIdATraversalHandsOutResolves) {
    const DAGCircuit dag = DAGCircuit::from_circuit(random_circuit(8));
    std::vector<int> ids = dag.topological_sort();
    for (int id : dag.front_layer()) ids.push_back(id);
    for (int id : dag.topological_sort()) {
        for (int s : dag.successors(id)) ids.push_back(s);
        for (int p : dag.predecessors(id)) ids.push_back(p);
    }
    for (int id : ids) EXPECT_NE(dag.node(id), nullptr) << "id " << id;
}

TEST(V11291DagNode, AnIdTheDagDoesNotHoldIsNull) {
    const DAGCircuit dag = DAGCircuit::from_circuit(random_circuit(9));
    int max_id = 0;
    for (const DAGNode& n : dag.nodes) max_id = std::max(max_id, n.node_id);
    for (int id : {-1, max_id + 1, max_id + 1000, INT_MAX, INT_MIN}) {
        EXPECT_EQ(dag.node(id), nullptr) << "id " << id;
    }
    EXPECT_EQ(DAGCircuit().node(0), nullptr);
}

TEST(V11291DagNode, TheMutableOverloadWritesThroughToTheDag) {
    DAGCircuit dag = DAGCircuit::from_circuit(random_circuit(10));
    int op_id = -1;
    for (const DAGNode& n : dag.nodes)
        if (n.type == DAGNode::Type::OP) { op_id = n.node_id; break; }
    ASSERT_GE(op_id, 0);

    DAGNode* mut = dag.node(op_id);
    ASSERT_NE(mut, nullptr);
    mut->op.label = "marked";

    const DAGCircuit& view = dag;
    EXPECT_EQ(view.node(op_id)->op.label, "marked");
}

TEST(V11291DagNode, ARemovedNodeNoLongerResolvesAndTheRestStillDo) {
    DAGCircuit dag = DAGCircuit::from_circuit(random_circuit(11));
    std::vector<int> op_ids;
    for (const DAGNode& n : dag.nodes)
        if (n.type == DAGNode::Type::OP) op_ids.push_back(n.node_id);
    ASSERT_GE(op_ids.size(), 3u);

    const int gone = op_ids[op_ids.size() / 2];
    dag.remove_node(gone);
    EXPECT_EQ(dag.node(gone), nullptr);
    for (int id : op_ids) {
        if (id == gone) continue;
        const DAGNode* n = dag.node(id);
        ASSERT_NE(n, nullptr) << "id " << id;
        EXPECT_EQ(n->node_id, id);
    }
}

TEST(V11291DagNode, ASubstitutedNodeIsReplacedByResolvableOnes) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).h(1);
    DAGCircuit dag = DAGCircuit::from_circuit(qc);
    int cx_id = -1;
    for (const DAGNode& n : dag.nodes)
        if (n.type == DAGNode::Type::OP && n.op.type == GT::CX) cx_id = n.node_id;
    ASSERT_GE(cx_id, 0);

    QuantumCircuit rep(2);
    rep.h(1).cz(0, 1).h(1);
    dag.substitute_node(cx_id, DAGCircuit::from_circuit(rep));

    EXPECT_EQ(dag.node(cx_id), nullptr);
    int cz = 0;
    for (int id : dag.topological_sort()) {
        const DAGNode* n = dag.node(id);
        ASSERT_NE(n, nullptr) << "id " << id;
        if (n->type == DAGNode::Type::OP && n->op.type == GT::CZ) ++cz;
    }
    EXPECT_EQ(cz, 1);
}
