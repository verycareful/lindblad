// 1.1.26.1 test wave - where an anchor fires, and what it hands over.
//
// An anchor is the whole addressing scheme of the observation harness: it
// names a point in a run without the circuit carrying anything, so every claim
// about WHEN an observer is invoked rests on it. The release shipped nine
// anchor kinds and no test constructs one.
//
// The two properties worth separating are membership and order. A suite that
// probes one firing establishes that an anchor fired somewhere; these compare
// whole index sequences, because an anchor that fires at the right count in
// the wrong order is a different defect and an equally silent one.
//
// The layering expectation is hand computed in the oracle header rather than
// taken from a run, on a circuit chosen so that a layering computed any other
// plausible way gives a different answer.
//
// Two behaviours here are asserted rather than assumed because they read as
// surprising and are load bearing: an instruction the backend SKIPS (a
// barrier, a condition that does not hold) still fires its anchors, and on a
// route that defers measurement to a sampling pass the measurement anchors
// fire around an instruction that did not collapse anything.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace lindblad;

using v11261::Firing;
using v11261::label_instruction;
using v11261::layered_circuit;
using v11261::layered_circuit_layer_ends;
using v11261::plan_with;
using v11261::RecorderPtr;
using v11261::recorder;
using v11261::run_sv;

namespace {

// Captures the outcome distribution at each firing. Small enough to live here,
// and deliberately not one of the built-in observers: the claim under test is
// about what the STATE looks like at an anchor, so the reader should be able
// to see exactly what was read without leaving the file.
class ProbabilitySpy : public Observer {
public:
    void observe(const ObservationContext& ctx) override {
        readings_.push_back(ctx.state.statevector().probabilities());
    }

    const std::vector<std::vector<double>>& readings() const { return readings_; }

private:
    std::vector<std::vector<double>> readings_;
};

}  // namespace

// =============================================================================
// The endpoints
// =============================================================================

TEST(V11261Anchors, AtStartFiresOnceBeforeTheFirstInstruction) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    run_sv(qc, plan_with(Anchor::at_start(), rec));

    ASSERT_EQ(rec->count(), 1u);
    // -1 is the documented index at the start: no instruction has run, and any
    // non-negative value would name one that has.
    EXPECT_EQ(rec->firings()[0].instruction, -1);
    EXPECT_EQ(rec->firings()[0].anchor, "at_start");
    EXPECT_EQ(rec->firings()[0].n_qubits, qc.n_qubits);
}

TEST(V11261Anchors, AtEndFiresOnceAfterTheLastInstruction) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    run_sv(qc, plan_with(Anchor::at_end(), rec));

    ASSERT_EQ(rec->count(), 1u);
    EXPECT_EQ(rec->firings()[0].instruction,
              static_cast<int>(qc.instructions.size()) - 1);
    EXPECT_EQ(rec->firings()[0].anchor, "at_end");
}

// =============================================================================
// Positional anchors
// =============================================================================

TEST(V11261Anchors, AfterInstructionFiresAtExactlyThatIndex) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    run_sv(qc, plan_with(Anchor::after_instruction(3), rec));

    ASSERT_EQ(rec->count(), 1u);
    EXPECT_EQ(rec->firings()[0].instruction, 3);
    EXPECT_EQ(rec->firings()[0].anchor, "after_instruction(3)");
}

TEST(V11261Anchors, AfterLabelFiresOncePerInstructionCarryingIt) {
    QuantumCircuit qc = layered_circuit();
    label_instruction(qc, 0, "mark");
    label_instruction(qc, 2, "mark");
    label_instruction(qc, 4, "mark");
    label_instruction(qc, 1, "other");

    RecorderPtr rec = recorder();
    run_sv(qc, plan_with(Anchor::after_label("mark"), rec));

    // One label on three instructions is three firings, not one. The anchor
    // names a label and a label is not unique.
    EXPECT_EQ(rec->indices(), (std::vector<int>{0, 2, 4}));
    for (const Firing& f : rec->firings()) EXPECT_EQ(f.anchor, "after_label(mark)");
}

TEST(V11261Anchors, EveryInstructionFiresOncePerInstructionInOrder) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    run_sv(qc, plan_with(Anchor::every_instruction(), rec));

    EXPECT_EQ(rec->indices(), (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST(V11261Anchors, EveryLayerMatchesTheHandComputedLayering) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    run_sv(qc, plan_with(Anchor::every_layer(), rec));

    EXPECT_EQ(rec->indices(), layered_circuit_layer_ends());
}

TEST(V11261Anchors, WhereFiresOnlyWhereThePredicateHolds) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    RunPlan plan = plan_with(
        Anchor::where([](const Instruction& inst, int) {
            return inst.type == Instruction::GateType::CX;
        }),
        rec);
    run_sv(qc, plan);

    // Indices 2, 3 and 5 are the CX gates in layered_circuit().
    EXPECT_EQ(rec->indices(), (std::vector<int>{2, 3, 5}));
    for (const Firing& f : rec->firings()) EXPECT_EQ(f.anchor, "where(predicate)");
}

TEST(V11261Anchors, ThePredicateReceivesTheIndexItIsJudging) {
    const QuantumCircuit qc = layered_circuit();
    std::vector<int> seen;

    RunPlan plan = plan_with(
        Anchor::where([&seen](const Instruction&, int index) {
            seen.push_back(index);
            return false;
        }),
        recorder());
    run_sv(qc, plan);

    EXPECT_EQ(seen, (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

// =============================================================================
// Instructions the backend skips
// =============================================================================

TEST(V11261Anchors, ABarrierStillFires) {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.barrier();
    qc.cx(0, 1);

    RecorderPtr rec = recorder();
    run_sv(qc, plan_with(Anchor::every_instruction(), rec));

    // The backend executes nothing at index 1, and the anchor fires anyway. An
    // anchor that silently did not fire is indistinguishable from one that
    // fired and found nothing.
    EXPECT_EQ(rec->indices(), (std::vector<int>{0, 1, 2}));
}

TEST(V11261Anchors, AConditionedGateThatDoesNotRunStillFires) {
    // Qubit 0 is measured in |0>, so the condition clreg[0] == 1 never holds
    // and the X at index 1 never runs. Nothing here is random, so the claim is
    // exact rather than probable.
    QuantumCircuit qc(2, 2);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});
    qc.measure(1, 1);

    RecorderPtr rec = recorder();
    const int shots = 3;
    StatevectorSimulator sim;
    auto result = sim.run(qc, shots, 20261, plan_with(Anchor::every_instruction(), rec));

    EXPECT_EQ(rec->indices(), (std::vector<int>{0, 1, 2, 0, 1, 2, 0, 1, 2}));

    // Every shot reads 00, which is what proves the conditioned X did not run
    // while its anchor fired on every one of those shots.
    EXPECT_EQ(result.counts.at("00"), shots);
    EXPECT_EQ(result.counts.size(), 1u);
}

// =============================================================================
// Shots
// =============================================================================

TEST(V11261Anchors, FiringOrderIsShotThenInstruction) {
    QuantumCircuit qc(2, 2);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});

    RecorderPtr rec = recorder();
    StatevectorSimulator sim;
    sim.run(qc, 3, 20261, plan_with(Anchor::every_instruction(), rec));

    EXPECT_EQ(rec->shots(), (std::vector<int>{0, 0, 1, 1, 2, 2}));
    EXPECT_EQ(rec->indices(), (std::vector<int>{0, 1, 0, 1, 0, 1}));
}

TEST(V11261Anchors, BeginRunCarriesTheRegisterAndTheShotCount) {
    QuantumCircuit qc(2, 2);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});

    RecorderPtr rec = recorder();
    StatevectorSimulator sim;
    sim.run(qc, 5, 20261, plan_with(Anchor::at_end(), rec));

    EXPECT_EQ(rec->begins(), 1);
    EXPECT_EQ(rec->ends(), 1);
    EXPECT_EQ(rec->begin_qubits(), 2);
    EXPECT_EQ(rec->begin_shots(), 5);
    for (const Firing& f : rec->firings()) EXPECT_EQ(f.n_shots, 5);
}

TEST(V11261Anchors, BeginRunAndEndRunHappenOnceAroundTheWholeRun) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    run_sv(qc, plan_with(Anchor::every_instruction(), rec));

    // Once each, not once per firing and not once per shot.
    EXPECT_EQ(rec->begins(), 1);
    EXPECT_EQ(rec->ends(), 1);
}

// =============================================================================
// Coordinates
// =============================================================================

TEST(V11261Anchors, OneObserverOnTwoAnchorsTellsThemApart) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), rec);
    plan.observations.observe(Anchor::after_instruction(2), rec);
    run_sv(qc, plan);

    // Without the anchor name on the context these two readings are a sequence
    // of length two whose entries cannot be told apart, and the coordinates
    // are decoration.
    ASSERT_EQ(rec->count(), 2u);
    EXPECT_EQ(rec->anchors(), (std::vector<std::string>{"at_start", "after_instruction(2)"}));
    EXPECT_EQ(rec->indices(), (std::vector<int>{-1, 2}));
}

TEST(V11261Anchors, AnchorNamesAreTheDocumentedSpelling) {
    // FiringPoint::anchor carries exactly these, so they are caller-facing
    // surface and are pinned as strings rather than inferred from a run.
    EXPECT_EQ(Anchor::at_start().name(), "at_start");
    EXPECT_EQ(Anchor::at_end().name(), "at_end");
    EXPECT_EQ(Anchor::after_instruction(3).name(), "after_instruction(3)");
    EXPECT_EQ(Anchor::after_label("qft_done").name(), "after_label(qft_done)");
    EXPECT_EQ(Anchor::every_instruction().name(), "every_instruction");
    EXPECT_EQ(Anchor::every_layer().name(), "every_layer");
    EXPECT_EQ(Anchor::before_each_measurement().name(), "before_each_measurement");
    EXPECT_EQ(Anchor::after_each_measurement().name(), "after_each_measurement");
    EXPECT_EQ(Anchor::where([](const Instruction&, int) { return true; }).name(),
              "where(predicate)");
}

TEST(V11261Anchors, ObserversRunOnTheThreadThatCalledRun) {
    const QuantumCircuit qc = layered_circuit();
    RecorderPtr rec = recorder();

    run_sv(qc, plan_with(Anchor::every_instruction(), rec));

    // The interface promises the calling thread and no concurrency within a
    // run, which is what lets an implementation hold mutable state unlocked.
    // The gate kernels do run OpenMP, so this is the assertion that keeps a
    // future parallel shot loop from breaking that contract quietly.
    ASSERT_EQ(rec->count(), 6u);
    std::set<std::thread::id> threads;
    for (const Firing& f : rec->firings()) threads.insert(f.thread);
    ASSERT_EQ(threads.size(), 1u);
    EXPECT_EQ(*threads.begin(), std::this_thread::get_id());
}

// =============================================================================
// Measurement anchors
// =============================================================================

TEST(V11261Anchors, MeasurementAnchorsFireAroundEachMeasure) {
    QuantumCircuit qc(2, 2);
    qc.h(0);
    qc.measure(0, 0);
    qc.cx(0, 1);
    qc.measure(1, 1);

    RecorderPtr before = recorder();
    RecorderPtr after = recorder();

    RunPlan plan;
    plan.observations.observe(Anchor::before_each_measurement(), before);
    plan.observations.observe(Anchor::after_each_measurement(), after);

    StatevectorSimulator sim;
    sim.run(qc, 1, 20261, plan);

    // Indices 1 and 3 are the two MEASURE instructions; the gates between them
    // fire neither anchor.
    EXPECT_EQ(before->indices(), (std::vector<int>{1, 3}));
    EXPECT_EQ(after->indices(), (std::vector<int>{1, 3}));
}

TEST(V11261Anchors, MeasurementAnchorsSeeAnUncollapsedStateOnTheTerminalRoute) {
    // Terminal-only measurement: the gate pass skips MEASURE entirely and the
    // outcomes are drawn afterwards by sampling one final state. The anchors
    // still fire around the skipped instruction, so what they hand over is the
    // state BEFORE any collapse, at both of them.
    //
    // This is defensible and it is not what the anchor names suggest, so it is
    // asserted here rather than discovered by a caller.
    QuantumCircuit qc(1, 1);
    qc.h(0);
    qc.measure(0, 0);

    auto before = std::make_shared<ProbabilitySpy>();
    auto after = std::make_shared<ProbabilitySpy>();

    RunPlan plan;
    plan.observations.observe(Anchor::before_each_measurement(), before);
    plan.observations.observe(Anchor::after_each_measurement(), after);

    StatevectorSimulator sim;
    sim.run(qc, 64, 20261, plan);

    ASSERT_EQ(before->readings().size(), 1u);
    ASSERT_EQ(after->readings().size(), 1u);
    ASSERT_EQ(after->readings()[0].size(), 2u);

    const double half = 1.0 / 2.0;
    EXPECT_NEAR(before->readings()[0][0], half, DEFAULT_PHYSICAL_ATOL);
    EXPECT_NEAR(after->readings()[0][0], half, DEFAULT_PHYSICAL_ATOL);
    EXPECT_NEAR(after->readings()[0][1], half, DEFAULT_PHYSICAL_ATOL);
}

TEST(V11261Anchors, MeasurementAnchorsSeeACollapsedStateOnThePerShotRoute) {
    // The same two anchors on a route that does collapse: a condition forces
    // per-shot trajectories, so after_each_measurement sees a basis state
    // while before_each_measurement still sees the superposition.
    QuantumCircuit qc(2, 2);
    qc.h(0);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});

    auto before = std::make_shared<ProbabilitySpy>();
    auto after = std::make_shared<ProbabilitySpy>();

    RunPlan plan;
    plan.observations.observe(Anchor::before_each_measurement(), before);
    plan.observations.observe(Anchor::after_each_measurement(), after);

    StatevectorSimulator sim;
    sim.run(qc, 4, 20261, plan);

    ASSERT_EQ(before->readings().size(), 4u);
    ASSERT_EQ(after->readings().size(), 4u);

    for (const auto& reading : after->readings()) {
        // Exactly one amplitude carries all the weight once the qubit has
        // collapsed, whichever outcome was drawn.
        double largest = 0.0;
        for (const double p : reading) largest = std::max(largest, p);
        EXPECT_NEAR(largest, 1.0, DEFAULT_PHYSICAL_ATOL);
    }
    for (const auto& reading : before->readings()) {
        EXPECT_NEAR(reading[0], 1.0 / 2.0, DEFAULT_PHYSICAL_ATOL);
    }
}

// =============================================================================
// Degenerate circuits
// =============================================================================

TEST(V11261Anchors, AnEmptyCircuitStillFiresStartAndEnd) {
    const QuantumCircuit qc(2);
    ASSERT_TRUE(qc.instructions.empty());

    RecorderPtr start = recorder();
    RecorderPtr end = recorder();
    RecorderPtr every = recorder();

    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), start);
    plan.observations.observe(Anchor::at_end(), end);
    plan.observations.observe(Anchor::every_instruction(), every);
    run_sv(qc, plan);

    ASSERT_EQ(start->count(), 1u);
    EXPECT_EQ(start->firings()[0].instruction, -1);

    // at_end reports the last index reached, and on a circuit with no
    // instructions that is the same -1 the start carries.
    ASSERT_EQ(end->count(), 1u);
    EXPECT_EQ(end->firings()[0].instruction, -1);

    EXPECT_EQ(every->count(), 0u);
}

TEST(V11261Anchors, EveryLayerFiresNothingOnAnEmptyCircuit) {
    const QuantumCircuit qc(2);
    RecorderPtr rec = recorder();

    run_sv(qc, plan_with(Anchor::every_layer(), rec));

    EXPECT_EQ(rec->count(), 0u);
}

TEST(V11261Anchors, ASingleInstructionCircuitIsOneLayer) {
    QuantumCircuit qc(2);
    qc.h(0);

    RecorderPtr rec = recorder();
    run_sv(qc, plan_with(Anchor::every_layer(), rec));

    EXPECT_EQ(rec->indices(), (std::vector<int>{0}));
}
