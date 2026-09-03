// 1.1.26.1 test wave - the four policy knobs.
//
// They are four separate questions and the release argues they must stay
// separate: one enumerator cannot mean both "convert, and throw when conversion
// is impossible" and "convert, and warn when it is". A suite that walked a
// diagonal through them would test that argument not at all, so Conversion and
// Response are crossed here in full, against each of the three things that can
// cause a refusal: no route exists, conversion was declined, and the guard.
//
// The guard is a ratio rather than a byte count, which is what lets it scale
// with the problem and the machine without tuning. That design has a
// consequence nothing documents and which is pinned below: on the MPS backend
// the live footprint GROWS with bond dimension, so the same observation can be
// refused early in a run and allowed later.
//
// Nothing here needs a large allocation to prove a refusal. A test that had to
// reserve the memory it claims is over budget would be unrunnable on a small
// machine and would defeat its own purpose, so every guard case is driven by
// the ratio between two cheap states.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace lindblad;

using v11261::capture_warnings;
using v11261::first_deliveries;
using v11261::label_instruction;
using v11261::layered_circuit;
using v11261::recorder;

namespace {

// Three qubits of entangling gates. Small enough that a density matrix
// conversion is cheap and large enough that the two footprints differ.
QuantumCircuit small_circuit() {
    QuantumCircuit qc(3);
    qc.h(0);
    qc.cx(0, 1);
    qc.cx(1, 2);
    return qc;
}

struct Outcome {
    bool success = false;
    std::string message;
    bool has_entry = false;
    std::vector<std::string> warnings;
};

// Runs a StateObserver asking for `wanted` under the given options, and reports
// every channel the verdict could have arrived on.
Outcome observe_as(StateForm wanted, RunPlan::Options options) {
    RunPlan plan;
    plan.options = options;
    plan.observations.observe(Anchor::at_end(),
                              std::make_shared<StateObserver>(wanted, "s"));

    StatevectorSimulator sim;
    StatevectorSimulator::Result result;
    Outcome out;
    out.warnings = capture_warnings([&] {
        result = sim.run(small_circuit(), 0, 20261, plan);
    });
    out.success = result.success;
    out.message = result.error_message;
    out.has_entry = result.observations.contains("s");
    return out;
}

RunPlan::Options knobs(Conversion conversion, Response response,
                       Cost cost = Cost::Unlimited) {
    RunPlan::Options o;
    o.conversion = conversion;
    o.response = response;
    o.cost = cost;
    return o;
}

}  // namespace

// =============================================================================
// Conversion crossed with Response, against a route that does not exist
// =============================================================================
// A stabilizer tableau cannot be recovered from arbitrary amplitudes. No knob
// makes it available, so Conversion is not what decides here: Response is.

TEST(V11261Knobs, AnImpossibleConversionThrowsUnderThrow) {
    for (const Conversion conversion : {Conversion::Convert, Conversion::Never}) {
        const Outcome o = observe_as(StateForm::Stabilizer,
                                     knobs(conversion, Response::Throw));
        EXPECT_FALSE(o.success);
        EXPECT_FALSE(o.has_entry);
    }
}

TEST(V11261Knobs, AnImpossibleConversionWarnsAndOmitsUnderWarn) {
    for (const Conversion conversion : {Conversion::Convert, Conversion::Never}) {
        const Outcome o = observe_as(StateForm::Stabilizer,
                                     knobs(conversion, Response::Warn));
        EXPECT_TRUE(o.success) << o.message;
        // Omitted rather than wrong. An entry produced here would be a
        // fabrication, and the absence is the honest answer.
        EXPECT_FALSE(o.has_entry);
        EXPECT_FALSE(o.warnings.empty());
    }
}

TEST(V11261Knobs, AnImpossibleConversionIsSilentUnderIgnore) {
    for (const Conversion conversion : {Conversion::Convert, Conversion::Never}) {
        const Outcome o = observe_as(StateForm::Stabilizer,
                                     knobs(conversion, Response::Ignore));
        EXPECT_TRUE(o.success) << o.message;
        EXPECT_FALSE(o.has_entry);
        EXPECT_TRUE(o.warnings.empty());
    }
}

// =============================================================================
// Conversion crossed with Response, against a route that DOES exist
// =============================================================================
// A statevector has a density matrix. Here Conversion is what decides, and
// Response only decides what the refusal looks like when Conversion declines.

TEST(V11261Knobs, APossibleConversionSucceedsUnderConvert) {
    const Outcome o = observe_as(StateForm::DensityMatrix,
                                 knobs(Conversion::Convert, Response::Throw));
    EXPECT_TRUE(o.success) << o.message;
    EXPECT_TRUE(o.has_entry);
}

TEST(V11261Knobs, APossibleConversionIsRefusedUnderNever) {
    // The distinction the two knobs exist to keep separate: this refusal is a
    // policy choice, and the one above it is a fact about the representations.
    const Outcome thrown = observe_as(StateForm::DensityMatrix,
                                      knobs(Conversion::Never, Response::Throw));
    EXPECT_FALSE(thrown.success);
    EXPECT_NE(thrown.message.find("Never"), std::string::npos) << thrown.message;

    const Outcome warned = observe_as(StateForm::DensityMatrix,
                                      knobs(Conversion::Never, Response::Warn));
    EXPECT_TRUE(warned.success) << warned.message;
    EXPECT_FALSE(warned.has_entry);
    EXPECT_FALSE(warned.warnings.empty());

    const Outcome ignored = observe_as(StateForm::DensityMatrix,
                                       knobs(Conversion::Never, Response::Ignore));
    EXPECT_TRUE(ignored.success) << ignored.message;
    EXPECT_FALSE(ignored.has_entry);
    EXPECT_TRUE(ignored.warnings.empty());
}

TEST(V11261Knobs, ANativeReadSucceedsUnderConversionNever) {
    // Never means hand back only what the backend already holds, so the native
    // form is exactly what it still permits.
    const Outcome o = observe_as(StateForm::Statevector,
                                 knobs(Conversion::Never, Response::Throw));
    EXPECT_TRUE(o.success) << o.message;
    EXPECT_TRUE(o.has_entry);
}

// =============================================================================
// The guard
// =============================================================================

TEST(V11261Knobs, TheGuardBoundaryIsExactAndInclusive) {
    // Three qubits: the statevector is 8 * 16 bytes and its density matrix is
    // 64 * 16, a ratio of exactly 8. The guard refuses what EXCEEDS the budget,
    // so the ratio itself passes and anything under it does not. Both figures
    // are derived from the register rather than transcribed.
    RunPlan::Options at_boundary = knobs(Conversion::Convert, Response::Throw, Cost::Guarded);
    at_boundary.guard_multiple = 8.0;
    const Outcome allowed = observe_as(StateForm::DensityMatrix, at_boundary);
    EXPECT_TRUE(allowed.success) << allowed.message;
    EXPECT_TRUE(allowed.has_entry);

    RunPlan::Options just_under = at_boundary;
    just_under.guard_multiple = 7.5;
    const Outcome refused = observe_as(StateForm::DensityMatrix, just_under);
    EXPECT_FALSE(refused.success);
    EXPECT_NE(refused.message.find("guard"), std::string::npos) << refused.message;
}

TEST(V11261Knobs, TheGuardRefusalNamesBothFootprints) {
    RunPlan::Options options = knobs(Conversion::Convert, Response::Throw, Cost::Guarded);
    options.guard_multiple = 1.0;
    const Outcome o = observe_as(StateForm::DensityMatrix, options);

    ASSERT_FALSE(o.success);
    // A caller cannot choose a better multiple without knowing what was
    // measured against what.
    EXPECT_NE(o.message.find("bytes"), std::string::npos) << o.message;
    EXPECT_NE(o.message.find("Cost::Unlimited"), std::string::npos) << o.message;
}

TEST(V11261Knobs, UnlimitedNeverRefusesOnSize) {
    RunPlan::Options options = knobs(Conversion::Convert, Response::Throw, Cost::Unlimited);
    options.guard_multiple = 1.0;  // ignored under Unlimited

    const Outcome o = observe_as(StateForm::DensityMatrix, options);
    EXPECT_TRUE(o.success) << o.message;
    EXPECT_TRUE(o.has_entry);
}

TEST(V11261Knobs, UnlimitedReportsTheCostItAllowed) {
    const Outcome o = observe_as(StateForm::DensityMatrix,
                                 knobs(Conversion::Convert, Response::Throw, Cost::Unlimited));
    ASSERT_TRUE(o.success) << o.message;
    // Allowed, but not silently: the caller who turned the guard off is told
    // what it would have caught.
    EXPECT_FALSE(o.warnings.empty());
}

TEST(V11261Knobs, UnlimitedStillRefusesImpossibility) {
    // Size is the only thing Unlimited stops caring about. A route that does
    // not exist is not a question of expense.
    const Outcome o = observe_as(StateForm::Stabilizer,
                                 knobs(Conversion::Convert, Response::Throw, Cost::Unlimited));
    EXPECT_FALSE(o.success);
    EXPECT_FALSE(o.has_entry);
}

TEST(V11261Knobs, ANativeReadUnderTheDefaultGuardIsAllowed) {
    // Handing back a copy of what the backend already holds costs exactly the
    // live state, which is what guard_multiple 1.0 is defined to permit. The
    // native path is not exempt from the comparison, it passes it.
    RunPlan::Options options = knobs(Conversion::Convert, Response::Throw, Cost::Guarded);
    options.guard_multiple = 1.0;

    const Outcome o = observe_as(StateForm::Statevector, options);
    EXPECT_TRUE(o.success) << o.message;
    EXPECT_TRUE(o.has_entry);
}

TEST(V11261Knobs, ANativeReadUnderASubCopyGuardIsRefused) {
    // The other side of the same rule, and worth pinning because it surprises:
    // a guard below 1.0 says no allocation may cost as much as the state
    // itself, and a copy of the state does. Nothing is exempt from the guard.
    RunPlan::Options options = knobs(Conversion::Convert, Response::Throw, Cost::Guarded);
    options.guard_multiple = 0.5;

    const Outcome o = observe_as(StateForm::Statevector, options);
    EXPECT_FALSE(o.success);
    EXPECT_NE(o.message.find("guard"), std::string::npos) << o.message;
}

TEST(V11261Knobs, TheGuardOnAnMpsMovesWithTheBondDimension) {
    // The guard measures against the LIVE state, and an MPS grows as it
    // entangles. So the denominator is not a property of the register the way
    // it is on every other backend, and the same request can be refused at one
    // anchor and allowed at a later one in the same run.
    //
    // Worked through, at three qubits and guard_multiple 1.0. The dense
    // conversion is fixed at 8 amplitudes * 16 bytes = 128 bytes throughout.
    //
    //   at_start, the product state: three tensors of shape 1x2x1, so 6
    //   complex entries = 96 bytes. 128 > 96, so the guard REFUSES.
    //
    //   at_end, the GHZ state: tensors of 1x2x2, 2x2x2 and 2x2x1, so 16
    //   complex entries = 256 bytes. 128 <= 256, so the guard ALLOWS it.
    //
    // Same plan, same circuit, same request, opposite verdicts.
    QuantumCircuit qc(3);
    qc.h(0);
    qc.cx(0, 1);
    qc.cx(1, 2);

    auto early = std::make_shared<StateObserver>(StateForm::Statevector);
    auto late = std::make_shared<StateObserver>(StateForm::Statevector);

    RunPlan plan;
    plan.options = knobs(Conversion::Convert, Response::Ignore, Cost::Guarded);
    plan.options.guard_multiple = 1.0;
    plan.observations.observe(Anchor::at_start(), early);
    plan.observations.observe(Anchor::at_end(), late);

    MPSSimulator sim;
    auto r = sim.run(qc, 8, 0, 20261, plan);

    EXPECT_EQ(r.final_state.n_qubits, 3);

    // Ignore, so neither refusal fails the run. What differs is whether each
    // observer captured anything, and that difference IS the finding: the
    // verdict depends on when it was asked, which is true on no other backend.
    EXPECT_NE(early->count(), late->count())
        << "the guard reached the same verdict at both ends of the run, so the "
           "live footprint did not move as expected";
    EXPECT_EQ(early->count(), 0u);
    EXPECT_EQ(late->count(), 1u);
}

// =============================================================================
// Warn is deduplicated by the channel
// =============================================================================

TEST(V11261Knobs, WarnIsDeliveredOnceAndThenTallied) {
    // The observer fires at every instruction and is refused every time. The
    // channel deduplicates, so the caller gets the message once rather than
    // once per gate, which is what makes Warn usable on a trace anchor at all.
    RunPlan plan;
    plan.options = knobs(Conversion::Convert, Response::Warn);
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<StateObserver>(StateForm::Stabilizer, "s"));

    StatevectorSimulator sim;
    StatevectorSimulator::Result result;
    const std::vector<std::string> msgs = capture_warnings([&] {
        result = sim.run(layered_circuit(), 0, 20261, plan);
    });

    ASSERT_TRUE(result.success) << result.error_message;
    const std::vector<std::string> first = first_deliveries(msgs);
    EXPECT_EQ(first.size(), 1u) << "the channel delivered " << first.size()
                                << " first-time notices for one repeated refusal";
}

TEST(V11261Knobs, ARefusedFiringRecordsNoPoint) {
    // The observer's own results and its firing points stay in step, because
    // a firing that produced nothing records nothing. A caller indexes one
    // against the other, so a drift here would misattribute every reading.
    auto observer = std::make_shared<StateObserver>(StateForm::Stabilizer, "s");

    RunPlan plan;
    plan.options = knobs(Conversion::Convert, Response::Ignore);
    plan.observations.observe(Anchor::every_instruction(), observer);

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(observer->count(), 0u);
    EXPECT_FALSE(r.observations.contains("s"));
}

// =============================================================================
// Fusion
// =============================================================================

TEST(V11261Knobs, SuppressKeepsAnchorsOnTheInstructionsTheCallerWrote) {
    // Fusion rewrites instructions into blocks, which renumbers positions and
    // drops labels. Suppress is the default for an observed run precisely so
    // every anchor keeps meaning what it said.
    StatevectorSimulator::Options options;
    options.fusion_enable = true;
    options.fusion_threshold = 2;  // force fusion to engage on a small circuit

    RunPlan plan;
    plan.options.fusion = RunPlan::Options::Fusion::Suppress;
    auto rec = recorder();
    plan.observations.observe(Anchor::every_instruction(), rec);

    const QuantumCircuit qc = layered_circuit();
    StatevectorSimulator sim(options);
    auto r = sim.run(qc, 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(rec->count(), qc.instructions.size());
    EXPECT_EQ(rec->indices(), (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST(V11261Knobs, KeepBindsAnchorsToTheFusedCircuit) {
    // Keep asks for fusion anyway and accepts the consequence: anchors then
    // index blocks, because blocks are what execute.
    StatevectorSimulator::Options options;
    options.fusion_enable = true;
    options.fusion_threshold = 2;

    RunPlan plan;
    plan.options.fusion = RunPlan::Options::Fusion::Keep;
    auto rec = recorder();
    plan.observations.observe(Anchor::every_instruction(), rec);

    const QuantumCircuit qc = layered_circuit();
    StatevectorSimulator sim(options);
    auto r = sim.run(qc, 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    // Fusion only helps if it merged something, so a firing count equal to the
    // instruction count would mean this test proved nothing.
    EXPECT_LT(rec->count(), qc.instructions.size())
        << "fusion did not merge anything, so Keep and Suppress are "
           "indistinguishable here and this test is not testing them";
}

TEST(V11261Knobs, ALabelFusionAbsorbedFailsTheRunUnderKeep) {
    // The alternative would be an anchor that silently never fires, which is
    // the one outcome the anchor design refuses to produce.
    StatevectorSimulator::Options options;
    options.fusion_enable = true;
    options.fusion_threshold = 2;

    QuantumCircuit qc = layered_circuit();
    label_instruction(qc, 2, "absorbed");

    RunPlan plan;
    plan.options.fusion = RunPlan::Options::Fusion::Keep;
    plan.observations.observe(Anchor::after_label("absorbed"), recorder());

    StatevectorSimulator sim(options);
    auto r = sim.run(qc, 0, 20261, plan);

    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error_message.find("absorbed"), std::string::npos) << r.error_message;
}

TEST(V11261Knobs, TheSameLabelResolvesUnderSuppress) {
    // The control for the test above: the label is fine, and only fusion
    // removed it.
    StatevectorSimulator::Options options;
    options.fusion_enable = true;
    options.fusion_threshold = 2;

    QuantumCircuit qc = layered_circuit();
    label_instruction(qc, 2, "absorbed");

    RunPlan plan;
    plan.options.fusion = RunPlan::Options::Fusion::Suppress;
    auto rec = recorder();
    plan.observations.observe(Anchor::after_label("absorbed"), rec);

    StatevectorSimulator sim(options);
    auto r = sim.run(qc, 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(rec->indices(), (std::vector<int>{2}));
}
