// 1.1.26.1 test wave - the whole plan is validated before the run, not during
// it. EVERY TEST IN THE FIRST TWO SECTIONS OF THIS FILE SHIPS RED.
//
// The harness states its own principle: an anchor is resolved "before any state
// is touched", because a failure found later costs the run and arrives as an
// ambiguous half-result. 1.1.26.0 applies that principle to anchors and to
// nothing else. Every other mistake a caller can make in a plan is found
// lazily: an out-of-range amplitude index and a malformed entropy region on the
// first firing, and a duplicate bundle label only at end_run, after the entire
// simulation has been paid for.
//
// These tests assert the intended contract rather than the shipped behaviour,
// so they are red until 1.1.26.2 corrects it. They are written to fail on
// exactly one assertion each: the run does fail today, so `success` is already
// false, and what is red is WHEN it failed.
//
// The discriminator is a witness. A recorder is attached to every_instruction
// alongside the offending observer at at_end. Under the intended contract the
// verdict is reached before any state is touched and the witness never fires;
// under the shipped behaviour the witness fires once per instruction and only
// then does the offending observer refuse. The count separates the two exactly,
// which a success flag alone cannot.
//
// The third section is NOT red and must never become red. It pins the boundary
// of this change: a conversion or cost refusal is what Response::Warn and
// Response::Ignore exist to omit, so moving those decisions earlier must move
// WHEN the verdict is reached without changing WHAT it is. A fix that turns
// every knob-governed refusal into a pre-flight throw would empty both of those
// enumerators of meaning, and these tests are what catches that.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace lindblad;

using v11261::capture_warnings;
using v11261::layered_circuit;
using v11261::RecorderPtr;
using v11261::recorder;

namespace {

// Runs `plan` on the four-qubit six-instruction circuit and reports how much of
// the run happened before the plan was refused. The witness is attached here so
// no test can forget it.
struct Verdict {
    bool failed = false;
    std::string message;
    std::size_t witness_firings = 0;
    std::size_t bundle_entries = 0;
};

Verdict run_and_watch(RunPlan plan) {
    RecorderPtr witness = recorder();
    plan.observations.observe(Anchor::every_instruction(), witness);

    StatevectorSimulator sim;
    auto result = sim.run(layered_circuit(), 0, 20261, plan);

    Verdict v;
    v.failed = !result.success;
    v.message = result.error_message;
    v.witness_firings = witness->count();
    v.bundle_entries = result.observations.size();
    return v;
}

// The plan is refused, and refused before the circuit ran at all.
void expect_refused_at_preflight(const Verdict& v) {
    EXPECT_TRUE(v.failed) << "the plan was accepted";
    EXPECT_EQ(v.witness_firings, 0u)
        << "the plan was refused only after " << v.witness_firings
        << " instructions had already run; the verdict was available before any "
           "state was touched";
}

}  // namespace

// =============================================================================
// EXPECTED-RED: caller errors, which must be refused before the run
// =============================================================================
// Each of these is decidable from the plan and the register width alone. None
// of them needs a circuit to have run, a state to exist, or an anchor to have
// fired. All of them are found today on the first firing.

TEST(V11261PreflightContract, AnOutOfRangeAmplitudeIndexIsRefusedBeforeTheRun) {
    // Four qubits hold sixteen amplitudes, so index 64 names nothing. The
    // observer's indices are fixed at construction and the register width is
    // fixed by the circuit, so nothing about this verdict needs a run.
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(),
        std::make_shared<AmplitudeObserver>(std::vector<std::size_t>{64}, "amps"));

    expect_refused_at_preflight(run_and_watch(std::move(plan)));
}

TEST(V11261PreflightContract, ARepeatedQubitInAnEntropyRegionIsRefusedBeforeTheRun) {
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(),
        std::make_shared<EntropyObserver>(std::vector<int>{1, 1}, 1.0, "s"));

    expect_refused_at_preflight(run_and_watch(std::move(plan)));
}

TEST(V11261PreflightContract, AQubitOutsideTheRegisterIsRefusedBeforeTheRun) {
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(),
        std::make_shared<EntropyObserver>(std::vector<int>{9}, 1.0, "s"));

    expect_refused_at_preflight(run_and_watch(std::move(plan)));
}

TEST(V11261PreflightContract, ACutNamingEveryQubitIsRefusedBeforeTheRun) {
    // A cut needs a side to be entangled with, and that this one has none is
    // visible from the region and the qubit count together.
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(),
        std::make_shared<EntropyObserver>(std::vector<int>{0, 1, 2, 3}, 1.0, "s"));

    expect_refused_at_preflight(run_and_watch(std::move(plan)));
}

TEST(V11261PreflightContract, ADuplicateLabelIsRefusedBeforeTheRun) {
    // The worst of the group. Two observers sharing a label is a property of
    // the PLAN: it needs no circuit, no state and no run, and today it is paid
    // for with the entire simulation and refused at the end of it.
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("same"));
    plan.observations.observe(Anchor::at_end(), std::make_shared<ProbabilityObserver>("same"));

    expect_refused_at_preflight(run_and_watch(std::move(plan)));
}

// =============================================================================
// EXPECTED-RED: a failed run hands back nothing
// =============================================================================

TEST(V11261PreflightContract, AFailedRunLeavesTheBundleEmpty) {
    // end_run walks the observers in order, so the ones ahead of a collision
    // have already written their entries. The result then carries success ==
    // false beside a partly populated bundle: a caller checking the flag learns
    // the run failed, and a caller reading the bundle finds real entries in it.
    // Whichever of the two they trust, one of them is lying.
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("same"));
    plan.observations.observe(Anchor::at_end(), std::make_shared<ProbabilityObserver>("same"));

    const Verdict v = run_and_watch(std::move(plan));

    ASSERT_TRUE(v.failed);
    EXPECT_EQ(v.bundle_entries, 0u)
        << "a failed run handed back " << v.bundle_entries
        << " observations, which read as a complete answer";
}

// =============================================================================
// EXPECTED-RED: knob-governed refusals, decided before the run
// =============================================================================
// These differ from the group above in WHAT the verdict is, not in when it
// should be reached. Under Response::Throw a refusal fails the run, so the
// witness applies here unchanged. Under Warn and Ignore the verdict is to omit,
// and that is the next section.

TEST(V11261PreflightContract, AnImpossibleConversionIsRefusedBeforeTheRun) {
    // No conversion from dense amplitudes to a stabilizer tableau exists at
    // all: convertible_to answers that from the two forms alone, and both are
    // known before the circuit runs.
    RunPlan plan;
    plan.options.response = Response::Throw;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<StateObserver>(StateForm::Stabilizer, "st"));

    expect_refused_at_preflight(run_and_watch(std::move(plan)));
}

TEST(V11261PreflightContract, AnOverGuardConversionIsRefusedBeforeTheRun) {
    // A density matrix costs 4^n against a statevector's 2^n, so at
    // guard_multiple 1.0 this is over budget on any register at all, and both
    // figures come from the qubit count rather than from a live state.
    RunPlan plan;
    plan.options.response = Response::Throw;
    plan.options.cost = Cost::Guarded;
    plan.options.guard_multiple = 1.0;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<StateObserver>(StateForm::DensityMatrix, "dm"));

    expect_refused_at_preflight(run_and_watch(std::move(plan)));
}

// =============================================================================
// GREEN, and the boundary of the change: the knobs still govern
// =============================================================================
// Nothing below is red. These pin what must NOT change when the decisions above
// move earlier. A refusal that Warn or Ignore is meant to absorb must stay
// absorbed: an observation the backend cannot produce is omissible by design,
// which is the whole reason Response exists as a separate knob from Conversion.

TEST(V11261PreflightContract, WarnOmitsTheObservationAndCompletesTheRun) {
    RunPlan plan;
    plan.options.response = Response::Warn;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<StateObserver>(StateForm::Stabilizer, "st"));

    StatevectorSimulator sim;
    StatevectorSimulator::Result result;
    const std::vector<std::string> msgs = capture_warnings([&] {
        result = sim.run(layered_circuit(), 0, 20261, plan);
    });

    EXPECT_TRUE(result.success) << result.error_message;
    EXPECT_FALSE(result.observations.contains("st"));
    EXPECT_FALSE(msgs.empty()) << "Warn produced no warning";
}

TEST(V11261PreflightContract, IgnoreOmitsTheObservationSilently) {
    RunPlan plan;
    plan.options.response = Response::Ignore;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<StateObserver>(StateForm::Stabilizer, "st"));

    StatevectorSimulator sim;
    StatevectorSimulator::Result result;
    const std::vector<std::string> msgs = capture_warnings([&] {
        result = sim.run(layered_circuit(), 0, 20261, plan);
    });

    EXPECT_TRUE(result.success) << result.error_message;
    EXPECT_FALSE(result.observations.contains("st"));
    EXPECT_TRUE(msgs.empty()) << msgs.front();
}

TEST(V11261PreflightContract, AnOmittedObservationDoesNotStopTheOthers) {
    // One observer refused must not take the rest of the plan with it. This is
    // what "the observation is omitted" has to mean for the knob to be useful,
    // and it is the property most at risk from moving refusals to a pre-flight.
    RunPlan plan;
    plan.options.response = Response::Ignore;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<StateObserver>(StateForm::Stabilizer, "refused"));
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("kept"));

    StatevectorSimulator sim;
    auto result = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_FALSE(result.observations.contains("refused"));
    EXPECT_TRUE(result.observations.contains("kept"));
}
