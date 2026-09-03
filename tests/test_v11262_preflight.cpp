// 1.1.26.2 test wave - the plan is validated before the run.
//
// The suite that pinned this defect asserted WHEN a verdict was reached, using
// a witness observer to prove the circuit had not run yet. Those tests are
// still in the tree and are now green. This file tests the mechanism that made
// them green, which is a different question: the pre-flight hook is public
// surface, so a caller can implement it, and the rules it must obey are not
// obvious from the fact that the eight pinned tests pass.
//
// Three properties matter most here and none of them is implied by the pins.
// The hook is consulted once per OBSERVER rather than once per attachment, so
// an observer on three anchors is asked once and does not collide with itself
// over its own label. An observer the pre-flight rules out is DROPPED, so it
// costs nothing per anchor afterwards, which is the only place the saving
// actually shows up. And a refusal that Warn or Ignore would have absorbed is
// still absorbed, because deciding sooner must change when a verdict is reached
// and never what it is.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

using v11261::capture_warnings;
using v11261::layered_circuit;
using v11261::recorder;

namespace {

// Counts what the harness asked of it and when. Uses only the public interface,
// which is the point: the hook has to be implementable from outside.
class CountingObserver : public Observer {
public:
    explicit CountingObserver(bool accept = true, std::string label = {})
        : accept_(accept), label_(std::move(label)) {}

    bool preflight(const PreflightContext& ctx) override {
        ++preflights_;
        seen_form_ = ctx.form;
        seen_qubits_ = ctx.n_qubits;
        return accept_;
    }

    const std::string& label() const override { return label_; }

    void observe(const ObservationContext&) override { ++observations_; }

    int preflights() const { return preflights_; }
    int observations() const { return observations_; }
    StateForm seen_form() const { return seen_form_; }
    int seen_qubits() const { return seen_qubits_; }

private:
    bool accept_;
    std::string label_;
    int preflights_ = 0;
    int observations_ = 0;
    StateForm seen_form_ = StateForm::Statevector;
    int seen_qubits_ = -1;
};

// Writes into the bundle DURING the run rather than at end_run, which is what
// makes a partly populated bundle reachable at all.
class EagerBundleWriter : public Observer {
public:
    explicit EagerBundleWriter(std::string label) : label_(std::move(label)) {}

    void observe(const ObservationContext& ctx) override {
        if (ctx.bundle != nullptr && !written_) {
            ctx.bundle->put(label_, 1.0);
            written_ = true;
        }
    }

private:
    std::string label_;
    bool written_ = false;
};

// Throws from the middle of the run, after another observer has already written.
class ThrowingObserver : public Observer {
public:
    explicit ThrowingObserver(int throw_on) : throw_on_(throw_on) {}

    void observe(const ObservationContext&) override {
        if (++calls_ >= throw_on_) {
            throw std::runtime_error("ThrowingObserver: as designed");
        }
    }

private:
    int throw_on_;
    int calls_ = 0;
};

}  // namespace

// =============================================================================
// The hook is public surface
// =============================================================================

TEST(V11262Preflight, ACallerWrittenObserverIsConsultedBeforeTheRun) {
    auto obs = std::make_shared<CountingObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(obs->preflights(), 1);
    EXPECT_EQ(obs->seen_form(), StateForm::Statevector);
    EXPECT_EQ(obs->seen_qubits(), 4);
}

TEST(V11262Preflight, TheHookIsAskedOncePerObserverNotOncePerAnchor) {
    // An observer on three anchors is ONE observer with one label. Asking it
    // three times would repeat its work and, worse, would make it collide with
    // itself over the label it claims.
    auto obs = std::make_shared<CountingObserver>(true, "shared");
    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), obs);
    plan.observations.observe(Anchor::at_end(), obs);
    plan.observations.observe(Anchor::every_instruction(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(obs->preflights(), 1);
    // Still fires at every anchor it was attached to: 1 + 1 + 6.
    EXPECT_EQ(obs->observations(), 8);
}

TEST(V11262Preflight, TheBackendsFormReachesTheHook) {
    const QuantumCircuit qc(2);
    {
        auto obs = std::make_shared<CountingObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), obs);
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        sim.run(qc, noise, 4, 20261, plan);
        EXPECT_EQ(obs->seen_form(), StateForm::DensityMatrix);
    }
    {
        auto obs = std::make_shared<CountingObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), obs);
        CliffordSimulator sim;
        sim.run(qc, 4, 20261, plan);
        EXPECT_EQ(obs->seen_form(), StateForm::Stabilizer);
    }
    {
        auto obs = std::make_shared<CountingObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), obs);
        MPSSimulator sim;
        sim.run(qc, 8, 4, 20261, plan);
        EXPECT_EQ(obs->seen_form(), StateForm::MPS);
    }
}

TEST(V11262Preflight, AnObserverThatDoesNotOverrideTheHookStillRuns) {
    // The default answers yes. An observer with nothing to decide early is not
    // required to know the hook exists.
    auto rec = recorder();
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), rec);

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(rec->count(), 6u);
}

// =============================================================================
// A ruled-out observer is dropped, not merely silenced
// =============================================================================

TEST(V11262Preflight, AnObserverRuledOutNeverFiresAtAll) {
    // Where the saving actually is. Refusing at every firing and refusing once
    // produce the same empty result, and only one of them stops doing the work.
    auto refused = std::make_shared<CountingObserver>(/*accept=*/false);
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), refused);

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(refused->preflights(), 1);
    EXPECT_EQ(refused->observations(), 0);
}

TEST(V11262Preflight, RulingOneObserverOutLeavesTheRestOfThePlanAlone) {
    auto refused = std::make_shared<CountingObserver>(/*accept=*/false);
    auto kept = std::make_shared<CountingObserver>(/*accept=*/true);

    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), refused);
    plan.observations.observe(Anchor::every_instruction(), kept);

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(refused->observations(), 0);
    EXPECT_EQ(kept->observations(), 6);
}

TEST(V11262Preflight, APlanWhoseEveryObserverIsRuledOutStillRuns) {
    auto refused = std::make_shared<CountingObserver>(/*accept=*/false);
    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), refused);
    plan.observations.observe(Anchor::at_end(), refused);

    const QuantumCircuit qc = layered_circuit();
    StatevectorSimulator sim;
    auto watched = sim.run(qc, 128, 20261, plan);
    auto unwatched = sim.run(qc, 128, 20261);

    ASSERT_TRUE(watched.success) << watched.error_message;
    EXPECT_EQ(refused->observations(), 0);
    EXPECT_EQ(watched.counts, unwatched.counts);
}

// =============================================================================
// The boundary: deciding early changes when, never what
// =============================================================================

TEST(V11262Preflight, WarnStillAbsorbsARefusalDecidedEarly) {
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
    EXPECT_FALSE(msgs.empty());
}

TEST(V11262Preflight, IgnoreStillAbsorbsARefusalDecidedEarly) {
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

TEST(V11262Preflight, TheMpsGuardIsStillDecidedAtFiringTime) {
    // The one cost the pre-flight deliberately does not decide. An MPS grows as
    // it entangles, so the same request is genuinely refused early in a run and
    // allowed later, and moving that decision forward would answer it wrongly
    // at one end or the other.
    QuantumCircuit qc(3);
    qc.h(0);
    qc.cx(0, 1);
    qc.cx(1, 2);

    auto early = std::make_shared<StateObserver>(StateForm::Statevector);
    auto late = std::make_shared<StateObserver>(StateForm::Statevector);

    RunPlan plan;
    plan.options.response = Response::Ignore;
    plan.options.cost = Cost::Guarded;
    plan.options.guard_multiple = 1.0;
    plan.observations.observe(Anchor::at_start(), early);
    plan.observations.observe(Anchor::at_end(), late);

    MPSSimulator sim;
    sim.run(qc, 8, 0, 20261, plan);

    EXPECT_EQ(early->count(), 0u);
    EXPECT_EQ(late->count(), 1u);
}

// =============================================================================
// Two observers cannot claim one label
// =============================================================================

TEST(V11262Preflight, TwoObserversSharingALabelAreRefusedBeforeTheRun) {
    auto witness = recorder();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("same"));
    plan.observations.observe(Anchor::at_end(), std::make_shared<ProbabilityObserver>("same"));
    plan.observations.observe(Anchor::every_instruction(), witness);

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error_message.find("same"), std::string::npos) << r.error_message;
    EXPECT_EQ(witness->count(), 0u);
}

TEST(V11262Preflight, OneObserverMayHoldItsLabelOnManyAnchors) {
    // The counterpart, and the reason the check is per observer: this is not a
    // collision and refusing it would make a labelled trace impossible.
    auto obs = std::make_shared<PurityObserver>("trace");
    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), obs);
    plan.observations.observe(Anchor::every_instruction(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(obs->count(), 7u);
    EXPECT_EQ(r.observations.size(), 7u);
}

TEST(V11262Preflight, UnlabelledObserversDoNotCollideWithEachOther) {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>());
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>());
    plan.observations.observe(Anchor::at_end(), std::make_shared<ProbabilityObserver>());

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.observations.size(), 0u);
}

// =============================================================================
// A failed run hands back nothing
// =============================================================================

TEST(V11262Preflight, AFailedRunLeavesNoObservationsBehind) {
    // The first observer writes into the bundle as it fires, so by the time the
    // second one throws there are real entries in it. A caller checking the
    // flag and a caller reading the bundle must not be told different things.
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<EagerBundleWriter>("written"));
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<ThrowingObserver>(3));

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_FALSE(r.success);
    EXPECT_EQ(r.observations.size(), 0u);
    EXPECT_FALSE(r.observations.contains("written"));
}

TEST(V11262Preflight, AFailedDensityMatrixRunLeavesNoObservationsBehind) {
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<EagerBundleWriter>("written"));
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<ThrowingObserver>(2));

    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);

    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(qc, noise, 4, 20261, plan);

    ASSERT_FALSE(r.success);
    EXPECT_EQ(r.observations.size(), 0u);
}

TEST(V11262Preflight, ASuccessfulRunStillReturnsEverythingItCollected) {
    // The other half: clearing on failure must not clear on success.
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("purity"));
    plan.observations.observe(Anchor::at_end(), std::make_shared<StateObserver>("state"));
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<EagerBundleWriter>("eager"));

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.observations.size(), 3u);
    EXPECT_TRUE(r.observations.contains("purity"));
    EXPECT_TRUE(r.observations.contains("state"));
    EXPECT_TRUE(r.observations.contains("eager"));
}

// =============================================================================
// An observer that throws fails the run, on every backend
// =============================================================================
// Three of the four backends fire their per-instruction anchors from a
// destructor, which is how an instruction the backend SKIPS still fires. A
// destructor is noexcept, so an observer throwing there ended the process
// rather than the run: no message, no failed Result, nothing a caller could
// catch or report. A bug in a caller's own observer is the likeliest way to
// reach this, and it is the one case where the harness could take the whole
// program down with it.

TEST(V11262Preflight, AThrowingObserverFailsTheStatevectorRun) {
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<ThrowingObserver>(2));

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error_message.find("as designed"), std::string::npos)
        << r.error_message;
}

TEST(V11262Preflight, AThrowingObserverFailsTheDensityMatrixRun) {
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<ThrowingObserver>(2));

    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);

    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(qc, noise, 4, 20261, plan);

    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error_message.find("as designed"), std::string::npos)
        << r.error_message;
}

TEST(V11262Preflight, AThrowingObserverFailsTheCliffordRun) {
    // No error channel on this Result, so the failure arrives as a throw. What
    // matters either way is that it arrives at all rather than calling
    // std::terminate on the way out of a destructor.
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<ThrowingObserver>(2));

    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);

    CliffordSimulator sim;
    EXPECT_THROW(sim.run(qc, 4, 20261, plan), std::runtime_error);
}

TEST(V11262Preflight, AThrowingObserverFailsTheMpsRun) {
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<ThrowingObserver>(2));

    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);

    MPSSimulator sim;
    EXPECT_THROW(sim.run(qc, 8, 4, 20261, plan), std::runtime_error);
}

TEST(V11262Preflight, AThrowingObserverOnTheLastInstructionStillFailsTheRun) {
    // The failure is held until somewhere it can be raised from, so the case
    // with no following instruction is the one that could quietly vanish.
    const QuantumCircuit qc = layered_circuit();
    RunPlan plan;
    plan.observations.observe(
        Anchor::every_instruction(),
        std::make_shared<ThrowingObserver>(static_cast<int>(qc.instructions.size())));

    StatevectorSimulator sim;
    auto r = sim.run(qc, 0, 20261, plan);

    EXPECT_FALSE(r.success);
}

// =============================================================================
// Which fault is reported when a plan has several
// =============================================================================

TEST(V11262Preflight, AnUnresolvableAnchorIsReportedBeforeAnyObserverFault) {
    // Both are faults in the same plan. The anchor is reported because it is the
    // more fundamental one: an anchor naming nothing is broken whatever is
    // attached to it, and a caller told about the observer first would fix that
    // and meet the anchor on the next run.
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(),
        std::make_shared<AmplitudeObserver>(std::vector<std::size_t>{999}, "amps"));
    plan.observations.observe(Anchor::after_instruction(99), recorder());

    StatevectorSimulator sim;
    auto r = sim.run(layered_circuit(), 0, 20261, plan);

    ASSERT_FALSE(r.success);
    EXPECT_NE(r.error_message.find("after_instruction(99)"), std::string::npos)
        << r.error_message;
}

TEST(V11262Preflight, AnAmplitudeIndexOutsideTheRegisterIsACallerMistakeUnderEveryResponse) {
    // Not an observation the backend cannot produce, so no response softens it.
    for (const Response response : {Response::Throw, Response::Warn, Response::Ignore}) {
        RunPlan plan;
        plan.options.response = response;
        plan.observations.observe(
            Anchor::at_end(),
            std::make_shared<AmplitudeObserver>(std::vector<std::size_t>{64}, "amps"));

        StatevectorSimulator sim;
        auto r = sim.run(layered_circuit(), 0, 20261, plan);
        EXPECT_FALSE(r.success);
    }
}

TEST(V11262Preflight, AMalformedEntropyRegionIsACallerMistakeUnderEveryResponse) {
    for (const Response response : {Response::Throw, Response::Warn, Response::Ignore}) {
        RunPlan plan;
        plan.options.response = response;
        plan.observations.observe(
            Anchor::at_end(),
            std::make_shared<EntropyObserver>(std::vector<int>{1, 1}));

        StatevectorSimulator sim;
        auto r = sim.run(layered_circuit(), 0, 20261, plan);
        EXPECT_FALSE(r.success);
    }
}
