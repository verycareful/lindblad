// 1.1.26.1 test wave - an anchor that cannot fire fails the run.
//
// This is the sharpest contract the observation harness makes. An observation
// that did not happen and an observation that happened and found nothing are
// the same empty result to a caller, so the library refuses to let the first
// one be silent: an anchor is resolved against the circuit at the pre-flight,
// before any state is touched, and one that resolves to nothing fails the run
// and names itself.
//
// The half worth testing hardest is that no policy softens it. Response::Warn
// and Response::Ignore govern an OBSERVATION the backend cannot produce, which
// is a capability question. An anchor that does not match its circuit is a
// mistake in the caller's code, and there is no reading of Ignore under which
// running a plan that does not describe the run is the helpful answer. Every
// failure below is therefore exercised under all three responses.
//
// HOW a failed run reports itself is a property of the backend and not of the
// failure. StatevectorSimulator and DensityMatrixSimulator carry an error
// channel on their Result and report through it; MPSSimulator and
// CliffordSimulator have no such field and throw. Both are asserted, each
// against the backend that uses it, through the two helpers in the oracle.
//
// Rejections that happen earlier still, at the point an Anchor or a plan is
// constructed, are pinned here too: those never reach a circuit at all, and a
// caller who gets a throw from the factory is told at the line that is wrong.

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
#include "lindblad/transpiler.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

using v11261::capture_warnings;
using v11261::label_instruction;
using v11261::layered_circuit;
using v11261::plan_with;
using v11261::RecorderPtr;
using v11261::recorder;
using v11261::sv_run_failure;
using v11261::throwing_run_failure;

namespace {

QuantumCircuit two_instructions() {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    return qc;
}

// A plan carrying one unresolvable anchor and the response knob under test.
RunPlan plan_with_response(Anchor anchor, Response response) {
    RunPlan plan = plan_with(std::move(anchor), recorder());
    plan.options.response = response;
    return plan;
}

const Response kAllResponses[] = {Response::Throw, Response::Warn, Response::Ignore};

bool mentions(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}

}  // namespace

// =============================================================================
// An index that names no instruction
// =============================================================================

TEST(V11261AnchorFailures, AnIndexEqualToTheInstructionCountFailsTheRun) {
    const QuantumCircuit qc = two_instructions();
    // Two instructions occupy 0 and 1, so 2 is the first index past the end and
    // the one an off-by-one produces.
    const std::string message =
        sv_run_failure(qc, plan_with(Anchor::after_instruction(2), recorder()));
    EXPECT_TRUE(mentions(message, "after_instruction(2)")) << message;
}

TEST(V11261AnchorFailures, AnIndexBeyondTheEndFailsTheRun) {
    const QuantumCircuit qc = two_instructions();
    const std::string message =
        sv_run_failure(qc, plan_with(Anchor::after_instruction(99), recorder()));
    EXPECT_TRUE(mentions(message, "after_instruction(99)")) << message;
}

TEST(V11261AnchorFailures, AnUnresolvableIndexFailsUnderEveryResponse) {
    const QuantumCircuit qc = two_instructions();
    for (const Response response : kAllResponses) {
        const std::string message =
            sv_run_failure(qc, plan_with_response(Anchor::after_instruction(7), response));
        EXPECT_TRUE(mentions(message, "after_instruction(7)")) << message;
    }
}

TEST(V11261AnchorFailures, AfterInstructionZeroDoesNotResolveOnAnEmptyCircuit) {
    const QuantumCircuit qc(2);
    // Index 0 is inside every circuit that has an instruction, and this one has
    // none, so the same index that is always valid elsewhere fails here.
    const std::string message =
        sv_run_failure(qc, plan_with(Anchor::after_instruction(0), recorder()));
    EXPECT_TRUE(mentions(message, "after_instruction(0)")) << message;
}

// =============================================================================
// A label no instruction carries
// =============================================================================

TEST(V11261AnchorFailures, AnAbsentLabelFailsTheRun) {
    QuantumCircuit qc = layered_circuit();
    label_instruction(qc, 0, "present");

    const std::string message =
        sv_run_failure(qc, plan_with(Anchor::after_label("absent"), recorder()));
    EXPECT_TRUE(mentions(message, "after_label(absent)")) << message;
}

TEST(V11261AnchorFailures, AnAbsentLabelFailsUnderEveryResponse) {
    const QuantumCircuit qc = layered_circuit();
    for (const Response response : kAllResponses) {
        const std::string message =
            sv_run_failure(qc, plan_with_response(Anchor::after_label("absent"), response));
        EXPECT_TRUE(mentions(message, "after_label(absent)")) << message;
    }
}

TEST(V11261AnchorFailures, ALabelRemovedByATranspilerPassFailsTheRun) {
    // The pass that removes the instruction removes its label with it, which is
    // the realistic way a plan and its circuit come apart: the caller labelled
    // something the optimiser then proved unnecessary.
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    qc.cx(0, 1);
    label_instruction(qc, 2, "second_cx");

    const QuantumCircuit optimised = transpile(qc, CouplingMap(), {}, 1);

    // The precondition is asserted rather than assumed: if cancellation stopped
    // happening, this test must fail loudly instead of passing because the
    // label survived for a different reason.
    ASSERT_LT(optimised.instructions.size(), qc.instructions.size());
    for (const Instruction& inst : optimised.instructions) {
        ASSERT_NE(inst.label, "second_cx");
    }

    const std::string message =
        sv_run_failure(optimised, plan_with(Anchor::after_label("second_cx"), recorder()));
    EXPECT_TRUE(mentions(message, "after_label(second_cx)")) << message;
}

// =============================================================================
// What the failure looks like
// =============================================================================

TEST(V11261AnchorFailures, TheFailureSaysWhyTheAnchorCouldNotResolve) {
    const QuantumCircuit qc = two_instructions();
    const std::string message =
        sv_run_failure(qc, plan_with(Anchor::after_instruction(9), recorder()));

    // Naming the anchor is not enough on its own: the caller also has to be
    // told what the circuit actually held, or the next guess is as blind.
    EXPECT_TRUE(mentions(message, "after_instruction(9)")) << message;
    EXPECT_TRUE(mentions(message, "2")) << message;
}

TEST(V11261AnchorFailures, TheFailureSaysWhyTheLabelCouldNotResolve) {
    const QuantumCircuit qc = layered_circuit();
    const std::string message =
        sv_run_failure(qc, plan_with(Anchor::after_label("qft_done"), recorder()));

    EXPECT_TRUE(mentions(message, "after_label(qft_done)")) << message;
    EXPECT_TRUE(mentions(message, "label")) << message;
}

TEST(V11261AnchorFailures, NothingIsObservedBeforeTheFailure) {
    const QuantumCircuit qc = two_instructions();
    RecorderPtr resolvable = recorder();

    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), resolvable);
    plan.observations.observe(Anchor::after_instruction(50), recorder());

    sv_run_failure(qc, plan);

    // Resolution happens before any state is touched, so the anchor that WOULD
    // have fired never did. A partially observed run is worse than none: its
    // bundle would look like a complete answer.
    EXPECT_EQ(resolvable->count(), 0u);
    EXPECT_EQ(resolvable->begins(), 0);
}

TEST(V11261AnchorFailures, TheFailureIsNotAlsoDeliveredThroughTheWarningChannel) {
    const QuantumCircuit qc = two_instructions();

    const std::vector<std::string> msgs = capture_warnings([&] {
        sv_run_failure(qc, plan_with_response(Anchor::after_instruction(4), Response::Warn));
    });

    // Warn is not a downgrade path for this failure. The run fails, and it does
    // not also emit: a caller watching the channel must not see a note
    // suggesting the run continued.
    EXPECT_TRUE(msgs.empty()) << msgs.front();
}

TEST(V11261AnchorFailures, AFailedRunCollectsNoObservations) {
    const QuantumCircuit qc = two_instructions();

    RunPlan plan;
    plan.observations.observe(Anchor::at_start(),
                              std::make_shared<StateObserver>("start"));
    plan.observations.observe(Anchor::after_instruction(50), recorder());

    StatevectorSimulator sim;
    auto result = sim.run(qc, 0, 20261, plan);

    ASSERT_FALSE(result.success);
    // An empty bundle on a failed run is the point: a bundle holding the one
    // anchor that did resolve would read as a complete answer to a plan that
    // was never run.
    EXPECT_EQ(result.observations.size(), 0u);
}

// =============================================================================
// Rejections at construction, before any circuit exists
// =============================================================================

TEST(V11261AnchorFailures, AnEmptyLabelIsRejectedAtConstruction) {
    EXPECT_THROW(Anchor::after_label(""), std::invalid_argument);
}

TEST(V11261AnchorFailures, ANegativeInstructionIndexIsRejectedAtConstruction) {
    // Resolution against a circuit only checks the upper bound, so this guard
    // is the only thing standing between a caller and an anchor that could
    // never match any index the runner presents.
    EXPECT_THROW(Anchor::after_instruction(-1), std::invalid_argument);
    EXPECT_THROW(Anchor::after_instruction(-99), std::invalid_argument);
}

TEST(V11261AnchorFailures, AnEmptyPredicateIsRejectedAtConstruction) {
    EXPECT_THROW(Anchor::where(Anchor::PredicateFn{}), std::invalid_argument);
}

TEST(V11261AnchorFailures, ANullObserverIsRejected) {
    ObservationPlan plan;
    EXPECT_THROW(plan.observe(Anchor::at_start(), nullptr), std::invalid_argument);
    EXPECT_TRUE(plan.empty());
}

TEST(V11261AnchorFailures, TheEndpointsResolveOnAnEmptyCircuit) {
    // at_start and at_end name a point in the RUN rather than an instruction,
    // so they resolve against a circuit that has none. Only positional anchors
    // can fail to resolve.
    const QuantumCircuit qc(2);
    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), recorder());
    plan.observations.observe(Anchor::at_end(), recorder());
    plan.observations.observe(Anchor::every_instruction(), recorder());
    plan.observations.observe(Anchor::every_layer(), recorder());
    plan.observations.observe(Anchor::before_each_measurement(), recorder());
    plan.observations.observe(Anchor::after_each_measurement(), recorder());

    StatevectorSimulator sim;
    auto result = sim.run(qc, 0, 20261, plan);
    EXPECT_TRUE(result.success) << result.error_message;
}

// =============================================================================
// Every backend refuses, each through its own channel
// =============================================================================

TEST(V11261AnchorFailures, EveryBackendRefusesAnUnresolvableAnchor) {
    const QuantumCircuit qc = two_instructions();

    {
        const std::string message =
            sv_run_failure(qc, plan_with(Anchor::after_instruction(5), recorder()));
        EXPECT_TRUE(mentions(message, "after_instruction(5)")) << message;
    }
    {
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        auto result = sim.run(qc, noise, 16, 20261,
                              plan_with(Anchor::after_instruction(5), recorder()));
        ASSERT_FALSE(result.success);
        EXPECT_TRUE(mentions(result.error_message, "after_instruction(5)"))
            << result.error_message;
    }
    {
        const std::string message = throwing_run_failure([&] {
            CliffordSimulator sim;
            sim.run(qc, 16, 20261, plan_with(Anchor::after_instruction(5), recorder()));
        });
        EXPECT_TRUE(mentions(message, "after_instruction(5)")) << message;
    }
    {
        const std::string message = throwing_run_failure([&] {
            MPSSimulator sim;
            sim.run(qc, 8, 16, 20261, plan_with(Anchor::after_instruction(5), recorder()));
        });
        EXPECT_TRUE(mentions(message, "after_instruction(5)")) << message;
    }
}

TEST(V11261AnchorFailures, EveryBackendRefusesAnAbsentLabel) {
    const QuantumCircuit qc = two_instructions();

    {
        const std::string message =
            sv_run_failure(qc, plan_with(Anchor::after_label("nope"), recorder()));
        EXPECT_TRUE(mentions(message, "after_label(nope)")) << message;
    }
    {
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        auto result =
            sim.run(qc, noise, 16, 20261, plan_with(Anchor::after_label("nope"), recorder()));
        ASSERT_FALSE(result.success);
        EXPECT_TRUE(mentions(result.error_message, "after_label(nope)"))
            << result.error_message;
    }
    {
        const std::string message = throwing_run_failure([&] {
            CliffordSimulator sim;
            sim.run(qc, 16, 20261, plan_with(Anchor::after_label("nope"), recorder()));
        });
        EXPECT_TRUE(mentions(message, "after_label(nope)")) << message;
    }
    {
        const std::string message = throwing_run_failure([&] {
            MPSSimulator sim;
            sim.run(qc, 8, 16, 20261, plan_with(Anchor::after_label("nope"), recorder()));
        });
        EXPECT_TRUE(mentions(message, "after_label(nope)")) << message;
    }
}
