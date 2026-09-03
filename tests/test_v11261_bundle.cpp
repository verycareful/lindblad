// 1.1.26.1 test wave - the label-keyed bundle.
//
// Two roads lead out of an observed run. The typed one is to construct an
// observer, attach it, and read it afterwards, which is what a C++ caller
// wants. The other is the bundle, which is the shape that survives being
// copied, stored, or carried across a language boundary, and which a C++
// observer object does not. One class serves both, and giving it a label is
// what opens the second road.
//
// The key scheme is where this gets interesting. An observer that fired once
// writes under its plain label; one that fired repeatedly writes each firing
// under label@<instruction>#<shot>, so nothing overwrites anything. That means
// the KEY depends on how many times the observer fired, which is not known
// until the run ends, and the tests below pin both halves of that rule.
//
// A string-keyed store cannot fail at compile time, so it fails loudly at the
// point of the mistake instead: a missing label and a wrong payload kind are
// both errors rather than empty values.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

using v11261::layered_circuit;

namespace {

QuantumCircuit bell() {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    return qc;
}

}  // namespace

// =============================================================================
// The two key schemes
// =============================================================================

TEST(V11261Bundle, OneFiringWritesThePlainLabel) {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("purity"));

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_TRUE(r.observations.contains("purity"));
    EXPECT_EQ(r.observations.size(), 1u);
    EXPECT_NEAR(r.observations.number("purity"), 1.0, 1e-9);
}

TEST(V11261Bundle, SeveralFiringsWriteQualifiedKeysAndOverwriteNothing) {
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<PurityObserver>("purity"));

    const QuantumCircuit qc = layered_circuit();
    StatevectorSimulator sim;
    auto r = sim.run(qc, 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    // Six firings, six entries. The plain label is NOT among them: a caller
    // reading "purity" here would be reading one arbitrary firing as though it
    // described the run.
    EXPECT_EQ(r.observations.size(), qc.instructions.size());
    EXPECT_FALSE(r.observations.contains("purity"));
    for (int i = 0; i < static_cast<int>(qc.instructions.size()); ++i) {
        const std::string key = "purity@" + std::to_string(i) + "#0";
        EXPECT_TRUE(r.observations.contains(key)) << key;
    }
}

TEST(V11261Bundle, TheQualifiedKeyCarriesTheShotAsWellAsTheInstruction) {
    // Two shots on a feedforward circuit, so the same instruction index recurs
    // and only the shot tells the entries apart.
    QuantumCircuit qc(2, 2);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});

    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(),
                              std::make_shared<PurityObserver>("p"));

    StatevectorSimulator sim;
    auto r = sim.run(qc, 2, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.observations.size(), 4u);
    EXPECT_TRUE(r.observations.contains("p@0#0"));
    EXPECT_TRUE(r.observations.contains("p@1#0"));
    EXPECT_TRUE(r.observations.contains("p@0#1"));
    EXPECT_TRUE(r.observations.contains("p@1#1"));
}

TEST(V11261Bundle, LabelsComeBackSorted) {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("zulu"));
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("alpha"));
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("mike"));

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    // The store is a hash map, so without the sort the order would vary by
    // build and by insertion history rather than by anything a caller controls.
    EXPECT_EQ(r.observations.labels(),
              (std::vector<std::string>{"alpha", "mike", "zulu"}));
}

TEST(V11261Bundle, AnUnlabelledObserverCollectsItsOwnResultsAndWritesNothing) {
    auto observer = std::make_shared<PurityObserver>();  // no label
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), observer);

    const QuantumCircuit qc = layered_circuit();
    StatevectorSimulator sim;
    auto r = sim.run(qc, 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_FALSE(observer->labelled());
    EXPECT_EQ(observer->count(), qc.instructions.size());
    EXPECT_EQ(observer->values().size(), qc.instructions.size());
    EXPECT_EQ(r.observations.size(), 0u);
}

// =============================================================================
// Reading it
// =============================================================================

TEST(V11261Bundle, AMissingLabelThrowsRatherThanReturningEmpty) {
    ObservationBundle bundle;
    bundle.put("here", 1.0);

    EXPECT_THROW(bundle.number("absent"), std::invalid_argument);
    EXPECT_THROW(bundle.reals("absent"), std::invalid_argument);
    EXPECT_THROW(bundle.amplitudes("absent"), std::invalid_argument);
    EXPECT_THROW(bundle.integers("absent"), std::invalid_argument);
    EXPECT_THROW(bundle.text("absent"), std::invalid_argument);
    EXPECT_FALSE(bundle.contains("absent"));
}

TEST(V11261Bundle, TheWrongAccessorForAStoredKindThrows) {
    ObservationBundle bundle;
    bundle.put("number", 2.5);
    bundle.put("reals", std::vector<double>{1.0, 2.0});
    bundle.put("integers", std::vector<int>{1, 2});
    bundle.put("text", std::string("hello"));

    EXPECT_NEAR(bundle.number("number"), 2.5, 1e-12);
    EXPECT_THROW(bundle.reals("number"), std::invalid_argument);
    EXPECT_THROW(bundle.number("reals"), std::invalid_argument);
    EXPECT_THROW(bundle.integers("reals"), std::invalid_argument);
    EXPECT_THROW(bundle.text("integers"), std::invalid_argument);
    EXPECT_THROW(bundle.number("text"), std::invalid_argument);
}

TEST(V11261Bundle, ADuplicateLabelIsRefusedRatherThanOverwriting) {
    ObservationBundle bundle;
    bundle.put("x", 1.0);

    // Two observations under one name leave one of them unreachable, and the
    // caller cannot tell which, so neither is accepted quietly.
    EXPECT_THROW(bundle.put("x", 2.0), std::invalid_argument);
    EXPECT_NEAR(bundle.number("x"), 1.0, 1e-12);
    EXPECT_EQ(bundle.size(), 1u);
}

TEST(V11261Bundle, AnEmptyLabelIsRefused) {
    ObservationBundle bundle;
    EXPECT_THROW(bundle.put("", 1.0), std::invalid_argument);
    EXPECT_EQ(bundle.size(), 0u);
}

TEST(V11261Bundle, AStatePayloadReportsItsFormAndRefusesTheOthers) {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<StateObserver>("state"));

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    ASSERT_TRUE(r.observations.contains("state"));
    EXPECT_EQ(r.observations.form("state"), StateForm::Statevector);

    const Statevector& sv = r.observations.statevector("state");
    EXPECT_NEAR(sv.probability(0), 1.0 / 2.0, 1e-9);
    EXPECT_NEAR(sv.probability(3), 1.0 / 2.0, 1e-9);

    // form() says which accessor is valid, and the others say so rather than
    // reinterpreting the payload.
    EXPECT_THROW(r.observations.density_matrix("state"), std::invalid_argument);
    EXPECT_THROW(r.observations.stabilizer("state"), std::invalid_argument);
    EXPECT_THROW(r.observations.mps("state"), std::invalid_argument);
    EXPECT_THROW(r.observations.number("state"), std::invalid_argument);
}

// =============================================================================
// Surviving being moved and copied
// =============================================================================

TEST(V11261Bundle, TheBundleSurvivesBeingMovedOutOfTheResult) {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<StateObserver>("state"));
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("purity"));

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    // The claim is that this shape outlives the run and the observers that
    // filled it. State payloads are held by shared pointer, so the moved
    // bundle still owns its states.
    const ObservationBundle moved = std::move(r.observations);

    EXPECT_EQ(moved.size(), 2u);
    EXPECT_NEAR(moved.number("purity"), 1.0, 1e-9);
    EXPECT_NEAR(moved.statevector("state").probability(3), 1.0 / 2.0, 1e-9);
}

TEST(V11261Bundle, TheBundleSurvivesBeingCopied) {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<StateObserver>("state"));

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    const ObservationBundle copy = r.observations;

    // Both refer to the same state, which is the point of the shared pointer:
    // a copy costs a refcount rather than 2^n amplitudes.
    ASSERT_TRUE(copy.contains("state"));
    EXPECT_EQ(&copy.statevector("state"), &r.observations.statevector("state"));
    EXPECT_NEAR(copy.statevector("state").probability(0), 1.0 / 2.0, 1e-9);
}

TEST(V11261Bundle, TheDensityMatrixBackendCarriesABundleToo) {
    // Every Result gained one, so the road out of a run does not depend on
    // which backend ran it.
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<PurityObserver>("purity"));

    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(bell(), noise, 8, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_TRUE(r.observations.contains("purity"));
    EXPECT_NEAR(r.observations.number("purity"), 1.0, 1e-9);
}

TEST(V11261Bundle, AnEmptyPlanLeavesTheBundleEmpty) {
    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.observations.size(), 0u);
    EXPECT_TRUE(r.observations.labels().empty());
}
