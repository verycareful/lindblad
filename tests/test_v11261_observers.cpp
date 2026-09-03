// 1.1.26.1 test wave - the built-in observer catalogue.
//
// The claim these have to earn is that they are ORDINARY implementations of a
// public interface with no privileged access, so that an observer written by a
// caller sits beside them rather than beneath them. A suite exercising only the
// built-ins would leave that claim untested however many of them it covered, so
// the last section of this file defines observers of its own, using nothing the
// library does not expose, and asks the same things of them.
//
// The catalogue is also where the reading is checked against an independent
// route wherever one exists: ProbabilityObserver against the backend's own
// probabilities(), ExpectationObserver against SparsePauliOp evaluated on the
// final state, PurityObserver against DensityMatrix::purity(). A reading that
// only agreed with itself would be evidence of nothing.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

using v11261::capture_warnings;
using v11261::layered_circuit;

namespace {

QuantumCircuit bell() {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    return qc;
}

constexpr double kTol = 1e-9;

}  // namespace

// =============================================================================
// StateObserver
// =============================================================================

TEST(V11261Observers, StateObserverCapturesTheNativeFormOnEveryBackend) {
    {
        auto obs = std::make_shared<StateObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), obs);
        StatevectorSimulator sim;
        auto r = sim.run(bell(), 0, 20261, plan);
        ASSERT_TRUE(r.success) << r.error_message;
        EXPECT_EQ(obs->form(), StateForm::Statevector);
        EXPECT_NEAR(obs->statevector().probability(3), 1.0 / 2.0, kTol);
    }
    {
        auto obs = std::make_shared<StateObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), obs);
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        auto r = sim.run(bell(), noise, 8, 20261, plan);
        ASSERT_TRUE(r.success) << r.error_message;
        EXPECT_EQ(obs->form(), StateForm::DensityMatrix);
        EXPECT_NEAR(obs->density_matrix().trace(), 1.0, kTol);
    }
    {
        auto obs = std::make_shared<StateObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), obs);
        CliffordSimulator sim;
        sim.run(bell(), 8, 20261, plan);
        EXPECT_EQ(obs->form(), StateForm::Stabilizer);
        EXPECT_EQ(obs->stabilizer().n_qubits, 2);
    }
    {
        auto obs = std::make_shared<StateObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), obs);
        MPSSimulator sim;
        sim.run(bell(), 8, 8, 20261, plan);
        EXPECT_EQ(obs->form(), StateForm::MPS);
        EXPECT_EQ(obs->mps().n_qubits, 2);
    }
}

TEST(V11261Observers, StateObserverConvertsWhenAskedForAReachableForm) {
    auto obs = std::make_shared<StateObserver>(StateForm::Statevector);
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::at_end(), obs);

    CliffordSimulator sim;
    sim.run(bell(), 8, 20261, plan);

    // The tableau route to dense amplitudes, read through the catalogue.
    ASSERT_EQ(obs->count(), 1u);
    EXPECT_EQ(obs->form(), StateForm::Statevector);
    EXPECT_NEAR(obs->statevector().probability(0), 1.0 / 2.0, kTol);
    EXPECT_NEAR(obs->statevector().probability(3), 1.0 / 2.0, kTol);
}

TEST(V11261Observers, StateObserverRefusesTheWrongTypedAccessor) {
    auto obs = std::make_shared<StateObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);
    StatevectorSimulator sim;
    sim.run(bell(), 0, 20261, plan);

    EXPECT_THROW(obs->density_matrix(), std::invalid_argument);
    EXPECT_THROW(obs->stabilizer(), std::invalid_argument);
    // Reading past what was captured names the count rather than crashing.
    EXPECT_THROW(obs->statevector(5), std::invalid_argument);
    EXPECT_THROW(obs->form(5), std::invalid_argument);
}

// =============================================================================
// ProbabilityObserver
// =============================================================================

TEST(V11261Observers, ProbabilityObserverAgreesWithTheBackendsOwnProbabilities) {
    auto obs = std::make_shared<ProbabilityObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    const std::vector<double> reference = r.final_state.probabilities();
    ASSERT_EQ(obs->probabilities().size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(obs->probabilities()[i], reference[i], kTol) << "index " << i;
    }
}

TEST(V11261Observers, ProbabilityObserverAveragesEntrywiseAcrossFirings) {
    // Two firings on one qubit: |0> at the start and (|0>+|1>)/sqrt(2) at the
    // end, so the entrywise mean is (1 + 0.5)/2 and (0 + 0.5)/2, derived rather
    // than transcribed.
    QuantumCircuit qc(1);
    qc.h(0);

    auto obs = std::make_shared<ProbabilityObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), obs);
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    sim.run(qc, 0, 20261, plan);

    ASSERT_EQ(obs->count(), 2u);
    const std::vector<double> mean = obs->average();
    ASSERT_EQ(mean.size(), 2u);
    EXPECT_NEAR(mean[0], (1.0 + 0.5) / 2.0, kTol);
    EXPECT_NEAR(mean[1], (0.0 + 0.5) / 2.0, kTol);
}

TEST(V11261Observers, ProbabilityObserverAverageOfNothingIsEmpty) {
    const ProbabilityObserver obs;
    EXPECT_TRUE(obs.average().empty());
}

// =============================================================================
// AmplitudeObserver
// =============================================================================

TEST(V11261Observers, AmplitudeObserverReturnsTheNamedIndicesInOrder) {
    auto obs = std::make_shared<AmplitudeObserver>(std::vector<std::size_t>{3, 0});
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    sim.run(bell(), 0, 20261, plan);

    ASSERT_EQ(obs->count(), 1u);
    ASSERT_EQ(obs->amplitudes().size(), 2u);
    // Asked for 3 then 0, so they come back in that order and not sorted.
    EXPECT_NEAR(obs->amplitudes()[0].norm(), INV_SQRT2, kTol);
    EXPECT_NEAR(obs->amplitudes()[1].norm(), INV_SQRT2, kTol);
    EXPECT_EQ(obs->indices(), (std::vector<std::size_t>{3, 0}));
}

// =============================================================================
// ExpectationObserver
// =============================================================================

TEST(V11261Observers, ExpectationObserverMatchesTheIndependentEvaluation) {
    // ZZ on a Bell pair is +1, and the observer's answer is checked against
    // SparsePauliOp evaluated on the returned state rather than against a
    // number written here.
    const SparsePauliOp zz({PauliString("ZZ")});

    auto obs = std::make_shared<ExpectationObserver>(zz, "zz");
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    EXPECT_NEAR(obs->value(), zz.expectation_value(r.final_state), kTol);
    EXPECT_NEAR(obs->value(), 1.0, kTol);
    EXPECT_NEAR(r.observations.number("zz"), obs->value(), kTol);
}

TEST(V11261Observers, ExpectationObserverOnTheTableauAgreesWithTheStatevector) {
    // The Clifford route sums term by term over the tableau and never expands
    // to amplitudes, so agreement here is two independent computations meeting.
    const SparsePauliOp zz({PauliString("ZZ")});

    auto tableau = std::make_shared<ExpectationObserver>(zz);
    RunPlan clifford_plan;
    clifford_plan.observations.observe(Anchor::at_end(), tableau);
    CliffordSimulator clifford;
    clifford.run(bell(), 8, 20261, clifford_plan);

    auto dense = std::make_shared<ExpectationObserver>(zz);
    RunPlan sv_plan;
    sv_plan.observations.observe(Anchor::at_end(), dense);
    StatevectorSimulator sv;
    sv.run(bell(), 0, 20261, sv_plan);

    ASSERT_EQ(tableau->count(), 1u);
    ASSERT_EQ(dense->count(), 1u);
    EXPECT_NEAR(tableau->value(), dense->value(), kTol);
}

TEST(V11261Observers, ExpectationObserverReportsMeanAndVarianceAcrossFirings) {
    // <Z> on one qubit: 1 before the H and 0 after it. Mean and variance are
    // derived from those two, not transcribed.
    QuantumCircuit qc(1);
    qc.h(0);
    const SparsePauliOp z({PauliString("Z")});

    auto obs = std::make_shared<ExpectationObserver>(z);
    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), obs);
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    sim.run(qc, 0, 20261, plan);

    ASSERT_EQ(obs->count(), 2u);
    const double a = obs->value(0);
    const double b = obs->value(1);
    const double mean = (a + b) / 2.0;
    EXPECT_NEAR(obs->mean(), mean, kTol);
    EXPECT_NEAR(obs->variance(),
                ((a - mean) * (a - mean) + (b - mean) * (b - mean)) / 2.0, kTol);
}

// =============================================================================
// PurityObserver
// =============================================================================

TEST(V11261Observers, PurityIsExactlyOneOnTheThreePureBackends) {
    {
        auto obs = std::make_shared<PurityObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::every_instruction(), obs);
        StatevectorSimulator sim;
        sim.run(bell(), 0, 20261, plan);
        for (const double p : obs->values()) EXPECT_NEAR(p, 1.0, kTol);
    }
    {
        auto obs = std::make_shared<PurityObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::every_instruction(), obs);
        CliffordSimulator sim;
        sim.run(bell(), 8, 20261, plan);
        ASSERT_GT(obs->count(), 0u);
        for (const double p : obs->values()) EXPECT_NEAR(p, 1.0, kTol);
    }
    {
        auto obs = std::make_shared<PurityObserver>();
        RunPlan plan;
        plan.observations.observe(Anchor::every_instruction(), obs);
        MPSSimulator sim;
        sim.run(bell(), 8, 8, 20261, plan);
        ASSERT_GT(obs->count(), 0u);
        for (const double p : obs->values()) EXPECT_NEAR(p, 1.0, kTol);
    }
}

TEST(V11261Observers, PurityFallsBelowOneUnderNoise) {
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.2), "h");

    auto obs = std::make_shared<PurityObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    DensityMatrixSimulator sim;
    auto r = sim.run(bell(), noise, 8, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    ASSERT_EQ(obs->count(), 1u);
    EXPECT_LT(obs->value(), 1.0);
    // Checked against the backend's own figure on the state it returned.
    EXPECT_NEAR(obs->value(), r.final_state.purity(), kTol);
}

// =============================================================================
// ClassicalRegisterObserver
// =============================================================================

TEST(V11261Observers, TheClassicalRegisterIsReadAsItStoodMidRun) {
    // Qubit 0 is flipped and measured, so clbit 0 is 1 from instruction 1
    // onward. The observer must see that at instruction 1, not the register's
    // final value handed back after the run.
    QuantumCircuit qc(2, 2);
    qc.x(0);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});
    qc.measure(1, 1);

    auto before = std::make_shared<ClassicalRegisterObserver>();
    auto after = std::make_shared<ClassicalRegisterObserver>();

    RunPlan plan;
    plan.observations.observe(Anchor::at_start(), before);
    plan.observations.observe(Anchor::after_instruction(1), after);

    StatevectorSimulator sim;
    auto r = sim.run(qc, 1, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    ASSERT_EQ(before->count(), 1u);
    ASSERT_EQ(after->count(), 1u);
    EXPECT_EQ(before->clbits()[0], 0);
    EXPECT_EQ(after->clbits()[0], 1);
}

// =============================================================================
// BondDimensionObserver / TruncationObserver
// =============================================================================

TEST(V11261Observers, BondDimensionsAreReportedOnAnMps) {
    auto obs = std::make_shared<BondDimensionObserver>("bonds");
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    MPSSimulator sim;
    auto r = sim.run(bell(), 8, 8, 20261, plan);

    ASSERT_EQ(obs->count(), 1u);
    EXPECT_EQ(obs->bond_dimensions().size(), 2u);
    // A Bell pair is exactly bond dimension 2 across its one cut.
    EXPECT_EQ(obs->bond_dimensions()[0], 2);
    EXPECT_TRUE(r.observations.contains("bonds"));
}

TEST(V11261Observers, BondAndTruncationObserversRefuseTheOtherBackends) {
    // Neither quantity EXISTS off an MPS, so this is impossibility rather than
    // expense, and it arrives through the response knob like any other refusal.
    for (const Response response : {Response::Warn, Response::Ignore}) {
        auto bonds = std::make_shared<BondDimensionObserver>();
        auto trunc = std::make_shared<TruncationObserver>();

        RunPlan plan;
        plan.options.response = response;
        plan.observations.observe(Anchor::at_end(), bonds);
        plan.observations.observe(Anchor::at_end(), trunc);

        StatevectorSimulator sim;
        auto r = sim.run(bell(), 0, 20261, plan);

        EXPECT_TRUE(r.success) << r.error_message;
        EXPECT_EQ(bonds->count(), 0u);
        EXPECT_EQ(trunc->count(), 0u);
    }

    auto bonds = std::make_shared<BondDimensionObserver>();
    RunPlan plan;
    plan.options.response = Response::Throw;
    plan.observations.observe(Anchor::at_end(), bonds);
    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);
    EXPECT_FALSE(r.success);
}

TEST(V11261Observers, TruncationIsReportedOnAnMps) {
    auto obs = std::make_shared<TruncationObserver>("discarded");
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    MPSSimulator sim;
    // Room for the state, so nothing is discarded.
    sim.run(bell(), 8, 8, 20261, plan);

    ASSERT_EQ(obs->count(), 1u);
    EXPECT_NEAR(obs->values()[0], 0.0, DEFAULT_PHYSICAL_ATOL);
}

// =============================================================================
// CallbackObserver
// =============================================================================

TEST(V11261Observers, TheCallbackReceivesAFullyFormedContext) {
    int calls = 0;
    int seen_index = -99;
    int seen_shot = -99;
    int seen_qubits = -99;
    std::string seen_anchor;

    auto obs = std::make_shared<CallbackObserver>([&](const ObservationContext& ctx) {
        ++calls;
        seen_index = ctx.instruction_index;
        seen_shot = ctx.shot;
        seen_qubits = ctx.state.n_qubits();
        seen_anchor = ctx.anchor;
        EXPECT_EQ(ctx.state.form(), StateForm::Statevector);
        EXPECT_EQ(ctx.n_shots, 1);
    });

    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);
    StatevectorSimulator sim;
    sim.run(bell(), 0, 20261, plan);

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(seen_index, 1);
    EXPECT_EQ(seen_shot, 0);
    EXPECT_EQ(seen_qubits, 2);
    EXPECT_EQ(seen_anchor, "at_end");
}

TEST(V11261Observers, ANullCallbackIsRejectedAtConstruction) {
    EXPECT_THROW(CallbackObserver(CallbackObserver::Fn{}), std::invalid_argument);
}

// =============================================================================
// An observer the library does not ship
// =============================================================================

namespace {

// Uses only the public interface: no friendship, no library change, no
// registration anywhere. If this works, the catalogue has no privileged access.
class HeaviestOutcomeObserver : public Observer {
public:
    explicit HeaviestOutcomeObserver(std::string label) : label_(std::move(label)) {}

    void begin_run(int n_qubits, int n_shots) override {
        qubits_ = n_qubits;
        shots_ = n_shots;
    }

    void observe(const ObservationContext& ctx) override {
        const std::vector<double> probs = ctx.state.statevector().probabilities();
        std::size_t best = 0;
        for (std::size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > probs[best]) best = i;
        }
        heaviest_.push_back(static_cast<int>(best));

        // Writing into the bundle is public surface too, so a caller-written
        // observer reaches the same road out of the run.
        if (ctx.bundle != nullptr) ctx.bundle->put(label_, heaviest_);
    }

    const std::vector<int>& heaviest() const { return heaviest_; }
    int qubits() const { return qubits_; }
    int shots() const { return shots_; }

private:
    std::string label_;
    std::vector<int> heaviest_;
    int qubits_ = -1;
    int shots_ = -1;
};

}  // namespace

TEST(V11261Observers, AnObserverWrittenHereIsAFirstClassCitizen) {
    QuantumCircuit qc(2);
    qc.x(0);  // |01> in bitstring terms, amplitude index 1

    auto obs = std::make_shared<HeaviestOutcomeObserver>("heaviest");
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(qc, 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(obs->qubits(), 2);
    EXPECT_EQ(obs->shots(), 1);
    ASSERT_EQ(obs->heaviest().size(), 1u);
    EXPECT_EQ(obs->heaviest()[0], 1);

    ASSERT_TRUE(r.observations.contains("heaviest"));
    EXPECT_EQ(r.observations.integers("heaviest"), (std::vector<int>{1}));
}

// =============================================================================
// Reuse across runs
// =============================================================================

TEST(V11261Observers, AnObserverReusedForASecondRunAccumulates) {
    // begin_run clears nothing, so a second run appends to the first run's
    // readings rather than replacing them. Pinned because it is not documented
    // anywhere and a caller reusing an instance would otherwise discover it by
    // reading a value that belongs to a run they had finished with.
    auto obs = std::make_shared<PurityObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    sim.run(bell(), 0, 20261, plan);
    ASSERT_EQ(obs->count(), 1u);

    sim.run(bell(), 0, 20261, plan);
    EXPECT_EQ(obs->count(), 2u);
    EXPECT_EQ(obs->values().size(), 2u);
}
