// 1.1.26.1 test wave - watching a run does not change it.
//
// This is the headline claim of the release, and it is the one a caller cannot
// check for themselves: they have no unwatched run to compare against. Nature
// does not let a state be read without disturbance, and the whole argument for
// keeping observation out of the circuit is that a SIMULATION does. If watching
// perturbs, the design is not merely inconvenient, it is wrong.
//
// The claim is attacked from two directions. Through the outputs: a watched run
// and an unwatched one, same circuit and same seed, must agree on counts and on
// the final state. And through the state's own invariants at EVERY firing
// rather than at the end, because a single reading cannot tell a state that was
// never harmed from one that was harmed and renormalised.
//
// The subtlest failure here is not a physics error at all. A StateObserver that
// stored handles on the live state instead of snapshots would leave every
// captured entry pure, normalised, and equal to the FINAL state. Purity would
// be 1 everywhere and every reading would be wrong. The aliasing test is what
// separates those, and no invariant check can.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;

using v11261::layered_circuit;
using v11261::recorder;

namespace {

constexpr double kTol = 1e-12;
constexpr double kLoose = 1e-9;

// Long enough to cross the paths that matter: entangling gates, a barrier, and
// a conditioned gate whose condition never holds. All Clifford, so the same
// circuit runs on all four backends.
QuantumCircuit watched_circuit() {
    QuantumCircuit qc(3, 3);
    qc.h(0);
    qc.cx(0, 1);
    qc.barrier();
    qc.h(2);
    qc.cz(1, 2);
    qc.measure(0, 0);
    qc.add_if(1, 1, Instruction::GateType::X, {2});
    return qc;
}

QuantumCircuit bell() {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    return qc;
}

// Records the norm and the outcome distribution at every firing, so a state
// damaged mid-run is caught where it was damaged.
class InvariantSpy : public Observer {
public:
    void observe(const ObservationContext& ctx) override {
        const Statevector& sv = ctx.state.statevector();
        norms_.push_back(sv.norm());
        firings_.push_back(ctx.instruction_index);
    }

    const std::vector<double>& norms() const { return norms_; }
    std::size_t count() const { return norms_.size(); }

private:
    std::vector<double> norms_;
    std::vector<int> firings_;
};

}  // namespace

// =============================================================================
// Purity holds at every firing, not just at the end
// =============================================================================

TEST(V11261Purity, PurityIsOneAtEveryFiringOnTheStatevector) {
    auto obs = std::make_shared<PurityObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), obs);
    plan.observations.observe(Anchor::at_start(), obs);
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(watched_circuit(), 4, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    ASSERT_GT(obs->count(), 0u);
    for (std::size_t k = 0; k < obs->values().size(); ++k) {
        EXPECT_NEAR(obs->values()[k], 1.0, kLoose) << "firing " << k;
    }
}

TEST(V11261Purity, PurityIsOneAtEveryFiringOnTheTableau) {
    auto obs = std::make_shared<PurityObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), obs);

    CliffordSimulator sim;
    sim.run(watched_circuit(), 4, 20261, plan);

    ASSERT_GT(obs->count(), 0u);
    for (std::size_t k = 0; k < obs->values().size(); ++k) {
        EXPECT_NEAR(obs->values()[k], 1.0, kLoose) << "firing " << k;
    }
}

TEST(V11261Purity, PurityIsOneAtEveryFiringOnAnMps) {
    auto obs = std::make_shared<PurityObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), obs);

    MPSSimulator sim;
    sim.run(watched_circuit(), 16, 4, 20261, plan);

    ASSERT_GT(obs->count(), 0u);
    for (std::size_t k = 0; k < obs->values().size(); ++k) {
        EXPECT_NEAR(obs->values()[k], 1.0, kLoose) << "firing " << k;
    }
}

TEST(V11261Purity, TheNormIsOneAtEveryFiring) {
    auto spy = std::make_shared<InvariantSpy>();
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), spy);

    StatevectorSimulator sim;
    auto r = sim.run(watched_circuit(), 4, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    ASSERT_GT(spy->count(), 0u);
    for (std::size_t k = 0; k < spy->norms().size(); ++k) {
        EXPECT_NEAR(spy->norms()[k], 1.0, kLoose) << "firing " << k;
    }
}

TEST(V11261Purity, TheDensityMatrixKeepsUnitTraceAtEveryFiring) {
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.1), "h");

    std::vector<double> traces;
    auto spy = std::make_shared<CallbackObserver>([&traces](const ObservationContext& ctx) {
        traces.push_back(ctx.state.density_matrix().trace());
    });

    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), spy);

    DensityMatrixSimulator sim;
    auto r = sim.run(watched_circuit(), noise, 4, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    ASSERT_FALSE(traces.empty());
    // Noise mixes the state; it does not destroy probability.
    for (std::size_t k = 0; k < traces.size(); ++k) {
        EXPECT_NEAR(traces[k], 1.0, kLoose) << "firing " << k;
    }
}

TEST(V11261Purity, WatchingANoisyRunDoesNotChangeHowMixedItGets) {
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.2), "h");
    // Two qubits, because cx acts on two. A one-qubit channel attached to a
    // two-qubit gate is not rejected anywhere and corrupts the matrix instead.
    noise.add_quantum_error(NoiseChannels::depolarizing(0.1, 2), "cx");

    auto obs = std::make_shared<PurityObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), obs);

    DensityMatrixSimulator sim;
    auto watched = sim.run(bell(), noise, 8, 20261, plan);
    auto unwatched = sim.run(bell(), noise, 8, 20261);

    ASSERT_TRUE(watched.success) << watched.error_message;
    ASSERT_TRUE(unwatched.success) << unwatched.error_message;

    ASSERT_EQ(obs->count(), 1u);
    EXPECT_LT(obs->value(), 1.0);
    // Watching added no mixing and removed none.
    EXPECT_NEAR(obs->value(), unwatched.final_state.purity(), kLoose);
    EXPECT_NEAR(watched.final_state.purity(), unwatched.final_state.purity(), kLoose);
}

// =============================================================================
// A conversion does not touch what it read
// =============================================================================

TEST(V11261Purity, ConvertingToADensityMatrixLeavesTheStatevectorIntact) {
    auto converted = std::make_shared<StateObserver>(StateForm::DensityMatrix);
    auto after = std::make_shared<PurityObserver>();

    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::every_instruction(), converted);
    plan.observations.observe(Anchor::at_end(), after);

    StatevectorSimulator sim;
    auto watched = sim.run(bell(), 0, 20261, plan);
    auto unwatched = sim.run(bell(), 0, 20261);

    ASSERT_TRUE(watched.success) << watched.error_message;
    ASSERT_GT(converted->count(), 0u);

    // produce_state clones on every route. This pins the clone rather than
    // assuming it: a conversion that normalised or reused the live buffer
    // would show up here and nowhere else.
    EXPECT_NEAR(after->value(), 1.0, kLoose);
    EXPECT_NEAR(watched.final_state.norm(), 1.0, kLoose);
    for (std::size_t i = 0; i < watched.final_state.dim; ++i) {
        EXPECT_NEAR(watched.final_state.probability(i),
                    unwatched.final_state.probability(i), kLoose)
            << "amplitude " << i;
    }
}

TEST(V11261Purity, ConvertingTheTableauToAmplitudesLeavesTheTableauIntact) {
    auto converted = std::make_shared<StateObserver>(StateForm::Statevector);
    auto purity = std::make_shared<PurityObserver>();

    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::every_instruction(), converted);
    plan.observations.observe(Anchor::at_end(), purity);

    CliffordSimulator sim;
    auto watched = sim.run(watched_circuit(), 64, 20261, plan);
    auto unwatched = sim.run(watched_circuit(), 64, 20261);

    ASSERT_GT(converted->count(), 0u);
    EXPECT_NEAR(purity->value(), 1.0, kLoose);
    EXPECT_EQ(watched.counts, unwatched.counts);
}

// =============================================================================
// Snapshots are snapshots, not handles
// =============================================================================

TEST(V11261Purity, EachCapturedStateIsTheStateAsItStoodAtThatFiring) {
    // One qubit, and every instruction moves it somewhere distinguishable:
    //   after 0 (H)  : an even superposition
    //   after 1 (Z)  : still an even superposition, opposite relative phase
    //   after 2 (H)  : back to a basis state, |1>
    // Reading the captures AFTER the run has finished is the point. If the
    // observer held handles rather than copies, all three would now read as the
    // final state, and every one of them would still be pure and normalised.
    QuantumCircuit qc(1);
    qc.h(0);
    qc.z(0);
    qc.h(0);

    auto obs = std::make_shared<StateObserver>();
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(qc, 0, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;
    ASSERT_EQ(obs->count(), 3u);

    EXPECT_NEAR(obs->statevector(0).probability(0), 1.0 / 2.0, kLoose);
    EXPECT_NEAR(obs->statevector(1).probability(0), 1.0 / 2.0, kLoose);
    EXPECT_NEAR(obs->statevector(2).probability(1), 1.0, kLoose);

    // The first two agree on probabilities and differ in phase, so a capture
    // that silently kept only magnitudes would pass the lines above.
    const Complex128 a1 = obs->statevector(0).amplitude(1);
    const Complex128 b1 = obs->statevector(1).amplitude(1);
    EXPECT_GT((a1 - b1).norm(), 0.5);
}

TEST(V11261Purity, CapturedStatesOutliveTheRunThatProducedThem) {
    auto obs = std::make_shared<StateObserver>();
    {
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), obs);
        StatevectorSimulator sim;
        auto r = sim.run(bell(), 0, 20261, plan);
        ASSERT_TRUE(r.success) << r.error_message;
    }
    // The simulator, the plan and the Result are all gone. The capture is held
    // by shared pointer and is still readable, which is what makes it a
    // snapshot rather than a view.
    ASSERT_EQ(obs->count(), 1u);
    EXPECT_NEAR(obs->statevector().probability(0), 1.0 / 2.0, kLoose);
    EXPECT_NEAR(obs->statevector().probability(3), 1.0 / 2.0, kLoose);
}

// =============================================================================
// A watched run and an unwatched run are the same run
// =============================================================================

TEST(V11261Purity, AnEmptyPlanChangesNothingOnEveryBackend) {
    const QuantumCircuit qc = watched_circuit();
    const RunPlan empty;

    {
        StatevectorSimulator sim;
        auto with = sim.run(qc, 128, 20261, empty);
        auto without = sim.run(qc, 128, 20261);
        EXPECT_EQ(with.counts, without.counts);
    }
    {
        CliffordSimulator sim;
        auto with = sim.run(qc, 128, 20261, empty);
        auto without = sim.run(qc, 128, 20261);
        EXPECT_EQ(with.counts, without.counts);
    }
    {
        MPSSimulator sim;
        auto with = sim.run(qc, 16, 128, 20261, empty);
        auto without = sim.run(qc, 16, 128, 20261);
        EXPECT_EQ(with.counts, without.counts);
    }
    {
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        auto with = sim.run(qc, noise, 128, 20261, empty);
        auto without = sim.run(qc, noise, 128, 20261);
        EXPECT_EQ(with.counts, without.counts);
    }
}

TEST(V11261Purity, AWatchedStatevectorRunAgreesWithAnUnwatchedOne) {
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), recorder());

    const QuantumCircuit qc = watched_circuit();
    StatevectorSimulator sim;
    auto watched = sim.run(qc, 256, 20261, plan);
    auto unwatched = sim.run(qc, 256, 20261);

    ASSERT_TRUE(watched.success) << watched.error_message;
    EXPECT_EQ(watched.counts, unwatched.counts);
}

TEST(V11261Purity, TheWatchedCliffordGatePassGivesTheUnwatchedAnswer) {
    // A watched Clifford run performs its gate pass on the row-major tableau
    // rather than the bit-sliced one. Two entirely different gate
    // implementations, and they have to produce one answer.
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), recorder());

    QuantumCircuit qc(4, 4);
    qc.h(0);
    qc.cx(0, 1);
    qc.h(2);
    qc.cz(1, 2);
    qc.s(3);
    qc.cx(2, 3);
    qc.measure_all();

    CliffordSimulator sim;
    auto watched = sim.run(qc, 512, 20261, plan);
    auto unwatched = sim.run(qc, 512, 20261);

    EXPECT_EQ(watched.counts, unwatched.counts);
}

TEST(V11261Purity, TheWatchedUnfusedStatevectorRunMatchesTheUnfusedOne) {
    // Watching suppresses fusion, so the comparison that isolates observation
    // from fusion is against a run with fusion off and nothing attached.
    StatevectorSimulator::Options options;
    options.fusion_enable = false;

    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), recorder());

    const QuantumCircuit qc = layered_circuit();
    StatevectorSimulator sim(options);
    auto watched = sim.run(qc, 0, 20261, plan);
    auto unwatched = sim.run(qc, 0, 20261);

    ASSERT_TRUE(watched.success) << watched.error_message;
    ASSERT_TRUE(unwatched.success) << unwatched.error_message;
    ASSERT_EQ(watched.final_state.dim, unwatched.final_state.dim);
    for (std::size_t i = 0; i < watched.final_state.dim; ++i) {
        EXPECT_NEAR(watched.final_state.probability(i),
                    unwatched.final_state.probability(i), kTol)
            << "amplitude " << i;
    }
}

TEST(V11261Purity, WatchingAnMpsChangesNeitherBondsNorDiscardedWeight) {
    // An observer that renormalised or re-truncated on its way past would move
    // both of these, and neither shows up in the counts.
    auto bonds = std::make_shared<BondDimensionObserver>();
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::at_end(), bonds);
    plan.observations.observe(Anchor::every_instruction(), recorder());

    const QuantumCircuit qc = watched_circuit();
    MPSSimulator sim;
    auto watched = sim.run(qc, 4, 64, 20261, plan);
    auto unwatched = sim.run(qc, 4, 64, 20261);

    EXPECT_NEAR(watched.final_state.truncation_error(),
                unwatched.final_state.truncation_error(), kTol);
    EXPECT_EQ(watched.final_state.current_max_bond_dim(),
              unwatched.final_state.current_max_bond_dim());
    EXPECT_EQ(watched.counts, unwatched.counts);
}

// =============================================================================
// Firing counts follow the execution strategy
// =============================================================================

TEST(V11261Purity, OneEvolutionServingEveryShotFiresOnce) {
    // Terminal-only measurement: nothing before the measurements is
    // stochastic, so one forward pass serves every shot and the single firing
    // describes all of them. Firing per shot here would be a lie about how
    // many independent states were visited.
    QuantumCircuit qc(2, 2);
    qc.h(0);
    qc.cx(0, 1);
    qc.measure_all();

    auto rec = recorder();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), rec);

    StatevectorSimulator sim;
    auto r = sim.run(qc, 64, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    EXPECT_EQ(rec->count(), 1u);
    EXPECT_EQ(rec->firings()[0].n_shots, 1);
}

TEST(V11261Purity, APerShotTrajectoryFiresOncePerShot) {
    // Feedforward forces independent trajectories, so each shot IS a separate
    // state and each one is reported.
    QuantumCircuit qc(2, 2);
    qc.h(0);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});

    auto rec = recorder();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), rec);

    const int shots = 7;
    StatevectorSimulator sim;
    auto r = sim.run(qc, shots, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    EXPECT_EQ(rec->count(), static_cast<std::size_t>(shots));
    EXPECT_EQ(rec->shots(), (std::vector<int>{0, 1, 2, 3, 4, 5, 6}));
    for (const auto& f : rec->firings()) EXPECT_EQ(f.n_shots, shots);
}

TEST(V11261Purity, ShotsZeroIsOneSeededTrajectory) {
    auto rec = recorder();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), rec);

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    EXPECT_EQ(rec->count(), 1u);
    EXPECT_EQ(rec->firings()[0].shot, 0);
    EXPECT_EQ(rec->firings()[0].n_shots, 1);
}
