// 1.1.26.2 test wave - what seeding a run may allocate.
//
// The cost guard is a ratio: nothing an OBSERVATION asks for should cost more
// than the simulation itself. On the write side it was measuring that ratio
// against the state the caller handed in rather than against the run, so it
// compared a destination with a source and refused three routes at every
// register size. Seeding a density matrix run from a pure state, which is the
// obvious use of the write side on that backend, could not be done at all
// without changing an option.
//
// The suite that found this reached most of the write side only under
// Cost::Unlimited, which is exactly what let it hide: relaxing the knob relaxed
// the thing under test. Every route below is therefore run at the DEFAULT
// options, and that is the point of the file.
//
// initial_cost is a separate field rather than a reuse of cost because the two
// have opposite right answers. An observation costing more than the simulation
// should be asked for explicitly; the state a run starts from IS the simulation
// and there is nothing to ask about, since the caller already chose the source,
// the backend and the register. It stays sayable for the one write-side
// allocation that is a genuine surprise, and that case is pinned at the end.

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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

const char* kFiveOnThree = "101";

QuantumCircuit measure_only(int n) {
    QuantumCircuit qc(n, n);
    qc.measure_all();
    return qc;
}

// |101> as dense amplitudes. Non-symmetric, so a reversed convention cannot
// pass.
std::shared_ptr<const Statevector> sv_five_on_three() {
    auto sv = std::make_shared<Statevector>(3);
    std::vector<Complex128> amps(8, Complex128(0.0, 0.0));
    amps[5] = Complex128(1.0, 0.0);
    sv->set_amplitudes(amps);
    return sv;
}

// The same state as a tableau: O(n^2) bits against the statevector's 2^n
// amplitudes, which is what made the old comparison refuse it.
std::shared_ptr<const StabilizerState> tableau_five_on_three() {
    auto st = std::make_shared<StabilizerState>(3);
    st->apply_x(0);
    st->apply_x(2);
    return st;
}

// And as a chain, built by running the circuit that prepares it.
std::shared_ptr<const MPSState> mps_five_on_three() {
    QuantumCircuit qc(3);
    qc.x(0);
    qc.x(2);
    MPSSimulator sim;
    auto r = sim.run(qc, 8, 0, 20261);
    return std::make_shared<const MPSState>(std::move(r.final_state));
}

}  // namespace

// =============================================================================
// The three routes that could not run at default options
// =============================================================================

TEST(V11262InitialCost, ATableauSeedsAStatevectorRunAtDefaultOptions) {
    RunPlan plan;
    plan.initial = InitialState::from(tableau_five_on_three());
    // Nothing touched. A caller who never mentions cost is the case that
    // matters, since Cost::Guarded is the default value of the read-side knob.
    EXPECT_EQ(plan.options.cost, Cost::Guarded);
    EXPECT_EQ(plan.options.initial_cost, Cost::Unlimited);

    StatevectorSimulator sim;
    auto r = sim.run(measure_only(3), 32, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
}

TEST(V11262InitialCost, AChainSeedsAStatevectorRunAtDefaultOptions) {
    RunPlan plan;
    plan.initial = InitialState::from(mps_five_on_three());

    StatevectorSimulator sim;
    auto r = sim.run(measure_only(3), 32, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
}

TEST(V11262InitialCost, AStatevectorSeedsADensityMatrixRunAtDefaultOptions) {
    // 4^n against 2^n, a ratio of 2^n, so this was refused on a register of any
    // size at all. The density matrix is not an extra allocation: that backend
    // holds one whether or not anybody supplied a state.
    RunPlan plan;
    plan.initial = InitialState::from(sv_five_on_three());

    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(measure_only(3), noise, 32, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
}

TEST(V11262InitialCost, ATableauSeedsADensityMatrixRunAtDefaultOptions) {
    RunPlan plan;
    plan.initial = InitialState::from(tableau_five_on_three());

    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(measure_only(3), noise, 32, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
}

// =============================================================================
// Every supplied form into every backend, all at the default
// =============================================================================
// The routes that already worked are here too, so a change that fixes the three
// above by loosening something cannot quietly break these.

TEST(V11262InitialCost, EverySupportedRouteRunsAtDefaultOptions) {
    const QuantumCircuit qc = measure_only(3);

    struct Route {
        const char* name;
        InitialState state;
    };

    {
        StatevectorSimulator sim;
        for (const auto& route : {Route{"statevector", InitialState::from(sv_five_on_three())},
                                  Route{"tableau", InitialState::from(tableau_five_on_three())},
                                  Route{"chain", InitialState::from(mps_five_on_three())}}) {
            RunPlan plan;
            plan.initial = route.state;
            auto r = sim.run(qc, 16, 20261, plan);
            EXPECT_TRUE(r.success) << route.name << ": " << r.error_message;
            if (r.success) EXPECT_EQ(r.counts.at(kFiveOnThree), 16) << route.name;
        }
    }
    {
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        for (const auto& route : {Route{"statevector", InitialState::from(sv_five_on_three())},
                                  Route{"tableau", InitialState::from(tableau_five_on_three())},
                                  Route{"chain", InitialState::from(mps_five_on_three())}}) {
            RunPlan plan;
            plan.initial = route.state;
            auto r = sim.run(qc, noise, 16, 20261, plan);
            EXPECT_TRUE(r.success) << route.name << ": " << r.error_message;
            if (r.success) EXPECT_EQ(r.counts.at(kFiveOnThree), 16) << route.name;
        }
    }
    {
        MPSSimulator sim;
        for (const auto& route : {Route{"statevector", InitialState::from(sv_five_on_three())},
                                  Route{"chain", InitialState::from(mps_five_on_three())}}) {
            RunPlan plan;
            plan.initial = route.state;
            auto r = sim.run(qc, 8, 16, 20261, plan);
            EXPECT_EQ(r.counts.at(kFiveOnThree), 16) << route.name;
        }
    }
    {
        CliffordSimulator sim;
        RunPlan plan;
        plan.initial = InitialState::from(tableau_five_on_three());
        auto r = sim.run(qc, 16, 20261, plan);
        EXPECT_EQ(r.counts.at(kFiveOnThree), 16);
    }
}

// =============================================================================
// What the two knobs still do
// =============================================================================

TEST(V11262InitialCost, TheReadSideGuardIsUnchanged) {
    // cost still defaults to Guarded and still refuses an over-budget
    // OBSERVATION. Exempting the write side must not reach across.
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<StateObserver>(StateForm::DensityMatrix, "dm"));

    QuantumCircuit qc(3);
    qc.h(0);
    qc.cx(0, 1);

    StatevectorSimulator sim;
    auto r = sim.run(qc, 0, 20261, plan);

    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error_message.find("guard"), std::string::npos) << r.error_message;
}

TEST(V11262InitialCost, ConversionNeverStillRefusesTheWriteSide) {
    // The write side answers to what may be translated even though it no longer
    // answers to what that may cost. Those are different questions.
    RunPlan plan;
    plan.initial = InitialState::from(sv_five_on_three());
    plan.options.conversion = Conversion::Never;

    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(measure_only(3), noise, 8, 20261, plan);

    EXPECT_FALSE(r.success);
}

TEST(V11262InitialCost, AnImpossibleWriteStillFailsUnderEveryResponse) {
    // Unchanged and worth re-pinning beside the loosening: a run has to start
    // somewhere, so there is nothing here for Warn or Ignore to omit.
    for (const Response response : {Response::Throw, Response::Warn, Response::Ignore}) {
        RunPlan plan;
        plan.initial = InitialState::from(sv_five_on_three());
        plan.options.response = response;

        CliffordSimulator sim;
        EXPECT_THROW(sim.run(measure_only(3), 8, 20261, plan), std::invalid_argument);
    }
}

TEST(V11262InitialCost, AGuardedWriteRefusesTheOneAllocationThatSurprises) {
    // Seeding an MPS from a compact state materialises a full 2^n dense array
    // inside a backend chosen to avoid exactly that, and it is the only
    // write-side cost that is not already implied by the caller's own choices.
    // Guarded is not the default, so asking for this is deliberate.
    RunPlan plan;
    plan.initial = InitialState::from(tableau_five_on_three());
    plan.options.initial_cost = Cost::Guarded;
    plan.options.guard_multiple = 1.0;

    MPSSimulator sim;
    EXPECT_THROW(sim.run(measure_only(3), 8, 16, 20261, plan), std::invalid_argument);
}

TEST(V11262InitialCost, AGuardedWriteAllowsWhatTheRunWouldHoldAnyway) {
    // The same knob on the routes where the conversion produces exactly the
    // state the backend was going to allocate: a ratio of one, so it passes.
    RunPlan plan;
    plan.initial = InitialState::from(sv_five_on_three());
    plan.options.initial_cost = Cost::Guarded;
    plan.options.guard_multiple = 1.0;

    {
        StatevectorSimulator sim;
        auto r = sim.run(measure_only(3), 16, 20261, plan);
        EXPECT_TRUE(r.success) << r.error_message;
    }
    {
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        auto r = sim.run(measure_only(3), noise, 16, 20261, plan);
        EXPECT_TRUE(r.success) << r.error_message;
    }
}

TEST(V11262InitialCost, AGuardedWriteAllowsHandingAChainItsOwnAmplitudes) {
    // A caller supplying a statevector to an MPS run already holds those
    // amplitudes. Nothing is materialised that did not exist, so the guard has
    // nothing to catch even when it is switched on.
    RunPlan plan;
    plan.initial = InitialState::from(sv_five_on_three());
    plan.options.initial_cost = Cost::Guarded;
    plan.options.guard_multiple = 1.0;

    MPSSimulator sim;
    auto r = sim.run(measure_only(3), 8, 16, 20261, plan);
    EXPECT_EQ(r.counts.at(kFiveOnThree), 16);
}
