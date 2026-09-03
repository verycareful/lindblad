// 1.1.26.1 test wave - where the run starts.
//
// InitialState is the whole write side of the harness, and it answers to the
// knobs differently from everything else: a read the backend cannot perform is
// omissible, and Response::Warn and Response::Ignore exist to omit it, but
// there is no such thing as omitting the state a run starts from. The
// alternative to the state the caller asked for is silently simulating a
// different circuit, so an unproducible initial state fails the run under every
// response. That asymmetry is the single most important claim here.
//
// Every basis index used below is NON-SYMMETRIC. Five on three qubits is |101>,
// which reads differently in each direction, so a reversed convention cannot
// pass. A symmetric value would mask exactly the bug this project's convention
// rule exists to catch.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
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

using v11261::sv_run_failure;
using v11261::throwing_run_failure;

namespace {

// Three qubits, measured into three clbits. Nothing acts on the state, so the
// counts report the initial state and nothing else.
QuantumCircuit measure_only(int n) {
    QuantumCircuit qc(n, n);
    qc.measure_all();
    return qc;
}

// |101> as a dense statevector: index 5 of eight, everything else zero.
std::shared_ptr<const Statevector> five_on_three() {
    auto sv = std::make_shared<Statevector>(3);
    std::vector<Complex128> amps(8, Complex128(0.0, 0.0));
    amps[5] = Complex128(1.0, 0.0);
    sv->set_amplitudes(amps);
    return sv;
}

// (|00> + |11>)/sqrt(2). Needs bond dimension 2 as an MPS, so a run capped at 1
// has to truncate it.
std::shared_ptr<const Statevector> bell_pair() {
    auto sv = std::make_shared<Statevector>(2);
    std::vector<Complex128> amps(4, Complex128(0.0, 0.0));
    amps[0] = Complex128(INV_SQRT2, 0.0);
    amps[3] = Complex128(INV_SQRT2, 0.0);
    sv->set_amplitudes(amps);
    return sv;
}

const char* kFiveOnThree = "101";

}  // namespace

// =============================================================================
// The default
// =============================================================================

TEST(V11261InitialState, TheDefaultIsTheAllZeroStateOnEveryBackend) {
    const QuantumCircuit qc = measure_only(3);
    const RunPlan empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(empty.initial.is_default());

    {
        StatevectorSimulator sim;
        auto r = sim.run(qc, 32, 20261, empty);
        EXPECT_EQ(r.counts.at("000"), 32);
    }
    {
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        auto r = sim.run(qc, noise, 32, 20261, empty);
        EXPECT_EQ(r.counts.at("000"), 32);
    }
    {
        CliffordSimulator sim;
        auto r = sim.run(qc, 32, 20261, empty);
        EXPECT_EQ(r.counts.at("000"), 32);
    }
    {
        MPSSimulator sim;
        auto r = sim.run(qc, 8, 32, 20261, empty);
        EXPECT_EQ(r.counts.at("000"), 32);
    }
}

// =============================================================================
// basis(k), with a value that reads differently in each direction
// =============================================================================

TEST(V11261InitialState, BasisFiveOnThreeQubitsIsTheStateOneZeroOne) {
    RunPlan plan;
    plan.initial = InitialState::basis(5);
    EXPECT_TRUE(plan.initial.is_basis());
    EXPECT_EQ(plan.initial.basis_index(), 5u);
    EXPECT_FALSE(plan.empty());

    StatevectorSimulator sim;
    auto r = sim.run(QuantumCircuit(3), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    // Qubit 0 carries the ones place, so K = 5 is q0 = 1, q1 = 0, q2 = 1.
    EXPECT_NEAR(r.final_state.probability(5), 1.0, DEFAULT_PHYSICAL_ATOL);
}

TEST(V11261InitialState, BasisFiveReadsAsOneZeroOneOnEveryBackend) {
    const QuantumCircuit qc = measure_only(3);
    RunPlan plan;
    plan.initial = InitialState::basis(5);

    {
        StatevectorSimulator sim;
        auto r = sim.run(qc, 32, 20261, plan);
        EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
    }
    {
        DensityMatrixSimulator sim;
        const NoiseModel noise;
        auto r = sim.run(qc, noise, 32, 20261, plan);
        EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
    }
    {
        CliffordSimulator sim;
        auto r = sim.run(qc, 32, 20261, plan);
        EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
    }
    {
        MPSSimulator sim;
        auto r = sim.run(qc, 8, 32, 20261, plan);
        EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
    }
}

TEST(V11261InitialState, ABasisIndexOutsideTheRegisterFails) {
    // Three qubits hold eight basis states, so 8 is the first index that names
    // none of them.
    RunPlan plan;
    plan.initial = InitialState::basis(8);

    const std::string message = sv_run_failure(QuantumCircuit(3), plan);
    EXPECT_NE(message.find("8"), std::string::npos) << message;

    const std::string clifford = throwing_run_failure([&] {
        CliffordSimulator sim;
        sim.run(QuantumCircuit(3), 8, 20261, plan);
    });
    EXPECT_NE(clifford.find("8"), std::string::npos) << clifford;
}

// =============================================================================
// A supplied state, into the backend that holds that form natively
// =============================================================================

TEST(V11261InitialState, ASuppliedStatevectorSeedsTheStatevectorBackend) {
    RunPlan plan;
    plan.initial = InitialState::from(five_on_three());
    EXPECT_EQ(plan.initial.form(), StateForm::Statevector);
    EXPECT_FALSE(plan.initial.is_default());
    EXPECT_FALSE(plan.initial.is_basis());

    StatevectorSimulator sim;
    auto r = sim.run(QuantumCircuit(3), 0, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_NEAR(r.final_state.probability(5), 1.0, DEFAULT_PHYSICAL_ATOL);
}

TEST(V11261InitialState, ASuppliedStabilizerSeedsTheCliffordBackend) {
    auto seed = std::make_shared<StabilizerState>(3);
    seed->apply_x(0);
    seed->apply_x(2);  // |101>

    RunPlan plan;
    plan.initial = InitialState::from(seed);

    CliffordSimulator sim;
    auto r = sim.run(measure_only(3), 32, 20261, plan);
    EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
}

// EXPECTED-RED. The cost guard refuses this, and refusing it is wrong.
//
// apply_initial_state builds its StateView on the SUPPLIED state, so the guard
// compares the 4^n density matrix it is about to produce against the 2^n
// statevector it was handed. That ratio is 2^n, so it exceeds guard_multiple
// 1.0 on every register of any size: seeding a density matrix simulator from a
// pure state, which is the obvious use of the write side on that backend, fails
// at the default setting and always will.
//
// The allocation being refused is not even an extra cost. The backend allocates
// 4^n either way; that IS its state. The guard exists to stop an OBSERVATION
// quietly allocating more than the simulation itself, and here it is measuring
// against the wrong side of the conversion.
//
// It is also incoherent with the decision immediately next to it. The response
// knob was deliberately exempted from the initial state, because there is no
// such thing as omitting the state a run starts from. The cost knob gates it
// anyway, and refuses the run for being expensive when nothing was saved.
TEST(V11261InitialState, ASuppliedStatevectorIsConvertedForTheDensityMatrixBackend) {
    // The one conversion the write side actually performs: a pure state has a
    // density matrix, and producing it is arithmetic rather than inference.
    RunPlan plan;
    plan.initial = InitialState::from(five_on_three());

    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(measure_only(3), noise, 32, 20261, plan);

    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
}

TEST(V11261InitialState, ASuppliedStatevectorIsFactorisedForTheMpsBackend) {
    RunPlan plan;
    plan.initial = InitialState::from(five_on_three());

    MPSSimulator sim;
    auto r = sim.run(measure_only(3), 8, 32, 20261, plan);

    EXPECT_EQ(r.counts.at(kFiveOnThree), 32);
}

// =============================================================================
// A supplied state the backend cannot be given
// =============================================================================

TEST(V11261InitialState, AStatevectorCannotSeedTheCliffordBackend) {
    // Not expense: a general statevector is not a stabilizer state at all, and
    // recovering a tableau from amplitudes is not a conversion that exists.
    RunPlan plan;
    plan.initial = InitialState::from(five_on_three());

    const std::string message = throwing_run_failure([&] {
        CliffordSimulator sim;
        sim.run(measure_only(3), 8, 20261, plan);
    });
    EXPECT_NE(message.find("stabilizer"), std::string::npos) << message;
}

TEST(V11261InitialState, ADensityMatrixCannotSeedTheStatevectorBackend) {
    auto dm = std::make_shared<DensityMatrix>(2);
    RunPlan plan;
    plan.initial = InitialState::from(dm);

    const std::string message = sv_run_failure(QuantumCircuit(2), plan);
    EXPECT_FALSE(message.empty());
}

TEST(V11261InitialState, AnUnproducibleInitialStateFailsUnderWarnAndIgnoreToo) {
    // THE claim of this file. Warn and Ignore govern observations, which are
    // omissible. A run has to start somewhere, so there is nothing to omit
    // here: the only alternative to the caller's state is silently simulating a
    // different circuit.
    for (const Response response : {Response::Throw, Response::Warn, Response::Ignore}) {
        RunPlan plan;
        plan.initial = InitialState::from(five_on_three());
        plan.options.response = response;

        const std::string message = throwing_run_failure([&] {
            CliffordSimulator sim;
            sim.run(measure_only(3), 8, 20261, plan);
        });
        EXPECT_FALSE(message.empty());
    }
}

TEST(V11261InitialState, AnUnproducibleInitialStateFailsUnderConversionNever) {
    RunPlan plan;
    plan.initial = InitialState::from(five_on_three());
    plan.options.conversion = Conversion::Never;

    // The density matrix backend CAN take a statevector, but not when the
    // caller has declined conversion.
    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(measure_only(3), noise, 8, 20261, plan);
    EXPECT_FALSE(r.success);
}

TEST(V11261InitialState, ASuppliedStateOfTheWrongWidthFails) {
    RunPlan plan;
    plan.initial = InitialState::from(five_on_three());  // three qubits

    const std::string message = sv_run_failure(QuantumCircuit(2), plan);
    EXPECT_NE(message.find("qubit"), std::string::npos) << message;
}

// =============================================================================
// MPS truncates rather than refusing
// =============================================================================

// EXPECTED-RED. The truncation happens; the report of it does not.
//
// mps_from_sv drives each split through svd_truncate_verified, which returns
// the weight that split chose to throw away in SvdTruncation::discarded_weight,
// a field whose own comment says a caller accumulates it into its truncation
// total. mps_from_sv never does, so a state factorised past the bond cap comes
// back silently approximate with truncation_error() reading zero.
//
// apply_initial_state promises the opposite in as many words: a supplied state
// too large for the cap "is TRUNCATED here rather than refused ... and the
// discarded weight is what truncation_error reports". A caller who trusts that
// figure to decide whether their cap was adequate is told it was.
TEST(V11261InitialState, AnMpsTruncatesASuppliedStateItCannotHold) {
    // A Bell pair needs bond dimension 2. Capping the run at 1 does not make
    // the state unavailable, it makes it approximate: that is what running at
    // this cap MEANS, and the discarded weight is what truncation_error is for.
    RunPlan plan;
    plan.initial = InitialState::from(bell_pair());

    MPSSimulator sim;
    auto r = sim.run(QuantumCircuit(2), 1, 0, 20261, plan);

    EXPECT_GT(r.final_state.truncation_error(), 0.0);
}

TEST(V11261InitialState, AnMpsKeepsASuppliedStateItCanHold) {
    RunPlan plan;
    plan.initial = InitialState::from(bell_pair());

    MPSSimulator sim;
    auto r = sim.run(QuantumCircuit(2), 8, 0, 20261, plan);

    // Room for the state, so nothing is discarded and the pair survives whole.
    EXPECT_NEAR(r.final_state.truncation_error(), 0.0, DEFAULT_PHYSICAL_ATOL);
    const Statevector dense = r.final_state.to_statevector();
    EXPECT_NEAR(dense.probability(0), 1.0 / 2.0, 1e-9);
    EXPECT_NEAR(dense.probability(3), 1.0 / 2.0, 1e-9);
}

// =============================================================================
// Rejections at construction
// =============================================================================

TEST(V11261InitialState, ANullSuppliedStateIsRejectedAtConstruction) {
    EXPECT_THROW(InitialState::from(std::shared_ptr<const Statevector>{}),
                 std::invalid_argument);
    EXPECT_THROW(InitialState::from(std::shared_ptr<const DensityMatrix>{}),
                 std::invalid_argument);
    EXPECT_THROW(InitialState::from(std::shared_ptr<const StabilizerState>{}),
                 std::invalid_argument);
    EXPECT_THROW(InitialState::from(std::shared_ptr<const MPSState>{}),
                 std::invalid_argument);
}

TEST(V11261InitialState, AskingADefaultInitialStateForItsFormThrows) {
    // is_default and is_basis are the questions that ARE answerable; form()
    // reports the representation of a supplied state and there is none here.
    const InitialState zero;
    EXPECT_THROW(zero.form(), std::invalid_argument);

    const InitialState basis = InitialState::basis(5);
    EXPECT_THROW(basis.form(), std::invalid_argument);
}

TEST(V11261InitialState, APlanCarryingOnlyAnInitialStateIsNotEmpty) {
    // empty() drives the zero-overhead path, and a plan that watches nothing
    // but starts somewhere else still has work to do.
    RunPlan plan;
    plan.initial = InitialState::basis(5);

    EXPECT_FALSE(plan.empty());
    EXPECT_TRUE(plan.observations.empty());

    StatevectorSimulator sim;
    auto r = sim.run(QuantumCircuit(3), 0, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;
    EXPECT_NEAR(r.final_state.probability(5), 1.0, DEFAULT_PHYSICAL_ATOL);
}
