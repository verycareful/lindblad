// 1.1.26.2 test wave - a truncated chain says so.
//
// Rebuilding an MPS from dense amplitudes is a sequential SVD, one truncated
// split per bond, and every split returns the weight it chose to throw away.
// The reconstruction discarded all of them, so a chain factorised past its bond
// cap came back silently approximate with truncation_error() reading zero. A
// caller using that figure to decide whether their cap was adequate was told it
// was.
//
// The reconstruction also REPLACED the state it was rebuilding, which reset the
// counters along with it. That is the worse half: it happens mid-run, on every
// gate with no compact MPS form, so a chain that had already lost weight to
// earlier gates reported none.
//
// The values here are derived rather than observed. A Bell pair has Schmidt
// coefficients 1/sqrt(2) and 1/sqrt(2), so their squares are 1/2 each and
// keeping one of them discards exactly 1/2. Asserting a non-zero figure would
// have passed on any arithmetic at all.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 1e-9;

// (|00> + |11>)/sqrt(2): bond dimension 2 across its one cut.
Statevector bell_pair() {
    Statevector sv(2);
    std::vector<Complex128> amps(4, Complex128(0.0, 0.0));
    amps[0] = Complex128(INV_SQRT2, 0.0);
    amps[3] = Complex128(INV_SQRT2, 0.0);
    sv.set_amplitudes(amps);
    return sv;
}

// The weight one of two equal Schmidt values carries. Derived from the
// amplitude rather than written as 0.5.
const double kHalfWeight = INV_SQRT2 * INV_SQRT2;

}  // namespace

// =============================================================================
// The figure is reported, and it is the right figure
// =============================================================================

TEST(V11262MpsTruncation, ACappedRebuildReportsTheWeightItDiscarded) {
    MPSState mps(2, /*max_bond_dim=*/1);
    mps.rebuild_from_statevector(bell_pair());

    // Two equal Schmidt values, one kept, so exactly half the weight goes.
    EXPECT_NEAR(mps.truncation_error(), kHalfWeight, kTol);
}

TEST(V11262MpsTruncation, ARebuildWithRoomDiscardsNothing) {
    // The other side of the same claim: the fix must not make every chain look
    // truncated.
    MPSState mps(2, /*max_bond_dim=*/8);
    mps.rebuild_from_statevector(bell_pair());

    EXPECT_NEAR(mps.truncation_error(), 0.0, DEFAULT_PHYSICAL_ATOL);
}

TEST(V11262MpsTruncation, TheCountersAccumulateAcrossRebuilds) {
    // Two rebuilds at the same cap discard the same weight twice. This is the
    // half that mattered mid-run: the reconstruction used to replace the state
    // and take its zeroed counters with it.
    MPSState mps(2, /*max_bond_dim=*/1);
    mps.rebuild_from_statevector(bell_pair());
    const double after_one = mps.truncation_error();

    mps.rebuild_from_statevector(bell_pair());
    const double after_two = mps.truncation_error();

    EXPECT_NEAR(after_one, kHalfWeight, kTol);
    EXPECT_NEAR(after_two, 2.0 * kHalfWeight, kTol);
}

TEST(V11262MpsTruncation, EverySplitIsCounted) {
    // A chain of n qubits is n-1 bonds, so a rebuild performs n-1 splits. The
    // call count is the denominator the rescue counts are read against, and it
    // was being thrown away with everything else.
    MPSState mps(4, /*max_bond_dim=*/4);
    Statevector sv(4);
    std::vector<Complex128> amps(16, Complex128(0.0, 0.0));
    amps[0] = Complex128(INV_SQRT2, 0.0);
    amps[15] = Complex128(INV_SQRT2, 0.0);   // a four-qubit GHZ state
    sv.set_amplitudes(amps);

    mps.rebuild_from_statevector(sv);

    EXPECT_EQ(mps.svd_call_count(), 3u);
}

TEST(V11262MpsTruncation, TruncationAtSeveralBondsAddsUp) {
    // A GHZ chain capped at one bond loses weight at every bond rather than at
    // one of them, so the total exceeds what a single split could account for.
    MPSState mps(4, /*max_bond_dim=*/1);
    Statevector sv(4);
    std::vector<Complex128> amps(16, Complex128(0.0, 0.0));
    amps[0] = Complex128(INV_SQRT2, 0.0);
    amps[15] = Complex128(INV_SQRT2, 0.0);
    sv.set_amplitudes(amps);

    mps.rebuild_from_statevector(sv);

    EXPECT_EQ(mps.svd_call_count(), 3u);
    EXPECT_GT(mps.truncation_error(), 0.0);
}

// =============================================================================
// The rebuild leaves the rest of the state alone
// =============================================================================

TEST(V11262MpsTruncation, ARebuildKeepsTheCapAndTheCutoff) {
    MPSState mps(2, /*max_bond_dim=*/3, /*cutoff=*/1e-12);
    mps.rebuild_from_statevector(bell_pair());

    EXPECT_EQ(mps.max_bond_dim, 3);
    EXPECT_NEAR(mps.cutoff, 1e-12, 1e-18);
    EXPECT_EQ(mps.n_qubits, 2);
}

TEST(V11262MpsTruncation, ARebuildOfTheWrongWidthIsRefused) {
    MPSState mps(3);
    EXPECT_THROW(mps.rebuild_from_statevector(bell_pair()), std::invalid_argument);
}

TEST(V11262MpsTruncation, ARebuiltChainHoldsTheStateItWasGiven) {
    // The counters are the subject of this file, so the amplitudes are checked
    // too: a rebuild that reported its truncation correctly and factorised
    // wrongly would pass everything above.
    MPSState mps(2, /*max_bond_dim=*/8);
    mps.rebuild_from_statevector(bell_pair());

    const Statevector dense = mps.to_statevector();
    EXPECT_NEAR(dense.probability(0), kHalfWeight, 1e-9);
    EXPECT_NEAR(dense.probability(3), kHalfWeight, 1e-9);
    EXPECT_NEAR(dense.probability(1), 0.0, 1e-9);
    EXPECT_NEAR(dense.probability(2), 0.0, 1e-9);
}

// =============================================================================
// Through the run, which is how a caller meets it
// =============================================================================

TEST(V11262MpsTruncation, SeedingPastTheCapIsReportedByTheRun) {
    auto seed = std::make_shared<Statevector>(bell_pair());
    RunPlan plan;
    plan.initial = InitialState::from(seed);

    MPSSimulator sim;
    auto r = sim.run(QuantumCircuit(2), /*max_bond_dim=*/1, 0, 20261, plan);

    EXPECT_NEAR(r.final_state.truncation_error(), kHalfWeight, kTol);
}

TEST(V11262MpsTruncation, SeedingReportsWhatSeedingCostAndNothingElse) {
    // The initial state is a FRESH chain, so its total describes this run
    // rather than carrying anything in. Accumulation is right mid-run and wrong
    // for the state a run starts from.
    auto seed = std::make_shared<Statevector>(bell_pair());
    RunPlan plan;
    plan.initial = InitialState::from(seed);

    MPSSimulator sim;
    auto first = sim.run(QuantumCircuit(2), 1, 0, 20261, plan);
    auto second = sim.run(QuantumCircuit(2), 1, 0, 20261, plan);

    EXPECT_NEAR(first.final_state.truncation_error(),
                second.final_state.truncation_error(), kTol);
}

TEST(V11262MpsTruncation, SeedingWithinTheCapReportsNothingDiscarded) {
    auto seed = std::make_shared<Statevector>(bell_pair());
    RunPlan plan;
    plan.initial = InitialState::from(seed);

    MPSSimulator sim;
    auto r = sim.run(QuantumCircuit(2), 8, 0, 20261, plan);

    EXPECT_NEAR(r.final_state.truncation_error(), 0.0, DEFAULT_PHYSICAL_ATOL);
}
