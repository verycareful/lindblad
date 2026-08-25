// R.1.21.1 test wave - coverage owed by R.1.20.3 and R.1.20.5.
//
// Those two releases were patches and so had no test slot of their own. Most of
// what they shipped is covered: R.1.20.2's line-ending tool has its own Python
// suite under ctest, R.1.20.4's non-finite work is covered by the two-leg
// floating-point pair and its rank-counting fix is pinned in the MPS Shor
// diagnostic. What is left is the R.1.20.5 ladder accessors, whose own contract
// as accessors was never stated, and that is what this file covers.

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/types.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

std::array<Complex128, 4> hadamard_4() {
    constexpr double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

std::array<Complex128, 16> cnot_16() {
    std::array<Complex128, 16> m{};
    m[0 * 4 + 0] = Complex128(1.0, 0.0);
    m[1 * 4 + 3] = Complex128(1.0, 0.0);
    m[2 * 4 + 2] = Complex128(1.0, 0.0);
    m[3 * 4 + 1] = Complex128(1.0, 0.0);
    return m;
}

} // namespace

// The R.1.20.3 half of this backfill is NOT here. Both BDC warnings latch
// process-globally (warn_bdc_broken_once, and its qudit twin), so exactly one
// test in the binary can ever observe the text, and two already do:
// R1131Mps.BdcSelectionEmitsBrokenWarning and R1131Qudit.MpsBdcSelectionWarns.
// The assertion R.1.20.3 owed, that the message cites nothing a reader cannot
// reach, is added to those two rather than duplicated into a suite that would
// always find the latch already spent.

// =============================================================================
// R.1.20.5 - the SVD ladder counters
// =============================================================================
// The counters are read as diagnostics by two existing suites, which assert
// relationships between them on a particular circuit. Their own contract as
// accessors has never been stated: what they read on a fresh state, what
// increments them, and what a healthy run looks like.

TEST(R1211BackfillLadder, CountersStartAtZero) {
    const MPSState mps(4);
    EXPECT_EQ(mps.svd_call_count(), 0u)
        << "a state that has done no bond splits must report none; a nonzero "
           "denominator here would make every fallback ratio wrong";
    EXPECT_EQ(mps.gram_fallback_count(), 0u);
    EXPECT_EQ(mps.max_verify_residual_excess(), 0.0);
}

TEST(R1211BackfillLadder, SingleQubitGatesDoNotSplitBonds) {
    // svd_call_count is the denominator, so it has to count the thing it
    // claims to: bond splits, not gate applications.
    MPSState mps(4);
    for (int q = 0; q < 4; ++q) mps.apply_single_qubit_gate(hadamard_4(), q);
    EXPECT_EQ(mps.svd_call_count(), 0u)
        << "a one-qubit gate acts on a single tensor and splits no bond";
}

TEST(R1211BackfillLadder, EachTwoQubitGateSplitsAtLeastOneBond) {
    MPSState mps(4);
    mps.apply_single_qubit_gate(hadamard_4(), 0);

    std::size_t previous = mps.svd_call_count();
    for (int q = 0; q + 1 < 4; ++q) {
        mps.apply_two_qubit_gate(cnot_16(), q, q + 1);
        EXPECT_GT(mps.svd_call_count(), previous)
            << "the gate on (" << q << ", " << q + 1
            << ") did not register a bond split";
        previous = mps.svd_call_count();
    }
}

TEST(R1211BackfillLadder, FallbacksNeverExceedCalls) {
    // The fallback count is a subset of the calls, so the ratio it exists to
    // support is only meaningful while this holds.
    MPSState mps(6);
    mps.apply_single_qubit_gate(hadamard_4(), 0);
    for (int q = 0; q + 1 < 6; ++q) mps.apply_two_qubit_gate(cnot_16(), q, q + 1);

    EXPECT_GT(mps.svd_call_count(), 0u);
    EXPECT_LE(mps.gram_fallback_count(), mps.svd_call_count());
}

TEST(R1211BackfillLadder, HealthyRunKeepsTheResidualExcessNearMachineEpsilon) {
    // The accessor reports the worst factorisation error verification
    // accepted, as a fraction of the squared Frobenius norm, stated as the
    // excess over the exact identity. A clean run therefore sits near the
    // square of machine epsilon, and the number being small is what makes a
    // large one meaningful.
    MPSState mps(6);
    mps.apply_single_qubit_gate(hadamard_4(), 0);
    for (int q = 0; q + 1 < 6; ++q) mps.apply_two_qubit_gate(cnot_16(), q, q + 1);

    const double excess = mps.max_verify_residual_excess();
    EXPECT_GE(excess, 0.0)
        << "the excess is measured over the exact identity and cannot be "
           "negative; got " << excess;
    EXPECT_LT(excess, 1e-20)
        << "a GHZ ladder is about the easiest thing an SVD can be asked to "
           "factor. A healthy factorisation of these matrices measures around "
           "1e-30, and the defect R.1.20.5 fixed measured 7.4e-16, so a bound "
           "here at 1e-20 sits ten orders above clean and four below broken; "
           "got " << excess;
}

TEST(R1211BackfillLadder, CountersAreIndependentPerState) {
    // They are per-state members rather than process-wide counters, which is
    // what lets a caller attribute a fallback to the run that produced it.
    MPSState busy(4);
    busy.apply_single_qubit_gate(hadamard_4(), 0);
    for (int q = 0; q + 1 < 4; ++q) busy.apply_two_qubit_gate(cnot_16(), q, q + 1);
    ASSERT_GT(busy.svd_call_count(), 0u);

    const MPSState idle(4);
    EXPECT_EQ(idle.svd_call_count(), 0u)
        << "a second state saw the first state's bond splits, so the counters "
           "are shared where they must not be";
}
