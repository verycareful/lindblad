// R.1.20.1 test suite — quiet_nan_strict / is_finite_strict, FAST-MATH leg.
//
// This translation unit compiles under the project-wide -ffast-math, which
// implies -ffinite-math-only. Its twin, test_r1201_strict_fp_nofast.cpp,
// includes the SAME checks under -fno-fast-math. Comparing the two answers the
// question the helpers exist to settle: does a guard survive the FP model the
// library is actually built with?
//
// R.1.20.0 added quiet_nan_strict() as the companion to the existing
// is_finite_strict(). The pair matters because the failure they prevent is
// silent in both directions: a NaN guard that folds to "always finite" stops
// reporting, and a NaN marker that is never materialised stops being
// detectable. Neither leaves a warning on GCC, and neither changes any
// observable behaviour until the moment something has actually gone wrong.
//
// The shared body lives in r1201_fp_checks.hpp; see the note there.

#include "r1201_fp_checks.hpp"

#include <gtest/gtest.h>

namespace r1201_fp {

// The one non-inline definition in the pair. Deliberately in its own function
// in its own translation unit so the value crosses a boundary the optimiser
// cannot see through, rather than staying in a register it has already drawn
// conclusions about. Both legs link against this single definition.
double round_trip(double x) { return x; }

} // namespace r1201_fp

namespace {
const std::string kLeg = "fast-math";
} // namespace

TEST(R1201StrictFPFastMath, QuietNanStrictIsNonFinite) {
    r1201_fp::check_quiet_nan_strict_is_non_finite(kLeg);
}

TEST(R1201StrictFPFastMath, IsFiniteStrictAcceptsFiniteValues) {
    r1201_fp::check_is_finite_strict_accepts_finite_values(kLeg);
}

TEST(R1201StrictFPFastMath, IsFiniteStrictRejectsInfinities) {
    r1201_fp::check_is_finite_strict_rejects_infinities(kLeg);
}

TEST(R1201StrictFPFastMath, MarkerSurvivesStorageAndCalls) {
    r1201_fp::check_marker_survives_storage_and_calls(kLeg);
}

TEST(R1201StrictFPFastMath, ValidityFlagPatternRanksCorrectly) {
    r1201_fp::check_validity_flag_pattern_ranks_correctly(kLeg);
}

// Prints rather than asserts; see the note on report_std_isfinite_behaviour.
// Read the two legs side by side:
//   ./build-clang/tests/lindblad_tests --gtest_filter='R1201StrictFP*'
TEST(R1201StrictFPFastMath, RecordStdIsfiniteBehaviour) {
    r1201_fp::report_std_isfinite_behaviour(kLeg);
    SUCCEED();
}
