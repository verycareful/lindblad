// R.1.20.1 test suite — quiet_nan_strict / is_finite_strict, FAST-MATH leg.
//
// This translation unit compiles under the project-wide flags. Its twin,
// test_r1201_strict_fp_nofast.cpp, includes the SAME checks under
// -fno-fast-math. Comparing the two answers the question the helpers exist to
// settle: does a guard mean the same thing under the FP model the library is
// actually built with?
//
// The pair matters because the failure it catches is silent in both directions:
// a NaN guard that folds to "always finite" stops reporting, and a NaN marker
// that is never materialised stops being detectable. Neither leaves a warning
// on GCC, and neither changes any observable behaviour until the moment
// something has actually gone wrong.
//
// This leg also carries the build-configuration pin, being the one compiled
// with the project's own flags.
//
// The shared body lives in r1201_fp_checks.hpp; see the note there.

#include "r1201_fp_checks.hpp"

#include <gtest/gtest.h>

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

// The build-configuration pin, and the only check here that does not go through
// a value. Under -ffinite-math-only clang marks floating-point parameters and
// return values nofpclass(nan inf), so a NaN crossing any function boundary is
// poison and reads back as whatever occupies the register. Every non-finite
// guard in the library is then inoperative, and no guard can report that about
// itself, which is why this asserts the flag state directly.
TEST(R1201StrictFPFastMath, BuildOmitsFiniteMathOnly) {
#if defined(__FINITE_MATH_ONLY__)
    EXPECT_EQ(__FINITE_MATH_ONLY__, 0)
        << "-ffinite-math-only is in effect for a translation unit built with "
           "the project-wide flags; every non-finite guard in the library is "
           "inoperative";
#else
    GTEST_SKIP() << "compiler does not define __FINITE_MATH_ONLY__";
#endif
}

// Prints rather than asserts; see the note on report_std_isfinite_behaviour.
// Read the two legs side by side:
//   ./build-clang/tests/lindblad_tests --gtest_filter='R1201StrictFP*'
TEST(R1201StrictFPFastMath, RecordStdIsfiniteBehaviour) {
    r1201_fp::report_std_isfinite_behaviour(kLeg);
    SUCCEED();
}

// The counterpart of LegIsCompiledWithoutFastMath in the strict twin. Together
// the two establish that the pair spans two floating-point models rather than
// running the same one under two names.
// __FAST_MATH__ is NOT the discriminator: it requires the whole -ffast-math
// bundle, and this build pairs -ffast-math with -fno-finite-math-only, so it is
// defined in neither leg. The individually-observable bundle members are what
// separate them, since -fno-fast-math turns all of them off.
#if defined(__FAST_MATH__) || defined(__ASSOCIATIVE_MATH__) || \
    defined(__RECIPROCAL_MATH__) || defined(__NO_MATH_ERRNO__)
#define LINDBLAD_TEST_FAST_MATH_FAMILY 1
#else
#define LINDBLAD_TEST_FAST_MATH_FAMILY 0
#endif

TEST(R1201StrictFPFastMath, LegIsCompiledWithFastMath) {
#if defined(__GNUC__) || defined(__clang__)
    EXPECT_EQ(LINDBLAD_TEST_FAST_MATH_FAMILY, 1)
        << "the project-wide -ffast-math did not reach this translation unit, "
           "so both legs of the pair now compile under the strict model and "
           "the fast-math half of the coverage is gone";
#else
    GTEST_SKIP() << "the -ffast-math flag is set only for GCC and Clang; this "
                    "compiler spells its fast-math model differently.";
#endif
}
