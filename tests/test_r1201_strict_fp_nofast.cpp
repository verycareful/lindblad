// R.1.20.1 test suite — quiet_nan_strict / is_finite_strict, STRICT-FP leg.
//
// Compiled WITHOUT fast math (-fno-fast-math, set per-source in
// tests/CMakeLists.txt) while its twin test_r1201_strict_fp.cpp compiles under
// the project-wide flags. Same header, same checks, same inputs, and the
// floating-point model is the only variable.
//
// What the pair establishes:
//
//   Both legs green  is_finite_strict and quiet_nan_strict answer the same way
//                    under both FP models the tree is built with, so a guard
//                    written with them means the same thing in either, which is
//                    the property #68 needed and the POSIX spellings could not
//                    provide.
//
//   Legs disagree    the assertion that differs names precisely which property
//                    the FP model changed, with everything else held constant.
//                    That is a far stronger diagnostic than a single failing
//                    test in one build, where "the compiler folded it" and "the
//                    helper is wrong" look identical.

#include "r1201_fp_checks.hpp"

#include <gtest/gtest.h>

namespace {
const std::string kLeg = "strict-fp";
} // namespace

TEST(R1201StrictFPStrict, QuietNanStrictIsNonFinite) {
    r1201_fp::check_quiet_nan_strict_is_non_finite(kLeg);
}

TEST(R1201StrictFPStrict, IsFiniteStrictAcceptsFiniteValues) {
    r1201_fp::check_is_finite_strict_accepts_finite_values(kLeg);
}

TEST(R1201StrictFPStrict, IsFiniteStrictRejectsInfinities) {
    r1201_fp::check_is_finite_strict_rejects_infinities(kLeg);
}

TEST(R1201StrictFPStrict, MarkerSurvivesStorageAndCalls) {
    r1201_fp::check_marker_survives_storage_and_calls(kLeg);
}

TEST(R1201StrictFPStrict, ValidityFlagPatternRanksCorrectly) {
    r1201_fp::check_validity_flag_pattern_ranks_correctly(kLeg);
}

// Under this FP model std::isfinite is expected to behave; the fast-math twin
// is where the answer may differ. Printed, not asserted — the comparison
// between the two lines is the evidence, and neither line is a contract.
TEST(R1201StrictFPStrict, RecordStdIsfiniteBehaviour) {
    r1201_fp::report_std_isfinite_behaviour(kLeg);
    SUCCEED();
}

// The two legs share their assertions through a header and differ only in the
// floating-point flags their translation unit is compiled with. That makes the
// flag state itself a contract: if both legs were to compile under one model,
// every check above would run twice against identical codegen while appearing
// to cover two.
//
// This leg carries -fno-fast-math, set on the source file in tests/CMakeLists.
//
// __FAST_MATH__ alone cannot express the distinction. It is defined only when
// the whole -ffast-math bundle is in force, and this build pairs -ffast-math
// with -fno-finite-math-only, so neither leg defines it. The other bundle
// members are individually observable and -fno-fast-math turns all of them
// off, so their presence is what actually separates the two translation units.
#if defined(__FAST_MATH__) || defined(__ASSOCIATIVE_MATH__) || \
    defined(__RECIPROCAL_MATH__) || defined(__NO_MATH_ERRNO__)
#define LINDBLAD_TEST_FAST_MATH_FAMILY 1
#else
#define LINDBLAD_TEST_FAST_MATH_FAMILY 0
#endif

TEST(R1201StrictFPStrict, LegIsCompiledWithoutFastMath) {
    EXPECT_EQ(LINDBLAD_TEST_FAST_MATH_FAMILY, 0)
        << "the strict leg sees a fast-math relaxation, so its -fno-fast-math "
           "source property has stopped taking effect and both legs now "
           "measure one model";
}
