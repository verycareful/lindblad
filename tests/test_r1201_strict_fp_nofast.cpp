// R.1.20.1 test suite — quiet_nan_strict / is_finite_strict, STRICT-FP leg.
//
// Compiled WITHOUT fast math (-fno-fast-math, set per-source in
// tests/CMakeLists.txt) while its twin test_r1201_strict_fp.cpp compiles under
// the project-wide -ffast-math. Same header, same checks, same inputs — the
// floating-point model is the only variable.
//
// What the pair establishes:
//
//   Both legs green  the helpers do what they claim. is_finite_strict and
//                    quiet_nan_strict are model-independent, so guards written
//                    with them mean the same thing in every build of the
//                    library, which is the property #68 needed and the POSIX
//                    spellings could not provide.
//
//   Legs disagree    the assertion that differs names precisely which property
//                    -ffinite-math-only broke, with everything else held
//                    constant. That is a far stronger diagnostic than a single
//                    failing test in a fast-math build, where "the compiler
//                    folded it" and "the helper is wrong" look identical.
//
// The non-inline round_trip() the checks call is defined once, in the fast-math
// twin; this TU links against it.

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
