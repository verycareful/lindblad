// =============================================================================
// tests/test_visualiser_ascii_fixtures.cpp : R.1.10.1 AsciiFixtureTest
// =============================================================================
// Golden-file byte-exact comparison for the ASCII renderer. Each fixture is
// rendered with a specific DrawOptions configuration and compared against a
// committed golden file under tests/golden/visualisation/ascii/. Plus a
// handful of structural assertions (no UTF-8 bytes in ascii_safe mode,
// output ends with newline, empty circuits produce non-empty output) that
// do not depend on the golden tree.
//
// Regeneration: run lindblad_visualiser_regen after the build to refresh
// every golden file from the current implementation. Each diff must be
// reviewed against docs/plans/circuit-visualiser-design.md before committing.

#include "lindblad/circuit.hpp"
#include "lindblad/visualisation.hpp"

#include "golden_helpers.hpp"
#include "visualiser_fixtures.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

using namespace lindblad;
using namespace lindblad::vfx;

namespace {

bool has_non_ascii(const std::string& s) {
    return std::any_of(s.begin(), s.end(), [](char c) {
        return static_cast<unsigned char>(c) >= 0x80;
    });
}

DrawOptions ascii_safe_opts() {
    DrawOptions o;
    o.ascii_safe = true;
    return o;
}

DrawOptions show_clbits_opts() {
    DrawOptions o;
    o.show_clbits = true;
    return o;
}

DrawOptions raw_param_opts() {
    DrawOptions o;
    o.param_format = ParamFormat::Raw;
    return o;
}

DrawOptions no_params_opts() {
    DrawOptions o;
    o.show_params = false;
    return o;
}

} // namespace

// =============================================================================
// Golden-file fixtures
// =============================================================================

TEST(AsciiFixtureTest, BellMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::ASCII, {}),
              load_golden("ascii/bell.txt"));
}

TEST(AsciiFixtureTest, BellAsciiSafeMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::ASCII, ascii_safe_opts()),
              load_golden("ascii/bell.ascii_safe.txt"));
}

TEST(AsciiFixtureTest, BellShowClbitsMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::ASCII, show_clbits_opts()),
              load_golden("ascii/bell.show_clbits.txt"));
}

TEST(AsciiFixtureTest, GhzMatchesGolden) {
    EXPECT_EQ(ghz_3q().draw(DrawMode::ASCII, {}),
              load_golden("ascii/ghz.txt"));
}

TEST(AsciiFixtureTest, ParametricMatchesGolden) {
    EXPECT_EQ(parametric_rotations().draw(DrawMode::ASCII, {}),
              load_golden("ascii/parametric.txt"));
}

TEST(AsciiFixtureTest, ParametricRawParamsMatchesGolden) {
    EXPECT_EQ(parametric_rotations().draw(DrawMode::ASCII, raw_param_opts()),
              load_golden("ascii/parametric.raw.txt"));
}

TEST(AsciiFixtureTest, ParametricNoParamsMatchesGolden) {
    EXPECT_EQ(parametric_rotations().draw(DrawMode::ASCII, no_params_opts()),
              load_golden("ascii/parametric.no_params.txt"));
}

TEST(AsciiFixtureTest, BarrierAndMeasureMatchesGolden) {
    EXPECT_EQ(barrier_and_measure().draw(DrawMode::ASCII, {}),
              load_golden("ascii/barrier_measure.txt"));
}

TEST(AsciiFixtureTest, FeedforwardShowClbitsMatchesGolden) {
    EXPECT_EQ(conditional_feedforward().draw(DrawMode::ASCII, show_clbits_opts()),
              load_golden("ascii/feedforward.show_clbits.txt"));
}

TEST(AsciiFixtureTest, NonContigUnitaryMatchesGolden) {
    EXPECT_EQ(non_contiguous_unitary().draw(DrawMode::ASCII, {}),
              load_golden("ascii/noncontig_unitary.txt"));
}

TEST(AsciiFixtureTest, TallBoxMatchesGolden) {
    EXPECT_EQ(tallbox_demo().draw(DrawMode::ASCII, {}),
              load_golden("ascii/tallbox.txt"));
}

TEST(AsciiFixtureTest, AllSingleQubitGatesMatchesGolden) {
    EXPECT_EQ(all_single_qubit_unparam().draw(DrawMode::ASCII, {}),
              load_golden("ascii/all_1q_unparam.txt"));
}

TEST(AsciiFixtureTest, ResetThenHadamardMatchesGolden) {
    EXPECT_EQ(reset_and_h().draw(DrawMode::ASCII, {}),
              load_golden("ascii/reset_h.txt"));
}

// =============================================================================
// Structural assertions independent of the golden tree
// =============================================================================

TEST(AsciiFixtureTest, AsciiSafeOutputContainsNoUtf8) {
    const std::string out =
        bell_with_measures().draw(DrawMode::ASCII, ascii_safe_opts());
    EXPECT_FALSE(has_non_ascii(out));
}

TEST(AsciiFixtureTest, OutputEndsWithNewline) {
    const std::string out = bell_with_measures().draw(DrawMode::ASCII, {});
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(out.back(), '\n');
}

TEST(AsciiFixtureTest, EmptyCircuitProducesNonEmptyOutput) {
    // Even a circuit with no instructions should render qubit wire prefixes.
    const std::string out = empty_2q().draw(DrawMode::ASCII, {});
    EXPECT_FALSE(out.empty());
}
