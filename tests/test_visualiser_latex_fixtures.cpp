// =============================================================================
// tests/test_visualiser_latex_fixtures.cpp : R.1.10.1 LatexFixtureTest
// =============================================================================
// Golden-file byte-exact comparison for the LaTeX (Quantikz) renderer plus
// structural invariants. The renderer emits a `quantikz` environment with no
// document shell; tests verify the envelope, the row prefixes, the default
// \qw fill, and the per-gate token mapping.

#include "lindblad/circuit.hpp"
#include "lindblad/visualisation.hpp"

#include "golden_helpers.hpp"
#include "visualiser_fixtures.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace lindblad;
using namespace lindblad::vfx;

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

DrawOptions show_clbits_opts() {
    DrawOptions o;
    o.show_clbits = true;
    return o;
}

DrawOptions legend_opts() {
    DrawOptions o;
    o.include_legend = true;
    return o;
}

} // namespace

// =============================================================================
// Golden-file fixtures
// =============================================================================

TEST(LatexFixtureTest, BellMatchesGolden) {
    EXPECT_EQ(bell_unmeasured().draw(DrawMode::LATEX, {}),
              load_golden("latex/bell_unmeasured.txt"));
}

TEST(LatexFixtureTest, BellWithMeasuresMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::LATEX, {}),
              load_golden("latex/bell.txt"));
}

TEST(LatexFixtureTest, BellShowClbitsMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::LATEX, show_clbits_opts()),
              load_golden("latex/bell.show_clbits.txt"));
}

TEST(LatexFixtureTest, GhzMatchesGolden) {
    EXPECT_EQ(ghz_3q().draw(DrawMode::LATEX, {}),
              load_golden("latex/ghz.txt"));
}

TEST(LatexFixtureTest, ParametricMatchesGolden) {
    EXPECT_EQ(parametric_rotations().draw(DrawMode::LATEX, {}),
              load_golden("latex/parametric.txt"));
}

TEST(LatexFixtureTest, TallBoxMatchesGolden) {
    EXPECT_EQ(tallbox_demo().draw(DrawMode::LATEX, {}),
              load_golden("latex/tallbox.txt"));
}

TEST(LatexFixtureTest, AllThreeQubitMatchesGolden) {
    EXPECT_EQ(all_three_qubit().draw(DrawMode::LATEX, {}),
              load_golden("latex/all_3q.txt"));
}

// =============================================================================
// Structural invariants
// =============================================================================

TEST(LatexFixtureTest, OutputStartsWithBeginQuantikz) {
    const std::string out = bell_unmeasured().draw(DrawMode::LATEX, {});
    EXPECT_EQ(out.find("\\begin{quantikz}"), 0u);
}

TEST(LatexFixtureTest, OutputClosesWithEndQuantikz) {
    const std::string out = bell_unmeasured().draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\end{quantikz}"));
}

TEST(LatexFixtureTest, RowsUseLstickWithMathQubitLabel) {
    const std::string out = ghz_3q().draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\lstick{$q_{0}$}"));
    EXPECT_TRUE(contains(out, "\\lstick{$q_{1}$}"));
    EXPECT_TRUE(contains(out, "\\lstick{$q_{2}$}"));
}

TEST(LatexFixtureTest, EmptyCellsDefaultToQw) {
    const std::string out = ghz_3q().draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\qw"));
}

TEST(LatexFixtureTest, CWireRowAppearsWhenShowClbits) {
    const std::string out =
        bell_with_measures().draw(DrawMode::LATEX, show_clbits_opts());
    EXPECT_TRUE(contains(out, "\\lstick{$c$}"));
    EXPECT_TRUE(contains(out, "\\setwiretype{c}"));
}

TEST(LatexFixtureTest, NoDocumentShellEmitted) {
    // The renderer must NOT emit \documentclass or \begin{document}: the
    // output is meant to be pasted into the caller's existing LaTeX source.
    const std::string out = bell_unmeasured().draw(DrawMode::LATEX, {});
    EXPECT_EQ(out.find("\\documentclass"),     std::string::npos);
    EXPECT_EQ(out.find("\\begin{document}"),   std::string::npos);
}

TEST(LatexFixtureTest, LegendOptionAddsCommentBlock) {
    // The include_legend toggle currently appends a comment annotation; the
    // exact contents are tested via the structural assertion that something
    // about the legend appears in the output.
    const std::string out = bell_unmeasured().draw(DrawMode::LATEX, legend_opts());
    EXPECT_TRUE(contains(out, "legend"));
}
