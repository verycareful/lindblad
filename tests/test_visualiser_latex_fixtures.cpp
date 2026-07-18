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

// True when any byte is outside 7-bit ASCII. A compilable pdflatex document
// must not carry raw UTF-8 codepoints in the Quantikz body, so the LaTeX
// renderer's output must be pure ASCII.
bool has_non_ascii(const std::string& s) {
    for (unsigned char c : s) {
        if (c >= 128) { return true; }
    }
    return false;
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

// =============================================================================
// pdflatex compilability : no raw UTF-8 glyphs (R.1.17.2, #65)
// =============================================================================
// The pretty parameter formatter emits UTF-8 pi (U+03C0) and middle dot
// (U+00B7). Earlier the LaTeX renderer wrapped those raw bytes in \text{...},
// which pdflatex cannot typeset ("Unicode character not set up for use with
// LaTeX"). The renderer now translates the known glyphs to math-mode control
// sequences (\pi, \cdot) and emits them in math mode, so the .tex compiles.

TEST(LatexFixtureTest, PiRendersAsMathCommandNotRawGlyph) {
    // parametric_rotations uses RX(pi/2), RY(pi/4), CRX(pi/3): every angle
    // snaps to a pi multiple, so the output must carry \pi in math mode and
    // no raw UTF-8 pi byte.
    const std::string out = parametric_rotations().draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\pi"));
    EXPECT_FALSE(contains(out, "\\text{"));
    EXPECT_FALSE(has_non_ascii(out));
}

TEST(LatexFixtureTest, MultiParamPiRendersAsMathCommand) {
    // The U(pi/2, pi/4, pi/6) fixture exercises a multi-parameter suffix: each
    // pi becomes \pi, and the comma-separated list stays in math mode.
    const std::string out = all_single_qubit_param().draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\pi"));
    EXPECT_FALSE(has_non_ascii(out));
}

TEST(LatexFixtureTest, MiddleDotRendersAsCdot) {
    // A symbolic multiplication parameter (a*b) formats with the UTF-8 middle
    // dot in the pretty formatter. The LaTeX renderer must translate it to the
    // \cdot math command with no raw byte surviving.
    QuantumCircuit qc(1, 0, "middot");
    qc.rx(0.0, 0);
    qc.instructions.back().param_exprs.push_back(
        ParamExpr::make_binary('*', ParamExpr::make_name("a"),
                                    ParamExpr::make_name("b")));
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\cdot"));
    EXPECT_FALSE(has_non_ascii(out));
}

TEST(LatexFixtureTest, RawParamFormatKeepsAsciiTextWrap) {
    // Raw (%.4f) parameters have no non-ASCII glyph, so they keep the prior
    // \text{...} behaviour: the fix is scoped to glyph-bearing suffixes only.
    DrawOptions raw;
    raw.param_format = ParamFormat::Raw;
    const std::string out = parametric_rotations().draw(DrawMode::LATEX, raw);
    EXPECT_FALSE(has_non_ascii(out));
    EXPECT_TRUE(contains(out, "\\text{"));
}
