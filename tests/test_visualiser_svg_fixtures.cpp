// =============================================================================
// tests/test_visualiser_svg_fixtures.cpp : R.1.10.1 SvgFixtureTest
// =============================================================================
// Golden-file byte-exact comparison for the SVG renderer plus structural
// invariants that hold across every fixture (XML prolog, encoding declaration,
// class palette presence, data-* attribute presence on glyph groups).

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

} // namespace

// =============================================================================
// Golden-file fixtures
// =============================================================================

TEST(SvgFixtureTest, BellMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::SVG, {}),
              load_golden("svg/bell.txt"));
}

TEST(SvgFixtureTest, BellShowClbitsMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::SVG, show_clbits_opts()),
              load_golden("svg/bell.show_clbits.txt"));
}

TEST(SvgFixtureTest, GhzMatchesGolden) {
    EXPECT_EQ(ghz_3q().draw(DrawMode::SVG, {}),
              load_golden("svg/ghz.txt"));
}

TEST(SvgFixtureTest, ParametricMatchesGolden) {
    EXPECT_EQ(parametric_rotations().draw(DrawMode::SVG, {}),
              load_golden("svg/parametric.txt"));
}

TEST(SvgFixtureTest, TallBoxMatchesGolden) {
    EXPECT_EQ(tallbox_demo().draw(DrawMode::SVG, {}),
              load_golden("svg/tallbox.txt"));
}

TEST(SvgFixtureTest, NonContigUnitaryMatchesGolden) {
    EXPECT_EQ(non_contiguous_unitary().draw(DrawMode::SVG, {}),
              load_golden("svg/noncontig_unitary.txt"));
}

// =============================================================================
// Structural invariants
// =============================================================================

TEST(SvgFixtureTest, OutputStartsWithXmlProlog) {
    const std::string out = bell_with_measures().draw(DrawMode::SVG, {});
    EXPECT_EQ(out.find("<?xml"), 0u);
}

TEST(SvgFixtureTest, OutputDeclaresUtf8Encoding) {
    const std::string out = bell_with_measures().draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, "encoding=\"UTF-8\""));
}

TEST(SvgFixtureTest, OutputIncludesInlineStyleBlock) {
    const std::string out = bell_with_measures().draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, "<style>"));
    EXPECT_TRUE(contains(out, "</style>"));
}

TEST(SvgFixtureTest, GlyphCarriesDataAttributes) {
    const std::string out = ghz_3q().draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, "data-gate="));
    EXPECT_TRUE(contains(out, "data-col="));
    EXPECT_TRUE(contains(out, "data-qubits="));
}

TEST(SvgFixtureTest, RootElementClosesWithEndTag) {
    const std::string out = bell_with_measures().draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, "</svg>"));
}

TEST(SvgFixtureTest, OneWireLinePerQubit) {
    // GHZ has 3 qubits; the SVG should contain at least 3 wire lines.
    const std::string out = ghz_3q().draw(DrawMode::SVG, {});
    std::string::size_type pos = 0;
    int count = 0;
    while ((pos = out.find("class=\"lb-wire\"", pos)) != std::string::npos) {
        ++count;
        pos += 1;
    }
    EXPECT_EQ(count, 3);
}
