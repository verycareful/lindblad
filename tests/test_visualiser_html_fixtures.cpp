// =============================================================================
// tests/test_visualiser_html_fixtures.cpp : R.1.10.1 HtmlFixtureTest
// =============================================================================
// Golden-file byte-exact comparison for the HTML renderer plus structural
// invariants. The HTML renderer wraps the SVG output in a styled page shell;
// tests verify the DOCTYPE / head / body shape, the embedded SVG, the
// lb-meta caption, and the optional legend toggle.

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

DrawOptions legend_opts() {
    DrawOptions o;
    o.include_legend = true;
    return o;
}

} // namespace

// =============================================================================
// Golden-file fixtures
// =============================================================================

TEST(HtmlFixtureTest, BellMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::HTML, {}),
              load_golden("html/bell.txt"));
}

TEST(HtmlFixtureTest, GhzMatchesGolden) {
    EXPECT_EQ(ghz_3q().draw(DrawMode::HTML, {}),
              load_golden("html/ghz.txt"));
}

TEST(HtmlFixtureTest, BellWithLegendMatchesGolden) {
    EXPECT_EQ(bell_with_measures().draw(DrawMode::HTML, legend_opts()),
              load_golden("html/bell_legend.txt"));
}

// =============================================================================
// Structural invariants
// =============================================================================

TEST(HtmlFixtureTest, OutputStartsWithDoctype) {
    const std::string out = bell_with_measures().draw(DrawMode::HTML, {});
    EXPECT_EQ(out.find("<!DOCTYPE html>"), 0u);
}

TEST(HtmlFixtureTest, OutputCarriesUtf8Charset) {
    const std::string out = bell_with_measures().draw(DrawMode::HTML, {});
    EXPECT_TRUE(contains(out, "charset=\"utf-8\""));
}

TEST(HtmlFixtureTest, HtmlShellWrapsBody) {
    const std::string out = bell_with_measures().draw(DrawMode::HTML, {});
    EXPECT_TRUE(contains(out, "<html"));
    EXPECT_TRUE(contains(out, "<head>"));
    EXPECT_TRUE(contains(out, "<body>"));
    EXPECT_TRUE(contains(out, "</body>"));
    EXPECT_TRUE(contains(out, "</html>"));
}

TEST(HtmlFixtureTest, EmbeddedSvgPresent) {
    const std::string out = bell_with_measures().draw(DrawMode::HTML, {});
    EXPECT_TRUE(contains(out, "<svg"));
    EXPECT_TRUE(contains(out, "</svg>"));
}

TEST(HtmlFixtureTest, LbMetaCaptionPresent) {
    const std::string out = bell_with_measures().draw(DrawMode::HTML, {});
    EXPECT_TRUE(contains(out, "lb-meta"));
}

TEST(HtmlFixtureTest, HoverRulePresent) {
    // Pure-CSS hover styling for .lb-glyph is the value-add over plain SVG.
    const std::string out = bell_with_measures().draw(DrawMode::HTML, {});
    EXPECT_TRUE(contains(out, ".lb-glyph:hover"));
}

TEST(HtmlFixtureTest, LegendAbsentByDefault) {
    const std::string out = bell_with_measures().draw(DrawMode::HTML, {});
    EXPECT_EQ(out.find("lb-legend"), std::string::npos);
}

TEST(HtmlFixtureTest, LegendPresentWhenIncludeLegend) {
    const std::string out =
        bell_with_measures().draw(DrawMode::HTML, legend_opts());
    EXPECT_TRUE(contains(out, "lb-legend"));
}
