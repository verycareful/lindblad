// R.1.12.1 total-coverage suite, Batch 5: circuit visualisation options.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 5".
//
// Exercises draw() across ASCII/SVG/LaTeX/HTML and the DrawOptions knobs
// (fold_width, show_clbits, show_params, ascii_safe, ParamFormat) on a fixture
// circuit containing controls, swaps, barriers, measures, conditions and
// parameters; plus draw_to_file (content equality + unwritable-path throw) and
// the to_ascii() compatibility wrapper. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace lindblad;

namespace {

// A circuit covering every render glyph class.
QuantumCircuit fixture() {
    QuantumCircuit qc(3, 2, "viz_fixture");
    qc.h(0);
    qc.cx(0, 1);            // control + target
    qc.swap(1, 2);          // swap glyph
    qc.rz(PI_2, 0);         // parameterised (pi-snappable)
    qc.barrier();           // barrier glyph
    qc.p_if(0.5, 2, 0);     // conditioned gate
    qc.measure(0, 0);       // measure glyph
    return qc;
}

bool is_pure_ascii(const std::string& s) {
    for (unsigned char c : s) if (c >= 128) return false;
    return true;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace

// =============================================================================
// Renderer backends
// =============================================================================

TEST(R1121Viz, EveryModeProducesOutput) {
    auto qc = fixture();
    EXPECT_FALSE(qc.draw(DrawMode::ASCII).empty());
    EXPECT_NE(qc.draw(DrawMode::SVG).find("<svg"), std::string::npos);
    EXPECT_FALSE(qc.draw(DrawMode::LATEX).empty());
    EXPECT_FALSE(qc.draw(DrawMode::HTML).empty());
}

TEST(R1121Viz, ToAsciiMatchesDrawAscii) {
    auto qc = fixture();
    EXPECT_EQ(qc.to_ascii(), qc.draw(DrawMode::ASCII, DrawOptions{}));
}

// =============================================================================
// DrawOptions knobs
// =============================================================================

TEST(R1121Viz, AsciiSafeAvoidsNonAsciiBytes) {
    auto qc = fixture();
    DrawOptions safe;
    safe.ascii_safe = true;
    safe.param_format = ParamFormat::Raw;  // decimal params, no pi glyph
    EXPECT_TRUE(is_pure_ascii(qc.draw(DrawMode::ASCII, safe)))
        << "ascii_safe must avoid UTF-8 box-drawing characters";
}

TEST(R1121Viz, FoldWidthIsAcceptedAndNonDestructive) {
    // The ASCII renderer currently emits unfolded output regardless of
    // fold_width (folding is a documented future refinement; see
    // test_visualiser_folding.cpp). Pin the current contract: every fold_width
    // is accepted, never crashes, and yields non-empty output; fold_width = 0
    // (the disable sentinel) matches the default.
    QuantumCircuit wide(2);
    for (int i = 0; i < 40; ++i) (i % 2 == 0) ? wide.h(0) : wide.x(0);
    for (int w : {0, 10, 40, 120}) {
        DrawOptions opts;
        opts.fold_width = w;
        EXPECT_FALSE(wide.draw(DrawMode::ASCII, opts).empty());
    }
    DrawOptions zero;
    zero.fold_width = 0;
    EXPECT_EQ(wide.draw(DrawMode::ASCII, zero), wide.draw(DrawMode::ASCII, DrawOptions{}));
}

TEST(R1121Viz, ShowClbitsTogglesClassicalWire) {
    auto qc = fixture();
    DrawOptions hidden;
    hidden.show_clbits = false;
    DrawOptions shown;
    shown.show_clbits = true;
    EXPECT_NE(qc.draw(DrawMode::ASCII, hidden), qc.draw(DrawMode::ASCII, shown));
}

TEST(R1121Viz, ShowParamsTogglesParameterText) {
    auto qc = fixture();
    DrawOptions with;
    with.show_params = true;
    DrawOptions without;
    without.show_params = false;
    EXPECT_NE(qc.draw(DrawMode::ASCII, with), qc.draw(DrawMode::ASCII, without));
}

TEST(R1121Viz, ParamFormatPrettyVsRaw) {
    QuantumCircuit qc(1);
    qc.rz(PI_2, 0);  // exactly pi/2 -> Pretty should snap, Raw should not
    DrawOptions pretty;
    pretty.param_format = ParamFormat::Pretty;
    DrawOptions raw;
    raw.param_format = ParamFormat::Raw;
    EXPECT_NE(qc.draw(DrawMode::ASCII, pretty), qc.draw(DrawMode::ASCII, raw));
}

// The full option grid across every renderer must produce non-empty, valid
// output and never crash. ascii_safe output stays pure ASCII in ASCII mode.
TEST(R1121Viz, OptionGridProducesValidOutputEveryMode) {
    auto qc = fixture();
    const DrawMode modes[] = {DrawMode::ASCII, DrawMode::SVG, DrawMode::LATEX,
                              DrawMode::HTML};
    for (bool show_clbits : {false, true})
     for (bool show_params : {false, true})
      for (bool ascii_safe : {false, true})
       for (ParamFormat pf : {ParamFormat::Pretty, ParamFormat::Raw})
        for (int legend : {0, 1}) {
            DrawOptions o;
            o.show_clbits = show_clbits;
            o.show_params = show_params;
            o.ascii_safe = ascii_safe;
            o.param_format = pf;
            o.include_legend = (legend != 0);
            for (DrawMode m : modes) {
                std::string out = qc.draw(m, o);
                EXPECT_FALSE(out.empty());
                if (m == DrawMode::SVG) EXPECT_NE(out.find("<svg"), std::string::npos);
                if (m == DrawMode::ASCII && ascii_safe && pf == ParamFormat::Raw)
                    EXPECT_TRUE(is_pure_ascii(out));
            }
        }
}

// pi-snap table: Pretty snaps recognised multiples of pi (output differs from
// Raw), while a non-rational-multiple value formats identically in both.
TEST(R1121Viz, ParamFormatPiSnapTable) {
    DrawOptions pretty; pretty.param_format = ParamFormat::Pretty;
    DrawOptions raw;    raw.param_format = ParamFormat::Raw;
    for (double snap : {PI_2, PI, PI / 4.0, 3.0 * PI / 4.0}) {
        QuantumCircuit qc(1);
        qc.rz(snap, 0);
        SCOPED_TRACE("snap angle " + std::to_string(snap));
        EXPECT_NE(qc.draw(DrawMode::ASCII, pretty), qc.draw(DrawMode::ASCII, raw))
            << "Pretty must snap a pi-multiple";
    }
    // A non-pi value renders identically under both formats (%.4f fallback).
    QuantumCircuit plain(1);
    plain.rz(0.37, 0);
    EXPECT_EQ(plain.draw(DrawMode::ASCII, pretty), plain.draw(DrawMode::ASCII, raw));
}

TEST(R1121Viz, CellSizeAffectsSvgAndHtmlGeometry) {
    auto qc = fixture();
    DrawOptions small; small.cell_width_px = 48; small.cell_height_px = 48;
    DrawOptions big;   big.cell_width_px = 96;   big.cell_height_px = 96;
    EXPECT_NE(qc.draw(DrawMode::SVG, small), qc.draw(DrawMode::SVG, big))
        << "SVG geometry must scale with cell size";
    EXPECT_NE(qc.draw(DrawMode::HTML, small), qc.draw(DrawMode::HTML, big));
}

TEST(R1121Viz, IncludeLegendChangesLatexAndHtml) {
    auto qc = fixture();
    DrawOptions with;    with.include_legend = true;
    DrawOptions without; without.include_legend = false;
    EXPECT_NE(qc.draw(DrawMode::LATEX, with), qc.draw(DrawMode::LATEX, without));
    EXPECT_NE(qc.draw(DrawMode::HTML, with), qc.draw(DrawMode::HTML, without));
    // The legend variant is not shorter than the plain one.
    EXPECT_GE(qc.draw(DrawMode::LATEX, with).size(),
              qc.draw(DrawMode::LATEX, without).size());
}

TEST(R1121Viz, ShowClbitsAffectsSvgRendering) {
    auto qc = fixture();
    DrawOptions hidden; hidden.show_clbits = false;
    DrawOptions shown;  shown.show_clbits = true;
    EXPECT_NE(qc.draw(DrawMode::SVG, hidden), qc.draw(DrawMode::SVG, shown))
        << "the classical wire toggles in SVG too";
}

// =============================================================================
// draw_to_file
// =============================================================================

TEST(R1121Viz, DrawToFileWritesRenderedContent) {
    auto qc = fixture();
    const std::string path = "r1121_viz_out.svg";
    qc.draw_to_file(path, DrawMode::SVG);
    EXPECT_EQ(read_file(path), qc.draw(DrawMode::SVG));
    std::remove(path.c_str());
}

TEST(R1121Viz, DrawToFileThrowsOnUnwritablePath) {
    auto qc = fixture();
    EXPECT_THROW(
        qc.draw_to_file("/no_such_dir_r1121/inner/out.txt", DrawMode::ASCII),
        std::runtime_error);
}
