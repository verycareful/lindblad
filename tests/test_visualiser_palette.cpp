// =============================================================================
// tests/test_visualiser_palette.cpp : R.1.10.1 PaletteTest
// =============================================================================
// Black-box palette assertions on each backend. Verifies the glyph character
// set chosen at runtime: ASCII Unicode vs ASCII-safe, SVG inline CSS classes,
// LaTeX Quantikz tokens for daggers / subscripted rotations / barriers /
// resets / measurements. Each test exercises one palette dimension on a
// representative fixture; large-scale per-gate coverage lives in the fixture
// suites.

#include "lindblad/circuit.hpp"
#include "lindblad/visualisation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

using namespace lindblad;

namespace {

// UTF-8 byte sequences for the ASCII renderer's default palette glyphs.
const std::string kUnicodeWire    = "\xe2\x94\x80"; // U+2500 BOX DRAWINGS LIGHT HORIZONTAL
const std::string kUnicodeBoxL    = "\xe2\x94\xa4"; // U+2524
const std::string kUnicodeBoxR    = "\xe2\x94\x9c"; // U+251c
const std::string kUnicodeCtrl    = "\xe2\x97\x8f"; // U+25cf
const std::string kUnicodeXor     = "\xe2\x8a\x95"; // U+2295
const std::string kUnicodeSwap    = "\xe2\x9c\x95"; // U+2715
const std::string kUnicodeStrutV  = "\xe2\x94\x82"; // U+2502
const std::string kUnicodeStrutX  = "\xe2\x94\xbc"; // U+253c
const std::string kUnicodeBarrier = "\xe2\x94\x8a"; // U+250a

// Helper: does `s` contain any byte >= 0x80 (i.e. any non-ASCII)?
bool has_non_ascii(const std::string& s) {
    return std::any_of(s.begin(), s.end(), [](char c) {
        return static_cast<unsigned char>(c) >= 0x80;
    });
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

} // namespace

// =============================================================================
// ASCII palette : Unicode defaults
// =============================================================================

TEST(PaletteTest, AsciiUnicodeWireOnH) {
    QuantumCircuit qc(1);
    qc.h(0);
    const std::string out = qc.draw(DrawMode::ASCII, {});
    EXPECT_TRUE(contains(out, kUnicodeWire));
}

TEST(PaletteTest, AsciiUnicodeBoxBordersOnSingleQubitGate) {
    QuantumCircuit qc(1);
    qc.h(0);
    const std::string out = qc.draw(DrawMode::ASCII, {});
    EXPECT_TRUE(contains(out, kUnicodeBoxL));
    EXPECT_TRUE(contains(out, kUnicodeBoxR));
}

TEST(PaletteTest, AsciiUnicodeControlAndXorOnCx) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    const std::string out = qc.draw(DrawMode::ASCII, {});
    EXPECT_TRUE(contains(out, kUnicodeCtrl));
    EXPECT_TRUE(contains(out, kUnicodeXor));
}

TEST(PaletteTest, AsciiUnicodeStrutPipeAcrossCxGap) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    const std::string out = qc.draw(DrawMode::ASCII, {});
    EXPECT_TRUE(contains(out, kUnicodeStrutV));
}

TEST(PaletteTest, AsciiUnicodeSwapGlyphsOnSwap) {
    QuantumCircuit qc(2);
    qc.swap(0, 1);
    const std::string out = qc.draw(DrawMode::ASCII, {});
    EXPECT_TRUE(contains(out, kUnicodeSwap));
}

TEST(PaletteTest, AsciiUnicodeBarrierGlyph) {
    QuantumCircuit qc(2);
    qc.barrier({});
    const std::string out = qc.draw(DrawMode::ASCII, {});
    EXPECT_TRUE(contains(out, kUnicodeBarrier));
}

// =============================================================================
// ASCII palette : ascii_safe fallback
// =============================================================================

TEST(PaletteTest, AsciiSafeProducesNoMultibyteBytes) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).swap(0, 1).barrier({});
    DrawOptions opts;
    opts.ascii_safe = true;
    const std::string out = qc.draw(DrawMode::ASCII, opts);
    EXPECT_FALSE(has_non_ascii(out))
        << "ascii_safe mode must not emit any byte >= 0x80";
}

TEST(PaletteTest, AsciiSafeReplacesWireWithDash) {
    QuantumCircuit qc(1);
    qc.h(0);
    DrawOptions opts;
    opts.ascii_safe = true;
    const std::string out = qc.draw(DrawMode::ASCII, opts);
    EXPECT_TRUE(contains(out, "-"));
}

TEST(PaletteTest, AsciiSafeReplacesBoxBorders) {
    QuantumCircuit qc(1);
    qc.h(0);
    DrawOptions opts;
    opts.ascii_safe = true;
    const std::string out = qc.draw(DrawMode::ASCII, opts);
    EXPECT_TRUE(contains(out, "[H]"));
}

TEST(PaletteTest, AsciiSafeReplacesControlBulletWithStar) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    DrawOptions opts;
    opts.ascii_safe = true;
    const std::string out = qc.draw(DrawMode::ASCII, opts);
    EXPECT_TRUE(contains(out, "*"));
}

// =============================================================================
// SVG palette : inline style classes + colour attributes
// =============================================================================

TEST(PaletteTest, SvgIncludesEveryLbClass) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    const std::string out = qc.draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, ".lb-wire"));
    EXPECT_TRUE(contains(out, ".lb-gate"));
    EXPECT_TRUE(contains(out, ".lb-ctrl"));
    EXPECT_TRUE(contains(out, ".lb-label"));
    EXPECT_TRUE(contains(out, ".lb-strut"));
    EXPECT_TRUE(contains(out, ".lb-rowlabel"));
}

TEST(PaletteTest, SvgGlyphCarriesDataGateAttribute) {
    QuantumCircuit qc(1);
    qc.h(0);
    const std::string out = qc.draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, "data-gate=\"h\""));
}

TEST(PaletteTest, SvgUsesHadamardCatalogueFill) {
    // The Tier 1 catalogue assigns #e8eef9 to the Hadamard family.
    QuantumCircuit qc(1);
    qc.h(0);
    const std::string out = qc.draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, "#e8eef9"));
}

TEST(PaletteTest, SvgUsesPauliFamilyFill) {
    // Pauli gates share #fce8e8 fill.
    QuantumCircuit qc(1);
    qc.x(0);
    const std::string out = qc.draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, "#fce8e8"));
}

TEST(PaletteTest, SvgBarrierUsesDashedStroke) {
    QuantumCircuit qc(2);
    qc.barrier({});
    const std::string out = qc.draw(DrawMode::SVG, {});
    EXPECT_TRUE(contains(out, "stroke-dasharray"));
}

// =============================================================================
// LaTeX palette : Quantikz tokens
// =============================================================================

TEST(PaletteTest, LatexHadamardEmitsPlainGateMacro) {
    QuantumCircuit qc(1);
    qc.h(0);
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\gate{H}"));
}

TEST(PaletteTest, LatexDaggerEmitsTexBackslashDagger) {
    QuantumCircuit qc(1);
    qc.sdg(0);
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "S^{\\dagger}"));
}

TEST(PaletteTest, LatexRotationUsesSubscript) {
    QuantumCircuit qc(1);
    qc.rx(1.5708, 0);
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "R_X"));
}

TEST(PaletteTest, LatexCxEmitsCtrlAndTarg) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\ctrl{1}"));
    EXPECT_TRUE(contains(out, "\\targ{}"));
}

TEST(PaletteTest, LatexSwapEmitsSwapAndTargX) {
    QuantumCircuit qc(2);
    qc.swap(0, 1);
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\swap{1}"));
    EXPECT_TRUE(contains(out, "\\targX{}"));
}

TEST(PaletteTest, LatexMeasureEmitsMeterMacro) {
    QuantumCircuit qc(1, 1);
    qc.measure(0, 0);
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\meter{}"));
}

TEST(PaletteTest, LatexResetEmitsKetZero) {
    QuantumCircuit qc(1);
    qc.reset(0);
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\push{\\ket{0}}"));
}

TEST(PaletteTest, LatexBarrierEmitsDashedBarrier) {
    QuantumCircuit qc(2);
    qc.barrier({});
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\barrier[\\dashed]"));
}

TEST(PaletteTest, LatexTallBoxUsesMultiWireGateMacro) {
    QuantumCircuit qc(2);
    qc.rxx(1.5708, 0, 1);
    const std::string out = qc.draw(DrawMode::LATEX, {});
    EXPECT_TRUE(contains(out, "\\gate[2]"));
}
