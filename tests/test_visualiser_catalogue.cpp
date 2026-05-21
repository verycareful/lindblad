// =============================================================================
// tests/test_visualiser_catalogue.cpp : R.1.10.1 GateCatalogueTest
// =============================================================================
// Direct assertions on the Tier 1, Tier 2, and Tier 3 catalogues plus
// glyph dispatch via build_glyph. Failures here mean a catalogue entry is
// missing or carries the wrong structure (e.g. CZ rendered as Ctrl+Dot
// rather than Ctrl+Ctrl), independent of how it is later rendered.

#include "lindblad/circuit.hpp"
#include "lindblad/visualisation.hpp"

#include "../src/visualisation/composite_catalogue.hpp"
#include "../src/visualisation/document.hpp"
#include "../src/visualisation/gate_builders.hpp"
#include "../src/visualisation/gate_symbols.hpp"
#include "../src/visualisation/layout.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <variant>

using namespace lindblad;
using namespace lindblad::viz;
using GT = Instruction::GateType;

namespace {

// Tiny factory: synthesise an Instruction with a given gate type and qubit
// list, no params. Lets each test exercise build_glyph in isolation.
Instruction make_inst(GT type, std::vector<int> qubits,
                      std::vector<double> params = {},
                      std::vector<int> clbits = {}) {
    Instruction inst;
    inst.type    = type;
    inst.qubits  = std::move(qubits);
    inst.params  = std::move(params);
    inst.clbits  = std::move(clbits);
    return inst;
}

// Return true iff the glyph contains a part of the given variant kind at
// the specified qubit row.
template <typename T>
bool has_part_at(const Glyph& g, int qrow) {
    for (const auto& kv : g.parts) {
        if (kv.first == qrow && std::holds_alternative<T>(kv.second)) {
            return true;
        }
    }
    return false;
}

template <typename T>
const T* find_part_at(const Glyph& g, int qrow) {
    for (const auto& kv : g.parts) {
        if (kv.first == qrow) {
            if (auto* p = std::get_if<T>(&kv.second)) { return p; }
        }
    }
    return nullptr;
}

DrawOptions default_opts() { return DrawOptions{}; }

} // namespace

// =============================================================================
// Tier 1 : symbol_catalogue() coverage
// =============================================================================

TEST(GateCatalogueTest, SymbolCatalogueIsNonEmpty) {
    EXPECT_GT(symbol_catalogue().size(), 0u);
}

TEST(GateCatalogueTest, HasEntryForEverySingleQubitBoxGate) {
    const auto& cat = symbol_catalogue();
    const std::vector<GT> required = {
        GT::H,   GT::X,   GT::Y,   GT::Z,
        GT::S,   GT::SDG, GT::T,   GT::TDG,
        GT::SX,  GT::SXDG,
        GT::RX,  GT::RY,  GT::RZ,  GT::P,
        GT::U,   GT::U1,  GT::U2,  GT::U3,
        GT::PARAM_RX, GT::PARAM_RY, GT::PARAM_RZ,
        GT::PARAM_P,  GT::PARAM_U
    };
    for (GT t : required) {
        EXPECT_NE(cat.find(t), cat.end())
            << "missing Tier 1 catalogue entry for GateType " << static_cast<int>(t);
    }
}

TEST(GateCatalogueTest, HadamardLabelAndShowParams) {
    const auto& sym = symbol_catalogue().at(GT::H);
    EXPECT_EQ(sym.label, "H");
    EXPECT_FALSE(sym.show_params);
}

TEST(GateCatalogueTest, RotationGatesShowParams) {
    EXPECT_TRUE(symbol_catalogue().at(GT::RX).show_params);
    EXPECT_TRUE(symbol_catalogue().at(GT::RY).show_params);
    EXPECT_TRUE(symbol_catalogue().at(GT::RZ).show_params);
}

TEST(GateCatalogueTest, PauliGatesShareColourPalette) {
    const auto& x = symbol_catalogue().at(GT::X);
    const auto& y = symbol_catalogue().at(GT::Y);
    const auto& z = symbol_catalogue().at(GT::Z);
    EXPECT_EQ(x.svg_fill,   y.svg_fill);
    EXPECT_EQ(x.svg_fill,   z.svg_fill);
    EXPECT_EQ(x.svg_stroke, y.svg_stroke);
    EXPECT_EQ(x.svg_stroke, z.svg_stroke);
}

TEST(GateCatalogueTest, DaggerLabelsContainDaggerCharacter) {
    // The catalogue uses the UTF-8 dagger (U+2020) in display labels.
    const std::string dagger = "\xe2\x80\xa0";
    EXPECT_NE(symbol_catalogue().at(GT::SDG).label.find(dagger),  std::string::npos);
    EXPECT_NE(symbol_catalogue().at(GT::TDG).label.find(dagger),  std::string::npos);
    EXPECT_NE(symbol_catalogue().at(GT::SXDG).label.find(dagger), std::string::npos);
}

TEST(GateCatalogueTest, ParamVariantsShareLabelWithNumericSiblings) {
    EXPECT_EQ(symbol_catalogue().at(GT::PARAM_RX).label,
              symbol_catalogue().at(GT::RX).label);
    EXPECT_EQ(symbol_catalogue().at(GT::PARAM_RY).label,
              symbol_catalogue().at(GT::RY).label);
    EXPECT_EQ(symbol_catalogue().at(GT::PARAM_RZ).label,
              symbol_catalogue().at(GT::RZ).label);
    EXPECT_EQ(symbol_catalogue().at(GT::PARAM_P).label,
              symbol_catalogue().at(GT::P).label);
}

TEST(GateCatalogueTest, LatexMacroFilledOnlyWhenDefaultIsWrong) {
    // Plain gates that take the default \gate{label} keep latex_macro empty.
    EXPECT_TRUE(symbol_catalogue().at(GT::H).latex_macro.empty());
    EXPECT_TRUE(symbol_catalogue().at(GT::X).latex_macro.empty());
    // Daggers and subscripted rotations have an explicit override.
    EXPECT_FALSE(symbol_catalogue().at(GT::SDG).latex_macro.empty());
    EXPECT_FALSE(symbol_catalogue().at(GT::RX).latex_macro.empty());
}

// =============================================================================
// Tier 2 : composite_catalogue() coverage
// =============================================================================

TEST(GateCatalogueTest, CompositeCatalogueIsNonEmpty) {
    EXPECT_GT(composite_catalogue().size(), 0u);
}

TEST(GateCatalogueTest, HasEntryForEveryControlledGate) {
    const auto& cat = composite_catalogue();
    const std::vector<GT> required = {
        GT::CX, GT::CY, GT::CZ, GT::CH,
        GT::SWAP, GT::ISWAP,
        GT::CRX, GT::CRY, GT::CRZ, GT::CP, GT::CU,
        GT::CCX, GT::CCZ, GT::CSWAP, GT::RCCX,
        GT::RXX, GT::RYY, GT::RZZ, GT::RZX, GT::ECR
    };
    for (GT t : required) {
        EXPECT_NE(cat.find(t), cat.end())
            << "missing Tier 2 catalogue entry for GateType " << static_cast<int>(t);
    }
}

TEST(GateCatalogueTest, CxIsCtrlBulletPlusXorTarget) {
    Glyph g = build_glyph(make_inst(GT::CX, {0, 1}), default_opts());
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 0));
    EXPECT_TRUE(has_part_at<XorTargetPart>(g, 1));
    EXPECT_TRUE(g.has_strut);
}

TEST(GateCatalogueTest, CzIsTwoCtrlBullets) {
    Glyph g = build_glyph(make_inst(GT::CZ, {0, 1}), default_opts());
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 0));
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 1));
    EXPECT_FALSE(has_part_at<XorTargetPart>(g, 1));
}

TEST(GateCatalogueTest, CczIsThreeCtrlBullets) {
    Glyph g = build_glyph(make_inst(GT::CCZ, {0, 1, 2}), default_opts());
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 0));
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 1));
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 2));
}

TEST(GateCatalogueTest, CcxIsTwoCtrlBulletsPlusXor) {
    Glyph g = build_glyph(make_inst(GT::CCX, {0, 1, 2}), default_opts());
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 0));
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 1));
    EXPECT_TRUE(has_part_at<XorTargetPart>(g, 2));
}

TEST(GateCatalogueTest, CyIsCtrlPlusYBox) {
    Glyph g = build_glyph(make_inst(GT::CY, {0, 1}), default_opts());
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 0));
    const BoxPart* box = find_part_at<BoxPart>(g, 1);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->label, "Y");
}

TEST(GateCatalogueTest, ChIsCtrlPlusHBox) {
    Glyph g = build_glyph(make_inst(GT::CH, {0, 1}), default_opts());
    const BoxPart* box = find_part_at<BoxPart>(g, 1);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->label, "H");
}

TEST(GateCatalogueTest, SwapIsTwoSwapX) {
    Glyph g = build_glyph(make_inst(GT::SWAP, {0, 1}), default_opts());
    EXPECT_TRUE(has_part_at<SwapXPart>(g, 0));
    EXPECT_TRUE(has_part_at<SwapXPart>(g, 1));
    EXPECT_TRUE(g.has_strut);
}

TEST(GateCatalogueTest, CswapIsCtrlPlusTwoSwapX) {
    Glyph g = build_glyph(make_inst(GT::CSWAP, {0, 1, 2}), default_opts());
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 0));
    EXPECT_TRUE(has_part_at<SwapXPart>(g, 1));
    EXPECT_TRUE(has_part_at<SwapXPart>(g, 2));
}

TEST(GateCatalogueTest, CrxParamLabelIncludesValue) {
    constexpr double kPi = 3.141592653589793;
    Glyph g = build_glyph(make_inst(GT::CRX, {0, 1}, {kPi / 2.0}), default_opts());
    const BoxPart* box = find_part_at<BoxPart>(g, 1);
    ASSERT_NE(box, nullptr);
    EXPECT_NE(box->label.find("RX"), std::string::npos);
    EXPECT_NE(box->label.find("("),  std::string::npos);
    EXPECT_NE(box->label.find(")"),  std::string::npos);
}

TEST(GateCatalogueTest, CrxHonoursShowParamsFalse) {
    DrawOptions opts;
    opts.show_params = false;
    constexpr double kPi = 3.141592653589793;
    Glyph g = build_glyph(make_inst(GT::CRX, {0, 1}, {kPi / 2.0}), opts);
    const BoxPart* box = find_part_at<BoxPart>(g, 1);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->label.find("("), std::string::npos);
}

TEST(GateCatalogueTest, IswapBothSlotsCarryBoxLabel) {
    Glyph g = build_glyph(make_inst(GT::ISWAP, {0, 1}), default_opts());
    const BoxPart* a = find_part_at<BoxPart>(g, 0);
    const BoxPart* b = find_part_at<BoxPart>(g, 1);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(a->label, b->label);
}

// =============================================================================
// Tier 2 : TallBox structure for interaction gates
// =============================================================================

TEST(GateCatalogueTest, RxxIsSingleTallBoxNoStrut) {
    Glyph g = build_glyph(make_inst(GT::RXX, {0, 1}, {1.5708}), default_opts());
    // TallBox emits exactly one BoxPart with rowspan = 2 and no strut.
    ASSERT_EQ(g.parts.size(), 1u);
    const BoxPart* box = find_part_at<BoxPart>(g, 0);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->rowspan, 2);
    EXPECT_FALSE(g.has_strut);
}

TEST(GateCatalogueTest, RyyRzzRzxAllTallBoxes) {
    constexpr double kPi = 3.141592653589793;
    for (GT t : { GT::RYY, GT::RZZ, GT::RZX }) {
        Glyph g = build_glyph(make_inst(t, {0, 1}, {kPi / 4.0}), default_opts());
        ASSERT_EQ(g.parts.size(), 1u) << "TallBox should emit one part: type "
                                       << static_cast<int>(t);
        const BoxPart* box = find_part_at<BoxPart>(g, 0);
        ASSERT_NE(box, nullptr);
        EXPECT_EQ(box->rowspan, 2);
        EXPECT_FALSE(g.has_strut);
    }
}

TEST(GateCatalogueTest, EcrIsTallBoxWithoutParams) {
    Glyph g = build_glyph(make_inst(GT::ECR, {0, 1}), default_opts());
    const BoxPart* box = find_part_at<BoxPart>(g, 0);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->rowspan, 2);
    // ECR has no parameters, so the label is the bare "ECR".
    EXPECT_EQ(box->label, "ECR");
}

TEST(GateCatalogueTest, RxxLabelIncludesParam) {
    constexpr double kPi = 3.141592653589793;
    Glyph g = build_glyph(make_inst(GT::RXX, {0, 1}, {kPi / 4.0}), default_opts());
    const BoxPart* box = find_part_at<BoxPart>(g, 0);
    ASSERT_NE(box, nullptr);
    EXPECT_NE(box->label.find("RXX"), std::string::npos);
    EXPECT_NE(box->label.find("("),   std::string::npos);
}

// =============================================================================
// Tier 3 : hand-written builders
// =============================================================================

TEST(GateCatalogueTest, BarrierBuilderEmitsBarrierPartPerQubit) {
    Instruction inst = make_inst(GT::BARRIER, {0, 2});
    Glyph g = build_barrier_glyph(inst, default_opts());
    EXPECT_EQ(g.parts.size(), 2u);
    EXPECT_TRUE(has_part_at<BarrierPart>(g, 0));
    EXPECT_TRUE(has_part_at<BarrierPart>(g, 2));
    EXPECT_FALSE(g.has_strut);
}

TEST(GateCatalogueTest, BarrierBuilderEmitsNoPartsForEmptyQubitList) {
    Instruction inst = make_inst(GT::BARRIER, {});
    Glyph g = build_barrier_glyph(inst, default_opts());
    // Builder leaves the parts empty; the layout pass promotes the
    // instruction to a full-width column break separately.
    EXPECT_TRUE(g.parts.empty());
}

TEST(GateCatalogueTest, MeasureBuilderRecordsClbitOnPart) {
    Instruction inst = make_inst(GT::MEASURE, {1}, {}, {3});
    Glyph g = build_measure_glyph(inst, default_opts());
    const MeasurePart* mp = find_part_at<MeasurePart>(g, 1);
    ASSERT_NE(mp, nullptr);
    EXPECT_EQ(mp->clbit, 3);
}

TEST(GateCatalogueTest, MeasureBuilderUsesSentinelClbitWhenAbsent) {
    Instruction inst = make_inst(GT::MEASURE, {0});
    Glyph g = build_measure_glyph(inst, default_opts());
    const MeasurePart* mp = find_part_at<MeasurePart>(g, 0);
    ASSERT_NE(mp, nullptr);
    EXPECT_EQ(mp->clbit, -1);
}

TEST(GateCatalogueTest, ResetBuilderEmitsResetPartOnTarget) {
    Instruction inst = make_inst(GT::RESET, {2});
    Glyph g = build_reset_glyph(inst, default_opts());
    EXPECT_TRUE(has_part_at<ResetPart>(g, 2));
    EXPECT_FALSE(g.has_strut);
}

TEST(GateCatalogueTest, UnitaryBuilderUsesProvidedLabel) {
    Instruction inst;
    inst.type   = GT::UNITARY;
    inst.qubits = {0, 1};
    inst.label  = "MyUop";
    Glyph g = build_unitary_glyph(inst, default_opts());
    const BoxPart* box = find_part_at<BoxPart>(g, 0);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->label, "MyUop");
    EXPECT_EQ(box->rowspan, 2);
}

TEST(GateCatalogueTest, UnitaryBuilderFallsBackToUWhenLabelEmpty) {
    Instruction inst;
    inst.type   = GT::UNITARY;
    inst.qubits = {0};
    Glyph g = build_unitary_glyph(inst, default_opts());
    const BoxPart* box = find_part_at<BoxPart>(g, 0);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->label, "U");
}

TEST(GateCatalogueTest, UnitaryBuilderRowspanCoversNonContiguousRange) {
    Instruction inst;
    inst.type   = GT::UNITARY;
    inst.qubits = {0, 3};
    inst.label  = "BigU";
    Glyph g = build_unitary_glyph(inst, default_opts());
    const BoxPart* box = find_part_at<BoxPart>(g, 0);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->rowspan, 4); // rows 0..3 inclusive
}

TEST(GateCatalogueTest, UnitaryBuilderEmptyQubitsReturnsEmptyGlyph) {
    Instruction inst;
    inst.type = GT::UNITARY;
    Glyph g = build_unitary_glyph(inst, default_opts());
    EXPECT_TRUE(g.parts.empty());
}

// =============================================================================
// build_glyph dispatch : Tier 3 -> Tier 2 -> Tier 1
// =============================================================================

TEST(GateCatalogueTest, BuildGlyphDispatchesMeasureToTier3) {
    Glyph g = build_glyph(make_inst(GT::MEASURE, {0}, {}, {0}), default_opts());
    EXPECT_TRUE(has_part_at<MeasurePart>(g, 0));
}

TEST(GateCatalogueTest, BuildGlyphDispatchesCxToTier2) {
    Glyph g = build_glyph(make_inst(GT::CX, {0, 1}), default_opts());
    EXPECT_TRUE(has_part_at<CtrlBulletPart>(g, 0));
}

TEST(GateCatalogueTest, BuildGlyphDispatchesHadamardToTier1) {
    Glyph g = build_glyph(make_inst(GT::H, {0}), default_opts());
    const BoxPart* box = find_part_at<BoxPart>(g, 0);
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(box->label, "H");
}

TEST(GateCatalogueTest, BuildGlyphStampsDataGateFromInstructionName) {
    Glyph g = build_glyph(make_inst(GT::H, {0}), default_opts());
    EXPECT_EQ(g.data_gate, "h");
}
