// =============================================================================
// tests/test_visualiser_layout.cpp : R.1.10.1 DocumentLayoutTest
// =============================================================================
// Black-box coverage for lindblad::viz::build_document. Every test exercises
// one ASAP-packing rule or one edge case from the design spec without
// involving any renderer. Failures here mean the layout pass disagrees with
// the documented contract, regardless of which backend formats the result.

#include "lindblad/circuit.hpp"
#include "lindblad/visualisation.hpp"

#include "../src/visualisation/document.hpp"

#include "visualiser_fixtures.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <variant>

using namespace lindblad;
using namespace lindblad::viz;

namespace {

// Convenience: build a default-options CircuitDocument from a circuit.
CircuitDocument build_default(const QuantumCircuit& qc) {
    return build_document(qc, DrawOptions{});
}

// Count glyphs that hold at least one part of type T.
template <typename T>
int count_glyphs_with_part(const CircuitDocument& doc) {
    int n = 0;
    for (const auto& layer : doc.layers) {
        for (const auto& g : layer.glyphs) {
            for (const auto& kv : g.parts) {
                if (std::holds_alternative<T>(kv.second)) {
                    ++n;
                    break;
                }
            }
        }
    }
    return n;
}

// Look up the first glyph whose data_gate matches `name`. Returns a pointer
// so tests can assert on null when expected.
const Glyph* find_glyph(const CircuitDocument& doc, const std::string& name) {
    for (const auto& layer : doc.layers) {
        for (const auto& g : layer.glyphs) {
            if (g.data_gate == name) { return &g; }
        }
    }
    return nullptr;
}

} // namespace

// =============================================================================
// build_document : basic packing
// =============================================================================

TEST(DocumentLayoutTest, EmptyCircuitProducesEmptyDocument) {
    QuantumCircuit qc(2);
    auto doc = build_default(qc);
    EXPECT_EQ(doc.n_qubits, 2);
    EXPECT_TRUE(doc.layers.empty());
}

TEST(DocumentLayoutTest, SingleGateOccupiesColumnZero) {
    QuantumCircuit qc(1);
    qc.h(0);
    auto doc = build_default(qc);
    ASSERT_EQ(doc.layers.size(), 1u);
    EXPECT_EQ(doc.layers[0].column, 0);
    EXPECT_EQ(doc.layers[0].glyphs.size(), 1u);
    EXPECT_EQ(doc.layers[0].glyphs[0].column, 0);
}

TEST(DocumentLayoutTest, DisjointSingleQubitGatesPackIntoOneColumn) {
    QuantumCircuit qc(3);
    qc.h(0).x(1).y(2);
    auto doc = build_default(qc);
    // All three single-qubit gates touch disjoint qubits: ASAP packs them
    // into column 0 as three glyphs in the same layer.
    ASSERT_EQ(doc.layers.size(), 1u);
    EXPECT_EQ(doc.layers[0].glyphs.size(), 3u);
}

TEST(DocumentLayoutTest, OverlappingGatesSerialiseIntoSeparateColumns) {
    QuantumCircuit qc(1);
    qc.h(0).x(0);
    auto doc = build_default(qc);
    // Two gates on the same qubit must serialise.
    ASSERT_EQ(doc.layers.size(), 2u);
    EXPECT_EQ(doc.layers[0].column, 0);
    EXPECT_EQ(doc.layers[1].column, 1);
}

TEST(DocumentLayoutTest, GlyphColumnMatchesLayerColumn) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    auto doc = build_default(qc);
    for (const auto& layer : doc.layers) {
        for (const auto& g : layer.glyphs) {
            EXPECT_EQ(g.column, layer.column);
        }
    }
}

TEST(DocumentLayoutTest, MultipleGlyphsPerLayerKeepInsertionOrder) {
    QuantumCircuit qc(3);
    qc.h(0).x(1).y(2);
    auto doc = build_default(qc);
    ASSERT_EQ(doc.layers.size(), 1u);
    ASSERT_EQ(doc.layers[0].glyphs.size(), 3u);
    EXPECT_EQ(doc.layers[0].glyphs[0].data_gate, "h");
    EXPECT_EQ(doc.layers[0].glyphs[1].data_gate, "x");
    EXPECT_EQ(doc.layers[0].glyphs[2].data_gate, "y");
}

// =============================================================================
// build_document : multi-qubit gate packing
// =============================================================================

TEST(DocumentLayoutTest, CxBetweenAdjacentQubitsHasStrut) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    auto doc = build_default(qc);
    ASSERT_EQ(doc.layers.size(), 1u);
    const Glyph& g = doc.layers[0].glyphs[0];
    EXPECT_TRUE(g.has_strut);
    EXPECT_EQ(g.strut_top, 0);
    EXPECT_EQ(g.strut_bot, 1);
}

TEST(DocumentLayoutTest, CxBetweenDistantQubitsBlocksMiddleRows) {
    // CX(0, 3) on a 4-qubit circuit. Spec section 5.2: only the explicitly
    // touched qubits (0 and 3) are reserved; intermediate qubits (1, 2)
    // remain free for other gates to pack into the same column.
    QuantumCircuit qc(4);
    qc.cx(0, 3).x(1).x(2);
    auto doc = build_default(qc);
    ASSERT_EQ(doc.layers.size(), 1u);
    // Three glyphs share column 0: the CX (rows 0+3) plus the two X gates.
    EXPECT_EQ(doc.layers[0].glyphs.size(), 3u);
    const Glyph* cx = find_glyph(doc, "cx");
    ASSERT_NE(cx, nullptr);
    EXPECT_EQ(cx->strut_top, 0);
    EXPECT_EQ(cx->strut_bot, 3);
}

TEST(DocumentLayoutTest, CcxStrutSpansMinToMaxQubit) {
    QuantumCircuit qc(3);
    qc.ccx(0, 1, 2);
    auto doc = build_default(qc);
    ASSERT_EQ(doc.layers.size(), 1u);
    const Glyph& g = doc.layers[0].glyphs[0];
    EXPECT_TRUE(g.has_strut);
    EXPECT_EQ(g.strut_top, 0);
    EXPECT_EQ(g.strut_bot, 2);
}

TEST(DocumentLayoutTest, SerialisedCxGatesProduceTwoLayers) {
    QuantumCircuit qc(3);
    qc.cx(0, 1).cx(1, 2);
    auto doc = build_default(qc);
    // Both CX gates touch qubit 1: they must serialise.
    EXPECT_EQ(doc.layers.size(), 2u);
}

// =============================================================================
// build_document : barriers
// =============================================================================

TEST(DocumentLayoutTest, BarrierForcesFullWidthColumnBreak) {
    QuantumCircuit qc(3);
    qc.h(0).h(1).barrier({}).x(2);
    auto doc = build_default(qc);
    // Layer 0: H(0), H(1). Layer 1: BARRIER (full-width). Layer 2: X(2)
    // because the barrier reserved column 1 for every row, pushing X(2)
    // to column 2.
    ASSERT_GE(doc.layers.size(), 3u);
    EXPECT_EQ(doc.layers[0].column, 0);
    EXPECT_EQ(doc.layers[1].column, 1);
    EXPECT_EQ(doc.layers[2].column, 2);
}

TEST(DocumentLayoutTest, EmptyBarrierStillBlocksAllRows) {
    QuantumCircuit qc(2);
    qc.h(0).barrier({}).x(1);
    auto doc = build_default(qc);
    // Without the barrier, H(0) and X(1) would share column 0. With it
    // they straddle the barrier and end up in columns 0 and 2.
    ASSERT_EQ(doc.layers.size(), 3u);
    EXPECT_EQ(doc.layers[0].glyphs.size(), 1u);
    EXPECT_EQ(doc.layers[2].glyphs.size(), 1u);
}

TEST(DocumentLayoutTest, BarrierWithExplicitQubitsStillForcesFullBreak) {
    // Per spec section 5.2, BARRIER always forces a full-width break,
    // regardless of whether qubits are listed explicitly. A subsequent gate
    // on a qubit NOT in the barrier list must still pack to the right of it.
    QuantumCircuit qc(3);
    qc.h(0).barrier({0, 1}).x(2);
    auto doc = build_default(qc);
    ASSERT_GE(doc.layers.size(), 3u);
    const Glyph* x = find_glyph(doc, "x");
    ASSERT_NE(x, nullptr);
    EXPECT_GE(x->column, 2);
}

// =============================================================================
// build_document : non-contiguous UNITARY
// =============================================================================

TEST(DocumentLayoutTest, NonContiguousUnitaryReservesIntermediateRows) {
    QuantumCircuit qc(4);
    std::vector<Complex128> mat(16, {0.0, 0.0});
    for (int i = 0; i < 4; ++i) { mat[i * 4 + i] = {1.0, 0.0}; }
    qc.unitary(mat, {0, 3}, "U").x(1).x(2);
    auto doc = build_default(qc);
    // UNITARY on {0, 3} reserves rows 0..3, so X(1) and X(2) must serialise
    // after it. Locate the UNITARY by its data_gate: Instruction::gate_name()
    // returns the gate's `label` field when non-empty (so "U" here), and
    // falls back to "unitary" only when the label is absent. The fixture
    // passes "U" so the lookup uses that.
    const Glyph* u = find_glyph(doc, "U");
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->column, 0);
    const Glyph* x1 = find_glyph(doc, "x");
    ASSERT_NE(x1, nullptr);
    EXPECT_GE(x1->column, 1);
}

TEST(DocumentLayoutTest, ContiguousUnitaryDoesNotInflate) {
    QuantumCircuit qc(3);
    std::vector<Complex128> mat(4, {0.0, 0.0});
    mat[0] = {1.0, 0.0};
    mat[3] = {1.0, 0.0};
    qc.unitary(mat, {0}, "U").x(1);
    auto doc = build_default(qc);
    // UNITARY on {0} reserves row 0 only; X(1) can pack into column 0.
    ASSERT_EQ(doc.layers.size(), 1u);
    EXPECT_EQ(doc.layers[0].glyphs.size(), 2u);
}

// =============================================================================
// build_document : conditional gates
// =============================================================================

TEST(DocumentLayoutTest, ConditionalGateWithClbitsOffDoesNotSerialise) {
    // With show_clbits = false, conditional gates that touch different
    // qubits should still share a column (conditional info is decorative).
    QuantumCircuit qc(2, 2);
    qc.p_if(1.0, 0, 0, 1).p_if(1.0, 1, 1, 1);
    DrawOptions opts; // show_clbits defaults to false
    auto doc = build_document(qc, opts);
    ASSERT_EQ(doc.layers.size(), 1u);
    EXPECT_EQ(doc.layers[0].glyphs.size(), 2u);
}

TEST(DocumentLayoutTest, ConditionalGateWithClbitsOnSerialisesOnCWire) {
    // With show_clbits = true, every conditional gate touches the shared
    // c-wire row, so successive conditionals serialise.
    QuantumCircuit qc(2, 2);
    qc.p_if(1.0, 0, 0, 1).p_if(1.0, 1, 1, 1);
    DrawOptions opts;
    opts.show_clbits = true;
    auto doc = build_document(qc, opts);
    EXPECT_EQ(doc.layers.size(), 2u);
}

TEST(DocumentLayoutTest, ConditionalGateRecordsClbitAndValue) {
    QuantumCircuit qc(2, 1);
    qc.p_if(3.14, 1, 0, 1);
    auto doc = build_default(qc);
    ASSERT_EQ(doc.layers.size(), 1u);
    const Glyph& g = doc.layers[0].glyphs[0];
    EXPECT_EQ(g.condition_clbit, 0);
    EXPECT_EQ(g.condition_value, 1);
}

TEST(DocumentLayoutTest, NonConditionalGateLeavesConditionFieldsAtSentinels) {
    QuantumCircuit qc(1);
    qc.h(0);
    auto doc = build_default(qc);
    const Glyph& g = doc.layers[0].glyphs[0];
    EXPECT_EQ(g.condition_clbit, -1);
    EXPECT_EQ(g.condition_value, 0);
}

// =============================================================================
// build_document : c-wire strut extension for MEASURE and conditionals
// =============================================================================

TEST(DocumentLayoutTest, MeasureWithClbitsOnExtendsStrutToCWire) {
    QuantumCircuit qc(2, 2);
    qc.measure(0, 0);
    DrawOptions opts;
    opts.show_clbits = true;
    auto doc = build_document(qc, opts);
    ASSERT_EQ(doc.layers.size(), 1u);
    const Glyph& g = doc.layers[0].glyphs[0];
    EXPECT_TRUE(g.has_strut);
    EXPECT_EQ(g.strut_bot, qc.n_qubits); // virtual c-wire row index
}

TEST(DocumentLayoutTest, MeasureWithClbitsOffOmitsCWireStrut) {
    QuantumCircuit qc(2, 2);
    qc.measure(0, 0);
    auto doc = build_default(qc); // show_clbits defaults to false
    ASSERT_EQ(doc.layers.size(), 1u);
    const Glyph& g = doc.layers[0].glyphs[0];
    EXPECT_FALSE(g.has_strut);
}

TEST(DocumentLayoutTest, ConditionalGateWithClbitsOnExtendsStrut) {
    QuantumCircuit qc(2, 1);
    qc.p_if(1.0, 1, 0, 1);
    DrawOptions opts;
    opts.show_clbits = true;
    auto doc = build_document(qc, opts);
    const Glyph& g = doc.layers[0].glyphs[0];
    EXPECT_TRUE(g.has_strut);
    EXPECT_EQ(g.strut_bot, qc.n_qubits);
}

// =============================================================================
// build_document : qubit and clbit labels
// =============================================================================

TEST(DocumentLayoutTest, QubitLabelsFollowQiskitConvention) {
    QuantumCircuit qc(3);
    auto doc = build_default(qc);
    ASSERT_EQ(doc.qubit_labels.size(), 3u);
    EXPECT_EQ(doc.qubit_labels[0], "q[0]");
    EXPECT_EQ(doc.qubit_labels[1], "q[1]");
    EXPECT_EQ(doc.qubit_labels[2], "q[2]");
}

TEST(DocumentLayoutTest, ClbitLabelsEmptyWhenShowClbitsFalse) {
    QuantumCircuit qc(2, 2);
    auto doc = build_default(qc);
    EXPECT_TRUE(doc.clbit_labels.empty());
}

TEST(DocumentLayoutTest, ClbitLabelsPopulatedWhenShowClbitsTrue) {
    QuantumCircuit qc(2, 3);
    DrawOptions opts;
    opts.show_clbits = true;
    auto doc = build_document(qc, opts);
    ASSERT_EQ(doc.clbit_labels.size(), 3u);
    EXPECT_EQ(doc.clbit_labels[0], "c[0]");
    EXPECT_EQ(doc.clbit_labels[1], "c[1]");
    EXPECT_EQ(doc.clbit_labels[2], "c[2]");
}

TEST(DocumentLayoutTest, TenQubitLabelsRenderInBracketForm) {
    QuantumCircuit qc(10);
    auto doc = build_default(qc);
    ASSERT_EQ(doc.qubit_labels.size(), 10u);
    EXPECT_EQ(doc.qubit_labels[9], "q[9]");
}

// =============================================================================
// build_document : data_gate stamping and options capture
// =============================================================================

TEST(DocumentLayoutTest, DataGateMatchesInstructionGateName) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rx(1.5708, 0);
    auto doc = build_default(qc);
    EXPECT_NE(find_glyph(doc, "h"),  nullptr);
    EXPECT_NE(find_glyph(doc, "cx"), nullptr);
    EXPECT_NE(find_glyph(doc, "rx"), nullptr);
}

TEST(DocumentLayoutTest, OptionsCapturedOnDocument) {
    QuantumCircuit qc(1);
    DrawOptions opts;
    opts.show_clbits  = true;
    opts.fold_width   = 100;
    opts.ascii_safe   = true;
    opts.param_format = ParamFormat::Raw;
    auto doc = build_document(qc, opts);
    EXPECT_EQ(doc.options.show_clbits,  true);
    EXPECT_EQ(doc.options.fold_width,   100);
    EXPECT_EQ(doc.options.ascii_safe,   true);
    EXPECT_EQ(doc.options.param_format, ParamFormat::Raw);
}

// =============================================================================
// build_document : measure / reset / barrier glyph parts
// =============================================================================

TEST(DocumentLayoutTest, MeasureProducesMeasurePartOnTargetQubit) {
    QuantumCircuit qc(2, 2);
    qc.measure(1, 0);
    auto doc = build_default(qc);
    EXPECT_EQ(count_glyphs_with_part<MeasurePart>(doc), 1);
    const Glyph& g = doc.layers[0].glyphs[0];
    EXPECT_EQ(g.parts.front().first, 1);
}

TEST(DocumentLayoutTest, ResetProducesResetPart) {
    QuantumCircuit qc(1);
    qc.reset(0);
    auto doc = build_default(qc);
    EXPECT_EQ(count_glyphs_with_part<ResetPart>(doc), 1);
}

TEST(DocumentLayoutTest, BarrierProducesBarrierPartPerQubit) {
    QuantumCircuit qc(3);
    qc.barrier({0, 1});
    auto doc = build_default(qc);
    int parts = 0;
    for (const auto& layer : doc.layers) {
        for (const auto& g : layer.glyphs) {
            for (const auto& kv : g.parts) {
                if (std::holds_alternative<BarrierPart>(kv.second)) { ++parts; }
            }
        }
    }
    EXPECT_EQ(parts, 2);
}

// =============================================================================
// build_document : layer count == circuit depth in simple cases
// =============================================================================

TEST(DocumentLayoutTest, LayerCountMatchesDepthForSerialChain) {
    QuantumCircuit qc(1);
    qc.h(0).x(0).y(0).z(0);
    auto doc = build_default(qc);
    EXPECT_EQ(doc.layers.size(), 4u);
}

TEST(DocumentLayoutTest, LayerCountReflectsPackingNotInstructionCount) {
    QuantumCircuit qc(3);
    qc.h(0).h(1).h(2); // three disjoint single-qubit gates
    auto doc = build_default(qc);
    EXPECT_EQ(doc.layers.size(), 1u);
}
