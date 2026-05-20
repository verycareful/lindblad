// =============================================================================
// src/visualisation/gate_builders.cpp : Tier 3 hand-written gate builders
// =============================================================================
// Four small builders for the gates whose visual structure does not fit the
// Tier 1 or Tier 2 catalogues:
//
//   BARRIER : one BarrierPart per touched qubit. Empty qubit list means the
//             full register: the builder enumerates 0..n_qubits-1. Since the
//             builder does not know n_qubits, it returns an empty parts list
//             in that case and lets the layout pass expand to a full-width
//             barrier (which it already does via the full_width_break path).
//   MEASURE : MeasurePart on inst.qubits[0] carrying the destination clbit.
//             The c-wire strut is set up by the layout pass (which knows the
//             c-wire row index); this builder only records the part itself.
//   RESET   : ResetPart on inst.qubits[0]. No strut.
//   UNITARY : a single tall BoxPart with rowspan = max-min+1 anchored at the
//             topmost qubit row. Label sources from inst.label, falling back
//             to "U" when the caller did not provide one.

#include "gate_builders.hpp"

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <algorithm>
#include <string>

namespace lindblad::viz {

// =============================================================================
// build_barrier_glyph : one BarrierPart per touched qubit
// =============================================================================
// Empty inst.qubits is the "implicit all qubits" form: layout.cpp already
// promotes that to a full-width column break, so we leave the parts list empty
// in that case. The renderer's full-width barrier pass paints every wire row
// when it encounters a Glyph at a barrier column with no parts.

Glyph build_barrier_glyph(const Instruction& inst, const DrawOptions& /*opts*/) {
    Glyph g;
    g.parts.reserve(inst.qubits.size());
    for (int q : inst.qubits) {
        g.parts.emplace_back(q, BarrierPart{});
    }
    g.has_strut = false;
    return g;
}

// =============================================================================
// build_measure_glyph : MeasurePart on the measured qubit
// =============================================================================
// inst.qubits[0] is the measured qubit; inst.clbits[0] is the destination
// classical bit (or -1 if missing, which the parser should never produce but
// we tolerate defensively). The c-wire strut is the layout pass's
// responsibility because the builder cannot know the document's c-wire row
// index.

Glyph build_measure_glyph(const Instruction& inst, const DrawOptions& /*opts*/) {
    Glyph g;
    if (inst.qubits.empty()) {
        return g;
    }
    MeasurePart part;
    part.clbit = inst.clbits.empty() ? -1 : inst.clbits[0];
    g.parts.emplace_back(inst.qubits[0], part);
    g.has_strut = false;
    return g;
}

// =============================================================================
// build_reset_glyph : ResetPart on the target qubit
// =============================================================================
// No strut. The renderer paints the |0> reset marker (or "|0>" in ASCII-safe
// mode) at the resolved row.

Glyph build_reset_glyph(const Instruction& inst, const DrawOptions& /*opts*/) {
    Glyph g;
    if (inst.qubits.empty()) {
        return g;
    }
    g.parts.emplace_back(inst.qubits[0], ResetPart{});
    g.has_strut = false;
    return g;
}

// =============================================================================
// build_unitary_glyph : tall labelled box spanning min..max qubit rows
// =============================================================================
// Anchor at min(qubits) with rowspan = max(qubits) - min(qubits) + 1. Label
// from inst.label, fallback "U". Non-contiguous qubit sets (e.g. {0, 3})
// produce a box spanning every intermediate row even though those qubits do
// not participate in the unitary; the layout pass reserves those rows too so
// no later gate slips under the box.

Glyph build_unitary_glyph(const Instruction& inst, const DrawOptions& /*opts*/) {
    Glyph g;
    if (inst.qubits.empty()) {
        return g;
    }
    int qmin = *std::min_element(inst.qubits.begin(), inst.qubits.end());
    int qmax = *std::max_element(inst.qubits.begin(), inst.qubits.end());

    BoxPart box;
    box.label   = inst.label.empty() ? std::string("U") : inst.label;
    box.rowspan = qmax - qmin + 1;
    // svg_fill and svg_stroke intentionally left empty: the SVG renderer
    // falls back to a neutral palette for unitary boxes (custom unitaries do
    // not carry GateSymbol entries).
    g.parts.emplace_back(qmin, std::move(box));
    g.has_strut = false;
    return g;
}

} // namespace lindblad::viz
