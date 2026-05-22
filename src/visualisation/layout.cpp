// =============================================================================
// src/visualisation/layout.cpp : ASAP packing and glyph dispatch
// =============================================================================
// Implementation of the visualiser's layout pass. Every QuantumCircuit funnels
// through build_document(): we walk the instruction list in order, place each
// gate in the leftmost column where every touched row is free (ASAP packing),
// and group the resulting glyphs into ordered Layers.
//
// Three Glyph builders feed the document:
//   - build_box_glyph        : Tier 1 single-qubit labelled box dispatcher
//   - build_composite_glyph  : Tier 2 controlled / multi-bullet dispatcher
//   - build_glyph            : top-level routing per spec section 5.3
//
// Several edge cases drive the packing rules (per spec section 5.2):
//   - BARRIER (with or without qubits) forces a full-width column break so
//     subsequent gates restart from a clean column. This holds even for the
//     empty-qubits case, which we treat as a full-width barrier.
//   - Non-contiguous UNITARY (e.g. qubits {0, 3}) reserves the full row range
//     [min..max]; the renderer paints a single tall box across all rows in
//     that range. Intermediate rows must be reserved to prevent another gate
//     from slipping under the box.
//   - Conditional gates with opts.show_clbits = true: the bundled c-wire row
//     (virtual row index n_qubits) participates in collision detection, so
//     later gates on the same condition_clbit serialise. When show_clbits is
//     false, the condition info is preserved on the Glyph but does not affect
//     packing.
//   - MEASURE / conditional gates with show_clbits = true also need a strut
//     down to the c-wire row. We extend this in build_document() after the
//     dispatcher returns because the dispatcher does not know the document's
//     n_qubits geometry.

#include "layout.hpp"

#include "composite_catalogue.hpp"
#include "document.hpp"
#include "format_params.hpp"
#include "gate_builders.hpp"
#include "gate_symbols.hpp"

#include "lindblad/circuit.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lindblad::viz {

// =============================================================================
// build_box_glyph : Tier 1 single-wire labelled box dispatcher
// =============================================================================
// Emits a single BoxPart at inst.qubits[0] with label and colours sourced from
// the supplied GateSymbol. No strut, no rowspan: this is the simplest gate
// visual. The label is assembled by format_gate_label so the caller's
// show_params / ParamFormat preferences are honoured.

Glyph build_box_glyph(const Instruction& inst,
                      const GateSymbol& sym,
                      const DrawOptions& opts) {
    Glyph g;
    BoxPart part;
    part.label       = format_gate_label(inst, sym, opts);
    part.svg_fill    = sym.svg_fill;
    part.svg_stroke  = sym.svg_stroke;
    part.rowspan     = 1;
    // R.1.10.2: thread the LaTeX gate-symbol override through to the
    // renderer so subscripted rotations and daggered gates render in math
    // mode. Empty when the catalogue accepts the default \gate{<label>}
    // emission.
    part.latex_macro = sym.latex_macro;
    // Tier 1 is single-qubit by construction; the catalogue never targets a
    // box gate with more than one qubit slot. Reading qubits[0] is always
    // safe for the gate types listed in symbol_catalogue().
    g.parts.emplace_back(inst.qubits[0], std::move(part));
    return g;
}

// =============================================================================
// build_composite_glyph : Tier 2 controlled / multi-bullet dispatcher
// =============================================================================
// Walks rule.parts and emits one GlyphPart per entry, placed at
// inst.qubits[part.qubit_slot]. The optional strut spans [min..max] of every
// placed row when rule.draw_strut is true.
//
// TallBox is special: when ANY slot carries Role::TallBox, the gate renders as
// a single labelled BoxPart at min(qubit_rows) with rowspan = #slots. No strut
// is drawn because the tall box itself spans the rows. This convention is
// used by symmetric two-qubit interaction gates (RXX RYY RZZ RZX ECR).

Glyph build_composite_glyph(const Instruction& inst,
                            const CompositeGate& rule,
                            const DrawOptions& opts) {
    Glyph g;

    // ------------------------------------------------------------------------
    // TallBox special case: detect ANY tall slot up front. If present, emit a
    // single BoxPart at the topmost qubit row with rowspan covering every slot
    // listed in the rule, and skip strut emission entirely. We resolve every
    // slot's row first so we can compute min(rows) and rowspan precisely.
    // ------------------------------------------------------------------------
    bool any_tall = false;
    for (const CompositePart& cp : rule.parts) {
        if (cp.role == CompositePart::Role::TallBox) {
            any_tall = true;
            break;
        }
    }

    if (any_tall) {
        std::vector<int> rows;
        rows.reserve(rule.parts.size());
        for (const CompositePart& cp : rule.parts) {
            rows.push_back(inst.qubits[cp.qubit_slot]);
        }
        // Defensive guard: empty rule.parts would be malformed catalogue data;
        // bail without producing a glyph rather than dereferencing nothing.
        if (rows.empty()) {
            return g;
        }
        int row_min = *std::min_element(rows.begin(), rows.end());
        int row_max = *std::max_element(rows.begin(), rows.end());

        BoxPart box;
        // Build the label: rule.box_label, optionally suffixed with the
        // instruction's parameters when the catalogue opts in and the caller
        // has not disabled show_params globally. We reuse the Tier 1
        // formatter by synthesising a temporary GateSymbol; this keeps the
        // pi-snap / ParamFormat logic in one place.
        GateSymbol pseudo;
        pseudo.label       = rule.box_label;
        pseudo.show_params = rule.box_show_params;
        box.label   = format_gate_label(inst, pseudo, opts);
        box.rowspan = row_max - row_min + 1;
        // SVG fill / stroke are intentionally left empty: the composite
        // catalogue does not currently expose interaction-gate colours, and
        // the SVG renderer falls back to its default styling when these are
        // blank.
        g.parts.emplace_back(row_min, std::move(box));
        g.has_strut = false;
        return g;
    }

    // ------------------------------------------------------------------------
    // Standard composite path: emit one part per slot, optionally followed by
    // a strut spanning the placed rows.
    // ------------------------------------------------------------------------
    std::vector<int> rows;
    rows.reserve(rule.parts.size());

    for (const CompositePart& cp : rule.parts) {
        int row = inst.qubits[cp.qubit_slot];
        rows.push_back(row);

        switch (cp.role) {
            case CompositePart::Role::CtrlBullet: {
                CtrlBulletPart p;
                p.anti = false;
                g.parts.emplace_back(row, p);
                break;
            }
            case CompositePart::Role::AntiCtrl: {
                CtrlBulletPart p;
                p.anti = true;
                g.parts.emplace_back(row, p);
                break;
            }
            case CompositePart::Role::XorTarget: {
                g.parts.emplace_back(row, XorTargetPart{});
                break;
            }
            case CompositePart::Role::SwapX: {
                g.parts.emplace_back(row, SwapXPart{});
                break;
            }
            case CompositePart::Role::Box: {
                BoxPart box;
                GateSymbol pseudo;
                pseudo.label       = rule.box_label;
                pseudo.show_params = rule.box_show_params;
                box.label   = format_gate_label(inst, pseudo, opts);
                box.rowspan = 1;
                g.parts.emplace_back(row, std::move(box));
                break;
            }
            case CompositePart::Role::TallBox: {
                // Unreachable: the any_tall branch above handles this role
                // exclusively. A TallBox mixed with other roles is a
                // catalogue authoring error; if it ever happens, we degrade
                // gracefully to a single-row Box at this slot rather than
                // crashing.
                BoxPart box;
                GateSymbol pseudo;
                pseudo.label       = rule.box_label;
                pseudo.show_params = rule.box_show_params;
                box.label   = format_gate_label(inst, pseudo, opts);
                box.rowspan = 1;
                g.parts.emplace_back(row, std::move(box));
                break;
            }
        }
    }

    if (rule.draw_strut && !rows.empty()) {
        g.has_strut = true;
        g.strut_top = *std::min_element(rows.begin(), rows.end());
        g.strut_bot = *std::max_element(rows.begin(), rows.end());
    }

    return g;
}

// =============================================================================
// build_glyph : top-level dispatcher per spec section 5.3
// =============================================================================
// Three-tier routing: hand-written builders for the irregular gates first,
// then the composite catalogue, then the box-gate catalogue as a fallback.
// data_gate is stamped on every returned glyph so SVG / HTML backends can
// emit a stable data-gate attribute.

Glyph build_glyph(const Instruction& inst, const DrawOptions& opts) {
    Glyph g;

    // Tier 3 special cases: hand-written builders for the four irregular
    // gates whose visuals do not fit the catalogue shapes.
    switch (inst.type) {
        case Instruction::GateType::BARRIER:
            g = build_barrier_glyph(inst, opts);
            g.data_gate = inst.gate_name();
            return g;
        case Instruction::GateType::MEASURE:
            g = build_measure_glyph(inst, opts);
            g.data_gate = inst.gate_name();
            return g;
        case Instruction::GateType::RESET:
            g = build_reset_glyph(inst, opts);
            g.data_gate = inst.gate_name();
            return g;
        case Instruction::GateType::UNITARY:
            g = build_unitary_glyph(inst, opts);
            g.data_gate = inst.gate_name();
            return g;
        default:
            break;
    }

    // Tier 2: composite catalogue lookup. The composite map is the seam for
    // controlled / multi-bullet gates (CX CY CZ CH SWAP ISWAP CRX ... ECR).
    const auto& comp = composite_catalogue();
    if (auto it = comp.find(inst.type); it != comp.end()) {
        g = build_composite_glyph(inst, it->second, opts);
        g.data_gate = inst.gate_name();
        return g;
    }

    // Tier 1 fallback: every remaining gate is expected to live in the
    // single-qubit box catalogue. A missing entry indicates a catalogue gap
    // rather than a user error; the .at() throw surfaces the gap loudly so
    // it cannot regress silently.
    g = build_box_glyph(inst, symbol_catalogue().at(inst.type), opts);
    g.data_gate = inst.gate_name();
    return g;
}

// =============================================================================
// build_document : ASAP packing driver
// =============================================================================
// Walks the circuit's instruction list in order and places each gate in the
// leftmost column where every touched row is free. The per-row "next free
// column" cursor is sized to n_qubits when show_clbits is false and
// n_qubits + 1 when show_clbits is true: the trailing virtual row models the
// bundled c-wire so MEASURE drops and conditional gates serialise on it.
//
// Layout proceeds in three phases:
//   1. Compute the touched-row set per instruction (per spec section 5.2).
//   2. Compute col = max(next_free_col[r] for r in touched) and advance
//      next_free_col[r] = col + 1 for every touched row.
//   3. Call build_glyph(), patch the column, and (for MEASURE / conditional
//      under show_clbits) extend the strut down to the c-wire row.
// Finally we group glyphs by column into ordered Layers.

CircuitDocument build_document(const QuantumCircuit& qc, const DrawOptions& opts) {
    CircuitDocument doc;
    doc.n_qubits = qc.n_qubits;
    doc.n_clbits = qc.n_clbits;
    doc.options  = opts;

    // Qubit labels follow Qiskit's "q[0]", "q[1]", ... convention. Always
    // emitted regardless of show_clbits because qubit wires are always drawn.
    doc.qubit_labels.reserve(qc.n_qubits);
    for (int q = 0; q < qc.n_qubits; ++q) {
        doc.qubit_labels.push_back("q[" + std::to_string(q) + "]");
    }
    // Classical-bit labels are only populated when the bundled c-wire is
    // visible. The vector stays empty otherwise so renderers can short-circuit
    // on emptiness alone.
    if (opts.show_clbits) {
        doc.clbit_labels.reserve(qc.n_clbits);
        for (int c = 0; c < qc.n_clbits; ++c) {
            doc.clbit_labels.push_back("c[" + std::to_string(c) + "]");
        }
    }

    // Cursor row count: n_qubits qubit rows plus one virtual c-wire row when
    // show_clbits is true. The virtual row is indexed n_qubits and represents
    // every classical bit collectively (the c-wire is bundled). Tracking a
    // single shared cursor for the bundle preserves serialisation order across
    // gates that touch any clbit.
    const int cursor_rows = qc.n_qubits + (opts.show_clbits ? 1 : 0);
    const int cwire_row   = qc.n_qubits; // valid only when show_clbits = true
    std::vector<int> next_free_col(cursor_rows, 0);

    std::vector<Glyph> glyphs;
    glyphs.reserve(qc.instructions.size());

    for (const Instruction& inst : qc.instructions) {
        // --------------------------------------------------------------------
        // Phase 1a: build the glyph up front so we can inspect its visual
        // footprint when computing the touched-row set. Any glyph that emits
        // a tall BoxPart (rowspan > 1) visually owns every wire row in the
        // span; the layout pass must reserve those rows even when the
        // underlying instruction lists only the endpoints. UNITARY and Tier 2
        // TallBox composites (RXX, RYY, RZZ, RZX, ECR) both hit this case;
        // RZZ(0, 2) was the first non-adjacent TallBox in the demo set and
        // exposed the original bug where RX(1) packed into the same column
        // as the tall RZZ box and overlapped on q1's wire row.
        // --------------------------------------------------------------------
        Glyph g = build_glyph(inst, opts);

        // --------------------------------------------------------------------
        // Phase 1b: compute the touched-row set per instruction type.
        // --------------------------------------------------------------------
        std::vector<int> touched;
        bool full_width_break = false;

        // A glyph "owns intermediate rows" when its visual footprint covers
        // wire rows the underlying instruction does not list explicitly.
        // Two cases trigger this:
        //   - a tall BoxPart (rowspan > 1) drawn THROUGH the intermediate
        //     wires (TallBox composites RXX/RYY/RZZ/RZX/ECR, plus UNITARY)
        //   - a vertical strut spanning at least one intermediate qubit row
        //     (non-adjacent CX/CP/CZ/SWAP and the controlled rotations
        //     when the listed qubits skip a wire; the strut crosses those
        //     intermediate wires and reads as ambiguous when a single-qubit
        //     gate also packs into the same column)
        // Both cases must reserve every row in [min(qubits)..max(qubits)]
        // so no later gate packs underneath the box or strut crossing.
        // Empirically this matches what the human eye expects from QFT and
        // QAOA renderings; the trade-off is wider diagrams in exchange for
        // unambiguous structure.
        bool glyph_owns_span = false;
        for (const auto& kv : g.parts) {
            if (auto* box = std::get_if<BoxPart>(&kv.second)) {
                if (box->rowspan > 1) { glyph_owns_span = true; break; }
            }
        }
        if (g.has_strut && (g.strut_bot - g.strut_top) >= 2) {
            // Strut span of 2 or more qubit indices crosses at least one
            // intermediate wire. Adjacent struts (CX(0,1) etc.) keep the
            // default two-row touched set.
            glyph_owns_span = true;
        }

        if (inst.type == Instruction::GateType::BARRIER) {
            // BARRIER (with or without an explicit qubit list) forces a full-
            // width column break: every row's cursor advances past `col` so
            // no subsequent gate can pack underneath it. The empty-qubits
            // case is the "implicit all qubits" barrier from the public API.
            full_width_break = true;
            touched.reserve(cursor_rows);
            for (int r = 0; r < cursor_rows; ++r) {
                touched.push_back(r);
            }
        } else if (glyph_owns_span) {
            // Tall BoxPart: reserve every wire row in [min..max] of the
            // listed qubits so intermediate rows are locked under the box.
            // Catches non-adjacent TallBox composites (RZZ(0, 2)), UNITARY,
            // and any future glyph type that produces a multi-row BoxPart.
            // Contiguous spans collapse to the same range trivially.
            if (!inst.qubits.empty()) {
                int qmin = *std::min_element(inst.qubits.begin(), inst.qubits.end());
                int qmax = *std::max_element(inst.qubits.begin(), inst.qubits.end());
                touched.reserve(qmax - qmin + 1);
                for (int r = qmin; r <= qmax; ++r) {
                    touched.push_back(r);
                }
            }
        } else {
            // Default: every named qubit is touched. Then, when show_clbits is
            // true, a conditional gate also touches the bundled c-wire row so
            // subsequent gates on the same clbit serialise behind it. Spec
            // section 5.2 leaves MEASURE's clbit row out of `touched` because
            // the c-wire strut is decorative; only the qubit row participates
            // in collision detection.
            touched.reserve(inst.qubits.size() + 1);
            for (int q : inst.qubits) {
                touched.push_back(q);
            }
            if (opts.show_clbits && inst.condition_clbit >= 0) {
                touched.push_back(cwire_row);
            }
        }

        // --------------------------------------------------------------------
        // Phase 2: choose the leftmost column where every touched row is free
        // and advance every touched cursor past it. For full-width breaks we
        // bump every cursor, not just the touched subset; this is redundant
        // with `touched = all rows` but kept explicit for clarity and for
        // symmetry with the spec pseudocode.
        // --------------------------------------------------------------------
        int col = 0;
        if (!touched.empty()) {
            for (int r : touched) {
                if (r >= 0 && r < cursor_rows && next_free_col[r] > col) {
                    col = next_free_col[r];
                }
            }
        }

        if (full_width_break) {
            for (int r = 0; r < cursor_rows; ++r) {
                next_free_col[r] = col + 1;
            }
        } else {
            for (int r : touched) {
                if (r >= 0 && r < cursor_rows) {
                    next_free_col[r] = col + 1;
                }
            }
        }

        // --------------------------------------------------------------------
        // Phase 3: patch document-level metadata onto the glyph already
        // built above (column index, conditional info, c-wire strut
        // extension). The glyph itself was constructed in Phase 1a so the
        // touched-row computation could inspect its visual footprint.
        // --------------------------------------------------------------------
        g.column = col;

        // Preserve conditional metadata on every glyph so renderers can emit
        // the `if c[k]=v` tag even when show_clbits = false (the tag is
        // inline next to the gate in that mode).
        g.condition_clbit = inst.condition_clbit;
        g.condition_value = inst.condition_value;

        // Extend the strut down to the c-wire row when the gate has a classical
        // hook AND the bundled c-wire is visible. Two triggers:
        //   - a MeasurePart inside the glyph (MEASURE)
        //   - condition_clbit >= 0 (conditional gate)
        // The strut top stays at the topmost row already involved (existing
        // strut top if any, else the topmost qubit row used by the glyph's
        // parts), the strut bottom drops to cwire_row.
        if (opts.show_clbits) {
            bool has_measure = false;
            for (const auto& [row, part] : g.parts) {
                (void)row;
                if (std::holds_alternative<MeasurePart>(part)) {
                    has_measure = true;
                    break;
                }
            }
            const bool has_condition = inst.condition_clbit >= 0;

            if (has_measure || has_condition) {
                // Pick a sensible top: prefer an existing strut top, else
                // collapse over the rows owned by this glyph. Without any
                // parts we fall back to the c-wire row itself (a zero-length
                // strut), which keeps the structure consistent without
                // painting an invalid line.
                int existing_top = g.has_strut ? g.strut_top : cwire_row;
                int parts_top    = cwire_row;
                bool have_parts_row = false;
                for (const auto& [row, part] : g.parts) {
                    (void)part;
                    if (!have_parts_row || row < parts_top) {
                        parts_top = row;
                        have_parts_row = true;
                    }
                }
                int new_top = existing_top;
                if (have_parts_row && parts_top < new_top) {
                    new_top = parts_top;
                }
                g.has_strut = true;
                g.strut_top = new_top;
                g.strut_bot = cwire_row;
            }
        }

        glyphs.push_back(std::move(g));
    }

    // ------------------------------------------------------------------------
    // Final phase: group glyphs by column into ordered Layers. We sort by
    // column index because instructions may not produce columns in strictly
    // monotonic order (a later instruction can pack into an earlier free
    // column). Within each layer we preserve insertion order so renderers
    // get deterministic ordering for ties.
    // ------------------------------------------------------------------------
    // Stable sort keeps insertion order within equal columns, which downstream
    // renderers rely on when painting overlapping parts (e.g. two single-qubit
    // gates in the same column).
    std::stable_sort(glyphs.begin(), glyphs.end(),
                     [](const Glyph& a, const Glyph& b) {
                         return a.column < b.column;
                     });

    for (Glyph& g : glyphs) {
        if (doc.layers.empty() || doc.layers.back().column != g.column) {
            Layer layer;
            layer.column = g.column;
            doc.layers.push_back(std::move(layer));
        }
        doc.layers.back().glyphs.push_back(std::move(g));
    }

    return doc;
}

} // namespace lindblad::viz
