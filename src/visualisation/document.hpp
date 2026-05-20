#pragma once

// =============================================================================
// src/visualisation/document.hpp : circuit visualiser data model
// =============================================================================
// Pure-data types shared by the layout pass and every backend renderer. The
// layout pass produces a CircuitDocument; each renderer consumes one. There
// is exactly one document model: no per-backend forking of layout state.
//
// GlyphPart is a *closed* variant of seven kinds. Adding a new visual
// primitive (e.g. a hypothetical "phase wheel") requires editing this header
// AND each renderer's std::visit. Considered acceptable because primitives
// evolve far slower than gates: gates flow through the Tier 1/2/3 catalogues
// without touching this header.
//
// Namespace lindblad::viz is internal: these types are not part of the public
// API. The only public surface is QuantumCircuit::draw() in circuit.hpp.

#include "lindblad/circuit.hpp"

#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lindblad::viz {

// =============================================================================
// GlyphPart variant : the seven visual primitives a renderer must handle
// =============================================================================
// Each part is placed on a single wire row at one column. Renderers pattern-
// match via std::visit. The 7-kind closed set covers every gate visual the
// catalogues can produce; multi-qubit gates compose multiple parts inside a
// single Glyph.

// BoxPart : a labelled rectangle drawn around a wire (single or multi-qubit).
// rowspan = 1 is a single-wire box. rowspan > 1 marks a tall box that covers
// `rowspan` consecutive wire rows starting at the part's host row (used by
// UNITARY and the TallBox composite role for RXX / RYY / RZZ / RZX / ECR).
struct BoxPart {
    std::string label;       // "H", "RX(pi/2)", "U(theta,phi,lambda)"
    std::string svg_fill;    // ignored by ASCII / LaTeX backends
    std::string svg_stroke;  // ignored by ASCII / LaTeX backends
    int         rowspan = 1; // multi-qubit boxes span consecutive rows
};

// CtrlBulletPart : a control marker. anti = false renders as a filled disc;
// anti = true renders as an open ring (anti-control). CZ / CCZ use this kind
// on every involved qubit because the phase action is symmetric: there is no
// dedicated "dot target" primitive.
struct CtrlBulletPart {
    bool anti = false;
};

// XorTargetPart : the standard CNOT target marker (circle with cross).
struct XorTargetPart {};

// SwapXPart : a single arm of a SWAP gate (drawn as an x).
struct SwapXPart {};

// MeasurePart : a measurement box on a qubit wire. clbit = -1 means no
// classical drop line should be drawn (used when show_clbits is false). When
// non-negative, the renderer drops a strut to the named classical-bit row.
struct MeasurePart {
    int clbit = -1;
};

// ResetPart : the |0> reset marker.
struct ResetPart {};

// BarrierPart : the dashed-vertical barrier marker (one per touched qubit).
struct BarrierPart {};

using GlyphPart = std::variant<BoxPart,
                               CtrlBulletPart,
                               XorTargetPart,
                               SwapXPart,
                               MeasurePart,
                               ResetPart,
                               BarrierPart>;

// =============================================================================
// Glyph : one logical gate occupying one column
// =============================================================================
// A glyph owns every visual primitive a single gate contributes. Multi-qubit
// gates carry multiple (qubit_row, part) pairs in `parts`. The strut fields
// describe the optional vertical connector that links the placed parts
// (e.g. the line between a CX control and its target). condition_* preserve
// classical conditioning so renderers can draw the `if c[k]=v` tag.

struct Glyph {
    int column = -1;                                                 // 0-indexed; assigned by layout
    std::vector<std::pair<int /*qubit_row*/, GlyphPart>> parts;      // one entry per visual primitive
    bool has_strut = false;                                          // true => draw the vertical connector
    int strut_top = -1;                                              // min qubit row spanned by the strut
    int strut_bot = -1;                                              // max qubit row spanned by the strut
    int condition_clbit = -1;                                        // -1 = unconditional gate
    int condition_value = 0;                                         // value compared against condition_clbit
    std::string data_gate;                                           // SVG/HTML data-gate attribute; set by build_glyph()
                                                                     // to Instruction::gate_name()
};

// =============================================================================
// Layer : every glyph sharing the same column
// =============================================================================
// Layout groups glyphs into layers after ASAP packing. Renderers walk layers
// left-to-right and within each layer paint every glyph independently (parts
// never collide within a layer because the packer reserves every touched
// row).

struct Layer {
    int                column = -1;
    std::vector<Glyph> glyphs;
};

// =============================================================================
// CircuitDocument : the complete backend-agnostic render input
// =============================================================================
// Captures the qubit / clbit counts, ordered layers, row labels, and the
// DrawOptions used to build it. The options are kept so renderers can read
// formatting knobs (cell sizes, ascii_safe, include_legend) without needing a
// separate handle to the caller's struct.

struct CircuitDocument {
    int                       n_qubits = 0;
    int                       n_clbits = 0;
    std::vector<Layer>        layers;
    std::vector<std::string>  qubit_labels;   // raw "q[0]", "q[1]", ... (Qiskit convention)
    std::vector<std::string>  clbit_labels;   // raw "c[0]", ... populated only when show_clbits
    DrawOptions               options;        // captured at build time
};

// Build a CircuitDocument from a QuantumCircuit using ASAP packing.
// qc   = the circuit to render
// opts = caller's DrawOptions (folding, c-wire visibility, etc.)
// Defined in src/visualisation/layout.cpp.
CircuitDocument build_document(const QuantumCircuit& qc, const DrawOptions& opts);

} // namespace lindblad::viz
