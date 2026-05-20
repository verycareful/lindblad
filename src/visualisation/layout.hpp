#pragma once

// =============================================================================
// src/visualisation/layout.hpp : ASAP packing and glyph dispatch
// =============================================================================
// Declares the entry points for the visualiser's layout pass. The free
// function build_document() in document.hpp drives this file; the helpers
// below construct individual Glyph values for the dispatcher to attach to a
// column.
//
// Dispatch order in build_glyph():
//   1. Tier 3 special cases (BARRIER / MEASURE / RESET / UNITARY) by switch
//   2. Tier 2 composite catalogue lookup (CX, CCZ, SWAP, RXX, ...)
//   3. Tier 1 box-gate catalogue fallback (H, RX, U, ...)
//
// build_box_glyph and build_composite_glyph are the catalogue-driven helpers
// invoked by build_glyph. They are exposed here so that future renderers or
// tests can construct glyphs without going through the full dispatcher.
//
// Definitions live in layout.cpp (added in a later task).

#include "composite_catalogue.hpp"
#include "document.hpp"
#include "gate_symbols.hpp"
#include "lindblad/circuit.hpp"

namespace lindblad::viz {

// Top-level dispatch: routes an Instruction to the correct builder.
// inst = the instruction to render
// opts = caller options threaded through to the chosen builder
Glyph build_glyph(const Instruction& inst, const DrawOptions& opts);

// Tier 1 path: build a single-wire labelled box glyph.
// inst = the instruction (provides qubit row and optional params)
// sym  = the matching Tier 1 catalogue entry
// opts = caller options (drives param suffix via format_gate_label)
Glyph build_box_glyph(const Instruction& inst,
                      const GateSymbol& sym,
                      const DrawOptions& opts);

// Tier 2 path: walk a composite rule and emit one part per slot.
// inst = the instruction (provides the qubit list indexed by slot)
// rule = the matching Tier 2 catalogue entry
// opts = caller options (drives strut emission and box-label assembly)
Glyph build_composite_glyph(const Instruction& inst,
                            const CompositeGate& rule,
                            const DrawOptions& opts);

} // namespace lindblad::viz
