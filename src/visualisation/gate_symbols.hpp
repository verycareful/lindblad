#pragma once

// =============================================================================
// src/visualisation/gate_symbols.hpp : Tier 1 single-qubit box gate catalogue
// =============================================================================
// Declares the GateSymbol struct and the symbol_catalogue() accessor. The
// catalogue covers single-qubit "box" gates: H X Y Z S SDG T TDG SX SXDG
// RX RY RZ P U U1 U2 U3 plus the symbolic PARAM_* variants. Two-qubit
// interaction gates (RXX / RYY / RZZ / RZX / ECR) live in the composite
// catalogue because they render as a single tall box spanning two wires.
//
// The catalogue is the seam a maintainer edits to add or restyle a box gate;
// layout, struts, and every renderer remain untouched. Definitions live in
// gate_symbols.cpp (added in a later task).

#include "lindblad/circuit.hpp"

#include <string>
#include <unordered_map>

namespace lindblad::viz {

// GateSymbol : per-gate appearance metadata.
// label        = text drawn inside the box
// show_params  = whether to append "(p1, p2, ...)" to the label when the
//                instruction has numeric parameters and the caller has not
//                disabled show_params in DrawOptions
// svg_fill     = SVG / HTML rect fill colour
// svg_stroke   = SVG / HTML rect border colour
// latex_macro  = explicit Quantikz token (e.g. "\\gate{R_X}"); empty string
//                falls back to the default "\\gate{label}" assembly
struct GateSymbol {
    std::string label;
    bool        show_params = false;
    std::string svg_fill;
    std::string svg_stroke;
    std::string latex_macro;
};

// Returns a reference to the static catalogue. The map is built once on
// first call and never mutated thereafter. Lookup is O(1) average.
const std::unordered_map<Instruction::GateType, GateSymbol>& symbol_catalogue();

} // namespace lindblad::viz
