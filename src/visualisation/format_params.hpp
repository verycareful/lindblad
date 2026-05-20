#pragma once

// =============================================================================
// src/visualisation/format_params.hpp : parameter and label formatting helpers
// =============================================================================
// Three small helpers shared by every gate builder:
//
//   format_param(double, ParamFormat)        : numeric -> string
//   format_param(const ParamExpr&, ParamFormat) : symbolic expression -> string
//   format_gate_label(inst, sym, opts)        : assemble the full box label
//
// Numeric formatting in Pretty mode snaps recognised rational multiples of
// pi (0, +/- pi/8, pi/6, pi/4, pi/3, pi/2, 2pi/3, 3pi/4, 5pi/6, pi, 3pi/2,
// 2pi, 3pi, 4pi) within tolerance 1e-6, then falls back to "%.4f". Raw mode
// always uses "%.4f". Symbolic names pass through unchanged; binary-op
// expressions render with their operator (e.g. "theta+pi/4", "2*theta").
//
// Label assembly follows: if GateSymbol::show_params AND DrawOptions::
// show_params AND inst has any parameters (numeric or symbolic), append a
// parenthesised comma-separated list to GateSymbol::label. Otherwise return
// the bare label.
//
// Definitions live in format_params.cpp (added in a later task).

#include "gate_symbols.hpp"
#include "lindblad/circuit.hpp"

#include <string>

namespace lindblad::viz {

// Format a single numeric parameter value.
// v   = the value to render
// fmt = Pretty (pi-snap then %.4f) or Raw (always %.4f)
std::string format_param(double v, ParamFormat fmt);

// Format a single symbolic parameter expression.
// e   = the expression tree (literal, name, or binary op)
// fmt = applied to embedded literals; names pass through unchanged
std::string format_param(const ParamExpr& e, ParamFormat fmt);

// Assemble the final label for a Tier 1 box gate.
// inst = the instruction whose params drive the suffix
// sym  = the Tier 1 catalogue entry (provides base label and show_params)
// opts = caller options (provides global show_params toggle and ParamFormat)
std::string format_gate_label(const Instruction& inst,
                              const GateSymbol& sym,
                              const DrawOptions& opts);

} // namespace lindblad::viz
