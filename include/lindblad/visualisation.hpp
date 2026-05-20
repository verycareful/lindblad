#pragma once

// =============================================================================
// lindblad/visualisation.hpp : public re-exports of circuit visualiser options
// =============================================================================
// Convenience header for callers that need DrawMode, ParamFormat, or
// DrawOptions without pulling in the full QuantumCircuit interface. The
// canonical definitions live in lindblad/circuit.hpp; this header simply
// re-exports them via using-declarations so user code can write:
//
//   #include "lindblad/visualisation.hpp"
//   lindblad::DrawOptions opts;
//   opts.fold_width = 80;
//
// All four renderer backends (ASCII, SVG, LATEX, HTML) are reached through
// QuantumCircuit::draw(); this header carries no additional API surface.

#include "lindblad/circuit.hpp"

namespace lindblad {

// Re-exported option types. Defined in lindblad/circuit.hpp; aliased here so
// that downstream code can include only this lighter header. The aliases are
// no-ops in this same namespace (DrawMode etc. are already visible), but they
// document the intent and keep the public-API surface explicit.
using ::lindblad::DrawMode;
using ::lindblad::ParamFormat;
using ::lindblad::DrawOptions;

} // namespace lindblad
