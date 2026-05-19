// trivial_layout.cpp — TrivialLayout pass
//
// TrivialLayout: Identity mapping — logical qubit i → physical qubit i.
// No reordering is performed; the circuit is returned unchanged.
// SabreLayout lives in sabre_layout.cpp.

#include "lindblad/transpiler.hpp"

namespace lindblad {

// =============================================================================
// TrivialLayout — identity qubit mapping
// =============================================================================

DAGCircuit TrivialLayout::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    return dag;
}

} // namespace lindblad
