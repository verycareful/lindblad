// trivial_layout.cpp — TrivialLayout pass
//
// TrivialLayout: Identity mapping — logical qubit i → physical qubit i.
// No reordering is performed. When a coupling map is present, the DAG is
// expanded to n_physical_qubits by identity embedding (see
// layout_expansion.hpp), so the output honours the same invariant as
// SabreLayout:
//
//     out.n_qubits == coupling_map.n_physical_qubits
//
// This makes circuits smaller than the device routable: every physical slot
// holds a (possibly idle) logical wire, so SABRE SWAP candidates through
// otherwise-empty slots exist and idle-wire SWAPs are legal (Qiskit emits
// the same). Circuits larger than the device throw std::invalid_argument.
// With no coupling map (n_physical == 0), the circuit passes through
// unchanged. SabreLayout lives in sabre_layout.cpp.

#include "lindblad/transpiler.hpp"

#include "layout_expansion.hpp"

namespace lindblad {

// =============================================================================
// TrivialLayout — identity qubit mapping, expanded to the device width
// =============================================================================

DAGCircuit TrivialLayout::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    return transpiler_detail::expand_to_physical(dag, ctx, "TrivialLayout");
}

} // namespace lindblad
