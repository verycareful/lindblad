// layout_expansion.hpp — shared identity-embedding expansion for layout passes
//
// INTERNAL header (src/, not include/): implementation detail of the layout
// passes, not public API.
//
// Both layout passes deliver the same output invariant:
//
//     dag.n_qubits == coupling_map.n_physical_qubits
//
// i.e. every physical slot of the device holds a (possibly idle) logical
// wire. TrivialLayout achieves it by identity embedding alone; SabreLayout
// expands its INPUT with this helper before running the SABRE search.
//
// Why this exists (R.1.15.0, frozen-slot defect): SABRE SWAP candidates
// require both physical slots of an edge to be occupied. When
// n_circuit < n_physical the occupied set was frozen at the initial layout
// image, while the scoring heuristic used full-graph BFS distances whose
// shortest paths run through empty slots that could never be entered. On
// degree-sparse maps (heavy-hex) routing thrashed until the SWAP-budget
// guard threw a misleading "disconnected components" error. With every slot
// occupied, idle-wire SWAPs are legal (Qiskit emits the same) and the
// frozen-set situation cannot arise.
//
// The identity embedding is exact: existing wires keep their indices, new
// wires n_circuit..n_physical-1 are idle (no instructions), and
// DAGCircuit::from_circuit sizes the IN-node array for all physical wires.

#pragma once

#include "lindblad/transpiler.hpp"

#include <stdexcept>
#include <string>

namespace lindblad {
namespace transpiler_detail {

// Expand `dag` to coupling_map.n_physical_qubits wires by identity embedding.
//
// dag       = input circuit DAG (logical qubit indices)
// ctx       = transpilation context; only the coupling map is read
// pass_name = calling pass, used in error messages
//
// Returns dag unchanged when the map is unconstrained (n_physical == 0,
// i.e. default CouplingMap()) or already matches. Throws std::invalid_argument
// when the circuit cannot fit the device, which would otherwise index the
// distance matrix out of range in the SABRE passes.
inline DAGCircuit expand_to_physical(
    const DAGCircuit& dag,
    const TranspilationContext& ctx,
    const char* pass_name
) {
    const int n_physical = ctx.coupling_map.n_physical_qubits;
    if (n_physical == 0) return dag;  // unconstrained routing: nothing to embed into

    if (dag.n_qubits > n_physical) {
        throw std::invalid_argument(
            std::string("lindblad::") + pass_name + ": circuit has " +
            std::to_string(dag.n_qubits) + " qubits but the coupling map has only " +
            std::to_string(n_physical) + " physical qubits");
    }
    if (dag.n_qubits == n_physical) return dag;

    QuantumCircuit qc = dag.to_circuit();
    qc.n_qubits = n_physical;
    return DAGCircuit::from_circuit(qc);
}

} // namespace transpiler_detail
} // namespace lindblad
