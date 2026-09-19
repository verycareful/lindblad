// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

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
