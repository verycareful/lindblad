// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// remove_diagonal.cpp — Remove diagonal gates before measurements
//                       Remove redundant resets on known-zero qubits
//
// RemoveDiagonalGatesBeforeMeasure:
//   Diagonal gates (RZ, P, T, S, Z, U1, TDG, SDG) only affect phase.
//   If the only successor on a wire is MEASURE, the gate has no observable
//   effect and can be removed. Pure performance win — no semantic change.
//
// RemoveResetInZeroState:
//   Remove each RESET on a qubit known to be in |0⟩, the rule in
//   detail/trivial_resets.hpp that the simulators apply at run time: a qubit
//   is known |0⟩ from the start until something acts on it, and after an
//   unconditioned RESET; a conditioned RESET leaves it known only if it
//   already was, since it may not run. The pass sees the circuit alone and
//   takes every qubit to start at |0⟩, so a circuit to be run from a supplied
//   initial state must not be given it; no preset level includes it.

#include "lindblad/transpiler.hpp"
#include "lindblad/detail/trivial_resets.hpp"

#include <vector>

namespace lindblad {

// =============================================================================
// RemoveDiagonalGatesBeforeMeasure
// =============================================================================

static bool is_diagonal_gate(const Instruction& inst) {
    using GT = Instruction::GateType;
    switch (inst.type) {
        case GT::Z: case GT::S: case GT::SDG: case GT::T: case GT::TDG:
        case GT::RZ: case GT::P: case GT::U1:
            return true;
        default:
            return false;
    }
}

DAGCircuit RemoveDiagonalGatesBeforeMeasure::run(
    const DAGCircuit& dag, const TranspilationContext& /*ctx*/
) const {
    QuantumCircuit qc = dag.to_circuit();
    int n = static_cast<int>(qc.instructions.size());
    std::vector<bool> removed(n, false);

    // For each diagonal gate, check if the next operation on that wire is MEASURE.
    // Scan backwards: mark the "last real gate before measure" per wire.
    // Actually easier to scan forward: for each diagonal gate on qubit q,
    // look ahead to see if the next gate on q is MEASURE (or another diagonal
    // which itself will be checked).

    // Build per-qubit instruction index lists
    std::vector<std::vector<int>> wire_ops(qc.n_qubits);
    for (int i = 0; i < n; ++i) {
        for (int q : qc.instructions[i].qubits) {
            wire_ops[q].push_back(i);
        }
    }

    // For each wire, walk backwards from the end removing diagonal gates
    // that appear just before MEASURE (possibly with other diagonal gates between)
    for (int q = 0; q < qc.n_qubits; ++q) {
        auto& ops = wire_ops[q];
        if (ops.empty()) continue;

        // Walk from end
        int k = static_cast<int>(ops.size()) - 1;

        // Find the last operation on this wire
        while (k >= 0 && removed[ops[k]]) --k;
        if (k < 0) continue;

        // If the last op is not MEASURE, skip this wire
        if (qc.instructions[ops[k]].type != Instruction::GateType::MEASURE) continue;

        // Now walk backwards removing diagonal gates
        --k;
        while (k >= 0) {
            int idx = ops[k];
            if (removed[idx]) { --k; continue; }

            const auto& inst = qc.instructions[idx];
            // Only remove single-qubit diagonal gates on this wire
            if (is_diagonal_gate(inst) && inst.qubits.size() == 1 && inst.qubits[0] == q) {
                removed[idx] = true;
                --k;
            } else {
                break;  // non-diagonal gate encountered, stop
            }
        }
    }

    QuantumCircuit optimized(qc.n_qubits, qc.n_clbits);
    for (int i = 0; i < n; ++i) {
        if (!removed[i]) {
            optimized.instructions.push_back(qc.instructions[i]);
        }
    }
    return DAGCircuit::from_circuit(optimized);
}

// =============================================================================
// RemoveResetInZeroState
// =============================================================================

DAGCircuit RemoveResetInZeroState::run(
    const DAGCircuit& dag, const TranspilationContext& /*ctx*/
) const {
    QuantumCircuit qc = dag.to_circuit();
    QuantumCircuit optimized(qc.n_qubits, qc.n_clbits);

    const std::vector<bool> trivial = detail::trivial_resets(
        qc, std::vector<bool>(static_cast<std::size_t>(qc.n_qubits), true));
    for (std::size_t i = 0; i < qc.instructions.size(); ++i)
        if (!trivial[i]) optimized.instructions.push_back(qc.instructions[i]);

    return DAGCircuit::from_circuit(optimized);
}

} // namespace lindblad
