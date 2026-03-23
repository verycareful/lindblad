// remove_diagonal.cpp — Remove diagonal gates before measurements
//                       Remove redundant resets on known-zero qubits
//
// RemoveDiagonalGatesBeforeMeasure:
//   Diagonal gates (RZ, P, T, S, Z, U1, TDG, SDG) only affect phase.
//   If the only successor on a wire is MEASURE, the gate has no observable
//   effect and can be removed. Pure performance win — no semantic change.
//
// RemoveResetInZeroState:
//   Track which qubits are known to be in |0⟩ (at circuit start, or after
//   a RESET). Remove RESET instructions on qubits already in |0⟩.

#include "qpp/transpiler.hpp"

#include <unordered_set>
#include <vector>

namespace qpp {

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

    // Track which qubits are known to be |0⟩
    // All qubits start in |0⟩ state
    std::unordered_set<int> known_zero;
    for (int q = 0; q < qc.n_qubits; ++q) {
        known_zero.insert(q);
    }

    for (const auto& inst : qc.instructions) {
        if (inst.type == Instruction::GateType::RESET) {
            int q = inst.qubits[0];
            if (known_zero.count(q)) {
                // Already |0⟩, skip the reset
                continue;
            }
            // After reset, qubit is |0⟩
            known_zero.insert(q);
            optimized.instructions.push_back(inst);
        } else {
            // Any gate on a qubit makes it no longer known-zero
            // (except BARRIER which is a no-op)
            if (inst.type != Instruction::GateType::BARRIER) {
                for (int q : inst.qubits) {
                    known_zero.erase(q);
                }
            }
            optimized.instructions.push_back(inst);
        }
    }

    return DAGCircuit::from_circuit(optimized);
}

} // namespace qpp
