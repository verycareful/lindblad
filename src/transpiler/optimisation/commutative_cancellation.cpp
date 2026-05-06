// commutative_cancellation.cpp — Commutative gate cancellation
//
// For each wire, identify pairs of self-inverse or parameterised rotation gates
// that can cancel through commuting intermediate gates. Uses the Pauli
// commutation rules: diagonal gates commute with each other, X-type gates
// commute with each other, etc.
//
// Example: RZ(a) - CX(other) - RZ(-a) → the two RZ gates cancel because
// RZ commutes through the control wire of CX (both are Z-diagonal).

#include "lindblad/transpiler.hpp"

#include <unordered_set>
#include <vector>

namespace lindblad {

// A gate is Z-diagonal if it only applies phase shifts (no amplitude mixing).
// These commute with each other and commute through the control of CX/CZ.
static bool is_z_diagonal(const Instruction& inst) {
    using GT = Instruction::GateType;
    switch (inst.type) {
        case GT::Z: case GT::S: case GT::SDG: case GT::T: case GT::TDG:
        case GT::RZ: case GT::P: case GT::U1:
            return true;
        default:
            return false;
    }
}

// Check if a gate is a rotation that can cancel with its inverse
static bool is_cancellable_rotation(const Instruction& inst) {
    using GT = Instruction::GateType;
    switch (inst.type) {
        case GT::RX: case GT::RY: case GT::RZ: case GT::P: case GT::U1:
            return true;
        default:
            return false;
    }
}

// Check if two rotation gates cancel (same type, opposite parameters)
static bool rotations_cancel(const Instruction& a, const Instruction& b, double atol = 1e-10) {
    if (a.type != b.type) return false;
    if (a.qubits != b.qubits) return false;
    if (a.params.size() != b.params.size()) return false;
    for (size_t i = 0; i < a.params.size(); ++i) {
        if (std::abs(a.params[i] + b.params[i]) > atol) return false;
    }
    return true;
}

// Check if two rotation gates of the same type can be merged
static bool rotations_merge(const Instruction& a, const Instruction& b) {
    return a.type == b.type && a.qubits == b.qubits &&
           a.params.size() == 1 && b.params.size() == 1;
}

// Check if inst commutes through gate on the same wire.
// Conservative: only recognize Z-diagonal through Z-diagonal,
// and self-inverse gates through themselves.
static bool commutes_on_wire(const Instruction& moving, const Instruction& barrier_gate, int wire) {
    // Z-diagonal gates commute with each other on any shared wire
    if (is_z_diagonal(moving) && is_z_diagonal(barrier_gate)) return true;

    // A Z-diagonal gate commutes through the CONTROL wire of CX/CZ/CP/CRZ
    // (the control only checks |1⟩, and Z-diagonal preserves computational basis)
    using GT = Instruction::GateType;
    if (is_z_diagonal(moving)) {
        if ((barrier_gate.type == GT::CX || barrier_gate.type == GT::CZ ||
             barrier_gate.type == GT::CP || barrier_gate.type == GT::CRZ) &&
            barrier_gate.qubits.size() >= 2 && barrier_gate.qubits[0] == wire) {
            return true;  // moving gate is on the control wire
        }
    }

    // CX commutes with CX on the same qubits (they're self-inverse)
    if (moving.type == GT::CX && barrier_gate.type == GT::CX &&
        moving.qubits == barrier_gate.qubits) {
        return true;
    }

    return false;
}

DAGCircuit CommutativeCancellation::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    QuantumCircuit qc = dag.to_circuit();
    std::vector<Instruction> insts = qc.instructions;
    int n = static_cast<int>(insts.size());
    std::vector<bool> removed(n, false);

    // Fixed-point: repeat until a full pass makes no changes.
    // A single forward pass misses cancellations exposed by earlier merges.
    bool changed = true;
    while (changed) {
        changed = false;

        for (int i = 0; i < n; ++i) {
            if (removed[i]) continue;
            const auto& inst_i = insts[i];

            if (!is_cancellable_rotation(inst_i) && !is_z_diagonal(inst_i)) continue;

            int wire = inst_i.qubits[0];

            for (int j = i + 1; j < n; ++j) {
                if (removed[j]) continue;
                const auto& inst_j = insts[j];

                bool touches_wire = false;
                for (int q : inst_j.qubits) {
                    if (q == wire) { touches_wire = true; break; }
                }

                if (!touches_wire) continue;

                if (rotations_cancel(inst_i, inst_j)) {
                    removed[i] = true;
                    removed[j] = true;
                    changed = true;
                    break;
                }

                if (rotations_merge(inst_i, inst_j)) {
                    double merged = inst_i.params[0] + inst_j.params[0];
                    if (std::abs(merged) < 1e-10) {
                        removed[i] = true;
                        removed[j] = true;
                    } else {
                        insts[i].params[0] = merged;
                        removed[j] = true;
                    }
                    changed = true;
                    break;
                }

                if (commutes_on_wire(inst_i, inst_j, wire)) {
                    continue;
                }

                break;
            }
        }
    }

    QuantumCircuit optimized(qc.n_qubits, qc.n_clbits);
    for (int i = 0; i < n; ++i) {
        if (!removed[i]) {
            optimized.instructions.push_back(insts[i]);
        }
    }

    return DAGCircuit::from_circuit(optimized);
}

} // namespace lindblad
