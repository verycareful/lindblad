#include "qpp/transpiler.hpp"

namespace qpp {

// =============================================================================
// Optimize1qGates — merge adjacent single-qubit gates on the same qubit
// =============================================================================

DAGCircuit Optimize1qGates::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    // Convert to circuit, merge adjacent 1q gates, convert back
    QuantumCircuit qc = dag.to_circuit();
    QuantumCircuit optimized(qc.n_qubits, qc.n_clbits);

    // Track last single-qubit gate per qubit for merging
    // Simple version: cancel consecutive identical self-inverse gates (X, H, etc.)
    std::vector<int> last_gate_idx(qc.n_qubits, -1);

    for (size_t i = 0; i < qc.instructions.size(); ++i) {
        const auto& inst = qc.instructions[i];

        if (inst.qubits.size() == 1) {
            int q = inst.qubits[0];
            // Check if previous instruction on this qubit is the same self-inverse gate
            if (last_gate_idx[q] >= 0) {
                auto& prev = optimized.instructions[last_gate_idx[q]];
                bool cancel = false;

                using GT = Instruction::GateType;
                // Self-inverse gates: X, Y, Z, H, CX
                if (prev.type == inst.type) {
                    switch (inst.type) {
                        case GT::X: case GT::Y: case GT::Z: case GT::H:
                            cancel = true;
                            break;
                        default:
                            break;
                    }
                }
                // S + SDG = I
                if ((prev.type == GT::S && inst.type == GT::SDG) ||
                    (prev.type == GT::SDG && inst.type == GT::S)) {
                    cancel = true;
                }
                // T + TDG = I
                if ((prev.type == GT::T && inst.type == GT::TDG) ||
                    (prev.type == GT::TDG && inst.type == GT::T)) {
                    cancel = true;
                }

                if (cancel) {
                    // Remove previous gate
                    optimized.instructions.erase(
                        optimized.instructions.begin() + last_gate_idx[q]);
                    last_gate_idx[q] = -1;
                    continue;
                }
            }

            last_gate_idx[q] = static_cast<int>(optimized.instructions.size());
        } else {
            // Multi-qubit gate: reset tracking for all involved qubits
            for (int q : inst.qubits) {
                last_gate_idx[q] = -1;
            }
        }

        optimized.instructions.push_back(inst);
    }

    return DAGCircuit::from_circuit(optimized);
}

// =============================================================================
// CXCancellation — cancel adjacent CX gates: CX.CX = I
// =============================================================================

DAGCircuit CXCancellation::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    QuantumCircuit qc = dag.to_circuit();
    QuantumCircuit optimized(qc.n_qubits, qc.n_clbits);

    for (size_t i = 0; i < qc.instructions.size(); ++i) {
        const auto& inst = qc.instructions[i];

        if (inst.type == Instruction::GateType::CX && !optimized.instructions.empty()) {
            const auto& prev = optimized.instructions.back();
            if (prev.type == Instruction::GateType::CX &&
                prev.qubits[0] == inst.qubits[0] &&
                prev.qubits[1] == inst.qubits[1]) {
                // Cancel CX.CX
                optimized.instructions.pop_back();
                continue;
            }
        }

        optimized.instructions.push_back(inst);
    }

    return DAGCircuit::from_circuit(optimized);
}

// =============================================================================
// ConsolidateBlocks — merge blocks of 2-qubit gates into single unitary
// =============================================================================

DAGCircuit ConsolidateBlocks::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    // For now, pass through
    // Full implementation would identify 2-qubit blocks, compute their
    // combined unitary, and KAK-decompose to minimal CNOT count
    return dag;
}

// =============================================================================
// PassManager
// =============================================================================

void PassManager::append(std::unique_ptr<TranspilationPass> pass) {
    passes.push_back(std::move(pass));
}

DAGCircuit PassManager::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    DAGCircuit current = dag;
    for (const auto& pass : passes) {
        current = pass->run(current, ctx);
    }
    return current;
}

// =============================================================================
// Preset pass managers
// =============================================================================

PassManager preset_pass_manager(
    int optimization_level,
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates
) {
    PassManager pm;

    if (optimization_level >= 0) {
        pm.append(std::make_unique<TrivialLayout>());
    }

    if (optimization_level >= 1) {
        pm.append(std::make_unique<SabreSwap>());
        pm.append(std::make_unique<Optimize1qGates>());
        pm.append(std::make_unique<CXCancellation>());
    }

    if (optimization_level >= 2) {
        pm.append(std::make_unique<Optimize1qGates>());
        pm.append(std::make_unique<CXCancellation>());
    }

    if (optimization_level >= 3) {
        pm.append(std::make_unique<ConsolidateBlocks>());
        pm.append(std::make_unique<Optimize1qGates>());
        pm.append(std::make_unique<CXCancellation>());
    }

    return pm;
}

// =============================================================================
// Convenience transpile function
// =============================================================================

QuantumCircuit transpile(
    const QuantumCircuit& circuit,
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates,
    int optimization_level
) {
    auto dag = DAGCircuit::from_circuit(circuit);

    TranspilationContext ctx;
    ctx.coupling_map = coupling_map;
    ctx.basis_gates = basis_gates;
    ctx.optimization_level = optimization_level;

    auto pm = preset_pass_manager(optimization_level, coupling_map, basis_gates);
    auto result_dag = pm.run(dag, ctx);

    return result_dag.to_circuit();
}

} // namespace qpp
