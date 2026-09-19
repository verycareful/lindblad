// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// scheduling.cpp — ASAP and ALAP scheduling passes
//
// Assigns a time slot (cycle) to each instruction based on data dependencies.
// Uses the DAG structure to determine when each gate's operands are available.
//
// ASAP: each gate is assigned the earliest cycle when all its operand wires
//       are free. Forward topological pass.
// ALAP: each gate is assigned the latest cycle it can occupy without
//       increasing circuit depth. Reverse topological pass.
//
// The time slot is stored in Instruction::schedule_time (dedicated field).
// condition_clbit / condition_value are left untouched for classical conditioning.

#include "lindblad/transpiler.hpp"

#include <algorithm>
#include <vector>

namespace lindblad {

// =============================================================================
// ASAPSchedule — As Soon As Possible
// =============================================================================

std::vector<int> asap_schedule_times(const QuantumCircuit& qc) {
    std::vector<int> times(qc.instructions.size(), 0);

    // Track the next available cycle for each qubit wire
    std::vector<int> wire_available(static_cast<std::size_t>(qc.n_qubits), 0);

    for (std::size_t i = 0; i < qc.instructions.size(); ++i) {
        const Instruction& inst = qc.instructions[i];
        if (inst.type == Instruction::GateType::BARRIER) {
            // Barrier synchronizes all involved wires to the latest cycle
            int max_cycle = 0;
            for (int q : inst.qubits) {
                max_cycle = std::max(max_cycle, wire_available[static_cast<std::size_t>(q)]);
            }
            for (int q : inst.qubits) {
                wire_available[static_cast<std::size_t>(q)] = max_cycle;
            }
            times[i] = max_cycle;
            continue;
        }

        // Find the earliest cycle when all operand wires are free
        int start_cycle = 0;
        for (int q : inst.qubits) {
            start_cycle = std::max(start_cycle, wire_available[static_cast<std::size_t>(q)]);
        }
        times[i] = start_cycle;

        // Mark wires as occupied until the next cycle
        for (int q : inst.qubits) {
            wire_available[static_cast<std::size_t>(q)] = start_cycle + 1;
        }
    }
    return times;
}

DAGCircuit ASAPSchedule::run(
    const DAGCircuit& dag, const TranspilationContext& /*ctx*/
) const {
    QuantumCircuit qc = dag.to_circuit();
    const std::vector<int> times = asap_schedule_times(qc);
    for (std::size_t i = 0; i < qc.instructions.size(); ++i) {
        qc.instructions[i].schedule_time = times[i];
    }
    return DAGCircuit::from_circuit(qc);
}

// =============================================================================
// ALAPSchedule — As Late As Possible
// =============================================================================

DAGCircuit ALAPSchedule::run(
    const DAGCircuit& dag, const TranspilationContext& /*ctx*/
) const {
    QuantumCircuit qc = dag.to_circuit();

    // First compute circuit depth via ASAP to know the total depth
    std::vector<int> wire_available(qc.n_qubits, 0);
    int n = static_cast<int>(qc.instructions.size());

    // Forward pass: compute ASAP start times
    std::vector<int> asap_start(n, 0);
    for (int i = 0; i < n; ++i) {
        const auto& inst = qc.instructions[i];
        if (inst.type == Instruction::GateType::BARRIER) {
            int max_cycle = 0;
            for (int q : inst.qubits) max_cycle = std::max(max_cycle, wire_available[q]);
            for (int q : inst.qubits) wire_available[q] = max_cycle;
            asap_start[i] = max_cycle;
            continue;
        }
        int start = 0;
        for (int q : inst.qubits) start = std::max(start, wire_available[q]);
        asap_start[i] = start;
        for (int q : inst.qubits) wire_available[q] = start + 1;
    }

    int total_depth = *std::max_element(wire_available.begin(), wire_available.end());

    // Reverse pass: compute ALAP start times
    // Each wire's "deadline" starts at total_depth
    std::vector<int> wire_deadline(qc.n_qubits, total_depth);
    std::vector<int> alap_start(n, 0);

    for (int i = n - 1; i >= 0; --i) {
        const auto& inst = qc.instructions[i];
        if (inst.type == Instruction::GateType::BARRIER) {
            int min_deadline = total_depth;
            for (int q : inst.qubits) min_deadline = std::min(min_deadline, wire_deadline[q]);
            for (int q : inst.qubits) wire_deadline[q] = min_deadline;
            alap_start[i] = min_deadline;
            continue;
        }
        // Latest this gate can start: min deadline across its wires, minus 1 (for the gate itself)
        int latest = total_depth;
        for (int q : inst.qubits) {
            latest = std::min(latest, wire_deadline[q] - 1);
        }
        latest = std::max(latest, asap_start[i]);  // can't be earlier than ASAP
        alap_start[i] = latest;
        for (int q : inst.qubits) {
            wire_deadline[q] = latest;
        }
    }

    // Build output with ALAP annotations
    QuantumCircuit scheduled(qc.n_qubits, qc.n_clbits);
    scheduled.name = qc.name;
    for (int i = 0; i < n; ++i) {
        Instruction inst = qc.instructions[i];
        inst.schedule_time = alap_start[i];
        scheduled.instructions.push_back(std::move(inst));
    }

    return DAGCircuit::from_circuit(scheduled);
}

} // namespace lindblad
