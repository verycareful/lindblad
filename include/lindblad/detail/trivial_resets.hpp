// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

// =============================================================================
// trivial_resets - which RESETs cannot change the state
// =============================================================================
// A RESET collapses its qubit, so a simulator that samples every shot from one
// forward pass would collapse it once for all of them; a circuit holding one
// has to run shot by shot. A RESET on a qubit known to be |0> is the exception:
// the collapse has one outcome and the state is unchanged, so it cannot make
// shots differ. `reset q;` at the top of a QASM file is the common case.
//
// A qubit is known to be |0> when the initial state puts it there and nothing
// has acted on it since, or when an unconditioned RESET has just put it there.
// A conditioned RESET leaves a qubit known |0> only if it already was, since it
// may not run. Any other instruction on the qubit, a measurement included,
// ends the knowledge; a barrier does not.

#include "lindblad/circuit.hpp"
#include "lindblad/observation.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lindblad {
namespace detail {

// Which qubits hold |0> before the first instruction: all of them under the
// default initial state; under a basis state, those whose bit of the index is
// 0 (qubits past bit 63 among them, as the seeding leaves them at |0>); none
// under a supplied state, whose qubits are not known.
inline std::vector<bool> zero_at_start(const InitialState& initial, int n_qubits) {
    std::vector<bool> zero(static_cast<std::size_t>(n_qubits), false);
    if (initial.is_default()) {
        zero.assign(static_cast<std::size_t>(n_qubits), true);
    } else if (initial.is_basis()) {
        const std::uint64_t index = initial.basis_index();
        for (int q = 0; q < n_qubits; ++q)
            zero[static_cast<std::size_t>(q)] = q >= 64 || ((index >> q) & 1u) == 0;
    }
    return zero;
}

// Per instruction: true for a RESET whose qubit is known to be |0> when it
// runs. `known_zero` is zero_at_start's answer, consumed as the walk goes.
inline std::vector<bool> trivial_resets(const QuantumCircuit& circuit,
                                        std::vector<bool> known_zero) {
    using GT = Instruction::GateType;
    std::vector<bool> trivial(circuit.instructions.size(), false);
    for (std::size_t i = 0; i < circuit.instructions.size(); ++i) {
        const Instruction& inst = circuit.instructions[i];
        if (inst.type == GT::BARRIER) continue;
        if (inst.type == GT::RESET) {
            const auto q = static_cast<std::size_t>(inst.qubits[0]);
            trivial[i] = known_zero[q];
            if (inst.condition_clbit < 0) known_zero[q] = true;
            continue;
        }
        for (int q : inst.qubits) known_zero[static_cast<std::size_t>(q)] = false;
    }
    return trivial;
}

// Whether the circuit holds a RESET that can change the state, and so has to
// run shot by shot on a backend that collapses it.
inline bool has_nontrivial_reset(const QuantumCircuit& circuit,
                                 const InitialState& initial) {
    const std::vector<bool> trivial =
        trivial_resets(circuit, zero_at_start(initial, circuit.n_qubits));
    for (std::size_t i = 0; i < circuit.instructions.size(); ++i)
        if (circuit.instructions[i].type == Instruction::GateType::RESET && !trivial[i])
            return true;
    return false;
}

}  // namespace detail
}  // namespace lindblad
