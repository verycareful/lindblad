// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/circuit.hpp"

#include <functional>
#include <string>

namespace lindblad {
namespace detail {

// =============================================================================
// preflight_instructions - what the circuit alone decides, before the first gate
// =============================================================================
// One pass over a circuit before any state is touched, refusing everything the
// instructions decide on their own, so none of it can fail a run partway:
//
//   - an unbound parameterised gate (a PARAM_* type, or symbolic expressions
//     bind_parameters() has not resolved)
//   - a gate naming the wrong number of qubits for its type, or one qubit
//     twice, or carrying fewer parameters than its type reads
//   - a non-finite gate parameter
//   - a UNITARY whose matrix is not (2^k)^2 entries for its k operands
//   - a PERMUTATION whose map is not a bijection of [0, 2^k)
//   - whatever `backend_check` refuses, given its reason (empty = accepted)
//
// Each refusal is an InvalidArgument whose message starts with `entry_point`
// and names the instruction: its index, gate and qubits. Operand indices are
// the caller's validate_operands(), which runs first.
void preflight_instructions(
    const QuantumCircuit& circuit, const char* entry_point,
    const std::function<std::string(const Instruction&, int n_qubits)>& backend_check = {});

}  // namespace detail
}  // namespace lindblad
