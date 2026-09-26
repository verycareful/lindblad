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
// pauli_rules - what makes a Pauli string, and an operator built from them, a
//               valid observable
// =============================================================================
//
// Alphabet. A Pauli string is written with I, X, Y and Z, uppercase, and
// nothing else. The kernels test for exactly those characters, so any other
// one, a lowercase letter included, would read as identity on some paths and
// as its Pauli on others. It is refused wherever a string is accepted.
//
// Width. A Pauli string names one Pauli per qubit and carries no qubit labels,
// so its length IS the register it acts on. Every term of one operator
// therefore has one length, and an operator evaluated against a state has the
// state's length exactly. The expectation kernels build their bit masks from
// the string: a term wider than the state sets mask bits past the last qubit,
// which indexes past the amplitude arrays for an X or Y and reads a missing
// qubit as |0> for a Z; a shorter term is identity on qubits it never names,
// which is a guess about what the caller meant rather than something the
// string says.
//
// Emptiness. An operator with no terms has no width. The zero operator on n
// qubits is SparsePauliOp::zero(n), a single all-identity term with
// coefficient 0, and simplify() returns exactly that when every term cancels,
// so an operator with no terms reaching an evaluation is a caller mistake, not
// a value.
//
// Hermiticity. Every Pauli string is Hermitian, so a sum of them is Hermitian
// exactly when every coefficient is real once repeated labels are merged. The
// expectation of a non-Hermitian operator is complex, and every evaluation
// here returns a double, so returning the real part would report a different
// number under the caller's name. Imaginary parts within
// DEFAULT_PHYSICAL_ATOL count as real, as they do for MA-QAOA's mixer.
//
// Every refusal is std::invalid_argument and names `where`.

#include <string>
#include <vector>

namespace lindblad {

struct PauliString;
class SparsePauliOp;

namespace detail {

// Throws unless every character of `label` is I, X, Y or Z.
void check_pauli_label(const std::string& label, const char* where);

// The width every term shares, or -1 for no terms. Checks every label's
// alphabet, and throws when two terms differ in width, naming the first that
// disagrees with term 0.
int uniform_pauli_width(const std::vector<PauliString>& terms, const char* where);

// The width every term shares. Throws when there are no terms, and otherwise
// as uniform_pauli_width.
int required_pauli_width(const std::vector<PauliString>& terms, const char* where);

// Throws unless `op` is Hermitian: every coefficient real to within
// DEFAULT_PHYSICAL_ATOL after merging repeated labels. When every coefficient
// is already real term by term, nothing is merged or allocated.
void check_hermitian(const SparsePauliOp& op, const char* where);

// Everything an evaluation that returns a real expectation value needs: at
// least one term, a valid alphabet, every term exactly `n_qubits` wide, and a
// Hermitian operator. `against` names what supplies the width in the message
// ("state", "circuit").
void check_observable(const SparsePauliOp& op, int n_qubits, const char* where,
                      const char* against = "state");

}  // namespace detail
}  // namespace lindblad
