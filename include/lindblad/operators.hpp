// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/types.hpp"

#include <string>
#include <vector>

namespace lindblad {

class Statevector;
class DensityMatrix;
class QuantumCircuit;

// =============================================================================
// PauliString — tensor product of single-qubit Paulis
// =============================================================================
// Qubit ordering convention (project-wide, LSB-first; see
// docs/Architecture.md "Conventions"):
//   pauli[q] acts on qubit q. pauli[0] is qubit 0 (the LEAST significant
//   qubit), pauli[N-1] is qubit N-1.
//
//   Example: PauliString("XIZ") on a 3-qubit system means:
//     X on qubit 0, I on qubit 1, Z on qubit 2.
//
//   NOTE: this is the opposite character order from Qiskit's labels, and it
//   reads in the opposite direction from measurement bitstrings (whose
//   rightmost character is qubit 0). "XI" (X on qubit 0) marks the state
//   counted under the key "01".
//
//   tensor(): this->pauli occupies the LOW qubits of the result, other the
//   HIGH qubits (labels concatenate left to right as qubit index grows).

struct PauliString {
    std::string pauli;    // e.g., "XYZII" — index q acts on qubit q (LSB-first)
    Complex128 coeff;

    PauliString() : coeff(1.0, 0.0) {}
    // Throws std::invalid_argument for any character other than I, X, Y or Z,
    // uppercase (lowercase included), as Qiskit's labels do.
    PauliString(const std::string& p, Complex128 c = Complex128(1.0, 0.0));

    int n_qubits() const { return static_cast<int>(pauli.size()); }
    PauliString compose(const PauliString& other) const;
    PauliString adjoint() const;
    bool commutes_with(const PauliString& other) const;
};

// =============================================================================
// SparsePauliOp — sum of Pauli strings
// =============================================================================
// Width rule. A Pauli string names one Pauli per qubit and carries no qubit
// labels, so its length is the register it acts on: every term of one operator
// has the same length, and an operator evaluated against a state has exactly
// the state's qubit count. Construction (the vector constructor, from_list,
// operator+, simplify) refuses terms of different lengths, and every
// evaluation (expectation_value, expectation_value_batch, to_matrix,
// DensityMatrix::expectation_value_sparse, Estimator, ExpectationObserver)
// refuses a term whose length is not the state's. Both refusals are
// std::invalid_argument. `terms` is a public vector, so the evaluation check
// stands on its own rather than trusting construction.
//
// A term is written with I, X, Y and Z, uppercase; any other character is
// refused on construction and again at evaluation, as a width is.
//
// Every evaluation returns a real number, so it refuses an operator that is
// not Hermitian: one whose coefficients, once repeated labels are merged, are
// not all real to within DEFAULT_PHYSICAL_ATOL. to_matrix() returns the
// matrix itself and accepts any coefficients.
//
// An operator with no terms has no width, and every evaluation refuses it.
// The zero operator on n qubits is zero(n), one all-identity term with
// coefficient 0; simplify() returns that when every term cancels, so H - H
// evaluates to 0. A default-constructed operator is still the way to build
// one term by term, and QAOA and MA-QAOA read a mixer with no terms as "use
// the default mixer".

class SparsePauliOp {
public:
    std::vector<PauliString> terms;

    SparsePauliOp() = default;
    // Throws std::invalid_argument when the terms differ in length.
    explicit SparsePauliOp(const std::vector<PauliString>& terms);

    // Merges terms with the same label and drops those whose coefficient has
    // magnitude at most atol. When nothing survives, the result is the zero
    // operator at the input's width rather than an operator with no terms.
    SparsePauliOp simplify(double atol = 1e-8) const;
    SparsePauliOp compose(const SparsePauliOp& other) const;
    SparsePauliOp adjoint() const;
    SparsePauliOp tensor(const SparsePauliOp& other) const;
    SparsePauliOp operator+(const SparsePauliOp& other) const;
    SparsePauliOp operator*(double scalar) const;

    std::vector<Complex128> to_matrix() const;
    double expectation_value(const Statevector& sv) const;

    // Compute ⟨H⟩ for a batch of statevectors simultaneously.
    // Amortises the Pauli mask precomputation across all states.
    // Evaluations are parallelised across states with OpenMP.
    std::vector<double> expectation_value_batch(
        const std::vector<const Statevector*>& states
    ) const;

    // The width of the first term, and 0 for an operator with no terms (which
    // has no width; see the width rule above).
    int n_qubits() const;
    size_t size() const { return terms.size(); }

    static SparsePauliOp from_list(
        const std::vector<std::pair<std::string, Complex128>>& label_coeff
    );
    static SparsePauliOp identity(int n_qubits);
    static SparsePauliOp zero(int n_qubits);
};

// =============================================================================
// Operator — general matrix operator
// =============================================================================

class Operator {
public:
    std::vector<Complex128> data;
    int n_qubits;

    Operator() : n_qubits(0) {}
    Operator(std::vector<Complex128> d, int nq) : data(std::move(d)), n_qubits(nq) {}

    static Operator from_circuit(const QuantumCircuit& circuit);
    static Operator from_pauli(const SparsePauliOp& op);

    Operator compose(const Operator& other) const;
    Operator tensor(const Operator& other) const;
    Operator adjoint() const;
    Operator power(int n) const;

    // The tolerance matches ValidationOptions::atol, so asking an Operator
    // whether it is unitary gives the same verdict the primitives give when
    // they are handed the same matrix. Two defaults an order of magnitude
    // apart would let a matrix pass is_unitary() and then be rejected by the
    // kernel it was checked for.
    bool is_unitary(double atol = 1e-12) const;
    bool is_hermitian(double atol = 1e-12) const;

    Complex128 trace() const;
    size_t dim() const { return 1ULL << n_qubits; }
};

// =============================================================================
// Quantum information metrics
// =============================================================================

namespace QuantumInfo {
    double state_fidelity(const Statevector& sv1, const Statevector& sv2);
    double state_fidelity(const DensityMatrix& rho1, const DensityMatrix& rho2);

    // Squared process fidelity (Hilbert-Schmidt inner product):
    //   F_proc(U, V) = |Tr(U† V)|² / d²
    // where d is the Hilbert space dimension.
    // This is the SQUARED quantity. For the unsquared version, take sqrt().
    // Related: average_gate_fidelity = (d * F_proc + 1) / (d + 1)  [Nielsen 2002]
    double process_fidelity(const Operator& channel1, const Operator& channel2);

    // Average gate fidelity: F_avg = (d * F_proc + 1) / (d + 1)
    // where F_proc is the squared process fidelity above. See: Nielsen (2002),
    // "A simple formula for the average gate fidelity of a quantum dynamical operation".
    double average_gate_fidelity(const Operator& channel, const Operator& target);
    double entropy(const DensityMatrix& rho, double base = 2.0);
    double entanglement_entropy(const Statevector& sv, const std::vector<int>& subsystem);
    double concurrence(const DensityMatrix& rho);
    DensityMatrix partial_trace(const DensityMatrix& rho, const std::vector<int>& qargs);
    DensityMatrix partial_trace(const Statevector& sv, const std::vector<int>& qargs);

    std::vector<double> pauli_expectation_values(
        const Statevector& sv,
        const std::vector<std::string>& paulis
    );
}

} // namespace lindblad
