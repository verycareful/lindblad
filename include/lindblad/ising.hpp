#pragma once

#include "lindblad/operators.hpp"

#include <string>
#include <vector>

namespace lindblad {

// =============================================================================
// IsingHamiltonian — H = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + offset
//
// Native representation of Ising cost Hamiltonians as they arise from QUBO
// problems. Eliminates the Python round-trip for problem setup.
//
// QUBO form: minimize x^T Q x, x ∈ {0,1}^n
// Ising form: minimize s^T J s + h^T s + const, s ∈ {-1,+1}^n
// Substitution: x_i = (1 - s_i) / 2
// =============================================================================

struct IsingHamiltonian {
    std::vector<double> h;              // linear terms h[i] for Z_i (size = n_qubits)
    std::vector<std::vector<double>> J; // quadratic terms J[i][j] for Z_i Z_j (upper triangular used)
    double offset = 0.0;               // constant energy shift

    int n_qubits() const { return static_cast<int>(h.size()); }

    // Convert to SparsePauliOp for use with Estimator / VQE / QAOA.
    // Pauli string convention: index 0 = most significant qubit (Qiskit ordering).
    SparsePauliOp to_sparse_pauli_op() const;

    // Evaluate energy for a given bitstring (x ∈ {0,1}^n, MSB first).
    double evaluate(const std::string& bitstring) const;

    // Evaluate energy for a given spin assignment (s ∈ {-1,+1}^n).
    double evaluate_spins(const std::vector<int>& spins) const;

    // Construct from a QUBO matrix Q (n×n, row-major).
    // The QUBO problem is: minimize x^T Q x, x ∈ {0,1}^n.
    // penalty_A scales diagonal terms — leave at 1.0 unless rescaling is needed.
    static IsingHamiltonian from_qubo(
        const std::vector<std::vector<double>>& Q,
        double penalty_A = 1.0
    );

    // Construct directly from h and J vectors.
    static IsingHamiltonian from_hJ(
        const std::vector<double>& h,
        const std::vector<std::vector<double>>& J,
        double offset = 0.0
    );
};

} // namespace lindblad
