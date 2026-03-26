#include "qpp/ising.hpp"

#include <stdexcept>

namespace qpp {

// =============================================================================
// IsingHamiltonian::to_sparse_pauli_op
//
// H = offset + sum_i h[i] Z_i + sum_{i<j} J[i][j] Z_i Z_j
//
// Pauli string convention: pauli[0] acts on the most significant qubit (qubit
// n-1 in ket notation). So Z on qubit q maps to position (n-1-q) in the string.
// =============================================================================

SparsePauliOp IsingHamiltonian::to_sparse_pauli_op() const {
    int n = n_qubits();
    if (n == 0) return SparsePauliOp();

    std::vector<PauliString> terms;

    // Constant offset as identity
    if (offset != 0.0) {
        terms.push_back({std::string(n, 'I'), Complex128(offset, 0.0)});
    }

    // Linear terms: h[i] * Z_i
    for (int i = 0; i < n; ++i) {
        if (h[i] == 0.0) continue;
        std::string p(n, 'I');
        p[n - 1 - i] = 'Z';   // qubit i → position n-1-i (MSB-first)
        terms.push_back({p, Complex128(h[i], 0.0)});
    }

    // Quadratic terms: J[i][j] * Z_i Z_j  (i < j)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double v = J[i][j];
            if (v == 0.0) continue;
            std::string p(n, 'I');
            p[n - 1 - i] = 'Z';
            p[n - 1 - j] = 'Z';
            terms.push_back({p, Complex128(v, 0.0)});
        }
    }

    return SparsePauliOp(terms);
}

// =============================================================================
// IsingHamiltonian::evaluate
//
// bitstring: x ∈ {0,1}^n, MSB first (x[0] = qubit n-1).
// Spin mapping: s_i = 1 - 2*x_i  (x=0 → s=+1, x=1 → s=-1).
// Energy: offset + sum_i h[i]*s_i + sum_{i<j} J[i][j]*s_i*s_j
// =============================================================================

double IsingHamiltonian::evaluate(const std::string& bitstring) const {
    int n = n_qubits();
    if (static_cast<int>(bitstring.size()) != n) {
        throw std::invalid_argument("Bitstring length must match n_qubits");
    }

    std::vector<int> spins(n);
    for (int i = 0; i < n; ++i) {
        int x_i = (bitstring[n - 1 - i] == '1') ? 1 : 0;  // qubit i at position n-1-i
        spins[i] = 1 - 2 * x_i;
    }

    return evaluate_spins(spins);
}

double IsingHamiltonian::evaluate_spins(const std::vector<int>& spins) const {
    int n = n_qubits();
    double energy = offset;
    for (int i = 0; i < n; ++i) energy += h[i] * spins[i];
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            energy += J[i][j] * spins[i] * spins[j];
    return energy;
}

// =============================================================================
// IsingHamiltonian::from_qubo
//
// QUBO: minimize x^T Q x, x ∈ {0,1}^n
//   = sum_i Q[i][i] x_i + sum_{i<j} (Q[i][j] + Q[j][i]) x_i x_j
//
// Substitution x_i = (1 - s_i) / 2, s_i ∈ {-1, +1}:
//   x_i x_j = (1 - s_i)(1 - s_j) / 4
//           = (1 - s_i - s_j + s_i s_j) / 4
//
// QUBO cost = sum_i Q_ii * (1-s_i)/2
//           + sum_{i<j} (Q_ij+Q_ji) * (1-s_i-s_j+s_i s_j)/4
//
// Grouping by spin products:
//   h[i] = -Q_ii/2  - sum_{j≠i} (Q_ij+Q_ji)/4
//   J[i][j] = (Q_ij + Q_ji) / 4    (i < j)
//   offset  = sum_i Q_ii/2 + sum_{i<j} (Q_ij+Q_ji)/4
// =============================================================================

IsingHamiltonian IsingHamiltonian::from_qubo(
    const std::vector<std::vector<double>>& Q,
    double penalty_A
) {
    int n = static_cast<int>(Q.size());
    for (const auto& row : Q) {
        if (static_cast<int>(row.size()) != n) {
            throw std::invalid_argument("Q must be a square matrix");
        }
    }

    IsingHamiltonian ising;
    ising.h.resize(n, 0.0);
    ising.J.assign(n, std::vector<double>(n, 0.0));

    double off = 0.0;

    for (int i = 0; i < n; ++i) {
        double Qii = penalty_A * Q[i][i];
        // Diagonal term: Q_ii * x_i = Q_ii * (1 - s_i) / 2
        ising.h[i] -= Qii / 2.0;
        off += Qii / 2.0;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double Qij = penalty_A * (Q[i][j] + Q[j][i]);
            // Off-diagonal: (Q_ij+Q_ji) * x_i x_j = Qij * (1 - s_i - s_j + s_i s_j) / 4
            ising.h[i] -= Qij / 4.0;
            ising.h[j] -= Qij / 4.0;
            ising.J[i][j] = Qij / 4.0;
            off += Qij / 4.0;
        }
    }

    ising.offset = off;
    return ising;
}

IsingHamiltonian IsingHamiltonian::from_hJ(
    const std::vector<double>& h_in,
    const std::vector<std::vector<double>>& J_in,
    double offset_in
) {
    int n = static_cast<int>(h_in.size());
    if (static_cast<int>(J_in.size()) != n) {
        throw std::invalid_argument("J must have same dimension as h");
    }
    IsingHamiltonian ising;
    ising.h = h_in;
    ising.J = J_in;
    ising.offset = offset_in;
    return ising;
}

} // namespace qpp
