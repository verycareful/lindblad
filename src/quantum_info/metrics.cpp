// metrics.cpp — Quantum information metrics
// All quantities computed with full Eigen3 eigendecomposition (no approximations).
// - Von Neumann entropy: SelfAdjointEigenSolver on density matrix.
// - Mixed-state fidelity: Uhlmann-Jozsa F = (Tr sqrt(sqrt(rho1)*rho2*sqrt(rho1)))^2.
// - Concurrence: sqrt of square roots of eigenvalues of rho * rho_tilde.

#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace lindblad {
namespace QuantumInfo {

// =============================================================================
// Internal helpers: convert lindblad DensityMatrix to Eigen Hermitian matrix
// =============================================================================

static Eigen::MatrixXcd to_eigen(const DensityMatrix& rho) {
    int d = static_cast<int>(rho.dim);
    Eigen::MatrixXcd M(d, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            M(i, j) = std::complex<double>(rho.data[i * d + j].real,
                                           rho.data[i * d + j].imag);
    return M;
}

static Eigen::MatrixXcd to_eigen(const Operator& op) {
    int d = static_cast<int>(op.dim());
    Eigen::MatrixXcd M(d, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            M(i, j) = std::complex<double>(op.data[i * d + j].real,
                                           op.data[i * d + j].imag);
    return M;
}

// Matrix square root of a positive semidefinite Hermitian matrix via eigendecomposition
// sqrt(A) = V * diag(sqrt(eigenvalues)) * V†
static Eigen::MatrixXcd matrix_sqrt(const Eigen::MatrixXcd& A) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(A);
    const auto& vals = es.eigenvalues();
    const auto& vecs = es.eigenvectors();
    Eigen::VectorXd sqrt_vals = vals.array().max(0.0).sqrt();
    return vecs * sqrt_vals.asDiagonal() * vecs.adjoint();
}

// =============================================================================
// State fidelity: |⟨ψ1|ψ2⟩|²
// =============================================================================

double state_fidelity(const Statevector& sv1, const Statevector& sv2) {
    Complex128 inner = sv1.inner_product(sv2);
    return inner.norm_sq();
}

// Mixed-state fidelity: Uhlmann-Jozsa formula
// F(ρ, σ) = (Tr √(√ρ σ √ρ))²
double state_fidelity(const DensityMatrix& rho1, const DensityMatrix& rho2) {
    if (rho1.dim != rho2.dim)
        throw std::invalid_argument("Density matrix dimension mismatch");

    // Special case: if rho1 is a pure state |ψ⟩⟨ψ|, fidelity = ⟨ψ|rho2|ψ⟩
    // Check purity quickly
    auto M1 = to_eigen(rho1);
    auto M2 = to_eigen(rho2);

    // sqrt_rho1 = sqrt(rho1)
    Eigen::MatrixXcd sqrt_rho1 = matrix_sqrt(M1);

    // inner = sqrt(rho1) * rho2 * sqrt(rho1)
    Eigen::MatrixXcd inner_mat = sqrt_rho1 * M2 * sqrt_rho1;

    // sqrt of inner_mat
    Eigen::MatrixXcd sqrt_inner = matrix_sqrt(inner_mat);

    // Fidelity = (Tr(sqrt_inner))^2
    std::complex<double> tr = sqrt_inner.trace();
    double f = tr.real();
    return f * f;
}

// =============================================================================
// Process fidelity — SQUARED Hilbert-Schmidt inner product
// F_proc(U, V) = |Tr(U† V)|² / d²
// This returns the squared quantity. For unsquared, take sqrt() of result.
// =============================================================================

double process_fidelity(const Operator& channel1, const Operator& channel2) {
    auto M1 = to_eigen(channel1);
    auto M2 = to_eigen(channel2);
    double d = static_cast<double>(channel1.dim());
    std::complex<double> tr = (M1.adjoint() * M2).trace();
    return (tr * std::conj(tr)).real() / (d * d);
}

// =============================================================================
// Average gate fidelity: F_avg = (d * F_proc + 1) / (d + 1)
// =============================================================================

double average_gate_fidelity(const Operator& channel, const Operator& target) {
    double d = static_cast<double>(channel.dim());
    double f_proc = process_fidelity(channel, target);
    return (d * f_proc + 1.0) / (d + 1.0);
}

// =============================================================================
// Von Neumann entropy: -Tr(ρ log ρ) using full Eigen eigendecomposition
// =============================================================================

double entropy(const DensityMatrix& rho, double base) {
    auto M = to_eigen(rho);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
    const auto& vals = es.eigenvalues();

    double log_base = std::log(base);
    double result = 0.0;
    for (int i = 0; i < vals.size(); ++i) {
        double p = std::max(0.0, vals(i));
        if (p > 1e-15) {
            result -= p * std::log(p) / log_base;
        }
    }
    return result;
}

// =============================================================================
// Entanglement entropy: entropy of reduced density matrix
// =============================================================================

double entanglement_entropy(const Statevector& sv, const std::vector<int>& subsystem) {
    auto rho_reduced = partial_trace(sv, subsystem);
    return entropy(rho_reduced, 2.0);
}

// =============================================================================
// Concurrence (2-qubit) — Wootters formula
// C(ρ) = max(0, λ1 - λ2 - λ3 - λ4)
// where λ_i are square roots of eigenvalues of R = √ρ * ρ̃ * √ρ,
// ρ̃ = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
// =============================================================================

double concurrence(const DensityMatrix& rho) {
    if (rho.n_qubits != 2)
        throw std::invalid_argument("Concurrence requires a 2-qubit density matrix");

    auto M = to_eigen(rho);

    // Build σ_y ⊗ σ_y (4x4 real-valued anti-symmetric)
    // σ_y = [[0,-i],[i,0]] → σ_y ⊗ σ_y = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
    Eigen::MatrixXcd sysy = Eigen::MatrixXcd::Zero(4, 4);
    sysy(0, 3) = -1.0;
    sysy(1, 2) =  1.0;
    sysy(2, 1) =  1.0;
    sysy(3, 0) = -1.0;

    // rho_tilde = sysy * conj(rho) * sysy
    Eigen::MatrixXcd rho_tilde = sysy * M.conjugate() * sysy;

    // sqrt_rho
    Eigen::MatrixXcd sqrt_rho = matrix_sqrt(M);

    // R = sqrt_rho * rho_tilde * sqrt_rho
    Eigen::MatrixXcd R = sqrt_rho * rho_tilde * sqrt_rho;

    // Eigenvalues of R (it should be positive semi-definite)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(R);
    Eigen::VectorXd evals = es.eigenvalues().array().max(0.0);

    // Sort eigenvalues descending and take square roots
    std::vector<double> lambdas(evals.data(), evals.data() + evals.size());
    std::sort(lambdas.begin(), lambdas.end(), std::greater<double>());
    for (auto& l : lambdas) l = std::sqrt(l);

    double C = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3];
    return std::max(0.0, C);
}

// =============================================================================
// Partial trace
// =============================================================================

DensityMatrix partial_trace(const DensityMatrix& rho, const std::vector<int>& trace_out_qubits) {
    int nq = rho.n_qubits;
    int nt = static_cast<int>(trace_out_qubits.size());
    int nk = nq - nt;
    size_t dim_keep = 1ULL << nk;
    size_t dim_trace = 1ULL << nt;

    // Build set of kept qubits (complement of traced-out qubits)
    std::vector<int> keep_qubits;
    for (int q = 0; q < nq; ++q) {
        bool traced = false;
        for (int tq : trace_out_qubits) if (tq == q) { traced = true; break; }
        if (!traced) keep_qubits.push_back(q);
    }

    // The traced-out qubits
    std::vector<int> trace_qubits = trace_out_qubits;

    DensityMatrix result(nk);

    // Map full index -> kept sub-index and traced sub-index
    auto kept_sub = [&](size_t full) -> size_t {
        size_t sub = 0;
        for (int ki = 0; ki < nk; ++ki)
            if ((full >> keep_qubits[ki]) & 1) sub |= (size_t(1) << ki);
        return sub;
    };

    auto traced_sub = [&](size_t full) -> size_t {
        size_t sub = 0;
        for (int ti = 0; ti < nt; ++ti)
            if ((full >> trace_qubits[ti]) & 1) sub |= (size_t(1) << ti);
        return sub;
    };

    auto build_full = [&](size_t keep_idx, size_t trace_idx) -> size_t {
        size_t full = 0;
        for (int ki = 0; ki < nk; ++ki)
            if ((keep_idx >> ki) & 1) full |= (size_t(1) << keep_qubits[ki]);
        for (int ti = 0; ti < nt; ++ti)
            if ((trace_idx >> ti) & 1) full |= (size_t(1) << trace_qubits[ti]);
        return full;
    };
    (void)kept_sub; (void)traced_sub;

    for (size_t i = 0; i < dim_keep; ++i) {
        for (size_t j = 0; j < dim_keep; ++j) {
            Complex128 sum(0.0, 0.0);
            for (size_t t = 0; t < dim_trace; ++t) {
                size_t full_i = build_full(i, t);
                size_t full_j = build_full(j, t);
                sum += rho.data[full_i * rho.dim + full_j];
            }
            result(i, j) = sum;
        }
    }

    return result;
}

DensityMatrix partial_trace(const Statevector& sv, const std::vector<int>& trace_out_qubits) {
    DensityMatrix rho = DensityMatrix::from_statevector(sv);
    return partial_trace(rho, trace_out_qubits);
}

// =============================================================================
// Pauli expectation values
// =============================================================================

std::vector<double> pauli_expectation_values(
    const Statevector& sv,
    const std::vector<std::string>& paulis
) {
    std::vector<double> results;
    results.reserve(paulis.size());
    for (const auto& pauli_str : paulis) {
        SparsePauliOp op({{pauli_str, Complex128(1.0, 0.0)}});
        results.push_back(op.expectation_value(sv));
    }
    return results;
}

} // namespace QuantumInfo
} // namespace lindblad
