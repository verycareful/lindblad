#include "qpp/operators.hpp"
#include "qpp/statevector.hpp"
#include "qpp/simulators/density_matrix_sim.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace qpp {
namespace QuantumInfo {

// =============================================================================
// State fidelity: |⟨ψ1|ψ2⟩|²
// =============================================================================

double state_fidelity(const Statevector& sv1, const Statevector& sv2) {
    Complex128 inner = sv1.inner_product(sv2);
    return inner.norm_sq();
}

double state_fidelity(const DensityMatrix& rho1, const DensityMatrix& rho2) {
    // F(rho1, rho2) = [Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1)))]^2
    // Simplified for now: Tr(rho1 * rho2) — this is the Hilbert-Schmidt fidelity
    if (rho1.dim != rho2.dim) {
        throw std::invalid_argument("Dimension mismatch");
    }

    double result = 0.0;
    for (size_t i = 0; i < rho1.dim; ++i) {
        for (size_t j = 0; j < rho1.dim; ++j) {
            Complex128 prod = rho1.data[i * rho1.dim + j] * rho2.data[j * rho2.dim + i];
            result += prod.real;
        }
    }
    return result;
}

// =============================================================================
// Process fidelity
// =============================================================================

double process_fidelity(const Operator& channel1, const Operator& channel2) {
    // F_process = |Tr(U1† U2)|² / d²
    auto adj1 = channel1.adjoint();
    auto product = adj1.compose(channel2);
    Complex128 tr = product.trace();
    double d = static_cast<double>(channel1.dim());
    return tr.norm_sq() / (d * d);
}

// =============================================================================
// Average gate fidelity
// =============================================================================

double average_gate_fidelity(const Operator& channel, const Operator& target) {
    double d = static_cast<double>(channel.dim());
    double f_proc = process_fidelity(channel, target);
    return (d * f_proc + 1.0) / (d + 1.0);
}

// =============================================================================
// Von Neumann entropy: -Tr(rho * log(rho))
// =============================================================================

double entropy(const DensityMatrix& rho, double base) {
    // Compute eigenvalues of rho
    // For now, use the diagonal elements as approximation for diagonal states
    // TODO: implement full eigendecomposition

    double result = 0.0;
    double log_base = std::log(base);

    for (size_t i = 0; i < rho.dim; ++i) {
        double p = rho.data[i * rho.dim + i].real;
        if (p > 1e-15) {
            result -= p * std::log(p) / log_base;
        }
    }

    return result;
}

// =============================================================================
// Entanglement entropy
// =============================================================================

double entanglement_entropy(const Statevector& sv, const std::vector<int>& subsystem) {
    auto rho_reduced = partial_trace(sv, subsystem);
    return entropy(rho_reduced, 2.0);
}

// =============================================================================
// Concurrence (2-qubit entanglement measure)
// =============================================================================

double concurrence(const DensityMatrix& rho) {
    if (rho.n_qubits != 2) {
        throw std::invalid_argument("Concurrence requires a 2-qubit state");
    }

    // C(rho) = max(0, λ1 - λ2 - λ3 - λ4)
    // where λ_i are eigenvalues of sqrt(sqrt(rho) * rho_tilde * sqrt(rho))
    // rho_tilde = (σ_y ⊗ σ_y) * rho* * (σ_y ⊗ σ_y)

    // Simplified: for pure states, C = 2|ad - bc| where |ψ⟩ = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩
    // For mixed states, use the full formula

    // Build σ_y ⊗ σ_y
    std::vector<Complex128> sigma_yy(16, Complex128(0.0, 0.0));
    // σ_y = [[0, -i], [i, 0]]
    // σ_y ⊗ σ_y = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
    sigma_yy[0*4+3] = Complex128(-1, 0);
    sigma_yy[1*4+2] = Complex128(1, 0);
    sigma_yy[2*4+1] = Complex128(1, 0);
    sigma_yy[3*4+0] = Complex128(-1, 0);

    // rho_tilde = sigma_yy * conj(rho) * sigma_yy
    std::vector<Complex128> rho_conj(16);
    for (int i = 0; i < 16; ++i) rho_conj[i] = rho.data[i].conj();

    // temp = sigma_yy * rho_conj
    std::vector<Complex128> temp(16, Complex128(0,0));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                temp[i*4+j] += sigma_yy[i*4+k] * rho_conj[k*4+j];
            }
        }
    }

    // rho_tilde = temp * sigma_yy
    std::vector<Complex128> rho_tilde(16, Complex128(0,0));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                rho_tilde[i*4+j] += temp[i*4+k] * sigma_yy[k*4+j];
            }
        }
    }

    // R = rho * rho_tilde
    std::vector<Complex128> R(16, Complex128(0,0));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                R[i*4+j] += rho.data[i*4+k] * rho_tilde[k*4+j];
            }
        }
    }

    // Eigenvalues of R (using trace/det for 4x4 — simplified)
    // For now return 0 for mixed states
    return 0.0;
}

// =============================================================================
// Partial trace
// =============================================================================

DensityMatrix partial_trace(const DensityMatrix& rho, const std::vector<int>& keep_qubits) {
    int nq = rho.n_qubits;
    int nk = static_cast<int>(keep_qubits.size());
    int nt = nq - nk;
    size_t dim_keep = 1ULL << nk;
    size_t dim_trace = 1ULL << nt;

    // Determine traced-out qubits
    std::vector<int> trace_qubits;
    for (int q = 0; q < nq; ++q) {
        bool keep = false;
        for (int kq : keep_qubits) {
            if (kq == q) { keep = true; break; }
        }
        if (!keep) trace_qubits.push_back(q);
    }

    DensityMatrix result(nk);

    for (size_t i = 0; i < dim_keep; ++i) {
        for (size_t j = 0; j < dim_keep; ++j) {
            Complex128 sum(0.0, 0.0);
            for (size_t t = 0; t < dim_trace; ++t) {
                // Map (i, t) to full index for row
                size_t full_i = 0;
                int ki = 0, ti = 0;
                for (int q = 0; q < nq; ++q) {
                    bool is_keep = false;
                    for (int kq : keep_qubits) {
                        if (kq == q) { is_keep = true; break; }
                    }
                    if (is_keep) {
                        if ((i >> ki) & 1) full_i |= (1ULL << q);
                        ki++;
                    } else {
                        if ((t >> ti) & 1) full_i |= (1ULL << q);
                        ti++;
                    }
                }

                // Map (j, t) to full index for column
                size_t full_j = 0;
                ki = 0; ti = 0;
                for (int q = 0; q < nq; ++q) {
                    bool is_keep = false;
                    for (int kq : keep_qubits) {
                        if (kq == q) { is_keep = true; break; }
                    }
                    if (is_keep) {
                        if ((j >> ki) & 1) full_j |= (1ULL << q);
                        ki++;
                    } else {
                        if ((t >> ti) & 1) full_j |= (1ULL << q);
                        ti++;
                    }
                }

                sum += rho.data[full_i * rho.dim + full_j];
            }
            result(i, j) = sum;
        }
    }

    return result;
}

DensityMatrix partial_trace(const Statevector& sv, const std::vector<int>& keep_qubits) {
    DensityMatrix rho = DensityMatrix::from_statevector(sv);
    return partial_trace(rho, keep_qubits);
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
} // namespace qpp
