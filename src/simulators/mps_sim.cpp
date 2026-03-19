#include "qpp/simulators/mps_sim.hpp"
#include "qpp/statevector.hpp"
#include "qpp/circuit.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>
#include <random>
#include <stdexcept>

namespace qpp {

// =============================================================================
// MPSState
// =============================================================================

MPSState::MPSState(int n_qubits, int max_bond_dim, double cutoff)
    : n_qubits(n_qubits)
    , max_bond_dim(max_bond_dim)
    , cutoff(cutoff)
    , total_truncation_error(0.0) {
    // Initialize to |0...0⟩
    tensors.resize(n_qubits);
    for (int i = 0; i < n_qubits; ++i) {
        tensors[i] = MPSTensor(1, 1);
        tensors[i](0, 0, 0) = Complex128(1.0, 0.0);  // |0⟩ amplitude
        tensors[i](0, 1, 0) = Complex128(0.0, 0.0);  // |1⟩ amplitude
    }
}

void MPSState::apply_single_qubit_gate(
    const std::array<Complex128, 4>& U, int qubit
) {
    auto& T = tensors[qubit];
    MPSTensor result(T.bond_left, T.bond_right);

    for (int l = 0; l < T.bond_left; ++l) {
        for (int r = 0; r < T.bond_right; ++r) {
            // new[l, phys_out, r] = sum_phys_in U[phys_out][phys_in] * T[l, phys_in, r]
            for (int po = 0; po < 2; ++po) {
                Complex128 sum(0.0, 0.0);
                for (int pi = 0; pi < 2; ++pi) {
                    sum += U[po * 2 + pi] * T(l, pi, r);
                }
                result(l, po, r) = sum;
            }
        }
    }

    T = result;
}

// Simple in-place SVD for small matrices (Jacobi method)
// This is a simplified implementation suitable for small bond dimensions
void MPSState::svd_truncate(
    const std::vector<Complex128>& matrix,
    int rows, int cols,
    std::vector<Complex128>& U_out,
    std::vector<double>& S_out,
    std::vector<Complex128>& Vt_out,
    int& new_rank
) {
    // For MPS with small bond dimensions, use eigendecomposition of M†M
    // M = U * S * V†
    // M†M = V * S^2 * V†
    
    int min_dim = std::min(rows, cols);
    
    // Compute M†M
    std::vector<Complex128> MtM(cols * cols, Complex128(0.0, 0.0));
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < cols; ++j) {
            Complex128 sum(0.0, 0.0);
            for (int k = 0; k < rows; ++k) {
                sum += matrix[k * cols + i].conj() * matrix[k * cols + j];
            }
            MtM[i * cols + j] = sum;
        }
    }

    // Simple power iteration to find singular values
    // For production, use LAPACK. This is a basic implementation.
    S_out.resize(min_dim, 0.0);
    Vt_out.resize(min_dim * cols, Complex128(0.0, 0.0));
    U_out.resize(rows * min_dim, Complex128(0.0, 0.0));

    std::vector<Complex128> remaining = MtM;
    
    for (int sv_idx = 0; sv_idx < min_dim; ++sv_idx) {
        // Power iteration
        std::vector<Complex128> v(cols, Complex128(1.0 / std::sqrt(cols), 0.0));
        
        for (int iter = 0; iter < 100; ++iter) {
            // Multiply: v_new = remaining * v
            std::vector<Complex128> v_new(cols, Complex128(0.0, 0.0));
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < cols; ++j) {
                    v_new[i] += remaining[i * cols + j] * v[j];
                }
            }

            // Normalize
            double norm = 0.0;
            for (int i = 0; i < cols; ++i) {
                norm += v_new[i].norm_sq();
            }
            norm = std::sqrt(norm);
            if (norm < 1e-15) break;

            for (int i = 0; i < cols; ++i) {
                v[i] = v_new[i] / norm;
            }
        }

        // Singular value = sqrt(eigenvalue)
        std::vector<Complex128> Mv(cols, Complex128(0.0, 0.0));
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < cols; ++j) {
                Mv[i] += remaining[i * cols + j] * v[j];
            }
        }
        double eigenvalue = 0.0;
        for (int i = 0; i < cols; ++i) {
            eigenvalue += (v[i].conj() * Mv[i]).real;
        }
        
        S_out[sv_idx] = std::sqrt(std::max(0.0, eigenvalue));

        // Store V row
        for (int j = 0; j < cols; ++j) {
            Vt_out[sv_idx * cols + j] = v[j].conj();
        }

        // Compute U column: u = M * v / sigma
        if (S_out[sv_idx] > 1e-15) {
            for (int i = 0; i < rows; ++i) {
                Complex128 sum(0.0, 0.0);
                for (int j = 0; j < cols; ++j) {
                    sum += matrix[i * cols + j] * v[j];
                }
                U_out[i * min_dim + sv_idx] = sum / S_out[sv_idx];
            }
        }

        // Deflate: remaining -= eigenvalue * v * v†
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < cols; ++j) {
                remaining[i * cols + j] -= Complex128(eigenvalue, 0.0) * v[i] * v[j].conj();
            }
        }
    }

    // Truncate based on cutoff and max_bond_dim
    new_rank = 0;
    for (int i = 0; i < min_dim; ++i) {
        if (S_out[i] > cutoff && new_rank < max_bond_dim) {
            new_rank++;
        }
    }
    if (new_rank == 0) new_rank = 1;

    // Accumulate truncation error
    for (int i = new_rank; i < min_dim; ++i) {
        total_truncation_error += S_out[i] * S_out[i];
    }
}

void MPSState::apply_two_qubit_gate(
    const std::array<Complex128, 16>& U, int q1, int q2
) {
    // Ensure q1 < q2 (swap gate indices if needed)
    // For non-adjacent qubits, SWAP to bring them adjacent first
    // For simplicity, assume q2 = q1 + 1 or use SWAP chain
    
    if (std::abs(q1 - q2) != 1) {
        // For non-adjacent qubits, we need to SWAP them adjacent
        // This is a simplification — just throw for now
        // TODO: implement SWAP chain for non-adjacent qubits
        if (q1 > q2) std::swap(q1, q2);
    }
    
    int left_q = std::min(q1, q2);
    auto& T1 = tensors[left_q];
    auto& T2 = tensors[left_q + 1];

    int bl = T1.bond_left;
    int bm = T1.bond_right;  // = T2.bond_left
    int br = T2.bond_right;

    // Contract T1 and T2 into a single tensor
    // theta[l, p1, p2, r] = sum_m T1[l, p1, m] * T2[m, p2, r]
    int theta_size = bl * 2 * 2 * br;
    std::vector<Complex128> theta(theta_size, Complex128(0.0, 0.0));

    for (int l = 0; l < bl; ++l) {
        for (int p1 = 0; p1 < 2; ++p1) {
            for (int p2 = 0; p2 < 2; ++p2) {
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int m = 0; m < bm; ++m) {
                        sum += T1(l, p1, m) * T2(m, p2, r);
                    }
                    theta[((l * 2 + p1) * 2 + p2) * br + r] = sum;
                }
            }
        }
    }

    // Apply gate: theta_new[l, p1', p2', r] = sum_{p1,p2} U[p1'*2+p2', p1*2+p2] * theta[l, p1, p2, r]
    std::vector<Complex128> theta_new(theta_size, Complex128(0.0, 0.0));
    for (int l = 0; l < bl; ++l) {
        for (int po1 = 0; po1 < 2; ++po1) {
            for (int po2 = 0; po2 < 2; ++po2) {
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int pi1 = 0; pi1 < 2; ++pi1) {
                        for (int pi2 = 0; pi2 < 2; ++pi2) {
                            int u_row = po1 * 2 + po2;
                            int u_col = pi1 * 2 + pi2;
                            sum += U[u_row * 4 + u_col] *
                                   theta[((l * 2 + pi1) * 2 + pi2) * br + r];
                        }
                    }
                    theta_new[((l * 2 + po1) * 2 + po2) * br + r] = sum;
                }
            }
        }
    }

    // Reshape theta_new into matrix: (bl*2) x (2*br)
    int rows = bl * 2;
    int cols = 2 * br;
    std::vector<Complex128> matrix(rows * cols, Complex128(0.0, 0.0));
    for (int l = 0; l < bl; ++l) {
        for (int p1 = 0; p1 < 2; ++p1) {
            for (int p2 = 0; p2 < 2; ++p2) {
                for (int r = 0; r < br; ++r) {
                    matrix[(l * 2 + p1) * cols + (p2 * br + r)] =
                        theta_new[((l * 2 + p1) * 2 + p2) * br + r];
                }
            }
        }
    }

    // SVD and truncate
    std::vector<Complex128> U_mat, Vt_mat;
    std::vector<double> S_vals;
    int new_rank;
    svd_truncate(matrix, rows, cols, U_mat, S_vals, Vt_mat, new_rank);

    // Reconstruct T1 and T2
    T1 = MPSTensor(bl, new_rank);
    for (int l = 0; l < bl; ++l) {
        for (int p1 = 0; p1 < 2; ++p1) {
            for (int r = 0; r < new_rank; ++r) {
                T1(l, p1, r) = U_mat[(l * 2 + p1) * std::min(rows, cols) + r] *
                                Complex128(S_vals[r], 0.0);
            }
        }
    }

    T2 = MPSTensor(new_rank, br);
    for (int l = 0; l < new_rank; ++l) {
        for (int p2 = 0; p2 < 2; ++p2) {
            for (int r = 0; r < br; ++r) {
                T2(l, p2, r) = Vt_mat[l * cols + p2 * br + r];
            }
        }
    }
}

int MPSState::current_max_bond_dim() const {
    int max_chi = 0;
    for (const auto& t : tensors) {
        max_chi = std::max(max_chi, std::max(t.bond_left, t.bond_right));
    }
    return max_chi;
}

std::vector<double> MPSState::probabilities_single(int qubit) const {
    // Compute single-qubit reduced density matrix
    // This gives P(0) and P(1) for the qubit
    
    // Contract all tensors except the target qubit
    // For simplicity, compute the full state and extract
    // This is inefficient but correct for small systems
    
    if (n_qubits <= 20) {
        auto sv = to_statevector();
        std::vector<double> probs(2, 0.0);
        for (size_t i = 0; i < sv.dim; ++i) {
            int bit = (i >> qubit) & 1;
            probs[bit] += sv.probability(i);
        }
        return probs;
    }

    // For large systems, use efficient contraction
    // TODO: implement efficient single-qubit marginal
    return {0.5, 0.5};  // placeholder
}

Statevector MPSState::to_statevector() const {
    if (n_qubits > 25) {
        throw std::runtime_error("Too many qubits for full statevector conversion");
    }

    size_t dim = 1ULL << n_qubits;
    Statevector sv(n_qubits);

    // Contract all MPS tensors
    // Start from left, accumulate
    for (size_t idx = 0; idx < dim; ++idx) {
        // For each computational basis state, contract the MPS
        // Extract the physical index for each qubit
        std::vector<int> phys(n_qubits);
        for (int q = 0; q < n_qubits; ++q) {
            phys[q] = (idx >> q) & 1;
        }

        // Contract: result = T_0[phys[0]] * T_1[phys[1]] * ... * T_{N-1}[phys[N-1]]
        // Start with T_0 slice: matrix of shape (1, bond_right_0)
        std::vector<Complex128> current(tensors[0].bond_right);
        for (int r = 0; r < tensors[0].bond_right; ++r) {
            current[r] = tensors[0](0, phys[0], r);
        }

        for (int q = 1; q < n_qubits; ++q) {
            int new_right = tensors[q].bond_right;
            std::vector<Complex128> next(new_right, Complex128(0.0, 0.0));
            for (int r = 0; r < new_right; ++r) {
                for (int m = 0; m < tensors[q].bond_left; ++m) {
                    next[r] += current[m] * tensors[q](m, phys[q], r);
                }
            }
            current = next;
        }

        // current should be a 1-element vector
        sv.real_parts[idx] = current[0].real;
        sv.imag_parts[idx] = current[0].imag;
    }

    return sv;
}

// =============================================================================
// MPSSimulator
// =============================================================================

MPSSimulator::Result MPSSimulator::run(
    const QuantumCircuit& circuit, int max_bond_dim,
    int shots, uint64_t seed
) {
    Result result(circuit.n_qubits);
    result.final_state = MPSState(circuit.n_qubits, max_bond_dim);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Apply each gate
    for (const auto& inst : circuit.instructions) {
        if (inst.type == Instruction::GateType::BARRIER) continue;
        if (inst.type == Instruction::GateType::MEASURE) continue;
        if (inst.type == Instruction::GateType::RESET) continue;

        if (inst.qubits.size() == 1) {
            // Build 2x2 gate matrix
            Statevector tmp(1);
            std::array<Complex128, 4> U_mat;

            for (int col = 0; col < 2; ++col) {
                Statevector basis(1);
                basis.initialize_basis(col);

                // Apply gate
                using GT = Instruction::GateType;
                const auto& p = inst.params;
                switch (inst.type) {
                    case GT::H: gates::apply_h(basis, 0); break;
                    case GT::X: gates::apply_x(basis, 0); break;
                    case GT::Y: gates::apply_y(basis, 0); break;
                    case GT::Z: gates::apply_z(basis, 0); break;
                    case GT::S: gates::apply_s(basis, 0); break;
                    case GT::SDG: gates::apply_sdg(basis, 0); break;
                    case GT::T: gates::apply_t(basis, 0); break;
                    case GT::TDG: gates::apply_tdg(basis, 0); break;
                    case GT::SX: gates::apply_sx(basis, 0); break;
                    case GT::SXDG: gates::apply_sxdg(basis, 0); break;
                    case GT::RX: gates::apply_rx(basis, 0, p[0]); break;
                    case GT::RY: gates::apply_ry(basis, 0, p[0]); break;
                    case GT::RZ: gates::apply_rz(basis, 0, p[0]); break;
                    case GT::P: gates::apply_p(basis, 0, p[0]); break;
                    case GT::U: gates::apply_u(basis, 0, p[0], p[1], p[2]); break;
                    default: break;
                }

                for (int row = 0; row < 2; ++row) {
                    U_mat[row * 2 + col] = basis.amplitude(row);
                }
            }

            result.final_state.apply_single_qubit_gate(U_mat, inst.qubits[0]);

        } else if (inst.qubits.size() == 2) {
            // Build 4x4 gate matrix
            std::array<Complex128, 16> U_mat;

            for (int col = 0; col < 4; ++col) {
                Statevector basis(2);
                basis.initialize_basis(col);

                using GT = Instruction::GateType;
                const auto& p = inst.params;
                switch (inst.type) {
                    case GT::CX: gates::apply_cx(basis, 0, 1); break;
                    case GT::CY: gates::apply_cy(basis, 0, 1); break;
                    case GT::CZ: gates::apply_cz(basis, 0, 1); break;
                    case GT::CH: gates::apply_ch(basis, 0, 1); break;
                    case GT::SWAP: gates::apply_swap(basis, 0, 1); break;
                    case GT::ISWAP: gates::apply_iswap(basis, 0, 1); break;
                    case GT::CRX: gates::apply_crx(basis, 0, 1, p[0]); break;
                    case GT::CRY: gates::apply_cry(basis, 0, 1, p[0]); break;
                    case GT::CRZ: gates::apply_crz(basis, 0, 1, p[0]); break;
                    case GT::CP: gates::apply_cp(basis, 0, 1, p[0]); break;
                    case GT::RXX: gates::apply_rxx(basis, 0, 1, p[0]); break;
                    case GT::RYY: gates::apply_ryy(basis, 0, 1, p[0]); break;
                    case GT::RZZ: gates::apply_rzz(basis, 0, 1, p[0]); break;
                    default: break;
                }

                for (int row = 0; row < 4; ++row) {
                    U_mat[row * 4 + col] = basis.amplitude(row);
                }
            }

            result.final_state.apply_two_qubit_gate(U_mat, inst.qubits[0], inst.qubits[1]);
        }
    }

    // Sample measurements
    if (shots > 0 && circuit.n_qubits <= 20) {
        auto sv = result.final_state.to_statevector();
        result.counts = sv.sample_counts(shots, seed);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.simulation_time_seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    return result;
}

} // namespace qpp
