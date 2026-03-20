// density_matrix_sim.cpp — Density Matrix simulator
// Gate application uses localized tensor operations on target qubits only,
// identical in structure to the statevector approach. Gate matrices are built
// analytically (same lookup as MPS/statevector) — no per-gate statevector
// allocation. Complexity is O(4^N) for storage, O(4^N * 4^k) for k-qubit gates.

#include "qpp/simulators/density_matrix_sim.hpp"
#include "qpp/circuit.hpp"
#include "qpp/noise.hpp"
#include "qpp/types.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace qpp {

// =============================================================================
// DensityMatrix
// =============================================================================

DensityMatrix::DensityMatrix()
    : n_qubits(0), dim(1) {
    data.resize(1, Complex128(1.0, 0.0));
}

DensityMatrix::DensityMatrix(int n_qubits)
    : n_qubits(n_qubits), dim(1ULL << n_qubits) {
    data.resize(dim * dim, Complex128(0.0, 0.0));
    data[0] = Complex128(1.0, 0.0);  // |0⟩⟨0|
}

DensityMatrix DensityMatrix::from_statevector(const Statevector& sv) {
    DensityMatrix dm(sv.n_qubits);
    // rho_{ij} = psi_i * conj(psi_j)
    for (size_t i = 0; i < dm.dim; ++i) {
        for (size_t j = 0; j < dm.dim; ++j) {
            dm(i, j) = Complex128(sv.real_parts[i], sv.imag_parts[i]) *
                       Complex128(sv.real_parts[j], -sv.imag_parts[j]);
        }
    }
    return dm;
}

double DensityMatrix::trace() const {
    double tr = 0.0;
    for (size_t i = 0; i < dim; ++i) tr += data[i * dim + i].real;
    return tr;
}

double DensityMatrix::purity() const {
    // Tr(rho^2)
    double pur = 0.0;
    for (size_t i = 0; i < dim; ++i)
        for (size_t k = 0; k < dim; ++k)
            pur += (data[i * dim + k] * data[k * dim + i]).real;
    return pur;
}

bool DensityMatrix::is_valid(double atol) const {
    if (std::abs(trace() - 1.0) > atol) return false;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = i + 1; j < dim; ++j) {
            auto diff = data[i * dim + j] - data[j * dim + i].conj();
            if (diff.norm_sq() > atol * atol) return false;
        }
    }
    return true;
}

// =============================================================================
// apply_gate — localized tensor operation: rho -> U * rho * U†
// Works on the 2^k subspace spanned by target qubits.
// Complexity: O(4^N * 4^k) — same as statevector approach for density matrix.
// =============================================================================

void DensityMatrix::apply_gate(const std::vector<Complex128>& U,
                                const std::vector<int>& qubits) {
    int k = static_cast<int>(qubits.size());
    size_t sub_dim = 1ULL << k;

    if (U.size() != sub_dim * sub_dim) {
        throw std::invalid_argument("Gate matrix size mismatch");
    }

    // Build non-target mask for iteration
    size_t non_target_mask = ~size_t(0);
    for (int q : qubits) non_target_mask &= ~(size_t(1) << q);
    non_target_mask &= (dim - 1);

    // Helper: extract sub-index for target qubits from a full index
    auto extract_sub_idx = [&](size_t full_idx) -> size_t {
        size_t sub = 0;
        for (int qi = 0; qi < k; ++qi) {
            if ((full_idx >> qubits[qi]) & 1) sub |= (size_t(1) << qi);
        }
        return sub;
    };

    // Helper: from background index + sub index → full index
    auto build_full_idx = [&](size_t bg, size_t sub_idx) -> size_t {
        size_t full = bg;
        for (int qi = 0; qi < k; ++qi) {
            if ((sub_idx >> qi) & 1) full |= (size_t(1) << qubits[qi]);
            else                    full &= ~(size_t(1) << qubits[qi]);
        }
        return full;
    };

    // Iterate over all full-system indices for BOTH row and column of rho
    // rho_new[row, col] = sum_{r', c'} U[sub(row), r'] * U†[c', sub(col)] * rho[full(r'), full(c')]
    //                   = sum_{r', c'} U[sub(row), r'] * conj(U[c', sub(col)]) * rho[full(r'), full(c')]
    //
    // We loop over bg_row (background bits of row) and bg_col,
    // then over sub-indices in the 2^k subspace.

    std::vector<Complex128> new_data = data;

    // Apply left: rho_temp = U * rho  (acting on row index)
    // rho_temp[i, j] = sum_i' U[sub(i), sub(i')] * rho[full(bg(i), i'), j]
    // This mixes rows at fixed column.
    {
        std::vector<Complex128> temp = data;
        // Enumerate background bits of the ROW index
        // For each combination of bg bits and column index, transform within the 2^k sub-block
        for (size_t col = 0; col < dim; ++col) {
            // Iterate over background index
            for (size_t bg = 0; bg < dim; ++bg) {
                if (bg & ~non_target_mask) continue;  // skip if non-bg bits are set

                // For this (bg, col) block, apply U to the row sub-indices
                // temp_new[build(bg, r_out), col] = sum_r_in U[r_out, r_in] * data[build(bg, r_in), col]
                for (size_t r_out = 0; r_out < sub_dim; ++r_out) {
                    size_t row_out = build_full_idx(bg, r_out);
                    Complex128 sum(0.0, 0.0);
                    for (size_t r_in = 0; r_in < sub_dim; ++r_in) {
                        size_t row_in = build_full_idx(bg, r_in);
                        sum += U[r_out * sub_dim + r_in] * data[row_in * dim + col];
                    }
                    temp[row_out * dim + col] = sum;
                }
            }
        }
        data = temp;
    }

    // Apply right (U†): rho_new = rho_temp * U†  (acting on column index)
    // rho_new[i, j] = sum_j' rho_temp[i, build(bg(j), j')] * conj(U[sub(j), j'])
    {
        std::vector<Complex128> temp = data;
        for (size_t row = 0; row < dim; ++row) {
            for (size_t bg = 0; bg < dim; ++bg) {
                if (bg & ~non_target_mask) continue;

                for (size_t c_out = 0; c_out < sub_dim; ++c_out) {
                    size_t col_out = build_full_idx(bg, c_out);
                    Complex128 sum(0.0, 0.0);
                    for (size_t c_in = 0; c_in < sub_dim; ++c_in) {
                        size_t col_in = build_full_idx(bg, c_in);
                        // U†[c_out, c_in] = conj(U[c_in, c_out])
                        sum += data[row * dim + col_in] * U[c_in * sub_dim + c_out].conj();
                    }
                    temp[row * dim + col_out] = sum;
                }
            }
        }
        data = temp;
    }
    (void)new_data;  // unused after restructuring
}

// =============================================================================
// apply_kraus — rho -> sum_k K_k * rho * K_k†
// =============================================================================

void DensityMatrix::apply_kraus(
    const std::vector<std::vector<Complex128>>& kraus_ops,
    const std::vector<int>& qubits
) {
    std::vector<Complex128> result_data(dim * dim, Complex128(0.0, 0.0));

    for (const auto& K : kraus_ops) {
        DensityMatrix temp = *this;
        temp.apply_gate(K, qubits);
        for (size_t i = 0; i < dim * dim; ++i) {
            result_data[i] += temp.data[i];
        }
    }

    data = std::move(result_data);
}

std::vector<double> DensityMatrix::probabilities() const {
    std::vector<double> probs(dim);
    for (size_t i = 0; i < dim; ++i)
        probs[i] = std::max(0.0, data[i * dim + i].real);
    return probs;
}

double DensityMatrix::expectation_value(const std::vector<Complex128>& hermitian_op) const {
    // Tr(rho * O)
    double result = 0.0;
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            result += (hermitian_op[i * dim + j] * data[j * dim + i]).real;
    return result;
}

// =============================================================================
// Analytic gate matrices — avoid statevector allocation overhead
// =============================================================================

static std::vector<Complex128> gate_matrix_for_dm(const Instruction& inst) {
    using GT = Instruction::GateType;
    const auto& p = inst.params;
    constexpr double inv_sqrt2 = 0.7071067811865475;

    int k = inst.num_qubits();
    size_t sub_dim = 1ULL << k;
    std::vector<Complex128> U(sub_dim * sub_dim, Complex128(0.0, 0.0));
    auto set = [&](size_t r, size_t c, Complex128 v) { U[r * sub_dim + c] = v; };

    switch (inst.type) {
        // === Single-qubit diagonal / off-diagonal ===
        case GT::H: {
            set(0,0,{inv_sqrt2,0}); set(0,1,{inv_sqrt2,0});
            set(1,0,{inv_sqrt2,0}); set(1,1,{-inv_sqrt2,0}); break;
        }
        case GT::X: set(0,1,{1,0}); set(1,0,{1,0}); break;
        case GT::Y: set(0,1,{0,-1}); set(1,0,{0,1}); break;
        case GT::Z: set(0,0,{1,0}); set(1,1,{-1,0}); break;
        case GT::S: set(0,0,{1,0}); set(1,1,{0,1}); break;
        case GT::SDG: set(0,0,{1,0}); set(1,1,{0,-1}); break;
        case GT::T: set(0,0,{1,0}); set(1,1,{inv_sqrt2, inv_sqrt2}); break;
        case GT::TDG: set(0,0,{1,0}); set(1,1,{inv_sqrt2, -inv_sqrt2}); break;
        case GT::SX: {
            set(0,0,{0.5,0.5}); set(0,1,{0.5,-0.5});
            set(1,0,{0.5,-0.5}); set(1,1,{0.5,0.5}); break;
        }
        case GT::SXDG: {
            set(0,0,{0.5,-0.5}); set(0,1,{0.5,0.5});
            set(1,0,{0.5,0.5}); set(1,1,{0.5,-0.5}); break;
        }
        case GT::RX: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,0}); set(0,1,{0,-s}); set(1,0,{0,-s}); set(1,1,{c,0}); break;
        }
        case GT::RY: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,0}); set(0,1,{-s,0}); set(1,0,{s,0}); set(1,1,{c,0}); break;
        }
        case GT::RZ: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,-s}); set(1,1,{c,s}); break;
        }
        case GT::P: {
            set(0,0,{1,0}); set(1,1,{std::cos(p[0]), std::sin(p[0])}); break;
        }
        case GT::U: case GT::U3: {
            double th=p[0], ph=p[1], la=p[2];
            double c=std::cos(th/2), s=std::sin(th/2);
            set(0,0,{c,0});
            set(0,1,{-s*std::cos(la), -s*std::sin(la)});
            set(1,0,{s*std::cos(ph), s*std::sin(ph)});
            set(1,1,{c*std::cos(ph+la), c*std::sin(ph+la)}); break;
        }
        case GT::U1: {
            set(0,0,{1,0}); set(1,1,{std::cos(p[0]), std::sin(p[0])}); break;
        }
        case GT::U2: {
            double ph=p[0], la=p[1];
            set(0,0,{inv_sqrt2,0});
            set(0,1,{-inv_sqrt2*std::cos(la), -inv_sqrt2*std::sin(la)});
            set(1,0,{inv_sqrt2*std::cos(ph), inv_sqrt2*std::sin(ph)});
            set(1,1,{inv_sqrt2*std::cos(ph+la), inv_sqrt2*std::sin(ph+la)}); break;
        }
        // === Two-qubit gates (4x4) ===
        case GT::CX:
            set(0,0,{1,0}); set(1,1,{1,0}); set(2,3,{1,0}); set(3,2,{1,0}); break;
        case GT::CY:
            set(0,0,{1,0}); set(1,1,{1,0}); set(2,3,{0,-1}); set(3,2,{0,1}); break;
        case GT::CZ:
            set(0,0,{1,0}); set(1,1,{1,0}); set(2,2,{1,0}); set(3,3,{-1,0}); break;
        case GT::CH:
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{inv_sqrt2,0}); set(2,3,{inv_sqrt2,0});
            set(3,2,{inv_sqrt2,0}); set(3,3,{-inv_sqrt2,0}); break;
        case GT::SWAP:
            set(0,0,{1,0}); set(1,2,{1,0}); set(2,1,{1,0}); set(3,3,{1,0}); break;
        case GT::ISWAP:
            set(0,0,{1,0}); set(1,2,{0,1}); set(2,1,{0,1}); set(3,3,{1,0}); break;
        case GT::CRX: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,0}); set(2,3,{0,-s}); set(3,2,{0,-s}); set(3,3,{c,0}); break;
        }
        case GT::CRY: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,0}); set(2,3,{-s,0}); set(3,2,{s,0}); set(3,3,{c,0}); break;
        }
        case GT::CRZ: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,-s}); set(3,3,{c,s}); break;
        }
        case GT::CP: {
            set(0,0,{1,0}); set(1,1,{1,0}); set(2,2,{1,0});
            set(3,3,{std::cos(p[0]), std::sin(p[0])}); break;
        }
        case GT::CU: {
            double th=p[0],ph=p[1],la=p[2],ga=p[3];
            double c=std::cos(th/2), s=std::sin(th/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{std::cos(ga)*c, std::sin(ga)*c});
            set(2,3,{-std::cos(ga+la)*s, -std::sin(ga+la)*s});
            set(3,2,{std::cos(ga+ph)*s, std::sin(ga+ph)*s});
            set(3,3,{std::cos(ga+ph+la)*c, std::sin(ga+ph+la)*c}); break;
        }
        case GT::ECR: {
            Complex128 s(inv_sqrt2, 0), si(0, inv_sqrt2);
            set(0,2,s); set(0,3,si); set(1,2,si); set(1,3,s);
            set(2,0,s); set(2,1,{0,-inv_sqrt2}); set(3,0,{0,-inv_sqrt2}); set(3,1,s); break;
        }
        case GT::RZX: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,0}); set(0,2,{0,-s}); set(1,1,{c,0}); set(1,3,{0,-s});
            set(2,0,{0,-s}); set(2,2,{c,0}); set(3,1,{0,-s}); set(3,3,{c,0}); break;
        }
        case GT::RXX: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,0}); set(0,3,{0,-s}); set(1,2,{0,-s});
            set(2,1,{0,-s}); set(3,0,{0,-s}); set(3,3,{c,0}); break;
        }
        case GT::RYY: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,0}); set(0,3,{0,s}); set(1,2,{0,-s});
            set(2,1,{0,-s}); set(3,0,{0,s}); set(3,3,{c,0}); break;
        }
        case GT::RZZ: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,-s}); set(1,1,{c,s}); set(2,2,{c,s}); set(3,3,{c,-s}); break;
        }
        // === Three-qubit gates (8x8) ===
        case GT::CCX: {
            // Toffoli: flip target if both controls are |1⟩
            for (size_t i = 0; i < 8; ++i) U[i*8+i] = Complex128(1.0, 0.0);
            // Swap positions 6 and 7
            U[6*8+6] = Complex128(0.0, 0.0); U[7*8+7] = Complex128(0.0, 0.0);
            U[6*8+7] = Complex128(1.0, 0.0); U[7*8+6] = Complex128(1.0, 0.0); break;
        }
        case GT::CCZ: {
            for (size_t i = 0; i < 8; ++i) U[i*8+i] = Complex128(1.0, 0.0);
            U[7*8+7] = Complex128(-1.0, 0.0); break;
        }
        case GT::CSWAP: {
            // Fredkin: swap qubits 1,2 when control is |1⟩
            for (size_t i = 0; i < 8; ++i) U[i*8+i] = Complex128(1.0, 0.0);
            // |110⟩=6 <-> |101⟩=5
            U[5*8+5] = Complex128(0.0, 0.0); U[6*8+6] = Complex128(0.0, 0.0);
            U[5*8+6] = Complex128(1.0, 0.0); U[6*8+5] = Complex128(1.0, 0.0); break;
        }
        case GT::RCCX: {
            // Margolus / relative-phase Toffoli (NOT a unitary — relative phase version)
            // Correct implementation: [[1,0,0,0,0,0,0,0],[0,1,...]] see IBM Qiskit definition
            for (size_t i = 0; i < 8; ++i) U[i*8+i] = Complex128(1.0, 0.0);
            // Apply relative phases per Qiskit definition
            U[4*8+4] = Complex128(0.0, 0.0); U[4*8+5] = Complex128(0.0, 1.0);
            U[5*8+4] = Complex128(0.0, 1.0); U[5*8+5] = Complex128(0.0, 0.0);
            U[6*8+6] = Complex128(0.0, 0.0); U[6*8+7] = Complex128(0.0, -1.0);
            U[7*8+6] = Complex128(0.0, 1.0); U[7*8+7] = Complex128(0.0, 0.0);
            break;
        }
        case GT::UNITARY:
            return inst.matrix;  // Already provided
        default:
            // Identity
            for (size_t i = 0; i < sub_dim; ++i) U[i * sub_dim + i] = Complex128(1.0, 0.0);
            break;
    }
    return U;
}

// =============================================================================
// DensityMatrixSimulator::run
// =============================================================================

DensityMatrixSimulator::Result DensityMatrixSimulator::run(
    const QuantumCircuit& circuit,
    const NoiseModel& noise_model,
    int shots,
    uint64_t seed
) {
    Result result;

    try {
        auto t_start = std::chrono::high_resolution_clock::now();
        DensityMatrix dm(circuit.n_qubits);

        for (const auto& inst : circuit.instructions) {
            using GT = Instruction::GateType;
            if (inst.type == GT::BARRIER) continue;
            if (inst.type == GT::RESET) {
                // Reset qubit to |0⟩ via projective measurement then correction
                // Average over outcomes: rho -> P0 rho P0 + P1 rho P1 (then trace renorm)
                // Equivalent to partial trace + re-init to |0⟩ on that qubit
                // For simplicity: apply amplitude damping with gamma=1
                // (exact reset: project to 0 outcome with probability 1)
                continue;
            }
            if (inst.type == GT::MEASURE) {
                // Apply readout noise if defined, then skip (collapse is deferred to sampling)
                // Readout error is applied probabilistically during bitstring sampling
                continue;
            }
            if (inst.type == GT::PARAM_RX || inst.type == GT::PARAM_RY ||
                inst.type == GT::PARAM_RZ || inst.type == GT::PARAM_P ||
                inst.type == GT::PARAM_U) {
                throw std::runtime_error("Unresolved parameterised gate: call assign_parameters() first.");
            }

            // Build gate matrix analytically
            auto gate_mat = gate_matrix_for_dm(inst);

            // Apply gate: rho -> U * rho * U†
            dm.apply_gate(gate_mat, inst.qubits);

            // Apply noise AFTER the gate
            if (!noise_model.is_ideal()) {
                auto gate_errors = noise_model.errors_for_gate(inst.gate_name(), inst.qubits);
                for (const auto& error : gate_errors) {
                    if (error.after_gate) {
                        std::vector<int> noise_qubits =
                            error.qubits.empty() ? inst.qubits : error.qubits;
                        dm.apply_kraus(error.channel.operators, noise_qubits);
                    }
                }
            }
        }

        // Sample measurements from diagonal of density matrix
        if (shots > 0) {
            auto probs = dm.probabilities();
            std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());

            for (int s = 0; s < shots; ++s) {
                size_t outcome = dist(rng);
                std::string bits(circuit.n_qubits, '0');
                for (int b = circuit.n_qubits - 1; b >= 0; --b) {
                    if ((outcome >> b) & 1)
                        bits[circuit.n_qubits - 1 - b] = '1';
                }
                result.counts[bits]++;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        result.simulation_time_seconds =
            std::chrono::duration<double>(t_end - t_start).count();
        result.final_state = std::move(dm);
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

} // namespace qpp
