#include "qpp/simulators/density_matrix_sim.hpp"
#include "qpp/circuit.hpp"
#include "qpp/noise.hpp"

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
    data[0] = Complex128(1.0, 0.0);  // |0><0|
}

DensityMatrix DensityMatrix::from_statevector(const Statevector& sv) {
    DensityMatrix dm(sv.n_qubits);
    // rho = |psi><psi|
    // rho_{ij} = psi_i * conj(psi_j)
    for (size_t i = 0; i < dm.dim; ++i) {
        for (size_t j = 0; j < dm.dim; ++j) {
            Complex128 psi_i(sv.real_parts[i], sv.imag_parts[i]);
            Complex128 psi_j_conj(sv.real_parts[j], -sv.imag_parts[j]);
            dm(i, j) = psi_i * psi_j_conj;
        }
    }
    return dm;
}

double DensityMatrix::trace() const {
    double tr = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        tr += data[i * dim + i].real;
    }
    return tr;
}

double DensityMatrix::purity() const {
    // Tr(rho^2)
    double pur = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t k = 0; k < dim; ++k) {
            Complex128 prod = data[i * dim + k] * data[k * dim + i];
            pur += prod.real;
        }
    }
    return pur;
}

bool DensityMatrix::is_valid(double atol) const {
    // Check trace = 1
    if (std::abs(trace() - 1.0) > atol) return false;

    // Check Hermitian: rho_{ij} = conj(rho_{ji})
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = i + 1; j < dim; ++j) {
            auto diff = data[i * dim + j] - data[j * dim + i].conj();
            if (diff.norm_sq() > atol * atol) return false;
        }
    }

    return true;
}

void DensityMatrix::apply_gate(const std::vector<Complex128>& U,
                                const std::vector<int>& qubits) {
    // For simplicity, convert to full-space unitary and apply
    // rho -> U * rho * U†
    // This is O(dim^3) — acceptable for small systems
    
    int k = static_cast<int>(qubits.size());
    size_t sub_dim = 1ULL << k;

    if (U.size() != sub_dim * sub_dim) {
        throw std::invalid_argument("Gate matrix size mismatch");
    }

    // For single-qubit gates on small systems, apply directly
    // rho -> U * rho * U†
    // Build full-system unitary would be expensive, so instead we
    // work on the density matrix indices directly

    // Create a temporary copy
    std::vector<Complex128> new_data = data;

    // Apply U on left: new_rho = U * rho
    // Then apply U† on right: new_rho = new_rho * U†

    // For multi-qubit gates, this requires sophisticated index manipulation
    // For now, use the straightforward approach for small systems
    
    // Build full-space unitary
    std::vector<Complex128> full_U(dim * dim, Complex128(0.0, 0.0));
    
    // Identity on non-target qubits, U on target qubits
    std::vector<size_t> target_masks(k);
    for (int i = 0; i < k; ++i) {
        target_masks[i] = 1ULL << qubits[i];
    }

    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            // Check if i and j differ only on target qubits
            size_t non_target_mask = 0;
            for (int b = 0; b < n_qubits; ++b) {
                bool is_target = false;
                for (int ti = 0; ti < k; ++ti) {
                    if (qubits[ti] == b) { is_target = true; break; }
                }
                if (!is_target) non_target_mask |= (1ULL << b);
            }

            if ((i & non_target_mask) != (j & non_target_mask)) continue;

            // Extract target qubit indices
            size_t ti = 0, tj = 0;
            for (int q = 0; q < k; ++q) {
                if ((i >> qubits[q]) & 1) ti |= (1ULL << q);
                if ((j >> qubits[q]) & 1) tj |= (1ULL << q);
            }

            full_U[i * dim + j] = U[ti * sub_dim + tj];
        }
    }

    // rho_new = full_U * rho * full_U†
    // Step 1: temp = full_U * rho
    std::vector<Complex128> temp(dim * dim, Complex128(0.0, 0.0));
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            Complex128 sum(0.0, 0.0);
            for (size_t m = 0; m < dim; ++m) {
                sum += full_U[i * dim + m] * data[m * dim + j];
            }
            temp[i * dim + j] = sum;
        }
    }

    // Step 2: new_data = temp * full_U†
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            Complex128 sum(0.0, 0.0);
            for (size_t m = 0; m < dim; ++m) {
                sum += temp[i * dim + m] * full_U[j * dim + m].conj();
            }
            new_data[i * dim + j] = sum;
        }
    }

    data = new_data;
}

void DensityMatrix::apply_kraus(
    const std::vector<std::vector<Complex128>>& kraus_ops,
    const std::vector<int>& qubits
) {
    // rho -> sum_k K_k * rho * K_k†
    std::vector<Complex128> new_data(dim * dim, Complex128(0.0, 0.0));

    for (const auto& K : kraus_ops) {
        // Save current state, apply K as a gate, accumulate
        DensityMatrix temp = *this;
        temp.apply_gate(K, qubits);

        for (size_t i = 0; i < dim * dim; ++i) {
            new_data[i] += temp.data[i];
        }
    }

    data = new_data;
}

std::vector<double> DensityMatrix::probabilities() const {
    std::vector<double> probs(dim);
    for (size_t i = 0; i < dim; ++i) {
        probs[i] = data[i * dim + i].real;
    }
    return probs;
}

double DensityMatrix::expectation_value(const std::vector<Complex128>& hermitian_op) const {
    // Tr(rho * O)
    double result = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            Complex128 prod = hermitian_op[i * dim + j] * data[j * dim + i];
            result += prod.real;
        }
    }
    return result;
}

// =============================================================================
// DensityMatrixSimulator
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
            if (inst.type == Instruction::GateType::BARRIER) continue;
            if (inst.type == Instruction::GateType::MEASURE) continue;
            if (inst.type == Instruction::GateType::RESET) continue;

            // Build the gate matrix for density matrix application
            // For simple gates, construct the 2x2 or 4x4 matrix
            // and apply via dm.apply_gate()

            // For now, apply ideal gate then noise
            // Build a statevector sim to get the gate matrix
            int k = inst.num_qubits();
            size_t sub_dim = 1ULL << k;
            std::vector<Complex128> gate_matrix(sub_dim * sub_dim);

            // Construct gate matrix by applying gate to basis states
            for (size_t col = 0; col < sub_dim; ++col) {
                Statevector basis(k);
                basis.initialize_basis(col);

                // Apply the instruction to this small statevector
                // Map qubits to [0..k-1]
                Instruction local_inst = inst;
                for (int qi = 0; qi < k; ++qi) {
                    local_inst.qubits[qi] = qi;
                }

                // Use the statevector sim to apply
                StatevectorSimulator sv_sim;
                QuantumCircuit mini_circ(k);
                mini_circ.instructions.push_back(local_inst);
                sv_sim.run(mini_circ);

                // Actually, let's build the matrix directly by applying to basis states
                Statevector sv(k);
                sv.initialize_basis(col);

                // Apply gate directly using gate functions
                // This is a small helper — apply the local instruction
                using GT = Instruction::GateType;
                const auto& p = local_inst.params;
                switch (local_inst.type) {
                    case GT::H: gates::apply_h(sv, 0); break;
                    case GT::X: gates::apply_x(sv, 0); break;
                    case GT::Y: gates::apply_y(sv, 0); break;
                    case GT::Z: gates::apply_z(sv, 0); break;
                    case GT::S: gates::apply_s(sv, 0); break;
                    case GT::SDG: gates::apply_sdg(sv, 0); break;
                    case GT::T: gates::apply_t(sv, 0); break;
                    case GT::TDG: gates::apply_tdg(sv, 0); break;
                    case GT::SX: gates::apply_sx(sv, 0); break;
                    case GT::SXDG: gates::apply_sxdg(sv, 0); break;
                    case GT::RX: gates::apply_rx(sv, 0, p[0]); break;
                    case GT::RY: gates::apply_ry(sv, 0, p[0]); break;
                    case GT::RZ: gates::apply_rz(sv, 0, p[0]); break;
                    case GT::P: gates::apply_p(sv, 0, p[0]); break;
                    case GT::U: gates::apply_u(sv, 0, p[0], p[1], p[2]); break;
                    case GT::U1: gates::apply_u1(sv, 0, p[0]); break;
                    case GT::U2: gates::apply_u2(sv, 0, p[0], p[1]); break;
                    case GT::U3: gates::apply_u3(sv, 0, p[0], p[1], p[2]); break;
                    case GT::CX: gates::apply_cx(sv, 0, 1); break;
                    case GT::CY: gates::apply_cy(sv, 0, 1); break;
                    case GT::CZ: gates::apply_cz(sv, 0, 1); break;
                    case GT::CH: gates::apply_ch(sv, 0, 1); break;
                    case GT::SWAP: gates::apply_swap(sv, 0, 1); break;
                    case GT::ISWAP: gates::apply_iswap(sv, 0, 1); break;
                    case GT::CRX: gates::apply_crx(sv, 0, 1, p[0]); break;
                    case GT::CRY: gates::apply_cry(sv, 0, 1, p[0]); break;
                    case GT::CRZ: gates::apply_crz(sv, 0, 1, p[0]); break;
                    case GT::CP: gates::apply_cp(sv, 0, 1, p[0]); break;
                    case GT::CU: gates::apply_cu(sv, 0, 1, p[0], p[1], p[2], p[3]); break;
                    case GT::ECR: gates::apply_ecr(sv, 0, 1); break;
                    case GT::RZX: gates::apply_rzx(sv, 0, 1, p[0]); break;
                    case GT::RXX: gates::apply_rxx(sv, 0, 1, p[0]); break;
                    case GT::RYY: gates::apply_ryy(sv, 0, 1, p[0]); break;
                    case GT::RZZ: gates::apply_rzz(sv, 0, 1, p[0]); break;
                    case GT::CCX: gates::apply_ccx(sv, 0, 1, 2); break;
                    case GT::CCZ: gates::apply_ccz(sv, 0, 1, 2); break;
                    case GT::CSWAP: gates::apply_cswap(sv, 0, 1, 2); break;
                    case GT::RCCX: gates::apply_rccx(sv, 0, 1, 2); break;
                    case GT::UNITARY: gates::apply_unitary(sv, local_inst.qubits, local_inst.matrix); break;
                    default: break;
                }

                for (size_t row = 0; row < sub_dim; ++row) {
                    gate_matrix[row * sub_dim + col] = sv.amplitude(row);
                }
            }

            dm.apply_gate(gate_matrix, inst.qubits);

            // Apply noise (if any errors defined for this gate)
            auto gate_errors = noise_model.errors_for_gate(inst.gate_name(), inst.qubits);
            for (const auto& error : gate_errors) {
                if (error.after_gate) {
                    std::vector<int> noise_qubits = error.qubits.empty() ? inst.qubits : error.qubits;
                    dm.apply_kraus(error.channel.operators, noise_qubits);
                }
            }
        }

        // Sample if requested
        if (shots > 0) {
            auto probs = dm.probabilities();
            std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());

            for (int s = 0; s < shots; ++s) {
                size_t outcome = dist(rng);
                std::string bits(circuit.n_qubits, '0');
                for (int b = 0; b < circuit.n_qubits; ++b) {
                    if ((outcome >> (circuit.n_qubits - 1 - b)) & 1) bits[b] = '1';
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
