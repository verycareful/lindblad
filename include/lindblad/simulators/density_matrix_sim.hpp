#pragma once

#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {

// Forward declarations
class QuantumCircuit;
class NoiseModel;
class SparsePauliOp;

// =============================================================================
// DensityMatrix — density operator ρ for mixed states
// =============================================================================

class DensityMatrix {
public:
    int n_qubits;
    size_t dim;        // 2^n_qubits
    // density_matrix[i*dim + j] = rho_{ij}
    std::vector<Complex128> data;  // row-major, dim × dim

public:
    DensityMatrix();
    explicit DensityMatrix(int n_qubits);

    // Initialise from pure state
    static DensityMatrix from_statevector(const Statevector& sv);

    // Properties
    double trace() const;
    double purity() const;  // Tr(rho^2)
    bool is_valid(double atol = 1e-8) const;

    // Gate application: rho -> U.rho.U†
    void apply_gate(const std::vector<Complex128>& U,
                    const std::vector<int>& qubits);

    // Kraus channel: rho -> sum_k K_k.rho.K_k†
    void apply_kraus(
        const std::vector<std::vector<Complex128>>& kraus_ops,
        const std::vector<int>& qubits
    );

    // Measurement probabilities
    std::vector<double> probabilities() const;
    double expectation_value(const std::vector<Complex128>& hermitian_op) const;
    double expectation_value_sparse(const SparsePauliOp& hamiltonian) const;

    // Element access
    Complex128& operator()(size_t i, size_t j) { return data[i * dim + j]; }
    const Complex128& operator()(size_t i, size_t j) const { return data[i * dim + j]; }
};

// =============================================================================
// DensityMatrixSimulator
// =============================================================================

class DensityMatrixSimulator {
public:
    struct Result {
        DensityMatrix final_state;
        std::unordered_map<std::string, int> counts;
        double simulation_time_seconds = 0.0;
        bool success = true;
        std::string error_message;
    };

    Result run(
        const QuantumCircuit& circuit,
        const NoiseModel& noise_model,
        int shots = 1024,
        uint64_t seed = 0
    );
};

} // namespace lindblad
