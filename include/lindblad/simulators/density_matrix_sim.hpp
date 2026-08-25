#pragma once

#include "lindblad/statevector.hpp"
#include "lindblad/validation.hpp"
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

    // Reset in place to |0...0><0...0| (reuses the existing allocation; used
    // for per-shot buffer reuse in the trajectory path).
    void initialize();

    // Initialise from pure state
    static DensityMatrix from_statevector(const Statevector& sv);

    // Properties
    double trace() const;
    double purity() const;  // Tr(rho^2)
    // Checks trace==1 and Hermiticity only. Positive semi-definiteness is NOT verified
    // (full PSD check requires eigendecomposition and is O(4^N)).
    bool is_valid(double atol = 1e-8) const;

    // Gate application: rho -> U.rho.U†
    // validation = policy and tolerance for the unitarity of U.
    void apply_gate(const std::vector<Complex128>& U,
                    const std::vector<int>& qubits,
                    ValidationOptions validation = {});

    // Apply a k-qubit channel given directly as its superoperator, in ONE
    // pass over rho (R.1.17). S is (4^k × 4^k) row-major, external
    // convention (bit b of every sub-index addresses qubits[b], LSB-first,
    // matching KrausChannel / apply_unitary):
    //   S[(r_out·2^k + c_out)·4^k + (r_in·2^k + c_in)]
    //   rho'_block = S · vec(rho_block)  per background pair.
    // For a Kraus channel {K_k}: S[(ro,co),(ri,ci)] = Σ_k K[ro,ri]·conj(K[co,ci]).
    // Trace preservation is checked under `validation`: the condition on a
    // superoperator is Σ_ro S[(ro,ro),(ri,ci)] = δ_ri,ci at every input pair.
    // apply_kraus() builds this superoperator internally and is the
    // preferred entry point when only the operators are at hand.
    void apply_channel_superop(const std::vector<Complex128>& S,
                               const std::vector<int>& qubits,
                               ValidationOptions validation = {});

    // Kraus channel: rho -> sum_k K_k.rho.K_k†
    void apply_kraus(
        const std::vector<std::vector<Complex128>>& kraus_ops,
        const std::vector<int>& qubits,
        ValidationOptions validation = {}
    );

    // Full-register basis permutation: rho -> P.rho.P† where full_perm[a] is
    // the image of basis index a (a bijection of [0, dim)). Applied by row/
    // column relabel, no dense matrix (used for MCX and PERMUTATION).
    void apply_permutation(const std::vector<int>& full_perm);

    // Diagonal multi-controlled phase: rho -> D.rho.D† where D multiplies
    // basis index a by exp(i*lambda) iff (a & mask) == mask (used for MCP).
    void apply_mcp_phase(size_t mask, double lambda);

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
