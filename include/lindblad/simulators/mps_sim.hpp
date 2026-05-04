#pragma once

#include "lindblad/types.hpp"

#include <array>
#include <random>
#include <unordered_map>
#include <vector>

namespace lindblad {

class Statevector;
class QuantumCircuit;

// =============================================================================
// MPSTensor — tensor for one qubit site
// =============================================================================

struct MPSTensor {
    // Shape: (bond_left, physical_dim=2, bond_right)
    int bond_left;
    int bond_right;
    // data layout: data[left * 2 * bond_right + phys * bond_right + right]
    std::vector<Complex128> data;

    MPSTensor() : bond_left(1), bond_right(1), data(2, Complex128(0.0, 0.0)) {}
    MPSTensor(int bl, int br) : bond_left(bl), bond_right(br),
        data(bl * 2 * br, Complex128(0.0, 0.0)) {}

    Complex128& operator()(int left, int phys, int right) {
        return data[left * 2 * bond_right + phys * bond_right + right];
    }
    const Complex128& operator()(int left, int phys, int right) const {
        return data[left * 2 * bond_right + phys * bond_right + right];
    }
};

// =============================================================================
// MPSState — Matrix Product State
// =============================================================================

class MPSState {
public:
    int n_qubits;
    int max_bond_dim;    // chi — controls accuracy vs memory tradeoff
    double cutoff;       // singular value truncation threshold
    std::vector<MPSTensor> tensors;

public:
    MPSState(int n_qubits, int max_bond_dim = 64, double cutoff = 1e-12);

    // Gate application via SVD
    void apply_single_qubit_gate(
        const std::array<Complex128, 4>& U, int qubit
    );
    void apply_two_qubit_gate(
        const std::array<Complex128, 16>& U, int q1, int q2
    );

    // Truncation info
    double truncation_error() const { return total_truncation_error; }
    int current_max_bond_dim() const;

    // Measurements and expectation values
    std::vector<double> probabilities_single(int qubit) const;

    // Sequential measurement: sample a full bitstring respecting correlations.
    // Measures qubit 0, conditions on outcome, propagates boundary, repeats.
    // O(N * chi^3) per shot. Modifies internal state (projects measured qubits).
    std::string measure_sequential(std::mt19937_64& rng);

    // Convert to exact statevector (expensive, for small N only)
    Statevector to_statevector() const;

private:
    double total_truncation_error = 0.0;

    // SVD helper
    void svd_truncate(
        const std::vector<Complex128>& matrix,
        int rows, int cols,
        std::vector<Complex128>& U,
        std::vector<double>& S,
        std::vector<Complex128>& Vt,
        int& new_rank
    );

    // Adjacent two-qubit gate application (internal)
    void apply_two_qubit_gate_adjacent(
        const std::array<Complex128, 16>& U, int q1
    );

    // Adjacent SWAP gate (internal)
    void apply_swap_adjacent(int q);
};

// =============================================================================
// MPSSimulator
// =============================================================================

class MPSSimulator {
public:
    struct Result {
        MPSState final_state;
        std::unordered_map<std::string, int> counts;
        double simulation_time_seconds = 0.0;

        Result(int n) : final_state(n) {}
        Result(Result&&) = default;
        Result& operator=(Result&&) = default;
    };

    Result run(const QuantumCircuit& circuit, int max_bond_dim = 64,
               int shots = 1024, uint64_t seed = 0);
};

} // namespace lindblad
