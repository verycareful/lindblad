#pragma once

#include "lindblad/types.hpp"

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {

class QuantumCircuit;
class DensityMatrix;

// =============================================================================
// StabilizerState — Tableau representation for Clifford circuits
// =============================================================================

class StabilizerState {
public:
    int n_qubits;
    // Tableau: 2N rows × (2N+1) columns binary matrix
    // Rows 0..N-1: destabilizers
    // Rows N..2N-1: stabilizers
    // Columns 0..N-1: X part
    // Columns N..2N-1: Z part
    // Column 2N: phase bit (0 = +1, 1 = -1)
    std::vector<std::vector<bool>> tableau;

public:
    explicit StabilizerState(int n_qubits);

    // Clifford gates
    void apply_h(int qubit);
    void apply_s(int qubit);
    void apply_sdg(int qubit);
    void apply_cx(int control, int target);
    void apply_x(int qubit);
    void apply_y(int qubit);
    void apply_z(int qubit);

    // Measurement — returns 0 or 1.
    // rng is only consumed when the outcome is random (stabilizer has X-support on qubit).
    int measure(int qubit, bool random, std::mt19937_64& rng);

    // Expectation of Pauli string (+1, -1, or 0)
    int expectation_pauli(const std::string& pauli) const;

private:
    void rowmult(int dest, int src);
};

// =============================================================================
// CliffordSimulator
// =============================================================================

class CliffordSimulator {
public:
    static bool is_clifford(const QuantumCircuit& circuit);

    struct Result {
        StabilizerState final_state;
        std::unordered_map<std::string, int> counts;

        Result(int n) : final_state(n) {}
        Result(Result&&) = default;
        Result& operator=(Result&&) = default;
    };

    Result run(const QuantumCircuit& circuit, int shots = 1024,
               uint64_t seed = 0);
};

} // namespace lindblad
