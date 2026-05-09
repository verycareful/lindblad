#pragma once

#include "lindblad/types.hpp"

#include <cstdint>
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
//
// Aaronson-Gottesman (2004) tableau with word-packed row storage.
// Rows 0..N-1: destabilizers; rows N..2N-1: stabilizers.
// Columns 0..N-1: X part; columns N..2N-1: Z part.
// Phase bit stored separately per row in ph[] (0=+1, 1=-1).
//
// Each row's X/Z bits are packed into ceil(2N/64) uint64_t words.
// rowmult XOR uses word-level operations: ~64× faster than vector<bool> at N>100.

class StabilizerState {
public:
    int n_qubits;

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
    int              wpr;      // words per row: ceil(2*n_qubits / 64)
    int              num_rows; // normally 2*n_qubits; +1 during scratch measurement
    std::vector<uint64_t> tab; // flat row-major X/Z bits; row i at [i*wpr, (i+1)*wpr)
    std::vector<uint8_t>  ph;  // phase bit per row (0=+1, 1=-1)

    bool get_xz(int row, int col) const;
    void set_xz(int row, int col, bool v);
    void flip_xz(int row, int col);
    void xor_row(int dest, int src);  // word-level XOR of X/Z bits
    void copy_row(int dest, int src);
    void zero_row(int row);
    void push_scratch();
    void pop_scratch();

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
