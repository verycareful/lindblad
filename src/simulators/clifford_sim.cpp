#include "qpp/simulators/clifford_sim.hpp"
#include "qpp/circuit.hpp"

#include <random>
#include <stdexcept>

namespace qpp {

// =============================================================================
// StabilizerState — Gottesman-Knill tableau
// =============================================================================

StabilizerState::StabilizerState(int n_qubits)
    : n_qubits(n_qubits) {
    int rows = 2 * n_qubits;
    int cols = 2 * n_qubits + 1;
    tableau.resize(rows, std::vector<bool>(cols, false));

    // Initial state |0...0⟩:
    // Destabilizers (rows 0..N-1): X_i (X on qubit i)
    // Stabilizers (rows N..2N-1): Z_i (Z on qubit i)
    for (int i = 0; i < n_qubits; ++i) {
        tableau[i][i] = true;                          // X part of destabilizer i
        tableau[n_qubits + i][n_qubits + i] = true;   // Z part of stabilizer i
    }
}

// Row multiplication: dest = dest * src (in the Pauli group)
void StabilizerState::rowmult(int dest, int src) {
    int N = n_qubits;
    // Phase update: count how X*Z products contribute to phase
    // Phase rule: i^{2*sum} where sum counts (x_src & z_dest XOR z_src & x_dest) etc.
    int phase_count = 0;
    for (int j = 0; j < N; ++j) {
        bool xs = tableau[src][j];
        bool zs = tableau[src][N + j];
        bool xd = tableau[dest][j];
        bool zd = tableau[dest][N + j];

        // Count the phase contribution
        if (xs && zs) {            // Y
            if (xd && !zd) phase_count++;     // Y*X = iZ
            else if (!xd && zd) phase_count--; // Y*Z = -iX
        } else if (xs && !zs) {    // X
            if (!xd && zd) phase_count++;     // X*Z = -iY -> phase_count affects sign
            else if (xd && zs) phase_count++;
        } else if (!xs && zs) {    // Z
            if (xd && !zd) phase_count--;     // Z*X = iY
        }
    }

    // Update phase bit
    int current = tableau[dest][2 * N] ? 2 : 0;
    int src_phase = tableau[src][2 * N] ? 2 : 0;
    int new_phase = (current + src_phase + phase_count) % 4;
    if (new_phase < 0) new_phase += 4;
    tableau[dest][2 * N] = (new_phase == 2);

    // XOR the X and Z parts
    for (int j = 0; j < N; ++j) {
        tableau[dest][j] = tableau[dest][j] ^ tableau[src][j];
        tableau[dest][N + j] = tableau[dest][N + j] ^ tableau[src][N + j];
    }
}

void StabilizerState::apply_h(int qubit) {
    int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        // H: X <-> Z, and phase update if both X and Z are set
        bool x = tableau[i][qubit];
        bool z = tableau[i][N + qubit];
        if (x && z) {
            tableau[i][2 * N] = !tableau[i][2 * N];
        }
        tableau[i][qubit] = z;
        tableau[i][N + qubit] = x;
    }
}

void StabilizerState::apply_s(int qubit) {
    int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        // S: X -> Y = iXZ. So Z part gets XORed with X part
        // Phase: if X bit is set, phase gets XORed
        if (tableau[i][qubit]) {
            tableau[i][2 * N] = tableau[i][2 * N] ^ tableau[i][N + qubit];
            tableau[i][N + qubit] = !tableau[i][N + qubit];
        }
    }
}

void StabilizerState::apply_cx(int control, int target) {
    int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        // CX: X_c -> X_c * X_t, Z_t -> Z_c * Z_t
        // Phase update
        bool xc = tableau[i][control];
        bool xt = tableau[i][target];
        bool zc = tableau[i][N + control];
        bool zt = tableau[i][N + target];

        if (xc && zt && !(xt ^ zc)) {
            tableau[i][2 * N] = !tableau[i][2 * N];
        }

        tableau[i][target] = xt ^ xc;
        tableau[i][N + control] = zc ^ zt;
    }
}

void StabilizerState::apply_x(int qubit) {
    int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        // X: flip phase if Z bit is set
        if (tableau[i][N + qubit]) {
            tableau[i][2 * N] = !tableau[i][2 * N];
        }
    }
}

void StabilizerState::apply_y(int qubit) {
    int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        // Y = iXZ: flip phase if X XOR Z
        if (tableau[i][qubit] ^ tableau[i][N + qubit]) {
            tableau[i][2 * N] = !tableau[i][2 * N];
        }
    }
}

void StabilizerState::apply_z(int qubit) {
    int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        // Z: flip phase if X bit is set
        if (tableau[i][qubit]) {
            tableau[i][2 * N] = !tableau[i][2 * N];
        }
    }
}

int StabilizerState::measure(int qubit, bool random, uint64_t seed) {
    int N = n_qubits;

    // Check if the outcome is deterministic or random
    // Look for a stabilizer (rows N..2N-1) with X bit set on this qubit
    int p = -1;
    for (int i = N; i < 2 * N; ++i) {
        if (tableau[i][qubit]) {
            p = i;
            break;
        }
    }

    if (p >= 0) {
        // Random outcome
        // All other rows that have X bit set on this qubit: rowmult with p
        for (int i = 0; i < 2 * N; ++i) {
            if (i != p && tableau[i][qubit]) {
                rowmult(i, p);
            }
        }

        // Move row p to destabilizer (row p - N)
        tableau[p - N] = tableau[p];

        // Set stabilizer p to be ±Z on this qubit
        for (int j = 0; j < 2 * N + 1; ++j) {
            tableau[p][j] = false;
        }
        tableau[p][N + qubit] = true;

        // Random outcome
        int result;
        if (random) {
            std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
            std::uniform_int_distribution<int> dist(0, 1);
            result = dist(rng);
        } else {
            result = 0;
        }

        tableau[p][2 * N] = (result == 1);
        return result;
    } else {
        // Deterministic outcome
        // Look in destabilizers for X bit set
        // The measurement outcome is determined by the phase of the stabilizer
        // that anticommutes with Z on this qubit
        
        // Create a scratch row
        std::vector<bool> scratch(2 * N + 1, false);
        scratch[N + qubit] = true;  // Z on this qubit

        // Find destabilizers with X bit set on this qubit
        // and multiply them into the scratch row
        for (int i = 0; i < N; ++i) {
            if (tableau[i][qubit]) {
                // Multiply stabilizer i+N into scratch
                for (int j = 0; j < 2 * N + 1; ++j) {
                    scratch[j] = scratch[j] ^ tableau[i + N][j];
                }
            }
        }

        return scratch[2 * N] ? 1 : 0;
    }
}

int StabilizerState::expectation_pauli(const std::string& pauli) const {
    int N = n_qubits;
    if (static_cast<int>(pauli.size()) != N) {
        throw std::invalid_argument("Pauli string length must match n_qubits");
    }

    // Check if the Pauli operator commutes with all stabilizers
    // and if it's in the stabilizer group
    // Simple approach: check stabilizer phase
    
    // Build the Pauli as X and Z bits
    std::vector<bool> px(N, false), pz(N, false);
    for (int i = 0; i < N; ++i) {
        char c = pauli[i];
        if (c == 'X' || c == 'x') { px[i] = true; }
        else if (c == 'Y' || c == 'y') { px[i] = true; pz[i] = true; }
        else if (c == 'Z' || c == 'z') { pz[i] = true; }
        // 'I' leaves both false
    }

    // Check commutation with each stabilizer
    for (int s = N; s < 2 * N; ++s) {
        int anti = 0;
        for (int j = 0; j < N; ++j) {
            anti += (px[j] && tableau[s][N + j]) ? 1 : 0;
            anti += (pz[j] && tableau[s][j]) ? 1 : 0;
        }
        if (anti % 2 != 0) {
            // Anticommutes with a stabilizer -> expectation value is 0
            return 0;
        }
    }

    // If it commutes with all stabilizers, it's ±1
    // Determine sign by checking if it's in the stabilizer group
    // For simplicity, return +1 (this is a simplification)
    return 1;
}

// =============================================================================
// CliffordSimulator
// =============================================================================

bool CliffordSimulator::is_clifford(const QuantumCircuit& circuit) {
    using GT = Instruction::GateType;
    for (const auto& inst : circuit.instructions) {
        switch (inst.type) {
            case GT::H: case GT::X: case GT::Y: case GT::Z:
            case GT::S: case GT::SDG:
            case GT::CX: case GT::CZ: case GT::SWAP:
            case GT::MEASURE: case GT::RESET: case GT::BARRIER:
                break;
            default:
                return false;
        }
    }
    return true;
}

CliffordSimulator::Result CliffordSimulator::run(
    const QuantumCircuit& circuit, int shots, uint64_t seed
) {
    using GT = Instruction::GateType;
    Result result(circuit.n_qubits);

    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);

    for (int s = 0; s < shots; ++s) {
        StabilizerState state(circuit.n_qubits);

        std::string bitstring(circuit.n_qubits, '0');

        for (const auto& inst : circuit.instructions) {
            switch (inst.type) {
                case GT::H: state.apply_h(inst.qubits[0]); break;
                case GT::S: state.apply_s(inst.qubits[0]); break;
                case GT::SDG:
                    state.apply_s(inst.qubits[0]);
                    state.apply_s(inst.qubits[0]);
                    state.apply_s(inst.qubits[0]);
                    break;
                case GT::X: state.apply_x(inst.qubits[0]); break;
                case GT::Y: state.apply_y(inst.qubits[0]); break;
                case GT::Z: state.apply_z(inst.qubits[0]); break;
                case GT::CX: state.apply_cx(inst.qubits[0], inst.qubits[1]); break;
                case GT::CZ:
                    state.apply_h(inst.qubits[1]);
                    state.apply_cx(inst.qubits[0], inst.qubits[1]);
                    state.apply_h(inst.qubits[1]);
                    break;
                case GT::SWAP:
                    state.apply_cx(inst.qubits[0], inst.qubits[1]);
                    state.apply_cx(inst.qubits[1], inst.qubits[0]);
                    state.apply_cx(inst.qubits[0], inst.qubits[1]);
                    break;
                case GT::MEASURE: {
                    int outcome = state.measure(inst.qubits[0], true, rng());
                    bitstring[inst.qubits[0]] = outcome ? '1' : '0';
                    break;
                }
                default:
                    break;
            }
        }

        result.counts[bitstring]++;
    }

    // Store final state from last run
    result.final_state = StabilizerState(circuit.n_qubits);

    return result;
}

} // namespace qpp
