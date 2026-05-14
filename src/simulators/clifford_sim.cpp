#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/circuit.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace lindblad {

// =============================================================================
// StabilizerState — word-packed Gottesman-Knill tableau
// =============================================================================

// ----- inline bit helpers -----

inline bool StabilizerState::get_xz(int row, int col) const {
    return (tab[row * wpr + col / 64] >> (col % 64)) & 1;
}

inline void StabilizerState::set_xz(int row, int col, bool v) {
    uint64_t& w = tab[row * wpr + col / 64];
    const uint64_t mask = 1ULL << (col % 64);
    w = v ? (w | mask) : (w & ~mask);
}

inline void StabilizerState::flip_xz(int row, int col) {
    tab[row * wpr + col / 64] ^= (1ULL << (col % 64));
}

// Word-level XOR of the X/Z bits of row src into row dest.
// This is the hot path in rowmult: ~64× faster than element-by-element.
inline void StabilizerState::xor_row(int dest, int src) {
    const int bd = dest * wpr, bs = src * wpr;
    for (int w = 0; w < wpr; ++w) tab[bd + w] ^= tab[bs + w];
}

inline void StabilizerState::copy_row(int dest, int src) {
    const int bd = dest * wpr, bs = src * wpr;
    for (int w = 0; w < wpr; ++w) tab[bd + w] = tab[bs + w];
    ph[dest] = ph[src];
}

inline void StabilizerState::zero_row(int row) {
    const int b = row * wpr;
    for (int w = 0; w < wpr; ++w) tab[b + w] = 0;
    ph[row] = 0;
}

// Append a zeroed scratch row (used by deterministic measurement path).
inline void StabilizerState::push_scratch() {
    tab.insert(tab.end(), wpr, 0ULL);
    ph.push_back(0);
    ++num_rows;
}

inline void StabilizerState::pop_scratch() {
    tab.resize(tab.size() - wpr);
    ph.pop_back();
    --num_rows;
}

// ----- constructor -----

StabilizerState::StabilizerState(int n_qubits)
    : n_qubits(n_qubits),
      wpr((2 * n_qubits + 63) / 64),
      num_rows(2 * n_qubits),
      tab(static_cast<size_t>(2 * n_qubits) * static_cast<size_t>(wpr), 0ULL),
      ph(2 * n_qubits, 0)
{
    // Reserve space for the scratch row appended during deterministic measurement.
    tab.reserve(static_cast<size_t>(2 * n_qubits + 1) * static_cast<size_t>(wpr));
    ph.reserve(2 * n_qubits + 1);

    // Initial state |0...0⟩:
    // Destabilizers (rows 0..N-1): X_i
    // Stabilizers (rows N..2N-1): Z_i
    for (int i = 0; i < n_qubits; ++i) {
        set_xz(i, i, true);
        set_xz(n_qubits + i, n_qubits + i, true);
    }
}

// ----- rowmult -----
// dest = dest * src in the Pauli group.
// Phase per Aaronson-Gottesman 2004, Table 1: g(xd,zd, xs,zs).
// dest is the LEFT factor; outer condition dispatches on (xd,zd).
// After phase update, word-level XOR merges X/Z bits in O(N/64) steps.
void StabilizerState::rowmult(int dest, int src) {
    const int N = n_qubits;
    int phase_count = 0;
    for (int j = 0; j < N; ++j) {
        const bool xs = get_xz(src,  j);
        const bool zs = get_xz(src,  N + j);
        const bool xd = get_xz(dest, j);
        const bool zd = get_xz(dest, N + j);

        if (!xd && !zd) {
            // dest=I: no phase contribution
        } else if (xd && !zd) {
            // dest=X
            if      ( xs &&  zs) phase_count++;  // X*Y = +iZ
            else if (!xs &&  zs) phase_count--;  // X*Z = -iY
        } else if (!xd && zd) {
            // dest=Z
            if      ( xs && !zs) phase_count++;  // Z*X = +iY
            else if ( xs &&  zs) phase_count--;  // Z*Y = -iX
        } else {
            // dest=Y
            if      ( xs && !zs) phase_count--;  // Y*X = -iZ
            else if (!xs &&  zs) phase_count++;  // Y*Z = +iX
        }
    }

    const int cur = ph[dest] ? 2 : 0;
    const int sp  = ph[src]  ? 2 : 0;
    const int np  = ((cur + sp + phase_count) % 4 + 4) % 4;
    ph[dest] = (np == 2) ? 1 : 0;

    xor_row(dest, src);
}

// ----- Clifford gate applications -----

void StabilizerState::apply_h(int qubit) {
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        const bool x = get_xz(i, qubit);
        const bool z = get_xz(i, N + qubit);
        if (x && z) ph[i] ^= 1;
        set_xz(i, qubit,     z);
        set_xz(i, N + qubit, x);
    }
}

void StabilizerState::apply_s(int qubit) {
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, qubit)) {
            ph[i] ^= (uint8_t)get_xz(i, N + qubit);
            flip_xz(i, N + qubit);
        }
    }
}

void StabilizerState::apply_sdg(int qubit) {
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, qubit)) {
            flip_xz(i, N + qubit);
            ph[i] ^= (uint8_t)get_xz(i, N + qubit);
        }
    }
}

void StabilizerState::apply_cx(int control, int target) {
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        const bool xc = get_xz(i, control);
        const bool xt = get_xz(i, target);
        const bool zc = get_xz(i, N + control);
        const bool zt = get_xz(i, N + target);
        if (xc && zt && !(xt ^ zc)) ph[i] ^= 1;
        if (xc) flip_xz(i, target);       // X_target ^= X_control
        if (zt) flip_xz(i, N + control);  // Z_control ^= Z_target
    }
}

void StabilizerState::apply_x(int qubit) {
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, N + qubit)) ph[i] ^= 1;
    }
}

void StabilizerState::apply_y(int qubit) {
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, qubit) ^ get_xz(i, N + qubit)) ph[i] ^= 1;
    }
}

void StabilizerState::apply_z(int qubit) {
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, qubit)) ph[i] ^= 1;
    }
}

// ----- measurement -----

int StabilizerState::measure(int qubit, bool random, std::mt19937_64& rng) {
    const int N = n_qubits;

    // Find a stabilizer (rows N..2N-1) with X bit set on this qubit.
    int p = -1;
    for (int i = N; i < 2 * N; ++i) {
        if (get_xz(i, qubit)) { p = i; break; }
    }

    if (p >= 0) {
        // Random outcome.
        // Eliminate X bit on this qubit from all other rows via rowmult with row p.
        for (int i = 0; i < 2 * N; ++i) {
            if (i != p && get_xz(i, qubit)) rowmult(i, p);
        }

        // Move stabilizer p to destabilizer slot p-N.
        copy_row(p - N, p);

        // Reset stabilizer p to ±Z on this qubit.
        zero_row(p);
        set_xz(p, N + qubit, true);

        int result = 0;
        if (random) {
            std::uniform_int_distribution<int> dist(0, 1);
            result = dist(rng);
        }
        ph[p] = (uint8_t)result;
        return result;
    } else {
        // Deterministic outcome.
        // Use a scratch row to accumulate the product of destabilizers.
        push_scratch();
        const int scratch = num_rows - 1;
        zero_row(scratch);
        set_xz(scratch, N + qubit, true);  // start with Z on this qubit

        for (int i = 0; i < N; ++i) {
            if (get_xz(i, qubit)) rowmult(scratch, i + N);
        }

        const int outcome = ph[scratch] ? 1 : 0;
        pop_scratch();
        return outcome;
    }
}

// ----- expectation_pauli -----

int StabilizerState::expectation_pauli(const std::string& pauli) const {
    const int N = n_qubits;
    if (static_cast<int>(pauli.size()) != N) {
        throw std::invalid_argument("Pauli string length must match n_qubits");
    }

    // Build the target Pauli's X and Z bits.
    // Phase exponent mod 4: 0=+1, 1=+i, 2=-1, 3=-i.
    // Y = iXZ contributes one factor of i per Y qubit.
    std::vector<bool> px(N, false), pz(N, false);
    int p_phase = 0;
    for (int i = 0; i < N; ++i) {
        const char c = pauli[i];
        if      (c == 'X' || c == 'x') { px[i] = true; }
        else if (c == 'Y' || c == 'y') { px[i] = true; pz[i] = true; }
        else if (c == 'Z' || c == 'z') { pz[i] = true; }
    }

    // Check if P commutes with each stabilizer.
    // P anticommutes with g iff symplectic inner product is 1:
    // anti = sum_j (px_j & gz_j) XOR (pz_j & gx_j)  (mod 2)
    for (int s = N; s < 2 * N; ++s) {
        int anti = 0;
        for (int j = 0; j < N; ++j) {
            if (px[j] && get_xz(s, N + j)) anti ^= 1;
            if (pz[j] && get_xz(s, j))     anti ^= 1;
        }
        if (anti & 1) return 0;  // anticommutes → ⟨P⟩ = 0
    }

    // P commutes with all stabilizers.
    // Express P as a product of stabilizers via GF(2) Gaussian elimination.
    // Working copy of stabilizer block (rows N..2N-1).
    const int rows = N;
    const int cols = 2 * N;
    std::vector<std::vector<bool>> mat(rows, std::vector<bool>(cols, false));
    std::vector<int> mat_phase(rows, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) mat[i][j] = get_xz(N + i, j);
        mat_phase[i] = ph[N + i] ? 2 : 0;
    }

    // Target row: [px | pz] with accumulated phase
    std::vector<bool> target(cols, false);
    int target_phase = p_phase;
    for (int j = 0; j < N; ++j) target[j] = px[j];
    for (int j = 0; j < N; ++j) target[N + j] = pz[j];

    // Phase of row1 * row2 per Aaronson-Gottesman Table 1.
    // ph1, ph2, ph_out are exponents mod 4.
    auto pauli_phase_update = [&](const std::vector<bool>& row1, int ph1,
                                   const std::vector<bool>& row2, int ph2,
                                   int& ph_out) {
        int phase_count = 0;
        for (int j = 0; j < N; ++j) {
            const bool x1 = row1[j], z1 = row1[N + j];
            const bool x2 = row2[j], z2 = row2[N + j];
            if      ( x1 &&  z1 &&  x2 && !z2) phase_count--;  // Y*X = -iZ
            else if ( x1 &&  z1 && !x2 &&  z2) phase_count++;  // Y*Z = +iX
            else if ( x1 && !z1 &&  x2 &&  z2) phase_count++;  // X*Y = +iZ
            else if (!x1 &&  z1 &&  x2 &&  z2) phase_count--;  // Z*Y = -iX
            else if ( x1 && !z1 && !x2 &&  z2) phase_count--;  // X*Z = -iY
            else if (!x1 &&  z1 &&  x2 && !z2) phase_count++;  // Z*X = +iY
        }
        ph_out = ((ph1 + ph2 + phase_count) % 4 + 4) % 4;
    };

    int pivot_col = 0;
    for (int col = 0; col < cols && pivot_col < rows; ++col) {
        int pivot = -1;
        for (int r = pivot_col; r < rows; ++r) {
            if (mat[r][col]) { pivot = r; break; }
        }
        if (pivot < 0) continue;

        std::swap(mat[pivot], mat[pivot_col]);
        std::swap(mat_phase[pivot], mat_phase[pivot_col]);

        for (int r = 0; r < rows; ++r) {
            if (r != pivot_col && mat[r][col]) {
                int new_ph;
                pauli_phase_update(mat[r], mat_phase[r], mat[pivot_col], mat_phase[pivot_col], new_ph);
                for (int j = 0; j < cols; ++j) mat[r][j] = mat[r][j] ^ mat[pivot_col][j];
                mat_phase[r] = new_ph;
            }
        }

        if (target[col]) {
            int new_ph;
            pauli_phase_update(target, target_phase, mat[pivot_col], mat_phase[pivot_col], new_ph);
            for (int j = 0; j < cols; ++j) target[j] = target[j] ^ mat[pivot_col][j];
            target_phase = new_ph;
        }

        pivot_col++;
    }

    // If target is all-zero, P is in the stabilizer group; phase gives the eigenvalue.
    for (int j = 0; j < cols; ++j) {
        if (target[j]) return 0;  // not in stabilizer group
    }

    return (target_phase == 0) ? +1 : (target_phase == 2) ? -1 : 0;
}

// =============================================================================
// CliffordSimulator
// =============================================================================

bool CliffordSimulator::is_clifford(const QuantumCircuit& circuit) {
    using GT = Instruction::GateType;
    const double pi = M_PI;
    for (const auto& inst : circuit.instructions) {
        switch (inst.type) {
            case GT::H: case GT::X: case GT::Y: case GT::Z:
            case GT::S: case GT::SDG:
            case GT::CX: case GT::CZ: case GT::SWAP:
            case GT::MEASURE: case GT::RESET: case GT::BARRIER:
                break;
            case GT::P: {
                // Accept P only if the angle maps to a Clifford gate.
                if (inst.params.empty()) return false;
                double a = std::fmod(inst.params[0], 2.0 * pi);
                if (a < 0) a += 2.0 * pi;
                if (!(std::abs(a) < 1e-9 ||
                      std::abs(a - pi / 2.0) < 1e-9 ||
                      std::abs(a - pi)        < 1e-9 ||
                      std::abs(a - 3.0 * pi / 2.0) < 1e-9))
                    return false;
                break;
            }
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

    const int n_clbits = circuit.n_clbits > 0 ? circuit.n_clbits : circuit.n_qubits;
    const double pi = M_PI;

    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);

    for (int s = 0; s < shots; ++s) {
        StabilizerState state(circuit.n_qubits);
        std::vector<int> clreg(n_clbits, 0);

        for (const auto& inst : circuit.instructions) {
            // Classical condition check (feedforward)
            if (inst.condition_clbit >= 0) {
                int cv = (inst.condition_clbit < n_clbits)
                         ? clreg[inst.condition_clbit] : 0;
                if (cv != inst.condition_value) continue;
            }

            switch (inst.type) {
                case GT::H: state.apply_h(inst.qubits[0]); break;
                case GT::S: state.apply_s(inst.qubits[0]); break;
                case GT::SDG:
                    state.apply_sdg(inst.qubits[0]);
                    break;
                case GT::X: state.apply_x(inst.qubits[0]); break;
                case GT::Y: state.apply_y(inst.qubits[0]); break;
                case GT::Z: state.apply_z(inst.qubits[0]); break;
                case GT::P: {
                    // Map to the equivalent Clifford gate by angle.
                    double a = std::fmod(inst.params[0], 2.0 * pi);
                    if (a < 0) a += 2.0 * pi;
                    if (std::abs(a) < 1e-9 || std::abs(a - 2.0 * pi) < 1e-9) {
                        // P(0) = identity
                    } else if (std::abs(a - pi / 2.0) < 1e-9) {
                        state.apply_s(inst.qubits[0]);
                    } else if (std::abs(a - pi) < 1e-9) {
                        state.apply_z(inst.qubits[0]);
                    } else if (std::abs(a - 3.0 * pi / 2.0) < 1e-9) {
                        state.apply_sdg(inst.qubits[0]);
                    } else {
                        throw std::runtime_error(
                            "CliffordSimulator: P(" + std::to_string(inst.params[0]) +
                            ") is not Clifford. Only P(0), P(π/2), P(π), P(3π/2) are supported.");
                    }
                    break;
                }
                case GT::CX:
                    state.apply_cx(inst.qubits[0], inst.qubits[1]);
                    break;
                case GT::CZ:
                    // CZ = H(target) · CX · H(target)
                    state.apply_h(inst.qubits[1]);
                    state.apply_cx(inst.qubits[0], inst.qubits[1]);
                    state.apply_h(inst.qubits[1]);
                    break;
                case GT::SWAP:
                    // SWAP = CX(a,b)·CX(b,a)·CX(a,b)
                    state.apply_cx(inst.qubits[0], inst.qubits[1]);
                    state.apply_cx(inst.qubits[1], inst.qubits[0]);
                    state.apply_cx(inst.qubits[0], inst.qubits[1]);
                    break;
                case GT::MEASURE: {
                    int q = inst.qubits[0];
                    int clbit = inst.clbits.empty() ? q : inst.clbits[0];
                    int outcome = state.measure(q, true, rng);
                    if (clbit >= 0 && clbit < n_clbits)
                        clreg[clbit] = outcome;
                    break;
                }
                case GT::RESET: {
                    int q = inst.qubits[0];
                    int outcome = state.measure(q, true, rng);
                    if (outcome == 1) state.apply_x(q);
                    break;
                }
                case GT::BARRIER: break;
                default: break;
            }
        }

        // Build bitstring: clbit 0 = LSB (rightmost), highest clbit = MSB.
        std::string bitstring(n_clbits, '0');
        for (int c = 0; c < n_clbits; ++c) {
            if (clreg[c]) bitstring[n_clbits - 1 - c] = '1';
        }
        result.counts[bitstring]++;
        result.final_state = std::move(state);
    }

    return result;
}

} // namespace lindblad
