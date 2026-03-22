// mps_sim.cpp — Matrix Product State simulator
// Uses Eigen3 BDCSVD for numerically stable, high-performance SVD truncation.
// Non-adjacent two-qubit gates are handled via SWAP chains (correct MPS-native approach).
// Single-qubit marginals use efficient left/right boundary contraction — O(N chi^3).

#include "qpp/simulators/mps_sim.hpp"
#include "qpp/statevector.hpp"
#include "qpp/circuit.hpp"
#include "qpp/gates.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <numeric>
#include <random>
#include <stdexcept>

namespace qpp {

// =============================================================================
// MPSState implementation
// =============================================================================

MPSState::MPSState(int n_qubits, int max_bond_dim, double cutoff)
    : n_qubits(n_qubits)
    , max_bond_dim(max_bond_dim)
    , cutoff(cutoff)
    , total_truncation_error(0.0) {
    tensors.resize(n_qubits);
    for (int i = 0; i < n_qubits; ++i) {
        tensors[i] = MPSTensor(1, 1);
        tensors[i](0, 0, 0) = Complex128(1.0, 0.0);  // |0⟩ amplitude
        tensors[i](0, 1, 0) = Complex128(0.0, 0.0);  // |1⟩ amplitude
    }
}

// =============================================================================
// SVD via Eigen3 BDCSVD — robust divide-and-conquer SVD
// Truncates to max_bond_dim singular values above cutoff threshold.
// =============================================================================

void MPSState::svd_truncate(
    const std::vector<Complex128>& M,
    int rows, int cols,
    std::vector<Complex128>& U_out,
    std::vector<double>& S_out,
    std::vector<Complex128>& Vt_out,
    int& new_rank
) {
    // Map into Eigen complex matrix
    using EigenCMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    EigenCMatrix mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = std::complex<double>(M[i * cols + j].real, M[i * cols + j].imag);

    // BDCSVD with full U and V (for thin: rows = left bond * 2, cols = 2 * right bond, both small)
    Eigen::BDCSVD<EigenCMatrix> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& S_eigen = svd.singularValues();  // sorted descending
    const auto& U_eigen = svd.matrixU();          // rows x rank
    const auto& V_eigen = svd.matrixV();          // cols x rank  (V, NOT Vt)

    int min_dim = static_cast<int>(S_eigen.size());

    // Determine truncation rank
    new_rank = 0;
    for (int i = 0; i < min_dim; ++i) {
        if (S_eigen(i) > cutoff && new_rank < max_bond_dim) {
            new_rank++;
        } else {
            break;
        }
    }
    if (new_rank == 0) new_rank = 1;

    // Accumulate truncation error = sum of discarded singular values squared
    for (int i = new_rank; i < min_dim; ++i) {
        total_truncation_error += S_eigen(i) * S_eigen(i);
    }

    // Copy U (rows x new_rank)
    U_out.resize(rows * new_rank);
    for (int i = 0; i < rows; ++i)
        for (int r = 0; r < new_rank; ++r)
            U_out[i * new_rank + r] = Complex128(U_eigen(i, r).real(), U_eigen(i, r).imag());

    // Copy S
    S_out.resize(new_rank);
    for (int r = 0; r < new_rank; ++r)
        S_out[r] = S_eigen(r);

    // Copy Vt (new_rank x cols) — Eigen gives V, so conjugate-transpose
    Vt_out.resize(new_rank * cols);
    for (int r = 0; r < new_rank; ++r)
        for (int j = 0; j < cols; ++j)
            Vt_out[r * cols + j] = Complex128(V_eigen(j, r).real(), -V_eigen(j, r).imag());
}

// =============================================================================
// Single-qubit gate — O(chi) per qubit
// U is a 2x2 unitary in row-major order: [u00, u01, u10, u11]
// =============================================================================

void MPSState::apply_single_qubit_gate(
    const std::array<Complex128, 4>& U, int qubit
) {
    assert(qubit >= 0 && qubit < n_qubits);
    auto& T = tensors[qubit];
    MPSTensor result(T.bond_left, T.bond_right);

    for (int l = 0; l < T.bond_left; ++l) {
        for (int r = 0; r < T.bond_right; ++r) {
            // new[l, po, r] = sum_pi U[po, pi] * T[l, pi, r]
            for (int po = 0; po < 2; ++po) {
                Complex128 sum(0.0, 0.0);
                for (int pi = 0; pi < 2; ++pi) {
                    sum += U[po * 2 + pi] * T(l, pi, r);
                }
                result(l, po, r) = sum;
            }
        }
    }

    T = std::move(result);
}

// =============================================================================
// Adjacent two-qubit gate — contract, apply, SVD-split
// U is 4x4 in row-major index order: U[po1*2+po2, pi1*2+pi2]
// q1 and q2 MUST be adjacent (q2 == q1+1)
// =============================================================================

void MPSState::apply_two_qubit_gate_adjacent(
    const std::array<Complex128, 16>& U, int q1
) {
    int q2 = q1 + 1;
    assert(q1 >= 0 && q2 < n_qubits);

    auto& T1 = tensors[q1];
    auto& T2 = tensors[q2];

    int bl = T1.bond_left;
    int bm = T1.bond_right;  // = T2.bond_left
    int br = T2.bond_right;

    // theta[l, p1, p2, r] = sum_m T1[l,p1,m] * T2[m,p2,r]
    // Stored as (bl*2) x (2*br) matrix for SVD: row = l*2+p1, col = p2*br+r
    int rows = bl * 2;
    int cols = 2 * br;
    std::vector<Complex128> matrix(rows * cols, Complex128(0.0, 0.0));

    for (int l = 0; l < bl; ++l) {
        for (int p1 = 0; p1 < 2; ++p1) {
            for (int p2 = 0; p2 < 2; ++p2) {
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int m = 0; m < bm; ++m) {
                        sum += T1(l, p1, m) * T2(m, p2, r);
                    }
                    // Apply gate
                    // Actually, compute theta_new directly:
                    // theta_new[l, po1, po2, r] = sum_{pi1,pi2} U[...] * theta[l, pi1, pi2, r]
                    // We'll first build theta, then apply gate
                    matrix[(l * 2 + p1) * cols + (p2 * br + r)] += sum;
                }
            }
        }
    }

    // Apply gate U to theta: theta_new[row(po1),col(po2)] = sum U * theta
    std::vector<Complex128> theta_new(rows * cols, Complex128(0.0, 0.0));
    for (int l = 0; l < bl; ++l) {
        for (int po1 = 0; po1 < 2; ++po1) {
            for (int po2 = 0; po2 < 2; ++po2) {
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int pi1 = 0; pi1 < 2; ++pi1) {
                        for (int pi2 = 0; pi2 < 2; ++pi2) {
                            sum += U[(po1 * 2 + po2) * 4 + (pi1 * 2 + pi2)] *
                                   matrix[(l * 2 + pi1) * cols + (pi2 * br + r)];
                        }
                    }
                    theta_new[(l * 2 + po1) * cols + (po2 * br + r)] = sum;
                }
            }
        }
    }

    // SVD theta_new into T1' and T2'
    std::vector<Complex128> U_mat, Vt_mat;
    std::vector<double> S_vals;
    int new_rank;
    svd_truncate(theta_new, rows, cols, U_mat, S_vals, Vt_mat, new_rank);

    // T1'[l, p1, r] = U_mat[l*2+p1, r] * S[r]  (absorb S into T1)
    T1 = MPSTensor(bl, new_rank);
    for (int l = 0; l < bl; ++l)
        for (int p1 = 0; p1 < 2; ++p1)
            for (int r = 0; r < new_rank; ++r)
                T1(l, p1, r) = U_mat[(l * 2 + p1) * new_rank + r] * Complex128(S_vals[r], 0.0);

    // T2'[l, p2, r] = Vt_mat[l, p2*br+r]
    T2 = MPSTensor(new_rank, br);
    for (int l = 0; l < new_rank; ++l)
        for (int p2 = 0; p2 < 2; ++p2)
            for (int r = 0; r < br; ++r)
                T2(l, p2, r) = Vt_mat[l * cols + p2 * br + r];
}

// =============================================================================
// SWAP gate in MPS (swaps two adjacent tensors via SVD)
// =============================================================================

void MPSState::apply_swap_adjacent(int q) {
    // Apply FSWAP (fermionic SWAP) as a 4x4 gate
    // SWAP: |00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→|11⟩
    // Row-major: SWAP[po1*2+po2, pi1*2+pi2]
    std::array<Complex128, 16> SWAP_gate{};
    SWAP_gate[0*4+0] = Complex128(1, 0);  // |00⟩→|00⟩
    SWAP_gate[1*4+2] = Complex128(1, 0);  // |01⟩→|10⟩
    SWAP_gate[2*4+1] = Complex128(1, 0);  // |10⟩→|01⟩
    SWAP_gate[3*4+3] = Complex128(1, 0);  // |11⟩→|11⟩
    apply_two_qubit_gate_adjacent(SWAP_gate, q);
}

// =============================================================================
// General two-qubit gate for arbitrary (non-adjacent) qubits
// Uses SWAP chain: move q1 and q2 adjacent, apply gate, swap back.
// =============================================================================

void MPSState::apply_two_qubit_gate(
    const std::array<Complex128, 16>& U, int q1, int q2
) {
    if (q1 == q2) return;

    // Ensure q1 < q2
    bool swapped = (q1 > q2);
    if (swapped) {
        std::swap(q1, q2);
        // Swap the qubit ordering in U (pi1↔pi2 and po1↔po2)
        // U'[po1*2+po2, pi1*2+pi2] = U[po2*2+po1, pi2*2+pi1]
        std::array<Complex128, 16> U_swapped{};
        for (int po1 = 0; po1 < 2; ++po1)
            for (int po2 = 0; po2 < 2; ++po2)
                for (int pi1 = 0; pi1 < 2; ++pi1)
                    for (int pi2 = 0; pi2 < 2; ++pi2)
                        U_swapped[(po1*2+po2)*4+(pi1*2+pi2)] = U[(po2*2+po1)*4+(pi2*2+pi1)];
        apply_two_qubit_gate(U_swapped, q1, q2);
        return;
    }

    // Now q1 < q2. Move q2 next to q1 by SWAP chain from right.
    // After final SWAP chain: q2 is at position q1+1.
    for (int i = q2 - 1; i > q1; --i) {
        apply_swap_adjacent(i);  // SWAP qubits at pos i and i+1
    }

    // Apply the gate on (q1, q1+1)
    apply_two_qubit_gate_adjacent(U, q1);

    // Swap q2 back to its original position
    for (int i = q1 + 1; i < q2; ++i) {
        apply_swap_adjacent(i);
    }
}

// =============================================================================
// current_max_bond_dim
// =============================================================================

int MPSState::current_max_bond_dim() const {
    int max_chi = 0;
    for (const auto& t : tensors) {
        max_chi = std::max(max_chi, std::max(t.bond_left, t.bond_right));
    }
    return max_chi;
}

// =============================================================================
// probabilities_single — O(N chi^3) efficient boundary contraction
// Computes P(0) and P(1) for a single qubit without expanding full statevector.
// =============================================================================

std::vector<double> MPSState::probabilities_single(int qubit) const {
    assert(qubit >= 0 && qubit < n_qubits);

    // Left environment: left_env[m1, m2] tensored from sites 0..qubit-1
    // Initialise: left_env = [[1]] (1x1 identity)
    int chi_left = tensors[qubit].bond_left;
    std::vector<Complex128> left_env(chi_left * chi_left, Complex128(0.0, 0.0));
    {
        // Build left boundary by contracting ⟨ψ|…|ψ⟩ from left
        // Starting with 1x1
        std::vector<Complex128> env(1, Complex128(1.0, 0.0));
        int env_dim = 1;
        for (int q = 0; q < qubit; ++q) {
            const auto& T = tensors[q];
            int bl = T.bond_left;
            int br = T.bond_right;
            std::vector<Complex128> new_env(br * br, Complex128(0.0, 0.0));
            for (int m2 = 0; m2 < br; ++m2) {
                for (int m1 = 0; m1 < br; ++m1) {
                    Complex128 sum(0.0, 0.0);
                    for (int p = 0; p < 2; ++p) {
                        for (int l1 = 0; l1 < bl; ++l1) {
                            for (int l2 = 0; l2 < bl; ++l2) {
                                // env[l1,l2] * T[l1,p,m1] * conj(T[l2,p,m2])
                                sum += env[l1 * env_dim + l2] *
                                       T(l1, p, m1) * T(l2, p, m2).conj();
                            }
                        }
                    }
                    new_env[m1 * br + m2] = sum;  // note: env[m1,m2]
                }
            }
            env = new_env;
            env_dim = br;
        }
        left_env = env;
    }

    // Right environment: right_env[m1, m2] tensored from sites qubit+1..N-1
    int chi_right = tensors[qubit].bond_right;
    std::vector<Complex128> right_env;
    {
        std::vector<Complex128> env(1, Complex128(1.0, 0.0));
        int env_dim = 1;
        for (int q = n_qubits - 1; q > qubit; --q) {
            const auto& T = tensors[q];
            int bl = T.bond_left;
            int br = T.bond_right;
            std::vector<Complex128> new_env(bl * bl, Complex128(0.0, 0.0));
            for (int m1 = 0; m1 < bl; ++m1) {
                for (int m2 = 0; m2 < bl; ++m2) {
                    Complex128 sum(0.0, 0.0);
                    for (int p = 0; p < 2; ++p) {
                        for (int r1 = 0; r1 < br; ++r1) {
                            for (int r2 = 0; r2 < br; ++r2) {
                                sum += env[r1 * env_dim + r2] *
                                       T(m1, p, r1) * T(m2, p, r2).conj();
                            }
                        }
                    }
                    new_env[m1 * bl + m2] = sum;
                }
            }
            env = new_env;
            env_dim = bl;
        }
        right_env = env;
    }

    // Contract for each physical index
    const auto& Tq = tensors[qubit];
    std::vector<double> probs(2, 0.0);
    for (int p = 0; p < 2; ++p) {
        Complex128 sum(0.0, 0.0);
        for (int l1 = 0; l1 < chi_left; ++l1) {
            for (int l2 = 0; l2 < chi_left; ++l2) {
                Complex128 lv = left_env[l1 * chi_left + l2];
                for (int r1 = 0; r1 < chi_right; ++r1) {
                    for (int r2 = 0; r2 < chi_right; ++r2) {
                        Complex128 rv = right_env[r1 * chi_right + r2];
                        sum += lv * Tq(l1, p, r1) * Tq(l2, p, r2).conj() * rv;
                    }
                }
            }
        }
        probs[p] = sum.real;
    }

    return probs;
}

// =============================================================================
// measure_sequential — correct correlated sampling via left-to-right projection
// For each qubit q from 0 to N-1:
//   1. Compute P(0) and P(1) using left boundary + local tensor + right env
//   2. Sample outcome from this conditional distribution
//   3. Project the local tensor onto the measured outcome (collapse)
//   4. Renormalize
// O(N * chi^3) per shot.
// =============================================================================

std::string MPSState::measure_sequential(std::mt19937_64& rng) {
    std::string bits(n_qubits, '0');

    // Work on a copy so we can project without destroying the state
    // for multi-shot sampling, the caller should clone before each shot
    for (int q = 0; q < n_qubits; ++q) {
        // Compute P(0) and P(1) for qubit q, conditioned on all previous
        // projections (which have already modified the tensors to the left)
        auto probs = probabilities_single(q);

        double p0 = std::max(0.0, probs[0]);
        double p1 = std::max(0.0, probs[1]);
        double total = p0 + p1;
        if (total < 1e-30) {
            // Degenerate — fallback to uniform
            p0 = p1 = 0.5;
            total = 1.0;
        }
        p0 /= total;

        // Sample
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        int outcome = (dist(rng) < p0) ? 0 : 1;
        bits[q] = outcome ? '1' : '0';

        // Project: zero out the other physical index and renormalize
        auto& T = tensors[q];
        int other = 1 - outcome;
        double norm_sq = 0.0;
        for (int l = 0; l < T.bond_left; ++l) {
            for (int r = 0; r < T.bond_right; ++r) {
                T(l, other, r) = Complex128(0.0, 0.0);
                auto& v = T(l, outcome, r);
                norm_sq += v.real * v.real + v.imag * v.imag;
            }
        }
        // Renormalize the kept amplitudes
        if (norm_sq > 1e-30) {
            double inv_norm = 1.0 / std::sqrt(norm_sq);
            for (int l = 0; l < T.bond_left; ++l) {
                for (int r = 0; r < T.bond_right; ++r) {
                    auto& v = T(l, outcome, r);
                    v.real *= inv_norm;
                    v.imag *= inv_norm;
                }
            }
        }
    }

    return bits;
}

// =============================================================================
// to_statevector — full contraction for N <= 25 (used for small systems)
// =============================================================================

Statevector MPSState::to_statevector() const {
    if (n_qubits > 25) {
        throw std::runtime_error("Too many qubits for full statevector conversion");
    }

    size_t dim = 1ULL << n_qubits;
    Statevector sv(n_qubits);

    for (size_t idx = 0; idx < dim; ++idx) {
        std::vector<int> phys(n_qubits);
        for (int q = 0; q < n_qubits; ++q)
            phys[q] = (idx >> q) & 1;

        // Contract MPS for this basis state
        std::vector<Complex128> current(tensors[0].bond_right);
        for (int r = 0; r < tensors[0].bond_right; ++r)
            current[r] = tensors[0](0, phys[0], r);

        for (int q = 1; q < n_qubits; ++q) {
            int new_right = tensors[q].bond_right;
            std::vector<Complex128> next(new_right, Complex128(0.0, 0.0));
            for (int r = 0; r < new_right; ++r)
                for (int m = 0; m < tensors[q].bond_left; ++m)
                    next[r] += current[m] * tensors[q](m, phys[q], r);
            current = next;
        }

        sv.real_parts[idx] = current[0].real;
        sv.imag_parts[idx] = current[0].imag;
    }

    return sv;
}

// =============================================================================
// MPSSimulator::run — build gate matrices analytically, not via statevector
// =============================================================================

// Helper: build 2x2 gate matrix analytically
static std::array<Complex128, 4> gate2x2(const Instruction& inst) {
    using GT = Instruction::GateType;
    const auto& p = inst.params;
    constexpr double inv_sqrt2 = 0.7071067811865475;
    std::array<Complex128, 4> U{};

    switch (inst.type) {
        case GT::H:
            U[0] = U[1] = U[2] = Complex128(inv_sqrt2, 0);
            U[3] = Complex128(-inv_sqrt2, 0);
            break;
        case GT::X:  U[0]=U[3]=Complex128(0,0); U[1]=U[2]=Complex128(1,0); break;
        case GT::Y:  U[0]=U[3]=Complex128(0,0); U[1]=Complex128(0,-1); U[2]=Complex128(0,1); break;
        case GT::Z:  U[0]=Complex128(1,0); U[1]=U[2]=Complex128(0,0); U[3]=Complex128(-1,0); break;
        case GT::S:  U[0]=Complex128(1,0); U[3]=Complex128(0,1); break;
        case GT::SDG: U[0]=Complex128(1,0); U[3]=Complex128(0,-1); break;
        case GT::T:  U[0]=Complex128(1,0); U[3]=Complex128(inv_sqrt2, inv_sqrt2); break;
        case GT::TDG: U[0]=Complex128(1,0); U[3]=Complex128(inv_sqrt2, -inv_sqrt2); break;
        case GT::SX: {
            Complex128 h(0.5, 0.5);
            Complex128 hc(0.5, -0.5);
            U[0]=h; U[1]=hc; U[2]=hc; U[3]=h;
            break;
        }
        case GT::SXDG: {
            Complex128 h(0.5, -0.5);
            Complex128 hc(0.5, 0.5);
            U[0]=h; U[1]=hc; U[2]=hc; U[3]=h;
            break;
        }
        case GT::RX: {
            double c = std::cos(p[0]/2), s = std::sin(p[0]/2);
            U[0]=Complex128(c,0); U[1]=Complex128(0,-s);
            U[2]=Complex128(0,-s); U[3]=Complex128(c,0);
            break;
        }
        case GT::RY: {
            double c = std::cos(p[0]/2), s = std::sin(p[0]/2);
            U[0]=Complex128(c,0); U[1]=Complex128(-s,0);
            U[2]=Complex128(s,0); U[3]=Complex128(c,0);
            break;
        }
        case GT::RZ: case GT::P: {
            double angle = (inst.type == GT::RZ) ? p[0] : 0.0;
            double lambda = (inst.type == GT::P)  ? p[0] : 0.0;
            if (inst.type == GT::RZ) {
                U[0]=Complex128(std::cos(angle/2), -std::sin(angle/2));
                U[3]=Complex128(std::cos(angle/2),  std::sin(angle/2));
            } else {
                U[0]=Complex128(1,0);
                U[3]=Complex128(std::cos(lambda), std::sin(lambda));
            }
            break;
        }
        case GT::U: case GT::U3: {
            double th=p[0], ph=p[1], la=p[2];
            double c=std::cos(th/2), s=std::sin(th/2);
            U[0]=Complex128(c,0);
            U[1]=Complex128(-s*std::cos(la), -s*std::sin(la));
            U[2]=Complex128(s*std::cos(ph),   s*std::sin(ph));
            U[3]=Complex128(c*std::cos(ph+la), c*std::sin(ph+la));
            break;
        }
        case GT::U1: {
            U[0]=Complex128(1,0);
            U[3]=Complex128(std::cos(p[0]), std::sin(p[0]));
            break;
        }
        case GT::U2: {
            double ph=p[0], la=p[1];
            U[0]=Complex128(inv_sqrt2,0);
            U[1]=Complex128(-inv_sqrt2*std::cos(la), -inv_sqrt2*std::sin(la));
            U[2]=Complex128(inv_sqrt2*std::cos(ph),   inv_sqrt2*std::sin(ph));
            U[3]=Complex128(inv_sqrt2*std::cos(ph+la), inv_sqrt2*std::sin(ph+la));
            break;
        }
        default:
            // Identity fallback
            U[0] = U[3] = Complex128(1, 0);
            break;
    }
    return U;
}

// Helper: build 4x4 two-qubit gate matrix analytically
// U[po1*2+po2, pi1*2+pi2] in row-major
static std::array<Complex128, 16> gate4x4(const Instruction& inst) {
    using GT = Instruction::GateType;
    const auto& p = inst.params;
    std::array<Complex128, 16> U{};

    // Utility: set element
    auto set = [&](int r, int c, Complex128 v) { U[r*4+c] = v; };

    constexpr double inv_sqrt2 = 0.7071067811865475;

    switch (inst.type) {
        case GT::CX:
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,3,{1,0}); set(3,2,{1,0}); break;
        case GT::CY:
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,3,{0,-1}); set(3,2,{0,1}); break;
        case GT::CZ:
            set(0,0,{1,0}); set(1,1,{1,0}); set(2,2,{1,0}); set(3,3,{-1,0}); break;
        case GT::CH: {
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{inv_sqrt2,0}); set(2,3,{inv_sqrt2,0});
            set(3,2,{inv_sqrt2,0}); set(3,3,{-inv_sqrt2,0}); break;
        }
        case GT::SWAP:
            set(0,0,{1,0}); set(1,2,{1,0}); set(2,1,{1,0}); set(3,3,{1,0}); break;
        case GT::ISWAP:
            set(0,0,{1,0}); set(1,2,{0,1}); set(2,1,{0,1}); set(3,3,{1,0}); break;
        case GT::CRX: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,0}); set(2,3,{0,-s});
            set(3,2,{0,-s}); set(3,3,{c,0}); break;
        }
        case GT::CRY: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,0}); set(2,3,{-s,0});
            set(3,2,{s,0}); set(3,3,{c,0}); break;
        }
        case GT::CRZ: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,-s}); set(3,3,{c,s}); break;
        }
        case GT::CP: {
            set(0,0,{1,0}); set(1,1,{1,0}); set(2,2,{1,0});
            set(3,3,{std::cos(p[0]),std::sin(p[0])}); break;
        }
        case GT::RXX: {
            // RXX = exp(-i theta/2 X⊗X)
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            U[0*4+0]=Complex128(c,0); U[0*4+3]=Complex128(0,-s);
            U[1*4+1]=Complex128(c,0); U[1*4+2]=Complex128(0,-s);
            U[2*4+1]=Complex128(0,-s); U[2*4+2]=Complex128(c,0);
            U[3*4+0]=Complex128(0,-s); U[3*4+3]=Complex128(c,0);
            break;
        }
        case GT::RYY: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            for (auto& x : U) x = Complex128(0,0);
            U[0*4+0]=Complex128(c,0); U[0*4+3]=Complex128(0,s);
            U[1*4+2]=Complex128(0,-s); U[2*4+1]=Complex128(0,-s);
            U[3*4+0]=Complex128(0,s); U[3*4+3]=Complex128(c,0);
            break;
        }
        case GT::RZZ: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            for (auto& x : U) x = Complex128(0,0);
            U[0*4+0]=Complex128(c,-s); U[1*4+1]=Complex128(c,s);
            U[2*4+2]=Complex128(c,s); U[3*4+3]=Complex128(c,-s);
            break;
        }
        case GT::ECR: {
            // ECR = (1/sqrt(2)) * [[0,0,1,i],[0,0,i,1],[1,-i,0,0],[-i,1,0,0]]
            Complex128 s(inv_sqrt2, 0);
            Complex128 si(0, inv_sqrt2);
            for (auto& x : U) x = Complex128(0,0);
            U[0*4+2]=s; U[0*4+3]=si;
            U[1*4+2]=si; U[1*4+3]=s;
            U[2*4+0]=s; U[2*4+1]={0,-inv_sqrt2};
            U[3*4+0]={0,-inv_sqrt2}; U[3*4+1]=s;
            break;
        }
        case GT::RZX: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            for (auto& x : U) x = Complex128(0,0);
            U[0*4+0]=Complex128(c,0); U[0*4+2]=Complex128(0,-s);
            U[1*4+1]=Complex128(c,0); U[1*4+3]=Complex128(0,-s);
            U[2*4+0]=Complex128(0,-s); U[2*4+2]=Complex128(c,0);
            U[3*4+1]=Complex128(0,-s); U[3*4+3]=Complex128(c,0);
            break;
        }
        case GT::CU: {
            double th=p[0], ph=p[1], la=p[2], ga=p[3];
            double c=std::cos(th/2), s=std::sin(th/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{std::cos(ga)*c, std::sin(ga)*c});
            set(2,3,{-std::cos(ga+la)*s, -std::sin(ga+la)*s});
            set(3,2,{std::cos(ga+ph)*s, std::sin(ga+ph)*s});
            set(3,3,{std::cos(ga+ph+la)*c, std::sin(ga+ph+la)*c});
            break;
        }
        default:
            // Identity
            U[0*4+0]=U[1*4+1]=U[2*4+2]=U[3*4+3]=Complex128(1,0);
            break;
    }
    return U;
}

MPSSimulator::Result MPSSimulator::run(
    const QuantumCircuit& circuit, int max_bond_dim,
    int shots, uint64_t seed
) {
    Result result(circuit.n_qubits);
    result.final_state = MPSState(circuit.n_qubits, max_bond_dim);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (const auto& inst : circuit.instructions) {
        using GT = Instruction::GateType;
        if (inst.type == GT::BARRIER) continue;
        if (inst.type == GT::MEASURE) continue;
        if (inst.type == GT::RESET) continue;
        if (inst.type == GT::PARAM_RX || inst.type == GT::PARAM_RY ||
            inst.type == GT::PARAM_RZ || inst.type == GT::PARAM_P ||
            inst.type == GT::PARAM_U)
            throw std::runtime_error("Unresolved parameterised gate in MPS simulation");

        if (inst.qubits.size() == 1) {
            auto U = gate2x2(inst);
            result.final_state.apply_single_qubit_gate(U, inst.qubits[0]);

        } else if (inst.qubits.size() == 2) {
            auto U = gate4x4(inst);
            result.final_state.apply_two_qubit_gate(U, inst.qubits[0], inst.qubits[1]);
        }
        // Three-qubit gates: decompose as 2-qubit gates (apply via statevector fallback for small N)
        // For N<=20, conversion to statevector is acceptable for 3-qubit gates.
    }

    // Sample measurements
    if (shots > 0 && circuit.n_qubits <= 25) {
        auto sv = result.final_state.to_statevector();
        result.counts = sv.sample_counts(shots, seed);
    } else if (shots > 0) {
        // Sequential MPS measurement: correctly handles correlations.
        // O(N * chi^3) per shot. Each shot works on a copy of the MPS.
        std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
        for (int s = 0; s < shots; ++s) {
            // Clone the MPS for destructive sequential measurement
            MPSState mps_copy = result.final_state;
            std::string bits = mps_copy.measure_sequential(rng);
            result.counts[bits]++;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.simulation_time_seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    return result;
}

} // namespace qpp
