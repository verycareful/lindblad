// density_matrix_sim.cpp — Density Matrix simulator
// Gate application uses localized tensor operations on target qubits only,
// identical in structure to the statevector approach. Gate matrices are built
// analytically (same lookup as MPS/statevector) — no per-gate statevector
// allocation. Complexity is O(4^N) for storage, O(4^N * 4^k) for k-qubit gates.

#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

namespace lindblad {

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
//
// Background indices (those with all target bits = 0) are enumerated directly
// via bit-insertion, eliminating the O(4^N) branch-per-element loop.
// Each sub-block update uses an O(2^k) scratch buffer — no full-matrix copy.
// =============================================================================

void DensityMatrix::apply_gate(const std::vector<Complex128>& U,
                                const std::vector<int>& qubits) {
    int k = static_cast<int>(qubits.size());
    size_t sub_dim = 1ULL << k;

    if (U.size() != sub_dim * sub_dim) {
        throw std::invalid_argument("Gate matrix size mismatch");
    }

    // Mark target qubits for background-index enumeration.
    std::vector<bool> is_tgt(n_qubits, false);
    for (int q : qubits) is_tgt[q] = true;

    // Map sub-index → physical address using the ORIGINAL qubit order.
    // Gate matrices are MSB-first: qubits[0] is the MSB of the matrix index.
    // Sub-index bit qi (LSB=0) therefore maps to physical qubit qubits[k-1-qi].
    std::vector<size_t> sub_offsets(sub_dim);
    for (size_t s = 0; s < sub_dim; ++s) {
        size_t off = 0;
        for (int qi = 0; qi < k; ++qi)
            if ((s >> qi) & 1) off |= (size_t(1) << qubits[k - 1 - qi]);
        sub_offsets[s] = off;
    }

    // Precompute the 2^(N-k) background indices by inserting zero bits at
    // each target position into a compact (N-k)-bit counter.
    size_t n_bg = dim >> k;
    std::vector<size_t> bg_indices(n_bg);
    for (size_t bg_idx = 0; bg_idx < n_bg; ++bg_idx) {
        size_t bg = 0;
        int src_bit = 0;
        for (int b = 0; b < n_qubits; ++b) {
            if (!is_tgt[b]) {
                if ((bg_idx >> src_bit) & 1) bg |= (size_t(1) << b);
                ++src_bit;
            }
        }
        bg_indices[bg_idx] = bg;
    }

    // Scratch buffer: O(2^k) — thread-local to avoid repeated allocation
    thread_local std::vector<Complex128> scratch;
    if (scratch.size() < sub_dim) scratch.resize(sub_dim);

    // ---- Left multiply: data = U * data  (updates row index) ----
    // For each background row index bg, and for each column col:
    //   Read the sub_dim-vector data[bg|sub_offsets[s], col] into scratch,
    //   apply U in-place, write back.
    for (size_t bi = 0; bi < n_bg; ++bi) {
        const size_t bg = bg_indices[bi];
        for (size_t col = 0; col < dim; ++col) {
            // Read
            for (size_t s = 0; s < sub_dim; ++s)
                scratch[s] = data[(bg | sub_offsets[s]) * dim + col];
            // Apply U and write
            for (size_t r_out = 0; r_out < sub_dim; ++r_out) {
                Complex128 sum(0.0, 0.0);
                for (size_t r_in = 0; r_in < sub_dim; ++r_in)
                    sum += U[r_out * sub_dim + r_in] * scratch[r_in];
                data[(bg | sub_offsets[r_out]) * dim + col] = sum;
            }
        }
    }

    // ---- Right multiply: data = data * U†  (updates column index) ----
    // For each row row, and for each background col index bg:
    //   Read the sub_dim-vector data[row, bg|sub_offsets[s]] into scratch,
    //   apply U† (= conj(U^T)) in-place, write back.
    for (size_t row = 0; row < dim; ++row) {
        Complex128* row_ptr = data.data() + row * dim;
        for (size_t bi = 0; bi < n_bg; ++bi) {
            const size_t bg = bg_indices[bi];
            // Read
            for (size_t s = 0; s < sub_dim; ++s)
                scratch[s] = row_ptr[bg | sub_offsets[s]];
            // Apply U†: new[c_out] = sum_{c_in} conj(U[c_out, c_in]) * scratch[c_in]
            for (size_t c_out = 0; c_out < sub_dim; ++c_out) {
                Complex128 sum(0.0, 0.0);
                for (size_t c_in = 0; c_in < sub_dim; ++c_in)
                    sum += scratch[c_in] * U[c_out * sub_dim + c_in].conj();
                row_ptr[bg | sub_offsets[c_out]] = sum;
            }
        }
    }
}

// =============================================================================
// apply_kraus — rho -> sum_k K_k * rho * K_k†
// =============================================================================

// Permute a 2^k x 2^k matrix by k-bit reversal of row and column indices.
// Translates between the project-wide external convention (bit 0 of the
// matrix index = qubits[0]; see docs/Architecture.md "Conventions") and
// DensityMatrix::apply_gate's internal MSB-first sub-block addressing.
static std::vector<Complex128> dm_bit_reverse_matrix(
    const std::vector<Complex128>& M, int k
) {
    const size_t d = 1ULL << k;
    auto rev = [k](size_t idx) {
        size_t r = 0;
        for (int b = 0; b < k; ++b)
            if ((idx >> b) & 1) r |= (size_t(1) << (k - 1 - b));
        return r;
    };
    std::vector<Complex128> out(d * d, Complex128(0.0, 0.0));
    for (size_t r = 0; r < d; ++r)
        for (size_t c = 0; c < d; ++c)
            out[rev(r) * d + rev(c)] = M[r * d + c];
    return out;
}

void DensityMatrix::apply_kraus(
    const std::vector<std::vector<Complex128>>& kraus_ops,
    const std::vector<int>& qubits
) {
    // Save the original rho once so we can restore it before each K application
    // without re-allocating a new DensityMatrix per Kraus operator.
    const auto original = data;
    std::vector<Complex128> result_data(dim * dim, Complex128(0.0, 0.0));

    // Convention bridge: KrausChannel operator matrices follow the external
    // qubits[0]-is-LSB contract; apply_gate addresses qubits[0] as the MSB.
    const int k = static_cast<int>(qubits.size());

    for (const auto& K : kraus_ops) {
        data = original;        // restore rho (overwrites existing allocation)
        if (k >= 2) {
            apply_gate(dm_bit_reverse_matrix(K, k), qubits);
        } else {
            apply_gate(K, qubits);  // rho -> K * rho * K†  in-place
        }
        for (size_t i = 0; i < dim * dim; ++i) result_data[i] += data[i];
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

double DensityMatrix::expectation_value_sparse(const SparsePauliOp& hamiltonian) const {
    const int nq = hamiltonian.n_qubits();
    if (nq != n_qubits) {
        throw std::invalid_argument("Hamiltonian qubit count mismatch");
    }

    // Tr(ρH) = Σ_terms coeff · Tr(ρ · P_term)
    // For each Pauli string, P[col, row] ≠ 0 only when col = row ⊕ flip_mask,
    // where flip_mask covers X and Y qubit positions.
    // phase(row) = Π_{Z qubits} (-1)^{bit} · Π_{Y qubits} i·(-1)^{bit}
    double result = 0.0;
    for (const auto& term : hamiltonian.terms) {
        // Project convention (docs/Architecture.md "Conventions"): Pauli
        // strings are LSB-first, pauli[q] acts on qubit q = bit q.
        size_t flip_mask = 0;
        std::vector<int> z_bits, y_bits;
        for (int q = 0; q < nq; ++q) {
            const char p = term.pauli[q];
            const int bit = q;
            if (p == 'X' || p == 'x') {
                flip_mask |= (1ULL << bit);
            } else if (p == 'Y' || p == 'y') {
                flip_mask |= (1ULL << bit);
                y_bits.push_back(bit);
            } else if (p == 'Z' || p == 'z') {
                z_bits.push_back(bit);
            }
        }

        double tr_re = 0.0, tr_im = 0.0;
        for (size_t row = 0; row < dim; ++row) {
            const size_t col = row ^ flip_mask;

            // Build phase from Z and Y qubits.
            double ph_re = 1.0, ph_im = 0.0;
            for (int b : z_bits) {
                if ((row >> b) & 1) { ph_re = -ph_re; ph_im = -ph_im; }
            }
            for (int b : y_bits) {
                // multiply by i: (re,im) → (-im, re)
                double t = ph_re; ph_re = -ph_im; ph_im = t;
                // then by (-1)^{bit}
                if ((row >> b) & 1) { ph_re = -ph_re; ph_im = -ph_im; }
            }

            const Complex128& rho_rc = data[row * dim + col];
            tr_re += rho_rc.real * ph_re - rho_rc.imag * ph_im;
            tr_im += rho_rc.real * ph_im + rho_rc.imag * ph_re;
        }

        result += term.coeff.real * tr_re - term.coeff.imag * tr_im;
    }

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
            // exp(-i t/2 Z(x)X), Z on qubits[0] (MSB of this builder's index),
            // X on qubits[1] (LSB). X couples index pairs differing in the LSB;
            // the sign of the off-diagonal follows the Z eigenvalue of the MSB:
            // rows 0,1 (Z=+1) get -i*sin, rows 2,3 (Z=-1) get +i*sin.
            // Matches gates::apply_rzx (statevector reference implementation).
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,0}); set(0,1,{0,-s}); set(1,0,{0,-s}); set(1,1,{c,0});
            set(2,2,{c,0}); set(2,3,{0,s});  set(3,2,{0,s});  set(3,3,{c,0}); break;
        }
        case GT::RXX: {
            // exp(-i t/2 X(x)X): cos on the full diagonal, -i*sin on the
            // anti-diagonal. Matches gates::apply_rxx.
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,0}); set(1,1,{c,0}); set(2,2,{c,0}); set(3,3,{c,0});
            set(0,3,{0,-s}); set(1,2,{0,-s});
            set(2,1,{0,-s}); set(3,0,{0,-s}); break;
        }
        case GT::RYY: {
            // exp(-i t/2 Y(x)Y): cos on the full diagonal, +i*sin on the outer
            // anti-diagonal and -i*sin on the inner. Matches gates::apply_ryy.
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{c,0}); set(1,1,{c,0}); set(2,2,{c,0}); set(3,3,{c,0});
            set(0,3,{0,s}); set(1,2,{0,-s});
            set(2,1,{0,-s}); set(3,0,{0,s}); break;
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
            // Margolus / relative-phase Toffoli: H T CX Tdg CX T CX Tdg H
            // Matches statevector apply_rccx decomposition exactly.
            // |000⟩-|011⟩ identity, |100⟩→|100⟩, |101⟩→-|101⟩,
            // |110⟩→i|111⟩, |111⟩→-i|110⟩
            for (size_t i = 0; i < 8; ++i) U[i*8+i] = Complex128(1.0, 0.0);
            U[5*8+5] = Complex128(-1.0, 0.0);   // |101⟩ → -|101⟩
            U[6*8+6] = Complex128(0.0, 0.0);     // clear diagonal
            U[6*8+7] = Complex128(0.0, -1.0);    // |111⟩ contributes -i to row 6
            U[7*8+6] = Complex128(0.0, 1.0);     // |110⟩ → i|111⟩
            U[7*8+7] = Complex128(0.0, 0.0);     // clear diagonal
            break;
        }
        case GT::UNITARY: {
            // Convention bridge. Instruction::matrix follows the project-wide
            // contract (docs/Architecture.md "Conventions"): bit 0 (LSB) of the
            // matrix index is qubits[0]. DensityMatrix::apply_gate addresses
            // the sub-block with qubits[0] as the MSB, so permute rows and
            // columns by k-bit reversal. Mirrors the bridge in mps_sim.cpp.
            if (k <= 1 || inst.matrix.empty()) return inst.matrix;
            return dm_bit_reverse_matrix(inst.matrix, k);
        }
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

// True when no instruction (other than BARRIER) acts on a qubit after that
// qubit has been measured: outcomes can then be sampled from the final
// density matrix in one pass. A second MEASURE or RESET on a measured qubit
// forces the per-shot trajectory path.
static bool dm_measures_are_terminal(const QuantumCircuit& circuit) {
    std::vector<bool> measured(static_cast<size_t>(circuit.n_qubits), false);
    for (const auto& inst : circuit.instructions) {
        if (inst.type == Instruction::GateType::BARRIER) continue;
        for (int q : inst.qubits)
            if (q >= 0 && q < circuit.n_qubits &&
                measured[static_cast<size_t>(q)])
                return false;
        if (inst.type == Instruction::GateType::MEASURE)
            measured[static_cast<size_t>(inst.qubits[0])] = true;
    }
    return true;
}

DensityMatrixSimulator::Result DensityMatrixSimulator::run(
    const QuantumCircuit& circuit,
    const NoiseModel& noise_model,
    int shots,
    uint64_t seed
) {
    Result result;

    try {
        auto t_start = std::chrono::high_resolution_clock::now();

        // Execution strategy (see docs/api/simulators.md, Execution semantics):
        // per-shot trajectories whenever a classical condition exists OR a
        // measurement is followed by further operations on its qubit (the
        // collapse then influences later evolution and must be drawn per
        // shot). Otherwise: one pass with MEASURE deferred to sampling.
        // shots == 0 on the per-shot path runs a single seeded trajectory.
        bool has_feedforward = false;
        bool has_measure = false;
        const int n_clbits = circuit.n_clbits > 0 ? circuit.n_clbits : circuit.n_qubits;
        for (const auto& inst : circuit.instructions) {
            if (inst.condition_clbit >= 0) has_feedforward = true;
            if (inst.type == Instruction::GateType::MEASURE) has_measure = true;
        }
        const bool needs_per_shot =
            has_feedforward ||
            (has_measure && !dm_measures_are_terminal(circuit));

        // Reusable Kraus operators for RESET (|0><0| and |0><1| channels).
        const std::vector<Complex128> K0_reset = {
            Complex128(1,0), Complex128(0,0),
            Complex128(0,0), Complex128(0,0)
        };
        const std::vector<Complex128> K1_reset = {
            Complex128(0,0), Complex128(1,0),
            Complex128(0,0), Complex128(0,0)
        };

        // Helper: apply a single instruction (except MEASURE/BARRIER) to a DM.
        auto apply_inst = [&](DensityMatrix& dm, const Instruction& inst) {
            using GT = Instruction::GateType;
            if (inst.type == GT::RESET) {
                dm.apply_kraus({K0_reset, K1_reset}, inst.qubits);
                return;
            }
            if (inst.type == GT::PARAM_RX || inst.type == GT::PARAM_RY ||
                inst.type == GT::PARAM_RZ || inst.type == GT::PARAM_P ||
                inst.type == GT::PARAM_U) {
                throw std::runtime_error("Unresolved parameterised gate: call assign_parameters() first.");
            }
            auto gate_mat = gate_matrix_for_dm(inst);
            if (!noise_model.is_ideal()) {
                auto gate_errors = noise_model.errors_for_gate(inst.gate_name(), inst.qubits);
                for (const auto& error : gate_errors) {
                    if (!error.after_gate) {
                        std::vector<int> noise_qubits =
                            error.qubits.empty() ? inst.qubits : error.qubits;
                        dm.apply_kraus(error.channel.operators, noise_qubits);
                    }
                }
                dm.apply_gate(gate_mat, inst.qubits);
                for (const auto& error : gate_errors) {
                    if (error.after_gate) {
                        std::vector<int> noise_qubits =
                            error.qubits.empty() ? inst.qubits : error.qubits;
                        dm.apply_kraus(error.channel.operators, noise_qubits);
                    }
                }
            } else {
                dm.apply_gate(gate_mat, inst.qubits);
            }
        };

        if (needs_per_shot) {
            // Per-shot trajectory path (feedforward and/or mid-circuit
            // measurement). Each shot: fresh DM, iterate instructions with
            // condition checks, collapse DM on MEASURE, record to clreg.
            // shots == 0 runs exactly one seeded trajectory for final_state.
            std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
            std::uniform_real_distribution<double> udist(0.0, 1.0);
            std::vector<int> clreg(n_clbits, 0);

            const int n_shots = std::max(1, shots);
            DensityMatrix dm_last(circuit.n_qubits);

            for (int shot = 0; shot < n_shots; ++shot) {
                DensityMatrix dm(circuit.n_qubits);
                clreg.assign(n_clbits, 0);

                for (const auto& inst : circuit.instructions) {
                    using GT = Instruction::GateType;
                    if (inst.type == GT::BARRIER) continue;

                    // Classical condition check
                    if (inst.condition_clbit >= 0) {
                        int cv = (inst.condition_clbit < n_clbits)
                                 ? clreg[inst.condition_clbit] : 0;
                        if (cv != inst.condition_value) continue;
                    }

                    if (inst.type == GT::MEASURE) {
                        const int qubit = inst.qubits[0];
                        const int clbit = inst.clbits.empty() ? qubit : inst.clbits[0];

                        // P(qubit=0) = sum of diagonal elements with qubit bit = 0
                        double prob0 = 0.0;
                        for (size_t i = 0; i < dm.dim; ++i) {
                            if (!((i >> qubit) & 1))
                                prob0 += dm.data[i * dm.dim + i].real;
                        }
                        prob0 = std::max(0.0, std::min(1.0, prob0));

                        const int outcome = (udist(rng) < prob0) ? 0 : 1;
                        const double p_out = (outcome == 0) ? prob0 : (1.0 - prob0);

                        // Project: zero out rho_{ij} where qubit bit of i or j != outcome
                        for (size_t i = 0; i < dm.dim; ++i) {
                            const int bi = (i >> qubit) & 1;
                            for (size_t j = 0; j < dm.dim; ++j) {
                                const int bj = (j >> qubit) & 1;
                                if (bi != outcome || bj != outcome)
                                    dm.data[i * dm.dim + j] = Complex128(0.0, 0.0);
                            }
                        }
                        // Renormalize
                        if (p_out > 1e-15) {
                            const double inv_p = 1.0 / p_out;
                            for (auto& v : dm.data) { v.real *= inv_p; v.imag *= inv_p; }
                        }

                        if (clbit >= 0 && clbit < n_clbits) clreg[clbit] = outcome;
                        continue;
                    }

                    apply_inst(dm, inst);
                }

                // Record shot result
                if (shots > 0) {
                    std::string bits(n_clbits, '0');
                    for (int c = 0; c < n_clbits; ++c) {
                        if (clreg[c]) bits[n_clbits - 1 - c] = '1';
                    }
                    result.counts[bits]++;
                }

                // Keep only the last trajectory's state (moved, not copied:
                // a full 4^N copy per shot is pure overhead otherwise).
                if (shot == n_shots - 1) dm_last = std::move(dm);
            }

            result.final_state = std::move(dm_last);

        } else {
            // Standard single-pass mode: gates applied once, MEASURE deferred.
            DensityMatrix dm(circuit.n_qubits);

            for (const auto& inst : circuit.instructions) {
                using GT = Instruction::GateType;
                if (inst.type == GT::BARRIER) continue;
                if (inst.type == GT::MEASURE) continue;  // deferred to sampling below
                apply_inst(dm, inst);
            }

            // Sample measurements from the diagonal of the density matrix.
            // Keys follow the qubit -> clbit mapping of the (terminal)
            // MEASURE instructions; clbit 0 is the rightmost character. When
            // the circuit has no MEASURE at all, the full register is sampled
            // with qubit-indexed keys (legacy behaviour).
            if (shots > 0) {
                std::vector<std::pair<int, int>> meas;  // (qubit, clbit)
                for (const auto& inst : circuit.instructions)
                    if (inst.type == Instruction::GateType::MEASURE)
                        meas.emplace_back(inst.qubits[0],
                                          inst.clbits.empty() ? inst.qubits[0]
                                                              : inst.clbits[0]);

                auto probs = dm.probabilities();
                // Build cumulative probability array and sample via binary search —
                // avoids the O(2^N) alias table built by std::discrete_distribution.
                std::vector<double> cum(probs.size());
                std::partial_sum(probs.begin(), probs.end(), cum.begin());

                std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
                std::uniform_real_distribution<double> udist(0.0, 1.0);

                for (int s = 0; s < shots; ++s) {
                    double r = udist(rng);
                    auto it = std::lower_bound(cum.begin(), cum.end(), r);
                    size_t outcome = static_cast<size_t>(std::distance(cum.begin(), it));
                    if (outcome >= dm.dim) outcome = dm.dim - 1;

                    if (meas.empty()) {
                        std::string bits(circuit.n_qubits, '0');
                        for (int b = circuit.n_qubits - 1; b >= 0; --b) {
                            if ((outcome >> b) & 1)
                                bits[circuit.n_qubits - 1 - b] = '1';
                        }
                        result.counts[bits]++;
                    } else {
                        std::string bits(n_clbits, '0');
                        for (const auto& [q, c] : meas) {
                            if (c < 0 || c >= n_clbits) continue;
                            if ((outcome >> q) & 1) bits[n_clbits - 1 - c] = '1';
                        }
                        result.counts[bits]++;
                    }
                }
            }

            result.final_state = std::move(dm);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        result.simulation_time_seconds =
            std::chrono::duration<double>(t_end - t_start).count();
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

} // namespace lindblad
