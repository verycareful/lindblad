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

void DensityMatrix::initialize() {
    std::fill(data.begin(), data.end(), Complex128(0.0, 0.0));
    data[0] = Complex128(1.0, 0.0);  // |0...0><0...0|
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
// Localized sub-block application helpers (R.1.13, audit F-1/F-2)
//
// Rewrite of the previous serial, column-strided apply_gate. rho is row-major
// dim×dim, so a full row is contiguous. The ket (left) multiply therefore
// becomes a complex AXPY over contiguous dim-length rows, parallelised over
// background groups; the bra (right) multiply stays row-local and parallelises
// over rows. An out-of-place ket variant lets apply_kraus avoid the per-Kraus
// full-matrix restore copy entirely.
// =============================================================================

namespace {

// OMP-gate threshold: total complex elements touched by a pass. Below this the
// parallel/simd machinery costs more than it saves (small-register DM tests).
constexpr size_t DM_PAR_THRESHOLD = 1u << 16;

// Build the sub-block offset table (MSB-first internal addressing: sub-index
// bit qi maps to physical qubit qubits[k-1-qi]) and the background-index table
// (all target bits zero) for a k-qubit operator on `qubits`.
void dm_build_tables(int n_qubits, size_t dim, const std::vector<int>& qubits,
                     std::vector<size_t>& sub_off, std::vector<size_t>& bg) {
    const int k = static_cast<int>(qubits.size());
    const size_t sub_dim = size_t(1) << k;
    std::vector<bool> is_tgt(static_cast<size_t>(n_qubits), false);
    for (int q : qubits) is_tgt[static_cast<size_t>(q)] = true;

    sub_off.resize(sub_dim);
    for (size_t s = 0; s < sub_dim; ++s) {
        size_t off = 0;
        for (int qi = 0; qi < k; ++qi)
            if ((s >> qi) & 1) off |= (size_t(1) << qubits[static_cast<size_t>(k - 1 - qi)]);
        sub_off[s] = off;
    }

    const size_t n_bg = dim >> k;
    bg.resize(n_bg);
    for (size_t bidx = 0; bidx < n_bg; ++bidx) {
        size_t b = 0;
        int src_bit = 0;
        for (int bb = 0; bb < n_qubits; ++bb) {
            if (!is_tgt[static_cast<size_t>(bb)]) {
                if ((bidx >> src_bit) & 1) b |= (size_t(1) << bb);
                ++src_bit;
            }
        }
        bg[bidx] = b;
    }
}

// Ket (left) multiply, out-of-place: dst = U·rho on the row index.
// Reads `src`, writes `dst` (must NOT alias). Every row is written exactly
// once, so `dst` needs no pre-zeroing. Inner loop streams two contiguous
// complex arrays (a complex AXPY) and vectorises.
void dm_ket_apply(const Complex128* __restrict src, Complex128* __restrict dst,
                  const std::vector<Complex128>& U,
                  const std::vector<size_t>& sub_off,
                  const std::vector<size_t>& bg,
                  size_t sub_dim, size_t dim) {
    const int n_bg = static_cast<int>(bg.size());
    #pragma omp parallel for schedule(static) if(bg.size() * dim >= DM_PAR_THRESHOLD)
    for (int bi = 0; bi < n_bg; ++bi) {
        const size_t base = bg[static_cast<size_t>(bi)];
        for (size_t ro = 0; ro < sub_dim; ++ro) {
            Complex128* __restrict d = dst + (base + sub_off[ro]) * dim;
            const Complex128 u0 = U[ro * sub_dim];
            const Complex128* __restrict s0 = src + (base + sub_off[0]) * dim;
            for (size_t c = 0; c < dim; ++c) d[c] = u0 * s0[c];
            for (size_t ri = 1; ri < sub_dim; ++ri) {
                const Complex128 u = U[ro * sub_dim + ri];
                if (u.real == 0.0 && u.imag == 0.0) continue;
                const Complex128* __restrict s = src + (base + sub_off[ri]) * dim;
                for (size_t c = 0; c < dim; ++c) d[c] += u * s[c];
            }
        }
    }
}

// Ket (left) multiply, in-place: rho = U·rho on the row index. Output rows
// overlap input rows, so each background group's sub_dim rows are snapshotted
// (thread-local, grows to the largest sub_dim*dim seen — reused across gates
// so no per-gate allocation) before the AXPY writes back.
void dm_ket_apply_inplace(Complex128* __restrict data,
                          const std::vector<Complex128>& U,
                          const std::vector<size_t>& sub_off,
                          const std::vector<size_t>& bg,
                          size_t sub_dim, size_t dim) {
    const int n_bg = static_cast<int>(bg.size());
    #pragma omp parallel if(bg.size() * dim >= DM_PAR_THRESHOLD)
    {
        thread_local std::vector<Complex128> snap;
        if (snap.size() < sub_dim * dim) snap.resize(sub_dim * dim);
        #pragma omp for schedule(static)
        for (int bi = 0; bi < n_bg; ++bi) {
            const size_t base = bg[static_cast<size_t>(bi)];
            for (size_t s = 0; s < sub_dim; ++s) {
                const Complex128* row = data + (base + sub_off[s]) * dim;
                std::copy(row, row + dim, snap.begin() + static_cast<ptrdiff_t>(s * dim));
            }
            for (size_t ro = 0; ro < sub_dim; ++ro) {
                Complex128* __restrict d = data + (base + sub_off[ro]) * dim;
                const Complex128 u0 = U[ro * sub_dim];
                const Complex128* s0 = snap.data();
                for (size_t c = 0; c < dim; ++c) d[c] = u0 * s0[c];
                for (size_t ri = 1; ri < sub_dim; ++ri) {
                    const Complex128 u = U[ro * sub_dim + ri];
                    if (u.real == 0.0 && u.imag == 0.0) continue;
                    const Complex128* s = snap.data() + ri * dim;
                    for (size_t c = 0; c < dim; ++c) d[c] += u * s[c];
                }
            }
        }
    }
}

// Bra (right) multiply, in-place: rho = rho·U† on the column index. Row-local
// (mixes only the sub_dim target columns within each row), parallelised over
// rows. new[c_out] = Σ_{c_in} rho[c_in] · conj(U[c_out, c_in]).
void dm_bra_apply_inplace(Complex128* __restrict data,
                          const std::vector<Complex128>& U,
                          const std::vector<size_t>& sub_off,
                          const std::vector<size_t>& bg,
                          size_t sub_dim, size_t dim) {
    const int ndim = static_cast<int>(dim);
    #pragma omp parallel if(dim * bg.size() >= DM_PAR_THRESHOLD)
    {
        thread_local std::vector<Complex128> tmp;
        if (tmp.size() < sub_dim) tmp.resize(sub_dim);
        #pragma omp for schedule(static)
        for (int row = 0; row < ndim; ++row) {
            Complex128* __restrict rp = data + static_cast<size_t>(row) * dim;
            for (size_t bidx = 0; bidx < bg.size(); ++bidx) {
                const size_t base = bg[bidx];
                for (size_t ci = 0; ci < sub_dim; ++ci)
                    tmp[ci] = rp[base + sub_off[ci]];
                for (size_t co = 0; co < sub_dim; ++co) {
                    Complex128 sum(0.0, 0.0);
                    for (size_t ci = 0; ci < sub_dim; ++ci)
                        sum += tmp[ci] * U[co * sub_dim + ci].conj();
                    rp[base + sub_off[co]] = sum;
                }
            }
        }
    }
}

}  // namespace

// =============================================================================
// apply_gate — localized tensor operation: rho -> U * rho * U†
// In-place ket AXPY followed by in-place bra multiply (audit F-1). Background
// indices (all target bits = 0) are enumerated directly; each pass is
// OpenMP-parallel and streams contiguous rows.
// =============================================================================

void DensityMatrix::apply_gate(const std::vector<Complex128>& U,
                                const std::vector<int>& qubits) {
    const int k = static_cast<int>(qubits.size());
    const size_t sub_dim = size_t(1) << k;

    if (U.size() != sub_dim * sub_dim) {
        throw std::invalid_argument("Gate matrix size mismatch");
    }

    std::vector<size_t> sub_off, bg;
    dm_build_tables(n_qubits, dim, qubits, sub_off, bg);

    dm_ket_apply_inplace(data.data(), U, sub_off, bg, sub_dim, dim);
    dm_bra_apply_inplace(data.data(), U, sub_off, bg, sub_dim, dim);
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
    // Out-of-place accumulation (R.1.13, audit F-2): the previous loop did
    // `data = original` (a full 4^N copy) before each in-place K application.
    // Here `data` is left untouched as the source and each K's contribution is
    // built in `scratch` via the out-of-place ket multiply (no restore copy),
    // then accumulated into `result`. Peak memory is unchanged (result +
    // scratch), but the per-operator full-matrix copy is gone.
    const int k = static_cast<int>(qubits.size());
    const size_t sub_dim = size_t(1) << k;

    std::vector<size_t> sub_off, bg;
    dm_build_tables(n_qubits, dim, qubits, sub_off, bg);

    std::vector<Complex128> result(dim * dim, Complex128(0.0, 0.0));
    std::vector<Complex128> scratch(dim * dim);
    const Complex128* orig = data.data();  // untouched until the final move

    // Convention bridge: KrausChannel operator matrices follow the external
    // qubits[0]-is-LSB contract; the internal sub-block addressing here treats
    // qubits[0] as the MSB, so bit-reverse for k >= 2 (a no-op for k <= 1).
    for (const auto& K : kraus_ops) {
        const std::vector<Complex128>* Kp = &K;
        std::vector<Complex128> Krev;
        if (k >= 2) { Krev = dm_bit_reverse_matrix(K, k); Kp = &Krev; }

        dm_ket_apply(orig, scratch.data(), *Kp, sub_off, bg, sub_dim, dim);   // scratch = K·rho
        dm_bra_apply_inplace(scratch.data(), *Kp, sub_off, bg, sub_dim, dim); // scratch = scratch·K†

        const size_t total = dim * dim;
        #pragma omp parallel for schedule(static) if(total >= DM_PAR_THRESHOLD)
        for (int i = 0; i < static_cast<int>(total); ++i) result[static_cast<size_t>(i)] += scratch[static_cast<size_t>(i)];
    }

    data = std::move(result);
}

void DensityMatrix::apply_permutation(const std::vector<int>& full_perm) {
    // new_rho[full_perm[a], full_perm[b]] = rho[a, b].
    std::vector<Complex128> out(dim * dim, Complex128(0.0, 0.0));
    #pragma omp parallel for schedule(static) if(dim * dim >= DM_PAR_THRESHOLD)
    for (long long a = 0; a < static_cast<long long>(dim); ++a) {
        const size_t fa = static_cast<size_t>(full_perm[static_cast<size_t>(a)]);
        const Complex128* __restrict src = data.data() + static_cast<size_t>(a) * dim;
        Complex128* __restrict drow = out.data() + fa * dim;
        for (size_t b = 0; b < dim; ++b)
            drow[static_cast<size_t>(full_perm[b])] = src[b];
    }
    data = std::move(out);
}

void DensityMatrix::apply_mcp_phase(size_t mask, double lambda) {
    // phase(a) = e^{i*lambda} iff (a & mask) == mask, else 1.
    // new_rho[a,b] = phase(a) * conj(phase(b)) * rho[a,b]. Only entries where
    // exactly one of a,b satisfies the mask acquire a net +-lambda phase.
    const double c = std::cos(lambda), s = std::sin(lambda);
    #pragma omp parallel for schedule(static) if(dim * dim >= DM_PAR_THRESHOLD)
    for (long long al = 0; al < static_cast<long long>(dim); ++al) {
        const size_t a = static_cast<size_t>(al);
        const bool pa = (a & mask) == mask;
        Complex128* __restrict row = data.data() + a * dim;
        for (size_t b = 0; b < dim; ++b) {
            const bool pb = (b & mask) == mask;
            if (pa == pb) continue;               // net phase 1
            const double ss = pa ? s : -s;        // pa&&!pb: +lambda; !pa&&pb: -lambda
            const double r = row[b].real, im = row[b].imag;
            row[b].real = r * c - im * ss;
            row[b].imag = r * ss + im * c;
        }
    }
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
    // For each Pauli string, P[k, row] ≠ 0 only when k = row ⊕ flip_mask
    // (flip_mask covers X and Y positions). The per-row phase is
    // i^{#Y} · (-1)^{popcount(row & sign_mask)} where sign_mask covers Z and Y
    // positions. R.1.13 (audit F-14): the i^{#Y} factor is a per-term constant
    // folded into the coefficient (same trick as SparsePauliOp to_matrix), the
    // per-row sign is one popcount parity, and the row loop is an OpenMP
    // reduction — replacing the per-row z_bits/y_bits vector walks.
    double result = 0.0;
    for (const auto& term : hamiltonian.terms) {
        // Project convention (docs/Architecture.md "Conventions"): Pauli
        // strings are LSB-first, pauli[q] acts on qubit q = bit q.
        size_t flip_mask = 0, sign_mask = 0;
        int y_count = 0;
        for (int q = 0; q < nq; ++q) {
            const char p = term.pauli[q];
            if (p == 'X' || p == 'x') {
                flip_mask |= (1ULL << q);
            } else if (p == 'Y' || p == 'y') {
                flip_mask |= (1ULL << q);
                sign_mask |= (1ULL << q);
                ++y_count;
            } else if (p == 'Z' || p == 'z') {
                sign_mask |= (1ULL << q);
            }
        }

        // c = coeff · i^{#Y} (constant per term).
        double ty_r = 1.0, ty_i = 0.0;
        switch (y_count & 3) {
            case 1: ty_r = 0.0; ty_i = 1.0; break;
            case 2: ty_r = -1.0; ty_i = 0.0; break;
            case 3: ty_r = 0.0; ty_i = -1.0; break;
            default: break;
        }
        const double c_re = term.coeff.real * ty_r - term.coeff.imag * ty_i;
        const double c_im = term.coeff.real * ty_i + term.coeff.imag * ty_r;

        double tr_re = 0.0, tr_im = 0.0;
        #pragma omp parallel for reduction(+:tr_re,tr_im) schedule(static) if(dim >= (1u << 12))
        for (long long r = 0; r < static_cast<long long>(dim); ++r) {
            const size_t row = static_cast<size_t>(r);
            const Complex128& rho_rc = data[row * dim + (row ^ flip_mask)];
            if (LINDBLAD_POPCOUNT64(row & sign_mask) & 1) {
                tr_re -= rho_rc.real; tr_im -= rho_rc.imag;
            } else {
                tr_re += rho_rc.real; tr_im += rho_rc.imag;
            }
        }

        result += c_re * tr_re - c_im * tr_im;
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

        // Per-instruction plan, resolved ONCE (audit F-12/F-13). Previously
        // apply_inst deep-copied every Kraus matrix (errors_for_gate returns by
        // value) and rebuilt the gate matrix on every call — i.e. per shot per
        // gate. Both are constant across shots, so precompute here: gate
        // matrices, and noise as pointers into the (immutable) NoiseModel with
        // the effective qubit list resolved (empty error.qubits -> inst.qubits).
        struct ResolvedError {
            const std::vector<std::vector<Complex128>>* ops;
            std::vector<int> qubits;
            bool after_gate;
        };
        const size_t n_inst = circuit.instructions.size();
        std::vector<std::vector<Complex128>> gate_mats(n_inst);
        std::vector<std::vector<ResolvedError>> inst_errors(n_inst);
        // Structured ops applied without a dense matrix (audit F-7/F-9):
        //   op_kind 0 = matrix gate, 1 = permutation, 2 = mcp phase.
        std::vector<char> op_kind(n_inst, 0);
        std::vector<std::vector<int>> op_perm(n_inst);
        std::vector<size_t> op_mask(n_inst, 0);
        std::vector<double> op_lambda(n_inst, 0.0);
        const size_t dim = size_t(1) << circuit.n_qubits;  // full-register dim
        const bool ideal = noise_model.is_ideal();
        {
            using GT = Instruction::GateType;
            for (size_t ii = 0; ii < n_inst; ++ii) {
                const auto& inst = circuit.instructions[ii];
                if (inst.type == GT::MEASURE || inst.type == GT::BARRIER ||
                    inst.type == GT::RESET)
                    continue;
                if (inst.type == GT::PARAM_RX || inst.type == GT::PARAM_RY ||
                    inst.type == GT::PARAM_RZ || inst.type == GT::PARAM_P ||
                    inst.type == GT::PARAM_U) {
                    throw std::runtime_error(
                        "Unresolved parameterised gate: call assign_parameters() first.");
                }
                // Structured ops: precompute the full-register permutation
                // (MCX/PERMUTATION) or phase mask (MCP) once.
                if (inst.type == GT::MCX || inst.type == GT::PERMUTATION) {
                    op_kind[ii] = 1;
                    std::vector<int> fp(dim);
                    if (inst.type == GT::MCX) {
                        size_t ctrl_mask = 0;
                        for (size_t c = 0; c + 1 < inst.qubits.size(); ++c)
                            ctrl_mask |= (1ULL << inst.qubits[c]);
                        const size_t tbit = 1ULL << inst.qubits.back();
                        for (size_t a = 0; a < dim; ++a)
                            fp[a] = static_cast<int>(
                                ((a & ctrl_mask) == ctrl_mask) ? (a ^ tbit) : a);
                    } else {
                        const int k = static_cast<int>(inst.qubits.size());
                        const size_t sub_dim = size_t(1) << k;
                        size_t tgt_mask = 0;
                        std::vector<size_t> sub_off(sub_dim, 0);
                        for (int q : inst.qubits) tgt_mask |= (1ULL << q);
                        for (size_t x = 0; x < sub_dim; ++x)
                            for (int b = 0; b < k; ++b)
                                if ((x >> b) & 1)
                                    sub_off[x] |= (1ULL << inst.qubits[static_cast<size_t>(b)]);
                        for (size_t a = 0; a < dim; ++a) {
                            size_t x = 0;
                            for (int b = 0; b < k; ++b)
                                if ((a >> inst.qubits[static_cast<size_t>(b)]) & 1) x |= (size_t(1) << b);
                            fp[a] = static_cast<int>(
                                (a & ~tgt_mask) | sub_off[static_cast<size_t>(inst.permutation[x])]);
                        }
                    }
                    op_perm[ii] = std::move(fp);
                } else if (inst.type == GT::MCP) {
                    op_kind[ii] = 2;
                    size_t mask = 0;
                    for (int q : inst.qubits) mask |= (1ULL << q);
                    op_mask[ii] = mask;
                    op_lambda[ii] = inst.params[0];
                } else {
                    gate_mats[ii] = gate_matrix_for_dm(inst);
                }
                if (!ideal) {
                    const auto* bucket =
                        noise_model.errors_for_gate_ref(inst.gate_name());
                    if (bucket) {
                        for (const auto& ge : *bucket) {
                            if (!ge.qubits.empty() && ge.qubits != inst.qubits)
                                continue;
                            inst_errors[ii].push_back(ResolvedError{
                                &ge.channel.operators,
                                ge.qubits.empty() ? inst.qubits : ge.qubits,
                                ge.after_gate});
                        }
                    }
                }
            }
        }

        // Apply instruction `ii` (except MEASURE/BARRIER) to a DM, using the
        // pre-resolved plan (no per-call Kraus copy or gate-matrix rebuild).
        auto apply_inst = [&](DensityMatrix& dm, size_t ii) {
            const auto& inst = circuit.instructions[ii];
            if (inst.type == Instruction::GateType::RESET) {
                dm.apply_kraus({K0_reset, K1_reset}, inst.qubits);
                return;
            }
            for (const auto& re : inst_errors[ii])
                if (!re.after_gate) dm.apply_kraus(*re.ops, re.qubits);
            if (op_kind[ii] == 1)      dm.apply_permutation(op_perm[ii]);
            else if (op_kind[ii] == 2) dm.apply_mcp_phase(op_mask[ii], op_lambda[ii]);
            else                       dm.apply_gate(gate_mats[ii], inst.qubits);
            for (const auto& re : inst_errors[ii])
                if (re.after_gate) dm.apply_kraus(*re.ops, re.qubits);
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
            // Reuse ONE density-matrix buffer across shots (audit F-13):
            // re-initialise to |0><0| per shot instead of allocating a fresh
            // 4^N matrix each time. After the loop `dm` holds the last shot's
            // (collapsed) state, which becomes final_state.
            DensityMatrix dm(circuit.n_qubits);

            for (int shot = 0; shot < n_shots; ++shot) {
                if (shot > 0) dm.initialize();
                clreg.assign(n_clbits, 0);

                for (size_t ii = 0; ii < n_inst; ++ii) {
                    const auto& inst = circuit.instructions[ii];
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

                        // Readout error (issue #33): flip the RECORDED value
                        // with the qubit's confusion probability. The state
                        // stays collapsed to the true outcome; only the
                        // classical record (and any feedforward reading it)
                        // sees the noisy bit. Uses the run RNG so seeds stay
                        // reproducible; ideal models draw nothing extra.
                        int recorded = outcome;
                        auto ro_it = noise_model.readout_errors.find(qubit);
                        if (ro_it != noise_model.readout_errors.end()) {
                            const double flip_p = (outcome == 0)
                                ? ro_it->second.prob_meas_1_prep_0
                                : ro_it->second.prob_meas_0_prep_1;
                            if (flip_p > 0.0 && udist(rng) < flip_p) recorded ^= 1;
                        }
                        if (clbit >= 0 && clbit < n_clbits) clreg[clbit] = recorded;
                        continue;
                    }

                    apply_inst(dm, ii);
                }

                // Record shot result
                if (shots > 0) {
                    std::string bits(n_clbits, '0');
                    for (int c = 0; c < n_clbits; ++c) {
                        if (clreg[c]) bits[n_clbits - 1 - c] = '1';
                    }
                    result.counts[bits]++;
                }
            }

            // `dm` is the last shot's final (collapsed) state.
            result.final_state = std::move(dm);

        } else {
            // Standard single-pass mode: gates applied once, MEASURE deferred.
            DensityMatrix dm(circuit.n_qubits);

            for (size_t ii = 0; ii < n_inst; ++ii) {
                const auto& inst = circuit.instructions[ii];
                using GT = Instruction::GateType;
                if (inst.type == GT::BARRIER) continue;
                if (inst.type == GT::MEASURE) continue;  // deferred to sampling below
                apply_inst(dm, ii);
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
                            int bit = static_cast<int>((outcome >> q) & 1);
                            // Readout error (issue #33): per-shot, per-qubit
                            // flip of the recorded bit through the confusion
                            // matrix. Ideal models draw nothing extra, so
                            // seeded ideal runs are bit-for-bit unchanged.
                            auto ro_it = noise_model.readout_errors.find(q);
                            if (ro_it != noise_model.readout_errors.end()) {
                                const double flip_p = (bit == 0)
                                    ? ro_it->second.prob_meas_1_prep_0
                                    : ro_it->second.prob_meas_0_prep_1;
                                if (flip_p > 0.0 && udist(rng) < flip_p) bit ^= 1;
                            }
                            if (bit) bits[n_clbits - 1 - c] = '1';
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
