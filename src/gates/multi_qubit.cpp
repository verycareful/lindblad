#include "lindblad/gates.hpp"

#include "lindblad/detail/validate.hpp"
#include "lindblad/detail/validate_physical.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace lindblad {
namespace gates {

// =============================================================================
// CCX (Toffoli): flip target when both controls are 1
// =============================================================================
// Three-level stride pattern: visits only dim/8 pairs that need work instead
// of dim iterations with 7/8 wasted (no-op) passes.  4–8× faster for large N.
void apply_ccx(Statevector& sv, int c1, int c2, int tgt) {
    detail::check_qubit(c1, sv.n_qubits, "ccx");
    detail::check_qubit(c2, sv.n_qubits, "ccx");
    detail::check_qubit(tgt, sv.n_qubits, "ccx");
    detail::check_all_distinct({c1, c2, tgt}, "ccx");
    std::array<int,3> qs = {c1, c2, tgt};
    std::sort(qs.begin(), qs.end());
    const size_t s0  = 1ULL << qs[0];
    const size_t s1  = 1ULL << qs[1];
    const size_t s2  = 1ULL << qs[2];
    const size_t c1b = 1ULL << c1;
    const size_t c2b = 1ULL << c2;
    const size_t tb  = 1ULL << tgt;
    const int n_outer = static_cast<int>(sv.dim / (2 * s2));

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (int oi = 0; oi < n_outer; ++oi) {
        const size_t k2 = static_cast<size_t>(oi) * 2 * s2;
        for (size_t k1 = 0; k1 < s2; k1 += 2 * s1) {
            for (size_t k0 = 0; k0 < s1; k0 += 2 * s0) {
                for (size_t k = 0; k < s0; ++k) {
                    const size_t base = k2 + k1 + k0 + k;
                    const size_t i = base | c1b | c2b;       // c1=1, c2=1, tgt=0
                    const size_t j = base | c1b | c2b | tb;  // c1=1, c2=1, tgt=1
                    std::swap(sv.real_parts[i], sv.real_parts[j]);
                    std::swap(sv.imag_parts[i], sv.imag_parts[j]);
                }
            }
        }
    }
}

// =============================================================================
// CCZ: negate phase when all three qubits are 1
// =============================================================================
void apply_ccz(Statevector& sv, int c1, int c2, int tgt) {
    detail::check_qubit(c1, sv.n_qubits, "ccz");
    detail::check_qubit(c2, sv.n_qubits, "ccz");
    detail::check_qubit(tgt, sv.n_qubits, "ccz");
    detail::check_all_distinct({c1, c2, tgt}, "ccz");
    std::array<int,3> qs = {c1, c2, tgt};
    std::sort(qs.begin(), qs.end());
    const size_t s0  = 1ULL << qs[0];
    const size_t s1  = 1ULL << qs[1];
    const size_t s2  = 1ULL << qs[2];
    const size_t c1b = 1ULL << c1;
    const size_t c2b = 1ULL << c2;
    const size_t tb  = 1ULL << tgt;
    const int n_outer = static_cast<int>(sv.dim / (2 * s2));

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (int oi = 0; oi < n_outer; ++oi) {
        const size_t k2 = static_cast<size_t>(oi) * 2 * s2;
        for (size_t k1 = 0; k1 < s2; k1 += 2 * s1) {
            for (size_t k0 = 0; k0 < s1; k0 += 2 * s0) {
                for (size_t k = 0; k < s0; ++k) {
                    const size_t i = (k2 + k1 + k0 + k) | c1b | c2b | tb;  // all three = 1
                    sv.real_parts[i] = -sv.real_parts[i];
                    sv.imag_parts[i] = -sv.imag_parts[i];
                }
            }
        }
    }
}

// =============================================================================
// CSWAP (Fredkin): swap q1, q2 when ctrl=1
// =============================================================================
void apply_cswap(Statevector& sv, int ctrl, int q1, int q2) {
    detail::check_qubit(ctrl, sv.n_qubits, "cswap");
    detail::check_qubit(q1, sv.n_qubits, "cswap");
    detail::check_qubit(q2, sv.n_qubits, "cswap");
    detail::check_all_distinct({ctrl, q1, q2}, "cswap");
    std::array<int,3> qs = {ctrl, q1, q2};
    std::sort(qs.begin(), qs.end());
    const size_t s0    = 1ULL << qs[0];
    const size_t s1    = 1ULL << qs[1];
    const size_t s2    = 1ULL << qs[2];
    const size_t ctrlb = 1ULL << ctrl;
    const size_t q1b   = 1ULL << q1;
    const size_t q2b   = 1ULL << q2;
    const int n_outer  = static_cast<int>(sv.dim / (2 * s2));

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (int oi = 0; oi < n_outer; ++oi) {
        const size_t k2 = static_cast<size_t>(oi) * 2 * s2;
        for (size_t k1 = 0; k1 < s2; k1 += 2 * s1) {
            for (size_t k0 = 0; k0 < s1; k0 += 2 * s0) {
                for (size_t k = 0; k < s0; ++k) {
                    const size_t base = k2 + k1 + k0 + k;
                    const size_t i = base | ctrlb | q1b;  // ctrl=1, q1=1, q2=0
                    const size_t j = base | ctrlb | q2b;  // ctrl=1, q1=0, q2=1
                    std::swap(sv.real_parts[i], sv.real_parts[j]);
                    std::swap(sv.imag_parts[i], sv.imag_parts[j]);
                }
            }
        }
    }
}

// =============================================================================
// RCCX (Margolus / simplified Toffoli)
// Equivalent to CCX up to relative phases.
//
// Exact action in the |c1 c2 t⟩ basis (verified against the gate sequence
// below and pinned by the B4 regression tests):
//   |101⟩ → -|101⟩            (c1=1, c2=0, t=1 picks up a sign)
//   |110⟩ →  i|111⟩
//   |111⟩ → -i|110⟩           (the controlled-X swap carries the ±i pair)
//   all other basis states are unchanged.
//
// This matches the simplified Toffoli used in Qiskit:
// RCCX = H(tgt) . T(tgt) . CX(c2,tgt) . Tdg(tgt) . CX(c1,tgt) .
//        T(tgt) . CX(c2,tgt) . Tdg(tgt) . H(tgt)
//
// Applied as ONE three-level-stride pass over the dim/8
// base groups (every coefficient of the exact action above is +-1 or +-i, so
// the kernel is exact with zero floating-point rounding), replacing the
// 9-kernel ladder that swept the full statevector nine times. The ladder
// remains the reference decomposition in the transpiler and MPS paths.
// =============================================================================
void apply_rccx(Statevector& sv, int c1, int c2, int tgt) {
    detail::check_qubit(c1, sv.n_qubits, "rccx");
    detail::check_qubit(c2, sv.n_qubits, "rccx");
    detail::check_qubit(tgt, sv.n_qubits, "rccx");
    detail::check_all_distinct({c1, c2, tgt}, "rccx");
    std::array<int,3> qs = {c1, c2, tgt};
    std::sort(qs.begin(), qs.end());
    const size_t s0  = 1ULL << qs[0];
    const size_t s1  = 1ULL << qs[1];
    const size_t s2  = 1ULL << qs[2];
    const size_t c1b = 1ULL << c1;
    const size_t c2b = 1ULL << c2;
    const size_t tb  = 1ULL << tgt;
    const int n_outer = static_cast<int>(sv.dim / (2 * s2));

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (int oi = 0; oi < n_outer; ++oi) {
        const size_t k2 = static_cast<size_t>(oi) * 2 * s2;
        for (size_t k1 = 0; k1 < s2; k1 += 2 * s1) {
            for (size_t k0 = 0; k0 < s1; k0 += 2 * s0) {
                for (size_t k = 0; k < s0; ++k) {
                    const size_t base = k2 + k1 + k0 + k;
                    // |101>: c1=1, c2=0, t=1 picks up a -1 phase
                    const size_t i101 = base | c1b | tb;
                    sv.real_parts[i101] = -sv.real_parts[i101];
                    sv.imag_parts[i101] = -sv.imag_parts[i101];
                    // c1=1, c2=1 pair:
                    //   new|110> = -i * old|111>,  new|111> = i * old|110>
                    const size_t i110 = base | c1b | c2b;
                    const size_t i111 = i110 | tb;
                    const double r0 = sv.real_parts[i110], im0 = sv.imag_parts[i110];
                    const double r1 = sv.real_parts[i111], im1 = sv.imag_parts[i111];
                    sv.real_parts[i110] = im1;   // -i*(r1 + i*im1) = im1 - i*r1
                    sv.imag_parts[i110] = -r1;
                    sv.real_parts[i111] = -im0;  //  i*(r0 + i*im0) = -im0 + i*r0
                    sv.imag_parts[i111] = r0;
                }
            }
        }
    }
}

// =============================================================================
// Arbitrary N-qubit unitary
// =============================================================================
// Apply an arbitrary 2^k × 2^k unitary to k target qubits.
// The matrix is given in row-major order.

void apply_unitary(
    Statevector& sv,
    const std::vector<int>& targets,
    const std::vector<Complex128>& matrix_in,
    ValidationOptions validation
) {
    detail::check_qubits(targets, sv.n_qubits, "unitary");
    detail::check_all_distinct(targets, "unitary");
    const int k = static_cast<int>(targets.size());
    const size_t block_size = 1ULL << k;

    if (matrix_in.size() != block_size * block_size) {
        throw std::invalid_argument(
            "Unitary matrix size mismatch: expected " +
            std::to_string(block_size * block_size) +
            ", got " + std::to_string(matrix_in.size())
        );
    }

    // Under Fix the repair lands in `fixed` and `matrix` binds to it; under
    // every other policy `matrix` is the caller's operand and nothing is
    // copied. The caller's own matrix is never modified either way.
    std::vector<Complex128> fixed;
    const std::vector<Complex128>& matrix = detail::check_unitary_fixing(
        matrix_in, block_size, validation, "unitary", fixed);

    // Precompute target masks
    std::vector<size_t> target_masks(k);
    for (int i = 0; i < k; ++i) {
        target_masks[i] = 1ULL << targets[i];
    }

    // Sort targets to iterate in a canonical order
    std::vector<int> sorted_targets = targets;
    std::sort(sorted_targets.begin(), sorted_targets.end());

    // For each "background" index (indices of amplitudes not in the target qubits),
    // apply the matrix to the 2^k dimensional subspace.

    // Iterate over all indices, but only process each group once.
    // A group is identified by the bits NOT in the target set.

    // Count number of background groups
    size_t n_groups = sv.dim >> k;

    // Work-shape dispatch. Parallelising over background
    // groups starves when k is large: an n-qubit oracle has ONE group and ran
    // fully serial with O(4^n) work. With few groups but large blocks,
    // parallelise the ROW loop of the in-group multiply instead (rows are
    // independent once the group snapshot is taken).
    const bool par_groups = n_groups > (1 << 15);
    const bool par_rows =
        !par_groups && block_size >= (1 << 8) &&
        static_cast<unsigned long long>(n_groups) * block_size * block_size >=
            (1ULL << 20);

    if (par_rows) {
        std::vector<size_t> indices(block_size);
        std::vector<double> old_real(block_size);
        std::vector<double> old_imag(block_size);

        for (size_t g = 0; g < n_groups; ++g) {
            // Map group index g to a background index (zeros at target bits).
            size_t bg_idx = 0;
            size_t g_bits = g;
            int target_idx = 0;
            for (int b = 0; b < sv.n_qubits; ++b) {
                if (target_idx < k && b == sorted_targets[target_idx]) {
                    target_idx++;
                } else {
                    if (g_bits & 1) bg_idx |= (1ULL << b);
                    g_bits >>= 1;
                }
            }

            // Subspace indices + snapshot (serial: O(block_size * k), small
            // relative to the O(block_size^2) multiply below).
            for (size_t s = 0; s < block_size; ++s) {
                size_t idx = bg_idx;
                for (int ti = 0; ti < k; ++ti) {
                    if ((s >> ti) & 1) idx |= target_masks[ti];
                }
                indices[s] = idx;
                old_real[s] = sv.real_parts[idx];
                old_imag[s] = sv.imag_parts[idx];
            }

            #pragma omp parallel for schedule(static)
            for (int row = 0; row < static_cast<int>(block_size); ++row) {
                const Complex128* mrow =
                    matrix.data() + static_cast<size_t>(row) * block_size;
                double new_r = 0.0;
                double new_i = 0.0;
                for (size_t col = 0; col < block_size; ++col) {
                    new_r += mrow[col].real * old_real[col] - mrow[col].imag * old_imag[col];
                    new_i += mrow[col].real * old_imag[col] + mrow[col].imag * old_real[col];
                }
                sv.real_parts[indices[row]] = new_r;
                sv.imag_parts[indices[row]] = new_i;
            }
        }
        return;
    }

    // Thread-local buffers eliminate 3 heap allocations per OpenMP iteration.
    // Sized on first use; resized only when block_size grows.
    thread_local std::vector<size_t> tl_indices;
    thread_local std::vector<double> tl_old_real;
    thread_local std::vector<double> tl_old_imag;

    #pragma omp parallel for schedule(static) if(n_groups > (1<<15)) \
        firstprivate(block_size)
    for (int gg = 0; gg < static_cast<int>(n_groups); ++gg) {
        // Resize thread-local buffers only when needed
        if (tl_indices.size() < block_size) {
            tl_indices.resize(block_size);
            tl_old_real.resize(block_size);
            tl_old_imag.resize(block_size);
        }

        size_t g = gg;
        // Map group index g to a background index (zeros at all target bit positions)
        size_t bg_idx = 0;
        size_t g_bits = g;
        int target_idx = 0;

        for (int b = 0; b < sv.n_qubits; ++b) {
            if (target_idx < k && b == sorted_targets[target_idx]) {
                target_idx++;
            } else {
                if (g_bits & 1) bg_idx |= (1ULL << b);
                g_bits >>= 1;
            }
        }

        // Compute all 2^k indices in this subspace
        for (size_t s = 0; s < block_size; ++s) {
            size_t idx = bg_idx;
            for (int ti = 0; ti < k; ++ti) {
                if ((s >> ti) & 1) idx |= target_masks[ti];
            }
            tl_indices[s] = idx;
        }

        // Snapshot current amplitudes
        for (size_t s = 0; s < block_size; ++s) {
            tl_old_real[s] = sv.real_parts[tl_indices[s]];
            tl_old_imag[s] = sv.imag_parts[tl_indices[s]];
        }

        // Apply matrix: new[row] = sum_col matrix[row*block_size + col] * old[col]
        for (size_t row = 0; row < block_size; ++row) {
            double new_r = 0.0;
            double new_i = 0.0;
            for (size_t col = 0; col < block_size; ++col) {
                const Complex128& m = matrix[row * block_size + col];
                new_r += m.real * tl_old_real[col] - m.imag * tl_old_imag[col];
                new_i += m.real * tl_old_imag[col] + m.imag * tl_old_real[col];
            }
            sv.real_parts[tl_indices[row]] = new_r;
            sv.imag_parts[tl_indices[row]] = new_i;
        }
    }
}

// =============================================================================
// MCX — multi-controlled X. Flips `target` on every amplitude
// whose control qubits are all |1>. Replaces Grover's dense 2^n x 2^n
// diffusion matrix with an O(dim) strided pass over disjoint pairs.
// =============================================================================
void apply_mcx(Statevector& sv, const std::vector<int>& controls,
               int target) {
    detail::check_qubits(controls, sv.n_qubits, "mcx");
    detail::check_qubit(target, sv.n_qubits, "mcx");
    detail::check_all_distinct(controls, "mcx");
    detail::check_require(
        std::find(controls.begin(), controls.end(), target) == controls.end(),
        "mcx", "target must not be among the controls");
    size_t ctrl_mask = 0;
    for (int c : controls) ctrl_mask |= (1ULL << c);
    const size_t tbit = 1ULL << target;
    const size_t dim = sv.dim;
    double* __restrict__ rp = sv.real_parts;
    double* __restrict__ ip = sv.imag_parts;

    // Iterate indices with target bit 0; each visited pair (i, i|tbit) is
    // disjoint, so the parallel swaps do not race.
    #pragma omp parallel for schedule(static) if(dim >= (1<<20))
    for (long long ii = 0; ii < static_cast<long long>(dim); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        if ((i & tbit) != 0) continue;
        if ((i & ctrl_mask) != ctrl_mask) continue;
        const size_t j = i | tbit;
        std::swap(rp[i], rp[j]);
        std::swap(ip[i], ip[j]);
    }
}

// =============================================================================
// MCP — multi-controlled phase. Multiplies by exp(i*lambda) where every listed
// qubit is |1>.
// =============================================================================
void apply_mcp(Statevector& sv, const std::vector<int>& qubits,
               double lambda) {
    detail::check_qubits(qubits, sv.n_qubits, "mcp");
    detail::check_all_distinct(qubits, "mcp");
    size_t mask = 0;
    for (int q : qubits) mask |= (1ULL << q);
    const double c = std::cos(lambda), s = std::sin(lambda);
    const size_t dim = sv.dim;
    double* __restrict__ rp = sv.real_parts;
    double* __restrict__ ip = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(dim >= (1<<20))
    for (long long ii = 0; ii < static_cast<long long>(dim); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        if ((i & mask) != mask) continue;
        const double r = rp[i], im = ip[i];
        rp[i] = r * c - im * s;
        ip[i] = r * s + im * c;
    }
}

// =============================================================================
// PERMUTATION — apply |x> -> |perm[x]> on the target subspace.
// The reversible-classical-oracle building block: Shor's modular
// multiplication is a permutation, applied here as an O(dim) gather instead of
// a dense 2^k x 2^k matrix multiply.
// =============================================================================
void apply_permutation(Statevector& sv, const std::vector<int>& qubits,
                       const std::vector<int>& perm) {
    detail::check_qubits(qubits, sv.n_qubits, "permutation");
    detail::check_all_distinct(qubits, "permutation");
    const int k = static_cast<int>(qubits.size());
    const size_t sub_dim = size_t(1) << k;
    if (perm.size() != sub_dim)
        throw std::invalid_argument("apply_permutation: perm size != 2^k");
    // perm must be a bijection of [0, 2^k): an out-of-range image indexes
    // sub_off out of bounds (UB); a repeated image silently drops basis states.
    {
        std::vector<char> seen(sub_dim, 0);
        for (int v : perm) {
            if (v < 0 || static_cast<size_t>(v) >= sub_dim)
                detail::throw_structure("permutation",
                    "perm entry " + std::to_string(v) + " out of range [0, " +
                    std::to_string(sub_dim) + ")");
            if (seen[static_cast<size_t>(v)])
                detail::throw_structure("permutation",
                    "perm must be a bijection (repeated image " +
                    std::to_string(v) + ")");
            seen[static_cast<size_t>(v)] = 1;
        }
    }

    // Physical offset of each target sub-state (LSB = qubits[0]).
    std::vector<size_t> sub_off(sub_dim);
    for (size_t x = 0; x < sub_dim; ++x) {
        size_t o = 0;
        for (int b = 0; b < k; ++b)
            if ((x >> b) & 1) o |= (1ULL << qubits[static_cast<size_t>(b)]);
        sub_off[x] = o;
    }

    std::vector<int> is_tgt(static_cast<size_t>(sv.n_qubits), 0);
    for (int q : qubits) is_tgt[static_cast<size_t>(q)] = 1;
    const size_t n_bg = sv.dim >> k;

    double* __restrict__ rp = sv.real_parts;
    double* __restrict__ ip = sv.imag_parts;

    #pragma omp parallel
    {
        thread_local std::vector<double> tr, ti;
        if (tr.size() < sub_dim) { tr.resize(sub_dim); ti.resize(sub_dim); }
        #pragma omp for schedule(static)
        for (long long bgll = 0; bgll < static_cast<long long>(n_bg); ++bgll) {
            // Map compact group index -> background base (target bits zero).
            size_t base = 0, gbits = static_cast<size_t>(bgll);
            int src_bit = 0;
            for (int b = 0; b < sv.n_qubits; ++b) {
                if (!is_tgt[static_cast<size_t>(b)]) {
                    if ((gbits >> src_bit) & 1) base |= (1ULL << b);
                    ++src_bit;
                }
            }
            for (size_t x = 0; x < sub_dim; ++x) {
                tr[x] = rp[base + sub_off[x]];
                ti[x] = ip[base + sub_off[x]];
            }
            for (size_t x = 0; x < sub_dim; ++x) {
                const size_t dst = base + sub_off[static_cast<size_t>(perm[x])];
                rp[dst] = tr[x];
                ip[dst] = ti[x];
            }
        }
    }
}

} // namespace gates
} // namespace lindblad

