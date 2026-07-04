#include "lindblad/gates.hpp"

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
void apply_ccx(Statevector& sv, int c1, int c2, int tgt) noexcept {
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
void apply_ccz(Statevector& sv, int c1, int c2, int tgt) noexcept {
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
void apply_cswap(Statevector& sv, int ctrl, int q1, int q2) noexcept {
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
// This is the simplified Toffoli used in Qiskit:
// RCCX = H(tgt) . T(tgt) . CX(c2,tgt) . Tdg(tgt) . CX(c1,tgt) .
//        T(tgt) . CX(c2,tgt) . Tdg(tgt) . H(tgt)
// =============================================================================
void apply_rccx(Statevector& sv, int c1, int c2, int tgt) noexcept {
    // For simplicity and correctness, implement as the gate sequence:
    // H(tgt), T(tgt), CX(c2,tgt), Tdg(tgt), CX(c1,tgt),
    // T(tgt), CX(c2,tgt), Tdg(tgt), H(tgt)
    apply_h(sv, tgt);
    apply_t(sv, tgt);
    apply_cx(sv, c2, tgt);
    apply_tdg(sv, tgt);
    apply_cx(sv, c1, tgt);
    apply_t(sv, tgt);
    apply_cx(sv, c2, tgt);
    apply_tdg(sv, tgt);
    apply_h(sv, tgt);
}

// =============================================================================
// Arbitrary N-qubit unitary
// =============================================================================
// Apply an arbitrary 2^k × 2^k unitary to k target qubits.
// The matrix is given in row-major order.

void apply_unitary(
    Statevector& sv,
    const std::vector<int>& targets,
    const std::vector<Complex128>& matrix
) {
    const int k = static_cast<int>(targets.size());
    const size_t block_size = 1ULL << k;

    if (matrix.size() != block_size * block_size) {
        throw std::invalid_argument(
            "Unitary matrix size mismatch: expected " +
            std::to_string(block_size * block_size) +
            ", got " + std::to_string(matrix.size())
        );
    }

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

} // namespace gates
} // namespace lindblad

