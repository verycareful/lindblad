#include "qpp/gates.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qpp {
namespace gates {

// =============================================================================
// CCX (Toffoli): flip target when both controls are 1
// =============================================================================
void apply_ccx(Statevector& sv, int c1, int c2, int tgt) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        // Act when c1=1, c2=1, tgt=0
        if (((i >> c1) & 1) && ((i >> c2) & 1) && !((i >> tgt) & 1)) {
            size_t j = i | (1ULL << tgt);
            std::swap(sv.real_parts[i], sv.real_parts[j]);
            std::swap(sv.imag_parts[i], sv.imag_parts[j]);
        }
    }
}

// =============================================================================
// CCZ: negate phase when all three qubits are 1
// =============================================================================
void apply_ccz(Statevector& sv, int c1, int c2, int tgt) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        if (((i >> c1) & 1) && ((i >> c2) & 1) && ((i >> tgt) & 1)) {
            sv.real_parts[i] = -sv.real_parts[i];
            sv.imag_parts[i] = -sv.imag_parts[i];
        }
    }
}

// =============================================================================
// CSWAP (Fredkin): swap q1, q2 when ctrl=1
// =============================================================================
void apply_cswap(Statevector& sv, int ctrl, int q1, int q2) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        // Act when ctrl=1, q1 and q2 bits differ
        if (!((i >> ctrl) & 1)) continue;
        int b1 = (i >> q1) & 1;
        int b2 = (i >> q2) & 1;
        if (b1 == 0 && b2 == 1) {
            size_t j = (i | (1ULL << q1)) & ~(1ULL << q2);
            std::swap(sv.real_parts[i], sv.real_parts[j]);
            std::swap(sv.imag_parts[i], sv.imag_parts[j]);
        }
    }
}

// =============================================================================
// RCCX (Margolus / simplified Toffoli)
// Equivalent to CCX up to relative phase on |101⟩ and |111⟩
//
// Matrix in computational basis (sorted by c1,c2,tgt):
// Acts as identity except in the c1=1,c2=1 subspace where it applies
// a simplified X rotation. The exact implementation:
//   |110⟩ → |111⟩
//   |111⟩ → |110⟩
// with a relative phase on |101⟩ (multiply by i) and |100⟩ (multiply by -i)
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

    // Create a mask of all non-target bits
    size_t target_mask_all = 0;
    for (int i = 0; i < k; ++i) {
        target_mask_all |= (1ULL << targets[i]);
    }

    // Count number of background groups
    size_t n_groups = sv.dim >> k;

    #pragma omp parallel for schedule(static) if(n_groups > (1<<15))
    for (size_t g = 0; g < n_groups; ++g) {
        // Map group index g to a background index
        // Insert zeros at all target bit positions
        size_t bg_idx = 0;
        size_t g_bits = g;
        int bit_pos = 0;
        int target_idx = 0;

        for (int b = 0; b < sv.n_qubits; ++b) {
            if (target_idx < k && b == sorted_targets[target_idx]) {
                // This is a target bit — skip it (set to 0)
                target_idx++;
            } else {
                // Background bit — take from g_bits
                if (g_bits & 1) {
                    bg_idx |= (1ULL << b);
                }
                g_bits >>= 1;
            }
        }

        // Now bg_idx has zeros at all target positions.
        // Compute all 2^k indices in this subspace
        std::vector<size_t> indices(block_size);
        for (size_t s = 0; s < block_size; ++s) {
            size_t idx = bg_idx;
            for (int ti = 0; ti < k; ++ti) {
                if ((s >> ti) & 1) {
                    idx |= target_masks[ti];
                }
            }
            indices[s] = idx;
        }

        // Read current amplitudes
        std::vector<double> old_real(block_size);
        std::vector<double> old_imag(block_size);
        for (size_t s = 0; s < block_size; ++s) {
            old_real[s] = sv.real_parts[indices[s]];
            old_imag[s] = sv.imag_parts[indices[s]];
        }

        // Apply matrix: new[row] = sum_col matrix[row*block_size + col] * old[col]
        for (size_t row = 0; row < block_size; ++row) {
            double new_r = 0.0;
            double new_i = 0.0;
            for (size_t col = 0; col < block_size; ++col) {
                const Complex128& m = matrix[row * block_size + col];
                new_r += m.real * old_real[col] - m.imag * old_imag[col];
                new_i += m.real * old_imag[col] + m.imag * old_real[col];
            }
            sv.real_parts[indices[row]] = new_r;
            sv.imag_parts[indices[row]] = new_i;
        }
    }
}

} // namespace gates
} // namespace qpp
