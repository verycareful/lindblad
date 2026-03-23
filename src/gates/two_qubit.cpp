#include "qpp/gates.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>

namespace qpp {
namespace gates {

// =============================================================================
// Cache-optimised two-qubit gate core
// =============================================================================
// All two-qubit gates share the same nested lo/hi-step loop structure.
// By iterating in block order (outer → hi_step, middle → lo_step, inner → contiguous),
// we get sequential memory access within cache lines and enable SIMD vectorisation.
// This eliminates the branch-per-element pattern of the naive approach.

// =============================================================================
// Helper: Controlled single-qubit gate (cache-optimised + SIMD)
// For each index where ctrl bit = 1:
//   Apply the 2x2 matrix [[a,b],[c,d]] to the target qubit subspace.
// =============================================================================

static inline void apply_controlled_matrix(
    Statevector& sv, int ctrl, int tgt,
    double ar, double ai,
    double br, double bi,
    double cr, double ci,
    double dr, double di
) noexcept {
    assert(ctrl != tgt);
    const size_t dim = sv.dim;
    const size_t ctrl_step = 1ULL << ctrl;
    const size_t tgt_step  = 1ULL << tgt;

    // Sort the two qubit positions to iterate in memory order
    const int lo = std::min(ctrl, tgt);
    const int hi = std::max(ctrl, tgt);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            // Within this block, process pairs where ctrl=1, tgt=0
            // The offset for ctrl=1 depends on whether ctrl is hi or lo
            size_t base = k + j;

            // Compute offsets: we need ctrl bit set, tgt bit clear
            // When ctrl > tgt: ctrl is hi, tgt is lo
            //   ctrl=1 → add hi_step; tgt=0 → don't add lo_step
            //   partner (tgt=1) → add lo_step
            // When tgt > ctrl: tgt is hi, ctrl is lo
            //   ctrl=1 → add lo_step; tgt=0 → don't add hi_step
            //   partner (tgt=1) → add hi_step
            size_t off_ctrl1_tgt0, off_partner;
            if (ctrl > tgt) {
                off_ctrl1_tgt0 = hi_step;       // ctrl(hi)=1, tgt(lo)=0
                off_partner    = lo_step;        // tgt(lo)=1
            } else {
                off_ctrl1_tgt0 = lo_step;        // ctrl(lo)=1, tgt(hi)=0
                off_partner    = hi_step;        // tgt(hi)=1
            }

            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx0 = base + i + off_ctrl1_tgt0;
                size_t idx1 = idx0 + off_partner;

                double r0 = sv.real_parts[idx0];
                double i0 = sv.imag_parts[idx0];
                double r1 = sv.real_parts[idx1];
                double i1 = sv.imag_parts[idx1];

                sv.real_parts[idx0] = (ar*r0 - ai*i0) + (br*r1 - bi*i1);
                sv.imag_parts[idx0] = (ar*i0 + ai*r0) + (br*i1 + bi*r1);
                sv.real_parts[idx1] = (cr*r0 - ci*i0) + (dr*r1 - di*i1);
                sv.imag_parts[idx1] = (cr*i0 + ci*r0) + (dr*i1 + di*r1);
            }
        }
    }
}

// =============================================================================
// Helper: Controlled phase — specialised diagonal path
// Only modifies amplitude when ctrl=1 AND tgt=1.
// No amplitude mixing → half the memory traffic of apply_controlled_matrix.
// =============================================================================

static inline void apply_controlled_phase(
    Statevector& sv, int ctrl, int tgt,
    double cos_p, double sin_p
) noexcept {
    assert(ctrl != tgt);
    const size_t dim = sv.dim;
    const int lo = std::min(ctrl, tgt);
    const int hi = std::max(ctrl, tgt);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            // Both bits set: offset = hi_step + lo_step
            size_t base = k + j + hi_step + lo_step;

            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx = base + i;
                double r = sv.real_parts[idx];
                double im = sv.imag_parts[idx];
                sv.real_parts[idx] = r * cos_p - im * sin_p;
                sv.imag_parts[idx] = r * sin_p + im * cos_p;
            }
        }
    }
}

// =============================================================================
// Helper: Two-qubit swap pair — cache-optimised loop for gates that swap
// amplitudes between |01⟩ and |10⟩ subspaces (SWAP, iSWAP, CX).
// Calls a lambda(idx_a, idx_b) for each pair.
// =============================================================================

// =============================================================================
// CX (CNOT): ctrl=1 flips target — cache-optimised
// =============================================================================
void apply_cx(Statevector& sv, int ctrl, int tgt) noexcept {
    assert(ctrl != tgt);
    const size_t dim = sv.dim;
    const int lo = std::min(ctrl, tgt);
    const int hi = std::max(ctrl, tgt);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            size_t base = k + j;
            size_t off_ctrl1_tgt0, off_partner;
            if (ctrl > tgt) {
                off_ctrl1_tgt0 = hi_step;
                off_partner    = lo_step;
            } else {
                off_ctrl1_tgt0 = lo_step;
                off_partner    = hi_step;
            }

            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx0 = base + i + off_ctrl1_tgt0;
                size_t idx1 = idx0 + off_partner;

                double tmp_r = sv.real_parts[idx0];
                double tmp_i = sv.imag_parts[idx0];
                sv.real_parts[idx0] = sv.real_parts[idx1];
                sv.imag_parts[idx0] = sv.imag_parts[idx1];
                sv.real_parts[idx1] = tmp_r;
                sv.imag_parts[idx1] = tmp_i;
            }
        }
    }
}

// =============================================================================
// CY: ctrl=1, apply Y to target
// Y = [[0, -i], [i, 0]]
// =============================================================================
void apply_cy(Statevector& sv, int ctrl, int tgt) noexcept {
    apply_controlled_matrix(sv, ctrl, tgt,
        0.0, 0.0,    // a = 0
        0.0, -1.0,   // b = -i
        0.0, 1.0,    // c = i
        0.0, 0.0     // d = 0
    );
}

// =============================================================================
// CZ: ctrl=1, tgt=1 → negate phase — specialised diagonal path
// =============================================================================
void apply_cz(Statevector& sv, int ctrl, int tgt) noexcept {
    assert(ctrl != tgt);
    const size_t dim = sv.dim;
    const int lo = std::min(ctrl, tgt);
    const int hi = std::max(ctrl, tgt);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            // Both bits = 1: offset = hi_step + lo_step
            size_t base = k + j + hi_step + lo_step;

            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx = base + i;
                sv.real_parts[idx] = -sv.real_parts[idx];
                sv.imag_parts[idx] = -sv.imag_parts[idx];
            }
        }
    }
}

// =============================================================================
// CH: ctrl=1, apply H to target
// =============================================================================
void apply_ch(Statevector& sv, int ctrl, int tgt) noexcept {
    apply_controlled_matrix(sv, ctrl, tgt,
        INV_SQRT2, 0.0,   // a = 1/sqrt2
        INV_SQRT2, 0.0,   // b = 1/sqrt2
        INV_SQRT2, 0.0,   // c = 1/sqrt2
       -INV_SQRT2, 0.0    // d = -1/sqrt2
    );
}

// =============================================================================
// SWAP: exchange q1 and q2 amplitudes — cache-optimised
// Only swaps |01⟩ <-> |10⟩ pairs
// =============================================================================
void apply_swap(Statevector& sv, int q1, int q2) noexcept {
    assert(q1 != q2);
    const size_t dim = sv.dim;
    const int lo = std::min(q1, q2);
    const int hi = std::max(q1, q2);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            // |01⟩: lo=1, hi=0 → base + lo_step
            // |10⟩: lo=0, hi=1 → base + hi_step
            size_t off_01 = k + j + lo_step;
            size_t off_10 = k + j + hi_step;

            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx_a = off_01 + i;
                size_t idx_b = off_10 + i;

                double tmp_r = sv.real_parts[idx_a];
                double tmp_i = sv.imag_parts[idx_a];
                sv.real_parts[idx_a] = sv.real_parts[idx_b];
                sv.imag_parts[idx_a] = sv.imag_parts[idx_b];
                sv.real_parts[idx_b] = tmp_r;
                sv.imag_parts[idx_b] = tmp_i;
            }
        }
    }
}

// =============================================================================
// iSWAP: swap |01⟩ <-> |10⟩ and multiply each by i — cache-optimised
// =============================================================================
void apply_iswap(Statevector& sv, int q1, int q2) noexcept {
    assert(q1 != q2);
    const size_t dim = sv.dim;
    const int lo = std::min(q1, q2);
    const int hi = std::max(q1, q2);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            size_t off_01 = k + j + lo_step;
            size_t off_10 = k + j + hi_step;

            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx_a = off_01 + i;  // |01⟩
                size_t idx_b = off_10 + i;  // |10⟩

                double r_a = sv.real_parts[idx_a];
                double i_a = sv.imag_parts[idx_a];
                double r_b = sv.real_parts[idx_b];
                double i_b = sv.imag_parts[idx_b];

                // |01⟩ gets i * old|10⟩: i*(r_b + i*i_b) = -i_b + i*r_b
                sv.real_parts[idx_a] = -i_b;
                sv.imag_parts[idx_a] = r_b;
                // |10⟩ gets i * old|01⟩: i*(r_a + i*i_a) = -i_a + i*r_a
                sv.real_parts[idx_b] = -i_a;
                sv.imag_parts[idx_b] = r_a;
            }
        }
    }
}

// =============================================================================
// CRX: controlled RX(theta)
// =============================================================================
void apply_crx(Statevector& sv, int ctrl, int tgt, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    apply_controlled_matrix(sv, ctrl, tgt,
        cos_half, 0.0,           // a = cos(t/2)
        0.0,     -sin_half,      // b = -i*sin(t/2)
        0.0,     -sin_half,      // c = -i*sin(t/2)
        cos_half, 0.0            // d = cos(t/2)
    );
}

// =============================================================================
// CRY: controlled RY(theta)
// =============================================================================
void apply_cry(Statevector& sv, int ctrl, int tgt, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    apply_controlled_matrix(sv, ctrl, tgt,
        cos_half,  0.0,
       -sin_half,  0.0,
        sin_half,  0.0,
        cos_half,  0.0
    );
}

// =============================================================================
// CRZ: controlled RZ(theta) — specialised diagonal path
// When ctrl=1: diag(exp(-i*t/2), exp(i*t/2)) on target
// =============================================================================
void apply_crz(Statevector& sv, int ctrl, int tgt, double theta) noexcept {
    assert(ctrl != tgt);
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;
    const int lo = std::min(ctrl, tgt);
    const int hi = std::max(ctrl, tgt);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            size_t base = k + j;

            // ctrl=1, tgt=0: apply exp(-i*t/2)
            size_t off_c1t0, off_c1t1;
            if (ctrl > tgt) {
                off_c1t0 = hi_step;                // ctrl(hi)=1, tgt(lo)=0
                off_c1t1 = hi_step + lo_step;      // ctrl(hi)=1, tgt(lo)=1
            } else {
                off_c1t0 = lo_step;                // ctrl(lo)=1, tgt(hi)=0
                off_c1t1 = lo_step + hi_step;      // ctrl(lo)=1, tgt(hi)=1
            }

            // tgt=0: multiply by exp(-i*t/2) = cos - i*sin
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx = base + off_c1t0 + i;
                double r = sv.real_parts[idx];
                double im = sv.imag_parts[idx];
                sv.real_parts[idx] = r * cos_half + im * sin_half;
                sv.imag_parts[idx] = im * cos_half - r * sin_half;
            }

            // tgt=1: multiply by exp(+i*t/2) = cos + i*sin
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx = base + off_c1t1 + i;
                double r = sv.real_parts[idx];
                double im = sv.imag_parts[idx];
                sv.real_parts[idx] = r * cos_half - im * sin_half;
                sv.imag_parts[idx] = im * cos_half + r * sin_half;
            }
        }
    }
}

// =============================================================================
// CP: controlled P(lambda) — uses specialised diagonal path
// =============================================================================
void apply_cp(Statevector& sv, int ctrl, int tgt, double lambda) noexcept {
    apply_controlled_phase(sv, ctrl, tgt, std::cos(lambda), std::sin(lambda));
}

// =============================================================================
// CU: controlled U(theta, phi, lambda) with global phase gamma
// =============================================================================
void apply_cu(Statevector& sv, int ctrl, int tgt,
              double theta, double phi, double lambda, double gamma) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const double cos_lam = std::cos(lambda);
    const double sin_lam = std::sin(lambda);
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);
    const double cos_pl = std::cos(phi + lambda);
    const double sin_pl = std::sin(phi + lambda);
    const double cos_g = std::cos(gamma);
    const double sin_g = std::sin(gamma);

    double ar = cos_g * cos_half;
    double ai = sin_g * cos_half;

    double gl_r = cos_g * cos_lam - sin_g * sin_lam;
    double gl_i = cos_g * sin_lam + sin_g * cos_lam;
    double br = -gl_r * sin_half;
    double bi = -gl_i * sin_half;

    double gp_r = cos_g * cos_phi - sin_g * sin_phi;
    double gp_i = cos_g * sin_phi + sin_g * cos_phi;
    double cr_val = gp_r * sin_half;
    double ci_val = gp_i * sin_half;

    double gpl_r = cos_g * cos_pl - sin_g * sin_pl;
    double gpl_i = cos_g * sin_pl + sin_g * cos_pl;
    double dr_val = gpl_r * cos_half;
    double di_val = gpl_i * cos_half;

    apply_controlled_matrix(sv, ctrl, tgt,
        ar, ai, br, bi, cr_val, ci_val, dr_val, di_val
    );
}

// =============================================================================
// ECR: Echoed cross-resonance gate — cache-optimised 4-index group
// (1/sqrt2) * [[0,0,1,i],[0,0,i,1],[1,-i,0,0],[-i,1,0,0]]
// =============================================================================
void apply_ecr(Statevector& sv, int q1, int q2) noexcept {
    assert(q1 != q2);
    const size_t dim = sv.dim;
    const int lo = std::min(q1, q2);
    const int hi = std::max(q1, q2);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    // Precompute offsets for 4 basis states |q1,q2⟩
    // Need to map bit positions to lo/hi positions
    size_t off_00 = 0;
    size_t off_01, off_10, off_11;
    if (q1 > q2) {
        // q1=hi, q2=lo
        off_01 = lo_step;                // q1=0, q2=1
        off_10 = hi_step;                // q1=1, q2=0
        off_11 = hi_step + lo_step;      // q1=1, q2=1
    } else {
        // q1=lo, q2=hi
        off_01 = hi_step;                // q1=0, q2=1
        off_10 = lo_step;                // q1=1, q2=0
        off_11 = lo_step + hi_step;      // q1=1, q2=1
    }

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t base = k + j + i;
                size_t i00 = base + off_00;
                size_t i01 = base + off_01;
                size_t i10 = base + off_10;
                size_t i11 = base + off_11;

                double r00 = sv.real_parts[i00], im00 = sv.imag_parts[i00];
                double r01 = sv.real_parts[i01], im01 = sv.imag_parts[i01];
                double r10 = sv.real_parts[i10], im10 = sv.imag_parts[i10];
                double r11 = sv.real_parts[i11], im11 = sv.imag_parts[i11];

                // Row 0 (|00⟩): 1/sqrt2 * (|10⟩ + i|11⟩)
                sv.real_parts[i00] = INV_SQRT2 * (r10 - im11);
                sv.imag_parts[i00] = INV_SQRT2 * (im10 + r11);

                // Row 1 (|01⟩): 1/sqrt2 * (i|10⟩ + |11⟩)
                sv.real_parts[i01] = INV_SQRT2 * (-im10 + r11);
                sv.imag_parts[i01] = INV_SQRT2 * (r10 + im11);

                // Row 2 (|10⟩): 1/sqrt2 * (|00⟩ - i|01⟩)
                sv.real_parts[i10] = INV_SQRT2 * (r00 + im01);
                sv.imag_parts[i10] = INV_SQRT2 * (im00 - r01);

                // Row 3 (|11⟩): 1/sqrt2 * (-i|00⟩ + |01⟩)
                sv.real_parts[i11] = INV_SQRT2 * (im00 + r01);
                sv.imag_parts[i11] = INV_SQRT2 * (-r00 + im01);
            }
        }
    }
}

// =============================================================================
// RZX(theta): exp(-i * theta/2 * Z*X) — cache-optimised
// =============================================================================
void apply_rzx(Statevector& sv, int q1, int q2, double theta) noexcept {
    assert(q1 != q2);
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;
    const int lo = std::min(q1, q2);
    const int hi = std::max(q1, q2);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    // Precompute offsets for the 4 computational basis states |q1,q2⟩
    size_t off_00 = 0, off_01, off_10, off_11;
    if (q1 > q2) {
        off_01 = lo_step; off_10 = hi_step; off_11 = hi_step + lo_step;
    } else {
        off_01 = hi_step; off_10 = lo_step; off_11 = lo_step + hi_step;
    }

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t base = k + j + i;

                // Process q1=0 pair: |00⟩ <-> |01⟩ with sign=-1
                {
                    size_t ia = base + off_00;
                    size_t ib = base + off_01;
                    double r0 = sv.real_parts[ia], i0 = sv.imag_parts[ia];
                    double r1 = sv.real_parts[ib], i1 = sv.imag_parts[ib];

                    sv.real_parts[ia] = cos_half * r0 - sin_half * i1;
                    sv.imag_parts[ia] = cos_half * i0 + sin_half * r1;
                    sv.real_parts[ib] = cos_half * r1 - sin_half * i0;
                    sv.imag_parts[ib] = cos_half * i1 + sin_half * r0;
                }

                // Process q1=1 pair: |10⟩ <-> |11⟩ with sign=+1
                {
                    size_t ia = base + off_10;
                    size_t ib = base + off_11;
                    double r0 = sv.real_parts[ia], i0 = sv.imag_parts[ia];
                    double r1 = sv.real_parts[ib], i1 = sv.imag_parts[ib];

                    sv.real_parts[ia] = cos_half * r0 + sin_half * i1;
                    sv.imag_parts[ia] = cos_half * i0 - sin_half * r1;
                    sv.real_parts[ib] = cos_half * r1 + sin_half * i0;
                    sv.imag_parts[ib] = cos_half * i1 - sin_half * r0;
                }
            }
        }
    }
}

// =============================================================================
// RXX(theta): exp(-i * theta/2 * X*X) — cache-optimised 4-index group
// =============================================================================
void apply_rxx(Statevector& sv, int q1, int q2, double theta) noexcept {
    assert(q1 != q2);
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;
    const int lo = std::min(q1, q2);
    const int hi = std::max(q1, q2);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    size_t off_00 = 0, off_01, off_10, off_11;
    if (q1 > q2) {
        off_01 = lo_step; off_10 = hi_step; off_11 = hi_step + lo_step;
    } else {
        off_01 = hi_step; off_10 = lo_step; off_11 = lo_step + hi_step;
    }

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t base = k + j + i;
                size_t i00 = base + off_00;
                size_t i01 = base + off_01;
                size_t i10 = base + off_10;
                size_t i11 = base + off_11;

                double r00 = sv.real_parts[i00], im00 = sv.imag_parts[i00];
                double r01 = sv.real_parts[i01], im01 = sv.imag_parts[i01];
                double r10 = sv.real_parts[i10], im10 = sv.imag_parts[i10];
                double r11 = sv.real_parts[i11], im11 = sv.imag_parts[i11];

                sv.real_parts[i00] = cos_half * r00 + sin_half * im11;
                sv.imag_parts[i00] = cos_half * im00 - sin_half * r11;

                sv.real_parts[i01] = cos_half * r01 + sin_half * im10;
                sv.imag_parts[i01] = cos_half * im01 - sin_half * r10;

                sv.real_parts[i10] = sin_half * im01 + cos_half * r10;
                sv.imag_parts[i10] = -sin_half * r01 + cos_half * im10;

                sv.real_parts[i11] = sin_half * im00 + cos_half * r11;
                sv.imag_parts[i11] = -sin_half * r00 + cos_half * im11;
            }
        }
    }
}

// =============================================================================
// RYY(theta): exp(-i * theta/2 * Y*Y) — cache-optimised 4-index group
// =============================================================================
void apply_ryy(Statevector& sv, int q1, int q2, double theta) noexcept {
    assert(q1 != q2);
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;
    const int lo = std::min(q1, q2);
    const int hi = std::max(q1, q2);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    size_t off_00 = 0, off_01, off_10, off_11;
    if (q1 > q2) {
        off_01 = lo_step; off_10 = hi_step; off_11 = hi_step + lo_step;
    } else {
        off_01 = hi_step; off_10 = lo_step; off_11 = lo_step + hi_step;
    }

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t base = k + j + i;
                size_t i00 = base + off_00;
                size_t i01 = base + off_01;
                size_t i10 = base + off_10;
                size_t i11 = base + off_11;

                double r00 = sv.real_parts[i00], im00 = sv.imag_parts[i00];
                double r01 = sv.real_parts[i01], im01 = sv.imag_parts[i01];
                double r10 = sv.real_parts[i10], im10 = sv.imag_parts[i10];
                double r11 = sv.real_parts[i11], im11 = sv.imag_parts[i11];

                sv.real_parts[i00] = cos_half * r00 - sin_half * im11;
                sv.imag_parts[i00] = cos_half * im00 + sin_half * r11;

                sv.real_parts[i01] = cos_half * r01 + sin_half * im10;
                sv.imag_parts[i01] = cos_half * im01 - sin_half * r10;

                sv.real_parts[i10] = sin_half * im01 + cos_half * r10;
                sv.imag_parts[i10] = -sin_half * r01 + cos_half * im10;

                sv.real_parts[i11] = -sin_half * im00 + cos_half * r11;
                sv.imag_parts[i11] = sin_half * r00 + cos_half * im11;
            }
        }
    }
}

// =============================================================================
// RZZ(theta): exp(-i * theta/2 * Z*Z) — diagonal, cache-optimised
// diag(e^{-it/2}, e^{it/2}, e^{it/2}, e^{-it/2})
// Parity-based: same-parity gets exp(-it/2), different-parity gets exp(+it/2)
// =============================================================================
void apply_rzz(Statevector& sv, int q1, int q2, double theta) noexcept {
    assert(q1 != q2);
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;
    const int lo = std::min(q1, q2);
    const int hi = std::max(q1, q2);
    const size_t lo_step = 1ULL << lo;
    const size_t hi_step = 1ULL << hi;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t k = 0; k < dim; k += 2 * hi_step) {
        for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
            size_t base = k + j;

            // |00⟩: parity=0, exp(-it/2) = cos - i*sin
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx = base + i;
                double r = sv.real_parts[idx];
                double im = sv.imag_parts[idx];
                sv.real_parts[idx] = r * cos_half + im * sin_half;
                sv.imag_parts[idx] = im * cos_half - r * sin_half;
            }

            // |01⟩: parity=1, exp(+it/2) = cos + i*sin
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx = base + lo_step + i;
                double r = sv.real_parts[idx];
                double im = sv.imag_parts[idx];
                sv.real_parts[idx] = r * cos_half - im * sin_half;
                sv.imag_parts[idx] = im * cos_half + r * sin_half;
            }

            // |10⟩: parity=1, exp(+it/2)
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx = base + hi_step + i;
                double r = sv.real_parts[idx];
                double im = sv.imag_parts[idx];
                sv.real_parts[idx] = r * cos_half - im * sin_half;
                sv.imag_parts[idx] = im * cos_half + r * sin_half;
            }

            // |11⟩: parity=0, exp(-it/2)
            #pragma omp simd aligned(sv.real_parts, sv.imag_parts: 64)
            for (size_t i = 0; i < lo_step; ++i) {
                size_t idx = base + hi_step + lo_step + i;
                double r = sv.real_parts[idx];
                double im = sv.imag_parts[idx];
                sv.real_parts[idx] = r * cos_half + im * sin_half;
                sv.imag_parts[idx] = im * cos_half - r * sin_half;
            }
        }
    }
}

} // namespace gates
} // namespace qpp
