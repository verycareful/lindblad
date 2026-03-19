#include "qpp/gates.hpp"

#include <cmath>
#include <algorithm>

namespace qpp {
namespace gates {

// =============================================================================
// Helper: Controlled single-qubit gate
// =============================================================================
// For each index where ctrl bit = 1:
//   Apply the 2x2 matrix [[a,b],[c,d]] to the target qubit subspace.

static inline void apply_controlled_matrix(
    Statevector& sv, int ctrl, int tgt,
    double ar, double ai,
    double br, double bi,
    double cr, double ci,
    double dr, double di
) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        // Only act when ctrl=1 and tgt=0
        if (((i >> ctrl) & 1) && !((i >> tgt) & 1)) {
            size_t j = i | (1ULL << tgt);  // partner with tgt=1

            double r0 = sv.real_parts[i];
            double i0 = sv.imag_parts[i];
            double r1 = sv.real_parts[j];
            double i1 = sv.imag_parts[j];

            sv.real_parts[i] = (ar*r0 - ai*i0) + (br*r1 - bi*i1);
            sv.imag_parts[i] = (ar*i0 + ai*r0) + (br*i1 + bi*r1);
            sv.real_parts[j] = (cr*r0 - ci*i0) + (dr*r1 - di*i1);
            sv.imag_parts[j] = (cr*i0 + ci*r0) + (dr*i1 + di*r1);
        }
    }
}

// Helper: Controlled phase (only modify phase when ctrl=1, tgt=1)
static inline void apply_controlled_phase(
    Statevector& sv, int ctrl, int tgt,
    double cos_p, double sin_p
) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        if (((i >> ctrl) & 1) && ((i >> tgt) & 1)) {
            double r = sv.real_parts[i];
            double im = sv.imag_parts[i];
            sv.real_parts[i] = r * cos_p - im * sin_p;
            sv.imag_parts[i] = r * sin_p + im * cos_p;
        }
    }
}

// =============================================================================
// CX (CNOT): ctrl=1 flips target
// =============================================================================
void apply_cx(Statevector& sv, int ctrl, int tgt) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        // Act when ctrl=1 and tgt=0
        if (((i >> ctrl) & 1) && !((i >> tgt) & 1)) {
            size_t j = i | (1ULL << tgt);
            std::swap(sv.real_parts[i], sv.real_parts[j]);
            std::swap(sv.imag_parts[i], sv.imag_parts[j]);
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
// CZ: ctrl=1, tgt=1 → negate phase
// =============================================================================
void apply_cz(Statevector& sv, int ctrl, int tgt) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        if (((i >> ctrl) & 1) && ((i >> tgt) & 1)) {
            sv.real_parts[i] = -sv.real_parts[i];
            sv.imag_parts[i] = -sv.imag_parts[i];
        }
    }
}

// =============================================================================
// CH: ctrl=1, apply H to target
// =============================================================================
void apply_ch(Statevector& sv, int ctrl, int tgt) noexcept {
    apply_controlled_matrix(sv, ctrl, tgt,
        INV_SQRT2, 0.0,   // a = 1/√2
        INV_SQRT2, 0.0,   // b = 1/√2
        INV_SQRT2, 0.0,   // c = 1/√2
       -INV_SQRT2, 0.0    // d = -1/√2
    );
}

// =============================================================================
// SWAP: exchange q1 and q2 amplitudes
// =============================================================================
void apply_swap(Statevector& sv, int q1, int q2) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        int b1 = (i >> q1) & 1;
        int b2 = (i >> q2) & 1;
        // Only swap when bits differ: (q1=0,q2=1) <-> (q1=1,q2=0)
        if (b1 == 0 && b2 == 1) {
            size_t j = (i | (1ULL << q1)) & ~(1ULL << q2);
            std::swap(sv.real_parts[i], sv.real_parts[j]);
            std::swap(sv.imag_parts[i], sv.imag_parts[j]);
        }
    }
}

// =============================================================================
// iSWAP:
// [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
// In the |q1,q2⟩ = |01⟩ and |10⟩ subspace: swap and multiply by i
// =============================================================================
void apply_iswap(Statevector& sv, int q1, int q2) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        int b1 = (i >> q1) & 1;
        int b2 = (i >> q2) & 1;
        if (b1 == 0 && b2 == 1) {
            size_t j = (i | (1ULL << q1)) & ~(1ULL << q2);

            double r_i = sv.real_parts[i];
            double im_i = sv.imag_parts[i];
            double r_j = sv.real_parts[j];
            double im_j = sv.imag_parts[j];

            // amp_i (|01⟩) → i * amp_j (gets i * old |10⟩ value)
            sv.real_parts[i] = -im_j;
            sv.imag_parts[i] = r_j;
            // amp_j (|10⟩) → i * amp_i (gets i * old |01⟩ value)
            sv.real_parts[j] = -im_i;
            sv.imag_parts[j] = r_i;
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
// CRZ: controlled RZ(theta)
// =============================================================================
void apply_crz(Statevector& sv, int ctrl, int tgt, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // When ctrl=1: diag(exp(-i*t/2), exp(i*t/2)) on target
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        if (!((i >> ctrl) & 1)) continue;

        double r = sv.real_parts[i];
        double im = sv.imag_parts[i];

        if (!((i >> tgt) & 1)) {
            // Target = 0: multiply by exp(-i*t/2)
            sv.real_parts[i] = r * cos_half + im * sin_half;
            sv.imag_parts[i] = im * cos_half - r * sin_half;
        } else {
            // Target = 1: multiply by exp(i*t/2)
            sv.real_parts[i] = r * cos_half - im * sin_half;
            sv.imag_parts[i] = im * cos_half + r * sin_half;
        }
    }
}

// =============================================================================
// CP: controlled P(lambda)
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

    // U with global phase: exp(i*gamma) * U(theta, phi, lambda)
    // a = exp(i*gamma) * cos(t/2)
    // b = -exp(i*(gamma+lambda)) * sin(t/2)
    // c = exp(i*(gamma+phi)) * sin(t/2)
    // d = exp(i*(gamma+phi+lambda)) * cos(t/2)

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
// ECR: Echoed cross-resonance gate
// (1/√2) * [[0,0,1,i],[0,0,i,1],[1,-i,0,0],[-i,1,0,0]]
// =============================================================================
void apply_ecr(Statevector& sv, int q1, int q2) noexcept {
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        int b1 = (i >> q1) & 1;
        int b2 = (i >> q2) & 1;
        // Only process once per 4-index group: require b1=0, b2=0
        if (b1 != 0 || b2 != 0) continue;

        size_t i00 = i;
        size_t i01 = i | (1ULL << q2);
        size_t i10 = i | (1ULL << q1);
        size_t i11 = i | (1ULL << q1) | (1ULL << q2);

        double r00 = sv.real_parts[i00], im00 = sv.imag_parts[i00];
        double r01 = sv.real_parts[i01], im01 = sv.imag_parts[i01];
        double r10 = sv.real_parts[i10], im10 = sv.imag_parts[i10];
        double r11 = sv.real_parts[i11], im11 = sv.imag_parts[i11];

        // Row 0 (|00⟩): 1/√2 * (|10⟩ + i|11⟩)
        sv.real_parts[i00] = INV_SQRT2 * (r10 - im11);
        sv.imag_parts[i00] = INV_SQRT2 * (im10 + r11);

        // Row 1 (|01⟩): 1/√2 * (i|10⟩ + |11⟩)
        sv.real_parts[i01] = INV_SQRT2 * (-im10 + r11);
        sv.imag_parts[i01] = INV_SQRT2 * (r10 + im11);

        // Row 2 (|10⟩): 1/√2 * (|00⟩ - i|01⟩)
        sv.real_parts[i10] = INV_SQRT2 * (r00 + im01);
        sv.imag_parts[i10] = INV_SQRT2 * (im00 - r01);

        // Row 3 (|11⟩): 1/√2 * (-i|00⟩ + |01⟩)
        sv.real_parts[i11] = INV_SQRT2 * (im00 + r01);
        sv.imag_parts[i11] = INV_SQRT2 * (-r00 + im01);
    }
}

// =============================================================================
// RZX(theta): exp(-i * theta/2 * Z⊗X)
// [[cos(t/2), -i*sin(t/2), 0, 0],
//  [-i*sin(t/2), cos(t/2), 0, 0],
//  [0, 0, cos(t/2), i*sin(t/2)],
//  [0, 0, i*sin(t/2), cos(t/2)]]
// =============================================================================
void apply_rzx(Statevector& sv, int q1, int q2, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        int b1 = (i >> q1) & 1;
        int b2 = (i >> q2) & 1;
        if (b2 != 0) continue;  // Process pairs with q2=0

        size_t j = i | (1ULL << q2);  // q2=1 partner

        double r0 = sv.real_parts[i];
        double i0 = sv.imag_parts[i];
        double r1 = sv.real_parts[j];
        double i1 = sv.imag_parts[j];

        double sign = (b1 == 0) ? -1.0 : 1.0;

        // For q1=0: [[cos, -i*sin], [-i*sin, cos]]
        // For q1=1: [[cos, i*sin], [i*sin, cos]]
        sv.real_parts[i] = cos_half * r0 + sign * sin_half * i1;
        sv.imag_parts[i] = cos_half * i0 - sign * sin_half * r1;
        sv.real_parts[j] = cos_half * r1 + sign * sin_half * i0;
        sv.imag_parts[j] = cos_half * i1 - sign * sin_half * r0;
    }
}

// =============================================================================
// RXX(theta): exp(-i * theta/2 * X⊗X)
// =============================================================================
void apply_rxx(Statevector& sv, int q1, int q2, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        int b1 = (i >> q1) & 1;
        int b2 = (i >> q2) & 1;
        if (b1 != 0 || b2 != 0) continue;

        size_t i00 = i;
        size_t i01 = i | (1ULL << q2);
        size_t i10 = i | (1ULL << q1);
        size_t i11 = i | (1ULL << q1) | (1ULL << q2);

        double r00 = sv.real_parts[i00], im00 = sv.imag_parts[i00];
        double r01 = sv.real_parts[i01], im01 = sv.imag_parts[i01];
        double r10 = sv.real_parts[i10], im10 = sv.imag_parts[i10];
        double r11 = sv.real_parts[i11], im11 = sv.imag_parts[i11];

        // RXX: |00⟩ → cos|00⟩ - i*sin|11⟩
        //       |01⟩ → cos|01⟩ - i*sin|10⟩
        //       |10⟩ → -i*sin|01⟩ + cos|10⟩
        //       |11⟩ → -i*sin|00⟩ + cos|11⟩
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

// =============================================================================
// RYY(theta): exp(-i * theta/2 * Y⊗Y)
// =============================================================================
void apply_ryy(Statevector& sv, int q1, int q2, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        int b1 = (i >> q1) & 1;
        int b2 = (i >> q2) & 1;
        if (b1 != 0 || b2 != 0) continue;

        size_t i00 = i;
        size_t i01 = i | (1ULL << q2);
        size_t i10 = i | (1ULL << q1);
        size_t i11 = i | (1ULL << q1) | (1ULL << q2);

        double r00 = sv.real_parts[i00], im00 = sv.imag_parts[i00];
        double r01 = sv.real_parts[i01], im01 = sv.imag_parts[i01];
        double r10 = sv.real_parts[i10], im10 = sv.imag_parts[i10];
        double r11 = sv.real_parts[i11], im11 = sv.imag_parts[i11];

        // RYY: |00⟩ → cos|00⟩ + i*sin|11⟩
        //       |01⟩ → cos|01⟩ - i*sin|10⟩
        //       |10⟩ → -i*sin|01⟩ + cos|10⟩
        //       |11⟩ → i*sin|00⟩ + cos|11⟩
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

// =============================================================================
// RZZ(theta): exp(-i * theta/2 * Z⊗Z)
// Diagonal: diag(e^{-it/2}, e^{it/2}, e^{it/2}, e^{-it/2})
// =============================================================================
void apply_rzz(Statevector& sv, int q1, int q2, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const size_t dim = sv.dim;

    #pragma omp parallel for schedule(static) if(dim > (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        int b1 = (i >> q1) & 1;
        int b2 = (i >> q2) & 1;
        int parity = b1 ^ b2;

        double r = sv.real_parts[i];
        double im = sv.imag_parts[i];

        if (parity == 0) {
            // exp(-i*t/2): cos - i*sin
            sv.real_parts[i] = r * cos_half + im * sin_half;
            sv.imag_parts[i] = im * cos_half - r * sin_half;
        } else {
            // exp(+i*t/2): cos + i*sin
            sv.real_parts[i] = r * cos_half - im * sin_half;
            sv.imag_parts[i] = im * cos_half + r * sin_half;
        }
    }
}

} // namespace gates
} // namespace qpp
