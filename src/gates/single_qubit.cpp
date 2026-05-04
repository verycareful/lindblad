#include "lindblad/gates.hpp"

#include <cmath>

namespace lindblad {
namespace gates {

// =============================================================================
// Helper: Generic single-qubit gate application
// =============================================================================
// Applies 2x2 unitary [[a, b], [c, d]] to qubit q.
// For each pair (index with bit q=0, index with bit q=1):
//   new_0 = a * old_0 + b * old_1
//   new_1 = c * old_0 + d * old_1

static inline void apply_single_qubit_matrix(
    Statevector& sv, int qubit,
    double ar, double ai,  // a = matrix[0][0]
    double br, double bi,  // b = matrix[0][1]
    double cr, double ci,  // c = matrix[1][0]
    double dr, double di   // d = matrix[1][1]
) noexcept {
    const size_t step = 1ULL << qubit;
    double* __restrict__ real_ptr = sv.real_parts;
    double* __restrict__ imag_ptr = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (size_t i = 0; i < sv.dim; i += 2 * step) {
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i; j < i + step; ++j) {
            double r0 = real_ptr[j];
            double i0 = imag_ptr[j];
            double r1 = real_ptr[j + step];
            double i1 = imag_ptr[j + step];

            // new_0 = a * old_0 + b * old_1
            real_ptr[j]        = (ar*r0 - ai*i0) + (br*r1 - bi*i1);
            imag_ptr[j]        = (ar*i0 + ai*r0) + (br*i1 + bi*r1);

            // new_1 = c * old_0 + d * old_1
            real_ptr[j + step] = (cr*r0 - ci*i0) + (dr*r1 - di*i1);
            imag_ptr[j + step] = (cr*i0 + ci*r0) + (dr*i1 + di*r1);
        }
    }
}

// Helper: Apply diagonal gate diag(exp(i*phase0), exp(i*phase1)) to qubit q
static inline void apply_diagonal_phase(
    Statevector& sv, int qubit,
    double cos0, double sin0,  // cos/sin of phase0
    double cos1, double sin1   // cos/sin of phase1
) noexcept {
    const size_t step = 1ULL << qubit;
    double* __restrict__ real_ptr = sv.real_parts;
    double* __restrict__ imag_ptr = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (size_t i = 0; i < sv.dim; i += 2 * step) {
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i; j < i + step; ++j) {
            // Apply phase0 to qubit=0 amplitudes
            double r = real_ptr[j];
            double im = imag_ptr[j];
            real_ptr[j] = r * cos0 - im * sin0;
            imag_ptr[j] = r * sin0 + im * cos0;
        }

        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i + step; j < i + 2 * step; ++j) {
            // Apply phase1 to qubit=1 amplitudes
            double r = real_ptr[j];
            double im = imag_ptr[j];
            real_ptr[j] = r * cos1 - im * sin1;
            imag_ptr[j] = r * sin1 + im * cos1;
        }
    }
}

// =============================================================================
// Pauli X: [[0,1],[1,0]] — swap qubit=0 and qubit=1 amplitudes
// =============================================================================
void apply_x(Statevector& sv, int q) noexcept {
    const size_t step = 1ULL << q;
    double* __restrict__ real_ptr = sv.real_parts;
    double* __restrict__ imag_ptr = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (size_t i = 0; i < sv.dim; i += 2 * step) {
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i; j < i + step; ++j) {
            std::swap(real_ptr[j], real_ptr[j + step]);
            std::swap(imag_ptr[j], imag_ptr[j + step]);
        }
    }
}

// =============================================================================
// Pauli Y: [[0,-i],[i,0]]
// =============================================================================
void apply_y(Statevector& sv, int q) noexcept {
    const size_t step = 1ULL << q;
    double* __restrict__ real_ptr = sv.real_parts;
    double* __restrict__ imag_ptr = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (size_t i = 0; i < sv.dim; i += 2 * step) {
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i; j < i + step; ++j) {
            double r0 = real_ptr[j];
            double i0 = imag_ptr[j];
            double r1 = real_ptr[j + step];
            double i1 = imag_ptr[j + step];

            // new_0 = -i * old_1 = (i1, -r1)
            real_ptr[j]        = i1;
            imag_ptr[j]        = -r1;
            // new_1 = i * old_0 = (-i0, r0)
            real_ptr[j + step] = -i0;
            imag_ptr[j + step] = r0;
        }
    }
}

// =============================================================================
// Pauli Z: [[1,0],[0,-1]] — negate qubit=1 amplitudes
// =============================================================================
void apply_z(Statevector& sv, int q) noexcept {
    const size_t step = 1ULL << q;
    double* __restrict__ real_ptr = sv.real_parts;
    double* __restrict__ imag_ptr = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (size_t i = 0; i < sv.dim; i += 2 * step) {
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i + step; j < i + 2 * step; ++j) {
            real_ptr[j] = -real_ptr[j];
            imag_ptr[j] = -imag_ptr[j];
        }
    }
}

// =============================================================================
// Hadamard: (1/√2)[[1,1],[1,-1]]
// =============================================================================
void apply_h(Statevector& sv, int q) noexcept {
    const size_t step = 1ULL << q;
    double* __restrict__ real_ptr = sv.real_parts;
    double* __restrict__ imag_ptr = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (size_t i = 0; i < sv.dim; i += 2 * step) {
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i; j < i + step; ++j) {
            double r0 = real_ptr[j];
            double i0 = imag_ptr[j];
            double r1 = real_ptr[j + step];
            double i1 = imag_ptr[j + step];

            real_ptr[j]        = INV_SQRT2 * (r0 + r1);
            imag_ptr[j]        = INV_SQRT2 * (i0 + i1);
            real_ptr[j + step] = INV_SQRT2 * (r0 - r1);
            imag_ptr[j + step] = INV_SQRT2 * (i0 - i1);
        }
    }
}

// =============================================================================
// S gate: [[1,0],[0,i]] — phase gate with lambda=pi/2
// =============================================================================
void apply_s(Statevector& sv, int q) noexcept {
    const size_t step = 1ULL << q;
    double* __restrict__ real_ptr = sv.real_parts;
    double* __restrict__ imag_ptr = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (size_t i = 0; i < sv.dim; i += 2 * step) {
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i + step; j < i + 2 * step; ++j) {
            // Multiply by i: (r + im*i) * i = -im + r*i
            double r = real_ptr[j];
            double im = imag_ptr[j];
            real_ptr[j] = -im;
            imag_ptr[j] = r;
        }
    }
}

// =============================================================================
// S† gate: [[1,0],[0,-i]]
// =============================================================================
void apply_sdg(Statevector& sv, int q) noexcept {
    const size_t step = 1ULL << q;
    double* __restrict__ real_ptr = sv.real_parts;
    double* __restrict__ imag_ptr = sv.imag_parts;

    #pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
    for (size_t i = 0; i < sv.dim; i += 2 * step) {
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t j = i + step; j < i + 2 * step; ++j) {
            // Multiply by -i: (r + im*i) * (-i) = im - r*i
            double r = real_ptr[j];
            double im = imag_ptr[j];
            real_ptr[j] = im;
            imag_ptr[j] = -r;
        }
    }
}

// =============================================================================
// T gate: [[1,0],[0,exp(i*pi/4)]]
// =============================================================================
void apply_t(Statevector& sv, int q) noexcept {
    // exp(i*pi/4) = cos(pi/4) + i*sin(pi/4) = (1+i)/sqrt(2)
    apply_diagonal_phase(sv, q, 1.0, 0.0, INV_SQRT2, INV_SQRT2);
}

// =============================================================================
// T† gate: [[1,0],[0,exp(-i*pi/4)]]
// =============================================================================
void apply_tdg(Statevector& sv, int q) noexcept {
    apply_diagonal_phase(sv, q, 1.0, 0.0, INV_SQRT2, -INV_SQRT2);
}

// =============================================================================
// SX (sqrt X): 0.5 * [[1+i, 1-i], [1-i, 1+i]]
// =============================================================================
void apply_sx(Statevector& sv, int q) noexcept {
    // Matrix: 0.5 * [[1+i, 1-i], [1-i, 1+i]]
    apply_single_qubit_matrix(sv, q,
        0.5,  0.5,   // a = (1+i)/2
        0.5, -0.5,   // b = (1-i)/2
        0.5, -0.5,   // c = (1-i)/2
        0.5,  0.5    // d = (1+i)/2
    );
}

// =============================================================================
// SX† (sqrt X dagger): 0.5 * [[1-i, 1+i], [1+i, 1-i]]
// =============================================================================
void apply_sxdg(Statevector& sv, int q) noexcept {
    apply_single_qubit_matrix(sv, q,
        0.5, -0.5,   // a = (1-i)/2
        0.5,  0.5,   // b = (1+i)/2
        0.5,  0.5,   // c = (1+i)/2
        0.5, -0.5    // d = (1-i)/2
    );
}

// =============================================================================
// RX(theta): [[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]
// =============================================================================
void apply_rx(Statevector& sv, int q, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // Matrix: [[cos, -i*sin], [-i*sin, cos]]
    apply_single_qubit_matrix(sv, q,
        cos_half, 0.0,           // a = cos(t/2)
        0.0,     -sin_half,      // b = -i*sin(t/2)
        0.0,     -sin_half,      // c = -i*sin(t/2)
        cos_half, 0.0            // d = cos(t/2)
    );
}

// =============================================================================
// RY(theta): [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
// =============================================================================
void apply_ry(Statevector& sv, int q, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    apply_single_qubit_matrix(sv, q,
        cos_half,  0.0,          // a = cos(t/2)
       -sin_half,  0.0,          // b = -sin(t/2)
        sin_half,  0.0,          // c = sin(t/2)
        cos_half,  0.0           // d = cos(t/2)
    );
}

// =============================================================================
// RZ(theta): [[exp(-i*t/2), 0], [0, exp(i*t/2)]]
// =============================================================================
void apply_rz(Statevector& sv, int q, double theta) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // phase0 = exp(-i*t/2) = cos(t/2) - i*sin(t/2)
    // phase1 = exp(+i*t/2) = cos(t/2) + i*sin(t/2)
    apply_diagonal_phase(sv, q, cos_half, -sin_half, cos_half, sin_half);
}

// =============================================================================
// Phase gate P(lambda): [[1, 0], [0, exp(i*lambda)]]
// =============================================================================
void apply_p(Statevector& sv, int q, double lambda) noexcept {
    apply_diagonal_phase(sv, q, 1.0, 0.0, std::cos(lambda), std::sin(lambda));
}

// =============================================================================
// U gate (general single-qubit unitary):
// U(theta, phi, lambda) = [[cos(t/2), -exp(i*lam)*sin(t/2)],
//                          [exp(i*phi)*sin(t/2), exp(i*(phi+lam))*cos(t/2)]]
// =============================================================================
void apply_u(Statevector& sv, int q,
             double theta, double phi, double lambda) noexcept {
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);
    const double cos_lam = std::cos(lambda);
    const double sin_lam = std::sin(lambda);
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);
    const double cos_pl = std::cos(phi + lambda);
    const double sin_pl = std::sin(phi + lambda);

    // a = cos(t/2)
    // b = -exp(i*lam)*sin(t/2) = -sin(t/2)*(cos(lam) + i*sin(lam))
    // c = exp(i*phi)*sin(t/2) = sin(t/2)*(cos(phi) + i*sin(phi))
    // d = exp(i*(phi+lam))*cos(t/2) = cos(t/2)*(cos(phi+lam) + i*sin(phi+lam))

    apply_single_qubit_matrix(sv, q,
        cos_half, 0.0,
        -sin_half * cos_lam, -sin_half * sin_lam,
         sin_half * cos_phi,  sin_half * sin_phi,
         cos_half * cos_pl,   cos_half * sin_pl
    );
}

// =============================================================================
// U1(lambda) = P(lambda) = [[1, 0], [0, exp(i*lambda)]]
// =============================================================================
void apply_u1(Statevector& sv, int q, double lambda) noexcept {
    apply_p(sv, q, lambda);
}

// =============================================================================
// U2(phi, lambda) = U(pi/2, phi, lambda)
// = (1/sqrt(2)) * [[1, -exp(i*lam)], [exp(i*phi), exp(i*(phi+lam))]]
// =============================================================================
void apply_u2(Statevector& sv, int q, double phi, double lambda) noexcept {
    apply_u(sv, q, PI_2, phi, lambda);
}

// =============================================================================
// U3(theta, phi, lambda) = U(theta, phi, lambda)
// =============================================================================
void apply_u3(Statevector& sv, int q,
              double theta, double phi, double lambda) noexcept {
    apply_u(sv, q, theta, phi, lambda);
}

} // namespace gates
} // namespace lindblad

