// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/gates.hpp"

#include "lindblad/detail/sv_groups.hpp"
#include "lindblad/detail/validate.hpp"

#include <cmath>
#include <utility>

namespace lindblad {
namespace gates {

// =============================================================================
// One-qubit sweep: the gate as one formula over every pair
// =============================================================================
// A pair is member 0 (bit q clear) and member 1 (bit q set). The kernel's
// formula g(r, i) runs on doubles or on Lane8 blocks of eight pairs, whichever
// detail::for_each_group picks for q (sv_groups.hpp); Touched names the
// members g reads or writes.
inline constexpr unsigned kPairBoth = 0b11;  // members 0 and 1
inline constexpr unsigned kPairSet = 0b10;   // member 1 only

template <unsigned Touched, class G>
static inline void for_each_pair(Statevector& sv, int q, const G& g) {
    const int qs[1] = {q};
    detail::for_each_group<1, Touched>(sv.real_parts, sv.imag_parts, sv.dim, qs, g);
}

// A diagonal gate scales each member on its own: d(x, r, i) updates member x
// (a std::integral_constant) in place (detail::for_each_diagonal).
template <unsigned Touched, class D>
static inline void for_each_pair_diagonal(Statevector& sv, int q, const D& d) {
    const int qs[1] = {q};
    detail::for_each_diagonal<1, Touched>(sv.real_parts, sv.imag_parts, sv.dim, qs, d);
}

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
) {
    for_each_pair<kPairBoth>(sv, qubit, [=](auto* r, auto* i) {
        const auto r0 = r[0];
        const auto i0 = i[0];
        const auto r1 = r[1];
        const auto i1 = i[1];

        // new_0 = a * old_0 + b * old_1
        r[0] = (ar*r0 - ai*i0) + (br*r1 - bi*i1);
        i[0] = (ar*i0 + ai*r0) + (br*i1 + bi*r1);

        // new_1 = c * old_0 + d * old_1
        r[1] = (cr*r0 - ci*i0) + (dr*r1 - di*i1);
        i[1] = (cr*i0 + ci*r0) + (dr*i1 + di*r1);
    });
}

// Helper: Apply diagonal gate diag(exp(i*phase0), exp(i*phase1)) to qubit q
static inline void apply_diagonal_phase(
    Statevector& sv, int qubit,
    double cos0, double sin0,  // cos/sin of phase0
    double cos1, double sin1   // cos/sin of phase1
) {
    for_each_pair_diagonal<kPairBoth>(sv, qubit, [=](auto x, auto& r, auto& i) {
        const auto r0 = r;
        const auto i0 = i;
        if constexpr (decltype(x)::value == 0) {
            // Apply phase0 to qubit=0 amplitudes
            r = r0 * cos0 - i0 * sin0;
            i = r0 * sin0 + i0 * cos0;
        } else {
            // Apply phase1 to qubit=1 amplitudes
            r = r0 * cos1 - i0 * sin1;
            i = r0 * sin1 + i0 * cos1;
        }
    });
}

// =============================================================================
// Pauli X: [[0,1],[1,0]] — swap qubit=0 and qubit=1 amplitudes
// =============================================================================
void apply_x(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "x");
    for_each_pair<kPairBoth>(sv, q, [](auto* r, auto* i) {
        std::swap(r[0], r[1]);
        std::swap(i[0], i[1]);
    });
}

// =============================================================================
// Pauli Y: [[0,-i],[i,0]]
// =============================================================================
void apply_y(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "y");
    for_each_pair<kPairBoth>(sv, q, [](auto* r, auto* i) {
        const auto r0 = r[0];
        const auto i0 = i[0];
        const auto r1 = r[1];
        const auto i1 = i[1];

        // new_0 = -i * old_1 = (i1, -r1)
        r[0] = i1;
        i[0] = -r1;
        // new_1 = i * old_0 = (-i0, r0)
        r[1] = -i0;
        i[1] = r0;
    });
}

// =============================================================================
// Pauli Z: [[1,0],[0,-1]] — negate qubit=1 amplitudes
// =============================================================================
void apply_z(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "z");
    for_each_pair_diagonal<kPairSet>(sv, q, [](auto, auto& r, auto& i) {
        r = -r;
        i = -i;
    });
}

// =============================================================================
// Hadamard: (1/√2)[[1,1],[1,-1]]
// =============================================================================
void apply_h(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "h");
    for_each_pair<kPairBoth>(sv, q, [](auto* r, auto* i) {
        const auto r0 = r[0];
        const auto i0 = i[0];
        const auto r1 = r[1];
        const auto i1 = i[1];

        r[0] = INV_SQRT2 * (r0 + r1);
        i[0] = INV_SQRT2 * (i0 + i1);
        r[1] = INV_SQRT2 * (r0 - r1);
        i[1] = INV_SQRT2 * (i0 - i1);
    });
}

// =============================================================================
// S gate: [[1,0],[0,i]] — phase gate with lambda=pi/2
// =============================================================================
void apply_s(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "s");
    for_each_pair_diagonal<kPairSet>(sv, q, [](auto, auto& r, auto& i) {
        // Multiply by i: (r + im*i) * i = -im + r*i
        const auto r1 = r;
        r = -i;
        i = r1;
    });
}

// =============================================================================
// S† gate: [[1,0],[0,-i]]
// =============================================================================
void apply_sdg(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "sdg");
    for_each_pair_diagonal<kPairSet>(sv, q, [](auto, auto& r, auto& i) {
        // Multiply by -i: (r + im*i) * (-i) = im - r*i
        const auto r1 = r;
        r = i;
        i = -r1;
    });
}

// =============================================================================
// T gate: [[1,0],[0,exp(i*pi/4)]]
// =============================================================================
void apply_t(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "t");
    // exp(i*pi/4) = cos(pi/4) + i*sin(pi/4) = (1+i)/sqrt(2)
    apply_diagonal_phase(sv, q, 1.0, 0.0, INV_SQRT2, INV_SQRT2);
}

// =============================================================================
// T† gate: [[1,0],[0,exp(-i*pi/4)]]
// =============================================================================
void apply_tdg(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "tdg");
    apply_diagonal_phase(sv, q, 1.0, 0.0, INV_SQRT2, -INV_SQRT2);
}

// =============================================================================
// SX (sqrt X): 0.5 * [[1+i, 1-i], [1-i, 1+i]]
// =============================================================================
void apply_sx(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "sx");
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
void apply_sxdg(Statevector& sv, int q) {
    detail::check_qubit(q, sv.n_qubits, "sxdg");
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
void apply_rx(Statevector& sv, int q, double theta) {
    detail::check_qubit(q, sv.n_qubits, "rx");
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
void apply_ry(Statevector& sv, int q, double theta) {
    detail::check_qubit(q, sv.n_qubits, "ry");
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
void apply_rz(Statevector& sv, int q, double theta) {
    detail::check_qubit(q, sv.n_qubits, "rz");
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // phase0 = exp(-i*t/2) = cos(t/2) - i*sin(t/2)
    // phase1 = exp(+i*t/2) = cos(t/2) + i*sin(t/2)
    apply_diagonal_phase(sv, q, cos_half, -sin_half, cos_half, sin_half);
}

// =============================================================================
// Phase gate P(lambda): [[1, 0], [0, exp(i*lambda)]]
// =============================================================================
void apply_p(Statevector& sv, int q, double lambda) {
    detail::check_qubit(q, sv.n_qubits, "p");
    apply_diagonal_phase(sv, q, 1.0, 0.0, std::cos(lambda), std::sin(lambda));
}

// =============================================================================
// U gate (general single-qubit unitary):
// U(theta, phi, lambda) = [[cos(t/2), -exp(i*lam)*sin(t/2)],
//                          [exp(i*phi)*sin(t/2), exp(i*(phi+lam))*cos(t/2)]]
// =============================================================================
void apply_u(Statevector& sv, int q,
             double theta, double phi, double lambda) {
    detail::check_qubit(q, sv.n_qubits, "u");
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
void apply_u1(Statevector& sv, int q, double lambda) {
    detail::check_qubit(q, sv.n_qubits, "u1");
    apply_p(sv, q, lambda);
}

// =============================================================================
// U2(phi, lambda) = U(pi/2, phi, lambda)
// = (1/sqrt(2)) * [[1, -exp(i*lam)], [exp(i*phi), exp(i*(phi+lam))]]
// =============================================================================
void apply_u2(Statevector& sv, int q, double phi, double lambda) {
    detail::check_qubit(q, sv.n_qubits, "u2");
    apply_u(sv, q, PI_2, phi, lambda);
}

// =============================================================================
// U3(theta, phi, lambda) = U(theta, phi, lambda)
// =============================================================================
void apply_u3(Statevector& sv, int q,
              double theta, double phi, double lambda) {
    detail::check_qubit(q, sv.n_qubits, "u3");
    apply_u(sv, q, theta, phi, lambda);
}

} // namespace gates
} // namespace lindblad

