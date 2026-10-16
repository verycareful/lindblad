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
#include <algorithm>
#include <cassert>
#include <utility>

namespace lindblad {
namespace gates {

// =============================================================================
// Two-qubit sweep: the gate as one formula over every group of four
// =============================================================================
// for_each_quad(sv, a, b, g) runs g(r, i) on every group of four amplitudes,
// member x = bit_a + 2 * bit_b, on doubles or on Lane8 blocks of eight groups,
// whichever detail::for_each_group picks for the qubits (sv_groups.hpp).
// Touched names the members g reads or writes; the others are not moved.
template <unsigned Touched, class G>
static inline void for_each_quad(Statevector& sv, int a, int b, const G& g) {
    const int qs[2] = {a, b};
    detail::for_each_group<2, Touched>(sv.real_parts, sv.imag_parts, sv.dim, qs, g);
}

// A diagonal gate scales each member on its own: d(x, r, i) updates member x
// (a std::integral_constant) in place (detail::for_each_diagonal).
template <unsigned Touched, class D>
static inline void for_each_quad_diagonal(Statevector& sv, int a, int b, const D& d) {
    const int qs[2] = {a, b};
    detail::for_each_diagonal<2, Touched>(sv.real_parts, sv.imag_parts, sv.dim, qs, d);
}

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
    // Members: 1 = ctrl 1, tgt 0; 3 = ctrl 1, tgt 1.
    for_each_quad<0b1010>(sv, ctrl, tgt, [=](auto* r, auto* i) {
        const auto r0 = r[1];
        const auto i0 = i[1];
        const auto r1 = r[3];
        const auto i1 = i[3];

        r[1] = (ar*r0 - ai*i0) + (br*r1 - bi*i1);
        i[1] = (ar*i0 + ai*r0) + (br*i1 + bi*r1);
        r[3] = (cr*r0 - ci*i0) + (dr*r1 - di*i1);
        i[3] = (cr*i0 + ci*r0) + (dr*i1 + di*r1);
    });
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
    // Member 3 = ctrl 1, tgt 1.
    for_each_quad_diagonal<0b1000>(sv, ctrl, tgt, [=](auto, auto& r, auto& i) {
        const auto re = r;
        const auto im = i;
        r = re * cos_p - im * sin_p;
        i = re * sin_p + im * cos_p;
    });
}

// =============================================================================
// CX (CNOT): ctrl=1 flips target — cache-optimised
// =============================================================================
void apply_cx(Statevector& sv, int ctrl, int tgt) {
    detail::check_qubit(ctrl, sv.n_qubits, "cx");
    detail::check_qubit(tgt, sv.n_qubits, "cx");
    detail::check_distinct2(ctrl, tgt, "cx");
    // Members: 1 = ctrl 1, tgt 0; 3 = ctrl 1, tgt 1.
    for_each_quad<0b1010>(sv, ctrl, tgt, [](auto* r, auto* i) {
        std::swap(r[1], r[3]);
        std::swap(i[1], i[3]);
    });
}

// =============================================================================
// CY: ctrl=1, apply Y to target
// Y = [[0, -i], [i, 0]]
// =============================================================================
void apply_cy(Statevector& sv, int ctrl, int tgt) {
    detail::check_qubit(ctrl, sv.n_qubits, "cy");
    detail::check_qubit(tgt, sv.n_qubits, "cy");
    detail::check_distinct2(ctrl, tgt, "cy");
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
void apply_cz(Statevector& sv, int ctrl, int tgt) {
    detail::check_qubit(ctrl, sv.n_qubits, "cz");
    detail::check_qubit(tgt, sv.n_qubits, "cz");
    detail::check_distinct2(ctrl, tgt, "cz");
    // Member 3 = ctrl 1, tgt 1.
    for_each_quad_diagonal<0b1000>(sv, ctrl, tgt, [](auto, auto& r, auto& i) {
        r = -r;
        i = -i;
    });
}

// =============================================================================
// CH: ctrl=1, apply H to target
// =============================================================================
void apply_ch(Statevector& sv, int ctrl, int tgt) {
    detail::check_qubit(ctrl, sv.n_qubits, "ch");
    detail::check_qubit(tgt, sv.n_qubits, "ch");
    detail::check_distinct2(ctrl, tgt, "ch");
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
void apply_swap(Statevector& sv, int q1, int q2) {
    detail::check_qubit(q1, sv.n_qubits, "swap");
    detail::check_qubit(q2, sv.n_qubits, "swap");
    detail::check_distinct2(q1, q2, "swap");
    // Members: 1 = q1 1, q2 0; 2 = q1 0, q2 1.
    for_each_quad<0b0110>(sv, q1, q2, [](auto* r, auto* i) {
        std::swap(r[1], r[2]);
        std::swap(i[1], i[2]);
    });
}

// =============================================================================
// iSWAP: swap |01⟩ <-> |10⟩ and multiply each by i — cache-optimised
// =============================================================================
void apply_iswap(Statevector& sv, int q1, int q2) {
    detail::check_qubit(q1, sv.n_qubits, "iswap");
    detail::check_qubit(q2, sv.n_qubits, "iswap");
    detail::check_distinct2(q1, q2, "iswap");
    // Members: 1 = q1 1, q2 0; 2 = q1 0, q2 1.
    for_each_quad<0b0110>(sv, q1, q2, [](auto* r, auto* i) {
        const auto r_a = r[1];
        const auto i_a = i[1];
        const auto r_b = r[2];
        const auto i_b = i[2];

        // member 1 gets i * old member 2: i*(r_b + i*i_b) = -i_b + i*r_b
        r[1] = -i_b;
        i[1] = r_b;
        // member 2 gets i * old member 1: i*(r_a + i*i_a) = -i_a + i*r_a
        r[2] = -i_a;
        i[2] = r_a;
    });
}

// =============================================================================
// CRX: controlled RX(theta)
// =============================================================================
void apply_crx(Statevector& sv, int ctrl, int tgt, double theta) {
    detail::check_qubit(ctrl, sv.n_qubits, "crx");
    detail::check_qubit(tgt, sv.n_qubits, "crx");
    detail::check_distinct2(ctrl, tgt, "crx");
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
void apply_cry(Statevector& sv, int ctrl, int tgt, double theta) {
    detail::check_qubit(ctrl, sv.n_qubits, "cry");
    detail::check_qubit(tgt, sv.n_qubits, "cry");
    detail::check_distinct2(ctrl, tgt, "cry");
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
void apply_crz(Statevector& sv, int ctrl, int tgt, double theta) {
    detail::check_qubit(ctrl, sv.n_qubits, "crz");
    detail::check_qubit(tgt, sv.n_qubits, "crz");
    detail::check_distinct2(ctrl, tgt, "crz");
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // Members: 1 = ctrl 1, tgt 0; 3 = ctrl 1, tgt 1.
    for_each_quad_diagonal<0b1010>(sv, ctrl, tgt, [=](auto x, auto& r, auto& i) {
        const auto r0 = r;
        const auto i0 = i;
        if constexpr (decltype(x)::value == 1) {
            // tgt=0: multiply by exp(-i*t/2) = cos - i*sin
            r = r0 * cos_half + i0 * sin_half;
            i = i0 * cos_half - r0 * sin_half;
        } else {
            // tgt=1: multiply by exp(+i*t/2) = cos + i*sin
            r = r0 * cos_half - i0 * sin_half;
            i = i0 * cos_half + r0 * sin_half;
        }
    });
}

// =============================================================================
// CP: controlled P(lambda) — uses specialised diagonal path
// =============================================================================
void apply_cp(Statevector& sv, int ctrl, int tgt, double lambda) {
    detail::check_qubit(ctrl, sv.n_qubits, "cp");
    detail::check_qubit(tgt, sv.n_qubits, "cp");
    detail::check_distinct2(ctrl, tgt, "cp");
    apply_controlled_phase(sv, ctrl, tgt, std::cos(lambda), std::sin(lambda));
}

// =============================================================================
// CU: controlled U(theta, phi, lambda) with global phase gamma
// =============================================================================
void apply_cu(Statevector& sv, int ctrl, int tgt,
              double theta, double phi, double lambda, double gamma) {
    detail::check_qubit(ctrl, sv.n_qubits, "cu");
    detail::check_qubit(tgt, sv.n_qubits, "cu");
    detail::check_distinct2(ctrl, tgt, "cu");
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
void apply_ecr(Statevector& sv, int q1, int q2) {
    detail::check_qubit(q1, sv.n_qubits, "ecr");
    detail::check_qubit(q2, sv.n_qubits, "ecr");
    detail::check_distinct2(q1, q2, "ecr");

    // Member x = bit_q2 + 2 * bit_q1, so x written in binary is |q1 q2⟩.
    for_each_quad<0b1111>(sv, q2, q1, [](auto* r, auto* i) {
        const auto r00 = r[0], im00 = i[0];
        const auto r01 = r[1], im01 = i[1];
        const auto r10 = r[2], im10 = i[2];
        const auto r11 = r[3], im11 = i[3];

        // Row 0 (|00⟩): 1/sqrt2 * (|10⟩ + i|11⟩)
        r[0] = INV_SQRT2 * (r10 - im11);
        i[0] = INV_SQRT2 * (im10 + r11);

        // Row 1 (|01⟩): 1/sqrt2 * (i|10⟩ + |11⟩)
        r[1] = INV_SQRT2 * (-im10 + r11);
        i[1] = INV_SQRT2 * (r10 + im11);

        // Row 2 (|10⟩): 1/sqrt2 * (|00⟩ - i|01⟩)
        r[2] = INV_SQRT2 * (r00 + im01);
        i[2] = INV_SQRT2 * (im00 - r01);

        // Row 3 (|11⟩): 1/sqrt2 * (-i|00⟩ + |01⟩)
        r[3] = INV_SQRT2 * (im00 + r01);
        i[3] = INV_SQRT2 * (-r00 + im01);
    });
}

// =============================================================================
// RZX(theta): exp(-i * theta/2 * Z*X) — cache-optimised
// =============================================================================
void apply_rzx(Statevector& sv, int q1, int q2, double theta) {
    detail::check_qubit(q1, sv.n_qubits, "rzx");
    detail::check_qubit(q2, sv.n_qubits, "rzx");
    detail::check_distinct2(q1, q2, "rzx");
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // Member x = bit_q2 + 2 * bit_q1, so x written in binary is |q1 q2⟩.
    for_each_quad<0b1111>(sv, q2, q1, [=](auto* r, auto* i) {
        // q1=0 pair: |00⟩ <-> |01⟩, Z=+1, applies exp(-i*t/2*X): off-diag = -i*sin
        {
            const auto r0 = r[0], i0 = i[0];
            const auto r1 = r[1], i1 = i[1];
            r[0] = cos_half * r0 + sin_half * i1;
            i[0] = cos_half * i0 - sin_half * r1;
            r[1] = cos_half * r1 + sin_half * i0;
            i[1] = cos_half * i1 - sin_half * r0;
        }

        // q1=1 pair: |10⟩ <-> |11⟩, Z=-1, applies exp(+i*t/2*X): off-diag = +i*sin
        {
            const auto r0 = r[2], i0 = i[2];
            const auto r1 = r[3], i1 = i[3];
            r[2] = cos_half * r0 - sin_half * i1;
            i[2] = cos_half * i0 + sin_half * r1;
            r[3] = cos_half * r1 - sin_half * i0;
            i[3] = cos_half * i1 + sin_half * r0;
        }
    });
}

// =============================================================================
// RXX(theta): exp(-i * theta/2 * X*X) — cache-optimised 4-index group
// =============================================================================
void apply_rxx(Statevector& sv, int q1, int q2, double theta) {
    detail::check_qubit(q1, sv.n_qubits, "rxx");
    detail::check_qubit(q2, sv.n_qubits, "rxx");
    detail::check_distinct2(q1, q2, "rxx");
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // Member x = bit_q2 + 2 * bit_q1, so x written in binary is |q1 q2⟩.
    for_each_quad<0b1111>(sv, q2, q1, [=](auto* r, auto* i) {
        const auto r00 = r[0], im00 = i[0];
        const auto r01 = r[1], im01 = i[1];
        const auto r10 = r[2], im10 = i[2];
        const auto r11 = r[3], im11 = i[3];

        r[0] = cos_half * r00 + sin_half * im11;
        i[0] = cos_half * im00 - sin_half * r11;

        r[1] = cos_half * r01 + sin_half * im10;
        i[1] = cos_half * im01 - sin_half * r10;

        r[2] = sin_half * im01 + cos_half * r10;
        i[2] = -sin_half * r01 + cos_half * im10;

        r[3] = sin_half * im00 + cos_half * r11;
        i[3] = -sin_half * r00 + cos_half * im11;
    });
}

// =============================================================================
// RYY(theta): exp(-i * theta/2 * Y*Y) — cache-optimised 4-index group
// =============================================================================
void apply_ryy(Statevector& sv, int q1, int q2, double theta) {
    detail::check_qubit(q1, sv.n_qubits, "ryy");
    detail::check_qubit(q2, sv.n_qubits, "ryy");
    detail::check_distinct2(q1, q2, "ryy");
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // Member x = bit_q2 + 2 * bit_q1, so x written in binary is |q1 q2⟩.
    for_each_quad<0b1111>(sv, q2, q1, [=](auto* r, auto* i) {
        const auto r00 = r[0], im00 = i[0];
        const auto r01 = r[1], im01 = i[1];
        const auto r10 = r[2], im10 = i[2];
        const auto r11 = r[3], im11 = i[3];

        r[0] = cos_half * r00 - sin_half * im11;
        i[0] = cos_half * im00 + sin_half * r11;

        r[1] = cos_half * r01 + sin_half * im10;
        i[1] = cos_half * im01 - sin_half * r10;

        r[2] = sin_half * im01 + cos_half * r10;
        i[2] = -sin_half * r01 + cos_half * im10;

        r[3] = -sin_half * im00 + cos_half * r11;
        i[3] = sin_half * r00 + cos_half * im11;
    });
}

// =============================================================================
// RZZ(theta): exp(-i * theta/2 * Z*Z) — diagonal, cache-optimised
// diag(e^{-it/2}, e^{it/2}, e^{it/2}, e^{-it/2})
// Parity-based: same-parity gets exp(-it/2), different-parity gets exp(+it/2)
// =============================================================================
void apply_rzz(Statevector& sv, int q1, int q2, double theta) {
    detail::check_qubit(q1, sv.n_qubits, "rzz");
    detail::check_qubit(q2, sv.n_qubits, "rzz");
    detail::check_distinct2(q1, q2, "rzz");
    const double cos_half = std::cos(theta / 2.0);
    const double sin_half = std::sin(theta / 2.0);

    // Member x = bit_q1 + 2 * bit_q2; the phase depends on its parity only.
    for_each_quad_diagonal<0b1111>(sv, q1, q2, [=](auto x, auto& r, auto& i) {
        constexpr std::size_t X = decltype(x)::value;
        const auto r0 = r;
        const auto i0 = i;
        if constexpr (X == 0 || X == 3) {
            // Even parity: exp(-it/2) = cos - i*sin
            r = r0 * cos_half + i0 * sin_half;
            i = i0 * cos_half - r0 * sin_half;
        } else {
            // Odd parity: exp(+it/2) = cos + i*sin
            r = r0 * cos_half - i0 * sin_half;
            i = i0 * cos_half + r0 * sin_half;
        }
    });
}

} // namespace gates
} // namespace lindblad

