#include "lindblad/qudit/qudit_clifford.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace lindblad {

// =============================================================================
// Static helpers
// =============================================================================

int QuditCliffordSimulator::mod(int a, int m) {
    int r = a % m;
    if (r < 0) r += m;
    return r;
}

// Fermat's little theorem: for prime p and a !=  0 mod p, a^{p-2} ≡ a^{-1} (mod p).
int QuditCliffordSimulator::mod_inv(int a, int p) {
    int base = mod(a, p);
    if (base == 0)
        throw std::invalid_argument(
            "QuditCliffordSimulator::mod_inv: 0 has no modular inverse");
    long long result = 1;
    long long b = base;
    int exp = p - 2;
    while (exp > 0) {
        if (exp & 1) result = (result * b) % p;
        b = (b * b) % p;
        exp >>= 1;
    }
    return static_cast<int>(result);
}

bool QuditCliffordSimulator::is_prime(int d) {
    if (d < 2) return false;
    if (d == 2) return true;
    if ((d % 2) == 0) return false;
    for (int i = 3; static_cast<long long>(i) * i <= d; i += 2)
        if ((d % i) == 0) return false;
    return true;
}

// =============================================================================
// Construction — initial state |0...0>
// =============================================================================

QuditCliffordSimulator::QuditCliffordSimulator(int n_qudits_, int d_)
    : n_qudits(n_qudits_)
    , d(d_)
{
    if (n_qudits_ < 1)
        throw std::invalid_argument(
            "QuditCliffordSimulator: n_qudits must be >= 1");
    if (!is_prime(d_))
        throw std::invalid_argument(
            "QuditCliffordSimulator: d must be prime");

    const int rows = 2 * n_qudits;
    xbits.assign(rows, std::vector<int>(n_qudits, 0));
    zbits.assign(rows, std::vector<int>(n_qudits, 0));
    phase.assign(rows, 0);

    // Destabilizer rows [0, n): D_j = X_j
    for (int j = 0; j < n_qudits; ++j) {
        xbits[j][j] = 1;
    }
    // Stabilizer rows [n, 2n): S_j = Z_j
    for (int j = 0; j < n_qudits; ++j) {
        zbits[n_qudits + j][j] = 1;
    }
}

// =============================================================================
// Single-qudit gates
// =============================================================================

void QuditCliffordSimulator::apply_X(int q, int m) {
    // X^m Z^b X^{-m} = tau^{-2 m b} Z^b  =>  phase -= 2 m * zbits[r][q]
    const int rows = 2 * n_qudits;
    const int two_d = 2 * d;
    const int m_red = mod(m, d);
    if (m_red == 0) return;
    for (int r = 0; r < rows; ++r) {
        phase[r] = mod(phase[r] - 2 * m_red * zbits[r][q], two_d);
    }
}

void QuditCliffordSimulator::apply_Z(int q, int m) {
    // Z^m X^a Z^{-m} = tau^{+2 m a} X^a  =>  phase += 2 m * xbits[r][q]
    const int rows = 2 * n_qudits;
    const int two_d = 2 * d;
    const int m_red = mod(m, d);
    if (m_red == 0) return;
    for (int r = 0; r < rows; ++r) {
        phase[r] = mod(phase[r] + 2 * m_red * xbits[r][q], two_d);
    }
}

void QuditCliffordSimulator::apply_H(int q) {
    // H X H^dag = Z,  H Z H^dag = X^{d-1}.
    // Heisenberg: a row's (x_q, z_q) component
    //   tau^c X^{x_q} Z^{z_q}
    //   |-> tau^c Z^{x_q} X^{-z_q}
    //   = tau^c X^{-z_q} Z^{x_q} * tau^{2 * x_q * (-z_q)}
    //   = tau^{c - 2 x_q z_q} X^{d-z_q} Z^{x_q}.
    const int rows = 2 * n_qudits;
    const int two_d = 2 * d;
    for (int r = 0; r < rows; ++r) {
        const int old_x = xbits[r][q];
        const int old_z = zbits[r][q];
        xbits[r][q] = old_z;
        zbits[r][q] = mod(-old_x, d);
        phase[r] = mod(phase[r] - 2 * old_x * old_z, two_d);
    }
}

void QuditCliffordSimulator::apply_P(int q) {
    if (d != 2) {
        throw std::runtime_error(
            "QuditCliffordSimulator::apply_P: not implemented for d > 2");
    }
    // d = 2: S gate.  S X S^dag = i X Z = Y,  S Z S^dag = Z.
    // Tableau update per row:
    //   (x, z, phi)  ->  (x, x XOR z, phi + x)   in Z_2 / Z_4 conventions.
    // Derivation: S (X^x Z^z) S^dag = (S X S^dag)^x (S Z S^dag)^z
    //   = (iXZ)^x Z^z = i^x X^x Z^x Z^z = i^x X^x Z^{x+z}.
    // tau = i for d=2, so the extra phase i^x is tau^x  =>  phase += x.
    const int rows = 2 * n_qudits;
    const int two_d = 2 * d;  // = 4
    for (int r = 0; r < rows; ++r) {
        const int old_x = xbits[r][q];
        const int old_z = zbits[r][q];
        zbits[r][q] = mod(old_z + old_x, d);
        phase[r] = mod(phase[r] + old_x, two_d);
    }
}

// =============================================================================
// Two-qudit gate: CSUM and its adjoint
// =============================================================================

void QuditCliffordSimulator::apply_CSUM(int q_control, int q_target) {
    // Conjugation rules under CSUM(c -> t):
    //   X_c -> X_c X_t,   X_t -> X_t
    //   Z_c -> Z_c,       Z_t -> Z_c^{d-1} Z_t
    //
    // On a row (x_c, z_c, x_t, z_t):
    //   new x_t = (x_t + x_c) mod d
    //   new z_c = (z_c - z_t) mod d
    //   x_c, z_t unchanged
    //
    // Phase update (Gottesman-style cross term, derived from reordering the
    // X_t^{x_c} factor past the existing Z's when normalising to standard form):
    //   phase -= 2 * z_c * x_t   (using OLD values, before updating)
    const int rows = 2 * n_qudits;
    const int two_d = 2 * d;
    const int c = q_control;
    const int t = q_target;
    for (int r = 0; r < rows; ++r) {
        const int old_xc = xbits[r][c];
        const int old_xt = xbits[r][t];
        const int old_zc = zbits[r][c];
        const int old_zt = zbits[r][t];

        phase[r] = mod(phase[r] - 2 * old_zc * old_xt, two_d);
        xbits[r][t] = mod(old_xt + old_xc, d);
        zbits[r][c] = mod(old_zc - old_zt, d);
        // xbits[r][c], zbits[r][t] unchanged.
    }
}

void QuditCliffordSimulator::apply_CSUM_dag(int q_control, int q_target) {
    // CSUM^dag : |x_c, x_t> -> |x_c, (x_t - x_c) mod d>.
    // Conjugation rules:
    //   X_c -> X_c X_t^{d-1},   X_t -> X_t
    //   Z_c -> Z_c Z_t,         Z_t -> Z_t
    //
    // Row update:
    //   new x_t = (x_t - x_c) mod d
    //   new z_c = (z_c + z_t) mod d
    //   x_c, z_t unchanged
    // Phase: opposite sign of the CSUM cross term  =>  phase += 2 * z_c * x_t.
    const int rows = 2 * n_qudits;
    const int two_d = 2 * d;
    const int c = q_control;
    const int t = q_target;
    for (int r = 0; r < rows; ++r) {
        const int old_xc = xbits[r][c];
        const int old_xt = xbits[r][t];
        const int old_zc = zbits[r][c];
        const int old_zt = zbits[r][t];

        phase[r] = mod(phase[r] + 2 * old_zc * old_xt, two_d);
        xbits[r][t] = mod(old_xt - old_xc, d);
        zbits[r][c] = mod(old_zc + old_zt, d);
    }
}

// =============================================================================
// Symplectic product and row multiplication
// =============================================================================

int QuditCliffordSimulator::symplectic_product(int row_a, int row_b) const {
    long long acc = 0;
    for (int q = 0; q < n_qudits; ++q) {
        acc += static_cast<long long>(xbits[row_a][q]) * zbits[row_b][q]
             - static_cast<long long>(zbits[row_a][q]) * xbits[row_b][q];
    }
    return mod(static_cast<int>(acc % d), d);
}

void QuditCliffordSimulator::row_multiply(int r, int s) {
    // G_r <- G_r * G_s
    //
    // Per qudit: (tau^{c1} X^{a1} Z^{b1}) * (tau^{c2} X^{a2} Z^{b2})
    //          = tau^{c1+c2 - 2 b1 a2} X^{a1+a2} Z^{b1+b2}.
    //
    // The cross-phase per qudit is -2 * zbits[r][q] * xbits[s][q] (using OLD
    // values of zbits[r][q] before updating).
    const int two_d = 2 * d;
    long long phase_delta = 0;
    for (int q = 0; q < n_qudits; ++q) {
        phase_delta -= 2LL * zbits[r][q] * xbits[s][q];
        xbits[r][q] = mod(xbits[r][q] + xbits[s][q], d);
        zbits[r][q] = mod(zbits[r][q] + zbits[s][q], d);
    }
    int delta_mod = mod(static_cast<int>(phase_delta % two_d), two_d);
    phase[r] = mod(phase[r] + delta_mod + phase[s], two_d);
}

// =============================================================================
// Measurement
// =============================================================================

int QuditCliffordSimulator::measure_qudit(int q, uint64_t seed) {
    const int n = n_qudits;
    const int rows = 2 * n;
    const int two_d = 2 * d;

    // Find first stabilizer row p in [n, 2n) with xbits[p][q] != 0.
    int p = -1;
    for (int r = n; r < rows; ++r) {
        if (xbits[r][q] != 0) { p = r; break; }
    }

    if (p != -1) {
        // ── Indeterminate (random) outcome ───────────────────────────────────
        //
        // Eliminate the X_q component from every OTHER row by row-multiplying
        // with row p.  Because d is prime, xbits[p][q] is invertible mod d.
        const int inv_xpq = mod_inv(xbits[p][q], d);

        for (int r = 0; r < rows; ++r) {
            if (r == p) continue;
            if (xbits[r][q] == 0) continue;
            // We need: xbits[r][q] + k * xbits[p][q] ≡ 0 (mod d), so
            //   k = -xbits[r][q] * inv_xpq (mod d)
            int k = mod(-xbits[r][q] * inv_xpq, d);
            for (int i = 0; i < k; ++i) row_multiply(r, p);
        }

        // Move the old stabilizer row p into its destabilizer slot (p - n).
        const int dest_row = p - n;
        xbits[dest_row] = xbits[p];
        zbits[dest_row] = zbits[p];
        phase[dest_row] = phase[p];

        // Sample the random outcome uniformly from {0, ..., d-1}.
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> dist(0, d - 1);
        const int outcome = dist(rng);

        // Install the new stabilizer Z_q with the correct phase so it
        // stabilises |outcome>:  tau^{phi} Z_q |k> = tau^{phi + 2k} |k>,
        // and we want this to equal |k> for k = outcome  =>  phi = -2*outcome.
        std::fill(xbits[p].begin(), xbits[p].end(), 0);
        std::fill(zbits[p].begin(), zbits[p].end(), 0);
        zbits[p][q] = 1;
        phase[p] = mod(-2 * outcome, two_d);

        return outcome;
    }

    // ── Determined outcome ───────────────────────────────────────────────────
    //
    // No stabilizer anticommutes with Z_q, so Z_q (or some power of it) lives
    // in the stabilizer group.  Multiply together every stabilizer row that
    // contains a Z_q component into a scratch row.  In the determined case for
    // the algorithms we care about, the resulting scratch row reduces to a
    // pure Z_q^m on qudit q, and its phase encodes the outcome k via the
    // eigenvalue equation
    //     phase + 2 * m * k ≡ 0 (mod 2d)
    // i.e. k = -phase / (2m) (mod d).
    std::vector<int> scratch_x(n, 0);
    std::vector<int> scratch_z(n, 0);
    int scratch_phase = 0;

    for (int r = n; r < rows; ++r) {
        if (zbits[r][q] == 0) continue;
        // scratch *= row r  (using the same product formula as row_multiply).
        long long delta = 0;
        for (int qq = 0; qq < n; ++qq) {
            delta -= 2LL * scratch_z[qq] * xbits[r][qq];
            scratch_x[qq] = mod(scratch_x[qq] + xbits[r][qq], d);
            scratch_z[qq] = mod(scratch_z[qq] + zbits[r][qq], d);
        }
        int delta_mod = mod(static_cast<int>(delta % two_d), two_d);
        scratch_phase = mod(scratch_phase + delta_mod + phase[r], two_d);
    }

    const int m = scratch_z[q];
    if (m == 0) {
        // Fallback: no Z_q component found anywhere.  This can occur only for
        // pathological states (e.g. when the stabilizer for qudit q has been
        // entirely scrambled into a global identity).  Return 0 by convention.
        return 0;
    }

    // Solve phase + 2*m*k ≡ 0 (mod 2d) for k in {0,...,d-1}.
    // The phase of a valid stabilizer is always even (since stabilizer-group
    // elements square to identity-up-to-phase, and the phase is in Z_{2d}).
    // We assume scratch_phase is even here; for odd d, 2 is invertible mod d
    // and we use the half-phase directly.
    const int ph = mod(scratch_phase, two_d);
    if (d == 2) {
        // 2*m*k ≡ -ph (mod 4).  m and ph are in {0,1,2,3} with ph even.
        // Solutions: k = ((4 - ph) / 2) * m^{-1} mod 2.  For prime d=2,
        // m must be 1 (only nonzero value mod 2), so k = (4 - ph)/2 mod 2.
        return mod((-ph / 2), 2);
    }
    // Odd prime d: ph is even, so ph/2 is a well-defined integer; reduce mod d.
    const int ph_half = mod(ph / 2, d);
    const int inv_m = mod_inv(m, d);
    return mod(-ph_half * inv_m, d);
}

std::vector<int> QuditCliffordSimulator::measure(uint64_t seed) {
    std::vector<int> result;
    result.reserve(n_qudits);
    for (int q = 0; q < n_qudits; ++q) {
        result.push_back(measure_qudit(q, seed + static_cast<uint64_t>(q)));
    }
    return result;
}

} // namespace lindblad
