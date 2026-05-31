#pragma once
#include "lindblad/types.hpp"
#include <cstdint>
#include <vector>

namespace lindblad {

// =============================================================================
// QuditCliffordSimulator — Heisenberg-picture stabilizer tableau (prime d >= 2)
//
// State: 2n rows (n destabilizers + n stabilizers) of Heisenberg-Weyl Paulis.
// Each row r encodes: (phase[r] in Z_{2d}, xbits[r][q] in Z_d, zbits[r][q] in Z_d)
// representing the operator
//     tau^{phase[r]} * prod_q X_q^{xbits[r][q]} * Z_q^{zbits[r][q]}
// where tau = exp(i*pi*(d+1)/d), omega = tau^2 = exp(2*pi*i/d), X = shift,
// Z = clock.
//
// Row layout:
//   rows [n, 2n) = stabilizers (S_0,...,S_{n-1})
//   rows [0, n)  = destabilizers (D_0,...,D_{n-1}), paired with S_j at row n+j.
//
// Initial state |0...0>:
//   stabilizer row n+j: Z_j      (zbits[n+j][j] = 1, all else 0, phase 0)
//   destabilizer row j: X_j      (xbits[j][j] = 1,   all else 0, phase 0)
//
// Throws std::invalid_argument from the constructor if d is not prime.
// =============================================================================

class QuditCliffordSimulator {
public:
    int n_qudits;
    int d;  // must be prime

    // 2n rows total (first n = destabilizers, last n = stabilizers)
    std::vector<std::vector<int>> xbits;  // [row][qudit], mod d
    std::vector<std::vector<int>> zbits;  // [row][qudit], mod d
    std::vector<int> phase;               // [row], mod 2d

    // Construct and initialise to |0...0> (stabilised by Z_0,...,Z_{n-1}).
    QuditCliffordSimulator(int n_qudits, int d);

    // ── Gate application (Heisenberg-picture tableau update) ─────────────────

    // X^m on qudit q:  |k> -> |(k+m) mod d>
    // Conjugation:     tau^c X^a Z^b  ->  tau^{c - 2m*b_q} X^a Z^b
    void apply_X(int q, int m = 1);

    // Z^m on qudit q:  |k> -> omega^{m*k}|k>
    // Conjugation:     tau^c X^a Z^b  ->  tau^{c + 2m*a_q} X^a Z^b
    void apply_Z(int q, int m = 1);

    // Qudit QFT (Hadamard) on qudit q:  H X H^dag = Z,   H Z H^dag = X^{d-1}
    // Per row: swap x_q <-> z_q (then negate the new z_q), phase update.
    void apply_H(int q);

    // Phase gate P on qudit q.
    //   d = 2:     S gate,  S X S^dag = i X Z = Y,  S Z S^dag = Z.
    //   odd prime: canonical qudit phase gate P = sum_k omega^{2^{-1} k(k-1)} |k><k|
    //              (Howard & Vala 2012);  P X P^dag = X Z,  P Z P^dag = Z.
    //              Heisenberg update: z_q -> z_q + x_q,  phase += x_q(x_q-1) mod 2d.
    void apply_P(int q);

    // CSUM(c, t):   |x_c>|x_t>  ->  |x_c>|(x_c + x_t) mod d>
    // Conjugation rules:
    //   X_c -> X_c X_t,   X_t -> X_t
    //   Z_c -> Z_c,       Z_t -> Z_c^{d-1} Z_t
    void apply_CSUM(int q_control, int q_target);

    // CSUM^dag(c, t):  |x_c>|x_t>  ->  |x_c>|(x_t - x_c) mod d>
    void apply_CSUM_dag(int q_control, int q_target);

    // ── Measurement ──────────────────────────────────────────────────────────

    // Measure qudit q in the Z basis. Returns an outcome in {0,...,d-1}
    // and updates the tableau to the post-measurement state.
    int measure_qudit(int q, uint64_t seed = 0);

    // Measure all n qudits (seed perturbed per qudit for independence).
    std::vector<int> measure(uint64_t seed = 0);

    // ── Utilities ────────────────────────────────────────────────────────────

    static bool is_prime(int d);

    // Symplectic inner product of two rows over all qudits:
    //   sum_q (xbits[a][q]*zbits[b][q] - zbits[a][q]*xbits[b][q]) mod d
    int symplectic_product(int row_a, int row_b) const;

    // Row multiply: G_{row_r} <- G_{row_r} * G_{row_s}.
    // Combines phase, xbits, zbits using the Heisenberg-Weyl product rule:
    //   tau^c1 X^a1 Z^b1 * tau^c2 X^a2 Z^b2
    //   = tau^{c1 + c2 - 2 b1 a2} X^{a1+a2} Z^{b1+b2}.
    void row_multiply(int r, int s);

private:
    static int mod(int a, int m);
    static int mod_inv(int a, int p);  // Fermat: a^{p-2} mod p
};

} // namespace lindblad
