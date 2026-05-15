#pragma once

#include "lindblad/types.hpp"

#include <vector>

namespace lindblad {
namespace qudit_gates {

// =============================================================================
// Qudit gate matrix generators
//
// All matrices are row-major: entry (row, col) is at index [row * ncols + col].
// =============================================================================

// d×d qudit QFT matrix.
// F[j*d + k] = (1/sqrt(d)) * exp(2*pi*i * j*k / d)
// With omega = exp(2*pi*i/d): F[j,k] = omega^{j*k} / sqrt(d)
std::vector<Complex128> qft_matrix(int d);

// d×d inverse qudit QFT (conjugate transpose of qft_matrix(d)).
// F_dag[j*d + k] = (1/sqrt(d)) * exp(-2*pi*i * j*k / d)
std::vector<Complex128> iqft_matrix(int d);

// d×d shift-by-m gate: X^m[j,k] = 1 if j == (k + m) mod d, else 0.
// Column k has a 1 in row (k+m)%d: X|k> = |(k+m) mod d> (forward shift).
// Default m=1 gives the standard qudit shift (generalised X gate).
std::vector<Complex128> shift_matrix(int d, int m = 1);

// d²×d² controlled-ADD gate for secret component s:
//   |x_control> |y_target>  ->  |x_control> |(y_target + s*x_control) mod d>
//
// Row index r = new_control*d + new_target
// Col index c = old_control*d + old_target
// U[r*d^2 + c] = 1 if new_control == old_control
//                  and new_target == (old_target + s*old_control) mod d,
//                else 0.
std::vector<Complex128> cadd_matrix(int d, int s);

// d²×d² controlled-power gate: clock qudit c applies U^{c*k} to target qudit.
// Row index r = r_ctrl*d + r_tgt; col index c = c_ctrl*d + c_tgt.
// M[r, c] = delta(r_ctrl, c_ctrl) * (U^{c_ctrl*k})[r_tgt, c_tgt].
// U must be a d×d row-major unitary matrix.
// k may be any non-negative integer (including 0 = identity; 1 = U itself).
std::vector<Complex128> controlled_power_matrix(int d,
                                                const std::vector<Complex128>& U,
                                                int k);

} // namespace qudit_gates
} // namespace lindblad
