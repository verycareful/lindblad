#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/types.hpp"

#include <cmath>

namespace lindblad {
namespace qudit_gates {

// ---------------------------------------------------------------------------
// qft_matrix — F[j,k] = omega^{j*k} / sqrt(d),  omega = exp(2*pi*i/d)
// ---------------------------------------------------------------------------

std::vector<Complex128> qft_matrix(int d) {
    std::vector<Complex128> F(static_cast<size_t>(d * d));
    const double inv_sqrt_d    = 1.0 / std::sqrt(static_cast<double>(d));
    const double two_pi_over_d = 2.0 * PI / static_cast<double>(d);
    for (int j = 0; j < d; ++j)
        for (int k = 0; k < d; ++k)
            F[static_cast<size_t>(j * d + k)] =
                Complex128::exp_i(two_pi_over_d * static_cast<double>(j * k))
                * inv_sqrt_d;
    return F;
}

// ---------------------------------------------------------------------------
// iqft_matrix — F_dag[j,k] = omega^{-j*k} / sqrt(d) (conjugate transpose of QFT)
// ---------------------------------------------------------------------------

std::vector<Complex128> iqft_matrix(int d) {
    std::vector<Complex128> Fd(static_cast<size_t>(d * d));
    const double inv_sqrt_d    = 1.0 / std::sqrt(static_cast<double>(d));
    const double two_pi_over_d = 2.0 * PI / static_cast<double>(d);
    for (int j = 0; j < d; ++j)
        for (int k = 0; k < d; ++k)
            Fd[static_cast<size_t>(j * d + k)] =
                Complex128::exp_i(-two_pi_over_d * static_cast<double>(j * k))
                * inv_sqrt_d;
    return Fd;
}

// ---------------------------------------------------------------------------
// shift_matrix — X^m[j,k] = 1 if j == (k + m) mod d, else 0
// ---------------------------------------------------------------------------
// Column k has a 1 in row (k+m)%d, so X|k> = |(k+m) mod d> (forward shift).

std::vector<Complex128> shift_matrix(int d, int m) {
    std::vector<Complex128> X(static_cast<size_t>(d * d), Complex128(0.0, 0.0));
    // Normalise m into [0, d) to allow negative or large m
    int m_norm = ((m % d) + d) % d;
    for (int k = 0; k < d; ++k)
        X[static_cast<size_t>(((k + m_norm) % d) * d + k)] =
            Complex128(1.0, 0.0);
    return X;
}

// ---------------------------------------------------------------------------
// cadd_matrix — |x>|y> -> |x>|(y + s*x) mod d>
// ---------------------------------------------------------------------------
// Convention matches apply_2qudit (project LSB-first): the FIRST operand
// (the control, x) is the LEAST significant digit of the matrix index:
//   row r = new_tgt*d + new_ctrl, col c = old_tgt*d + old_ctrl.
// Non-zero entry where new_ctrl == old_ctrl AND new_tgt == (old_tgt + s*old_ctrl) mod d.

std::vector<Complex128> cadd_matrix(int d, int s) {
    const int dd = d * d;
    std::vector<Complex128> U(static_cast<size_t>(dd * dd), Complex128(0.0, 0.0));
    // Normalise s into [0, d)
    const int s_norm = ((s % d) + d) % d;
    for (int c0 = 0; c0 < d; ++c0) {
        for (int c1 = 0; c1 < d; ++c1) {
            const int r0  = c0;
            const int r1  = (c1 + (s_norm * c0) % d) % d;
            const int row = r1 * d + r0;
            const int col = c1 * d + c0;
            U[static_cast<size_t>(row * dd + col)] = Complex128(1.0, 0.0);
        }
    }
    return U;
}

// ---------------------------------------------------------------------------
// mat_mul — multiply two d×d matrices (row-major): C = A * B
// ---------------------------------------------------------------------------
static std::vector<Complex128> mat_mul(const std::vector<Complex128>& A,
                                       const std::vector<Complex128>& B,
                                       int d)
{
    std::vector<Complex128> C(static_cast<size_t>(d * d), Complex128(0.0, 0.0));
    for (int r = 0; r < d; ++r)
        for (int k = 0; k < d; ++k) {
            const Complex128 a = A[static_cast<size_t>(r * d + k)];
            if (a.real == 0.0 && a.imag == 0.0) continue;
            for (int c = 0; c < d; ++c)
                C[static_cast<size_t>(r * d + c)] +=
                    a * B[static_cast<size_t>(k * d + c)];
        }
    return C;
}

// ---------------------------------------------------------------------------
// mat_pow — U^m via binary exponentiation (U is d×d, row-major, m >= 0)
// ---------------------------------------------------------------------------
static std::vector<Complex128> mat_pow(const std::vector<Complex128>& U,
                                       int d, int m)
{
    if (m == 0) {
        std::vector<Complex128> I(static_cast<size_t>(d * d), Complex128(0.0, 0.0));
        for (int i = 0; i < d; ++i)
            I[static_cast<size_t>(i * d + i)] = Complex128(1.0, 0.0);
        return I;
    }
    if (m == 1) return U;
    if (m % 2 == 0) {
        auto half = mat_pow(U, d, m / 2);
        return mat_mul(half, half, d);
    }
    return mat_mul(U, mat_pow(U, d, m - 1), d);
}

// ---------------------------------------------------------------------------
// controlled_power_matrix — d²×d² gate: clock ctrl=c applies U^{c*k} to target
// ---------------------------------------------------------------------------
// Convention matches apply_2qudit (project LSB-first): the FIRST operand (the
// control) is the LEAST significant digit:
//   row r = r_tgt*d + r_ctrl, col = c_tgt*d + c_ctrl.
// For ctrl value c: target block is U^{c*k}. k=0 → identity; k=1 → standard CU.

std::vector<Complex128> controlled_power_matrix(int d,
                                                 const std::vector<Complex128>& U,
                                                 int k)
{
    const int dd = d * d;
    std::vector<Complex128> M(static_cast<size_t>(dd * dd), Complex128(0.0, 0.0));
    for (int ctrl = 0; ctrl < d; ++ctrl) {
        const auto Uk = mat_pow(U, d, ctrl * k);
        for (int r_t = 0; r_t < d; ++r_t)
            for (int c_t = 0; c_t < d; ++c_t) {
                const int row = r_t * d + ctrl;
                const int col = c_t * d + ctrl;
                M[static_cast<size_t>(row * dd + col)] =
                    Uk[static_cast<size_t>(r_t * d + c_t)];
            }
    }
    return M;
}

} // namespace qudit_gates
} // namespace lindblad
