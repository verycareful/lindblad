#include "lindblad/qudit/qudit_noise_model.hpp"
#include "lindblad/qudit/qudit_gates.hpp"

#include <cmath>
#include <stdexcept>

namespace lindblad {

// =============================================================================
// Local helpers (not exposed in header)
// =============================================================================

// Multiply two d×d row-major matrices: C = A * B
static std::vector<Complex128> mat_mul_dd(const std::vector<Complex128>& A,
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

// Raise a d×d matrix to a non-negative integer power via repeated squaring
static std::vector<Complex128> mat_pow_dd(const std::vector<Complex128>& M,
                                          int d, int p)
{
    if (p == 0) {
        std::vector<Complex128> I(static_cast<size_t>(d * d), Complex128(0.0, 0.0));
        for (int i = 0; i < d; ++i)
            I[static_cast<size_t>(i * d + i)] = Complex128(1.0, 0.0);
        return I;
    }
    if (p == 1) return M;
    if (p % 2 == 0) {
        auto half = mat_pow_dd(M, d, p / 2);
        return mat_mul_dd(half, half, d);
    }
    return mat_mul_dd(M, mat_pow_dd(M, d, p - 1), d);
}

// =============================================================================
// QuditNoiseModel factory methods
// =============================================================================

// Generalised depolarising channel:
//   K_0 = sqrt(1-p) * I
//   K_{a,b} = sqrt(p/(d²-1)) * X^a * Z^b   for (a,b) != (0,0)
// where X = shift_matrix(d,1) and Z is the diagonal clock gate Z[j,j] = omega^j.
QuditKrausChannel QuditNoiseModel::depolarizing_channel(int d, double p)
{
    if (d < 2)
        throw std::invalid_argument("depolarizing_channel: d must be >= 2");
    if (p < 0.0 || p > 1.0)
        throw std::invalid_argument("depolarizing_channel: p must be in [0,1]");

    const int dd = d * d;
    const double two_pi_over_d = 2.0 * PI / static_cast<double>(d);

    // Build X (shift) and Z (clock) matrices
    const auto X = qudit_gates::shift_matrix(d, 1);

    std::vector<Complex128> Z(static_cast<size_t>(dd), Complex128(0.0, 0.0));
    for (int j = 0; j < d; ++j)
        Z[static_cast<size_t>(j * d + j)] =
            Complex128::exp_i(two_pi_over_d * static_cast<double>(j));

    // Precompute powers of X: X_pow[a] = X^a
    std::vector<std::vector<Complex128>> X_pow(static_cast<size_t>(d));
    X_pow[0] = mat_pow_dd(X, d, 0);  // identity
    for (int a = 1; a < d; ++a)
        X_pow[static_cast<size_t>(a)] =
            mat_pow_dd(X, d, a);

    // Precompute powers of Z: Z_pow[b] = Z^b
    std::vector<std::vector<Complex128>> Z_pow(static_cast<size_t>(d));
    Z_pow[0] = mat_pow_dd(Z, d, 0);  // identity
    for (int b = 1; b < d; ++b)
        Z_pow[static_cast<size_t>(b)] =
            mat_pow_dd(Z, d, b);

    QuditKrausChannel ch;
    ch.ops.reserve(static_cast<size_t>(dd));

    const double scale0   = std::sqrt(1.0 - p);
    const double scale_ab = (dd > 1) ? std::sqrt(p / static_cast<double>(dd - 1)) : 0.0;

    for (int a = 0; a < d; ++a) {
        for (int b = 0; b < d; ++b) {
            // Compute K_{a,b} = X^a * Z^b
            auto K = mat_mul_dd(X_pow[static_cast<size_t>(a)],
                                Z_pow[static_cast<size_t>(b)], d);

            const double sc = (a == 0 && b == 0) ? scale0 : scale_ab;
            for (auto& v : K) v = v * sc;

            ch.ops.push_back(std::move(K));
        }
    }

    return ch;
}

// Amplitude damping for a d-level system:
//   K_0[k,k] = sqrt(1 - k*gamma)   (diagonal)
//   K_j[j-1, j] = sqrt(j * gamma)  for j = 1..d-1  (lower-diagonal)
// gamma must satisfy (d-1)*gamma <= 1.
QuditKrausChannel QuditNoiseModel::amplitude_damping_channel(int d, double gamma)
{
    if (d < 2)
        throw std::invalid_argument("amplitude_damping_channel: d must be >= 2");
    if (gamma < 0.0)
        throw std::invalid_argument("amplitude_damping_channel: gamma must be >= 0");

    const int dd = d * d;
    QuditKrausChannel ch;
    ch.ops.reserve(static_cast<size_t>(d));

    // K_0: diagonal
    {
        std::vector<Complex128> K0(static_cast<size_t>(dd), Complex128(0.0, 0.0));
        for (int k = 0; k < d; ++k) {
            const double val = 1.0 - static_cast<double>(k) * gamma;
            // Clamp to avoid sqrt of negative due to floating-point noise
            K0[static_cast<size_t>(k * d + k)] =
                Complex128(std::sqrt(val > 0.0 ? val : 0.0), 0.0);
        }
        ch.ops.push_back(std::move(K0));
    }

    // K_j: single off-diagonal entry at (j-1, j)
    for (int j = 1; j < d; ++j) {
        std::vector<Complex128> Kj(static_cast<size_t>(dd), Complex128(0.0, 0.0));
        const double val = static_cast<double>(j) * gamma;
        Kj[static_cast<size_t>((j - 1) * d + j)] =
            Complex128(std::sqrt(val > 0.0 ? val : 0.0), 0.0);
        ch.ops.push_back(std::move(Kj));
    }

    return ch;
}

// Phase damping for a d-level system:
//   K_0 = sqrt(1-p) * I
//   K_j = sqrt(p) * |j><j|  for j = 0..d-1
QuditKrausChannel QuditNoiseModel::phase_damping_channel(int d, double p)
{
    if (d < 2)
        throw std::invalid_argument("phase_damping_channel: d must be >= 2");
    if (p < 0.0 || p > 1.0)
        throw std::invalid_argument("phase_damping_channel: p must be in [0,1]");

    const int dd = d * d;
    QuditKrausChannel ch;
    ch.ops.reserve(static_cast<size_t>(1 + d));

    // K_0 = sqrt(1-p) * I
    {
        std::vector<Complex128> K0(static_cast<size_t>(dd), Complex128(0.0, 0.0));
        const double sc = std::sqrt(1.0 - p);
        for (int i = 0; i < d; ++i)
            K0[static_cast<size_t>(i * d + i)] = Complex128(sc, 0.0);
        ch.ops.push_back(std::move(K0));
    }

    // K_j = sqrt(p) * |j><j|
    const double sc = std::sqrt(p);
    for (int j = 0; j < d; ++j) {
        std::vector<Complex128> Kj(static_cast<size_t>(dd), Complex128(0.0, 0.0));
        Kj[static_cast<size_t>(j * d + j)] = Complex128(sc, 0.0);
        ch.ops.push_back(std::move(Kj));
    }

    return ch;
}

// Amplitude damping Lindblad operator:
//   L[j-1, j] = sqrt(gamma * j) for j=1..d-1, rate=1.
QuditLindbladOp QuditNoiseModel::amplitude_damping_lindblad(int d, double gamma)
{
    if (d < 2)
        throw std::invalid_argument("amplitude_damping_lindblad: d must be >= 2");
    if (gamma < 0.0)
        throw std::invalid_argument("amplitude_damping_lindblad: gamma must be >= 0");

    const int dd = d * d;
    std::vector<Complex128> L(static_cast<size_t>(dd), Complex128(0.0, 0.0));
    for (int j = 1; j < d; ++j) {
        const double val = gamma * static_cast<double>(j);
        L[static_cast<size_t>((j - 1) * d + j)] =
            Complex128(std::sqrt(val > 0.0 ? val : 0.0), 0.0);
    }
    return QuditLindbladOp{std::move(L), 1.0};
}

// Dephasing Lindblad operators:
//   L_j[k,k] = sqrt(gamma) * omega^{j*k}  for j=1..d-1, rate=1 each.
std::vector<QuditLindbladOp> QuditNoiseModel::dephasing_lindblad(int d, double gamma)
{
    if (d < 2)
        throw std::invalid_argument("dephasing_lindblad: d must be >= 2");
    if (gamma < 0.0)
        throw std::invalid_argument("dephasing_lindblad: gamma must be >= 0");

    const int dd = d * d;
    const double sqrt_gamma    = std::sqrt(gamma);
    const double two_pi_over_d = 2.0 * PI / static_cast<double>(d);

    std::vector<QuditLindbladOp> ops;
    ops.reserve(static_cast<size_t>(d - 1));

    for (int j = 1; j < d; ++j) {
        std::vector<Complex128> L(static_cast<size_t>(dd), Complex128(0.0, 0.0));
        for (int k = 0; k < d; ++k) {
            // L_j[k,k] = sqrt(gamma) * omega^{j*k}
            const double theta = two_pi_over_d * static_cast<double>(j * k);
            L[static_cast<size_t>(k * d + k)] =
                Complex128::exp_i(theta) * sqrt_gamma;
        }
        ops.push_back(QuditLindbladOp{std::move(L), 1.0});
    }

    return ops;
}

// =============================================================================
// Convenience methods — build channels and add to per_qudit map
// =============================================================================

void QuditNoiseModel::add_depolarizing(int q, int d, double p)
{
    auto ch = depolarizing_channel(d, p);
    per_qudit[q].kraus.ops.insert(
        per_qudit[q].kraus.ops.end(),
        std::make_move_iterator(ch.ops.begin()),
        std::make_move_iterator(ch.ops.end()));
}

void QuditNoiseModel::add_amplitude_damping(int q, int d, double gamma)
{
    auto ch = amplitude_damping_channel(d, gamma);
    per_qudit[q].kraus.ops.insert(
        per_qudit[q].kraus.ops.end(),
        std::make_move_iterator(ch.ops.begin()),
        std::make_move_iterator(ch.ops.end()));
}

void QuditNoiseModel::add_phase_damping(int q, int d, double p)
{
    auto ch = phase_damping_channel(d, p);
    per_qudit[q].kraus.ops.insert(
        per_qudit[q].kraus.ops.end(),
        std::make_move_iterator(ch.ops.begin()),
        std::make_move_iterator(ch.ops.end()));
}

void QuditNoiseModel::add_lindblad_op(int q, std::vector<Complex128> L, double rate)
{
    per_qudit[q].lindblad.push_back(QuditLindbladOp{std::move(L), rate});
}

} // namespace lindblad
