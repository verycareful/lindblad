#include "lindblad/qudit/qudit_density_matrix.hpp"

#include "lindblad/detail/validate.hpp"
#include "lindblad/detail/validate_physical.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace lindblad {

// =============================================================================
// Local helpers
// =============================================================================

// d×d row-major matrix multiply: C = A * B
static std::vector<Complex128> dm_mat_mul(const std::vector<Complex128>& A,
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

// Conjugate transpose of a d×d matrix
static std::vector<Complex128> dm_dag(const std::vector<Complex128>& M, int d)
{
    std::vector<Complex128> Md(static_cast<size_t>(d * d));
    for (int r = 0; r < d; ++r)
        for (int c = 0; c < d; ++c)
            Md[static_cast<size_t>(c * d + r)] =
                M[static_cast<size_t>(r * d + c)].conj();
    return Md;
}

// =============================================================================
// QuditDensityMatrix — construction
// =============================================================================

size_t QuditDensityMatrix::ipow(size_t base, int exp) noexcept {
    size_t result = 1;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

QuditDensityMatrix::QuditDensityMatrix(int n_qudits_, int d_)
    : n_qudits(n_qudits_)
    , d(d_)
    , dim(ipow(static_cast<size_t>(d_), n_qudits_))
    , dim_sq(dim * dim)
    , rho(dim * dim, Complex128(0.0, 0.0))
{
    if (d_ < 2)
        throw std::invalid_argument("QuditDensityMatrix: d must be >= 2");
    if (n_qudits_ < 1)
        throw std::invalid_argument("QuditDensityMatrix: n_qudits must be >= 1");
    rho[0] = Complex128(1.0, 0.0);  // |0...0><0...0|
}

QuditDensityMatrix::QuditDensityMatrix(const QuditStatevector& sv)
    : n_qudits(sv.n_qudits)
    , d(sv.d)
    , dim(sv.dim)
    , dim_sq(sv.dim * sv.dim)
    , rho(sv.dim * sv.dim, Complex128(0.0, 0.0))
{
    // ρ = |ψ><ψ|:  rho[i*dim+j] = amp[i] * conj(amp[j])
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            rho[i * dim + j] = sv.amplitudes[i] * sv.amplitudes[j].conj();
}

// =============================================================================
// Initialisation helpers
// =============================================================================

void QuditDensityMatrix::initialize() {
    std::fill(rho.begin(), rho.end(), Complex128(0.0, 0.0));
    rho[0] = Complex128(1.0, 0.0);
}

void QuditDensityMatrix::symmetrize() {
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = i + 1; j < dim; ++j) {
            const Complex128 a = rho[i * dim + j];
            const Complex128 b = rho[j * dim + i];
            const Complex128 avg_real(0.5 * (a.real + b.real),
                                      0.5 * (a.imag - b.imag));
            rho[i * dim + j] = avg_real;
            rho[j * dim + i] = avg_real.conj();
        }
        // Diagonal must be real
        rho[i * dim + i].imag = 0.0;
    }
}

void QuditDensityMatrix::normalize() {
    const double tr = trace();
    if (std::abs(tr) < 1e-15) return;
    const double inv = 1.0 / tr;
    for (auto& v : rho) { v.real *= inv; v.imag *= inv; }
}

double QuditDensityMatrix::trace() const {
    double tr = 0.0;
    for (size_t i = 0; i < dim; ++i)
        tr += rho[i * dim + i].real;
    return tr;
}

double QuditDensityMatrix::purity() const {
    // Tr(ρ²) = Σ_{i,j} |ρ_{ij}|²  (because ρ is Hermitian, this equals Σ_{i,j} ρ_{ij} ρ_{ji})
    double s = 0.0;
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            s += rho[i * dim + j].norm_sq();
    return s;
}

// =============================================================================
// apply_to_ket — apply d×d gate U to ket (row) index of ρ for qudit q
//
// Stride of qudit q in the ket index = d^q.
// For each group of d ket rows that differ only in the value of qudit q:
//   new_rho[(base + k*stride)*dim + j] = Σ_l U[k,l] * rho[(base + l*stride)*dim + j]
// We batch over all bra columns j together for cache efficiency.
// =============================================================================

void QuditDensityMatrix::apply_to_ket(int q, const std::vector<Complex128>& U)
{
    const size_t stride = ipow(static_cast<size_t>(d), q);
    const size_t block  = stride * static_cast<size_t>(d);

    std::vector<Complex128> old_row(static_cast<size_t>(d));
    std::vector<Complex128> new_row(static_cast<size_t>(d));

    for (size_t outer = 0; outer < dim; outer += block) {
        for (size_t inner = 0; inner < stride; ++inner) {
            const size_t base = outer + inner;
            // For each bra column j, apply U to the d ket amplitudes
            for (size_t j = 0; j < dim; ++j) {
                // Extract d values: old_row[k] = rho[(base + k*stride)*dim + j]
                for (int k = 0; k < d; ++k)
                    old_row[static_cast<size_t>(k)] =
                        rho[(base + static_cast<size_t>(k) * stride) * dim + j];
                // Transform: new_row[k] = Σ_l U[k*d+l] * old_row[l]
                for (int k = 0; k < d; ++k) {
                    Complex128 acc(0.0, 0.0);
                    for (int l = 0; l < d; ++l)
                        acc += U[static_cast<size_t>(k * d + l)]
                             * old_row[static_cast<size_t>(l)];
                    new_row[static_cast<size_t>(k)] = acc;
                }
                // Write back
                for (int k = 0; k < d; ++k)
                    rho[(base + static_cast<size_t>(k) * stride) * dim + j] =
                        new_row[static_cast<size_t>(k)];
            }
        }
    }
}

// =============================================================================
// apply_to_bra — apply U† to bra (column) index of ρ for qudit q
//
// Equivalent to applying conj(U)^T to the column index.
// rho[i*dim + (base + k*stride)] ← Σ_l conj(U[l,k]) * rho[i*dim + (base + l*stride)]
// which is: apply U† to the column qudit-q slice for each ket row i.
// =============================================================================

void QuditDensityMatrix::apply_to_bra(int q, const std::vector<Complex128>& U)
{
    const size_t stride = ipow(static_cast<size_t>(d), q);
    const size_t block  = stride * static_cast<size_t>(d);

    std::vector<Complex128> old_col(static_cast<size_t>(d));
    std::vector<Complex128> new_col(static_cast<size_t>(d));

    for (size_t outer = 0; outer < dim; outer += block) {
        for (size_t inner = 0; inner < stride; ++inner) {
            const size_t base = outer + inner;
            for (size_t i = 0; i < dim; ++i) {
                for (int k = 0; k < d; ++k)
                    old_col[static_cast<size_t>(k)] =
                        rho[i * dim + (base + static_cast<size_t>(k) * stride)];
                for (int k = 0; k < d; ++k) {
                    Complex128 acc(0.0, 0.0);
                    for (int l = 0; l < d; ++l)
                        acc += U[static_cast<size_t>(k * d + l)].conj()
                             * old_col[static_cast<size_t>(l)];
                    new_col[static_cast<size_t>(k)] = acc;
                }
                for (int k = 0; k < d; ++k)
                    rho[i * dim + (base + static_cast<size_t>(k) * stride)] =
                        new_col[static_cast<size_t>(k)];
            }
        }
    }
}

// =============================================================================
// apply_1qudit — ρ → U_q ρ U_q†
// =============================================================================

void QuditDensityMatrix::apply_1qudit(int q, const std::vector<Complex128>& U,
                                      ValidationOptions validation)
{
    detail::check_qudit(q, n_qudits, "QuditDensityMatrix::apply_1qudit");
    detail::check_size(U.size(), static_cast<size_t>(d) * static_cast<size_t>(d),
                       "QuditDensityMatrix::apply_1qudit", "matrix");
    detail::check_unitary(U, static_cast<size_t>(d), validation,
                          "QuditDensityMatrix::apply_1qudit");
    apply_to_ket(q, U);
    apply_to_bra(q, U);
}

// =============================================================================
// apply_to_ket_2q — apply d²×d² gate U to ket indices for (q0, q1)
//
// Mirrors QuditStatevector::apply_2qudit but operates on rho's ket dimension.
// Matrix index convention (project LSB-first): the FIRST argument is the
// LEAST significant digit, sub-index = x1 * d + x0.
// For each base index (where both qudit q0 and q1 = 0 in the ket):
//   - For each bra column j, extract d² amplitudes, multiply by U, write back.
// =============================================================================

void QuditDensityMatrix::apply_to_ket_2q(int q0, int q1,
                                         const std::vector<Complex128>& U)
{
    if (q0 == q1)
        throw std::invalid_argument("apply_to_ket_2q: q0 must not equal q1");

    const size_t stride0 = ipow(static_cast<size_t>(d), q0);
    const size_t stride1 = ipow(static_cast<size_t>(d), q1);
    const int dd = d * d;

    std::vector<Complex128> old_amp(static_cast<size_t>(dd));
    std::vector<Complex128> new_amp(static_cast<size_t>(dd));

    for (size_t idx = 0; idx < dim; ++idx) {
        if ((idx / stride0) % static_cast<size_t>(d) != 0) continue;
        if ((idx / stride1) % static_cast<size_t>(d) != 0) continue;

        // For each bra column j
        for (size_t j = 0; j < dim; ++j) {
            // Extract d² ket values
            for (int x0 = 0; x0 < d; ++x0)
                for (int x1 = 0; x1 < d; ++x1)
                    old_amp[static_cast<size_t>(x1 * d + x0)] =
                        rho[(idx + static_cast<size_t>(x0) * stride0
                                 + static_cast<size_t>(x1) * stride1) * dim + j];

            // Matrix multiply
            for (int r = 0; r < dd; ++r) {
                Complex128 acc(0.0, 0.0);
                for (int c = 0; c < dd; ++c)
                    acc += U[static_cast<size_t>(r * dd + c)]
                         * old_amp[static_cast<size_t>(c)];
                new_amp[static_cast<size_t>(r)] = acc;
            }

            // Write back
            for (int x0 = 0; x0 < d; ++x0)
                for (int x1 = 0; x1 < d; ++x1)
                    rho[(idx + static_cast<size_t>(x0) * stride0
                              + static_cast<size_t>(x1) * stride1) * dim + j] =
                        new_amp[static_cast<size_t>(x1 * d + x0)];
        }
    }
}

// =============================================================================
// apply_to_bra_2q — apply U† to bra indices for (q0, q1)
// Matrix index convention as apply_to_ket_2q: sub-index = x1 * d + x0
// (first argument = least significant digit).
// =============================================================================

void QuditDensityMatrix::apply_to_bra_2q(int q0, int q1,
                                          const std::vector<Complex128>& U)
{
    if (q0 == q1)
        throw std::invalid_argument("apply_to_bra_2q: q0 must not equal q1");

    const size_t stride0 = ipow(static_cast<size_t>(d), q0);
    const size_t stride1 = ipow(static_cast<size_t>(d), q1);
    const int dd = d * d;

    std::vector<Complex128> old_amp(static_cast<size_t>(dd));
    std::vector<Complex128> new_amp(static_cast<size_t>(dd));

    for (size_t idx = 0; idx < dim; ++idx) {
        if ((idx / stride0) % static_cast<size_t>(d) != 0) continue;
        if ((idx / stride1) % static_cast<size_t>(d) != 0) continue;

        // For each ket row i
        for (size_t i = 0; i < dim; ++i) {
            // Extract d² bra values
            for (int x0 = 0; x0 < d; ++x0)
                for (int x1 = 0; x1 < d; ++x1)
                    old_amp[static_cast<size_t>(x1 * d + x0)] =
                        rho[i * dim + (idx + static_cast<size_t>(x0) * stride0
                                           + static_cast<size_t>(x1) * stride1)];

            // Matrix multiply with conj(U)
            for (int r = 0; r < dd; ++r) {
                Complex128 acc(0.0, 0.0);
                for (int c = 0; c < dd; ++c)
                    acc += U[static_cast<size_t>(r * dd + c)].conj()
                         * old_amp[static_cast<size_t>(c)];
                new_amp[static_cast<size_t>(r)] = acc;
            }

            // Write back
            for (int x0 = 0; x0 < d; ++x0)
                for (int x1 = 0; x1 < d; ++x1)
                    rho[i * dim + (idx + static_cast<size_t>(x0) * stride0
                                       + static_cast<size_t>(x1) * stride1)] =
                        new_amp[static_cast<size_t>(x1 * d + x0)];
        }
    }
}

// =============================================================================
// apply_2qudit — ρ → U_{q0,q1} ρ U_{q0,q1}†
// =============================================================================

void QuditDensityMatrix::apply_2qudit(int q0, int q1,
                                      const std::vector<Complex128>& U,
                                      ValidationOptions validation)
{
    detail::check_qudit(q0, n_qudits, "QuditDensityMatrix::apply_2qudit");
    detail::check_qudit(q1, n_qudits, "QuditDensityMatrix::apply_2qudit");
    detail::check_distinct2(q0, q1, "QuditDensityMatrix::apply_2qudit", "qudits");
    detail::check_size(U.size(), static_cast<size_t>(d) * d * d * d,
                       "QuditDensityMatrix::apply_2qudit", "matrix");
    detail::check_unitary(U, static_cast<size_t>(d) * static_cast<size_t>(d),
                          validation, "QuditDensityMatrix::apply_2qudit");
    apply_to_ket_2q(q0, q1, U);
    apply_to_bra_2q(q0, q1, U);
}

// =============================================================================
// apply_kraus_1qudit — ρ → Σ_k K_k ρ K_k†
// =============================================================================

void QuditDensityMatrix::apply_kraus_1qudit(
    int q, const std::vector<std::vector<Complex128>>& K_ops,
    ValidationOptions validation)
{
    detail::check_qudit(q, n_qudits, "QuditDensityMatrix::apply_kraus_1qudit");
    for (const auto& K : K_ops)
        detail::check_size(K.size(), static_cast<size_t>(d) * static_cast<size_t>(d),
                           "QuditDensityMatrix::apply_kraus_1qudit", "Kraus operator");
    detail::check_kraus_tp(K_ops, static_cast<size_t>(d), validation,
                           "QuditDensityMatrix::apply_kraus_1qudit");
    std::vector<Complex128> rho_new(dim_sq, Complex128(0.0, 0.0));

    for (const auto& K : K_ops) {
        // temp = copy of rho, then apply K on ket and K† on bra
        std::vector<Complex128> temp = rho;

        // Apply K to ket index
        const size_t stride = ipow(static_cast<size_t>(d), q);
        const size_t block  = stride * static_cast<size_t>(d);

        std::vector<Complex128> old_v(static_cast<size_t>(d));
        std::vector<Complex128> new_v(static_cast<size_t>(d));

        // Ket pass
        for (size_t outer = 0; outer < dim; outer += block) {
            for (size_t inner = 0; inner < stride; ++inner) {
                const size_t base = outer + inner;
                for (size_t j = 0; j < dim; ++j) {
                    for (int k = 0; k < d; ++k)
                        old_v[static_cast<size_t>(k)] =
                            temp[(base + static_cast<size_t>(k) * stride) * dim + j];
                    for (int k = 0; k < d; ++k) {
                        Complex128 acc(0.0, 0.0);
                        for (int l = 0; l < d; ++l)
                            acc += K[static_cast<size_t>(k * d + l)]
                                 * old_v[static_cast<size_t>(l)];
                        new_v[static_cast<size_t>(k)] = acc;
                    }
                    for (int k = 0; k < d; ++k)
                        temp[(base + static_cast<size_t>(k) * stride) * dim + j] =
                            new_v[static_cast<size_t>(k)];
                }
            }
        }

        // Bra pass (apply conj(K) to bra index of temp)
        for (size_t outer = 0; outer < dim; outer += block) {
            for (size_t inner = 0; inner < stride; ++inner) {
                const size_t base = outer + inner;
                for (size_t i = 0; i < dim; ++i) {
                    for (int k = 0; k < d; ++k)
                        old_v[static_cast<size_t>(k)] =
                            temp[i * dim + (base + static_cast<size_t>(k) * stride)];
                    for (int k = 0; k < d; ++k) {
                        Complex128 acc(0.0, 0.0);
                        for (int l = 0; l < d; ++l)
                            acc += K[static_cast<size_t>(k * d + l)].conj()
                                 * old_v[static_cast<size_t>(l)];
                        new_v[static_cast<size_t>(k)] = acc;
                    }
                    for (int k = 0; k < d; ++k)
                        temp[i * dim + (base + static_cast<size_t>(k) * stride)] =
                            new_v[static_cast<size_t>(k)];
                }
            }
        }

        // Accumulate
        for (size_t idx = 0; idx < dim_sq; ++idx)
            rho_new[idx] += temp[idx];
    }

    rho = std::move(rho_new);
}

// =============================================================================
// apply_kraus_2qudit — ρ → Σ_k K_k ρ K_k†  (2-qudit Kraus)
// =============================================================================

void QuditDensityMatrix::apply_kraus_2qudit(
    int q0, int q1,
    const std::vector<std::vector<Complex128>>& K_ops,
    ValidationOptions validation)
{
    detail::check_qudit(q0, n_qudits, "QuditDensityMatrix::apply_kraus_2qudit");
    detail::check_qudit(q1, n_qudits, "QuditDensityMatrix::apply_kraus_2qudit");
    detail::check_distinct2(q0, q1, "QuditDensityMatrix::apply_kraus_2qudit", "qudits");
    for (const auto& K : K_ops)
        detail::check_size(K.size(), static_cast<size_t>(d) * d * d * d,
                           "QuditDensityMatrix::apply_kraus_2qudit", "Kraus operator");
    detail::check_kraus_tp(K_ops, static_cast<size_t>(d) * static_cast<size_t>(d),
                           validation,
                           "QuditDensityMatrix::apply_kraus_2qudit");
    std::vector<Complex128> rho_new(dim_sq, Complex128(0.0, 0.0));

    for (const auto& K : K_ops) {
        std::vector<Complex128> temp = rho;

        // Apply K to ket
        {
            QuditDensityMatrix tmp_dm(n_qudits, d);
            tmp_dm.rho = temp;
            tmp_dm.apply_to_ket_2q(q0, q1, K);
            tmp_dm.apply_to_bra_2q(q0, q1, K);
            temp = std::move(tmp_dm.rho);
        }

        for (size_t idx = 0; idx < dim_sq; ++idx)
            rho_new[idx] += temp[idx];
    }

    rho = std::move(rho_new);
}

// =============================================================================
// apply_lindblad_step — first-order Euler step of Lindblad master equation:
//   ρ ← ρ + dt * Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
//
// For qudit q, L_k acts on the d-dimensional subspace of that qudit.
// We expand L_k and L_k†L_k as full operators on the qudit subspace,
// then apply them to the ket/bra indices of ρ.
// =============================================================================

void QuditDensityMatrix::apply_lindblad_step(
    int q, const std::vector<QuditLindbladOp>& ops, double dt)
{
    for (const auto& [L, gamma] : ops) {
        const double coeff = dt * gamma;
        const auto Ld = dm_dag(L, d);
        const auto LdL = dm_mat_mul(Ld, L, d);

        // Save the original ρ; all three terms are computed from rho_orig
        const std::vector<Complex128> rho_orig = rho;

        const size_t stride = ipow(static_cast<size_t>(d), q);
        const size_t block  = stride * static_cast<size_t>(d);

        std::vector<Complex128> old_v(static_cast<size_t>(d));
        std::vector<Complex128> new_v(static_cast<size_t>(d));

        // --- Term 1: + coeff * L ρ_orig L† ---
        {
            std::vector<Complex128> temp = rho_orig;

            // Apply L to ket
            for (size_t outer = 0; outer < dim; outer += block) {
                for (size_t inner = 0; inner < stride; ++inner) {
                    const size_t base = outer + inner;
                    for (size_t j = 0; j < dim; ++j) {
                        for (int k = 0; k < d; ++k)
                            old_v[static_cast<size_t>(k)] =
                                temp[(base + static_cast<size_t>(k) * stride) * dim + j];
                        for (int k = 0; k < d; ++k) {
                            Complex128 acc(0.0, 0.0);
                            for (int l = 0; l < d; ++l)
                                acc += L[static_cast<size_t>(k * d + l)]
                                     * old_v[static_cast<size_t>(l)];
                            new_v[static_cast<size_t>(k)] = acc;
                        }
                        for (int k = 0; k < d; ++k)
                            temp[(base + static_cast<size_t>(k) * stride) * dim + j] =
                                new_v[static_cast<size_t>(k)];
                    }
                }
            }

            // Apply conj(L) to bra
            for (size_t outer = 0; outer < dim; outer += block) {
                for (size_t inner = 0; inner < stride; ++inner) {
                    const size_t base = outer + inner;
                    for (size_t i = 0; i < dim; ++i) {
                        for (int k = 0; k < d; ++k)
                            old_v[static_cast<size_t>(k)] =
                                temp[i * dim + (base + static_cast<size_t>(k) * stride)];
                        for (int k = 0; k < d; ++k) {
                            Complex128 acc(0.0, 0.0);
                            for (int l = 0; l < d; ++l)
                                acc += L[static_cast<size_t>(k * d + l)].conj()
                                     * old_v[static_cast<size_t>(l)];
                            new_v[static_cast<size_t>(k)] = acc;
                        }
                        for (int k = 0; k < d; ++k)
                            temp[i * dim + (base + static_cast<size_t>(k) * stride)] =
                                new_v[static_cast<size_t>(k)];
                    }
                }
            }

            // Add coeff * (L ρ_orig L†) to rho
            for (size_t idx = 0; idx < dim_sq; ++idx)
                rho[idx] += temp[idx] * coeff;
        }

        // --- Term 2: - 0.5 * coeff * L†L ρ_orig  (LdL acts on ket index) ---
        {
            const double c2 = 0.5 * coeff;

            for (size_t outer = 0; outer < dim; outer += block) {
                for (size_t inner = 0; inner < stride; ++inner) {
                    const size_t base = outer + inner;
                    for (size_t j = 0; j < dim; ++j) {
                        for (int k = 0; k < d; ++k)
                            old_v[static_cast<size_t>(k)] =
                                rho_orig[(base + static_cast<size_t>(k) * stride) * dim + j];
                        for (int k = 0; k < d; ++k) {
                            Complex128 acc(0.0, 0.0);
                            for (int l = 0; l < d; ++l)
                                acc += LdL[static_cast<size_t>(k * d + l)]
                                     * old_v[static_cast<size_t>(l)];
                            new_v[static_cast<size_t>(k)] = acc;
                        }
                        for (int k = 0; k < d; ++k)
                            rho[(base + static_cast<size_t>(k) * stride) * dim + j] -=
                                new_v[static_cast<size_t>(k)] * c2;
                    }
                }
            }
        }

        // --- Term 3: - 0.5 * coeff * ρ_orig L†L  (LdL acts on bra index) ---
        // Applying M to bra: rho'[i,j] = Σ_k rho[i,k] * M†[k,j]
        // Since LdL is Hermitian (M† = M), bra application gives:
        // new_col[k] = Σ_l rho_orig[i, base+l*stride] * LdL[l,k]
        {
            const double c2 = 0.5 * coeff;

            for (size_t outer = 0; outer < dim; outer += block) {
                for (size_t inner = 0; inner < stride; ++inner) {
                    const size_t base = outer + inner;
                    for (size_t i = 0; i < dim; ++i) {
                        for (int k = 0; k < d; ++k)
                            old_v[static_cast<size_t>(k)] =
                                rho_orig[i * dim + (base + static_cast<size_t>(k) * stride)];
                        for (int k = 0; k < d; ++k) {
                            Complex128 acc(0.0, 0.0);
                            for (int l = 0; l < d; ++l)
                                // Applying M to bra (rho * M†, M=LdL Hermitian → M†=M):
                                // new_rho[i, base+k*stride] = Σ_l rho_orig[i,base+l*stride] * LdL[l,k]
                                acc += LdL[static_cast<size_t>(l * d + k)]
                                     * old_v[static_cast<size_t>(l)];
                            new_v[static_cast<size_t>(k)] = acc;
                        }
                        for (int k = 0; k < d; ++k)
                            rho[i * dim + (base + static_cast<size_t>(k) * stride)] -=
                                new_v[static_cast<size_t>(k)] * c2;
                    }
                }
            }
        }
    }
}

// =============================================================================
// apply_noise — apply all Kraus channels from a noise model
// If dt > 0, also apply Lindblad steps.
// =============================================================================

void QuditDensityMatrix::apply_noise(const QuditNoiseModel& model, double dt)
{
    for (const auto& [q, noise] : model.per_qudit) {
        if (!noise.kraus.ops.empty())
            apply_kraus_1qudit(q, noise.kraus.ops);
    }

    if (dt > 0.0) {
        for (const auto& [q, spec] : model.per_qudit) {
            if (!spec.lindblad.empty()) {
                apply_lindblad_step(q, spec.lindblad, dt);
            }
        }
    }
}

// =============================================================================
// apply_phase_oracle — ρ_{ij} ← phase(i) * conj(phase(j)) * ρ_{ij}
// =============================================================================

void QuditDensityMatrix::apply_phase_oracle(
    const std::function<Complex128(const std::vector<int>&)>& phase_fn)
{
    // Precompute phases for all basis states
    std::vector<Complex128> phases(dim);
    for (size_t i = 0; i < dim; ++i) {
        auto digits = QuditStatevector::index_to_digits(i, d, n_qudits);
        phases[i] = phase_fn(digits);
    }

    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            rho[i * dim + j] = rho[i * dim + j] * phases[i] * phases[j].conj();
}

// =============================================================================
// apply_function_oracle — |x>|y> → |x>|(y + f(x)) mod d>
// =============================================================================

void QuditDensityMatrix::apply_function_oracle(
    int n_query, int n_output,
    const std::function<std::vector<int>(const std::vector<int>&)>& f)
{
    if (n_query < 1 || n_output < 0)
        throw std::invalid_argument(
            "apply_function_oracle: n_query >= 1 and n_output >= 0 required");
    if (n_query + n_output > n_qudits)
        throw std::invalid_argument(
            "apply_function_oracle: n_query + n_output exceeds n_qudits");

    // Build permutation perm[old_idx] = new_idx
    std::vector<size_t> perm(dim);
    for (size_t idx = 0; idx < dim; ++idx) {
        auto digits = QuditStatevector::index_to_digits(idx, d, n_qudits);
        std::vector<int> x(digits.begin(), digits.begin() + n_query);
        const auto fx = f(x);
        if (static_cast<int>(fx.size()) != n_output)
            throw std::invalid_argument(
                "apply_function_oracle: f returned wrong number of digits");
        for (int i = 0; i < n_output; ++i) {
            const int v = fx[static_cast<size_t>(i)];
            if (v < 0 || v >= d)
                throw std::invalid_argument(
                    "apply_function_oracle: f returned value out of [0, d)");
            const size_t qi = static_cast<size_t>(n_query + i);
            digits[qi] = (digits[qi] + v) % d;
        }
        perm[idx] = QuditStatevector::digits_to_index(digits, d);
    }

    // Apply permutation: rho_new[perm[i]*dim + perm[j]] = rho[i*dim + j]
    std::vector<Complex128> rho_new(dim_sq, Complex128(0.0, 0.0));
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            rho_new[perm[i] * dim + perm[j]] = rho[i * dim + j];

    rho = std::move(rho_new);
}

// =============================================================================
// measure — sample from diagonal; collapse ρ
// =============================================================================

std::vector<int> QuditDensityMatrix::measure(uint64_t seed)
{
    std::mt19937_64 rng(seed == 0
        ? static_cast<uint64_t>(std::random_device{}())
        : seed);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    const double roll = udist(rng);
    double cumulative = 0.0;
    size_t chosen = dim - 1;
    for (size_t i = 0; i < dim; ++i) {
        cumulative += rho[i * dim + i].real;
        if (roll <= cumulative) {
            chosen = i;
            break;
        }
    }

    // Collapse to post-measurement state
    std::fill(rho.begin(), rho.end(), Complex128(0.0, 0.0));
    rho[chosen * dim + chosen] = Complex128(1.0, 0.0);

    return QuditStatevector::index_to_digits(chosen, d, n_qudits);
}

// =============================================================================
// partial_trace — keep `keep_qudits`, trace out all others
// =============================================================================

QuditDensityMatrix QuditDensityMatrix::partial_trace(
    const std::vector<int>& keep_qudits) const
{
    if (keep_qudits.empty()) {
        throw std::invalid_argument("QuditDensityMatrix::partial_trace: keep_qudits must not be empty");
    }

    const int n_keep  = static_cast<int>(keep_qudits.size());
    const int n_trace = n_qudits - n_keep;

    // Build a boolean mask of which qudits are kept
    std::vector<bool> is_kept(static_cast<size_t>(n_qudits), false);
    for (int q : keep_qudits)
        is_kept[static_cast<size_t>(q)] = true;

    // Collect traced-out qudit indices in order
    std::vector<int> trace_qudits;
    trace_qudits.reserve(static_cast<size_t>(n_trace));
    for (int q = 0; q < n_qudits; ++q)
        if (!is_kept[static_cast<size_t>(q)])
            trace_qudits.push_back(q);

    const size_t dim_keep  = ipow(static_cast<size_t>(d), n_keep);
    const size_t dim_trace = ipow(static_cast<size_t>(d), n_trace);

    QuditDensityMatrix result(n_keep, d);
    // Reset to zero (constructor sets rho[0]=1)
    std::fill(result.rho.begin(), result.rho.end(), Complex128(0.0, 0.0));

    // Enumerate i_keep in [0, dim_keep) and j_keep in [0, dim_keep)
    for (size_t i_keep = 0; i_keep < dim_keep; ++i_keep) {
        // Digits of i_keep in keep_qudits order
        auto i_keep_digits = QuditStatevector::index_to_digits(
            i_keep, d, n_keep);

        for (size_t j_keep = 0; j_keep < dim_keep; ++j_keep) {
            auto j_keep_digits = QuditStatevector::index_to_digits(
                j_keep, d, n_keep);

            Complex128 acc(0.0, 0.0);

            // Sum over all traced-out configurations
            for (size_t x_trace = 0; x_trace < dim_trace; ++x_trace) {
                auto x_trace_digits = QuditStatevector::index_to_digits(
                    x_trace, d, n_trace);

                // Build full digit arrays for ket index i_full and bra index j_full
                std::vector<int> i_full_digits(static_cast<size_t>(n_qudits), 0);
                std::vector<int> j_full_digits(static_cast<size_t>(n_qudits), 0);

                // Fill kept qudits
                for (int ki = 0; ki < n_keep; ++ki) {
                    const int q = keep_qudits[static_cast<size_t>(ki)];
                    i_full_digits[static_cast<size_t>(q)] =
                        i_keep_digits[static_cast<size_t>(ki)];
                    j_full_digits[static_cast<size_t>(q)] =
                        j_keep_digits[static_cast<size_t>(ki)];
                }

                // Fill traced-out qudits (same value for ket and bra)
                for (int ti = 0; ti < n_trace; ++ti) {
                    const int q = trace_qudits[static_cast<size_t>(ti)];
                    i_full_digits[static_cast<size_t>(q)] =
                        x_trace_digits[static_cast<size_t>(ti)];
                    j_full_digits[static_cast<size_t>(q)] =
                        x_trace_digits[static_cast<size_t>(ti)];
                }

                const size_t i_full =
                    QuditStatevector::digits_to_index(i_full_digits, d);
                const size_t j_full =
                    QuditStatevector::digits_to_index(j_full_digits, d);

                acc += rho[i_full * dim + j_full];
            }

            result.rho[i_keep * dim_keep + j_keep] = acc;
        }
    }

    return result;
}

} // namespace lindblad
