#include "lindblad/qudit/qudit_mps.hpp"

#include "lindblad/detail/validate.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

namespace lindblad {

// =============================================================================
// Local helpers — conversion between Complex128 and std::complex<double>
// =============================================================================

static inline std::complex<double> to_std(const Complex128& c) noexcept {
    return std::complex<double>(c.real, c.imag);
}

static inline Complex128 from_std(const std::complex<double>& z) noexcept {
    return Complex128(z.real(), z.imag());
}

// One-time loud warning when the (currently broken) BDCSVD backend is selected
// on the qudit MPS. Mirrors the qubit-layer warning (audit F-23).
static void warn_bdc_broken_once_qudit() {
    static bool warned = false;
    if (warned) return;
    warned = true;
    std::cerr <<
        "\n***************************************************************\n"
        "[lindblad] WARNING: SVDMethod::BDC (Eigen BDCSVD) is SELECTED on the\n"
        "  qudit MPS but is CURRENTLY BROKEN for complex / degenerate inputs\n"
        "  (R.1.11.2 bug, docs/plans/eigen-bdcsvd-bug.md). Results may be\n"
        "  SILENTLY WRONG. Use SVDMethod::Jacobi (the default) until fixed.\n"
        "***************************************************************\n";
}

// SVD with the selected backend. Jacobi is accurate (default); BDC is faster
// but currently broken, so it warns loudly. Writes thin U, singular values S,
// and V (not V^dagger).
static void qmps_svd(const Eigen::MatrixXcd& M, SVDMethod method,
                     Eigen::MatrixXcd& U, Eigen::VectorXd& S, Eigen::MatrixXcd& V) {
    if (method == SVDMethod::BDC) {
        warn_bdc_broken_once_qudit();
        Eigen::BDCSVD<Eigen::MatrixXcd> svd(
            M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        U = svd.matrixU();
        S = svd.singularValues();
        V = svd.matrixV();
    } else {
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
            M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        U = svd.matrixU();
        S = svd.singularValues();
        V = svd.matrixV();
    }
}

// =============================================================================
// MPSSiteTensor
// =============================================================================

MPSSiteTensor::MPSSiteTensor(int d_, int chi_L_, int chi_R_)
    : d(d_), chi_L(chi_L_), chi_R(chi_R_),
      data(static_cast<size_t>(d_) * static_cast<size_t>(chi_L_) *
               static_cast<size_t>(chi_R_),
           Complex128(0.0, 0.0))
{}

Complex128& MPSSiteTensor::at(int sigma, int aL, int aR) {
    const size_t idx = static_cast<size_t>(sigma) *
                           static_cast<size_t>(chi_L) *
                           static_cast<size_t>(chi_R) +
                       static_cast<size_t>(aL) * static_cast<size_t>(chi_R) +
                       static_cast<size_t>(aR);
    return data[idx];
}

const Complex128& MPSSiteTensor::at(int sigma, int aL, int aR) const {
    const size_t idx = static_cast<size_t>(sigma) *
                           static_cast<size_t>(chi_L) *
                           static_cast<size_t>(chi_R) +
                       static_cast<size_t>(aL) * static_cast<size_t>(chi_R) +
                       static_cast<size_t>(aR);
    return data[idx];
}

// Row index = sigma * chi_L + aL; col index = aR.
Eigen::MatrixXcd MPSSiteTensor::as_left_matrix() const {
    Eigen::MatrixXcd M(d * chi_L, chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int aR = 0; aR < chi_R; ++aR)
                M(sigma * chi_L + aL, aR) = to_std(at(sigma, aL, aR));
    return M;
}

// Row index = aL; col index = sigma * chi_R + aR.
Eigen::MatrixXcd MPSSiteTensor::as_right_matrix() const {
    Eigen::MatrixXcd M(chi_L, d * chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int aR = 0; aR < chi_R; ++aR)
                M(aL, sigma * chi_R + aR) = to_std(at(sigma, aL, aR));
    return M;
}

MPSSiteTensor MPSSiteTensor::from_left_matrix(const Eigen::MatrixXcd& M,
                                              int d, int chi_L) {
    const int chi_R = static_cast<int>(M.cols());
    if (M.rows() != static_cast<Eigen::Index>(d) * chi_L)
        throw std::invalid_argument(
            "MPSSiteTensor::from_left_matrix: row count must be d * chi_L");
    MPSSiteTensor T(d, chi_L, chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int aR = 0; aR < chi_R; ++aR)
                T.at(sigma, aL, aR) = from_std(M(sigma * chi_L + aL, aR));
    return T;
}

MPSSiteTensor MPSSiteTensor::from_right_matrix(const Eigen::MatrixXcd& M,
                                               int d, int chi_R) {
    const int chi_L = static_cast<int>(M.rows());
    if (M.cols() != static_cast<Eigen::Index>(d) * chi_R)
        throw std::invalid_argument(
            "MPSSiteTensor::from_right_matrix: col count must be d * chi_R");
    MPSSiteTensor T(d, chi_L, chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int aR = 0; aR < chi_R; ++aR)
                T.at(sigma, aL, aR) = from_std(M(aL, sigma * chi_R + aR));
    return T;
}

// =============================================================================
// QuditMPS — construction
// =============================================================================

size_t QuditMPS::ipow(size_t base, int exp) noexcept {
    size_t result = 1;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

QuditMPS::QuditMPS(int n_qudits_, int d_, int max_bond_dim_, double svd_cutoff_)
    : n_qudits(n_qudits_), d(d_),
      max_bond_dim(max_bond_dim_), svd_cutoff(svd_cutoff_)
{
    if (d_ < 2)
        throw std::invalid_argument("QuditMPS: d must be >= 2");
    if (n_qudits_ < 1)
        throw std::invalid_argument("QuditMPS: n_qudits must be >= 1");
    if (max_bond_dim_ < 1)
        throw std::invalid_argument("QuditMPS: max_bond_dim must be >= 1");

    tensors.reserve(static_cast<size_t>(n_qudits_));
    for (int q = 0; q < n_qudits_; ++q) {
        MPSSiteTensor T(d_, 1, 1);
        T.at(0, 0, 0) = Complex128(1.0, 0.0);  // |0> on every site
        tensors.push_back(std::move(T));
    }
}

QuditMPS::QuditMPS(const QuditStatevector& sv,
                   int max_bond_dim_, double svd_cutoff_)
    : n_qudits(sv.n_qudits), d(sv.d),
      max_bond_dim(max_bond_dim_), svd_cutoff(svd_cutoff_)
{
    if (max_bond_dim_ < 1)
        throw std::invalid_argument("QuditMPS: max_bond_dim must be >= 1");

    tensors.reserve(static_cast<size_t>(n_qudits));

    // Pack the amplitudes into a dense complex matrix.
    //
    // At step q the residual matrix has
    //   rows  = chi_L * d   (left bond of current site times physical leg)
    //   cols  = d^(n-1-q)   (flat index of qudits q+1..n-1, little-endian)
    //
    // Initial reshape (q=0, chi_L=1): rows = d, cols = d^(n-1).
    //   M[sigma_0, col] = sv.amplitudes[sigma_0 * d^0 + col * d]
    //                   = sv.amplitudes[sigma_0 + col * d]
    //
    // Subsequent residuals come from absorbing S * V^dagger into the left side
    // of the next reshape: M_new[(aL_prev * d) + sigma_q, col] = residual[..]
    //   where aL_prev runs over chi (truncated rank from previous step).

    int chi_L = 1;
    size_t cols = ipow(static_cast<size_t>(d), n_qudits - 1);

    // Initial residual matrix from sv.amplitudes
    Eigen::MatrixXcd M(static_cast<Eigen::Index>(d) * chi_L,
                       static_cast<Eigen::Index>(cols));
    for (int sigma = 0; sigma < d; ++sigma)
        for (size_t col = 0; col < cols; ++col)
            M(sigma, static_cast<Eigen::Index>(col)) =
                to_std(sv.amplitudes[static_cast<size_t>(sigma) + col *
                                     static_cast<size_t>(d)]);

    for (int q = 0; q < n_qudits - 1; ++q) {
        Eigen::MatrixXcd U, V;
        Eigen::VectorXd S;
        qmps_svd(M, svd_method, U, S, V);

        // Discarded-weight truncation, identical to MPSState::svd_truncate:
        // svd_cutoff is the fraction of total weight (sum of sigma^2) a split
        // may throw away, NOT a magnitude threshold. The qudit layer mirrors
        // the qubit layer at general d, and that has to extend to what a bond
        // dimension means; a magnitude rule also makes retained chi depend on
        // the input's scale and on how the target rounded its way there.
        const int full_rank = static_cast<int>(S.size());
        double total = 0.0;
        for (int i = 0; i < full_rank; ++i) total += S(i) * S(i);
        const double budget = svd_cutoff * total;

        int chi = full_rank;
        double discarded = 0.0;
        while (chi > 1) {
            const double w = S(chi - 1) * S(chi - 1);
            if (discarded + w > budget) break;
            discarded += w;
            --chi;
        }
        if (chi > max_bond_dim) chi = max_bond_dim;
        if (chi == 0) chi = 1;

        // Left tensor: shape (d, chi_L, chi) from leftmost chi columns of U.
        // The residual matrix M is built with rows ordered (left-bond major):
        //   row = aL * d + sigma   (see the M_next reshape below and the initial
        //   reshape where chi_L = 1). U inherits that row order, so the physical
        //   index sigma is the LOW digit and the left bond aL is the HIGH digit.
        // Decode with the SAME ordering (this also matches the final-site read
        // `M(aL * d + sigma, 0)`). Using `sigma * chi_L + aL` here transposed the
        // physical/bond indices on intermediate sites (chi_L > 1), corrupting any
        // state needing >= 3 sites with a nontrivial interior bond (R.1.11.2 fix).
        MPSSiteTensor Tq(d, chi_L, chi);
        for (int sigma = 0; sigma < d; ++sigma)
            for (int aL = 0; aL < chi_L; ++aL)
                for (int alpha = 0; alpha < chi; ++alpha)
                    Tq.at(sigma, aL, alpha) =
                        from_std(U(aL * d + sigma, alpha));
        tensors.push_back(std::move(Tq));

        // Build residual M' = diag(S_truncated) * V^dagger_truncated
        // Vt rows are indexed by the singular values; we keep top `chi`.
        Eigen::MatrixXcd Vt_trunc(chi, static_cast<Eigen::Index>(cols));
        for (int alpha = 0; alpha < chi; ++alpha)
            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols); ++c)
                Vt_trunc(alpha, c) = std::conj(V(c, alpha));

        Eigen::MatrixXcd residual(chi, static_cast<Eigen::Index>(cols));
        for (int alpha = 0; alpha < chi; ++alpha)
            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols); ++c)
                residual(alpha, c) = S(alpha) * Vt_trunc(alpha, c);

        // Reshape for next step.  Residual columns currently index the
        // sub-register of qudits (q+1..n-1) in little-endian order, so qudit
        // (q+1) has stride 1 and the higher qudits have stride d, d^2, ...
        //   old_col = sigma_{q+1} + next_col * d
        // where next_col indexes qudits (q+2..n-1) — again little-endian.
        // We fold sigma_{q+1} into the row dimension:
        //   M_next[aL_prev * d + sigma_{q+1}, next_col] =
        //       residual[aL_prev, sigma_{q+1} + next_col * d]
        const size_t new_cols = cols / static_cast<size_t>(d);
        Eigen::MatrixXcd M_next(static_cast<Eigen::Index>(chi) * d,
                                static_cast<Eigen::Index>(new_cols));
        for (int aL_prev = 0; aL_prev < chi; ++aL_prev)
            for (int sigma_next = 0; sigma_next < d; ++sigma_next)
                for (size_t next_col = 0; next_col < new_cols; ++next_col)
                    M_next(aL_prev * d + sigma_next,
                           static_cast<Eigen::Index>(next_col)) =
                        residual(aL_prev,
                                 static_cast<Eigen::Index>(
                                     static_cast<size_t>(sigma_next) +
                                     next_col * static_cast<size_t>(d)));
        M = std::move(M_next);
        chi_L = chi;
        cols = new_cols;
    }

    // Final site: M has shape (chi_L * d, 1).
    {
        MPSSiteTensor Tlast(d, chi_L, 1);
        for (int sigma = 0; sigma < d; ++sigma)
            for (int aL = 0; aL < chi_L; ++aL)
                Tlast.at(sigma, aL, 0) = from_std(M(aL * d + sigma, 0));
        tensors.push_back(std::move(Tlast));
    }
}

// =============================================================================
// to_statevector — left-to-right contraction
// =============================================================================

QuditStatevector QuditMPS::to_statevector() const {
    // C has shape (dim_so_far, chi).
    // Start with chi = chi_L of site 0 (= 1), dim_so_far = 1, C = [[1]].
    Eigen::MatrixXcd C(1, 1);
    C(0, 0) = std::complex<double>(1.0, 0.0);
    size_t dim_so_far = 1;

    for (int q = 0; q < n_qudits; ++q) {
        const auto& T = tensors[static_cast<size_t>(q)];
        const int chi_L = T.chi_L;
        const int chi_R = T.chi_R;
        const size_t new_dim = dim_so_far * static_cast<size_t>(d);

        Eigen::MatrixXcd next(static_cast<Eigen::Index>(new_dim), chi_R);
        next.setZero();
        for (size_t i = 0; i < dim_so_far; ++i) {
            for (int sigma = 0; sigma < d; ++sigma) {
                // new row = i + sigma * dim_so_far (little-endian: qudit q
                // is the most-significant new digit relative to qudits 0..q-1)
                const size_t new_row = i + static_cast<size_t>(sigma) * dim_so_far;
                for (int aR = 0; aR < chi_R; ++aR) {
                    std::complex<double> acc(0.0, 0.0);
                    for (int aL = 0; aL < chi_L; ++aL)
                        acc += C(static_cast<Eigen::Index>(i), aL) *
                               to_std(T.at(sigma, aL, aR));
                    next(static_cast<Eigen::Index>(new_row), aR) = acc;
                }
            }
        }
        C = std::move(next);
        dim_so_far = new_dim;
    }

    QuditStatevector sv(n_qudits, d);
    // After the loop chi should be 1.
    for (size_t i = 0; i < sv.dim; ++i)
        sv.amplitudes[i] = from_std(C(static_cast<Eigen::Index>(i), 0));
    return sv;
}

// =============================================================================
// norm_sq / normalize
// =============================================================================

double QuditMPS::norm_sq() const {
    // Left-to-right transfer matrix contraction:
    //   E_{q+1}[aR', aR] = sum_{sigma, aL', aL} conj(A_q[sigma,aL',aR']) *
    //                                              E_q[aL', aL] * A_q[sigma,aL,aR]
    // Boundary: E_0 = [[1]] (1x1).
    Eigen::MatrixXcd E(1, 1);
    E(0, 0) = std::complex<double>(1.0, 0.0);

    for (int q = 0; q < n_qudits; ++q) {
        const auto& T = tensors[static_cast<size_t>(q)];
        const int chi_L = T.chi_L;
        const int chi_R = T.chi_R;

        // First contract: tmp[aL', sigma, aR] = sum_{aL} E[aL', aL] * A[sigma,aL,aR]
        // Then: E_new[aR', aR] = sum_{sigma, aL'} conj(A[sigma,aL',aR']) * tmp[aL',sigma,aR]
        Eigen::MatrixXcd E_new(chi_R, chi_R);
        E_new.setZero();

        // Build A as a (d*chi_L) x chi_R matrix for one multiplication.
        Eigen::MatrixXcd A_left = T.as_left_matrix();  // (d*chi_L, chi_R)

        // tmp[(sigma, aL'), aR] = sum_{aL} E[aL', aL] * A[(sigma,aL), aR]
        // Build as block: tmp(sigma*chi_L + aL', aR)
        //   = sum_aL E(aL', aL) * A_left(sigma*chi_L + aL, aR)
        Eigen::MatrixXcd tmp(static_cast<Eigen::Index>(d) * chi_L, chi_R);
        for (int sigma = 0; sigma < d; ++sigma) {
            // Block product: E (chi_L x chi_L) * A_block (chi_L x chi_R)
            Eigen::MatrixXcd A_block = A_left.block(
                sigma * chi_L, 0, chi_L, chi_R);
            tmp.block(sigma * chi_L, 0, chi_L, chi_R) = E * A_block;
        }

        // E_new(aR', aR) = sum_{sigma, aL'} conj(A_left(sigma*chi_L + aL', aR'))
        //                                    * tmp(sigma*chi_L + aL', aR)
        // = (A_left.adjoint() * tmp) sized (chi_R x chi_R)
        E_new = A_left.adjoint() * tmp;
        E = std::move(E_new);
    }

    // Final E is 1x1 (since chi_R[n-1] = 1).
    return E(0, 0).real();
}

void QuditMPS::normalize() {
    const double n2 = norm_sq();
    if (n2 < 1e-30) return;
    const double inv = 1.0 / std::sqrt(n2);
    auto& T0 = tensors[0];
    for (auto& v : T0.data) { v.real *= inv; v.imag *= inv; }
}

// =============================================================================
// apply_1qudit — O(d^2 * chi_L * chi_R)
// =============================================================================

void QuditMPS::apply_1qudit(int q, const std::vector<Complex128>& U) {
    detail::check_qudit(q, n_qudits, "QuditMPS::apply_1qudit");
    detail::check_size(U.size(), static_cast<size_t>(d) * static_cast<size_t>(d),
                       "QuditMPS::apply_1qudit", "matrix");

    auto& T = tensors[static_cast<size_t>(q)];
    const int chi_L = T.chi_L;
    const int chi_R = T.chi_R;

    std::vector<Complex128> old_v(static_cast<size_t>(d));
    std::vector<Complex128> new_v(static_cast<size_t>(d));

    for (int aL = 0; aL < chi_L; ++aL) {
        for (int aR = 0; aR < chi_R; ++aR) {
            for (int s = 0; s < d; ++s)
                old_v[static_cast<size_t>(s)] = T.at(s, aL, aR);
            for (int so = 0; so < d; ++so) {
                Complex128 acc(0.0, 0.0);
                for (int si = 0; si < d; ++si)
                    acc += U[static_cast<size_t>(so * d + si)] *
                           old_v[static_cast<size_t>(si)];
                new_v[static_cast<size_t>(so)] = acc;
            }
            for (int s = 0; s < d; ++s)
                T.at(s, aL, aR) = new_v[static_cast<size_t>(s)];
        }
    }
}

// =============================================================================
// contract_two_sites — Theta[sigma_q*chi_L + aL, sigma_{q+1}*chi_R + aR]
//   = sum_{am} A_q[sigma_q, aL, am] * A_{q+1}[sigma_{q+1}, am, aR]
// =============================================================================

Eigen::MatrixXcd QuditMPS::contract_two_sites(int q) const {
    const auto& T0 = tensors[static_cast<size_t>(q)];
    const auto& T1 = tensors[static_cast<size_t>(q + 1)];
    const int chi_L = T0.chi_L;
    const int chi_M = T0.chi_R;  // = T1.chi_L
    const int chi_R = T1.chi_R;

    if (T1.chi_L != chi_M)
        throw std::runtime_error("contract_two_sites: bond dimension mismatch");

    Eigen::MatrixXcd Theta(static_cast<Eigen::Index>(d) * chi_L,
                           static_cast<Eigen::Index>(d) * chi_R);
    Theta.setZero();
    for (int s0 = 0; s0 < d; ++s0) {
        for (int aL = 0; aL < chi_L; ++aL) {
            for (int s1 = 0; s1 < d; ++s1) {
                for (int aR = 0; aR < chi_R; ++aR) {
                    std::complex<double> acc(0.0, 0.0);
                    for (int am = 0; am < chi_M; ++am)
                        acc += to_std(T0.at(s0, aL, am)) *
                               to_std(T1.at(s1, am, aR));
                    Theta(s0 * chi_L + aL, s1 * chi_R + aR) = acc;
                }
            }
        }
    }
    return Theta;
}

// =============================================================================
// split_two_sites — SVD-truncate Theta into tensors[q] and tensors[q+1]
// Theta shape: (d*chi_L) x (d*chi_R).
// =============================================================================

void QuditMPS::split_two_sites(int q, const Eigen::MatrixXcd& Theta) {
    const int chi_L = tensors[static_cast<size_t>(q)].chi_L;
    const int chi_R = tensors[static_cast<size_t>(q + 1)].chi_R;

    if (Theta.rows() != static_cast<Eigen::Index>(d) * chi_L ||
        Theta.cols() != static_cast<Eigen::Index>(d) * chi_R)
        throw std::runtime_error("split_two_sites: shape mismatch");

    Eigen::MatrixXcd U, V;
    Eigen::VectorXd S;
    qmps_svd(Theta, svd_method, U, S, V);

    // Discarded-weight truncation: see the statevector constructor above.
    const int full_rank = static_cast<int>(S.size());
    double total = 0.0;
    for (int i = 0; i < full_rank; ++i) total += S(i) * S(i);
    const double budget = svd_cutoff * total;

    int chi = full_rank;
    double discarded = 0.0;
    while (chi > 1) {
        const double w = S(chi - 1) * S(chi - 1);
        if (discarded + w > budget) break;
        discarded += w;
        --chi;
    }
    if (chi > max_bond_dim) chi = max_bond_dim;
    if (chi == 0) chi = 1;

    // Left tensor: shape (d, chi_L, chi) from leftmost chi columns of U.
    MPSSiteTensor Tq(d, chi_L, chi);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int alpha = 0; alpha < chi; ++alpha)
                Tq.at(sigma, aL, alpha) =
                    from_std(U(sigma * chi_L + aL, alpha));

    // Right tensor: shape (d, chi, chi_R) absorbing S into V^dagger.
    //   right[sigma_{q+1}, alpha, aR] = S(alpha) * conj(V(sigma_{q+1}*chi_R + aR, alpha))
    MPSSiteTensor Tq1(d, chi, chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int alpha = 0; alpha < chi; ++alpha)
            for (int aR = 0; aR < chi_R; ++aR) {
                const std::complex<double> vt =
                    std::conj(V(sigma * chi_R + aR, alpha));
                Tq1.at(sigma, alpha, aR) =
                    from_std(std::complex<double>(S(alpha), 0.0) * vt);
            }

    tensors[static_cast<size_t>(q)]     = std::move(Tq);
    tensors[static_cast<size_t>(q + 1)] = std::move(Tq1);
}

// =============================================================================
// apply_2qudit_adjacent — d^2 x d^2 gate on sites (q, q+1)
// =============================================================================

void QuditMPS::apply_2qudit_adjacent(int q, const std::vector<Complex128>& U) {
    detail::check_qudit(q, n_qudits, "QuditMPS::apply_2qudit_adjacent");
    detail::check_qudit(q + 1, n_qudits, "QuditMPS::apply_2qudit_adjacent");
    const size_t d2 = static_cast<size_t>(d) * static_cast<size_t>(d);
    detail::check_size(U.size(), d2 * d2,
                       "QuditMPS::apply_2qudit_adjacent", "matrix");

    Eigen::MatrixXcd Theta = contract_two_sites(q);
    const int chi_L = tensors[static_cast<size_t>(q)].chi_L;
    const int chi_R = tensors[static_cast<size_t>(q + 1)].chi_R;

    // Matrix index convention (project LSB-first, docs/Architecture.md
    // "Conventions"): the FIRST site of the pair is the LEAST significant
    // digit of the U index:
    //   Theta_new[out_q*chi_L + aL, out_{q+1}*chi_R + aR]
    //     = sum_{in_q, in_{q+1}}
    //           U[(out_{q+1}*d + out_q), (in_{q+1}*d + in_q)] *
    //           Theta[in_q*chi_L + aL, in_{q+1}*chi_R + aR]
    Eigen::MatrixXcd Theta_new(static_cast<Eigen::Index>(d) * chi_L,
                               static_cast<Eigen::Index>(d) * chi_R);
    Theta_new.setZero();

    for (int aL = 0; aL < chi_L; ++aL) {
        for (int aR = 0; aR < chi_R; ++aR) {
            for (int so0 = 0; so0 < d; ++so0) {
                for (int so1 = 0; so1 < d; ++so1) {
                    std::complex<double> acc(0.0, 0.0);
                    const size_t u_row = static_cast<size_t>(so1) *
                                             static_cast<size_t>(d) +
                                         static_cast<size_t>(so0);
                    for (int si0 = 0; si0 < d; ++si0) {
                        for (int si1 = 0; si1 < d; ++si1) {
                            const size_t u_col = static_cast<size_t>(si1) *
                                                     static_cast<size_t>(d) +
                                                 static_cast<size_t>(si0);
                            const Complex128& u_el =
                                U[u_row * d2 + u_col];
                            acc += to_std(u_el) *
                                   Theta(si0 * chi_L + aL, si1 * chi_R + aR);
                        }
                    }
                    Theta_new(so0 * chi_L + aL, so1 * chi_R + aR) = acc;
                }
            }
        }
    }

    split_two_sites(q, Theta_new);
}

// =============================================================================
// apply_swap — SWAP gate between adjacent qudits (q, q+1)
// =============================================================================

void QuditMPS::apply_swap(int q) {
    const size_t d2 = static_cast<size_t>(d) * static_cast<size_t>(d);
    std::vector<Complex128> swap_mat(d2 * d2, Complex128(0.0, 0.0));
    // swap[(out_{q+1}*d + out_q)*d^2 + (in_{q+1}*d + in_q)]
    //   = delta(out_q, in_{q+1}) * delta(out_{q+1}, in_q)
    // (the SWAP matrix is invariant under exchanging the digit roles, so the
    // construction below is valid in the LSB-first convention as well)
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const size_t row = static_cast<size_t>(j) * static_cast<size_t>(d) +
                               static_cast<size_t>(i);
            const size_t col = static_cast<size_t>(i) * static_cast<size_t>(d) +
                               static_cast<size_t>(j);
            swap_mat[row * d2 + col] = Complex128(1.0, 0.0);
        }
    }
    apply_2qudit_adjacent(q, swap_mat);
}

// =============================================================================
// apply_2qudit — arbitrary pair (q0, q1); SWAP chain for non-adjacent pairs
// =============================================================================

void QuditMPS::apply_2qudit(int q0, int q1, const std::vector<Complex128>& U) {
    detail::check_qudit(q0, n_qudits, "QuditMPS::apply_2qudit");
    detail::check_qudit(q1, n_qudits, "QuditMPS::apply_2qudit");
    detail::check_distinct2(q0, q1, "QuditMPS::apply_2qudit", "qudits");
    const size_t d2 = static_cast<size_t>(d) * static_cast<size_t>(d);
    detail::check_size(U.size(), d2 * d2, "QuditMPS::apply_2qudit", "matrix");

    // Normalise so q0 < q1, exchanging the two digit roles of U if necessary
    // (valid in any fixed digit convention: it relabels which operand owns
    // which digit, here the LSB-first encoding of docs/Architecture.md).
    if (q0 > q1) {
        std::swap(q0, q1);
        // U'[(out_b*d + out_a), (in_b*d + in_a)] = U[(out_a*d + out_b), (in_a*d + in_b)]
        std::vector<Complex128> U_swapped(d2 * d2, Complex128(0.0, 0.0));
        for (int o0 = 0; o0 < d; ++o0)
            for (int o1 = 0; o1 < d; ++o1)
                for (int i0 = 0; i0 < d; ++i0)
                    for (int i1 = 0; i1 < d; ++i1) {
                        const size_t r_new =
                            static_cast<size_t>(o0) * static_cast<size_t>(d) +
                            static_cast<size_t>(o1);
                        const size_t c_new =
                            static_cast<size_t>(i0) * static_cast<size_t>(d) +
                            static_cast<size_t>(i1);
                        const size_t r_old =
                            static_cast<size_t>(o1) * static_cast<size_t>(d) +
                            static_cast<size_t>(o0);
                        const size_t c_old =
                            static_cast<size_t>(i1) * static_cast<size_t>(d) +
                            static_cast<size_t>(i0);
                        U_swapped[r_new * d2 + c_new] = U[r_old * d2 + c_old];
                    }
        apply_2qudit(q0, q1, U_swapped);
        return;
    }

    if (q1 == q0 + 1) {
        apply_2qudit_adjacent(q0, U);
        return;
    }

    // Bring qudit q0 up next to q1 by swapping it rightward.
    // After this sequence the physical leg originally at q0 sits at position q1 - 1
    // and the leg originally at q1 stays at position q1.  The gate then acts on
    // adjacent sites (q1 - 1, q1) and we undo the swaps to restore the order.
    for (int i = q0; i < q1 - 1; ++i)
        apply_swap(i);

    apply_2qudit_adjacent(q1 - 1, U);

    for (int i = q1 - 2; i >= q0; --i)
        apply_swap(i);
}

// =============================================================================
// apply_phase_oracle / apply_function_oracle — fallback via statevector
// =============================================================================

void QuditMPS::apply_phase_oracle(
    const std::function<Complex128(const std::vector<int>&)>& phase_fn)
{
    auto sv = to_statevector();
    sv.apply_phase_oracle(phase_fn);
    *this = QuditMPS(sv, max_bond_dim, svd_cutoff);
}

void QuditMPS::apply_function_oracle(int n_query, int n_output,
                                     const std::function<int(int)>& f)
{
    auto sv = to_statevector();
    const int d_local = d;
    // Wrap the int->int oracle into the vector<int>->vector<int> signature
    // that QuditStatevector::apply_function_oracle expects.
    auto f_digits = [&f, d_local, n_query, n_output]
        (const std::vector<int>& x) -> std::vector<int>
    {
        // Decode query digits into a flat integer (little-endian: x[0] is LSB).
        int x_flat = 0;
        for (int i = n_query - 1; i >= 0; --i)
            x_flat = x_flat * d_local + x[static_cast<size_t>(i)];
        int y = f(x_flat);
        std::vector<int> result(static_cast<size_t>(n_output));
        for (int i = 0; i < n_output; ++i) {
            result[static_cast<size_t>(i)] = y % d_local;
            y /= d_local;
        }
        return result;
    };
    sv.apply_function_oracle(n_query, n_output, f_digits);
    *this = QuditMPS(sv, max_bond_dim, svd_cutoff);
}

// Sequential environment sampling (audit F-5): previously this contracted the
// whole MPS to a dense d^n statevector and sampled that, defeating MPS
// compactness for wide registers. Now it precomputes the right environments
// once (build_right_envs, O(n·chi^3)) and samples left-to-right read-only,
// carrying the left environment incrementally — O(n·chi^3) total, memory
// bounded by the bond dimension. Mirrors the qubit-layer mps_sample.
std::vector<int> QuditMPS::measure(uint64_t seed) {
    std::mt19937_64 rng(seed == 0
        ? static_cast<uint64_t>(std::random_device{}())
        : seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // right_envs[q] sized chi_L(q) x chi_L(q); right_envs[n] = [[1]].
    const auto right_envs = build_right_envs();

    std::vector<int> digits(static_cast<size_t>(n_qudits), 0);
    Eigen::MatrixXcd left(1, 1);
    left(0, 0) = std::complex<double>(1.0, 0.0);

    for (int q = 0; q < n_qudits; ++q) {
        const auto& T = tensors[static_cast<size_t>(q)];
        const int cl = T.chi_L, cr = T.chi_R;
        const Eigen::MatrixXcd& R = right_envs[static_cast<size_t>(q + 1)];  // cr x cr

        // prob[sigma] = Re Σ_{l1,l2} left[l1,l2] · B_sigma[l1,l2],
        //   B_sigma[l1,l2] = Σ_{r1,r2} T[sigma,l1,r1] R[r1,r2] conj(T[sigma,l2,r2]).
        // Two-stage O(chi^3): A = T_sigma·R, then contract with conj(T_sigma).
        std::vector<double> probs(static_cast<size_t>(d), 0.0);
        for (int s = 0; s < d; ++s) {
            Eigen::MatrixXcd A(cl, cr);
            for (int l1 = 0; l1 < cl; ++l1)
                for (int r2 = 0; r2 < cr; ++r2) {
                    std::complex<double> a(0.0, 0.0);
                    for (int r1 = 0; r1 < cr; ++r1)
                        a += to_std(T.at(s, l1, r1)) * R(r1, r2);
                    A(l1, r2) = a;
                }
            std::complex<double> acc(0.0, 0.0);
            for (int l1 = 0; l1 < cl; ++l1)
                for (int l2 = 0; l2 < cl; ++l2) {
                    std::complex<double> b(0.0, 0.0);
                    for (int r2 = 0; r2 < cr; ++r2)
                        b += A(l1, r2) * std::conj(to_std(T.at(s, l2, r2)));
                    acc += left(l1, l2) * b;
                }
            probs[static_cast<size_t>(s)] = acc.real();
        }

        double total = 0.0;
        for (double& x : probs) { if (x < 0.0) x = 0.0; total += x; }

        int sel;
        if (total < 1e-30) {
            sel = 0;  // degenerate marginal: pick digit 0
        } else {
            const double roll = dist(rng) * total;
            double c = 0.0;
            sel = d - 1;
            for (int s = 0; s < d; ++s) {
                c += probs[static_cast<size_t>(s)];
                if (roll <= c) { sel = s; break; }
            }
        }
        digits[static_cast<size_t>(q)] = sel;

        const double p_out = probs[static_cast<size_t>(sel)];
        const double inv_p = (p_out > 1e-30) ? 1.0 / p_out : 1.0;

        // new_left[m1,m2] = inv_p · Σ_{l1,l2} left[l1,l2] T[sel,l1,m1] conj(T[sel,l2,m2]).
        //   C[l2,m1] = Σ_l1 left[l1,l2] T[sel,l1,m1]
        //   new_left[m1,m2] = Σ_l2 C[l2,m1] conj(T[sel,l2,m2]) · inv_p
        Eigen::MatrixXcd C(cl, cr);
        for (int l2 = 0; l2 < cl; ++l2)
            for (int m1 = 0; m1 < cr; ++m1) {
                std::complex<double> a(0.0, 0.0);
                for (int l1 = 0; l1 < cl; ++l1)
                    a += left(l1, l2) * to_std(T.at(sel, l1, m1));
                C(l2, m1) = a;
            }
        Eigen::MatrixXcd new_left(cr, cr);
        for (int m1 = 0; m1 < cr; ++m1)
            for (int m2 = 0; m2 < cr; ++m2) {
                std::complex<double> a(0.0, 0.0);
                for (int l2 = 0; l2 < cl; ++l2)
                    a += C(l2, m1) * std::conj(to_std(T.at(sel, l2, m2)));
                new_left(m1, m2) = a * inv_p;
            }
        left = std::move(new_left);
    }

    return digits;
}

// =============================================================================
// left_canonicalize / right_canonicalize via SVD
// =============================================================================

void QuditMPS::left_canonicalize() {
    for (int q = 0; q < n_qudits - 1; ++q) {
        auto& Tq = tensors[static_cast<size_t>(q)];
        // M = as_left_matrix has shape (d * chi_L, chi_R).
        Eigen::MatrixXcd M = Tq.as_left_matrix();

        Eigen::MatrixXcd U, V;
        Eigen::VectorXd S;
        qmps_svd(M, svd_method, U, S, V);

        const int full_rank = static_cast<int>(S.size());
        const double smax = (full_rank > 0) ? S(0) : 0.0;
        const double threshold = smax * svd_cutoff;
        int chi = 0;
        for (int i = 0; i < full_rank; ++i) {
            if (S(i) > threshold && chi < max_bond_dim) ++chi;
            else break;
        }
        if (chi == 0) chi = 1;

        // tensors[q] = U_truncated reshaped from (d*chi_L, chi) back to (d, chi_L, chi).
        const int chi_L = Tq.chi_L;
        MPSSiteTensor Tnew(d, chi_L, chi);
        for (int sigma = 0; sigma < d; ++sigma)
            for (int aL = 0; aL < chi_L; ++aL)
                for (int alpha = 0; alpha < chi; ++alpha)
                    Tnew.at(sigma, aL, alpha) =
                        from_std(U(sigma * chi_L + aL, alpha));
        tensors[static_cast<size_t>(q)] = std::move(Tnew);

        // Absorb S * V^dagger into tensors[q+1].
        //   tensors[q+1] new chi_L = chi (replacing the old chi_R of tensors[q]).
        //   absorb: A_{q+1}_new[sigma, alpha, aR] = sum_{aL_old} (S*Vt)(alpha, aL_old) *
        //                                          A_{q+1}_old[sigma, aL_old, aR]
        auto& Tq1 = tensors[static_cast<size_t>(q + 1)];
        const int aL_old_dim = Tq1.chi_L;
        const int aR_dim     = Tq1.chi_R;
        if (V.rows() != static_cast<Eigen::Index>(aL_old_dim))
            throw std::runtime_error("left_canonicalize: V row count mismatch");

        // SV[alpha, aL_old] = S(alpha) * conj(V(aL_old, alpha))
        Eigen::MatrixXcd SVt(chi, aL_old_dim);
        for (int alpha = 0; alpha < chi; ++alpha)
            for (int aL_old = 0; aL_old < aL_old_dim; ++aL_old)
                SVt(alpha, aL_old) = std::complex<double>(S(alpha), 0.0) *
                                     std::conj(V(aL_old, alpha));

        MPSSiteTensor Tq1_new(d, chi, aR_dim);
        for (int sigma = 0; sigma < d; ++sigma) {
            for (int alpha = 0; alpha < chi; ++alpha) {
                for (int aR = 0; aR < aR_dim; ++aR) {
                    std::complex<double> acc(0.0, 0.0);
                    for (int aL_old = 0; aL_old < aL_old_dim; ++aL_old)
                        acc += SVt(alpha, aL_old) *
                               to_std(Tq1.at(sigma, aL_old, aR));
                    Tq1_new.at(sigma, alpha, aR) = from_std(acc);
                }
            }
        }
        tensors[static_cast<size_t>(q + 1)] = std::move(Tq1_new);
    }
}

void QuditMPS::right_canonicalize() {
    for (int q = n_qudits - 1; q > 0; --q) {
        auto& Tq = tensors[static_cast<size_t>(q)];
        // M = as_right_matrix has shape (chi_L, d * chi_R).
        Eigen::MatrixXcd M = Tq.as_right_matrix();

        Eigen::MatrixXcd U, V;
        Eigen::VectorXd S;
        qmps_svd(M, svd_method, U, S, V);

        const int full_rank = static_cast<int>(S.size());
        const double smax = (full_rank > 0) ? S(0) : 0.0;
        const double threshold = smax * svd_cutoff;
        int chi = 0;
        for (int i = 0; i < full_rank; ++i) {
            if (S(i) > threshold && chi < max_bond_dim) ++chi;
            else break;
        }
        if (chi == 0) chi = 1;

        // tensors[q] = V^dagger_truncated reshaped from (chi, d*chi_R) -> (d, chi, chi_R).
        const int chi_R = Tq.chi_R;
        MPSSiteTensor Tnew(d, chi, chi_R);
        // Vt[alpha, sigma*chi_R + aR] = conj(V(sigma*chi_R + aR, alpha))
        for (int sigma = 0; sigma < d; ++sigma)
            for (int alpha = 0; alpha < chi; ++alpha)
                for (int aR = 0; aR < chi_R; ++aR)
                    Tnew.at(sigma, alpha, aR) =
                        from_std(std::conj(V(sigma * chi_R + aR, alpha)));
        tensors[static_cast<size_t>(q)] = std::move(Tnew);

        // Absorb U * S into tensors[q-1].
        //   tensors[q-1] new chi_R = chi.
        //   A_{q-1}_new[sigma, aL, alpha] = sum_{aR_old} A_{q-1}_old[sigma, aL, aR_old] *
        //                                                (U*S)(aR_old, alpha)
        auto& Tqm1 = tensors[static_cast<size_t>(q - 1)];
        const int aL_dim     = Tqm1.chi_L;
        const int aR_old_dim = Tqm1.chi_R;
        if (U.rows() != static_cast<Eigen::Index>(aR_old_dim))
            throw std::runtime_error("right_canonicalize: U row count mismatch");

        Eigen::MatrixXcd US(aR_old_dim, chi);
        for (int aR_old = 0; aR_old < aR_old_dim; ++aR_old)
            for (int alpha = 0; alpha < chi; ++alpha)
                US(aR_old, alpha) = U(aR_old, alpha) *
                                    std::complex<double>(S(alpha), 0.0);

        MPSSiteTensor Tqm1_new(d, aL_dim, chi);
        for (int sigma = 0; sigma < d; ++sigma) {
            for (int aL = 0; aL < aL_dim; ++aL) {
                for (int alpha = 0; alpha < chi; ++alpha) {
                    std::complex<double> acc(0.0, 0.0);
                    for (int aR_old = 0; aR_old < aR_old_dim; ++aR_old)
                        acc += to_std(Tqm1.at(sigma, aL, aR_old)) *
                               US(aR_old, alpha);
                    Tqm1_new.at(sigma, aL, alpha) = from_std(acc);
                }
            }
        }
        tensors[static_cast<size_t>(q - 1)] = std::move(Tqm1_new);
    }
}

// =============================================================================
// build_right_envs — diagnostic helper (unused by the public API)
// Returns the right environments E_q s.t. E_n = [[1]] and
//   E_q[aL', aL] = sum_{sigma, aR', aR} A_q[sigma, aL', aR']
//                                       * E_{q+1}[aR', aR]
//                                       * conj(A_q[sigma, aL, aR])
// is the network of all sites from q..n-1 traced over their physical legs.
// =============================================================================

std::vector<Eigen::MatrixXcd> QuditMPS::build_right_envs() const {
    std::vector<Eigen::MatrixXcd> envs(static_cast<size_t>(n_qudits + 1));
    envs[static_cast<size_t>(n_qudits)] = Eigen::MatrixXcd(1, 1);
    envs[static_cast<size_t>(n_qudits)](0, 0) = std::complex<double>(1.0, 0.0);

    for (int q = n_qudits - 1; q >= 0; --q) {
        const auto& T = tensors[static_cast<size_t>(q)];
        const int chi_L = T.chi_L;
        const int chi_R = T.chi_R;
        const auto& E_next = envs[static_cast<size_t>(q + 1)];

        Eigen::MatrixXcd E_new(chi_L, chi_L);
        E_new.setZero();
        for (int aLp = 0; aLp < chi_L; ++aLp) {
            for (int aL = 0; aL < chi_L; ++aL) {
                std::complex<double> acc(0.0, 0.0);
                for (int sigma = 0; sigma < d; ++sigma) {
                    for (int aRp = 0; aRp < chi_R; ++aRp) {
                        for (int aR = 0; aR < chi_R; ++aR) {
                            acc += to_std(T.at(sigma, aLp, aRp)) *
                                   E_next(aRp, aR) *
                                   std::conj(to_std(T.at(sigma, aL, aR)));
                        }
                    }
                }
                E_new(aLp, aL) = acc;
            }
        }
        envs[static_cast<size_t>(q)] = std::move(E_new);
    }

    return envs;
}

} // namespace lindblad
