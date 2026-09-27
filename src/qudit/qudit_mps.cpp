// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/qudit/qudit_mps.hpp"

#include "lindblad/detail/validate.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/detail/eigen_backend.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <string>
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

// Complex128 is layout-identical to std::complex<double>, so site data is
// handed to the QR and mapped for products in place.
static inline std::complex<double>* as_std(Complex128* p) noexcept {
    return reinterpret_cast<std::complex<double>*>(p);
}
static inline const std::complex<double>* as_std(const Complex128* p) noexcept {
    return reinterpret_cast<const std::complex<double>*>(p);
}

using RowMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::RowMajor>;
using ColMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::ColMajor>;

// Below this a marginal is treated as zero: sampling falls back to its
// degenerate rule and renormalisation is skipped rather than dividing by noise.
static constexpr double MARGINAL_FLOOR = 1e-30;

// One-time note when either Jacobi kernel is selected on the qudit MPS. The
// latch is per layer, so this fires even when a qubit MPS in the same process
// has already noted its own selection. The cost figures are the qubit layer's.
static void warn_jacobi_slower_once_qudit(SVDMethod m) {
    static bool warned = false;
    if (warned) return;
    warned = true;
    emit_warning(
        std::string("note: SVDMethod::") + to_string(m) +
        " selected for the qudit MPS. BDC is the default (autonne divide and conquer) "
        "and is faster as the block grows and the spectrum "
        "decays: measured on a 128x128 decaying spectrum, 2.7x over Jacobi "
        "and 19x over EigenJacobi. Jacobi resolves the tail of a "
        "graded spectrum with relative accuracy, which is why it remains "
        "selectable; select it for that, not for speed.");
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
detail::DenseMatrix MPSSiteTensor::as_left_matrix() const {
    detail::DenseMatrix M(d * chi_L, chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int aR = 0; aR < chi_R; ++aR)
                M(sigma * chi_L + aL, aR) = to_std(at(sigma, aL, aR));
    return M;
}

// Row index = aL; col index = sigma * chi_R + aR.
detail::DenseMatrix MPSSiteTensor::as_right_matrix() const {
    detail::DenseMatrix M(chi_L, d * chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int aR = 0; aR < chi_R; ++aR)
                M(aL, sigma * chi_R + aR) = to_std(at(sigma, aL, aR));
    return M;
}

MPSSiteTensor MPSSiteTensor::from_left_matrix(const detail::DenseMatrix& M,
                                              int d, int chi_L) {
    const int chi_R = M.cols();
    if (M.rows() != d * chi_L)
        throw std::invalid_argument(
            "MPSSiteTensor::from_left_matrix: row count must be d * chi_L");
    MPSSiteTensor T(d, chi_L, chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int aR = 0; aR < chi_R; ++aR)
                T.at(sigma, aL, aR) = from_std(M(sigma * chi_L + aL, aR));
    return T;
}

MPSSiteTensor MPSSiteTensor::from_right_matrix(const detail::DenseMatrix& M,
                                               int d, int chi_R) {
    const int chi_L = M.rows();
    if (M.cols() != d * chi_R)
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

// Every bond split in this class goes through here, which is what keeps the
// four call sites from selecting a rank four different ways. The ladder itself
// (kernel, Jacobi rescue, Gram rescue, throw, and why each rung exists) is
// shared with the qubit layer; see include/lindblad/detail/svd_truncate.hpp.
//
// Eigen matrices are column-major and Complex128 is layout-identical to
// std::complex<double>, so the block is mapped in place rather than copied.
detail::SvdTruncation QuditMPS::truncate_block(const detail::DenseMatrix& M,
                                               const char* ctx,
                                               detail::FidelityLedger& ledger) {
    if (svd_method == SVDMethod::Jacobi || svd_method == SVDMethod::EigenJacobi)
        warn_jacobi_slower_once_qudit(svd_method);
    // Bracketing the whole ladder, rescue included, as the qubit layer does.
    const auto svd_t0 = std::chrono::steady_clock::now();
    detail::SvdTruncation r = detail::svd_truncate_verified(
        reinterpret_cast<const Complex128*>(M.data()),
        M.rows(), M.cols(),
        detail::MatrixOrder::ColMajor, max_bond_dim, svd_cutoff, svd_method,
        svd_rescue, ctx);
    svd_nanos += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - svd_t0).count());

    // Counted once the ladder has returned, so every figure covers splits that
    // completed and the rescue counts report rescues that SUCCEEDED: a failed
    // rescue does not return. The Gram route's floor-rejected weight is booked
    // beside the truncation total, never inside it.
    ++svd_calls;
    if (r.used_jacobi_rescue) ++jacobi_rescues;
    if (r.used_gram_fallback) ++gram_fallbacks;
    floor_rejected += r.floor_rejected_weight;
    total_truncation_error += r.discarded_weight;
    max_verify_resid_excess =
        std::max(max_verify_resid_excess, r.residual_excess);

    double kept = 0.0;
    for (int i = 0; i < r.rank; ++i) kept += r.S(i) * r.S(i);
    ledger.record(kept, r.discarded_weight);
    return r;
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

    tensors_.reserve(static_cast<size_t>(n_qudits_));
    for (int q = 0; q < n_qudits_; ++q) {
        MPSSiteTensor T(d_, 1, 1);
        T.at(0, 0, 0) = Complex128(1.0, 0.0);  // |0> on every site
        tensors_.push_back(std::move(T));
    }
    // A bond-1 unit vector is orthonormal both ways, so any single site could
    // be named the centre of |0...0>; site 0 is where sampling starts.
    span_lo = 0;
    span_hi = 0;
}

QuditMPS::QuditMPS(const QuditStatevector& sv,
                   int max_bond_dim_, double svd_cutoff_)
    : n_qudits(sv.n_qudits), d(sv.d),
      max_bond_dim(max_bond_dim_), svd_cutoff(svd_cutoff_)
{
    if (max_bond_dim_ < 1)
        throw std::invalid_argument("QuditMPS: max_bond_dim must be >= 1");
    rebuild_from(sv);
}

void QuditMPS::rebuild_from(const QuditStatevector& sv) {
    if (sv.n_qudits != n_qudits || sv.d != d) {
        throw std::invalid_argument(
            "QuditMPS: the amplitudes cover " + std::to_string(sv.n_qudits) +
            " qudits of dimension " + std::to_string(sv.d) + ", this chain " +
            std::to_string(n_qudits) + " of dimension " + std::to_string(d));
    }

    // Built beside the chain rather than into it, so a throw part way through
    // the sweep leaves the chain and its fidelity figures as they were.
    std::vector<MPSSiteTensor> sites;
    sites.reserve(static_cast<size_t>(n_qudits));
    detail::FidelityLedger ledger = fidelity;

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
    detail::DenseMatrix M(d * chi_L, static_cast<int>(cols));
    for (int sigma = 0; sigma < d; ++sigma)
        for (size_t col = 0; col < cols; ++col)
            M(sigma, static_cast<int>(col)) =
                to_std(sv.amplitudes[static_cast<size_t>(sigma) + col *
                                     static_cast<size_t>(d)]);

    for (int q = 0; q < n_qudits - 1; ++q) {
        // Canonical by construction: every site before this one is an
        // isometry and the residual is the whole remainder of the state, so
        // the discarded fraction is a fraction of the state.
        const detail::SvdTruncation split =
            truncate_block(M, "QuditMPS statevector decomposition", ledger);
        const detail::DenseMatrix& U = split.U;
        const detail::DenseMatrix& V = split.V;
        const detail::RealVector& S = split.S;
        const int chi = split.rank;

        // Left tensor: shape (d, chi_L, chi) from leftmost chi columns of U.
        // The residual matrix M is built with rows ordered (left-bond major):
        //   row = aL * d + sigma   (see the M_next reshape below and the initial
        //   reshape where chi_L = 1). U inherits that row order, so the physical
        //   index sigma is the LOW digit and the left bond aL is the HIGH digit.
        // Decode with the SAME ordering (this also matches the final-site read
        // `M(aL * d + sigma, 0)`). Using `sigma * chi_L + aL` here transposed the
        // physical/bond indices on intermediate sites (chi_L > 1), corrupting any
        // state needing >= 3 sites with a nontrivial interior bond.
        MPSSiteTensor Tq(d, chi_L, chi);
        for (int sigma = 0; sigma < d; ++sigma)
            for (int aL = 0; aL < chi_L; ++aL)
                for (int alpha = 0; alpha < chi; ++alpha)
                    Tq.at(sigma, aL, alpha) =
                        from_std(U(aL * d + sigma, alpha));
        sites.push_back(std::move(Tq));

        // Build residual M' = diag(S_truncated) * V^dagger_truncated
        // Vt rows are indexed by the singular values; we keep top `chi`.
        detail::DenseMatrix Vt_trunc(chi, static_cast<int>(cols));
        for (int alpha = 0; alpha < chi; ++alpha)
            for (int c = 0; c < static_cast<int>(cols); ++c)
                Vt_trunc(alpha, c) = std::conj(V(c, alpha));

        detail::DenseMatrix residual(chi, static_cast<int>(cols));
        for (int alpha = 0; alpha < chi; ++alpha)
            for (int c = 0; c < static_cast<int>(cols); ++c)
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
        detail::DenseMatrix M_next(chi * d, static_cast<int>(new_cols));
        for (int aL_prev = 0; aL_prev < chi; ++aL_prev)
            for (int sigma_next = 0; sigma_next < d; ++sigma_next)
                for (size_t next_col = 0; next_col < new_cols; ++next_col)
                    M_next(aL_prev * d + sigma_next,
                           static_cast<int>(next_col)) =
                        residual(aL_prev,
                                 static_cast<int>(
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
        sites.push_back(std::move(Tlast));
    }

    // Sites 0..n-2 are the sweep's isometries, so the centre is the last site.
    tensors_ = std::move(sites);
    span_lo = n_qudits - 1;
    span_hi = n_qudits - 1;
    fidelity = ledger;
}

// =============================================================================
// set_tensors / canonicalize - replacing the chain and moving its centre
// =============================================================================

void QuditMPS::set_tensors(std::vector<MPSSiteTensor> sites) {
    const char* ctx = "QuditMPS::set_tensors";
    const auto fail = [ctx](const std::string& what) {
        throw std::invalid_argument(std::string(ctx) + ": " + what);
    };
    if (static_cast<int>(sites.size()) != n_qudits) {
        fail("expected " + std::to_string(n_qudits) + " site tensors, got " +
             std::to_string(sites.size()));
    }
    for (int q = 0; q < n_qudits; ++q) {
        const MPSSiteTensor& t = sites[static_cast<size_t>(q)];
        const std::string site = "site " + std::to_string(q);
        if (t.d != d) {
            fail(site + " has physical dimension " + std::to_string(t.d) +
                 " where the chain's is " + std::to_string(d));
        }
        if (t.chi_L < 1 || t.chi_R < 1) {
            fail(site + " has a bond below 1 (" + std::to_string(t.chi_L) +
                 ", " + std::to_string(t.chi_R) + ")");
        }
        if (q == 0 && t.chi_L != 1) fail("the left end bond must be 1");
        if (q == n_qudits - 1 && t.chi_R != 1) {
            fail("the right end bond must be 1");
        }
        if (q > 0 && sites[static_cast<size_t>(q - 1)].chi_R != t.chi_L) {
            fail(site + " has left bond " + std::to_string(t.chi_L) +
                 " where its neighbour's right bond is " +
                 std::to_string(sites[static_cast<size_t>(q - 1)].chi_R));
        }
        const size_t expected = static_cast<size_t>(t.d) *
                                static_cast<size_t>(t.chi_L) *
                                static_cast<size_t>(t.chi_R);
        if (t.data.size() != expected) {
            fail(site + " holds " + std::to_string(t.data.size()) +
                 " entries where its shape needs " + std::to_string(expected));
        }
        for (const Complex128& c : t.data) {
            if (!is_finite_strict(c.real) || !is_finite_strict(c.imag)) {
                fail(site + " holds a non-finite entry");
            }
        }
    }

    tensors_ = std::move(sites);
    span_lo = 0;
    span_hi = n_qudits - 1;
    fidelity.reset();
}

void QuditMPS::canonicalize(int site) {
    detail::check_qudit(site, n_qudits, "QuditMPS::canonicalize");
    focus(site, site);
}

// =============================================================================
// Moving the orthogonality centre
// =============================================================================
//
// A site's data, [sigma][aL][aR], is row-major as its (d chi_L) x chi_R left
// matrix, so the rightward step hands it to the QR in place. As a
// chi_L x (d chi_R) right matrix it is d separate chi_L x chi_R blocks, so the
// leftward step assembles that matrix first and the rightward step multiplies
// R into the neighbour block by block.

void QuditMPS::shift_right(int q) {
    MPSSiteTensor& A = tensors_[static_cast<size_t>(q)];
    MPSSiteTensor& B = tensors_[static_cast<size_t>(q + 1)];
    const int rows = d * A.chi_L;
    const int chi = A.chi_R;
    const int k = std::min(rows, chi);

    std::vector<std::complex<double>> Q(static_cast<size_t>(rows) * k);
    std::vector<std::complex<double>> R(static_cast<size_t>(k) * chi);
    detail::qr_thin(as_std(A.data.data()), rows, chi,
                    detail::MatrixOrder::RowMajor, Q.data(), R.data());

    MPSSiteTensor A_new(d, A.chi_L, k);
    Eigen::Map<RowMajorC>(as_std(A_new.data.data()), rows, k) =
        Eigen::Map<const ColMajorC>(Q.data(), rows, k);

    const int chi_R = B.chi_R;
    const size_t in_block = static_cast<size_t>(chi) * chi_R;
    const size_t out_block = static_cast<size_t>(k) * chi_R;
    const Eigen::Map<const ColMajorC> Rm(R.data(), k, chi);
    MPSSiteTensor B_new(d, k, chi_R);
    for (int sigma = 0; sigma < d; ++sigma) {
        Eigen::Map<RowMajorC>(as_std(B_new.data.data()) + sigma * out_block, k,
                              chi_R) =
            Rm * Eigen::Map<const RowMajorC>(
                     as_std(B.data.data()) + sigma * in_block, chi, chi_R);
    }

    A = std::move(A_new);
    B = std::move(B_new);
}

void QuditMPS::shift_left(int q) {
    MPSSiteTensor& A = tensors_[static_cast<size_t>(q - 1)];
    MPSSiteTensor& B = tensors_[static_cast<size_t>(q)];
    const int chi = B.chi_L;
    const int chi_R = B.chi_R;
    const int cols = d * chi_R;
    const int k = std::min(chi, cols);

    // B as its chi x (d chi_R) right matrix, row-major, column sigma*chi_R + aR.
    std::vector<std::complex<double>> Mb(static_cast<size_t>(chi) * cols);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi; ++aL)
            for (int aR = 0; aR < chi_R; ++aR)
                Mb[static_cast<size_t>(aL) * cols + sigma * chi_R + aR] =
                    to_std(B.at(sigma, aL, aR));

    // Read column-major that buffer is the transpose, so this factorises
    // B^T = Q R, hence B = R^T Q^T (see detail::qr_thin).
    std::vector<std::complex<double>> Q(static_cast<size_t>(cols) * k);
    std::vector<std::complex<double>> R(static_cast<size_t>(k) * chi);
    detail::qr_thin(Mb.data(), cols, chi, detail::MatrixOrder::ColMajor,
                    Q.data(), R.data());

    // Q^T is the column-major Q buffer read row-major: row alpha, column
    // sigma*chi_R + aR.
    MPSSiteTensor B_new(d, k, chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int alpha = 0; alpha < k; ++alpha)
            for (int aR = 0; aR < chi_R; ++aR)
                B_new.at(sigma, alpha, aR) = from_std(
                    Q[static_cast<size_t>(alpha) * cols + sigma * chi_R + aR]);

    // L = R^T, chi x k, is the column-major R buffer read row-major, and A's
    // left matrix multiplies it in place.
    const int rows = d * A.chi_L;
    MPSSiteTensor A_new(d, A.chi_L, k);
    Eigen::Map<RowMajorC>(as_std(A_new.data.data()), rows, k) =
        Eigen::Map<const RowMajorC>(as_std(A.data.data()), rows, chi) *
        Eigen::Map<const RowMajorC>(R.data(), chi, k);

    A = std::move(A_new);
    B = std::move(B_new);
}

void QuditMPS::focus(int a, int b) {
    while (span_lo < a) {
        shift_right(span_lo);
        ++span_lo;
        span_hi = std::max(span_hi, span_lo);
    }
    while (span_hi > b) {
        shift_left(span_hi);
        --span_hi;
        span_lo = std::min(span_lo, span_hi);
    }
}

bool QuditMPS::split_needs_centre(int q) const {
    if (canonical_form == CanonicalForm::Always) return true;
    if (svd_cutoff > MPS_DEFAULT_CUTOFF) return true;
    const MPSSiteTensor& A = tensors_[static_cast<size_t>(q)];
    const MPSSiteTensor& B = tensors_[static_cast<size_t>(q + 1)];
    return std::min(d * A.chi_L, d * B.chi_R) > max_bond_dim;
}

std::vector<double> QuditMPS::centre_marginals(int site) const {
    const MPSSiteTensor& T = tensors_[static_cast<size_t>(site)];
    const size_t block = static_cast<size_t>(T.chi_L) * T.chi_R;
    std::vector<double> p(static_cast<size_t>(d), 0.0);
    for (int sigma = 0; sigma < d; ++sigma)
        for (size_t i = 0; i < block; ++i) {
            const Complex128& c = T.data[sigma * block + i];
            p[static_cast<size_t>(sigma)] += c.real * c.real + c.imag * c.imag;
        }
    return p;
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
        const auto& T = tensors_[static_cast<size_t>(q)];
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
    // Left-to-right transfer matrix contraction over the open span:
    //   E_{q+1}[aR', aR] = sum_{sigma, aL', aL} conj(A_q[sigma,aL',aR']) *
    //                                              E_q[aL', aL] * A_q[sigma,aL,aR]
    // The sites left of the span are left-orthonormal, so E arrives at its
    // first site as the identity; the sites right of it are right-orthonormal,
    // so what waits past its last site is the identity too, and closing on it
    // is a trace. A single-site span is that site's squared Frobenius norm.
    if (span_lo == span_hi) {
        double sum = 0.0;
        for (const Complex128& c : tensors_[static_cast<size_t>(span_lo)].data)
            sum += c.real * c.real + c.imag * c.imag;
        return sum;
    }

    const int chi_first = tensors_[static_cast<size_t>(span_lo)].chi_L;
    Eigen::MatrixXcd E = Eigen::MatrixXcd::Identity(chi_first, chi_first);

    for (int q = span_lo; q <= span_hi; ++q) {
        const auto& T = tensors_[static_cast<size_t>(q)];
        const int chi_L = T.chi_L;
        const int chi_R = T.chi_R;

        // First contract: tmp[aL', sigma, aR] = sum_{aL} E[aL', aL] * A[sigma,aL,aR]
        // Then: E_new[aR', aR] = sum_{sigma, aL'} conj(A[sigma,aL',aR']) * tmp[aL',sigma,aR]
        Eigen::MatrixXcd E_new(chi_R, chi_R);
        E_new.setZero();

        // Build A as a (d*chi_L) x chi_R matrix for one multiplication. The
        // accessor returns the library's own storage; it is mapped here because
        // the block arithmetic below is what needs a backend, not the tensor.
        const detail::DenseMatrix A_left_store = T.as_left_matrix();
        Eigen::Map<const Eigen::MatrixXcd> A_left(
            A_left_store.data(), A_left_store.rows(), A_left_store.cols());

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

    return E.trace().real();
}

void QuditMPS::normalize() {
    const double n2 = norm_sq();
    const double n = std::sqrt(n2);
    // Refuses rather than returning quietly, matching the other state classes.
    // The guard is on the norm rather than its square so that every layer
    // rejects the same set of states: squaring first would put the effective
    // floor at the square root of this one.
    if (!is_normalizable(n)) {
        throw std::runtime_error(
            "QuditMPS::normalize: no norm to divide out; the state is zero or "
            "non-finite");
    }
    const double inv = 1.0 / n;
    auto& T = tensors_[static_cast<size_t>(span_lo)];
    for (auto& v : T.data) { v.real *= inv; v.imag *= inv; }
}

// norm_sq() is the measurement, as in the qubit MPS, and for the same reason:
// an MPS holds no flat amplitude array, so there is no sum to quarantine and the
// chain contraction runs under the project-wide flags.
bool QuditMPS::is_normalized(double atol) const {
    return std::abs(norm_sq() - 1.0) <= atol;
}

void QuditMPS::check_normalized(ValidationOptions validation) {
    // Returns before measuring when nothing would consume the residual: the
    // measurement is a transfer matrix contraction down the whole chain.
    if (detail::measurement_unused(validation)) return;
    const char* ctx = "QuditMPS::check_normalized";
    const double ns = norm_sq();
    if (detail::check_normalized(ns, validation, ctx) &&
        detail::normalization_repairable(ns, validation, ctx,
                                         detail::STATE_NORMALIZATION)) {
        normalize();
    }
}

// =============================================================================
// apply_1qudit — O(d^2 * chi_L * chi_R)
// =============================================================================

void QuditMPS::apply_1qudit(int q, const std::vector<Complex128>& U_in,
                            ValidationOptions validation) {
    detail::check_qudit(q, n_qudits, "QuditMPS::apply_1qudit");
    detail::check_size(U_in.size(), static_cast<size_t>(d) * static_cast<size_t>(d),
                       "QuditMPS::apply_1qudit", "matrix");
    std::vector<Complex128> U_fixed;
    const std::vector<Complex128>& U = detail::check_unitary_fixing(
        U_in, static_cast<size_t>(d), validation, "QuditMPS::apply_1qudit",
        U_fixed);

    auto& T = tensors_[static_cast<size_t>(q)];
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

detail::DenseMatrix QuditMPS::contract_two_sites(int q) const {
    const auto& T0 = tensors_[static_cast<size_t>(q)];
    const auto& T1 = tensors_[static_cast<size_t>(q + 1)];
    const int chi_L = T0.chi_L;
    const int chi_M = T0.chi_R;  // = T1.chi_L
    const int chi_R = T1.chi_R;

    if (T1.chi_L != chi_M)
        throw std::runtime_error("contract_two_sites: bond dimension mismatch");

    detail::DenseMatrix Theta(d * chi_L, d * chi_R);
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
// split_two_sites - SVD-truncate Theta into tensors[q] and tensors[q+1]
// Theta shape: (d*chi_L) x (d*chi_R).
// =============================================================================

void QuditMPS::split_two_sites(int q, const detail::DenseMatrix& Theta,
                               Absorb absorb) {
    const int chi_L = tensors_[static_cast<size_t>(q)].chi_L;
    const int chi_R = tensors_[static_cast<size_t>(q + 1)].chi_R;

    if (Theta.rows() != d * chi_L || Theta.cols() != d * chi_R)
        throw std::runtime_error("split_two_sites: shape mismatch");

    const detail::SvdTruncation split =
        truncate_block(Theta, "QuditMPS two-site split", fidelity);
    const detail::DenseMatrix& U = split.U;
    const detail::DenseMatrix& V = split.V;
    const detail::RealVector& S = split.S;
    const int chi = split.rank;
    const bool right = (absorb == Absorb::Right);

    // Left tensor: shape (d, chi_L, chi) from leftmost chi columns of U, times
    // S when the singular values go left.
    MPSSiteTensor Tq(d, chi_L, chi);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int aL = 0; aL < chi_L; ++aL)
            for (int alpha = 0; alpha < chi; ++alpha) {
                const std::complex<double> u = U(sigma * chi_L + aL, alpha);
                Tq.at(sigma, aL, alpha) = from_std(right ? u : u * S(alpha));
            }

    // Right tensor: shape (d, chi, chi_R) from V^dagger, times S when the
    // singular values go right.
    //   right[sigma_{q+1}, alpha, aR] = S(alpha) * conj(V(sigma_{q+1}*chi_R + aR, alpha))
    MPSSiteTensor Tq1(d, chi, chi_R);
    for (int sigma = 0; sigma < d; ++sigma)
        for (int alpha = 0; alpha < chi; ++alpha)
            for (int aR = 0; aR < chi_R; ++aR) {
                const std::complex<double> vt =
                    std::conj(V(sigma * chi_R + aR, alpha));
                Tq1.at(sigma, alpha, aR) = from_std(right ? S(alpha) * vt : vt);
            }

    tensors_[static_cast<size_t>(q)]     = std::move(Tq);
    tensors_[static_cast<size_t>(q + 1)] = std::move(Tq1);

    // As MPSState: the site holding S is always in the span afterwards, and
    // the site holding an isometry drops out of it when no site beyond it, on
    // its own side, was in the span.
    if (right) {
        if (span_lo >= q) span_lo = q + 1;
        span_hi = std::max(span_hi, q + 1);
    } else {
        if (span_hi <= q + 1) span_hi = q;
        span_lo = std::min(span_lo, q);
    }
}

// =============================================================================
// apply_2qudit_adjacent - d^2 x d^2 gate on sites (q, q+1)
// =============================================================================

void QuditMPS::apply_2qudit_adjacent(int q, const std::vector<Complex128>& U_in,
                                     ValidationOptions validation) {
    detail::check_qudit(q, n_qudits, "QuditMPS::apply_2qudit_adjacent");
    detail::check_qudit(q + 1, n_qudits, "QuditMPS::apply_2qudit_adjacent");
    const size_t d2 = static_cast<size_t>(d) * static_cast<size_t>(d);
    detail::check_size(U_in.size(), d2 * d2,
                       "QuditMPS::apply_2qudit_adjacent", "matrix");
    std::vector<Complex128> U_fixed;
    const std::vector<Complex128>& U = detail::check_unitary_fixing(
        U_in, d2, validation, "QuditMPS::apply_2qudit_adjacent", U_fixed);

    gate_adjacent(q, U, Absorb::Right);
}

void QuditMPS::gate_adjacent(int q, const std::vector<Complex128>& U,
                             Absorb absorb) {
    const size_t d2 = static_cast<size_t>(d) * static_cast<size_t>(d);

    // Before the contraction, because moving the centre rewrites the two
    // sites being contracted.
    if (split_needs_centre(q)) focus(q, q + 1);

    detail::DenseMatrix Theta = contract_two_sites(q);
    const int chi_L = tensors_[static_cast<size_t>(q)].chi_L;
    const int chi_R = tensors_[static_cast<size_t>(q + 1)].chi_R;

    // Matrix index convention (project LSB-first, docs/Architecture.md
    // "Conventions"): the FIRST site of the pair is the LEAST significant
    // digit of the U index:
    //   Theta_new[out_q*chi_L + aL, out_{q+1}*chi_R + aR]
    //     = sum_{in_q, in_{q+1}}
    //           U[(out_{q+1}*d + out_q), (in_{q+1}*d + in_q)] *
    //           Theta[in_q*chi_L + aL, in_{q+1}*chi_R + aR]
    detail::DenseMatrix Theta_new(d * chi_L, d * chi_R);

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

    split_two_sites(q, Theta_new, absorb);
}

// =============================================================================
// apply_swap - SWAP gate between adjacent qudits (q, q+1)
// =============================================================================

void QuditMPS::apply_swap(int q, Absorb absorb) {
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
    gate_adjacent(q, swap_mat, absorb);
}

// =============================================================================
// apply_2qudit - arbitrary pair (q0, q1); SWAP chain for non-adjacent pairs
// =============================================================================

void QuditMPS::apply_2qudit(int q0, int q1, const std::vector<Complex128>& U_in,
                            ValidationOptions validation) {
    detail::check_qudit(q0, n_qudits, "QuditMPS::apply_2qudit");
    detail::check_qudit(q1, n_qudits, "QuditMPS::apply_2qudit");
    detail::check_distinct2(q0, q1, "QuditMPS::apply_2qudit", "qudits");
    const size_t d2 = static_cast<size_t>(d) * static_cast<size_t>(d);
    detail::check_size(U_in.size(), d2 * d2, "QuditMPS::apply_2qudit", "matrix");
    std::vector<Complex128> U_fixed;
    const std::vector<Complex128>& U = detail::check_unitary_fixing(
        U_in, d2, validation, "QuditMPS::apply_2qudit", U_fixed);

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
        gate_adjacent(q0, U, Absorb::Right);
        return;
    }

    // Bring qudit q0 up next to q1 by swapping it rightward.
    // After this sequence the physical leg originally at q0 sits at position q1 - 1
    // and the leg originally at q1 stays at position q1.  The gate then acts on
    // adjacent sites (q1 - 1, q1) and we undo the swaps to restore the order.
    //
    // Each split sends its singular values toward the block the chain touches
    // next, so when splits move the centre the next one finds it already in
    // place: rightward on the way in, leftward from the gate on the way back.
    for (int i = q0; i < q1 - 1; ++i)
        apply_swap(i, Absorb::Right);

    gate_adjacent(q1 - 1, U, Absorb::Left);

    for (int i = q1 - 2; i >= q0; --i)
        apply_swap(i, Absorb::Left);
}

// =============================================================================
// apply_phase_oracle / apply_function_oracle - fallback via statevector
// =============================================================================

void QuditMPS::apply_phase_oracle(
    const std::function<Complex128(const std::vector<int>&)>& phase_fn)
{
    auto sv = to_statevector();
    sv.apply_phase_oracle(phase_fn);
    rebuild_from(sv);
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
    rebuild_from(sv);
}

// =============================================================================
// measure - sample from the chain centred on qudit 0
// =============================================================================
// With the centre on qudit 0 every other site is right-orthonormal, so the
// environment right of any qudit is the identity, and the sample needs only a
// row vector v on the bond left of the current qudit, carrying the digits
// drawn so far:
//
//   w_s = v · A_q[s],   P(s | digits so far) = |w_s|² / Σ_s' |w_s'|²,
//   then v <- w_sel / |w_sel|.
//
// O(n·d·chi²), memory bounded by the bond dimension, and the chain is not
// collapsed. Mirrors the qubit layer's terminal sampler.

std::vector<int> QuditMPS::measure(uint64_t seed) {
    std::mt19937_64 rng(seed == 0
        ? static_cast<uint64_t>(std::random_device{}())
        : seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    focus(0, 0);

    std::vector<int> digits(static_cast<size_t>(n_qudits), 0);
    std::vector<std::complex<double>> v{std::complex<double>(1.0, 0.0)};
    std::vector<std::vector<std::complex<double>>> w(static_cast<size_t>(d));

    for (int q = 0; q < n_qudits; ++q) {
        const auto& T = tensors_[static_cast<size_t>(q)];
        const int cl = T.chi_L, cr = T.chi_R;

        std::vector<double> probs(static_cast<size_t>(d), 0.0);
        for (int s = 0; s < d; ++s) {
            auto& ws = w[static_cast<size_t>(s)];
            ws.assign(static_cast<size_t>(cr), std::complex<double>(0.0, 0.0));
            for (int l = 0; l < cl; ++l) {
                const std::complex<double> vl = v[static_cast<size_t>(l)];
                for (int r = 0; r < cr; ++r)
                    ws[static_cast<size_t>(r)] += vl * to_std(T.at(s, l, r));
            }
            for (const auto& z : ws) probs[static_cast<size_t>(s)] += std::norm(z);
        }

        double total = 0.0;
        for (double x : probs) total += x;

        int sel;
        if (total < MARGINAL_FLOOR) {
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
        const double inv = (p_out > MARGINAL_FLOOR) ? 1.0 / std::sqrt(p_out) : 1.0;
        v = w[static_cast<size_t>(sel)];
        for (auto& z : v) z *= inv;
    }

    return digits;
}

// =============================================================================
// measure_qudit - measure one qudit at the centre and collapse onto it
// =============================================================================

int QuditMPS::measure_qudit(int q, std::mt19937_64& rng) {
    detail::check_qudit(q, n_qudits, "QuditMPS::measure_qudit");
    focus(q, q);

    // At the centre the slice norms ARE the raw marginals, summing to the
    // state's norm². One uniform is drawn whatever the marginals, so the
    // generator advances the same way on every call.
    const std::vector<double> probs = centre_marginals(q);
    double total = 0.0;
    for (double x : probs) total += x;
    const double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

    int sel = 0;
    if (total >= MARGINAL_FLOOR) {
        const double target = roll * total;
        double c = 0.0;
        sel = d - 1;
        for (int s = 0; s < d; ++s) {
            c += probs[static_cast<size_t>(s)];
            if (target <= c) { sel = s; break; }
        }
    }

    // Collapse: every other digit's slice is zeroed and the chosen one divided
    // by the square root of its raw marginal, leaving unit norm.
    MPSSiteTensor& T = tensors_[static_cast<size_t>(q)];
    const size_t block = static_cast<size_t>(T.chi_L) * T.chi_R;
    const double p_sel = probs[static_cast<size_t>(sel)];
    const double inv = (p_sel > MARGINAL_FLOOR) ? 1.0 / std::sqrt(p_sel) : 1.0;
    for (int s = 0; s < d; ++s)
        for (size_t i = 0; i < block; ++i) {
            Complex128& c = T.data[static_cast<size_t>(s) * block + i];
            if (s == sel) { c.real *= inv; c.imag *= inv; }
            else          { c = Complex128(0.0, 0.0); }
        }
    fidelity.invalidate();
    return sel;
}

// =============================================================================
// left_canonicalize / right_canonicalize via SVD
// =============================================================================

void QuditMPS::left_canonicalize() {
    // From site 0, every split below has only isometries to its left and
    // right-orthonormal sites to its right: canonical gauge throughout.
    focus(0, 0);
    for (int q = 0; q < n_qudits - 1; ++q) {
        auto& Tq = tensors_[static_cast<size_t>(q)];
        // M = as_left_matrix has shape (d * chi_L, chi_R).
        const detail::DenseMatrix M = Tq.as_left_matrix();

        const detail::SvdTruncation split =
            truncate_block(M, "QuditMPS left canonicalisation", fidelity);
        const detail::DenseMatrix& U = split.U;
        const detail::DenseMatrix& V = split.V;
        const detail::RealVector& S = split.S;
        const int chi = split.rank;

        // tensors[q] = U_truncated reshaped from (d*chi_L, chi) back to (d, chi_L, chi).
        const int chi_L = Tq.chi_L;
        MPSSiteTensor Tnew(d, chi_L, chi);
        for (int sigma = 0; sigma < d; ++sigma)
            for (int aL = 0; aL < chi_L; ++aL)
                for (int alpha = 0; alpha < chi; ++alpha)
                    Tnew.at(sigma, aL, alpha) =
                        from_std(U(sigma * chi_L + aL, alpha));
        tensors_[static_cast<size_t>(q)] = std::move(Tnew);

        // Absorb S * V^dagger into tensors[q+1].
        //   tensors[q+1] new chi_L = chi (replacing the old chi_R of tensors[q]).
        //   absorb: A_{q+1}_new[sigma, alpha, aR] = sum_{aL_old} (S*Vt)(alpha, aL_old) *
        //                                          A_{q+1}_old[sigma, aL_old, aR]
        auto& Tq1 = tensors_[static_cast<size_t>(q + 1)];
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
        tensors_[static_cast<size_t>(q + 1)] = std::move(Tq1_new);
        span_lo = span_hi = q + 1;
    }
}

void QuditMPS::right_canonicalize() {
    // The mirror image: from the last site, every split has right-orthonormal
    // sites to its right and left-orthonormal ones to its left.
    focus(n_qudits - 1, n_qudits - 1);
    for (int q = n_qudits - 1; q > 0; --q) {
        auto& Tq = tensors_[static_cast<size_t>(q)];
        // M = as_right_matrix has shape (chi_L, d * chi_R).
        const detail::DenseMatrix M = Tq.as_right_matrix();

        const detail::SvdTruncation split =
            truncate_block(M, "QuditMPS right canonicalisation", fidelity);
        const detail::DenseMatrix& U = split.U;
        const detail::DenseMatrix& V = split.V;
        const detail::RealVector& S = split.S;
        const int chi = split.rank;

        // tensors[q] = V^dagger_truncated reshaped from (chi, d*chi_R) -> (d, chi, chi_R).
        const int chi_R = Tq.chi_R;
        MPSSiteTensor Tnew(d, chi, chi_R);
        // Vt[alpha, sigma*chi_R + aR] = conj(V(sigma*chi_R + aR, alpha))
        for (int sigma = 0; sigma < d; ++sigma)
            for (int alpha = 0; alpha < chi; ++alpha)
                for (int aR = 0; aR < chi_R; ++aR)
                    Tnew.at(sigma, alpha, aR) =
                        from_std(std::conj(V(sigma * chi_R + aR, alpha)));
        tensors_[static_cast<size_t>(q)] = std::move(Tnew);

        // Absorb U * S into tensors[q-1].
        //   tensors[q-1] new chi_R = chi.
        //   A_{q-1}_new[sigma, aL, alpha] = sum_{aR_old} A_{q-1}_old[sigma, aL, aR_old] *
        //                                                (U*S)(aR_old, alpha)
        auto& Tqm1 = tensors_[static_cast<size_t>(q - 1)];
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
        tensors_[static_cast<size_t>(q - 1)] = std::move(Tqm1_new);
        span_lo = span_hi = q - 1;
    }
}

} // namespace lindblad
