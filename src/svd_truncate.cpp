// =============================================================================
// detail::svd_truncate_verified - SELECT -> VERIFY -> FALLBACK -> THROW
// =============================================================================
// Contract and the reason this file is quarantined: include/lindblad/detail/
// svd_truncate.hpp. Everything here is the implementation of that ladder.

#include "lindblad/detail/svd_truncate.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lindblad {
namespace detail {

namespace {

using RowMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::RowMajor>;
using ColMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::ColMajor>;

// The Gram route squares the condition number, so sigmas below sqrt(eps) times
// sigma_max carry no information there however much weight they hold. Relative
// to sigma_max, so it means the same thing on every target. This is a validity
// floor, NOT a truncation knob: weight below it is still honestly counted as
// discarded.
constexpr double kGramValidityFloorRel = 1.5e-8;  // ~sqrt(DBL_EPSILON)

// Slack on the backward error a stable SVD is entitled to, in units of
// N*eps*‖M‖_F. Generous because the two errors are not symmetric: a false
// REJECT costs one Gram recomputation, while a false ACCEPT is a wrong state
// that nothing downstream will catch. Even so the gate lands far below the
// defects this ladder exists for. On the degenerate rank-deficient thetas of a
// 13-qubit period-finding circuit, Eigen 3.4.0 has been measured returning a
// factorisation whose excess over the identity is 7.4e-16 of ‖M‖_F² (a 2.7e-8
// relative reconstruction error, which is not backward-stable at any dimension)
// while a healthy factorisation of the same matrices measures 2.8e-30. Ten
// orders of clearance above, four below.
constexpr double kBackwardErrorSlack = 64.0;

// One rung of the ladder: build the kept slice from a candidate factorisation
// and decide whether it is trustworthy. Returns false to reject it, leaving
// `out` untouched, so the caller can try the next route.
//
// sigma_floor rejects values that are not trustworthy DATA at all. The direct
// SVD passes 0: a backend SVD resolves sigmas down to ~eps*sigma_max, so every
// finite value it reports is real, and how much of it to keep belongs to the
// weight budget alone. The Gram route passes its own floor.
template <typename MatT>
bool attempt(const MatT& mat, int rows, int cols,
             const Eigen::VectorXd& S_try, const Eigen::MatrixXcd& U_try,
             const Eigen::MatrixXcd& V_try, double sigma_floor,
             int max_bond_dim, double cutoff, double m_fro_sq,
             SvdTruncation& out) {
    const int md = static_cast<int>(S_try.size());

    // SELECT. Every bit-level-finite sigma is a candidate WHEREVER IT SITS in
    // S, which is what makes this immune to ordering corruption: interleaved
    // garbage displacing real sigmas is one of the observed failure modes, and
    // a scan that stopped at the first small value would silently drop real
    // directions behind it. Each candidate carries its source index so the
    // matching U and V columns are gathered individually.
    std::vector<std::pair<double, int>> fs;  // (sigma, source index)
    fs.reserve(static_cast<std::size_t>(md));
    double below_floor = 0.0;  // weight rejected as untrustworthy
    for (int i = 0; i < md; ++i) {
        const double s = S_try(i);
        // A NaN sigma satisfies 's > sigma_floor' and would be kept, growing
        // the rank with garbage.
        if (!is_finite_strict(s)) continue;  // artifact, not weight
        if (s > sigma_floor) fs.push_back({s, i});
        else                 below_floor += s * s;
    }
    std::stable_sort(fs.begin(), fs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    // Walk up from the smallest survivor, dropping while the running discarded
    // weight stays inside the budget. Weight already rejected by sigma_floor
    // counts against it, so a Gram route that lost a lot to the validity floor
    // does not then also truncate aggressively.
    double total = below_floor;
    for (const auto& p : fs) total += p.first * p.first;
    const double budget = cutoff * total;

    int k = static_cast<int>(fs.size());
    {
        double d = below_floor;
        while (k > 1) {
            const double w = fs[static_cast<std::size_t>(k - 1)].first *
                             fs[static_cast<std::size_t>(k - 1)].first;
            if (d + w > budget) break;
            d += w;
            --k;
        }
    }
    k = std::min<int>(k, max_bond_dim);
    if (k == 0) {
        // Numerically-zero matrix or fully corrupt spectrum: keep the single
        // largest finite sigma if one exists at all.
        int best = -1;
        double best_val = -1.0;
        for (int i = 0; i < md; ++i) {
            const double s = S_try(i);
            if (is_finite_strict(s) && s > best_val) {
                best_val = s;
                best = i;
            }
        }
        if (best < 0) return false;
        // The rescued sigma may have sat below sigma_floor, in which case its
        // weight is already in below_floor. It is now KEPT, so take it back out
        // or it would be counted as kept and discarded at once.
        if (!(best_val > sigma_floor)) below_floor -= best_val * best_val;
        fs.assign(1, {best_val, best});
        k = 1;
    }

    // Discarded weight = every finite sigma NOT kept (beyond-rank AND
    // below-floor; non-finite entries are artifacts, not weight).
    //
    // Sum the discarded buckets DIRECTLY. Do not compute this as
    // `total - kept`: for a normalised state both are ~1.0 while the real
    // difference is ~1e-30, so the subtraction cannot resolve it and returns
    // multiples of eps instead, and the truncation error then reports ~1e-15 of
    // phantom loss for a bond that discarded nothing. Adding up the small
    // values keeps every term at its own scale.
    double discarded = below_floor;
    for (std::size_t i = static_cast<std::size_t>(k); i < fs.size(); ++i)
        discarded += fs[i].first * fs[i].first;

    // Gather, in descending sigma order.
    Eigen::MatrixXcd U_k(rows, k);
    Eigen::MatrixXcd V_k(cols, k);
    Eigen::VectorXd S_k(k);
    for (int r = 0; r < k; ++r) {
        const int src = fs[static_cast<std::size_t>(r)].second;
        S_k(r) = fs[static_cast<std::size_t>(r)].first;
        U_k.col(r) = U_try.col(src);
        V_k.col(r) = V_try.col(src);
    }

    // VERIFY 1: kept slice bit-finite.
    for (int i = 0; i < k; ++i)
        if (!is_finite_strict(S_k(i))) return false;
    for (int c = 0; c < k; ++c) {
        for (int r = 0; r < rows; ++r) {
            const auto z = U_k(r, c);
            if (!is_finite_strict(z.real()) || !is_finite_strict(z.imag()))
                return false;
        }
        for (int r = 0; r < cols; ++r) {
            const auto z = V_k(r, c);
            if (!is_finite_strict(z.real()) || !is_finite_strict(z.imag()))
                return false;
        }
    }

    // VERIFY 2: Frobenius identity of the truncated factorisation.
    //
    // ‖M - U_k S_k V_k†‖_F² == discarded holds with EQUALITY for a truncated
    // SVD in exact arithmetic, so the slack added to `discarded` is the entire
    // allowance a COMPUTED factorisation gets. The standard backward-error
    // bound is ‖M - U S V†‖_F <= c*N*eps*‖M‖_F with N the larger dimension.
    //
    // The reconstruction is materialised into the INPUT's storage order before
    // subtracting. Eigen's coefficient-wise binary operations require both
    // operands to agree on row- versus column-major, and the kept slice is
    // built column-major while a caller's block may be either.
    typename MatT::PlainObject recon = U_k * S_k.asDiagonal() * V_k.adjoint();
    const double resid = (mat - recon).squaredNorm();
    if (!is_finite_strict(resid)) return false;
    const double bwd = kBackwardErrorSlack *
                       static_cast<double>(std::max(rows, cols)) *
                       std::numeric_limits<double>::epsilon();
    // The comparison is stated and made in the AMPLITUDE domain:
    //   ‖M - U_k S_k V_k†‖_F <= sqrt(discarded) + bwd * ‖M‖_F
    // Squaring it here would drop the cross term
    // 2*bwd*sqrt(discarded*‖M‖_F²), which is the dominant allowance whenever
    // truncation is heavy. That term is not optional slack: with `discarded` at
    // the scale of ‖M‖_F², resid and discarded are two large nearly-equal
    // quantities computed by different routes, so their difference carries
    // FIRST-order rounding (~eps*discarded) while a squared-domain bound offers
    // only second-order room. It vanishes as discarded -> 0, so a bond that
    // truncated nothing is still held to the strict backward-error bound alone.
    const double allowed = std::sqrt(discarded) + bwd * std::sqrt(m_fro_sq);
    if (resid > allowed * allowed + 1e-18) return false;

    out.U = std::move(U_k);
    out.S = std::move(S_k);
    out.V = std::move(V_k);
    out.rank = k;
    out.discarded_weight = discarded;
    // Subtracting `discarded` leaves only the factorisation's own error.
    // Clamped because the two sides are computed differently and can cross by
    // an ulp when both are dust.
    out.residual_excess =
        (m_fro_sq > 0.0) ? std::max(0.0, resid - discarded) / m_fro_sq : 0.0;
    return true;
}

template <typename MapT>
SvdTruncation run_ladder(const MapT& mat, int rows, int cols, int max_bond_dim,
                         double cutoff, SVDMethod method, const char* ctx) {
    using PlainT = typename MapT::PlainObject;

    Eigen::MatrixXcd U_eigen, V_eigen;
    Eigen::VectorXd S_eigen;
    if (method == SVDMethod::BDC) {
        Eigen::BDCSVD<PlainT> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        S_eigen = svd.singularValues();
        U_eigen = svd.matrixU();
        V_eigen = svd.matrixV();
    } else {
        Eigen::JacobiSVD<PlainT> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        S_eigen = svd.singularValues();
        U_eigen = svd.matrixU();
        V_eigen = svd.matrixV();
    }

    // Accumulated one entry at a time in memory order rather than through a
    // vectorised reduction, so the value does not depend on how the target
    // chose to partition the sum. It feeds the acceptance bound below, and a
    // bound that moves with the hardware is not a bound.
    double m_fro_sq = 0.0;
    {
        const std::complex<double>* q = mat.data();
        const std::size_t n = static_cast<std::size_t>(rows) *
                              static_cast<std::size_t>(cols);
        for (std::size_t t = 0; t < n; ++t)
            m_fro_sq += q[t].real() * q[t].real() + q[t].imag() * q[t].imag();
    }

    SvdTruncation out;
    if (attempt(mat, rows, cols, S_eigen, U_eigen, V_eigen, /*sigma_floor=*/0.0,
                max_bond_dim, cutoff, m_fro_sq, out)) {
        return out;
    }

    // FALLBACK. Recompute through a route that shares no code with the one that
    // just failed: the Gram matrix of whichever side is smaller, through a
    // self-adjoint eigendecomposition, which is robust on exactly-degenerate
    // Hermitian input. sigma = sqrt(max(lambda, 0)); the partner factor is
    // built only for sigmas above the validity floor, so no tiny divisions.
    const bool tall = rows >= cols;
    const Eigen::MatrixXcd G = tall ? Eigen::MatrixXcd(mat.adjoint() * mat)
                                    : Eigen::MatrixXcd(mat * mat.adjoint());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(G);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            std::string(ctx) +
            ": Gram-route eigendecomposition failed on a " +
            std::to_string(rows) + "x" + std::to_string(cols) + " matrix");
    }
    const int gd = static_cast<int>(es.eigenvalues().size());
    Eigen::VectorXd Sg(gd);
    for (int i = 0; i < gd; ++i) {
        // Eigenvalues ascend; emit sigmas descending.
        Sg(i) = std::sqrt(std::max(0.0, es.eigenvalues()(gd - 1 - i)));
    }
    const double smax = (gd > 0) ? Sg(0) : 0.0;
    const double floor_g = kGramValidityFloorRel * smax;

    Eigen::MatrixXcd Ug(rows, gd), Vg(cols, gd);
    if (tall) {
        for (int i = 0; i < gd; ++i) Vg.col(i) = es.eigenvectors().col(gd - 1 - i);
        for (int i = 0; i < gd; ++i) {
            if (Sg(i) > floor_g) Ug.col(i) = (mat * Vg.col(i)) / Sg(i);
            else Ug.col(i).setZero();  // never selected: below the floor
        }
    } else {
        for (int i = 0; i < gd; ++i) Ug.col(i) = es.eigenvectors().col(gd - 1 - i);
        for (int i = 0; i < gd; ++i) {
            if (Sg(i) > floor_g) Vg.col(i) = (mat.adjoint() * Ug.col(i)) / Sg(i);
            else Vg.col(i).setZero();
        }
    }

    if (!attempt(mat, rows, cols, Sg, Ug, Vg, floor_g, max_bond_dim, cutoff,
                 m_fro_sq, out)) {
        // THROW. Never continue with a corrupt tensor: silent propagation is
        // exactly how this class of defect manifests.
        throw std::runtime_error(
            std::string(ctx) +
            ": both the SVD backend and the Gram-eigendecomposition fallback "
            "failed verification on a " +
            std::to_string(rows) + "x" + std::to_string(cols) +
            " matrix (non-finite or non-reconstructing factorisation); "
            "refusing to continue with a corrupt tensor");
    }
    out.used_gram_fallback = true;
    return out;
}

} // namespace

SvdTruncation svd_truncate_verified(const Complex128* data, int rows, int cols,
                                    MatrixOrder order, int max_bond_dim,
                                    double cutoff, SVDMethod method,
                                    const char* ctx) {
    // Complex128 {double real, double imag} is layout-identical to
    // std::complex<double>, so both orders map in place and neither caller pays
    // an O(rows*cols) copy to hand a block over.
    const auto* p = reinterpret_cast<const std::complex<double>*>(data);
    if (order == MatrixOrder::RowMajor) {
        Eigen::Map<const RowMajorC> mat(p, rows, cols);
        return run_ladder(mat, rows, cols, max_bond_dim, cutoff, method, ctx);
    }
    Eigen::Map<const ColMajorC> mat(p, rows, cols);
    return run_ladder(mat, rows, cols, max_bond_dim, cutoff, method, ctx);
}

} // namespace detail
} // namespace lindblad
