// =============================================================================
// detail::svd_truncate_verified - SELECT -> VERIFY -> FALLBACK -> THROW
// =============================================================================
// Contract, and which pieces of the ladder are compiled strict and why:
// include/lindblad/detail/svd_truncate.hpp. Everything here is the
// implementation of that ladder, under the project-wide flags.

#include "lindblad/detail/svd_truncate.hpp"

#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/detail/svd_verify.hpp"

#include <Eigen/Dense>

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
             const RealVector& S_try, const DenseMatrix& U_try,
             const DenseMatrix& V_try, double sigma_floor,
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

    // Weight absent from the kept slice, in TWO buckets, because they mean
    // different things and only one of them is truncation.
    //
    // `truncated` is weight this call chose to drop: directions the budget or
    // the bond cap rejected. That is what a caller's truncation error is asking
    // about.
    //
    // `below_floor` is weight the validity floor rejected as untrustworthy, and
    // on the Gram route that is not loss, it is noise the route manufactured.
    // Forming G = M†M squares the condition number, so an eigenvalue that is
    // exactly zero in M comes back at the scale of eps, and its sqrt is ~1e-8.
    // Three such values carry ~1e-16 of weight that was never in the matrix.
    // Reporting it as truncation error describes a bond that discarded nothing
    // as having lost something. The primary route passes sigma_floor = 0, so
    // this bucket is empty there and the distinction costs it nothing.
    //
    // Both buckets are summed DIRECTLY rather than as `total - kept`: for a
    // normalised state both of those are ~1.0 while the real difference is
    // ~1e-30, so the subtraction cannot resolve it and returns multiples of eps
    // instead. Adding up the small values keeps every term at its own scale.
    double truncated = 0.0;
    for (std::size_t i = static_cast<std::size_t>(k); i < fs.size(); ++i)
        truncated += fs[i].first * fs[i].first;

    // Everything missing from the kept slice, which is what the reconstruction
    // has to account for. The acceptance bound below compares against this, not
    // against `truncated`: a direction rejected by the floor is just as absent
    // from U_k S_k V_k† as one rejected by the budget.
    const double discarded = below_floor + truncated;

    // Gather, in descending sigma order. A source column is copied whole: both
    // sides are column-major, so a column is contiguous in each.
    DenseMatrix U_k(rows, k);
    DenseMatrix V_k(cols, k);
    RealVector S_k(k);
    for (int r = 0; r < k; ++r) {
        const int src = fs[static_cast<std::size_t>(r)].second;
        S_k(r) = fs[static_cast<std::size_t>(r)].first;
        std::copy(U_try.col(src), U_try.col(src) + rows, U_k.col(r));
        std::copy(V_try.col(src), V_try.col(src) + cols, V_k.col(r));
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
    // The residual is computed in its own strict-FP translation unit: it
    // subtracts two nearly identical matrices, and a value perturbed too small
    // would admit exactly the factorisations this rung exists to reject.
    const double resid = svd_reconstruction_residual_sq(
        mat.data(), rows, cols,
        static_cast<bool>(MatT::IsRowMajor) ? MatrixOrder::RowMajor
                                            : MatrixOrder::ColMajor,
        U_k.data(), S_k.data(), V_k.data(), k);
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
    out.discarded_weight = truncated;
    out.floor_rejected_weight = below_floor;
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
    // The input's storage order travels with the call rather than being
    // normalised here, so a caller's block is mapped in place on both paths.
    const MatrixOrder order = static_cast<bool>(MapT::IsRowMajor)
                                  ? MatrixOrder::RowMajor
                                  : MatrixOrder::ColMajor;
    const int kdim = std::min(rows, cols);
    DenseMatrix U_primary(rows, kdim), V_primary(cols, kdim);
    RealVector S_primary(kdim);
    const bool primary_ok =
        svd_thin(mat.data(), rows, cols, order, method, U_primary.data(),
                 S_primary.data(), V_primary.data());

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

    // A backend that reported failure has left its outputs unspecified, so the
    // verify rung is not given them to judge: the fallback is taken directly.
    SvdTruncation out;
    if (primary_ok &&
        attempt(mat, rows, cols, S_primary, U_primary, V_primary,
                /*sigma_floor=*/0.0,
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
    const int gd = tall ? cols : rows;
    RealVector g_evals(gd);
    DenseMatrix g_evecs(gd, gd);
    if (!eigh(G.data(), gd, MatrixOrder::ColMajor, g_evals.data(),
              g_evecs.data())) {
        throw std::runtime_error(
            std::string(ctx) +
            ": Gram-route eigendecomposition failed on a " +
            std::to_string(rows) + "x" + std::to_string(cols) + " matrix");
    }
    RealVector Sg(gd);
    for (int i = 0; i < gd; ++i) {
        // Eigenvalues ascend; emit sigmas descending.
        Sg(i) = std::sqrt(std::max(0.0, g_evals(gd - 1 - i)));
    }
    const double smax = (gd > 0) ? Sg(0) : 0.0;
    const double floor_g = kGramValidityFloorRel * smax;

    // Eigenvectors arrive ascending and are reversed into descending sigma
    // order by copying whole columns; both sides are column-major, so a column
    // is contiguous in each.
    //
    // The partner factor is a matrix-vector product. Columns left at zero are
    // the ones below the validity floor, which SELECT never keeps, so they are
    // never read: the zero is what makes that explicit rather than leaving
    // uninitialised storage behind a rank the caller might raise.
    using CVec = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
    DenseMatrix Ug(rows, gd), Vg(cols, gd);
    if (tall) {
        for (int i = 0; i < gd; ++i)
            std::copy(g_evecs.col(gd - 1 - i), g_evecs.col(gd - 1 - i) + gd,
                      Vg.col(i));
        for (int i = 0; i < gd; ++i) {
            if (!(Sg(i) > floor_g)) continue;
            Eigen::Map<const CVec> v(Vg.col(i), cols);
            Eigen::Map<CVec>(Ug.col(i), rows) = (mat * v) / Sg(i);
        }
    } else {
        for (int i = 0; i < gd; ++i)
            std::copy(g_evecs.col(gd - 1 - i), g_evecs.col(gd - 1 - i) + gd,
                      Ug.col(i));
        for (int i = 0; i < gd; ++i) {
            if (!(Sg(i) > floor_g)) continue;
            Eigen::Map<const CVec> u(Ug.col(i), rows);
            Eigen::Map<CVec>(Vg.col(i), cols) = (mat.adjoint() * u) / Sg(i);
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
