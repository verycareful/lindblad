// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// =============================================================================
// detail::svd_truncate_verified - kernel -> Jacobi -> Gram -> THROW
// =============================================================================
// Contract, and which pieces of the ladder are compiled strict and why:
// include/lindblad/detail/svd_truncate.hpp. Everything here is the
// implementation of that ladder, under the project-wide flags.

#include "lindblad/detail/svd_truncate.hpp"

#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/detail/report.hpp"
#include "lindblad/detail/svd_verify.hpp"
#include "lindblad/validation.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
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

// The Gram route's validity floor: below it a sigma is not trustworthy DATA on
// that route, however much weight it holds. Forming G = M†M squares the
// condition number, and a self-adjoint eigensolver's error on an eigenvalue
// is bounded by slack * n * eps * ||G||, with ||G|| = sigma_max². An eigenvalue
// that is exactly zero in M therefore comes back anywhere in [0, that bound],
// and its square root anywhere in [0, sqrt(slack * n * eps) * sigma_max]: the
// floor is the top of that band, so noise on a null direction cannot pass as
// a direction. A floor at bare sqrt(eps) * sigma_max sits INSIDE the band and
// keeps a null direction whenever the solver's error on it exceeds one eps,
// which it routinely does. The same bound is the eigenvector's accuracy: a
// sigma at the floor has its direction resolved to about one part in slack,
// and one below it worse, so a direction the floor rejects is one the route
// could not have built correctly anyway.
//
// Relative to sigma_max, so it means the same thing on every target. A
// validity floor, NOT a truncation knob: weight below it is still counted as
// absent from the factorisation, in its own bucket.
double gram_validity_floor(int gd, double sigma_max) {
    return std::sqrt(kBackwardErrorSlack * static_cast<double>(gd) *
                     std::numeric_limits<double>::epsilon()) *
           sigma_max;
}

// What a rung produced: a verified factorisation, a factorisation that failed
// verification, or none at all (the kernel declined, or no singular value was
// finite).
enum class Candidate { Accepted, Rejected, None };

// One rung of the ladder: build the kept slice from a candidate factorisation
// and decide whether it is trustworthy. A rejected candidate leaves `out`
// untouched unless `keep_unverified` asks for it (SvdRejection::Ignore), in
// which case `out` holds it, flagged used_unverified, and the result is still
// Rejected so the caller knows what it holds.
//
// sigma_floor rejects values that are not trustworthy DATA at all. The direct
// SVD passes 0: a backend SVD resolves sigmas down to ~eps*sigma_max, so every
// finite value it reports is real, and how much of it to keep belongs to the
// weight budget alone. The Gram route passes its own floor.
// A number for a diagnostic, short and unambiguous about its magnitude.
std::string sci(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3e", v);
    return buf;
}

// `why` receives the reason for a rejection, in the terms the rung measured
// it, so the warning that reports a descent can say what was wrong rather than
// only that something was.
template <typename MatT>
Candidate attempt(const MatT& mat, int rows, int cols,
                  const RealVector& S_try, const DenseMatrix& U_try,
                  const DenseMatrix& V_try, double sigma_floor,
                  int max_bond_dim, double cutoff, double m_fro_sq,
                  bool keep_unverified, SvdTruncation& out, std::string& why) {
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
        if (best < 0) {
            why = "no finite singular value in the spectrum";
            return Candidate::None;
        }
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
    // exactly zero in M comes back at the scale of eps, and its sqrt at ~1e-8.
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
    bool verified = true;
    for (int i = 0; i < k && verified; ++i) {
        if (!is_finite_strict(S_k(i))) {
            why = "a kept singular value is non-finite";
            verified = false;
        }
    }
    for (int c = 0; c < k && verified; ++c) {
        for (int r = 0; r < rows && verified; ++r) {
            const auto z = U_k(r, c);
            if (!is_finite_strict(z.real()) || !is_finite_strict(z.imag())) {
                why = "a kept left singular vector holds a non-finite entry";
                verified = false;
            }
        }
        for (int r = 0; r < cols && verified; ++r) {
            const auto z = V_k(r, c);
            if (!is_finite_strict(z.real()) || !is_finite_strict(z.imag())) {
                why = "a kept right singular vector holds a non-finite entry";
                verified = false;
            }
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
    double resid = std::numeric_limits<double>::infinity();
    if (verified) {
        resid = svd_reconstruction_residual_sq(
            mat.data(), rows, cols,
            static_cast<bool>(MatT::IsRowMajor) ? MatrixOrder::RowMajor
                                                : MatrixOrder::ColMajor,
            U_k.data(), S_k.data(), V_k.data(), k);
        if (!is_finite_strict(resid)) {
            why = "the reconstruction residual is non-finite";
            verified = false;
        } else {
            const double bwd = kBackwardErrorSlack *
                               static_cast<double>(std::max(rows, cols)) *
                               std::numeric_limits<double>::epsilon();
            // The comparison is stated and made in the AMPLITUDE domain:
            //   ‖M - U_k S_k V_k†‖_F <= sqrt(discarded) + bwd * ‖M‖_F
            // Squaring it here would drop the cross term
            // 2*bwd*sqrt(discarded*‖M‖_F²), which is the dominant allowance
            // whenever truncation is heavy. That term is not optional slack:
            // with `discarded` at the scale of ‖M‖_F², resid and discarded are
            // two large nearly-equal quantities computed by different routes,
            // so their difference carries FIRST-order rounding (~eps*discarded)
            // while a squared-domain bound offers only second-order room. It
            // vanishes as discarded -> 0, so a bond that truncated nothing is
            // still held to the strict backward-error bound alone.
            const double allowed = std::sqrt(discarded) + bwd * std::sqrt(m_fro_sq);
            if (resid > allowed * allowed + 1e-18) {
                why = "reconstruction residual " + sci(std::sqrt(resid)) +
                      " exceeds the allowance " + sci(allowed) + " (rank " +
                      std::to_string(k) + " of " + std::to_string(md) +
                      ", discarded weight " + sci(discarded) + ", |M|_F " +
                      sci(std::sqrt(m_fro_sq)) + ")";
                verified = false;
            }
        }
    }
    if (!verified && !keep_unverified) return Candidate::Rejected;

    out.U = std::move(U_k);
    out.S = std::move(S_k);
    out.V = std::move(V_k);
    out.rank = k;
    out.discarded_weight = truncated;
    out.floor_rejected_weight = below_floor;
    // Subtracting `discarded` leaves only the factorisation's own error.
    // Clamped because the two sides are computed differently and can cross by
    // an ulp when both are dust. A residual that could not be measured reports
    // as infinite.
    out.residual_excess =
        !is_finite_strict(resid) ? std::numeric_limits<double>::infinity()
        : (m_fro_sq > 0.0)       ? std::max(0.0, resid - discarded) / m_fro_sq
                                 : 0.0;
    out.used_unverified = !verified;
    return verified ? Candidate::Accepted : Candidate::Rejected;
}

// One SVD rung: factorise with `method`, then SELECT and VERIFY the result. A
// kernel that reported failure has left its outputs unspecified, so there is no
// candidate to judge, or to keep: Candidate::None.
template <typename MapT>
Candidate try_kernel(const MapT& mat, int rows, int cols, MatrixOrder order,
                     SVDMethod method, int max_bond_dim, double cutoff,
                     double m_fro_sq, bool keep_unverified, SvdTruncation& out,
                     std::string& why) {
    const int kdim = std::min(rows, cols);
    DenseMatrix U(rows, kdim), V(cols, kdim);
    RealVector S(kdim);
    if (!svd_thin(mat.data(), rows, cols, order, method, U.data(), S.data(),
                  V.data())) {
        why = "the kernel declined the block";
        return Candidate::None;
    }
    return attempt(mat, rows, cols, S, U, V, /*sigma_floor=*/0.0, max_bond_dim,
                   cutoff, m_fro_sq, keep_unverified, out, why);
}

// The Gram rung: recompute through a route that shares no code with either
// SVD kernel. The Gram matrix of whichever side is smaller goes through a
// self-adjoint eigendecomposition, which is robust on exactly-degenerate
// Hermitian input. sigma = sqrt(max(lambda, 0)); the partner factor is built
// only for sigmas above the validity floor, so no tiny divisions. Returns false
// when the candidate is rejected, and Candidate::None when the
// eigendecomposition itself fails, since there is then no candidate to judge.
template <typename MapT>
Candidate try_gram(const MapT& mat, int rows, int cols, int max_bond_dim,
                   double cutoff, double m_fro_sq, bool keep_unverified,
                   SvdTruncation& out, std::string& why) {
    const bool tall = rows >= cols;
    const Eigen::MatrixXcd G = tall ? Eigen::MatrixXcd(mat.adjoint() * mat)
                                    : Eigen::MatrixXcd(mat * mat.adjoint());
    const int gd = tall ? cols : rows;
    RealVector g_evals(gd);
    DenseMatrix g_evecs(gd, gd);
    if (!eigh(G.data(), gd, MatrixOrder::ColMajor, g_evals.data(),
              g_evecs.data())) {
        why = "the Gram-route eigendecomposition failed";
        return Candidate::None;
    }
    RealVector Sg(gd);
    for (int i = 0; i < gd; ++i) {
        // Eigenvalues ascend; emit sigmas descending.
        Sg(i) = std::sqrt(std::max(0.0, g_evals(gd - 1 - i)));
    }
    const double smax = (gd > 0) ? Sg(0) : 0.0;
    const double floor_g = gram_validity_floor(gd, smax);

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

    return attempt(mat, rows, cols, Sg, Ug, Vg, floor_g, max_bond_dim, cutoff,
                   m_fro_sq, keep_unverified, out, why);
}

template <typename MapT>
SvdTruncation run_ladder(const MapT& mat, int rows, int cols, int max_bond_dim,
                         double cutoff, SVDMethod method, const SvdPolicy& policy,
                         const char* ctx) {
    // The input's storage order travels with the call rather than being
    // normalised here, so a caller's block is mapped in place on both paths.
    const MatrixOrder order = static_cast<bool>(MapT::IsRowMajor)
                                  ? MatrixOrder::RowMajor
                                  : MatrixOrder::ColMajor;

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

    const bool keep = policy.rejection == SvdRejection::Ignore;
    const bool report = policy.report == SvdReport::Warn;

    // Rung 1: the kernel the caller selected.
    SvdTruncation out;
    std::string why;
    Candidate got = try_kernel(mat, rows, cols, order, method, max_bond_dim, cutoff,
                               m_fro_sq, keep, out, why);
    if (got == Candidate::Accepted) return out;

    const std::string block = std::string("the ") + to_string(method) +
                              " factorisation of a " + std::to_string(rows) +
                              "x" + std::to_string(cols) + " block";
    const std::string where = std::string(ctx) + ": " + block;
    // What a rung's outcome was, in the words of a warning or a refusal.
    const auto outcome = [](Candidate c, const std::string& reason) {
        return std::string(c == Candidate::None ? " produced nothing ("
                                                : " failed verification (") +
               reason + ")";
    };

    if (policy.rejection == SvdRejection::Throw) {
        // The caller asked for this kernel and no other. Stopping here is the
        // answer they chose over a tensor from a kernel they did not name.
        raise<RuntimeFailure>(ctx, block + outcome(got, why) +
                                       " and svd_rejection is SvdRejection::Throw; "
                                       "refusing to continue with it");
    }
    if (got == Candidate::Rejected && keep) {
        if (report) {
            emit_warning(where + outcome(got, why) +
                         "; used as it is, unverified, under SvdRejection::Ignore");
        }
        return out;
    }

    // The ladder: under Fix for any rejection, under Ignore only when the kernel
    // produced nothing to use. Under Ignore a rung's candidate is used whether
    // or not it verifies.
    std::string trail = block + outcome(got, why);

    // Rung 2: autonne's Jacobi, an independent road to the same factorisation.
    // Pointless when it was the kernel that just failed, so that case goes
    // straight to the Gram route.
    if (method != SVDMethod::Jacobi) {
        if (report) emit_warning(where + outcome(got, why) + "; retrying with SVDMethod::Jacobi");
        got = try_kernel(mat, rows, cols, order, SVDMethod::Jacobi, max_bond_dim, cutoff,
                         m_fro_sq, keep, out, why);
        if (got == Candidate::Accepted || (got == Candidate::Rejected && keep)) {
            out.used_jacobi_rescue = true;
            if (out.used_unverified && report) {
                emit_warning(where + ": the Jacobi rescue" + outcome(got, why) +
                             "; used as it is, unverified, under SvdRejection::Ignore");
            }
            return out;
        }
        trail += "; the Jacobi rescue" + outcome(got, why);
    }

    // Rung 3: the Gram route, only when the caller accepts it.
    if (!policy.accept_gram) {
        raise<RuntimeFailure>(ctx, trail +
                                       "; the Gram route is the remaining rung and "
                                       "svd_accept_gram is off; refusing to continue "
                                       "with a corrupt tensor");
    }
    if (report) emit_warning(where + trail.substr(block.size()) +
                             "; recomputing through the Gram route");
    got = try_gram(mat, rows, cols, max_bond_dim, cutoff, m_fro_sq, keep, out, why);
    if (got == Candidate::Accepted || (got == Candidate::Rejected && keep)) {
        out.used_gram_fallback = true;
        if (out.used_unverified && report) {
            emit_warning(where + ": the Gram route" + outcome(got, why) +
                         "; used as it is, unverified, under SvdRejection::Ignore");
        }
        return out;
    }

    // THROW. Never continue with a corrupt tensor: silent propagation is
    // exactly how this class of defect manifests.
    raise<RuntimeFailure>(ctx, trail + "; the Gram route" + outcome(got, why) +
                                   "; every permitted rung failed, refusing to "
                                   "continue with a corrupt tensor");
}

} // namespace

SvdTruncation svd_truncate_verified(const Complex128* data, int rows, int cols,
                                    MatrixOrder order, int max_bond_dim,
                                    double cutoff, SVDMethod method,
                                    const SvdPolicy& policy, const char* ctx) {
    // Complex128 {double real, double imag} is layout-identical to
    // std::complex<double>, so both orders map in place and neither caller pays
    // an O(rows*cols) copy to hand a block over.
    const auto* p = reinterpret_cast<const std::complex<double>*>(data);
    if (order == MatrixOrder::RowMajor) {
        Eigen::Map<const RowMajorC> mat(p, rows, cols);
        return run_ladder(mat, rows, cols, max_bond_dim, cutoff, method, policy, ctx);
    }
    Eigen::Map<const ColMajorC> mat(p, rows, cols);
    return run_ladder(mat, rows, cols, max_bond_dim, cutoff, method, policy, ctx);
}

void note_nondefault_svd_policy(SvdLayer layer, const SvdPolicy& policy) {
    // One flag per layer and per setting, so each note is emitted once in the
    // process whichever thread splits first.
    enum Setting { Throw, Ignore, AcceptGram, Silent, SettingCount };
    static std::atomic<bool> noted[2][SettingCount] = {};
    const int l = layer == SvdLayer::Qubit ? 0 : 1;
    const char* name = layer == SvdLayer::Qubit ? "the qubit MPS" : "the qudit MPS";
    const auto once = [&](Setting setting, const char* text) {
        if (!noted[l][setting].exchange(true)) {
            emit_warning(std::string("note: ") + name + " has " + text);
        }
    };
    if (policy.rejection == SvdRejection::Throw) {
        once(Throw, "svd_rejection = SvdRejection::Throw, not the default (Fix): a "
                    "factorisation that fails verification ends the run instead of "
                    "being repaired. Be alert to it.");
    }
    if (policy.rejection == SvdRejection::Ignore) {
        once(Ignore, "svd_rejection = SvdRejection::Ignore, not the default (Fix): a "
                     "factorisation that fails verification is used as it is, so the "
                     "state can be wrong with nothing further to say so. Be alert to "
                     "the results.");
    }
    if (policy.accept_gram) {
        once(AcceptGram, "svd_accept_gram on, not the default: the ladder may end on "
                         "the Gram route, which drops singular values below its "
                         "validity floor. Be alert to floor_rejected_weight().");
    }
    if (policy.report == SvdReport::Silent) {
        once(Silent, "svd_report = SvdReport::Silent, not the default (Warn): repaired "
                     "and unverified factorisations are counted but not reported. Be "
                     "alert to the counters.");
    }
}

} // namespace detail
} // namespace lindblad
