// mps_sim.cpp — Matrix Product State simulator
// SVD truncation defaults to Eigen JacobiSVD (accurate). BDCSVD is a faster
// opt-in but is CURRENTLY BROKEN for complex/degenerate inputs (R.1.11.2:
// on a 36x36 degenerate complex matrix it violates U·S·V† = M and Frobenius
// norm preservation, under strict FP too — a genuine Eigen 3.4.0 defect);
// selecting it emits a loud runtime warning.
// JacobiSVD itself has a narrower Eigen 3.4.0 defect (R.1.16.0, issue #44):
// on degenerate rank-deficient inputs it can emit NaN inside NULL-SPACE
// singular vectors. svd_truncate below is hardened against it (the garbage
// is discardable by construction; anything non-finite in the KEPT slice
// throws). Reproducers for both defects: tests/diag_r1160_matrices.hpp.
// Non-adjacent two-qubit gates are handled via SWAP chains (correct MPS-native approach).
// Single-qubit marginals use efficient left/right boundary contraction — O(N chi^3).

#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/detail/validate.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/gates.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace lindblad {

// Emit a one-time, very visible warning when the (currently broken) BDCSVD
// backend is selected. Eigen's BDCSVD produces inaccurate results for
// complex/degenerate inputs; Jacobi is the accurate default.
static void warn_bdc_broken_once() {
    static bool warned = false;
    if (warned) return;
    warned = true;
    emit_warning(
        "***************************************************************\n"
        "  WARNING: SVDMethod::BDC (Eigen BDCSVD) is SELECTED but is\n"
        "  CURRENTLY BROKEN for complex / degenerate inputs: MPS results may\n"
        "  be SILENTLY WRONG.\n"
        "  Use SVDMethod::Jacobi (the default) until the upstream fix lands.\n"
        "***************************************************************");
}

// =============================================================================
// MPSState implementation
// =============================================================================

MPSState::MPSState(int n_qubits, int max_bond_dim, double cutoff)
    : n_qubits(n_qubits)
    , max_bond_dim(max_bond_dim)
    , cutoff(cutoff)
    , total_truncation_error(0.0) {
    // A bond dimension below 1 retains no singular values at all. Left
    // unchecked it reaches svd_truncate as k = min(k, max_bond_dim) == 0,
    // which is indistinguishable there from a numerically corrupt spectrum:
    // the rescue branch keeps one sigma, reports nothing discarded, and the
    // verify step then measures a rank-1 residual against a bound sized for a
    // factorisation that dropped nothing. Both routes fail and the throw
    // blames the SVD backend for what is a caller argument.
    detail::check_require(max_bond_dim >= 1, "MPSState",
                          "max_bond_dim must be >= 1 (got " +
                              std::to_string(max_bond_dim) + ")");
    tensors.resize(n_qubits);
    for (int i = 0; i < n_qubits; ++i) {
        tensors[i] = MPSTensor(1, 1);
        tensors[i](0, 0, 0) = Complex128(1.0, 0.0);  // |0⟩ amplitude
        tensors[i](0, 1, 0) = Complex128(0.0, 0.0);  // |1⟩ amplitude
    }
}

// =============================================================================
// SVD via Eigen3 BDCSVD — robust divide-and-conquer SVD
// Truncates to max_bond_dim singular values above cutoff threshold.
// =============================================================================

void MPSState::svd_truncate(
    const std::vector<Complex128>& M,
    int rows, int cols,
    std::vector<Complex128>& U_out,
    std::vector<double>& S_out,
    std::vector<Complex128>& Vt_out,
    int& new_rank
) {
    // Complex128 {double real, double imag} is layout-identical to std::complex<double>.
    // Zero-copy Eigen::Map avoids the O(rows*cols) element-by-element copy before SVD.
    using EigenCMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const EigenCMatrix> mat(
        reinterpret_cast<const std::complex<double>*>(M.data()),
        rows, cols
    );

    // SVD backend: Jacobi by default (accurate, avoids the BDCSVD accuracy
    // defect), BDC opt-in for speed at large chi. Compute into shared locals so
    // the truncation logic below is backend-agnostic.
    Eigen::MatrixXcd U_eigen, V_eigen;
    Eigen::VectorXd S_eigen;
    if (svd_method == SVDMethod::BDC) {
        warn_bdc_broken_once();
        Eigen::BDCSVD<EigenCMatrix> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        S_eigen = svd.singularValues();
        U_eigen = svd.matrixU();
        V_eigen = svd.matrixV();
    } else {
        Eigen::JacobiSVD<EigenCMatrix> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        S_eigen = svd.singularValues();
        U_eigen = svd.matrixU();
        V_eigen = svd.matrixV();
    }

    // ------------------------------------------------------------------
    // SELECT -> VERIFY -> FALLBACK -> THROW  (issue #44)
    //
    // Eigen 3.4.0's SVD on degenerate rank-deficient inputs (the 13-qubit
    // Shor IQFT two-site thetas, spectrum {1,1,1,1,0,...}) returns broken
    // factorisations whose SHAPE MORPHS with tiny input and codegen
    // perturbations — observed across seven diagnostic runs: NaN inside
    // null-space singular vectors (both FP models), non-finite entries
    // interleaved into S displacing real sigmas, a KEPT singular vector
    // that is wrong but perfectly finite (fidelity fell to (3/4)^2 with
    // zero truncation error), and an all-non-finite S. Unmarked garbage
    // cannot be pattern-matched away, so this routine TRUSTS NOTHING:
    //   1. SELECT: every bit-level-finite sigma is a candidate, wherever it
    //      sits in S (immune to ordering corruption); truncate by DISCARDED
    //      WEIGHT (below); cap at max_bond_dim; gather matching U/V columns
    //      individually.
    //   2. VERIFY: kept slice bit-finite AND the Frobenius identity
    //        ||M - U_k S_k V_k^H||_F <= sqrt(discarded) + c*N*eps*||M||_F
    //      which holds with EQUALITY at zero slack for any true truncated
    //      SVD, the allowance being only the backward error a stable SVD is
    //      entitled to (see the gate itself for the sizing). Every failure
    //      mode observed so far trips it. Costs one rank-slice GEMM — small
    //      next to the SVD itself, and correctness outranks it (golden rule
    //      #1 over #2).
    //   3. FALLBACK on failure: recompute via the Gram route (M^H M or
    //      M M^H, whichever is smaller, SelfAdjointEigenSolver — robust on
    //      exactly-degenerate Hermitian input), sigma = sqrt(max(lambda,0)),
    //      partner factor built only for kept sigmas (no tiny divisions).
    //      Effective floor sqrt(eps)*sigma_max: Gram squares the condition
    //      number, so smaller sigmas are noise there; the extra discarded
    //      weight is honestly accounted in truncation_error().
    //   4. THROW if the fallback fails verification too — never continue
    //      with a corrupt tensor (silent NaN propagation is exactly how
    //      issue #44 manifests).
    // Which rung ran is otherwise invisible to the caller, since a rescued
    // bond and a clean one produce equally valid tensors. svd_call_count()
    // and gram_fallback_count() report it.
    // Reproducers: tests/diag_r1160_matrices.hpp and the DiagR1160* suites.
    // ------------------------------------------------------------------

    ++svd_calls;

    double m_fro_sq = 0.0;
    for (const auto& c : M) m_fro_sq += c.real * c.real + c.imag * c.imag;

    double pending_discarded = 0.0;
    double pending_resid_excess = 0.0;

    // TRUNCATION RULE — discarded weight, not a magnitude threshold.
    //
    // `cutoff` is the fraction of total weight (sum of sigma^2) that truncation
    // may throw away. Comparing each sigma against `cutoff` as an ABSOLUTE
    // magnitude instead would ask a question about the noise rather than about
    // the data: a value's distance from a fixed constant depends on the scale of
    // the matrix it came from AND on how the target rounded its way there. On
    // one 13-qubit state that counts 12 significant directions on x86-64 and 17
    // on arm64 — twelve real sigmas at 2.9e-01, everything else numerical zero,
    // and a fixed threshold sitting inside the noise band instead of inside the
    // eleven-order gap above it.
    //
    // A weight fraction is scale-free, so the same spectrum classifies
    // identically on any target, and it bounds the physical error directly
    // instead of proxying it. Note the budget is a CEILING, not a quota: on a
    // bimodal spectrum (stabiliser, GHZ, Shor) there is nothing between the
    // noise and the budget, so nothing extra is discarded and the result is
    // bit-for-bit what the magnitude rule produced.
    //
    // `sigma_floor` is a separate concept and NOT a truncation knob: it rejects
    // values that are not trustworthy data at all (the Gram route squares the
    // condition number, so anything under sqrt(eps)*sigma_max there is noise
    // regardless of how much weight it carries). Weight below it is still
    // honestly counted as discarded.

    auto attempt = [&](const Eigen::VectorXd& S_try,
                       const Eigen::MatrixXcd& U_try,
                       const Eigen::MatrixXcd& V_try,
                       double sigma_floor) -> bool {
        const int md = static_cast<int>(S_try.size());
        std::vector<std::pair<double, int>> fs;  // (sigma, source index)
        fs.reserve(static_cast<size_t>(md));
        double below_floor = 0.0;  // weight rejected as untrustworthy
        for (int i = 0; i < md; ++i) {
            const double s = S_try(i);
            // A NaN sigma satisfies 's > sigma_floor' and would be kept,
            // growing chi with garbage.
            if (!is_finite_strict(s)) continue;  // artifact, not weight
            if (s > sigma_floor) fs.push_back({s, i});
            else                 below_floor += s * s;
        }
        std::stable_sort(fs.begin(), fs.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        // Walk up from the smallest survivor, dropping while the running
        // discarded weight stays inside the budget. Weight already rejected by
        // sigma_floor counts against it, so a Gram route that lost a lot to the
        // validity floor does not then also truncate aggressively.
        double total = below_floor;
        for (const auto& p : fs) total += p.first * p.first;
        const double budget = cutoff * total;

        int k = static_cast<int>(fs.size());
        {
            double d = below_floor;
            while (k > 1) {
                const double w = fs[static_cast<size_t>(k - 1)].first *
                                 fs[static_cast<size_t>(k - 1)].first;
                if (d + w > budget) break;
                d += w;
                --k;
            }
        }
        k = std::min<int>(k, max_bond_dim);
        if (k == 0) {
            // Numerically-zero matrix or fully corrupt spectrum: keep the
            // single largest finite sigma if one exists at all.
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
            // The rescued sigma may have sat below sigma_floor, in which case
            // its weight is already in below_floor. It is now KEPT, so take it
            // back out or it would be counted as kept and discarded at once.
            if (!(best_val > sigma_floor)) below_floor -= best_val * best_val;
            fs.assign(1, {best_val, best});
            k = 1;
        }

        // Discarded weight = every finite sigma NOT kept (beyond-rank AND
        // below-floor; non-finite entries are artifacts, not weight).
        //
        // Sum the discarded buckets DIRECTLY. Do not compute this as
        // `total - kept`: for a normalised state both are ~1.0 while the real
        // difference is ~1e-30, so the subtraction cannot resolve it and
        // returns multiples of eps instead — truncation_error() then reports
        // ~1e-15 of phantom loss for a bond that discarded nothing. Adding up
        // the small values keeps every term at its own scale.
        double discarded = below_floor;
        for (size_t i = static_cast<size_t>(k); i < fs.size(); ++i)
            discarded += fs[i].first * fs[i].first;

        // Gather (descending sigma order: downstream conventions unchanged).
        S_out.resize(static_cast<size_t>(k));
        U_out.resize(static_cast<size_t>(rows) * k);
        Vt_out.resize(static_cast<size_t>(k) * cols);
        for (int r = 0; r < k; ++r) {
            const int src = fs[static_cast<size_t>(r)].second;
            S_out[static_cast<size_t>(r)] = fs[static_cast<size_t>(r)].first;
            for (int row = 0; row < rows; ++row) {
                const auto u = U_try(row, src);
                U_out[static_cast<size_t>(row) * k + r] =
                    Complex128(u.real(), u.imag());
            }
            for (int c = 0; c < cols; ++c) {
                const auto v = std::conj(V_try(c, src));  // row r of V^dagger
                Vt_out[static_cast<size_t>(r) * cols + c] =
                    Complex128(v.real(), v.imag());
            }
        }

        // Verify 1: kept slice bit-finite.
        for (int i = 0; i < k; ++i)
            if (!is_finite_strict(S_out[static_cast<size_t>(i)])) return false;
        for (const auto& c : U_out)
            if (!is_finite_strict(c.real) || !is_finite_strict(c.imag)) return false;
        for (const auto& c : Vt_out)
            if (!is_finite_strict(c.real) || !is_finite_strict(c.imag)) return false;

        // Verify 2: Frobenius identity of the truncated factorisation.
        //
        // ||M - U_k S_k V_k^H||_F^2 == discarded holds with EQUALITY for a
        // truncated SVD in exact arithmetic, so the slack added to `discarded`
        // is the entire allowance a COMPUTED factorisation gets. Size it from
        // the backward error a stable SVD is entitled to and no larger: the
        // standard bound is ||M - U S V^H||_F <= c * N * eps * ||M||_F with N
        // the larger dimension, and this compares squared norms, so the bound
        // enters squared.
        //
        // c is generous because the two errors are not symmetric: a false
        // REJECT costs one Gram recomputation, while a false ACCEPT is a wrong
        // state that nothing downstream will catch. Even so the gate lands far
        // below the defects this ladder exists for. On the degenerate
        // rank-deficient thetas of a 13-qubit period-finding circuit, Eigen
        // 3.4.0 has been measured returning a factorisation whose excess over
        // the identity is 7.4e-16 of ||M||_F^2 (a 2.7e-8 relative
        // reconstruction error, which is not backward-stable at any dimension)
        // while a healthy factorisation of the same matrices measures 2.8e-30.
        // Ten orders of clearance above, four below.
        Eigen::Map<const EigenCMatrix> Um(
            reinterpret_cast<const std::complex<double>*>(U_out.data()), rows, k);
        Eigen::Map<const EigenCMatrix> Vtm(
            reinterpret_cast<const std::complex<double>*>(Vt_out.data()), k, cols);
        Eigen::Map<const Eigen::VectorXd> Sm(S_out.data(), k);
        const double resid =
            (mat - Um * Sm.asDiagonal() * Vtm).squaredNorm();
        if (!is_finite_strict(resid)) return false;
        constexpr double kBackwardErrorSlack = 64.0;
        const double bwd = kBackwardErrorSlack *
                           static_cast<double>(std::max(rows, cols)) *
                           std::numeric_limits<double>::epsilon();
        // The comparison is stated and made in the AMPLITUDE domain:
        //   ||M - U_k S_k V_k^H||_F <= sqrt(discarded) + bwd * ||M||_F
        // Squaring it here would drop the cross term
        // 2*bwd*sqrt(discarded*||M||_F^2), which is the dominant allowance
        // whenever truncation is heavy. That term is not optional slack: with
        // `discarded` at the scale of ||M||_F^2, resid and discarded are two
        // large nearly-equal quantities computed by different routes, so their
        // difference carries FIRST-order rounding (~eps*discarded) while a
        // squared-domain bound offers only second-order room. It vanishes as
        // discarded -> 0, so a bond that truncated nothing is still held to
        // the strict backward-error bound alone.
        const double allowed =
            std::sqrt(discarded) + bwd * std::sqrt(m_fro_sq);
        if (resid > allowed * allowed + 1e-18) return false;

        new_rank = k;
        pending_discarded = discarded;
        // Excess over a perfect truncated SVD, which satisfies the Frobenius
        // identity with EQUALITY (resid == discarded). Subtracting `discarded`
        // leaves only the factorisation's own error, so the figure means the
        // same thing on a bond that truncated heavily and one that truncated
        // nothing. Clamped because the two sides are computed differently and
        // can cross by an ulp when both are dust.
        pending_resid_excess =
            (m_fro_sq > 0.0) ? std::max(0.0, resid - discarded) / m_fro_sq : 0.0;
        return true;
    };

    // Direct SVD: no validity floor. A backend SVD resolves sigmas down to
    // ~eps*sigma_max, so every finite value it reports is trustworthy data;
    // how much of it to keep is the weight budget's decision alone.
    if (!attempt(S_eigen, U_eigen, V_eigen, 0.0)) {
        // Gram-route fallback.
        const bool tall = rows >= cols;
        const Eigen::MatrixXcd G =
            tall ? Eigen::MatrixXcd(mat.adjoint() * mat)
                 : Eigen::MatrixXcd(mat * mat.adjoint());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(G);
        if (es.info() != Eigen::Success) {
            throw std::runtime_error(
                "MPS svd_truncate: Gram-route eigendecomposition failed on a " +
                std::to_string(rows) + "x" + std::to_string(cols) + " matrix");
        }
        const int gd = static_cast<int>(es.eigenvalues().size());
        Eigen::VectorXd Sg(gd);
        for (int i = 0; i < gd; ++i) {
            // Eigenvalues ascend; emit sigmas descending.
            Sg(i) = std::sqrt(std::max(0.0, es.eigenvalues()(gd - 1 - i)));
        }
        // Validity floor for THIS route only (not a truncation knob): the Gram
        // matrix squares the condition number, so sigmas under sqrt(eps) times
        // sigma_max carry no information here however much weight they hold.
        // Relative to sigma_max, so it means the same thing on every target.
        constexpr double kGramValidityFloorRel = 1.5e-8;  // ~sqrt(DBL_EPSILON)
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

        if (!attempt(Sg, Ug, Vg, floor_g)) {
            throw std::runtime_error(
                "MPS svd_truncate: both the SVD backend and the "
                "Gram-eigendecomposition fallback failed verification on a " +
                std::to_string(rows) + "x" + std::to_string(cols) +
                " matrix (non-finite or non-reconstructing factorisation); "
                "refusing to continue with a corrupt tensor");
        }
        // Counted only here, past the throw: gram_fallback_count() reports
        // rescues that succeeded, and a failed rescue does not return.
        ++gram_fallbacks;
    }

    total_truncation_error += pending_discarded;
    max_verify_resid_excess =
        std::max(max_verify_resid_excess, pending_resid_excess);
}

// =============================================================================
// Single-qubit gate — O(chi) per qubit
// U is a 2x2 unitary in row-major order: [u00, u01, u10, u11]
// =============================================================================

void MPSState::apply_single_qubit_gate(
    const std::array<Complex128, 4>& U, int qubit,
    ValidationOptions validation
) {
    detail::check_qubit(qubit, n_qubits, "MPSState::apply_single_qubit_gate");
    detail::check_unitary(U.data(), 2, validation,
                          "MPSState::apply_single_qubit_gate");
    auto& T = tensors[qubit];
    MPSTensor result(T.bond_left, T.bond_right);

    for (int l = 0; l < T.bond_left; ++l) {
        for (int r = 0; r < T.bond_right; ++r) {
            // new[l, po, r] = sum_pi U[po, pi] * T[l, pi, r]
            for (int po = 0; po < 2; ++po) {
                Complex128 sum(0.0, 0.0);
                for (int pi = 0; pi < 2; ++pi) {
                    sum += U[po * 2 + pi] * T(l, pi, r);
                }
                result(l, po, r) = sum;
            }
        }
    }

    T = std::move(result);
}

// =============================================================================
// Adjacent two-qubit gate — contract, apply, SVD-split
// U is 4x4 in row-major index order: U[po1*2+po2, pi1*2+pi2]
// q1 and q2 MUST be adjacent (q2 == q1+1)
// =============================================================================

void MPSState::apply_two_qubit_gate_adjacent(
    const std::array<Complex128, 16>& U, int q1
) {
    int q2 = q1 + 1;
    detail::check_qubit(q1, n_qubits, "MPSState::apply_two_qubit_gate_adjacent");
    detail::check_qubit(q2, n_qubits, "MPSState::apply_two_qubit_gate_adjacent");

    auto& T1 = tensors[q1];
    auto& T2 = tensors[q2];

    int bl = T1.bond_left;
    int bm = T1.bond_right;  // = T2.bond_left
    int br = T2.bond_right;

    // theta[l, p1, p2, r] = sum_m T1[l,p1,m] * T2[m,p2,r]
    // Stored as (bl*2) x (2*br) matrix for SVD: row = l*2+p1, col = p2*br+r.
    //
    // This contraction is a single zero-copy Eigen GEMM.
    // MPSTensor data is contiguous row-major in exactly the needed shapes:
    //   T1[(l*2+p1), m] is (bl*2) x bm,  T2[m, (p2*br+r)] is bm x (2*br),
    // and Complex128 is layout-identical to std::complex<double>, so both map
    // in place. theta = M1 * M2 replaces the O(4*chi^3) scalar loop with BLAS3.
    // (The 4x4 gate contraction below stays scalar — it is only O(16*chi^2).)
    int rows = bl * 2;
    int cols = 2 * br;
    std::vector<Complex128> matrix(static_cast<size_t>(rows) * cols, Complex128(0.0, 0.0));
    {
        using EigenCM = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const EigenCM> M1(
            reinterpret_cast<const std::complex<double>*>(T1.data.data()), rows, bm);
        Eigen::Map<const EigenCM> M2(
            reinterpret_cast<const std::complex<double>*>(T2.data.data()), bm, cols);
        Eigen::Map<EigenCM>(
            reinterpret_cast<std::complex<double>*>(matrix.data()), rows, cols) = M1 * M2;
    }

    // Apply gate U to theta: theta_new[row(po1),col(po2)] = sum U * theta
    std::vector<Complex128> theta_new(rows * cols, Complex128(0.0, 0.0));
    for (int l = 0; l < bl; ++l) {
        for (int po1 = 0; po1 < 2; ++po1) {
            for (int po2 = 0; po2 < 2; ++po2) {
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int pi1 = 0; pi1 < 2; ++pi1) {
                        for (int pi2 = 0; pi2 < 2; ++pi2) {
                            sum += U[(po1 * 2 + po2) * 4 + (pi1 * 2 + pi2)] *
                                   matrix[(l * 2 + pi1) * cols + (pi2 * br + r)];
                        }
                    }
                    theta_new[(l * 2 + po1) * cols + (po2 * br + r)] = sum;
                }
            }
        }
    }

    // SVD theta_new into T1' and T2'
    std::vector<Complex128> U_mat, Vt_mat;
    std::vector<double> S_vals;
    int new_rank;
    svd_truncate(theta_new, rows, cols, U_mat, S_vals, Vt_mat, new_rank);

    // T1'[l, p1, r] = U_mat[l*2+p1, r] * S[r]  (absorb S into T1)
    T1 = MPSTensor(bl, new_rank);
    for (int l = 0; l < bl; ++l)
        for (int p1 = 0; p1 < 2; ++p1)
            for (int r = 0; r < new_rank; ++r)
                T1(l, p1, r) = U_mat[(l * 2 + p1) * new_rank + r] * Complex128(S_vals[r], 0.0);

    // T2'[l, p2, r] = Vt_mat[l, p2*br+r]
    T2 = MPSTensor(new_rank, br);
    for (int l = 0; l < new_rank; ++l)
        for (int p2 = 0; p2 < 2; ++p2)
            for (int r = 0; r < br; ++r)
                T2(l, p2, r) = Vt_mat[l * cols + p2 * br + r];
}

// =============================================================================
// SWAP gate in MPS (swaps two adjacent tensors via SVD)
// =============================================================================

void MPSState::apply_swap_adjacent(int q) {
    // Apply FSWAP (fermionic SWAP) as a 4x4 gate
    // SWAP: |00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→|11⟩
    // Row-major: SWAP[po1*2+po2, pi1*2+pi2]
    std::array<Complex128, 16> SWAP_gate{};
    SWAP_gate[0*4+0] = Complex128(1, 0);  // |00⟩→|00⟩
    SWAP_gate[1*4+2] = Complex128(1, 0);  // |01⟩→|10⟩
    SWAP_gate[2*4+1] = Complex128(1, 0);  // |10⟩→|01⟩
    SWAP_gate[3*4+3] = Complex128(1, 0);  // |11⟩→|11⟩
    apply_two_qubit_gate_adjacent(SWAP_gate, q);
}

// =============================================================================
// General two-qubit gate for arbitrary (non-adjacent) qubits
// Uses SWAP chain: move q1 and q2 adjacent, apply gate, swap back.
// =============================================================================

void MPSState::apply_two_qubit_gate(
    const std::array<Complex128, 16>& U, int q1, int q2,
    ValidationOptions validation
) {
    detail::check_qubit(q1, n_qubits, "MPSState::apply_two_qubit_gate");
    detail::check_qubit(q2, n_qubits, "MPSState::apply_two_qubit_gate");
    detail::check_distinct2(q1, q2, "MPSState::apply_two_qubit_gate");
    detail::check_unitary(U.data(), 4, validation,
                          "MPSState::apply_two_qubit_gate");

    // Ensure q1 < q2
    bool swapped = (q1 > q2);
    if (swapped) {
        std::swap(q1, q2);
        // Swap the qubit ordering in U (pi1↔pi2 and po1↔po2)
        // U'[po1*2+po2, pi1*2+pi2] = U[po2*2+po1, pi2*2+pi1]
        std::array<Complex128, 16> U_swapped{};
        for (int po1 = 0; po1 < 2; ++po1)
            for (int po2 = 0; po2 < 2; ++po2)
                for (int pi1 = 0; pi1 < 2; ++pi1)
                    for (int pi2 = 0; pi2 < 2; ++pi2)
                        U_swapped[(po1*2+po2)*4+(pi1*2+pi2)] = U[(po2*2+po1)*4+(pi2*2+pi1)];
        apply_two_qubit_gate(U_swapped, q1, q2, {Validation::Ignore});
        return;
    }

    // Now q1 < q2. Move q2 next to q1 by SWAP chain from right.
    // After final SWAP chain: q2 is at position q1+1.
    for (int i = q2 - 1; i > q1; --i) {
        apply_swap_adjacent(i);  // SWAP qubits at pos i and i+1
    }

    // Apply the gate on (q1, q1+1)
    apply_two_qubit_gate_adjacent(U, q1);

    // Swap q2 back to its original position
    for (int i = q1 + 1; i < q2; ++i) {
        apply_swap_adjacent(i);
    }
}

// =============================================================================
// current_max_bond_dim
// =============================================================================

int MPSState::current_max_bond_dim() const {
    int max_chi = 0;
    for (const auto& t : tensors) {
        max_chi = std::max(max_chi, std::max(t.bond_left, t.bond_right));
    }
    return max_chi;
}

// =============================================================================
// probabilities_single — O(N chi^3) efficient boundary contraction
// Computes P(0) and P(1) for a single qubit without expanding full statevector.
// =============================================================================

std::vector<double> MPSState::probabilities_single(int qubit) const {
    detail::check_qubit(qubit, n_qubits, "MPSState::probabilities_single");

    // Left environment: left_env[m1, m2] tensored from sites 0..qubit-1
    // Initialise: left_env = [[1]] (1x1 identity)
    int chi_left = tensors[qubit].bond_left;
    std::vector<Complex128> left_env(chi_left * chi_left, Complex128(0.0, 0.0));
    {
        // Build left boundary by contracting ⟨ψ|…|ψ⟩ from left
        // Starting with 1x1
        std::vector<Complex128> env(1, Complex128(1.0, 0.0));
        int env_dim = 1;
        for (int q = 0; q < qubit; ++q) {
            const auto& T = tensors[q];
            int bl = T.bond_left;
            int br = T.bond_right;
            std::vector<Complex128> new_env(br * br, Complex128(0.0, 0.0));
            for (int m2 = 0; m2 < br; ++m2) {
                for (int m1 = 0; m1 < br; ++m1) {
                    Complex128 sum(0.0, 0.0);
                    for (int p = 0; p < 2; ++p) {
                        for (int l1 = 0; l1 < bl; ++l1) {
                            for (int l2 = 0; l2 < bl; ++l2) {
                                // env[l1,l2] * T[l1,p,m1] * conj(T[l2,p,m2])
                                sum += env[l1 * env_dim + l2] *
                                       T(l1, p, m1) * T(l2, p, m2).conj();
                            }
                        }
                    }
                    new_env[m1 * br + m2] = sum;  // note: env[m1,m2]
                }
            }
            env = new_env;
            env_dim = br;
        }
        left_env = env;
    }

    // Right environment: right_env[m1, m2] tensored from sites qubit+1..N-1
    int chi_right = tensors[qubit].bond_right;
    std::vector<Complex128> right_env;
    {
        std::vector<Complex128> env(1, Complex128(1.0, 0.0));
        int env_dim = 1;
        for (int q = n_qubits - 1; q > qubit; --q) {
            const auto& T = tensors[q];
            int bl = T.bond_left;
            int br = T.bond_right;
            std::vector<Complex128> new_env(bl * bl, Complex128(0.0, 0.0));
            for (int m1 = 0; m1 < bl; ++m1) {
                for (int m2 = 0; m2 < bl; ++m2) {
                    Complex128 sum(0.0, 0.0);
                    for (int p = 0; p < 2; ++p) {
                        for (int r1 = 0; r1 < br; ++r1) {
                            for (int r2 = 0; r2 < br; ++r2) {
                                sum += env[r1 * env_dim + r2] *
                                       T(m1, p, r1) * T(m2, p, r2).conj();
                            }
                        }
                    }
                    new_env[m1 * bl + m2] = sum;
                }
            }
            env = new_env;
            env_dim = bl;
        }
        right_env = env;
    }

    // Contract for each physical index
    const auto& Tq = tensors[qubit];
    std::vector<double> probs(2, 0.0);
    for (int p = 0; p < 2; ++p) {
        Complex128 sum(0.0, 0.0);
        for (int l1 = 0; l1 < chi_left; ++l1) {
            for (int l2 = 0; l2 < chi_left; ++l2) {
                Complex128 lv = left_env[l1 * chi_left + l2];
                for (int r1 = 0; r1 < chi_right; ++r1) {
                    for (int r2 = 0; r2 < chi_right; ++r2) {
                        Complex128 rv = right_env[r1 * chi_right + r2];
                        sum += lv * Tq(l1, p, r1) * Tq(l2, p, r2).conj() * rv;
                    }
                }
            }
        }
        probs[p] = sum.real;
    }

    return probs;
}

// =============================================================================
// measure_sequential — correct correlated sampling via left-to-right projection
// Precomputes all right environments O(N·chi^3) then processes left-to-right:
//   1. Compute P(0) and P(1) using precomputed right_env + incremental left_env + tensor
//   2. Sample outcome from this conditional distribution
//   3. Project the local tensor onto the measured outcome (collapse)
//   4. Renormalize; update left_env incrementally
// O(N·chi^3) per shot — avoids the O(N^2·chi^3) cost of rebuilding environments per qubit.
// =============================================================================

std::string MPSState::measure_sequential(std::mt19937_64& rng) {
    std::string bits(n_qubits, '0');

    // === Precompute right environments O(N·chi^3) total ===
    // right_envs[q] = density env for sites q..N-1, size bond_left[q] x bond_left[q].
    // right_envs[N] = [[1]] (1x1 identity).
    // Computed from original (unmodified) tensors — valid because we process left-to-right
    // and tensors[q+1..N-1] are untouched when computing probs for qubit q.
    std::vector<std::vector<Complex128>> right_envs(n_qubits + 1);
    right_envs[n_qubits] = {Complex128(1.0, 0.0)};

    for (int q = n_qubits - 1; q >= 0; --q) {
        const auto& T = tensors[q];
        const int bl = T.bond_left;
        const int br = T.bond_right;

        std::vector<Complex128> new_env(bl * bl, Complex128(0.0, 0.0));
        for (int m1 = 0; m1 < bl; ++m1) {
            for (int m2 = 0; m2 < bl; ++m2) {
                Complex128 sum(0.0, 0.0);
                for (int p = 0; p < 2; ++p) {
                    for (int r1 = 0; r1 < br; ++r1) {
                        for (int r2 = 0; r2 < br; ++r2) {
                            sum += right_envs[q + 1][r1 * br + r2] *
                                   T(m1, p, r1) * T(m2, p, r2).conj();
                        }
                    }
                }
                new_env[m1 * bl + m2] = sum;
            }
        }
        right_envs[q] = std::move(new_env);
    }

    // === Sequential measurement with incremental left environment O(N·chi^3) total ===
    // left_env starts as 1x1 identity; updated after each projection.
    std::vector<Complex128> left_env = {Complex128(1.0, 0.0)};

    for (int q = 0; q < n_qubits; ++q) {
        auto& Tq = tensors[q];
        const int chi_left  = Tq.bond_left;
        const int chi_right = Tq.bond_right;

        // Compute P(0) and P(1) using left_env, Tq, and precomputed right_envs[q+1]
        std::vector<double> probs(2, 0.0);
        for (int p = 0; p < 2; ++p) {
            Complex128 sum(0.0, 0.0);
            for (int l1 = 0; l1 < chi_left; ++l1) {
                for (int l2 = 0; l2 < chi_left; ++l2) {
                    const Complex128 lv = left_env[l1 * chi_left + l2];
                    for (int r1 = 0; r1 < chi_right; ++r1) {
                        for (int r2 = 0; r2 < chi_right; ++r2) {
                            sum += lv * Tq(l1, p, r1) * Tq(l2, p, r2).conj() *
                                   right_envs[q + 1][r1 * chi_right + r2];
                        }
                    }
                }
            }
            probs[p] = sum.real;
        }

        double p0 = std::max(0.0, probs[0]);
        double p1 = std::max(0.0, probs[1]);
        double total = p0 + p1;
        if (total < 1e-30) { p0 = p1 = 0.5; total = 1.0; }
        p0 /= total;

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        int outcome = (dist(rng) < p0) ? 0 : 1;
        // Project bitstring convention: qubit 0 is the RIGHTMOST character
        // (matches Statevector::sample_counts and the per-shot paths).
        bits[n_qubits - 1 - q] = outcome ? '1' : '0';

        // Project: zero out the other physical index
        const int other = 1 - outcome;
        for (int l = 0; l < chi_left; ++l)
            for (int r = 0; r < chi_right; ++r)
                Tq(l, other, r) = Complex128(0.0, 0.0);

        // Renormalize
        const double prob_outcome = (outcome == 0) ? probs[0] : probs[1];
        if (prob_outcome > 1e-30) {
            const double inv_norm = 1.0 / std::sqrt(prob_outcome);
            for (int l = 0; l < chi_left; ++l)
                for (int r = 0; r < chi_right; ++r) {
                    auto& v = Tq(l, outcome, r);
                    v.real *= inv_norm;
                    v.imag *= inv_norm;
                }
        }

        // Incrementally update left environment O(chi^3):
        // left_env_new[m1,m2] = sum_{l1,l2} left_env[l1,l2] * Tq[l1,o,m1] * conj(Tq[l2,o,m2])
        // Only outcome physical index survives (other is zeroed above).
        std::vector<Complex128> new_left(chi_right * chi_right, Complex128(0.0, 0.0));
        for (int m1 = 0; m1 < chi_right; ++m1) {
            for (int m2 = 0; m2 < chi_right; ++m2) {
                Complex128 sum(0.0, 0.0);
                for (int l1 = 0; l1 < chi_left; ++l1) {
                    for (int l2 = 0; l2 < chi_left; ++l2) {
                        sum += left_env[l1 * chi_left + l2] *
                               Tq(l1, outcome, m1) * Tq(l2, outcome, m2).conj();
                    }
                }
                new_left[m1 * chi_right + m2] = sum;
            }
        }
        left_env = std::move(new_left);
    }

    return bits;
}

// =============================================================================
// to_statevector — full contraction for N <= 25 (used for small systems)
// =============================================================================

// Hard memory limit: 2^25 complex doubles ≈ 512 MB. Distinct from the
// performance crossover (MPS_SV_CROSSOVER) used in MPSSimulator::run.
static constexpr int MPS_SV_MAX_QUBITS = 25;

// Performance crossover: for N <= this value sequential MPS sampling
// (O(shots * N * chi^3)) is slower than full statevector sampling
// (O(N * chi^2 * 2^N)). Empirically ~18–20 for typical bond dimensions.
static constexpr int MPS_SV_CROSSOVER = 18;

Statevector MPSState::to_statevector() const {
    if (n_qubits > MPS_SV_MAX_QUBITS) {
        throw std::runtime_error("Too many qubits for full statevector conversion");
    }

    // Standard left-to-right site contraction: maintains a (dim_so_far x chi) matrix
    // that grows 2x per site.  O(N) allocations vs O(N * 2^N) for the per-basis-state loop.
    //
    // After site q: current[idx, r] = amplitude of basis state idx (0..2^(q+1)-1)
    //               with bond index r ∈ [0, bond_right[q]).
    //
    // Expansion step: new_current[idx*2 + p, r'] = sum_m current[idx, m] * T[m, p, r']
    int dim_so_far = 1;
    std::vector<Complex128> current(1, Complex128(1.0, 0.0));  // 1x1 identity

    for (int q = 0; q < n_qubits; ++q) {
        const auto& T  = tensors[q];
        const int bl   = T.bond_left;
        const int br   = T.bond_right;
        const int new_dim = dim_so_far * 2;

        std::vector<Complex128> next(new_dim * br, Complex128(0.0, 0.0));
        for (int idx = 0; idx < dim_so_far; ++idx) {
            for (int p = 0; p < 2; ++p) {
                const int new_row = idx * 2 + p;
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int m = 0; m < bl; ++m)
                        sum += current[idx * bl + m] * T(m, p, r);
                    next[new_row * br + r] = sum;
                }
            }
        }
        current = std::move(next);
        dim_so_far = new_dim;
    }

    // current now has shape (2^N x 1).
    // The left-to-right contraction places qubit 0 in the MSB position of each
    // index (new_row = idx*2 + p shifts previous bits left and appends p as LSB,
    // so the first qubit processed occupies the most-significant bit).
    //
    // The Statevector convention — shared by all gate implementations and
    // sample_counts — uses qubit q as bit q (LSB = qubit 0):
    //   index i  ↔  qubit q has value (i >> q) & 1
    //
    // Reconcile by bit-reversing each index when writing the output.
    Statevector sv(n_qubits);
    for (size_t idx = 0; idx < static_cast<size_t>(dim_so_far); ++idx) {
        // Reverse the N-bit representation of idx so that qubit 0 maps to bit 0.
        size_t rev = 0;
        for (int b = 0; b < n_qubits; ++b)
            rev |= ((idx >> b) & 1ULL) << (n_qubits - 1 - b);
        sv.real_parts[rev] = current[idx].real;
        sv.imag_parts[rev] = current[idx].imag;
    }
    return sv;
}

// =============================================================================
// mps_from_sv — reconstruct MPS from a statevector via sequential SVD
// Used as a fallback when a gate cannot be applied natively in MPS form.
// =============================================================================

static MPSState mps_from_sv(const Statevector& sv, int n, int max_bond_dim, double cutoff) {
    MPSState result(n, max_bond_dim, cutoff);
    size_t dim = 1ULL << n;

    int left_bond = 1;
    int right_cols = (int)dim;
    std::vector<Complex128> block(dim);
    // sv uses qubit q at bit q (LSB = qubit 0); the MPS sequential SVD expects
    // qubit 0 at the MSB of each index (site 0 = MSB).  Bit-reverse each index
    // so the two conventions are consistent — mirrors the reversal in to_statevector.
    for (size_t i = 0; i < dim; ++i) {
        size_t rev = 0;
        for (int b = 0; b < n; ++b)
            rev |= ((i >> b) & 1ULL) << (n - 1 - b);
        block[i] = {sv.real_parts[rev], sv.imag_parts[rev]};
    }

    for (int site = 0; site < n - 1; ++site) {
        int half_cols = right_cols / 2;
        int rows = left_bond * 2;

        // Reshape: M[(alpha*2 + p), c2] = block[alpha*right_cols + p*half_cols + c2]
        // p ∈ {0,1} is the physical index; c2 ∈ [0,half_cols) indexes remaining sites.
        Eigen::MatrixXcd M(rows, half_cols);
        for (int alpha = 0; alpha < left_bond; ++alpha)
            for (int p = 0; p < 2; ++p)
                for (int c2 = 0; c2 < half_cols; ++c2)
                    M(alpha * 2 + p, c2) = std::complex<double>(
                        block[alpha * right_cols + p * half_cols + c2].real,
                        block[alpha * right_cols + p * half_cols + c2].imag);

        // JacobiSVD for accuracy: this is the reconstruction fallback, and the
        // BDCSVD defect makes composite states unreliable.
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& svals = svd.singularValues();
        // Same discarded-weight rule as svd_truncate: `cutoff` is the fraction
        // of total weight truncation may drop, not a magnitude threshold. The
        // two construction paths have to agree, or a state's bond dimension
        // would depend on which one built it.
        const int nsv = (int)svals.size();
        double total = 0.0;
        for (int i = 0; i < nsv; ++i) total += svals(i) * svals(i);
        const double budget = cutoff * total;

        int k = nsv;
        double discarded = 0.0;
        while (k > 1) {
            const double w = svals(k - 1) * svals(k - 1);
            if (discarded + w > budget) break;
            discarded += w;
            --k;
        }
        k = std::min(k, max_bond_dim);

        result.tensors[site] = MPSTensor(left_bond, k);
        for (int alpha = 0; alpha < left_bond; ++alpha)
            for (int p = 0; p < 2; ++p)
                for (int r = 0; r < k; ++r)
                    result.tensors[site](alpha, p, r) = {
                        svd.matrixU()(alpha * 2 + p, r).real(),
                        svd.matrixU()(alpha * 2 + p, r).imag()};

        // New block = S * V†
        block.resize(k * half_cols);
        for (int r = 0; r < k; ++r)
            for (int c2 = 0; c2 < half_cols; ++c2) {
                auto v = svals(r) * std::conj(svd.matrixV()(c2, r));
                block[r * half_cols + c2] = {v.real(), v.imag()};
            }

        left_bond = k;
        right_cols = half_cols;
    }

    result.tensors[n - 1] = MPSTensor(left_bond, 1);
    for (int alpha = 0; alpha < left_bond; ++alpha)
        for (int p = 0; p < 2; ++p)
            result.tensors[n - 1](alpha, p, 0) = block[alpha * 2 + p];

    return result;
}

// =============================================================================
// MPSSimulator::run — build gate matrices analytically, not via statevector
// =============================================================================

// Helper: build 2x2 gate matrix analytically
static std::array<Complex128, 4> gate2x2(const Instruction& inst) {
    using GT = Instruction::GateType;
    const auto& p = inst.params;
    constexpr double inv_sqrt2 = INV_SQRT2;
    std::array<Complex128, 4> U{};

    switch (inst.type) {
        case GT::H:
            U[0] = U[1] = U[2] = Complex128(inv_sqrt2, 0);
            U[3] = Complex128(-inv_sqrt2, 0);
            break;
        case GT::X:  U[0]=U[3]=Complex128(0,0); U[1]=U[2]=Complex128(1,0); break;
        case GT::Y:  U[0]=U[3]=Complex128(0,0); U[1]=Complex128(0,-1); U[2]=Complex128(0,1); break;
        case GT::Z:  U[0]=Complex128(1,0); U[1]=U[2]=Complex128(0,0); U[3]=Complex128(-1,0); break;
        case GT::S:  U[0]=Complex128(1,0); U[3]=Complex128(0,1); break;
        case GT::SDG: U[0]=Complex128(1,0); U[3]=Complex128(0,-1); break;
        case GT::T:  U[0]=Complex128(1,0); U[3]=Complex128(inv_sqrt2, inv_sqrt2); break;
        case GT::TDG: U[0]=Complex128(1,0); U[3]=Complex128(inv_sqrt2, -inv_sqrt2); break;
        case GT::SX: {
            Complex128 h(0.5, 0.5);
            Complex128 hc(0.5, -0.5);
            U[0]=h; U[1]=hc; U[2]=hc; U[3]=h;
            break;
        }
        case GT::SXDG: {
            Complex128 h(0.5, -0.5);
            Complex128 hc(0.5, 0.5);
            U[0]=h; U[1]=hc; U[2]=hc; U[3]=h;
            break;
        }
        case GT::RX: {
            double c = std::cos(p[0]/2), s = std::sin(p[0]/2);
            U[0]=Complex128(c,0); U[1]=Complex128(0,-s);
            U[2]=Complex128(0,-s); U[3]=Complex128(c,0);
            break;
        }
        case GT::RY: {
            double c = std::cos(p[0]/2), s = std::sin(p[0]/2);
            U[0]=Complex128(c,0); U[1]=Complex128(-s,0);
            U[2]=Complex128(s,0); U[3]=Complex128(c,0);
            break;
        }
        case GT::RZ: case GT::P: {
            double angle = (inst.type == GT::RZ) ? p[0] : 0.0;
            double lambda = (inst.type == GT::P)  ? p[0] : 0.0;
            if (inst.type == GT::RZ) {
                U[0]=Complex128(std::cos(angle/2), -std::sin(angle/2));
                U[3]=Complex128(std::cos(angle/2),  std::sin(angle/2));
            } else {
                U[0]=Complex128(1,0);
                U[3]=Complex128(std::cos(lambda), std::sin(lambda));
            }
            break;
        }
        case GT::U: case GT::U3: {
            double th=p[0], ph=p[1], la=p[2];
            double c=std::cos(th/2), s=std::sin(th/2);
            U[0]=Complex128(c,0);
            U[1]=Complex128(-s*std::cos(la), -s*std::sin(la));
            U[2]=Complex128(s*std::cos(ph),   s*std::sin(ph));
            U[3]=Complex128(c*std::cos(ph+la), c*std::sin(ph+la));
            break;
        }
        case GT::U1: {
            U[0]=Complex128(1,0);
            U[3]=Complex128(std::cos(p[0]), std::sin(p[0]));
            break;
        }
        case GT::U2: {
            double ph=p[0], la=p[1];
            U[0]=Complex128(inv_sqrt2,0);
            U[1]=Complex128(-inv_sqrt2*std::cos(la), -inv_sqrt2*std::sin(la));
            U[2]=Complex128(inv_sqrt2*std::cos(ph),   inv_sqrt2*std::sin(ph));
            U[3]=Complex128(inv_sqrt2*std::cos(ph+la), inv_sqrt2*std::sin(ph+la));
            break;
        }
        default:
            // Identity fallback
            U[0] = U[3] = Complex128(1, 0);
            break;
    }
    return U;
}

// Helper: build 4x4 two-qubit gate matrix analytically
// U[po1*2+po2, pi1*2+pi2] in row-major
static std::array<Complex128, 16> gate4x4(const Instruction& inst) {
    using GT = Instruction::GateType;
    const auto& p = inst.params;
    std::array<Complex128, 16> U{};

    // Utility: set element
    auto set = [&](int r, int c, Complex128 v) { U[r*4+c] = v; };

    constexpr double inv_sqrt2 = INV_SQRT2;

    switch (inst.type) {
        case GT::CX:
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,3,{1,0}); set(3,2,{1,0}); break;
        case GT::CY:
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,3,{0,-1}); set(3,2,{0,1}); break;
        case GT::CZ:
            set(0,0,{1,0}); set(1,1,{1,0}); set(2,2,{1,0}); set(3,3,{-1,0}); break;
        case GT::CH: {
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{inv_sqrt2,0}); set(2,3,{inv_sqrt2,0});
            set(3,2,{inv_sqrt2,0}); set(3,3,{-inv_sqrt2,0}); break;
        }
        case GT::SWAP:
            set(0,0,{1,0}); set(1,2,{1,0}); set(2,1,{1,0}); set(3,3,{1,0}); break;
        case GT::ISWAP:
            set(0,0,{1,0}); set(1,2,{0,1}); set(2,1,{0,1}); set(3,3,{1,0}); break;
        case GT::CRX: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,0}); set(2,3,{0,-s});
            set(3,2,{0,-s}); set(3,3,{c,0}); break;
        }
        case GT::CRY: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,0}); set(2,3,{-s,0});
            set(3,2,{s,0}); set(3,3,{c,0}); break;
        }
        case GT::CRZ: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{c,-s}); set(3,3,{c,s}); break;
        }
        case GT::CP: {
            set(0,0,{1,0}); set(1,1,{1,0}); set(2,2,{1,0});
            set(3,3,{std::cos(p[0]),std::sin(p[0])}); break;
        }
        case GT::RXX: {
            // RXX = exp(-i theta/2 X⊗X)
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            U[0*4+0]=Complex128(c,0); U[0*4+3]=Complex128(0,-s);
            U[1*4+1]=Complex128(c,0); U[1*4+2]=Complex128(0,-s);
            U[2*4+1]=Complex128(0,-s); U[2*4+2]=Complex128(c,0);
            U[3*4+0]=Complex128(0,-s); U[3*4+3]=Complex128(c,0);
            break;
        }
        case GT::RYY: {
            // exp(-i t/2 Y(x)Y): cos on the full diagonal, +i*sin outer /
            // -i*sin inner anti-diagonal. Matches gates::apply_ryy.
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            for (auto& x : U) x = Complex128(0,0);
            U[0*4+0]=Complex128(c,0); U[1*4+1]=Complex128(c,0);
            U[2*4+2]=Complex128(c,0); U[3*4+3]=Complex128(c,0);
            U[0*4+3]=Complex128(0,s); U[1*4+2]=Complex128(0,-s);
            U[2*4+1]=Complex128(0,-s); U[3*4+0]=Complex128(0,s);
            break;
        }
        case GT::RZZ: {
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            for (auto& x : U) x = Complex128(0,0);
            U[0*4+0]=Complex128(c,-s); U[1*4+1]=Complex128(c,s);
            U[2*4+2]=Complex128(c,s); U[3*4+3]=Complex128(c,-s);
            break;
        }
        case GT::ECR: {
            // ECR = (1/sqrt(2)) * [[0,0,1,i],[0,0,i,1],[1,-i,0,0],[-i,1,0,0]]
            Complex128 s(inv_sqrt2, 0);
            Complex128 si(0, inv_sqrt2);
            for (auto& x : U) x = Complex128(0,0);
            U[0*4+2]=s; U[0*4+3]=si;
            U[1*4+2]=si; U[1*4+3]=s;
            U[2*4+0]=s; U[2*4+1]={0,-inv_sqrt2};
            U[3*4+0]={0,-inv_sqrt2}; U[3*4+1]=s;
            break;
        }
        case GT::RZX: {
            // exp(-i t/2 Z(x)X), Z on the first qubit (MSB of the pair label),
            // X on the second (LSB). Rows 0,1 (Z=+1) couple with -i*sin; rows
            // 2,3 (Z=-1) couple with +i*sin. Matches gates::apply_rzx.
            double c=std::cos(p[0]/2), s=std::sin(p[0]/2);
            for (auto& x : U) x = Complex128(0,0);
            U[0*4+0]=Complex128(c,0); U[0*4+1]=Complex128(0,-s);
            U[1*4+0]=Complex128(0,-s); U[1*4+1]=Complex128(c,0);
            U[2*4+2]=Complex128(c,0); U[2*4+3]=Complex128(0,s);
            U[3*4+2]=Complex128(0,s); U[3*4+3]=Complex128(c,0);
            break;
        }
        case GT::CU: {
            double th=p[0], ph=p[1], la=p[2], ga=p[3];
            double c=std::cos(th/2), s=std::sin(th/2);
            set(0,0,{1,0}); set(1,1,{1,0});
            set(2,2,{std::cos(ga)*c, std::sin(ga)*c});
            set(2,3,{-std::cos(ga+la)*s, -std::sin(ga+la)*s});
            set(3,2,{std::cos(ga+ph)*s, std::sin(ga+ph)*s});
            set(3,3,{std::cos(ga+ph+la)*c, std::sin(ga+ph+la)*c});
            break;
        }
        default:
            // Identity
            U[0*4+0]=U[1*4+1]=U[2*4+2]=U[3*4+3]=Complex128(1,0);
            break;
    }
    return U;
}

// Helper: apply one instruction to an MPS state.
// Handles RESET, all gate types. MEASURE and BARRIER must NOT be passed here.
static void mps_apply_instruction(MPSState& mps, const Instruction& inst,
                                  std::mt19937_64& rng) {
    using GT = Instruction::GateType;

    if (inst.type == GT::RESET) {
        int qubit = inst.qubits[0];
        // probabilities_single returns RAW marginals <psi|P_k|psi> (they sum
        // to the state's norm^2, not necessarily 1); normalise for sampling.
        auto probs = mps.probabilities_single(qubit);
        const double p0_raw = std::max(0.0, probs[0]);
        const double p1_raw = std::max(0.0, probs[1]);
        double total = p0_raw + p1_raw;
        if (total < 1e-30) total = 1.0;
        std::uniform_real_distribution<double> udist(0.0, 1.0);
        int outcome = (udist(rng) < p0_raw / total) ? 0 : 1;
        int keep = outcome, zero_phys = 1 - outcome;
        auto& T = mps.tensors[qubit];
        for (int l = 0; l < T.bond_left; ++l)
            for (int r = 0; r < T.bond_right; ++r)
                T(l, zero_phys, r) = Complex128(0.0, 0.0);
        // Renormalise by the environment-contracted marginal of the sampled
        // outcome: the projected state's global norm^2 equals that marginal.
        // The local Frobenius norm is only valid in canonical form, which gate
        // application does not maintain.
        const double p_raw = (outcome == 0) ? p0_raw : p1_raw;
        const double inv_norm = (p_raw > 1e-30) ? 1.0 / std::sqrt(p_raw) : 1.0;
        for (int l = 0; l < T.bond_left; ++l)
            for (int r = 0; r < T.bond_right; ++r) {
                T(l, keep, r).real *= inv_norm;
                T(l, keep, r).imag *= inv_norm;
            }
        if (outcome == 1) {
            // Flipping the collapsed qubit to |1>. X is built here, so it
            // carries Ignore like the other locally-built factors.
            const std::array<Complex128, 4> X_g = {
                Complex128(0,0), Complex128(1,0),
                Complex128(1,0), Complex128(0,0)
            };
            mps.apply_single_qubit_gate(X_g, qubit, {Validation::Ignore});
        }
        return;
    }

    if (inst.type == GT::PARAM_RX || inst.type == GT::PARAM_RY ||
        inst.type == GT::PARAM_RZ || inst.type == GT::PARAM_P ||
        inst.type == GT::PARAM_U)
        throw std::runtime_error("Unresolved parameterised gate in MPS simulation");

    // Multi-controlled X reduces to X/CX/CCX for <= 2 controls (native MPS
    // path). Wider MCX and the MCP/PERMUTATION structured ops have no compact
    // MPS form, so they take the same bounded statevector fallback as a 3+ qubit
    // UNITARY (to_statevector -> apply -> mps_from_sv). This keeps e.g. Shor's
    // PERMUTATION oracle runnable on the MPS backend, as its dense UNITARY was.
    if (inst.type == GT::MCX && inst.qubits.size() <= 3) {
        Instruction sub = inst;
        const size_t nq = inst.qubits.size();
        sub.type = (nq == 1) ? GT::X : (nq == 2) ? GT::CX : GT::CCX;
        mps_apply_instruction(mps, sub, rng);
        return;
    }
    if (inst.type == GT::MCX || inst.type == GT::MCP ||
        inst.type == GT::PERMUTATION) {
        if (mps.n_qubits > MPS_SV_MAX_QUBITS) {
            throw std::runtime_error(
                "MPS simulator: " + inst.gate_name() + " on n_qubits=" +
                std::to_string(mps.n_qubits) + " exceeds the statevector-"
                "fallback limit (" + std::to_string(MPS_SV_MAX_QUBITS) +
                "); decompose to 1/2-qubit gates or use the statevector/"
                "density-matrix backend");
        }
        auto sv = mps.to_statevector();
        if (inst.type == GT::MCX) {
            std::vector<int> controls(inst.qubits.begin(), inst.qubits.end() - 1);
            gates::apply_mcx(sv, controls, inst.qubits.back());
        } else if (inst.type == GT::MCP) {
            gates::apply_mcp(sv, inst.qubits, inst.params[0]);
        } else {
            gates::apply_permutation(sv, inst.qubits, inst.permutation);
        }
        mps = mps_from_sv(sv, mps.n_qubits, mps.max_bond_dim, mps.cutoff);
        return;
    }

    // UNITARY gates store the matrix directly in inst.matrix.
    //
    // 1-qubit and 2-qubit UNITARYs route to MPSState's direct
    // apply_single_qubit_gate / apply_two_qubit_gate, which contract the
    // matrix into the affected site tensors (with truncated SVD for the
    // 2-qubit case, and a SWAP network for non-adjacent qubit pairs). Memory
    // cost stays bounded by the bond dimension and is independent of
    // n_qubits — so MPS circuits with arbitrary register widths can now
    // contain user-supplied 1q/2q unitaries.
    //
    // 3+ qubit UNITARYs fall back to the full statevector path. That path
    // is bounded by MPS_SV_MAX_QUBITS (= 25) inside to_statevector(); for
    // wider circuits with a multi-qubit UNITARY we raise a clearer error
    // naming the gate and qubit count rather than letting the generic
    // "Too many qubits for full statevector conversion" message surface
    // from a call site far from the offending instruction.
    if (inst.type == GT::UNITARY) {
        if (inst.qubits.size() == 1) {
            if (inst.matrix.size() != 4)
                throw std::runtime_error("MPS UNITARY: 1-qubit matrix must have 4 entries");
            std::array<Complex128, 4> U{
                inst.matrix[0], inst.matrix[1],
                inst.matrix[2], inst.matrix[3]
            };
            mps.apply_single_qubit_gate(U, inst.qubits[0],
                                        {Validation::Ignore});
            return;
        }
        if (inst.qubits.size() == 2) {
            if (inst.matrix.size() != 16)
                throw std::runtime_error("MPS UNITARY: 2-qubit matrix must have 16 entries");
            // Convention bridge. apply_unitary indexes its matrix with
            // bit 0 (LSB) = targets[0] state, bit 1 = targets[1] state.
            // MPSState::apply_two_qubit_gate indexes its 4x4 with
            // bit 1 (MSB) = first qubit arg, bit 0 = second qubit arg
            // (matching the SV/DM 2q-gate convention used elsewhere in MPS).
            // Swap bits 0 and 1 of both row and column to translate.
            auto swap01 = [](int idx) {
                return ((idx & 1) << 1) | ((idx >> 1) & 1);
            };
            std::array<Complex128, 16> U{};
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    U[r * 4 + c] = inst.matrix[swap01(r) * 4 + swap01(c)];
                }
            }
            mps.apply_two_qubit_gate(U, inst.qubits[0], inst.qubits[1],
                                     {Validation::Ignore});
            return;
        }
        if (mps.n_qubits > MPS_SV_MAX_QUBITS) {
            throw std::runtime_error(
                "MPS UNITARY: cannot apply a " +
                std::to_string(inst.qubits.size()) +
                "-qubit UNITARY on an MPS state with n_qubits=" +
                std::to_string(mps.n_qubits) + " (limit " +
                std::to_string(MPS_SV_MAX_QUBITS) +
                " for the full-statevector fallback path). Decompose the "
                "unitary into 1- and 2-qubit factors and apply them via the "
                "direct MPS tensor-contraction path.");
        }
        auto sv = mps.to_statevector();
        gates::apply_unitary(sv, inst.qubits, inst.matrix, {Validation::Ignore});
        mps = mps_from_sv(sv, mps.n_qubits, mps.max_bond_dim, mps.cutoff);
        return;
    }

    if (inst.qubits.size() == 1) {
        auto U = gate2x2(inst);
        mps.apply_single_qubit_gate(U, inst.qubits[0], {Validation::Ignore});

    } else if (inst.qubits.size() == 2) {
        auto U = gate4x4(inst);
        mps.apply_two_qubit_gate(U, inst.qubits[0], inst.qubits[1],
                                 {Validation::Ignore});

    } else if (inst.qubits.size() == 3) {
        int q0 = inst.qubits[0], q1 = inst.qubits[1], q2 = inst.qubits[2];

        // The factors below are built here rather than supplied, so every
        // application of them passes Ignore: their unitarity is a property of
        // this file, and checking it once per gate per shot would measure the
        // same four constants for the life of the run.
        constexpr double s2 = INV_SQRT2;
        const std::array<Complex128, 4> H_g = {
            Complex128(s2,0), Complex128(s2,0),
            Complex128(s2,0), Complex128(-s2,0)
        };
        const std::array<Complex128, 4> T_g = {
            Complex128(1,0), Complex128(0,0),
            Complex128(0,0), Complex128(s2, s2)
        };
        const std::array<Complex128, 4> Tdg_g = {
            Complex128(1,0), Complex128(0,0),
            Complex128(0,0), Complex128(s2, -s2)
        };
        std::array<Complex128, 16> CX_g{};
        CX_g[0] = CX_g[5] = CX_g[11] = CX_g[14] = Complex128(1,0);

        // CCX decomposition: standard 6-CNOT Toffoli
        auto apply_ccx = [&](int c1, int c2, int tgt) {
            mps.apply_single_qubit_gate(H_g, tgt, {Validation::Ignore});
            mps.apply_two_qubit_gate(CX_g, c2, tgt, {Validation::Ignore});
            mps.apply_single_qubit_gate(Tdg_g, tgt, {Validation::Ignore});
            mps.apply_two_qubit_gate(CX_g, c1, tgt, {Validation::Ignore});
            mps.apply_single_qubit_gate(T_g, tgt, {Validation::Ignore});
            mps.apply_two_qubit_gate(CX_g, c2, tgt, {Validation::Ignore});
            mps.apply_single_qubit_gate(Tdg_g, tgt, {Validation::Ignore});
            mps.apply_two_qubit_gate(CX_g, c1, tgt, {Validation::Ignore});
            mps.apply_single_qubit_gate(T_g, c2, {Validation::Ignore});
            mps.apply_single_qubit_gate(T_g, tgt, {Validation::Ignore});
            mps.apply_single_qubit_gate(H_g, tgt, {Validation::Ignore});
            mps.apply_two_qubit_gate(CX_g, c1, c2, {Validation::Ignore});
            mps.apply_single_qubit_gate(T_g, c1, {Validation::Ignore});
            mps.apply_single_qubit_gate(Tdg_g, c2, {Validation::Ignore});
            mps.apply_two_qubit_gate(CX_g, c1, c2, {Validation::Ignore});
        };

        switch (inst.type) {
            case GT::CCX:
                apply_ccx(q0, q1, q2);
                break;
            case GT::CCZ:
                mps.apply_single_qubit_gate(H_g, q2, {Validation::Ignore});
                apply_ccx(q0, q1, q2);
                mps.apply_single_qubit_gate(H_g, q2, {Validation::Ignore});
                break;
            case GT::CSWAP:
                mps.apply_two_qubit_gate(CX_g, q2, q1, {Validation::Ignore});
                apply_ccx(q0, q1, q2);
                mps.apply_two_qubit_gate(CX_g, q2, q1, {Validation::Ignore});
                break;
            case GT::RCCX:
                mps.apply_single_qubit_gate(H_g, q2, {Validation::Ignore});
                mps.apply_single_qubit_gate(T_g, q2, {Validation::Ignore});
                mps.apply_two_qubit_gate(CX_g, q1, q2, {Validation::Ignore});
                mps.apply_single_qubit_gate(Tdg_g, q2, {Validation::Ignore});
                mps.apply_two_qubit_gate(CX_g, q0, q2, {Validation::Ignore});
                mps.apply_single_qubit_gate(T_g, q2, {Validation::Ignore});
                mps.apply_two_qubit_gate(CX_g, q1, q2, {Validation::Ignore});
                mps.apply_single_qubit_gate(Tdg_g, q2, {Validation::Ignore});
                mps.apply_single_qubit_gate(H_g, q2, {Validation::Ignore});
                break;
            case GT::UNITARY: {
                auto sv = mps.to_statevector();
                gates::apply_unitary(sv, inst.qubits, inst.matrix,
                                     {Validation::Ignore});
                mps = mps_from_sv(sv, mps.n_qubits, mps.max_bond_dim, mps.cutoff);
                break;
            }
            default:
                throw std::runtime_error(
                    "MPS simulator: unsupported 3-qubit gate type " +
                    std::to_string(static_cast<int>(inst.type)));
        }
    } else {
        throw std::runtime_error(
            "MPS simulator: unsupported " + std::to_string(inst.qubits.size()) +
            "-qubit gate");
    }
}

// True when no instruction (other than BARRIER) acts on a qubit after that
// qubit has been measured. The pre-measurement state is then deterministic and
// outcomes can be sampled from a single forward pass instead of per-shot
// trajectories. A second MEASURE or a RESET on a measured qubit also counts as
// "acting on it" and forces the per-shot path.
static bool mps_measures_are_terminal(const QuantumCircuit& circuit) {
    std::vector<bool> measured(static_cast<size_t>(circuit.n_qubits), false);
    for (const auto& inst : circuit.instructions) {
        if (inst.type == Instruction::GateType::BARRIER) continue;
        for (int q : inst.qubits)
            if (q >= 0 && q < circuit.n_qubits &&
                measured[static_cast<size_t>(q)])
                return false;
        if (inst.type == Instruction::GateType::MEASURE)
            measured[static_cast<size_t>(inst.qubits[0])] = true;
    }
    return true;
}

// =============================================================================
// Hoisted read-only sequential sampling
//
// The right environments E_q (transfer-operator contraction of sites q..N-1)
// are INVARIANT across shots and across the left-to-right projection, so they
// are computed once. Each shot then samples read-only: the left environment is
// carried incrementally and each site's outcome slice is read directly from the
// (unmodified) tensors, scaled by 1/p_outcome — no per-shot MPS copy and no
// environment rebuild. Every contraction is the two-stage O(chi^3) form (the
// previous per-shot measure_sequential rebuilt O(N chi^3) environments AND
// deep-copied the whole MPS per shot; its 4-index contractions were O(chi^4)).
// =============================================================================

// right_envs[q] = E_q sized (bond_left[q] x bond_left[q]); right_envs[N] = [[1]].
static std::vector<std::vector<Complex128>> mps_right_envs(
    const std::vector<MPSTensor>& tensors, int n
) {
    std::vector<std::vector<Complex128>> envs(static_cast<size_t>(n + 1));
    envs[static_cast<size_t>(n)] = {Complex128(1.0, 0.0)};

    for (int q = n - 1; q >= 0; --q) {
        const auto& T = tensors[static_cast<size_t>(q)];
        const int bl = T.bond_left, br = T.bond_right;
        const auto& En = envs[static_cast<size_t>(q + 1)];  // br x br

        std::vector<Complex128> out(static_cast<size_t>(bl) * bl, Complex128(0.0, 0.0));
        // For each physical index p (two-stage O(chi^3)):
        //   Y[m1, r2] = sum_r1 T[m1,p,r1] * En[r1,r2]
        //   out[m1,m2] += sum_r2 Y[m1,r2] * conj(T[m2,p,r2])
        std::vector<Complex128> Y(static_cast<size_t>(bl) * br);
        for (int p = 0; p < 2; ++p) {
            for (int m1 = 0; m1 < bl; ++m1)
                for (int r2 = 0; r2 < br; ++r2) {
                    Complex128 acc(0.0, 0.0);
                    for (int r1 = 0; r1 < br; ++r1)
                        acc += T(m1, p, r1) * En[static_cast<size_t>(r1) * br + r2];
                    Y[static_cast<size_t>(m1) * br + r2] = acc;
                }
            for (int m1 = 0; m1 < bl; ++m1)
                for (int m2 = 0; m2 < bl; ++m2) {
                    Complex128 acc(0.0, 0.0);
                    for (int r2 = 0; r2 < br; ++r2)
                        acc += Y[static_cast<size_t>(m1) * br + r2] * T(m2, p, r2).conj();
                    out[static_cast<size_t>(m1) * bl + m2] += acc;
                }
        }
        envs[static_cast<size_t>(q)] = std::move(out);
    }
    return envs;
}

// Sample one full bitstring read-only, using precomputed right environments.
// Convention: qubit 0 is the RIGHTMOST character (matches measure_sequential
// and the statevector sampling paths).
static std::string mps_sample(
    const std::vector<MPSTensor>& tensors, int n,
    const std::vector<std::vector<Complex128>>& right_envs,
    std::mt19937_64& rng
) {
    std::string bits(static_cast<size_t>(n), '0');
    std::vector<Complex128> left = {Complex128(1.0, 0.0)};  // 1x1
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int q = 0; q < n; ++q) {
        const auto& T = tensors[static_cast<size_t>(q)];
        const int cl = T.bond_left, cr = T.bond_right;
        const auto& R = right_envs[static_cast<size_t>(q + 1)];  // cr x cr

        double probs[2] = {0.0, 0.0};
        for (int p = 0; p < 2; ++p) {
            // A[l1,r2] = sum_r1 T[l1,p,r1] R[r1,r2]
            std::vector<Complex128> A(static_cast<size_t>(cl) * cr);
            for (int l1 = 0; l1 < cl; ++l1)
                for (int r2 = 0; r2 < cr; ++r2) {
                    Complex128 acc(0.0, 0.0);
                    for (int r1 = 0; r1 < cr; ++r1)
                        acc += T(l1, p, r1) * R[static_cast<size_t>(r1) * cr + r2];
                    A[static_cast<size_t>(l1) * cr + r2] = acc;
                }
            // B[l1,l2] = sum_r2 A[l1,r2] conj(T[l2,p,r2]);  prob = Re sum left[l1,l2] B[l1,l2]
            Complex128 sum(0.0, 0.0);
            for (int l1 = 0; l1 < cl; ++l1)
                for (int l2 = 0; l2 < cl; ++l2) {
                    Complex128 b(0.0, 0.0);
                    for (int r2 = 0; r2 < cr; ++r2)
                        b += A[static_cast<size_t>(l1) * cr + r2] * T(l2, p, r2).conj();
                    sum += left[static_cast<size_t>(l1) * cl + l2] * b;
                }
            probs[p] = sum.real;
        }

        double p0 = std::max(0.0, probs[0]);
        double p1 = std::max(0.0, probs[1]);
        double total = p0 + p1;
        if (total < 1e-30) { p0 = p1 = 0.5; total = 1.0; }
        const int outcome = (dist(rng) < p0 / total) ? 0 : 1;
        bits[static_cast<size_t>(n - 1 - q)] = outcome ? '1' : '0';

        const double p_out = (outcome == 0) ? probs[0] : probs[1];
        const double inv_p = (p_out > 1e-30) ? 1.0 / p_out : 1.0;

        // new_left[m1,m2] = (1/p_out) sum_{l1,l2} left[l1,l2] T[l1,out,m1] conj(T[l2,out,m2])
        //   C[l2,m1] = sum_l1 left[l1,l2] T[l1,out,m1]
        //   new_left[m1,m2] = sum_l2 C[l2,m1] conj(T[l2,out,m2]) * inv_p
        std::vector<Complex128> C(static_cast<size_t>(cl) * cr, Complex128(0.0, 0.0));
        for (int l2 = 0; l2 < cl; ++l2)
            for (int m1 = 0; m1 < cr; ++m1) {
                Complex128 acc(0.0, 0.0);
                for (int l1 = 0; l1 < cl; ++l1)
                    acc += left[static_cast<size_t>(l1) * cl + l2] * T(l1, outcome, m1);
                C[static_cast<size_t>(l2) * cr + m1] = acc;
            }
        std::vector<Complex128> new_left(static_cast<size_t>(cr) * cr, Complex128(0.0, 0.0));
        for (int m1 = 0; m1 < cr; ++m1)
            for (int m2 = 0; m2 < cr; ++m2) {
                Complex128 acc(0.0, 0.0);
                for (int l2 = 0; l2 < cl; ++l2)
                    acc += C[static_cast<size_t>(l2) * cr + m1] * T(l2, outcome, m2).conj();
                new_left[static_cast<size_t>(m1) * cr + m2] = acc * inv_p;
            }
        left = std::move(new_left);
    }
    return bits;
}

MPSSimulator::Result MPSSimulator::run(
    const QuantumCircuit& circuit, int max_bond_dim,
    int shots, uint64_t seed
) {
    ScopedWarningFlush flush_on_exit;
    // Checked here as well as in the MPSState constructor so the message names
    // this call. The argument order differs from StatevectorSimulator::run
    // (circuit, shots, seed), so run(qc, 0, 0) meaning shots is a live way to
    // arrive here.
    detail::check_require(max_bond_dim >= 1, "MPSSimulator::run",
                          "max_bond_dim must be >= 1 (got " +
                              std::to_string(max_bond_dim) + ")");
    Result result(circuit.n_qubits);
    result.final_state = MPSState(circuit.n_qubits, max_bond_dim);

    // Pre-flight: reject any out-of-range operand index up front (this backend
    // surfaces errors by throwing, consistent with its other run() guards).
    circuit.validate_operands();
    circuit.validate_physical();

    auto t_start = std::chrono::high_resolution_clock::now();
    std::mt19937_64 rng(seed == 0 ? static_cast<uint64_t>(std::random_device{}()) : seed);

    // Execution strategy (see docs/api/simulators.md, Execution semantics):
    //   1. Terminal-only measurements (no feedforward, nothing acting on a
    //      qubit after it was measured): ONE forward pass, then sample
    //      outcomes from the final state with the qubit -> clbit mapping.
    //   2. Mid-circuit measurement or feedforward with shots > 0: per-shot
    //      trajectories (each stochastic collapse drawn independently).
    //   3. shots == 0: a single seeded trajectory; classical conditions are
    //      honoured and MEASURE outcomes recorded along the way.
    bool has_measure = false;
    bool has_condition = false;
    int n_clbits = circuit.n_clbits > 0 ? circuit.n_clbits : circuit.n_qubits;
    for (const auto& inst : circuit.instructions) {
        if (inst.type == Instruction::GateType::MEASURE) has_measure = true;
        if (inst.condition_clbit >= 0) has_condition = true;
    }
    const bool terminal_only =
        has_measure && !has_condition && mps_measures_are_terminal(circuit);

    // Collapse one qubit to a sampled outcome. probabilities_single returns
    // RAW marginals <psi|P_k|psi>; sampling normalises by their sum, and the
    // post-projection state is renormalised by the sampled outcome's marginal
    // (the projected global norm^2), NOT by the local tensor Frobenius norm,
    // which is only valid in canonical form.
    auto collapse_qubit = [&](MPSState& state, int qubit) -> int {
        auto probs = state.probabilities_single(qubit);
        const double p0_raw = std::max(0.0, probs[0]);
        const double p1_raw = std::max(0.0, probs[1]);
        double total = p0_raw + p1_raw;
        if (total < 1e-30) total = 1.0;
        std::uniform_real_distribution<double> udist(0.0, 1.0);
        const int outcome = (udist(rng) < p0_raw / total) ? 0 : 1;
        auto& T = state.tensors[qubit];
        const int other = 1 - outcome;
        for (int l = 0; l < T.bond_left; ++l)
            for (int r = 0; r < T.bond_right; ++r)
                T(l, other, r) = Complex128(0.0, 0.0);
        const double p_raw = (outcome == 0) ? p0_raw : p1_raw;
        const double inv_norm = (p_raw > 1e-30) ? 1.0 / std::sqrt(p_raw) : 1.0;
        for (int l = 0; l < T.bond_left; ++l)
            for (int r = 0; r < T.bond_right; ++r) {
                T(l, outcome, r).real *= inv_norm;
                T(l, outcome, r).imag *= inv_norm;
            }
        return outcome;
    };

    // One trajectory: honours classical conditions, records MEASURE outcomes.
    auto run_trajectory = [&](MPSState& state, std::vector<int>& clreg) {
        for (const auto& inst : circuit.instructions) {
            using GT = Instruction::GateType;
            if (inst.type == GT::BARRIER) continue;
            if (inst.condition_clbit >= 0) {
                int cv = (inst.condition_clbit < n_clbits)
                         ? clreg[inst.condition_clbit] : 0;
                if (cv != inst.condition_value) continue;
            }
            if (inst.type == GT::MEASURE) {
                const int qubit = inst.qubits[0];
                const int clbit = inst.clbits.empty() ? -1 : inst.clbits[0];
                const int outcome = collapse_qubit(state, qubit);
                if (clbit >= 0 && clbit < n_clbits) clreg[clbit] = outcome;
                continue;
            }
            mps_apply_instruction(state, inst, rng);
        }
    };

    std::vector<int> clreg(n_clbits, 0);

    if (shots > 0 && has_measure && !terminal_only) {
        // Per-shot trajectories: re-initialise the MPS to |0...0⟩ and
        // re-simulate for every shot so that each MEASURE collapses the state
        // independently (required for mid-circuit measurement / feedforward).
        result.counts.clear();
        for (int shot = 0; shot < shots; ++shot) {
            result.final_state = MPSState(circuit.n_qubits, max_bond_dim);
            clreg.assign(n_clbits, 0);
            run_trajectory(result.final_state, clreg);

            // Build bitstring: clbit 0 is LSB (rightmost), highest clbit is MSB.
            std::string bits(n_clbits, '0');
            for (int c = 0; c < n_clbits; ++c) {
                if (clreg[c]) bits[n_clbits - 1 - c] = '1';
            }
            result.counts[bits]++;
        }
    } else {
        if (shots == 0) {
            // Single seeded trajectory (collapses measures, honours
            // conditions); final_state is one reproducible trajectory.
            run_trajectory(result.final_state, clreg);
        } else {
            // Terminal-only measurements (or none): one forward pass with
            // MEASURE skipped; outcomes are sampled from the final state.
            // Conditions can only reference initial-zero clbits here.
            for (const auto& inst : circuit.instructions) {
                using GT = Instruction::GateType;
                if (inst.type == GT::BARRIER || inst.type == GT::MEASURE) continue;
                if (inst.condition_clbit >= 0) {
                    int cv = (inst.condition_clbit < n_clbits)
                             ? clreg[inst.condition_clbit] : 0;
                    if (cv != inst.condition_value) continue;
                }
                mps_apply_instruction(result.final_state, inst, rng);
            }
        }

        if (shots > 0) {
            // qubit -> clbit map of the terminal measurements. Empty when the
            // circuit has no MEASURE: sample the full register, qubit-indexed.
            std::vector<std::pair<int, int>> meas;
            for (const auto& inst : circuit.instructions)
                if (inst.type == Instruction::GateType::MEASURE)
                    meas.emplace_back(inst.qubits[0],
                                      inst.clbits.empty() ? inst.qubits[0]
                                                          : inst.clbits[0]);

            const int nq = circuit.n_qubits;
            auto record = [&](const std::string& qubit_bits, int count) {
                // qubit_bits: full register, qubit q at position nq-1-q.
                if (meas.empty()) {
                    result.counts[qubit_bits] += count;
                    return;
                }
                std::string key(n_clbits, '0');
                for (const auto& [q, c] : meas) {
                    if (c < 0 || c >= n_clbits) continue;
                    if (qubit_bits[nq - 1 - q] == '1') key[n_clbits - 1 - c] = '1';
                }
                result.counts[key] += count;
            };

            // Use full statevector contraction for small N (MPS_SV_CROSSOVER)
            // where it outperforms sequential MPS sampling. MPS_SV_MAX_QUBITS
            // is the hard memory limit for to_statevector() and is
            // intentionally larger than the crossover.
            const bool use_sv = circuit.n_qubits <= MPS_SV_CROSSOVER;
            if (use_sv) {
                auto sv = result.final_state.to_statevector();
                auto raw = sv.sample_counts(shots, seed);
                for (const auto& [bits, cnt] : raw) record(bits, cnt);
            } else {
                // Sequential MPS measurement: right environments are
                // shot-invariant, so compute them ONCE and sample each shot
                // read-only (no per-shot MPS copy, no environment rebuild).
                // O(N * chi^3) per shot.
                const auto right_envs =
                    mps_right_envs(result.final_state.tensors, circuit.n_qubits);
                for (int s = 0; s < shots; ++s) {
                    record(mps_sample(result.final_state.tensors,
                                      circuit.n_qubits, right_envs, rng), 1);
                }
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.simulation_time_seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    return result;
}

} // namespace lindblad
