// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// mps_sim.cpp — Matrix Product State simulator
// SVD truncation defaults to BDC, which is divide-and-conquer and pulls away
// from Jacobi as the block grows; Jacobi is selectable and notes once that it
// is slower. Neither is trusted on its word: every factorisation goes through
// the SELECT -> VERIFY -> FALLBACK -> THROW ladder in svd_truncate, which
// rebuilds the block from the kept slice and compares it against the input, so
// a factorisation that is finite but wrong is rejected rather than propagated.
// That ladder exists because degenerate rank-deficient blocks have produced
// NaN inside null-space singular vectors and, worse, finite-but-wrong kept
// vectors carrying no marker at all. Reproducers: tests/diag_r1160_matrices.hpp.
// Non-adjacent two-qubit gates are handled via SWAP chains (correct MPS-native approach).
// The chain tracks its open span (see mps_sim.hpp) and moves its orthogonality
// centre by QR and LQ steps, so marginals, collapse, the norm and sampling read
// the centre locally instead of contracting environments over the whole chain.

#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/detail/validate.hpp"
#include "lindblad/detail/failure_collector.hpp"
#include "lindblad/detail/memory_budget.hpp"
#include "lindblad/detail/preflight.hpp"
#include "lindblad/detail/report.hpp"
#include "lindblad/detail/trivial_resets.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/detail/svd_truncate.hpp"
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/detail/theta_harvest.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/hw_info.hpp"

#include <optional>
#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace lindblad {

// Emit a one-time note when either Jacobi kernel is selected. Both are
// correct, and both are the slower algorithm as the block grows: autonne's
// Jacobi by 2.7x against BDC on a 128x128 decaying spectrum, Eigen's by 19x. A
// caller selecting one deliberately is entitled to, and is told the cost once
// rather than on every split.
//
// The latch is per layer rather than per process: the qudit MPS carries its own
// so that selecting Jacobi there is reported even when a qubit simulation in
// the same process already noted it. Two layers, two things a caller needs
// told.
static bool is_jacobi(SVDMethod m) {
    return m == SVDMethod::Jacobi || m == SVDMethod::EigenJacobi;
}

static void warn_jacobi_slower_once(SVDMethod m) {
    static std::atomic<bool> warned{false};
    if (warned.exchange(true)) return;
    emit_warning(
        std::string("note: SVDMethod::") + to_string(m) +
        " selected for the qubit MPS. BDC is the default (autonne divide and conquer) "
        "and is faster as the block grows and the spectrum "
        "decays: measured on a 128x128 decaying spectrum, 2.7x over Jacobi "
        "and 19x over EigenJacobi. Jacobi resolves the tail of a "
        "graded spectrum with relative accuracy, which is why it remains "
        "selectable; select it for that, not for speed.");
}

// The one other kernel that is not the default. Eigen's BDCSVD costs about what
// autonne's does and is kept so a caller can hold the two providers against each
// other, so its note says only that it is in force, once per layer.
static void note_eigen_bdc_once() {
    static std::atomic<bool> noted{false};
    if (noted.exchange(true)) return;
    emit_warning("note: SVDMethod::EigenBDC selected for the qubit MPS, not the "
                 "default (BDC): Eigen's divide and conquer runs every bond split. "
                 "Be alert to it.");
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
    tensors_.resize(n_qubits);
    for (int i = 0; i < n_qubits; ++i) {
        tensors_[i] = MPSTensor(1, 1);
        tensors_[i](0, 0, 0) = Complex128(1.0, 0.0);  // |0⟩ amplitude
        tensors_[i](0, 1, 0) = Complex128(0.0, 0.0);  // |1⟩ amplitude
    }
    // A bond-1 unit vector is orthonormal both ways, so any single site could
    // be named the centre of |0...0>; site 0 is where sampling starts.
    span_lo = 0;
    span_hi = n_qubits > 0 ? 0 : -1;
}

// =============================================================================
// set_tensors / canonicalize - replacing the chain and moving its centre
// =============================================================================

void MPSState::set_tensors(std::vector<MPSTensor> sites) {
    const char* ctx = "MPSState::set_tensors";
    const auto fail = [ctx](const std::string& what) {
        throw std::invalid_argument(std::string(ctx) + ": " + what);
    };
    if (static_cast<int>(sites.size()) != n_qubits) {
        fail("expected " + std::to_string(n_qubits) + " site tensors, got " +
             std::to_string(sites.size()));
    }
    for (int q = 0; q < n_qubits; ++q) {
        const MPSTensor& t = sites[static_cast<std::size_t>(q)];
        const std::string site = "site " + std::to_string(q);
        if (t.bond_left < 1 || t.bond_right < 1) {
            fail(site + " has a bond below 1 (" + std::to_string(t.bond_left) +
                 ", " + std::to_string(t.bond_right) + ")");
        }
        if (q == 0 && t.bond_left != 1) fail("the left end bond must be 1");
        if (q == n_qubits - 1 && t.bond_right != 1) {
            fail("the right end bond must be 1");
        }
        if (q > 0 &&
            sites[static_cast<std::size_t>(q - 1)].bond_right != t.bond_left) {
            fail(site + " has left bond " + std::to_string(t.bond_left) +
                 " where its neighbour's right bond is " +
                 std::to_string(sites[static_cast<std::size_t>(q - 1)].bond_right));
        }
        const std::size_t expected = static_cast<std::size_t>(t.bond_left) * 2 *
                                     static_cast<std::size_t>(t.bond_right);
        if (t.data.size() != expected) {
            fail(site + " holds " + std::to_string(t.data.size()) +
                 " entries where its bonds need " + std::to_string(expected));
        }
        for (const Complex128& c : t.data) {
            if (!is_finite_strict(c.real) || !is_finite_strict(c.imag)) {
                fail(site + " holds a non-finite entry");
            }
        }
    }

    tensors_ = std::move(sites);
    span_lo = 0;
    span_hi = n_qubits - 1;
    fidelity.reset();
}

void MPSState::canonicalize(int site) {
    detail::check_qubit(site, n_qubits, "MPSState::canonicalize");
    focus(site, site);
}

// =============================================================================
// Moving the orthogonality centre
// =============================================================================
//
// Both steps go through the one QR in detail::eigen_backend and then multiply
// the leftover factor into the neighbour. Complex128 is layout-identical to
// std::complex<double>, so every site is mapped in place: a site's data is
// row-major both as a (2 chi_L) x chi_R matrix and as a chi_L x (2 chi_R) one.

namespace {

using RowMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::RowMajor>;
using ColMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::ColMajor>;

std::complex<double>* as_std(Complex128* p) {
    return reinterpret_cast<std::complex<double>*>(p);
}

}  // namespace

void MPSState::shift_right(int q) {
    MPSTensor& A = tensors_[static_cast<std::size_t>(q)];
    MPSTensor& B = tensors_[static_cast<std::size_t>(q + 1)];
    const int rows = 2 * A.bond_left;
    const int chi = A.bond_right;
    const int k = std::min(rows, chi);

    std::vector<std::complex<double>> Q(static_cast<std::size_t>(rows) * k);
    std::vector<std::complex<double>> R(static_cast<std::size_t>(k) * chi);
    detail::qr_thin(as_std(A.data.data()), rows, chi,
                    detail::MatrixOrder::RowMajor, Q.data(), R.data());

    MPSTensor A_new(A.bond_left, k);
    Eigen::Map<RowMajorC>(as_std(A_new.data.data()), rows, k) =
        Eigen::Map<const ColMajorC>(Q.data(), rows, k);

    const int cols = 2 * B.bond_right;
    MPSTensor B_new(k, B.bond_right);
    Eigen::Map<RowMajorC>(as_std(B_new.data.data()), k, cols) =
        Eigen::Map<const ColMajorC>(R.data(), k, chi) *
        Eigen::Map<const RowMajorC>(as_std(B.data.data()), chi, cols);

    A = std::move(A_new);
    B = std::move(B_new);
}

void MPSState::shift_left(int q) {
    MPSTensor& A = tensors_[static_cast<std::size_t>(q - 1)];
    MPSTensor& B = tensors_[static_cast<std::size_t>(q)];
    const int chi = B.bond_left;
    const int cols = 2 * B.bond_right;
    const int k = std::min(chi, cols);

    // B's row-major chi x (2 chi_R) buffer, read column-major, is its
    // transpose, so this factorises B^T = Q R and B = R^T Q^T (see qr_thin).
    std::vector<std::complex<double>> Q(static_cast<std::size_t>(cols) * k);
    std::vector<std::complex<double>> R(static_cast<std::size_t>(k) * chi);
    detail::qr_thin(as_std(B.data.data()), cols, chi,
                    detail::MatrixOrder::ColMajor, Q.data(), R.data());

    // Q^T, k x (2 chi_R) row-major, is the column-major Q buffer as it stands.
    MPSTensor B_new(k, B.bond_right);
    std::copy(Q.begin(), Q.end(), as_std(B_new.data.data()));

    // L = R^T, chi x k, is the column-major R buffer read row-major.
    const int rows = 2 * A.bond_left;
    MPSTensor A_new(A.bond_left, k);
    Eigen::Map<RowMajorC>(as_std(A_new.data.data()), rows, k) =
        Eigen::Map<const RowMajorC>(as_std(A.data.data()), rows, chi) *
        Eigen::Map<const RowMajorC>(R.data(), chi, k);

    A = std::move(A_new);
    B = std::move(B_new);
}

void MPSState::focus(int a, int b) {
    if (n_qubits == 0) return;
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

bool MPSState::split_needs_centre(int q) const {
    if (canonical_form == CanonicalForm::Always) return true;
    if (cutoff > MPS_DEFAULT_CUTOFF) return true;
    const MPSTensor& A = tensors_[static_cast<std::size_t>(q)];
    const MPSTensor& B = tensors_[static_cast<std::size_t>(q + 1)];
    return std::min(2 * A.bond_left, 2 * B.bond_right) > max_bond_dim;
}

// =============================================================================
// Collapse at the centre
// =============================================================================

// Every draw below compares a uniform against p0 / total, which is exactly 1
// when the other marginal is 0 and exactly 0 when p0 is, so an outcome of
// weight zero is never drawn and the outcome's marginal is always positive.
// Each routine first refuses a chain with no norm (zero or non-finite), so the
// total it divides by is finite and positive.

void MPSState::collapse_centre(int site, int outcome, double p_outcome) {
    MPSTensor& T = tensors_[static_cast<std::size_t>(site)];
    const int other = 1 - outcome;
    const double inv_norm = 1.0 / std::sqrt(p_outcome);
    for (int l = 0; l < T.bond_left; ++l)
        for (int r = 0; r < T.bond_right; ++r) {
            T(l, other, r) = Complex128(0.0, 0.0);
            T(l, outcome, r).real *= inv_norm;
            T(l, outcome, r).imag *= inv_norm;
        }
    fidelity.invalidate();
}

int MPSState::measure_qubit(int qubit, std::mt19937_64& rng) {
    detail::check_qubit(qubit, n_qubits, "MPSState::measure_qubit");
    focus(qubit, qubit);
    // At the centre the site's slice norms ARE the raw marginals ⟨ψ|P_k|ψ⟩,
    // summing to the state's norm², so sampling normalises by their sum and
    // the collapse divides out the chosen one, leaving unit norm.
    const std::array<double, 2> p = centre_marginals(qubit);
    const double total = p[0] + p[1];
    detail::require_norm_to_sample(std::sqrt(total), "MPSState::measure_qubit");
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    const int outcome = (udist(rng) < p[0] / total) ? 0 : 1;
    collapse_centre(qubit, outcome, p[static_cast<std::size_t>(outcome)]);
    return outcome;
}

// =============================================================================
// svd_truncate - adapter onto the shared verified truncation
// =============================================================================
//
// The ladder itself (kernel, Jacobi rescue, Gram rescue, throw, and why each
// rung exists) lives in include/lindblad/detail/svd_truncate.hpp and is shared with
// the qudit layer, so the two cannot drift apart in what they guarantee. What
// stays here is this layer's storage convention and its own counters: the
// blocks are row-major, the caller wants V-dagger rather than V, and the
// running truncation error and rescue counts belong to this state object.

// Called once the ladder has returned, so every figure covers splits that
// completed and the rescue counts report rungs that produced the split: a
// split no permitted rung produced does not return. The Gram route's
// floor-rejected weight is booked beside the truncation total, never inside it.
void MPSState::account_split(const detail::SvdTruncation& split,
                             std::uint64_t nanos) {
    ++svd_calls;
    svd_nanos += nanos;
    if (split.used_jacobi_rescue) ++jacobi_rescues;
    if (split.used_gram_fallback) ++gram_fallbacks;
    if (split.used_unverified) ++ignored_rejections;
    floor_rejected += split.floor_rejected_weight;
    total_truncation_error += split.discarded_weight;
    max_verify_resid_excess =
        std::max(max_verify_resid_excess, split.residual_excess);
}

detail::SvdPolicy MPSState::svd_policy() const {
    return detail::SvdPolicy{svd_rejection, svd_accept_gram, svd_report};
}

// Σ sigma² over the singular values a split kept.
static double kept_weight(const detail::SvdTruncation& split) {
    double kept = 0.0;
    for (int i = 0; i < split.rank; ++i) kept += split.S(i) * split.S(i);
    return kept;
}

void MPSState::svd_truncate(
    const std::vector<Complex128>& M,
    int rows, int cols,
    std::vector<Complex128>& U_out,
    std::vector<double>& S_out,
    std::vector<Complex128>& Vt_out,
    int& new_rank
) {
    if (is_jacobi(svd_method)) warn_jacobi_slower_once(svd_method);
    if (svd_method == SVDMethod::EigenBDC) note_eigen_bdc_once();
    detail::note_nondefault_svd_policy(detail::SvdLayer::Qubit, svd_policy());

    // Bracketing the ladder rather than the factorisation alone: the rung that
    // recomputes through the Gram route is part of what a split costs, and a
    // reading that excluded it would understate exactly the splits that were
    // hardest to factor.
    const auto svd_t0 = std::chrono::steady_clock::now();
    const detail::SvdTruncation r = detail::svd_truncate_verified(
        M.data(), rows, cols, detail::MatrixOrder::RowMajor,
        max_bond_dim, cutoff, svd_method, svd_policy(), "MPS svd_truncate");
    account_split(r, static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - svd_t0).count()));
    // The Gram route's floored weight counts toward the bound alone; an
    // unverified factorisation leaves the figures nothing they can describe.
    fidelity.record(kept_weight(r), r.discarded_weight, r.floor_rejected_weight);
    if (r.used_unverified) fidelity.invalidate();

    const int k = r.rank;
    new_rank = k;

    S_out.resize(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) S_out[static_cast<size_t>(i)] = r.S(i);

    U_out.resize(static_cast<size_t>(rows) * k);
    for (int row = 0; row < rows; ++row) {
        for (int c = 0; c < k; ++c) {
            const auto z = r.U(row, c);
            U_out[static_cast<size_t>(row) * k + c] =
                Complex128(z.real(), z.imag());
        }
    }

    // Row c of V-dagger is the conjugate of column c of V.
    Vt_out.resize(static_cast<size_t>(k) * cols);
    for (int c = 0; c < k; ++c) {
        for (int col = 0; col < cols; ++col) {
            const auto z = std::conj(r.V(col, c));
            Vt_out[static_cast<size_t>(c) * cols + col] =
                Complex128(z.real(), z.imag());
        }
    }
}

// =============================================================================
// Single-qubit gate — O(chi) per qubit
// U is a 2x2 unitary in row-major order: [u00, u01, u10, u11]
// =============================================================================

void MPSState::apply_single_qubit_gate(
    const std::array<Complex128, 4>& U_in, int qubit,
    ValidationOptions validation
) {
    detail::check_qubit(qubit, n_qubits, "MPSState::apply_single_qubit_gate");
    std::array<Complex128, 4> U_fixed;
    const std::array<Complex128, 4>& U = detail::check_unitary_fixing(
        U_in, 2, validation, "MPSState::apply_single_qubit_gate", U_fixed);
    gate_one_site(U, qubit,
                  detail::gate_keeps_unitarity(U.data(), 2, validation,
                                               unchecked_gates));
}

void MPSState::gate_one_site(const std::array<Complex128, 4>& U, int qubit,
                             bool unitary) {
    auto& T = tensors_[qubit];
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

    // A site outside the open span is held orthonormal, and every read at the
    // centre relies on it; a matrix that is not unitary voids that, so the
    // span widens over the site and the next move of the centre sweeps it.
    // The span stays contiguous, taking in whatever lies between.
    if (!unitary) {
        if (qubit < span_lo) span_lo = qubit;
        if (qubit > span_hi) span_hi = qubit;
        fidelity.invalidate();
    }
}

// =============================================================================
// Run budget - what an MPS run's growth is charged against
// =============================================================================

// Bytes the chain's site tensors hold.
static std::uint64_t chain_bytes(const std::vector<MPSTensor>& tensors) {
    std::uint64_t bytes = 0;
    for (const auto& tensor : tensors) {
        bytes = detail::saturating_add(bytes,
                                       detail::complex_bytes(tensor.data.size()));
    }
    return bytes;
}

// Where a linked run is, for a refusal: built only when one is raised, so the
// instruction loop stores two ints and a pointer per gate and copies nothing.
static FailurePoint link_point(const detail::BudgetLink& link) {
    FailurePoint point;
    point.shot = link.shot;
    point.instruction = link.instruction;
    if (link.inst != nullptr) {
        point.gate = link.inst->gate_name();
        point.qubits = link.inst->qubits;
    }
    return point;
}

// What a two-site update at bonds (bl, br) allocates beyond the chain before
// it releases anything, in bytes: the contracted block and the gated block,
// each (2 bl) x (2 br); the factorisation's U and V at the kept rank k and the
// row-major copies taken of them; the two new site tensors; and the singular
// values twice. A kernel's own workspace is not counted, so the budget bounds
// what this code allocates rather than every byte a library below it touches.
static std::uint64_t two_site_peak_bytes(int bl, int br, int max_bond_dim) {
    const std::uint64_t l = static_cast<std::uint64_t>(bl);
    const std::uint64_t r = static_cast<std::uint64_t>(br);
    std::uint64_t k = std::min<std::uint64_t>(2 * l, 2 * r);
    if (max_bond_dim > 0) k = std::min<std::uint64_t>(k, static_cast<std::uint64_t>(max_bond_dim));
    const std::uint64_t blocks = detail::saturating_mul(8, detail::saturating_mul(l, r));
    const std::uint64_t factors = detail::saturating_mul(6, detail::saturating_mul(k, l + r));
    return detail::saturating_add(
        detail::complex_bytes(detail::saturating_add(blocks, factors)),
        detail::saturating_mul(2 * k, sizeof(double)));
}

// A dense fallback holds, at its peak, at most four arrays of 2^n amplitudes:
// to_statevector's contraction keeps its last two rows of the expansion and
// then one of them beside the Statevector it fills, and the rebuild holds the
// Statevector, its bit-reversed block, and the first split's factor of the
// same size. Checked before to_statevector is called.
static void charge_dense_fallback(const MPSState& mps, const std::string& what) {
    detail::RunBudget* budget = mps.budget_link.budget;
    if (budget == nullptr) return;
    const std::uint64_t peak = detail::saturating_mul(
        4, detail::complex_bytes(detail::pow2_saturating(mps.n_qubits)));
    if (!budget->fits(peak)) budget->check_peak(peak, what, link_point(mps.budget_link));
}

// =============================================================================
// Adjacent two-qubit gate — contract, apply, SVD-split
// U is 4x4 in row-major index order: U[po1*2+po2, pi1*2+pi2]
// q1 and q2 MUST be adjacent (q2 == q1+1)
// =============================================================================

void MPSState::apply_two_qubit_gate_adjacent(
    const std::array<Complex128, 16>& U, int q1, Absorb absorb
) {
    int q2 = q1 + 1;
    detail::check_qubit(q1, n_qubits, "MPSState::apply_two_qubit_gate_adjacent");
    detail::check_qubit(q2, n_qubits, "MPSState::apply_two_qubit_gate_adjacent");

    // Before the contraction, because moving the centre rewrites the two
    // sites being contracted.
    if (split_needs_centre(q1)) focus(q1, q2);

    auto& T1 = tensors_[q1];
    auto& T2 = tensors_[q2];

    int bl = T1.bond_left;
    int bm = T1.bond_right;  // = T2.bond_left
    int br = T2.bond_right;

    // Before anything is allocated, so a run over its budget stops with the
    // chain as it was.
    if (budget_link.budget != nullptr) {
        const std::uint64_t peak = two_site_peak_bytes(bl, br, max_bond_dim);
        if (!budget_link.budget->fits(peak)) {
            budget_link.budget->check_peak(peak, "a two-site update", link_point(budget_link));
        }
    }

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

#ifdef LINDBLAD_MPS_THETA_HARVEST
    // Offered here rather than reconstructed later: the block exists only
    // between the gate contraction above and the factorisation below, and
    // rebuilding it from the site tensors afterwards would duplicate this
    // function's layout logic and drift from what actually gets factorised.
    detail::theta_harvest_offer(theta_new, rows, cols);
#endif

    // SVD theta_new into T1' and T2'
    std::vector<Complex128> U_mat, Vt_mat;
    std::vector<double> S_vals;
    int new_rank;
    svd_truncate(theta_new, rows, cols, U_mat, S_vals, Vt_mat, new_rank);

    // The singular values go to one side and the other side keeps its
    // isometry: U is left-orthonormal as T1', V-dagger right-orthonormal as
    // T2'. Absorb::Right gives T1' = U, T2' = S V-dagger; Absorb::Left gives
    // T1' = U S, T2' = V-dagger.
    const bool right = (absorb == Absorb::Right);
    T1 = MPSTensor(bl, new_rank);
    for (int l = 0; l < bl; ++l)
        for (int p1 = 0; p1 < 2; ++p1)
            for (int r = 0; r < new_rank; ++r) {
                const Complex128 u = U_mat[(l * 2 + p1) * new_rank + r];
                T1(l, p1, r) = right ? u : u * Complex128(S_vals[r], 0.0);
            }

    T2 = MPSTensor(new_rank, br);
    for (int l = 0; l < new_rank; ++l)
        for (int p2 = 0; p2 < 2; ++p2)
            for (int r = 0; r < br; ++r) {
                const Complex128 v = Vt_mat[l * cols + p2 * br + r];
                T2(l, p2, r) = right ? Complex128(S_vals[l], 0.0) * v : v;
            }

    // The site holding S is always in the span afterwards. The site holding an
    // isometry drops out of it when no site beyond it, on its own side, was in
    // the span. After a split that moved the centre first the span was inside
    // [q1, q2], so this leaves exactly the site holding S.
    if (right) {
        if (span_lo >= q1) span_lo = q2;
        span_hi = std::max(span_hi, q2);
    } else {
        if (span_hi <= q2) span_hi = q1;
        span_lo = std::min(span_lo, q1);
    }

    if (budget_link.budget != nullptr) budget_link.budget->set_state(chain_bytes(tensors_));
}

// =============================================================================
// SWAP gate in MPS (swaps two adjacent tensors via SVD)
// =============================================================================

void MPSState::apply_swap_adjacent(int q, Absorb absorb) {
    // Apply FSWAP (fermionic SWAP) as a 4x4 gate
    // SWAP: |00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→|11⟩
    // Row-major: SWAP[po1*2+po2, pi1*2+pi2]
    std::array<Complex128, 16> SWAP_gate{};
    SWAP_gate[0*4+0] = Complex128(1, 0);  // |00⟩→|00⟩
    SWAP_gate[1*4+2] = Complex128(1, 0);  // |01⟩→|10⟩
    SWAP_gate[2*4+1] = Complex128(1, 0);  // |10⟩→|01⟩
    SWAP_gate[3*4+3] = Complex128(1, 0);  // |11⟩→|11⟩
    apply_two_qubit_gate_adjacent(SWAP_gate, q, absorb);
}

// =============================================================================
// General two-qubit gate for arbitrary (non-adjacent) qubits
// Uses SWAP chain: move q1 and q2 adjacent, apply gate, swap back.
// =============================================================================

// Exchanges which operand owns bit 0 and which owns bit 1 of a 4x4 gate's
// index, on rows and columns alike. The exchange is its own inverse, so it
// converts in either direction between the project's LSB-first matrices
// (bit 0 = first operand, as gates::apply_unitary reads them) and the
// MSB-first order the two-site contraction and gate4x4's builders use.
static std::array<Complex128, 16> exchange_operand_bits(
    const std::array<Complex128, 16>& U) {
    const auto swap01 = [](int idx) { return ((idx & 1) << 1) | ((idx >> 1) & 1); };
    std::array<Complex128, 16> out{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[r * 4 + c] = U[swap01(r) * 4 + swap01(c)];
    return out;
}

void MPSState::apply_two_qubit_gate(
    const std::array<Complex128, 16>& U_in, int q1, int q2,
    ValidationOptions validation
) {
    detail::check_qubit(q1, n_qubits, "MPSState::apply_two_qubit_gate");
    detail::check_qubit(q2, n_qubits, "MPSState::apply_two_qubit_gate");
    detail::check_distinct2(q1, q2, "MPSState::apply_two_qubit_gate");
    std::array<Complex128, 16> U_fixed;
    const std::array<Complex128, 16>& U = detail::check_unitary_fixing(
        U_in, 4, validation, "MPSState::apply_two_qubit_gate", U_fixed);
    gate_two_site(U, q1, q2,
                  detail::gate_keeps_unitarity(U.data(), 4, validation,
                                               unchecked_gates));
}

void MPSState::gate_two_site(const std::array<Complex128, 16>& U, int q1,
                             int q2, bool unitary) {
    // The caller's matrix is LSB-first (q1 is bit 0), the contraction reads
    // MSB-first. Exchanging the two bits permutes rows and columns alike, so
    // it preserves unitarity and `unitary` holds for what is applied.
    apply_two_qubit_gate_msb(exchange_operand_bits(U), q1, q2);
    // The splits restore the gauge whatever U was; the fidelity bound does
    // not survive a gate that is not unitary.
    if (!unitary) fidelity.invalidate();
}

void MPSState::apply_two_qubit_gate_msb(
    const std::array<Complex128, 16>& U, int q1, int q2
) {
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
        apply_two_qubit_gate_msb(U_swapped, q1, q2);
        return;
    }

    // Now q1 < q2. Move q2 next to q1 by SWAP chain from right.
    // After final SWAP chain: q2 is at position q1+1.
    //
    // Each split sends its singular values toward the block the chain touches
    // next, so when splits move the centre the next one finds it already in
    // place: leftward on the way in, rightward from the gate on the way back.
    for (int i = q2 - 1; i > q1; --i) {
        apply_swap_adjacent(i, Absorb::Left);  // SWAP qubits at pos i and i+1
    }

    // Apply the gate on (q1, q1+1)
    apply_two_qubit_gate_adjacent(U, q1, Absorb::Right);

    // Swap q2 back to its original position
    for (int i = q1 + 1; i < q2; ++i) {
        apply_swap_adjacent(i, Absorb::Right);
    }
}

// =============================================================================
// current_max_bond_dim
// =============================================================================

int MPSState::current_max_bond_dim() const {
    int max_chi = 0;
    for (const auto& t : tensors_) {
        max_chi = std::max(max_chi, std::max(t.bond_left, t.bond_right));
    }
    return max_chi;
}

// =============================================================================
// absorb_profile
// =============================================================================

// The tallies add and the residual takes the max, because that is how each
// figure is accumulated across the splits of one chain: a chain that absorbed
// another reports exactly what one chain performing both sets of splits would.
void MPSState::absorb_profile(const MPSState& other) {
    svd_calls += other.svd_calls;
    svd_nanos += other.svd_nanos;
    jacobi_rescues += other.jacobi_rescues;
    gram_fallbacks += other.gram_fallbacks;
    ignored_rejections += other.ignored_rejections;
    floor_rejected += other.floor_rejected;
    total_truncation_error += other.total_truncation_error;
    max_verify_resid_excess =
        std::max(max_verify_resid_excess, other.max_verify_resid_excess);
}

// ⟨ψ|ψ⟩ by left-to-right transfer-matrix contraction over the open span:
//   E_{q+1}[aR', aR] = Σ_{phys, aL', aL} conj(A_q[aL', phys, aR'])
//                                        · E_q[aL', aL] · A_q[aL, phys, aR]
// Every site left of the span is left-orthonormal, so the environment arriving
// at its first site is the identity; every site right of it is
// right-orthonormal, so the environment waiting past its last site is the
// identity too, and closing on it is a trace. The imaginary part is zero by
// construction (the contraction is ⟨ψ|ψ⟩) and is discarded rather than
// checked.
//
// A_q is reshaped once per site into a (2·chi_L) x chi_R matrix so each step is
// two matrix products rather than a five-deep index loop. A single-site span
// needs none of it: the trace is that site's squared Frobenius norm.
double MPSState::norm_sq() const {
    if (n_qubits == 0) return 1.0;
    if (span_lo == span_hi) {
        double sum = 0.0;
        for (const Complex128& c : tensors_[static_cast<size_t>(span_lo)].data)
            sum += c.real * c.real + c.imag * c.imag;
        return sum;
    }

    const int chi_first = tensors_[static_cast<size_t>(span_lo)].bond_left;
    Eigen::MatrixXcd E = Eigen::MatrixXcd::Identity(chi_first, chi_first);

    for (int q = span_lo; q <= span_hi; ++q) {
        const MPSTensor& T = tensors_[static_cast<size_t>(q)];
        const int chi_L = T.bond_left;
        const int chi_R = T.bond_right;

        // Row index (phys · chi_L + aL) keeps the left bond contiguous, which
        // is what lets E multiply each physical block as one dense product.
        Eigen::MatrixXcd A_left(2 * chi_L, chi_R);
        for (int phys = 0; phys < 2; ++phys)
            for (int aL = 0; aL < chi_L; ++aL)
                for (int aR = 0; aR < chi_R; ++aR) {
                    const Complex128& c = T(aL, phys, aR);
                    A_left(phys * chi_L + aL, aR) =
                        std::complex<double>(c.real, c.imag);
                }

        Eigen::MatrixXcd tmp(2 * chi_L, chi_R);
        for (int phys = 0; phys < 2; ++phys)
            tmp.block(phys * chi_L, 0, chi_L, chi_R) =
                E * A_left.block(phys * chi_L, 0, chi_L, chi_R);

        E = A_left.adjoint() * tmp;
    }

    return E.trace().real();
}

void MPSState::normalize() {
    const double n = std::sqrt(norm_sq());
    // Refuses rather than returning quietly, matching every other state class:
    // handing back an unnormalized state from a call named normalize reports
    // nothing to a caller who asked for exactly one thing.
    if (!is_normalizable(n)) {
        detail::raise<RuntimeFailure>("MPSState::normalize",
            "no norm to divide out; the state is zero or non-finite");
    }
    if (n_qubits == 0) return;
    const double inv = 1.0 / n;
    for (Complex128& c : tensors_[static_cast<size_t>(span_lo)].data) {
        c.real *= inv;
        c.imag *= inv;
    }
}

// norm_sq() is the measurement here, unlike the dense classes, because there is
// nothing else to measure: an MPS holds no flat amplitude array, so there is no
// sum for the strict-FP balanced summation to quarantine. The contraction runs
// under the project-wide flags like the rest of this file, which is why the
// tolerance carries the target-dependence the dense path does not have to.
bool MPSState::is_normalized(double atol) const {
    return std::abs(norm_sq() - 1.0) <= atol;
}

void MPSState::check_normalized(ValidationOptions validation) {
    // Returns before measuring when nothing would consume the residual. The
    // measurement is an O(n·chi³) contraction, which is the most expensive of
    // any state class here, so the opt-out matters most on this one.
    if (detail::measurement_unused(validation)) return;
    const char* ctx = "MPSState::check_normalized";
    const double ns = norm_sq();
    if (detail::check_normalized(ns, validation, ctx) &&
        detail::normalization_repairable(ns, validation, ctx,
                                         detail::STATE_NORMALIZATION)) {
        normalize();
    }
}

// =============================================================================
// probabilities_single - marginals read through the open span
// =============================================================================
//
// P(k) = Re Tr[ A_k† L A_k R^T ], where A_k is the qubit's site restricted to
// physical index k (a chi_L x chi_R matrix), L the environment of every site
// left of it and R of every site right of it, both indexed (bra, ket). A site
// left of the open span is left-orthonormal and one right of it
// right-orthonormal, and each contracts to the identity, so L is built only
// from the span's sites left of the qubit and R only from its sites right of
// it. With the centre on the qubit both are identities and this reads one site.

namespace {

// Site `t` restricted to physical index `phys`, as a chi_L x chi_R matrix.
Eigen::MatrixXcd physical_slice(const MPSTensor& t, int phys) {
    Eigen::MatrixXcd a(t.bond_left, t.bond_right);
    for (int l = 0; l < t.bond_left; ++l)
        for (int r = 0; r < t.bond_right; ++r) {
            const Complex128& c = t(l, phys, r);
            a(l, r) = std::complex<double>(c.real, c.imag);
        }
    return a;
}

// Environment arriving at the left bond of site `stop` from sites
// first..stop-1, starting from the identity on the left bond of `first`:
// E <- Σ_p A_p† E A_p per site.
Eigen::MatrixXcd left_environment(const std::vector<MPSTensor>& t, int first,
                                  int stop) {
    const int chi = t[static_cast<size_t>(first)].bond_left;
    Eigen::MatrixXcd E = Eigen::MatrixXcd::Identity(chi, chi);
    for (int q = first; q < stop; ++q) {
        const MPSTensor& site = t[static_cast<size_t>(q)];
        Eigen::MatrixXcd next =
            Eigen::MatrixXcd::Zero(site.bond_right, site.bond_right);
        for (int p = 0; p < 2; ++p) {
            const Eigen::MatrixXcd a = physical_slice(site, p);
            next.noalias() += a.adjoint() * (E * a);
        }
        E = std::move(next);
    }
    return E;
}

// Environment arriving at the right bond of site `stop` from sites
// first..stop+1, walking left from the identity on the right bond of
// `first`: R <- Σ_p conj(A_p) R A_p^T per site.
Eigen::MatrixXcd right_environment(const std::vector<MPSTensor>& t, int first,
                                   int stop) {
    const int chi = t[static_cast<size_t>(first)].bond_right;
    Eigen::MatrixXcd R = Eigen::MatrixXcd::Identity(chi, chi);
    for (int q = first; q > stop; --q) {
        const MPSTensor& site = t[static_cast<size_t>(q)];
        Eigen::MatrixXcd next =
            Eigen::MatrixXcd::Zero(site.bond_left, site.bond_left);
        for (int p = 0; p < 2; ++p) {
            const Eigen::MatrixXcd a = physical_slice(site, p);
            next.noalias() += a.conjugate() * (R * a.transpose());
        }
        R = std::move(next);
    }
    return R;
}

}  // namespace

std::vector<double> MPSState::probabilities_single(int qubit) const {
    detail::check_qubit(qubit, n_qubits, "MPSState::probabilities_single");

    const bool left_is_identity = span_lo >= qubit;
    const bool right_is_identity = span_hi <= qubit;
    if (left_is_identity && right_is_identity) {
        const std::array<double, 2> p = centre_marginals(qubit);
        return {p[0], p[1]};
    }

    const Eigen::MatrixXcd L =
        left_environment(tensors_, std::min(span_lo, qubit), qubit);
    const Eigen::MatrixXcd R =
        right_environment(tensors_, std::max(span_hi, qubit), qubit);

    const MPSTensor& site = tensors_[static_cast<size_t>(qubit)];
    std::vector<double> probs(2, 0.0);
    for (int p = 0; p < 2; ++p) {
        const Eigen::MatrixXcd a = physical_slice(site, p);
        probs[static_cast<size_t>(p)] =
            (a.adjoint() * L * a * R.transpose()).trace().real();
    }
    return probs;
}

// Squared norms of the two physical slices of `site`. With every site to its
// left left-orthonormal and every site to its right right-orthonormal these
// ARE the raw marginals ⟨ψ|P_k|ψ⟩, which is the only way it is called.
std::array<double, 2> MPSState::centre_marginals(int site) const {
    const MPSTensor& T = tensors_[static_cast<std::size_t>(site)];
    std::array<double, 2> p{0.0, 0.0};
    for (int l = 0; l < T.bond_left; ++l)
        for (int phys = 0; phys < 2; ++phys)
            for (int r = 0; r < T.bond_right; ++r) {
                const Complex128& c = T(l, phys, r);
                p[static_cast<std::size_t>(phys)] +=
                    c.real * c.real + c.imag * c.imag;
            }
    return p;
}

// =============================================================================
// measure_sequential - measure every qubit at the centre, left to right
// =============================================================================
// The centre starts on qubit 0. Each qubit's marginals are then its site's
// slice norms, conditional on every outcome before it because the chain has
// already collapsed onto those, and after the collapse one QR step carries the
// centre to the next qubit. O(N·chi³) per call, with no environments.

std::string MPSState::measure_sequential(std::mt19937_64& rng) {
    std::string bits(static_cast<size_t>(n_qubits), '0');
    if (n_qubits == 0) return bits;
    focus(0, 0);

    // With the centre on qubit 0 its marginals sum to the chain's norm², so
    // the refusal is taken there, before the first draw. Each collapse leaves
    // unit norm, so every later total is 1 to rounding.
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int q = 0; q < n_qubits; ++q) {
        const std::array<double, 2> probs = centre_marginals(q);
        const double total = probs[0] + probs[1];
        if (q == 0)
            detail::require_norm_to_sample(std::sqrt(total),
                                           "MPSState::measure_sequential");
        const int outcome = (dist(rng) < probs[0] / total) ? 0 : 1;
        // Project bitstring convention: qubit 0 is the RIGHTMOST character
        // (matches Statevector::sample_counts and the per-shot paths).
        bits[static_cast<size_t>(n_qubits - 1 - q)] = outcome ? '1' : '0';

        collapse_centre(q, outcome, probs[static_cast<size_t>(outcome)]);
        if (q + 1 < n_qubits) focus(q + 1, q + 1);
    }
    return bits;
}

// =============================================================================
// to_statevector - full contraction, up to the chain's dense limit
// =============================================================================

// Every dense route this backend takes stops at max_mps_dense_qubits(
// qubit_limit): 25 qubits under Enforce (2^25 complex doubles, 512 MiB), 31
// under Lift. Terminal sampling stops earlier still (see
// dense_sampling_is_cheaper). This is the refusal each of those routes gives,
// naming the limit in force and, when one exists, how to lift it.
static std::string dense_limit_text(int n_qubits, QubitLimit limit) {
    std::string text = std::to_string(n_qubits) +
                       " qubits exceed the dense-fallback limit (" +
                       std::to_string(max_mps_dense_qubits(limit)) + ")";
    if (n_qubits > LIFTED_MPS_DENSE_MAX_QUBITS) {
        text += ", which no setting raises past " +
                std::to_string(LIFTED_MPS_DENSE_MAX_QUBITS) +
                ": the chain is rebuilt by factorising a 2 x 2^(n-1) block, "
                "and that is the widest the factorisation can address";
    } else if (limit == QubitLimit::Enforce) {
        text += "; qubit_limit = QubitLimit::Lift raises it to " +
                std::to_string(LIFTED_MPS_DENSE_MAX_QUBITS);
    }
    return text;
}

Statevector MPSState::to_statevector() const {
    if (n_qubits > max_mps_dense_qubits(qubit_limit)) {
        detail::raise<InvalidArgument>("MPSState::to_statevector",
                                       dense_limit_text(n_qubits, qubit_limit));
    }

    // Standard left-to-right site contraction: maintains a (dim_so_far x chi) matrix
    // that grows 2x per site.  O(N) allocations vs O(N * 2^N) for the per-basis-state loop.
    //
    // After site q: current[idx, r] = amplitude of basis state idx (0..2^(q+1)-1)
    //               with bond index r ∈ [0, bond_right[q]).
    //
    // Expansion step: new_current[idx*2 + p, r'] = sum_m current[idx, m] * T[m, p, r']
    //
    // Row counts and flat offsets are size_t: past 30 qubits, which Lift
    // allows, 2^n rows no longer fit an int.
    std::size_t dim_so_far = 1;
    std::vector<Complex128> current(1, Complex128(1.0, 0.0));  // 1x1 identity

    for (int q = 0; q < n_qubits; ++q) {
        const auto& T  = tensors_[q];
        const int bl   = T.bond_left;
        const int br   = T.bond_right;
        const std::size_t new_dim = dim_so_far * 2;
        const std::size_t ubl = static_cast<std::size_t>(bl);
        const std::size_t ubr = static_cast<std::size_t>(br);

        std::vector<Complex128> next(new_dim * ubr, Complex128(0.0, 0.0));
        for (std::size_t idx = 0; idx < dim_so_far; ++idx) {
            for (int p = 0; p < 2; ++p) {
                const std::size_t new_row = idx * 2 + static_cast<std::size_t>(p);
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int m = 0; m < bl; ++m)
                        sum += current[idx * ubl + static_cast<std::size_t>(m)] * T(m, p, r);
                    next[new_row * ubr + static_cast<std::size_t>(r)] = sum;
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
    Statevector sv(n_qubits, qubit_limit);
    for (size_t idx = 0; idx < dim_so_far; ++idx) {
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
// MPSState::rebuild_from_statevector — reconstruct the chain by sequential SVD
// Used as a fallback when a gate cannot be applied natively in MPS form, and to
// seed a run from a supplied dense state.
// =============================================================================

void MPSState::rebuild_from_statevector(const Statevector& sv) {
    const int n = n_qubits;
    if (sv.n_qubits != n) {
        detail::raise<InvalidArgument>("MPSState::rebuild_from_statevector",
            "the amplitudes cover " + std::to_string(sv.n_qubits) + " qubits, this chain " +
            std::to_string(n));
    }
    if (n == 0) return;
    // The first split factorises a 2 x 2^(n-1) block, and the factorisation
    // takes its dimensions as int.
    if (n > LIFTED_MPS_DENSE_MAX_QUBITS) {
        detail::raise<InvalidArgument>("MPSState::rebuild_from_statevector",
            std::to_string(n) + " qubits is wider than the " +
            std::to_string(LIFTED_MPS_DENSE_MAX_QUBITS) +
            " the rebuild can factorise; its first block is 2 x 2^(n-1)");
    }

    // Built beside this state rather than into it, so a throw part way through
    // the sweep leaves the chain and its fidelity figures as they were rather
    // than half rewritten.
    MPSState result(n, max_bond_dim, cutoff);
    detail::FidelityLedger ledger = fidelity;
    size_t dim = 1ULL << n;

    // right_cols is 2^n before the first split, one past what an int holds at
    // n = 31, so it and every flat offset below are size_t. half_cols, which
    // the factorisation receives, is at most 2^30.
    int left_bond = 1;
    std::size_t right_cols = dim;
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
        const int half_cols = static_cast<int>(right_cols / 2);
        const int rows = left_bond * 2;

        // The block IS the matrix this split needs, so it is handed over in
        // place. right_cols == 2 * half_cols, so the reshape index
        // alpha*right_cols + p*half_cols + c2 equals (alpha*2 + p)*half_cols + c2,
        // which is row-major element (alpha*2 + p, c2) of a rows x half_cols
        // matrix. p ∈ {0,1} is the physical index; c2 ∈ [0,half_cols) indexes
        // the remaining sites.
        //
        // Through the shared ladder rather than a bare factorisation: this path
        // selects a rank from singular values, and a rank chosen from values it
        // has not verified is the defect the ladder exists to prevent. With
        // this chain's selected kernel, like every other split it performs: a
        // caller who chose one is owed it here as much as on the gate path,
        // and a kernel this build cannot provide throws here as it does there.
        if (is_jacobi(svd_method)) warn_jacobi_slower_once(svd_method);
        if (svd_method == SVDMethod::EigenBDC) note_eigen_bdc_once();
        detail::note_nondefault_svd_policy(detail::SvdLayer::Qubit, svd_policy());
        const auto svd_t0 = std::chrono::steady_clock::now();
        const detail::SvdTruncation split = detail::svd_truncate_verified(
            block.data(), rows, half_cols, detail::MatrixOrder::RowMajor,
            max_bond_dim, cutoff, svd_method, svd_policy(),
            "MPSState::rebuild_from_statevector");
        const std::uint64_t svd_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - svd_t0).count());
        const int k = split.rank;

        // The same figures MPSState::svd_truncate records, because they
        // describe the STATE rather than the route that produced it. A split
        // this sweep performed is one this chain paid for, and a caller reading
        // these counters is asking about the chain and not about which function
        // built it. That applies to the timing too: a dense fallback's rebuild
        // is bond-split work the run spent, so leaving it out would let a
        // circuit hide its most expensive splits behind a >2q gate.
        account_split(split, svd_ns);
        // Canonical by construction: every site before this one is an
        // isometry and the block is the whole remainder of the state, so the
        // discarded fraction is a fraction of the state.
        ledger.record(kept_weight(split), split.discarded_weight,
                      split.floor_rejected_weight);
        if (split.used_unverified) ledger.invalidate();

        result.tensors_[site] = MPSTensor(left_bond, k);
        for (int alpha = 0; alpha < left_bond; ++alpha)
            for (int p = 0; p < 2; ++p)
                for (int r = 0; r < k; ++r)
                    result.tensors_[site](alpha, p, r) = {
                        split.U(alpha * 2 + p, r).real(),
                        split.U(alpha * 2 + p, r).imag()};

        // New block = S * V†
        const std::size_t uhalf = static_cast<std::size_t>(half_cols);
        block.resize(static_cast<std::size_t>(k) * uhalf);
        for (int r = 0; r < k; ++r)
            for (int c2 = 0; c2 < half_cols; ++c2) {
                auto v = split.S(r) * std::conj(split.V(c2, r));
                block[static_cast<std::size_t>(r) * uhalf + static_cast<std::size_t>(c2)] =
                    {v.real(), v.imag()};
            }

        left_bond = k;
        right_cols = uhalf;
    }

    result.tensors_[n - 1] = MPSTensor(left_bond, 1);
    for (int alpha = 0; alpha < left_bond; ++alpha)
        for (int p = 0; p < 2; ++p)
            result.tensors_[n - 1](alpha, p, 0) = block[alpha * 2 + p];

    // Only the tensors move across. The counters above were accumulated into
    // this object as the sweep ran, and taking result's would reset them.
    // Sites 0..n-2 are the sweep's isometries, so the centre is the last site.
    tensors_ = std::move(result.tensors_);
    span_lo = n - 1;
    span_hi = n - 1;
    fidelity = ledger;
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

// =============================================================================
// detail::MPSDispatch - run()'s route onto the chain's gate helpers
// =============================================================================
// Below run()'s pre-flight no matrix is judged again, so the dispatcher does not
// go through the public gate entries, whose policy would measure a caller's
// matrix once per gate per shot. It hands the chain each gate together with
// whether it is unitary, which is all the chain's records need: a gate this
// file builds is unitary by construction and is never measured, and a
// circuit's own matrix is judged for the records under its instruction's
// policy (detail::gate_keeps_unitarity). The structural checks stay, since a
// circuit can reach here without passing through the builders that make them.

namespace detail {

struct MPSDispatch {
    static void one_site(MPSState& s, const std::array<Complex128, 4>& U,
                         int qubit, bool unitary) {
        check_qubit(qubit, s.n_qubits, "MPSState::apply_single_qubit_gate");
        s.gate_one_site(U, qubit, unitary);
    }
    static void two_site(MPSState& s, const std::array<Complex128, 16>& U,
                         int q1, int q2, bool unitary) {
        check_qubit(q1, s.n_qubits, "MPSState::apply_two_qubit_gate");
        check_qubit(q2, s.n_qubits, "MPSState::apply_two_qubit_gate");
        check_distinct2(q1, q2, "MPSState::apply_two_qubit_gate");
        s.gate_two_site(U, q1, q2, unitary);
    }
    // A dense fallback rebuilds the chain in canonical gauge whatever the
    // matrix was; only the fidelity bound needs telling.
    static void not_unitary(MPSState& s) { s.fidelity.invalidate(); }
    // Zero exactly the figures absorb_profile folds, so a chain copied from a
    // shared start then absorbed reports only the splits it performed itself.
    static void clear_profile(MPSState& s) {
        s.svd_calls = 0;
        s.svd_nanos = 0;
        s.jacobi_rescues = 0;
        s.gram_fallbacks = 0;
        s.ignored_rejections = 0;
        s.floor_rejected = 0.0;
        s.total_truncation_error = 0.0;
        s.max_verify_resid_excess = 0.0;
    }
};

}  // namespace detail

// Why the dispatcher below cannot apply `inst` to an n_qubits chain under
// `limit`, or "" when it can. run() asks this for every instruction before the
// first gate, after preflight_instructions has checked each gate's operand and
// parameter counts, so what is left is the dense fallback: MCX with more than
// two controls, MCP, PERMUTATION and a UNITARY on three or more qubits have no
// compact form and are applied to the chain's dense amplitudes, which stop at
// the chain's limit. Every other gate the dispatcher reaches has a native or
// decomposed route (CCX, CCZ, CSWAP and RCCX decompose into one- and two-qubit
// gates). The messages are the dispatcher's own, which stay as its guards.
static std::string mps_rejection(const Instruction& inst, int n_qubits, QubitLimit limit) {
    using GT = Instruction::GateType;
    const bool dense =
        (inst.type == GT::MCX && inst.qubits.size() > 3) || inst.type == GT::MCP ||
        inst.type == GT::PERMUTATION ||
        (inst.type == GT::UNITARY && inst.qubits.size() >= 3);
    if (!dense || n_qubits <= max_mps_dense_qubits(limit)) return {};
    if (inst.type == GT::UNITARY) {
        return "a " + std::to_string(inst.qubits.size()) +
               "-qubit UNITARY is applied through the dense fallback, and " +
               dense_limit_text(n_qubits, limit) +
               ". Otherwise decompose the unitary into 1- and 2-qubit factors, "
               "which the chain applies by direct tensor contraction";
    }
    return inst.gate_name() + " is applied through the dense fallback, and " +
           dense_limit_text(n_qubits, limit) +
           ". Otherwise decompose it to 1- and 2-qubit gates or use the "
           "statevector or density-matrix backend";
}

// Helper: apply one instruction to an MPS state.
// Handles RESET, all gate types. MEASURE and BARRIER must NOT be passed here.
static void mps_apply_instruction(MPSState& mps, const Instruction& inst,
                                  std::mt19937_64& rng) {
    using GT = Instruction::GateType;

    if (inst.type == GT::RESET) {
        // Measure, then flip a 1 back to 0: the collapse is the same one a
        // MEASURE performs, so a reset draws from the same distribution.
        const int qubit = inst.qubits[0];
        const int outcome = mps.measure_qubit(qubit, rng);
        if (outcome == 1) {
            // Flipping the collapsed |1> back to |0>. X is built here, so it
            // goes to the chain as unitary, like the other locally-built
            // factors.
            const std::array<Complex128, 4> X_g = {
                Complex128(0,0), Complex128(1,0),
                Complex128(1,0), Complex128(0,0)
            };
            detail::MPSDispatch::one_site(mps, X_g, qubit, true);
        }
        return;
    }

    if (inst.type == GT::PARAM_RX || inst.type == GT::PARAM_RY ||
        inst.type == GT::PARAM_RZ || inst.type == GT::PARAM_P ||
        inst.type == GT::PARAM_U)
        detail::raise_internal("MPSSimulator::run",
            "an unbound parameterised gate reached the chain past the pass before "
            "the first gate, which refuses one");

    // Multi-controlled X reduces to X/CX/CCX for <= 2 controls (native MPS
    // path). Wider MCX and the MCP/PERMUTATION structured ops have no compact
    // MPS form, so they take the same bounded statevector fallback as a 3+ qubit
    // UNITARY (to_statevector -> apply -> rebuild). This keeps e.g. Shor's
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
        if (mps.n_qubits > max_mps_dense_qubits(mps.qubit_limit)) {
            detail::raise_internal("MPSSimulator::run",
                inst.gate_name() + " reached the dense fallback past the pass before "
                "the first gate, which refuses it when " +
                dense_limit_text(mps.n_qubits, mps.qubit_limit));
        }
        charge_dense_fallback(mps, "the dense fallback for " + inst.gate_name());
        auto sv = mps.to_statevector();
        if (inst.type == GT::MCX) {
            std::vector<int> controls(inst.qubits.begin(), inst.qubits.end() - 1);
            gates::apply_mcx(sv, controls, inst.qubits.back());
        } else if (inst.type == GT::MCP) {
            gates::apply_mcp(sv, inst.qubits, inst.params[0]);
        } else {
            gates::apply_permutation(sv, inst.qubits, inst.permutation);
        }
        mps.rebuild_from_statevector(sv);
        return;
    }

    // UNITARY gates store the matrix directly in inst.matrix.
    //
    // 1-qubit and 2-qubit UNITARYs route to the chain's one- and two-site
    // gates through MPSDispatch, judged for the chain's records under the
    // instruction's own policy, which contract the matrix into the affected
    // site tensors (with truncated SVD for the 2-qubit case, and a SWAP
    // network for non-adjacent qubit pairs). Memory
    // cost stays bounded by the bond dimension and is independent of
    // n_qubits — so MPS circuits with arbitrary register widths can now
    // contain user-supplied 1q/2q unitaries.
    //
    // 3+ qubit UNITARYs fall back to the full statevector path, which
    // to_statevector() bounds by the chain's dense limit. A wider register is
    // refused by the pass before the first gate (mps_rejection), naming the
    // gate and its width; the check here only guards against that pass and
    // this dispatcher disagreeing.
    if (inst.type == GT::UNITARY) {
        if (inst.qubits.size() == 1) {
            if (inst.matrix.size() != 4)
                detail::raise_internal("MPSSimulator::run",
                    "a 1-qubit UNITARY without 4 matrix entries reached the chain past "
                    "the pass before the first gate");
            std::array<Complex128, 4> U{
                inst.matrix[0], inst.matrix[1],
                inst.matrix[2], inst.matrix[3]
            };
            detail::MPSDispatch::one_site(
                mps, U, inst.qubits[0],
                detail::gate_keeps_unitarity(U.data(), 2, inst.validation,
                                             mps.unchecked_gates));
            return;
        }
        if (inst.qubits.size() == 2) {
            if (inst.matrix.size() != 16)
                detail::raise_internal("MPSSimulator::run",
                    "a 2-qubit UNITARY without 16 matrix entries reached the chain past "
                    "the pass before the first gate");
            // inst.matrix is qubits[0]-is-LSB, which is apply_two_qubit_gate's
            // own convention, so it is handed over as it stands.
            std::array<Complex128, 16> U{};
            std::copy(inst.matrix.begin(), inst.matrix.end(), U.begin());
            detail::MPSDispatch::two_site(
                mps, U, inst.qubits[0], inst.qubits[1],
                detail::gate_keeps_unitarity(U.data(), 4, inst.validation,
                                             mps.unchecked_gates));
            return;
        }
        if (mps.n_qubits > max_mps_dense_qubits(mps.qubit_limit)) {
            detail::raise_internal("MPSSimulator::run",
                "a " + std::to_string(inst.qubits.size()) +
                "-qubit UNITARY reached the dense fallback past the pass before the "
                "first gate, which refuses it when " +
                dense_limit_text(mps.n_qubits, mps.qubit_limit));
        }
        charge_dense_fallback(mps, "the dense fallback for a " +
                                       std::to_string(inst.qubits.size()) + "-qubit UNITARY");
        auto sv = mps.to_statevector();
        gates::apply_unitary(sv, inst.qubits, inst.matrix, {Validation::Ignore});
        mps.rebuild_from_statevector(sv);
        const std::size_t rows = std::size_t{1} << inst.qubits.size();
        if (!detail::gate_keeps_unitarity(inst.matrix.data(), rows,
                                          inst.validation, mps.unchecked_gates))
            detail::MPSDispatch::not_unitary(mps);
        return;
    }

    if (inst.qubits.size() == 1) {
        auto U = gate2x2(inst);
        detail::MPSDispatch::one_site(mps, U, inst.qubits[0], true);

    } else if (inst.qubits.size() == 2) {
        // gate4x4 builds in its MSB-first frame (first operand = bit 1);
        // apply_two_qubit_gate takes the project's LSB-first order.
        const auto U = exchange_operand_bits(gate4x4(inst));
        detail::MPSDispatch::two_site(mps, U, inst.qubits[0], inst.qubits[1], true);

    } else if (inst.qubits.size() == 3) {
        int q0 = inst.qubits[0], q1 = inst.qubits[1], q2 = inst.qubits[2];

        // The factors below are built here rather than supplied, so every
        // application of them goes to the chain as unitary: their unitarity is
        // a property of this file, and measuring it once per gate per shot
        // would measure the same four constants for the life of the run.
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
        // CX with its control on the first operand, which is bit 0 of the
        // index in the project's LSB-first order: the control set flips the
        // target, so basis states 1 and 3 exchange.
        std::array<Complex128, 16> CX_g{};
        CX_g[0 * 4 + 0] = CX_g[1 * 4 + 3] = CX_g[2 * 4 + 2] = CX_g[3 * 4 + 1] =
            Complex128(1, 0);

        // CCX decomposition: standard 6-CNOT Toffoli
        auto apply_ccx = [&](int c1, int c2, int tgt) {
            detail::MPSDispatch::one_site(mps, H_g, tgt, true);
            detail::MPSDispatch::two_site(mps, CX_g, c2, tgt, true);
            detail::MPSDispatch::one_site(mps, Tdg_g, tgt, true);
            detail::MPSDispatch::two_site(mps, CX_g, c1, tgt, true);
            detail::MPSDispatch::one_site(mps, T_g, tgt, true);
            detail::MPSDispatch::two_site(mps, CX_g, c2, tgt, true);
            detail::MPSDispatch::one_site(mps, Tdg_g, tgt, true);
            detail::MPSDispatch::two_site(mps, CX_g, c1, tgt, true);
            detail::MPSDispatch::one_site(mps, T_g, c2, true);
            detail::MPSDispatch::one_site(mps, T_g, tgt, true);
            detail::MPSDispatch::one_site(mps, H_g, tgt, true);
            detail::MPSDispatch::two_site(mps, CX_g, c1, c2, true);
            detail::MPSDispatch::one_site(mps, T_g, c1, true);
            detail::MPSDispatch::one_site(mps, Tdg_g, c2, true);
            detail::MPSDispatch::two_site(mps, CX_g, c1, c2, true);
        };

        switch (inst.type) {
            case GT::CCX:
                apply_ccx(q0, q1, q2);
                break;
            case GT::CCZ:
                detail::MPSDispatch::one_site(mps, H_g, q2, true);
                apply_ccx(q0, q1, q2);
                detail::MPSDispatch::one_site(mps, H_g, q2, true);
                break;
            case GT::CSWAP:
                detail::MPSDispatch::two_site(mps, CX_g, q2, q1, true);
                apply_ccx(q0, q1, q2);
                detail::MPSDispatch::two_site(mps, CX_g, q2, q1, true);
                break;
            case GT::RCCX:
                detail::MPSDispatch::one_site(mps, H_g, q2, true);
                detail::MPSDispatch::one_site(mps, T_g, q2, true);
                detail::MPSDispatch::two_site(mps, CX_g, q1, q2, true);
                detail::MPSDispatch::one_site(mps, Tdg_g, q2, true);
                detail::MPSDispatch::two_site(mps, CX_g, q0, q2, true);
                detail::MPSDispatch::one_site(mps, T_g, q2, true);
                detail::MPSDispatch::two_site(mps, CX_g, q1, q2, true);
                detail::MPSDispatch::one_site(mps, Tdg_g, q2, true);
                detail::MPSDispatch::one_site(mps, H_g, q2, true);
                break;
            case GT::UNITARY: {
                charge_dense_fallback(mps, "the dense fallback for a 3-qubit UNITARY");
                auto sv = mps.to_statevector();
                gates::apply_unitary(sv, inst.qubits, inst.matrix,
                                     {Validation::Ignore});
                mps.rebuild_from_statevector(sv);
                break;
            }
            default:
                detail::raise_internal("MPSSimulator::run",
                    "3-qubit gate type " + std::to_string(static_cast<int>(inst.type)) +
                    " reached the chain past the pass before the first gate, which "
                    "refuses every type the chain has no route for");
        }
    } else {
        detail::raise_internal("MPSSimulator::run",
            "a " + std::to_string(inst.qubits.size()) + "-qubit " + inst.gate_name() +
            " reached the chain past the pass before the first gate, which refuses "
            "every gate the chain has no route for");
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
// Terminal sampling from a chain centred on qubit 0
//
// With the centre on qubit 0 every other site is right-orthonormal, so the
// environment right of any qubit is the identity, and a shot needs only a row
// vector v on the bond left of the current qubit, carrying the outcomes so
// far:
//
//   w_p = v · A_q[p],   P(p | outcomes so far) = |w_p|² / (|w_0|² + |w_1|²),
//   then v <- w_out / |w_out|.
//
// O(N·chi²) per shot with nothing precomputed, and read-only: the chain is not
// collapsed, so every shot samples the same state. Qubit 0 is the RIGHTMOST
// character (matches measure_sequential and the statevector sampling paths).
// =============================================================================

static std::string mps_sample(const MPSState& state, std::mt19937_64& rng) {
    if (state.open_span() != std::pair<int, int>{0, 0}) {
        detail::raise_internal("MPSSimulator::run",
            "mps_sample: the chain must be centred on qubit 0, so that every "
            "site right of it is right-orthonormal");
    }
    const std::vector<MPSTensor>& tensors = state.tensors();
    const int n = state.n_qubits;
    std::string bits(static_cast<size_t>(n), '0');
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<std::complex<double>> v{std::complex<double>(1.0, 0.0)};
    std::array<std::vector<std::complex<double>>, 2> w;
    for (int q = 0; q < n; ++q) {
        const MPSTensor& T = tensors[static_cast<size_t>(q)];
        const int cl = T.bond_left, cr = T.bond_right;

        std::array<double, 2> probs{0.0, 0.0};
        for (int p = 0; p < 2; ++p) {
            auto& wp = w[static_cast<size_t>(p)];
            wp.assign(static_cast<size_t>(cr), std::complex<double>(0.0, 0.0));
            for (int l = 0; l < cl; ++l) {
                const std::complex<double> vl = v[static_cast<size_t>(l)];
                for (int r = 0; r < cr; ++r) {
                    const Complex128& c = T(l, p, r);
                    wp[static_cast<size_t>(r)] +=
                        vl * std::complex<double>(c.real, c.imag);
                }
            }
            for (const auto& z : wp) probs[static_cast<size_t>(p)] += std::norm(z);
        }

        // The caller refused a chain with no norm, so the first total is the
        // chain's positive norm² and every later one is 1 to rounding.
        const double total = probs[0] + probs[1];
        const int outcome = (dist(rng) < probs[0] / total) ? 0 : 1;
        bits[static_cast<size_t>(n - 1 - q)] = outcome ? '1' : '0';

        const double p_out = probs[static_cast<size_t>(outcome)];
        const double inv = 1.0 / std::sqrt(p_out);
        v = w[static_cast<size_t>(outcome)];
        for (auto& z : v) z *= inv;
    }
    return bits;
}

// =============================================================================
// Choosing the terminal-sampling path
// =============================================================================
//
// Two paths sample the same distribution, the chain's own, so the choice is
// cost alone. The dense path contracts the chain into amplitudes once and then
// draws each shot for next to nothing. The MPS sampler above draws each shot by
// walking the chain. Both costs follow from the bond profile, so the choice is
// made before either runs:
//
//   dense ~ Σ_q 2^(q+1) chi_L(q) chi_R(q)             multiply-accumulates in
//                                                    to_statevector's contraction
//   mps   ~ shots x Σ_q chi_L(q) chi_R(q) x RATIO    the sampler's per-shot walk
//
// RATIO is the measured cost of one unit of sampler work against one dense
// multiply-accumulate: 2.1 ns against 1.2 ns, medians over brickwork circuits at
// n = 14 to 24 and chi = 8 to 64, where each model stayed within a factor of
// two of its median across the grid. Only the ratio enters the rule, and both
// sides are scalar complex loops over the same tensors.
static constexpr double MPS_SAMPLER_COST_RATIO = 1.8;

// The dense path also allocates 2^n amplitudes where the sampler's memory
// follows the bond dimension, so it is taken only while those amplitudes fit
// in one last-level cache instance (hw::llc_bytes(), per instance for the
// reason given there). That caps the allocation sampling can cause at a size
// the machine already holds as working memory, whatever the shot count. When
// detection reports no cache, 4 MiB stands in, at the small end of current
// last-level caches, so unknown hardware is not credited with more cache than
// it is likely to have.
static constexpr std::size_t MPS_DENSE_SAMPLING_LLC_FALLBACK = std::size_t(4) << 20;
static constexpr std::size_t MPS_BYTES_PER_AMPLITUDE = sizeof(Complex128);

// Widest register the dense path may take, detected once. Never above
// ENFORCED_MPS_DENSE_MAX_QUBITS, whatever the chain's qubit_limit: this only
// chooses between two sampling paths and refuses nothing, so lifting the limit
// has no reason to widen it.
static int dense_sampling_max_qubits() {
    static const int cached = [] {
        std::size_t llc = hw::llc_bytes();
        if (llc == 0) llc = MPS_DENSE_SAMPLING_LLC_FALLBACK;
        int n = 0;
        while (n < ENFORCED_MPS_DENSE_MAX_QUBITS &&
               (MPS_BYTES_PER_AMPLITUDE << (n + 1)) <= llc)
            ++n;
        return n;
    }();
    return cached;
}

static bool dense_sampling_is_cheaper(const MPSState& state, int shots) {
    if (state.n_qubits > dense_sampling_max_qubits()) return false;
    double dense = 0.0;
    double walk = 0.0;
    const std::vector<MPSTensor>& t = state.tensors();
    for (int q = 0; q < state.n_qubits; ++q) {
        const double w = static_cast<double>(t[static_cast<size_t>(q)].bond_left) *
                         t[static_cast<size_t>(q)].bond_right;
        walk += w;
        dense += std::ldexp(w, q + 1);
    }
    // At or below, as docs/api/simulators.md states the rule: when the two
    // costs tie, the dense path runs.
    return dense <= MPS_SAMPLER_COST_RATIO * static_cast<double>(shots) * walk;
}

// Whether `chain` is a state a run may hand back: every tensor entry finite
// and the norm above zero.
static bool mps_is_finite_with_norm(const MPSState& chain) {
    for (const MPSTensor& tensor : chain.tensors()) {
        for (const Complex128& entry : tensor.data) {
            if (!is_finite_strict(entry.real) || !is_finite_strict(entry.imag)) return false;
        }
    }
    return chain.norm_sq() > 0.0;
}

// Every setting of the simulator, for a failed run's record.
static void record_options(detail::FailureCollector& failure, const MPSSimulator& sim,
                           int max_bond_dim) {
    using detail::option_value;
    failure.add_option("max_bond_dim", option_value(max_bond_dim));
    failure.add_option("svd_method", option_value(sim.svd_method));
    failure.add_option("svd_rejection", option_value(sim.svd_rejection));
    failure.add_option("svd_accept_gram", option_value(sim.svd_accept_gram));
    failure.add_option("svd_report", option_value(sim.svd_report));
    failure.add_option("canonical_form", option_value(sim.canonical_form));
    failure.add_option("unchecked_gates", option_value(sim.unchecked_gates));
    failure.add_option("qubit_limit", option_value(sim.qubit_limit));
    failure.add_option("max_memory_mb", option_value(sim.max_memory_mb));
}

MPSSimulator::Result MPSSimulator::run(
    const QuantumCircuit& circuit_in, int max_bond_dim,
    int shots, uint64_t seed, const RunPlan& plan
) {
    ScopedWarningFlush flush_on_exit;
    detail::check_circuit_has_qubits(circuit_in.n_qubits, "MPSSimulator::run");
    // Checked here as well as in the MPSState constructor so the message names
    // this call. The argument order differs from StatevectorSimulator::run
    // (circuit, shots, seed), so run(qc, 0, 0) meaning shots is a live way to
    // arrive here.
    detail::check_require(max_bond_dim >= 1, "MPSSimulator::run",
                          "max_bond_dim must be >= 1 (got " +
                              std::to_string(max_bond_dim) + ")");
    Result result(circuit_in.n_qubits);
    // Every chain this run builds starts as a copy of this one, which is the
    // one place the simulator's settings are copied onto a chain: the
    // constructor carries the bond cap and the cutoff, and the rest are
    // assigned here.
    MPSState prototype(circuit_in.n_qubits, max_bond_dim);
    prototype.svd_method = svd_method;
    prototype.svd_rejection = svd_rejection;
    prototype.svd_accept_gram = svd_accept_gram;
    prototype.svd_report = svd_report;
    prototype.canonical_form = canonical_form;
    prototype.unchecked_gates = unchecked_gates;
    prototype.qubit_limit = qubit_limit;
    result.final_state = prototype;

    // Everything the instructions decide on their own, operand indices first,
    // before any state is touched, including every gate the dense fallback
    // would refuse at this width.
    detail::preflight_instructions(
        circuit_in, "MPSSimulator::run",
        [limit = qubit_limit](const Instruction& inst, int n) {
            return mps_rejection(inst, n, limit);
        });
    // Under Repair::Attempt a repaired copy is executed and the caller's
    // circuit is left exactly as it was handed over; Repair::None binds
    // straight to it and nothing is copied.
    std::optional<QuantumCircuit> repaired_storage =
        circuit_in.validated_physical();
    const QuantumCircuit& circuit =
        repaired_storage ? *repaired_storage : circuit_in;

    // Declared outside the try block so the failure path can still reach them:
    // the collector, the harness, and every chain the run may be evolving when
    // it fails (the shared start while its prefix runs, a shot's trajectory,
    // or result.final_state on the single-pass paths).
    detail::FailureCollector failure("MPSSimulator::run", "mps", circuit_in, shots, plan);
    std::optional<detail::ObservationRunner> runner;
    std::optional<MPSState> start_slot;
    std::optional<MPSState> trajectory_slot;
    bool prefix_running = false;

    try {
        auto t_start = std::chrono::high_resolution_clock::now();
        const uint64_t base_seed =
            seed == 0 ? static_cast<uint64_t>(std::random_device{}()) : seed;
        failure.set_seed(base_seed);
        std::mt19937_64 rng(base_seed);

        // Execution strategy (see docs/api/simulators.md, Execution semantics):
        //   1. Terminal-only measurements (no feedforward, nothing acting on a
        //      qubit after it was measured): ONE forward pass, then sample
        //      outcomes from the final state with the qubit -> clbit mapping.
        //   2. Mid-circuit measurement, feedforward, or a RESET that can change
        //      the state, with shots > 0: per-shot trajectories (each stochastic
        //      collapse drawn independently).
        //   3. shots == 0: a single seeded trajectory; classical conditions are
        //      honoured and MEASURE outcomes recorded along the way.
        bool has_measure = false;
        bool has_condition = false;
        int n_clbits = circuit.n_clbits > 0 ? circuit.n_clbits : circuit.n_qubits;
        for (const auto& inst : circuit.instructions) {
            if (inst.type == Instruction::GateType::MEASURE) has_measure = true;
            if (inst.condition_clbit >= 0) has_condition = true;
        }
        // A RESET collapses its qubit, and one pass would collapse it once for
        // every shot. One on a qubit known to be |0> changes nothing and leaves
        // the one-pass path open (detail::trivial_resets).
        const bool has_reset = detail::has_nontrivial_reset(circuit, plan.initial);
        const bool terminal_only = has_measure && !has_condition && !has_reset &&
                                   mps_measures_are_terminal(circuit);
        const bool per_shot =
            shots > 0 && ((has_measure && !terminal_only) || has_reset);

        // One trajectory: honours classical conditions, records MEASURE outcomes.
        // Anchors resolve against the circuit before any state is touched, so an
        // anchor that cannot fire stops the run here.
        runner.emplace(plan, circuit, StateForm::MPS, "MPSSimulator::run");
        runner->set_bundle(&result.observations);
        detail::ObservationRunner* watcher = runner->active() ? &*runner : nullptr;

        // Everything the run allocates is checked here before it is allocated: the
        // evolving chain's growth at every two-site update, its dense fallbacks,
        // and the copies observers take. The chain being evolved is linked to it
        // (MPSState::budget_link) and reports its own size as `state`; chains the
        // run keeps but is not evolving are `held`. An MPS run has no fixed
        // footprint to refuse before the first gate, so the cap is met here.
        detail::RunBudget budget(detail::resolve_memory_cap_bytes(max_memory_mb), 0,
                                 "MPSSimulator::run");
        runner->set_budget(&budget);

        // `first` = the instruction to start from: 0, or the end of a prefix a
        // shared start chain has already run.
        auto run_trajectory = [&](MPSState& state, std::vector<int>& clreg,
                                  std::size_t first) {
            const StateView view(StateForm::MPS, &state, circuit.n_qubits);
            if (watcher) watcher->at_start(view);

            int index = static_cast<int>(first) - 1;
            for (std::size_t i = first; i < circuit.instructions.size(); ++i) {
                const Instruction& inst = circuit.instructions[i];
                using GT = Instruction::GateType;
                ++index;
                state.budget_link.instruction = index;
                state.budget_link.inst = &inst;
                if (watcher) watcher->before_instruction(index, inst, view);
                failure.at_instruction(index, &inst);
                detail::FiringGuard fire(watcher, index, inst, view);
                if (inst.type == GT::BARRIER) continue;
                if (inst.condition_clbit >= 0) {
                    int cv = (inst.condition_clbit < n_clbits)
                             ? clreg[inst.condition_clbit] : 0;
                    if (cv != inst.condition_value) continue;
                }
                if (inst.type == GT::MEASURE) {
                    const int qubit = inst.qubits[0];
                    const int clbit = inst.clbits.empty() ? -1 : inst.clbits[0];
                    const int outcome = state.measure_qubit(qubit, rng);
                    if (clbit >= 0 && clbit < n_clbits) clreg[clbit] = outcome;
                    continue;
                }
                mps_apply_instruction(state, inst, rng);
            }

            if (watcher) watcher->at_end(view, index);
            failure.leave_instructions();
        };

        std::vector<int> clreg(n_clbits, 0);

        if (per_shot) {
            // Per-shot trajectories: every shot runs on its own chain from the
            // initial state so that each collapse is drawn independently (required
            // for mid-circuit measurement, feedforward and reset). Each trajectory
            // then absorbs the profile figures the run has gathered so far and
            // becomes result.final_state, so the chain the caller reads afterwards
            // is the last trajectory in every respect (tensors, cap, cutoff,
            // kernel) and carries the run's totals.
            //
            // Everything before the first instruction that draws or reads a clbit
            // (a MEASURE, a RESET, a conditioned gate) is the same in every shot
            // and draws nothing. With no observer watching and more than one shot,
            // it runs once, seeding included, into a start chain every trajectory
            // copies, and the random stream, so the seeded counts, are those of a
            // rerun. The start's splits are counted once, as performed, and each
            // trajectory adds only its own. An observed run reruns every shot, so
            // each anchor fires once per shot with that shot's index.
            const bool reuse = !watcher && shots > 1;
            std::size_t prefix_end = 0;
            MPSState& start = start_slot.emplace(prototype);
            if (reuse) {
                detail::apply_initial_state(plan, start, "MPSSimulator::run");
                prefix_running = true;
                start.budget_link.attach(&budget);
                budget.set_state(chain_bytes(start.tensors()));
                using GT = Instruction::GateType;
                while (prefix_end < circuit.instructions.size()) {
                    const Instruction& inst = circuit.instructions[prefix_end];
                    if (inst.type == GT::MEASURE || inst.type == GT::RESET ||
                        inst.condition_clbit >= 0)
                        break;
                    start.budget_link.instruction = static_cast<int>(prefix_end);
                    start.budget_link.inst = &inst;
                    failure.at_instruction(static_cast<int>(prefix_end), &inst);
                    if (inst.type != GT::BARRIER) mps_apply_instruction(start, inst, rng);
                    ++prefix_end;
                }
                start.budget_link.detach();
                prefix_running = false;
                failure.leave_instructions();
                result.final_state.absorb_profile(start);
                detail::MPSDispatch::clear_profile(start);
            }

            result.counts.clear();
            runner->begin_run(circuit.n_qubits, shots);
            for (int shot = 0; shot < shots; ++shot) {
                failure.set_shot(shot);
                MPSState& trajectory = trajectory_slot.emplace(reuse ? start : prototype);
                if (!reuse) detail::apply_initial_state(plan, trajectory, "MPSSimulator::run");
                // Held while this shot runs: the shared start and the previous
                // shot's chain, which result.final_state keeps.
                budget.set_held(detail::saturating_add(
                    reuse ? chain_bytes(start.tensors()) : 0,
                    chain_bytes(result.final_state.tensors())));
                trajectory.budget_link.attach(&budget);
                trajectory.budget_link.shot = shot;
                budget.set_state(chain_bytes(trajectory.tensors()));
                clreg.assign(n_clbits, 0);
                runner->begin_shot(shot, clreg);
                run_trajectory(trajectory, clreg, prefix_end);

                std::string bits;
                if (has_measure) {
                    // Build bitstring: clbit 0 is LSB (rightmost), highest clbit
                    // is MSB.
                    bits.assign(static_cast<std::size_t>(n_clbits), '0');
                    for (int c = 0; c < n_clbits; ++c) {
                        if (clreg[c]) bits[n_clbits - 1 - c] = '1';
                    }
                } else {
                    // No MEASURE: the shot is one sample of the whole register
                    // from this trajectory's end state, qubit-indexed, as the
                    // one-pass path samples a circuit with no MEASURE. Read-only,
                    // so the returned chain is the trajectory's end state.
                    trajectory.canonicalize(0);
                    detail::require_norm_to_sample(std::sqrt(trajectory.norm_sq()),
                                                   "MPSSimulator::run");
                    bits = mps_sample(trajectory, rng);
                }
                result.counts[bits]++;
                failure.shot_done();

                trajectory.absorb_profile(result.final_state);
                result.final_state = std::move(trajectory);
                trajectory_slot.reset();
            }
        } else {
            detail::apply_initial_state(plan, result.final_state, "MPSSimulator::run");
            result.final_state.budget_link.attach(&budget);
            budget.set_state(chain_bytes(result.final_state.tensors()));
            runner->begin_run(circuit.n_qubits, 1);
            runner->begin_shot(0, clreg);

            if (shots == 0) {
                // Single seeded trajectory (collapses measures, honours
                // conditions); final_state is one reproducible trajectory.
                run_trajectory(result.final_state, clreg, 0);
            } else {
                // Terminal-only measurements (or none): one forward pass with
                // MEASURE skipped; outcomes are sampled from the final state. That
                // one evolution describes every shot, so the observers fire once.
                const StateView view(StateForm::MPS, &result.final_state,
                                     circuit.n_qubits);
                if (watcher) watcher->at_start(view);

                int index = -1;
                for (const auto& inst : circuit.instructions) {
                    using GT = Instruction::GateType;
                    ++index;
                    result.final_state.budget_link.instruction = index;
                    result.final_state.budget_link.inst = &inst;
                    if (watcher) watcher->before_instruction(index, inst, view);
                    failure.at_instruction(index, &inst);
                    detail::FiringGuard fire(watcher, index, inst, view);
                    if (inst.type == GT::BARRIER || inst.type == GT::MEASURE) continue;
                    if (inst.condition_clbit >= 0) {
                        int cv = (inst.condition_clbit < n_clbits)
                                 ? clreg[inst.condition_clbit] : 0;
                        if (cv != inst.condition_value) continue;
                    }
                    mps_apply_instruction(result.final_state, inst, rng);
                }
                if (watcher) watcher->at_end(view, index);
                failure.leave_instructions();
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

                // Whichever path is cheaper for this chain and shot count; both
                // sample the same distribution (see dense_sampling_is_cheaper).
                // Either path refuses a chain with no norm before its first draw,
                // naming this call rather than the state class underneath it.
                if (dense_sampling_is_cheaper(result.final_state, shots)) {
                    // to_statevector's last two expansion rows, then one of them
                    // beside the Statevector it fills.
                    const std::uint64_t dense = detail::saturating_mul(
                        2, detail::complex_bytes(detail::pow2_saturating(nq)));
                    budget.check_peak(dense, "sampling the final chain through its dense form");
                    auto sv = result.final_state.to_statevector();
                    detail::require_norm_to_sample(sv.norm(), "MPSSimulator::run");
                    auto raw = sv.sample_counts(shots, seed);
                    for (const auto& [bits, cnt] : raw) record(bits, cnt);
                } else {
                    // Sequential MPS sampling from the centre on qubit 0: moving
                    // it there is a gauge change, so the returned chain holds the
                    // same state, and each shot then carries a vector rather than
                    // an environment. O(N * chi^2) per shot.
                    result.final_state.canonicalize(0);
                    detail::require_norm_to_sample(
                        std::sqrt(result.final_state.norm_sq()), "MPSSimulator::run");
                    for (int s = 0; s < shots; ++s) {
                        record(mps_sample(result.final_state, rng), 1);
                    }
                }
            }
        }

        // Flushes every labelled observer into result.observations, before the
        // timer stops: collecting what was observed is part of the work.
        runner->end_run();

        auto t_end = std::chrono::high_resolution_clock::now();
        result.simulation_time_seconds =
            std::chrono::duration<double>(t_end - t_start).count();

        // No state at all, zero or non-finite, is refused rather than returned as
        // an answer. Only a waived physical check or a supplied chain with no norm
        // can produce one.
        if (!mps_is_finite_with_norm(result.final_state)) {
            detail::raise<RuntimeFailure>("MPSSimulator::run",
                "the final state is zero or not finite, so there is no state to return; "
                "a matrix let through by ValidationOptions Warn or Ignore, or a starting "
                "chain with no norm, can do this");
        }

        // The budget ends with this block; the chain handed back must not refer
        // to it.
        result.final_state.budget_link.detach();

    } catch (...) {
        // The failure path, as in the statevector backend: what the run had
        // computed goes into the failed-run record and the failure is
        // rethrown; one before the first instruction is rethrown untouched.
        FailedRun::State state;
        if (failure.work_started()) {
            try {
                std::vector<std::string> notes;
                if (runner) runner->flush_on_failure(notes);
                for (auto& text : notes) failure.note(std::move(text));
                record_options(failure, *this, max_bond_dim);
            } catch (...) {
                // These allocate, and may fail under the memory pressure that
                // failed the run; the run's own failure is what the caller
                // must see.
            }
            if (trajectory_slot) {
                state = std::move(*trajectory_slot);
            } else if (prefix_running && start_slot) {
                state = std::move(*start_slot);
            } else {
                state = std::move(result.final_state);
            }
        }
        failure.fail(std::current_exception(), std::move(result.counts),
                     std::move(result.observations), std::move(state));
    }

    return result;
}

// =============================================================================
// apply_initial_state - MPS
// =============================================================================
// Defined here rather than beside the other three, because the statevector to
// MPS route is the sequential-SVD rebuild above, which is a member of MPSState.

namespace detail {

// A |0...0> chain with every setting of `like`. Re-seeding builds a fresh
// chain, and the constructor carries the bond cap and the weight cutoff but
// not the settings that are assigned (the factorisation, the rejection policy,
// the canonical-form policy, the dense limit). Every branch below that rebuilds a chain goes
// through here, or a caller's choice would be silently replaced by the
// default before the first gate is applied.
static MPSState fresh_chain_like(const MPSState& like) {
    MPSState chain(like.n_qubits, like.max_bond_dim, like.cutoff);
    chain.svd_method = like.svd_method;
    chain.svd_rejection = like.svd_rejection;
    chain.svd_accept_gram = like.svd_accept_gram;
    chain.svd_report = like.svd_report;
    chain.canonical_form = like.canonical_form;
    chain.unchecked_gates = like.unchecked_gates;
    chain.qubit_limit = like.qubit_limit;
    return chain;
}

void apply_initial_state(const RunPlan& plan, MPSState& mps, std::string_view entry_point) {
    const InitialState& initial = plan.initial;
    const int n = mps.n_qubits;

    // A chain supplied as an MPS brings its own settings, which is the one
    // case where the caller has already answered the question; every other
    // branch keeps the ones `mps` arrived with.
    if (initial.is_default()) {
        mps = fresh_chain_like(mps);
        return;
    }

    if (initial.is_basis()) {
        const std::uint64_t index = initial.basis_index();
        if (n < 64 && index >= (std::uint64_t{1} << n)) {
            raise<InvalidArgument>(entry_point,
                "InitialState::basis(" + std::to_string(index) +
                ") is outside a " + std::to_string(n) + " qubit register");
        }
        mps = fresh_chain_like(mps);
        // A product state costs nothing in bond dimension, so this is an X on
        // each set digit rather than a dense build and a factorisation.
        //
        // Applied as unitary with no policy, because this matrix is the
        // library's own: it is exactly unitary by construction, and a check
        // here would judge our constant against a caller's tolerance while
        // overriding the policy they chose.
        const std::array<Complex128, 4> pauli_x = {
            Complex128(0.0, 0.0), Complex128(1.0, 0.0),
            Complex128(1.0, 0.0), Complex128(0.0, 0.0)};
        for (int q = 0; q < n && q < 64; ++q) {
            if ((index >> q) & 1ULL) {
                detail::MPSDispatch::one_site(mps, pauli_x, q, true);
            }
        }
        return;
    }

    if (initial.form() == StateForm::MPS) {
        const auto& source = *static_cast<const MPSState*>(initial.state());
        if (source.n_qubits != n) {
            raise<InvalidArgument>(entry_point,
                "InitialState: the supplied state covers " +
                std::to_string(source.n_qubits) + " qubits, the circuit " +
                std::to_string(n));
        }
        // A chain with no norm is no state at all, and the caller's input
        // decides that, so it is refused here rather than at the first
        // collapse or by the check on the final state. A truncated chain whose
        // norm has fallen below 1 is a state, and runs as given.
        if (!mps_is_finite_with_norm(source)) {
            raise<InvalidArgument>(entry_point,
                "InitialState: the supplied chain has no norm (it is zero or not "
                "finite), so a run cannot start from it");
        }
        mps = source;
        return;
    }

    // Everything else arrives as amplitudes and is factorised. The bond cap is
    // this run's, so a state needing more than it holds is TRUNCATED here
    // rather than refused: that is what running it at this cap means, and the
    // discarded weight is what truncation_error reports.
    // Measured against the SUPPLIED state rather than against the chain, which
    // is the one destination whose footprint is not known before the
    // factorisation runs. It is also the one worth measuring: a compact state
    // expanding into a full 2^n dense array is the surprise this backend exists
    // to avoid, while a caller who already holds those amplitudes has paid for
    // them once and is only handing them over.
    const StateView supplied(initial.form(), initial.state(), n);
    auto produced = produce_initial_state(supplied, StateForm::Statevector,
                                          plan.options, supplied.state_bytes(), entry_point);
    if (!produced) {
        raise<InvalidArgument>(entry_point,
            "InitialState: a " + std::string(to_string(initial.form())) +
            " cannot be turned into the amplitudes this backend factorises "
            "into an MPS, and a run has to start somewhere.");
    }

    const Statevector& sv = *static_cast<const Statevector*>(produced.get());
    if (sv.n_qubits != n) {
        raise<InvalidArgument>(entry_point,
            "InitialState: the supplied state covers " +
            std::to_string(sv.n_qubits) + " qubits, the circuit " +
            std::to_string(n));
    }
    // A fresh chain, so truncation_error() and the fidelity figures on the
    // result report what seeding cost and nothing else.
    // rebuild_from_statevector accumulates by design, which is right mid-run
    // and wrong for the state a run starts from.
    mps = fresh_chain_like(mps);
    mps.rebuild_from_statevector(sv);
}

}  // namespace detail

} // namespace lindblad
