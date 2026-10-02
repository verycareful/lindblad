// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.1 test wave - the qudit MPS keeps its centre (#128, qudit layer).
//
// QuditMPS mirrors the qubit layer at general d: the same open span, the same
// QR and LQ centre steps, the same CanonicalForm rule with d in the rank bound
// min(d chi_L, d chi_R). This file holds it to the same checks as
// test_v11301_canonical_form.cpp, at d = 3 and at an even d = 4 so no claim
// rests on the qubit case, plus what only this layer has: left_canonicalize and
// right_canonicalize, which truncate by SVD and must do so on the state's
// Schmidt values; measure(), which samples without collapsing; and the two
// oracles, which rebuild the chain in place and must keep its settings and
// counters.
//
// References, as in the qubit file: dense amplitudes, their Schmidt
// decomposition through the strict seam, and each site's orthonormality read
// off its entries. The sequential-SVD reference for the canonicalising sweeps
// is the dense constructor, which performs exactly that sweep on the dense
// state, left to right, and on the digit-reversed state for the mirror.

#include <gtest/gtest.h>

#include "v11301_mps_oracle.hpp"

#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;
using v11301::Amplitudes;
using v11301::Cplx;
using v11301::is_canonical;
using v11301::kEps;
using v11301::kSlack;
using v11301::same_sites;

namespace {

using Span = std::pair<int, int>;

std::vector<Complex128> random_gate(int dim, std::uint64_t seed) {
    return v11301::as_c128(v11301::random_unitary(dim, seed));
}

std::size_t pow_size(int base, int exp) {
    std::size_t r = 1;
    for (int i = 0; i < exp; ++i) r *= static_cast<std::size_t>(base);
    return r;
}

Amplitudes apply_dense_1(const Amplitudes& a, int n, int d, int q,
                         const std::vector<Complex128>& U) {
    QuditStatevector sv = v11301::to_qudit_statevector(a, n, d);
    sv.apply_1qudit(q, U, {Validation::Ignore});
    return v11301::dense(sv);
}

Amplitudes apply_dense_2(const Amplitudes& a, int n, int d, int q0, int q1,
                         const std::vector<Complex128>& U) {
    QuditStatevector sv = v11301::to_qudit_statevector(a, n, d);
    sv.apply_2qudit(q0, q1, U, {Validation::Ignore});
    return v11301::dense(sv);
}

QuditMPS chain_of(const std::vector<MPSSiteTensor>& sites, int d, int cap,
                  CanonicalForm form = CanonicalForm::Always,
                  double cutoff = MPS_DEFAULT_CUTOFF) {
    QuditMPS s(static_cast<int>(sites.size()), d, cap, cutoff);
    s.canonical_form = form;
    s.set_tensors(sites);
    return s;
}

double amp_tol(int n, int d, std::size_t ops, double norm) {
    return v11301::amplitude_tol(ops, pow_size(d, n), std::max(norm, 1.0));
}

// As in the qubit file: splits in canonical gauge move the state by at most
// sqrt(splits * truncation_error()) in total.
double truncation_distance(const QuditMPS& s) {
    return std::sqrt(static_cast<double>(s.svd_call_count()) * s.truncation_error());
}

// The state with its digit order reversed: digit q becomes digit n-1-q.
Amplitudes reversed(const Amplitudes& a, int n, int d) {
    Amplitudes out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::size_t rest = i, j = 0;
        for (int q = 0; q < n; ++q) {
            const std::size_t digit = rest % static_cast<std::size_t>(d);
            rest /= static_cast<std::size_t>(d);
            j = j * static_cast<std::size_t>(d) + digit;
        }
        out[j] = a[i];
    }
    return out;
}

class WarningCapture {
public:
    WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
        set_warning_handler([this](const std::string& m) { lines_.push_back(m); });
    }
    ~WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
    }
    std::size_t count() const { return lines_.size(); }

private:
    std::vector<std::string> lines_;
};

// The (n, d, interior bonds) chains every test runs at, one odd d and one even.
struct Shape {
    int n;
    int d;
    std::vector<int> bonds;
};

const std::vector<Shape>& shapes() {
    static const std::vector<Shape> s = {{5, 3, {3, 4, 4, 3}}, {4, 4, {3, 5, 3}}};
    return s;
}

std::string shape_name(const Shape& s) {
    return "n=" + std::to_string(s.n) + " d=" + std::to_string(s.d);
}

}  // namespace

// =============================================================================
// Construction
// =============================================================================

TEST(V11301QuditCanonicalForm, EveryConstructorLeavesACanonicalChain) {
    for (int d : {2, 3, 4}) {
        for (int n = 1; n <= 5; ++n) {
            SCOPED_TRACE("d=" + std::to_string(d) + " n=" + std::to_string(n));
            const QuditMPS fresh(n, d);
            EXPECT_EQ(fresh.open_span(), (Span{0, 0}));
            EXPECT_TRUE(is_canonical(fresh));
            EXPECT_EQ(fresh.svd_call_count(), 0u);

            // The dense constructor's sweep leaves sites 0..n-2 isometries.
            const auto sites = v11301::random_qudit_chain(
                n, d, std::vector<int>(static_cast<std::size_t>(n - 1), 2),
                3000u + static_cast<std::uint64_t>(n * 10 + d));
            const Amplitudes want = v11301::dense(chain_of(sites, d, 64));
            const QuditMPS built(v11301::to_qudit_statevector(want, n, d), 64);
            EXPECT_EQ(built.open_span(), (Span{n - 1, n - 1}));
            EXPECT_TRUE(is_canonical(built));
            EXPECT_EQ(built.svd_call_count(), static_cast<std::size_t>(n - 1));
            const double norm = std::sqrt(v11301::norm_sq(want));
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(built), want),
                      amp_tol(n, d, n, norm) + truncation_distance(built));
        }
    }
}

TEST(V11301QuditCanonicalForm, ConstructorsRefuseWhatNoChainCanBe) {
    EXPECT_THROW(QuditMPS(0, 3), std::invalid_argument);
    EXPECT_THROW(QuditMPS(3, 1), std::invalid_argument);
    EXPECT_THROW(QuditMPS(3, 3, 0), std::invalid_argument);
    EXPECT_THROW(QuditMPS(QuditStatevector(2, 3), 0), std::invalid_argument);
}

// =============================================================================
// The invariant, operation by operation
// =============================================================================

TEST(V11301QuditCanonicalForm, SingleQuditGatesLeaveTheSpanWhereItIs) {
    for (const Shape& sh : shapes()) {
        SCOPED_TRACE(shape_name(sh));
        const auto sites = v11301::random_qudit_chain(sh.n, sh.d, sh.bonds, 3100);
        std::uint64_t seed = 3110;
        for (int centre = -1; centre < sh.n; ++centre) {
            SCOPED_TRACE(centre < 0 ? std::string("set_tensors span")
                                    : "centre on " + std::to_string(centre));
            QuditMPS s = chain_of(sites, sh.d, 256);
            if (centre >= 0) s.canonicalize(centre);
            const Span span = s.open_span();
            Amplitudes want = v11301::dense(s);
            for (int q = 0; q < sh.n; ++q) {
                const auto U = random_gate(sh.d, seed++);
                s.apply_1qudit(q, U);
                want = apply_dense_1(want, sh.n, sh.d, q, U);
                ASSERT_EQ(s.open_span(), span) << "a gate on qudit " << q;
                ASSERT_TRUE(is_canonical(s)) << "after a gate on qudit " << q;
            }
            const double norm = std::sqrt(v11301::norm_sq(want));
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
                      amp_tol(sh.n, sh.d, static_cast<std::size_t>(sh.n), norm));
        }
    }
}

TEST(V11301QuditCanonicalForm, EveryTwoQuditGateLeavesASingleSiteCentre) {
    // Adjacent through both entry points, and every ordered pair through
    // apply_2qudit, which routes non-adjacent pairs by a SWAP chain. The split
    // count is one for the gate plus two per site crossed.
    for (const Shape& sh : shapes()) {
        SCOPED_TRACE(shape_name(sh));
        const int cap = static_cast<int>(pow_size(sh.d, sh.n / 2));
        QuditMPS s = chain_of(v11301::random_qudit_chain(sh.n, sh.d, sh.bonds, 3200),
                              sh.d, cap);
        s.canonicalize(1);
        Amplitudes want = v11301::dense(s);
        std::uint64_t seed = 3210;
        std::size_t ops = 0;
        const int d2 = sh.d * sh.d;

        for (int q = 0; q + 1 < sh.n; ++q) {
            SCOPED_TRACE("apply_2qudit_adjacent(" + std::to_string(q) + ")");
            const auto U = random_gate(d2, seed++);
            const std::size_t calls = s.svd_call_count();
            s.apply_2qudit_adjacent(q, U);
            want = apply_dense_2(want, sh.n, sh.d, q, q + 1, U);
            ++ops;
            const auto [lo, hi] = s.open_span();
            EXPECT_EQ(lo, hi);
            EXPECT_GE(lo, q);
            EXPECT_LE(lo, q + 1);
            EXPECT_EQ(s.svd_call_count() - calls, 1u);
            ASSERT_TRUE(is_canonical(s));
        }
        for (int q0 = 0; q0 < sh.n; ++q0) {
            for (int q1 = 0; q1 < sh.n; ++q1) {
                if (q0 == q1) continue;
                SCOPED_TRACE("apply_2qudit(" + std::to_string(q0) + ", " +
                             std::to_string(q1) + ")");
                const auto U = random_gate(d2, seed++);
                const std::size_t calls = s.svd_call_count();
                s.apply_2qudit(q0, q1, U);
                want = apply_dense_2(want, sh.n, sh.d, q0, q1, U);
                ++ops;
                const auto [lo, hi] = s.open_span();
                EXPECT_EQ(lo, hi);
                EXPECT_GE(lo, std::min(q0, q1));
                EXPECT_LE(lo, std::max(q0, q1));
                const std::size_t crossed =
                    static_cast<std::size_t>(std::abs(q1 - q0) - 1);
                EXPECT_EQ(s.svd_call_count() - calls, 1 + 2 * crossed);
                ASSERT_TRUE(is_canonical(s));
            }
        }
        const double norm = std::sqrt(v11301::norm_sq(want));
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
                  amp_tol(sh.n, sh.d, 3 * ops, norm) + truncation_distance(s));
    }
}

TEST(V11301QuditCanonicalForm, AutoReadsTheRankBoundWithD) {
    // The block at (1, 2) of a chain with bonds of 2 has rank bound
    // min(d * 2, d * 2) = 2d. A cap of 2d cannot bind, so Auto splits in place
    // and the outer sites are untouched; a cap of 2d - 1 can, so it focuses.
    // A cutoff above the default makes Auto focus whatever the cap.
    for (int d : {3, 4}) {
        SCOPED_TRACE("d=" + std::to_string(d));
        const int n = 5;
        const auto sites = v11301::random_qudit_chain(n, d, {2, 2, 2, 2}, 3300);
        const auto U = random_gate(d * d, 3301);
        const int bound = std::min(d * sites[1].chi_L, d * sites[2].chi_R);
        ASSERT_EQ(bound, 2 * d);

        QuditMPS at_bound = chain_of(sites, d, bound, CanonicalForm::Auto);
        at_bound.apply_2qudit_adjacent(1, U);
        EXPECT_EQ(at_bound.open_span(), (Span{0, n - 1}));
        for (int q : {0, 3, 4}) {
            EXPECT_TRUE(v11301::same_site(at_bound.tensors()[static_cast<std::size_t>(q)],
                                          sites[static_cast<std::size_t>(q)]))
                << "Auto rewrote site " << q << " outside the block";
        }

        QuditMPS below = chain_of(sites, d, bound - 1, CanonicalForm::Auto);
        below.apply_2qudit_adjacent(1, U);
        const auto [lo, hi] = below.open_span();
        EXPECT_EQ(lo, hi) << "a cap that can bind makes Auto focus first";
        EXPECT_TRUE(is_canonical(below));

        QuditMPS raised = chain_of(sites, d, 256, CanonicalForm::Auto,
                                   std::nextafter(MPS_DEFAULT_CUTOFF, 1.0));
        raised.apply_2qudit_adjacent(1, U);
        const auto [rlo, rhi] = raised.open_span();
        EXPECT_EQ(rlo, rhi) << "a raised cutoff makes Auto focus first";
        EXPECT_TRUE(is_canonical(raised));
    }
}

TEST(V11301QuditCanonicalForm, CanonicalizeMovesTheCentreAndNothingElse) {
    for (const Shape& sh : shapes()) {
        SCOPED_TRACE(shape_name(sh));
        QuditMPS base = chain_of(v11301::random_qudit_chain(sh.n, sh.d, sh.bonds, 3400),
                                 sh.d, 2);
        base.apply_2qudit_adjacent(1, random_gate(sh.d * sh.d, 3401));
        ASSERT_LT(*base.fidelity_estimate(), 1.0) << "the capped split truncated nothing";
        const Amplitudes want = v11301::dense(base);
        const double norm = std::sqrt(v11301::norm_sq(want));
        const v11301::Profile before = v11301::profile_of(base);

        for (int c = 0; c < sh.n; ++c) {
            SCOPED_TRACE("canonicalize(" + std::to_string(c) + ")");
            QuditMPS s = base;
            s.canonicalize(c);
            EXPECT_EQ(s.open_span(), (Span{c, c}));
            EXPECT_TRUE(is_canonical(s));
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
                      amp_tol(sh.n, sh.d, 2 * static_cast<std::size_t>(sh.n), norm));
            EXPECT_TRUE(v11301::profile_of(s) == before)
                << "before " << v11301::describe(before) << ", after "
                << v11301::describe(v11301::profile_of(s));
            const auto settled = s.tensors();
            s.canonicalize(c);
            EXPECT_TRUE(same_sites(s.tensors(), settled));
        }
        QuditMPS s = base;
        EXPECT_THROW(s.canonicalize(-1), std::out_of_range);
        EXPECT_THROW(s.canonicalize(sh.n), std::out_of_range);
        EXPECT_TRUE(same_sites(s.tensors(), base.tensors()));
    }
}

TEST(V11301QuditCanonicalForm, CanonicalizeTrimsABondWiderThanItsRank) {
    // A QR step leaves the bond at min(rows, cols) of the site it factors.
    const int d = 3;
    const auto sites = v11301::random_qudit_chain(3, d, {7, 7}, 3500);
    const Amplitudes want = v11301::dense(chain_of(sites, d, 64));
    const double norm = std::sqrt(v11301::norm_sq(want));

    QuditMPS right = chain_of(sites, d, 64);
    right.canonicalize(2);
    EXPECT_EQ(right.tensors()[0].chi_R, std::min(d * 1, 7));
    EXPECT_EQ(right.tensors()[1].chi_R, std::min(d * std::min(d * 1, 7), 7));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(right), want), amp_tol(3, d, 4, norm));
    EXPECT_TRUE(is_canonical(right));

    QuditMPS left = chain_of(sites, d, 64);
    left.canonicalize(0);
    EXPECT_EQ(left.tensors()[1].chi_R, std::min(7, d * 1));
    EXPECT_EQ(left.tensors()[0].chi_R, std::min(7, d * std::min(7, d * 1)));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(left), want), amp_tol(3, d, 4, norm));
    EXPECT_TRUE(is_canonical(left));
}

TEST(V11301QuditCanonicalForm, NormAndNormalizeReadThroughAnySpan) {
    for (const Shape& sh : shapes()) {
        SCOPED_TRACE(shape_name(sh));
        const auto sites = v11301::random_qudit_chain(sh.n, sh.d, sh.bonds, 3600);
        for (int centre = -1; centre < sh.n; ++centre) {
            SCOPED_TRACE(centre < 0 ? std::string("set_tensors span")
                                    : "centre on " + std::to_string(centre));
            QuditMPS s = chain_of(sites, sh.d, 256);
            if (centre >= 0) s.canonicalize(centre);
            const Amplitudes a = v11301::dense(s);
            const double want = v11301::norm_sq(a);
            const double tol = kSlack * static_cast<double>(4 * sh.n + a.size()) * kEps * want;
            EXPECT_NEAR(s.norm_sq(), want, tol);

            const auto before = s.tensors();
            const Span span = s.open_span();
            const v11301::Profile prof = v11301::profile_of(s);
            s.normalize();
            EXPECT_EQ(s.open_span(), span);
            EXPECT_TRUE(v11301::profile_of(s) == prof);
            for (int q = 0; q < sh.n; ++q) {
                const bool same = v11301::same_site(s.tensors()[static_cast<std::size_t>(q)],
                                                    before[static_cast<std::size_t>(q)]);
                EXPECT_EQ(same, q != span.first) << "site " << q;
            }
            EXPECT_TRUE(is_canonical(s));
            EXPECT_NEAR(s.norm_sq(), 1.0, kSlack * static_cast<double>(4 * sh.n) * kEps);
            Amplitudes scaled = a;
            for (auto& z : scaled) z /= std::sqrt(want);
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), scaled),
                      amp_tol(sh.n, sh.d, 2 * static_cast<std::size_t>(sh.n), 1.0));
        }
    }
}

TEST(V11301QuditCanonicalForm, MeasureSamplesWithoutChangingTheState) {
    // measure() moves the centre to qudit 0 and walks the chain; it is a gauge
    // change, so the state, the counters and the fidelity figures are all
    // unchanged, and the same seed draws the same digits.
    for (const Shape& sh : shapes()) {
        SCOPED_TRACE(shape_name(sh));
        QuditMPS s = chain_of(v11301::random_qudit_chain(sh.n, sh.d, sh.bonds, 3700),
                              sh.d, 3);
        s.apply_2qudit(0, sh.n - 1, random_gate(sh.d * sh.d, 3701));
        ASSERT_LT(*s.fidelity_estimate(), 1.0);
        const Amplitudes want = v11301::dense(s);
        const double norm = std::sqrt(v11301::norm_sq(want));
        const v11301::Profile before = v11301::profile_of(s);

        const std::vector<int> digits = s.measure(3702);
        ASSERT_EQ(digits.size(), static_cast<std::size_t>(sh.n));
        for (int x : digits) {
            EXPECT_GE(x, 0);
            EXPECT_LT(x, sh.d);
        }
        EXPECT_EQ(s.open_span(), (Span{0, 0}));
        EXPECT_TRUE(is_canonical(s));
        EXPECT_TRUE(v11301::profile_of(s) == before)
            << "before " << v11301::describe(before) << ", after "
            << v11301::describe(v11301::profile_of(s));
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
                  amp_tol(sh.n, sh.d, 2 * static_cast<std::size_t>(sh.n), norm));
        EXPECT_EQ(s.measure(3702), digits) << "the same seed drew different digits";
    }
}

TEST(V11301QuditCanonicalForm, MeasureQuditCollapsesAtTheCentre) {
    for (const Shape& sh : shapes()) {
        SCOPED_TRACE(shape_name(sh));
        const auto sites = v11301::random_qudit_chain(sh.n, sh.d, sh.bonds, 3800);
        for (int q = 0; q < sh.n; ++q) {
            SCOPED_TRACE("measure qudit " + std::to_string(q));
            QuditMPS s = chain_of(sites, sh.d, 256);
            std::mt19937_64 rng(3800 + static_cast<std::uint64_t>(q));
            const int x = s.measure_qudit(q, rng);
            EXPECT_GE(x, 0);
            EXPECT_LT(x, sh.d);
            EXPECT_EQ(s.open_span(), (Span{q, q}));
            EXPECT_TRUE(is_canonical(s));
            EXPECT_NEAR(s.norm_sq(), 1.0, kSlack * static_cast<double>(4 * sh.n) * kEps);
            EXPECT_FALSE(s.fidelity_estimate().has_value());
            EXPECT_FALSE(s.fidelity_lower_bound().has_value());
        }
    }
}

TEST(V11301QuditCanonicalForm, ANonFiniteSiteReachesTheLadderThroughTheCentreSteps) {
    const int d = 3;
    QuditMPS s(3, d, 8);
    std::vector<Complex128> nan_gate(9, Complex128(0.0, 0.0));
    nan_gate[0] = Complex128(quiet_nan_strict(), 0.0);
    nan_gate[4] = Complex128(1.0, 0.0);
    nan_gate[8] = Complex128(1.0, 0.0);
    s.apply_1qudit(0, nan_gate, {Validation::Ignore});
    ASSERT_FALSE(is_finite_strict(s.tensors()[0].data[0].real))
        << "the gate under Ignore did not carry the NaN into site 0";
    EXPECT_THROW(s.apply_2qudit_adjacent(1, random_gate(d * d, 3900)),
                 std::runtime_error);
}

// =============================================================================
// #128 at general d, and the SVD sweeps
// =============================================================================

TEST(V11301QuditSchmidtTruncation, ACappedSplitFromADistortedGaugeKeepsTheLeadingSchmidtDirections) {
    // As the qubit reproducer: G G^-1 conditioned at 1e3 on the two bonds
    // around the block, a random d^2 x d^2 gate on (2, 3) at a cap of 2, and
    // the result must be the projection of the gated state onto its two
    // leading Schmidt directions across the cut 2 | 3, under both policies. A
    // split of the distorted block itself must land measurably elsewhere.
    for (const Shape& sh : {Shape{6, 3, {3, 4, 4, 4, 3}}, Shape{5, 4, {4, 4, 4, 4}}}) {
        SCOPED_TRACE(shape_name(sh));
        const int n = sh.n, d = sh.d, q = 2, keep = 2;
        auto sites = v11301::random_qudit_chain(n, d, sh.bonds, 4000);
        v11301::distort_bond(sites, q - 1, 1e3, 4001);
        v11301::distort_bond(sites, q + 1, 1e3, 4002);
        const auto U = random_gate(d * d, 4003);

        const Amplitudes before = v11301::dense(chain_of(sites, d, 256));
        const Amplitudes gated = apply_dense_2(before, n, d, q, q + 1, U);
        const v11301::Schmidt sch = v11301::schmidt(gated, d, n, q + 1);
        const Amplitudes optimal = v11301::project_top(gated, d, n, q + 1, keep);
        double kept = 0.0, total = 0.0;
        for (std::size_t i = 0; i < sch.sigma.size(); ++i) {
            const double w = sch.sigma[i] * sch.sigma[i];
            total += w;
            if (static_cast<int>(i) < keep) kept += w;
        }
        const double norm = std::sqrt(total);
        const double tol = amp_tol(n, d, 4 * static_cast<std::size_t>(n), norm);
        const double gap = sch.sigma[static_cast<std::size_t>(keep - 1)] -
                           sch.sigma[static_cast<std::size_t>(keep)];
        ASSERT_GT(gap, 1e3 * tol);
        const double proj_tol = tol * (1.0 + sch.sigma[0] / gap);

        // The control: a split of the distorted block itself, done here
        // through the seam since a binding cap makes both policies focus,
        // keeps the block's leading directions, which the distortion has
        // moved away from the state's.
        {
            const MPSSiteTensor& A = sites[static_cast<std::size_t>(q)];
            const MPSSiteTensor& B = sites[static_cast<std::size_t>(q + 1)];
            const int cl = A.chi_L, cm = A.chi_R, cr = B.chi_R;
            const int rows = d * cl, cols = d * cr;
            std::vector<Cplx> theta(static_cast<std::size_t>(rows) * cols, Cplx(0.0, 0.0));
            for (int so0 = 0; so0 < d; ++so0)
                for (int so1 = 0; so1 < d; ++so1)
                    for (int l = 0; l < cl; ++l)
                        for (int r = 0; r < cr; ++r) {
                            Cplx acc(0.0, 0.0);
                            for (int si0 = 0; si0 < d; ++si0)
                                for (int si1 = 0; si1 < d; ++si1) {
                                    Cplx block(0.0, 0.0);
                                    for (int m = 0; m < cm; ++m)
                                        block += v11301::to_std(A.at(si0, l, m)) *
                                                 v11301::to_std(B.at(si1, m, r));
                                    const std::size_t row =
                                        static_cast<std::size_t>(so1 * d + so0);
                                    const std::size_t col =
                                        static_cast<std::size_t>(si1 * d + si0);
                                    acc += v11301::to_std(
                                               U[row * static_cast<std::size_t>(d * d) + col]) *
                                           block;
                                }
                            theta[static_cast<std::size_t>(so1 * cr + r) * rows +
                                  static_cast<std::size_t>(so0 * cl + l)] = acc;
                        }
            const int k = std::min(rows, cols);
            std::vector<Cplx> Um(static_cast<std::size_t>(rows) * k),
                Vm(static_cast<std::size_t>(cols) * k);
            std::vector<double> S(static_cast<std::size_t>(k));
            ASSERT_TRUE(detail::svd_thin(theta.data(), rows, cols,
                                         detail::MatrixOrder::ColMajor,
                                         SVDMethod::EigenJacobi, Um.data(), S.data(),
                                         Vm.data()));
            auto split = sites;
            MPSSiteTensor A2(d, cl, keep), B2(d, keep, cr);
            for (int s0 = 0; s0 < d; ++s0)
                for (int l = 0; l < cl; ++l)
                    for (int j = 0; j < keep; ++j)
                        A2.at(s0, l, j) = v11301::to_c128(
                            Um[static_cast<std::size_t>(j) * rows +
                               static_cast<std::size_t>(s0 * cl + l)] *
                            S[static_cast<std::size_t>(j)]);
            for (int s1 = 0; s1 < d; ++s1)
                for (int j = 0; j < keep; ++j)
                    for (int r = 0; r < cr; ++r)
                        B2.at(s1, j, r) = v11301::to_c128(std::conj(
                            Vm[static_cast<std::size_t>(j) * cols +
                               static_cast<std::size_t>(s1 * cr + r)]));
            split[static_cast<std::size_t>(q)] = std::move(A2);
            split[static_cast<std::size_t>(q + 1)] = std::move(B2);
            QuditMPS naive(n, d, 256);
            naive.set_tensors(split);
            ASSERT_GT(v11301::max_abs_diff(v11301::dense(naive), optimal), 1e3 * proj_tol)
                << "the distorted gauge does not change which directions a split "
                   "keeps, so this fixture cannot detect #128";
        }

        for (CanonicalForm form : {CanonicalForm::Always, CanonicalForm::Auto}) {
            SCOPED_TRACE(to_string(form));
            QuditMPS s = chain_of(sites, d, keep, form);
            const double norm_before = s.norm_sq();
            s.apply_2qudit_adjacent(q, U);
            EXPECT_EQ(s.svd_call_count(), 1u);
            EXPECT_EQ(s.tensors()[static_cast<std::size_t>(q)].chi_R, keep);
            EXPECT_TRUE(is_canonical(s));
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), optimal), proj_tol);

            const double weight_tol = kSlack * static_cast<double>(gated.size()) * kEps * total;
            EXPECT_NEAR(s.truncation_error(), total - kept, weight_tol);
            EXPECT_NEAR(norm_before - s.norm_sq(), s.truncation_error(), weight_tol);
            ASSERT_TRUE(s.fidelity_estimate().has_value());
            ASSERT_TRUE(s.fidelity_lower_bound().has_value());
            EXPECT_NEAR(*s.fidelity_estimate(), kept / total, weight_tol / total);
            EXPECT_NEAR(*s.fidelity_lower_bound(), kept / total, weight_tol / total);
        }
    }
}

TEST(V11301QuditSchmidtTruncation, LeftCanonicalizeIsTheSequentialSvdOfTheState) {
    // From a distorted chain at a cap that binds, left_canonicalize must give
    // the state the dense constructor's left-to-right sweep gives, which
    // truncates each bond on the Schmidt values of what the sweep has kept so
    // far. Both sweeps count their splits and fold them into the figures
    // alike, so the figures and the discarded weight match too.
    for (const Shape& sh : shapes()) {
        SCOPED_TRACE(shape_name(sh));
        const int cap = 2;
        auto sites = v11301::random_qudit_chain(sh.n, sh.d, sh.bonds, 4100);
        for (int b = 0; b + 1 < sh.n; ++b)
            v11301::distort_bond(sites, b, 1e2, 4101 + static_cast<std::uint64_t>(b));
        QuditMPS s = chain_of(sites, sh.d, cap);
        const Amplitudes original = v11301::dense(s);
        const std::size_t calls = s.svd_call_count();

        s.left_canonicalize();
        const QuditMPS ref(v11301::to_qudit_statevector(original, sh.n, sh.d), cap);

        EXPECT_EQ(s.open_span(), (Span{sh.n - 1, sh.n - 1}));
        EXPECT_TRUE(is_canonical(s));
        EXPECT_EQ(s.svd_call_count() - calls, static_cast<std::size_t>(sh.n - 1));
        for (int b = 0; b + 1 < sh.n; ++b)
            EXPECT_LE(s.tensors()[static_cast<std::size_t>(b)].chi_R, cap);

        const double norm = std::sqrt(v11301::norm_sq(original));
        const double tol = amp_tol(sh.n, sh.d, 4 * static_cast<std::size_t>(sh.n), norm);
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), v11301::dense(ref)), tol);
        EXPECT_NEAR(s.truncation_error(), ref.truncation_error(), tol * norm);
        ASSERT_TRUE(s.fidelity_estimate().has_value());
        EXPECT_NEAR(*s.fidelity_estimate(), *ref.fidelity_estimate(), tol);
        EXPECT_NEAR(*s.fidelity_lower_bound(), *ref.fidelity_lower_bound(), tol);
        EXPECT_LT(*s.fidelity_estimate(), 1.0) << "the cap did not bind";
    }
}

TEST(V11301QuditSchmidtTruncation, RightCanonicalizeIsTheMirrorSweep) {
    // The same from the right: the sweep equals the dense constructor's sweep
    // over the digit-reversed state, reversed back.
    for (const Shape& sh : shapes()) {
        SCOPED_TRACE(shape_name(sh));
        const int cap = 2;
        auto sites = v11301::random_qudit_chain(sh.n, sh.d, sh.bonds, 4200);
        for (int b = 0; b + 1 < sh.n; ++b)
            v11301::distort_bond(sites, b, 1e2, 4201 + static_cast<std::uint64_t>(b));
        QuditMPS s = chain_of(sites, sh.d, cap);
        const Amplitudes original = v11301::dense(s);

        s.right_canonicalize();
        const QuditMPS ref(
            v11301::to_qudit_statevector(reversed(original, sh.n, sh.d), sh.n, sh.d), cap);

        EXPECT_EQ(s.open_span(), (Span{0, 0}));
        EXPECT_TRUE(is_canonical(s));
        const double norm = std::sqrt(v11301::norm_sq(original));
        const double tol = amp_tol(sh.n, sh.d, 4 * static_cast<std::size_t>(sh.n), norm);
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s),
                                       reversed(v11301::dense(ref), sh.n, sh.d)),
                  tol);
        EXPECT_NEAR(s.truncation_error(), ref.truncation_error(), tol * norm);
        EXPECT_NEAR(*s.fidelity_estimate(), *ref.fidelity_estimate(), tol);
        EXPECT_LT(*s.fidelity_estimate(), 1.0) << "the cap did not bind";
    }
}

// =============================================================================
// set_tensors
// =============================================================================

TEST(V11301QuditSetTensors, EveryRuleRefusesAndLeavesTheChainUntouched) {
    const int n = 4, d = 3;
    const auto good = v11301::random_qudit_chain(n, d, {2, 3, 2}, 4300);
    const double nan = quiet_nan_strict();
    const double inf = std::numeric_limits<double>::infinity();

    struct Bad {
        std::string what;
        std::string message;
        std::function<void(std::vector<MPSSiteTensor>&)> mutate;
    };
    const std::vector<Bad> cases = {
        {"no sites", "expected 4 site tensors, got 0", [](auto& s) { s.clear(); }},
        {"one site short", "expected 4 site tensors, got 3", [](auto& s) { s.pop_back(); }},
        {"one site over", "expected 4 site tensors, got 5",
         [](auto& s) { s.push_back(MPSSiteTensor(3, 1, 1)); }},
        {"the wrong physical dimension",
         "site 1 has physical dimension 2 where the chain's is 3",
         [](auto& s) { s[1] = MPSSiteTensor(2, s[1].chi_L, s[1].chi_R); }},
        {"a zero left bond", "site 2 has a bond below 1", [](auto& s) { s[2].chi_L = 0; }},
        {"a negative right bond", "site 1 has a bond below 1",
         [](auto& s) { s[1].chi_R = -1; }},
        {"a left end bond above 1", "the left end bond must be 1",
         [](auto& s) { s[0] = MPSSiteTensor(3, 2, s[0].chi_R); }},
        {"a right end bond above 1", "the right end bond must be 1",
         [](auto& s) { s[3] = MPSSiteTensor(3, s[3].chi_L, 2); }},
        {"neighbouring bonds that disagree",
         "site 2 has left bond 4 where its neighbour's right bond is 3",
         [](auto& s) { s[2] = MPSSiteTensor(3, 4, s[2].chi_R); }},
        {"a data array one short", "site 1 holds 17 entries where its shape needs 18",
         [](auto& s) { s[1].data.pop_back(); }},
        {"a NaN real part", "site 2 holds a non-finite entry",
         [nan](auto& s) { s[2].data[1].real = nan; }},
        {"a NaN imaginary part", "site 0 holds a non-finite entry",
         [nan](auto& s) { s[0].data[2].imag = nan; }},
        {"an infinity", "site 3 holds a non-finite entry",
         [inf](auto& s) { s[3].data[0].real = -inf; }},
        {"two violations, the first named", "site 1 holds 17 entries",
         [nan](auto& s) { s[1].data.pop_back(); s[2].data[0].real = nan; }},
    };

    for (const auto& bad : cases) {
        SCOPED_TRACE(bad.what);
        QuditMPS s(n, d, 8);
        s.apply_2qudit_adjacent(1, random_gate(d * d, 4301));
        const auto sites_before = s.tensors();
        const Span span_before = s.open_span();
        const v11301::Profile prof = v11301::profile_of(s);
        auto sites = good;
        bad.mutate(sites);
        try {
            s.set_tensors(sites);
            ADD_FAILURE() << "accepted a chain with " << bad.what;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(bad.message), std::string::npos)
                << "the refusal does not name the violation: " << e.what();
            EXPECT_NE(std::string(e.what()).find("QuditMPS::set_tensors"), std::string::npos)
                << e.what();
        }
        EXPECT_TRUE(same_sites(s.tensors(), sites_before));
        EXPECT_EQ(s.open_span(), span_before);
        EXPECT_TRUE(v11301::profile_of(s) == prof);
    }
}

TEST(V11301QuditSetTensors, AValidChainOpensTheSpanAndResetsOnlyTheFigures) {
    const int n = 4, d = 3;
    QuditMPS s(n, d, 2);
    s.apply_2qudit(0, 3, random_gate(d * d, 4400));
    s.apply_2qudit(1, 2, random_gate(d * d, 4401));
    ASSERT_LT(*s.fidelity_estimate(), 1.0);
    const v11301::Profile before = v11301::profile_of(s);

    const auto sites = v11301::random_qudit_chain(n, d, {2, 3, 2}, 4402);
    s.set_tensors(sites);
    EXPECT_TRUE(same_sites(s.tensors(), sites));
    EXPECT_EQ(s.open_span(), (Span{0, n - 1}));
    EXPECT_EQ(s.fidelity_estimate(), std::optional<double>(1.0));
    EXPECT_EQ(s.fidelity_lower_bound(), std::optional<double>(1.0));
    EXPECT_EQ(s.svd_call_count(), before.svd_calls);
    EXPECT_EQ(s.truncation_error(), before.truncation);
    EXPECT_EQ(s.svd_time_ns(), before.nanos);
    EXPECT_EQ(s.max_bond_dim, 2);
}

// =============================================================================
// The oracles rebuild in place
// =============================================================================

TEST(V11301QuditOracles, APhaseOracleKeepsTheSettingsAndAccumulatesTheCounters) {
    // The oracle goes dense and rebuilds. It must do so as this object, with
    // its own kernel, rescue choice, policy, cap and cutoff, adding its splits
    // to the counters and figures it already had rather than starting over,
    // and leaving the centre where the sweep ends.
    const int n = 4, d = 3, cap = 5;
    const double cutoff = 2.0 * MPS_DEFAULT_CUTOFF;
    QuditMPS s(n, d, cap, cutoff);
    s.svd_method = SVDMethod::EigenBDC;
    s.svd_rejection = SvdRejection::Throw;
    s.svd_accept_gram = true;
    s.svd_report = SvdReport::Silent;
    s.canonical_form = CanonicalForm::Auto;
    s.apply_1qudit(0, random_gate(d, 4500));
    s.apply_2qudit(0, 3, random_gate(d * d, 4501));
    s.apply_2qudit(1, 2, random_gate(d * d, 4502));
    const v11301::Profile before = v11301::profile_of(s);
    ASSERT_GT(before.svd_calls, 0u);
    const Amplitudes prior = v11301::dense(s);

    const auto phase = [d](const std::vector<int>& x) {
        double angle = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i)
            angle += (PI / (2.0 + static_cast<double>(i))) * x[i] / d;
        return Complex128(std::cos(angle), std::sin(angle));
    };
    s.apply_phase_oracle(phase);

    EXPECT_EQ(s.svd_method, SVDMethod::EigenBDC);
    EXPECT_EQ(s.svd_rejection, SvdRejection::Throw);
    EXPECT_TRUE(s.svd_accept_gram);
    EXPECT_EQ(s.svd_report, SvdReport::Silent);
    EXPECT_EQ(s.canonical_form, CanonicalForm::Auto);
    EXPECT_EQ(s.max_bond_dim, cap);
    EXPECT_EQ(s.svd_cutoff, cutoff);
    EXPECT_EQ(s.open_span(), (Span{n - 1, n - 1}));
    EXPECT_TRUE(is_canonical(s));
    EXPECT_EQ(s.svd_call_count(), before.svd_calls + static_cast<std::size_t>(n - 1))
        << "the rebuild's splits were not added to the count";
    EXPECT_GE(s.truncation_error(), before.truncation);

    // The rebuilt chain is the sequential SVD of the oracle's dense output at
    // this cap and cutoff, and its figures are the prior ones extended by
    // that sweep's.
    QuditStatevector dense_after = v11301::to_qudit_statevector(prior, n, d);
    dense_after.apply_phase_oracle(phase);
    const QuditMPS ref(dense_after, cap, cutoff);
    const double norm = std::sqrt(v11301::norm_sq(prior));
    const double tol = amp_tol(n, d, 4 * static_cast<std::size_t>(n), norm);
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), v11301::dense(ref)), tol);
    EXPECT_NEAR(s.truncation_error() - before.truncation, ref.truncation_error(), tol * norm);
    ASSERT_TRUE(before.estimate.has_value());
    EXPECT_NEAR(*s.fidelity_estimate(), *before.estimate * *ref.fidelity_estimate(), tol);
}

TEST(V11301QuditOracles, AFunctionOracleKeepsTheSettingsAndAccumulatesTheCounters) {
    const int n = 4, d = 3, cap = 9;
    QuditMPS s(n, d, cap);
    s.svd_method = SVDMethod::EigenBDC;
    s.svd_rejection = SvdRejection::Throw;
    s.svd_accept_gram = true;
    s.svd_report = SvdReport::Silent;
    s.canonical_form = CanonicalForm::Auto;
    s.apply_1qudit(0, random_gate(d, 4600));
    s.apply_1qudit(1, random_gate(d, 4601));
    s.apply_2qudit(0, 1, random_gate(d * d, 4602));
    const v11301::Profile before = v11301::profile_of(s);
    const Amplitudes prior = v11301::dense(s);

    // f on the two query digits' flat index, into the two output digits.
    const auto f = [](int x) { return (5 * x + 1) % 9; };
    s.apply_function_oracle(2, 2, f);

    EXPECT_EQ(s.svd_method, SVDMethod::EigenBDC);
    EXPECT_EQ(s.svd_rejection, SvdRejection::Throw);
    EXPECT_TRUE(s.svd_accept_gram);
    EXPECT_EQ(s.svd_report, SvdReport::Silent);
    EXPECT_EQ(s.canonical_form, CanonicalForm::Auto);
    EXPECT_EQ(s.max_bond_dim, cap);
    EXPECT_EQ(s.open_span(), (Span{n - 1, n - 1}));
    EXPECT_TRUE(is_canonical(s));
    EXPECT_EQ(s.svd_call_count(), before.svd_calls + static_cast<std::size_t>(n - 1));

    // At a cap of d^(n/2) the rebuild is exact, so the chain is the oracle
    // applied to the prior state, digit by digit: |x>|y> -> |x>|y + f(x)>.
    Amplitudes want(prior.size(), Cplx(0.0, 0.0));
    for (std::size_t i = 0; i < prior.size(); ++i) {
        const int x = static_cast<int>(i % 9);
        const int y = static_cast<int>(i / 9);
        const int fx = f(x);
        const int y0 = (y % 3 + fx % 3) % 3;
        const int y1 = (y / 3 + fx / 3) % 3;
        want[static_cast<std::size_t>(x + 9 * (y0 + 3 * y1))] = prior[i];
    }
    const double norm = std::sqrt(v11301::norm_sq(prior));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
              amp_tol(n, d, 4 * static_cast<std::size_t>(n), norm) + truncation_distance(s));
}

// =============================================================================
// Operands answer to the caller's policy
// =============================================================================

TEST(V11301QuditOperands, ReversedAndSwapRoutedOperandsAnswerToTheCallersPolicy) {
    // The caller's matrix is validated once, under the caller's policy, and
    // then applied however the chain routes it. apply_2qudit normalises a
    // reversed pair by exchanging the matrix's digit roles and calling itself,
    // and that second call must not judge the exchanged matrix again under a
    // policy the caller did not pass. Under Ignore a non-unitary matrix is
    // applied as given, a caller's wider atol is honoured, and Warn reports
    // once and applies.
    const int n = 4, d = 3;
    std::vector<Complex128> U = random_gate(d * d, 4700);
    U[10] = U[10] * 1.001;  // no longer unitary, by about 1e-3
    std::vector<Complex128> V = random_gate(d * d, 4701);
    V[20] = V[20] + Complex128(1e-9, 0.0);  // off by about 1e-9

    for (const auto& [q0, q1] : std::vector<Span>{{0, 1}, {1, 0}, {3, 0}, {0, 3}, {2, 1}}) {
        SCOPED_TRACE("operands (" + std::to_string(q0) + ", " + std::to_string(q1) + ")");
        const auto sites = v11301::random_qudit_chain(n, d, {2, 3, 2}, 4702);

        QuditMPS s = chain_of(sites, d, 81);
        const Amplitudes before = v11301::dense(s);
        bool applied = true;
        try {
            s.apply_2qudit(q0, q1, U, {Validation::Ignore});
        } catch (const std::exception& e) {
            applied = false;
            ADD_FAILURE() << "Ignore measures nothing, so nothing may reject the "
                             "operand, but the call threw: " << e.what();
        }
        if (applied) {
            const Amplitudes want = apply_dense_2(before, n, d, q0, q1, U);
            const double norm = std::sqrt(v11301::norm_sq(want));
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
                      amp_tol(n, d, 8, norm) + truncation_distance(s));
        }

        QuditMPS t = chain_of(sites, d, 81);
        EXPECT_NO_THROW(t.apply_2qudit(q0, q1, V, {Validation::Throw, 1e-6}))
            << "inside the caller's atol, so nothing may reject it";

        QuditMPS w = chain_of(sites, d, 81);
        {
            WarningCapture cap;
            EXPECT_NO_THROW(w.apply_2qudit(q0, q1, U, {Validation::Warn}));
            EXPECT_EQ(cap.count(), 1u) << "Warn reports the operand once";
        }

        QuditMPS r = chain_of(sites, d, 81);
        EXPECT_THROW(r.apply_2qudit(q0, q1, U), std::invalid_argument)
            << "the default policy still throws";
        EXPECT_TRUE(same_sites(r.tensors(), sites)) << "a rejected gate changed the chain";
    }
}
