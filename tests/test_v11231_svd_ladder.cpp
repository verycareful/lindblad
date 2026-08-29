// 1.1.23.1 test wave - the shared truncation ladder, driven directly.
//
// 1.1.23.0 moved SELECT -> VERIFY -> FALLBACK -> THROW out of the qubit MPS
// into src/svd_truncate.cpp, where both MPS layers call it as a free function
// taking a raw block. That created a seam: the ladder can now be handed a
// crafted matrix instead of being reached only through a bond split whose
// spectrum happens to misbehave on one build. Nothing used that seam.
//
// WHAT IS REACHABLE FROM HERE, and what is not. The routine computes its own
// factorisation, so a test controls the INPUT matrix and nothing else. That
// covers SELECT completely (the weight budget, the bond cap, the zero-matrix
// rescue, the descending order of what is kept), the accounting it reports,
// both storage orders, and the THROW rung, which a non-finite block reaches
// because neither route can produce a finite spectrum from one.
//
// The Gram rescue SUCCEEDING is deliberately not tested here, and that is a
// statement about what is possible rather than an omission. Reaching it needs
// the backend factorisation to fail verification while the Gram route survives,
// and a backend that fails on demand is exactly the thing this ladder exists
// because we cannot produce. Pinning it needs a seam one level lower, where the
// candidate factorisation is injected rather than computed. That is worth
// having and is noted in TODO; it is not something this file can fake.

#include <gtest/gtest.h>

#include "lindblad/detail/svd_truncate.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// A clean factorisation satisfies the Frobenius identity with equality, so its
// reported excess sits near eps^2. The defect the ladder was built to catch
// measured 7.4e-16 of the same quantity, so anything below this separates the
// two by orders rather than by a hair.
constexpr double kHealthyExcess = 1e-20;

// Row-major and column-major buffers for the same logical matrix, so the two
// orders can be handed the same content and compared.
std::vector<Complex128> row_major(const std::vector<std::vector<double>>& m) {
    std::vector<Complex128> out;
    for (const auto& row : m)
        for (double v : row) out.push_back(Complex128(v, 0.0));
    return out;
}

std::vector<Complex128> col_major(const std::vector<std::vector<double>>& m) {
    const std::size_t rows = m.size();
    const std::size_t cols = m[0].size();
    std::vector<Complex128> out;
    out.reserve(rows * cols);
    for (std::size_t c = 0; c < cols; ++c)
        for (std::size_t r = 0; r < rows; ++r)
            out.push_back(Complex128(m[r][c], 0.0));
    return out;
}

// Square diagonal matrix, whose singular values are exactly the entries.
std::vector<std::vector<double>> diagonal(const std::vector<double>& d) {
    const std::size_t n = d.size();
    std::vector<std::vector<double>> m(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) m[i][i] = d[i];
    return m;
}

detail::SvdTruncation truncate(const std::vector<std::vector<double>>& m,
                               int max_bond_dim, double cutoff,
                               detail::MatrixOrder order = detail::MatrixOrder::RowMajor,
                               SVDMethod method = SVDMethod::Jacobi) {
    const int rows = static_cast<int>(m.size());
    const int cols = static_cast<int>(m[0].size());
    const std::vector<Complex128> buf =
        (order == detail::MatrixOrder::RowMajor) ? row_major(m) : col_major(m);
    return detail::svd_truncate_verified(buf.data(), rows, cols, order,
                                         max_bond_dim, cutoff, method,
                                         "V11231SvdLadder");
}

// Sum of squares of the entries, which for a diagonal matrix is the total
// singular weight the budget is a fraction of.
double weight(const std::vector<double>& d) {
    double w = 0.0;
    for (double v : d) w += v * v;
    return w;
}

}  // namespace

// =============================================================================
// The primary route on a healthy block
// =============================================================================

TEST(V11231SvdLadder, CleanBlockTakesThePrimaryRouteAndKeepsEverything) {
    const std::vector<double> sigmas = {1.0, 0.5, 0.25};
    const auto t = truncate(diagonal(sigmas), 64, 1e-16);

    EXPECT_EQ(t.rank, 3) << "nothing is discardable inside this budget";
    EXPECT_FALSE(t.used_gram_fallback)
        << "a healthy factorisation must not need rescuing";
    EXPECT_LT(t.residual_excess, kHealthyExcess)
        << "excess over the Frobenius identity, got " << t.residual_excess;
    EXPECT_LT(t.discarded_weight, kHealthyExcess)
        << "a split that kept everything discarded nothing";
}

// The accounting is summed from the discarded buckets rather than as
// (total - kept). For a block that truncates nothing, the subtraction would
// return multiples of eps against two values near 1, reporting phantom loss.
TEST(V11231SvdLadder, DiscardedWeightIsNotADifferenceOfLargeNumbers) {
    // A normalised block: total weight exactly 1, so total and kept are both 1
    // and their difference is where a subtraction loses all resolution.
    const std::vector<double> sigmas = {0.6, 0.8};
    ASSERT_DOUBLE_EQ(weight(sigmas), 1.0);

    const auto t = truncate(diagonal(sigmas), 64, 1e-16);
    ASSERT_EQ(t.rank, 2);
    EXPECT_LT(t.discarded_weight, kHealthyExcess)
        << "discarding nothing must report as nothing, got " << t.discarded_weight;
}

TEST(V11231SvdLadder, KeptSingularValuesComeBackInDescendingOrder) {
    // Supplied smallest-first, so a routine that merely echoed its input order
    // would fail here.
    const auto t = truncate(diagonal({0.25, 1.0, 0.5}), 64, 1e-16);
    ASSERT_EQ(t.rank, 3);
    for (int i = 1; i < t.rank; ++i)
        EXPECT_GE(t.S(i - 1), t.S(i)) << "singular value " << i << " out of order";
    EXPECT_NEAR(t.S(0), 1.0, 1e-12);
}

TEST(V11231SvdLadder, TheKeptSliceReconstructsTheBlockWhenNothingIsTruncated) {
    const std::vector<std::vector<double>> m = {{1.0, 0.5, 0.0},
                                                {0.0, 2.0, 0.25},
                                                {0.5, 0.0, 1.5}};
    const auto t = truncate(m, 64, 1e-16);
    ASSERT_EQ(t.rank, 3);

    const Eigen::MatrixXcd recon = t.U * t.S.asDiagonal() * t.V.adjoint();
    ASSERT_EQ(static_cast<int>(recon.rows()), 3);
    ASSERT_EQ(static_cast<int>(recon.cols()), 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(recon(r, c).real(), m[static_cast<std::size_t>(r)]
                                             [static_cast<std::size_t>(c)],
                        1e-12)
                << "entry (" << r << "," << c << ")";
}

// =============================================================================
// SELECT: the weight budget and the bond cap
// =============================================================================

TEST(V11231SvdLadder, TheBudgetDropsTheSmallestDirectionsAndCountsThem) {
    const std::vector<double> sigmas = {1.0, 0.5, 0.25};
    const double total = weight(sigmas);          // 1.3125

    // A budget that admits 0.25^2 but not 0.5^2 on top of it.
    const double cutoff = 0.0625 * 1.05 / total;
    const auto t = truncate(diagonal(sigmas), 64, cutoff);

    EXPECT_EQ(t.rank, 2) << "exactly the smallest direction is discardable";
    EXPECT_NEAR(t.discarded_weight, 0.25 * 0.25, 1e-12)
        << "the discarded weight is the squared sigma that was dropped";
}

// The budget is a ceiling, not a quota: a bimodal spectrum has nothing between
// the noise and the budget, so nothing extra goes just because room remains.
TEST(V11231SvdLadder, TheBudgetIsACeilingRatherThanAQuota) {
    const double noise = 1e-9;
    const std::vector<double> sigmas = {1.0, 1.0, noise};
    const auto t = truncate(diagonal(sigmas), 64, 0.1);

    EXPECT_EQ(t.rank, 2)
        << "the two unit directions hold weight far above the budget, so the "
           "budget cannot reach them however much of it is unspent";
    EXPECT_NEAR(t.discarded_weight, noise * noise, noise * noise * 1e-6)
        << "only the noise direction was discardable";
}

TEST(V11231SvdLadder, RankNeverExceedsMaxBondDim) {
    const auto sigmas = diagonal({1.0, 0.9, 0.8, 0.7, 0.6});
    for (int cap : {1, 2, 3, 5}) {
        SCOPED_TRACE("max_bond_dim = " + std::to_string(cap));
        const auto t = truncate(sigmas, cap, 1e-16);
        EXPECT_LE(t.rank, cap);
        EXPECT_EQ(static_cast<int>(t.U.cols()), t.rank)
            << "U carries exactly the kept columns";
        EXPECT_EQ(static_cast<int>(t.V.cols()), t.rank)
            << "V carries exactly the kept columns";
        EXPECT_EQ(static_cast<int>(t.S.size()), t.rank);
    }
}

TEST(V11231SvdLadder, ACapOfOneStillKeepsTheLargestDirection) {
    const auto t = truncate(diagonal({0.25, 1.0, 0.5}), 1, 1e-16);
    ASSERT_EQ(t.rank, 1);
    EXPECT_NEAR(t.S(0), 1.0, 1e-12) << "the direction kept is the largest one";
}

// A numerically zero block has no direction at all. Rather than returning an
// empty factorisation, the rescue keeps one, so a caller always receives a
// tensor with a usable bond.
TEST(V11231SvdLadder, AZeroBlockKeepsASingleDirection) {
    const auto t = truncate(diagonal({0.0, 0.0, 0.0}), 64, 1e-16);
    EXPECT_EQ(t.rank, 1) << "a bond may not collapse to zero width";
    EXPECT_FALSE(t.used_gram_fallback);
    EXPECT_DOUBLE_EQ(t.S(0), 0.0);
}

TEST(V11231SvdLadder, ARankDeficientBlockKeepsOnlyItsRealDirections) {
    // Spectrum {1, 1, 0, 0}: the shape the observed Eigen defect appeared on.
    const auto t = truncate(diagonal({1.0, 1.0, 0.0, 0.0}), 64, 1e-16);
    EXPECT_EQ(t.rank, 2)
        << "the two null directions carry no weight and are not real bonds";
    EXPECT_LT(t.discarded_weight, kHealthyExcess);
}

TEST(V11231SvdLadder, AnExactlyDegenerateSpectrumIsResolved) {
    const auto t = truncate(diagonal({1.0, 1.0, 1.0, 1.0}), 64, 1e-16);
    EXPECT_EQ(t.rank, 4) << "degeneracy is not rank deficiency";
    EXPECT_LT(t.residual_excess, kHealthyExcess);
    for (int i = 0; i < t.rank; ++i) EXPECT_NEAR(t.S(i), 1.0, 1e-12);
}

// =============================================================================
// Storage order
// =============================================================================

// Both orders are accepted so neither MPS layer copies a block to hand it over.
// A non-symmetric matrix is used deliberately: for a symmetric one the two
// buffers are identical and the test would pass without exercising anything.
TEST(V11231SvdLadder, BothStorageOrdersDescribeTheSameFactorisation) {
    const std::vector<std::vector<double>> m = {{1.0, 2.0, 3.0},
                                                {0.0, 0.5, 0.25},
                                                {4.0, 0.0, 0.125}};
    ASSERT_NE(row_major(m)[1].real, col_major(m)[1].real)
        << "the fixture must actually differ between layouts";

    const auto r = truncate(m, 64, 1e-16, detail::MatrixOrder::RowMajor);
    const auto c = truncate(m, 64, 1e-16, detail::MatrixOrder::ColMajor);

    ASSERT_EQ(r.rank, c.rank);
    for (int i = 0; i < r.rank; ++i)
        EXPECT_NEAR(r.S(i), c.S(i), 1e-12) << "singular value " << i;
    EXPECT_NEAR(r.discarded_weight, c.discarded_weight, 1e-18);
    EXPECT_EQ(r.used_gram_fallback, c.used_gram_fallback);
}

// =============================================================================
// Non-square blocks
// =============================================================================

// The Gram route forms the smaller of M†M and MM†, so both shapes have to be
// accepted and both have to report the same rank as the other orientation.
TEST(V11231SvdLadder, TallAndWideBlocksBothFactorise) {
    const std::vector<std::vector<double>> tall = {{1.0, 0.0},
                                                   {0.0, 0.5},
                                                   {0.0, 0.0},
                                                   {0.0, 0.0}};
    const std::vector<std::vector<double>> wide = {{1.0, 0.0, 0.0, 0.0},
                                                   {0.0, 0.5, 0.0, 0.0}};

    const auto t = truncate(tall, 64, 1e-16);
    const auto w = truncate(wide, 64, 1e-16);

    EXPECT_EQ(t.rank, 2);
    EXPECT_EQ(w.rank, 2);
    EXPECT_EQ(static_cast<int>(t.U.rows()), 4);
    EXPECT_EQ(static_cast<int>(t.V.rows()), 2);
    EXPECT_EQ(static_cast<int>(w.U.rows()), 2);
    EXPECT_EQ(static_cast<int>(w.V.rows()), 4);
    for (int i = 0; i < 2; ++i)
        EXPECT_NEAR(t.S(i), w.S(i), 1e-12)
            << "a block and its transpose share a spectrum";
}

TEST(V11231SvdLadder, ASingleEntryBlockFactorises) {
    const std::vector<std::vector<double>> one = {{2.0}};
    const auto t = truncate(one, 64, 1e-16);
    EXPECT_EQ(t.rank, 1);
    EXPECT_NEAR(t.S(0), 2.0, 1e-12);
    EXPECT_LT(t.residual_excess, kHealthyExcess);
}

// =============================================================================
// THROW: never return a corrupt tensor
// =============================================================================

// A non-finite block produces no finite spectrum on either route, so both rungs
// reject and the ladder raises. The alternative, returning whatever came back,
// is the silent-wrong-result mode this whole file exists because of.
TEST(V11231SvdLadder, ANonFiniteBlockThrowsRatherThanReturningATensor) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::vector<double>> m = {{nan, nan}, {nan, nan}};

    EXPECT_THROW(truncate(m, 64, 1e-16), std::runtime_error);
}

TEST(V11231SvdLadder, TheExceptionNamesTheCallerThatAskedForIt) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double>> m = {{nan, nan}, {nan, nan}};

    try {
        truncate(m, 64, 1e-16);
        FAIL() << "a block with no finite spectrum must not yield a tensor";
    } catch (const std::runtime_error& e) {
        // ctx is threaded through so a failure names the layer it came from
        // rather than a shared routine every layer calls.
        EXPECT_NE(std::string(e.what()).find("V11231SvdLadder"), std::string::npos)
            << "got: " << e.what();
    }
}

// =============================================================================
// Backend selection
// =============================================================================

// Both backends go through the same ladder, so selecting one does not change
// what a caller is guaranteed. The BDC warning belongs to the MPS layers, not
// here, so exercising it directly does not consume the latch those tests read.
TEST(V11231SvdLadder, BothBackendsProduceAVerifiedFactorisation) {
    const auto sigmas = diagonal({1.0, 0.5, 0.25});

    const auto jacobi = truncate(sigmas, 64, 1e-16,
                                 detail::MatrixOrder::RowMajor, SVDMethod::Jacobi);
    const auto bdc = truncate(sigmas, 64, 1e-16,
                              detail::MatrixOrder::RowMajor, SVDMethod::BDC);

    EXPECT_EQ(jacobi.rank, bdc.rank);
    EXPECT_LT(jacobi.residual_excess, kHealthyExcess);
    EXPECT_LT(bdc.residual_excess, kHealthyExcess)
        << "whichever backend ran, what comes back has been verified";
    for (int i = 0; i < jacobi.rank; ++i)
        EXPECT_NEAR(jacobi.S(i), bdc.S(i), 1e-10) << "singular value " << i;
}
