// 1.1.28.2 test wave - the Gram rescue's validity floor sits above the noise
// band, not inside it.
//
// The Gram route squares the condition number. A singular value that is
// exactly zero in M comes back from the eigendecomposition of M†M as the
// square root of the solver's own error on a zero eigenvalue, which the
// backward-error bound puts anywhere in [0, c * n * eps * sigma_max²], so the
// sigma lands anywhere in [0, sqrt(c * n * eps) * sigma_max]. A validity
// floor at bare sqrt(eps) * sigma_max sits inside that band and keeps a null
// direction as data whenever the solver's error on it exceeds one eps, which
// it routinely does: the same degenerate theta has returned a null sigma at
// 2e-16 and at 1.6e-8 from one Eigen version, one rounding pattern apart. The
// floor now sits at the top of the band, with the same slack the verify rung
// grants, so the rank the rescue reports is the rank of the input.
//
// The route is pinned through the in-test replica in diag_r1160_matrices.hpp,
// which mirrors the library rule, for the reason that header gives: the
// library's rescue runs only when the primary factorisation fails
// verification, and a backend that fails on demand is what this project cannot
// produce. The replica's floor is asserted to be the library's derivation
// rather than a copy of a number.
//
// Rank-deficient inputs here are products A * B with A of r columns and B of
// r rows, so the rank is r by construction and every direction beyond it is
// solver noise. Many seeds and several shapes, because the failure is a
// rounding pattern that any one input can miss.

#include <gtest/gtest.h>

#include "diag_r1160_matrices.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// Slack a backward-stable decomposition is entitled to, in units of
// n * eps: the shape the truncation ladder's verify rung allows.
constexpr double kSlack = 64.0;

Eigen::MatrixXcd random_block(int rows, int cols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    Eigen::MatrixXcd m(rows, cols);
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
            m(r, c) = std::complex<double>(unit(rng), unit(rng));
    return m;
}

// rows x cols of rank exactly r: an r-column factor times an r-row factor.
Eigen::MatrixXcd rank_deficient(int rows, int cols, int r, std::uint64_t seed) {
    return random_block(rows, r, seed) * random_block(r, cols, seed + 1);
}

// (rows, cols, rank). Both orientations of the Gram matrix, square blocks,
// and a rank close to the Gram dimension so the null space is small and the
// noise directions sit right against real ones.
const std::vector<std::tuple<int, int, int>> kShapes = {
    {8, 8, 4},   {8, 32, 5},   {32, 8, 3},   {16, 16, 12},
    {16, 16, 15}, {64, 64, 20}, {64, 128, 40}, {128, 64, 60},
};

constexpr int kSeeds = 24;

std::uint64_t seed_for(int rows, int cols, int seed) {
    return static_cast<std::uint64_t>(seed) * 1000 +
           static_cast<std::uint64_t>(rows) * 10 + static_cast<std::uint64_t>(cols);
}

}  // namespace

TEST(V11282GramFloor, FloorIsTheTopOfTheSolverErrorBand) {
    // The replica's floor, against its own derivation, on both sides: above
    // bare sqrt(eps) * sigma_max for every dimension, and exactly the bound
    // a null sigma cannot exceed if the eigensolver honours slack * n * eps.
    for (const int gd : {1, 2, 8, 16, 64, 128}) {
        const double sigma_max = 0.75;
        const double floor = diag_r1160::gram_validity_floor(gd, sigma_max);
        EXPECT_GT(floor, std::sqrt(kEps) * sigma_max) << gd;
        EXPECT_DOUBLE_EQ(floor, std::sqrt(kSlack * gd * kEps) * sigma_max) << gd;
    }
    EXPECT_EQ(diag_r1160::gram_validity_floor(8, 0.0), 0.0)
        << "a zero matrix has no scale for the floor to be relative to";
}

TEST(V11282GramFloor, GramRouteReportsTheRankOfTheInput) {
    // Every seed, every shape: the rescue counts exactly r directions and the
    // kept slice is clean. The reconstruction is held to what the route is
    // entitled to: the kept subspace is resolved to about n * eps times the
    // squared condition number of the kept part, so that factor multiplies
    // the usual backward-error allowance. Checked where the report exposes
    // the r-th sigma (it carries the leading fourteen).
    for (const auto& [rows, cols, r] : kShapes) {
        const int gd = std::min(rows, cols);
        for (int seed = 1; seed <= kSeeds; ++seed) {
            const Eigen::MatrixXcd M = rank_deficient(rows, cols, r, seed_for(rows, cols, seed));
            const auto rep = diag_r1160::run_gram_route_report(M);
            EXPECT_EQ(rep.rank, r)
                << rows << "x" << cols << " seed " << seed << ": a null "
                << "direction passed the floor as data, smallest kept sigma "
                << rep.smallest_pos;
            EXPECT_FALSE(rep.kept_slice_bad) << rows << "x" << cols << " seed " << seed;
            if (r <= static_cast<int>(rep.top.size())) {
                const double sigma_max = rep.top[0];
                const double kappa = sigma_max / rep.top[static_cast<std::size_t>(r - 1)];
                EXPECT_LT(rep.trunc_recon_err,
                          kSlack * gd * kEps * sigma_max * kappa * kappa)
                    << rows << "x" << cols << " seed " << seed;
            }
        }
    }
}

TEST(V11282GramFloor, NullSigmasStayInsideTheBand) {
    // What the floor is built on, measured rather than assumed: over every
    // seed and shape, the largest sigma the null space returns stays below
    // the floor. This is the eigensolver's backward-error bound restated in
    // sigma units, and it is the assertion the floor depends on. Read
    // straight off the eigendecomposition the route uses, ascending, so the
    // null space is the leading gd - r values.
    for (const auto& [rows, cols, r] : kShapes) {
        const int gd = std::min(rows, cols);
        ASSERT_LT(r, gd);
        for (int seed = 1; seed <= kSeeds; ++seed) {
            const Eigen::MatrixXcd M = rank_deficient(rows, cols, r, seed_for(rows, cols, seed));
            const Eigen::MatrixXcd G = (rows >= cols) ? Eigen::MatrixXcd(M.adjoint() * M)
                                                      : Eigen::MatrixXcd(M * M.adjoint());
            Eigen::VectorXd evals;
            Eigen::MatrixXcd evecs;
            ASSERT_TRUE(diag_r1160::seam_eigh(G, evals, evecs));
            const double sigma_max = std::sqrt(std::max(0.0, evals(gd - 1)));
            const double null_top = std::sqrt(std::max(0.0, evals(gd - r - 1)));
            EXPECT_LT(null_top, diag_r1160::gram_validity_floor(gd, sigma_max))
                << rows << "x" << cols << " seed " << seed << ": null sigma "
                << null_top << " against sigma_max " << sigma_max;
        }
    }
}
