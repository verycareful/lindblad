// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.1 test wave - detail::qr_thin, the decomposition 1.1.30.0 added to the
// seam.
//
// Both MPS layers move their orthogonality centre with it: a rightward step is
// a thin QR of one site read as a (d chi_L) x chi_R matrix, a leftward step a
// thin LQ of a site read as chi_L x (d chi_R). The LQ has no entry of its own.
// A row-major buffer read column-major is its transpose, so the layers pass a
// row-major M to qr_thin as a column-major M^T and read both factors back
// transposed. Every canonical-form guarantee the layers make rests on this one
// function, which had no test of its own.
//
// What is asserted is the header's contract, each clause separately:
//
//   - Q R == M to backward-stable accuracy, for tall, wide and square shapes,
//     rank-deficient and zero inputs included;
//   - Q has orthonormal columns (a unitary Q when the input is wide);
//   - R is upper trapezoidal, exactly: the entries below the diagonal are
//     written as zero rather than left at rounding level;
//   - both storage orders factorise the same matrix identically;
//   - the LQ-by-transpose identity the layers use: B = R^T Q^T with Q^T
//     orthonormal by rows and R^T lower trapezoidal;
//   - a finite input always factorises, a non-finite one propagates rather
//     than returning finite factors, and only an empty shape returns false.

#include <gtest/gtest.h>

#include "v11301_mps_oracle.hpp"

#include "lindblad/detail/eigen_backend.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using lindblad::detail::MatrixOrder;
using lindblad::detail::qr_thin;
using v11301::ComplexStream;
using v11301::Cplx;
using v11301::kEps;
using v11301::kSlack;

namespace {

// A rows x cols matrix held column-major, the order every check below reads.
struct Mat {
    int rows = 0;
    int cols = 0;
    std::vector<Cplx> a;  // column-major

    Mat(int r, int c) : rows(r), cols(c), a(static_cast<std::size_t>(r) * c) {}
    Cplx& operator()(int i, int j) { return a[static_cast<std::size_t>(j) * rows + i]; }
    const Cplx& operator()(int i, int j) const {
        return a[static_cast<std::size_t>(j) * rows + i];
    }
};

Mat random_mat(int rows, int cols, std::uint64_t seed) {
    ComplexStream s(seed);
    Mat m(rows, cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) m(i, j) = s.next();
    return m;
}

// rows x r times r x cols: rank r whenever r <= min(rows, cols), generically.
Mat rank_deficient(int rows, int cols, int r, std::uint64_t seed) {
    const Mat A = random_mat(rows, r, seed);
    const Mat B = random_mat(r, cols, seed + 1);
    Mat m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            Cplx acc(0.0, 0.0);
            for (int k = 0; k < r; ++k) acc += A(i, k) * B(k, j);
            m(i, j) = acc;
        }
    return m;
}

std::vector<Cplx> row_major(const Mat& m) {
    std::vector<Cplx> out(m.a.size());
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out[static_cast<std::size_t>(i) * m.cols + j] = m(i, j);
    return out;
}

double frobenius(const Mat& m) {
    double s = 0.0;
    for (const auto& z : m.a) s += std::norm(z);
    return std::sqrt(s);
}

struct Factors {
    bool ok = false;
    Mat Q;  // rows x k
    Mat R;  // k x cols
    Factors(int rows, int cols)
        : Q(rows, std::min(rows, cols)), R(std::min(rows, cols), cols) {}
};

Factors factorise(const Mat& m, MatrixOrder order) {
    Factors f(m.rows, m.cols);
    if (order == MatrixOrder::ColMajor) {
        f.ok = qr_thin(m.a.data(), m.rows, m.cols, order, f.Q.a.data(), f.R.a.data());
    } else {
        const std::vector<Cplx> rm = row_major(m);
        f.ok = qr_thin(rm.data(), m.rows, m.cols, order, f.Q.a.data(), f.R.a.data());
    }
    return f;
}

// max |(Q R - M)_ij|
double reconstruction_error(const Factors& f, const Mat& m) {
    const int k = f.Q.cols;
    double worst = 0.0;
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j) {
            Cplx acc(0.0, 0.0);
            for (int t = 0; t < k; ++t) acc += f.Q(i, t) * f.R(t, j);
            worst = std::max(worst, std::abs(acc - m(i, j)));
        }
    return worst;
}

// max |(Q^H Q - I)_ij|
double orthonormality_error(const Mat& Q) {
    double worst = 0.0;
    for (int i = 0; i < Q.cols; ++i)
        for (int j = 0; j < Q.cols; ++j) {
            Cplx acc(0.0, 0.0);
            for (int r = 0; r < Q.rows; ++r) acc += std::conj(Q(r, i)) * Q(r, j);
            worst = std::max(worst, std::abs(acc - Cplx(i == j ? 1.0 : 0.0, 0.0)));
        }
    return worst;
}

// The entries of R below its diagonal, which must be exactly zero.
int nonzero_below_diagonal(const Mat& R) {
    int count = 0;
    for (int i = 0; i < R.rows; ++i)
        for (int j = 0; j < std::min(i, R.cols); ++j)
            if (R(i, j) != Cplx(0.0, 0.0)) ++count;
    return count;
}

bool all_finite(const Mat& m) {
    for (const auto& z : m.a)
        if (!lindblad::is_finite_strict(z.real()) || !lindblad::is_finite_strict(z.imag()))
            return false;
    return true;
}

// Householder QR is backward stable: the reconstruction is exact to a few eps
// of the matrix's norm per row it touches, and Q is orthonormal to a few eps of
// its height.
double reconstruction_tol(const Mat& m) {
    return kSlack * static_cast<double>(std::max(m.rows, m.cols)) * kEps *
           std::max(frobenius(m), 1.0);
}
double orthonormality_tol(const Mat& Q) {
    return kSlack * static_cast<double>(Q.rows) * kEps;
}

// Tall, wide and square, including the vector shapes and the MPS site shapes
// the layers actually factorise: (2 chi) x chi and chi x (2 chi).
const std::vector<std::pair<int, int>>& shapes() {
    static const std::vector<std::pair<int, int>> s = {
        {1, 1}, {2, 1}, {1, 2}, {2, 2}, {5, 3}, {3, 5}, {8, 4}, {4, 8},
        {6, 6}, {16, 8}, {8, 16}, {32, 16}, {16, 32}, {27, 9}, {9, 27}};
    return s;
}

std::string shape_name(int rows, int cols, MatrixOrder order) {
    return std::to_string(rows) + "x" + std::to_string(cols) +
           (order == MatrixOrder::RowMajor ? " row-major" : " column-major");
}

}  // namespace

// =============================================================================
// The factorisation
// =============================================================================

TEST(V11301QrSeam, FactorsReconstructTheInputInEveryShapeAndOrder) {
    std::uint64_t seed = 1;
    for (const auto& [rows, cols] : shapes()) {
        for (MatrixOrder order : {MatrixOrder::RowMajor, MatrixOrder::ColMajor}) {
            SCOPED_TRACE(shape_name(rows, cols, order));
            const Mat m = random_mat(rows, cols, seed++);
            const Factors f = factorise(m, order);
            ASSERT_TRUE(f.ok) << "a finite nonempty input must factorise";
            EXPECT_LE(reconstruction_error(f, m), reconstruction_tol(m));
            EXPECT_LE(orthonormality_error(f.Q), orthonormality_tol(f.Q))
                << "Q does not have orthonormal columns";
            EXPECT_EQ(nonzero_below_diagonal(f.R), 0)
                << "R has entries below its diagonal";
        }
    }
}

TEST(V11301QrSeam, AWideInputGetsASquareUnitaryQ) {
    // k = rows when the input is wide, so Q is square, and orthonormal columns
    // make it unitary: Q Q^H == I as well.
    for (const auto& [rows, cols] : shapes()) {
        if (rows > cols) continue;
        SCOPED_TRACE(shape_name(rows, cols, MatrixOrder::ColMajor));
        const Mat m = random_mat(rows, cols, 700u + static_cast<std::uint64_t>(rows * 97 + cols));
        const Factors f = factorise(m, MatrixOrder::ColMajor);
        ASSERT_TRUE(f.ok);
        ASSERT_EQ(f.Q.rows, f.Q.cols);
        double worst = 0.0;
        for (int i = 0; i < f.Q.rows; ++i)
            for (int j = 0; j < f.Q.rows; ++j) {
                Cplx acc(0.0, 0.0);
                for (int t = 0; t < f.Q.cols; ++t) acc += f.Q(i, t) * std::conj(f.Q(j, t));
                worst = std::max(worst, std::abs(acc - Cplx(i == j ? 1.0 : 0.0, 0.0)));
            }
        EXPECT_LE(worst, orthonormality_tol(f.Q)) << "Q Q^H is not the identity";
    }
}

TEST(V11301QrSeam, BothStorageOrdersGiveTheSameFactorisation) {
    // The two orders describe one matrix, so they must yield one answer: a
    // layer that stores its sites row-major and one that holds column-major
    // blocks rely on the seam treating them alike. Compared to the bit.
    std::uint64_t seed = 300;
    for (const auto& [rows, cols] : shapes()) {
        SCOPED_TRACE(std::to_string(rows) + "x" + std::to_string(cols));
        const Mat m = random_mat(rows, cols, seed++);
        const Factors a = factorise(m, MatrixOrder::RowMajor);
        const Factors b = factorise(m, MatrixOrder::ColMajor);
        ASSERT_TRUE(a.ok);
        ASSERT_TRUE(b.ok);
        for (std::size_t i = 0; i < a.Q.a.size(); ++i) {
            ASSERT_TRUE(v11301::same_bits(a.Q.a[i].real(), b.Q.a[i].real()) &&
                        v11301::same_bits(a.Q.a[i].imag(), b.Q.a[i].imag()))
                << "Q entry " << i << " differs between the storage orders";
        }
        for (std::size_t i = 0; i < a.R.a.size(); ++i) {
            ASSERT_TRUE(v11301::same_bits(a.R.a[i].real(), b.R.a[i].real()) &&
                        v11301::same_bits(a.R.a[i].imag(), b.R.a[i].imag()))
                << "R entry " << i << " differs between the storage orders";
        }
    }
}

TEST(V11301QrSeam, RankDeficientInputsStillFactoriseExactly) {
    // An MPS site is routinely rank-deficient: a bond wider than the rank its
    // neighbour can carry, or a block a truncation left short. Householder QR
    // has no pivoting and no iteration to fail, so the contract holds there
    // too, and Q stays orthonormal even where R's diagonal falls to rounding.
    std::uint64_t seed = 500;
    for (const auto& [rows, cols] : shapes()) {
        const int full = std::min(rows, cols);
        for (int r = 1; r < full; ++r) {
            SCOPED_TRACE(std::to_string(rows) + "x" + std::to_string(cols) +
                         " rank " + std::to_string(r));
            const Mat m = rank_deficient(rows, cols, r, seed);
            seed += 2;
            for (MatrixOrder order : {MatrixOrder::RowMajor, MatrixOrder::ColMajor}) {
                const Factors f = factorise(m, order);
                ASSERT_TRUE(f.ok);
                EXPECT_LE(reconstruction_error(f, m), reconstruction_tol(m));
                EXPECT_LE(orthonormality_error(f.Q), orthonormality_tol(f.Q));
                EXPECT_EQ(nonzero_below_diagonal(f.R), 0);
            }
        }
    }
}

TEST(V11301QrSeam, TheZeroMatrixFactorisesToAnOrthonormalQAndAZeroR) {
    // A collapsed or all-zero site reaches the centre steps as a zero block.
    // It must come back as a zero R over an orthonormal Q, so the neighbour it
    // multiplies into becomes zero rather than non-finite.
    for (const auto& [rows, cols] : shapes()) {
        SCOPED_TRACE(std::to_string(rows) + "x" + std::to_string(cols));
        Mat m(rows, cols);
        const Factors f = factorise(m, MatrixOrder::ColMajor);
        ASSERT_TRUE(f.ok);
        EXPECT_TRUE(all_finite(f.Q)) << "Q of a zero matrix is not finite";
        EXPECT_TRUE(all_finite(f.R)) << "R of a zero matrix is not finite";
        EXPECT_LE(orthonormality_error(f.Q), orthonormality_tol(f.Q));
        for (const auto& z : f.R.a) EXPECT_EQ(std::abs(z), 0.0) << "R is not zero";
    }
}

// =============================================================================
// The LQ the leftward step builds from it
// =============================================================================

TEST(V11301QrSeam, TheTransposedCallIsAnLqOfTheRowMajorBuffer) {
    // shift_left hands a site's row-major chi_L x (d chi_R) buffer to qr_thin
    // as a column-major (d chi_R) x chi_L matrix, which is B^T. So B^T = Q R,
    // B = R^T Q^T, and the layers read both factors back through the same
    // transposition: the column-major R buffer read row-major is L = R^T
    // (chi_L x k), the column-major Q buffer read row-major is Q^T
    // (k x d chi_R). What the layers then rely on is checked here: B == L Q^T,
    // Q^T orthonormal by rows (so the site it becomes is right-orthonormal),
    // and L lower trapezoidal.
    std::uint64_t seed = 900;
    for (const auto& shape : shapes()) {
        // Named copies, not a structured binding: the Qt lambda below captures
        // b_cols, and clang 18 refuses a lambda capture of a structured binding
        // in any file compiled with -fopenmp.
        const int b_rows = shape.first;
        const int b_cols = shape.second;
        SCOPED_TRACE("B " + std::to_string(b_rows) + "x" + std::to_string(b_cols));
        const Mat B = random_mat(b_rows, b_cols, seed++);
        const std::vector<Cplx> buffer = row_major(B);  // what a site holds
        const int k = std::min(b_rows, b_cols);
        std::vector<Cplx> q(static_cast<std::size_t>(b_cols) * k);
        std::vector<Cplx> r(static_cast<std::size_t>(k) * b_rows);
        ASSERT_TRUE(qr_thin(buffer.data(), b_cols, b_rows, MatrixOrder::ColMajor,
                            q.data(), r.data()));

        // L(i, t) = R(t, i): the R buffer, k x b_rows column-major, read as
        // b_rows x k row-major. Qt(t, j) = Q(j, t): the Q buffer, b_cols x k
        // column-major, read as k x b_cols row-major.
        const auto L = [&](int i, int t) { return r[static_cast<std::size_t>(i) * k + t]; };
        const auto Qt = [&](int t, int j) { return q[static_cast<std::size_t>(t) * b_cols + j]; };

        double recon = 0.0;
        for (int i = 0; i < b_rows; ++i)
            for (int j = 0; j < b_cols; ++j) {
                Cplx acc(0.0, 0.0);
                for (int t = 0; t < k; ++t) acc += L(i, t) * Qt(t, j);
                recon = std::max(recon, std::abs(acc - B(i, j)));
            }
        EXPECT_LE(recon, reconstruction_tol(B)) << "B != L Q^T";

        double ortho = 0.0;
        for (int s = 0; s < k; ++s)
            for (int t = 0; t < k; ++t) {
                Cplx acc(0.0, 0.0);
                for (int j = 0; j < b_cols; ++j) acc += Qt(s, j) * std::conj(Qt(t, j));
                ortho = std::max(ortho, std::abs(acc - Cplx(s == t ? 1.0 : 0.0, 0.0)));
            }
        EXPECT_LE(ortho, kSlack * static_cast<double>(b_cols) * kEps)
            << "Q^T is not orthonormal by rows, so the site it becomes is not "
               "right-orthonormal";

        int above = 0;
        for (int i = 0; i < b_rows; ++i)
            for (int t = i + 1; t < k; ++t)
                if (L(i, t) != Cplx(0.0, 0.0)) ++above;
        EXPECT_EQ(above, 0) << "L has entries above its diagonal";
    }
}

// =============================================================================
// Refusals and non-finite input
// =============================================================================

TEST(V11301QrSeam, OnlyAnEmptyShapeReturnsFalse) {
    // The outputs are never written for an empty shape, so sentinels survive.
    const Cplx sentinel(7.0, -7.0);
    std::vector<Cplx> in(4, Cplx(1.0, 0.0));
    for (const auto& [rows, cols] :
         std::vector<std::pair<int, int>>{{0, 0}, {0, 3}, {3, 0}, {-1, 2}, {2, -1}}) {
        for (MatrixOrder order : {MatrixOrder::RowMajor, MatrixOrder::ColMajor}) {
            SCOPED_TRACE(shape_name(rows, cols, order));
            std::vector<Cplx> q(4, sentinel), r(4, sentinel);
            EXPECT_FALSE(qr_thin(in.data(), rows, cols, order, q.data(), r.data()));
            for (const auto& z : q) EXPECT_EQ(z, sentinel) << "Q was written";
            for (const auto& z : r) EXPECT_EQ(z, sentinel) << "R was written";
        }
    }
}

TEST(V11301QrSeam, ANonFiniteEntryIsCarriedIntoTheFactorsNotReported) {
    // The contract: a non-finite input still returns true, and the factors do
    // not come back entirely finite. That is what lets the SVD ladder downstream
    // of a centre step see the damage and throw, where a finite-looking Q and R
    // would hide it. Every position is tried, in every shape, for NaN and for
    // both infinities.
    const double values[] = {lindblad::quiet_nan_strict(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
    for (const auto& [rows, cols] :
         std::vector<std::pair<int, int>>{{4, 2}, {2, 4}, {3, 3}, {6, 2}, {2, 6}}) {
        const Mat base = random_mat(rows, cols, 4000u + static_cast<std::uint64_t>(rows * 10 + cols));
        for (double bad : values) {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    SCOPED_TRACE(std::to_string(rows) + "x" + std::to_string(cols) +
                                 " at (" + std::to_string(i) + ", " + std::to_string(j) +
                                 ") value " + std::to_string(bad));
                    Mat m = base;
                    m(i, j) = Cplx(bad, 0.0);
                    const Factors f = factorise(m, MatrixOrder::ColMajor);
                    EXPECT_TRUE(f.ok) << "non-finite input is propagated, not refused";
                    EXPECT_FALSE(all_finite(f.Q) && all_finite(f.R))
                        << "a non-finite input came back as entirely finite factors";
                }
        }
    }
}
