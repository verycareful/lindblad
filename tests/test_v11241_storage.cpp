// test_v11241_storage.cpp - the library's own dense storage, and the qudit
// matricisations that now cross it.
//
// detail::DenseMatrix and detail::RealVector replaced Eigen types in
// SvdTruncation and in the qudit MPSSiteTensor accessors, so a factorisation or
// a site tensor no longer obliges its holder to have Eigen. That conversion
// touched only element access, which is exactly the kind of change that
// transposes an index and produces a result that is wrong, finite and
// self-consistent.
//
// Two things are pinned here.
//
// The LAYOUT, because it is load-bearing rather than incidental: DenseMatrix is
// column-major so that data() reaches the decomposition seam with no transpose,
// and code that maps the buffer with a backend gets the layout it assumed. A
// silent switch to row-major would leave every element accessor correct and
// every mapped read wrong.
//
// The MATRICISATIONS, exhaustively over every (sigma, aL, aR) rather than at a
// single value. The accessors are pure index shuffles, so exhaustive costs
// nothing on these shapes, and every shape below gives d, chi_L and chi_R
// distinct values. Equal dimensions would hide precisely the transposition
// being hunted.

#include <gtest/gtest.h>

#include "lindblad/detail/dense_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"

#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

using lindblad::Complex128;
using lindblad::detail::DenseMatrix;
using lindblad::detail::RealVector;
using lindblad::MPSSiteTensor;

using Z = std::complex<double>;

namespace {

// Injective over the index triple and exact in binary floating point, so a
// swapped pair of indices cannot coincide with the value it displaced and no
// rounding enters a test about layout.
Z tensor_value(int sigma, int aL, int aR) {
    return Z(1.0 + 100.0 * sigma + 10.0 * aL + aR,
             -(1.0 + sigma + 2.0 * aL + 3.0 * aR));
}

Z matrix_value(int r, int c) {
    return Z(1.0 + 32.0 * r + c, -(2.0 + r + 8.0 * c));
}

void fill(MPSSiteTensor& t) {
    for (int sigma = 0; sigma < t.d; ++sigma)
        for (int aL = 0; aL < t.chi_L; ++aL)
            for (int aR = 0; aR < t.chi_R; ++aR) {
                const Z v = tensor_value(sigma, aL, aR);
                t.at(sigma, aL, aR) = Complex128(v.real(), v.imag());
            }
}

// Shapes with all three dimensions distinct, in both orderings, plus the
// boundary case chi_L == 1 that opens and closes a chain.
const std::vector<std::array<int, 3>>& shapes() {
    static const std::vector<std::array<int, 3>> s{
        {3, 2, 4}, {2, 4, 3}, {4, 3, 2}, {3, 1, 5}, {2, 5, 1}};
    return s;
}

}  // namespace

// =============================================================================
// DenseMatrix
// =============================================================================

TEST(V11241Storage, DenseMatrixIsColumnMajorInItsBuffer) {
    // The whole reason the type exists in this layout: data() is handed
    // straight to the decomposition seam, which reads column-major.
    const int rows = 5, cols = 3;
    DenseMatrix m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) m(r, c) = matrix_value(r, c);

    const Z* buf = m.data();
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            EXPECT_EQ(buf[static_cast<size_t>(c) * rows + r], matrix_value(r, c))
                << "buffer is not column-major at (" << r << "," << c << ")";
}

TEST(V11241Storage, DenseMatrixColumnPointerAddressesOneContiguousColumn) {
    const int rows = 4, cols = 3;
    DenseMatrix m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) m(r, c) = matrix_value(r, c);

    for (int c = 0; c < cols; ++c) {
        const Z* col = m.col(c);
        for (int r = 0; r < rows; ++r)
            EXPECT_EQ(col[r], matrix_value(r, c))
                << "col(" << c << ")[" << r << "]";
    }
}

TEST(V11241Storage, DenseMatrixConstructionZeroes) {
    // Callers fill only the columns above a validity floor and rely on the rest
    // already being zero, so this is a contract and not an implementation
    // detail.
    DenseMatrix m(4, 6);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c) EXPECT_EQ(m(r, c), Z(0.0, 0.0));
}

TEST(V11241Storage, DenseMatrixResizeZeroesWhetherItGrowsOrShrinks) {
    DenseMatrix m(3, 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m(r, c) = matrix_value(r, c);

    m.resize(5, 2);
    EXPECT_EQ(m.rows(), 5);
    EXPECT_EQ(m.cols(), 2);
    EXPECT_EQ(m.size(), 10u);
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 2; ++c)
            EXPECT_EQ(m(r, c), Z(0.0, 0.0)) << "growth left a stale value";

    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 2; ++c) m(r, c) = matrix_value(r, c);
    m.resize(2, 2);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            EXPECT_EQ(m(r, c), Z(0.0, 0.0)) << "shrink left a stale value";
}

TEST(V11241Storage, DenseMatrixSetZeroKeepsTheShape) {
    DenseMatrix m(3, 4);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) m(r, c) = matrix_value(r, c);
    m.set_zero();
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) EXPECT_EQ(m(r, c), Z(0.0, 0.0));
}

TEST(V11241Storage, DenseMatrixDefaultIsEmpty) {
    DenseMatrix m;
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_EQ(m.size(), 0u);

    DenseMatrix n(2, 2);
    EXPECT_FALSE(n.empty());
    EXPECT_EQ(n.size(), 4u);
}

// =============================================================================
// RealVector
// =============================================================================

TEST(V11241Storage, RealVectorZeroesAndIndexesItsBuffer) {
    RealVector v(4);
    EXPECT_EQ(v.size(), 4);
    EXPECT_FALSE(v.empty());
    for (int i = 0; i < 4; ++i) EXPECT_EQ(v(i), 0.0);

    for (int i = 0; i < 4; ++i) v(i) = 1.0 + i;
    for (int i = 0; i < 4; ++i) EXPECT_EQ(v.data()[i], 1.0 + i);
}

TEST(V11241Storage, RealVectorResizeZeroes) {
    RealVector v(3);
    for (int i = 0; i < 3; ++i) v(i) = 1.0 + i;
    v.resize(5);
    EXPECT_EQ(v.size(), 5);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(v(i), 0.0);

    RealVector d;
    EXPECT_TRUE(d.empty());
    EXPECT_EQ(d.size(), 0);
}

// =============================================================================
// MPSSiteTensor - flat index and both matricisations
// =============================================================================

TEST(V11241Storage, SiteTensorAtMatchesTheDocumentedFlatIndex) {
    for (const auto& s : shapes()) {
        MPSSiteTensor t(s[0], s[1], s[2]);
        fill(t);
        for (int sigma = 0; sigma < t.d; ++sigma)
            for (int aL = 0; aL < t.chi_L; ++aL)
                for (int aR = 0; aR < t.chi_R; ++aR) {
                    const size_t flat =
                        static_cast<size_t>(sigma) * t.chi_L * t.chi_R +
                        static_cast<size_t>(aL) * t.chi_R + aR;
                    const Z want = tensor_value(sigma, aL, aR);
                    EXPECT_EQ(t.data[flat].real, want.real())
                        << "d=" << t.d << " chi_L=" << t.chi_L
                        << " chi_R=" << t.chi_R << " at (" << sigma << ","
                        << aL << "," << aR << ")";
                    EXPECT_EQ(t.data[flat].imag, want.imag());
                }
    }
}

TEST(V11241Storage, LeftMatricisationUsesTheDocumentedRowIndex) {
    for (const auto& s : shapes()) {
        MPSSiteTensor t(s[0], s[1], s[2]);
        fill(t);
        const DenseMatrix M = t.as_left_matrix();
        ASSERT_EQ(M.rows(), t.d * t.chi_L);
        ASSERT_EQ(M.cols(), t.chi_R);
        for (int sigma = 0; sigma < t.d; ++sigma)
            for (int aL = 0; aL < t.chi_L; ++aL)
                for (int aR = 0; aR < t.chi_R; ++aR)
                    EXPECT_EQ(M(sigma * t.chi_L + aL, aR),
                              tensor_value(sigma, aL, aR))
                        << "d=" << t.d << " chi_L=" << t.chi_L
                        << " chi_R=" << t.chi_R << " at (" << sigma << ","
                        << aL << "," << aR << ")";
    }
}

TEST(V11241Storage, RightMatricisationUsesTheDocumentedColumnIndex) {
    for (const auto& s : shapes()) {
        MPSSiteTensor t(s[0], s[1], s[2]);
        fill(t);
        const DenseMatrix M = t.as_right_matrix();
        ASSERT_EQ(M.rows(), t.chi_L);
        ASSERT_EQ(M.cols(), t.d * t.chi_R);
        for (int sigma = 0; sigma < t.d; ++sigma)
            for (int aL = 0; aL < t.chi_L; ++aL)
                for (int aR = 0; aR < t.chi_R; ++aR)
                    EXPECT_EQ(M(aL, sigma * t.chi_R + aR),
                              tensor_value(sigma, aL, aR))
                        << "d=" << t.d << " chi_L=" << t.chi_L
                        << " chi_R=" << t.chi_R << " at (" << sigma << ","
                        << aL << "," << aR << ")";
    }
}

TEST(V11241Storage, LeftMatricisationRoundTripsExactlyAtEveryIndex) {
    for (const auto& s : shapes()) {
        MPSSiteTensor t(s[0], s[1], s[2]);
        fill(t);
        const MPSSiteTensor back =
            MPSSiteTensor::from_left_matrix(t.as_left_matrix(), t.d, t.chi_L);

        ASSERT_EQ(back.d, t.d);
        ASSERT_EQ(back.chi_L, t.chi_L);
        ASSERT_EQ(back.chi_R, t.chi_R);
        // A reshape moves bits and never arithmetic, so equality is exact and
        // a tolerance here would hide a genuine index error.
        for (int sigma = 0; sigma < t.d; ++sigma)
            for (int aL = 0; aL < t.chi_L; ++aL)
                for (int aR = 0; aR < t.chi_R; ++aR) {
                    EXPECT_EQ(back.at(sigma, aL, aR).real,
                              t.at(sigma, aL, aR).real)
                        << "d=" << t.d << " chi_L=" << t.chi_L
                        << " chi_R=" << t.chi_R << " at (" << sigma << ","
                        << aL << "," << aR << ")";
                    EXPECT_EQ(back.at(sigma, aL, aR).imag,
                              t.at(sigma, aL, aR).imag);
                }
    }
}

TEST(V11241Storage, RightMatricisationRoundTripsExactlyAtEveryIndex) {
    for (const auto& s : shapes()) {
        MPSSiteTensor t(s[0], s[1], s[2]);
        fill(t);
        const MPSSiteTensor back =
            MPSSiteTensor::from_right_matrix(t.as_right_matrix(), t.d, t.chi_R);

        ASSERT_EQ(back.d, t.d);
        ASSERT_EQ(back.chi_L, t.chi_L);
        ASSERT_EQ(back.chi_R, t.chi_R);
        for (int sigma = 0; sigma < t.d; ++sigma)
            for (int aL = 0; aL < t.chi_L; ++aL)
                for (int aR = 0; aR < t.chi_R; ++aR) {
                    EXPECT_EQ(back.at(sigma, aL, aR).real,
                              t.at(sigma, aL, aR).real)
                        << "d=" << t.d << " chi_L=" << t.chi_L
                        << " chi_R=" << t.chi_R << " at (" << sigma << ","
                        << aL << "," << aR << ")";
                    EXPECT_EQ(back.at(sigma, aL, aR).imag,
                              t.at(sigma, aL, aR).imag);
                }
    }
}

TEST(V11241Storage, FromLeftMatrixRejectsARowCountItCannotReshape) {
    // d * chi_L is the only row count that reshapes, and a mismatch is a caller
    // error rather than something to reinterpret.
    DenseMatrix M(7, 3);
    EXPECT_THROW(MPSSiteTensor::from_left_matrix(M, /*d=*/3, /*chi_L=*/2),
                 std::invalid_argument);
    EXPECT_NO_THROW(MPSSiteTensor::from_left_matrix(DenseMatrix(6, 3), 3, 2));
}
