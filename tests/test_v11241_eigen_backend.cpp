// test_v11241_eigen_backend.cpp - the decomposition seam's own contract.
//
// detail::eigen_backend became the single point at which this library performs
// a matrix decomposition, and nothing asserted what it promises. Every MPS
// test reaches it, so it is exercised constantly, but only through callers that
// would fail for a dozen other reasons; nothing states the seam's terms.
//
// Two of those terms fail SILENTLY when broken, which is why they are tested
// here rather than left to the callers:
//
//   MatrixOrder. The header calls it a parameter rather than an assumption
//   because a matrix read in the wrong order still decomposes. Worse, the
//   spectrum cannot detect it: reading a square row-major buffer as
//   column-major yields the transpose, and a transpose has exactly the same
//   singular values. Only the VECTORS differ, so every order test below
//   reconstructs U S V-dagger and compares against the logical matrix, and
//   every fixture is deliberately non-symmetric. A symmetric one would pass
//   whatever the seam did with the flag.
//
//   Ascending eigenvalues. eigh returns the solver's own order and does not
//   normalise it, so a caller wanting sigmas reverses explicitly. A seam that
//   quietly started returning descending values would leave every caller
//   reading the null space as the leading directions.
//
// The suite also pins the accuracy that chose the Eigen pin: on degenerate
// Hermitian input, the null-space tail must land at machine epsilon rather than
// its square root. That is the property the Gram rescue depends on, since it
// factorises exactly such matrices.

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/detail/svd_verify.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

using lindblad::SVDMethod;
using lindblad::detail::eigh;
using lindblad::detail::MatrixOrder;
using lindblad::detail::svd_reconstruction_residual_sq;
using lindblad::detail::svd_thin;

using Z = std::complex<double>;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// Slack a backward-stable decomposition is entitled to, in units of
// n * eps * scale. The same shape the truncation ladder's verify rung allows,
// so this suite and the ladder cannot disagree about what "correct" means.
constexpr double kSlack = 64.0;

double tol(int n, double scale = 1.0) {
    return kSlack * static_cast<double>(n) * kEps * scale;
}

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------
// Every entry is a dyadic rational, so the fixture itself is exact in binary
// floating point and no rounding enters before the decomposition does.
//
// Deliberately NOT symmetric and NOT Hermitian: m(r, c) != m(c, r) for every
// off-diagonal pair, which is what makes the MatrixOrder tests able to fail.
Z asym(int r, int c) {
    return Z(1.0 + static_cast<double>(r) + 2.0 * static_cast<double>(c),
             0.5 + 3.0 * static_cast<double>(r) - static_cast<double>(c));
}

std::vector<Z> asym_rowmajor(int rows, int cols) {
    std::vector<Z> a(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            a[static_cast<size_t>(r) * cols + c] = asym(r, c);
    return a;
}

std::vector<Z> asym_colmajor(int rows, int cols) {
    std::vector<Z> a(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            a[static_cast<size_t>(c) * rows + r] = asym(r, c);
    return a;
}

// Holds one factorisation, with the buffers sized as the seam requires.
struct Svd {
    std::vector<Z> U, V;
    std::vector<double> S;
    int rows = 0, cols = 0, k = 0;
    bool ok = false;

    Z u(int r, int j) const { return U[static_cast<size_t>(j) * rows + r]; }
    Z v(int c, int j) const { return V[static_cast<size_t>(j) * cols + c]; }

    // M(r, c) = sum_j U(r,j) S(j) conj(V(c,j)). V is returned as V and not as
    // V-dagger, so the conjugate belongs here rather than in the seam.
    Z recon(int r, int c) const {
        Z acc(0.0, 0.0);
        for (int j = 0; j < k; ++j) acc += u(r, j) * S[static_cast<size_t>(j)] *
                                          std::conj(v(c, j));
        return acc;
    }
};

Svd run_svd(const std::vector<Z>& buf, int rows, int cols, MatrixOrder order,
            SVDMethod method) {
    Svd f;
    f.rows = rows;
    f.cols = cols;
    f.k = std::min(rows, cols);
    f.U.assign(static_cast<size_t>(rows) * f.k, Z(0.0, 0.0));
    f.V.assign(static_cast<size_t>(cols) * f.k, Z(0.0, 0.0));
    f.S.assign(static_cast<size_t>(f.k), 0.0);
    f.ok = svd_thin(buf.data(), rows, cols, order, method, f.U.data(),
                    f.S.data(), f.V.data());
    return f;
}

double frob_sq(const std::vector<Z>& a) {
    double t = 0.0;
    for (const Z& z : a) t += std::norm(z);
    return t;
}

// The eigenvalues a degenerate fixture is built from. Distinct so the ordering
// is checkable, dyadic so they are exact, and descending so `rank` of them
// followed by zeros is what an ascending solver must return reversed.
const std::vector<double>& gram_eigenvalues() {
    static const std::vector<double> l{1.0, 0.5, 0.25, 0.125};
    return l;
}

// An n x n Hermitian matrix of EXACTLY the given rank, with a known spectrum:
// G = sum_m lambda_m v_m v_m-dagger over the first `rank` columns of the
// unitary discrete Fourier matrix, which are orthonormal by construction.
//
// Built this way rather than as B B-dagger over some convenient B because rank
// then holds by construction instead of by luck. The obvious B, whose entries
// are affine in the row index, has rank 2 whatever its column count, and a
// fixture that is secretly rank-deficient would make the null-space bound below
// pass for the wrong reason.
//
// The Fourier vectors are complex for every m > 0, so G carries a nonzero
// imaginary part. That is required, not incidental: a real Hermitian matrix is
// its own transpose, so it cannot tell a storage-order flip from itself.
std::vector<Z> degenerate_gram(int n, int rank) {
    const auto& lambda = gram_eigenvalues();
    std::vector<Z> G(static_cast<size_t>(n) * n, Z(0.0, 0.0));
    const double inv_n = 1.0 / static_cast<double>(n);

    for (int m = 0; m < rank; ++m) {
        const double lm = lambda[static_cast<size_t>(m)];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // v_m(r) = exp(2 pi i m r / n) / sqrt(n), so
                // v_m(i) conj(v_m(j)) = exp(2 pi i m (i - j) / n) / n.
                const double phase = lindblad::TWO_PI * static_cast<double>(m) *
                                     static_cast<double>(i - j) * inv_n;
                G[static_cast<size_t>(j) * n + i] +=
                    lm * inv_n * Z(std::cos(phase), std::sin(phase));
            }
        }
    }
    return G;
}

}  // namespace

// =============================================================================
// svd_thin - shape and the documented output conventions
// =============================================================================

TEST(V11241EigenBackend, ThinShapesAreTheSmallerDimension) {
    for (auto [rows, cols] : {std::pair{6, 6}, std::pair{8, 3}, std::pair{3, 8}}) {
        const auto buf = asym_colmajor(rows, cols);
        const auto f = run_svd(buf, rows, cols, MatrixOrder::ColMajor,
                               SVDMethod::BDC);
        ASSERT_TRUE(f.ok) << rows << "x" << cols;
        EXPECT_EQ(f.k, std::min(rows, cols));
        EXPECT_EQ(f.U.size(), static_cast<size_t>(rows) * f.k);
        EXPECT_EQ(f.V.size(), static_cast<size_t>(cols) * f.k);
        EXPECT_EQ(f.S.size(), static_cast<size_t>(f.k));
    }
}

TEST(V11241EigenBackend, ReconstructsTheMatrixInBothOrientations) {
    for (auto [rows, cols] : {std::pair{6, 6}, std::pair{8, 3}, std::pair{3, 8}}) {
        const auto buf = asym_colmajor(rows, cols);
        const double scale = std::sqrt(frob_sq(buf));
        for (auto method : {SVDMethod::Jacobi, SVDMethod::BDC}) {
            const auto f = run_svd(buf, rows, cols, MatrixOrder::ColMajor, method);
            ASSERT_TRUE(f.ok);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    EXPECT_NEAR(std::abs(f.recon(r, c) - asym(r, c)), 0.0,
                                tol(std::max(rows, cols), scale))
                        << rows << "x" << cols << " at (" << r << "," << c << ")";
                }
            }
        }
    }
}

TEST(V11241EigenBackend, SingularValuesAreDescending) {
    const auto buf = asym_colmajor(7, 7);
    for (auto method : {SVDMethod::Jacobi, SVDMethod::BDC}) {
        const auto f = run_svd(buf, 7, 7, MatrixOrder::ColMajor, method);
        ASSERT_TRUE(f.ok);
        for (int i = 1; i < f.k; ++i) {
            EXPECT_LE(f.S[static_cast<size_t>(i)], f.S[static_cast<size_t>(i - 1)])
                << "sigma[" << i << "] rose above its predecessor";
        }
        EXPECT_GE(f.S[0], 0.0) << "singular values are non-negative by definition";
    }
}

TEST(V11241EigenBackend, FactorColumnsAreOrthonormal) {
    const int rows = 6, cols = 4;
    const auto buf = asym_colmajor(rows, cols);
    const auto f = run_svd(buf, rows, cols, MatrixOrder::ColMajor, SVDMethod::BDC);
    ASSERT_TRUE(f.ok);

    // U†U = I over the kept columns, and likewise V†V. The thin factors carry
    // k columns each, so both products are k x k.
    for (int a = 0; a < f.k; ++a) {
        for (int b = 0; b < f.k; ++b) {
            Z du(0.0, 0.0), dv(0.0, 0.0);
            for (int r = 0; r < rows; ++r) du += std::conj(f.u(r, a)) * f.u(r, b);
            for (int c = 0; c < cols; ++c) dv += std::conj(f.v(c, a)) * f.v(c, b);
            const double want = (a == b) ? 1.0 : 0.0;
            EXPECT_NEAR(std::abs(du - Z(want, 0.0)), 0.0, tol(rows))
                << "U column " << a << " against " << b;
            EXPECT_NEAR(std::abs(dv - Z(want, 0.0)), 0.0, tol(cols))
                << "V column " << a << " against " << b;
        }
    }
}

TEST(V11241EigenBackend, VIsReturnedAsVRatherThanVDagger) {
    // M V = U S is the identity that distinguishes the two conventions. Had the
    // seam returned V-dagger, this product would come out conjugated and the
    // imaginary parts would flip sign, which the asymmetric fixture exposes.
    const int n = 5;
    const auto buf = asym_colmajor(n, n);
    const double scale = std::sqrt(frob_sq(buf));
    const auto f = run_svd(buf, n, n, MatrixOrder::ColMajor, SVDMethod::BDC);
    ASSERT_TRUE(f.ok);

    for (int j = 0; j < f.k; ++j) {
        for (int r = 0; r < n; ++r) {
            Z mv(0.0, 0.0);
            for (int c = 0; c < n; ++c) mv += asym(r, c) * f.v(c, j);
            const Z us = f.u(r, j) * f.S[static_cast<size_t>(j)];
            EXPECT_NEAR(std::abs(mv - us), 0.0, tol(n, scale))
                << "(M V)[" << r << "," << j << "] != (U S)[" << r << "," << j
                << "]";
        }
    }
}

TEST(V11241EigenBackend, BothBackendsAgreeOnTheSpectrum) {
    const int n = 8;
    const auto buf = asym_colmajor(n, n);
    const double scale = std::sqrt(frob_sq(buf));
    const auto j = run_svd(buf, n, n, MatrixOrder::ColMajor, SVDMethod::Jacobi);
    const auto b = run_svd(buf, n, n, MatrixOrder::ColMajor, SVDMethod::BDC);
    ASSERT_TRUE(j.ok);
    ASSERT_TRUE(b.ok);
    ASSERT_EQ(j.k, b.k);
    for (int i = 0; i < j.k; ++i) {
        EXPECT_NEAR(j.S[static_cast<size_t>(i)], b.S[static_cast<size_t>(i)],
                    tol(n, scale))
            << "backends disagree at sigma[" << i << "]";
    }
}

// =============================================================================
// svd_thin - MatrixOrder, the parameter that fails silently
// =============================================================================

TEST(V11241EigenBackend, RowMajorAndColumnMajorDescribeTheSameMatrix) {
    // The same logical matrix in two layouts must give the same factorisation
    // of THAT matrix. Asserted through the reconstruction rather than the
    // spectrum: a buffer read in the wrong order is the transpose, and a
    // transpose carries identical singular values, so the spectrum is blind to
    // exactly the bug this test exists for.
    const int n = 6;
    const auto rm = asym_rowmajor(n, n);
    const auto cm = asym_colmajor(n, n);
    const double scale = std::sqrt(frob_sq(cm));

    const auto fr = run_svd(rm, n, n, MatrixOrder::RowMajor, SVDMethod::BDC);
    const auto fc = run_svd(cm, n, n, MatrixOrder::ColMajor, SVDMethod::BDC);
    ASSERT_TRUE(fr.ok);
    ASSERT_TRUE(fc.ok);

    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            EXPECT_NEAR(std::abs(fr.recon(r, c) - asym(r, c)), 0.0,
                        tol(n, scale))
                << "row-major reconstruction wrong at (" << r << "," << c << ")";
            EXPECT_NEAR(std::abs(fc.recon(r, c) - asym(r, c)), 0.0,
                        tol(n, scale))
                << "column-major reconstruction wrong at (" << r << "," << c
                << ")";
        }
    }
}

TEST(V11241EigenBackend, RowMajorHonoursRectangularShape) {
    // Non-square is where an ignored order flag cannot even be shaped away, so
    // this covers both orientations rather than the square case alone.
    for (auto [rows, cols] : {std::pair{7, 3}, std::pair{3, 7}}) {
        const auto rm = asym_rowmajor(rows, cols);
        const double scale = std::sqrt(frob_sq(rm));
        const auto f = run_svd(rm, rows, cols, MatrixOrder::RowMajor,
                               SVDMethod::Jacobi);
        ASSERT_TRUE(f.ok) << rows << "x" << cols;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                EXPECT_NEAR(std::abs(f.recon(r, c) - asym(r, c)), 0.0,
                            tol(std::max(rows, cols), scale))
                    << rows << "x" << cols << " at (" << r << "," << c << ")";
    }
}

TEST(V11241EigenBackend, NonFiniteInputIsRefusedRatherThanFactorised) {
    const int n = 4;
    auto buf = asym_colmajor(n, n);
    buf[5] = Z(std::numeric_limits<double>::quiet_NaN(), 0.0);
    const auto f = run_svd(buf, n, n, MatrixOrder::ColMajor, SVDMethod::Jacobi);
    EXPECT_FALSE(f.ok)
        << "a NaN entry was accepted; the caller's fallback route never runs "
           "and the unspecified outputs are read as a factorisation";
}

// =============================================================================
// eigh
// =============================================================================

TEST(V11241EigenBackend, EighReconstructsTheHermitianMatrix) {
    const int n = 6;
    const auto G = degenerate_gram(n, 4);
    const double scale = std::sqrt(frob_sq(G));
    std::vector<double> vals(static_cast<size_t>(n));
    std::vector<Z> vecs(static_cast<size_t>(n) * n);
    ASSERT_TRUE(eigh(G.data(), n, MatrixOrder::ColMajor, vals.data(),
                     vecs.data()));

    // A = sum_j lambda_j v_j v_j†
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            Z acc(0.0, 0.0);
            for (int j = 0; j < n; ++j) {
                acc += vals[static_cast<size_t>(j)] *
                       vecs[static_cast<size_t>(j) * n + r] *
                       std::conj(vecs[static_cast<size_t>(j) * n + c]);
            }
            EXPECT_NEAR(std::abs(acc - G[static_cast<size_t>(c) * n + r]), 0.0,
                        tol(n, scale))
                << "at (" << r << "," << c << ")";
        }
    }
}

TEST(V11241EigenBackend, EigenvaluesAreAscending) {
    // Documented and deliberate: the seam preserves the solver's order and the
    // caller reverses. A silent switch to descending would make every caller
    // read the null space as the leading directions.
    const int n = 6;
    const auto G = degenerate_gram(n, 4);
    std::vector<double> vals(static_cast<size_t>(n));
    std::vector<Z> vecs(static_cast<size_t>(n) * n);
    ASSERT_TRUE(eigh(G.data(), n, MatrixOrder::ColMajor, vals.data(),
                     vecs.data()));
    for (int i = 1; i < n; ++i) {
        EXPECT_GE(vals[static_cast<size_t>(i)], vals[static_cast<size_t>(i - 1)])
            << "eigenvalue " << i << " fell below its predecessor";
    }
}

TEST(V11241EigenBackend, EighAcceptsANullEigenvectorBuffer) {
    // Callers wanting only a spectrum pass nullptr, and must get the identical
    // eigenvalues rather than a different code path's answer.
    const int n = 6;
    const auto G = degenerate_gram(n, 4);
    std::vector<double> with(static_cast<size_t>(n)), without(static_cast<size_t>(n));
    std::vector<Z> vecs(static_cast<size_t>(n) * n);
    ASSERT_TRUE(eigh(G.data(), n, MatrixOrder::ColMajor, with.data(),
                     vecs.data()));
    ASSERT_TRUE(eigh(G.data(), n, MatrixOrder::ColMajor, without.data(),
                     nullptr));
    for (int i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(with[static_cast<size_t>(i)],
                         without[static_cast<size_t>(i)])
            << "spectrum-only path disagreed at " << i;
}

TEST(V11241EigenBackend, EighHonoursMatrixOrderOnAComplexHermitianMatrix) {
    // The trap the header names: a Hermitian matrix read in the opposite order
    // is its own conjugate, so it decomposes successfully and returns a wrong
    // answer. The eigenvalues are identical either way, so only the
    // eigenvectors can catch it, and only when the matrix has a nonzero
    // imaginary part. Both are true of this Gram matrix.
    const int n = 6;
    const auto cm = degenerate_gram(n, 4);
    std::vector<Z> rm(cm.size());
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            rm[static_cast<size_t>(r) * n + c] = cm[static_cast<size_t>(c) * n + r];

    const double scale = std::sqrt(frob_sq(cm));

    // A real Hermitian matrix equals its own conjugate, so an order flip on one
    // would be undetectable. The fixture must therefore differ from its
    // conjugate by more than the tolerance this test compares against, or the
    // test proves nothing about the flag.
    double conj_gap = 0.0;
    for (const Z& z : cm) conj_gap = std::max(conj_gap, 2.0 * std::abs(z.imag()));
    ASSERT_GT(conj_gap, tol(n, scale))
        << "the fixture is within tolerance of its own conjugate, so a storage "
           "order flip could not be observed through it";
    std::vector<double> vals(static_cast<size_t>(n));
    std::vector<Z> vecs(static_cast<size_t>(n) * n);
    ASSERT_TRUE(eigh(rm.data(), n, MatrixOrder::RowMajor, vals.data(),
                     vecs.data()));

    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            Z acc(0.0, 0.0);
            for (int j = 0; j < n; ++j) {
                acc += vals[static_cast<size_t>(j)] *
                       vecs[static_cast<size_t>(j) * n + r] *
                       std::conj(vecs[static_cast<size_t>(j) * n + c]);
            }
            EXPECT_NEAR(std::abs(acc - cm[static_cast<size_t>(c) * n + r]), 0.0,
                        tol(n, scale))
                << "row-major read reconstructed the conjugate at (" << r << ","
                << c << ")";
        }
    }
}

TEST(V11241EigenBackend, DegenerateNullSpaceResolvesToMachineEpsilon) {
    // The property the Eigen pin was chosen on. The fixture is built from an
    // orthonormal basis with a chosen spectrum, so BOTH halves are known: the
    // nonzero eigenvalues, and an exact null space of dimension n - rank whose
    // returned values are pure solver error.
    //
    // A solver resolving only to the SQUARE ROOT of epsilon puts that tail near
    // 1e-8, which is exactly what the Gram rescue's validity floor exists to
    // refuse; one resolving to epsilon puts it near 1e-16. The bound is derived
    // rather than transcribed, and is asserted to sit below sqrt(eps) so it
    // genuinely separates the two regimes.
    const int n = 6, rank = 4;
    const auto G = degenerate_gram(n, rank);
    const double scale = std::sqrt(frob_sq(G));
    std::vector<double> vals(static_cast<size_t>(n));
    ASSERT_TRUE(eigh(G.data(), n, MatrixOrder::ColMajor, vals.data(), nullptr));

    const double bound = tol(n, scale);
    ASSERT_LT(bound, std::sqrt(kEps))
        << "the bound must lie below sqrt(eps) or it cannot separate a "
           "squared-condition solver from an accurate one";

    for (int i = 0; i < n - rank; ++i) {
        EXPECT_LT(std::abs(vals[static_cast<size_t>(i)]), bound)
            << "null-space eigenvalue " << i << " is "
            << vals[static_cast<size_t>(i)] << ", the signature of a solver "
            << "resolving only to sqrt(eps)";
    }

    // The nonzero half, ascending against the chosen values descending. This is
    // what makes the test above meaningful: without it a solver returning all
    // zeros would satisfy the null-space bound perfectly.
    const auto& lambda = gram_eigenvalues();
    for (int i = 0; i < rank; ++i) {
        const double want = lambda[static_cast<size_t>(rank - 1 - i)];
        EXPECT_NEAR(vals[static_cast<size_t>(n - rank + i)], want, tol(n, scale))
            << "eigenvalue " << (n - rank + i) << " should be " << want;
    }
}

// =============================================================================
// svd_verify - the residual the truncation ladder accepts or rejects on
// =============================================================================

TEST(V11241SvdVerify, ExactFactorisationHasVanishingResidual) {
    const int n = 6;
    const auto buf = asym_colmajor(n, n);
    const double fsq = frob_sq(buf);
    const auto f = run_svd(buf, n, n, MatrixOrder::ColMajor, SVDMethod::BDC);
    ASSERT_TRUE(f.ok);

    const double resid = svd_reconstruction_residual_sq(
        buf.data(), n, n, MatrixOrder::ColMajor, f.U.data(), f.S.data(),
        f.V.data(), f.k);
    EXPECT_LT(resid, tol(n, fsq))
        << "a full-rank factorisation left " << resid << " of residual";
}

TEST(V11241SvdVerify, TruncatedResidualEqualsTheDiscardedWeight) {
    // A truncated SVD satisfies the Frobenius identity with equality, so the
    // residual at rank k is exactly the weight of the dropped directions. That
    // makes the expected value derivable from the spectrum the same call
    // returned, rather than transcribed from a run.
    const int n = 7;
    const auto buf = asym_colmajor(n, n);
    const double fsq = frob_sq(buf);
    const auto f = run_svd(buf, n, n, MatrixOrder::ColMajor, SVDMethod::Jacobi);
    ASSERT_TRUE(f.ok);

    for (int k = 1; k <= f.k; ++k) {
        double dropped = 0.0;
        for (int i = k; i < f.k; ++i)
            dropped += f.S[static_cast<size_t>(i)] * f.S[static_cast<size_t>(i)];

        const double resid = svd_reconstruction_residual_sq(
            buf.data(), n, n, MatrixOrder::ColMajor, f.U.data(), f.S.data(),
            f.V.data(), k);
        EXPECT_NEAR(resid, dropped, tol(n, fsq))
            << "at k=" << k << " the residual and the discarded weight parted";
    }
}

TEST(V11241SvdVerify, ResidualAgreesAcrossStorageOrders) {
    const int n = 5;
    const auto rm = asym_rowmajor(n, n);
    const auto cm = asym_colmajor(n, n);
    const double fsq = frob_sq(cm);

    const auto fr = run_svd(rm, n, n, MatrixOrder::RowMajor, SVDMethod::BDC);
    const auto fc = run_svd(cm, n, n, MatrixOrder::ColMajor, SVDMethod::BDC);
    ASSERT_TRUE(fr.ok);
    ASSERT_TRUE(fc.ok);

    for (int k = 1; k <= fr.k; ++k) {
        const double a = svd_reconstruction_residual_sq(
            rm.data(), n, n, MatrixOrder::RowMajor, fr.U.data(), fr.S.data(),
            fr.V.data(), k);
        const double b = svd_reconstruction_residual_sq(
            cm.data(), n, n, MatrixOrder::ColMajor, fc.U.data(), fc.S.data(),
            fc.V.data(), k);
        EXPECT_NEAR(a, b, tol(n, fsq)) << "orders disagree at k=" << k;
    }
}
