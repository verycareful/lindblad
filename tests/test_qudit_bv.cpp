#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_simulator.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;
using namespace lindblad::qudit_gates;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double kTol = 1e-10;

// Matrix multiply: C(rows×cols) = A(rows×inner) · B(inner×cols), all row-major.
static std::vector<Complex128> mat_mul(
    const std::vector<Complex128>& A,
    const std::vector<Complex128>& B,
    int rows, int inner, int cols)
{
    std::vector<Complex128> C(
        static_cast<size_t>(rows * cols), Complex128(0.0, 0.0));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            Complex128 acc(0.0, 0.0);
            for (int k = 0; k < inner; ++k)
                acc += A[static_cast<size_t>(r * inner + k)]
                     * B[static_cast<size_t>(k * cols + c)];
            C[static_cast<size_t>(r * cols + c)] = acc;
        }
    return C;
}

// Conjugate transpose of an n×n row-major matrix.
static std::vector<Complex128> conj_transpose(
    const std::vector<Complex128>& M, int n)
{
    std::vector<Complex128> Mt(static_cast<size_t>(n * n));
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            Mt[static_cast<size_t>(k * n + j)] =
                M[static_cast<size_t>(j * n + k)].conj();
    return Mt;
}

// Unitarity check: M · M† ≈ I.
static bool is_unitary(const std::vector<Complex128>& M, int n) {
    auto Md = conj_transpose(M, n);
    auto I  = mat_mul(M, Md, n, n, n);
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
            const double expected_re = (j == k) ? 1.0 : 0.0;
            if (std::abs(I[static_cast<size_t>(j * n + k)].real - expected_re) > kTol)
                return false;
            if (std::abs(I[static_cast<size_t>(j * n + k)].imag) > kTol)
                return false;
        }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditStatevector
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditStatevector, InitializesToZeroState) {
    QuditStatevector sv(3, 3);
    EXPECT_EQ(sv.dim, 27u);
    EXPECT_NEAR(sv.amplitudes[0].real, 1.0, kTol);
    EXPECT_NEAR(sv.amplitudes[0].imag, 0.0, kTol);
    for (size_t i = 1; i < sv.dim; ++i) {
        EXPECT_NEAR(sv.amplitudes[i].real, 0.0, kTol) << "at index " << i;
        EXPECT_NEAR(sv.amplitudes[i].imag, 0.0, kTol) << "at index " << i;
    }
}

TEST(QuditStatevector, NormSqIsOne) {
    QuditStatevector sv(4, 2);
    EXPECT_NEAR(sv.norm_sq(), 1.0, kTol);
}

TEST(QuditStatevector, InvalidDThrows) {
    EXPECT_THROW(QuditStatevector(1, 1), std::invalid_argument);
}

TEST(QuditStatevector, InvalidNQuditsThrows) {
    EXPECT_THROW(QuditStatevector(0, 3), std::invalid_argument);
}

TEST(QuditStatevector, Apply1QuditShiftD3) {
    QuditStatevector sv(1, 3);  // |0>
    auto X = shift_matrix(3, 1);
    sv.apply_1qudit(0, X);
    EXPECT_NEAR(sv.amplitudes[0].real, 0.0, kTol);
    EXPECT_NEAR(sv.amplitudes[1].real, 1.0, kTol);
    EXPECT_NEAR(sv.amplitudes[2].real, 0.0, kTol);
}

TEST(QuditStatevector, Apply1QuditShiftTwiceD3) {
    QuditStatevector sv(1, 3);
    auto X = shift_matrix(3, 1);
    sv.apply_1qudit(0, X);
    sv.apply_1qudit(0, X);
    EXPECT_NEAR(sv.amplitudes[2].real, 1.0, kTol);
}

TEST(QuditStatevector, Apply1QuditShiftThreeTimesD3Cycles) {
    QuditStatevector sv(1, 3);
    auto X = shift_matrix(3, 1);
    sv.apply_1qudit(0, X);
    sv.apply_1qudit(0, X);
    sv.apply_1qudit(0, X);
    EXPECT_NEAR(sv.amplitudes[0].real, 1.0, kTol);
}

TEST(QuditStatevector, Apply1QuditPreservesNorm) {
    // Apply QFT — must preserve unit norm
    QuditStatevector sv(3, 3);
    auto F = qft_matrix(3);
    sv.apply_1qudit(1, F);
    EXPECT_NEAR(sv.norm_sq(), 1.0, kTol);
}

TEST(QuditStatevector, IndexToDigitsRoundtrip) {
    // d=3, n=3: idx 5 = 2·3^0 + 1·3^1 = {2, 1, 0}
    auto digits = QuditStatevector::index_to_digits(5, 3, 3);
    ASSERT_EQ(digits.size(), 3u);
    EXPECT_EQ(digits[0], 2);
    EXPECT_EQ(digits[1], 1);
    EXPECT_EQ(digits[2], 0);
    EXPECT_EQ(QuditStatevector::digits_to_index(digits, 3), 5u);
}

TEST(QuditStatevector, IndexToDigitsAllZeros) {
    auto digits = QuditStatevector::index_to_digits(0, 4, 4);
    for (int x : digits) EXPECT_EQ(x, 0);
    EXPECT_EQ(QuditStatevector::digits_to_index(digits, 4), 0u);
}

TEST(QuditStatevector, MeasureAfterShiftReturnsDigit) {
    // Apply shift twice to a single qutrit: state = |2>, measurement = {2}
    QuditStatevector sv(1, 3);
    auto X = shift_matrix(3, 1);
    sv.apply_1qudit(0, X);
    sv.apply_1qudit(0, X);
    auto outcome = sv.measure(42);
    ASSERT_EQ(outcome.size(), 1u);
    EXPECT_EQ(outcome[0], 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditGates
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditGates, QFTMatrixUnitaryD2) { EXPECT_TRUE(is_unitary(qft_matrix(2), 2)); }
TEST(QuditGates, QFTMatrixUnitaryD3) { EXPECT_TRUE(is_unitary(qft_matrix(3), 3)); }
TEST(QuditGates, QFTMatrixUnitaryD4) { EXPECT_TRUE(is_unitary(qft_matrix(4), 4)); }
TEST(QuditGates, QFTMatrixUnitaryD5) { EXPECT_TRUE(is_unitary(qft_matrix(5), 5)); }
TEST(QuditGates, QFTMatrixUnitaryD7) { EXPECT_TRUE(is_unitary(qft_matrix(7), 7)); }

TEST(QuditGates, IQFTIsConjugateTransposeOfQFT) {
    for (int d : {2, 3, 4, 5, 7}) {
        auto F  = qft_matrix(d);
        auto Fi = iqft_matrix(d);
        auto Ft = conj_transpose(F, d);
        for (int j = 0; j < d * d; ++j) {
            EXPECT_NEAR(Fi[static_cast<size_t>(j)].real,
                        Ft[static_cast<size_t>(j)].real, kTol)
                << "d=" << d << " j=" << j;
            EXPECT_NEAR(Fi[static_cast<size_t>(j)].imag,
                        Ft[static_cast<size_t>(j)].imag, kTol)
                << "d=" << d << " j=" << j;
        }
    }
}

TEST(QuditGates, QFTReducesToHadamardForD2) {
    // For d=2: F[j,k] = (1/sqrt(2)) * (-1)^{jk}
    auto F = qft_matrix(2);
    const double h = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(F[0].real,  h, kTol);  // F[0,0]
    EXPECT_NEAR(F[1].real,  h, kTol);  // F[0,1]
    EXPECT_NEAR(F[2].real,  h, kTol);  // F[1,0]
    EXPECT_NEAR(F[3].real, -h, kTol);  // F[1,1]
    for (auto& a : F) EXPECT_NEAR(a.imag, 0.0, kTol);
}

TEST(QuditGates, ShiftMatrixD3) {
    auto X = shift_matrix(3, 1);
    // Column 0 (input |0>): non-zero in row 1 → |1>
    EXPECT_NEAR(X[static_cast<size_t>(1 * 3 + 0)].real, 1.0, kTol);
    // Column 1 (input |1>): non-zero in row 2 → |2>
    EXPECT_NEAR(X[static_cast<size_t>(2 * 3 + 1)].real, 1.0, kTol);
    // Column 2 (input |2>): non-zero in row 0 → |0>
    EXPECT_NEAR(X[static_cast<size_t>(0 * 3 + 2)].real, 1.0, kTol);
}

TEST(QuditGates, ShiftMatrixIsUnitary) {
    for (int d : {2, 3, 4, 5, 7}) {
        EXPECT_TRUE(is_unitary(shift_matrix(d, 1), d)) << "d=" << d;
    }
}

TEST(QuditGates, CADDMatrixIsUnitary_D2_S1) {
    EXPECT_TRUE(is_unitary(cadd_matrix(2, 1), 4));  // d² = 4
}
TEST(QuditGates, CADDMatrixIsUnitary_D3_S1) {
    EXPECT_TRUE(is_unitary(cadd_matrix(3, 1), 9));
}
TEST(QuditGates, CADDMatrixIsUnitary_D3_S2) {
    EXPECT_TRUE(is_unitary(cadd_matrix(3, 2), 9));
}
TEST(QuditGates, CADDMatrixIsUnitary_D4_S3) {
    EXPECT_TRUE(is_unitary(cadd_matrix(4, 3), 16));
}
TEST(QuditGates, CADDMatrixIsUnitary_D5_S4) {
    EXPECT_TRUE(is_unitary(cadd_matrix(5, 4), 25));
}

TEST(QuditGates, CADDMatrixCorrect_D3_S1) {
    // CADD_1: |x>|y> → |x>|(y+x) mod 3>
    auto U = cadd_matrix(3, 1);
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            QuditStatevector sv(2, 3);
            std::fill(sv.amplitudes.begin(), sv.amplitudes.end(),
                      Complex128(0.0, 0.0));
            // Set to |x, y>: index = x*3^0 + y*3^1 = x + 3y
            sv.amplitudes[static_cast<size_t>(x + 3 * y)] = Complex128(1.0, 0.0);
            sv.apply_2qudit(0, 1, U);
            const int expected_idx = x + 3 * ((y + x) % 3);
            EXPECT_NEAR(sv.amplitudes[static_cast<size_t>(expected_idx)].real,
                        1.0, kTol)
                << "CADD_1 failed for |" << x << "," << y << ">";
        }
    }
}

TEST(QuditGates, CADDMatrixIdentityWhenS0) {
    auto U = cadd_matrix(3, 0);
    const int dd = 9;
    for (int r = 0; r < dd; ++r)
        for (int c = 0; c < dd; ++c) {
            const double expected = (r == c) ? 1.0 : 0.0;
            EXPECT_NEAR(U[static_cast<size_t>(r * dd + c)].real, expected, kTol);
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditBernsteinVazirani
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditBV, RecoversSecret_D2_101) {
    // d=2 is qubit BV — must give same result as standard BV
    std::vector<int> secret = {1, 0, 1};
    auto result = QuditBernsteinVazirani::solve(secret, 2);
    EXPECT_EQ(result.secret, secret);
    EXPECT_EQ(result.d, 2);
    EXPECT_EQ(result.n, 3);
}

TEST(QuditBV, RecoversSecret_D2_AllZeros) {
    std::vector<int> secret = {0, 0, 0};
    auto result = QuditBernsteinVazirani::solve(secret, 2);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, RecoversSecret_D2_AllOnes) {
    std::vector<int> secret = {1, 1, 1, 1};
    auto result = QuditBernsteinVazirani::solve(secret, 2);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, RecoversSecret_D3_210) {
    std::vector<int> secret = {2, 1, 0};
    auto result = QuditBernsteinVazirani::solve(secret, 3);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, RecoversSecret_D3_AllZeros) {
    std::vector<int> secret = {0, 0, 0};
    auto result = QuditBernsteinVazirani::solve(secret, 3);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, RecoversSecret_D3_AllTwos) {
    std::vector<int> secret = {2, 2, 2};
    auto result = QuditBernsteinVazirani::solve(secret, 3);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, RecoversSecret_D4_3201) {
    std::vector<int> secret = {3, 2, 0, 1};
    auto result = QuditBernsteinVazirani::solve(secret, 4);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, RecoversSecret_D5_4321) {
    std::vector<int> secret = {4, 3, 2, 1};
    auto result = QuditBernsteinVazirani::solve(secret, 5);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, RecoversSecret_D7_SingleQudit) {
    std::vector<int> secret = {5};
    auto result = QuditBernsteinVazirani::solve(secret, 7);
    EXPECT_EQ(result.secret, secret);
    EXPECT_EQ(result.n, 1);
}

TEST(QuditBV, RecoversSecret_D3_N6_Random) {
    std::vector<int> secret = {2, 0, 1, 2, 1, 0};
    auto result = QuditBernsteinVazirani::solve(secret, 3);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, RecoversSecret_D6_Composite) {
    // d=6 composite — Z_6 is a ring, not a field; BV still works
    std::vector<int> secret = {5, 3, 1, 4};
    auto result = QuditBernsteinVazirani::solve(secret, 6);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, MultipleShots_StillCorrect) {
    std::vector<int> secret = {1, 2, 0};
    auto result = QuditBernsteinVazirani::solve(secret, 3, 5, 42);
    EXPECT_EQ(result.secret, secret);
}

TEST(QuditBV, OracleGateIsUnitary_D3_S1) {
    auto U = QuditBernsteinVazirani::oracle_gate(3, 1);
    EXPECT_TRUE(is_unitary(U, 9));
}

TEST(QuditBV, OracleGateIsUnitary_D4_S3) {
    auto U = QuditBernsteinVazirani::oracle_gate(4, 3);
    EXPECT_TRUE(is_unitary(U, 16));
}

TEST(QuditBV, InvalidInput_DLessThan2_Throws) {
    EXPECT_THROW(
        QuditBernsteinVazirani::solve({1, 0}, 1),
        std::invalid_argument);
}

TEST(QuditBV, InvalidInput_EmptySecret_Throws) {
    EXPECT_THROW(
        QuditBernsteinVazirani::solve({}, 3),
        std::invalid_argument);
}

TEST(QuditBV, InvalidInput_SecretValueOutOfRange_Throws) {
    EXPECT_THROW(
        QuditBernsteinVazirani::solve({1, 3}, 3),  // 3 >= d=3
        std::invalid_argument);
}

TEST(QuditBV, InvalidInput_NegativeSecretValue_Throws) {
    EXPECT_THROW(
        QuditBernsteinVazirani::solve({1, -1}, 3),
        std::invalid_argument);
}
