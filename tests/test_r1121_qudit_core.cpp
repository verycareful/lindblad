// R.1.12.1 total-coverage suite, Batch 4: qudit core (QuditStatevector +
// qudit_gates). Plan: docs (R.1.12.1 coverage plan), "Batch 4".
//
// Index/digit conversions, ipow, gate-matrix unitarity, the d=2 cross-checks to
// the qubit layer, the LSB-first apply_2qudit/cadd convention, oracles, and
// seeded measurement. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

void expect_unitary(const std::vector<Complex128>& M, int d, double tol = 1e-9) {
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) {
            Complex128 acc(0, 0);
            for (int k = 0; k < d; ++k)
                acc += M[k * d + i].conj() * M[k * d + j];
            EXPECT_NEAR(acc.real, (i == j) ? 1.0 : 0.0, tol) << "(" << i << "," << j << ")";
            EXPECT_NEAR(acc.imag, 0.0, tol);
        }
}

void set_basis(QuditStatevector& qsv, size_t idx) {
    std::fill(qsv.amplitudes.begin(), qsv.amplitudes.end(), Complex128(0, 0));
    qsv.amplitudes[idx] = Complex128(1, 0);
}

}  // namespace

// =============================================================================
// index <-> digits, ipow
// =============================================================================

TEST(R1121QuditCore, IndexDigitRoundTripAsymmetric) {
    for (int d : {2, 3, 5, 7}) {
        std::vector<int> digits = {d - 1, 0, (d > 1 ? 1 : 0)};
        size_t idx = QuditStatevector::digits_to_index(digits, d);
        auto back = QuditStatevector::index_to_digits(idx, d, 3);
        SCOPED_TRACE("d = " + std::to_string(d));
        EXPECT_EQ(back, digits);
    }
    // explicit value: d=5, digits {2,0,4} -> 2 + 0*5 + 4*25 = 102.
    EXPECT_EQ(QuditStatevector::digits_to_index({2, 0, 4}, 5), 102u);
}

TEST(R1121QuditCore, IpowExactInteger) {
    EXPECT_EQ(QuditStatevector::ipow(3, 0), 1u);
    EXPECT_EQ(QuditStatevector::ipow(3, 4), 81u);
    EXPECT_EQ(QuditStatevector::ipow(7, 3), 343u);
}

// =============================================================================
// Construction / norm
// =============================================================================

TEST(R1121QuditCore, ConstructInitialiseNorm) {
    QuditStatevector qsv(3, 4);
    EXPECT_EQ(qsv.n_qudits, 3);
    EXPECT_EQ(qsv.d, 4);
    EXPECT_EQ(qsv.dim, 64u);  // 4^3
    EXPECT_NEAR(qsv.norm_sq(), 1.0, 1e-12);
    EXPECT_NEAR(qsv.amplitudes[0].real, 1.0, 1e-12);  // |0...0>
}

// =============================================================================
// qudit_gates: unitarity and d=2 cross-checks
// =============================================================================

TEST(R1121QuditCore, QftAndIqftAreUnitaryAndInverse) {
    for (int d : {2, 3, 4, 5}) {
        auto F = qudit_gates::qft_matrix(d);
        auto Fd = qudit_gates::iqft_matrix(d);
        SCOPED_TRACE("d = " + std::to_string(d));
        expect_unitary(F, d);
        // F * Fdag == I.
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) {
                Complex128 acc(0, 0);
                for (int k = 0; k < d; ++k) acc += F[i * d + k] * Fd[k * d + j];
                EXPECT_NEAR(acc.real, (i == j) ? 1.0 : 0.0, 1e-9);
                EXPECT_NEAR(acc.imag, 0.0, 1e-9);
            }
    }
}

TEST(R1121QuditCore, DimensionTwoGatesMatchQubitGates) {
    auto X = qudit_gates::shift_matrix(2, 1);
    EXPECT_NEAR(X[0].real, 0.0, 1e-12);
    EXPECT_NEAR(X[1].real, 1.0, 1e-12);
    EXPECT_NEAR(X[2].real, 1.0, 1e-12);
    EXPECT_NEAR(X[3].real, 0.0, 1e-12);

    auto H = qudit_gates::qft_matrix(2);
    const double s = INV_SQRT2;
    EXPECT_NEAR(H[0].real, s, 1e-12);
    EXPECT_NEAR(H[1].real, s, 1e-12);
    EXPECT_NEAR(H[2].real, s, 1e-12);
    EXPECT_NEAR(H[3].real, -s, 1e-12);
}

TEST(R1121QuditCore, ShiftMatrixIsCyclic) {
    const int d = 3;
    QuditStatevector qsv(1, d);  // |0>
    qsv.apply_1qudit(0, qudit_gates::shift_matrix(d, 1));
    EXPECT_EQ(qsv.measure(1)[0], 1);  // |0> -> |1>
    qsv.apply_1qudit(0, qudit_gates::shift_matrix(d, 1));
    qsv.apply_1qudit(0, qudit_gates::shift_matrix(d, 1));
    EXPECT_EQ(qsv.measure(1)[0], 0);  // back to |0> after d shifts
}

// =============================================================================
// apply_2qudit: LSB-first cadd convention
// =============================================================================

TEST(R1121QuditCore, CaddUsesFirstOperandAsControlLsb) {
    const int d = 3;
    QuditStatevector qsv(2, d);
    set_basis(qsv, QuditStatevector::digits_to_index({1, 0}, d));  // control=1, target=0
    qsv.apply_2qudit(0, 1, qudit_gates::cadd_matrix(d, 1));        // q0=control, q1=target
    auto m = qsv.measure(5);
    EXPECT_EQ(m[0], 1) << "control unchanged";
    EXPECT_EQ(m[1], 1) << "target += 1*control = 1";
}

TEST(R1121QuditCore, ApplyKQuditRejectsDuplicateIndices) {
    QuditStatevector qsv(2, 2);
    std::vector<Complex128> I16(16, Complex128(0, 0));
    for (int i = 0; i < 4; ++i) I16[i * 4 + i] = Complex128(1, 0);
    EXPECT_THROW(qsv.apply_kqudit({0, 0}, I16), std::invalid_argument);
}

// apply_kqudit on k=3 NON-CONTIGUOUS, UNSORTED qudits. A product gate
// U = U_c (x) U_b (x) U_a (LSB = qudits[0] = a) must act exactly like applying
// the three single-qudit gates separately — a convention-pinned cross-check
// that needs no hand-computed reference.
TEST(R1121QuditCore, ApplyKQuditThreeNonContiguousMatchesSequential) {
    const int d = 3;
    auto Ua = qudit_gates::shift_matrix(d, 1);
    auto Ub = qudit_gates::qft_matrix(d);
    auto Uc = qudit_gates::shift_matrix(d, 2);
    // qudits[0]=3 (LSB) -> Ua, qudits[1]=0 -> Ub, qudits[2]=2 -> Uc.
    const std::vector<int> qudits = {3, 0, 2};
    const int D = d * d * d;
    std::vector<Complex128> U(static_cast<size_t>(D) * D, Complex128(0, 0));
    for (int x0p = 0; x0p < d; ++x0p)
     for (int x1p = 0; x1p < d; ++x1p)
      for (int x2p = 0; x2p < d; ++x2p)
       for (int x0 = 0; x0 < d; ++x0)
        for (int x1 = 0; x1 < d; ++x1)
         for (int x2 = 0; x2 < d; ++x2) {
            int r = x2p * d * d + x1p * d + x0p;
            int c = x2 * d * d + x1 * d + x0;
            U[size_t(r) * D + c] = Ua[x0p * d + x0] * Ub[x1p * d + x1] * Uc[x2p * d + x2];
         }

    // Spread the amplitudes first so the test is not trivially diagonal.
    QuditStatevector a(4, d);
    a.apply_1qudit(0, qudit_gates::qft_matrix(d));
    a.apply_1qudit(2, qudit_gates::shift_matrix(d, 1));
    QuditStatevector b = a;  // copy

    a.apply_kqudit(qudits, U);
    b.apply_1qudit(3, Ua);
    b.apply_1qudit(0, Ub);
    b.apply_1qudit(2, Uc);

    ASSERT_EQ(a.amplitudes.size(), b.amplitudes.size());
    for (size_t i = 0; i < a.amplitudes.size(); ++i) {
        EXPECT_NEAR(a.amplitudes[i].real, b.amplitudes[i].real, 1e-9) << "re @ " << i;
        EXPECT_NEAR(a.amplitudes[i].imag, b.amplitudes[i].imag, 1e-9) << "im @ " << i;
    }
}

TEST(R1121QuditCore, ShiftNegativeAndCyclicAcrossDimensions) {
    for (int d = 2; d <= 7; ++d) {
        SCOPED_TRACE("d = " + std::to_string(d));
        // X^{-1}|0> = |d-1>.
        QuditStatevector qsv(1, d);
        qsv.apply_1qudit(0, qudit_gates::shift_matrix(d, -1));
        EXPECT_EQ(qsv.measure(7)[0], d - 1);
        // X^{-1} then X returns to |0>.
        qsv.apply_1qudit(0, qudit_gates::shift_matrix(d, 1));
        EXPECT_EQ(qsv.measure(7)[0], 0);
        // shift(d, m) == shift(d, m + d) (cyclic / normalised exponent).
        auto a = qudit_gates::shift_matrix(d, 2);
        auto bm = qudit_gates::shift_matrix(d, 2 + d);
        for (size_t i = 0; i < a.size(); ++i)
            EXPECT_NEAR(a[i].real, bm[i].real, 1e-12);
    }
}

TEST(R1121QuditCore, CaddActionTableAllInputs) {
    const int d = 3;
    for (int s : {1, 2}) {
        for (int x = 0; x < d; ++x)       // control (qudit 0, LSB)
            for (int y = 0; y < d; ++y) {  // target (qudit 1)
                QuditStatevector qsv(2, d);
                set_basis(qsv, QuditStatevector::digits_to_index({x, y}, d));
                qsv.apply_2qudit(0, 1, qudit_gates::cadd_matrix(d, s));
                auto m = qsv.measure(11);
                SCOPED_TRACE("s=" + std::to_string(s) + " x=" + std::to_string(x) +
                             " y=" + std::to_string(y));
                EXPECT_EQ(m[0], x) << "control preserved";
                EXPECT_EQ(m[1], (y + s * x) % d) << "target += s*control mod d";
            }
    }
}

TEST(R1121QuditCore, CaddDimensionTwoEqualsCx) {
    // cadd(2,1) with control = LSB equals the qubit-layer CX permutation
    // 0->0, 1->3, 2->2, 3->1 (control q0, target q1).
    auto U = qudit_gates::cadd_matrix(2, 1);
    const int perm[4] = {0, 3, 2, 1};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            double want = (row == perm[col]) ? 1.0 : 0.0;
            EXPECT_NEAR(U[row * 4 + col].real, want, 1e-12) << "(" << row << "," << col << ")";
        }
}

TEST(R1121QuditCore, ControlledPowerActionAndIdentityAtKZero) {
    const int d = 3;
    auto X = qudit_gates::shift_matrix(d, 1);
    // k = 2, U = shift: control c applies X^{2c} -> target = (t + 2c) mod d.
    auto M = qudit_gates::controlled_power_matrix(d, X, 2);
    expect_unitary(M, d * d);
    for (int c = 0; c < d; ++c)
        for (int t = 0; t < d; ++t) {
            QuditStatevector qsv(2, d);
            set_basis(qsv, QuditStatevector::digits_to_index({c, t}, d));  // ctrl=LSB
            qsv.apply_2qudit(0, 1, M);
            auto m = qsv.measure(5);
            SCOPED_TRACE("c=" + std::to_string(c) + " t=" + std::to_string(t));
            EXPECT_EQ(m[0], c);
            EXPECT_EQ(m[1], (t + 2 * c) % d);
        }
    // k = 0 is the identity regardless of control.
    auto M0 = qudit_gates::controlled_power_matrix(d, X, 0);
    for (int i = 0; i < d * d; ++i)
        for (int j = 0; j < d * d; ++j)
            EXPECT_NEAR(M0[i * d * d + j].real, (i == j) ? 1.0 : 0.0, 1e-12);
}

TEST(R1121QuditCore, QftUnitaryAcrossDimensionsTwoToSeven) {
    for (int d = 2; d <= 7; ++d) {
        SCOPED_TRACE("d = " + std::to_string(d));
        expect_unitary(qudit_gates::qft_matrix(d), d);
        expect_unitary(qudit_gates::iqft_matrix(d), d);
        // QFT of a uniform shift eigenstate concentrates — sanity that QFT|0>
        // is the uniform superposition (all amplitudes 1/sqrt(d)).
        QuditStatevector qsv(1, d);
        qsv.apply_1qudit(0, qudit_gates::qft_matrix(d));
        for (int k = 0; k < d; ++k)
            EXPECT_NEAR(qsv.amplitudes[k].real, 1.0 / std::sqrt((double)d), 1e-9);
    }
}

// =============================================================================
// Oracles and measurement
// =============================================================================

TEST(R1121QuditCore, FunctionOracleAddsImageToOutputRegister) {
    const int d = 3;
    QuditStatevector qsv(2, d);  // qudit0 = query, qudit1 = output
    set_basis(qsv, QuditStatevector::digits_to_index({1, 0}, d));  // query=1, output=0
    qsv.apply_function_oracle(1, 1, [](const std::vector<int>& x) {
        return std::vector<int>{(x[0] + 1) % 3};
    });
    auto m = qsv.measure(2);
    EXPECT_EQ(m[0], 1) << "query preserved";
    EXPECT_EQ(m[1], 2) << "output = (0 + f(1)) where f(1) = 2";
}

TEST(R1121QuditCore, PhaseOracleMultipliesAmplitude) {
    QuditStatevector qsv(2, 3);  // |00>
    qsv.apply_phase_oracle([](const std::vector<int>&) { return Complex128(0, 1); });
    EXPECT_NEAR(qsv.amplitudes[0].real, 0.0, 1e-12);
    EXPECT_NEAR(qsv.amplitudes[0].imag, 1.0, 1e-12);
}

TEST(R1121QuditCore, MeasureIsSeedDeterministic) {
    QuditStatevector qsv(2, 3);
    qsv.apply_1qudit(0, qudit_gates::qft_matrix(3));  // superposition on qudit 0
    EXPECT_EQ(qsv.measure(123), qsv.measure(123));
}
