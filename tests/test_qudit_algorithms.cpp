#include "lindblad/algorithms.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool all_zero(const std::vector<int>& v) {
    for (int x : v) if (x != 0) return false;
    return true;
}

// Coset-canonical Simon oracle: f(x) = lexicographic min of {x, x+s, x+2s,...} mod d
static std::function<std::vector<int>(const std::vector<int>&)>
make_simon_f(const std::vector<int>& s, int n, int d) {
    return [s, n, d](const std::vector<int>& x) -> std::vector<int> {
        std::vector<int> best = x;
        std::vector<int> cur(static_cast<size_t>(n));
        for (int k = 1; k < d; ++k) {
            for (int i = 0; i < n; ++i)
                cur[static_cast<size_t>(i)] =
                    (x[static_cast<size_t>(i)] +
                     k * s[static_cast<size_t>(i)]) % d;
            if (cur < best) best = cur;
        }
        return best;
    };
}

// Eigenstate of shift_matrix(d,1) for eigenvalue exp(2*pi*i*k/d).
// Forward shift X|j> = |(j+1)%d> has eigenstates |ũ_k>[j] = ω^{-jk}/√d.
static std::vector<Complex128> shift_eigenstate(int d, int k) {
    std::vector<Complex128> psi(static_cast<size_t>(d));
    const double norm = 1.0 / std::sqrt(static_cast<double>(d));
    const double two_pi_over_d = 2.0 * M_PI / static_cast<double>(d);
    for (int j = 0; j < d; ++j)
        psi[static_cast<size_t>(j)] =
            Complex128::exp_i(-two_pi_over_d * static_cast<double>(k * j)) * norm;
    return psi;
}

// Check that p is a non-zero scalar multiple of s in GF(d) (d prime).
// Any such p is a valid Simon period for the same hidden subgroup.
static bool is_simon_period(const std::vector<int>& p,
                             const std::vector<int>& s, int d) {
    for (int k = 1; k < d; ++k) {
        bool match = true;
        for (int i = 0; i < static_cast<int>(p.size()) && match; ++i)
            match = ((k * s[static_cast<size_t>(i)]) % d == p[static_cast<size_t>(i)]);
        if (match) return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditDeutschJozsa
// ─────────────────────────────────────────────────────────────────────────────

class QuditDeutschJozsaTest : public ::testing::Test {};

// Constant functions
TEST_F(QuditDeutschJozsaTest, ConstantZero_d2_n1) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    auto r = QuditDeutschJozsa::solve(1, 2, f, 42);
    EXPECT_EQ(r.verdict, QuditDeutschJozsa::Verdict::CONSTANT);
    EXPECT_EQ(r.d, 2); EXPECT_EQ(r.n, 1);
}
TEST_F(QuditDeutschJozsaTest, ConstantOne_d2_n2) {
    auto f = [](const std::vector<int>&) -> int { return 1; };
    EXPECT_EQ(QuditDeutschJozsa::solve(2, 2, f).verdict,
              QuditDeutschJozsa::Verdict::CONSTANT);
}
TEST_F(QuditDeutschJozsaTest, ConstantZero_d3_n1) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    EXPECT_EQ(QuditDeutschJozsa::solve(1, 3, f).verdict,
              QuditDeutschJozsa::Verdict::CONSTANT);
}
TEST_F(QuditDeutschJozsaTest, ConstantTwo_d3_n2) {
    auto f = [](const std::vector<int>&) -> int { return 2; };
    EXPECT_EQ(QuditDeutschJozsa::solve(2, 3, f).verdict,
              QuditDeutschJozsa::Verdict::CONSTANT);
}
TEST_F(QuditDeutschJozsaTest, ConstantZero_d5_n1) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    EXPECT_EQ(QuditDeutschJozsa::solve(1, 5, f).verdict,
              QuditDeutschJozsa::Verdict::CONSTANT);
}
TEST_F(QuditDeutschJozsaTest, ConstantZero_d7_n2) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    EXPECT_EQ(QuditDeutschJozsa::solve(2, 7, f).verdict,
              QuditDeutschJozsa::Verdict::CONSTANT);
}

// Balanced functions: f(x) = x[0] is balanced for any n, d (each value in {0..d-1}
// appears d^{n-1} times since x[0] determines output, remaining n-1 qudits free)
TEST_F(QuditDeutschJozsaTest, Balanced_d2_n1) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    EXPECT_EQ(QuditDeutschJozsa::solve(1, 2, f).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}
TEST_F(QuditDeutschJozsaTest, Balanced_d2_n2) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    EXPECT_EQ(QuditDeutschJozsa::solve(2, 2, f).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}
TEST_F(QuditDeutschJozsaTest, Balanced_d2_n3) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    EXPECT_EQ(QuditDeutschJozsa::solve(3, 2, f).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}
TEST_F(QuditDeutschJozsaTest, Balanced_d3_n1) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    EXPECT_EQ(QuditDeutschJozsa::solve(1, 3, f).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}
TEST_F(QuditDeutschJozsaTest, Balanced_d3_n2) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    EXPECT_EQ(QuditDeutschJozsa::solve(2, 3, f).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}
TEST_F(QuditDeutschJozsaTest, Balanced_d5_n1) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    EXPECT_EQ(QuditDeutschJozsa::solve(1, 5, f).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}
TEST_F(QuditDeutschJozsaTest, Balanced_d5_n2) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    EXPECT_EQ(QuditDeutschJozsa::solve(2, 5, f).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}
TEST_F(QuditDeutschJozsaTest, Balanced_d7_n1) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    EXPECT_EQ(QuditDeutschJozsa::solve(1, 7, f).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}

// d=2 agrees with standard D-J behavior
TEST_F(QuditDeutschJozsaTest, d2_ConstantMatchesStandardDJ) {
    // Standard DJ constant → all-zero measurement → CONSTANT
    auto f = [](const std::vector<int>&) -> int { return 1; };
    EXPECT_EQ(QuditDeutschJozsa::solve(3, 2, f).verdict,
              QuditDeutschJozsa::Verdict::CONSTANT);
}

// Validation
TEST_F(QuditDeutschJozsaTest, ThrowsWhenD_LessThan2) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    EXPECT_THROW(QuditDeutschJozsa::solve(1, 1, f), std::invalid_argument);
}
TEST_F(QuditDeutschJozsaTest, ThrowsWhenN_LessThan1) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    EXPECT_THROW(QuditDeutschJozsa::solve(0, 2, f), std::invalid_argument);
}
TEST_F(QuditDeutschJozsaTest, ThrowsWhenF_ReturnsOutOfRange) {
    auto f = [](const std::vector<int>&) -> int { return 5; };  // out of [0,3)
    EXPECT_THROW(QuditDeutschJozsa::solve(1, 3, f), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditGrover
// ─────────────────────────────────────────────────────────────────────────────

class QuditGroverTest : public ::testing::Test {};

// d=2, n=1 (N=2): Grover provides no amplitude amplification for N=2 — both
// states remain at P=0.5 after 1 iteration regardless of target. Covered by
// AutoIterationsD2_n1_IsOne which verifies R=1 is chosen.

TEST_F(QuditGroverTest, FindsTarget_d2_n2_target01) {
    auto r = QuditGrover::search(2, 2, {0, 1}, -1, 300, 42);
    EXPECT_EQ(r.solution, (std::vector<int>{0, 1}));
    EXPECT_GT(r.probability, 0.5);
}
TEST_F(QuditGroverTest, FindsTarget_d2_n3_target101) {
    auto r = QuditGrover::search(3, 2, {1, 0, 1}, -1, 400, 7);
    EXPECT_EQ(r.solution, (std::vector<int>{1, 0, 1}));
    EXPECT_GT(r.probability, 0.5);
}
TEST_F(QuditGroverTest, FindsTarget_d3_n1_target0) {
    auto r = QuditGrover::search(1, 3, {0}, -1, 200, 42);
    EXPECT_EQ(r.solution, std::vector<int>({0}));
    EXPECT_GT(r.probability, 0.7);
}
TEST_F(QuditGroverTest, FindsTarget_d3_n1_target2) {
    auto r = QuditGrover::search(1, 3, {2}, -1, 200, 42);
    EXPECT_EQ(r.solution, std::vector<int>({2}));
    EXPECT_GT(r.probability, 0.7);
}
TEST_F(QuditGroverTest, FindsTarget_d3_n2_target12) {
    auto r = QuditGrover::search(2, 3, {1, 2}, -1, 400, 42);
    EXPECT_EQ(r.solution, (std::vector<int>{1, 2}));
    EXPECT_GT(r.probability, 0.5);
}
TEST_F(QuditGroverTest, FindsTarget_d4_n1_target3) {
    auto r = QuditGrover::search(1, 4, {3}, -1, 200, 42);
    EXPECT_EQ(r.solution, std::vector<int>({3}));
    EXPECT_GT(r.probability, 0.7);
}
TEST_F(QuditGroverTest, FindsTarget_d5_n1_target4) {
    auto r = QuditGrover::search(1, 5, {4}, -1, 200, 42);
    EXPECT_EQ(r.solution, std::vector<int>({4}));
    EXPECT_GT(r.probability, 0.7);
}
TEST_F(QuditGroverTest, SearchWithOracle_Predicate_d2_n2) {
    auto r = QuditGrover::search_with_oracle(2, 2,
        [](const std::vector<int>& x) { return x[0] == 1 && x[1] == 0; },
        -1, 300, 42);
    EXPECT_EQ(r.solution, (std::vector<int>{1, 0}));
    EXPECT_GT(r.probability, 0.5);
}
TEST_F(QuditGroverTest, SearchWithOracle_Predicate_d3_n1) {
    auto r = QuditGrover::search_with_oracle(1, 3,
        [](const std::vector<int>& x) { return x[0] == 1; },
        -1, 200, 42);
    EXPECT_EQ(r.solution, std::vector<int>({1}));
    EXPECT_GT(r.probability, 0.7);
}
TEST_F(QuditGroverTest, AutoIterationsD2_n1_IsOne) {
    // N=2: theta=pi/4, R=max(1, round(pi/(4*pi/4)-0.5))=max(1,round(0.5))=1
    auto r = QuditGrover::search(1, 2, {0}, -1, 100, 1);
    EXPECT_EQ(r.num_iterations, 1);
}
TEST_F(QuditGroverTest, ExplicitIterations) {
    auto r = QuditGrover::search(1, 3, {1}, 1, 100, 42);
    EXPECT_EQ(r.num_iterations, 1);
}
TEST_F(QuditGroverTest, ResultContainsCorrectMetadata) {
    auto r = QuditGrover::search(2, 3, {0, 1}, -1, 10, 42);
    EXPECT_EQ(r.d, 3);
    EXPECT_EQ(r.n, 2);
    EXPECT_GE(r.num_iterations, 1);
}

// Validation
TEST_F(QuditGroverTest, ThrowsWhenD_LessThan2) {
    EXPECT_THROW(QuditGrover::search(1, 1, {0}), std::invalid_argument);
}
TEST_F(QuditGroverTest, ThrowsWhenN_LessThan1) {
    EXPECT_THROW(QuditGrover::search(0, 2, {}), std::invalid_argument);
}
TEST_F(QuditGroverTest, ThrowsWhenTargetWrongSize) {
    EXPECT_THROW(QuditGrover::search(2, 2, {0}), std::invalid_argument);
}
TEST_F(QuditGroverTest, ThrowsWhenTargetDigitOutOfRange) {
    EXPECT_THROW(QuditGrover::search(1, 3, {3}), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditPhaseEstimation
// ─────────────────────────────────────────────────────────────────────────────

class QuditPhaseEstimationTest : public ::testing::Test {};

// Use X = shift_matrix(d,1) whose eigenstates have known phases k/d
TEST_F(QuditPhaseEstimationTest, ShiftEigenphase_d2_k0_m3) {
    // Eigenstate |+> of X (d=2), eigenvalue +1, phi=0
    const auto U = qudit_gates::shift_matrix(2, 1);
    const auto psi = shift_eigenstate(2, 0);
    auto r = QuditPhaseEstimation::estimate(3, 2, U, psi, 42);
    EXPECT_EQ(r.m, 3); EXPECT_EQ(r.d, 2);
    EXPECT_NEAR(r.phase_estimate, 0.0, 1.0 / 8.0 + 1e-9);
}
TEST_F(QuditPhaseEstimationTest, ShiftEigenphase_d2_k1_m3) {
    // Eigenstate |-> of X (d=2), eigenvalue -1, phi=0.5
    const auto U = qudit_gates::shift_matrix(2, 1);
    const auto psi = shift_eigenstate(2, 1);
    auto r = QuditPhaseEstimation::estimate(3, 2, U, psi, 42);
    EXPECT_NEAR(r.phase_estimate, 0.5, 1.0 / 8.0 + 1e-9);
}
TEST_F(QuditPhaseEstimationTest, ShiftEigenphase_d3_k0_m2) {
    const auto U = qudit_gates::shift_matrix(3, 1);
    const auto psi = shift_eigenstate(3, 0);
    auto r = QuditPhaseEstimation::estimate(2, 3, U, psi, 42);
    EXPECT_NEAR(r.phase_estimate, 0.0, 1.0 / 9.0 + 1e-9);
}
TEST_F(QuditPhaseEstimationTest, ShiftEigenphase_d3_k1_m2) {
    // phi = 1/3
    const auto U = qudit_gates::shift_matrix(3, 1);
    const auto psi = shift_eigenstate(3, 1);
    auto r = QuditPhaseEstimation::estimate(2, 3, U, psi, 42);
    EXPECT_NEAR(r.phase_estimate, 1.0 / 3.0, 1.0 / 9.0 + 1e-9);
}
TEST_F(QuditPhaseEstimationTest, ShiftEigenphase_d3_k2_m2) {
    // phi = 2/3
    const auto U = qudit_gates::shift_matrix(3, 1);
    const auto psi = shift_eigenstate(3, 2);
    auto r = QuditPhaseEstimation::estimate(2, 3, U, psi, 42);
    EXPECT_NEAR(r.phase_estimate, 2.0 / 3.0, 1.0 / 9.0 + 1e-9);
}
TEST_F(QuditPhaseEstimationTest, ShiftEigenphase_d5_k2_m2) {
    // phi = 2/5
    const auto U = qudit_gates::shift_matrix(5, 1);
    const auto psi = shift_eigenstate(5, 2);
    auto r = QuditPhaseEstimation::estimate(2, 5, U, psi, 42);
    EXPECT_NEAR(r.phase_estimate, 2.0 / 5.0, 1.0 / 25.0 + 1e-9);
}
TEST_F(QuditPhaseEstimationTest, ResultHasCorrectNumberOfPhaseDigits) {
    const auto U = qudit_gates::shift_matrix(3, 1);
    const auto psi = shift_eigenstate(3, 1);
    auto r = QuditPhaseEstimation::estimate(3, 3, U, psi, 42);
    EXPECT_EQ(static_cast<int>(r.phase_digits.size()), 3);
}
TEST_F(QuditPhaseEstimationTest, PhaseDigitsInRange) {
    const auto U = qudit_gates::shift_matrix(3, 1);
    const auto psi = shift_eigenstate(3, 1);
    auto r = QuditPhaseEstimation::estimate(2, 3, U, psi, 42);
    for (int digit : r.phase_digits) {
        EXPECT_GE(digit, 0);
        EXPECT_LT(digit, 3);
    }
}

// Validation
TEST_F(QuditPhaseEstimationTest, ThrowsWhenD_LessThan2) {
    const auto U = qudit_gates::shift_matrix(2, 1);
    const auto psi = shift_eigenstate(2, 0);
    EXPECT_THROW(QuditPhaseEstimation::estimate(3, 1, U, psi), std::invalid_argument);
}
TEST_F(QuditPhaseEstimationTest, ThrowsWhenM_LessThan1) {
    const auto U = qudit_gates::shift_matrix(2, 1);
    const auto psi = shift_eigenstate(2, 0);
    EXPECT_THROW(QuditPhaseEstimation::estimate(0, 2, U, psi), std::invalid_argument);
}
TEST_F(QuditPhaseEstimationTest, ThrowsWhenU_WrongSize) {
    std::vector<Complex128> bad_U(5, Complex128(0.0, 0.0));
    const auto psi = shift_eigenstate(2, 0);
    EXPECT_THROW(QuditPhaseEstimation::estimate(2, 2, bad_U, psi), std::invalid_argument);
}
TEST_F(QuditPhaseEstimationTest, ThrowsWhenEigenstate_WrongSize) {
    const auto U = qudit_gates::shift_matrix(2, 1);
    std::vector<Complex128> bad_psi(3, Complex128(0.0, 0.0));
    EXPECT_THROW(QuditPhaseEstimation::estimate(2, 2, U, bad_psi), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditSimon
// ─────────────────────────────────────────────────────────────────────────────

class QuditSimonTest : public ::testing::Test {};

TEST_F(QuditSimonTest, TrivialPeriod_d2_n1) {
    // f(x) = x → injective, period = 0
    auto r = QuditSimon::solve(1, 2,
        [](const std::vector<int>& x) -> std::vector<int> { return x; },
        3, 42);
    EXPECT_TRUE(r.is_trivial);
    EXPECT_EQ(r.d, 2); EXPECT_EQ(r.n, 1);
    EXPECT_TRUE(all_zero(r.period));
}
TEST_F(QuditSimonTest, TrivialPeriod_d3_n2) {
    auto r = QuditSimon::solve(2, 3,
        [](const std::vector<int>& x) -> std::vector<int> { return x; },
        3, 42);
    EXPECT_TRUE(r.is_trivial);
    EXPECT_TRUE(all_zero(r.period));
}
TEST_F(QuditSimonTest, Period_d2_n2_s10) {
    std::vector<int> s = {1, 0};
    auto r = QuditSimon::solve(2, 2, make_simon_f(s, 2, 2), 5, 42);
    EXPECT_EQ(r.period, s);
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, Period_d2_n2_s11) {
    std::vector<int> s = {1, 1};
    auto r = QuditSimon::solve(2, 2, make_simon_f(s, 2, 2), 5, 42);
    EXPECT_EQ(r.period, s);
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, Period_d2_n3_s101) {
    std::vector<int> s = {1, 0, 1};
    auto r = QuditSimon::solve(3, 2, make_simon_f(s, 3, 2), 5, 42);
    EXPECT_EQ(r.period, s);
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, Period_d3_n2_s10) {
    std::vector<int> s = {1, 0};
    auto r = QuditSimon::solve(2, 3, make_simon_f(s, 2, 3), 5, 42);
    EXPECT_EQ(r.period, s);
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, Period_d3_n2_s12) {
    std::vector<int> s = {1, 2};
    auto r = QuditSimon::solve(2, 3, make_simon_f(s, 2, 3), 5, 42);
    // Any non-zero scalar multiple of s in GF(3) is a valid period.
    EXPECT_TRUE(is_simon_period(r.period, s, 3));
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, Period_d3_n3_s102) {
    std::vector<int> s = {1, 0, 2};
    auto r = QuditSimon::solve(3, 3, make_simon_f(s, 3, 3), 5, 42);
    EXPECT_TRUE(is_simon_period(r.period, s, 3));
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, Period_d5_n2_s14) {
    std::vector<int> s = {1, 4};
    auto r = QuditSimon::solve(2, 5, make_simon_f(s, 2, 5), 5, 42);
    EXPECT_TRUE(is_simon_period(r.period, s, 5));
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, Period_d7_n2_s16) {
    std::vector<int> s = {1, 6};
    auto r = QuditSimon::solve(2, 7, make_simon_f(s, 2, 7), 5, 42);
    EXPECT_TRUE(is_simon_period(r.period, s, 7));
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, QuantumQueriesPositive) {
    std::vector<int> s = {1, 0};
    auto r = QuditSimon::solve(2, 2, make_simon_f(s, 2, 2), 3, 42);
    EXPECT_GT(r.quantum_queries, 0);
}
TEST_F(QuditSimonTest, ResultMetadata_d3_n2) {
    std::vector<int> s = {1, 2};
    auto r = QuditSimon::solve(2, 3, make_simon_f(s, 2, 3), 3, 42);
    EXPECT_EQ(r.d, 3);
    EXPECT_EQ(r.n, 2);
}

// Validation
TEST_F(QuditSimonTest, ThrowsWhenD_LessThan2) {
    auto f = [](const std::vector<int>& x) -> std::vector<int> { return x; };
    EXPECT_THROW(QuditSimon::solve(1, 1, f), std::invalid_argument);
}
// Composite d is now supported (R.1.11.0): the kernel is computed over the ring
// Z_d via integer Smith Normal Form, replacing the former prime-only restriction.
// Comprehensive composite-d coverage lands in the R.1.11.1 test suite.
TEST_F(QuditSimonTest, CompositeD_RecoversPeriod_d4_n2) {
    std::vector<int> s = {2, 0};
    auto r = QuditSimon::solve(2, 4, make_simon_f(s, 2, 4), 6, 42);
    EXPECT_TRUE(is_simon_period(r.period, s, 4));
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, CompositeD_RecoversPeriod_d6_n2) {
    std::vector<int> s = {1, 0};
    auto r = QuditSimon::solve(2, 6, make_simon_f(s, 2, 6), 6, 42);
    EXPECT_TRUE(is_simon_period(r.period, s, 6));
    EXPECT_FALSE(r.is_trivial);
}
TEST_F(QuditSimonTest, ThrowsWhenN_LessThan1) {
    auto f = [](const std::vector<int>& x) -> std::vector<int> { return x; };
    EXPECT_THROW(QuditSimon::solve(0, 2, f), std::invalid_argument);
}
TEST_F(QuditSimonTest, ThrowsWhenF_ReturnsWrongSize) {
    auto f = [](const std::vector<int>&) -> std::vector<int> { return {0, 0, 0}; };
    EXPECT_THROW(QuditSimon::solve(2, 2, f), std::invalid_argument);
}
