#include <gtest/gtest.h>
#include "qpp/algorithms.hpp"
#include "qpp/circuit.hpp"

using namespace qpp;
using namespace qpp::algorithms;

// =============================================================================
// Deutsch-Jozsa
// =============================================================================

static QuantumCircuit dj_constant_oracle(int n) {
    return QuantumCircuit(n + 1); // f(x) = 0: no gates
}

static QuantumCircuit dj_balanced_oracle(int n) {
    QuantumCircuit qc(n + 1);
    qc.cx(0, n); // f(x) = x[0]: CNOT from first query qubit to ancilla
    return qc;
}

TEST(DeutschJozsa, ConstantOracleDetected) {
    auto oracle = dj_constant_oracle(3);
    auto result = DeutschJozsa::solve(oracle, 3);
    EXPECT_EQ(result.type, DeutschJozsa::Result::CONSTANT);
}

TEST(DeutschJozsa, BalancedOracleDetected) {
    auto oracle = dj_balanced_oracle(3);
    auto result = DeutschJozsa::solve(oracle, 3);
    EXPECT_EQ(result.type, DeutschJozsa::Result::BALANCED);
}

TEST(DeutschJozsa, ConstantOracle1Qubit) {
    auto oracle = dj_constant_oracle(1);
    auto result = DeutschJozsa::solve(oracle, 1);
    EXPECT_EQ(result.type, DeutschJozsa::Result::CONSTANT);
}

TEST(DeutschJozsa, BalancedOracle1Qubit) {
    auto oracle = dj_balanced_oracle(1);
    auto result = DeutschJozsa::solve(oracle, 1);
    EXPECT_EQ(result.type, DeutschJozsa::Result::BALANCED);
}

// =============================================================================
// Bernstein-Vazirani
// =============================================================================

static QuantumCircuit bv_oracle(const std::string& s) {
    int n = s.size();
    QuantumCircuit qc(n + 1);
    for (int i = 0; i < n; ++i)
        if (s[i] == '1') qc.cx(i, n);
    return qc;
}

TEST(BernsteinVazirani, RecoversSecret_101) {
    std::string secret = "101";
    auto result = BernsteinVazirani::solve(bv_oracle(secret), 3);
    EXPECT_EQ(result.secret, secret);
}

TEST(BernsteinVazirani, RecoversSecret_1100) {
    std::string secret = "1100";
    auto result = BernsteinVazirani::solve(bv_oracle(secret), 4);
    EXPECT_EQ(result.secret, secret);
}

TEST(BernsteinVazirani, RecoversSecret_AllZeros) {
    std::string secret = "000";
    auto result = BernsteinVazirani::solve(bv_oracle(secret), 3);
    EXPECT_EQ(result.secret, secret);
}

TEST(BernsteinVazirani, RecoversSecret_AllOnes) {
    std::string secret = "111";
    auto result = BernsteinVazirani::solve(bv_oracle(secret), 3);
    EXPECT_EQ(result.secret, secret);
}

// =============================================================================
// Simon's Algorithm
// =============================================================================

// Standard textbook Simon oracle for period s.
// Copies query to output register, then XORs output with s controlled on
// the first set bit of s to implement the 2-to-1 mapping.
static QuantumCircuit simon_oracle(const std::string& s) {
    int n = s.size();
    QuantumCircuit qc(2 * n);
    for (int i = 0; i < n; ++i) qc.cx(i, n + i);
    int k = -1;
    for (int i = 0; i < n; ++i) if (s[i] == '1') { k = i; break; }
    if (k >= 0)
        for (int i = 0; i < n; ++i)
            if (s[i] == '1') qc.cx(k, n + i);
    return qc;
}

TEST(SimonsAlgorithm, FindsPeriod_110) {
    std::string period = "110";
    auto result = Simon::solve(simon_oracle(period), 3, 42);
    EXPECT_EQ(result.period, period);
}

TEST(SimonsAlgorithm, FindsPeriod_1010) {
    std::string period = "1010";
    auto result = Simon::solve(simon_oracle(period), 4, 42);
    EXPECT_EQ(result.period, period);
}

TEST(SimonsAlgorithm, FindsPeriod_101) {
    std::string period = "101";
    auto result = Simon::solve(simon_oracle(period), 3, 7);
    EXPECT_EQ(result.period, period);
}

// =============================================================================
// RecursiveBernsteinVazirani
// =============================================================================

TEST(RecursiveBernsteinVazirani, Depth1MatchesStandardBV) {
    // Depth-1 recursive BV is identical to a single standard BV call.
    std::string s = "1011";
    auto result = RecursiveBernsteinVazirani::solve({bv_oracle(s)}, 4);
    ASSERT_EQ(result.depth, 1);
    ASSERT_EQ(result.total_oracle_calls, 1);
    ASSERT_EQ(result.secrets.size(), 1u);
    EXPECT_EQ(result.secrets[0], s);
}

TEST(RecursiveBernsteinVazirani, Depth2DistinctSecrets) {
    // Two independent levels, each with a different 3-bit secret.
    std::string s0 = "101", s1 = "010";
    auto result = RecursiveBernsteinVazirani::solve(
        {bv_oracle(s0), bv_oracle(s1)}, 3);
    ASSERT_EQ(result.depth, 2);
    ASSERT_EQ(result.total_oracle_calls, 2);
    ASSERT_EQ(result.secrets.size(), 2u);
    EXPECT_EQ(result.secrets[0], s0);
    EXPECT_EQ(result.secrets[1], s1);
}

TEST(RecursiveBernsteinVazirani, Depth3AllDistinct) {
    // Three levels — validates that seed offsetting keeps each run independent.
    std::string s0 = "110", s1 = "001", s2 = "111";
    auto result = RecursiveBernsteinVazirani::solve(
        {bv_oracle(s0), bv_oracle(s1), bv_oracle(s2)}, 3, 1, 42);
    ASSERT_EQ(result.depth, 3);
    ASSERT_EQ(result.total_oracle_calls, 3);
    EXPECT_EQ(result.secrets[0], s0);
    EXPECT_EQ(result.secrets[1], s1);
    EXPECT_EQ(result.secrets[2], s2);
}

TEST(RecursiveBernsteinVazirani, Depth2AllZerosSecret) {
    // Degenerate secret — zero string must also be recovered correctly at both levels.
    std::string s0 = "000", s1 = "000";
    auto result = RecursiveBernsteinVazirani::solve(
        {bv_oracle(s0), bv_oracle(s1)}, 3);
    EXPECT_EQ(result.secrets[0], s0);
    EXPECT_EQ(result.secrets[1], s1);
}

// =============================================================================
// ProbabilisticBernsteinVazirani  (Shukla & Vedula 2023, arXiv:2301.10014)
// =============================================================================

TEST(ProbabilisticBernsteinVazirani, SingleKeyPoolAlwaysReturnsKey) {
    // A pool of one oracle is a degenerate probabilistic oracle — always returns
    // the same secret. Every shot must recover it.
    std::string s = "1100";
    auto result = ProbabilisticBernsteinVazirani::solve({bv_oracle(s)}, 4, {}, 20, 0);
    ASSERT_EQ(result.shots_used, 20);
    ASSERT_EQ(result.discovered_keys.size(), 1u);
    EXPECT_EQ(result.discovered_keys[0], s);
    EXPECT_EQ(result.key_counts.at(s), 20);
}

TEST(ProbabilisticBernsteinVazirani, TwoKeyPoolBothDiscovered) {
    // Uniform 2-key pool: coupon-collector expectation is 2·H_2 = 3 shots;
    // 40 shots gives overwhelming probability of seeing both.
    std::string s0 = "101", s1 = "010";
    auto result = ProbabilisticBernsteinVazirani::solve(
        {bv_oracle(s0), bv_oracle(s1)}, 3, {}, 40, 7);
    EXPECT_EQ(result.discovered_keys.size(), 2u);
    EXPECT_GT(result.key_counts.count(s0), 0u) << "key s0 never seen";
    EXPECT_GT(result.key_counts.count(s1), 0u) << "key s1 never seen";
}

TEST(ProbabilisticBernsteinVazirani, ThreeKeyPoolAllDiscovered) {
    // Uniform 3-key pool: coupon-collector expectation ≈ 3·ln(3) ≈ 3.3 shots;
    // 60 shots is more than sufficient to find all three.
    std::string s0 = "100", s1 = "010", s2 = "001";
    auto result = ProbabilisticBernsteinVazirani::solve(
        {bv_oracle(s0), bv_oracle(s1), bv_oracle(s2)}, 3, {}, 60, 42);
    EXPECT_EQ(result.discovered_keys.size(), 3u);
    EXPECT_GT(result.key_counts.count(s0), 0u) << "key s0 never seen";
    EXPECT_GT(result.key_counts.count(s1), 0u) << "key s1 never seen";
    EXPECT_GT(result.key_counts.count(s2), 0u) << "key s2 never seen";
}

TEST(ProbabilisticBernsteinVazirani, SkewedWeightsCountsAreProportional) {
    // Oracle 0 drawn with weight 9, oracle 1 with weight 1.
    // After 200 shots, key 0 should appear ~10× more often than key 1.
    std::string s0 = "110", s1 = "001";
    auto result = ProbabilisticBernsteinVazirani::solve(
        {bv_oracle(s0), bv_oracle(s1)}, 3, {9.0, 1.0}, 200, 99);
    ASSERT_GT(result.key_counts.count(s0), 0u);
    ASSERT_GT(result.key_counts.count(s1), 0u);
    // With 9:1 weighting, s0 should appear at least 5× more often than s1.
    EXPECT_GT(result.key_counts.at(s0), result.key_counts.at(s1) * 5);
}

TEST(ProbabilisticBernsteinVazirani, ShotsUsedFieldCorrect) {
    std::string s = "11";
    auto result = ProbabilisticBernsteinVazirani::solve({bv_oracle(s)}, 2, {}, 17, 0);
    EXPECT_EQ(result.shots_used, 17);
}

TEST(ProbabilisticBernsteinVazirani, DiscoveredKeysSorted) {
    // discovered_keys must be lexicographically sorted regardless of discovery order.
    std::string s0 = "001", s1 = "110";
    auto result = ProbabilisticBernsteinVazirani::solve(
        {bv_oracle(s1), bv_oracle(s0)}, 3, {}, 60, 5);  // pool in reverse order
    ASSERT_EQ(result.discovered_keys.size(), 2u);
    EXPECT_LT(result.discovered_keys[0], result.discovered_keys[1]);
}
