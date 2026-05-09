// IsingHamiltonian and SoftDispatchResult tests — entirely untested before R.1.3.1.
// Covers: from_qubo, from_hJ, evaluate, evaluate_spins, to_sparse_pauli_op,
//         SoftDispatchResult helpers.

#include <gtest/gtest.h>
#include "lindblad/ising.hpp"
#include "lindblad/dispatch.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace lindblad;

static constexpr double kTol = 1e-10;

// =============================================================================
// IsingHamiltonian — from_hJ
// =============================================================================

TEST(IsingHamiltonian, FromHJ_SingleQubit) {
    // H = h[0]·Z_0.  evaluate on |0⟩ (s=+1): energy = h[0].
    IsingHamiltonian ising = IsingHamiltonian::from_hJ({1.5}, {{}});
    // bitstring "0" → qubit 0 = 0 → spin = +1 → energy = h[0]*s = 1.5
    EXPECT_NEAR(ising.evaluate("0"), 1.5, kTol);
    // bitstring "1" → spin = -1 → energy = -1.5
    EXPECT_NEAR(ising.evaluate("1"), -1.5, kTol);
}

TEST(IsingHamiltonian, FromHJ_TwoQubits_ZZ) {
    // H = J[0][1]·Z_0·Z_1.
    std::vector<std::vector<double>> J = {{0.0, 2.0}, {0.0, 0.0}};
    IsingHamiltonian ising = IsingHamiltonian::from_hJ({0.0, 0.0}, J);
    // "00" → s0=+1, s1=+1 → J·(+1)(+1) = 2.0
    EXPECT_NEAR(ising.evaluate("00"), 2.0, kTol);
    // "01" → s0=+1, s1=-1 → J·(+1)(-1) = -2.0
    EXPECT_NEAR(ising.evaluate("01"), -2.0, kTol);
    // "10" → s0=-1, s1=+1 → -2.0
    EXPECT_NEAR(ising.evaluate("10"), -2.0, kTol);
    // "11" → s0=-1, s1=-1 → +2.0
    EXPECT_NEAR(ising.evaluate("11"), 2.0, kTol);
}

TEST(IsingHamiltonian, FromHJ_WithOffset) {
    IsingHamiltonian ising = IsingHamiltonian::from_hJ({1.0}, {{}}, 3.0);
    // energy = h*s + offset = 1.0*(+1) + 3.0 = 4.0
    EXPECT_NEAR(ising.evaluate("0"), 4.0, kTol);
}

// =============================================================================
// IsingHamiltonian — evaluate_spins
// =============================================================================

TEST(IsingHamiltonian, EvaluateSpins_SingleQubit) {
    IsingHamiltonian ising = IsingHamiltonian::from_hJ({2.0}, {{}});
    EXPECT_NEAR(ising.evaluate_spins({ 1}),  2.0, kTol);
    EXPECT_NEAR(ising.evaluate_spins({-1}), -2.0, kTol);
}

TEST(IsingHamiltonian, EvaluateSpins_TwoQubits_SameAsEvaluate) {
    std::vector<std::vector<double>> J = {{0.0, 1.5}, {0.0, 0.0}};
    IsingHamiltonian ising = IsingHamiltonian::from_hJ({0.5, -0.3}, J);

    // evaluate("01") == evaluate_spins({+1,-1})  (MSB first, "0"=qubit1, "1"=qubit0)
    // Wait: bit ordering is MSB first: bitstring[0] = qubit n-1.
    // "01": qubit1 = 0 → s1=+1, qubit0 = 1 → s0=-1.
    double e_bits = ising.evaluate("01");
    double e_spins = ising.evaluate_spins({-1, 1}); // s[0]=-1, s[1]=+1
    EXPECT_NEAR(e_bits, e_spins, kTol);
}

// =============================================================================
// IsingHamiltonian — from_qubo
// =============================================================================

TEST(IsingHamiltonian, FromQUBO_SingleVariable) {
    // Q = [[q]], minimize q*x^2 = q*x for x∈{0,1}.
    std::vector<std::vector<double>> Q = {{1.0}};
    IsingHamiltonian ising = IsingHamiltonian::from_qubo(Q);
    // x=0 (bitstring "0") → QUBO cost = 0; x=1 ("1") → cost = 1.
    double e0 = ising.evaluate("0");
    double e1 = ising.evaluate("1");
    EXPECT_LT(e0, e1);  // ground state at x=0
}

TEST(IsingHamiltonian, FromQUBO_MinimumAtAllZeros) {
    // Q = diagonal with positive entries → minimum at all zeros.
    std::vector<std::vector<double>> Q = {{2.0, 0.0}, {0.0, 3.0}};
    IsingHamiltonian ising = IsingHamiltonian::from_qubo(Q);
    double e00 = ising.evaluate("00");
    double e01 = ising.evaluate("01");
    double e10 = ising.evaluate("10");
    double e11 = ising.evaluate("11");
    EXPECT_LE(e00, e01);
    EXPECT_LE(e00, e10);
    EXPECT_LE(e00, e11);
}

TEST(IsingHamiltonian, FromQUBO_SymmetricOffDiagonal) {
    // Q = [[0,1],[0,0]]: minimize x0*x1; minimum at (0,0), (1,0), (0,1); max at (1,1).
    std::vector<std::vector<double>> Q = {{0.0, 1.0}, {0.0, 0.0}};
    IsingHamiltonian ising = IsingHamiltonian::from_qubo(Q);
    double e11 = ising.evaluate("11");
    double e00 = ising.evaluate("00");
    EXPECT_GT(e11, e00);
}

// =============================================================================
// IsingHamiltonian — to_sparse_pauli_op
// =============================================================================

TEST(IsingHamiltonian, ToSparsePauliOp_SingleQubit_EigenvalueMatchesEvaluate) {
    // H = 1.5·Z_0. For |0⟩: ⟨Z⟩=+1 → energy=1.5.
    IsingHamiltonian ising = IsingHamiltonian::from_hJ({1.5}, {{}});
    SparsePauliOp H = ising.to_sparse_pauli_op();

    Statevector sv0(1);  // |0⟩
    EXPECT_NEAR(H.expectation_value(sv0), 1.5 + ising.offset, kTol);

    Statevector sv1(1);
    gates::apply_x(sv1, 0);  // |1⟩
    EXPECT_NEAR(H.expectation_value(sv1), -1.5 + ising.offset, kTol);
}

TEST(IsingHamiltonian, ToSparsePauliOp_ZZ_MatchesEvaluate) {
    std::vector<std::vector<double>> J = {{0.0, 1.0}, {0.0, 0.0}};
    IsingHamiltonian ising = IsingHamiltonian::from_hJ({0.0, 0.0}, J);
    SparsePauliOp H = ising.to_sparse_pauli_op();

    // |00⟩: s0=+1, s1=+1 → ⟨ZZ⟩=+1 → energy = J*1 = 1.0
    Statevector sv00(2);
    EXPECT_NEAR(H.expectation_value(sv00), 1.0, kTol);

    // |01⟩: s0=-1, s1=+1 → ⟨ZZ⟩=-1 → energy = -1.0
    Statevector sv01(2);
    gates::apply_x(sv01, 0);
    EXPECT_NEAR(H.expectation_value(sv01), -1.0, kTol);
}

TEST(IsingHamiltonian, ToSparsePauliOp_NumQubits) {
    IsingHamiltonian ising = IsingHamiltonian::from_hJ({1,2,3}, {
        {0,0.5,0.3},{0,0,0.7},{0,0,0}
    });
    SparsePauliOp H = ising.to_sparse_pauli_op();
    EXPECT_EQ(H.n_qubits(), 3);
}

// =============================================================================
// SoftDispatchResult — bitstring dispatch helpers
// =============================================================================

TEST(SoftDispatchResult, ThresholdRound_AboveHalf_IsOne) {
    SoftDispatchResult dispatch({{"0", 30}, {"1", 70}});
    dispatch.compute();
    auto rounded = dispatch.threshold_round(0.5);
    EXPECT_EQ(rounded, "1");
}

TEST(SoftDispatchResult, TopK_ReturnsTopBitstrings) {
    SoftDispatchResult dispatch({{"000", 10}, {"110", 50}, {"101", 30}, {"011", 5}});
    dispatch.compute();
    auto top = dispatch.top_k(2);
    ASSERT_EQ(top.size(), 2u);
    EXPECT_EQ(top[0].first, "110");   // highest count
    EXPECT_EQ(top[1].first, "101");
}

TEST(SoftDispatchResult, ExpectedCost_WeightedAverage) {
    // 2-bit problem: cost("00")=0, cost("11")=4
    // Counts: "00"→50, "11"→50 → expected = 0*0.5 + 4*0.5 = 2.0
    SoftDispatchResult dispatch({{"00", 50}, {"11", 50}});
    dispatch.compute();

    auto cost_fn = [](const std::string& bits) -> double {
        int v = 0;
        for (char c : bits) v += (c == '1') ? 1 : 0;
        return static_cast<double>(v * v);
    };
    double ec = dispatch.expected_cost(cost_fn);
    EXPECT_NEAR(ec, 2.0, 1e-10);
}
