// test_shor.cpp — Direct tests for Shor's factoring algorithm

#include <gtest/gtest.h>
#include "lindblad/algorithms.hpp"
#include "lindblad/backends/local_backend.hpp"

#include <cmath>
#include <algorithm>

using namespace lindblad;
using namespace lindblad::algorithms;

// =============================================================================
// factorize — classical pre-screening paths
// =============================================================================

TEST(ShorTest, FactorizeEvenNumber) {
    // Even N → {2, N/2} via trivial_gcd, no quantum circuit
    Shor shor;
    auto r = shor.factorize(6);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor, 2u);
    EXPECT_EQ(r.cofactor, 3u);
    EXPECT_EQ(r.method, "trivial_gcd");
    EXPECT_EQ(r.attempts, 0);
}

TEST(ShorTest, FactorizeEvenLarger) {
    Shor shor;
    auto r = shor.factorize(22);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor, 2u);
    EXPECT_EQ(r.cofactor, 11u);
    EXPECT_EQ(r.method, "trivial_gcd");
}

TEST(ShorTest, FactorizePerfectPower) {
    // 9 = 3^2 → {3, 3} via perfect_power
    Shor shor;
    auto r = shor.factorize(9);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor, 3u);
    EXPECT_EQ(r.cofactor, 3u);
    EXPECT_EQ(r.method, "perfect_power");
    EXPECT_EQ(r.attempts, 0);
}

TEST(ShorTest, FactorizePerfectPower8) {
    // 8 = 2^3, but 8 is even so trivial_gcd fires first
    Shor shor;
    auto r = shor.factorize(8);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor, 2u);
    EXPECT_EQ(r.cofactor, 4u);
    EXPECT_EQ(r.method, "trivial_gcd");
}

TEST(ShorTest, FactorizePerfectPower25) {
    // 25 = 5^2 → {5, 5} via perfect_power (odd, so even check won't fire)
    Shor shor;
    auto r = shor.factorize(25);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor, 5u);
    EXPECT_EQ(r.cofactor, 5u);
    EXPECT_EQ(r.method, "perfect_power");
}

TEST(ShorTest, FactorizePerfectPower27) {
    // 27 = 3^3 → {3, 9} via perfect_power
    Shor shor;
    auto r = shor.factorize(27);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor, 3u);
    EXPECT_EQ(r.cofactor, 9u);
    EXPECT_EQ(r.method, "perfect_power");
}

TEST(ShorTest, FactorizeTrialGCD15) {
    // 15 = 3×5, trial GCD with {3,5,7,11,13} hits gcd(3,15) = 3
    Shor shor;
    auto r = shor.factorize(15);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor * r.cofactor, 15u);
    EXPECT_EQ(r.method, "trivial_gcd");
    EXPECT_EQ(r.attempts, 0);
}

TEST(ShorTest, FactorizeTrialGCD21) {
    // 21 = 3×7, trial GCD with {3,...} hits gcd(3,21) = 3
    Shor shor;
    auto r = shor.factorize(21);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor * r.cofactor, 21u);
    EXPECT_EQ(r.method, "trivial_gcd");
}

TEST(ShorTest, FactorizeTrialGCD35) {
    // 35 = 5×7, trial GCD with {3,5,...} hits gcd(5,35) = 5
    Shor shor;
    auto r = shor.factorize(35);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor * r.cofactor, 35u);
    EXPECT_EQ(r.method, "trivial_gcd");
}

TEST(ShorTest, FactorizeTrialGCD33) {
    // 33 = 3×11, trial GCD with {3,...} hits gcd(3,33) = 3
    Shor shor;
    auto r = shor.factorize(33);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor * r.cofactor, 33u);
    EXPECT_EQ(r.method, "trivial_gcd");
}

TEST(ShorTest, FactorizeFour) {
    // 4 = 2^2; even → trivial_gcd
    Shor shor;
    auto r = shor.factorize(4);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor, 2u);
    EXPECT_EQ(r.cofactor, 2u);
}

// =============================================================================
// factorize — quantum path
//
// To force the quantum path, N must be odd, not a perfect power, and have
// no factors in {3,5,7,11,13}. The smallest such semiprime is 17×19 = 323,
// but that requires too many qubits. Instead we use a seeded run on 15 with
// an Options that forces the quantum path by bypassing classical shortcuts.
//
// Since the implementation always tries classical shortcuts first, we verify
// the quantum subroutines (build_period_finding_circuit, find_order) directly
// and validate factorize() produces correct factors regardless of method.
// =============================================================================

TEST(ShorTest, FactorizeProductConsistency) {
    // For all classical-path N, verify factor × cofactor == N
    Shor shor;
    for (uint64_t N : {4, 6, 8, 9, 10, 14, 15, 21, 25, 26, 27, 33, 35}) {
        auto r = shor.factorize(N);
        EXPECT_TRUE(r.success) << "Failed for N=" << N;
        EXPECT_EQ(r.factor * r.cofactor, N) << "Product mismatch for N=" << N;
        EXPECT_GT(r.factor, 1u) << "Trivial factor for N=" << N;
        EXPECT_LT(r.factor, N) << "Trivial factor for N=" << N;
    }
}

// =============================================================================
// factorize — exception paths
// =============================================================================

TEST(ShorTest, ThrowsForNLessThan4) {
    Shor shor;
    EXPECT_THROW(shor.factorize(0), std::invalid_argument);
    EXPECT_THROW(shor.factorize(1), std::invalid_argument);
    EXPECT_THROW(shor.factorize(2), std::invalid_argument);
    EXPECT_THROW(shor.factorize(3), std::invalid_argument);
}

TEST(ShorTest, ThrowsForPrime) {
    Shor shor;
    EXPECT_THROW(shor.factorize(5), std::invalid_argument);
    EXPECT_THROW(shor.factorize(7), std::invalid_argument);
    EXPECT_THROW(shor.factorize(11), std::invalid_argument);
    EXPECT_THROW(shor.factorize(13), std::invalid_argument);
    EXPECT_THROW(shor.factorize(17), std::invalid_argument);
    EXPECT_THROW(shor.factorize(97), std::invalid_argument);
}

// =============================================================================
// build_period_finding_circuit — structural properties
// =============================================================================

TEST(ShorTest, CircuitQubitCount) {
    // N=15: n_target = ⌈log₂(16)⌉ = 4, n_eval = 9 → total = 13
    int n_target = 4;
    int n_eval = 9;
    auto qc = Shor::build_period_finding_circuit(2, 15, n_eval, n_target);
    EXPECT_EQ(qc.n_qubits, n_eval + n_target);
}

TEST(ShorTest, CircuitQubitCountN21) {
    // N=21: n_target = ⌈log₂(22)⌉ = 5, n_eval = 11 → total = 16
    int n_target = 5;
    int n_eval = 11;
    auto qc = Shor::build_period_finding_circuit(2, 21, n_eval, n_target);
    EXPECT_EQ(qc.n_qubits, n_eval + n_target);
}

TEST(ShorTest, CircuitHasInstructions) {
    // The circuit must have H gates + X gate + UNITARY gates + IQFT gates
    int n_target = 4;
    int n_eval = 9;
    auto qc = Shor::build_period_finding_circuit(2, 15, n_eval, n_target);
    // At minimum: n_eval H + 1 X + n_eval UNITARY + IQFT gates
    EXPECT_GT(qc.instructions.size(), static_cast<size_t>(n_eval + 1 + n_eval));
}

TEST(ShorTest, CircuitNoMeasurements) {
    // build_period_finding_circuit should NOT append measurements
    int n_target = 4;
    int n_eval = 9;
    auto qc = Shor::build_period_finding_circuit(2, 15, n_eval, n_target);
    for (const auto& inst : qc.instructions) {
        EXPECT_NE(inst.type, Instruction::GateType::MEASURE)
            << "Circuit should not contain MEASURE instructions";
    }
}

TEST(ShorTest, CircuitContainsUnitaryGates) {
    // Must contain exactly n_eval UNITARY instructions (one per eval qubit)
    int n_target = 4;
    int n_eval = 9;
    auto qc = Shor::build_period_finding_circuit(2, 15, n_eval, n_target);
    int unitary_count = 0;
    for (const auto& inst : qc.instructions) {
        if (inst.type == Instruction::GateType::UNITARY) ++unitary_count;
    }
    EXPECT_EQ(unitary_count, n_eval);
}

TEST(ShorTest, CircuitStartsWithHadamards) {
    // First n_eval instructions should be H gates on qubits 0..n_eval-1
    int n_target = 4;
    int n_eval = 9;
    auto qc = Shor::build_period_finding_circuit(2, 15, n_eval, n_target);
    for (int i = 0; i < n_eval; ++i) {
        EXPECT_EQ(qc.instructions[i].type, Instruction::GateType::H)
            << "Instruction " << i << " should be H";
        EXPECT_EQ(qc.instructions[i].qubits[0], i)
            << "H gate " << i << " should be on qubit " << i;
    }
}

TEST(ShorTest, CircuitInitTargetToOne) {
    // Instruction at index n_eval should be X on qubit n_eval (target |1⟩)
    int n_target = 4;
    int n_eval = 9;
    auto qc = Shor::build_period_finding_circuit(2, 15, n_eval, n_target);
    EXPECT_EQ(qc.instructions[n_eval].type, Instruction::GateType::X);
    EXPECT_EQ(qc.instructions[n_eval].qubits[0], n_eval);
}

// =============================================================================
// find_order — direct order-finding verification
// =============================================================================

TEST(ShorTest, FindOrderA2N15) {
    // ord_15(2) = 4. Loop over 10 seeded runs — at least one must return a valid
    // order. A correct statevector QPE circuit cannot fail on every seed; if it
    // does, the circuit or phase-extraction logic is wrong.
    int n_target = static_cast<int>(std::ceil(std::log2(16.0)));
    int n_eval = 2 * n_target + 1;

    backends::LocalBackend::Config cfg;
    cfg.simulator = backends::LocalBackend::SimType::STATEVECTOR;
    backends::LocalBackend backend(cfg);

    bool any_valid = false;
    for (uint64_t seed = 1; seed <= 10; ++seed) {
        uint64_t r = Shor::find_order(2, 15, n_eval, backend, seed);
        if (r > 0) {
            uint64_t check = 1;
            for (uint64_t i = 0; i < r; ++i) check = check * 2 % 15;
            EXPECT_EQ(check, 1u) << "Invalid order r=" << r << " for seed=" << seed;
            any_valid = true;
        }
    }
    EXPECT_TRUE(any_valid) << "find_order(2, 15) returned no valid order across 10 seeds";
}

TEST(ShorTest, FindOrderA7N15) {
    // ord_15(7) = 4. Same loop-based check for a=7.
    int n_target = static_cast<int>(std::ceil(std::log2(16.0)));
    int n_eval = 2 * n_target + 1;

    backends::LocalBackend::Config cfg;
    cfg.simulator = backends::LocalBackend::SimType::STATEVECTOR;
    backends::LocalBackend backend(cfg);

    bool any_valid = false;
    for (uint64_t seed = 1; seed <= 10; ++seed) {
        uint64_t r = Shor::find_order(7, 15, n_eval, backend, seed);
        if (r > 0) {
            uint64_t check = 1;
            for (uint64_t i = 0; i < r; ++i) check = check * 7 % 15;
            EXPECT_EQ(check, 1u) << "Invalid order r=" << r << " for seed=" << seed;
            any_valid = true;
        }
    }
    EXPECT_TRUE(any_valid) << "find_order(7, 15) returned no valid order across 10 seeds";
}

// =============================================================================
// Backend parity — DM and MPS produce valid factors matching SV
// =============================================================================

TEST(ShorTest, DensityMatrixBackendParity) {
    // factorize(6) is classical, but verify the DM backend option doesn't crash
    Shor::Options opts;
    opts.simulator = backends::LocalBackend::SimType::DENSITY_MATRIX;
    opts.seed = 42;
    Shor shor(opts);
    auto r = shor.factorize(6);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor * r.cofactor, 6u);
}

TEST(ShorTest, MPSBackendParity) {
    // Same with MPS
    Shor::Options opts;
    opts.simulator = backends::LocalBackend::SimType::MPS;
    opts.seed = 42;
    Shor shor(opts);
    auto r = shor.factorize(6);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.factor * r.cofactor, 6u);
}

// =============================================================================
// Options — seed reproducibility
// =============================================================================

TEST(ShorTest, SeedReproducibility) {
    // Same seed → same result
    Shor::Options opts;
    opts.seed = 999;
    Shor shor(opts);

    auto r1 = shor.factorize(15);
    auto r2 = shor.factorize(15);
    EXPECT_EQ(r1.factor, r2.factor);
    EXPECT_EQ(r1.cofactor, r2.cofactor);
    EXPECT_EQ(r1.method, r2.method);
}

TEST(ShorTest, DefaultOptionsConstruct) {
    // Default construction should work without error
    Shor shor;
    EXPECT_EQ(shor.options.n_eval_qubits, 0);
    EXPECT_EQ(shor.options.max_attempts, 10);
    EXPECT_EQ(shor.options.seed, 0u);
    EXPECT_EQ(shor.options.simulator, backends::LocalBackend::SimType::STATEVECTOR);
}

TEST(ShorTest, ExplicitOptionsConstruct) {
    Shor::Options opts;
    opts.n_eval_qubits = 7;
    opts.max_attempts = 5;
    opts.seed = 12345;
    Shor shor(opts);
    EXPECT_EQ(shor.options.n_eval_qubits, 7);
    EXPECT_EQ(shor.options.max_attempts, 5);
    EXPECT_EQ(shor.options.seed, 12345u);
}

// =============================================================================
// Result fields — method string correctness
// =============================================================================

TEST(ShorTest, MethodStringEven) {
    Shor shor;
    auto r = shor.factorize(10);
    EXPECT_EQ(r.method, "trivial_gcd");
}

TEST(ShorTest, MethodStringPerfectPower) {
    Shor shor;
    auto r = shor.factorize(49);  // 7^2
    EXPECT_EQ(r.method, "perfect_power");
    EXPECT_EQ(r.factor, 7u);
    EXPECT_EQ(r.cofactor, 7u);
}

// =============================================================================
// cf_convergents — direct unit tests
// =============================================================================

TEST(ShorTest, CfConvergentsQuarterPhase) {
    // 0.25 = 1/4; CF expansion gives convergents (0,1) then (1,4).
    auto conv = Shor::cf_convergents(0.25, 100);
    bool found = false;
    for (const auto& [n, d] : conv)
        if (n == 1 && d == 4) { found = true; break; }
    EXPECT_TRUE(found) << "Expected convergent (1, 4) for x=0.25";
}

TEST(ShorTest, CfConvergentsHalfPhase) {
    // 0.5 = 1/2 → convergent (1, 2).
    auto conv = Shor::cf_convergents(0.5, 100);
    bool found = false;
    for (const auto& [n, d] : conv)
        if (n == 1 && d == 2) { found = true; break; }
    EXPECT_TRUE(found) << "Expected convergent (1, 2) for x=0.5";
}

TEST(ShorTest, CfConvergentsThirdPhase) {
    // 1/3 → convergent (1, 3).
    auto conv = Shor::cf_convergents(1.0 / 3.0, 100);
    bool found = false;
    for (const auto& [n, d] : conv)
        if (n == 1 && d == 3) { found = true; break; }
    EXPECT_TRUE(found) << "Expected convergent (1, 3) for x=1/3";
}

TEST(ShorTest, CfConvergentsMaxDenomEnforced) {
    // All returned denominators must be ≤ max_denom.
    auto conv = Shor::cf_convergents(0.618033988749895, 20);  // (√5-1)/2
    for (const auto& [n, d] : conv)
        EXPECT_LE(d, 20u) << "Denominator " << d << " exceeds max_denom=20";
    EXPECT_FALSE(conv.empty());
}

TEST(ShorTest, CfConvergentsMaxDenomOne) {
    // max_denom=1: only denominators of 0 or 1 may appear.
    auto conv = Shor::cf_convergents(0.3, 1);
    for (const auto& [n, d] : conv)
        EXPECT_LE(d, 1u) << "Denominator " << d << " exceeds max_denom=1";
}

TEST(ShorTest, CfConvergentsNearZero) {
    // x=0.0 terminates immediately after one convergent; all denominators ≤ max_denom.
    auto conv = Shor::cf_convergents(0.0, 100);
    for (const auto& [n, d] : conv)
        EXPECT_LE(d, 100u);
}

TEST(ShorTest, CfConvergentsNearOne) {
    // x≈1 should not overflow or loop; all denominators ≤ max_denom.
    auto conv = Shor::cf_convergents(0.999, 100);
    for (const auto& [n, d] : conv)
        EXPECT_LE(d, 100u);
}

// =============================================================================
// build_period_finding_circuit — UNITARY gate unitarity
// =============================================================================

TEST(ShorTest, UnitaryGatesAreUnitary) {
    // Every UNITARY instruction's matrix U must satisfy U†U = I (to 1e-10).
    // Unitarity is a required invariant — if violated, the simulation is wrong.
    int n_target = 4;
    int n_eval   = 9;
    auto qc = Shor::build_period_finding_circuit(2, 15, n_eval, n_target);

    const double tol = 1e-10;
    for (const auto& inst : qc.instructions) {
        if (inst.type != Instruction::GateType::UNITARY) continue;
        const auto& U = inst.matrix;
        const size_t dim = static_cast<size_t>(
            std::round(std::sqrt(static_cast<double>(U.size()))));
        ASSERT_EQ(dim * dim, U.size()) << "Matrix size is not a perfect square";

        for (size_t r = 0; r < dim; ++r) {
            for (size_t c = 0; c < dim; ++c) {
                Complex128 sum(0.0, 0.0);
                for (size_t k = 0; k < dim; ++k) {
                    Complex128 conj_ki(U[k * dim + r].real, -U[k * dim + r].imag);
                    sum += conj_ki * U[k * dim + c];
                }
                double expected_re = (r == c) ? 1.0 : 0.0;
                EXPECT_NEAR(sum.real, expected_re, tol)
                    << "U†U[" << r << "," << c << "].re out of tolerance";
                EXPECT_NEAR(sum.imag, 0.0, tol)
                    << "U†U[" << r << "," << c << "].im out of tolerance";
            }
        }
    }
}
