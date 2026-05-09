// MPS simulator tests — entirely untested before R.1.3.1.
// Covers: basic gate application, Bell/GHZ state fidelity vs SV,
// to_statevector roundtrip, measure_sequential, RESET (C-2 MPS).

#include <gtest/gtest.h>
#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <cmath>
#include <random>
#include <string>

using namespace lindblad;

static constexpr double kTol  = 1e-8;
static constexpr double kFid  = 1e-6;  // fidelity tolerance for MPS (truncation may introduce error)

// Helper: state fidelity |⟨ψ|φ⟩|²
static double fidelity(const Statevector& a, const Statevector& b) {
    auto ip = a.inner_product(b);
    return ip.real * ip.real + ip.imag * ip.imag;
}

// =============================================================================
// Basic run — MPS simulator produces a valid result
// =============================================================================

TEST(MPSSim, BasicRun_H_Gate) {
    QuantumCircuit qc(1);
    qc.h(0);
    qc.measure_all();

    MPSSimulator sim;
    auto res = sim.run(qc, /*bond=*/16, /*shots=*/1000, /*seed=*/42);

    // |+⟩ should give ≈ 50/50
    EXPECT_GT(res.counts["0"], 400);
    EXPECT_GT(res.counts["1"], 400);
}

TEST(MPSSim, BasicRun_X_Gate) {
    QuantumCircuit qc(1);
    qc.x(0);
    qc.measure_all();

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 500, 42);
    EXPECT_EQ(res.counts.size(), 1u);
    EXPECT_EQ(res.counts.count("1"), 1u);
}

// =============================================================================
// Bell state — fidelity vs exact SV
// =============================================================================

TEST(MPSSim, BellState_Fidelity_VsSV) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);

    // Exact SV result
    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(qc);
    ASSERT_TRUE(sv_res.success);

    // MPS result
    MPSSimulator mps_sim;
    auto mps_res = mps_sim.run(qc, 16, 0, 42);  // 0 shots = no sampling

    Statevector mps_sv = mps_res.final_state.to_statevector();
    double F = fidelity(sv_res.final_state, mps_sv);
    EXPECT_NEAR(F, 1.0, kFid);
}

TEST(MPSSim, BellState_Counts_BothBitstringsSeen) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    qc.measure_all();

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 1000, 42);

    // Only |00⟩ and |11⟩ should appear
    for (const auto& [bits, count] : res.counts) {
        EXPECT_TRUE(bits == "00" || bits == "11")
            << "Unexpected bitstring: " << bits;
    }
    EXPECT_GT(res.counts["00"], 400);
    EXPECT_GT(res.counts["11"], 400);
}

// =============================================================================
// GHZ state — 3 qubits
// =============================================================================

TEST(MPSSim, GHZState_Fidelity_VsSV) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(0, 2);

    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(qc);

    MPSSimulator mps_sim;
    auto mps_res = mps_sim.run(qc, 16, 0, 42);

    Statevector mps_sv = mps_res.final_state.to_statevector();
    double F = fidelity(sv_res.final_state, mps_sv);
    EXPECT_NEAR(F, 1.0, kFid);
}

TEST(MPSSim, GHZState_Counts_OnlyCorners) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(0, 2);
    qc.measure_all();

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 1000, 42);
    for (const auto& [bits, count] : res.counts)
        EXPECT_TRUE(bits == "000" || bits == "111")
            << "Unexpected bitstring: " << bits;
}

// =============================================================================
// to_statevector roundtrip
// =============================================================================

TEST(MPSSim, ToStatevector_ProductState) {
    // |+⟩⊗|1⟩ on 2 qubits: exact, no truncation.
    QuantumCircuit qc(2);
    qc.h(0).x(1);

    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(qc);

    MPSSimulator mps_sim;
    auto mps_res = mps_sim.run(qc, 16, 0, 42);

    Statevector mps_sv = mps_res.final_state.to_statevector();
    double F = fidelity(sv_res.final_state, mps_sv);
    EXPECT_NEAR(F, 1.0, kFid);
}

TEST(MPSSim, ToStatevector_4Qubit_Circuit) {
    // Random-ish 4-qubit circuit with moderate entanglement.
    QuantumCircuit qc(4);
    qc.h(0).h(2).cx(0, 1).cx(2, 3).cx(1, 2);

    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(qc);

    MPSSimulator mps_sim;
    auto mps_res = mps_sim.run(qc, 64, 0, 42);

    Statevector mps_sv = mps_res.final_state.to_statevector();
    double F = fidelity(sv_res.final_state, mps_sv);
    EXPECT_NEAR(F, 1.0, 1e-5);
}

// =============================================================================
// measure_sequential — samples a correct bitstring distribution
// =============================================================================

TEST(MPSSim, MeasureSequential_BasisState) {
    // |11⟩ should always sample "11".
    QuantumCircuit qc(2);
    qc.x(0).x(1);

    MPSSimulator mps_sim;
    auto res = mps_sim.run(qc, 16, 200, 42);
    qc.measure_all();
    // Use measure_sequential directly on the final state
    std::mt19937_64 rng(42);
    std::string s = res.final_state.measure_sequential(rng);
    EXPECT_EQ(s, "11");
}

TEST(MPSSim, MeasureSequential_Bell_BothOutcomes) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);

    MPSSimulator mps_sim;
    auto res = mps_sim.run(qc, 16, 0, 42);

    std::mt19937_64 rng(123);
    int cnt0 = 0, cnt1 = 0;
    for (int i = 0; i < 200; ++i) {
        // Each call modifies state, so re-run to get a fresh state
        MPSSimulator sim2;
        auto r2 = sim2.run(qc, 16, 0, static_cast<uint64_t>(i + 1));
        std::mt19937_64 rng2(static_cast<uint64_t>(i));
        std::string s = r2.final_state.measure_sequential(rng2);
        EXPECT_TRUE(s == "00" || s == "11") << "Unexpected: " << s;
        if (s == "00") cnt0++;
        else            cnt1++;
    }
    EXPECT_GT(cnt0, 60);
    EXPECT_GT(cnt1, 60);
}

// =============================================================================
// C-2 (MPS) — MEASURE and RESET in MPSSimulator
// =============================================================================

TEST(MPSSim, Reset_ExcitedState_CollapseToZero) {
    // X|0⟩ = |1⟩, RESET → |0⟩, measure → always "0".
    QuantumCircuit qc(1, 1);
    qc.x(0).reset(0).measure(0, 0);

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 200, 42);
    EXPECT_EQ(res.counts.size(), 1u);
    EXPECT_EQ(res.counts.count("0"), 1u);
}

TEST(MPSSim, Reset_Superposition_CollapseToZero) {
    // H|0⟩ = |+⟩, RESET → |0⟩.
    QuantumCircuit qc(1, 1);
    qc.h(0).reset(0).measure(0, 0);

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 200, 42);
    EXPECT_EQ(res.counts.size(), 1u);
    EXPECT_EQ(res.counts.count("0"), 1u);
}

TEST(MPSSim, Measure_Deterministic_KnownState) {
    // X|0⟩ = |1⟩; all shots must give "1".
    QuantumCircuit qc(1, 1);
    qc.x(0).measure(0, 0);

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 200, 42);
    EXPECT_EQ(res.counts.size(), 1u);
    EXPECT_EQ(res.counts.count("1"), 1u);
}

// =============================================================================
// probabilities_single — marginal probabilities
// =============================================================================

TEST(MPSSim, ProbabilitiesSingle_HalfHalf) {
    // H|0⟩: P(0)=P(1)=0.5.
    QuantumCircuit qc(1);
    qc.h(0);

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 0, 42);

    auto probs = res.final_state.probabilities_single(0);
    ASSERT_EQ(probs.size(), 2u);
    EXPECT_NEAR(probs[0], 0.5, 1e-10);
    EXPECT_NEAR(probs[1], 0.5, 1e-10);
}

TEST(MPSSim, ProbabilitiesSingle_BasisState_X1) {
    // X|0⟩ = |1⟩: P(0)=0, P(1)=1.
    QuantumCircuit qc(1);
    qc.x(0);

    MPSSimulator sim;
    auto res = sim.run(qc, 16, 0, 42);

    auto probs = res.final_state.probabilities_single(0);
    EXPECT_NEAR(probs[0], 0.0, 1e-10);
    EXPECT_NEAR(probs[1], 1.0, 1e-10);
}

// =============================================================================
// Bond dimension — truncation error is bounded
// =============================================================================

TEST(MPSSim, TruncationError_ProductState_IsZero) {
    // Product state requires bond dim 1; no truncation.
    QuantumCircuit qc(4);
    qc.h(0).h(1).h(2).h(3);

    MPSSimulator sim;
    auto res = sim.run(qc, 4, 0, 42);  // bond=4 is more than enough
    EXPECT_NEAR(res.final_state.truncation_error(), 0.0, 1e-12);
}
