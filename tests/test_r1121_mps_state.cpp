// R.1.12.1 total-coverage suite, Batch 2: lindblad/simulators/mps_sim.hpp
// (MPSState, MPSSimulator). Plan: docs (R.1.12.1 coverage plan), "Batch 2".
//
// Single-qubit action and the to_statevector index convention are checked on
// MPSState directly; full two-qubit correctness is checked by comparing the
// MPS simulator's final statevector probabilities against the exact
// statevector simulator. Bond-dimension growth, sequential-measurement bit
// order, and the conversion bound are pinned. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;

namespace {
constexpr double kTol = 1e-6;

std::array<Complex128, 4> X_gate() {
    return {Complex128(0, 0), Complex128(1, 0), Complex128(1, 0), Complex128(0, 0)};
}

std::array<Complex128, 4> RY_gate(double theta) {
    double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
    return {Complex128(c, 0), Complex128(-s, 0), Complex128(s, 0), Complex128(c, 0)};
}

}  // namespace

// =============================================================================
// Single-qubit action and to_statevector LSB index
// =============================================================================

TEST(R1121Mps, SingleQubitXTargetsQubitZeroAsLsb) {
    MPSState mps(2);
    mps.apply_single_qubit_gate(X_gate(), 0);  // |00> -> q0=1 -> index 1
    auto sv = mps.to_statevector();
    EXPECT_NEAR(sv.probability(1), 1.0, kTol);
    EXPECT_NEAR(sv.probability(0), 0.0, kTol);
}

TEST(R1121Mps, ProbabilitiesSingleReflectsQubitState) {
    MPSState mps(2);
    mps.apply_single_qubit_gate(X_gate(), 0);  // q0 = 1, q1 = 0
    auto p0 = mps.probabilities_single(0);
    auto p1 = mps.probabilities_single(1);
    ASSERT_EQ(p0.size(), 2u);
    EXPECT_NEAR(p0[0], 0.0, kTol);
    EXPECT_NEAR(p0[1], 1.0, kTol);
    EXPECT_NEAR(p1[0], 1.0, kTol);
    EXPECT_NEAR(p1[1], 0.0, kTol);
}

TEST(R1121Mps, ToStatevectorThrowsAboveQubitLimit) {
    MPSState big(26);  // > MPS_SV_MAX_QUBITS (25)
    EXPECT_THROW(big.to_statevector(), std::runtime_error);
}

// =============================================================================
// Bond dimension
// =============================================================================

TEST(R1121Mps, ProductStateHasUnitBondDimension) {
    MPSState mps(3);
    mps.apply_single_qubit_gate(X_gate(), 1);  // still a product state
    EXPECT_EQ(mps.current_max_bond_dim(), 1);
}

TEST(R1121Mps, EntanglingCircuitGrowsBondDimension) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);  // Bell -> bond dimension 2
    MPSSimulator sim;
    auto res = sim.run(qc, 64, 0, 1);
    EXPECT_GE(res.final_state.current_max_bond_dim(), 2);
}

// =============================================================================
// MPS simulator vs statevector — probability agreement
// =============================================================================

TEST(R1121Mps, SimulatorProbabilitiesMatchStatevector) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).ry(0.7, 2).cx(1, 2).rz(0.4, 0).x(1);

    MPSSimulator msim;
    auto mres = msim.run(qc, 64, 0, 1);
    auto mps_sv = mres.final_state.to_statevector();

    StatevectorSimulator ssim;
    auto sres = ssim.run(qc, 0, 1);

    ASSERT_EQ(mps_sv.dim, sres.final_state.dim);
    for (size_t i = 0; i < mps_sv.dim; ++i) {
        EXPECT_NEAR(mps_sv.probability(i), sres.final_state.probability(i), kTol)
            << "probability mismatch at index " << i;
    }
}

TEST(R1121Mps, MaxBondDimensionIsEnforcedAndTruncates) {
    // A circuit whose exact Schmidt rank across the central cut exceeds 2.
    QuantumCircuit qc(4);
    qc.h(0).h(1).cx(0, 2).cx(1, 3).cx(0, 3).ry(0.6, 1).cx(1, 2);

    // Capped at chi = 2: the reported bond dimension never exceeds the cap, and
    // truncation error is recorded.
    MPSSimulator capped;
    auto cres = capped.run(qc, 2, 0, 1);
    EXPECT_LE(cres.final_state.current_max_bond_dim(), 2)
        << "max_bond_dim must be a hard cap";
    EXPECT_GE(cres.final_state.truncation_error(), 0.0);

    // Uncapped (large chi): no truncation, exact agreement with the statevector.
    MPSSimulator exact;
    auto eres = exact.run(qc, 64, 0, 1);
    auto mps_sv = eres.final_state.to_statevector();
    StatevectorSimulator ssim;
    auto sres = ssim.run(qc, 0, 1);
    for (size_t i = 0; i < mps_sv.dim; ++i)
        EXPECT_NEAR(mps_sv.probability(i), sres.final_state.probability(i), kTol)
            << "uncapped MPS must be exact at index " << i;
}

TEST(R1121Mps, NonAdjacentAndReversedTwoQubitGatesMatchStatevector) {
    // Non-adjacent control/target (qubits 0 and 2) and reversed operand order
    // both route through the internal swap network; deterministic outcomes pin
    // the routing exactly.
    {
        QuantumCircuit qc(3);
        qc.x(0).cx(0, 2);  // q0=1 controls q2 -> index q0=1,q2=1 = 5
        MPSSimulator sim;
        auto sv = sim.run(qc, 64, 0, 1).final_state.to_statevector();
        EXPECT_NEAR(sv.probability(5), 1.0, kTol);
    }
    {
        QuantumCircuit qc(3);
        qc.x(2).cx(2, 0);  // reversed: q2=1 controls q0 -> index 5
        MPSSimulator sim;
        auto sv = sim.run(qc, 64, 0, 1).final_state.to_statevector();
        EXPECT_NEAR(sv.probability(5), 1.0, kTol);
    }
    // A randomized circuit with non-adjacent entanglers: full probability vector
    // matches the statevector simulator.
    {
        QuantumCircuit qc(4);
        qc.h(0).cx(0, 3).ry(0.5, 1).cx(1, 3).rz(0.9, 2).cx(2, 0).x(3);
        MPSSimulator msim;
        auto mps_sv = msim.run(qc, 64, 0, 7).final_state.to_statevector();
        StatevectorSimulator ssim;
        auto sres = ssim.run(qc, 0, 7);
        for (size_t i = 0; i < mps_sv.dim; ++i)
            EXPECT_NEAR(mps_sv.probability(i), sres.final_state.probability(i), kTol)
                << "index " << i;
    }
}

TEST(R1121Mps, BondDimensionGrowsWithEntanglementDepth) {
    // GHZ across a line keeps bond dimension 2; layered random entanglers push
    // it higher (more Schmidt rank across the central cut).
    QuantumCircuit ghz(4);
    ghz.h(0).cx(0, 1).cx(1, 2).cx(2, 3);
    MPSSimulator sim;
    EXPECT_EQ(sim.run(ghz, 64, 0, 1).final_state.current_max_bond_dim(), 2)
        << "GHZ is a bond-dimension-2 state";

    // Two Bell pairs that BOTH cross the central cut {0,1}|{2,3}: Bell(0,2) and
    // Bell(1,3). The reduced state on {0,1} is (I/2)(x)(I/2) -> Schmidt rank 4,
    // so the central bond dimension is 4 > 2. (cx on a |+> target would be a
    // no-op; here each cx targets a |0>, so it genuinely entangles.)
    QuantumCircuit deep(4);
    deep.h(0).cx(0, 2).h(1).cx(1, 3);
    EXPECT_GT(sim.run(deep, 64, 0, 1).final_state.current_max_bond_dim(), 2)
        << "entanglers crossing the central cut raise the bond dimension above 2";
}

// =============================================================================
// Sequential measurement
// =============================================================================

TEST(R1121Mps, MeasureSequentialReproducesProductMarginals) {
    // Independent product state: q0 with P(1) = 0.25, q1 with P(1) = 0.7.
    // measure_sequential over many fresh copies must reproduce the independent
    // joint distribution. Outcome key is c1 c0 (q0 rightmost).
    const double th0 = 2.0 * std::asin(std::sqrt(0.25));
    const double th1 = 2.0 * std::asin(std::sqrt(0.70));
    std::mt19937_64 rng(4242);
    std::unordered_map<std::string, int> counts;
    const int N = 8000;
    for (int i = 0; i < N; ++i) {
        MPSState mps(2);
        mps.apply_single_qubit_gate(RY_gate(th0), 0);
        mps.apply_single_qubit_gate(RY_gate(th1), 1);
        counts[mps.measure_sequential(rng)]++;
    }
    auto freq = [&](const std::string& k) {
        return counts.count(k) ? double(counts[k]) / N : 0.0;
    };
    // p(q1,q0): "00"=0.75*0.75? no -> P(q0=1)=0.25, P(q1=1)=0.7.
    EXPECT_NEAR(freq("00"), 0.75 * 0.30, 0.03);  // q1=0,q0=0
    EXPECT_NEAR(freq("01"), 0.25 * 0.30, 0.03);  // q1=0,q0=1
    EXPECT_NEAR(freq("10"), 0.75 * 0.70, 0.03);  // q1=1,q0=0
    EXPECT_NEAR(freq("11"), 0.25 * 0.70, 0.03);  // q1=1,q0=1
}

TEST(R1121Mps, MeasuredNonUniformCircuitMatchesStatevectorDistribution) {
    QuantumCircuit qc(3, 3);
    qc.ry(1.1, 0).cx(0, 1).rz(0.4, 1).ry(0.8, 2).cx(1, 2).measure_all();
    MPSSimulator msim;
    StatevectorSimulator ssim;
    auto mc = msim.run(qc, 64, 8000, 13).counts;
    auto sc = ssim.run(qc, 8000, 13).counts;
    // Total-variation distance between the two empirical distributions.
    int tm = 0, ts = 0;
    for (auto& [k, n] : mc) tm += n;
    for (auto& [k, n] : sc) ts += n;
    double tv = 0.0;
    std::vector<std::string> keys;
    for (auto& [k, n] : mc) keys.push_back(k);
    for (auto& [k, n] : sc) if (mc.find(k) == mc.end()) keys.push_back(k);
    for (const auto& k : keys) {
        double pm = mc.count(k) ? double(mc.at(k)) / tm : 0.0;
        double ps = sc.count(k) ? double(sc.at(k)) / ts : 0.0;
        tv += std::abs(pm - ps);
    }
    EXPECT_LT(0.5 * tv, 0.06) << "MPS and SV sampling distributions must agree";
}

TEST(R1121Mps, MeasureSequentialBitOrderQubitZeroRightmost) {
    MPSState mps(2);
    mps.apply_single_qubit_gate(X_gate(), 0);  // q0 = 1, q1 = 0
    std::mt19937_64 rng(99);
    EXPECT_EQ(mps.measure_sequential(rng), "01") << "qubit 0 is the rightmost char";
}

TEST(R1121Mps, MeasuredBellCircuitOnlyCorrelatedKeys) {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).measure_all();
    MPSSimulator sim;
    auto res = sim.run(qc, 64, 2000, 5);
    int total = 0;
    for (const auto& [bits, n] : res.counts) {
        EXPECT_TRUE(bits == "00" || bits == "11") << "unexpected key " << bits;
        total += n;
    }
    EXPECT_EQ(total, 2000);
}
