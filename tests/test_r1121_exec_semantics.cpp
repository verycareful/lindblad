// R.1.12.1 total-coverage suite, Batch 2: execution semantics across the
// statevector, density-matrix and MPS backends. Plan section "Batch 2".
//
// Exercises terminal measurement, partial measurement into permuted clbits,
// feedforward (deterministic), shots in {0, 1, 1024}, cross-backend agreement
// on deterministic circuits, and the eval_expectation contract (happy path +
// throw on measure/conditional circuits). Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

std::unordered_map<std::string, int> sv_counts(const QuantumCircuit& qc,
                                               int shots, uint64_t seed) {
    StatevectorSimulator s;
    return s.run(qc, shots, seed).counts;
}
std::unordered_map<std::string, int> dm_counts(const QuantumCircuit& qc,
                                               int shots, uint64_t seed) {
    DensityMatrixSimulator s;
    NoiseModel ideal;
    return s.run(qc, ideal, shots, seed).counts;
}
std::unordered_map<std::string, int> mps_counts(const QuantumCircuit& qc,
                                                int shots, uint64_t seed) {
    MPSSimulator s;
    return s.run(qc, 64, shots, seed).counts;
}

}  // namespace

// =============================================================================
// Terminal measurement — width and totals on every backend
// =============================================================================

TEST(R1121Exec, TerminalMeasurementDeterministicAllBackends) {
    QuantumCircuit qc(3, 3);
    qc.x(0).x(2).measure_all();  // |101> -> "101" (q0 rightmost)
    for (auto counts : {sv_counts(qc, 512, 1), dm_counts(qc, 512, 1),
                        mps_counts(qc, 512, 1)}) {
        ASSERT_EQ(counts.size(), 1u);
        EXPECT_EQ(counts.begin()->first, "101");
        EXPECT_EQ(counts.begin()->second, 512);
    }
}

TEST(R1121Exec, SuperpositionSupportAgreesAcrossBackends) {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).measure_all();
    for (auto counts : {sv_counts(qc, 4000, 3), dm_counts(qc, 4000, 3),
                        mps_counts(qc, 4000, 3)}) {
        int total = 0;
        for (const auto& [bits, n] : counts) {
            EXPECT_TRUE(bits == "00" || bits == "11") << "unexpected key " << bits;
            total += n;
        }
        EXPECT_EQ(total, 4000);
    }
}

// =============================================================================
// Partial measurement into permuted classical bits
// =============================================================================

TEST(R1121Exec, PartialMeasurementUsesClbitMapAndWidth) {
    QuantumCircuit qc(2, 2);
    qc.x(0);
    qc.measure(0, 1);  // q0 -> c1
    qc.measure(1, 0);  // q1 -> c0
    auto counts = sv_counts(qc, 256, 9);
    ASSERT_EQ(counts.size(), 1u);
    // key = c1 c0 = q0 q1 = "1" "0" -> "10".
    EXPECT_EQ(counts.begin()->first, "10");
    EXPECT_EQ(counts.begin()->first.size(), 2u);
}

// =============================================================================
// Feedforward (deterministic) at shots 1024, 1, and 0
// =============================================================================

TEST(R1121Exec, DeterministicFeedforward) {
    QuantumCircuit qc(2, 2);
    qc.x(0);                       // q0 = 1
    qc.measure(0, 0);              // c0 = 1
    qc.add_if(0, 1, GT::X, {1});   // c0==1 -> X on q1 -> q1 = 1
    qc.measure(1, 1);              // c1 = 1

    auto c1024 = sv_counts(qc, 1024, 4);
    ASSERT_EQ(c1024.size(), 1u);
    EXPECT_EQ(c1024.begin()->first, "11");
    EXPECT_EQ(c1024.begin()->second, 1024);

    auto c1 = sv_counts(qc, 1, 4);
    ASSERT_EQ(c1.size(), 1u);
    EXPECT_EQ(c1.begin()->first, "11");

    // shots == 0: one seeded trajectory; conditions honoured and the state
    // collapses, but counts are NOT populated (the outcome lives in
    // final_state). |11> = amp index 3 (q0=1 + 2*q1=1).
    StatevectorSimulator sim;
    auto r0 = sim.run(qc, 0, 4);
    EXPECT_TRUE(r0.counts.empty()) << "shots==0 does not sample counts";
    EXPECT_NEAR(r0.final_state.probability(3), 1.0, 1e-9);
}

TEST(R1121Exec, FeedforwardAgreesOnDensityMatrixBackend) {
    QuantumCircuit qc(2, 2);
    qc.x(0).measure(0, 0);
    qc.add_if(0, 1, GT::X, {1});
    qc.measure(1, 1);
    auto counts = dm_counts(qc, 512, 8);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.begin()->first, "11");
}

// =============================================================================
// eval_expectation — happy path and throw on measure/conditional circuits
// =============================================================================

TEST(R1121Exec, EvalExpectationOnUnitaryCircuit) {
    QuantumCircuit qc(1);
    qc.h(0);  // |+>
    auto obs = SparsePauliOp::from_list({{"X", Complex128(1.0, 0.0)}});
    StatevectorSimulator sim;
    EXPECT_NEAR(sim.eval_expectation(qc, obs), 1.0, 1e-9);  // <+|X|+> = 1
}

TEST(R1121Exec, EvalExpectationThrowsOnMeasurement) {
    QuantumCircuit qc(1, 1);
    qc.h(0).measure(0, 0);
    auto obs = SparsePauliOp::from_list({{"Z", Complex128(1.0, 0.0)}});
    StatevectorSimulator sim;
    EXPECT_ANY_THROW(sim.eval_expectation(qc, obs));
}

// =============================================================================
// The execution-strategy matrix: {9 measurement scenarios} x {shots 0,1,1024}
// x {SV, DM, MPS}. Each scenario carries its analytic outcome distribution;
// every backend must match it (and therefore each other) statistically.
// =============================================================================

namespace {

using Dist = std::unordered_map<std::string, double>;
using Counts = std::unordered_map<std::string, int>;

// Total-variation distance between an empirical counts map and an analytic
// distribution: 0.5 * sum_k |freq_k - p_k| over the union of keys.
double tv_distance(const Counts& counts, const Dist& analytic) {
    int total = 0;
    for (const auto& [k, n] : counts) total += n;
    if (total == 0) return 1.0;
    double tv = 0.0;
    // keys present in the empirical distribution
    for (const auto& [k, n] : counts) {
        double p = 0.0;
        auto it = analytic.find(k);
        if (it != analytic.end()) p = it->second;
        tv += std::abs(static_cast<double>(n) / total - p);
    }
    // keys only in the analytic distribution
    for (const auto& [k, p] : analytic) {
        if (counts.find(k) == counts.end()) tv += p;
    }
    return 0.5 * tv;
}

struct Scenario {
    const char* name;
    QuantumCircuit (*build)();
    Dist analytic;
};

QuantumCircuit s_bell()        { QuantumCircuit qc(2,2); qc.h(0).cx(0,1).measure_all(); return qc; }
QuantumCircuit s_permuted()    { QuantumCircuit qc(2,2); qc.x(0).measure(0,1).measure(1,0); return qc; }
QuantumCircuit s_mid_gate()    { QuantumCircuit qc(1,2); qc.h(0).measure(0,0).x(0).measure(0,1); return qc; }
QuantumCircuit s_ff_det()      { QuantumCircuit qc(2,2); qc.x(0).measure(0,0).add_if(0,1,GT::X,{1}).measure(1,1); return qc; }
QuantumCircuit s_ff_stoch()    { QuantumCircuit qc(2,2); qc.h(0).measure(0,0).add_if(0,1,GT::X,{1}).measure(1,1); return qc; }
QuantumCircuit s_cond_nomeas() { QuantumCircuit qc(1,1); qc.add_if(0,1,GT::X,{0}).measure(0,0); return qc; }
QuantumCircuit s_multi_same()  { QuantumCircuit qc(1,1); qc.x(0).measure(0,0).measure(0,0); return qc; }
QuantumCircuit s_reset_after() { QuantumCircuit qc(1,2); qc.x(0).measure(0,0).reset(0).measure(0,1); return qc; }
QuantumCircuit s_reset_before(){ QuantumCircuit qc(1,1); qc.x(0).reset(0).measure(0,0); return qc; }

const Scenario kScenarios[] = {
    {"terminal_measure_all",      s_bell,        {{"00", 0.5}, {"11", 0.5}}},
    {"partial_permuted_clbits",   s_permuted,    {{"10", 1.0}}},
    {"mid_measure_then_gate",     s_mid_gate,    {{"10", 0.5}, {"01", 0.5}}},
    {"feedforward_deterministic", s_ff_det,      {{"11", 1.0}}},
    {"feedforward_stochastic",    s_ff_stoch,    {{"00", 0.5}, {"11", 0.5}}},
    {"conditional_no_measure",    s_cond_nomeas, {{"0", 1.0}}},
    {"multiple_measure_same_clbit", s_multi_same,{{"1", 1.0}}},
    {"reset_after_measure",       s_reset_after, {{"01", 1.0}}},
    {"reset_before_measure",      s_reset_before,{{"0", 1.0}}},
};

}  // namespace

TEST(R1121Exec, StrategyMatrixManyShotsAllBackendsMatchAnalytic) {
    const int shots = 8192;
    for (const auto& sc : kScenarios) {
        auto qc = sc.build();
        SCOPED_TRACE(sc.name);
        const Counts cs = sv_counts(qc, shots, 12345);
        const Counts cd = dm_counts(qc, shots, 12345);
        const Counts cm = mps_counts(qc, shots, 12345);
        // Each backend agrees with the analytic distribution (hence each other).
        EXPECT_LT(tv_distance(cs, sc.analytic), 0.06) << "SV";
        EXPECT_LT(tv_distance(cd, sc.analytic), 0.06) << "DM";
        EXPECT_LT(tv_distance(cm, sc.analytic), 0.06) << "MPS";
        // Key widths are uniform within a scenario across backends.
        if (!cs.empty() && !cm.empty())
            EXPECT_EQ(cs.begin()->first.size(), cm.begin()->first.size());
    }
}

TEST(R1121Exec, StrategyMatrixSingleShotKeyInSupport) {
    for (const auto& sc : kScenarios) {
        auto qc = sc.build();
        SCOPED_TRACE(sc.name);
        for (auto fn : {sv_counts, dm_counts, mps_counts}) {
            const Counts c = fn(qc, 1, 777);
            ASSERT_EQ(c.size(), 1u);
            const std::string& key = c.begin()->first;
            EXPECT_EQ(c.begin()->second, 1);
            EXPECT_GT(sc.analytic.count(key), 0u)
                << "single-shot key '" << key << "' must be in the analytic support";
        }
    }
}

TEST(R1121Exec, StrategyMatrixShotsZeroLeavesCountsEmpty) {
    // shots == 0 is one seeded trajectory: conditions honoured, MEASURE outcomes
    // recorded into the trajectory, but counts is never populated.
    for (const auto& sc : kScenarios) {
        auto qc = sc.build();
        SCOPED_TRACE(sc.name);
        EXPECT_TRUE(sv_counts(qc, 0, 1).empty()) << "SV";
        EXPECT_TRUE(dm_counts(qc, 0, 1).empty()) << "DM";
        EXPECT_TRUE(mps_counts(qc, 0, 1).empty()) << "MPS";
    }
}

TEST(R1121Exec, ShotsZeroTrajectoryIsSeedDeterministic) {
    // Two runs with the same seed give the same trajectory final state; a
    // stochastic scenario at shots==0 picks ONE branch reproducibly.
    auto qc = s_ff_stoch();
    StatevectorSimulator sim;
    auto a = sim.run(qc, 0, 555);
    auto b = sim.run(qc, 0, 555);
    ASSERT_EQ(a.final_state.dim, b.final_state.dim);
    for (size_t i = 0; i < a.final_state.dim; ++i) {
        EXPECT_NEAR(a.final_state.probability(i), b.final_state.probability(i), 1e-12);
    }
    // The chosen branch is a computational basis state (|00> or |11>).
    double p0 = a.final_state.probability(0), p3 = a.final_state.probability(3);
    EXPECT_NEAR(std::max(p0, p3), 1.0, 1e-9) << "feedforward collapses to a basis state";
}
