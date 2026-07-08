// R.1.13.1 test patch — Clifford terminal-measurement fast path (audit F-19).
// The fast path runs the deterministic gate pass once, then copies the tableau
// and measures per shot. It must reproduce the statevector distribution; the
// general per-shot trajectory (feedforward / reset) must stay correct.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <string>
#include <unordered_map>

using namespace lindblad;

TEST(R1131Clifford, IsCliffordClassification) {
    QuantumCircuit cliff(2, 2);
    cliff.h(0).cx(0, 1).s(0).measure_all();
    EXPECT_TRUE(CliffordSimulator::is_clifford(cliff));

    QuantumCircuit non_cliff(1, 1);
    non_cliff.h(0).t(0).measure(0, 0);    // T is non-Clifford
    EXPECT_FALSE(CliffordSimulator::is_clifford(non_cliff));
}

// Fast path: a GHZ state's terminal measurement yields only 000 / 111, ~50/50.
TEST(R1131Clifford, TerminalGhzDistribution) {
    QuantumCircuit qc(3, 3);
    qc.h(0).cx(0, 1).cx(1, 2).measure_all();

    CliffordSimulator sim;
    auto res = sim.run(qc, /*shots=*/4000, /*seed=*/7);

    int total = 0;
    for (const auto& [k, v] : res.counts) {
        EXPECT_TRUE(k == "000" || k == "111") << "unexpected key " << k;
        total += v;
    }
    EXPECT_EQ(total, 4000);
    EXPECT_GT(res.counts["000"], 1500);
    EXPECT_GT(res.counts["111"], 1500);
}

// Fast path vs reference: the terminal-measurement distribution must match the
// statevector simulator's sampled distribution for a non-symmetric Clifford
// state (support + per-key frequencies).
TEST(R1131Clifford, TerminalPathMatchesStatevectorSampler) {
    auto make = [] {
        QuantumCircuit qc(3, 3);
        qc.h(0).cx(0, 1).cx(0, 2).x(1).s(2).measure_all();
        return qc;
    };

    const int shots = 8000;
    CliffordSimulator csim;
    auto cliff = csim.run(make(), shots, 123);

    StatevectorSimulator svsim;
    auto sv = svsim.run(make(), shots, 123);

    // Same support.
    EXPECT_EQ(cliff.counts.size(), sv.counts.size());
    for (const auto& [k, v] : cliff.counts) {
        ASSERT_TRUE(sv.counts.count(k)) << "Clifford produced key absent in SV: " << k;
        const double fc = static_cast<double>(v) / shots;
        const double fs = static_cast<double>(sv.counts.at(k)) / shots;
        EXPECT_NEAR(fc, fs, 0.05) << "frequency mismatch for key " << k;
    }
}

// General per-shot path (reset forces the trajectory route) stays correct.
TEST(R1131Clifford, ResetGeneralPathDeterministic) {
    QuantumCircuit qc(1, 1);
    qc.x(0).reset(0).measure(0, 0);       // deterministic |0>

    CliffordSimulator sim;
    auto res = sim.run(qc, /*shots=*/500, /*seed=*/3);
    EXPECT_EQ(res.counts["0"], 500);
    EXPECT_EQ(res.counts.count("1"), 0u);
}
