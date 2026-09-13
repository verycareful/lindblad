// 1.1.28.1 test wave - the bond-split profile counters, and #126.
//
// 1.1.28.0 added MPSState::svd_time_ns(), the time a run spends in the
// bond-split ladder, accumulated over the same splits svd_call_count() counts
// and published as a share of each MPS benchmark row. The contract from the
// header: the interval covers the whole ladder (factorisation, verification,
// any Gram rescue), it is taken at BOTH call sites (the gate path and the
// rebuild from dense amplitudes), it accumulates rather than resets, and a
// split that threw contributes nothing because a run that threw has no result
// to profile. The last clause has no deterministic trigger and is not pinned
// here; the rest is.
//
// The per-shot path (#126). Any circuit with a mid-circuit measurement, a
// classical condition or a reset, run at shots > 0, re-simulates from a fresh
// chain per trajectory, and the returned chain is the last trajectory's. All
// five profile figures on it therefore describe one shot out of N. The
// documentation says so; the numbers are wrong all the same, because the
// purpose of the figures is to profile a run and on that path they do not.
// The three PerShotPath* tests below assert the run-level contract the other
// two paths already meet, and they are RED on 1.1.28.1: a test release does
// not edit library code. They define what the fix has to satisfy.
//
// The two remaining figures, gram_fallback_count() and
// max_verify_residual_excess(), are lost by the same rebuild and ride on the
// same fix. A Gram rescue has no deterministic trigger from the public API, so
// they are not pinned separately.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "compare_common.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace lindblad;
using lindblad_bench::load_corpus_circuit;

namespace {

constexpr int kShots = 0;
constexpr std::uint64_t kSeed = 42;
constexpr double kEps = std::numeric_limits<double>::epsilon();

QuantumCircuit one_split_circuit() {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    return qc;
}

// Mid-circuit measurement with a gate on the measured qubit afterwards, so the
// run takes the per-shot trajectory path. No classical condition, so every
// trajectory applies the same gates and performs the same number of splits,
// which is what makes N trajectories exactly N times one.
QuantumCircuit per_shot_circuit() {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1);
    qc.measure(0, 0);
    qc.x(0);
    qc.measure(1, 1);
    return qc;
}

// CX as MPSState::apply_two_qubit_gate takes it: rows and columns index
// (q1, q2) with q1 the high bit, so with q1 as control it swaps |10> and |11>.
std::array<Complex128, 16> cx_matrix() {
    std::array<Complex128, 16> u{};
    u[0 * 4 + 0] = Complex128(1.0, 0.0);
    u[1 * 4 + 1] = Complex128(1.0, 0.0);
    u[2 * 4 + 3] = Complex128(1.0, 0.0);
    u[3 * 4 + 2] = Complex128(1.0, 0.0);
    return u;
}

std::array<Complex128, 4> hadamard_matrix() {
    return {Complex128(INV_SQRT2, 0.0), Complex128(INV_SQRT2, 0.0),
            Complex128(INV_SQRT2, 0.0), Complex128(-INV_SQRT2, 0.0)};
}

// A normalised, non-symmetric n-qubit state: amplitude k+1 at index k, scaled
// by the exact sum of squares. Entangled across every cut, so a rebuild keeps
// every split it performs.
Statevector ramp_state(int n) {
    const std::size_t dim = std::size_t{1} << n;
    double sum_sq = 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
        const double a = static_cast<double>(k + 1);
        sum_sq += a * a;
    }
    std::vector<Complex128> amps(dim);
    for (std::size_t k = 0; k < dim; ++k) {
        amps[k] = Complex128(static_cast<double>(k + 1) / std::sqrt(sum_sq), 0.0);
    }
    Statevector sv(n);
    sv.set_amplitudes(amps);
    return sv;
}

// Everything the chain reports about its own splits, read in one place so a
// test compares whole readings rather than one figure at a time.
struct Profile {
    std::size_t splits = 0;
    std::uint64_t nanos = 0;
    std::size_t rescues = 0;
    double discarded = 0.0;
    double excess = 0.0;
};

Profile profile_of(const MPSState& s) {
    Profile p;
    p.splits = s.svd_call_count();
    p.nanos = s.svd_time_ns();
    p.rescues = s.gram_fallback_count();
    p.discarded = s.truncation_error();
    p.excess = s.max_verify_residual_excess();
    return p;
}

Profile profile_of_run(const QuantumCircuit& qc, int cap, int shots) {
    MPSSimulator sim;
    return profile_of(sim.run(qc, cap, shots, kSeed).final_state);
}

}  // namespace

// =============================================================================
// The chain's own bookkeeping
// =============================================================================

TEST(V11281SvdTime, FreshChainReportsZeroEverywhere) {
    const Profile p = profile_of(MPSState(3));
    EXPECT_EQ(p.splits, 0u);
    EXPECT_EQ(p.nanos, 0u);
    EXPECT_EQ(p.rescues, 0u);
    EXPECT_EQ(p.discarded, 0.0);
    EXPECT_EQ(p.excess, 0.0);
}

TEST(V11281SvdTime, SingleQubitGatesDoNotSplit) {
    QuantumCircuit qc(3);
    qc.h(0).x(1).ry(PI_4, 2).rz(PI_2, 0).h(2);
    const Profile p = profile_of_run(qc, 4, kShots);
    EXPECT_EQ(p.splits, 0u);
    EXPECT_EQ(p.nanos, 0u)
        << "time was charged to the ladder on a circuit that never entered it";
}

TEST(V11281SvdTime, TimeAndCountAdvanceTogether) {
    const Profile p = profile_of_run(one_split_circuit(), 4, kShots);
    EXPECT_EQ(p.splits, 1u);
    EXPECT_GT(p.nanos, 0u)
        << "a split takes microseconds against a nanosecond clock, so a zero "
           "reading means the bracket is not around the ladder";
}

TEST(V11281SvdTime, CountersAccumulateAcrossSplits) {
    // Directly on the chain, one adjacent CX at a time. The count is exact and
    // the time is strictly increasing: each split spends a positive interval
    // in the ladder and nothing between splits touches the accumulator.
    MPSState chain(2, 4);
    std::uint64_t last_nanos = 0;
    for (std::size_t i = 1; i <= 3; ++i) {
        chain.apply_two_qubit_gate(cx_matrix(), 0, 1);
        const Profile p = profile_of(chain);
        EXPECT_EQ(p.splits, i);
        EXPECT_GT(p.nanos, last_nanos) << "split " << i << " added no time";
        last_nanos = p.nanos;
    }
}

TEST(V11281SvdTime, SingleQubitGatesLeaveTheCountersAlone) {
    MPSState chain(2, 4);
    chain.apply_two_qubit_gate(cx_matrix(), 0, 1);
    const Profile before = profile_of(chain);
    chain.apply_single_qubit_gate(hadamard_matrix(), 0);
    chain.apply_single_qubit_gate(hadamard_matrix(), 1);
    const Profile after = profile_of(chain);
    EXPECT_EQ(after.splits, before.splits);
    EXPECT_EQ(after.nanos, before.nanos);
    EXPECT_EQ(after.discarded, before.discarded);
}

TEST(V11281SvdTime, RebuildPathIsCounted) {
    // The rebuild from dense amplitudes is the ladder's second call site. A
    // sequential rebuild of an n-site chain performs n - 1 splits, one per
    // bond, and the bracket at that site charges each of them.
    const int n = 4;
    MPSState chain(n, 16);
    chain.rebuild_from_statevector(ramp_state(n));
    const Profile p = profile_of(chain);
    EXPECT_EQ(p.splits, static_cast<std::size_t>(n - 1));
    EXPECT_GT(p.nanos, 0u) << "the rebuild path's splits are not timed";
}

TEST(V11281SvdTime, RebuildAccumulatesOntoEarlierSplits) {
    // Accumulate rather than reset, the same contract truncation_error()
    // documents: a chain rebuilt part way through still carries what came
    // before.
    const int n = 3;
    MPSState chain(n, 8);
    chain.apply_two_qubit_gate(cx_matrix(), 0, 1);
    const Profile before = profile_of(chain);
    ASSERT_EQ(before.splits, 1u);
    chain.rebuild_from_statevector(ramp_state(n));
    const Profile after = profile_of(chain);
    EXPECT_EQ(after.splits, before.splits + static_cast<std::size_t>(n - 1));
    EXPECT_GT(after.nanos, before.nanos);
}

TEST(V11281SvdTime, LadderTimeIsBoundedByTheRunWallClock) {
    // The intervals the accumulator sums lie inside the run, so their total
    // cannot exceed the run's own wall time. A bracket that double-counted, or
    // that read a different clock, would show here. qv_n8 gives a few hundred
    // splits, enough for the share to be a real fraction rather than noise.
    const auto qc = load_corpus_circuit("qv_n8.qasm", false);
    const int cap = 1 << (qc.n_qubits / 2);
    MPSSimulator sim;
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = sim.run(qc, cap, kShots, kSeed);
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count();
    const Profile p = profile_of(r.final_state);
    ASSERT_GT(p.splits, 0u);
    EXPECT_GT(p.nanos, 0u);
    EXPECT_LE(p.nanos, static_cast<std::uint64_t>(wall))
        << "the ladder reports more time than the whole run took";
    ::testing::Test::RecordProperty("qv_n8_ladder_share_percent",
                                    std::to_string(100.0 * static_cast<double>(p.nanos) /
                                                   static_cast<double>(wall)));
    ::testing::Test::RecordProperty("qv_n8_mean_split_ns",
                                    std::to_string(p.nanos / p.splits));
}

// =============================================================================
// Which splits a run's reading covers, path by path
// =============================================================================

TEST(V11281SvdTime, ZeroShotsIsOneTrajectory) {
    // shots == 0 runs one seeded trajectory, measurements collapsed, and the
    // figures cover exactly that trajectory: one CX, one split.
    const Profile p = profile_of_run(per_shot_circuit(), 4, kShots);
    EXPECT_EQ(p.splits, 1u);
    EXPECT_GT(p.nanos, 0u);
}

TEST(V11281SvdTime, TerminalOnlyPathCountsTheSinglePass) {
    // measure_all is one forward pass with the measures skipped, then sampling
    // from the final state. Sampling performs no split, so the reading at any
    // shot count equals the reading of the bare circuit at shots == 0.
    const auto bare = load_corpus_circuit("qv_n8.qasm", false);
    auto measured = load_corpus_circuit("qv_n8.qasm", true);
    const int cap = 1 << (bare.n_qubits / 2);
    const Profile once = profile_of_run(bare, cap, kShots);
    const Profile sampled = profile_of_run(measured, cap, 64);
    ASSERT_GT(once.splits, 0u);
    EXPECT_EQ(sampled.splits, once.splits)
        << "sampling from the final state changed the split count";
    EXPECT_EQ(sampled.discarded, once.discarded);
    EXPECT_GT(sampled.nanos, 0u);
}

// -----------------------------------------------------------------------------
// #126. RED on 1.1.28.1 by design: the contract is that the returned chain's
// figures describe the whole run on every path, as they do on the two above.
// -----------------------------------------------------------------------------

TEST(V11281SvdTime, PerShotPathCountsEverySplitOfEveryShot) {
    const int shots = 100;
    const Profile one = profile_of_run(per_shot_circuit(), 4, kShots);
    ASSERT_EQ(one.splits, 1u) << "the fixture is meant to split exactly once";
    const Profile all = profile_of_run(per_shot_circuit(), 4, shots);
    EXPECT_EQ(all.splits, static_cast<std::size_t>(shots) * one.splits)
        << "the per-shot path reports the last trajectory's splits rather than "
           "the run's (#126)";
}

TEST(V11281SvdTime, PerShotPathAccumulatesTime) {
    // A thousand trajectories spend roughly a thousand times one trajectory's
    // ladder time. The bar is a tenth of that, which is where a scheduling
    // stall inflating the single reading cannot reach, and where a last-shot
    // reading, being one trajectory's time, cannot reach either.
    const int shots = 1000;
    const int factor = 10;
    // One run first so the single reading is not the process's cold first
    // pass through the ladder.
    profile_of_run(per_shot_circuit(), 4, kShots);
    const Profile one = profile_of_run(per_shot_circuit(), 4, kShots);
    ASSERT_GT(one.nanos, 0u);
    const Profile all = profile_of_run(per_shot_circuit(), 4, shots);
    ::testing::Test::RecordProperty("single_trajectory_ns", std::to_string(one.nanos));
    ::testing::Test::RecordProperty("thousand_trajectories_ns", std::to_string(all.nanos));
    EXPECT_GE(all.nanos, static_cast<std::uint64_t>(factor) * one.nanos)
        << "the per-shot path reports the last trajectory's ladder time rather "
           "than the run's (#126)";
}

TEST(V11281SvdTime, PerShotPathAccumulatesDiscardedWeight) {
    // At a cap of 1 the Bell pair's split keeps one of two equal directions and
    // discards the other, before any measurement is drawn, so every trajectory
    // discards the same weight. N trajectories discard N times it, up to the
    // rounding of summing N equal terms.
    const int shots = 100;
    const Profile one = profile_of_run(per_shot_circuit(), 1, kShots);
    ASSERT_GT(one.discarded, 0.0) << "the fixture is meant to truncate";
    const Profile all = profile_of_run(per_shot_circuit(), 1, shots);
    EXPECT_GT(all.discarded, one.discarded)
        << "the per-shot path reports the last trajectory's discarded weight "
           "rather than the run's (#126)";
    const double expected = static_cast<double>(shots) * one.discarded;
    EXPECT_NEAR(all.discarded, expected,
                static_cast<double>(shots) * static_cast<double>(shots) * kEps * one.discarded);
}
