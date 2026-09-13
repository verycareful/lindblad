// 1.1.28.2 test wave - the per-shot path reports the run, not the last shot
// (#126).
//
// MPSSimulator::run re-simulates per shot whenever a circuit measures
// mid-way, resets, or branches on a clbit, and every trajectory runs on a
// chain of its own. The chain a caller reads afterwards carries the last
// trajectory's tensors, which is the only sensible choice, and the profile
// figures of ALL of them, which is the only useful one: the figures exist to
// profile a run, and a run of N trajectories is N trajectories' worth of
// splits. The three PerShotPath* tests in the 1.1.28.1 wave assert exactly
// that contract and go green with this release; this file covers what they
// left open.
//
// The mechanism is MPSState::absorb_profile, which folds one chain's five
// figures into another's: the four tallies add and the worst accepted
// residual takes the max, so a chain that absorbed another reports what one
// chain performing both sets of splits would. The first group here pins that
// method directly, since the simulator is one caller of it and a second could
// rely on the same reading.
//
// The second group is the per-shot path through the public entry point: one
// shot reports exactly what the zero-shot trajectory reports, the totals grow
// with the shot count in the way each figure's own contract says, seeding
// rebuilds are counted per shot alongside the gate splits, and the returned
// tensors are a trajectory's end state rather than the accumulator's.
//
// gram_fallback_count() rides on the same method and the same loop, and has no
// deterministic trigger from the public API, so it is pinned through
// absorb_profile directly and not through a run.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

using namespace lindblad;

namespace {

constexpr int kShots = 0;
constexpr std::uint64_t kSeed = 42;
constexpr double kEps = std::numeric_limits<double>::epsilon();

// Slack a handful of rounding steps are entitled to, in units of eps: the
// figure the truncation ladder's verify rung allows per unit of dimension.
constexpr double kRoundingSlack = 64.0;

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

// Mid-circuit measurement with a gate on the measured qubit afterwards, so the
// run takes the per-shot trajectory path. No classical condition, so every
// trajectory applies the same gates and splits exactly once. The end state of
// a trajectory is |01> or |10> (the X lands on the measured qubit), never the
// |00> a fresh chain holds.
QuantumCircuit per_shot_circuit() {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1);
    qc.measure(0, 0);
    qc.x(0);
    qc.measure(1, 1);
    return qc;
}

// Three qubits, two gate splits per trajectory, on the per-shot path.
QuantumCircuit per_shot_circuit_3q() {
    QuantumCircuit qc(3, 2);
    qc.h(0).cx(0, 1);
    qc.measure(0, 0);
    qc.x(0);
    qc.cx(1, 2);
    qc.measure(2, 1);
    return qc;
}

// A normalised, non-symmetric n-qubit state: amplitude k+1 at index k, scaled
// by the exact sum of squares. Entangled across every cut, so a rebuild keeps
// every split it performs.
std::shared_ptr<const Statevector> ramp_state(int n) {
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
    auto sv = std::make_shared<Statevector>(n);
    sv->set_amplitudes(amps);
    return sv;
}

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

// What absorbing `b` into a chain reading `a` must produce, by the method's
// own rule: tallies add, the residual takes the max.
Profile folded(const Profile& a, const Profile& b) {
    Profile p;
    p.splits = a.splits + b.splits;
    p.nanos = a.nanos + b.nanos;
    p.rescues = a.rescues + b.rescues;
    p.discarded = a.discarded + b.discarded;
    p.excess = std::max(a.excess, b.excess);
    return p;
}

void expect_same_profile(const Profile& got, const Profile& want) {
    EXPECT_EQ(got.splits, want.splits);
    EXPECT_EQ(got.nanos, want.nanos);
    EXPECT_EQ(got.rescues, want.rescues);
    // The method performs the same double addition the expectation does, so
    // the two are the same bits, not merely close.
    EXPECT_EQ(got.discarded, want.discarded);
    EXPECT_EQ(got.excess, want.excess);
}

// A two-site chain that has split once, and a three-site chain that has split
// twice, both at a cap of 1 so every split discards weight and the figures
// are nonzero on every axis a test can reach.
MPSState one_split_chain() {
    MPSState chain(2, 1);
    chain.apply_single_qubit_gate(hadamard_matrix(), 0);
    chain.apply_two_qubit_gate(cx_matrix(), 0, 1);
    return chain;
}

MPSState two_split_chain() {
    MPSState chain(3, 1);
    chain.apply_single_qubit_gate(hadamard_matrix(), 0);
    chain.apply_two_qubit_gate(cx_matrix(), 0, 1);
    chain.apply_single_qubit_gate(hadamard_matrix(), 1);
    chain.apply_two_qubit_gate(cx_matrix(), 1, 2);
    return chain;
}

bool same_amplitudes(const Statevector& a, const Statevector& b) {
    const auto va = a.amplitudes();
    const auto vb = b.amplitudes();
    if (va.size() != vb.size()) return false;
    for (std::size_t i = 0; i < va.size(); ++i) {
        if (va[i].real != vb[i].real || va[i].imag != vb[i].imag) return false;
    }
    return true;
}

MPSSimulator::Result run_default(const QuantumCircuit& qc, int cap, int shots) {
    MPSSimulator sim;
    return sim.run(qc, cap, shots, kSeed);
}

}  // namespace

// =============================================================================
// absorb_profile directly
// =============================================================================

TEST(V11282ProfileTotals, AbsorbAddsTheTalliesAndTakesTheWorstResidual) {
    MPSState a = one_split_chain();
    const MPSState b = two_split_chain();
    const Profile pa = profile_of(a);
    const Profile pb = profile_of(b);
    ASSERT_EQ(pa.splits, 1u);
    ASSERT_EQ(pb.splits, 2u);
    ASSERT_GT(pa.discarded, 0.0) << "the fixture is meant to truncate";
    ASSERT_GT(pb.discarded, 0.0) << "the fixture is meant to truncate";

    a.absorb_profile(b);
    expect_same_profile(profile_of(a), folded(pa, pb));
}

TEST(V11282ProfileTotals, AbsorbIsOrderIndependent) {
    // Integer sums commute, IEEE addition of two terms commutes, and max
    // commutes, so absorbing in either direction reads the same. A copy of a
    // chain carries its figures, which is what lets one source feed both.
    const MPSState a0 = one_split_chain();
    const MPSState b0 = two_split_chain();
    MPSState ab = a0;
    ab.absorb_profile(b0);
    MPSState ba = b0;
    ba.absorb_profile(a0);
    expect_same_profile(profile_of(ab), folded(profile_of(a0), profile_of(b0)));
    expect_same_profile(profile_of(ba), profile_of(ab));
}

TEST(V11282ProfileTotals, AbsorbLeavesTheStateAndItsSettingsAlone) {
    // Only the five figures move. The tensors, the register width, the cap,
    // the cutoff and the kernel selection are the absorbing chain's own, and
    // the source's differ on every one of them here so a leak would show.
    const double cutoff = 1e-12;
    MPSState a(2, 4, cutoff);
    a.svd_method = SVDMethod::Jacobi;
    a.apply_single_qubit_gate(hadamard_matrix(), 0);
    a.apply_two_qubit_gate(cx_matrix(), 0, 1);
    const Statevector before = a.to_statevector();

    MPSState b = two_split_chain();
    b.svd_method = SVDMethod::BDC;
    a.absorb_profile(b);

    EXPECT_TRUE(same_amplitudes(a.to_statevector(), before));
    EXPECT_EQ(a.n_qubits, 2);
    EXPECT_EQ(a.max_bond_dim, 4);
    EXPECT_EQ(a.cutoff, cutoff);
    EXPECT_EQ(a.svd_method, SVDMethod::Jacobi);
}

TEST(V11282ProfileTotals, AbsorbingAFreshChainChangesNothing) {
    // A fresh chain reads zero on every figure, so absorbing it is the
    // identity: the tallies gain zero and the max keeps the larger side.
    MPSState a = two_split_chain();
    const Profile before = profile_of(a);
    a.absorb_profile(MPSState(5, 8));
    expect_same_profile(profile_of(a), before);
}

TEST(V11282ProfileTotals, AbsorbingItselfDoublesTheTallies) {
    // The argument may alias the receiver. Every tally then doubles, and the
    // max of a value with itself is that value.
    MPSState a = two_split_chain();
    const Profile before = profile_of(a);
    a.absorb_profile(a);
    expect_same_profile(profile_of(a), folded(before, before));
}

TEST(V11282ProfileTotals, AbsorbAccumulatesAcrossRepeats) {
    // N absorptions of the same source read N times its tallies: the method
    // adds onto whatever the chain already holds rather than replacing it.
    const int repeats = 5;
    MPSState a(2, 4);
    const MPSState b = one_split_chain();
    const Profile pb = profile_of(b);
    for (int i = 0; i < repeats; ++i) a.absorb_profile(b);
    const Profile got = profile_of(a);
    EXPECT_EQ(got.splits, static_cast<std::size_t>(repeats) * pb.splits);
    EXPECT_EQ(got.nanos, static_cast<std::uint64_t>(repeats) * pb.nanos);
    EXPECT_EQ(got.rescues, static_cast<std::size_t>(repeats) * pb.rescues);
    EXPECT_NEAR(got.discarded, static_cast<double>(repeats) * pb.discarded,
                static_cast<double>(repeats) * static_cast<double>(repeats) *
                    kEps * pb.discarded);
    EXPECT_EQ(got.excess, pb.excess);
}

// =============================================================================
// The per-shot path through MPSSimulator::run
// =============================================================================

TEST(V11282ProfileTotals, OneShotReportsExactlyOneTrajectory) {
    // shots == 0 is one seeded trajectory on the single-pass branch; shots == 1
    // is one trajectory on the per-shot branch. Same seed, same draws, same
    // splits, so the two chains agree bit for bit on the state and on every
    // figure that does not read a clock.
    const auto single = run_default(per_shot_circuit(), 4, kShots);
    const auto looped = run_default(per_shot_circuit(), 4, 1);
    const Profile ps = profile_of(single.final_state);
    const Profile pl = profile_of(looped.final_state);
    EXPECT_EQ(pl.splits, ps.splits);
    EXPECT_EQ(pl.rescues, ps.rescues);
    EXPECT_EQ(pl.discarded, ps.discarded);
    EXPECT_EQ(pl.excess, ps.excess);
    EXPECT_GT(pl.nanos, 0u);
    EXPECT_TRUE(same_amplitudes(looped.final_state.to_statevector(),
                                single.final_state.to_statevector()))
        << "the returned chain is not the trajectory that ran";
}

TEST(V11282ProfileTotals, TotalsGrowWithTheShotCount) {
    // At a cap of 1 every trajectory's one split discards the same weight
    // before any measurement is drawn, so the totals are N times one shot's
    // to the rounding of summing N equal terms. The first of N trajectories
    // draws the same numbers as the one-shot run, so the worst residual over
    // N shots is at least the one-shot reading.
    const Profile one = profile_of(run_default(per_shot_circuit(), 1, 1).final_state);
    ASSERT_EQ(one.splits, 1u);
    ASSERT_GT(one.discarded, 0.0) << "the fixture is meant to truncate";
    for (const int shots : {2, 4, 8, 32}) {
        const Profile all = profile_of(run_default(per_shot_circuit(), 1, shots).final_state);
        EXPECT_EQ(all.splits, static_cast<std::size_t>(shots) * one.splits) << shots;
        EXPECT_NEAR(all.discarded, static_cast<double>(shots) * one.discarded,
                    static_cast<double>(shots) * static_cast<double>(shots) *
                        kEps * one.discarded)
            << shots;
        EXPECT_GE(all.excess, one.excess) << shots;
    }
}

TEST(V11282ProfileTotals, SeedingRebuildsAreCountedPerShot) {
    // A dense seed is factorised into every trajectory's chain before its
    // gates run, so each shot pays n - 1 rebuild splits on top of its gate
    // splits, and the run's total is N times that sum.
    const int n = 3;
    const int shots = 7;
    RunPlan plan;
    plan.initial = InitialState::from(ramp_state(n));
    MPSSimulator sim;
    const Profile one = profile_of(sim.run(per_shot_circuit_3q(), 4, kShots, kSeed, plan).final_state);
    ASSERT_EQ(one.splits, static_cast<std::size_t>(n - 1) + 2u)
        << "one trajectory is the rebuild's n - 1 splits plus two CX";
    const Profile all = profile_of(sim.run(per_shot_circuit_3q(), 4, shots, kSeed, plan).final_state);
    EXPECT_EQ(all.splits, static_cast<std::size_t>(shots) * one.splits);
}

TEST(V11282ProfileTotals, ReturnedTensorsAreATrajectorysEndState) {
    // The accumulator that gathers the figures starts as a fresh |00> chain.
    // The chain handed back must not be it: after per_shot_circuit every
    // trajectory ends in |01> or |10>, so the amplitude at index 0 is zero and
    // exactly one of the other two carries the whole norm.
    const auto r = run_default(per_shot_circuit(), 4, 16);
    const auto amps = r.final_state.to_statevector().amplitudes();
    ASSERT_EQ(amps.size(), 4u);
    auto weight = [&](std::size_t i) {
        return amps[i].real * amps[i].real + amps[i].imag * amps[i].imag;
    };
    const double tol = kRoundingSlack * kEps;
    EXPECT_NEAR(weight(0), 0.0, tol);
    EXPECT_NEAR(weight(3), 0.0, tol);
    EXPECT_NEAR(weight(1) + weight(2), 1.0, tol);
    EXPECT_TRUE(weight(1) < tol || weight(2) < tol)
        << "a trajectory ends in one basis state, not a superposition";
    EXPECT_EQ(r.final_state.svd_call_count(), 16u);
}

TEST(V11282ProfileTotals, ReturnedChainIsTheLastTrajectoryInEveryRespect) {
    // A supplied chain seeds every trajectory with its own cap, cutoff and
    // kernel, and the chain that comes back is that trajectory rather than
    // the simulator's accumulator wearing its tensors: every setting is the
    // supplied chain's, with the run's totals on top.
    const int cap = 8;
    const double cutoff = 1e-12;
    auto seed = std::make_shared<MPSState>(2, cap, cutoff);
    seed->svd_method = SVDMethod::Jacobi;
    RunPlan plan;
    plan.initial = InitialState::from(std::shared_ptr<const MPSState>(seed));
    MPSSimulator sim;
    sim.svd_method = SVDMethod::BDC;
    const int shots = 6;
    const auto r = sim.run(per_shot_circuit(), 4, shots, kSeed, plan);
    EXPECT_EQ(r.final_state.svd_method, SVDMethod::Jacobi);
    EXPECT_EQ(r.final_state.max_bond_dim, cap);
    EXPECT_EQ(r.final_state.cutoff, cutoff);
    EXPECT_EQ(r.final_state.svd_call_count(), static_cast<std::size_t>(shots));
}

TEST(V11282ProfileTotals, TerminalOnlyAndZeroShotPathsAreUnchanged) {
    // The other two paths run one forward pass on the returned chain itself
    // and absorb nothing. One split, whichever way the same circuit is run.
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1);
    qc.measure_all();
    EXPECT_EQ(run_default(qc, 4, kShots).final_state.svd_call_count(), 1u);
    EXPECT_EQ(run_default(qc, 4, 100).final_state.svd_call_count(), 1u);
}
