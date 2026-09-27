// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.1 test wave - the fidelity figures (#98).
//
// #98 asked for a fidelity bound from each split's discarded weight. 1.1.30.0
// found the requested figure, prod_k (1 - eps_k), is an estimate rather than a
// bound, and ships it beside a rigorous floor, max(0, 1 - Delta^2/2)^2 with
// Delta the summed distance each renormalised projection moves the state. Both
// live in detail::FidelityLedger, which both MPS layers carry.
//
// Three groups of checks:
//
//   V11301FidelityLedger  the ledger's arithmetic against closed forms: a single
//                         split, the two-rotation example that separates the
//                         figures, the rounding-level regime the cancellation-
//                         free distance exists for, the clamps, and the floor
//                         never exceeding the estimate, which follows from the
//                         two definitions.
//
//   V11301FidelityFigures what each operation does to the figures on both
//                         layers: kept by a copy, a gauge change and
//                         normalisation; reset by set_tensors; accumulated by a
//                         rebuild; not folded by absorb_profile; emptied for good
//                         by every collapse, in or out of a run.
//
//   V11301FidelityBound   the floor against the true fidelity, computed from a
//                         dense reference, on binding runs of both layers under
//                         both policies; the estimate shown to fall on either
//                         side of the truth on constructed chains; and
//                         truncation_error() equal to the fall in norm_sq().

#include <gtest/gtest.h>

#include "v11301_mps_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/fidelity_ledger.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;
using lindblad::detail::FidelityLedger;
using v11301::Amplitudes;
using v11301::kEps;
using v11301::kSlack;

namespace {

// The ledger's two figures by closed form, for checking it against.
double closed_estimate(const std::vector<double>& eps) {
    double r = 1.0;
    for (double e : eps) r *= 1.0 - e;
    return r;
}

// The distance a renormalised projection keeping 1 - eps of a unit vector
// moves it: sqrt(2 - 2 sqrt(1 - eps)), written without the cancellation.
double closed_delta(double eps) {
    return std::sqrt(2.0 * eps / (1.0 + std::sqrt(1.0 - eps)));
}

double closed_floor(const std::vector<double>& eps) {
    double delta = 0.0;
    for (double e : eps) delta += closed_delta(e);
    const double overlap = std::max(0.0, 1.0 - 0.5 * delta * delta);
    return overlap * overlap;
}

// Record a split that kept 1 - eps of a block of weight `w`.
void record_fraction(FidelityLedger& l, double eps, double w = 1.0) {
    l.record((1.0 - eps) * w, eps * w);
}

// The rotation by `a` inside span{|00>, |11>}, identity on |01> and |10>. It
// is symmetric under exchanging its operands, so no operand-order convention
// enters.
std::array<Complex128, 16> pair_rotation(double a) {
    const double c = std::cos(a), s = std::sin(a);
    std::array<Complex128, 16> U{};
    U[0 * 4 + 0] = Complex128(c, 0.0);
    U[0 * 4 + 3] = Complex128(-s, 0.0);
    U[1 * 4 + 1] = Complex128(1.0, 0.0);
    U[2 * 4 + 2] = Complex128(1.0, 0.0);
    U[3 * 4 + 0] = Complex128(s, 0.0);
    U[3 * 4 + 3] = Complex128(c, 0.0);
    return U;
}

bool same_opt(const std::optional<double>& a, const std::optional<double>& b) {
    if (a.has_value() != b.has_value()) return false;
    return !a.has_value() || v11301::same_bits(*a, *b);
}

// A truncated chain on each layer, for the operation tests. The qubit one is
// a brickwork whose middle bond wants 8 at a cap of `cap`.
MPSState truncated_qubit_chain(int cap = 2) {
    MPSSimulator sim;
    return sim.run(v11301::brickwork(6, 3), cap, 0, 9).final_state;
}

QuditMPS truncated_qudit_chain() {
    QuditMPS s(5, 3, 2);
    std::uint64_t seed = 60;
    for (int q = 0; q + 1 < 5; ++q)
        s.apply_2qudit_adjacent(q, v11301::as_c128(v11301::random_unitary(9, seed++)));
    s.apply_2qudit(0, 4, v11301::as_c128(v11301::random_unitary(9, seed++)));
    return s;
}

}  // namespace

// =============================================================================
// V11301FidelityLedger - the arithmetic
// =============================================================================

TEST(V11301FidelityLedger, AFreshLedgerReadsExact) {
    const FidelityLedger l;
    EXPECT_EQ(l.estimate(), std::optional<double>(1.0));
    EXPECT_EQ(l.lower_bound(), std::optional<double>(1.0));
}

TEST(V11301FidelityLedger, ASplitThatDiscardsNothingChangesNothing) {
    // eps = 0 multiplies the estimate by exactly 1 and adds exactly 0 to the
    // distance, at any block weight, and a block with no weight removes none.
    FidelityLedger l;
    for (double kept : {1.0, 0.5, 1e-300, 1e300, 0.0}) l.record(kept, 0.0);
    EXPECT_EQ(l.estimate(), std::optional<double>(1.0));
    EXPECT_EQ(l.lower_bound(), std::optional<double>(1.0));
}

TEST(V11301FidelityLedger, OneSplitGivesTheSameFigureBothWays) {
    // For one split the floor's overlap is sqrt(1 - eps), so both figures are
    // 1 - eps: a single renormalised projection is exactly that far from the
    // state it came from.
    for (double eps : {1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999}) {
        SCOPED_TRACE("eps=" + std::to_string(eps));
        FidelityLedger l;
        record_fraction(l, eps, 3.0);
        ASSERT_TRUE(l.estimate().has_value());
        ASSERT_TRUE(l.lower_bound().has_value());
        EXPECT_NEAR(*l.estimate(), 1.0 - eps, 4.0 * kEps);
        EXPECT_NEAR(*l.lower_bound(), 1.0 - eps, 16.0 * kEps);
    }
}

TEST(V11301FidelityLedger, TheTwoRotationExampleSeparatesTheFigures) {
    // Two rotations by a, each followed by a truncation back onto |0>: each
    // keeps cos^2 a, so the estimate is cos^4 a, while the true fidelity is
    // cos^2(2a), which is lower for every a in (0, pi/4) since
    // cos^4 a - cos^2(2a) = sin^2 a (3 cos^2 a - 1). Each projection moves the
    // state by 2 sin(a/2), so the floor is (1 - 8 sin^2(a/2))^2 = (4 cos a - 3)^2,
    // which lies below the truth since cos 2a - (4 cos a - 3) = 2 (1 - cos a)^2.
    for (double a : {0.01, 0.1, 0.3, 0.6}) {
        SCOPED_TRACE("a=" + std::to_string(a));
        const double c = std::cos(a), s = std::sin(a);
        FidelityLedger l;
        l.record(c * c, s * s);              // the first block has weight 1
        l.record(c * c * c * c, c * c * s * s);  // the second, what was kept
        const double truth = std::cos(2.0 * a) * std::cos(2.0 * a);
        const double floor = std::pow(std::max(0.0, 4.0 * c - 3.0), 2);

        EXPECT_NEAR(*l.estimate(), c * c * c * c, 8.0 * kEps);
        EXPECT_NEAR(*l.lower_bound(), floor, 64.0 * kEps);
        EXPECT_GT(*l.estimate(), truth) << "the estimate is not a bound";
        EXPECT_LE(*l.lower_bound(), truth + 64.0 * kEps) << "the floor is";
    }
}

TEST(V11301FidelityLedger, RoundingLevelSplitsStillAccumulateDistance) {
    // At eps = 1e-20 the direct form 2 - 2 sqrt(1 - eps) subtracts two numbers
    // that are equal in double and returns 0, so ten thousand such splits
    // would leave the floor at exactly 1. The distance is sqrt(eps) = 1e-10 per
    // split, so the floor is 1 - (N sqrt(eps))^2 = 1 - 1e-12 to leading order,
    // and the estimate, a product of numbers that each round to 1, stays 1.
    constexpr int kSplits = 10000;
    constexpr double kTiny = 1e-20;
    FidelityLedger l;
    for (int i = 0; i < kSplits; ++i) record_fraction(l, kTiny);
    const double expected_gap = static_cast<double>(kSplits) * kSplits * kTiny;
    EXPECT_EQ(*l.estimate(), 1.0);
    EXPECT_NEAR(1.0 - *l.lower_bound(), expected_gap, 4.0 * kEps)
        << "rounding-level splits were dropped from the distance";
}

TEST(V11301FidelityLedger, TheFloorClampsAtZeroAndAFullDiscardZeroesBoth) {
    FidelityLedger many;
    for (int i = 0; i < 10; ++i) record_fraction(many, 0.5);
    EXPECT_EQ(*many.estimate(), std::ldexp(1.0, -10)) << "ten halvings, exactly";
    EXPECT_EQ(*many.lower_bound(), 0.0)
        << "a summed distance past sqrt(2) leaves no overlap to guarantee";

    FidelityLedger all;
    all.record(0.0, 1.0);
    EXPECT_EQ(*all.estimate(), 0.0);
    EXPECT_EQ(*all.lower_bound(), 0.0);
}

TEST(V11301FidelityLedger, FractionsOutsideTheUnitIntervalAreClamped) {
    // Neither can arise from a split, which reports non-negative weights; the
    // ledger still refuses to multiply by more than 1 or less than 0.
    FidelityLedger negative_discard;
    negative_discard.record(1.0, -1e-3);
    EXPECT_EQ(negative_discard.estimate(), std::optional<double>(1.0));
    EXPECT_EQ(negative_discard.lower_bound(), std::optional<double>(1.0));

    FidelityLedger negative_kept;
    negative_kept.record(-0.5, 1.0);
    EXPECT_EQ(*negative_kept.estimate(), 0.0);
    EXPECT_EQ(*negative_kept.lower_bound(), 0.0);
}

TEST(V11301FidelityLedger, TheFiguresAreFractionsSoScaleDoesNotEnter) {
    // eps is a ratio, so scaling a split's two weights together changes
    // nothing, to the bit when the scale is a power of two.
    FidelityLedger a, b;
    for (double eps : {0.01, 0.2, 1e-9}) {
        record_fraction(a, eps, 1.0);
        record_fraction(b, eps, 8.0);
    }
    EXPECT_TRUE(same_opt(a.estimate(), b.estimate()));
    EXPECT_TRUE(same_opt(a.lower_bound(), b.lower_bound()));
}

TEST(V11301FidelityLedger, TheFloorNeverExceedsTheEstimate) {
    // prod (1 - d_k^2/2) >= 1 - sum d_k^2/2 >= 1 - (sum d_k)^2/2, and the
    // estimate is the square of the first, the floor of the last. Checked on
    // every prefix of seeded sequences spanning rounding level to heavy loss,
    // with each figure also held to its closed form.
    std::mt19937_64 rng(98);
    std::uniform_real_distribution<double> log_eps(-18.0, std::log10(0.5));
    std::uniform_int_distribution<int> length(1, 50);
    for (int trial = 0; trial < 200; ++trial) {
        FidelityLedger l;
        std::vector<double> eps;
        const int n = length(rng);
        for (int k = 0; k < n; ++k) {
            const double e = std::pow(10.0, log_eps(rng));
            eps.push_back(e);
            record_fraction(l, e);
            const double tol = 16.0 * static_cast<double>(eps.size()) * kEps;
            ASSERT_LE(*l.lower_bound(), *l.estimate() + tol)
                << "trial " << trial << " after " << eps.size() << " splits";
            ASSERT_NEAR(*l.estimate(), closed_estimate(eps), tol);
            ASSERT_NEAR(*l.lower_bound(), closed_floor(eps), tol);
        }
    }
}

TEST(V11301FidelityLedger, ACollapseEmptiesBothForGoodUntilReset) {
    FidelityLedger l;
    record_fraction(l, 0.1);
    l.invalidate();
    EXPECT_FALSE(l.estimate().has_value());
    EXPECT_FALSE(l.lower_bound().has_value());
    record_fraction(l, 0.2);
    l.record(1.0, 0.0);
    EXPECT_FALSE(l.estimate().has_value()) << "a later split restored the figure";
    EXPECT_FALSE(l.lower_bound().has_value());
    l.reset();
    EXPECT_EQ(l.estimate(), std::optional<double>(1.0));
    EXPECT_EQ(l.lower_bound(), std::optional<double>(1.0));
}

// =============================================================================
// V11301FidelityFigures - what each operation does to them
// =============================================================================

TEST(V11301FidelityFigures, ANewChainReadsExactOnBothLayers) {
    for (int n : {1, 2, 5}) {
        const MPSState q(n);
        EXPECT_EQ(q.fidelity_estimate(), std::optional<double>(1.0));
        EXPECT_EQ(q.fidelity_lower_bound(), std::optional<double>(1.0));
        const QuditMPS d(n, 3);
        EXPECT_EQ(d.fidelity_estimate(), std::optional<double>(1.0));
        EXPECT_EQ(d.fidelity_lower_bound(), std::optional<double>(1.0));
    }
}

TEST(V11301FidelityFigures, CopiesGaugeChangesAndNormalisationKeepThem) {
    // A copy is the same chain with the same history. Moving the centre, and
    // rescaling, change neither the truncated state's direction nor the
    // untruncated one's, and both figures compare normalised states.
    {
        const MPSState base = truncated_qubit_chain();
        ASSERT_LT(*base.fidelity_estimate(), 1.0);
        const MPSState copy = base;
        EXPECT_TRUE(same_opt(copy.fidelity_estimate(), base.fidelity_estimate()));
        EXPECT_TRUE(same_opt(copy.fidelity_lower_bound(), base.fidelity_lower_bound()));

        for (int c = 0; c < base.n_qubits; ++c) {
            MPSState s = base;
            s.canonicalize(c);
            s.normalize();
            EXPECT_TRUE(same_opt(s.fidelity_estimate(), base.fidelity_estimate())) << c;
            EXPECT_TRUE(same_opt(s.fidelity_lower_bound(), base.fidelity_lower_bound())) << c;
        }

        // Handed to a run as its initial state, the chain arrives as a copy.
        auto source = std::make_shared<MPSState>(base);
        RunPlan plan;
        plan.initial = InitialState::from(std::shared_ptr<const MPSState>(source));
        MPSSimulator sim;
        const MPSState out =
            sim.run(QuantumCircuit(base.n_qubits), 2, 0, 1, plan).final_state;
        EXPECT_TRUE(same_opt(out.fidelity_estimate(), base.fidelity_estimate()));
        EXPECT_TRUE(same_opt(out.fidelity_lower_bound(), base.fidelity_lower_bound()));
    }
    {
        const QuditMPS base = truncated_qudit_chain();
        ASSERT_LT(*base.fidelity_estimate(), 1.0);
        for (int c = 0; c < base.n_qudits; ++c) {
            QuditMPS s = base;
            s.canonicalize(c);
            s.normalize();
            s.measure(77);
            EXPECT_TRUE(same_opt(s.fidelity_estimate(), base.fidelity_estimate())) << c;
            EXPECT_TRUE(same_opt(s.fidelity_lower_bound(), base.fidelity_lower_bound())) << c;
        }
    }
}

TEST(V11301FidelityFigures, EveryCollapseEmptiesThemForGood) {
    // Projection renormalises the exact and the truncated state by different
    // factors, so neither figure describes the pair afterwards, and no later
    // split can bring one back.
    std::mt19937_64 rng(4);
    {
        MPSState s = truncated_qubit_chain();
        s.measure_qubit(2, rng);
        EXPECT_FALSE(s.fidelity_estimate().has_value());
        EXPECT_FALSE(s.fidelity_lower_bound().has_value());
        s.apply_two_qubit_gate(v11301::as_gate4(v11301::random_unitary(4, 5)), 0, 5);
        EXPECT_FALSE(s.fidelity_estimate().has_value()) << "a later split restored it";
        EXPECT_FALSE(s.fidelity_lower_bound().has_value());
    }
    {
        MPSState s = truncated_qubit_chain();
        s.measure_sequential(rng);
        EXPECT_FALSE(s.fidelity_estimate().has_value());
        EXPECT_FALSE(s.fidelity_lower_bound().has_value());
    }
    {
        QuditMPS s = truncated_qudit_chain();
        s.measure_qudit(1, rng);
        EXPECT_FALSE(s.fidelity_estimate().has_value());
        s.apply_2qudit(0, 4, v11301::as_c128(v11301::random_unitary(9, 6)));
        EXPECT_FALSE(s.fidelity_estimate().has_value()) << "a later split restored it";
        EXPECT_FALSE(s.fidelity_lower_bound().has_value());
    }

    // In a run: a MEASURE or a RESET in a single trajectory, and every shot of
    // a per-shot run, collapse the chain the run returns.
    const QuantumCircuit gates = v11301::brickwork(6, 3);
    MPSSimulator sim;
    for (const char* which : {"measure", "reset"}) {
        SCOPED_TRACE(which);
        QuantumCircuit qc(6, 6);
        qc.instructions = gates.instructions;
        if (std::string(which) == "measure") qc.measure(3, 3); else qc.reset(3);
        qc.cx(3, 4);
        const auto r = sim.run(qc, 2, 0, 12);
        EXPECT_FALSE(r.final_state.fidelity_estimate().has_value());
        EXPECT_FALSE(r.final_state.fidelity_lower_bound().has_value());
        const auto shots = sim.run(qc, 2, 8, 12);
        EXPECT_FALSE(shots.final_state.fidelity_estimate().has_value());
        EXPECT_FALSE(shots.final_state.fidelity_lower_bound().has_value());
    }
}

TEST(V11301FidelityFigures, TerminalMeasurementsDoNotCollapseTheReturnedChain) {
    // With only terminal measurements the run evolves once without the
    // MEASUREs and samples every shot from that chain, which is returned
    // uncollapsed: its figures are the forward pass's, bit for bit the figures
    // of the same gates run with no measurement at all, on either sampling
    // path (a few shots walk the chain, many contract it).
    const QuantumCircuit gates = v11301::brickwork(6, 3);
    QuantumCircuit measured = gates;
    measured.measure_all();
    MPSSimulator sim;
    const MPSState bare = sim.run(gates, 2, 0, 3).final_state;
    ASSERT_LT(*bare.fidelity_estimate(), 1.0);
    for (int shots : {1, 100000}) {
        SCOPED_TRACE("shots=" + std::to_string(shots));
        const MPSState out = sim.run(measured, 2, shots, 3).final_state;
        EXPECT_TRUE(same_opt(out.fidelity_estimate(), bare.fidelity_estimate()));
        EXPECT_TRUE(same_opt(out.fidelity_lower_bound(), bare.fidelity_lower_bound()));
    }
}

TEST(V11301FidelityFigures, AbsorbProfileDoesNotFoldThem) {
    // absorb_profile folds another chain's split tallies into this one; the
    // figures describe these tensors' own history and stay as they were.
    MPSState a = truncated_qubit_chain();
    MPSState b = truncated_qubit_chain();
    b.apply_two_qubit_gate(v11301::as_gate4(v11301::random_unitary(4, 8)), 1, 4);
    const auto est = a.fidelity_estimate();
    const auto low = a.fidelity_lower_bound();
    const std::size_t calls = a.svd_call_count() + b.svd_call_count();
    const double discarded = a.truncation_error() + b.truncation_error();

    a.absorb_profile(b);
    EXPECT_TRUE(same_opt(a.fidelity_estimate(), est));
    EXPECT_TRUE(same_opt(a.fidelity_lower_bound(), low));
    EXPECT_EQ(a.svd_call_count(), calls);
    EXPECT_EQ(a.truncation_error(), discarded);
}

TEST(V11301FidelityFigures, ARebuildExtendsTheFiguresItFinds) {
    // rebuild_from_statevector's splits count like any other, onto what the
    // chain already carries: the estimate multiplies by the sweep's own and
    // the floor's distance adds. The sweep's own figures are those of a fresh
    // chain rebuilt from the same amplitudes with the same settings, which
    // performs the same splits.
    const int n = 6;
    const int cap = 4;
    MPSState s = truncated_qubit_chain(cap);
    ASSERT_EQ(s.max_bond_dim, cap);
    const double est_before = *s.fidelity_estimate();
    const double low_before = *s.fidelity_lower_bound();
    const double disc_before = s.truncation_error();
    const std::size_t calls_before = s.svd_call_count();

    const auto sites = v11301::random_chain(n, {2, 4, 8, 4, 2}, 99);
    MPSState target(n, 64);
    target.set_tensors(sites);
    target.normalize();
    const Statevector sv = target.to_statevector();

    MPSState fresh(n, cap);
    fresh.rebuild_from_statevector(sv);
    ASSERT_LT(*fresh.fidelity_estimate(), 1.0) << "the rebuild did not truncate";

    s.rebuild_from_statevector(sv);
    const double tol = kSlack * static_cast<double>(2 * n) * kEps;
    EXPECT_NEAR(*s.fidelity_estimate(), est_before * *fresh.fidelity_estimate(), tol);
    // The floor's distance is additive, so recover each Delta from its floor.
    // The runs truncate heavily, so each floor is well away from 1 and the
    // inversion is well conditioned.
    // Inverting the floor needs it strictly between the clamp at 0 and 1.
    const auto delta_of = [](double floor) {
        return std::sqrt(2.0 * (1.0 - std::sqrt(floor)));
    };
    ASSERT_GT(low_before, 0.0) << "the fixture truncates too hard to invert its floor";
    ASSERT_GT(*fresh.fidelity_lower_bound(), 0.0);
    ASSERT_LT(low_before, 1.0 - 1e-6);
    ASSERT_LT(*fresh.fidelity_lower_bound(), 1.0 - 1e-6);
    const double delta = delta_of(low_before) + delta_of(*fresh.fidelity_lower_bound());
    const double expected_low = std::pow(std::max(0.0, 1.0 - 0.5 * delta * delta), 2);
    EXPECT_NEAR(*s.fidelity_lower_bound(), expected_low, std::sqrt(tol));
    EXPECT_NEAR(s.truncation_error(), disc_before + fresh.truncation_error(), tol);
    EXPECT_EQ(s.svd_call_count(), calls_before + static_cast<std::size_t>(n - 1));
}

TEST(V11301FidelityFigures, SeedingARunFromAmplitudesStartsFromExact) {
    // A run seeded from a statevector factorises it into a fresh chain, so the
    // figures it returns describe the seeding and nothing before it.
    const int n = 6;
    const int cap = 2;
    MPSState target(n, 64);
    target.set_tensors(v11301::random_chain(n, {2, 4, 8, 4, 2}, 100));
    target.normalize();
    auto sv = std::make_shared<Statevector>(target.to_statevector());

    MPSState fresh(n, cap);
    fresh.rebuild_from_statevector(*sv);

    RunPlan plan;
    plan.initial = InitialState::from(std::shared_ptr<const Statevector>(sv));
    MPSSimulator sim;
    const MPSState out = sim.run(QuantumCircuit(n), cap, 0, 1, plan).final_state;
    EXPECT_TRUE(same_opt(out.fidelity_estimate(), fresh.fidelity_estimate()));
    EXPECT_TRUE(same_opt(out.fidelity_lower_bound(), fresh.fidelity_lower_bound()));
    EXPECT_EQ(out.truncation_error(), fresh.truncation_error());
}

// =============================================================================
// V11301FidelityBound - against the truth
// =============================================================================

TEST(V11301FidelityBound, TheEstimateLiesOnEitherSideOfTheTruth) {
    // The two-rotation example built on a chain. A two-qubit chain at a cap of
    // 1 rotated by a inside span{|00>, |11>} keeps cos a of the amplitude and
    // drops the rest; rotating by a again keeps cos^2 a of the weight again.
    // The untruncated state is the rotation by 2a, so the truth is
    // cos^2(2a) and the estimate cos^4 a lies above it. Rotating back by -a
    // instead, the untruncated state returns to |00>, which the chain also
    // holds, so the truth is 1 and the same estimate lies below it. The floor
    // is (4 cos a - 3)^2 in both, and below the truth in both.
    const double a = 0.3;
    const double c = std::cos(a);
    const double floor = (4.0 * c - 3.0) * (4.0 * c - 3.0);
    const double tol = kSlack * 4.0 * kEps;

    struct Case {
        const char* name;
        double second;  // the second rotation
        double truth;
    };
    for (const Case& k : {Case{"rotate twice", a, std::pow(std::cos(2.0 * a), 2)},
                          Case{"rotate and undo", -a, 1.0}}) {
        SCOPED_TRACE(k.name);
        MPSState s(2, 1);
        s.apply_two_qubit_gate(pair_rotation(a), 0, 1);
        s.apply_two_qubit_gate(pair_rotation(k.second), 0, 1);
        ASSERT_EQ(s.svd_call_count(), 2u);
        ASSERT_EQ(s.current_max_bond_dim(), 1);

        // The truth, from the chain's own amplitudes against the untruncated
        // state computed densely.
        const Amplitudes got = v11301::dense(s);
        Amplitudes exact(4, v11301::Cplx(0.0, 0.0));
        exact[0] = std::cos(a + k.second);
        exact[3] = std::sin(a + k.second);
        const double truth = v11301::fidelity(got, exact);
        EXPECT_NEAR(truth, k.truth, tol) << "the fixture is not the example";

        EXPECT_NEAR(*s.fidelity_estimate(), c * c * c * c, tol);
        EXPECT_NEAR(*s.fidelity_lower_bound(), floor, tol);
        EXPECT_LE(*s.fidelity_lower_bound(), truth + tol);
        if (k.second > 0.0) {
            EXPECT_GT(*s.fidelity_estimate(), truth) << "the estimate is not a bound";
        } else {
            EXPECT_LT(*s.fidelity_estimate(), truth);
        }
    }
}

TEST(V11301FidelityBound, TheFloorHoldsOnBindingQubitRuns) {
    // Seeded brickwork at caps that bind, under both policies: the true
    // fidelity between the chain and the dense state, both normalised, never
    // falls below the floor. The estimate is held to [floor, 1] and must fall
    // below 1, since each run truncates. truncation_error() is the weight the
    // state lost, so with no collapse and no normalisation it equals how far
    // norm_sq() fell from the unit norm the run started at.
    for (const auto& [n, layers] : std::vector<std::pair<int, int>>{{8, 6}, {10, 5}, {12, 4}}) {
        const QuantumCircuit qc = v11301::brickwork(n, layers);
        StatevectorSimulator svsim;
        const Amplitudes exact = v11301::dense(svsim.run(qc).final_state);
        const std::size_t dim = exact.size();
        for (int cap : {2, 4, 8}) {
            for (CanonicalForm form : {CanonicalForm::Always, CanonicalForm::Auto}) {
                SCOPED_TRACE("n=" + std::to_string(n) + " layers=" + std::to_string(layers) +
                             " cap=" + std::to_string(cap) + " " + to_string(form));
                MPSSimulator sim;
                sim.canonical_form = form;
                const MPSState s = sim.run(qc, cap, 0, 21).final_state;
                const double truth = v11301::fidelity(v11301::dense(s), exact);
                const double tol =
                    kSlack * static_cast<double>(qc.instructions.size() + dim) * kEps;
                ASSERT_TRUE(s.fidelity_lower_bound().has_value());
                ASSERT_TRUE(s.fidelity_estimate().has_value());
                std::cout << "[v11301-bound] n=" << n << " cap=" << cap << " "
                          << to_string(form) << " truth=" << truth
                          << " estimate=" << *s.fidelity_estimate()
                          << " floor=" << *s.fidelity_lower_bound() << "\n";
                EXPECT_LE(*s.fidelity_lower_bound(), truth + tol)
                    << "the floor is not a bound";
                EXPECT_LE(*s.fidelity_lower_bound(), *s.fidelity_estimate() + tol);
                EXPECT_LE(*s.fidelity_estimate(), 1.0);
                EXPECT_LT(*s.fidelity_estimate(), 1.0) << "the cap did not bind";
                EXPECT_GT(s.truncation_error(), 0.0);

                const double norm_tol =
                    kSlack * static_cast<double>(s.svd_call_count() +
                                                 qc.instructions.size() * n) * kEps;
                EXPECT_NEAR(1.0 - s.norm_sq(), s.truncation_error(), norm_tol)
                    << "truncation_error() is not the weight the state lost";
                // From unit norm each canonical split removes eps_k of what is
                // left, so the weights telescope: sum_k w_k = 1 - prod (1 - eps_k).
                EXPECT_NEAR(s.truncation_error(), 1.0 - *s.fidelity_estimate(), norm_tol)
                    << "the discarded weight and the retained product disagree";
            }
        }
    }
}

TEST(V11301FidelityBound, TheFloorHoldsOnBindingQuditRuns) {
    // The same on the qudit layer: a seeded chain of random d^2 x d^2 gates on
    // adjacent and swap-routed pairs, at caps below d^(n/2).
    const int n = 6, d = 3;
    std::vector<std::pair<int, int>> pairs;
    std::vector<std::vector<Complex128>> gates;
    std::uint64_t seed = 2100;
    for (int layer = 0; layer < 4; ++layer) {
        for (int q = layer % 2; q + 1 < n; q += 2) {
            pairs.push_back({q, q + 1});
            gates.push_back(v11301::as_c128(v11301::random_unitary(d * d, seed++)));
        }
        pairs.push_back({layer % n, (layer + 3) % n});
        gates.push_back(v11301::as_c128(v11301::random_unitary(d * d, seed++)));
    }
    QuditStatevector dense_sv(n, d);
    for (std::size_t i = 0; i < pairs.size(); ++i)
        dense_sv.apply_2qudit(pairs[i].first, pairs[i].second, gates[i]);
    const Amplitudes exact = v11301::dense(dense_sv);

    for (int cap : {2, 3, 6}) {
        for (CanonicalForm form : {CanonicalForm::Always, CanonicalForm::Auto}) {
            SCOPED_TRACE("cap=" + std::to_string(cap) + " " + to_string(form));
            QuditMPS s(n, d, cap);
            s.canonical_form = form;
            for (std::size_t i = 0; i < pairs.size(); ++i)
                s.apply_2qudit(pairs[i].first, pairs[i].second, gates[i]);
            const double truth = v11301::fidelity(v11301::dense(s), exact);
            const double tol =
                kSlack * static_cast<double>(pairs.size() + exact.size()) * kEps;
            ASSERT_TRUE(s.fidelity_lower_bound().has_value());
            EXPECT_LE(*s.fidelity_lower_bound(), truth + tol) << "the floor is not a bound";
            EXPECT_LE(*s.fidelity_lower_bound(), *s.fidelity_estimate() + tol);
            EXPECT_LT(*s.fidelity_estimate(), 1.0) << "the cap did not bind";

            const double norm_tol =
                kSlack * static_cast<double>(s.svd_call_count() + pairs.size() * n) * kEps;
            EXPECT_NEAR(1.0 - s.norm_sq(), s.truncation_error(), norm_tol)
                << "truncation_error() is not the weight the state lost";
            EXPECT_NEAR(s.truncation_error(), 1.0 - *s.fidelity_estimate(), norm_tol)
                << "the discarded weight and the retained product disagree";
        }
    }
}
