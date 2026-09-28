// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.2 - every sampler and collapse draws from the state's own normalised
// distribution.
//
// The MPS chains already drew each outcome with probability weight / total. The
// dense samplers assumed the total was 1: a state short of unit norm had the
// weight it lacked pushed onto the last outcome, and the dense collapses took
// the second half's weight as 1 - p0, so a short state could collapse onto an
// outcome of weight zero and be left with no norm. The chains, for their part,
// treated a total under a hand-typed floor of 1e-30 as a degenerate marginal
// (an even split, digit 0, no renormalisation), which after the refusal of
// normless states reached only states normalize() accepts.
//
// Every draw now goes through one rule, detail::born_draw_* on the dense layer
// and u < p0 / total on the qubit chain: the first outcome whose running sum
// strictly exceeds u * total, which can never be an outcome of weight zero. A
// state with no norm is refused before the draw, on the same threshold as
// normalize(). Each check here is exact: the states are deterministic, so the
// old rules fail them outright rather than statistically.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/born_draw.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kSlack = 64.0;

using Counts = std::unordered_map<std::string, int>;

// The key of a register index with qubit (or clbit) 0 as its rightmost
// character.
std::string key_of(std::size_t index, int width) {
    std::string key(static_cast<std::size_t>(width), '0');
    for (int b = 0; b < width; ++b)
        if ((index >> b) & 1u) key[static_cast<std::size_t>(width - 1 - b)] = '1';
    return key;
}

std::vector<double> running(const std::vector<double>& w) {
    std::vector<double> cum(w.size());
    std::partial_sum(w.begin(), w.end(), cum.begin());
    return cum;
}

std::array<Complex128, 4> scale2(double t) {
    return {Complex128(t, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(t, 0.0)};
}

std::array<Complex128, 4> pauli_x() {
    return {Complex128(0.0, 0.0), Complex128(1.0, 0.0),
            Complex128(1.0, 0.0), Complex128(0.0, 0.0)};
}

// The digit shift |x> -> |x + k mod d>.
std::vector<Complex128> shift(int d, int k) {
    std::vector<Complex128> m(static_cast<std::size_t>(d * d), Complex128(0.0, 0.0));
    for (int x = 0; x < d; ++x)
        m[static_cast<std::size_t>(((x + k) % d) * d + x)] = Complex128(1.0, 0.0);
    return m;
}

std::vector<Complex128> scale_d(int d, double t) {
    std::vector<Complex128> m(static_cast<std::size_t>(d * d), Complex128(0.0, 0.0));
    for (int x = 0; x < d; ++x) m[static_cast<std::size_t>(x * d + x)] = Complex128(t, 0.0);
    return m;
}

}  // namespace

// =============================================================================
// The draw itself
// =============================================================================

TEST(V11302BornDraw, TheDrawNeverLandsOnAnOutcomeOfWeightZero) {
    // Zero weights at the front, between and behind two positive ones.
    const std::vector<double> w = {0.0, 0.0, 0.25, 0.0, 0.75, 0.0, 0.0};
    const std::vector<double> cum = running(w);
    const auto weight = [&w](std::size_t s) { return w[s]; };
    const double total = cum.back();
    const std::size_t first = 2, last = 4;

    // A draw of exactly 0 is the one a non-strict comparison would stop on the
    // front zeros for.
    EXPECT_EQ(detail::born_draw_cumulative(cum, 0.0), first);
    EXPECT_EQ(detail::born_draw_linear(w.size(), total, 0.0, weight), first);

    // The largest uniform below 1: whatever the product rounds to, the draw
    // lands on the last positive weight and never on the zeros behind it.
    const double top = std::nextafter(1.0, 0.0);
    EXPECT_EQ(detail::born_draw_cumulative(cum, top), last);
    EXPECT_EQ(detail::born_draw_linear(w.size(), total, top, weight), last);

    // A walk whose running sum ends a rounding step short of the total it was
    // handed (a caller summing the same weights with a different contraction)
    // leaves a top draw above every running sum; it falls back to the last
    // positive weight rather than past the end.
    const double nudged = std::nextafter(total, 2.0 * total);
    EXPECT_EQ(detail::born_draw_linear(w.size(), nudged, top, weight), last);

    // A draw exactly on a boundary belongs to the outcome above it, as the
    // chain's u < p0 / total gives a draw equal to p0 / total to outcome 1.
    const double boundary = w[first] / total;
    EXPECT_EQ(detail::born_draw_cumulative(cum, boundary), last);
    EXPECT_EQ(detail::born_draw_linear(w.size(), total, boundary, weight), last);
}

TEST(V11302BornDraw, TheDrawIsScaledByTheWeightsOwnTotal) {
    // Weights summing to 1/2. Normalised, outcome 1 owns [0, 1/4) and outcome 3
    // owns [1/4, 1); a draw that assumed a total of 1 would send every draw
    // above 1/2 past the end.
    const std::vector<double> w = {0.0, 0.125, 0.0, 0.375};
    const std::vector<double> cum = running(w);
    const auto weight = [&w](std::size_t s) { return w[s]; };
    const double total = cum.back();
    const double split = w[1] / total;
    for (const double u : {split / 2.0, (split + 1.0) / 2.0}) {
        SCOPED_TRACE("u=" + std::to_string(u));
        const std::size_t want = (u < split) ? 1u : 3u;
        EXPECT_EQ(detail::born_draw_cumulative(cum, u), want);
        EXPECT_EQ(detail::born_draw_linear(w.size(), total, u, weight), want);
    }
}

// =============================================================================
// Dense states short of unit norm
// =============================================================================

TEST(V11302ShortStates, AStatevectorIsSampledAsItsOwnDistribution) {
    // Amplitude 1/2 on |101>, nothing elsewhere: norm² 1/4, one outcome.
    const int n = 3;
    const std::size_t index = 5;
    std::vector<Complex128> amps(std::size_t{1} << n, Complex128(0.0, 0.0));
    amps[index] = Complex128(0.5, 0.0);
    Statevector sv(n);
    sv.set_amplitudes(amps, {Validation::Ignore});

    const int shots = 64;
    const Counts counts = sv.sample_counts(shots, 21);
    ASSERT_EQ(counts.size(), 1u) << "weight landed off the state's only outcome";
    EXPECT_EQ(counts.begin()->first, key_of(index, n));
    EXPECT_EQ(counts.begin()->second, shots);
    for (std::uint64_t seed = 1; seed <= 8; ++seed)
        EXPECT_EQ(sv.measure_once(seed), key_of(index, n)) << "seed " << seed;
}

namespace {

// |101>, then the caller's (1/2) I on qubit 1 under Ignore: a state of norm²
// 1/4 whose every outcome is fixed. `mid` measures qubit 1 part way and flips
// it afterwards, which puts the run on the per-shot path; clbit 1 still reads
// the collapse's 0.
QuantumCircuit short_state_circuit(bool mid) {
    QuantumCircuit qc(3, 3);
    qc.x(0).x(2);
    const auto half = scale2(0.5);
    qc.unitary(std::vector<Complex128>(half.begin(), half.end()), {1}, "half",
               {Validation::Ignore});
    if (mid) {
        qc.measure(1, 1).x(1);
        qc.measure(0, 0).measure(2, 2);
    } else {
        qc.measure_all();
    }
    return qc;
}

}  // namespace

TEST(V11302ShortStates, TheDenseSimulatorsSampleAndCollapseAShortState) {
    const std::string key = key_of(5, 3);
    const int shots = 64;
    for (const bool mid : {false, true}) {
        SCOPED_TRACE(mid ? "per-shot collapse" : "terminal sampling");
        StatevectorSimulator sv;
        const auto sr = sv.run(short_state_circuit(mid), shots, 23);
        ASSERT_TRUE(sr.success) << sr.error_message;
        ASSERT_EQ(sr.counts.size(), 1u);
        EXPECT_EQ(sr.counts.begin()->first, key);
        EXPECT_EQ(sr.counts.begin()->second, shots);

        DensityMatrixSimulator dm;
        const auto dr = dm.run(short_state_circuit(mid), NoiseModel{}, shots, 23);
        ASSERT_TRUE(dr.success) << dr.error_message;
        ASSERT_EQ(dr.counts.size(), 1u);
        EXPECT_EQ(dr.counts.begin()->first, key);
        EXPECT_EQ(dr.counts.begin()->second, shots);
    }

    // The collapse renormalises by the drawn half's own weight, so the last
    // trajectory ends as |111> at unit norm.
    StatevectorSimulator sv;
    const auto sr = sv.run(short_state_circuit(true), 1, 29);
    ASSERT_TRUE(sr.success) << sr.error_message;
    EXPECT_NEAR(sr.final_state.amplitudes()[7].norm_sq(), 1.0, kSlack * kEps);
}

TEST(V11302ShortStates, TheQuditDenseClassesSampleAShortState) {
    // Digits (2, 1) at d = 3: flat index 2 + 3 * 1 = 5, weight 1/4.
    const int n = 2, d = 3;
    const std::vector<int> digits = {2, 1};
    const std::size_t index = 5;

    QuditStatevector sv(n, d);
    std::fill(sv.amplitudes.begin(), sv.amplitudes.end(), Complex128(0.0, 0.0));
    sv.amplitudes[index] = Complex128(0.5, 0.0);
    for (std::uint64_t seed = 1; seed <= 8; ++seed)
        EXPECT_EQ(sv.measure(seed), digits) << "seed " << seed;

    for (std::uint64_t seed = 1; seed <= 8; ++seed) {
        QuditDensityMatrix rho(n, d);
        std::fill(rho.rho.begin(), rho.rho.end(), Complex128(0.0, 0.0));
        rho.rho[index * rho.dim + index] = Complex128(0.25, 0.0);
        EXPECT_EQ(rho.measure(seed), digits) << "seed " << seed;
        EXPECT_EQ(rho.rho[index * rho.dim + index].real, 1.0);
    }
}

// =============================================================================
// Chains with a tiny norm, and where the refusal sits
// =============================================================================

namespace {

// |q2 q1 q0> = |110>, then site 0 (the centre) scaled by t under Ignore: a
// chain of norm t whose every outcome is fixed.
MPSState tiny_qubit_chain(double t) {
    MPSState s(3, 8);
    s.apply_single_qubit_gate(pauli_x(), 1);
    s.apply_single_qubit_gate(pauli_x(), 2);
    s.apply_single_qubit_gate(scale2(t), 0, {Validation::Ignore});
    return s;
}

// Digits (1, 2, 0) at d = 3, site 0 scaled by t under Ignore.
QuditMPS tiny_qudit_chain(double t) {
    const int d = 3;
    QuditMPS s(3, d, 9);
    s.apply_1qudit(0, shift(d, 1));
    s.apply_1qudit(1, shift(d, 2));
    s.apply_1qudit(0, scale_d(d, t), {Validation::Ignore});
    return s;
}

// A norm normalize() accepts but a floor of 1e-30 on the SQUARED norm would
// have called degenerate: 4 eps, squared 16 eps² ~ 7.9e-31.
constexpr double kTinyNorm = 4.0 * kEps;
// A norm normalize() refuses: at or below eps.
constexpr double kNoNorm = kEps / 2.0;

}  // namespace

TEST(V11302TinyNorm, AChainOfTinyNormIsMeasuredAsItsOwnState) {
    const std::string key = key_of(6, 3);
    for (std::uint64_t seed = 1; seed <= 8; ++seed) {
        SCOPED_TRACE("seed " + std::to_string(seed));
        MPSState a = tiny_qubit_chain(kTinyNorm);
        std::mt19937_64 ra(seed);
        EXPECT_EQ(a.measure_qubit(0, ra), 0) << "qubit 0 holds |0> with certainty";
        MPSState b = tiny_qubit_chain(kTinyNorm);
        std::mt19937_64 rb(seed);
        EXPECT_EQ(b.measure_sequential(rb), key);

        QuditMPS q = tiny_qudit_chain(kTinyNorm);
        EXPECT_EQ(q.measure(seed), (std::vector<int>{1, 2, 0}));
        std::mt19937_64 rq(seed);
        EXPECT_EQ(q.measure_qudit(0, rq), 1);
    }

    // A run seeded with the chain samples the one outcome on either path.
    auto source = std::make_shared<MPSState>(tiny_qubit_chain(kTinyNorm));
    RunPlan plan;
    plan.initial = InitialState::from(std::shared_ptr<const MPSState>(source));
    QuantumCircuit qc(3);
    qc.measure_all();
    MPSSimulator sim;
    for (const int shots : {1, 1 << 12}) {
        SCOPED_TRACE("shots " + std::to_string(shots));
        const auto r = sim.run(qc, 8, shots, 31, plan);
        ASSERT_EQ(r.counts.size(), 1u);
        EXPECT_EQ(r.counts.begin()->first, key);
        EXPECT_EQ(r.counts.begin()->second, shots);
    }
}

TEST(V11302TinyNorm, EveryDrawRefusesExactlyWhereNormalizeRefuses) {
    for (const double t : {kTinyNorm, kNoNorm}) {
        SCOPED_TRACE("norm " + std::to_string(t / kEps) + " eps");
        const bool has_norm = t > kEps;

        MPSState ref = tiny_qubit_chain(t);
        if (has_norm) {
            EXPECT_NO_THROW(ref.normalize());
        } else {
            EXPECT_THROW(ref.normalize(), std::runtime_error);
        }

        MPSState a = tiny_qubit_chain(t);
        std::mt19937_64 ra(3);
        MPSState b = tiny_qubit_chain(t);
        std::mt19937_64 rb(3);
        QuditMPS q = tiny_qudit_chain(t);
        std::mt19937_64 rq(3);
        std::vector<Complex128> amps(8, Complex128(0.0, 0.0));
        amps[6] = Complex128(t, 0.0);
        Statevector sv(3);
        sv.set_amplitudes(amps, {Validation::Ignore});
        if (has_norm) {
            EXPECT_NO_THROW(a.measure_qubit(0, ra));
            EXPECT_NO_THROW(b.measure_sequential(rb));
            EXPECT_NO_THROW(q.measure_qudit(0, rq));
            EXPECT_NO_THROW(sv.sample_counts(4, 3));
        } else {
            EXPECT_THROW(a.measure_qubit(0, ra), std::runtime_error);
            EXPECT_THROW(b.measure_sequential(rb), std::runtime_error);
            EXPECT_THROW(q.measure_qudit(0, rq), std::runtime_error);
            EXPECT_THROW(sv.sample_counts(4, 3), std::runtime_error);
        }
    }
}
