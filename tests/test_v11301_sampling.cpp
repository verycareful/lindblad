// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.1 test wave - measurement and sampling at the centre.
//
// 1.1.30.0 moved every collapse and every sample onto the orthogonality centre.
// measure_qubit and measure_qudit collapse one site; measure_sequential walks
// the centre along the chain; terminal sampling in MPSSimulator either walks a
// vector along a chain centred on qubit 0 or contracts the chain once into
// amplitudes, choosing by a documented cost rule. This file checks each
// against the dense state it must agree with.
//
// Two instruments make the checks exact rather than statistical wherever that
// is possible.
//
//   - A draw is replayed. Each collapse draws its uniforms from the caller's
//     engine, so a copy of the engine yields the same numbers, and the outcome
//     the dense marginals predict for them must be the outcome returned, and
//     the engine must have advanced by exactly the draws documented. A draw
//     that lands within rounding of a threshold decides nothing and is skipped.
//
//   - The path is observable. The sampler moves the returned chain's centre to
//     qubit 0 and the dense path leaves it where the gates put it, so a circuit
//     whose last two-qubit gate leaves the centre on the last qubit reports,
//     through open_span(), which path ran. The cost rule is restated here as
//     docs/api/simulators.md documents it, and pinned on both sides of its flip
//     and of its cache ceiling.
//
// Where the check must be statistical (both paths sample the chain's own
// distribution), the bound is derived: E|p_hat - p| <= sqrt(p (1 - p) / N) per
// outcome gives the expected total variation distance, and McDiarmid's
// inequality, since one shot moves the TVD by at most 1/N, adds
// sqrt(ln(1/delta) / (2N)) for a failure probability delta per seeded run.

#include <gtest/gtest.h>

#include "v11301_mps_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "v11311_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace lindblad;
using v11301::Amplitudes;
using v11301::Cplx;
using v11301::kEps;
using v11301::kSlack;

namespace {

using Span = std::pair<int, int>;
using Counts = std::unordered_map<std::string, int>;

// The rule as docs/api/simulators.md ("Choosing the Sampling Path") states it,
// restated here because the rule is what is under test: the dense path when
// its contraction costs no more than `shots` walks, one unit of walk weighted
// at this many multiply-accumulates, and only while the 2^n amplitudes fit in
// one last-level cache instance, with this standing in when detection reports
// none. The dense route stops at kDocumentedDenseLimit qubits in any case.
constexpr double kDocumentedSamplerCostRatio = 1.8;
constexpr std::size_t kDocumentedLlcFallback = std::size_t(4) << 20;
constexpr int kDocumentedDenseLimit = 25;

// A seeded run's chance of failing a statistical bound when the library is
// right.
constexpr double kFailureProbability = 1e-9;

int dense_ceiling() {
    std::size_t llc = hw::llc_bytes();
    if (llc == 0) llc = kDocumentedLlcFallback;
    int n = 0;
    while (n < kDocumentedDenseLimit && (sizeof(Complex128) << (n + 1)) <= llc) ++n;
    return n;
}

struct PathCost {
    double dense = 0.0;  // sum_q 2^(q+1) chi_L chi_R
    double walk = 0.0;   // sum_q chi_L chi_R
};

PathCost cost_of(const MPSState& s) {
    PathCost c;
    const auto& t = s.tensors();
    for (int q = 0; q < s.n_qubits; ++q) {
        const double w = static_cast<double>(t[static_cast<std::size_t>(q)].bond_left) *
                         t[static_cast<std::size_t>(q)].bond_right;
        c.walk += w;
        c.dense += std::ldexp(w, q + 1);
    }
    return c;
}

// The fewest shots for which the rule prefers the dense path.
int first_dense_shots(const PathCost& c) {
    const auto dense_wins = [&](int shots) {
        return c.dense <= kDocumentedSamplerCostRatio * static_cast<double>(shots) * c.walk;
    };
    int s = std::max(1, static_cast<int>(c.dense / (kDocumentedSamplerCostRatio * c.walk)));
    while (!dense_wins(s)) ++s;
    while (s > 1 && dense_wins(s - 1)) --s;
    return s;
}

// Index of a bitstring key with qubit (or clbit) 0 as its rightmost character.
std::size_t index_of(const std::string& key) {
    std::size_t i = 0;
    for (char ch : key) i = (i << 1) | static_cast<std::size_t>(ch == '1');
    return i;
}

std::string key_of(std::size_t index, int width) {
    std::string key(static_cast<std::size_t>(width), '0');
    for (int b = 0; b < width; ++b)
        if ((index >> b) & 1u) key[static_cast<std::size_t>(width - 1 - b)] = '1';
    return key;
}

// The normalised distribution of a state.
std::vector<double> distribution(const Amplitudes& a) {
    const double total = v11301::norm_sq(a);
    std::vector<double> p(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) p[i] = std::norm(a[i]) / total;
    return p;
}

double tvd(const Counts& counts, const std::vector<double>& p) {
    double shots = 0.0;
    for (const auto& kv : counts) shots += kv.second;
    std::vector<bool> seen(p.size(), false);
    double sum = 0.0;
    for (const auto& [key, n] : counts) {
        const std::size_t i = index_of(key);
        seen[i] = true;
        sum += std::abs(static_cast<double>(n) / shots - p[i]);
    }
    for (std::size_t i = 0; i < p.size(); ++i)
        if (!seen[i]) sum += p[i];
    return 0.5 * sum;
}

double tvd_bound(const std::vector<double>& p, int shots) {
    const double N = static_cast<double>(shots);
    double expected = 0.0;
    for (double x : p) expected += std::sqrt(std::max(0.0, x * (1.0 - x)) / N);
    return 0.5 * expected + std::sqrt(std::log(1.0 / kFailureProbability) / (2.0 * N));
}

// Random u3 and cx on qubits [first, first + width), then a final cx on the
// last pair of the register so the centre ends on the last qubit.
void entangle_block(QuantumCircuit& qc, int first, int width, int layers,
                    std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(0.0, TWO_PI);
    const auto u3 = [&](int q) {
        const double theta = angle(rng);
        const double phi = angle(rng);
        const double lam = angle(rng);
        qc.u3(theta, phi, lam, q);
    };
    for (int layer = 0; layer < layers; ++layer) {
        for (int a = first + layer % 2; a + 1 < first + width; a += 2) {
            u3(a);
            u3(a + 1);
            qc.cx(a, a + 1);
            u3(a);
            u3(a + 1);
        }
    }
    qc.cx(qc.n_qubits - 2, qc.n_qubits - 1);
}

// A fixed basis pattern on qubits below `first`, a random block from `first`
// to the end: the distribution lives on at most 2^width outcomes, which keeps
// the statistics tight, while the register is wide enough that the sampler is
// chosen at tens of thousands of shots.
QuantumCircuit fixed_then_block(int n, int width, int layers, std::uint64_t seed) {
    QuantumCircuit qc(n);
    for (int q = 0; q < n - width; ++q)
        if (q % 3 == 1) qc.x(q);
    entangle_block(qc, n - width, width, layers, seed);
    return qc;
}

// The forward chain of a measurement-free circuit, which is the chain every
// shot of a terminal run samples.
MPSState forward_chain(const QuantumCircuit& qc, int cap) {
    MPSSimulator sim;
    return sim.run(qc, cap, 0, 1).final_state;
}

}  // namespace

// =============================================================================
// V11301SamplingPath - which path runs
// =============================================================================

TEST(V11301SamplingPath, TheChoiceFlipsAtTheDocumentedCostRatio) {
    // A product state keeps every bond at 1, so the costs are exact integers:
    // dense = 2^(n+1) - 2 and walk = n, and the flip sits far from rounding.
    const int n = 12;
    ASSERT_GE(dense_ceiling(), n);
    QuantumCircuit qc(n);
    for (int q = 0; q < n; ++q) qc.ry(PI / (q + 3.0), q);
    qc.swap(n - 2, n - 1);

    const MPSState fwd = forward_chain(qc, 8);
    ASSERT_EQ(fwd.open_span(), (Span{n - 1, n - 1})) << "the probe needs the centre off 0";
    ASSERT_EQ(fwd.current_max_bond_dim(), 1);
    const PathCost c = cost_of(fwd);
    ASSERT_EQ(c.dense, std::ldexp(1.0, n + 1) - 2.0);
    ASSERT_EQ(c.walk, static_cast<double>(n));
    const int flip = first_dense_shots(c);

    MPSSimulator sim;
    constexpr std::uint64_t kSeed = 29;
    const auto below = sim.run(qc, 8, flip - 1, kSeed);
    EXPECT_EQ(below.final_state.open_span(), (Span{0, 0}))
        << (flip - 1) << " shots cost less walked than contracted, so the sampler "
           "should have run and recentred the chain";
    int total = 0;
    for (const auto& kv : below.counts) total += kv.second;
    EXPECT_EQ(total, flip - 1);

    const auto at = sim.run(qc, 8, flip, kSeed);
    EXPECT_EQ(at.final_state.open_span(), (Span{n - 1, n - 1}))
        << flip << " shots make the contraction the cheaper path";
    EXPECT_EQ(at.counts, at.final_state.to_statevector().sample_counts(flip, kSeed))
        << "the dense path draws from the amplitudes with the run's seed";
}

TEST(V11301SamplingPath, TheDensePathStopsAtOneLastLevelCacheInstance) {
    // At the ceiling the rule alone decides and a shot count past the flip
    // goes dense. One qubit wider the amplitudes no longer fit, and the
    // sampler runs whatever the shots. A basis state keeps the bonds at 1 and
    // the outcome single, so the shot counts cost little either way.
    const int ceiling = dense_ceiling();
    ASSERT_GE(ceiling, 2) << "hw::llc_bytes() reported under 64 bytes";
    for (int n : {ceiling, ceiling + 1}) {
        SCOPED_TRACE("n=" + std::to_string(n) + ", ceiling " + std::to_string(ceiling));
        QuantumCircuit qc(n);
        std::size_t index = 0;
        for (int q = 0; q < n; ++q)
            if (q % 3 == 1) {
                qc.x(q);
                index |= std::size_t{1} << q;
            }
        qc.swap(n - 2, n - 1);
        // The swap exchanges the last two digits of the pattern.
        const std::size_t hi = (index >> (n - 1)) & 1u, lo = (index >> (n - 2)) & 1u;
        index &= ~((std::size_t{1} << (n - 1)) | (std::size_t{1} << (n - 2)));
        index |= (lo << (n - 1)) | (hi << (n - 2));

        const MPSState fwd = forward_chain(qc, 4);
        ASSERT_EQ(fwd.open_span(), (Span{n - 1, n - 1}));
        const int shots = 2 * first_dense_shots(cost_of(fwd));

        MPSSimulator sim;
        const auto r = sim.run(qc, 4, shots, 31);
        const Span expected = (n <= ceiling) ? Span{n - 1, n - 1} : Span{0, 0};
        EXPECT_EQ(r.final_state.open_span(), expected)
            << (n <= ceiling ? "inside the ceiling the cost rule decides"
                             : "above the ceiling the dense path must never run");
        ASSERT_EQ(r.counts.size(), 1u);
        EXPECT_EQ(r.counts.begin()->first, key_of(index, n));
        EXPECT_EQ(r.counts.begin()->second, shots);
    }
}

TEST(V11301SamplingPath, AZeroQubitRunIsRefusedUpFrontOnEveryBackend) {
    // A circuit over no qubits has one state and one outcome, and nothing a
    // caller sends on purpose. Every simulator refuses to run one, at every
    // shot count, before touching any state, with the one refusal every run()
    // gives it: exactly lindblad::InvalidArgument from that run, the same
    // wording on all four, no position and no failed-run record. A zero-qubit
    // MPSState stays a valid object; only the run refuses.
    for (const int clbits : {0, 2}) {
        for (const int shots : {0, 1, 1000}) {
            SCOPED_TRACE("clbits " + std::to_string(clbits) + ", shots " +
                         std::to_string(shots));
            const QuantumCircuit empty(0, clbits);

            const auto expect_refusal = [](const std::string& entry,
                                           const std::function<void()>& run) {
                const std::uint64_t stores = detail::failed_run_stores();
                const auto e = v11311::thrown<InvalidArgument>(run);
                EXPECT_EQ(detail::failed_run_stores(), stores) << entry << " left a record";
                ASSERT_TRUE(e.has_value()) << entry << " ran a zero-qubit circuit";
                EXPECT_EQ(e->entry_point(), entry);
                EXPECT_EQ(std::string(e->what()),
                          entry + ": the circuit must have at least 1 qubit (got 0)");
                EXPECT_FALSE(e->where().has_value());
            };

            MPSSimulator mps;
            expect_refusal("MPSSimulator::run", [&] { (void)mps.run(empty, 4, shots, 1); });
            CliffordSimulator clifford;
            expect_refusal("CliffordSimulator::run", [&] { (void)clifford.run(empty, shots, 1); });
            StatevectorSimulator sv;
            expect_refusal("StatevectorSimulator::run", [&] { (void)sv.run(empty, shots, 1); });
            DensityMatrixSimulator dm;
            expect_refusal("DensityMatrixSimulator::run",
                           [&] { (void)dm.run(empty, NoiseModel{}, shots, 1); });
        }
    }

    // The object the runs refuse to build is still a valid one.
    const MPSState chain(0);
    EXPECT_EQ(chain.open_span(), (Span{0, -1}));
    EXPECT_EQ(chain.norm_sq(), 1.0);
}

TEST(V11301SamplingPath, BothPathsKeyCountsByTheMeasuredClbits) {
    // Terminal measurements mapping qubits to clbits out of order, onto a
    // register with a clbit nothing writes. A basis state makes every shot the
    // same key on either path.
    const int n = 5;
    QuantumCircuit qc(n, n + 1);
    qc.x(1).x(2).x(4);
    qc.swap(n - 2, n - 1);  // qubit 3 now 1 and qubit 4 now 0
    const std::vector<int> clbit_of = {3, 0, 4, 1, 2};
    for (int q = 0; q < n; ++q) qc.measure(q, clbit_of[static_cast<std::size_t>(q)]);
    std::size_t key = 0;
    for (int q : {1, 2, 3}) key |= std::size_t{1} << clbit_of[static_cast<std::size_t>(q)];

    QuantumCircuit bare(n);
    bare.instructions.assign(qc.instructions.begin(), qc.instructions.begin() + 4);
    const int flip = first_dense_shots(cost_of(forward_chain(bare, 4)));
    MPSSimulator sim;
    for (int shots : {1, flip - 1, flip, 4 * flip}) {
        SCOPED_TRACE("shots=" + std::to_string(shots));
        if (shots < 1) continue;
        const auto r = sim.run(qc, 4, shots, 37);
        ASSERT_EQ(r.counts.size(), 1u);
        EXPECT_EQ(r.counts.begin()->first, key_of(key, n + 1));
        EXPECT_EQ(r.counts.begin()->second, shots);
    }
}

// =============================================================================
// V11301Sampling - what each path draws from
// =============================================================================

TEST(V11301Sampling, BothPathsSampleTheChainsOwnDistribution) {
    // An untruncated chain. Just below the flip the sampler runs, and at four
    // times the flip the dense path does; each must match the chain's own
    // normalised distribution at an outcome set with no symmetry to hide a
    // bit-order mistake.
    const int n = std::min(20, dense_ceiling());
    ASSERT_GE(n, 14) << "the ceiling leaves too few qubits for a sampler-sized run";
    const QuantumCircuit qc = fixed_then_block(n, 6, 3, 41);
    const MPSState fwd = forward_chain(qc, 64);
    ASSERT_EQ(fwd.open_span(), (Span{n - 1, n - 1}));
    const std::vector<double> p = distribution(v11301::dense(fwd));
    const int flip = first_dense_shots(cost_of(fwd));

    MPSSimulator sim;
    struct Leg {
        const char* name;
        int shots;
        Span span;
    };
    for (const Leg& leg : {Leg{"sampler", flip - 1, Span{0, 0}},
                           Leg{"dense", 4 * flip, Span{n - 1, n - 1}}}) {
        SCOPED_TRACE(std::string(leg.name) + ", " + std::to_string(leg.shots) + " shots");
        const auto r = sim.run(qc, 64, leg.shots, 43);
        ASSERT_EQ(r.final_state.open_span(), leg.span) << "the other path ran";
        const double bound = tvd_bound(p, leg.shots);
        ASSERT_LT(bound, 0.05) << "too few shots for this check to discriminate";
        EXPECT_LT(tvd(r.counts, p), bound);
    }
}

TEST(V11301Sampling, ATruncatedChainIsSampledAsItsOwnNormalisedState) {
    // Truncation removes weight without renormalising, so the chain a capped
    // run returns has norm_sq() below 1 by exactly what it discarded. Both
    // paths sample "the chain's own distribution", which for a state of norm
    // below 1 is |a_k|^2 / norm_sq(): the sampler divides by the running total
    // at every step, and the dense path must reach the same distribution. The
    // fixed qubits make the all-ones outcome impossible, so weight arriving
    // there is weight the dense draw did not normalise away.
    const int n = std::min(20, dense_ceiling());
    ASSERT_GE(n, 14);
    const QuantumCircuit qc = fixed_then_block(n, 6, 4, 47);
    const MPSState fwd = forward_chain(qc, 2);
    ASSERT_EQ(fwd.open_span(), (Span{n - 1, n - 1}));
    const double lost = 1.0 - fwd.norm_sq();
    const std::vector<double> p = distribution(v11301::dense(fwd));
    const std::size_t all_ones = (std::size_t{1} << n) - 1;
    ASSERT_EQ(p[all_ones], 0.0) << "the fixed qubits must rule out the all-ones outcome";
    const int flip = first_dense_shots(cost_of(fwd));

    MPSSimulator sim;
    struct Leg {
        const char* name;
        int shots;
        Span span;
    };
    for (const Leg& leg : {Leg{"sampler", flip - 1, Span{0, 0}},
                           Leg{"dense", 4 * flip, Span{n - 1, n - 1}}}) {
        SCOPED_TRACE(std::string(leg.name) + ", " + std::to_string(leg.shots) + " shots");
        const auto r = sim.run(qc, 2, leg.shots, 53);
        ASSERT_EQ(r.final_state.open_span(), leg.span) << "the other path ran";
        const double bound = tvd_bound(p, leg.shots);
        ASSERT_GT(lost, 3.0 * bound)
            << "the cap discarded too little for a normalisation error to show";
        const auto it = r.counts.find(key_of(all_ones, n));
        const int impossible = (it == r.counts.end()) ? 0 : it->second;
        EXPECT_EQ(impossible, 0)
            << impossible << " shots landed on an outcome of probability zero; the "
            << "chain had lost " << lost << " of its weight";
        EXPECT_LT(tvd(r.counts, p), bound)
            << "the chain's own distribution is |a|^2 / norm_sq()";
    }
}

// =============================================================================
// V11301Collapse - one draw, the dense rule, the dense post-state
// =============================================================================

namespace {

// A draw deciding nothing: within rounding of the threshold it is compared to.
bool ambiguous(double u, double threshold, std::size_t dim) {
    return std::abs(u - threshold) <= kSlack * static_cast<double>(dim) * kEps;
}

// The state projected onto `outcome` of qubit q and divided by the square
// root of that outcome's raw marginal, which is the collapse measure_qubit
// performs, phases included.
Amplitudes collapse_dense(const Amplitudes& a, int q, int outcome) {
    const auto p = v11301::qubit_marginals(a, q);
    const double scale = 1.0 / std::sqrt(p[static_cast<std::size_t>(outcome)]);
    Amplitudes out(a.size(), Cplx(0.0, 0.0));
    for (std::size_t i = 0; i < a.size(); ++i)
        if (static_cast<int>((i >> q) & 1u) == outcome) out[i] = a[i] * scale;
    return out;
}

}  // namespace

TEST(V11301Collapse, MeasureQubitDrawsOnceAndCollapsesAsTheDenseRuleSays) {
    // outcome = 0 when the one uniform drawn falls below P(0) / (P(0) + P(1)),
    // with the marginals raw, so an unnormalised chain is measured as its
    // normalised state would be. The chain afterwards is the dense projection
    // divided by the square root of the raw marginal, amplitude for amplitude.
    const int n = 5;
    const auto sites = v11301::random_chain(n, {2, 4, 4, 2}, 5000);
    int seen[2] = {0, 0};
    int decided = 0;
    for (int q = 0; q < n; ++q) {
        for (std::uint64_t seed = 1; seed <= 24; ++seed) {
            SCOPED_TRACE("qubit " + std::to_string(q) + " seed " + std::to_string(seed));
            MPSState s(n, 64);
            s.set_tensors(sites);
            s.canonicalize((q + 2) % n);
            const Amplitudes a = v11301::dense(s);
            const auto p = v11301::qubit_marginals(a, q);
            const double threshold = p[0] / (p[0] + p[1]);

            std::mt19937_64 rng(seed);
            std::mt19937_64 replay = rng;
            const double u = std::uniform_real_distribution<double>(0.0, 1.0)(replay);
            if (ambiguous(u, threshold, a.size())) continue;
            ++decided;
            const int want = (u < threshold) ? 0 : 1;

            const int got = s.measure_qubit(q, rng);
            EXPECT_EQ(got, want);
            EXPECT_EQ(rng, replay) << "measure_qubit did not draw exactly one uniform";
            ++seen[got];
            const Amplitudes post = collapse_dense(a, q, want);
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), post),
                      v11301::amplitude_tol(4 * n, a.size(), 1.0));
        }
    }
    EXPECT_GE(decided, 5 * 24 - 2) << "too many draws landed on a threshold";
    EXPECT_GT(seen[0], 0) << "the fixture never produced outcome 0";
    EXPECT_GT(seen[1], 0) << "the fixture never produced outcome 1";
}

TEST(V11301Collapse, ResetDrawsTheSameCollapseAsMeasureAndLeavesZero) {
    // RESET is MEASURE followed by an X on outcome 1. With the same seed a
    // single trajectory through either instruction consumes the same draw and
    // collapses the rest of the register the same way, so the RESET chain is
    // the MEASURE chain with the reset qubit's digit cleared.
    const int n = 4;
    const int target = 1;
    QuantumCircuit prep(n, n);
    prep.ry(PI / 3.0, 0).cx(0, 1).ry(PI / 5.0, 2).cx(1, 2).h(3).cx(3, 1).rz(PI / 7.0, 1);

    int seen[2] = {0, 0};
    for (std::uint64_t seed = 1; seed <= 48; ++seed) {
        SCOPED_TRACE("seed " + std::to_string(seed));
        QuantumCircuit m = prep;
        m.measure(target, target);
        QuantumCircuit r = prep;
        r.reset(target);
        MPSSimulator sim;
        const Amplitudes am = v11301::dense(sim.run(m, 16, 0, seed).final_state);
        const Amplitudes ar = v11301::dense(sim.run(r, 16, 0, seed).final_state);

        const auto pm = v11301::qubit_marginals(am, target);
        const int outcome = pm[1] > pm[0] ? 1 : 0;
        ++seen[outcome];
        EXPECT_NEAR(pm[static_cast<std::size_t>(outcome)], 1.0, kSlack * 16 * kEps)
            << "MEASURE left the qubit uncollapsed";
        const auto pr = v11301::qubit_marginals(ar, target);
        EXPECT_NEAR(pr[0], 1.0, kSlack * 16 * kEps) << "RESET did not leave |0>";

        Amplitudes cleared(am.size(), Cplx(0.0, 0.0));
        for (std::size_t i = 0; i < am.size(); ++i)
            cleared[i & ~(std::size_t{1} << target)] += am[i];
        EXPECT_LE(v11301::max_abs_diff(ar, cleared), v11301::amplitude_tol(32, am.size(), 1.0))
            << "RESET collapsed the register differently from MEASURE";
    }
    EXPECT_GT(seen[0], 0);
    EXPECT_GT(seen[1], 0);

    // On a qubit that is certainly |1> the flip always happens; on one that is
    // certainly |0> it never does.
    for (int certain : {0, 1}) {
        QuantumCircuit qc(2);
        if (certain) qc.x(0);
        qc.x(1).reset(0);
        MPSSimulator sim;
        const Amplitudes a = v11301::dense(sim.run(qc, 4, 0, 3).final_state);
        EXPECT_NEAR(std::abs(a[0b10]), 1.0, kSlack * 4 * kEps)
            << "RESET of a qubit certainly in |" << certain << "> did not leave |0>";
    }
}

TEST(V11301Collapse, MeasureSequentialFollowsTheConditionalMarginals) {
    // One uniform per qubit, qubit 0 first: each outcome is decided by the
    // qubit's marginal conditional on the outcomes before it, which the dense
    // replay computes by collapsing as it goes. The bitstring puts qubit 0
    // rightmost, and the chain afterwards is the dense state collapsed onto
    // every outcome, phase included.
    const int n = 5;
    const auto sites = v11301::random_chain(n, {2, 4, 4, 2}, 5100);
    std::map<std::string, int> seen;
    for (std::uint64_t seed = 1; seed <= 40; ++seed) {
        SCOPED_TRACE("seed " + std::to_string(seed));
        MPSState s(n, 64);
        s.set_tensors(sites);
        Amplitudes psi = v11301::dense(s);

        std::mt19937_64 rng(seed);
        std::mt19937_64 replay = rng;
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::string want(static_cast<std::size_t>(n), '0');
        bool decisive = true;
        for (int q = 0; q < n; ++q) {
            const auto p = v11301::qubit_marginals(psi, q);
            ASSERT_GT(p[0] + p[1], 0.0);
            const double u = unit(replay);
            const double threshold = p[0] / (p[0] + p[1]);
            if (ambiguous(u, threshold, psi.size())) decisive = false;
            const int outcome = (u < threshold) ? 0 : 1;
            want[static_cast<std::size_t>(n - 1 - q)] = outcome ? '1' : '0';
            psi = collapse_dense(psi, q, outcome);
        }
        if (!decisive) continue;

        const std::string got = s.measure_sequential(rng);
        EXPECT_EQ(got, want);
        EXPECT_EQ(rng, replay) << "measure_sequential did not draw one uniform per qubit";
        ++seen[got];
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), psi),
                  v11301::amplitude_tol(8 * n, psi.size(), 1.0));
    }
    EXPECT_GE(seen.size(), 3u) << "the fixture's outcomes are too concentrated to test";
}

TEST(V11301Collapse, MeasureQuditDrawsOnceAndCollapsesAsTheDenseRuleSays) {
    // At general d the rule is cumulative: the one uniform scaled by the total
    // selects the first digit whose running marginal reaches it.
    const int n = 4, d = 3;
    const auto sites = v11301::random_qudit_chain(n, d, {3, 4, 3}, 5200);
    std::vector<int> seen(static_cast<std::size_t>(d), 0);
    for (int q = 0; q < n; ++q) {
        for (std::uint64_t seed = 1; seed <= 24; ++seed) {
            SCOPED_TRACE("qudit " + std::to_string(q) + " seed " + std::to_string(seed));
            QuditMPS s(n, d, 64);
            s.set_tensors(sites);
            const Amplitudes a = v11301::dense(s);
            const std::vector<double> p = v11301::qudit_marginals(a, q, d);
            double total = 0.0;
            for (double x : p) total += x;

            std::mt19937_64 rng(seed);
            std::mt19937_64 replay = rng;
            const double u = std::uniform_real_distribution<double>(0.0, 1.0)(replay);
            const double target = u * total;
            int want = d - 1;
            bool decisive = true;
            double running = 0.0;
            for (int x = 0; x < d; ++x) {
                running += p[static_cast<std::size_t>(x)];
                if (ambiguous(target / total, running / total, a.size())) decisive = false;
                if (target <= running) { want = x; break; }
            }
            if (!decisive) continue;

            const int got = s.measure_qudit(q, rng);
            EXPECT_EQ(got, want);
            EXPECT_EQ(rng, replay) << "measure_qudit did not draw exactly one uniform";
            ++seen[static_cast<std::size_t>(got)];

            std::size_t stride = 1;
            for (int i = 0; i < q; ++i) stride *= static_cast<std::size_t>(d);
            Amplitudes post(a.size(), Cplx(0.0, 0.0));
            const double scale = 1.0 / std::sqrt(p[static_cast<std::size_t>(want)]);
            for (std::size_t i = 0; i < a.size(); ++i)
                if (static_cast<int>((i / stride) % static_cast<std::size_t>(d)) == want)
                    post[i] = a[i] * scale;
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), post),
                      v11301::amplitude_tol(4 * n, a.size(), 1.0));
        }
    }
    for (int x = 0; x < d; ++x) EXPECT_GT(seen[static_cast<std::size_t>(x)], 0) << "digit " << x;
}

// =============================================================================
// V11301NoNorm - a state with nothing to measure is refused, everywhere
// =============================================================================
// A zero state has no distribution, and a non-finite one has none that means
// anything, so every entry that collapses or samples refuses it, as normalize()
// already does on every state class and with the same type. The refusal comes
// before any draw, so an engine the caller shares with other work is left as
// it was. Such states arise only from a caller building one, so each fixture
// builds its own the way a caller would: zero entries handed in through
// set_tensors or the amplitude setters, and a NaN through a gate under Ignore.

namespace {

// Two qubit chains with no norm: a zero site handed in, and a NaN carried in
// by a gate that nothing checks.
std::vector<std::pair<std::string, MPSState>> normless_qubit_chains(int n) {
    std::vector<std::pair<std::string, MPSState>> out;
    {
        MPSState s(n, 8);
        std::vector<MPSTensor> sites = s.tensors();
        for (auto& z : sites[1].data) z = Complex128(0.0, 0.0);
        s.set_tensors(sites);
        out.emplace_back("zero", std::move(s));
    }
    {
        MPSState s(n, 8);
        const std::array<Complex128, 4> nan_gate = {
            Complex128(quiet_nan_strict(), 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
        s.apply_single_qubit_gate(nan_gate, 1, {Validation::Ignore});
        out.emplace_back("non-finite", std::move(s));
    }
    return out;
}

std::vector<std::pair<std::string, QuditMPS>> normless_qudit_chains(int n, int d) {
    std::vector<std::pair<std::string, QuditMPS>> out;
    {
        QuditMPS s(n, d, 8);
        std::vector<MPSSiteTensor> sites = s.tensors();
        for (auto& z : sites[1].data) z = Complex128(0.0, 0.0);
        s.set_tensors(sites);
        out.emplace_back("zero", std::move(s));
    }
    {
        QuditMPS s(n, d, 8);
        std::vector<Complex128> nan_gate(static_cast<std::size_t>(d * d), Complex128(0.0, 0.0));
        nan_gate[0] = Complex128(quiet_nan_strict(), 0.0);
        for (int x = 1; x < d; ++x)
            nan_gate[static_cast<std::size_t>(x * d + x)] = Complex128(1.0, 0.0);
        s.apply_1qudit(1, nan_gate, {Validation::Ignore});
        out.emplace_back("non-finite", std::move(s));
    }
    return out;
}

// A direct call on a state with no norm: exactly RuntimeFailure naming the
// call, whose message is the call's name and `what`.
void expect_no_norm(const std::string& entry_point, const std::function<void()>& call,
                    const std::string& what = "no norm to sample from; the state is zero or "
                                              "non-finite") {
    const auto e = v11311::thrown<RuntimeFailure>(call);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), entry_point);
    EXPECT_EQ(std::string(e->what()), entry_point + ": " + what);
}

std::vector<std::pair<std::string, std::vector<Complex128>>> normless_amplitudes(std::size_t dim) {
    std::vector<Complex128> zero(dim, Complex128(0.0, 0.0));
    std::vector<Complex128> nan(dim, Complex128(0.0, 0.0));
    nan[0] = Complex128(quiet_nan_strict(), 0.0);
    return {{"zero", zero}, {"non-finite", nan}};
}

}  // namespace

TEST(V11301NoNorm, TheQubitChainRefusesEveryCollapse) {
    const int n = 3;
    for (auto& [name, chain] : normless_qubit_chains(n)) {
        SCOPED_TRACE(name);
        for (int q = 0; q < n; ++q) {
            MPSState s = chain;
            std::mt19937_64 rng(6);
            const std::mt19937_64 untouched = rng;
            SCOPED_TRACE("qubit " + std::to_string(q));
            expect_no_norm("MPSState::measure_qubit", [&] { (void)s.measure_qubit(q, rng); });
            EXPECT_EQ(rng, untouched) << "a refused measurement drew from the engine";
        }
        MPSState s = chain;
        std::mt19937_64 rng(7);
        const std::mt19937_64 untouched = rng;
        expect_no_norm("MPSState::measure_sequential", [&] { (void)s.measure_sequential(rng); });
        EXPECT_EQ(rng, untouched);
    }
}

TEST(V11301NoNorm, TheQuditChainRefusesEveryCollapseAndSample) {
    const int n = 3, d = 3;
    for (auto& [name, chain] : normless_qudit_chains(n, d)) {
        SCOPED_TRACE(name);
        for (int q = 0; q < n; ++q) {
            QuditMPS s = chain;
            std::mt19937_64 rng(6);
            const std::mt19937_64 untouched = rng;
            SCOPED_TRACE("qudit " + std::to_string(q));
            expect_no_norm("QuditMPS::measure_qudit", [&] { (void)s.measure_qudit(q, rng); });
            EXPECT_EQ(rng, untouched);
        }
        QuditMPS s = chain;
        expect_no_norm("QuditMPS::measure", [&] { (void)s.measure(7); });
    }
}

TEST(V11301NoNorm, AnMpsRunRefusesANormlessStartingChainBeforeTheFirstGate) {
    // A chain with no norm handed to a run as its initial state is refused
    // before the first gate, whatever the circuit: there is no state to start
    // from. The refusal is exactly InvalidArgument from the run and leaves no
    // failed-run record, since nothing was computed. Every path that would
    // otherwise meet the missing norm later (either sampler, a MEASURE or a
    // RESET in a trajectory, or none at all) is refused the same way.
    const int n = 3;
    for (auto& [name, chain] : normless_qubit_chains(n)) {
        SCOPED_TRACE(name);
        auto source = std::make_shared<MPSState>(chain);
        RunPlan plan;
        plan.initial = InitialState::from(std::shared_ptr<const MPSState>(source));
        MPSSimulator sim;

        const auto expect_refused = [&](const QuantumCircuit& qc, int shots, const char* path) {
            SCOPED_TRACE(path);
            const std::uint64_t stores = detail::failed_run_stores();
            const auto e = v11311::thrown<InvalidArgument>(
                [&] { (void)sim.run(qc, 8, shots, 1, plan); });
            EXPECT_EQ(detail::failed_run_stores(), stores) << "a refusal left a record";
            ASSERT_TRUE(e.has_value());
            EXPECT_EQ(e->entry_point(), "MPSSimulator::run");
            EXPECT_EQ(std::string(e->what()),
                      "MPSSimulator::run: InitialState: the supplied chain has no norm (it is "
                      "zero or not finite), so a run cannot start from it");
            EXPECT_FALSE(e->where().has_value());
        };

        QuantumCircuit gates(n);
        gates.x(0);
        expect_refused(gates, 0, "gates only");
        QuantumCircuit measured = gates;
        measured.measure_all();
        expect_refused(measured, 1, "the sampler path");
        expect_refused(measured, 1 << 20, "the dense path");
        expect_refused(gates, 1 << 20, "sampling the whole register with no MEASURE");
        expect_refused(measured, 0, "a MEASURE in a single trajectory");
        QuantumCircuit reset(n);
        reset.reset(1);
        expect_refused(reset, 0, "a RESET");
    }
}

TEST(V11301NoNorm, AnMpsRunStartsFromAChainWhoseNormFellBelowOne) {
    // The other side of the refusal: a chain that kept some norm, as a
    // truncated chain from an earlier run does, is a state a run can start
    // from. Its weight is halved here, on the site that carries |1> on qubit 0.
    const int n = 3;
    MPSState chain(n, 8);
    const std::array<Complex128, 4> x = {Complex128(0.0, 0.0), Complex128(1.0, 0.0),
                                         Complex128(1.0, 0.0), Complex128(0.0, 0.0)};
    chain.apply_single_qubit_gate(x, 0);
    std::vector<MPSTensor> sites = chain.tensors();
    for (auto& z : sites[0].data) z = Complex128(z.real * INV_SQRT2, z.imag * INV_SQRT2);
    chain.set_tensors(sites);
    ASSERT_GT(chain.norm_sq(), 0.0);
    ASSERT_LT(chain.norm_sq(), 1.0);

    RunPlan plan;
    plan.initial = InitialState::from(std::make_shared<const MPSState>(chain));
    QuantumCircuit measured(n, n);
    measured.measure_all();
    MPSSimulator sim;
    std::optional<MPSSimulator::Result> r;
    ASSERT_NO_THROW(r.emplace(sim.run(measured, 8, 64, 1, plan)));
    ASSERT_EQ(r->counts.size(), 1u);
    EXPECT_EQ(r->counts.begin()->first, "001");
    EXPECT_EQ(r->counts.begin()->second, 64);
}

TEST(V11301NoNorm, TheDenseStatesRefuseEverySample) {
    const int n = 3;
    for (const auto& [name, amps] : normless_amplitudes(std::size_t{1} << n)) {
        SCOPED_TRACE(name);
        Statevector sv(n);
        sv.set_amplitudes(amps, {Validation::Ignore});
        expect_no_norm("Statevector::sample_counts", [&] { (void)sv.sample_counts(10, 5); });
        expect_no_norm("Statevector::measure_once", [&] { (void)sv.measure_once(5); });

        // The simulator's own collapse, reached one instruction at a time.
        QuantumCircuit ops(n, n);
        ops.measure(1, 1).reset(2);
        StatevectorSimulator ssim;
        for (const auto& inst : ops.instructions) {
            SCOPED_TRACE(inst.gate_name());
            Statevector s(n);
            s.set_amplitudes(amps, {Validation::Ignore});
            expect_no_norm("StatevectorSimulator::apply_instruction",
                           [&] { ssim.apply_instruction(s, inst); });
        }
    }
    for (const auto& [name, amps] : normless_amplitudes(27)) {
        SCOPED_TRACE(std::string("qudit ") + name);
        QuditStatevector q(3, 3);
        q.amplitudes = amps;
        expect_no_norm("QuditStatevector::measure", [&] { (void)q.measure(5); });

        QuditDensityMatrix rho(3, 3);
        std::fill(rho.rho.begin(), rho.rho.end(), Complex128(0.0, 0.0));
        if (name == "non-finite") rho.rho[0] = Complex128(quiet_nan_strict(), 0.0);
        expect_no_norm("QuditDensityMatrix::measure", [&] { (void)rho.measure(5); });
    }
}

TEST(V11301NoNorm, DenseRunsReportTheRefusal) {
    // A normless state cannot be handed to the statevector or density-matrix
    // simulator as its initial state, since seeding judges normalisation, so it
    // arises the one way a run allows: a matrix the caller marked Ignore, here
    // a zero matrix or diag(NaN, 1), applied as the circuit's first
    // instruction. Every run of it then fails mid-run with RuntimeFailure and
    // leaves a failed-run record: with no measurement at the end-of-run check
    // on the final state, with one at the sample or the collapse.
    //
    // Where it fails follows the walk. Terminal sampling and the end-of-run
    // check come after the instructions, so they name no instruction and no
    // shot. A trajectory walked shot by shot names the MEASURE and the shot it
    // is in, and a shots == 0 run is one trajectory: shot 0.
    const int n = 3;
    const std::vector<std::pair<std::string, std::vector<Complex128>>> killers = {
        {"zero", std::vector<Complex128>(4, Complex128(0.0, 0.0))},
        {"non-finite", {Complex128(quiet_nan_strict(), 0.0), Complex128(0.0, 0.0),
                        Complex128(0.0, 0.0), Complex128(1.0, 0.0)}}};
    for (const auto& [name, matrix] : killers) {
        SCOPED_TRACE(name);
        QuantumCircuit gates(n, n);
        gates.unitary(matrix, {0}, "normless", {Validation::Ignore}).x(1);
        QuantumCircuit terminal = gates;
        terminal.measure_all();
        QuantumCircuit trajectory = gates;
        trajectory.measure(1, 1).x(1);  // a gate after the MEASURE: per shot

        const std::string no_norm = "no norm to sample from; the state is zero or non-finite";
        const auto expect_failed = [](const std::string& entry, const std::string& what,
                                      std::optional<std::pair<int, int>> shot_instruction,
                                      const std::function<void()>& run) {
            const std::uint64_t stores = detail::failed_run_stores();
            const auto e = v11311::thrown<RuntimeFailure>(run);
            EXPECT_EQ(detail::failed_run_stores(), stores + 1) << "a mid-run failure left no record";
            (void)take_failed_run();
            ASSERT_TRUE(e.has_value());
            EXPECT_EQ(e->entry_point(), entry);
            EXPECT_EQ(std::string(e->what()).rfind(entry + ": " + what, 0), 0u) << e->what();
            if (!shot_instruction) {
                EXPECT_FALSE(e->where().has_value());
            } else {
                v11311::expect_point(e->where(), shot_instruction->first,
                                     shot_instruction->second, "measure", {1});
            }
        };
        const std::string sv_final =
            "the final state is zero or not finite, so there is no state to return; a matrix "
            "let through by ValidationOptions Warn or Ignore can do this";
        const std::string dm_final =
            "the final state is zero or not finite, so there is no state to return; a matrix or "
            "channel let through by ValidationOptions Warn or Ignore can do this";

        StatevectorSimulator ssim;
        const std::string sv = "StatevectorSimulator::run";
        {
            SCOPED_TRACE("statevector, no measurement");
            expect_failed(sv, sv_final, std::nullopt, [&] { (void)ssim.run(gates, 0, 1); });
        }
        {
            SCOPED_TRACE("statevector, terminal sampling");
            expect_failed(sv, no_norm, std::nullopt, [&] { (void)ssim.run(terminal, 16, 1); });
        }
        {
            SCOPED_TRACE("statevector, per-shot collapse");
            expect_failed(sv, no_norm, std::pair{0, 2}, [&] { (void)ssim.run(trajectory, 16, 1); });
        }
        {
            // The statevector's single trajectory is walked shot by shot, so
            // it names shot 0, the shot its observers are told.
            SCOPED_TRACE("statevector, a single trajectory");
            expect_failed(sv, no_norm, std::pair{0, 2}, [&] { (void)ssim.run(trajectory, 0, 1); });
        }

        DensityMatrixSimulator dsim;
        const NoiseModel quiet{};
        const std::string dm = "DensityMatrixSimulator::run";
        {
            SCOPED_TRACE("density matrix, no measurement");
            expect_failed(dm, dm_final, std::nullopt, [&] { (void)dsim.run(gates, quiet, 0, 1); });
        }
        {
            SCOPED_TRACE("density matrix, terminal sampling");
            expect_failed(dm, no_norm, std::nullopt, [&] { (void)dsim.run(terminal, quiet, 16, 1); });
        }
        {
            SCOPED_TRACE("density matrix, per-shot collapse");
            expect_failed(dm, no_norm, std::pair{0, 2},
                          [&] { (void)dsim.run(trajectory, quiet, 16, 1); });
        }
        {
            SCOPED_TRACE("density matrix, a single trajectory");
            expect_failed(dm, no_norm, std::pair{0, 2},
                          [&] { (void)dsim.run(trajectory, quiet, 0, 1); });
        }
    }
}

// =============================================================================
// V11301ResetSampling - a RESET is a collapse in every shot
// =============================================================================
// ry(pi/3) puts qubit 0 in |1> with probability sin^2(pi/6) = 1/4, the CX
// copies it onto qubit 1, and the RESET of qubit 0 collapses qubit 1 onto that
// outcome, afresh in every shot. Measured afterwards, clbit 0 (qubit 1) reads
// 1 with probability 1/4 and clbit 1 (qubit 0) always reads 0. A backend that
// applies the RESET once for every shot returns a single key instead.

namespace {

QuantumCircuit reset_then_measure(double theta) {
    QuantumCircuit qc(2, 2);
    qc.ry(theta, 0).cx(0, 1).reset(0).measure(1, 0).measure(0, 1);
    return qc;
}

std::vector<double> reset_distribution(double theta) {
    const double p1 = std::pow(std::sin(theta / 2.0), 2);
    return {1.0 - p1, p1, 0.0, 0.0};  // keys "00", "01", "10", "11"
}

constexpr int kResetShots = 20000;

}  // namespace

TEST(V11301ResetSampling, TheMpsBackendCollapsesEveryShot) {
    const double theta = PI / 3.0;
    const auto p = reset_distribution(theta);
    MPSSimulator sim;
    const auto r = sim.run(reset_then_measure(theta), 4, kResetShots, 61);
    EXPECT_LT(tvd(r.counts, p), tvd_bound(p, kResetShots));
    EXPECT_EQ(r.counts.size(), 2u) << "one collapse served every shot";
}

TEST(V11301ResetSampling, TheStatevectorBackendCollapsesEveryShot) {
    const double theta = PI / 3.0;
    const auto p = reset_distribution(theta);
    StatevectorSimulator sim;
    const auto r = sim.run(reset_then_measure(theta), kResetShots, 61);
    EXPECT_LT(tvd(r.counts, p), tvd_bound(p, kResetShots));
    EXPECT_EQ(r.counts.size(), 2u) << "one collapse served every shot";
}

TEST(V11301ResetSampling, TheDensityMatrixAndCliffordBackendsAgree) {
    // The controls: the density-matrix backend applies RESET as a channel,
    // which one pass describes exactly, and the Clifford backend already runs
    // every shot when a RESET is present. Clifford takes H for the rotation,
    // which is a stabilizer gate, so its qubit 1 reads 1 with probability 1/2.
    {
        const double theta = PI / 3.0;
        const auto p = reset_distribution(theta);
        DensityMatrixSimulator sim;
        const auto r = sim.run(reset_then_measure(theta), NoiseModel{}, kResetShots, 61);
        EXPECT_LT(tvd(r.counts, p), tvd_bound(p, kResetShots));
    }
    {
        QuantumCircuit qc(2, 2);
        qc.h(0).cx(0, 1).reset(0).measure(1, 0).measure(0, 1);
        const auto p = reset_distribution(PI / 2.0);
        CliffordSimulator sim;
        const auto r = sim.run(qc, kResetShots, 61);
        EXPECT_LT(tvd(r.counts, p), tvd_bound(p, kResetShots));
    }
}

TEST(V11301Collapse, QuditMeasureSamplesTheNormalisedChain) {
    // measure() does not collapse, so repeated calls sample one distribution:
    // the chain's own, normalised, even when the chain is not. Digits come
    // back in qudit order, which a basis state with no symmetry pins exactly.
    const int n = 4, d = 3;
    {
        QuditMPS s(n, d, 64);
        const std::vector<int> digits = {1, 2, 0, 1};
        for (int q = 0; q < n; ++q) {
            std::vector<Complex128> shift(9, Complex128(0.0, 0.0));
            for (int x = 0; x < d; ++x)
                shift[static_cast<std::size_t>(((x + digits[static_cast<std::size_t>(q)]) % d) * d + x)] =
                    Complex128(1.0, 0.0);
            s.apply_1qudit(q, shift);
        }
        EXPECT_EQ(s.measure(3), digits);
    }

    QuditMPS s(n, d, 64);
    s.set_tensors(v11301::random_qudit_chain(n, d, {3, 4, 3}, 5300));
    const std::vector<double> p = distribution(v11301::dense(s));
    constexpr int kShots = 40000;
    Counts counts;
    for (int shot = 1; shot <= kShots; ++shot) {
        const std::vector<int> x = s.measure(static_cast<std::uint64_t>(shot));
        std::size_t index = 0, stride = 1;
        for (int q = 0; q < n; ++q) {
            index += static_cast<std::size_t>(x[static_cast<std::size_t>(q)]) * stride;
            stride *= static_cast<std::size_t>(d);
        }
        // tvd() reads keys as binary; this layer's keys are the flat index
        // written in binary, which is the same bijection onto indices.
        ++counts[key_of(index, 8)];
    }
    EXPECT_LT(tvd(counts, p), tvd_bound(p, kShots));
}
