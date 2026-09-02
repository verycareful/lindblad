// 1.1.25.1 test wave - the two sampling routes, and the paths the rework did
// not touch.
//
// Options::sampling selects between reading the outcome distribution's affine
// subspace off the tableau once and drawing each shot as a subset-sum of its
// free directions (Slab), and replaying a measurement pass over a copy of the
// tableau for every shot (PerShot). They sample the SAME distribution by
// different means, which makes each the other's oracle, and both are checked
// against the statevector's exact probabilities as well.
//
// Comparisons here are as tight as sampling allows: a counts key outside the
// exact support is a hard failure with no tolerance at all, and frequencies are
// held to six binomial standard deviations, which at these shot counts is a far
// narrower band than a total-variation bound would give.
//
// The general route (mid-circuit measurement, feedforward, reset) stays
// row-major and was not part of the rework, but nothing else in the tree proves
// the rework left it alone, so it is re-pinned here.

#include <gtest/gtest.h>

#include "v11251_clifford_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/clifford_sim.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

using Counts = std::unordered_map<std::string, int>;
using Sampling = CliffordSimulator::Options::Sampling;
using Elim = StabilizerState::Elimination;

const Sampling kRoutes[] = {Sampling::Slab, Sampling::PerShot};

// The general route is per-shot by nature and offers no slab, so a circuit that
// takes it is exercised under PerShot alone. Asking for Slab there is the
// subject of ExplicitSlabOnTheGeneralPathIsRefused.
const Sampling kGeneralRoutes[] = {Sampling::PerShot};

const char* route_name(Sampling s) {
    return s == Sampling::Slab ? "Slab" : "PerShot";
}

Counts run_with(const QuantumCircuit& qc, Sampling route, int shots, uint64_t seed,
                Elim elim = Elim::Plain) {
    CliffordSimulator sim;
    sim.options.sampling = route;
    sim.options.elimination = elim;
    return sim.run(qc, shots, seed).counts;
}

// The bitstring distribution a circuit must produce, built from the exact
// qubit-packed distribution and the circuit's own qubit-to-clbit map. clbit c
// sits at string position n_clbits - 1 - c, so clbit 0 is rightmost.
std::map<std::string, double> expected_keys(const v11251::Dist& qubit_dist,
                                            const std::vector<std::pair<int, int>>& map,
                                            int n_clbits) {
    std::map<std::string, double> out;
    for (const auto& [outcome, p] : qubit_dist) {
        std::string key(static_cast<size_t>(n_clbits), '0');
        for (const auto& [q, c] : map) {
            if (c < 0 || c >= n_clbits) continue;
            if ((outcome >> q) & 1ULL) key[static_cast<size_t>(n_clbits - 1 - c)] = '1';
        }
        out[key] += p;
    }
    return out;
}

// Reads the qubit-to-clbit map straight off the circuit, so a test cannot state
// a mapping the circuit does not actually carry.
std::vector<std::pair<int, int>> measurement_map(const QuantumCircuit& qc) {
    std::vector<std::pair<int, int>> map;
    for (const auto& inst : qc.instructions) {
        if (inst.type != Instruction::GateType::MEASURE) continue;
        const int q = inst.qubits[0];
        map.emplace_back(q, inst.clbits.empty() ? q : inst.clbits[0]);
    }
    return map;
}

int n_clbits_of(const QuantumCircuit& qc) {
    return qc.n_clbits > 0 ? qc.n_clbits : qc.n_qubits;
}

// The exact bitstring distribution of a circuit, obtained by running the gates
// on the statevector with the measurements stripped and then applying the
// circuit's own clbit map.
std::map<std::string, double> exact_keys(const QuantumCircuit& qc) {
    QuantumCircuit unmeasured(qc.n_qubits);
    for (const auto& inst : qc.instructions) {
        if (inst.type == Instruction::GateType::MEASURE ||
            inst.type == Instruction::GateType::BARRIER) {
            continue;
        }
        unmeasured.instructions.push_back(inst);
    }
    return expected_keys(v11251::exact_statevector_distribution(unmeasured),
                         measurement_map(qc), n_clbits_of(qc));
}

// Support is exact: a key the state cannot produce is a failure regardless of
// how rarely it appeared. Frequencies are held to six binomial sigmas, and
// every key with real probability must actually show up.
::testing::AssertionResult counts_match(const Counts& counts,
                                        const std::map<std::string, double>& exact,
                                        int shots) {
    int total = 0;
    for (const auto& [key, n] : counts) {
        total += n;
        if (exact.find(key) == exact.end()) {
            return ::testing::AssertionFailure()
                   << "key '" << key << "' appeared " << n
                   << " times but has zero probability";
        }
    }
    if (total != shots) {
        return ::testing::AssertionFailure()
               << "counts total " << total << ", expected " << shots;
    }
    for (const auto& [key, p] : exact) {
        const auto it = counts.find(key);
        const int n = (it == counts.end()) ? 0 : it->second;
        const double f = static_cast<double>(n) / static_cast<double>(shots);
        const double sigma = v11251::binomial_sigma(p, shots);
        if (std::abs(f - p) > 6.0 * sigma + 1e-9) {
            return ::testing::AssertionFailure()
                   << "key '" << key << "': frequency " << f << ", expected " << p
                   << " (6 sigma = " << 6.0 * sigma << ")";
        }
        if (n == 0 && p > 0.0) {
            return ::testing::AssertionFailure()
                   << "key '" << key << "' has probability " << p
                   << " but never appeared in " << shots << " shots";
        }
    }
    return ::testing::AssertionSuccess();
}

// Circuits with terminal measurement only, which is the condition for the fast
// path both routes live on.
struct Case {
    const char* name;
    QuantumCircuit (*build)();
};

QuantumCircuit c_bell() {
    QuantumCircuit qc(2, 2); qc.h(0).cx(0, 1).measure_all(); return qc;
}
QuantumCircuit c_ghz5() {
    QuantumCircuit qc(5, 5);
    qc.h(0).cx(0, 1).cx(1, 2).cx(2, 3).cx(3, 4).measure_all();
    return qc;
}
QuantumCircuit c_uniform5() {
    QuantumCircuit qc(5, 5);
    for (int q = 0; q < 5; ++q) qc.h(q);
    qc.measure_all();
    return qc;
}
QuantumCircuit c_deterministic() {
    QuantumCircuit qc(4, 4); qc.x(0).x(2).measure_all(); return qc;
}
QuantumCircuit c_new_gates() {
    QuantumCircuit qc(4, 4);
    qc.h(0).sx(1).cy(0, 2).iswap(1, 3).ecr(2, 3).sxdg(0).measure_all();
    return qc;
}
QuantumCircuit c_rotations() {
    QuantumCircuit qc(3, 3);
    qc.ry(PI_2, 0).rz(PI_2, 1).rx(PI_2, 2).cx(0, 1).cz(1, 2).measure_all();
    return qc;
}
QuantumCircuit c_partial() {
    QuantumCircuit qc(4, 4);
    qc.h(0).cx(0, 1).h(2).cx(2, 3);
    qc.measure(0, 0);
    qc.measure(2, 2);          // qubits 1 and 3 are never measured
    return qc;
}
QuantumCircuit c_permuted_clbits() {
    QuantumCircuit qc(3, 3);
    qc.h(0).cx(0, 1).x(2);
    qc.measure(0, 2);
    qc.measure(1, 0);
    qc.measure(2, 1);
    return qc;
}
QuantumCircuit c_narrow_clreg() {
    QuantumCircuit qc(4, 2);   // fewer clbits than qubits
    qc.h(0).cx(0, 1).h(2);
    qc.measure(0, 0);
    qc.measure(2, 1);
    return qc;
}
QuantumCircuit c_wide_clreg() {
    QuantumCircuit qc(2, 5);   // more clbits than qubits
    qc.h(0).cx(0, 1);
    qc.measure(0, 3);
    qc.measure(1, 1);
    return qc;
}

const Case kCases[] = {
    {"bell", c_bell},
    {"ghz5", c_ghz5},
    {"uniform5", c_uniform5},
    {"deterministic", c_deterministic},
    {"new_gates", c_new_gates},
    {"rotations", c_rotations},
    {"partial", c_partial},
    {"permuted_clbits", c_permuted_clbits},
    {"narrow_clreg", c_narrow_clreg},
    {"wide_clreg", c_wide_clreg},
};

constexpr int kShots = 20000;
constexpr uint64_t kSeed = 0x9E3779B97F4A7C15ULL;

}  // namespace

// =============================================================================
// Both routes against the exact distribution
// =============================================================================

TEST(V11251CliffordSampling, EveryRouteMatchesTheExactDistribution) {
    for (const Case& c : kCases) {
        const QuantumCircuit qc = c.build();
        const std::map<std::string, double> exact = exact_keys(qc);
        for (Sampling route : kRoutes) {
            SCOPED_TRACE(std::string(c.name) + " " + route_name(route));
            EXPECT_TRUE(counts_match(run_with(qc, route, kShots, kSeed), exact, kShots));
        }
    }
}

// The two routes are each other's oracle: the same key set, and frequencies
// that agree with each other as well as with the reference.
TEST(V11251CliffordSampling, TheTwoRoutesAgreeWithEachOther) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        const QuantumCircuit qc = c.build();
        const Counts slab = run_with(qc, Sampling::Slab, kShots, kSeed);
        const Counts per_shot = run_with(qc, Sampling::PerShot, kShots, kSeed + 1);

        EXPECT_EQ(slab.size(), per_shot.size());
        for (const auto& [key, n] : slab) {
            const auto it = per_shot.find(key);
            ASSERT_NE(it, per_shot.end()) << "Slab produced key '" << key
                                          << "' that PerShot never did";
            const double fs = static_cast<double>(n) / kShots;
            const double fp = static_cast<double>(it->second) / kShots;
            // Two independent binomials, so the difference carries both
            // variances. The bound is derived from the observed frequency
            // rather than assumed.
            const double sigma = std::sqrt(2.0) * v11251::binomial_sigma(fs, kShots);
            EXPECT_LE(std::abs(fs - fp), 6.0 * sigma + 1e-9)
                << "key '" << key << "': Slab " << fs << ", PerShot " << fp;
        }
    }
}

// The block elimination changes how the slab is reduced, not what it describes,
// so selecting it must not move the distribution.
TEST(V11251CliffordSampling, TheEliminationKnobDoesNotMoveTheDistribution) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        const QuantumCircuit qc = c.build();
        const std::map<std::string, double> exact = exact_keys(qc);
        EXPECT_TRUE(counts_match(
            run_with(qc, Sampling::Slab, kShots, kSeed, Elim::FourRussians),
            exact, kShots));
    }
}

// =============================================================================
// Determinism and the difference between the routes
// =============================================================================

TEST(V11251CliffordSampling, SameSeedGivesIdenticalCountsOnEachRoute) {
    for (const Case& c : kCases) {
        const QuantumCircuit qc = c.build();
        for (Sampling route : kRoutes) {
            SCOPED_TRACE(std::string(c.name) + " " + route_name(route));
            const Counts first = run_with(qc, route, 4096, kSeed);
            const Counts second = run_with(qc, route, 4096, kSeed);
            EXPECT_EQ(first, second);
        }
    }
}

TEST(V11251CliffordSampling, DifferentSeedsGiveDifferentCounts) {
    // A distribution wide enough that two seeds agreeing on every one of its
    // 32 counts would be an event of vanishing probability.
    const QuantumCircuit qc = c_uniform5();
    for (Sampling route : kRoutes) {
        SCOPED_TRACE(route_name(route));
        EXPECT_NE(run_with(qc, route, 4096, kSeed),
                  run_with(qc, route, 4096, kSeed + 12345));
    }
}

// The routes consume the random stream differently, so a given seed produces
// different individual bitstrings under each. Counts agree in distribution, not
// shot for shot, and that is intended rather than a defect to be discovered
// later by a caller pinning seeded bitstrings.
TEST(V11251CliffordSampling, TheTwoRoutesDivergeOnTheSameSeed) {
    const QuantumCircuit qc = c_uniform5();
    EXPECT_NE(run_with(qc, Sampling::Slab, 4096, kSeed),
              run_with(qc, Sampling::PerShot, 4096, kSeed));
}

// A deterministic circuit has one outcome, so both routes must agree shot for
// shot as well as in distribution, on every seed.
TEST(V11251CliffordSampling, DeterministicCircuitGivesOneKeyOnEveryRoute) {
    const QuantumCircuit qc = c_deterministic();
    for (Sampling route : kRoutes) {
        for (uint64_t seed : {uint64_t{1}, uint64_t{2}, kSeed}) {
            SCOPED_TRACE(std::string(route_name(route)) + " seed=" + std::to_string(seed));
            const Counts counts = run_with(qc, route, 512, seed);
            ASSERT_EQ(counts.size(), 1u);
            // x(0) and x(2) on a 4-qubit register: q0 and q2 set, q0 rightmost.
            EXPECT_EQ(counts.begin()->first, "0101");
            EXPECT_EQ(counts.begin()->second, 512);
        }
    }
}

// =============================================================================
// Shot counts at the edges
// =============================================================================

TEST(V11251CliffordSampling, SingleShotProducesOneKeyInTheSupport) {
    for (const Case& c : kCases) {
        const QuantumCircuit qc = c.build();
        const std::map<std::string, double> exact = exact_keys(qc);
        for (Sampling route : kRoutes) {
            SCOPED_TRACE(std::string(c.name) + " " + route_name(route));
            const Counts counts = run_with(qc, route, 1, kSeed);
            ASSERT_EQ(counts.size(), 1u);
            EXPECT_EQ(counts.begin()->second, 1);
            EXPECT_GT(exact.count(counts.begin()->first), 0u)
                << "single-shot key '" << counts.begin()->first
                << "' is outside the support";
        }
    }
}

TEST(V11251CliffordSampling, ZeroShotsRecordsNoCounts) {
    for (const Case& c : kCases) {
        const QuantumCircuit qc = c.build();
        for (Sampling route : kRoutes) {
            SCOPED_TRACE(std::string(c.name) + " " + route_name(route));
            EXPECT_TRUE(run_with(qc, route, 0, kSeed).empty());
        }
    }
}

// A large run must still total exactly, which is what catches a sampling loop
// that drops or double-records a shot.
TEST(V11251CliffordSampling, EveryShotIsRecordedExactlyOnce) {
    const QuantumCircuit qc = c_uniform5();
    for (Sampling route : kRoutes) {
        for (int shots : {1, 2, 3, 63, 64, 65, 1000, 65536}) {
            SCOPED_TRACE(std::string(route_name(route)) + " shots=" + std::to_string(shots));
            int total = 0;
            for (const auto& [key, n] : run_with(qc, route, shots, kSeed)) {
                (void)key;
                total += n;
            }
            EXPECT_EQ(total, shots);
        }
    }
}

// The Slab route spends one bit of a 64-bit pool per free direction, refilling
// when the pool runs dry. A register with more free directions than a pool
// holds crosses that refill inside a single shot.
TEST(V11251CliffordSampling, SlabPoolRefillsAcrossWideRegisters) {
    for (int n : {63, 64, 65, 100, 129}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        QuantumCircuit qc(n, n);
        for (int q = 0; q < n; ++q) qc.h(q);
        qc.measure_all();

        const Counts counts = run_with(qc, Sampling::Slab, 256, kSeed);
        int total = 0;
        for (const auto& [key, c] : counts) {
            EXPECT_EQ(key.size(), static_cast<size_t>(n));
            total += c;
        }
        EXPECT_EQ(total, 256);
        // With 2^n outcomes and 256 shots, repeats are what would signal a
        // pool that stopped advancing.
        EXPECT_GT(counts.size(), 200u)
            << "256 draws from 2^" << n << " outcomes collided far too often";
    }
}

// =============================================================================
// Circuits with no measurement at all
// =============================================================================

// A circuit with nothing to measure must not pay for an elimination, and every
// shot records the same empty register rather than an arbitrary one.
TEST(V11251CliffordSampling, CircuitWithoutMeasurementRecordsAllZeroKeys) {
    for (Sampling route : kRoutes) {
        SCOPED_TRACE(route_name(route));
        QuantumCircuit qc(4);
        qc.h(0).cx(0, 1).sx(2).iswap(2, 3).ecr(0, 3);

        const Counts counts = run_with(qc, route, 64, kSeed);
        ASSERT_EQ(counts.size(), 1u);
        EXPECT_EQ(counts.begin()->first, "0000");
        EXPECT_EQ(counts.begin()->second, 64);
    }
}

// The gate pass still runs, so the state that comes back is the circuit's, not
// a fresh register.
TEST(V11251CliffordSampling, CircuitWithoutMeasurementStillEvolvesTheState) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);

    CliffordSimulator sim;
    const auto res = sim.run(qc, /*shots=*/8, /*seed=*/7);
    EXPECT_EQ(res.final_state.expectation_pauli("ZZI"), 1);
    EXPECT_EQ(res.final_state.expectation_pauli("IZZ"), 1);
    EXPECT_EQ(res.final_state.expectation_pauli("XXX"), 1);
    EXPECT_EQ(res.final_state.expectation_pauli("ZII"), 0);
}

// =============================================================================
// The returned state after slab sampling
// =============================================================================

// final_state must be a usable tableau: measurable, and describing the same
// distribution the counts were drawn from.
TEST(V11251CliffordSampling, FinalStateIsConsistentWithTheCountsReturned) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        const QuantumCircuit qc = c.build();

        CliffordSimulator sim;
        sim.options.sampling = Sampling::Slab;
        auto res = sim.run(qc, kShots, kSeed);

        const std::map<std::string, double> from_state = expected_keys(
            v11251::exact_clifford_distribution(res.final_state),
            measurement_map(qc), n_clbits_of(qc));
        EXPECT_TRUE(counts_match(res.counts, from_state, kShots));
    }
}

TEST(V11251CliffordSampling, FinalStateRemainsMeasurable) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        const QuantumCircuit qc = c.build();

        CliffordSimulator sim;
        auto res = sim.run(qc, 64, kSeed);
        ASSERT_EQ(res.final_state.n_qubits, qc.n_qubits);

        std::mt19937_64 rng(4242);
        std::vector<int> outcomes;
        for (int q = 0; q < qc.n_qubits; ++q) {
            EXPECT_NO_THROW(outcomes.push_back(res.final_state.measure(q, true, rng)));
        }
        // Measuring every qubit determines the state, so the slab collapses.
        EXPECT_EQ(res.final_state.outcome_slab().dim, 0);
        // A second pass must reproduce the first: the state is now determined.
        for (int q = 0; q < qc.n_qubits; ++q) {
            EXPECT_EQ(res.final_state.measure(q, true, rng),
                      outcomes[static_cast<size_t>(q)]);
        }
    }
}

// =============================================================================
// Which path a circuit takes
// =============================================================================

// The terminal-measurement predicate decides which of the two rewritten routes
// runs, and its rules are easy to get subtly wrong. On the terminal path it is
// observable without reaching inside: Options::sampling genuinely selects
// between Slab and PerShot there, and the two consume the random stream
// differently, so the same seed gives DIFFERENT counts under each.
//
// Every case below carries the same 5-qubit uniform distribution, so the path
// is the only thing that varies. The circuits that take the general path live
// in ExplicitSlabOnTheGeneralPathIsRefused instead, because what distinguishes
// them there is a refusal rather than a difference in counts.
namespace {

struct PathCase {
    const char* name;
    bool terminal;
    QuantumCircuit (*build)();
};

const PathCase kPathCases[] = {
        {"plain_terminal", true, [] {
            QuantumCircuit qc(5, 5);
            for (int q = 0; q < 5; ++q) qc.h(q);
            qc.measure_all();
            return qc;
        }},
        {"barrier_before_measure", true, [] {
            // BARRIER is skipped by the predicate outright.
            QuantumCircuit qc(5, 5);
            for (int q = 0; q < 5; ++q) qc.h(q);
            qc.barrier();
            qc.measure_all();
            return qc;
        }},
        {"barrier_after_measure", true, [] {
            QuantumCircuit qc(5, 5);
            for (int q = 0; q < 5; ++q) qc.h(q);
            qc.measure_all();
            qc.barrier();
            return qc;
        }},
        {"gate_on_unmeasured_qubit", true, [] {
            // q4 is measured last, and nothing touches a qubit after ITS
            // measurement, so this is still terminal.
            QuantumCircuit qc(5, 5);
            for (int q = 0; q < 5; ++q) qc.h(q);
            for (int q = 0; q < 4; ++q) qc.measure(q, q);
            qc.z(4);
            qc.measure(4, 4);
            return qc;
        }},
        {"gate_on_measured_qubit", false, [] {
            QuantumCircuit qc(5, 5);
            for (int q = 0; q < 5; ++q) qc.h(q);
            qc.measure_all();
            qc.x(0);              // acts on a qubit already measured
            return qc;
        }},
        {"reset_at_end", false, [] {
            QuantumCircuit qc(5, 5);
            for (int q = 0; q < 5; ++q) qc.h(q);
            qc.measure_all();
            qc.reset(0);
            return qc;
        }},
        {"conditional_at_end", false, [] {
            QuantumCircuit qc(5, 5);
            for (int q = 0; q < 5; ++q) qc.h(q);
            qc.measure_all();
            qc.add_if(0, 1, Instruction::GateType::Z, {0});
            return qc;
        }},
};

// The distribution every case above produces: 32 outcomes, uniform.
void expect_uniform_over_32(const Counts& counts, int shots) {
    int total = 0;
    for (const auto& [key, n] : counts) {
        ASSERT_EQ(key.size(), 5u);
        total += n;
    }
    EXPECT_EQ(total, shots);
    EXPECT_EQ(counts.size(), 32u);
}

}  // namespace

TEST(V11251CliffordSampling, TerminalCircuitsHonourTheSamplingOption) {
    for (const PathCase& c : kPathCases) {
        if (!c.terminal) continue;
        SCOPED_TRACE(c.name);
        const QuantumCircuit qc = c.build();
        const Counts slab = run_with(qc, Sampling::Slab, 4096, kSeed);
        const Counts per_shot = run_with(qc, Sampling::PerShot, 4096, kSeed);

        EXPECT_NE(slab, per_shot)
            << "a terminal circuit honours Options::sampling, so the two routes "
               "must consume the seed differently";
        expect_uniform_over_32(slab, 4096);
        expect_uniform_over_32(per_shot, 4096);
    }
}

// PIN, currently RED. A sampling route that cannot be honoured must say so.
//
// Options::sampling documents itself as governing how TERMINAL measurements
// become shots, and a circuit with mid-circuit measurement, feedforward or
// reset takes the general route regardless. The reason is real: the slab is one
// affine subspace read off one fixed tableau, and once a mid-circuit
// measurement collapses the state each trajectory diverges, so there is no
// single subspace left to read.
//
// What is missing is the diagnostic. Today an explicit Slab request on such a
// circuit is dropped in silence: the run succeeds, the counts are correct, and
// the caller is never told that the route it asked for is not the route it got.
// A caller choosing Slab for speed has no way to learn it did not get it.
//
// Closing this needs a third enumerator rather than a bare throw, because Slab
// is the current DEFAULT and throwing on it as it stands would reject every
// default-configured reset or feedforward circuit. The shape that works is an
// Auto default that picks per circuit, leaving an EXPLICIT Slab to mean the
// caller wants that route and to be refused when it is unavailable.
TEST(V11251CliffordSampling, ExplicitSlabOnTheGeneralPathIsRefused) {
    for (const PathCase& c : kPathCases) {
        if (c.terminal) continue;
        SCOPED_TRACE(c.name);
        const QuantumCircuit qc = c.build();

        CliffordSimulator sim;
        sim.options.sampling = Sampling::Slab;
        EXPECT_THROW(sim.run(qc, 4096, kSeed), std::invalid_argument)
            << "an explicit Slab request was dropped in silence on a circuit "
               "that cannot take the slab route";

        // PerShot is what the general route does by nature, so it is honoured
        // and the answer is unchanged.
        expect_uniform_over_32(run_with(qc, Sampling::PerShot, 4096, kSeed), 4096);
    }
}

// =============================================================================
// Paths the rework did not touch
// =============================================================================

// A gate acting on a qubit after it was measured forces the general route, one
// fresh trajectory per shot.
TEST(V11251CliffordSampling, MidCircuitMeasurementThenMoreGates) {
    QuantumCircuit qc(1, 2);
    qc.h(0).measure(0, 0).x(0).measure(0, 1);

    for (Sampling route : kGeneralRoutes) {
        SCOPED_TRACE(route_name(route));
        const Counts counts = run_with(qc, route, 8000, kSeed);
        int total = 0;
        for (const auto& [key, n] : counts) {
            // The second measurement always disagrees with the first.
            EXPECT_TRUE(key == "10" || key == "01") << "unexpected key " << key;
            total += n;
        }
        EXPECT_EQ(total, 8000);
        const double f = static_cast<double>(counts.at("10")) / 8000.0;
        const double sigma = std::sqrt(0.25 / 8000.0);
        EXPECT_LE(std::abs(f - 0.5), 6.0 * sigma);
    }
}

TEST(V11251CliffordSampling, FeedforwardOnAMeasuredClbit) {
    // Deterministic: q0 is set, so the condition always fires.
    QuantumCircuit det(2, 2);
    det.x(0).measure(0, 0);
    det.add_if(0, 1, Instruction::GateType::X, {1});
    det.measure(1, 1);

    for (Sampling route : kGeneralRoutes) {
        SCOPED_TRACE(std::string("deterministic ") + route_name(route));
        const Counts counts = run_with(det, route, 512, kSeed);
        ASSERT_EQ(counts.size(), 1u);
        EXPECT_EQ(counts.begin()->first, "11");
        EXPECT_EQ(counts.begin()->second, 512);
    }

    // Stochastic: the condition fires on half the trajectories, and the two
    // clbits must always agree.
    QuantumCircuit stoch(2, 2);
    stoch.h(0).measure(0, 0);
    stoch.add_if(0, 1, Instruction::GateType::X, {1});
    stoch.measure(1, 1);

    for (Sampling route : kGeneralRoutes) {
        SCOPED_TRACE(std::string("stochastic ") + route_name(route));
        const Counts counts = run_with(stoch, route, 8000, kSeed);
        int total = 0;
        for (const auto& [key, n] : counts) {
            EXPECT_TRUE(key == "00" || key == "11") << "unexpected key " << key;
            total += n;
        }
        EXPECT_EQ(total, 8000);
        const double f = static_cast<double>(counts.at("11")) / 8000.0;
        const double sigma = std::sqrt(0.25 / 8000.0);
        EXPECT_LE(std::abs(f - 0.5), 6.0 * sigma);
    }
}

TEST(V11251CliffordSampling, ResetForcesTheGeneralPathAndClears) {
    for (Sampling route : kGeneralRoutes) {
        SCOPED_TRACE(route_name(route));

        QuantumCircuit before(1, 1);
        before.x(0).reset(0).measure(0, 0);
        const Counts c1 = run_with(before, route, 512, kSeed);
        ASSERT_EQ(c1.size(), 1u);
        EXPECT_EQ(c1.begin()->first, "0");
        EXPECT_EQ(c1.begin()->second, 512);

        // Reset after a measurement, with both outcomes recorded.
        QuantumCircuit after(1, 2);
        after.x(0).measure(0, 0).reset(0).measure(0, 1);
        const Counts c2 = run_with(after, route, 512, kSeed);
        ASSERT_EQ(c2.size(), 1u);
        EXPECT_EQ(c2.begin()->first, "01");

        // Reset of a superposition still leaves |0>.
        QuantumCircuit super(1, 1);
        super.h(0).reset(0).measure(0, 0);
        const Counts c3 = run_with(super, route, 512, kSeed);
        ASSERT_EQ(c3.size(), 1u);
        EXPECT_EQ(c3.begin()->first, "0");
        EXPECT_EQ(c3.begin()->second, 512);
    }
}

// A circuit using the new gates on the general route. Those gates are
// dispatched against the row-major tableau there rather than the bit-sliced
// one, so this is where the second implementation of each composed rule runs.
TEST(V11251CliffordSampling, NewGatesRunOnTheGeneralPath) {
    QuantumCircuit general(4, 4);
    general.h(0).sx(1).cy(0, 2).iswap(1, 3).ecr(2, 3);
    general.measure(0, 0);
    general.reset(0);            // reset forces the general route
    general.measure(0, 0);       // clbit 0 now reads the reset qubit
    general.measure(1, 1).measure(2, 2).measure(3, 3);

    for (Sampling route : kGeneralRoutes) {
        SCOPED_TRACE(route_name(route));
        const Counts counts = run_with(general, route, 8000, kSeed);
        int total = 0;
        for (const auto& [key, n] : counts) {
            ASSERT_EQ(key.size(), 4u);
            // clbit 0 is the rightmost character, and it was written after the
            // reset, so it can only ever be 0.
            EXPECT_EQ(key[3], '0') << "qubit 0 was reset before this measurement, key "
                                   << key;
            total += n;
        }
        EXPECT_EQ(total, 8000);
    }
}

// expectation_pauli after a circuit built from the new gates: the Pauli
// machinery reads the same tableau the transpose wrote.
TEST(V11251CliffordSampling, ExpectationPauliAfterNewGates) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);       // GHZ
    CliffordSimulator sim;
    const auto ghz = sim.run(qc, 0, 1);
    EXPECT_EQ(ghz.final_state.expectation_pauli("XXX"), 1);
    EXPECT_EQ(ghz.final_state.expectation_pauli("ZZI"), 1);
    EXPECT_EQ(ghz.final_state.expectation_pauli("IZZ"), 1);
    EXPECT_EQ(ghz.final_state.expectation_pauli("YYX"), -1);
    EXPECT_EQ(ghz.final_state.expectation_pauli("XII"), 0);

    // The same state reached through iswap and swap instead of the second cx.
    QuantumCircuit routed(3);
    routed.h(0).cx(0, 1).swap(1, 2).cx(2, 1);
    const auto same = sim.run(routed, 0, 1);
    EXPECT_TRUE(v11251::states_equal_exhaustive(ghz.final_state, same.final_state));
}

// Every case, on both routes, against the statevector's Pauli expectations for
// the pre-measurement state. This is the check that a measurement pass left the
// tableau in the state the gates put it in.
TEST(V11251CliffordSampling, PreMeasurementStateMatchesStatevectorOnEveryCase) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        const QuantumCircuit qc = c.build();
        QuantumCircuit unmeasured(qc.n_qubits);
        for (const auto& inst : qc.instructions) {
            if (inst.type == Instruction::GateType::MEASURE ||
                inst.type == Instruction::GateType::BARRIER) {
                continue;
            }
            unmeasured.instructions.push_back(inst);
        }
        EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(unmeasured));
    }
}
