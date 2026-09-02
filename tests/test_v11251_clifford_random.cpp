// 1.1.25.1 test wave - randomised Clifford circuits against every oracle.
//
// Every other suite in this wave states a property and then picks the circuits
// that exercise it, which means each one only ever visits gate orderings
// somebody thought of. This file inverts that: it draws circuits from the whole
// dispatch uniformly and asserts the same properties over whatever comes out.
// It is the part of the wave that can find a defect nobody anticipated.
//
// Every draw is seeded, so a failure here is reproducible and names the circuit
// that produced it rather than being a run-to-run coin flip. The seeds are
// fixed constants and the generator is std::mt19937_64, so the corpus is the
// same on every platform and every run.
//
// The oracles are the strongest each size admits: all 4^n Pauli expectations
// where n allows it (a complete comparison, not a probe), the structural
// fingerprint at the packing boundaries, and the exact statevector distribution
// wherever a distribution is compared.

#include <gtest/gtest.h>

#include "v11251_clifford_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/clifford_sim.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;

namespace {

using v11251::apply_step;
using v11251::is_two_qubit;
using v11251::kLayoutOps;
using v11251::Op;
using v11251::Step;

constexpr int kLayoutOpCount = 14;

// A script over the gates both tableau layouts implement. Operands are drawn
// so a two-qubit gate never repeats one, which is the precondition its
// distinctness guard enforces.
std::vector<Step> random_script(int n, int depth, std::mt19937_64& rng) {
    std::vector<Step> script;
    script.reserve(static_cast<size_t>(depth));
    std::uniform_int_distribution<int> pick_op(0, kLayoutOpCount - 1);
    std::uniform_int_distribution<int> pick_qubit(0, n - 1);
    for (int d = 0; d < depth; ++d) {
        Op op = kLayoutOps[pick_op(rng)];
        const int a = pick_qubit(rng);
        if (is_two_qubit(op)) {
            if (n < 2) { op = Op::H; script.push_back({op, a, -1}); continue; }
            int b = a;
            while (b == a) b = pick_qubit(rng);
            script.push_back({op, a, b});
        } else {
            script.push_back({op, a, -1});
        }
    }
    return script;
}

// A circuit over the whole dispatch, including the rotations, which the script
// form cannot reach because neither tableau layout takes an angle. Angles are
// drawn from the four quarter turns, so every circuit is Clifford by
// construction and is_clifford() must say so.
QuantumCircuit random_circuit(int n, int depth, std::mt19937_64& rng) {
    QuantumCircuit qc(n);
    std::uniform_int_distribution<int> pick_op(0, 17);
    std::uniform_int_distribution<int> pick_qubit(0, n - 1);
    std::uniform_int_distribution<int> pick_turn(0, 3);

    for (int d = 0; d < depth; ++d) {
        int g = pick_op(rng);
        const int a = pick_qubit(rng);
        int b = a;
        if (n >= 2) {
            while (b == a) b = pick_qubit(rng);
        } else if (g >= 8 && g <= 13) {
            g = g - 8;  // a one-qubit register has no two-qubit gate to apply
        }
        const double angle = static_cast<double>(pick_turn(rng)) * PI_2;

        switch (g) {
            case 0:  qc.h(a); break;
            case 1:  qc.s(a); break;
            case 2:  qc.sdg(a); break;
            case 3:  qc.x(a); break;
            case 4:  qc.y(a); break;
            case 5:  qc.z(a); break;
            case 6:  qc.sx(a); break;
            case 7:  qc.sxdg(a); break;
            case 8:  qc.cx(a, b); break;
            case 9:  qc.cy(a, b); break;
            case 10: qc.cz(a, b); break;
            case 11: qc.swap(a, b); break;
            case 12: qc.iswap(a, b); break;
            case 13: qc.ecr(a, b); break;
            case 14: qc.rx(angle, a); break;
            case 15: qc.ry(angle, a); break;
            case 16: qc.rz(angle, a); break;
            default: qc.p(angle, a); break;
        }
    }
    return qc;
}

// A circuit carrying the constructs that force the general per-shot route:
// mid-circuit measurement with gates after it, reset, and feedforward. The
// clbit a condition reads is always one that has been written by then.
QuantumCircuit random_general_circuit(int n, int depth, std::mt19937_64& rng) {
    QuantumCircuit qc(n, n);
    std::uniform_int_distribution<int> pick_qubit(0, n - 1);
    std::uniform_int_distribution<int> pick_kind(0, 9);
    int written = 0;  // clbits 0..written-1 hold a measured value

    for (int d = 0; d < depth; ++d) {
        const int a = pick_qubit(rng);
        int b = a;
        if (n >= 2) while (b == a) b = pick_qubit(rng);

        switch (pick_kind(rng)) {
            case 0: qc.h(a); break;
            case 1: qc.s(a); break;
            case 2: qc.sx(a); break;
            case 3: qc.x(a); break;
            case 4: if (n >= 2) qc.cx(a, b); else qc.h(a); break;
            case 5: if (n >= 2) qc.cy(a, b); else qc.z(a); break;
            case 6:
                if (written < n) { qc.measure(a, written); ++written; }
                else { qc.h(a); }
                break;
            case 7: qc.reset(a); break;
            case 8:
                if (written > 0) {
                    std::uniform_int_distribution<int> pick_c(0, written - 1);
                    qc.add_if(pick_c(rng), 1, Instruction::GateType::X, {a});
                } else {
                    qc.y(a);
                }
                break;
            default: qc.z(a); break;
        }
    }
    // Leave every clbit written, so the key width is the same on both backends.
    for (int q = written; q < n; ++q) qc.measure(q, q);
    return qc;
}

std::string random_pauli(int n, std::mt19937_64& rng) {
    static const char letters[4] = {'I', 'X', 'Y', 'Z'};
    std::uniform_int_distribution<int> pick(0, 3);
    std::string p(static_cast<size_t>(n), 'I');
    for (int q = 0; q < n; ++q) p[static_cast<size_t>(q)] = letters[pick(rng)];
    return p;
}

// Total variation distance between a sampled distribution and a reference one.
double tv_distance(const std::unordered_map<std::string, int>& counts, int shots,
                   const std::unordered_map<std::string, int>& reference) {
    double tv = 0.0;
    for (const auto& [key, n] : counts) {
        const auto it = reference.find(key);
        const double q = (it == reference.end()) ? 0.0
                                                 : static_cast<double>(it->second) / shots;
        tv += std::abs(static_cast<double>(n) / shots - q);
    }
    for (const auto& [key, n] : reference) {
        if (counts.find(key) == counts.end()) tv += static_cast<double>(n) / shots;
    }
    return 0.5 * tv;
}

// Seeds are per test, so one test's corpus cannot shift when another changes,
// and each is offset by n so no two sizes draw the same circuits.
constexpr uint64_t kSeedCircuits = 110250001;
constexpr uint64_t kSeedScripts  = 110250002;
constexpr uint64_t kSeedSlab     = 110250003;
constexpr uint64_t kSeedSampling = 110250004;
constexpr uint64_t kSeedElim     = 110250005;
constexpr uint64_t kSeedMeasure  = 110250006;
constexpr uint64_t kSeedPauli    = 110250007;
constexpr uint64_t kSeedGeneral  = 110250008;
constexpr uint64_t kSeedReject   = 110250009;

}  // namespace

// =============================================================================
// Random circuits against the statevector
// =============================================================================

// The strongest statement this file makes: for a random Clifford circuit over
// the entire dispatch, EVERY Pauli expectation agrees with the statevector.
// Agreement on all 4^n Paulis is equality of the states up to the global phase
// the stabilizer formalism does not represent.
TEST(V11251CliffordRandom, RandomCircuitsMatchStatevectorExhaustively) {
    struct Plan { int n; int circuits; int depth; };
    const Plan plans[] = {
        {1, 120, 20}, {2, 120, 30}, {3, 120, 40}, {4, 100, 50}, {5, 80, 60},
    };
    for (const Plan& p : plans) {
        std::mt19937_64 rng(kSeedCircuits + static_cast<uint64_t>(p.n));
        for (int i = 0; i < p.circuits; ++i) {
            const QuantumCircuit qc = random_circuit(p.n, p.depth, rng);
            SCOPED_TRACE("n=" + std::to_string(p.n) + " circuit " + std::to_string(i));
            ASSERT_TRUE(v11251::clifford_matches_statevector_exhaustive(qc));
        }
    }
}

// Every drawn circuit is Clifford by construction, so the classifier must
// accept all of them and the dispatch must execute all of them. A rejection
// here would be a circuit needlessly pushed onto a slower simulator.
TEST(V11251CliffordRandom, RandomCircuitsAreAcceptedAndExecutable) {
    std::mt19937_64 rng(kSeedCircuits);
    CliffordSimulator sim;
    for (int n : {1, 2, 3, 5, 8}) {
        for (int i = 0; i < 60; ++i) {
            QuantumCircuit qc = random_circuit(n, 80, rng);
            SCOPED_TRACE("n=" + std::to_string(n) + " circuit " + std::to_string(i));
            EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
            EXPECT_NO_THROW(sim.run(qc, /*shots=*/2, /*seed=*/1));
        }
    }
}

// =============================================================================
// Random scripts across the two tableau layouts
// =============================================================================

// The bit-sliced gate pass and the row-major one implement each rule
// separately, so a random script is the sharpest test of the pair: it visits
// orderings and operand pairings no hand-written script covers.
TEST(V11251CliffordRandom, RandomScriptsProvedEqualAtSmallSizes) {
    for (int n : {1, 2, 3, 4, 5}) {
        std::mt19937_64 rng(kSeedScripts + static_cast<uint64_t>(n));
        for (int i = 0; i < 80; ++i) {
            const std::vector<Step> script = random_script(n, 60, rng);
            SCOPED_TRACE("n=" + std::to_string(n) + " script " + std::to_string(i));
            ASSERT_TRUE(v11251::states_equal_exhaustive(
                v11251::run_bit_sliced(n, script), v11251::run_row_major(n, script)));
        }
    }
}

// The same at the packing boundaries, where the padding word and the 64x64
// block walk are what the two layouts disagree about if anything does.
TEST(V11251CliffordRandom, RandomScriptsAgreeAtBoundarySizes) {
    for (int n : {31, 32, 33, 63, 64, 65, 95, 96, 127, 128, 129, 160}) {
        std::mt19937_64 rng(kSeedScripts + static_cast<uint64_t>(n));
        for (int i = 0; i < 12; ++i) {
            const std::vector<Step> script = random_script(n, 400, rng);
            SCOPED_TRACE("n=" + std::to_string(n) + " script " + std::to_string(i));
            ASSERT_TRUE(v11251::states_equal_structural(
                v11251::run_bit_sliced(n, script), v11251::run_row_major(n, script)));
        }
    }
}

// =============================================================================
// Random states through the outcome slab
// =============================================================================

// The slab claims to describe the whole computational-basis distribution. On a
// random state that claim is checked against the statevector exactly: same
// support, same probabilities, no sampling anywhere.
TEST(V11251CliffordRandom, RandomStatesSlabMatchesStatevectorExactly) {
    CliffordSimulator sim;
    for (int n : {2, 3, 4, 5}) {
        std::mt19937_64 rng(kSeedSlab + static_cast<uint64_t>(n));
        for (int i = 0; i < 80; ++i) {
            const QuantumCircuit qc = random_circuit(n, 50, rng);
            SCOPED_TRACE("n=" + std::to_string(n) + " circuit " + std::to_string(i));
            const StabilizerState st = sim.run(qc, /*shots=*/0, /*seed=*/1).final_state;

            const v11251::Dist from_slab = v11251::exact_clifford_distribution(st);
            ASSERT_TRUE(v11251::distributions_equal(
                from_slab, v11251::exact_statevector_distribution(qc)));

            // Uniform over a coset of size 2^dim, which is what makes a shot a
            // draw of dim bits rather than a measurement pass.
            const int dim = st.outcome_slab().dim;
            ASSERT_EQ(from_slab.size(), static_cast<size_t>(1) << dim);
            for (const auto& [outcome, p] : from_slab) {
                (void)outcome;
                EXPECT_NEAR(p, std::ldexp(1.0, -dim), 1e-12);
            }
        }
    }
}

// Both eliminations reduce to the same canonical form, so on a random state
// they must agree bit for bit. Random pivot patterns are exactly what a block
// method can get wrong while every structured case still passes.
TEST(V11251CliffordRandom, BothEliminationsAgreeOnRandomStates) {
    for (int n : {2, 5, 8, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129}) {
        std::mt19937_64 rng(kSeedElim + static_cast<uint64_t>(n));
        for (int i = 0; i < 12; ++i) {
            const std::vector<Step> script = random_script(n, 6 * n, rng);
            const StabilizerState st = v11251::run_row_major(n, script);
            SCOPED_TRACE("n=" + std::to_string(n) + " state " + std::to_string(i));

            const v11251::SlabKey plain =
                v11251::slab_key(st, StabilizerState::Elimination::Plain);
            const v11251::SlabKey block =
                v11251::slab_key(st, StabilizerState::Elimination::FourRussians);
            ASSERT_EQ(plain.dim, block.dim);
            EXPECT_EQ(plain.offset, block.offset);
            EXPECT_EQ(plain.basis, block.basis);
        }
    }
}

// =============================================================================
// Random circuits through both sampling routes
// =============================================================================

// A drawn circuit with terminal measurements, sampled on both routes and
// checked against the exact distribution the slab describes. Support is exact:
// a key the state cannot produce fails regardless of how rarely it appeared.
TEST(V11251CliffordRandom, RandomCircuitsSampleTheExactDistributionOnBothRoutes) {
    using Sampling = CliffordSimulator::Options::Sampling;
    constexpr int shots = 8000;

    for (int n : {2, 3, 4}) {
        std::mt19937_64 rng(kSeedSampling + static_cast<uint64_t>(n));
        for (int i = 0; i < 12; ++i) {
            QuantumCircuit qc = random_circuit(n, 40, rng);
            QuantumCircuit measured(n, n);
            for (const auto& inst : qc.instructions) measured.instructions.push_back(inst);
            measured.measure_all();

            CliffordSimulator ref;
            const v11251::Dist exact = v11251::exact_clifford_distribution(
                ref.run(qc, /*shots=*/0, /*seed=*/1).final_state);

            for (Sampling route : {Sampling::Slab, Sampling::PerShot}) {
                SCOPED_TRACE("n=" + std::to_string(n) + " circuit " + std::to_string(i) +
                             (route == Sampling::Slab ? " Slab" : " PerShot"));
                CliffordSimulator sim;
                sim.options.sampling = route;
                const auto counts = sim.run(measured, shots, /*seed=*/99 + i).counts;

                int total = 0;
                for (const auto& [key, c] : counts) {
                    total += c;
                    const uint64_t outcome = std::stoull(key, nullptr, 2);
                    ASSERT_GT(exact.count(outcome), 0u)
                        << "key '" << key << "' is outside the exact support";
                    // Six binomial sigmas around the true probability.
                    const double p = exact.at(outcome);
                    const double f = static_cast<double>(c) / shots;
                    const double sigma = v11251::binomial_sigma(p, shots);
                    EXPECT_LE(std::abs(f - p), 6.0 * sigma + 1e-9)
                        << "key '" << key << "' frequency " << f << " vs " << p;
                }
                EXPECT_EQ(total, shots);
                EXPECT_EQ(counts.size(), exact.size())
                    << "every outcome with probability must appear in " << shots
                    << " shots";
            }
        }
    }
}

// =============================================================================
// Gate algebra
// =============================================================================

// Each gate's ORDER: the smallest k for which G^k is the identity MATRIX. The
// stabilizer formalism drops global phase, so what is pinned is equality up to
// phase, which is what the tableau is allowed to represent.
//
// The orders are properties of the gate matrices rather than of this backend:
// S and SX are fourth roots (S^2 = Z, SX^2 = X), ISWAP is a fourth root because
// ISWAP^2 = diag(1, -1, -1, 1), and every other gate here is a Hermitian
// unitary and therefore an involution.
//
// The two halves are quantified differently, and mixing them up is easy:
//
//   G^order = I is UNIVERSAL. It must hold on every state, so it is asserted
//   inside the loop over states.
//
//   Minimality is EXISTENTIAL. G^j for j < order is not the identity matrix,
//   but it can still fix any particular state: X leaves every X-eigenstate
//   alone, CX leaves |00> alone, SWAP leaves anything symmetric in its operands
//   alone. So a state where G^j moves the tableau has to be WITNESSED across
//   the corpus, never demanded of each state.
TEST(V11251CliffordRandom, EveryGateHasTheOrderItsMatrixHas) {
    struct Row { Op op; int order; };
    const Row rows[] = {
        {Op::H, 2}, {Op::X, 2}, {Op::Y, 2}, {Op::Z, 2},
        {Op::S, 4}, {Op::SDG, 4}, {Op::SX, 4}, {Op::SXDG, 4},
        {Op::CX, 2}, {Op::CY, 2}, {Op::CZ, 2}, {Op::SWAP, 2},
        {Op::ISWAP, 4}, {Op::ECR, 2},
    };
    constexpr int n = 3;
    constexpr int kStates = 32;
    std::mt19937_64 rng(kSeedScripts + 900);

    for (const Row& row : rows) {
        for (int a = 0; a < n; ++a) {
            const int b = (a + 1) % n;
            SCOPED_TRACE(std::string(v11251::op_name(row.op)) +
                         " a=" + std::to_string(a) + " b=" + std::to_string(b));

            // moved[j] records that SOME state in the corpus was moved by G^j.
            std::vector<bool> moved(static_cast<size_t>(row.order), false);

            for (int s = 0; s < kStates; ++s) {
                const std::vector<Step> prefix = random_script(n, 40, rng);
                const StabilizerState start = v11251::run_row_major(n, prefix);
                StabilizerState st = start;

                for (int k = 1; k <= row.order; ++k) {
                    apply_step(st, {row.op, a, b});
                    const bool back =
                        static_cast<bool>(v11251::states_equal_exhaustive(st, start));
                    if (k == row.order) {
                        ASSERT_TRUE(back)
                            << "state " << s << " did not return after "
                            << row.order << " applications";
                    } else if (!back) {
                        moved[static_cast<size_t>(k)] = true;
                    }
                }
            }

            for (int j = 1; j < row.order; ++j) {
                EXPECT_TRUE(moved[static_cast<size_t>(j)])
                    << "G^" << j << " fixed every one of the " << kStates
                    << " states tried, so the gate's order is below " << row.order;
            }
        }
    }
}

// A random script followed by its inverse must return the state exactly. This
// is sharper than any single gate's order: it holds only if every gate's bit
// rule AND its sign rule invert correctly in an arbitrary context.
TEST(V11251CliffordRandom, ScriptFollowedByItsInverseIsTheIdentity) {
    auto inverse_of = [](const Step& s) {
        std::vector<Step> out;
        switch (s.op) {
            case Op::S:    out.push_back({Op::SDG, s.a, s.b}); break;
            case Op::SDG:  out.push_back({Op::S, s.a, s.b}); break;
            case Op::SX:   out.push_back({Op::SXDG, s.a, s.b}); break;
            case Op::SXDG: out.push_back({Op::SX, s.a, s.b}); break;
            // A fourth root, so its inverse is three more of it.
            case Op::ISWAP:
                for (int i = 0; i < 3; ++i) out.push_back({Op::ISWAP, s.a, s.b});
                break;
            default: out.push_back(s); break;  // the involutions
        }
        return out;
    };

    for (int n : {1, 2, 3, 4, 5}) {
        std::mt19937_64 rng(kSeedScripts + 500 + static_cast<uint64_t>(n));
        for (int i = 0; i < 40; ++i) {
            const std::vector<Step> script = random_script(n, 50, rng);
            std::vector<Step> round_trip = script;
            for (auto it = script.rbegin(); it != script.rend(); ++it) {
                for (const Step& inv : inverse_of(*it)) round_trip.push_back(inv);
            }
            SCOPED_TRACE("n=" + std::to_string(n) + " script " + std::to_string(i));
            ASSERT_TRUE(v11251::states_equal_exhaustive(
                v11251::run_row_major(n, round_trip), StabilizerState(n)));
            ASSERT_TRUE(v11251::states_equal_exhaustive(
                v11251::run_bit_sliced(n, round_trip), StabilizerState(n)));
        }
    }
}

// =============================================================================
// measure()
// =============================================================================

// Measuring the same qubit twice gives the same answer: the first measurement
// collapses it, so every later one takes the deterministic branch.
TEST(V11251CliffordRandom, MeasurementIsRepeatableOnRandomStates) {
    for (int n : {1, 2, 3, 5, 8}) {
        std::mt19937_64 rng(kSeedMeasure + static_cast<uint64_t>(n));
        for (int i = 0; i < 30; ++i) {
            const std::vector<Step> script = random_script(n, 40, rng);
            StabilizerState st = v11251::run_row_major(n, script);
            SCOPED_TRACE("n=" + std::to_string(n) + " state " + std::to_string(i));

            std::mt19937_64 mrng(kSeedMeasure + 7);
            for (int q = 0; q < n; ++q) {
                const int first = st.measure(q, true, mrng);
                for (int again = 0; again < 3; ++again) {
                    ASSERT_EQ(st.measure(q, true, mrng), first)
                        << "qubit " << q << " changed its answer after collapsing";
                }
            }
            EXPECT_EQ(st.outcome_slab().dim, 0)
                << "every qubit was measured, so nothing is free";
        }
    }
}

// The documented contract on the generator: it is consumed ONLY when the
// outcome is random. A caller interleaving measurements with its own draws
// depends on this, and it is observable by comparing against an untouched
// generator carrying the same seed.
TEST(V11251CliffordRandom, DeterministicMeasurementDoesNotConsumeTheGenerator) {
    for (int n : {1, 3, 8}) {
        SCOPED_TRACE("determined n=" + std::to_string(n));
        StabilizerState st(n);  // |0...0>: every qubit is determined
        std::mt19937_64 used(12345), untouched(12345);
        for (int q = 0; q < n; ++q) EXPECT_EQ(st.measure(q, true, used), 0);
        EXPECT_EQ(used(), untouched())
            << "a determined outcome must not draw from the generator";
    }

    for (int n : {1, 3, 8}) {
        SCOPED_TRACE("random n=" + std::to_string(n));
        StabilizerState st(n);
        st.apply_h(0);
        std::mt19937_64 used(12345), untouched(12345);
        (void)st.measure(0, true, used);
        EXPECT_NE(used(), untouched())
            << "a random outcome must draw from the generator";
    }
}

// The marginal probability of each qubit reading 1, estimated over independent
// copies of the same state, against the statevector's exact marginal.
TEST(V11251CliffordRandom, MeasurementMarginalsMatchTheStatevector) {
    constexpr int trials = 4000;
    CliffordSimulator sim;
    for (int n : {2, 3, 4}) {
        std::mt19937_64 rng(kSeedMeasure + 100 + static_cast<uint64_t>(n));
        for (int i = 0; i < 6; ++i) {
            const QuantumCircuit qc = random_circuit(n, 40, rng);
            const StabilizerState base = sim.run(qc, /*shots=*/0, /*seed=*/1).final_state;
            const v11251::Dist exact = v11251::exact_statevector_distribution(qc);

            for (int q = 0; q < n; ++q) {
                double p_one = 0.0;
                for (const auto& [outcome, p] : exact) {
                    if ((outcome >> q) & 1ULL) p_one += p;
                }
                // Summing the exact probabilities can land a few ulps outside
                // [0, 1], and a marginal of exactly 0 or 1 is the common case
                // here, so the bound is taken from the clamped value.
                if (p_one > 1.0) p_one = 1.0;
                std::mt19937_64 mrng(9000 + static_cast<uint64_t>(q));
                int ones = 0;
                for (int t = 0; t < trials; ++t) {
                    StabilizerState copy = base;
                    ones += copy.measure(q, true, mrng);
                }
                const double f = static_cast<double>(ones) / trials;
                const double sigma = v11251::binomial_sigma(p_one, trials);
                SCOPED_TRACE("n=" + std::to_string(n) + " circuit " +
                             std::to_string(i) + " q=" + std::to_string(q));
                EXPECT_LE(std::abs(f - p_one), 6.0 * sigma + 1e-9)
                    << "marginal " << f << " vs exact " << p_one;
            }
        }
    }
}

// =============================================================================
// expectation_pauli
// =============================================================================

// Random Paulis on random states, at widths where enumerating all 4^n is out of
// reach. The statevector reference is exact, so any disagreement is real.
TEST(V11251CliffordRandom, ExpectationPauliMatchesStatevectorOnRandomPaulis) {
    CliffordSimulator sim;
    for (int n : {6, 8, 10}) {
        std::mt19937_64 rng(kSeedPauli + static_cast<uint64_t>(n));
        for (int i = 0; i < 12; ++i) {
            const QuantumCircuit qc = random_circuit(n, 60, rng);
            const StabilizerState st = sim.run(qc, /*shots=*/0, /*seed=*/1).final_state;
            StatevectorSimulator ss;
            const auto sr = ss.run(qc, /*shots=*/0, /*seed=*/1);

            for (int k = 0; k < 40; ++k) {
                const std::string p = random_pauli(n, rng);
                SCOPED_TRACE("n=" + std::to_string(n) + " circuit " +
                             std::to_string(i) + " pauli " + p);
                ASSERT_EQ(st.expectation_pauli(p),
                          v11251::sv_expectation_pauli_sign(sr.final_state, p));
            }
        }
    }
}

// The identity string is in every stabilizer group with a + sign, and every
// expectation is one of exactly three values.
TEST(V11251CliffordRandom, ExpectationPauliRangeAndIdentity) {
    for (int n : {1, 2, 5, 9}) {
        std::mt19937_64 rng(kSeedPauli + 500 + static_cast<uint64_t>(n));
        for (int i = 0; i < 20; ++i) {
            const std::vector<Step> script = random_script(n, 60, rng);
            const StabilizerState st = v11251::run_row_major(n, script);
            SCOPED_TRACE("n=" + std::to_string(n) + " state " + std::to_string(i));

            EXPECT_EQ(st.expectation_pauli(std::string(static_cast<size_t>(n), 'I')), 1);
            for (int k = 0; k < 30; ++k) {
                const int e = st.expectation_pauli(random_pauli(n, rng));
                EXPECT_TRUE(e == -1 || e == 0 || e == 1)
                    << "expectation " << e << " is outside {-1, 0, +1}";
            }
        }
    }
}

// =============================================================================
// The general per-shot route
// =============================================================================

// Random circuits carrying mid-circuit measurement, reset and feedforward, so
// every one of them takes the general route. The statevector runs the identical
// circuit through its own per-shot trajectories, an implementation that shares
// no code with this one.
//
// PerShot is selected explicitly: the general route is per-shot by nature, and
// a Slab request there is not a route this backend can take.
TEST(V11251CliffordRandom, GeneralPathRandomCircuitsMatchStatevector) {
    constexpr int shots = 6000;
    for (int n : {2, 3, 4}) {
        std::mt19937_64 rng(kSeedGeneral + static_cast<uint64_t>(n));
        for (int i = 0; i < 8; ++i) {
            const QuantumCircuit qc = random_general_circuit(n, 30, rng);
            SCOPED_TRACE("n=" + std::to_string(n) + " circuit " + std::to_string(i));
            ASSERT_TRUE(CliffordSimulator::is_clifford(qc));

            CliffordSimulator cs;
            cs.options.sampling = CliffordSimulator::Options::Sampling::PerShot;
            const auto cliff = cs.run(qc, shots, /*seed=*/4242);
            StatevectorSimulator ss;
            const auto sv = ss.run(qc, shots, /*seed=*/4242);

            int total = 0;
            for (const auto& [key, n_key] : cliff.counts) {
                EXPECT_EQ(key.size(), static_cast<size_t>(n));
                total += n_key;
            }
            EXPECT_EQ(total, shots);
            EXPECT_LT(tv_distance(cliff.counts, shots, sv.counts), 0.06)
                << "tableau and statevector disagree on the general route";
        }
    }
}

// =============================================================================
// Rejection
// =============================================================================

// A random Clifford circuit with one non-Clifford instruction spliced in must
// be rejected by the classifier and must throw on a direct run. Both arms are
// covered: a gate outside the group, and a rotation off the quarter-turn grid.
TEST(V11251CliffordRandom, OneNonCliffordInstructionSpoilsTheWholeCircuit) {
    std::mt19937_64 rng(kSeedReject);
    CliffordSimulator sim;
    for (int n : {1, 2, 4}) {
        for (int i = 0; i < 20; ++i) {
            const QuantumCircuit clean = random_circuit(n, 30, rng);
            ASSERT_TRUE(CliffordSimulator::is_clifford(clean));

            std::uniform_int_distribution<int> pick_qubit(0, n - 1);
            const int q = pick_qubit(rng);

            {
                QuantumCircuit spoiled(n);
                for (const auto& inst : clean.instructions)
                    spoiled.instructions.push_back(inst);
                spoiled.t(q);
                SCOPED_TRACE("n=" + std::to_string(n) + " t-gate " + std::to_string(i));
                EXPECT_FALSE(CliffordSimulator::is_clifford(spoiled));
                EXPECT_THROW(sim.run(spoiled, 2, 1), std::invalid_argument);
            }

            {
                QuantumCircuit spoiled(n);
                for (const auto& inst : clean.instructions)
                    spoiled.instructions.push_back(inst);
                spoiled.rz(PI_2 / 3.0, q);
                SCOPED_TRACE("n=" + std::to_string(n) + " angle " + std::to_string(i));
                EXPECT_FALSE(CliffordSimulator::is_clifford(spoiled));
                EXPECT_THROW(sim.run(spoiled, 2, 1), std::runtime_error);
            }
        }
    }
}
