// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.28.1 test wave - the bond sweep's corpus binds every cap it names (#100).
//
// The MPS comparison benchmark swept bond caps 8, 16, 32 and 64 on a circuit
// whose Schmidt rank peaked at 2. No cap ever bound, so the four sweep points
// were one measurement published under four labels. This is the test that
// would have caught it, and without it nothing stops the same shape of defect
// returning the next time the corpus is regenerated.
//
// The criterion is threshold-free. A cap C binds when the run at C differs
// from the run at C + 1, in bond profile or in accumulated discarded weight.
// Both runs apply the same gates in the same order with the same kernel, and
// the cap is the only input that differs, so if no split ever has more than C
// singular values above cutoff the two runs keep identical values everywhere
// and are bit-identical. If some split has C + 1 or more, cap C keeps C and
// cap C + 1 keeps C + 1, and the two diverge from that split on.
//
// That is a stricter question than whether the rank REACHED the cap. A circuit
// whose rank lands exactly on C produces a peak bond of C while discarding
// nothing, and that run is as free of truncation work as one whose cap never
// came close. The miniature below builds exactly that case, because a test
// that cannot tell the two apart would have passed the original defect.
//
// Binding is monotone downward: the first split whose rank exceeds 32 is at
// or before the first that exceeds 64, and every run is exact up to its own
// first binding split, so if 64 binds then 32, 16 and 8 bind. The top pair is
// therefore the load-bearing assertion. The lower pairs are cheap and are kept
// because when the top fails they say where the rank stopped.
//
// Cost: the corpus circuit at chi = 64 is the expensive run in this file, on
// the order of seconds. It runs always. The alternative, an opt-in gate, would
// skip the one assertion this release exists for on every ordinary run.
//
// Two properties of the reading, both conservative in the direction that
// matters. The bond profile is read from the FINAL state, so it is the rank at
// the end of the circuit and a lower bound on the peak over the run: a cap
// reported as binding did bind. And the corpus files carry no terminal
// measurement, so loading them as they are is gate-only; sampling would
// project the measured qubits and collapse the bonds.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "compare_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace lindblad;
using lindblad_bench::load_corpus_circuit;

namespace {

// One seeded trajectory, which is what shots == 0 means. There is nothing to
// sample in a gate-only circuit, and a second run at the same seed is the
// same run, which is what makes two caps comparable at all.
constexpr int kShots = 0;
constexpr std::uint64_t kSeed = 42;

// The caps bench_compare_mps sweeps at n = 24.
constexpr int kSweptCaps[] = {8, 16, 32, 64};

// Bond dimension across every cut of the chain. Site i spans cut i|i+1 through
// its right bond, so the profile is one shorter than the register.
std::vector<int> bond_profile(const MPSState& state) {
    std::vector<int> bonds;
    if (state.tensors.size() < 2) return bonds;
    bonds.reserve(state.tensors.size() - 1);
    for (std::size_t i = 0; i + 1 < state.tensors.size(); ++i) {
        bonds.push_back(state.tensors[i].bond_right);
    }
    return bonds;
}

struct Reading {
    std::vector<int> bonds;
    int peak = 0;
    std::size_t splits = 0;
    // Accumulated across every split of the run rather than per split, so a
    // total above 1 is ordinary on a deep circuit.
    double discarded = 0.0;
};

Reading measure(const QuantumCircuit& qc, int cap) {
    MPSSimulator sim;
    const auto result = sim.run(qc, cap, kShots, kSeed);
    Reading r;
    r.bonds = bond_profile(result.final_state);
    r.peak = r.bonds.empty() ? 0 : *std::max_element(r.bonds.begin(), r.bonds.end());
    r.splits = result.final_state.svd_call_count();
    r.discarded = result.final_state.truncation_error();
    return r;
}

// The criterion. Exact equality on both figures is deliberate: two runs that
// never truncated are the same computation and agree to the bit, so any
// tolerance here would only widen the set of pairs called "identical".
bool same_run(const Reading& a, const Reading& b) {
    return a.bonds == b.bonds && a.discarded == b.discarded;
}

std::string profile_text(const std::vector<int>& bonds) {
    std::ostringstream ss;
    for (std::size_t i = 0; i < bonds.size(); ++i) {
        if (i) ss << ' ';
        ss << bonds[i];
    }
    return ss.str();
}

// A brickwork of random SU(4) blocks on a small register, the same pattern the
// corpus generator emits: u3 on both wires, then three rounds of cx followed
// by a fresh u3 pair, on alternating bond parities. Angles come from a seeded
// stream, so the circuit is fixed across runs and platforms.
QuantumCircuit small_brickwork(int n, int layers, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(0.0, TWO_PI);
    QuantumCircuit qc(n);
    // Drawn into named values first: the order in which a compiler evaluates
    // call arguments is unspecified, and the circuit must not depend on it.
    auto u3 = [&](int q) {
        const double theta = angle(rng);
        const double phi = angle(rng);
        const double lam = angle(rng);
        qc.u3(theta, phi, lam, q);
    };
    for (int layer = 0; layer < layers; ++layer) {
        for (int a = layer % 2; a + 1 < n; a += 2) {
            u3(a);
            u3(a + 1);
            for (int k = 0; k < 3; ++k) {
                qc.cx(a, a + 1);
                u3(a);
                u3(a + 1);
            }
        }
    }
    return qc;
}

// The rank across cut i|i+1 of an n-qubit chain cannot exceed 2^min(i+1, n-i-1),
// whatever the circuit does. This is the ceiling profile, and it is what a
// generic brickwork of sufficient depth reaches at every cut.
std::vector<int> ceiling_profile(int n) {
    std::vector<int> bonds;
    for (int i = 0; i + 1 < n; ++i) {
        const int k = std::min(i + 1, n - i - 1);
        bonds.push_back(1 << k);
    }
    return bonds;
}

int count_type(const QuantumCircuit& qc, Instruction::GateType t) {
    int n = 0;
    for (const auto& inst : qc.instructions) {
        if (inst.type == t) ++n;
    }
    return n;
}

}  // namespace

// =============================================================================
// The corpus. brickwork_n24.qasm is the file the published sweep runs on.
// =============================================================================

TEST(V11281BondSaturation, BrickworkCorpusBindsEverySweptCap) {
    const auto qc = load_corpus_circuit("brickwork_n24.qasm", false);
    ASSERT_EQ(qc.n_qubits, 24);

    for (int cap : kSweptCaps) {
        const Reading at = measure(qc, cap);
        const Reading above = measure(qc, cap + 1);
        ::testing::Test::RecordProperty(
            "discarded_weight_chi" + std::to_string(cap),
            std::to_string(at.discarded));
        EXPECT_FALSE(same_run(at, above))
            << "chi = " << cap << " does no truncation work on the corpus: the "
               "run at " << cap << " is identical to the run at " << (cap + 1)
            << ". The sweep row for this cap measures an untruncated run.\n"
               "  profile at chi = " << cap << ": " << profile_text(at.bonds)
            << "\n  profile at chi = " << (cap + 1) << ": "
            << profile_text(above.bonds);
        // A cap that binds was reached. The converse is what the miniature
        // below shows to be false, which is why this is the weaker check.
        EXPECT_EQ(at.peak, cap)
            << "the final-state peak bond at chi = " << cap
            << " is not the cap, so the bond profile was not read correctly";
    }
}

TEST(V11281BondSaturation, EveryCorpusBlockIsOneSplit) {
    // Every cx in the brickwork family is on an adjacent pair, so no swap chain
    // is inserted and each is exactly one bond split. The u3 gates split
    // nothing. This is the `lb splits` column on the benchmark page, derived
    // from the circuit rather than typed from the page.
    const auto qc = load_corpus_circuit("brickwork_n24.qasm", false);
    const int cx = count_type(qc, Instruction::GateType::CX);
    const int u3 = count_type(qc, Instruction::GateType::U3);
    ASSERT_GT(cx, 0);
    ASSERT_EQ(static_cast<std::size_t>(cx + u3), qc.instructions.size())
        << "the corpus file carries an instruction other than cx and u3";
    // Three cx per block, and a u3 pair before the first cx and after each.
    EXPECT_EQ(cx % 3, 0);
    EXPECT_EQ(u3, 2 * (cx / 3) + 2 * cx);

    const Reading r = measure(qc, kSweptCaps[0]);
    EXPECT_EQ(r.splits, static_cast<std::size_t>(cx));
}

// =============================================================================
// Controls. A criterion that can only say "binds" would pass the test above.
// =============================================================================

TEST(V11281BondSaturation, ScalingCorpusIsANonBindingControl) {
    // The family the sweep used to run on. Its rank peaks at 2, so no cap in
    // the sweep can bind and every pair must be the same run. This is the
    // original defect, kept as the control that proves the criterion can
    // answer no.
    const auto qc = load_corpus_circuit("scaling_n24.qasm", false);
    ASSERT_EQ(qc.n_qubits, 24);

    const Reading lowest = measure(qc, kSweptCaps[0]);
    EXPECT_LE(lowest.peak, 2)
        << "the scaling family's rank is documented to peak at 2";

    for (int cap : {kSweptCaps[0], kSweptCaps[3]}) {
        const Reading at = measure(qc, cap);
        const Reading above = measure(qc, cap + 1);
        EXPECT_TRUE(same_run(at, above))
            << "chi = " << cap << " reported as binding on a family whose "
               "rank never exceeds 2";
    }
}

TEST(V11281BondSaturation, DerivedParityCapCannotBind) {
    // bench_validate's MPS member runs qv_n8 at a cap DERIVED from the register
    // width: the widest cut of an 8-qubit chain splits it 4 | 4, so the rank
    // cannot exceed 2^4 and the cap provably never binds. That is what makes
    // the parity gate a check on the contraction and sampling path rather than
    // on two truncation policies. Pinned through the engine, not on paper.
    const auto qc = load_corpus_circuit("qv_n8.qasm", false);
    ASSERT_EQ(qc.n_qubits, 8);
    const int derived_cap = 1 << (qc.n_qubits / 2);

    const Reading at = measure(qc, derived_cap);
    const Reading above = measure(qc, derived_cap + 1);
    EXPECT_TRUE(same_run(at, above))
        << "a cap of 2^(n/2) bound on an n-qubit chain, which no circuit can "
           "make happen";
    EXPECT_LE(at.peak, derived_cap);
    // Not asserted zero: the weight cutoff drops rounding-level directions
    // even on an exact run, so the figure is tiny rather than absent. It is
    // recorded so the scale of "tiny" is on the record.
    ::testing::Test::RecordProperty("discarded_weight_exact_run",
                                    std::to_string(at.discarded));
}

// =============================================================================
// The miniature. Reaching a cap and being truncated by one are different
// events, and this is the case that separates them.
// =============================================================================

TEST(V11281BondSaturation, ReachingACapIsNotBindingIt) {
    // Six qubits, four layers. The middle cut 2|3 is an even bond, crossed by
    // layers 0 and 2: the first crossing acts on a product state and gives
    // rank 2, the second multiplies by up to 4, and 8 is the cut's ceiling.
    // The odd bonds 1|2 and 3|4 are crossed by layers 1 and 3 on already
    // entangled input and reach their ceiling of 4 at the first crossing. The
    // end bonds cannot exceed 2. So at depth 4 every cut sits exactly on its
    // ceiling, and a cap of 8 is REACHED at the middle without truncating.
    const int n = 6;
    const auto qc = small_brickwork(n, 4, kSeed);
    const auto ceiling = ceiling_profile(n);
    const int top = *std::max_element(ceiling.begin(), ceiling.end());
    ASSERT_EQ(top, 8);

    // Uncapped in effect: a cap above every ceiling truncates nothing real.
    const Reading exact = measure(qc, top + 1);
    ASSERT_EQ(exact.bonds, ceiling)
        << "the miniature did not reach the ceiling at every cut: "
        << profile_text(exact.bonds) << " against " << profile_text(ceiling);

    // At the ceiling: the peak bond equals the cap, and the run is identical
    // to the exact one. A criterion of peak >= cap calls this binding.
    const Reading at_top = measure(qc, top);
    EXPECT_EQ(at_top.peak, top);
    EXPECT_TRUE(same_run(at_top, exact))
        << "a cap equal to the rank ceiling was reported as binding";

    // One below: every pair below the ceiling differs, because the middle cut
    // has more directions than the cap keeps.
    for (int cap : {2, 4}) {
        const Reading at = measure(qc, cap);
        const Reading above = measure(qc, cap + 1);
        EXPECT_FALSE(same_run(at, above))
            << "chi = " << cap << " did not bind on a circuit whose middle "
               "rank is " << top;
        EXPECT_GT(at.discarded, 0.0);
    }
}

TEST(V11281BondSaturation, DiscardedWeightAloneCannotJudgeBinding) {
    // The figure a naive criterion would threshold. At the ceiling the run
    // discards only rounding-level directions the weight cutoff removes, and
    // one below it discards a definite amount. Both are positive in general,
    // so "discarded > 0" cannot tell them apart, and any threshold placed
    // between them is a constant chosen from an observed run. The pairwise
    // criterion needs none. The two readings are recorded so the gap a
    // threshold would have to sit in is on the record.
    const int n = 6;
    const auto qc = small_brickwork(n, 4, kSeed);
    const Reading at_top = measure(qc, 8);
    const Reading below = measure(qc, 4);
    ::testing::Test::RecordProperty("discarded_at_ceiling",
                                    std::to_string(at_top.discarded));
    ::testing::Test::RecordProperty("discarded_one_step_below",
                                    std::to_string(below.discarded));
    EXPECT_LT(at_top.discarded, below.discarded);
    EXPECT_GT(below.discarded, 0.0);
}

TEST(V11281BondSaturation, BondProfileIsReadFromTheFinalState) {
    // The profile is the rank at the END of the circuit. A gate that
    // disentangles after the peak lowers it, so the reading is a lower bound
    // on the peak over the run and a cap reported as binding did bind. Shown
    // with a circuit that entangles and then exactly undoes it.
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    const Reading entangled = measure(qc, 4);
    EXPECT_EQ(entangled.bonds, std::vector<int>{2});

    qc.cx(0, 1).h(0);
    const Reading undone = measure(qc, 4);
    EXPECT_EQ(undone.splits, 2u) << "both cx split, whatever the final rank";
    EXPECT_LE(undone.bonds[0], 2);
    // The exact rank after undoing is 1 up to the weight cutoff; the split
    // that undoes the entanglement sees one singular value at 1 and one at
    // rounding, and the cutoff discards the rounding.
    EXPECT_EQ(undone.bonds[0], 1)
        << "a disentangling split left a rounding-level direction in the bond";
}
