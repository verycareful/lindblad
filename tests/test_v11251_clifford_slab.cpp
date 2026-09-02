// 1.1.25.1 test wave - StabilizerState::outcome_slab and the elimination knob.
//
// The slab is new public surface with properties that can be asserted directly,
// without reference to sampling at all. A stabilizer state's computational
// basis distribution is uniform over a coset of a linear subspace of F_2^n and
// never anything more complicated, so `offset` plus `basis` is a complete
// description of it: every outcome is offset xor some subset-sum of basis, and
// each is equally likely. That claim is checked here against the statevector's
// exact probabilities, so the comparison carries no sampling error.
//
// The two eliminations are pinned harder than agreeing on the distribution.
// Both end in a reduced row echelon form over the same affine subspace: pivot
// columns ascending, offset zero at every free coordinate, and one basis vector
// per free coordinate carrying that coordinate alone among the free ones. An
// RREF is unique for a given row space, so the returned slab is canonical and
// the two methods must agree BIT FOR BIT, not merely in distribution. The
// intermediate generators they produce do differ, which is why nothing here
// compares those.
//
// The note behind Elimination::FourRussians is deduplicated by the warning
// channel rather than by a flag inside the backend, so it is observable from a
// clean channel in any order: a suite that selected the block route earlier
// cannot consume it, and the repeat tally is the channel's own.

#include <gtest/gtest.h>

#include "v11251_clifford_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/validation.hpp"

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

using Slab = StabilizerState::OutcomeSlab;
using Elim = StabilizerState::Elimination;

using v11251::capture_warnings;
using v11251::first_deliveries;

bool slab_bit(const std::vector<uint64_t>& v, int q) {
    return ((v[static_cast<size_t>(q / 64)] >> (q % 64)) & 1ULL) != 0;
}

// Every outcome the slab describes: offset xor each subset-sum of basis.
// Restricted to one word, which covers every size used here.
std::set<uint64_t> slab_support(const Slab& slab) {
    std::set<uint64_t> out;
    const uint64_t base = slab.offset.empty() ? 0ULL : slab.offset[0];
    for (uint64_t m = 0; m < (1ULL << slab.dim); ++m) {
        uint64_t y = base;
        for (int b = 0; b < slab.dim; ++b) {
            if ((m >> b) & 1ULL) y ^= slab.basis[static_cast<size_t>(b)][0];
        }
        out.insert(y);
    }
    return out;
}

// The statevector's exact support, as outcome integers in the same packing the
// slab uses: qubit q in bit q.
std::set<uint64_t> statevector_support(const QuantumCircuit& qc) {
    std::set<uint64_t> out;
    for (const auto& [outcome, p] : v11251::exact_statevector_distribution(qc)) {
        (void)p;
        out.insert(outcome);
    }
    return out;
}

StabilizerState state_of(const QuantumCircuit& qc) {
    CliffordSimulator sim;
    return sim.run(qc, /*shots=*/0, /*seed=*/1).final_state;
}

// Circuits whose slabs are checked against the statevector. Each is
// measurement-free so both backends can be asked for a final state.
//
// expected_dim is the free dimension worked out by hand from the stabilizer
// generators, which is an oracle independent of both backends. kDeriveDim marks
// a case whose generators are not worth deriving on paper; dim is still pinned
// there, against the size of the statevector's support.
constexpr int kDeriveDim = -1;

struct Case {
    const char* name;
    int n;
    void (*build)(QuantumCircuit&);
    int expected_dim;
};

const Case kCases[] = {
    {"all_zero", 3, [](QuantumCircuit&) {}, 0},
    {"single_x", 3, [](QuantumCircuit& c) { c.x(1); }, 0},
    {"all_x", 3, [](QuantumCircuit& c) { c.x(0); c.x(1); c.x(2); }, 0},
    {"one_plus", 3, [](QuantumCircuit& c) { c.h(0); }, 1},
    {"all_plus", 4, [](QuantumCircuit& c) { for (int q = 0; q < 4; ++q) c.h(q); }, 4},
    {"bell", 2, [](QuantumCircuit& c) { c.h(0); c.cx(0, 1); }, 1},
    {"ghz3", 3, [](QuantumCircuit& c) { c.h(0); c.cx(0, 1); c.cx(1, 2); }, 1},
    {"ghz5", 5, [](QuantumCircuit& c) { c.h(0); c.cx(0, 1); c.cx(1, 2);
                                        c.cx(2, 3); c.cx(3, 4); }, 1},
    {"two_bells", 4, [](QuantumCircuit& c) { c.h(0); c.cx(0, 1);
                                             c.h(2); c.cx(2, 3); }, 2},
    {"phase_only", 3, [](QuantumCircuit& c) { c.h(0); c.h(1); c.h(2);
                                              c.s(0); c.cz(1, 2); }, 3},
    {"y_basis", 3, [](QuantumCircuit& c) { c.h(0); c.s(0); c.h(1); c.sdg(1); }, 2},
    {"new_gates", 4, [](QuantumCircuit& c) { c.h(0); c.sx(1); c.cy(0, 2);
                                             c.iswap(1, 3); c.ecr(2, 3); }, kDeriveDim},
    {"swap_chain", 4, [](QuantumCircuit& c) { c.h(0); c.cx(0, 1);
                                              c.swap(1, 2); c.swap(2, 3); }, 1},
};

}  // namespace

// =============================================================================
// Shape of the returned slab
// =============================================================================

TEST(V11251CliffordSlab, VectorWidthsMatchWordsPerVector) {
    for (int n : {1, 2, 3, 8, 63, 64, 65, 127, 128, 129}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState st(n);
        for (int q = 0; q < n; ++q) st.apply_h(q);
        const Slab slab = st.outcome_slab();
        const size_t W = static_cast<size_t>(st.words_per_vector());
        EXPECT_EQ(slab.n_qubits, n);
        EXPECT_EQ(slab.offset.size(), W);
        EXPECT_EQ(slab.basis.size(), static_cast<size_t>(slab.dim));
        for (const auto& v : slab.basis) EXPECT_EQ(v.size(), W);
    }
}

// Nothing may be set past qubit n - 1. The elimination works in whole words and
// the last one is only partly meaningful, so a stray high bit would be a real
// outcome vector carrying a qubit the register does not have.
TEST(V11251CliffordSlab, NoBitIsSetPastTheLastQubit) {
    for (int n : {1, 2, 3, 8, 33, 63, 64, 65, 100, 127, 128, 129}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState st(n);
        for (int q = 0; q < n; ++q) { st.apply_h(q); st.apply_s(q); }
        for (int q = 0; q + 1 < n; ++q) st.apply_cx(q, q + 1);
        const Slab slab = st.outcome_slab();
        const int W = st.words_per_vector();
        const int tail = n % 64;
        const uint64_t mask = (tail == 0) ? ~0ULL : ((1ULL << tail) - 1ULL);
        EXPECT_EQ(slab.offset[static_cast<size_t>(W - 1)] & ~mask, 0ULL)
            << "offset carries a bit past qubit " << n - 1;
        for (size_t b = 0; b < slab.basis.size(); ++b) {
            EXPECT_EQ(slab.basis[b][static_cast<size_t>(W - 1)] & ~mask, 0ULL)
                << "basis vector " << b << " carries a bit past qubit " << n - 1;
        }
    }
}

// A basis is a basis: no vector is zero, and no two are equal. Both follow from
// the reduced form, where each vector owns a free coordinate no other one sets.
TEST(V11251CliffordSlab, BasisVectorsAreNonZeroAndDistinct) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        const Slab slab = state_of(qc).outcome_slab();
        std::set<std::vector<uint64_t>> seen;
        for (const auto& v : slab.basis) {
            bool any = false;
            for (uint64_t w : v) any = any || (w != 0);
            EXPECT_TRUE(any) << "a basis vector is all zero";
            EXPECT_TRUE(seen.insert(v).second) << "a basis vector is repeated";
        }
        EXPECT_EQ(seen.size(), static_cast<size_t>(slab.dim));
    }
}

// =============================================================================
// The slab describes the true distribution
// =============================================================================

// dim is the free dimension, so the support has exactly 2^dim members, all
// distinct. This is the claim the sampling loop rests on when it draws dim bits
// and calls the result a shot.
TEST(V11251CliffordSlab, SupportSizeIsTwoToTheDim) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        const Slab slab = state_of(qc).outcome_slab();
        if (c.expected_dim != kDeriveDim) EXPECT_EQ(slab.dim, c.expected_dim);
        EXPECT_EQ(slab_support(slab).size(), static_cast<size_t>(1) << slab.dim);
        // The same claim against an oracle that knows nothing about the slab:
        // the statevector's support has 2^dim members too.
        EXPECT_EQ(static_cast<size_t>(1) << slab.dim, statevector_support(qc).size());
    }
}

// The support the slab describes and the support the statevector actually has
// must be the same set. Neither side is sampled.
TEST(V11251CliffordSlab, SupportEqualsTheStatevectorSupport) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        EXPECT_EQ(slab_support(state_of(qc).outcome_slab()), statevector_support(qc));
    }
}

// offset is one point of the coset, so it must itself be an outcome the state
// can produce, and so must offset combined with any subset of basis. That is
// the whole content of "the constraints are satisfied", stated without reaching
// into the constraint rows.
TEST(V11251CliffordSlab, OffsetAndEverySubsetSumSatisfyTheConstraints) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        const Slab slab = state_of(qc).outcome_slab();
        const std::set<uint64_t> allowed = statevector_support(qc);

        const uint64_t base = slab.offset.empty() ? 0ULL : slab.offset[0];
        EXPECT_GT(allowed.count(base), 0u) << "offset is not a reachable outcome";

        for (uint64_t m = 0; m < (1ULL << slab.dim); ++m) {
            uint64_t y = base;
            for (int b = 0; b < slab.dim; ++b) {
                if ((m >> b) & 1ULL) y ^= slab.basis[static_cast<size_t>(b)][0];
            }
            EXPECT_GT(allowed.count(y), 0u)
                << "subset " << m << " leaves the support at outcome " << y;
        }
    }
}

// The distribution is uniform over the coset, which is the other half of what
// makes the sampling loop's fair coin per free direction correct.
TEST(V11251CliffordSlab, DistributionIsUniformOverTheCosetAndMatchesStatevector) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        const StabilizerState st = state_of(qc);
        const v11251::Dist from_slab = v11251::exact_clifford_distribution(st);
        EXPECT_TRUE(v11251::distributions_equal(
            from_slab, v11251::exact_statevector_distribution(qc)));
        const double uniform = std::ldexp(1.0, -st.outcome_slab().dim);
        for (const auto& [outcome, p] : from_slab) {
            (void)outcome;
            EXPECT_NEAR(p, uniform, 1e-12);
        }
    }
}

// =============================================================================
// The extremes of dim
// =============================================================================

TEST(V11251CliffordSlab, FullyDeterminedStateHasDimZero) {
    for (int n : {1, 2, 3, 8, 33, 64, 65}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState st(n);
        const Slab fresh = st.outcome_slab();
        EXPECT_EQ(fresh.dim, 0);
        EXPECT_TRUE(fresh.basis.empty());
        for (uint64_t w : fresh.offset) EXPECT_EQ(w, 0ULL);

        // A basis state other than |0...0> is equally determined, and its
        // offset is the state itself.
        StabilizerState flipped(n);
        for (int q = 0; q < n; q += 2) flipped.apply_x(q);
        const Slab s2 = flipped.outcome_slab();
        EXPECT_EQ(s2.dim, 0);
        for (int q = 0; q < n; ++q) {
            EXPECT_EQ(slab_bit(s2.offset, q), (q % 2) == 0)
                << "offset disagrees with the prepared basis state at qubit " << q;
        }
    }
}

// Measuring collapses the free directions away, so a state that was undetermined
// becomes determined and its dim drops to zero.
TEST(V11251CliffordSlab, MeasurementCollapseDropsDimToZero) {
    for (int n : {2, 3, 5, 8}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState st(n);
        for (int q = 0; q < n; ++q) st.apply_h(q);
        EXPECT_EQ(st.outcome_slab().dim, n);

        std::mt19937_64 rng(99);
        std::vector<int> outcomes;
        for (int q = 0; q < n; ++q) outcomes.push_back(st.measure(q, true, rng));

        const Slab after = st.outcome_slab();
        EXPECT_EQ(after.dim, 0);
        for (int q = 0; q < n; ++q) {
            EXPECT_EQ(slab_bit(after.offset, q), outcomes[static_cast<size_t>(q)] == 1)
                << "offset disagrees with the measured outcome at qubit " << q;
        }
    }
}

TEST(V11251CliffordSlab, UnconstrainedStateHasDimEqualToN) {
    for (int n : {1, 2, 3, 8, 33, 64, 65}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState st(n);
        for (int q = 0; q < n; ++q) st.apply_h(q);
        const Slab slab = st.outcome_slab();
        EXPECT_EQ(slab.dim, n);
        EXPECT_EQ(slab.basis.size(), static_cast<size_t>(n));
        // Every qubit is free, so the reduced basis is the identity: vector q
        // sets qubit q and nothing else.
        for (int b = 0; b < n; ++b) {
            for (int q = 0; q < n; ++q) {
                EXPECT_EQ(slab_bit(slab.basis[static_cast<size_t>(b)], q), b == q)
                    << "basis vector " << b << " at qubit " << q;
            }
        }
    }
}

// A GHZ state is the smallest case where dim is neither 0 nor n: n - 1 parity
// constraints leave exactly one free direction.
TEST(V11251CliffordSlab, GhzHasOneFreeDirectionAndTwoOutcomes) {
    for (int n : {2, 3, 4, 5, 8}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState st(n);
        st.apply_h(0);
        for (int q = 0; q + 1 < n; ++q) st.apply_cx(q, q + 1);

        const Slab slab = st.outcome_slab();
        EXPECT_EQ(slab.dim, 1);
        ASSERT_EQ(slab.basis.size(), 1u);

        const uint64_t all_ones =
            (n == 64) ? ~0ULL : ((static_cast<uint64_t>(1) << n) - 1ULL);
        EXPECT_EQ(slab_support(slab), (std::set<uint64_t>{0ULL, all_ones}));
    }
}

TEST(V11251CliffordSlab, ZeroQubitStateHasAnEmptySlab) {
    StabilizerState st(0);
    const Slab slab = st.outcome_slab();
    EXPECT_EQ(slab.n_qubits, 0);
    EXPECT_EQ(slab.dim, 0);
    EXPECT_TRUE(slab.offset.empty());
    EXPECT_TRUE(slab.basis.empty());

    // The block route has to reach the same conclusion on an empty register.
    const Slab block = st.outcome_slab(Elim::FourRussians);
    EXPECT_EQ(block.dim, 0);
    EXPECT_TRUE(block.offset.empty());
    EXPECT_TRUE(block.basis.empty());
}

// =============================================================================
// outcome_slab is a read
// =============================================================================

// The elimination runs in a local copy, so the state it was called on is left
// exactly as it was. A caller that samples and then measures the same state
// depends on this.
TEST(V11251CliffordSlab, DoesNotModifyTheStateItReads) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        StabilizerState st = state_of(qc);
        const v11251::Fingerprint before = v11251::fingerprint(st);

        (void)st.outcome_slab();
        (void)st.outcome_slab(Elim::Plain);
        (void)st.outcome_slab(Elim::FourRussians);

        EXPECT_TRUE(v11251::fingerprints_equal(before, v11251::fingerprint(st)));
    }
}

// Repeated calls return the same slab, which is what "canonical" means in
// practice for a caller holding on to one.
TEST(V11251CliffordSlab, RepeatedCallsReturnTheSameSlab) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        const StabilizerState st = state_of(qc);
        EXPECT_TRUE(v11251::slab_key(st) == v11251::slab_key(st));
        EXPECT_EQ(st.outcome_slab().dim, st.outcome_slab().dim);
    }
}

// =============================================================================
// The elimination knob
// =============================================================================

TEST(V11251CliffordSlab, PlainIsTheDefaultElimination) {
    CliffordSimulator sim;
    EXPECT_EQ(sim.options.elimination, Elim::Plain);

    // The default argument of outcome_slab is the same choice, which is what
    // makes the no-argument call in the rest of this file a Plain call.
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        const StabilizerState st = state_of(qc);
        EXPECT_TRUE(v11251::slab_key(st) == v11251::slab_key(st, Elim::Plain));
    }
}

// Both routes reduce to the same reduced row echelon form over the same affine
// subspace, and an RREF is unique for a given row space, so the slabs are equal
// bit for bit: same dim, same offset words, same basis vectors in the same
// order. Anything weaker would pass while the canonical form drifted.
TEST(V11251CliffordSlab, BothEliminationsProduceTheIdenticalCanonicalSlab) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        const StabilizerState st = state_of(qc);
        const v11251::SlabKey plain = v11251::slab_key(st, Elim::Plain);
        const v11251::SlabKey block = v11251::slab_key(st, Elim::FourRussians);
        EXPECT_EQ(plain.dim, block.dim);
        EXPECT_EQ(plain.offset, block.offset);
        EXPECT_EQ(plain.basis, block.basis);
        EXPECT_TRUE(plain == block);
    }
}

// The same statement at sizes past the block width, where the block route
// actually engages its table rather than falling through a single partial
// block. kBlock is 6, so a tableau needs well over 64 rows for the two routes
// to take genuinely different paths.
TEST(V11251CliffordSlab, EliminationsAgreeAtSizesPastTheBlockWidth) {
    for (int n : {7, 8, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState st(n);
        // A state mixing determined and free qubits, so the elimination has
        // both pivot and non-pivot columns to find at every size.
        for (int q = 0; q < n; ++q) {
            if (q % 3 == 0) st.apply_h(q);
            if (q % 3 == 1) st.apply_x(q);
            if (q % 3 == 2) { st.apply_h(q); st.apply_s(q); }
        }
        for (int q = 0; q + 1 < n; q += 2) st.apply_cx(q, q + 1);
        for (int q = 1; q + 1 < n; q += 3) st.apply_cz(q, q + 1);

        const v11251::SlabKey plain = v11251::slab_key(st, Elim::Plain);
        const v11251::SlabKey block = v11251::slab_key(st, Elim::FourRussians);
        EXPECT_EQ(plain.dim, block.dim);
        EXPECT_EQ(plain.offset, block.offset);
        EXPECT_EQ(plain.basis, block.basis);
    }
}

// The distribution comparison as well, since it is the property a caller
// actually depends on and it holds even if the canonical form were ever
// deliberately changed.
TEST(V11251CliffordSlab, EliminationsProduceTheSameDistribution) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        QuantumCircuit qc(c.n);
        c.build(qc);
        const StabilizerState st = state_of(qc);
        EXPECT_EQ(slab_support(st.outcome_slab(Elim::Plain)),
                  slab_support(st.outcome_slab(Elim::FourRussians)));
        EXPECT_EQ(slab_support(st.outcome_slab(Elim::FourRussians)),
                  statevector_support(qc));
    }
}

// Selecting the block route announces itself, saying which method was selected
// and that the default is the faster one at ordinary sizes.
//
// Deduplication belongs to the warning channel rather than to a flag inside the
// backend, so this is observable from a clean channel in any order, and a suite
// that selected the block route earlier cannot consume it.
TEST(V11251CliffordSlab, FourRussiansAnnouncesItself) {
    const std::vector<std::string> notes = capture_warnings([] {
        StabilizerState st(6);
        for (int q = 0; q < 6; ++q) st.apply_h(q);
        (void)st.outcome_slab(Elim::FourRussians);
    });
    ASSERT_EQ(notes.size(), 1u) << "one selection, one note";
    const std::string& note = notes.front();
    EXPECT_NE(note.find("note:"), std::string::npos) << note;
    EXPECT_NE(note.find("FourRussians"), std::string::npos) << note;
    EXPECT_NE(note.find("Plain"), std::string::npos) << note;
}

// A caller selecting the block route in a loop is told once and given a tally,
// not told on every iteration. That is the channel's repeat handling, which the
// backend now relies on instead of suppressing the message itself.
TEST(V11251CliffordSlab, RepeatedSelectionsAreCountedNotRepeated) {
    const std::vector<std::string> notes = capture_warnings([] {
        StabilizerState st(6);
        for (int q = 0; q < 6; ++q) st.apply_h(q);
        for (int i = 0; i < 4; ++i) (void)st.outcome_slab(Elim::FourRussians);
    });

    // One first delivery, whatever the repeat tally looks like.
    const std::vector<std::string> first = first_deliveries(notes);
    ASSERT_EQ(first.size(), 1u) << "the note must be delivered once, not four times";
    EXPECT_NE(first.front().find("FourRussians"), std::string::npos);

    // The other three are accounted for rather than dropped.
    ASSERT_EQ(notes.size(), 2u);
    EXPECT_NE(notes.back().find("[repeated 3 more times]"), std::string::npos)
        << notes.back();
}

// The default route says nothing at all.
TEST(V11251CliffordSlab, PlainEliminationEmitsNothing) {
    const std::vector<std::string> emitted = capture_warnings([] {
        StabilizerState st(6);
        for (int q = 0; q < 6; ++q) st.apply_h(q);
        for (int i = 0; i < 4; ++i) {
            (void)st.outcome_slab();
            (void)st.outcome_slab(Elim::Plain);
        }
    });
    EXPECT_TRUE(emitted.empty()) << "the default elimination must be silent";
}
