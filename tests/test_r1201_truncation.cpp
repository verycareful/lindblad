// R.1.20.1 test suite — the discarded-weight MPS truncation rule (#69).
//
// R.1.20.0 replaced the rule both MPS layers used to decide how many Schmidt
// directions to keep. The old rule compared each singular value against
// `cutoff` as an ABSOLUTE MAGNITUDE. The new rule treats `cutoff` as the
// maximum FRACTION OF TOTAL WEIGHT (sum of sigma²) a truncation may discard.
//
// Why that had to change: a magnitude threshold asks a question whose answer
// depends on the scale of the input and on how the target rounded its way
// there. The same 13-qubit state counted 12 significant directions on x86-64
// and 17 on arm64 — twelve real singular values at 2.9e-01, everything else
// numerical zero, and a fixed threshold sitting inside the noise band rather
// than inside the eleven-order gap above it. A weight fraction is scale-free,
// so the same spectrum classifies identically on any target.
//
// Testing that through the public API needs a discriminator: a case where the
// two rules give DIFFERENT answers. A Bell pair supplies one exactly. Its
// spectrum is {1/√2, 1/√2} — singular values ≈ 0.7071, weights 0.5 each — so
// the magnitude rule's decision point sits at 0.7071 and the weight rule's at
// 0.5. Between those two numbers the rules disagree about the rank, and that
// gap is where most of this file does its work.
//
// Covered here:
//   - cutoff is spent against WEIGHT, not magnitude (the discriminator above)
//   - it is a CEILING, not a quota: a clean spectral gap loses nothing
//   - the accumulated error never exceeds the declared budget
//   - a bond that discarded nothing reports EXACTLY zero, which is the
//     regression pin for the catastrophic-cancellation defect found while
//     R.1.20.0 was being written
//   - the qudit layer answers the same question as the qubit layer, which is
//     what the mirror convention requires and what R.1.20.0 changed it to do
//     (it previously used a third, different rule: relative magnitude)

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace lindblad;

namespace {

// The truncation budget both layers ship with. Written out here ON PURPOSE:
// this is the one place the documented default is the thing under test, so it
// has to be stated independently rather than read back from the object that is
// supposed to carry it.
constexpr double kDocumentedDefaultCutoff = 1e-16;

// Bell-pair spectrum landmarks. Both rules agree outside the interval between
// them and disagree inside it, which is what makes the pair a discriminator.
constexpr double kBellSingularValue = INV_SQRT2;  // ≈ 0.7071 — the magnitude
constexpr double kBellWeight        = 0.5;        // sigma²    — the weight

static_assert(kBellWeight < kBellSingularValue,
              "the discriminating interval must be non-empty");

// Hadamard on one qubit, in the 2x2 row-major form apply_single_qubit_gate
// wants. Library-sourced amplitude.
std::array<Complex128, 4> h2x2() {
    return {Complex128(INV_SQRT2, 0), Complex128(INV_SQRT2, 0),
            Complex128(INV_SQRT2, 0), Complex128(-INV_SQRT2, 0)};
}

// CNOT with the control on the FIRST operand and the target on the second.
//
// MPSState::apply_two_qubit_gate is a RAW BACKEND PRIMITIVE, and its matrix
// index is MSB-first in the operands: the contraction is documented as
// `row = l*2 + p1, col = p2*br + r` and the operand-reorder step spells it
// `U'[po1*2 + po2, pi1*2 + pi2]`, so for apply_two_qubit_gate(U, qa, qb) the
// index is (bit at qa)*2 + (bit at qb) — qa is the HIGH bit.
//
// That is NOT the qubits[0]-is-LSB layout the circuit-level API uses, and the
// difference is deliberate: the frozen conventions say backends with MSB-first
// internals bridge by bit reversal, and warn in as many words never to hand a
// raw matrix to a backend assuming MSB order. Getting this backwards puts the
// control on the wrong qubit, which for |+0> yields a PRODUCT state — every
// spectrum in this file would then be rank 1 and the suite would quietly stop
// testing truncation at all.
//
// Map with index = qa*2 + qb: 00->00, 01->01, 10->11, 11->10.
std::array<Complex128, 16> cnot_control_first_operand() {
    std::array<Complex128, 16> U{};
    U[0 * 4 + 0] = Complex128(1, 0);
    U[1 * 4 + 1] = Complex128(1, 0);
    U[3 * 4 + 2] = Complex128(1, 0);
    U[2 * 4 + 3] = Complex128(1, 0);
    return U;
}

// A two-qubit MPS carrying a Bell pair, built at the requested cutoff. One
// two-qubit gate means exactly ONE truncating SVD, so every figure this state
// reports is attributable to a single decision.
MPSState bell_state_at_cutoff(double cutoff) {
    MPSState st(2, /*max_bond_dim=*/64, cutoff);
    st.apply_single_qubit_gate(h2x2(), 0);
    st.apply_two_qubit_gate(cnot_control_first_operand(), 0, 1);
    return st;
}

// The same Bell pair as a d=2 qudit state, built from dense amplitudes rather
// than from gates. Going in through the statevector constructor sidesteps the
// operand-ordering question entirely: the spectrum is a property of the state,
// so an orientation mix-up cannot silently change what is being measured.
QuditMPS bell_qudit_at_cutoff(double svd_cutoff) {
    QuditStatevector sv(/*n_qudits=*/2, /*d=*/2);
    sv.amplitudes.assign(4, Complex128(0, 0));
    sv.amplitudes[0] = Complex128(INV_SQRT2, 0);  // |00>
    sv.amplitudes[3] = Complex128(INV_SQRT2, 0);  // |11>
    return QuditMPS(sv, /*max_bond_dim=*/64, svd_cutoff);
}

int qudit_max_bond(const QuditMPS& qm) {
    int chi = 1;
    for (const auto& t : qm.tensors) {
        chi = std::max(chi, t.chi_L);
        chi = std::max(chi, t.chi_R);
    }
    return chi;
}

// A GHZ chain: one Hadamard then a ladder of CNOTs. Schmidt rank is 2 at every
// cut no matter how long the chain, and the rest of each spectrum is exact
// zero — a bimodal spectrum with an enormous gap, which is the shape the
// "ceiling, not quota" claim is about.
MPSState ghz_state(int n, double cutoff) {
    MPSState st(n, /*max_bond_dim=*/64, cutoff);
    st.apply_single_qubit_gate(h2x2(), 0);
    for (int q = 0; q + 1 < n; ++q) {
        st.apply_two_qubit_gate(cnot_control_first_operand(), q, q + 1);
    }
    return st;
}

} // namespace

// =============================================================================
// The discriminator: weight, not magnitude
// =============================================================================

// Below the weight of the smallest direction, both rules keep everything, so
// this end of the sweep only establishes the baseline: the state really is
// entangled and really does have rank 2.
TEST(R1201Truncation, BellKeepsBothDirectionsBelowTheWeightThreshold) {
    for (double cutoff : {0.0, 1e-30, kDocumentedDefaultCutoff, 1e-9, 0.1, 0.4}) {
        MPSState st = bell_state_at_cutoff(cutoff);
        EXPECT_EQ(st.current_max_bond_dim(), 2)
            << "cutoff = " << cutoff
            << ": a Bell pair has two real Schmidt directions and the budget "
               "is too small to pay for dropping either";
    }
}

// The discriminating case, and the reason this file exists.
//
// At cutoff = 0.6 the two rules disagree outright:
//   weight rule     dropping one direction costs 0.5 of weight, and 0.5 <= 0.6,
//                   so it is affordable -> rank 1
//   magnitude rule  the singular value is 0.7071, which is ABOVE 0.6, so it
//                   would be kept -> rank 2
//
// Rank 1 here is therefore positive evidence that `cutoff` is being spent
// against weight. Rank 2 would mean the magnitude rule is back.
TEST(R1201Truncation, CutoffIsSpentAgainstWeightNotMagnitude) {
    ASSERT_LT(kBellWeight, 0.6);
    ASSERT_GT(kBellSingularValue, 0.6);

    MPSState st = bell_state_at_cutoff(0.6);
    EXPECT_EQ(st.current_max_bond_dim(), 1)
        << "a budget of 0.6 covers the 0.5 of weight one direction carries, so "
           "it must be discarded. Rank 2 means the singular value (0.7071) was "
           "compared against the cutoff directly — the magnitude rule #69 "
           "removed, and with it the platform-dependent rank count";
}

// The boundary from the other side: just under the weight, the same direction
// is unaffordable and must survive. Together with the case above this pins the
// decision point at 0.5 — the WEIGHT — rather than at 0.7071.
TEST(R1201Truncation, DecisionPointSitsAtTheWeightNotTheSingularValue) {
    EXPECT_EQ(bell_state_at_cutoff(0.45).current_max_bond_dim(), 2)
        << "0.45 does not cover 0.5 of weight; nothing may be dropped";
    EXPECT_EQ(bell_state_at_cutoff(0.55).current_max_bond_dim(), 1)
        << "0.55 does cover 0.5 of weight; one direction must go";

    // Both rules agree out here, so these are consistency checks rather than
    // discriminators — but a rule that got them wrong would be broken outright.
    EXPECT_EQ(bell_state_at_cutoff(0.9).current_max_bond_dim(), 1);
    EXPECT_EQ(bell_state_at_cutoff(1e-3).current_max_bond_dim(), 2);
}

// Rank is never driven below one, whatever the budget says. A state has to
// remain a state.
TEST(R1201Truncation, RankNeverFallsBelowOne) {
    for (double cutoff : {0.99, 1.0, 2.0, 1e6}) {
        MPSState st = bell_state_at_cutoff(cutoff);
        EXPECT_GE(st.current_max_bond_dim(), 1)
            << "cutoff = " << cutoff << ": truncated away the entire state";
    }
}

// =============================================================================
// Ceiling, not quota
// =============================================================================

// The property that makes the change accuracy-neutral in practice. Where a
// spectrum has a clean gap between real content and numerical noise, raising
// the budget across many orders of magnitude changes nothing, because there is
// nothing in the gap to spend it on. A rule that discarded weight merely
// BECAUSE it had budget would shed directions as the cutoff rose.
TEST(R1201Truncation, BudgetIsACeilingNotAQuotaOnBimodalSpectra) {
    for (int n : {2, 4, 6, 8}) {
        for (double cutoff : {1e-30, 1e-20, kDocumentedDefaultCutoff,
                              1e-12, 1e-8, 1e-4, 1e-2}) {
            MPSState st = ghz_state(n, cutoff);
            EXPECT_EQ(st.current_max_bond_dim(), 2)
                << "GHZ n=" << n << " at cutoff " << cutoff
                << ": Schmidt rank is exactly 2 at every cut and the rest of "
                   "the spectrum is exact zero, so no budget in this range can "
                   "buy anything";
        }
    }
}

// A product state has rank 1 and nothing to discard at any budget.
TEST(R1201Truncation, ProductStateStaysRankOne) {
    for (double cutoff : {1e-30, kDocumentedDefaultCutoff, 1e-6, 0.4}) {
        MPSState st(4, /*max_bond_dim=*/64, cutoff);
        st.apply_single_qubit_gate(h2x2(), 0);
        st.apply_single_qubit_gate(h2x2(), 1);
        st.apply_single_qubit_gate(h2x2(), 2);
        // A two-qubit gate is needed to reach the SVD path at all; a diagonal
        // gate on a product state leaves it a product state.
        std::array<Complex128, 16> diag{};
        diag[0] = diag[5] = diag[10] = diag[15] = Complex128(1, 0);
        st.apply_two_qubit_gate(diag, 0, 1);
        EXPECT_EQ(st.current_max_bond_dim(), 1) << "cutoff = " << cutoff;
    }
}

// =============================================================================
// The budget is honoured, and the accounting is sound
// =============================================================================

// The regression pin for a defect that was introduced and caught DURING
// R.1.20.0: the discarded weight was briefly computed as `total - kept`. For a
// normalised state both sides are ≈ 1.0 while the true difference is ~1e-30, so
// the subtraction cannot resolve it and quantises to multiples of eps —
// truncation_error() then reported ~8.9e-16 (four ULP of 1.0) of phantom loss
// for a bond that had discarded nothing at all.
//
// A Bell pair at the default cutoff discards nothing: the primary SVD route
// runs with a validity floor of zero, and both singular values are far above
// any budget. So the honest report is EXACTLY zero, and any eps-scale figure
// here means the accounting is differencing large numbers again.
TEST(R1201Truncation, DiscardingNothingReportsExactlyZero) {
    MPSState st = bell_state_at_cutoff(kDocumentedDefaultCutoff);
    ASSERT_EQ(st.current_max_bond_dim(), 2)
        << "precondition: nothing may have been discarded";

    EXPECT_EQ(st.truncation_error(), 0.0)
        << "reported " << st.truncation_error()
        << " of discarded weight for a bond that discarded nothing. A value "
           "near 8.9e-16 (four ULP of 1.0) is the signature of computing "
           "discarded weight as total-minus-kept instead of summing the "
           "discarded buckets directly";
}

// The same claim over a longer circuit, where a per-gate epsilon would
// accumulate visibly rather than hide in a single bond.
TEST(R1201Truncation, ExactCircuitsAccumulateNoPhantomError) {
    MPSState st = ghz_state(10, kDocumentedDefaultCutoff);
    ASSERT_EQ(st.current_max_bond_dim(), 2);
    EXPECT_EQ(st.truncation_error(), 0.0)
        << "nine truncating SVDs discarded nothing between them, so the "
           "accumulated figure must still be exactly zero; got "
           << st.truncation_error();
}

// The contract svd_truncate offers: a truncating SVD spends at most one budget,
// so N of them accumulate at most N budgets. Checked where truncation actually
// happens, since a bound is only meaningful when the quantity is non-zero.
TEST(R1201Truncation, AccumulatedErrorStaysInsideTheDeclaredBudget) {
    for (double cutoff : {0.55, 0.6, 0.75, 0.9}) {
        const int n = 6;
        MPSState st = ghz_state(n, cutoff);
        const int n_truncating_svds = n - 1;

        const double terr = st.truncation_error();
        EXPECT_GE(terr, 0.0) << "discarded weight cannot be negative";
        EXPECT_LE(terr, n_truncating_svds * cutoff * (1.0 + 1e-9))
            << "cutoff = " << cutoff << ": accumulated discarded weight "
            << terr << " exceeds " << n_truncating_svds
            << " budgets, so at least one SVD spent more than it was allowed";
    }
}

// Truncation must actually engage once the budget is large enough — otherwise
// every bound above would be satisfied trivially by a rule that never truncates.
TEST(R1201Truncation, TruncationDoesEngageWhenTheBudgetAllowsIt) {
    MPSState st = ghz_state(6, 0.75);
    EXPECT_GT(st.truncation_error(), 0.0)
        << "a budget of 0.75 per bond covers the 0.5 each GHZ direction "
           "carries; if nothing was discarded the rule is not spending at all";
    EXPECT_EQ(st.current_max_bond_dim(), 1);
}

// =============================================================================
// Defaults
// =============================================================================

// Both layers ship the same documented budget. The value is bounded at the
// order of the reconstruction error an SVD already carries (~1.1e-16 relative),
// so nothing the factorisation actually resolved is thrown away.
TEST(R1201Truncation, BothLayersDefaultToTheDocumentedBudget) {
    MPSState qubit_layer(4);
    EXPECT_EQ(qubit_layer.cutoff, kDocumentedDefaultCutoff)
        << "MPSState default cutoff changed";

    QuditMPS qudit_layer(/*n_qudits=*/4, /*d=*/3);
    EXPECT_EQ(qudit_layer.svd_cutoff, kDocumentedDefaultCutoff)
        << "QuditMPS default svd_cutoff changed";

    EXPECT_EQ(qubit_layer.cutoff, qudit_layer.svd_cutoff)
        << "the two layers must ship the same budget; the mirror convention "
           "requires them to answer the same question the same way";
}

// =============================================================================
// The mirror convention: both layers answer the same question
// =============================================================================

// Before R.1.20.0 the qudit layer used a THIRD rule — relative magnitude —
// which meant the same state could carry a different bond dimension depending
// on which layer was simulating it. The mirror convention says the qudit layer
// mirrors the qubit layer at general d, so the identical spectrum must be
// classified identically.
//
// The Bell pair is the same state in both layers (d = 2, two sites), so the
// crossover from rank 2 to rank 1 has to happen between the same two budgets.
TEST(R1201Truncation, QuditLayerClassifiesTheSameSpectrumTheSameWay) {
    struct Case { double cutoff; int expected_rank; };
    const Case cases[] = {
        {kDocumentedDefaultCutoff, 2},
        {1e-6,                     2},
        {0.45,                     2},  // below the weight: keep both
        {0.55,                     1},  // above the weight: drop one
        {0.9,                      1},
    };

    for (const auto& c : cases) {
        const int qubit_rank = bell_state_at_cutoff(c.cutoff)
                                   .current_max_bond_dim();
        const int qudit_rank = qudit_max_bond(bell_qudit_at_cutoff(c.cutoff));

        EXPECT_EQ(qubit_rank, c.expected_rank)
            << "qubit layer at cutoff " << c.cutoff;
        EXPECT_EQ(qudit_rank, c.expected_rank)
            << "qudit layer at cutoff " << c.cutoff;
        EXPECT_EQ(qubit_rank, qudit_rank)
            << "the layers disagree at cutoff " << c.cutoff
            << ": qubit kept " << qubit_rank << ", qudit kept " << qudit_rank
            << ". The same spectrum must classify identically in both, which "
               "is what the mirror convention requires";
    }
}

// The qudit layer's budget is a weight fraction too, checked at d = 3 so the
// claim is about general d rather than about the d = 2 special case that
// happens to coincide with the qubit layer.
//
// State: (|00> + |11> + |22>)/√3 — three equal Schmidt directions, weight 1/3
// each, singular values 1/√3 ≈ 0.5774. A budget of 0.5 covers one direction
// (1/3) and then a second would take it to 2/3, which it cannot afford, so
// exactly one goes. A magnitude rule at 0.5 would keep all three, since
// 0.5774 > 0.5.
TEST(R1201Truncation, QuditBudgetIsAWeightFractionAtGeneralD) {
    const double inv_sqrt3 = 1.0 / SQRT3;
    const double weight_each = 1.0 / 3.0;
    ASSERT_LT(weight_each, 0.5);
    ASSERT_GT(inv_sqrt3, 0.5) << "the discriminating interval must be non-empty";

    auto make = [&](double cutoff) {
        QuditStatevector sv(/*n_qudits=*/2, /*d=*/3);
        sv.amplitudes.assign(9, Complex128(0, 0));
        sv.amplitudes[0] = Complex128(inv_sqrt3, 0);  // |00>
        sv.amplitudes[4] = Complex128(inv_sqrt3, 0);  // |11>
        sv.amplitudes[8] = Complex128(inv_sqrt3, 0);  // |22>
        return qudit_max_bond(QuditMPS(sv, /*max_bond_dim=*/64, cutoff));
    };

    EXPECT_EQ(make(kDocumentedDefaultCutoff), 3)
        << "three equal Schmidt directions must all survive the default budget";
    EXPECT_EQ(make(0.3), 3) << "0.3 does not cover one direction's 1/3 weight";
    EXPECT_EQ(make(0.5), 2)
        << "0.5 covers one direction's 1/3 of weight but not two thirds. Rank 3 "
           "would mean the singular value (0.5774) was compared against the "
           "cutoff directly — the relative-magnitude rule R.1.20.0 replaced";
    EXPECT_EQ(make(0.9), 1) << "0.9 covers two directions' 2/3 of weight";
}
