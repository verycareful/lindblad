// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.1 test wave - the qubit MPS keeps its centre (#128).
//
// 1.1.30.0 made MPSState track an open span [lo, hi]: every site left of it
// left-orthonormal, every site right of it right-orthonormal, and a bond split
// that truncates does so on the state's Schmidt coefficients because the
// centre is moved onto the block first. Before that, splits truncated whatever
// block the gauge happened to present, and in a distorted gauge that keeps the
// wrong directions: the defect #128 reported.
//
// Covered here, each against a reference the chain did not compute:
//
//   - the invariant after every operation that moves or keeps the span, read
//     off the site entries (v11301::is_canonical), with the dense state
//     unchanged where the operation is a gauge change;
//   - the #128 reproducer: a capped split from a deliberately distorted gauge
//     keeps exactly the state's leading Schmidt directions, a split in the
//     distorted gauge demonstrably would not, and the figures the split
//     reports are the ones the dense Schmidt spectrum predicts;
//   - CanonicalForm: Always focuses before every split, Auto only where the
//     cap can bind or the cutoff is raised, the two agree in the exact regime,
//     and Auto keeps the wider bonds 1.1.30.0 measured on a 16-qubit brickwork;
//   - norm_sq, probabilities_single, normalize and the entropy observer's bond
//     spectrum, all of which read only the open span, on chains whose span sits
//     everywhere relative to the qubit or cut being read;
//   - set_tensors, every validation rule and what a valid chain resets;
//   - a chain with no qubits.

#include <gtest/gtest.h>

#include "v11301_mps_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace lindblad;
using v11301::Amplitudes;
using v11301::Cplx;
using v11301::is_canonical;
using v11301::kEps;
using v11301::kSlack;
using v11301::same_sites;

namespace {

using Span = std::pair<int, int>;

std::array<Complex128, 4> random_gate2(std::uint64_t seed) {
    return v11301::as_gate2(v11301::random_unitary(2, seed));
}

std::array<Complex128, 16> random_gate4(std::uint64_t seed) {
    return v11301::as_gate4(v11301::random_unitary(4, seed));
}

// The dense state with a gate applied through gates::apply_unitary, whose
// matrix convention (the first target is the least significant bit) is the
// one MPSState's gate entry points document.
Amplitudes apply_dense(const Amplitudes& a, int n, const std::vector<int>& targets,
                       const Complex128* U, std::size_t entries) {
    Statevector sv = v11301::to_statevector(a, n);
    gates::apply_unitary(sv, targets, std::vector<Complex128>(U, U + entries),
                         {Validation::Ignore});
    return v11301::dense(sv);
}

Amplitudes apply_dense(const Amplitudes& a, int n, int q,
                       const std::array<Complex128, 4>& U) {
    return apply_dense(a, n, {q}, U.data(), U.size());
}

Amplitudes apply_dense(const Amplitudes& a, int n, int q1, int q2,
                       const std::array<Complex128, 16>& U) {
    return apply_dense(a, n, {q1, q2}, U.data(), U.size());
}

// A chain holding `sites`, with the given settings.
MPSState chain_of(const std::vector<MPSTensor>& sites, int cap,
                  CanonicalForm form = CanonicalForm::Always,
                  double cutoff = MPS_DEFAULT_CUTOFF) {
    MPSState s(static_cast<int>(sites.size()), cap, cutoff);
    s.canonical_form = form;
    s.set_tensors(sites);
    return s;
}

// The rank across cut i|i+1 of an n-qubit state cannot exceed this.
int bond_ceiling(int n, int i) { return 1 << std::min(i + 1, n - i - 1); }

// Amplitude tolerance for a chain of `n` qubits reached through `ops` steps.
double amp_tol(int n, std::size_t ops, double norm) {
    return v11301::amplitude_tol(ops, std::size_t{1} << n, std::max(norm, 1.0));
}

// How far truncation can have moved a chain from the untruncated state. A split
// in canonical gauge moves the state by the square root of the weight it
// discarded, so by the triangle inequality the total is at most
// sum_k sqrt(w_k) <= sqrt(splits * truncation_error()). Every chain it is
// applied to below took its splits in canonical gauge.
double truncation_distance(const MPSState& s) {
    return std::sqrt(static_cast<double>(s.svd_call_count()) * s.truncation_error());
}

// Fidelity tolerance for a state reached through `ops` operations over `dim`
// amplitudes, plus what the weight budget lets `splits` splits discard. A
// fidelity is quadratic in the error, so rounding enters at eps and the
// budget's dust at its own size.
double fidelity_tol(std::size_t ops, std::size_t dim, std::size_t splits) {
    return kSlack * static_cast<double>(ops + dim) * kEps +
           MPS_DEFAULT_CUTOFF * static_cast<double>(splits);
}

// Every configuration of the open span a test below reads through, built from
// one random chain so the dense state is shared: the whole chain open (as
// set_tensors leaves it), the centre on each site in turn, and two partial
// spans an in-place Auto split leaves behind.
struct SpanCase {
    std::string name;
    MPSState chain;
};

std::vector<SpanCase> span_cases(const std::vector<MPSTensor>& sites, int cap) {
    const int n = static_cast<int>(sites.size());
    std::vector<SpanCase> cases;
    cases.push_back({"set_tensors", chain_of(sites, cap)});
    for (int c = 0; c < n; ++c) {
        MPSState s = chain_of(sites, cap);
        s.canonicalize(c);
        cases.push_back({"centre on " + std::to_string(c), std::move(s)});
    }
    // Centred on 0, then an in-place split under Auto at (2, 3): the span grows
    // to [0, 3] with sites 4.. still right-orthonormal. And centred on n-1,
    // then an in-place split at (1, 2): site 1 keeps an isometry and leaves the
    // span, which becomes [2, n-1].
    {
        MPSState s = chain_of(sites, cap, CanonicalForm::Auto);
        s.canonicalize(0);
        s.apply_two_qubit_gate(random_gate4(71), 2, 3);
        cases.push_back({"Auto span after an in-place split right of the centre",
                         std::move(s)});
    }
    {
        MPSState s = chain_of(sites, cap, CanonicalForm::Auto);
        s.canonicalize(n - 1);
        s.apply_two_qubit_gate(random_gate4(72), 1, 2);
        cases.push_back({"Auto span after an in-place split left of the centre",
                         std::move(s)});
    }
    return cases;
}

class WarningCapture {
public:
    WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
        set_warning_handler([this](const std::string& m) { lines_.push_back(m); });
    }
    ~WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
    }
    std::size_t count() const { return lines_.size(); }

private:
    std::vector<std::string> lines_;
};

}  // namespace

// =============================================================================
// Defaults and settings
// =============================================================================

TEST(V11301CanonicalForm, EveryTypeDefaultsToAlways) {
    EXPECT_EQ(MPSState(3).canonical_form, CanonicalForm::Always);
    EXPECT_EQ(MPSSimulator{}.canonical_form, CanonicalForm::Always);
    EXPECT_EQ(QuditMPS(3, 3).canonical_form, CanonicalForm::Always);
    EXPECT_EQ(QuditMPS(QuditStatevector(3, 3)).canonical_form, CanonicalForm::Always);

    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);
    MPSSimulator sim;
    EXPECT_EQ(sim.run(qc, 8, 0, 11).final_state.canonical_form, CanonicalForm::Always);

    EXPECT_STREQ(to_string(CanonicalForm::Auto), "Auto");
    EXPECT_STREQ(to_string(CanonicalForm::Always), "Always");
}

TEST(V11301CanonicalForm, TheSimulatorCopiesItsSettingsOntoEveryChainItBuilds) {
    // Every route a run takes to build a chain, and the settings each must
    // carry. A chain supplied as an MPS is the one exception: it brings its
    // own settings, bond cap included.
    MPSSimulator sim;
    sim.canonical_form = CanonicalForm::Auto;
    sim.svd_method = SVDMethod::EigenBDC;
    sim.svd_rejection = SvdRejection::Throw;
    sim.svd_accept_gram = true;
    sim.svd_report = SvdReport::Silent;
    constexpr int kCap = 6;

    const auto expect_sim_settings = [&](const MPSState& s, const char* route) {
        SCOPED_TRACE(route);
        EXPECT_EQ(s.canonical_form, CanonicalForm::Auto);
        EXPECT_EQ(s.svd_method, SVDMethod::EigenBDC);
        EXPECT_EQ(s.svd_rejection, SvdRejection::Throw);
        EXPECT_TRUE(s.svd_accept_gram);
        EXPECT_EQ(s.svd_report, SvdReport::Silent);
        EXPECT_EQ(s.max_bond_dim, kCap);
        EXPECT_EQ(s.cutoff, MPS_DEFAULT_CUTOFF);
    };

    QuantumCircuit gates(4);
    gates.h(0).cx(0, 1).cx(1, 2).cx(2, 3);
    expect_sim_settings(sim.run(gates, kCap, 0, 3).final_state, "single trajectory");

    QuantumCircuit terminal = gates;
    terminal.measure_all();
    expect_sim_settings(sim.run(terminal, kCap, 64, 3).final_state, "terminal sampling");

    // A gate on the measured qubit after its MEASURE forces per-shot runs.
    QuantumCircuit mid(4, 4);
    mid.h(0).cx(0, 1).measure(0, 0).cx(0, 2).cx(2, 3);
    expect_sim_settings(sim.run(mid, kCap, 8, 3).final_state, "per-shot trajectories");

    {
        RunPlan plan;
        plan.initial = InitialState::basis(5);
        expect_sim_settings(sim.run(gates, kCap, 0, 3, plan).final_state,
                            "basis-state seed");
    }
    {
        auto sv = std::make_shared<Statevector>(4);
        sv->initialize_basis(9);
        RunPlan plan;
        plan.initial = InitialState::from(std::shared_ptr<const Statevector>(sv));
        expect_sim_settings(sim.run(gates, kCap, 0, 3, plan).final_state,
                            "statevector seed");
    }
    {
        auto source = std::make_shared<MPSState>(4, 3);
        source->canonical_form = CanonicalForm::Always;
        source->svd_method = SVDMethod::BDC;
        source->svd_rejection = SvdRejection::Ignore;
        source->svd_accept_gram = false;
        source->svd_report = SvdReport::Warn;
        RunPlan plan;
        plan.initial = InitialState::from(std::shared_ptr<const MPSState>(source));
        const MPSState out = sim.run(gates, kCap, 0, 3, plan).final_state;
        EXPECT_EQ(out.canonical_form, CanonicalForm::Always)
            << "a chain supplied as an MPS brings its own policy";
        EXPECT_EQ(out.svd_method, SVDMethod::BDC);
        EXPECT_EQ(out.svd_rejection, SvdRejection::Ignore);
        EXPECT_FALSE(out.svd_accept_gram);
        EXPECT_EQ(out.svd_report, SvdReport::Warn);
        EXPECT_EQ(out.max_bond_dim, 3) << "and its own bond cap";
    }
}

// =============================================================================
// The invariant, operation by operation
// =============================================================================

TEST(V11301CanonicalForm, ANewChainIsCentredOnSiteZero) {
    for (int n = 1; n <= 6; ++n) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const MPSState s(n);
        EXPECT_EQ(s.open_span(), (Span{0, 0}));
        EXPECT_TRUE(is_canonical(s));
        EXPECT_EQ(s.fidelity_estimate(), std::optional<double>(1.0));
        EXPECT_EQ(s.fidelity_lower_bound(), std::optional<double>(1.0));
    }
}

TEST(V11301CanonicalForm, SingleQubitGatesLeaveTheSpanWhereItIs) {
    // A unitary on the physical index keeps both orthonormalities, so every
    // span configuration survives a 1q gate on any qubit, inside or outside it.
    const int n = 5;
    const auto sites = v11301::random_chain(n, {2, 4, 4, 2}, 101);
    std::uint64_t seed = 1000;
    for (auto& c : span_cases(sites, 16)) {
        SCOPED_TRACE(c.name);
        MPSState& s = c.chain;
        Amplitudes want = v11301::dense(s);
        const Span span = s.open_span();
        std::size_t ops = 0;
        for (int round = 0; round < 2; ++round) {
            for (int q = 0; q < n; ++q) {
                const auto U = random_gate2(seed++);
                s.apply_single_qubit_gate(U, q);
                want = apply_dense(want, n, q, U);
                ++ops;
                ASSERT_EQ(s.open_span(), span) << "a 1q gate on qubit " << q
                                               << " moved the span";
                ASSERT_TRUE(is_canonical(s)) << "after a 1q gate on qubit " << q;
            }
        }
        const double norm = std::sqrt(v11301::norm_sq(want));
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want), amp_tol(n, ops, norm));
    }
}

TEST(V11301CanonicalForm, EveryTwoQubitGateLeavesASingleSiteCentre) {
    // Under Always every split is taken at the centre and hands it on, so after
    // any two-qubit gate, adjacent or swap-routed, in either operand order, the
    // span is one site between the operands and the chain is canonical around
    // it. The splits are counted too: one for the gate, and two per site the
    // SWAP chain crosses (in and back); the centre steps count nothing.
    const int n = 6;
    MPSState s = chain_of(v11301::random_chain(n, {2, 4, 8, 4, 2}, 202), 64);
    s.canonicalize(2);
    Amplitudes want = v11301::dense(s);
    std::uint64_t seed = 2000;
    std::size_t ops = 0;
    for (int q1 = 0; q1 < n; ++q1) {
        for (int q2 = 0; q2 < n; ++q2) {
            if (q1 == q2) continue;
            SCOPED_TRACE("gate on (" + std::to_string(q1) + ", " + std::to_string(q2) + ")");
            const auto U = random_gate4(seed++);
            const std::size_t calls_before = s.svd_call_count();
            s.apply_two_qubit_gate(U, q1, q2);
            want = apply_dense(want, n, q1, q2, U);
            ++ops;

            const auto [lo, hi] = s.open_span();
            EXPECT_EQ(lo, hi) << "the span is not a single site";
            EXPECT_GE(lo, std::min(q1, q2));
            EXPECT_LE(lo, std::max(q1, q2));
            ASSERT_TRUE(is_canonical(s));

            const std::size_t crossed =
                static_cast<std::size_t>(std::abs(q2 - q1) - 1);
            EXPECT_EQ(s.svd_call_count() - calls_before, 1 + 2 * crossed)
                << "a gate spanning " << std::abs(q2 - q1)
                << " sites performs one split plus two per site crossed";

            const auto& t = s.tensors();
            for (int i = 0; i + 1 < n; ++i) {
                EXPECT_LE(t[static_cast<std::size_t>(i)].bond_right, bond_ceiling(n, i))
                    << "bond " << i << " is wider than any state can need";
            }
        }
    }
    const double norm = std::sqrt(v11301::norm_sq(want));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
              amp_tol(n, 3 * ops, norm) + truncation_distance(s));
}

TEST(V11301CanonicalForm, AlwaysFocusesBeforeASplitAndAutoRunsInPlace) {
    // From a chain whose span is the whole register, a split at (2, 3) under
    // Always moves the centre onto the pair first, rewriting the outer sites
    // with QR and LQ steps. Under Auto with a cap that cannot bind it runs where
    // the chain stands: only the two sites of the block change, and the span
    // stays as wide as it was.
    const int n = 6;
    const auto sites = v11301::random_chain(n, {2, 3, 3, 3, 2}, 303);
    const auto U = random_gate4(3030);
    const Amplitudes want = apply_dense(v11301::dense(chain_of(sites, 64)), n, 2, 3, U);

    MPSState always = chain_of(sites, 64, CanonicalForm::Always);
    always.apply_two_qubit_gate(U, 2, 3);
    EXPECT_EQ(always.open_span(), (Span{3, 3}));
    EXPECT_TRUE(is_canonical(always));
    EXPECT_FALSE(v11301::same_site(always.tensors()[0], sites[0]))
        << "Always did not move the centre through site 0";
    EXPECT_FALSE(v11301::same_site(always.tensors()[5], sites[5]))
        << "Always did not move the centre through site 5";

    MPSState in_place = chain_of(sites, 64, CanonicalForm::Auto);
    // The rank bound min(2 * 3, 2 * 3) = 6 does not exceed 64.
    in_place.apply_two_qubit_gate(U, 2, 3);
    EXPECT_EQ(in_place.open_span(), (Span{0, n - 1}));
    for (int q : {0, 1, 4, 5}) {
        EXPECT_TRUE(v11301::same_site(in_place.tensors()[static_cast<std::size_t>(q)],
                                      sites[static_cast<std::size_t>(q)]))
            << "Auto rewrote site " << q << ", outside the block it split";
    }
    EXPECT_TRUE(is_canonical(in_place));

    // The 6 x 6 block is full rank, so neither split discards anything, and
    // the in-place one is compared at rounding.
    ASSERT_EQ(in_place.truncation_error(), 0.0);
    const double norm = std::sqrt(v11301::norm_sq(want));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(always), want),
              amp_tol(n, 8, norm) + truncation_distance(always));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(in_place), want), amp_tol(n, 8, norm));
    EXPECT_EQ(always.svd_call_count(), 1u);
    EXPECT_EQ(in_place.svd_call_count(), 1u);
}

TEST(V11301CanonicalForm, AutoFocusesExactlyWhenTheCapCanBind) {
    // The rank bound of the block at (1, 2) is min(2 chi_L(1), 2 chi_R(2)) =
    // min(2 * 2, 2 * 2) = 4. A cap of 4 cannot bind, so Auto splits in place; a
    // cap of 3 can, so Auto focuses first. The comparison is strict.
    const int n = 5;
    const auto sites = v11301::random_chain(n, {2, 2, 2, 2}, 404);
    const auto U = random_gate4(4040);
    const int bound = std::min(2 * sites[1].bond_left, 2 * sites[2].bond_right);
    ASSERT_EQ(bound, 4);

    MPSState at_bound = chain_of(sites, bound, CanonicalForm::Auto);
    at_bound.apply_two_qubit_gate(U, 1, 2);
    EXPECT_EQ(at_bound.open_span(), (Span{0, n - 1}))
        << "a cap equal to the rank bound cannot bind, so the split runs in place";
    for (int q : {0, 3, 4}) {
        EXPECT_TRUE(v11301::same_site(at_bound.tensors()[static_cast<std::size_t>(q)],
                                      sites[static_cast<std::size_t>(q)]));
    }

    MPSState below = chain_of(sites, bound - 1, CanonicalForm::Auto);
    below.apply_two_qubit_gate(U, 1, 2);
    EXPECT_EQ(below.open_span(), (Span{2, 2}))
        << "a cap below the rank bound can bind, so the centre moves first";
    EXPECT_TRUE(is_canonical(below));
    EXPECT_FALSE(v11301::same_site(below.tensors()[4], sites[4]));
}

TEST(V11301CanonicalForm, AutoFocusesEverySplitOnceTheCutoffIsRaised) {
    // A cutoff above MPS_DEFAULT_CUTOFF asks for real weight to be dropped, and
    // a fraction of a distorted block is not a fraction of the state, so Auto
    // focuses whatever the cap. At the default itself it does not.
    const int n = 5;
    const auto sites = v11301::random_chain(n, {2, 2, 2, 2}, 505);
    const auto U = random_gate4(5050);

    MPSState raised = chain_of(sites, 64, CanonicalForm::Auto,
                               std::nextafter(MPS_DEFAULT_CUTOFF, 1.0));
    raised.apply_two_qubit_gate(U, 1, 2);
    EXPECT_EQ(raised.open_span(), (Span{2, 2}));
    EXPECT_TRUE(is_canonical(raised));

    MPSState at_default = chain_of(sites, 64, CanonicalForm::Auto, MPS_DEFAULT_CUTOFF);
    at_default.apply_two_qubit_gate(U, 1, 2);
    EXPECT_EQ(at_default.open_span(), (Span{0, n - 1}));
}

TEST(V11301CanonicalForm, CanonicalizeMovesTheCentreAndNothingElse) {
    // A gauge change: the state is unchanged to rounding, and nothing the chain
    // reports about its splits moves, the fidelity figures included, which
    // are compared to the bit. The chain first takes a capped split so those
    // figures are not trivially 1.
    const int n = 6;
    MPSState base = chain_of(v11301::random_chain(n, {2, 4, 5, 4, 2}, 606), 3);
    base.apply_two_qubit_gate(random_gate4(6060), 2, 3);
    ASSERT_LT(*base.fidelity_estimate(), 1.0) << "the capped split truncated nothing";
    const Amplitudes want = v11301::dense(base);
    const double norm = std::sqrt(v11301::norm_sq(want));
    const v11301::Profile before = v11301::profile_of(base);

    for (int c = 0; c < n; ++c) {
        SCOPED_TRACE("canonicalize(" + std::to_string(c) + ")");
        MPSState s = base;
        s.canonicalize(c);
        EXPECT_EQ(s.open_span(), (Span{c, c}));
        EXPECT_TRUE(is_canonical(s));
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want), amp_tol(n, 2 * n, norm));
        EXPECT_TRUE(v11301::profile_of(s) == before)
            << "before " << v11301::describe(before) << ", after "
            << v11301::describe(v11301::profile_of(s));

        const auto settled = s.tensors();
        s.canonicalize(c);
        EXPECT_TRUE(same_sites(s.tensors(), settled))
            << "canonicalize on the centre it already has changed the chain";
    }

    MPSState s = base;
    const auto sites_before = s.tensors();
    EXPECT_THROW(s.canonicalize(-1), std::out_of_range);
    EXPECT_THROW(s.canonicalize(n), std::out_of_range);
    EXPECT_TRUE(same_sites(s.tensors(), sites_before));
    EXPECT_EQ(s.open_span(), base.open_span());
}

TEST(V11301CanonicalForm, CanonicalizeTrimsABondWiderThanItsRank) {
    // A QR step leaves the bond at k = min(rows, cols) of the site it factors,
    // which is lossless. Moving right from a hand-built chain with bonds of 5,
    // site 0 is 2 x 5 and site 1 then 4 x 5; moving left, site 2 is 5 x 2 and
    // site 1 then 5 x 4.
    const auto sites = v11301::random_chain(3, {5, 5}, 707);
    const Amplitudes want = v11301::dense(chain_of(sites, 64));
    const double norm = std::sqrt(v11301::norm_sq(want));

    MPSState right = chain_of(sites, 64);
    right.canonicalize(2);
    EXPECT_EQ(right.tensors()[0].bond_right, std::min(2 * 1, 5));
    EXPECT_EQ(right.tensors()[1].bond_right, std::min(2 * 2, 5));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(right), want), amp_tol(3, 4, norm));

    MPSState left = chain_of(sites, 64);
    left.canonicalize(0);
    EXPECT_EQ(left.tensors()[1].bond_right, std::min(5, 2 * 1));
    EXPECT_EQ(left.tensors()[0].bond_right, std::min(5, 2 * 2));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(left), want), amp_tol(3, 4, norm));
    EXPECT_TRUE(is_canonical(left));
    EXPECT_TRUE(is_canonical(right));
}

TEST(V11301CanonicalForm, RebuildFromStatevectorCentresOnTheLastSite) {
    const int n = 5;
    MPSState s = chain_of(v11301::random_chain(n, {2, 3, 3, 2}, 808), 64);
    const auto target = v11301::random_chain(n, {2, 4, 4, 2}, 809);
    const Amplitudes want = v11301::dense(chain_of(target, 64));
    const std::size_t calls = s.svd_call_count();

    s.rebuild_from_statevector(v11301::to_statevector(want, n));
    EXPECT_EQ(s.open_span(), (Span{n - 1, n - 1}));
    EXPECT_TRUE(is_canonical(s));
    EXPECT_EQ(s.svd_call_count() - calls, static_cast<std::size_t>(n - 1));
    const double norm = std::sqrt(v11301::norm_sq(want));
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
              amp_tol(n, n, norm) + truncation_distance(s));

    // A register of the wrong width is refused before anything is touched.
    const auto sites_before = s.tensors();
    const v11301::Profile before = v11301::profile_of(s);
    const Span span_before = s.open_span();
    EXPECT_THROW(s.rebuild_from_statevector(Statevector(n + 1)), std::invalid_argument);
    EXPECT_TRUE(same_sites(s.tensors(), sites_before));
    EXPECT_TRUE(v11301::profile_of(s) == before);
    EXPECT_EQ(s.open_span(), span_before);
}

TEST(V11301CanonicalForm, ARebuildThatThrowsLeavesTheChainAsItWas) {
    // The rebuild is built beside the chain, so a sweep that throws part way
    // leaves the tensors, the span and the figures as they were. A non-finite
    // amplitude makes the first split fail on every rung, so no split
    // completes and the counters stay put as well.
    const int n = 4;
    MPSState s = chain_of(v11301::random_chain(n, {2, 3, 2}, 909), 2);
    s.apply_two_qubit_gate(random_gate4(9090), 1, 2);
    s.canonicalize(1);
    const auto sites_before = s.tensors();
    const v11301::Profile before = v11301::profile_of(s);
    const Span span_before = s.open_span();

    std::vector<Complex128> amps(std::size_t{1} << n, Complex128(0.25, 0.0));
    amps[5] = Complex128(quiet_nan_strict(), 0.0);
    Statevector bad(n);
    bad.set_amplitudes(amps, {Validation::Ignore});
    ASSERT_FALSE(is_finite_strict(bad.real_parts[5])) << "the poison did not land";

    EXPECT_THROW(s.rebuild_from_statevector(bad), std::runtime_error);
    EXPECT_TRUE(same_sites(s.tensors(), sites_before));
    EXPECT_EQ(s.open_span(), span_before);
    EXPECT_TRUE(v11301::profile_of(s) == before)
        << "before " << v11301::describe(before) << ", after "
        << v11301::describe(v11301::profile_of(s));
}

TEST(V11301CanonicalForm, MeasureQubitLeavesTheCentreOnTheQubit) {
    const int n = 5;
    const auto sites = v11301::random_chain(n, {2, 4, 4, 2}, 1111);
    for (int q = 0; q < n; ++q) {
        SCOPED_TRACE("measure qubit " + std::to_string(q));
        MPSState s = chain_of(sites, 64);
        s.normalize();
        std::mt19937_64 rng(1100 + static_cast<std::uint64_t>(q));
        s.measure_qubit(q, rng);
        EXPECT_EQ(s.open_span(), (Span{q, q}));
        EXPECT_TRUE(is_canonical(s));
        EXPECT_NEAR(s.norm_sq(), 1.0, kSlack * n * kEps)
            << "the collapse renormalises the centre, and the centre carries the norm";
        EXPECT_FALSE(s.fidelity_estimate().has_value());
        EXPECT_FALSE(s.fidelity_lower_bound().has_value());
    }
}

TEST(V11301CanonicalForm, MeasureSequentialLeavesACollapsedChainCentredOnTheLastQubit) {
    const int n = 5;
    MPSState s = chain_of(v11301::random_chain(n, {2, 4, 4, 2}, 1212), 64);
    s.normalize();
    std::mt19937_64 rng(1212);
    const std::string bits = s.measure_sequential(rng);
    ASSERT_EQ(bits.size(), static_cast<std::size_t>(n));
    EXPECT_EQ(s.open_span(), (Span{n - 1, n - 1}));
    EXPECT_TRUE(is_canonical(s));

    // The chain is now the measured basis state, qubit 0 the rightmost
    // character, with unit modulus there and nothing anywhere else.
    std::size_t index = 0;
    for (int q = 0; q < n; ++q)
        if (bits[static_cast<std::size_t>(n - 1 - q)] == '1') index |= std::size_t{1} << q;
    const Amplitudes a = v11301::dense(s);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i == index) {
            EXPECT_NEAR(std::abs(a[i]), 1.0, kSlack * n * kEps);
        } else {
            EXPECT_LE(std::abs(a[i]), kSlack * n * kEps) << "amplitude " << i;
        }
    }
}

TEST(V11301CanonicalForm, ARunWithMidCircuitCollapseReturnsACanonicalChain) {
    // MEASURE and RESET move the centre, collapse, and hand the chain on to the
    // next gate. Whatever path the run takes, the chain it returns satisfies
    // the invariant.
    // The cx on qubit 1 after its MEASURE makes the shots run per trajectory.
    QuantumCircuit qc(5, 5);
    qc.h(0).cx(0, 1).ry(PI / 3.0, 2).cx(1, 2).measure(1, 1).cx(1, 3).reset(2);
    qc.h(2).cx(2, 3).cx(0, 4).rz(PI / 5.0, 4).reset(0).cx(3, 4).measure(4, 4);
    for (CanonicalForm form : {CanonicalForm::Always, CanonicalForm::Auto}) {
        SCOPED_TRACE(to_string(form));
        MPSSimulator sim;
        sim.canonical_form = form;
        const auto one = sim.run(qc, 8, 0, 131);
        EXPECT_TRUE(is_canonical(one.final_state)) << "single trajectory";
        const auto shots = sim.run(qc, 8, 16, 131);
        EXPECT_TRUE(is_canonical(shots.final_state)) << "per-shot trajectories";
    }
}

TEST(V11301CanonicalForm, ANonFiniteSiteReachesTheLadderThroughTheCentreSteps) {
    // The poison sits on site 0 and the gate acts on (1, 2). Under Always the
    // split first moves the centre from site 0 onto the pair, and the QR step
    // carries the NaN into site 1, so the block the ladder receives is
    // non-finite and the split must throw rather than continue.
    MPSState s(3, 8);
    const double poison = quiet_nan_strict();
    const std::array<Complex128, 4> nan_gate = {
        Complex128(poison, 0.0), Complex128(0.0, 0.0),
        Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
    s.apply_single_qubit_gate(nan_gate, 0, {Validation::Ignore});
    ASSERT_FALSE(is_finite_strict(s.tensors()[0].data[0].real))
        << "the gate under Ignore did not carry the NaN into site 0";
    ASSERT_EQ(s.open_span(), (Span{0, 0}));

    std::array<Complex128, 16> CZ{};
    CZ[0] = CZ[5] = CZ[10] = Complex128(1.0, 0.0);
    CZ[15] = Complex128(-1.0, 0.0);
    EXPECT_THROW(s.apply_two_qubit_gate(CZ, 1, 2), std::runtime_error);
}

TEST(V11301CanonicalForm, ReversedAndSwapRoutedOperandsAnswerToTheCallersPolicy) {
    // The caller's matrix is validated once, under the caller's policy, and
    // then applied however the chain routes it: in either operand order and
    // across a SWAP chain. A matrix accepted under Ignore is applied as given,
    // one outside tolerance but inside a caller's wider atol is accepted, and
    // Warn reports once and applies.
    const int n = 4;
    std::array<Complex128, 16> U = random_gate4(1414);
    U[5] = U[5] * 1.001;  // no longer unitary, by about 1e-3
    const std::array<Complex128, 16> V = [] {
        auto m = random_gate4(1415);
        m[6] = m[6] + Complex128(1e-9, 0.0);  // off by about 1e-9
        return m;
    }();

    for (const auto& [q1, q2] : std::vector<Span>{{1, 0}, {3, 0}, {0, 3}, {2, 1}}) {
        SCOPED_TRACE("operands (" + std::to_string(q1) + ", " + std::to_string(q2) + ")");
        const auto sites = v11301::random_chain(n, {2, 3, 2}, 1416);
        MPSState s = chain_of(sites, 64);
        const Amplitudes before = v11301::dense(s);
        bool applied = true;
        try {
            s.apply_two_qubit_gate(U, q1, q2, {Validation::Ignore});
        } catch (const std::exception& e) {
            applied = false;
            ADD_FAILURE() << "Ignore measures nothing, so nothing may reject the "
                             "operand, but the call threw: " << e.what();
        }
        if (applied) {
            const Amplitudes want = apply_dense(before, n, q1, q2, U);
            const double norm = std::sqrt(v11301::norm_sq(want));
            EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), want),
                      amp_tol(n, 8, norm) + truncation_distance(s));
        }

        MPSState t = chain_of(sites, 64);
        EXPECT_NO_THROW(t.apply_two_qubit_gate(V, q1, q2, {Validation::Throw, 1e-6}))
            << "inside the caller's atol, so nothing may reject it";

        MPSState w = chain_of(sites, 64);
        {
            WarningCapture cap;
            EXPECT_NO_THROW(w.apply_two_qubit_gate(U, q1, q2, {Validation::Warn}));
            EXPECT_EQ(cap.count(), 1u) << "Warn reports the operand once";
        }

        MPSState r = chain_of(sites, 64);
        EXPECT_THROW(r.apply_two_qubit_gate(U, q1, q2), std::invalid_argument)
            << "the default policy still throws";
        EXPECT_TRUE(same_sites(r.tensors(), sites)) << "a rejected gate changed the chain";
    }
}

// =============================================================================
// #128: a capped split keeps the state's leading Schmidt directions
// =============================================================================

namespace {

// The bond split of a two-site block in whatever gauge the sites carry, done
// here the way a split without a centre does it: contract, apply U (MSB-first
// within the block, as the adjacent kernel reads it), SVD, keep `keep`, and
// put S on the left site. What #128 reported the library doing.
std::vector<MPSTensor> split_in_place(std::vector<MPSTensor> sites, int q,
                                      const std::array<Complex128, 16>& U_lsb,
                                      int keep) {
    // The project order has the first operand on bit 0; the block reads the
    // left site's physical index as bit 1.
    std::array<Cplx, 16> G{};
    const auto swap01 = [](int i) { return ((i & 1) << 1) | ((i >> 1) & 1); };
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            G[static_cast<std::size_t>(r * 4 + c)] =
                v11301::to_std(U_lsb[static_cast<std::size_t>(swap01(r) * 4 + swap01(c))]);

    const MPSTensor& A = sites[static_cast<std::size_t>(q)];
    const MPSTensor& B = sites[static_cast<std::size_t>(q + 1)];
    const int bl = A.bond_left, bm = A.bond_right, br = B.bond_right;
    const int rows = 2 * bl, cols = 2 * br;
    std::vector<Cplx> theta(static_cast<std::size_t>(rows) * cols, Cplx(0.0, 0.0));
    for (int l = 0; l < bl; ++l)
        for (int po1 = 0; po1 < 2; ++po1)
            for (int po2 = 0; po2 < 2; ++po2)
                for (int r = 0; r < br; ++r) {
                    Cplx acc(0.0, 0.0);
                    for (int pi1 = 0; pi1 < 2; ++pi1)
                        for (int pi2 = 0; pi2 < 2; ++pi2) {
                            Cplx block(0.0, 0.0);
                            for (int m = 0; m < bm; ++m)
                                block += v11301::to_std(A(l, pi1, m)) *
                                         v11301::to_std(B(m, pi2, r));
                            acc += G[static_cast<std::size_t>((po1 * 2 + po2) * 4 +
                                                              (pi1 * 2 + pi2))] * block;
                        }
                    // Column-major for the seam.
                    theta[static_cast<std::size_t>(po2 * br + r) * rows +
                          static_cast<std::size_t>(l * 2 + po1)] = acc;
                }
    const int k = std::min(rows, cols);
    std::vector<Cplx> Um(static_cast<std::size_t>(rows) * k), Vm(static_cast<std::size_t>(cols) * k);
    std::vector<double> S(static_cast<std::size_t>(k));
    if (!detail::svd_thin(theta.data(), rows, cols, detail::MatrixOrder::ColMajor,
                          SVDMethod::EigenJacobi, Um.data(), S.data(), Vm.data())) {
        throw std::logic_error("split_in_place: the seam refused the block");
    }
    MPSTensor A2(bl, keep), B2(keep, br);
    for (int l = 0; l < bl; ++l)
        for (int p = 0; p < 2; ++p)
            for (int j = 0; j < keep; ++j)
                A2(l, p, j) = v11301::to_c128(
                    Um[static_cast<std::size_t>(j) * rows + (l * 2 + p)] * S[static_cast<std::size_t>(j)]);
    for (int j = 0; j < keep; ++j)
        for (int p = 0; p < 2; ++p)
            for (int r = 0; r < br; ++r)
                B2(j, p, r) = v11301::to_c128(
                    std::conj(Vm[static_cast<std::size_t>(j) * cols + (p * br + r)]));
    sites[static_cast<std::size_t>(q)] = std::move(A2);
    sites[static_cast<std::size_t>(q + 1)] = std::move(B2);
    return sites;
}

}  // namespace

TEST(V11301SchmidtTruncation, ACappedSplitFromADistortedGaugeKeepsTheLeadingSchmidtDirections) {
    // A six-qubit chain handed in by set_tensors, with G G^-1 inserted on the
    // two bonds around the block, G conditioned at 1e3: the same state in a
    // gauge far from canonical. (On the bond inside the block the two factors
    // would meet in the contraction and cancel.) A random gate on (2, 3) at a
    // cap of 2 must leave
    // exactly the projection of the gated state onto its two leading Schmidt
    // directions across the cut 2 | 3, the optimal truncation, under both
    // policies (the cap binds, so Auto focuses too).
    const int n = 6;
    const int q = 2;
    const int keep = 2;
    auto sites = v11301::random_chain(n, {2, 4, 4, 4, 2}, 1515);
    v11301::distort_bond(sites, q - 1, 1e3, 1516);
    v11301::distort_bond(sites, q + 1, 1e3, 1518);
    const auto U = random_gate4(1517);

    const Amplitudes before = v11301::dense(chain_of(sites, 64));
    const Amplitudes gated = apply_dense(before, n, q, q + 1, U);
    const v11301::Schmidt sch = v11301::schmidt(gated, 2, n, q + 1);
    const Amplitudes optimal = v11301::project_top(gated, 2, n, q + 1, keep);

    double kept = 0.0, total = 0.0;
    for (std::size_t i = 0; i < sch.sigma.size(); ++i) {
        const double w = sch.sigma[i] * sch.sigma[i];
        total += w;
        if (static_cast<int>(i) < keep) kept += w;
    }
    const double norm = std::sqrt(total);
    const double tol = amp_tol(n, 4 * n, norm);
    const double gap = sch.sigma[static_cast<std::size_t>(keep - 1)] -
                       sch.sigma[static_cast<std::size_t>(keep)];
    ASSERT_GT(gap, 1e3 * tol)
        << "the fixture's Schmidt gap is too small for the kept subspace to be "
           "well defined";
    // Perturbing the state by E moves the kept subspace by at most ||E|| / gap
    // (Davis-Kahan), so the projection may move by that times the norm.
    const double proj_tol = tol * (1.0 + sch.sigma[0] / gap);

    // The control: the same split in the distorted gauge, as a split without a
    // centre performs it, lands measurably away from the optimum. Without this
    // the assertions below could pass on a fixture no gauge could spoil.
    {
        MPSState naive(n, 64);
        naive.set_tensors(split_in_place(sites, q, U, keep));
        ASSERT_GT(v11301::max_abs_diff(v11301::dense(naive), optimal), 1e3 * proj_tol)
            << "the distorted gauge does not change which directions a split "
               "keeps, so this fixture cannot detect #128";
    }

    for (CanonicalForm form : {CanonicalForm::Always, CanonicalForm::Auto}) {
        SCOPED_TRACE(to_string(form));
        MPSState s = chain_of(sites, keep, form);
        const double norm_before = s.norm_sq();
        s.apply_two_qubit_gate(U, q, q + 1);

        EXPECT_EQ(s.svd_call_count(), 1u);
        EXPECT_EQ(s.tensors()[static_cast<std::size_t>(q)].bond_right, keep);
        EXPECT_TRUE(is_canonical(s));
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), optimal), proj_tol)
            << "the split did not keep the state's leading Schmidt directions";

        // What the split reports is what the Schmidt spectrum predicts: the
        // weight of the discarded directions, which is also exactly how far
        // the norm fell, and as the fraction of the state it removed, the
        // estimate and the floor, which agree for a single split.
        const double weight_tol = kSlack * static_cast<double>(1 << n) * kEps * total;
        EXPECT_NEAR(s.truncation_error(), total - kept, weight_tol);
        EXPECT_NEAR(norm_before - s.norm_sq(), s.truncation_error(), weight_tol);
        ASSERT_TRUE(s.fidelity_estimate().has_value());
        ASSERT_TRUE(s.fidelity_lower_bound().has_value());
        EXPECT_NEAR(*s.fidelity_estimate(), kept / total, weight_tol / total);
        EXPECT_NEAR(*s.fidelity_lower_bound(), kept / total, weight_tol / total);
    }
}

// =============================================================================
// Auto against Always
// =============================================================================

TEST(V11301CanonicalForm, AutoAndAlwaysAgreeInTheExactRegime) {
    // n = 10 at a cap of 2^5 cannot truncate anything real. Both policies must
    // reproduce the dense state, and therefore each other, to rounding plus
    // the weight budget's dust, and neither may spend more than that dust.
    const int n = 10;
    const QuantumCircuit qc = v11301::brickwork(n, 6);
    StatevectorSimulator svsim;
    const Amplitudes want = v11301::dense(svsim.run(qc).final_state);

    Amplitudes got[2];
    std::size_t splits[2] = {0, 0};
    const CanonicalForm forms[2] = {CanonicalForm::Always, CanonicalForm::Auto};
    for (int i = 0; i < 2; ++i) {
        SCOPED_TRACE(to_string(forms[i]));
        MPSSimulator sim;
        sim.canonical_form = forms[i];
        const MPSState s = sim.run(qc, 1 << (n / 2), 0, 17).final_state;
        got[i] = v11301::dense(s);
        splits[i] = s.svd_call_count();
        const double tol = fidelity_tol(qc.instructions.size(), want.size(), splits[i]);
        EXPECT_GE(v11301::fidelity(got[i], want), 1.0 - tol);
        EXPECT_NEAR(v11301::norm_sq(got[i]), 1.0, tol);
        EXPECT_LE(s.truncation_error(),
                  s.cutoff * static_cast<double>(s.svd_call_count()))
            << "the exact regime discarded more than the budget's dust";
        EXPECT_TRUE(is_canonical(s));
    }
    EXPECT_GE(v11301::fidelity(got[0], got[1]),
              1.0 - fidelity_tol(2 * qc.instructions.size(), want.size(),
                                 splits[0] + splits[1]));
}

TEST(V11301CanonicalForm, AutoKeepsWiderBondsOnTheMeasuredBrickwork) {
    // The measurement behind Always being the default: on the 16-qubit, 8-layer
    // brickwork at a cap of 2^8, where nothing real can be truncated, splits
    // run in place keep directions that are rounding noise in the state but
    // not in their distorted blocks, so Auto's widest bond exceeds Always's at
    // the same fidelity. Asserted as that inequality, not as the pair measured.
    const int n = 16;
    const QuantumCircuit qc = v11301::brickwork(n, 8);
    StatevectorSimulator svsim;
    const Amplitudes want = v11301::dense(svsim.run(qc).final_state);
    int widest[2] = {0, 0};
    const CanonicalForm forms[2] = {CanonicalForm::Always, CanonicalForm::Auto};
    for (int i = 0; i < 2; ++i) {
        SCOPED_TRACE(to_string(forms[i]));
        MPSSimulator sim;
        sim.canonical_form = forms[i];
        const MPSState s = sim.run(qc, 1 << (n / 2), 0, 17).final_state;
        widest[i] = s.current_max_bond_dim();
        EXPECT_GE(v11301::fidelity(v11301::dense(s), want),
                  1.0 - fidelity_tol(qc.instructions.size(), want.size(),
                                     s.svd_call_count()));
    }
    EXPECT_GT(widest[1], widest[0])
        << "Auto's widest bond " << widest[1] << " does not exceed Always's "
        << widest[0] << " on the circuit where 1.1.30.0 measured it doing so";
}

// =============================================================================
// Reads through the open span
// =============================================================================

TEST(V11301CanonicalForm, NormAndMarginalsReadThroughAnySpan) {
    // norm_sq contracts only the span and probabilities_single only the span's
    // stretch on each side of the qubit, taking every site outside it as the
    // identity. On every span configuration, for every qubit, both must match
    // the dense state: raw figures, summing to the norm.
    const int n = 6;
    const auto sites = v11301::random_chain(n, {2, 3, 4, 3, 2}, 1717);
    for (const auto& c : span_cases(sites, 16)) {
        SCOPED_TRACE(c.name);
        const Amplitudes a = v11301::dense(c.chain);
        const double want = v11301::norm_sq(a);
        const double tol = kSlack * static_cast<double>(4 * n + (1 << n)) * kEps * want;
        EXPECT_NEAR(c.chain.norm_sq(), want, tol);
        for (int q = 0; q < n; ++q) {
            const auto p = c.chain.probabilities_single(q);
            const auto ref = v11301::qubit_marginals(a, q);
            ASSERT_EQ(p.size(), 2u);
            EXPECT_NEAR(p[0], ref[0], tol) << "qubit " << q;
            EXPECT_NEAR(p[1], ref[1], tol) << "qubit " << q;
            EXPECT_NEAR(p[0] + p[1], c.chain.norm_sq(), tol) << "qubit " << q;
        }
        EXPECT_THROW(c.chain.probabilities_single(-1), std::out_of_range);
        EXPECT_THROW(c.chain.probabilities_single(n), std::out_of_range);
    }
}

TEST(V11301CanonicalForm, NormalizeScalesOnlyTheFirstSiteOfTheSpan) {
    // Scaling a site outside the span would break its orthonormality, so
    // normalize touches the span's first site and nothing else, leaves the
    // span, the counters and the figures alone, and yields the dense state
    // divided by its norm.
    const int n = 6;
    const auto sites = v11301::random_chain(n, {2, 3, 4, 3, 2}, 1818);
    for (auto& c : span_cases(sites, 16)) {
        SCOPED_TRACE(c.name);
        MPSState& s = c.chain;
        const Amplitudes a = v11301::dense(s);
        const double norm = std::sqrt(v11301::norm_sq(a));
        const auto before = s.tensors();
        const Span span = s.open_span();
        const v11301::Profile prof = v11301::profile_of(s);

        s.normalize();
        EXPECT_EQ(s.open_span(), span);
        EXPECT_TRUE(v11301::profile_of(s) == prof);
        for (int q = 0; q < n; ++q) {
            const bool same = v11301::same_site(s.tensors()[static_cast<std::size_t>(q)],
                                                before[static_cast<std::size_t>(q)]);
            if (q == span.first) {
                EXPECT_FALSE(same) << "the first site of the span was not scaled";
            } else {
                EXPECT_TRUE(same) << "site " << q << " changed, outside the scaled site";
            }
        }
        EXPECT_TRUE(is_canonical(s));
        EXPECT_NEAR(s.norm_sq(), 1.0, kSlack * static_cast<double>(4 * n) * kEps);
        EXPECT_TRUE(s.is_normalized());
        Amplitudes scaled = a;
        for (auto& z : scaled) z /= norm;
        EXPECT_LE(v11301::max_abs_diff(v11301::dense(s), scaled), amp_tol(n, 2 * n, 1.0));
    }
}

TEST(V11301BondSpectrum, EntropyAtEveryCutReadsThroughAnySpan) {
    // The entropy observer reads a prefix or suffix cut of an MPS off the bond,
    // from two environment Grams that start from the identity wherever the
    // open span ends. Each chain is handed to a run as its initial state, which
    // keeps its span, and observed before any instruction; every cut, on every
    // span configuration, in both von Neumann and Renyi-2, must match the
    // entropy of the dense Schmidt spectrum.
    const int n = 6;
    const auto sites = v11301::random_chain(n, {2, 3, 4, 3, 2}, 1919);
    // The eigensolve route, as the entropy suite bounds it.
    constexpr double kLoose = 1e-7;

    for (const auto& c : span_cases(sites, 16)) {
        SCOPED_TRACE(c.name);
        const Amplitudes a = v11301::dense(c.chain);
        for (int cut = 1; cut < n; ++cut) {
            const v11301::Schmidt sch = v11301::schmidt(a, 2, n, cut);
            double total = 0.0;
            for (double s : sch.sigma) total += s * s;
            double vn = 0.0, purity = 0.0;
            for (double s : sch.sigma) {
                const double p = s * s / total;
                if (p > 0.0) vn -= p * std::log2(p);
                purity += p * p;
            }
            const double renyi2 = -std::log2(purity);

            std::vector<int> prefix, suffix;
            for (int q = 0; q < cut; ++q) prefix.push_back(q);
            for (int q = cut; q < n; ++q) suffix.push_back(q);

            for (const auto& [region, order, want] :
                 std::vector<std::tuple<std::vector<int>, double, double>>{
                     {prefix, 1.0, vn}, {suffix, 1.0, vn},
                     {prefix, 2.0, renyi2}, {suffix, 2.0, renyi2}}) {
                SCOPED_TRACE("cut " + std::to_string(cut) + (region == prefix ? " prefix" : " suffix") +
                             " order " + std::to_string(order));
                auto obs = std::make_shared<EntropyObserver>(region, order);
                RunPlan plan;
                plan.options.cost = Cost::Unlimited;
                plan.initial = InitialState::from(
                    std::shared_ptr<const MPSState>(std::make_shared<MPSState>(c.chain)));
                plan.observations.observe(Anchor::at_start(), obs);
                MPSSimulator sim;
                sim.run(QuantumCircuit(n), 16, 0, 5, plan);
                ASSERT_EQ(obs->count(), 1u);
                EXPECT_NEAR(obs->value(), want, kLoose);
                EXPECT_TRUE(obs->is_entanglement());
            }
        }
    }
}

// =============================================================================
// set_tensors
// =============================================================================

namespace {

struct BadChain {
    std::string what;       // the violation
    std::string message;    // a substring the refusal must carry
    std::function<void(std::vector<MPSTensor>&)> mutate;
};

}  // namespace

TEST(V11301SetTensors, AValidChainOpensTheSpanAndResetsOnlyTheFigures) {
    // The handed-in chain becomes the reference later truncation is measured
    // against, so both figures read exactly 1. The counters and
    // truncation_error() describe splits this object performed and are kept.
    const int n = 4;
    MPSState s(n, 2);
    s.apply_single_qubit_gate(random_gate2(1), 0);
    s.apply_two_qubit_gate(random_gate4(2), 0, 1);
    s.apply_two_qubit_gate(random_gate4(3), 1, 2);
    s.apply_two_qubit_gate(random_gate4(4), 2, 3);
    s.apply_two_qubit_gate(random_gate4(5), 0, 3);
    ASSERT_LT(*s.fidelity_estimate(), 1.0);
    const v11301::Profile before = v11301::profile_of(s);

    const auto sites = v11301::random_chain(n, {2, 3, 2}, 2020);
    s.set_tensors(sites);
    EXPECT_TRUE(same_sites(s.tensors(), sites)) << "the chain is not the one handed in";
    EXPECT_EQ(s.open_span(), (Span{0, n - 1}));
    EXPECT_EQ(s.fidelity_estimate(), std::optional<double>(1.0));
    EXPECT_EQ(s.fidelity_lower_bound(), std::optional<double>(1.0));
    EXPECT_EQ(s.svd_call_count(), before.svd_calls);
    EXPECT_EQ(s.truncation_error(), before.truncation);
    EXPECT_EQ(s.jacobi_rescue_count(), before.jacobi);
    EXPECT_EQ(s.gram_fallback_count(), before.gram);
    EXPECT_EQ(s.svd_time_ns(), before.nanos);
    EXPECT_EQ(s.max_bond_dim, 2) << "the settings are not the chain's to change";

    // A collapsed chain's empty figures are restored too.
    std::mt19937_64 rng(7);
    s.measure_qubit(1, rng);
    ASSERT_FALSE(s.fidelity_estimate().has_value());
    s.set_tensors(sites);
    EXPECT_EQ(s.fidelity_estimate(), std::optional<double>(1.0));
    EXPECT_EQ(s.fidelity_lower_bound(), std::optional<double>(1.0));
}

TEST(V11301SetTensors, EveryRuleRefusesAndLeavesTheChainUntouched) {
    const int n = 4;
    const auto good = v11301::random_chain(n, {2, 3, 2}, 2121);
    const double nan = quiet_nan_strict();
    const double inf = std::numeric_limits<double>::infinity();

    const std::vector<BadChain> cases = {
        {"no sites", "expected 4 site tensors, got 0",
         [](auto& s) { s.clear(); }},
        {"one site short", "expected 4 site tensors, got 3",
         [](auto& s) { s.pop_back(); }},
        {"one site over", "expected 4 site tensors, got 5",
         [](auto& s) { s.push_back(MPSTensor(1, 1)); }},
        {"a zero left bond", "site 2 has a bond below 1",
         [](auto& s) { s[2].bond_left = 0; }},
        {"a negative right bond", "site 1 has a bond below 1",
         [](auto& s) { s[1].bond_right = -1; }},
        {"a left end bond above 1", "the left end bond must be 1",
         [](auto& s) { s[0] = MPSTensor(2, s[0].bond_right); }},
        {"a right end bond above 1", "the right end bond must be 1",
         [](auto& s) { s[3] = MPSTensor(s[3].bond_left, 2); }},
        {"neighbouring bonds that disagree",
         "site 2 has left bond 4 where its neighbour's right bond is 3",
         [](auto& s) { s[2] = MPSTensor(4, s[2].bond_right); }},
        {"a data array one short", "site 1 holds 11 entries where its bonds need 12",
         [](auto& s) { s[1].data.pop_back(); }},
        {"a data array one over", "site 1 holds 13 entries where its bonds need 12",
         [](auto& s) { s[1].data.push_back(Complex128(0.0, 0.0)); }},
        {"a NaN real part", "site 2 holds a non-finite entry",
         [nan](auto& s) { s[2].data[3].real = nan; }},
        {"a NaN imaginary part", "site 2 holds a non-finite entry",
         [nan](auto& s) { s[2].data[0].imag = nan; }},
        {"a positive infinity", "site 3 holds a non-finite entry",
         [inf](auto& s) { s[3].data[1].real = inf; }},
        {"a negative infinity", "site 0 holds a non-finite entry",
         [inf](auto& s) { s[0].data[0].imag = -inf; }},
        {"two violations, the first named", "site 1 holds 11 entries",
         [nan](auto& s) { s[1].data.pop_back(); s[3].data[0].real = nan; }},
    };

    for (const auto& bad : cases) {
        SCOPED_TRACE(bad.what);
        MPSState s(n, 8);
        s.apply_two_qubit_gate(random_gate4(21), 1, 2);
        const auto sites_before = s.tensors();
        const Span span_before = s.open_span();
        const v11301::Profile prof = v11301::profile_of(s);

        auto sites = good;
        bad.mutate(sites);
        try {
            s.set_tensors(sites);
            ADD_FAILURE() << "accepted a chain with " << bad.what;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(bad.message), std::string::npos)
                << "the refusal does not name the violation: " << e.what();
            EXPECT_NE(std::string(e.what()).find("MPSState::set_tensors"), std::string::npos)
                << e.what();
        }
        EXPECT_TRUE(same_sites(s.tensors(), sites_before));
        EXPECT_EQ(s.open_span(), span_before);
        EXPECT_TRUE(v11301::profile_of(s) == prof);
    }

    // A one-site chain is both ends at once.
    MPSState one(1);
    std::vector<MPSTensor> wide{MPSTensor(1, 2)};
    EXPECT_THROW(one.set_tensors(wide), std::invalid_argument);
    std::vector<MPSTensor> fine{MPSTensor(1, 1)};
    fine[0].data[1] = Complex128(1.0, 0.0);
    EXPECT_NO_THROW(one.set_tensors(fine));
    EXPECT_EQ(one.open_span(), (Span{0, 0}));
}

// =============================================================================
// No qubits
// =============================================================================

TEST(V11301ZeroQubits, AChainWithNoSitesIsTheScalarOne) {
    MPSState s(0);
    EXPECT_EQ(s.open_span(), (Span{0, -1}));
    EXPECT_TRUE(s.tensors().empty());
    EXPECT_EQ(s.norm_sq(), 1.0);
    EXPECT_TRUE(s.is_normalized());
    EXPECT_NO_THROW(s.check_normalized());
    EXPECT_NO_THROW(s.normalize());
    EXPECT_EQ(s.norm_sq(), 1.0);
    EXPECT_EQ(s.open_span(), (Span{0, -1}));
    EXPECT_EQ(s.current_max_bond_dim(), 0);
    EXPECT_EQ(s.fidelity_estimate(), std::optional<double>(1.0));
    EXPECT_EQ(s.fidelity_lower_bound(), std::optional<double>(1.0));

    std::mt19937_64 rng(5);
    const std::mt19937_64 untouched = rng;
    EXPECT_EQ(s.measure_sequential(rng), "");
    EXPECT_EQ(rng, untouched) << "no qubit to measure, so no draw";

    EXPECT_THROW(s.canonicalize(0), std::out_of_range);
    EXPECT_THROW(s.probabilities_single(0), std::out_of_range);
    EXPECT_THROW(s.measure_qubit(0, rng), std::out_of_range);

    EXPECT_NO_THROW(s.set_tensors({}));
    EXPECT_EQ(s.open_span(), (Span{0, -1}));
    EXPECT_THROW(s.set_tensors({MPSTensor(1, 1)}), std::invalid_argument);
    // Running a zero-qubit circuit is refused by every simulator; that is
    // V11301SamplingPath.AZeroQubitRunIsRefusedUpFrontOnEveryBackend. The
    // object itself is valid, and this test is about the object.
}
