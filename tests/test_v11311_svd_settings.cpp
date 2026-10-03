// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - what an MPS split does with a factorisation it cannot use, under
// each setting, on every surface that splits.
//
// 1.1.31.0 replaced svd_rescue with three settings. svd_rejection says what a
// rejected or missing factorisation gets: Fix descends the ladder (Jacobi,
// then the Gram route when svd_accept_gram allows), Throw stops at the first,
// Ignore uses what a rung produced, verified or not. svd_report says whether a
// descent is reported (Warn) or only counted (Silent). Every non-default
// choice emits a one-time note per layer saying it is in force.
//
// A block of NaN is what drives a split down the ladder here: no kernel can
// factorise it, every rung declines, and that is the one rejection a test can
// cause without a seam. A factorisation that is produced and then fails
// verification cannot be caused at all (a kernel that is wrong on demand is
// exactly what the ladder exists because nobody can build), so these stay
// known gaps: Throw at a verified-then-rejected factorisation, Ignore using
// one and counting it in ignored_rejection_count(), the fidelity figures
// withdrawn after it, and the Gram route succeeding with its floored weight
// lowering fidelity_lower_bound().
//
// The surfaces: MPSState split directly, MPSSimulator::run, and QuditMPS. The
// simulator's run names an internal helper as the failure's entry point (a
// known red pinned in test_v11311_failure_point.cpp), so here its messages are
// matched by what follows that name.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "v11311_helpers.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

struct Settings {
    SvdRejection rejection = SvdRejection::Fix;
    bool accept_gram = false;
    SvdReport report = SvdReport::Warn;
};

// The three surfaces that split, each made to split a block of NaN under the
// given settings. Each returns what it threw and what the channel carried.
struct Surface {
    const char* name;
    const char* block;  // the factorisation as the messages name it
    std::function<std::optional<RuntimeFailure>(const Settings&)> split;
};

std::optional<RuntimeFailure> run_quietly(const std::function<void()>& fn) {
    return v11311::thrown<RuntimeFailure>(fn);
}

std::vector<Surface> surfaces() {
    const double nan = quiet_nan_strict();
    return {
        {"MPSState", "the BDC factorisation of a 2x2 block",
         [nan](const Settings& s) {
             MPSState state(3);
             state.svd_rejection = s.rejection;
             state.svd_accept_gram = s.accept_gram;
             state.svd_report = s.report;
             std::array<Complex128, 16> gate;
             gate.fill(Complex128(nan, 0.0));
             return run_quietly([&] { state.apply_two_qubit_gate(gate, 0, 1, {Validation::Ignore}); });
         }},
        {"MPSSimulator", "the BDC factorisation of a 2x2 block",
         [nan](const Settings& s) {
             QuantumCircuit qc(2);
             qc.unitary(std::vector<Complex128>(16, Complex128(nan, 0.0)), {0, 1}, "nan",
                        {Validation::Ignore});
             MPSSimulator sim;
             sim.svd_rejection = s.rejection;
             sim.svd_accept_gram = s.accept_gram;
             sim.svd_report = s.report;
             RunPlan plan;
             plan.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::DoNotSave;
             auto e = run_quietly([&] { (void)sim.run(qc, 8, 0, 1, plan); });
             (void)take_failed_run();
             return e;
         }},
        {"QuditMPS", "the BDC factorisation of a 3x3 block",
         [nan](const Settings& s) {
             QuditMPS mps(3, 3);
             mps.svd_rejection = s.rejection;
             mps.svd_accept_gram = s.accept_gram;
             mps.svd_report = s.report;
             const std::vector<Complex128> bad(81, Complex128(nan, 0.0));
             return run_quietly([&] { mps.apply_2qudit_adjacent(0, bad, {Validation::Ignore}); });
         }},
    };
}

const std::string kDeclined = " produced nothing (the kernel declined the block)";
const std::string kRetry = "; retrying with SVDMethod::Jacobi";
const std::string kGram = "; recomputing through the Gram route";

// What a split of NaN throws and reports under `s`: its message contains
// `ending`, and the channel carries `retries` Jacobi retries and `grams` Gram
// descents for that block.
void expect_split(const Surface& surface, const Settings& s, const std::string& ending,
                  std::size_t retries, std::size_t grams) {
    v11311::WarningCapture warnings;
    const auto e = surface.split(s);
    ASSERT_TRUE(e.has_value()) << "a block of NaN produced a tensor";
    v11311::expect_message(*e, {std::string(surface.block) + kDeclined, ending});
    EXPECT_EQ(warnings.count(kRetry), retries);
    EXPECT_EQ(warnings.count(kGram), grams);
}

const std::string kNoGram =
    "; the Jacobi rescue produced nothing (the kernel declined the block); the Gram route is the "
    "remaining rung and svd_accept_gram is off; refusing to continue with a corrupt tensor";
const std::string kEveryRung = "; every permitted rung failed, refusing to continue with a corrupt tensor";
const std::string kThrow = " and svd_rejection is SvdRejection::Throw; refusing to continue with it";

}  // namespace

// =============================================================================
// Fix, the default
// =============================================================================

TEST(V11311SvdSettings, FixWithoutTheGramRouteStopsWhereJacobiFails) {
    for (const Surface& surface : surfaces()) {
        SCOPED_TRACE(surface.name);
        expect_split(surface, Settings{}, kNoGram, 1, 0);
    }
}

TEST(V11311SvdSettings, FixWithTheGramRouteDescendsToItBeforeStopping) {
    for (const Surface& surface : surfaces()) {
        SCOPED_TRACE(surface.name);
        expect_split(surface, Settings{SvdRejection::Fix, true, SvdReport::Warn}, kEveryRung, 1, 1);
    }
}

// =============================================================================
// Throw
// =============================================================================

TEST(V11311SvdSettings, ThrowStopsAtTheFirstBlockTheKernelDeclines) {
    for (const Surface& surface : surfaces()) {
        SCOPED_TRACE(surface.name);
        for (const bool gram : {false, true}) {
            SCOPED_TRACE(gram ? "Gram route accepted" : "Gram route off");
            expect_split(surface, Settings{SvdRejection::Throw, gram, SvdReport::Warn}, kThrow, 0, 0);
        }
    }
}

// =============================================================================
// Ignore: a block that produced nothing still descends
// =============================================================================

TEST(V11311SvdSettings, IgnoreDescendsTheLadderForABlockThatProducedNothing) {
    for (const Surface& surface : surfaces()) {
        SCOPED_TRACE(surface.name);
        expect_split(surface, Settings{SvdRejection::Ignore, false, SvdReport::Warn}, kNoGram, 1, 0);
        expect_split(surface, Settings{SvdRejection::Ignore, true, SvdReport::Warn}, kEveryRung, 1, 1);
    }
}

// =============================================================================
// Silent
// =============================================================================

TEST(V11311SvdSettings, SilentReportsNoDescentAndStillStopsTheSameWay) {
    for (const Surface& surface : surfaces()) {
        SCOPED_TRACE(surface.name);
        expect_split(surface, Settings{SvdRejection::Fix, true, SvdReport::Silent}, kEveryRung, 0, 0);
        expect_split(surface, Settings{SvdRejection::Ignore, false, SvdReport::Silent}, kNoGram, 0, 0);
    }
}

namespace {

// Seven qutrits, 168 gates, every pair distance: the default kernel declines
// several of its blocks on current builds and Jacobi repairs each, so the
// rescue is counted without any seam.
std::size_t qutrit_chain_rescues(SvdReport report, std::size_t& retry_warnings) {
    constexpr int n = 7, d = 3;
    const auto F = qudit_gates::qft_matrix(d);
    const auto SUM = qudit_gates::cadd_matrix(d, 1);
    std::vector<Complex128> CP(81, Complex128(0.0, 0.0));
    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
            const std::size_t k = static_cast<std::size_t>(a + d * b);
            const double phi = TWO_PI * a * b / d;
            CP[k * 9 + k] = Complex128(std::cos(phi), std::sin(phi));
        }
    QuditMPS mps(n, d, 64);
    mps.svd_report = report;
    v11311::WarningCapture warnings;
    for (int step = 0; step < 56; ++step) {
        mps.apply_1qudit((3 * step + 1) % n, F);
        const int a = (5 * step) % n;
        int b = (2 * step + 3) % n;
        if (b == a) b = (b + 1) % n;
        mps.apply_2qudit(a, b, (step % 2 == 0) ? SUM : CP);
        const int c0 = (step * 4 + 2) % n;
        int c1 = (n - 1) - (step % n);
        if (c1 == c0) c1 = (c1 + 1) % n;
        mps.apply_2qudit(c0, c1, (step % 3 == 0) ? CP : SUM);
    }
    // The channel delivers a repeated message once and then a tally,
    // "<message> [repeated K more times]", so a tally stands for K more.
    retry_warnings = 0;
    for (const std::string& m : warnings.messages()) {
        if (m.find(kRetry) == std::string::npos) continue;
        const std::size_t at = m.find(" [repeated ");
        retry_warnings += at == std::string::npos
                              ? 1
                              : std::stoul(m.substr(at + std::strlen(" [repeated ")));
    }
    return mps.jacobi_rescue_count();
}

}  // namespace

TEST(V11311SvdSettings, SilentCountsEveryRescueItDoesNotReport) {
    std::size_t warned_retries = 0, silent_retries = 0;
    const std::size_t warned = qutrit_chain_rescues(SvdReport::Warn, warned_retries);
    const std::size_t silent = qutrit_chain_rescues(SvdReport::Silent, silent_retries);
    ::testing::Test::RecordProperty("qutrit_chain_jacobi_rescues", std::to_string(warned));
    EXPECT_EQ(warned_retries, warned) << "Warn reports each rescue once";
    EXPECT_EQ(silent, warned) << "Silent repairs the same blocks";
    EXPECT_EQ(silent_retries, 0u) << "Silent reported a rescue";
}

// =============================================================================
// The one-time notes, in a fresh process each
// =============================================================================
// The notes are latched once per process, so whichever test chose a setting
// first would consume them. Each check runs in a child process, which starts
// with every latch clear, and exits with 10 * (qubit notes) + (qudit notes).

namespace {

[[noreturn]] void exit_with_note_counts(const Settings& s, const std::string& setting) {
    std::size_t qubit = 0, qudit = 0;
    set_warning_handler([&](const std::string& m) {
        if (m.find("[repeated ") != std::string::npos) return;
        if (m.rfind("note: the qubit MPS has " + setting, 0) == 0) ++qubit;
        if (m.rfind("note: the qudit MPS has " + setting, 0) == 0) ++qudit;
        // A note for a setting this child did not choose is a failure too.
        if (m.rfind("note: the qubit MPS has ", 0) == 0 && m.find(setting) == std::string::npos) qubit += 100;
        if (m.rfind("note: the qudit MPS has ", 0) == 0 && m.find(setting) == std::string::npos) qudit += 100;
    });
    for (int rep = 0; rep < 2; ++rep) {
        QuantumCircuit qc(4);
        qc.h(0).cx(0, 1).cx(1, 2).cx(2, 3);
        MPSSimulator sim;
        sim.svd_rejection = s.rejection;
        sim.svd_accept_gram = s.accept_gram;
        sim.svd_report = s.report;
        (void)sim.run(qc, 8, 0, 1);

        QuditMPS mps(4, 3);
        mps.svd_rejection = s.rejection;
        mps.svd_accept_gram = s.accept_gram;
        mps.svd_report = s.report;
        mps.apply_1qudit(0, qudit_gates::qft_matrix(3));
        mps.apply_2qudit_adjacent(0, qudit_gates::cadd_matrix(3, 1));
        mps.apply_2qudit_adjacent(1, qudit_gates::cadd_matrix(3, 1));
    }
    flush_warnings();
    set_warning_handler(nullptr);
    std::exit(static_cast<int>(std::min<std::size_t>(10 * qubit + qudit, 255)));
}

}  // namespace

TEST(V11311SvdSettingsNotes, EachNonDefaultSettingIsNotedOncePerLayer) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(exit_with_note_counts({SvdRejection::Throw, false, SvdReport::Warn},
                                      "svd_rejection = SvdRejection::Throw, not the default (Fix)"),
                ::testing::ExitedWithCode(11), "");
    EXPECT_EXIT(exit_with_note_counts({SvdRejection::Ignore, false, SvdReport::Warn},
                                      "svd_rejection = SvdRejection::Ignore, not the default (Fix)"),
                ::testing::ExitedWithCode(11), "");
    EXPECT_EXIT(exit_with_note_counts({SvdRejection::Fix, true, SvdReport::Warn},
                                      "svd_accept_gram on, not the default"),
                ::testing::ExitedWithCode(11), "");
    EXPECT_EXIT(exit_with_note_counts({SvdRejection::Fix, false, SvdReport::Silent},
                                      "svd_report = SvdReport::Silent, not the default (Warn)"),
                ::testing::ExitedWithCode(11), "");
}

TEST(V11311SvdSettingsNotes, TheDefaultsAreNotNoted) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // Any note at all counts 100 in the child, so a clean exit means none.
    EXPECT_EXIT(exit_with_note_counts(Settings{}, "(no setting)"), ::testing::ExitedWithCode(0), "");
}
