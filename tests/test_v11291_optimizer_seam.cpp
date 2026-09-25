// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.1 test wave - the optimizer seam, held to the contract its header
// states.
//
// 1.1.29.0 moved VQE, QAOA and MA-QAOA onto one internal minimiser seam with
// FLOP's COBYLA as the default and NLopt's COBYLA, Nelder-Mead and BOBYQA
// selectable. The seam promises more than any one library does: a malformed
// request is refused before the objective runs, a non-finite objective value
// ends the run with the best finite point seen, an exception thrown by the
// objective reaches the caller unchanged even across NLopt's C frames, and
// every backend is handed the initial step explicitly. Each of those is a
// property a caller relies on without seeing the library underneath, so each
// is asserted here on all four backends.
//
// The objective in every test records the points it is called at. The seam's
// claims are about what the objective saw and what came back, and a recording
// objective observes both without reaching inside the seam.
//
// Starting points and steps in the initial-step tests are dyadic, so the first
// trial points are exact in binary and a backend that rescales by the step
// (NLopt does) lands on the same bits as one that adds it.
//
// ONE TEST SHIPS RED in this release:
//   V11291OptimizerSeam.BobyqaRefusingABoxTooNarrowForItsStepIsAnInvalidArgument
// NLopt's refusal currently reaches the caller as a runtime_error reporting an
// objective that produced no finite value. The seam change that turns it green
// belongs to the next patch release.

#include <gtest/gtest.h>

#include "lindblad/detail/optimizer.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using lindblad::detail::minimize;
using lindblad::detail::Objective;
using lindblad::detail::optimizer_name;
using lindblad::detail::optimizer_names;
using lindblad::detail::OptimizerBackend;
using lindblad::detail::OptimizerOutcome;
using lindblad::detail::OptimizerSpec;
using lindblad::detail::parse_optimizer_name;
using lindblad::detail::resolve_optimizer;

namespace {

constexpr const char* kWhere = "V11291OptimizerSeam";

constexpr OptimizerBackend kBackends[] = {
    OptimizerBackend::FlopCobyla,
    OptimizerBackend::NloptCobyla,
    OptimizerBackend::NloptNelderMead,
    OptimizerBackend::NloptBobyqa,
};

// Built from bit patterns so the project-wide -ffast-math cannot fold them.
double pos_inf() { return std::bit_cast<double>(0x7FF0000000000000ULL); }
double neg_inf() { return std::bit_cast<double>(0xFFF0000000000000ULL); }
double nan_bits() { return lindblad::quiet_nan_strict(); }

// Every call the seam makes, in order: the point and the value returned.
struct Trace {
    std::vector<std::vector<double>> points;
    std::vector<double> values;
    std::size_t calls() const { return points.size(); }
};

// Wraps f so every call is recorded before the value is returned.
Objective recording(Trace& trace, std::function<double(std::span<const double>)> f) {
    return [&trace, f = std::move(f)](std::span<const double> x) {
        trace.points.emplace_back(x.begin(), x.end());
        const double v = f(x);
        trace.values.push_back(v);
        return v;
    };
}

// A separable quadratic with its minimum at kCentre, so every backend has an
// unambiguous answer to converge to.
const std::vector<double> kCentre = {0.3, -0.7, 1.1};

double quadratic(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - kCentre[i];
        s += d * d;
    }
    return s;
}

OptimizerSpec spec_for(OptimizerBackend b, int cap = 400, double step = 0.3,
                       double xtol = 1e-8) {
    OptimizerSpec s;
    s.backend = b;
    s.max_evaluations = cap;
    s.xtol_rel = xtol;
    s.initial_step = step;
    return s;
}

std::string label(OptimizerBackend b) { return std::string(optimizer_name(b)); }

std::vector<std::string> capture_warnings(const std::function<void()>& fn) {
    lindblad::flush_warnings();
    std::vector<std::string> captured;
    lindblad::set_warning_handler(
        [&captured](const std::string& m) { captured.push_back(m); });
    fn();
    lindblad::set_warning_handler(nullptr);
    return captured;
}

// A malformed request must be refused before the objective runs even once.
// Returns the message so the caller can check what it names.
std::string refused(const OptimizerSpec& spec, std::vector<double> x0) {
    Trace trace;
    try {
        (void)minimize(spec, recording(trace, quadratic), x0, kWhere);
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(trace.calls(), 0u) << "the objective ran before the refusal";
        return e.what();
    }
    ADD_FAILURE() << "no std::invalid_argument for a malformed request";
    return {};
}

}  // namespace

// =============================================================================
// Names
// =============================================================================

TEST(V11291OptimizerSeam, TheNameTableIsTheDocumentedFour) {
    const auto names = optimizer_names();
    const std::vector<std::string_view> got(names.begin(), names.end());
    EXPECT_EQ(got, (std::vector<std::string_view>{"COBYLA", "NLOPT_COBYLA",
                                                  "NELDER_MEAD", "BOBYQA"}));
}

TEST(V11291OptimizerSeam, EveryNameParsesToTheBackendThatCarriesIt) {
    std::vector<OptimizerBackend> seen;
    for (std::string_view name : optimizer_names()) {
        SCOPED_TRACE(std::string(name));
        OptimizerBackend b = OptimizerBackend::FlopCobyla;
        ASSERT_TRUE(parse_optimizer_name(name, b));
        EXPECT_EQ(optimizer_name(b), name);
        seen.push_back(b);
    }
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(std::unique(seen.begin(), seen.end()), seen.end())
        << "two names resolve to one backend";
}

TEST(V11291OptimizerSeam, TheDefaultNameIsFlopCobyla) {
    OptimizerBackend b = OptimizerBackend::NloptBobyqa;
    ASSERT_TRUE(parse_optimizer_name("COBYLA", b));
    EXPECT_EQ(b, OptimizerBackend::FlopCobyla);
    ASSERT_TRUE(parse_optimizer_name("NLOPT_COBYLA", b));
    EXPECT_EQ(b, OptimizerBackend::NloptCobyla);
}

TEST(V11291OptimizerSeam, ParseRejectsNearMissesAndLeavesTheOutputAlone) {
    using namespace std::string_view_literals;
    // The last entry carries an embedded NUL, so a comparison that stops at
    // the first zero byte would read it as "COBYLA".
    const std::string_view near_misses[] = {
        ""sv,      "cobyla"sv,      "Cobyla"sv,       " COBYLA"sv, "COBYLA "sv,
        "POWELL"sv, "NELDER-MEAD"sv, "NLOPT_BOBYQA"sv, "COBYLA\0x"sv};
    for (std::string_view name : near_misses) {
        SCOPED_TRACE("'" + std::string(name) + "'");
        OptimizerBackend b = OptimizerBackend::NloptNelderMead;
        EXPECT_FALSE(parse_optimizer_name(name, b));
        EXPECT_EQ(b, OptimizerBackend::NloptNelderMead) << "output was written";
    }
}

TEST(V11291OptimizerSeam, AnUnknownNameWarnsOnceAndResolvesToTheDefault) {
    OptimizerBackend b = OptimizerBackend::NloptBobyqa;
    const auto msgs = capture_warnings(
        [&] { b = resolve_optimizer("POWELL", "SomeAlgorithm::entry"); });

    EXPECT_EQ(b, OptimizerBackend::FlopCobyla);
    ASSERT_EQ(msgs.size(), 1u);
    EXPECT_NE(msgs[0].find("SomeAlgorithm::entry"), std::string::npos) << msgs[0];
    EXPECT_NE(msgs[0].find("POWELL"), std::string::npos) << msgs[0];
    // The message lists what would have been accepted, so the caller can fix
    // the spelling without reading the source.
    for (std::string_view name : optimizer_names())
        EXPECT_NE(msgs[0].find(std::string(name)), std::string::npos) << msgs[0];
}

TEST(V11291OptimizerSeam, AKnownNameResolvesSilently) {
    for (std::string_view name : optimizer_names()) {
        SCOPED_TRACE(std::string(name));
        OptimizerBackend b = OptimizerBackend::FlopCobyla;
        const auto msgs =
            capture_warnings([&] { b = resolve_optimizer(std::string(name), kWhere); });
        EXPECT_TRUE(msgs.empty()) << msgs.front();
        EXPECT_EQ(optimizer_name(b), name);
    }
}

// =============================================================================
// A malformed request is refused before any evaluation
// =============================================================================

TEST(V11291OptimizerSeam, AnEmptyStartIsRefused) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        const std::string msg = refused(spec_for(b), {});
        EXPECT_NE(msg.find(kWhere), std::string::npos) << msg;
    }
}

TEST(V11291OptimizerSeam, ANonFiniteStartIsRefused) {
    for (OptimizerBackend b : kBackends) {
        for (double bad : {nan_bits(), pos_inf(), neg_inf()}) {
            SCOPED_TRACE(label(b));
            const std::string msg = refused(spec_for(b), {0.1, bad, 0.2});
            EXPECT_NE(msg.find(kWhere), std::string::npos) << msg;
        }
    }
}

TEST(V11291OptimizerSeam, ANonPositiveCapIsRefused) {
    for (OptimizerBackend b : kBackends) {
        for (int cap : {0, -1, -1000}) {
            SCOPED_TRACE(label(b) + " cap " + std::to_string(cap));
            const std::string msg = refused(spec_for(b, cap), {0.0, 0.0, 0.0});
            EXPECT_NE(msg.find(kWhere), std::string::npos) << msg;
        }
    }
}

TEST(V11291OptimizerSeam, ANonPositiveOrNonFiniteInitialStepIsRefused) {
    for (OptimizerBackend b : kBackends) {
        for (double step : {0.0, -0.0, -0.3, nan_bits(), pos_inf()}) {
            SCOPED_TRACE(label(b) + " step " + std::to_string(step));
            refused(spec_for(b, 100, step), {0.0, 0.0, 0.0});
        }
    }
}

TEST(V11291OptimizerSeam, ANegativeOrNonFiniteToleranceIsRefused) {
    for (OptimizerBackend b : kBackends) {
        for (double xtol : {-1e-6, nan_bits(), pos_inf()}) {
            SCOPED_TRACE(label(b) + " xtol " + std::to_string(xtol));
            refused(spec_for(b, 100, 0.3, xtol), {0.0, 0.0, 0.0});
        }
    }
}

TEST(V11291OptimizerSeam, AZeroToleranceIsAccepted) {
    // Zero asks the backend never to stop on x, which leaves the cap as the
    // only exit. That is a legitimate request, not a malformed one.
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        Trace trace;
        const OptimizerOutcome out =
            minimize(spec_for(b, 30, 0.3, 0.0), recording(trace, quadratic),
                     std::vector<double>{0.0, 0.0, 0.0}, kWhere);
        EXPECT_GT(trace.calls(), 0u);
        EXPECT_TRUE(lindblad::is_finite_strict(out.f));
    }
}

TEST(V11291OptimizerSeam, BoundsOfTheWrongSizeAreRefused) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        OptimizerSpec lopsided = spec_for(b);
        lopsided.lower = {-1.0, -1.0, -1.0};
        lopsided.upper = {1.0, 1.0};
        refused(lopsided, {0.0, 0.0, 0.0});

        OptimizerSpec short_box = spec_for(b);
        short_box.lower = {-1.0, -1.0};
        short_box.upper = {1.0, 1.0};
        refused(short_box, {0.0, 0.0, 0.0});
    }
}

TEST(V11291OptimizerSeam, AStartOutsideItsBoundsIsRefused) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        OptimizerSpec s = spec_for(b);
        s.lower = {-1.0, -1.0, -1.0};
        s.upper = {1.0, 1.0, 1.0};
        refused(s, {0.0, 1.5, 0.0});
        refused(s, {-1.0000001, 0.0, 0.0});
    }
}

TEST(V11291OptimizerSeam, AnInvertedOrNonFiniteBoundIsRefused) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        OptimizerSpec inverted = spec_for(b);
        inverted.lower = {-1.0, 1.0, -1.0};
        inverted.upper = {1.0, -1.0, 1.0};
        refused(inverted, {0.0, 0.0, 0.0});

        OptimizerSpec open = spec_for(b);
        open.lower = {-1.0, neg_inf(), -1.0};
        open.upper = {1.0, 1.0, 1.0};
        refused(open, {0.0, 0.0, 0.0});
    }
}

TEST(V11291OptimizerSeam, AStartOnItsBoundIsAccepted) {
    // A bound the start sits on is not one it violates.
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        OptimizerSpec s = spec_for(b, 60);
        s.lower = {-2.0, -2.0, -2.0};
        s.upper = {2.0, 2.0, 2.0};
        Trace trace;
        EXPECT_NO_THROW((void)minimize(s, recording(trace, quadratic),
                                       std::vector<double>{-2.0, 0.0, 2.0}, kWhere));
        EXPECT_GT(trace.calls(), 0u);
    }
}

// =============================================================================
// What comes back
// =============================================================================

TEST(V11291OptimizerSeam, TheReturnedValueIsTheObjectiveAtTheReturnedPoint) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        Trace trace;
        const OptimizerOutcome out =
            minimize(spec_for(b), recording(trace, quadratic),
                     std::vector<double>{0.0, 0.0, 0.0}, kWhere);

        ASSERT_EQ(out.x.size(), 3u);
        // Bit for bit: a returned pair the objective never produced would be a
        // number nobody computed.
        bool found = false;
        for (std::size_t i = 0; i < trace.calls(); ++i)
            if (trace.points[i] == out.x && trace.values[i] == out.f) found = true;
        EXPECT_TRUE(found) << "the returned (x, f) was never evaluated";
    }
}

TEST(V11291OptimizerSeam, EvaluationsCountsEveryObjectiveCall) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        Trace trace;
        const OptimizerOutcome out =
            minimize(spec_for(b), recording(trace, quadratic),
                     std::vector<double>{0.0, 0.0, 0.0}, kWhere);
        EXPECT_EQ(static_cast<std::size_t>(out.evaluations), trace.calls());
    }
}

TEST(V11291OptimizerSeam, ASolvableProblemConvergesOnItsTolerance) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        Trace trace;
        const OptimizerOutcome out =
            minimize(spec_for(b, 4000), recording(trace, quadratic),
                     std::vector<double>{0.0, 0.0, 0.0}, kWhere);

        EXPECT_TRUE(out.converged) << out.status;
        EXPECT_FALSE(out.hit_evaluation_cap) << out.status;
        EXPECT_FALSE(out.saw_non_finite);
        EXPECT_FALSE(out.status.empty());
        for (std::size_t i = 0; i < kCentre.size(); ++i)
            EXPECT_NEAR(out.x[i], kCentre[i], 1e-4) << "coordinate " << i;
    }
}

TEST(V11291OptimizerSeam, TheCapIsNeverExceededAndReachingItIsNotConvergence) {
    // Three parameters: caps below, at and above the n + 1 simplex and the
    // 2n + 1 design BOBYQA builds before its first model step.
    for (OptimizerBackend b : kBackends) {
        for (int cap : {1, 2, 3, 4, 5, 7, 8, 12}) {
            SCOPED_TRACE(label(b) + " cap " + std::to_string(cap));
            Trace trace;
            const OptimizerOutcome out =
                minimize(spec_for(b, cap, 0.3, 1e-12), recording(trace, quadratic),
                         std::vector<double>{0.0, 0.0, 0.0}, kWhere);

            EXPECT_LE(trace.calls(), static_cast<std::size_t>(cap));
            EXPECT_GE(trace.calls(), 1u);
            EXPECT_TRUE(lindblad::is_finite_strict(out.f));
            if (out.hit_evaluation_cap) {
                EXPECT_EQ(trace.calls(), static_cast<std::size_t>(cap));
                EXPECT_FALSE(out.converged) << out.status;
            }
        }
    }
}

TEST(V11291OptimizerSeam, AnUnreachableToleranceStopsOnTheCap) {
    // xtol 1e-15 cannot be met in 25 evaluations of a three-parameter problem,
    // so the cap is the reason the run ended and it must say so.
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        Trace trace;
        const OptimizerOutcome out =
            minimize(spec_for(b, 25, 0.3, 1e-15), recording(trace, quadratic),
                     std::vector<double>{0.0, 0.0, 0.0}, kWhere);
        EXPECT_TRUE(out.hit_evaluation_cap) << out.status;
        EXPECT_FALSE(out.converged) << out.status;
        EXPECT_EQ(trace.calls(), 25u);
    }
}

// =============================================================================
// A non-finite objective ends the run
// =============================================================================

TEST(V11291OptimizerSeam, ANonFiniteValueEndsTheRunWithTheBestFinitePoint) {
    constexpr std::size_t kFinite = 5;  // good evaluations before the bad one
    for (OptimizerBackend b : kBackends) {
        for (double bad : {nan_bits(), pos_inf(), neg_inf()}) {
            SCOPED_TRACE(label(b) + " bad " + std::to_string(bad));
            Trace trace;
            const auto f = [&trace, bad](std::span<const double> x) {
                return trace.calls() > kFinite ? bad : quadratic(x);
            };
            const OptimizerOutcome out =
                minimize(spec_for(b, 200), recording(trace, f),
                         std::vector<double>{0.0, 0.0, 0.0}, kWhere);

            EXPECT_EQ(trace.calls(), kFinite + 1) << "the run went on past the bad value";
            EXPECT_TRUE(out.saw_non_finite);
            EXPECT_FALSE(out.converged);

            const auto best = std::min_element(trace.values.begin(),
                                               trace.values.begin() + kFinite);
            const std::size_t at = static_cast<std::size_t>(best - trace.values.begin());
            // -inf in particular must not come back as the minimum it compares as.
            EXPECT_TRUE(lindblad::is_finite_strict(out.f));
            EXPECT_EQ(out.f, *best);
            EXPECT_EQ(out.x, trace.points[at]);
        }
    }
}

TEST(V11291OptimizerSeam, NoFiniteValueAtAllThrowsRuntimeError) {
    for (OptimizerBackend b : kBackends) {
        for (double bad : {nan_bits(), pos_inf(), neg_inf()}) {
            SCOPED_TRACE(label(b) + " bad " + std::to_string(bad));
            Trace trace;
            try {
                (void)minimize(spec_for(b), recording(trace, [bad](auto) { return bad; }),
                               std::vector<double>{0.0, 0.0, 0.0}, kWhere);
                ADD_FAILURE() << "a run with no finite value returned a result";
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find(kWhere), std::string::npos) << e.what();
            }
            EXPECT_EQ(trace.calls(), 1u);
        }
    }
}

// =============================================================================
// An exception thrown by the objective reaches the caller unchanged
// =============================================================================

namespace {
struct ObjectiveFailure final : std::runtime_error {
    using std::runtime_error::runtime_error;
};
}  // namespace

TEST(V11291OptimizerSeam, AThrownExceptionKeepsItsTypeAndMessage) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        Trace trace;
        const auto f = [&trace](std::span<const double> x) -> double {
            if (trace.calls() == 3) throw ObjectiveFailure("estimator gave up at call 3");
            return quadratic(x);
        };
        try {
            (void)minimize(spec_for(b), recording(trace, f),
                           std::vector<double>{0.0, 0.0, 0.0}, kWhere);
            ADD_FAILURE() << "the exception was swallowed";
        } catch (const ObjectiveFailure& e) {
            EXPECT_STREQ(e.what(), "estimator gave up at call 3");
        } catch (const std::exception& e) {
            ADD_FAILURE() << "the exception changed type: " << e.what();
        }
    }
}

TEST(V11291OptimizerSeam, ANonStandardExceptionAlsoArrivesUnchanged) {
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        Trace trace;
        const auto f = [&trace](std::span<const double> x) -> double {
            if (trace.calls() == 2) throw 1129;
            return quadratic(x);
        };
        try {
            (void)minimize(spec_for(b), recording(trace, f),
                           std::vector<double>{0.0, 0.0, 0.0}, kWhere);
            ADD_FAILURE() << "the exception was swallowed";
        } catch (int code) {
            EXPECT_EQ(code, 1129);
        } catch (...) {
            ADD_FAILURE() << "the exception changed type";
        }
    }
}

TEST(V11291OptimizerSeam, AnExceptionOnTheFirstCallIsNotAnEmptyResult) {
    // With nothing evaluated there is no best point, so the only honest
    // outcome is the objective's own exception, not the seam's runtime_error
    // for a run that produced no finite value.
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        Trace trace;
        const auto f = [](std::span<const double>) -> double {
            throw ObjectiveFailure("first call");
        };
        EXPECT_THROW((void)minimize(spec_for(b), recording(trace, f),
                                    std::vector<double>{0.0, 0.0, 0.0}, kWhere),
                     ObjectiveFailure);
    }
}

// =============================================================================
// Boxes
// =============================================================================

TEST(V11291OptimizerSeam, EveryPointEvaluatedAndReturnedLiesInsideTheBox) {
    // The quadratic's minimum sits outside the box on two axes, so every
    // backend is pulled against a wall for the whole run.
    for (OptimizerBackend b : kBackends) {
        SCOPED_TRACE(label(b));
        OptimizerSpec s = spec_for(b, 300, 0.3);
        s.lower = {-0.5, -0.5, -0.5};
        s.upper = {0.5, 0.5, 0.5};
        Trace trace;
        const OptimizerOutcome out =
            minimize(s, recording(trace, quadratic), std::vector<double>{0.0, 0.0, 0.0}, kWhere);

        auto inside = [&s](const std::vector<double>& p) {
            for (std::size_t i = 0; i < p.size(); ++i)
                if (!(p[i] >= s.lower[i] && p[i] <= s.upper[i])) return false;
            return true;
        };
        for (std::size_t k = 0; k < trace.calls(); ++k)
            EXPECT_TRUE(inside(trace.points[k])) << "evaluation " << k << " left the box";
        EXPECT_TRUE(inside(out.x)) << "the returned point is outside the box";
    }
}

TEST(V11291OptimizerSeam, FlopFitsItsOpeningStepsIntoABoxNarrowerThanTheStep) {
    OptimizerSpec s = spec_for(OptimizerBackend::FlopCobyla, 100, 1.5);
    s.lower = {-1.0, -1.0, -1.0};
    s.upper = {1.0, 1.0, 1.0};
    Trace trace;
    const OptimizerOutcome out =
        minimize(s, recording(trace, quadratic), std::vector<double>{0.0, 0.0, 0.0}, kWhere);
    EXPECT_GT(trace.calls(), 1u);
    EXPECT_TRUE(lindblad::is_finite_strict(out.f));
}

TEST(V11291OptimizerSeam, BobyqaRefusingABoxTooNarrowForItsStepIsAnInvalidArgument) {
    // Red in this release. BOBYQA needs every interval at least twice the step
    // wide and refuses before evaluating anything. The caller's arguments are
    // the cause, so the refusal is an invalid_argument naming the entry point
    // and the method, raised with the objective never called.
    OptimizerSpec s = spec_for(OptimizerBackend::NloptBobyqa, 100, 1.5);
    s.lower = {-1.0, -1.0, -1.0};
    s.upper = {1.0, 1.0, 1.0};
    const std::string msg = refused(s, {0.0, 0.0, 0.0});
    EXPECT_NE(msg.find(kWhere), std::string::npos) << msg;
    EXPECT_NE(msg.find("BOBYQA"), std::string::npos) << msg;
}

// =============================================================================
// The initial step reaches every backend
// =============================================================================

namespace {

// True when p equals base everywhere except coordinate axis, where it differs
// by exactly +step or -step.
bool is_axis_step(const std::vector<double>& p, const std::vector<double>& base,
                  std::size_t axis, double step) {
    for (std::size_t i = 0; i < base.size(); ++i) {
        if (i == axis) {
            if (p[i] != base[i] + step && p[i] != base[i] - step) return false;
        } else if (p[i] != base[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

TEST(V11291OptimizerSeam, TheFirstTrialPointsAreOneStepAlongEachAxis) {
    // Each opening trial point is one step along one axis from a point already
    // evaluated. Which point that is depends on the backend: a method may move
    // its base to any point that ties or improves on it (NLopt's Nelder-Mead
    // moves on a tie, so on a flat objective its opening vertices form a
    // staircase from the start rather than a star around it). Every opening
    // point is therefore compared with every point evaluated before it. Two
    // step sizes, so a backend using its own default for one cannot pass both.
    const std::vector<double> x0 = {0.5, -0.25, 0.75};
    const std::size_t n = x0.size();
    for (OptimizerBackend b : kBackends) {
        for (double step : {0.25, 0.125}) {
            SCOPED_TRACE(label(b) + " step " + std::to_string(step));
            Trace trace;
            (void)minimize(spec_for(b, 60, step), recording(trace, [](auto) { return 1.0; }),
                           x0, kWhere);

            const std::size_t window = std::min(trace.calls(), 2 * n + 1);
            ASSERT_GE(window, n + 1);
            EXPECT_EQ(trace.points[0], x0) << "the start was not evaluated first";
            for (std::size_t axis = 0; axis < n; ++axis) {
                bool found = false;
                for (std::size_t k = 1; k < window; ++k)
                    for (std::size_t m = 0; m < k; ++m)
                        if (is_axis_step(trace.points[k], trace.points[m], axis, step))
                            found = true;
                EXPECT_TRUE(found) << "no opening point one step along axis " << axis;
            }
        }
    }
}
