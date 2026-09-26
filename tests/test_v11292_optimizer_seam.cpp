// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.2 - the edges of the two seam fixes that the 1.1.29.1 pins leave open.
//
// NLopt can refuse a request before it evaluates anything, and the seam used
// to report that as "the optimiser produced no finite objective value", which
// names the wrong cause. The 1.1.29.1 pins check that the refusal is now an
// invalid_argument naming the entry point and the method. What they do not
// check is that NLopt's own reason reaches the caller, which is the part that
// tells them what to change, nor that MA-QAOA, whose two paths share the same
// box as QAOA, surfaces it the same way.
//
// uniform_in moves a result equal to hi to the largest double below it. On a
// degenerate range (lo == hi) every draw takes that branch, and there is no
// double strictly inside the range, so the draw must still return the range's
// one value rather than stepping outside it.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/optimizer.hpp"
#include "lindblad/operators.hpp"

#include <bit>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;
using namespace lindblad::detail;

namespace {

constexpr const char* kWhere = "V11292OptimizerSeam";

// The text after this marker is NLopt's own reason, or the code's name when
// NLopt left none.
constexpr const char* kReasonMarker = "refused the request before any evaluation: ";

std::string reason_in(const std::string& msg) {
    const auto at = msg.find(kReasonMarker);
    if (at == std::string::npos) return {};
    return msg.substr(at + std::string(kReasonMarker).size());
}

// Three-qubit cost with distinct weights, so nothing about the problem is
// symmetric.
SparsePauliOp cost() {
    return SparsePauliOp::from_list({{"ZZI", Complex128(1.0, 0.0)},
                                     {"IZZ", Complex128(0.5, 0.0)},
                                     {"ZIZ", Complex128(0.25, 0.0)}});
}

}  // namespace

// =============================================================================
// NLopt refusals
// =============================================================================

TEST(V11292OptimizerSeam, TheRefusalCarriesNloptsOwnReason) {
    // BOBYQA needs each interval at least twice the step wide; [-1, 1] with a
    // step of 1.5 fails that before any evaluation.
    OptimizerSpec spec;
    spec.backend = OptimizerBackend::NloptBobyqa;
    spec.max_evaluations = 100;
    spec.xtol_rel = 1e-8;
    spec.initial_step = 1.5;
    spec.lower = {-1.0, -1.0, -1.0};
    spec.upper = {1.0, 1.0, 1.0};

    std::size_t calls = 0;
    const Objective counted = [&calls](std::span<const double>) {
        ++calls;
        return 0.0;
    };
    try {
        (void)minimize(spec, counted, std::vector<double>{0.0, 0.0, 0.0}, kWhere);
        FAIL() << "BOBYQA ran with a step wider than half its box";
    } catch (const std::invalid_argument& e) {
        const std::string reason = reason_in(e.what());
        EXPECT_FALSE(reason.empty()) << e.what();
        EXPECT_NE(reason, "INVALID_ARGS")
            << "only the code's name arrived, not NLopt's explanation: " << e.what();
    }
    EXPECT_EQ(calls, 0u);
}

TEST(V11292OptimizerSeam, MaqaoaBobyqaWithAStepTooWideForTheBoxIsAnInvalidArgument) {
    // MA-QAOA's box is 4pi wide on both paths, so a 7 rad step does not fit.
    for (bool layerwise : {false, true}) {
        SCOPED_TRACE(layerwise ? "layerwise" : "full");
        MAQAOA m;
        m.options.optimizer = "BOBYQA";
        m.options.layerwise = layerwise;
        m.options.p = 2;
        m.options.seed = 11292;
        m.options.max_iterations = 40;
        m.options.initial_step = 7.0;
        try {
            (void)m.optimize(cost());
            ADD_FAILURE() << "BOBYQA ran with a step wider than half its box";
        } catch (const std::invalid_argument& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("MAQAOA::optimize"), std::string::npos) << msg;
            EXPECT_NE(msg.find("BOBYQA"), std::string::npos) << msg;
        }
    }
}

// =============================================================================
// uniform_in
// =============================================================================

TEST(V11292UniformDraw, ADegenerateRangeReturnsItsOneValue) {
    std::mt19937_64 rng(11292);
    for (int i = 0; i < 4096; ++i) {
        const double got = uniform_in(rng, PI, PI);
        ASSERT_EQ(std::bit_cast<std::uint64_t>(got), std::bit_cast<std::uint64_t>(PI))
            << "draw " << i;
    }
}
