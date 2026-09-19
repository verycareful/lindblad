// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/qudit/qudit_statevector.hpp"

#include <cstdint>
#include <vector>

namespace lindblad {

// =============================================================================
// QuditGateOp — a single gate application to be executed by QuditSimulator
// =============================================================================

struct QuditGateOp {
    enum class Type { SINGLE, TWO } type;
    int q0;          // qudit index (SINGLE) or control qudit (TWO)
    int q1 = -1;     // target qudit (TWO only; ignored for SINGLE)
    // d×d (SINGLE) or d²×d² (TWO), row-major. TWO matrices follow the
    // project LSB-first convention (docs/Architecture.md "Conventions"):
    // q0, the first operand, is the LEAST significant digit of the index
    // (row r = r_q1*d + r_q0), matching QuditStatevector::apply_2qudit.
    std::vector<Complex128> matrix;
};

// =============================================================================
// QuditSimulator — executes a sequence of QuditGateOps then measures
// =============================================================================

class QuditSimulator {
public:
    struct Result {
        std::vector<int> outcome;          // measured symbol per qudit, {0..d-1}
        double simulation_time_seconds = 0.0;
    };

    // Apply ops in order to sv, then measure once.
    // sv is modified in-place (gate applications are unitary; measure samples
    // from the final probability distribution without collapsing the state
    // here — collapse is unnecessary because we sample once and return).
    static Result run(QuditStatevector& sv,
                      const std::vector<QuditGateOp>& ops,
                      uint64_t seed = 0);
};

} // namespace lindblad
