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
    std::vector<Complex128> matrix;  // d×d (SINGLE) or d²×d² (TWO), row-major
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
