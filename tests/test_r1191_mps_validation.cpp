// R.1.19.1 test wave — MPS primitive fail-loud validation.
//
// R.1.19.0 replaced the Release-dead asserts in MPSState with real checks:
// apply_single_qubit_gate, apply_two_qubit_gate, and probabilities_single now
// throw std::out_of_range on an out-of-range qubit. apply_two_qubit_gate also
// gained distinctness enforcement: it used to `if (q1 == q2) return;` (a silent
// no-op), and now throws std::invalid_argument, so a caller can no longer feed
// an aliased pair and get a silently skipped gate.
//
// The MPS tensor network indexes `tensors[qubit]` directly, so an out-of-range
// qubit was an out-of-bounds vector access (UB) in the shipped Release build,
// where the guarding asserts were compiled out. These tests pin the loud
// behaviour for the below/at/above/far-above index taxonomy and the
// distinctness change, with positive controls proving valid calls still run.

#include <gtest/gtest.h>

#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/types.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

constexpr int kN = 4;   // 4-qubit MPS chain
const std::vector<int> kBadIndices = {-1, kN, kN + 1, 64};

std::array<Complex128, 4> id2() {
    return {Complex128(1.0, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
}

std::array<Complex128, 16> id4() {
    std::array<Complex128, 16> m;
    for (auto& e : m) e = Complex128(0.0, 0.0);
    m[0] = m[5] = m[10] = m[15] = Complex128(1.0, 0.0);
    return m;
}

} // namespace

// =============================================================================
// apply_single_qubit_gate
// =============================================================================

TEST(R1191MpsValidation, SingleQubitOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        MPSState mps(kN);
        EXPECT_THROW(mps.apply_single_qubit_gate(id2(), bad), std::out_of_range);
    }
}

TEST(R1191MpsValidation, SingleQubitValidDoesNotThrow) {
    MPSState mps(kN);
    EXPECT_NO_THROW(mps.apply_single_qubit_gate(id2(), 0));
    EXPECT_NO_THROW(mps.apply_single_qubit_gate(id2(), kN - 1));
}

// =============================================================================
// apply_two_qubit_gate
// =============================================================================

TEST(R1191MpsValidation, TwoQubitOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { MPSState mps(kN); EXPECT_THROW(mps.apply_two_qubit_gate(id4(), bad, 1), std::out_of_range); }
        { MPSState mps(kN); EXPECT_THROW(mps.apply_two_qubit_gate(id4(), 0, bad), std::out_of_range); }
    }
}

TEST(R1191MpsValidation, TwoQubitNonDistinctThrows) {
    // Regression pin for the behaviour change: apply_two_qubit_gate previously
    // returned silently when q1 == q2, so a bug that aliased the two operands
    // produced a skipped gate with no signal. It now throws.
    MPSState mps(kN);
    EXPECT_THROW(mps.apply_two_qubit_gate(id4(), 2, 2), std::invalid_argument);
}

TEST(R1191MpsValidation, TwoQubitValidDoesNotThrow) {
    MPSState mps(kN);
    EXPECT_NO_THROW(mps.apply_two_qubit_gate(id4(), 0, 1));  // adjacent
    EXPECT_NO_THROW(mps.apply_two_qubit_gate(id4(), 3, 0));  // reversed, non-adjacent
}

// =============================================================================
// probabilities_single
// =============================================================================

TEST(R1191MpsValidation, ProbabilitiesSingleOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        MPSState mps(kN);
        EXPECT_THROW(mps.probabilities_single(bad), std::out_of_range);
    }
}

TEST(R1191MpsValidation, ProbabilitiesSingleValidDoesNotThrow) {
    MPSState mps(kN);
    for (int q = 0; q < kN; ++q) {
        SCOPED_TRACE("q=" + std::to_string(q));
        EXPECT_NO_THROW(mps.probabilities_single(q));
    }
}

// =============================================================================
// Message format
// =============================================================================

TEST(R1191MpsValidation, MessageFormatMatchesCircuitLayer) {
    MPSState mps(kN);
    try {
        mps.apply_single_qubit_gate(id2(), 77);
        FAIL() << "expected std::out_of_range";
    } catch (const std::out_of_range& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("MPSState::apply_single_qubit_gate"), std::string::npos);
        EXPECT_NE(msg.find("77"), std::string::npos);
        EXPECT_NE(msg.find("out of range [0, 4)"), std::string::npos);
    }
}
