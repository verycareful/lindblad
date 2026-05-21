// =============================================================================
// tests/test_visualiser_folding.cpp : R.1.10.1 FoldingTest
// =============================================================================
// DrawOptions::fold_width is wired into the public API but the ASCII
// renderer currently emits unfolded output (folding is a documented future
// refinement in docs/api/visualisation.md). These tests pin the current
// behaviour: setting fold_width does not crash, does not regress non-fold
// output, and an explicit fold_width = 0 disables folding (the documented
// sentinel). When folding is implemented in a future patch, these tests
// migrate to the new behaviour and the placeholder assertions below are
// strengthened.

#include "lindblad/circuit.hpp"
#include "lindblad/visualisation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <string>

using namespace lindblad;

namespace {

// Build a wide circuit: N single-qubit gates on q0 chained to widen the
// rendered diagram. With N >= ~30, the diagram is wider than 120 chars and
// would fold under a future fold_width = 120 setting.
QuantumCircuit wide_circuit(int n_gates) {
    QuantumCircuit qc(2);
    for (int i = 0; i < n_gates; ++i) {
        if (i % 2 == 0) { qc.h(0); }
        else            { qc.x(0); }
    }
    return qc;
}

// Length of the longest line (by byte count, not codepoint) in `s`.
size_t longest_line(const std::string& s) {
    size_t max_len = 0;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == '\n') {
            max_len = std::max(max_len, i - start);
            start = i + 1;
        }
    }
    return max_len;
}

} // namespace

// =============================================================================
// fold_width = 0 disables folding (sentinel)
// =============================================================================

TEST(FoldingTest, FoldWidthZeroProducesNonEmptyOutput) {
    QuantumCircuit qc = wide_circuit(40);
    DrawOptions opts;
    opts.fold_width = 0;
    const std::string out = qc.draw(DrawMode::ASCII, opts);
    EXPECT_FALSE(out.empty());
}

TEST(FoldingTest, FoldWidthZeroMatchesDefaultFor40Gate) {
    // Pin: fold_width = 0 yields the same output as the default opts for
    // any circuit the current renderer handles without folding.
    QuantumCircuit qc = wide_circuit(40);
    DrawOptions zero;
    zero.fold_width = 0;
    DrawOptions def;
    EXPECT_EQ(qc.draw(DrawMode::ASCII, zero),
              qc.draw(DrawMode::ASCII, def));
}

// =============================================================================
// fold_width set to a small value: current behaviour is no-op
// =============================================================================

TEST(FoldingTest, SmallFoldWidthDoesNotCrash) {
    // Pin: the renderer must produce output without throwing or aborting
    // when fold_width is set to a small value (current implementation just
    // emits unfolded output; future implementations will respect it).
    QuantumCircuit qc = wide_circuit(40);
    DrawOptions opts;
    opts.fold_width = 20;
    std::string out;
    ASSERT_NO_THROW(out = qc.draw(DrawMode::ASCII, opts));
    EXPECT_FALSE(out.empty());
}

TEST(FoldingTest, NoFoldMarkerInCurrentOutput) {
    // Pin the current pre-folding behaviour: there is no "... fold ..."
    // separator anywhere in the output. When folding is added in a future
    // patch, this test must be inverted to expect the marker for wide
    // circuits with a small fold_width.
    QuantumCircuit qc = wide_circuit(40);
    DrawOptions opts;
    opts.fold_width = 20;
    const std::string out = qc.draw(DrawMode::ASCII, opts);
    EXPECT_EQ(out.find("... fold ..."), std::string::npos);
}

// =============================================================================
// Long lines are tolerated in the current implementation
// =============================================================================

TEST(FoldingTest, WideCircuitProducesLongLinesUnderCurrentRenderer) {
    // Pin: wide circuit + small fold_width currently produces lines longer
    // than fold_width. Documents the gap between the option's intent and
    // the implementation, so a future folding implementation has a clear
    // test target to flip.
    QuantumCircuit qc = wide_circuit(40);
    DrawOptions opts;
    opts.fold_width = 20;
    const std::string out = qc.draw(DrawMode::ASCII, opts);
    EXPECT_GT(longest_line(out), static_cast<size_t>(opts.fold_width));
}
