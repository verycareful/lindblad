// 1.1.25.1 test wave - fail-loud surface of the reworked Clifford backend.
//
// The 1.1.25.0 rework added five gates to StabilizerState and a whole second
// tableau layout, ColumnTableau, whose every gate carries its own bounds and
// distinctness checks. R1191CliffordValidation pins that surface for the gates
// that existed before it; this file holds every entry point the rework
// introduced to exactly the same contract, so a new method cannot index a
// column directly the way the pre-R.1.19.0 gates could.
//
// Two properties are pinned per method: the throw itself, and the message,
// which must name the method, the offending index and the valid range. A bare
// EXPECT_THROW would pass on a check that fires for the wrong reason.

#include <gtest/gtest.h>

#include "lindblad/simulators/clifford_sim.hpp"

#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

constexpr int kN = 3;

// -1 and kN bracket the range; kN + 1 is past it; 64 crosses the word boundary
// that the packed layout makes interesting.
const std::vector<int> kBadIndices = {-1, kN, kN + 1, 64};

using SqRow = std::pair<const char*, std::function<void(StabilizerState&, int)>>;
using TqRow = std::pair<const char*, std::function<void(StabilizerState&, int, int)>>;
using ColSqRow =
    std::pair<const char*, std::function<void(StabilizerState::ColumnTableau&, int)>>;
using ColTqRow =
    std::pair<const char*,
              std::function<void(StabilizerState::ColumnTableau&, int, int)>>;

std::vector<SqRow> state_single_qubit() {
    return {
        {"apply_sx",   [](StabilizerState& s, int q) { s.apply_sx(q); }},
        {"apply_sxdg", [](StabilizerState& s, int q) { s.apply_sxdg(q); }},
    };
}

std::vector<TqRow> state_two_qubit() {
    return {
        {"apply_cz",    [](StabilizerState& s, int a, int b) { s.apply_cz(a, b); }},
        {"apply_swap",  [](StabilizerState& s, int a, int b) { s.apply_swap(a, b); }},
        {"apply_cy",    [](StabilizerState& s, int a, int b) { s.apply_cy(a, b); }},
        {"apply_iswap", [](StabilizerState& s, int a, int b) { s.apply_iswap(a, b); }},
        {"apply_ecr",   [](StabilizerState& s, int a, int b) { s.apply_ecr(a, b); }},
    };
}

std::vector<ColSqRow> column_single_qubit() {
    using C = StabilizerState::ColumnTableau;
    return {
        {"apply_h",    [](C& c, int q) { c.apply_h(q); }},
        {"apply_s",    [](C& c, int q) { c.apply_s(q); }},
        {"apply_sdg",  [](C& c, int q) { c.apply_sdg(q); }},
        {"apply_x",    [](C& c, int q) { c.apply_x(q); }},
        {"apply_y",    [](C& c, int q) { c.apply_y(q); }},
        {"apply_z",    [](C& c, int q) { c.apply_z(q); }},
        {"apply_sx",   [](C& c, int q) { c.apply_sx(q); }},
        {"apply_sxdg", [](C& c, int q) { c.apply_sxdg(q); }},
    };
}

std::vector<ColTqRow> column_two_qubit() {
    using C = StabilizerState::ColumnTableau;
    return {
        {"apply_cx",    [](C& c, int a, int b) { c.apply_cx(a, b); }},
        {"apply_cy",    [](C& c, int a, int b) { c.apply_cy(a, b); }},
        {"apply_cz",    [](C& c, int a, int b) { c.apply_cz(a, b); }},
        {"apply_swap",  [](C& c, int a, int b) { c.apply_swap(a, b); }},
        {"apply_iswap", [](C& c, int a, int b) { c.apply_iswap(a, b); }},
        {"apply_ecr",   [](C& c, int a, int b) { c.apply_ecr(a, b); }},
    };
}

// The message a bounds check must produce, per detail::throw_index_oor.
void expect_oor_message(const std::string& msg, const std::string& scope,
                        const std::string& method, int bad, int n) {
    EXPECT_NE(msg.find(scope + "::" + method), std::string::npos)
        << "message does not name the method: " << msg;
    EXPECT_NE(msg.find("qubit index " + std::to_string(bad)), std::string::npos)
        << "message does not name the bad index: " << msg;
    EXPECT_NE(msg.find("out of range [0, " + std::to_string(n) + ")"),
              std::string::npos)
        << "message does not name the valid range: " << msg;
}

// The message a distinctness check must produce, per detail::throw_not_distinct.
void expect_distinct_message(const std::string& msg, const std::string& scope,
                             const std::string& method) {
    EXPECT_NE(msg.find(scope + "::" + method), std::string::npos)
        << "message does not name the method: " << msg;
    EXPECT_NE(msg.find("qubits must be distinct"), std::string::npos)
        << "message does not state the rule: " << msg;
}

}  // namespace

// =============================================================================
// StabilizerState - the gates the rework added
// =============================================================================

TEST(V11251CliffordFailLoud, StateSingleQubitOutOfRangeThrows) {
    for (const auto& [name, fn] : state_single_qubit()) {
        for (int bad : kBadIndices) {
            SCOPED_TRACE(std::string(name) + " q=" + std::to_string(bad));
            StabilizerState st(kN);
            EXPECT_THROW(fn(st, bad), std::out_of_range);
        }
    }
}

TEST(V11251CliffordFailLoud, StateSingleQubitMessageNamesMethodIndexAndRange) {
    for (const auto& [name, fn] : state_single_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(kN);
        try {
            fn(st, 88);
            ADD_FAILURE() << name << " accepted qubit 88 on a " << kN << "-qubit state";
        } catch (const std::out_of_range& e) {
            expect_oor_message(e.what(), "StabilizerState", name, 88, kN);
        }
    }
}

TEST(V11251CliffordFailLoud, StateSingleQubitValidDoesNotThrow) {
    for (const auto& [name, fn] : state_single_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(kN);
        for (int q = 0; q < kN; ++q) EXPECT_NO_THROW(fn(st, q));
    }
}

TEST(V11251CliffordFailLoud, StateTwoQubitOutOfRangeThrowsOnEitherOperand) {
    for (const auto& [name, fn] : state_two_qubit()) {
        for (int bad : kBadIndices) {
            SCOPED_TRACE(std::string(name) + " bad=" + std::to_string(bad));
            { StabilizerState st(kN); EXPECT_THROW(fn(st, bad, 1), std::out_of_range); }
            { StabilizerState st(kN); EXPECT_THROW(fn(st, 0, bad), std::out_of_range); }
        }
    }
}

TEST(V11251CliffordFailLoud, StateTwoQubitMessageNamesMethodIndexAndRange) {
    for (const auto& [name, fn] : state_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(kN);
        try {
            fn(st, 0, 88);
            ADD_FAILURE() << name << " accepted qubit 88 on a " << kN << "-qubit state";
        } catch (const std::out_of_range& e) {
            expect_oor_message(e.what(), "StabilizerState", name, 88, kN);
        }
    }
}

TEST(V11251CliffordFailLoud, StateTwoQubitNonDistinctThrows) {
    for (const auto& [name, fn] : state_two_qubit()) {
        for (int q = 0; q < kN; ++q) {
            SCOPED_TRACE(std::string(name) + " q=" + std::to_string(q));
            StabilizerState st(kN);
            EXPECT_THROW(fn(st, q, q), std::invalid_argument);
        }
    }
}

TEST(V11251CliffordFailLoud, StateTwoQubitNonDistinctMessageStatesTheRule) {
    for (const auto& [name, fn] : state_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(kN);
        try {
            fn(st, 1, 1);
            ADD_FAILURE() << name << " accepted a repeated operand";
        } catch (const std::invalid_argument& e) {
            expect_distinct_message(e.what(), "StabilizerState", name);
        }
    }
}

// A bad index must be rejected even when the OTHER operand is also bad, and the
// range check must run before the distinctness check: (88, 88) is both, and the
// out_of_range is the one a caller can act on.
TEST(V11251CliffordFailLoud, StateTwoQubitRangeCheckPrecedesDistinctness) {
    for (const auto& [name, fn] : state_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(kN);
        EXPECT_THROW(fn(st, 88, 88), std::out_of_range);
    }
}

TEST(V11251CliffordFailLoud, StateTwoQubitValidDoesNotThrow) {
    for (const auto& [name, fn] : state_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(kN);
        EXPECT_NO_THROW(fn(st, 0, 1));
        EXPECT_NO_THROW(fn(st, 2, 0));
        EXPECT_NO_THROW(fn(st, 1, 2));
    }
}

// =============================================================================
// StabilizerState::ColumnTableau - the layout the rework introduced
// =============================================================================

TEST(V11251CliffordFailLoud, ColumnSingleQubitOutOfRangeThrows) {
    for (const auto& [name, fn] : column_single_qubit()) {
        for (int bad : kBadIndices) {
            SCOPED_TRACE(std::string(name) + " q=" + std::to_string(bad));
            StabilizerState::ColumnTableau c(kN);
            EXPECT_THROW(fn(c, bad), std::out_of_range);
        }
    }
}

TEST(V11251CliffordFailLoud, ColumnSingleQubitMessageNamesTheNestedScope) {
    for (const auto& [name, fn] : column_single_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState::ColumnTableau c(kN);
        try {
            fn(c, 88);
            ADD_FAILURE() << name << " accepted qubit 88 on a " << kN << "-qubit tableau";
        } catch (const std::out_of_range& e) {
            expect_oor_message(e.what(), "StabilizerState::ColumnTableau", name, 88, kN);
        }
    }
}

TEST(V11251CliffordFailLoud, ColumnSingleQubitValidDoesNotThrow) {
    for (const auto& [name, fn] : column_single_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState::ColumnTableau c(kN);
        for (int q = 0; q < kN; ++q) EXPECT_NO_THROW(fn(c, q));
    }
}

TEST(V11251CliffordFailLoud, ColumnTwoQubitOutOfRangeThrowsOnEitherOperand) {
    for (const auto& [name, fn] : column_two_qubit()) {
        for (int bad : kBadIndices) {
            SCOPED_TRACE(std::string(name) + " bad=" + std::to_string(bad));
            { StabilizerState::ColumnTableau c(kN); EXPECT_THROW(fn(c, bad, 1), std::out_of_range); }
            { StabilizerState::ColumnTableau c(kN); EXPECT_THROW(fn(c, 0, bad), std::out_of_range); }
        }
    }
}

TEST(V11251CliffordFailLoud, ColumnTwoQubitMessageNamesTheNestedScope) {
    for (const auto& [name, fn] : column_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState::ColumnTableau c(kN);
        try {
            fn(c, 0, 88);
            ADD_FAILURE() << name << " accepted qubit 88 on a " << kN << "-qubit tableau";
        } catch (const std::out_of_range& e) {
            expect_oor_message(e.what(), "StabilizerState::ColumnTableau", name, 88, kN);
        }
    }
}

TEST(V11251CliffordFailLoud, ColumnTwoQubitNonDistinctThrows) {
    for (const auto& [name, fn] : column_two_qubit()) {
        for (int q = 0; q < kN; ++q) {
            SCOPED_TRACE(std::string(name) + " q=" + std::to_string(q));
            StabilizerState::ColumnTableau c(kN);
            EXPECT_THROW(fn(c, q, q), std::invalid_argument);
        }
    }
}

// The distinctness guard is load-bearing here rather than defensive: the
// two-operand column sweep takes __restrict__ pointers to four columns, so a
// repeated operand would alias two of them.
TEST(V11251CliffordFailLoud, ColumnTwoQubitNonDistinctMessageStatesTheRule) {
    for (const auto& [name, fn] : column_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState::ColumnTableau c(kN);
        try {
            fn(c, 1, 1);
            ADD_FAILURE() << name << " accepted a repeated operand";
        } catch (const std::invalid_argument& e) {
            expect_distinct_message(e.what(), "StabilizerState::ColumnTableau", name);
        }
    }
}

TEST(V11251CliffordFailLoud, ColumnTwoQubitRangeCheckPrecedesDistinctness) {
    for (const auto& [name, fn] : column_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState::ColumnTableau c(kN);
        EXPECT_THROW(fn(c, 88, 88), std::out_of_range);
    }
}

TEST(V11251CliffordFailLoud, ColumnTwoQubitValidDoesNotThrow) {
    for (const auto& [name, fn] : column_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState::ColumnTableau c(kN);
        EXPECT_NO_THROW(fn(c, 0, 1));
        EXPECT_NO_THROW(fn(c, 2, 0));
        EXPECT_NO_THROW(fn(c, 1, 2));
    }
}

// A rejected call must leave the tableau untouched, so a caller that catches
// and continues is not working from a half-applied gate.
TEST(V11251CliffordFailLoud, RejectedCallLeavesTheStateUnchanged) {
    for (const auto& [name, fn] : state_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(kN);
        st.apply_h(0);
        st.apply_cx(0, 1);
        const int before_zz = st.expectation_pauli("ZZI");
        const int before_xx = st.expectation_pauli("XXI");

        EXPECT_THROW(fn(st, 0, 88), std::out_of_range);
        EXPECT_THROW(fn(st, 2, 2), std::invalid_argument);

        EXPECT_EQ(st.expectation_pauli("ZZI"), before_zz);
        EXPECT_EQ(st.expectation_pauli("XXI"), before_xx);
    }
}

// =============================================================================
// A single-qubit register is a legal register, not an edge case to reject
// =============================================================================

TEST(V11251CliffordFailLoud, SingleQubitRegisterAcceptsQubitZeroAndRejectsOne) {
    for (const auto& [name, fn] : state_single_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(1);
        EXPECT_NO_THROW(fn(st, 0));
        EXPECT_THROW(fn(st, 1), std::out_of_range);
    }
    for (const auto& [name, fn] : column_single_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState::ColumnTableau c(1);
        EXPECT_NO_THROW(fn(c, 0));
        EXPECT_THROW(fn(c, 1), std::out_of_range);
    }
    // Every two-qubit form needs two distinct operands, and a one-qubit
    // register cannot supply them.
    for (const auto& [name, fn] : state_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState st(1);
        EXPECT_THROW(fn(st, 0, 0), std::invalid_argument);
        EXPECT_THROW(fn(st, 0, 1), std::out_of_range);
    }
    for (const auto& [name, fn] : column_two_qubit()) {
        SCOPED_TRACE(name);
        StabilizerState::ColumnTableau c(1);
        EXPECT_THROW(fn(c, 0, 0), std::invalid_argument);
        EXPECT_THROW(fn(c, 0, 1), std::out_of_range);
    }
}

// =============================================================================
// Register width
// =============================================================================

// A negative width reaches every buffer size as an enormous unsigned length, so
// it is rejected before any of them is computed rather than left to whatever
// the allocator does with it. Zero is a legal width and stays constructible.
TEST(V11251CliffordFailLoud, NegativeWidthIsRejected) {
    // Braced rather than parenthesised: `StabilizerState(bad)` is a declaration
    // of a variable named `bad`, not a temporary, and would not compile.
    for (int bad : {-1, -2, -64, -1000}) {
        SCOPED_TRACE("n=" + std::to_string(bad));
        EXPECT_THROW(StabilizerState{bad}, std::invalid_argument);
        EXPECT_THROW(StabilizerState::ColumnTableau{bad}, std::invalid_argument);
    }
}

TEST(V11251CliffordFailLoud, NegativeWidthMessageNamesTheTypeAndTheValue) {
    try {
        StabilizerState st(-7);
        ADD_FAILURE() << "a negative width was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("StabilizerState"), std::string::npos) << msg;
        EXPECT_NE(msg.find("n_qubits must be >= 0"), std::string::npos) << msg;
        EXPECT_NE(msg.find("-7"), std::string::npos) << msg;
    }
    try {
        StabilizerState::ColumnTableau c(-7);
        ADD_FAILURE() << "a negative width was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("StabilizerState::ColumnTableau"), std::string::npos) << msg;
        EXPECT_NE(msg.find("n_qubits must be >= 0"), std::string::npos) << msg;
        EXPECT_NE(msg.find("-7"), std::string::npos) << msg;
    }
}

TEST(V11251CliffordFailLoud, ZeroAndPositiveWidthsAreAccepted) {
    for (int n : {0, 1, 2, 64, 65}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        EXPECT_NO_THROW(StabilizerState{n});
        EXPECT_NO_THROW(StabilizerState::ColumnTableau{n});
    }
}

// =============================================================================
// expectation_pauli - the one method whose argument is a string
// =============================================================================

TEST(V11251CliffordFailLoud, ExpectationPauliRejectsWrongLength) {
    StabilizerState st(kN);
    EXPECT_THROW(st.expectation_pauli(""), std::invalid_argument);
    EXPECT_THROW(st.expectation_pauli("ZZ"), std::invalid_argument);
    EXPECT_THROW(st.expectation_pauli("ZZZZ"), std::invalid_argument);
    EXPECT_NO_THROW(st.expectation_pauli("ZZZ"));
}
