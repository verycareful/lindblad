// 1.1.25.1 test wave - word and block boundaries of the bit-sliced gate pass.
//
// 1.1.25.0 introduced two independent packings, and the corpus the rework was
// measured on (n = 20, 40, 80, 160) lands on almost none of the sizes where
// either one is interesting.
//
// StabilizerState packs a row's X and Z planes end to end and carries one
// padding word, because the Z plane begins at bit N and a 64-bit read lining
// the planes up per qubit therefore spans a word boundary. Whether that read
// touches the padding at all depends on N mod 64, so n = 63, 64, 65 and
// n = 127, 128, 129 exercise different arms of it than the corpus does.
//
// ColumnTableau transposes in 64x64 blocks. The block walk is driven by
// 2N rather than N, so the aligned sizes are the multiples of 32, and the
// column count is rounded up to a whole block, which means the gather always
// reads columns that carry no tableau data. Sizes on either side of each
// boundary, and sizes spanning several blocks, separate a correct walk from one
// that is right only when everything divides evenly.
//
// Both layouts are compared through v11251::states_equal_exhaustive at the
// sizes where all 4^n Pauli expectations are reachable, and through the
// structural fingerprint at the boundary sizes, where they are not.

#include <gtest/gtest.h>

#include "v11251_clifford_oracle.hpp"

#include "lindblad/simulators/clifford_sim.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

using v11251::apply_step;
using v11251::is_two_qubit;
using v11251::kLayoutOps;
using v11251::Op;
using v11251::op_name;
using v11251::run_bit_sliced;
using v11251::run_row_major;
using v11251::Step;

// Sizes where all 4^n Pauli strings are enumerable, so equality is proved
// rather than probed.
const int kExhaustiveSizes[] = {1, 2, 3, 4, 5, 6};

// One size on each side of every packing boundary, plus sizes spanning several
// transpose blocks. 2N is a multiple of 64 exactly at the multiples of 32.
const int kBoundarySizes[] = {1, 2, 7, 8, 31, 32, 33, 63, 64, 65,
                              95, 96, 127, 128, 129, 160};

// Qubit indices worth touching at width n: the ends, and both sides of every
// 64-bit word boundary the two planes can land on.
std::vector<int> interesting_qubits(int n) {
    std::vector<int> out;
    for (int q : {0, 1, 30, 31, 32, 33, 62, 63, 64, 65, 94, 95, 96, 127, 128}) {
        if (q < n) out.push_back(q);
    }
    if (n >= 1) out.push_back(n - 1);
    if (n >= 2) out.push_back(n - 2);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// A script touching every gate type at every interesting index. Two-qubit gates
// pair each interesting index with its neighbour and with the far end of the
// register, so a gate is applied both within one word and across the whole
// tableau.
std::vector<Step> full_script(int n) {
    std::vector<Step> script;
    const std::vector<int> qs = interesting_qubits(n);
    for (Op op : kLayoutOps) {
        for (int q : qs) {
            if (!is_two_qubit(op)) {
                script.push_back({op, q, -1});
                continue;
            }
            if (n < 2) continue;
            const int near = (q + 1 < n) ? q + 1 : q - 1;
            script.push_back({op, q, near});
            const int far = (q < n / 2) ? n - 1 : 0;
            if (far != q) script.push_back({op, q, far});
        }
    }
    return script;
}

}  // namespace

// =============================================================================
// The freshly constructed tableau
// =============================================================================

// The cheapest possible transpose test: with no gates applied at all, the
// column layout holds one set bit per column and to_state() must place every
// one of them on the diagonal of the row-major tableau. A block walk that is
// off by a block, a word or a lane moves at least one bit off that diagonal.
TEST(V11251CliffordTranspose, FreshTableauMatchesFreshStateAtEverySize) {
    for (int n : kBoundarySizes) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState::ColumnTableau cols(n);
        const StabilizerState from_columns = cols.to_state();
        const StabilizerState direct(n);
        EXPECT_EQ(from_columns.n_qubits, n);
        EXPECT_TRUE(v11251::states_equal_structural(from_columns, direct));
    }
}

TEST(V11251CliffordTranspose, FreshTableauProvedEqualAtSmallSizes) {
    for (int n : kExhaustiveSizes) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState::ColumnTableau cols(n);
        EXPECT_TRUE(v11251::states_equal_exhaustive(cols.to_state(), StabilizerState(n)));
    }
}

// The initial state read directly, independent of any comparison: |0...0> is
// the +1 eigenstate of every Z_q and has no X_q expectation. Restricted to the
// sizes where expectation_pauli's own elimination stays cheap.
TEST(V11251CliffordTranspose, FreshTableauIsTheAllZeroState) {
    for (int n : {1, 2, 7, 8, 31, 32, 33, 63, 64, 65}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const StabilizerState st = StabilizerState::ColumnTableau(n).to_state();
        for (int q : interesting_qubits(n)) {
            SCOPED_TRACE("q=" + std::to_string(q));
            std::string z(static_cast<size_t>(n), 'I');
            z[static_cast<size_t>(q)] = 'Z';
            EXPECT_EQ(st.expectation_pauli(z), 1);
            std::string x(static_cast<size_t>(n), 'I');
            x[static_cast<size_t>(q)] = 'X';
            EXPECT_EQ(st.expectation_pauli(x), 0);
            std::string y(static_cast<size_t>(n), 'I');
            y[static_cast<size_t>(q)] = 'Y';
            EXPECT_EQ(st.expectation_pauli(y), 0);
        }
    }
}

TEST(V11251CliffordTranspose, FreshTableauReportsItsWidth) {
    for (int n : kBoundarySizes) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState::ColumnTableau cols(n);
        EXPECT_EQ(cols.n_qubits(), n);
        EXPECT_EQ(cols.to_state().n_qubits, n);
    }
}

// =============================================================================
// One gate at a time
// =============================================================================

// Every gate type, on its own, from the initial state: the two layouts carry
// independent implementations of each rule (a word-lane update on one side, a
// row sweep on the other), so this is where a disagreement between the two
// rules shows up unmixed with anything else.
TEST(V11251CliffordTranspose, EachGateAloneProvedEqualAtSmallSizes) {
    for (int n : kExhaustiveSizes) {
        for (Op op : kLayoutOps) {
            if (is_two_qubit(op) && n < 2) continue;
            for (int a = 0; a < n; ++a) {
                for (int b = 0; b < n; ++b) {
                    if (is_two_qubit(op) && a == b) continue;
                    if (!is_two_qubit(op) && b != 0) continue;
                    const std::vector<Step> script = {{op, a, b}};
                    SCOPED_TRACE("n=" + std::to_string(n) + " " + op_name(op) +
                                 " a=" + std::to_string(a) + " b=" + std::to_string(b));
                    EXPECT_TRUE(v11251::states_equal_exhaustive(
                        run_bit_sliced(n, script), run_row_major(n, script)));
                }
            }
        }
    }
}

// The same, at the packing boundaries. A gate acting on qubit 63, 64 or 65
// reads and writes columns that sit on either side of a transpose block, and a
// gate acting on the last qubit is the one whose Z column is nearest the
// rounded-up column padding.
TEST(V11251CliffordTranspose, EachGateAloneMatchesAtBoundarySizes) {
    for (int n : kBoundarySizes) {
        for (Op op : kLayoutOps) {
            if (is_two_qubit(op) && n < 2) continue;
            for (int a : interesting_qubits(n)) {
                const int b = (a + 1 < n) ? a + 1 : a - 1;
                if (is_two_qubit(op) && (b < 0 || b == a)) continue;
                const std::vector<Step> script = {{op, a, b}};
                SCOPED_TRACE("n=" + std::to_string(n) + " " + op_name(op) +
                             " a=" + std::to_string(a) + " b=" + std::to_string(b));
                EXPECT_TRUE(v11251::states_equal_structural(
                    run_bit_sliced(n, script), run_row_major(n, script)));
            }
        }
    }
}

// =============================================================================
// Whole scripts
// =============================================================================

// Every gate type at every interesting index, in one run. A single gate leaves
// most of the tableau untouched, so a bug confined to rows that are still in
// their initial state can survive the one-gate cases and not this one.
TEST(V11251CliffordTranspose, FullScriptMatchesAtBoundarySizes) {
    for (int n : kBoundarySizes) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const std::vector<Step> script = full_script(n);
        ASSERT_FALSE(script.empty());
        EXPECT_TRUE(v11251::states_equal_structural(
            run_bit_sliced(n, script), run_row_major(n, script)));
    }
}

TEST(V11251CliffordTranspose, FullScriptProvedEqualAtSmallSizes) {
    for (int n : kExhaustiveSizes) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const std::vector<Step> script = full_script(n);
        ASSERT_FALSE(script.empty());
        EXPECT_TRUE(v11251::states_equal_exhaustive(
            run_bit_sliced(n, script), run_row_major(n, script)));
    }
}

// The script applied one step at a time, comparing after every step. When the
// whole-script test fails this is what says which gate broke it, and it also
// catches a pair of errors that would cancel by the end of a long script.
TEST(V11251CliffordTranspose, ScriptAgreesAfterEveryStep) {
    for (int n : {3, 33, 64, 65}) {
        const std::vector<Step> script = full_script(n);
        StabilizerState::ColumnTableau cols(n);
        StabilizerState st(n);
        for (size_t i = 0; i < script.size(); ++i) {
            apply_step(cols, script[i]);
            apply_step(st, script[i]);
            SCOPED_TRACE("n=" + std::to_string(n) + " step " + std::to_string(i) +
                         " " + op_name(script[i].op) + " a=" +
                         std::to_string(script[i].a) + " b=" +
                         std::to_string(script[i].b));
            ASSERT_TRUE(v11251::states_equal_structural(cols.to_state(), st));
        }
    }
}

// =============================================================================
// to_state() is a read
// =============================================================================

// The conversion must not disturb the tableau it reads, so calling it twice
// gives the same state and gates applied afterwards continue from where they
// left off rather than from a consumed buffer.
TEST(V11251CliffordTranspose, ToStateDoesNotConsumeTheTableau) {
    for (int n : {3, 33, 64, 65, 128}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState::ColumnTableau cols(n);
        const std::vector<Step> script = full_script(n);
        for (const Step& s : script) apply_step(cols, s);

        const StabilizerState first = cols.to_state();
        const StabilizerState second = cols.to_state();
        EXPECT_TRUE(v11251::states_equal_structural(first, second));

        // Continuing after a read must agree with a run that never read.
        cols.apply_h(0);
        StabilizerState reference = run_row_major(n, script);
        reference.apply_h(0);
        EXPECT_TRUE(v11251::states_equal_structural(cols.to_state(), reference));
    }
}

// =============================================================================
// The zero-qubit register
// =============================================================================

// Width zero is a degenerate but constructible register: the column buffer, the
// block walk and the row-major tableau all have nothing to do, and each must
// do that nothing without indexing anything.
TEST(V11251CliffordTranspose, ZeroQubitTableauConvertsToZeroQubitState) {
    StabilizerState::ColumnTableau cols(0);
    EXPECT_EQ(cols.n_qubits(), 0);
    const StabilizerState st = cols.to_state();
    EXPECT_EQ(st.n_qubits, 0);
    EXPECT_EQ(st.words_per_vector(), 0);

    const StabilizerState::OutcomeSlab slab = st.outcome_slab();
    EXPECT_EQ(slab.n_qubits, 0);
    EXPECT_EQ(slab.dim, 0);
    EXPECT_TRUE(slab.basis.empty());
}

// =============================================================================
// words_per_vector
// =============================================================================

// The packed outcome vector is what the slab and the sampling loop index into,
// so its width has to follow N exactly at the boundaries rather than being
// rounded to whatever the tableau row happens to use.
TEST(V11251CliffordTranspose, WordsPerVectorFollowsTheQubitCount) {
    for (int n : kBoundarySizes) {
        SCOPED_TRACE("n=" + std::to_string(n));
        StabilizerState st(n);
        EXPECT_EQ(st.words_per_vector(), (n + 63) / 64);
    }
    EXPECT_EQ(StabilizerState(0).words_per_vector(), 0);
    EXPECT_EQ(StabilizerState(1).words_per_vector(), 1);
    EXPECT_EQ(StabilizerState(64).words_per_vector(), 1);
    EXPECT_EQ(StabilizerState(65).words_per_vector(), 2);
    EXPECT_EQ(StabilizerState(128).words_per_vector(), 2);
    EXPECT_EQ(StabilizerState(129).words_per_vector(), 3);
}
