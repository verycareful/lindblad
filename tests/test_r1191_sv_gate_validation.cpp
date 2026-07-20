// R.1.19.1 test wave — statevector gate-kernel fail-loud validation.
//
// R.1.19.0 made every public gates::apply_* primitive check its operands
// before touching memory: an out-of-range qubit throws std::out_of_range, a
// bad operand structure (non-distinct qubits where required, wrong matrix or
// permutation size, a non-bijection permutation, a target among the controls)
// throws std::invalid_argument. Before that the kernels computed `1ULL << q`
// from the caller's index and strode through the amplitude array, so an
// out-of-range index was undefined behaviour (shift-width overflow past q=63,
// out-of-bounds strided writes for n<=q<64) and a non-distinct two-qubit pair
// silently corrupted state (the old `assert` compiled out under NDEBUG, the
// only supported build).
//
// This suite drives every single-, two-, and three-qubit kernel plus
// apply_unitary / apply_mcx / apply_mcp / apply_permutation through the whole
// negative-path matrix: index below range (-1), at range (n), above range
// (n+1), and far above (64, the old shift-overflow class); non-distinct
// operands; wrong matrix / permutation size; and a non-bijection permutation.
// Each family also has a positive control proving a valid call does NOT throw,
// so the guards are not over-eager. The throwing paths double as
// AddressSanitizer / UndefinedBehaviorSanitizer bait: a missing check would
// surface as UB on the sanitiser leg of the suite instead of a clean throw.

#include <gtest/gtest.h>

#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

// A 3-qubit register (dim 8) is the reference size throughout: every "bad"
// index below is out of range for it. n=3 is small enough that a valid call
// is cheap and large enough that distinct multi-qubit operands exist.
constexpr int kN = 3;

// Bad qubit indices spanning the whole failure taxonomy for an n=3 register:
// below range, at range, just above, and far above (the pre-R.1.19.0
// `1ULL << 64` shift-width-overflow case, now a plain bounds throw).
const std::vector<int> kBadIndices = {-1, kN, kN + 1, 64};

Statevector make_sv() {
    Statevector sv(kN);
    sv.initialize_basis(0);
    return sv;
}

// Row-major identity of the given matrix dimension (rows == cols).
std::vector<Complex128> identity(size_t rows) {
    std::vector<Complex128> m(rows * rows, Complex128(0.0, 0.0));
    for (size_t i = 0; i < rows; ++i) m[i * rows + i] = Complex128(1.0, 0.0);
    return m;
}

// ---- single-qubit kernels: (name, apply(sv, q)) ----
using SqGate = std::function<void(Statevector&, int)>;
std::vector<std::pair<const char*, SqGate>> single_qubit_gates() {
    return {
        {"x",    [](Statevector& s, int q) { gates::apply_x(s, q); }},
        {"y",    [](Statevector& s, int q) { gates::apply_y(s, q); }},
        {"z",    [](Statevector& s, int q) { gates::apply_z(s, q); }},
        {"h",    [](Statevector& s, int q) { gates::apply_h(s, q); }},
        {"s",    [](Statevector& s, int q) { gates::apply_s(s, q); }},
        {"sdg",  [](Statevector& s, int q) { gates::apply_sdg(s, q); }},
        {"t",    [](Statevector& s, int q) { gates::apply_t(s, q); }},
        {"tdg",  [](Statevector& s, int q) { gates::apply_tdg(s, q); }},
        {"sx",   [](Statevector& s, int q) { gates::apply_sx(s, q); }},
        {"sxdg", [](Statevector& s, int q) { gates::apply_sxdg(s, q); }},
        {"rx",   [](Statevector& s, int q) { gates::apply_rx(s, q, 0.7); }},
        {"ry",   [](Statevector& s, int q) { gates::apply_ry(s, q, 0.7); }},
        {"rz",   [](Statevector& s, int q) { gates::apply_rz(s, q, 0.7); }},
        {"p",    [](Statevector& s, int q) { gates::apply_p(s, q, 0.7); }},
        {"u",    [](Statevector& s, int q) { gates::apply_u(s, q, 0.3, 0.5, 0.7); }},
        {"u1",   [](Statevector& s, int q) { gates::apply_u1(s, q, 0.7); }},
        {"u2",   [](Statevector& s, int q) { gates::apply_u2(s, q, 0.5, 0.7); }},
        {"u3",   [](Statevector& s, int q) { gates::apply_u3(s, q, 0.3, 0.5, 0.7); }},
    };
}

// ---- two-qubit kernels: (name, apply(sv, a, b)) ----
using TqGate = std::function<void(Statevector&, int, int)>;
std::vector<std::pair<const char*, TqGate>> two_qubit_gates() {
    return {
        {"cx",    [](Statevector& s, int a, int b) { gates::apply_cx(s, a, b); }},
        {"cy",    [](Statevector& s, int a, int b) { gates::apply_cy(s, a, b); }},
        {"cz",    [](Statevector& s, int a, int b) { gates::apply_cz(s, a, b); }},
        {"ch",    [](Statevector& s, int a, int b) { gates::apply_ch(s, a, b); }},
        {"swap",  [](Statevector& s, int a, int b) { gates::apply_swap(s, a, b); }},
        {"iswap", [](Statevector& s, int a, int b) { gates::apply_iswap(s, a, b); }},
        {"crx",   [](Statevector& s, int a, int b) { gates::apply_crx(s, a, b, 0.7); }},
        {"cry",   [](Statevector& s, int a, int b) { gates::apply_cry(s, a, b, 0.7); }},
        {"crz",   [](Statevector& s, int a, int b) { gates::apply_crz(s, a, b, 0.7); }},
        {"cp",    [](Statevector& s, int a, int b) { gates::apply_cp(s, a, b, 0.7); }},
        {"cu",    [](Statevector& s, int a, int b) { gates::apply_cu(s, a, b, 0.3, 0.5, 0.7, 0.1); }},
        {"ecr",   [](Statevector& s, int a, int b) { gates::apply_ecr(s, a, b); }},
        {"rzx",   [](Statevector& s, int a, int b) { gates::apply_rzx(s, a, b, 0.7); }},
        {"rxx",   [](Statevector& s, int a, int b) { gates::apply_rxx(s, a, b, 0.7); }},
        {"ryy",   [](Statevector& s, int a, int b) { gates::apply_ryy(s, a, b, 0.7); }},
        {"rzz",   [](Statevector& s, int a, int b) { gates::apply_rzz(s, a, b, 0.7); }},
    };
}

// ---- three-qubit kernels: (name, apply(sv, a, b, c)) ----
using ThreeGate = std::function<void(Statevector&, int, int, int)>;
std::vector<std::pair<const char*, ThreeGate>> three_qubit_gates() {
    return {
        {"ccx",   [](Statevector& s, int a, int b, int c) { gates::apply_ccx(s, a, b, c); }},
        {"ccz",   [](Statevector& s, int a, int b, int c) { gates::apply_ccz(s, a, b, c); }},
        {"cswap", [](Statevector& s, int a, int b, int c) { gates::apply_cswap(s, a, b, c); }},
        {"rccx",  [](Statevector& s, int a, int b, int c) { gates::apply_rccx(s, a, b, c); }},
    };
}

} // namespace

// =============================================================================
// Single-qubit kernels
// =============================================================================

TEST(R1191SvGateValidation, SingleQubitOutOfRangeThrows) {
    for (const auto& [name, fn] : single_qubit_gates()) {
        for (int q : kBadIndices) {
            SCOPED_TRACE(std::string(name) + " q=" + std::to_string(q));
            Statevector sv = make_sv();
            EXPECT_THROW(fn(sv, q), std::out_of_range);
        }
    }
}

TEST(R1191SvGateValidation, SingleQubitValidDoesNotThrow) {
    for (const auto& [name, fn] : single_qubit_gates()) {
        SCOPED_TRACE(name);
        Statevector sv = make_sv();
        EXPECT_NO_THROW(fn(sv, 0));
        EXPECT_NO_THROW(fn(sv, kN - 1));  // the last valid index
    }
}

// =============================================================================
// Two-qubit kernels
// =============================================================================

TEST(R1191SvGateValidation, TwoQubitOutOfRangeThrows) {
    for (const auto& [name, fn] : two_qubit_gates()) {
        for (int bad : kBadIndices) {
            SCOPED_TRACE(std::string(name) + " bad=" + std::to_string(bad));
            {
                Statevector sv = make_sv();
                EXPECT_THROW(fn(sv, bad, 0), std::out_of_range);  // first operand
            }
            {
                Statevector sv = make_sv();
                EXPECT_THROW(fn(sv, 0, bad), std::out_of_range);  // second operand
            }
        }
    }
}

TEST(R1191SvGateValidation, TwoQubitNonDistinctThrows) {
    // The old assert(a != b) compiled out under NDEBUG; a Release build silently
    // corrupted state on an aliased index pair. Now it is a loud structure error.
    for (const auto& [name, fn] : two_qubit_gates()) {
        SCOPED_TRACE(name);
        Statevector sv = make_sv();
        EXPECT_THROW(fn(sv, 1, 1), std::invalid_argument);
    }
}

TEST(R1191SvGateValidation, TwoQubitValidDoesNotThrow) {
    for (const auto& [name, fn] : two_qubit_gates()) {
        SCOPED_TRACE(name);
        Statevector sv = make_sv();
        EXPECT_NO_THROW(fn(sv, 0, 1));
        EXPECT_NO_THROW(fn(sv, 2, 0));  // reversed / non-adjacent
    }
}

// =============================================================================
// Three-qubit kernels
// =============================================================================

TEST(R1191SvGateValidation, ThreeQubitOutOfRangeThrows) {
    for (const auto& [name, fn] : three_qubit_gates()) {
        for (int bad : kBadIndices) {
            SCOPED_TRACE(std::string(name) + " bad=" + std::to_string(bad));
            { Statevector sv = make_sv(); EXPECT_THROW(fn(sv, bad, 1, 2), std::out_of_range); }
            { Statevector sv = make_sv(); EXPECT_THROW(fn(sv, 0, bad, 2), std::out_of_range); }
            { Statevector sv = make_sv(); EXPECT_THROW(fn(sv, 0, 1, bad), std::out_of_range); }
        }
    }
}

TEST(R1191SvGateValidation, ThreeQubitNonDistinctThrows) {
    for (const auto& [name, fn] : three_qubit_gates()) {
        SCOPED_TRACE(name);
        { Statevector sv = make_sv(); EXPECT_THROW(fn(sv, 0, 0, 1), std::invalid_argument); }
        { Statevector sv = make_sv(); EXPECT_THROW(fn(sv, 0, 1, 0), std::invalid_argument); }
        { Statevector sv = make_sv(); EXPECT_THROW(fn(sv, 1, 0, 0), std::invalid_argument); }
    }
}

TEST(R1191SvGateValidation, ThreeQubitValidDoesNotThrow) {
    for (const auto& [name, fn] : three_qubit_gates()) {
        SCOPED_TRACE(name);
        Statevector sv = make_sv();
        EXPECT_NO_THROW(fn(sv, 0, 1, 2));
        EXPECT_NO_THROW(fn(sv, 2, 0, 1));  // permuted operands
    }
}

// =============================================================================
// apply_unitary — arbitrary N-qubit dense unitary
// =============================================================================

TEST(R1191SvGateValidation, UnitaryOutOfRangeThrows) {
    const auto U2 = identity(4);  // 2-qubit block
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { Statevector sv = make_sv(); EXPECT_THROW(gates::apply_unitary(sv, {bad, 1}, U2), std::out_of_range); }
        { Statevector sv = make_sv(); EXPECT_THROW(gates::apply_unitary(sv, {0, bad}, U2), std::out_of_range); }
    }
}

TEST(R1191SvGateValidation, UnitaryNonDistinctTargetsThrows) {
    Statevector sv = make_sv();
    EXPECT_THROW(gates::apply_unitary(sv, {1, 1}, identity(4)), std::invalid_argument);
}

TEST(R1191SvGateValidation, UnitaryWrongMatrixSizeThrows) {
    Statevector sv = make_sv();
    // Two targets need a 4x4 (16-entry) matrix; anything else is rejected.
    EXPECT_THROW(gates::apply_unitary(sv, {0, 1}, identity(2)), std::invalid_argument);
    EXPECT_THROW(gates::apply_unitary(sv, {0, 1}, std::vector<Complex128>(15, Complex128(0.0, 0.0))),
                 std::invalid_argument);
}

TEST(R1191SvGateValidation, UnitaryValidDoesNotThrow) {
    Statevector sv = make_sv();
    EXPECT_NO_THROW(gates::apply_unitary(sv, {0}, identity(2)));
    EXPECT_NO_THROW(gates::apply_unitary(sv, {0, 2}, identity(4)));
}

// =============================================================================
// apply_mcx — multi-controlled X
// =============================================================================

TEST(R1191SvGateValidation, McxOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { Statevector sv = make_sv(); EXPECT_THROW(gates::apply_mcx(sv, {bad, 1}, 2), std::out_of_range); }
        { Statevector sv = make_sv(); EXPECT_THROW(gates::apply_mcx(sv, {0, 1}, bad), std::out_of_range); }
    }
}

TEST(R1191SvGateValidation, McxNonDistinctControlsThrows) {
    Statevector sv = make_sv();
    EXPECT_THROW(gates::apply_mcx(sv, {0, 0}, 2), std::invalid_argument);
}

TEST(R1191SvGateValidation, McxTargetAmongControlsThrows) {
    Statevector sv = make_sv();
    EXPECT_THROW(gates::apply_mcx(sv, {0, 1}, 1), std::invalid_argument);
}

TEST(R1191SvGateValidation, McxValidDoesNotThrow) {
    Statevector sv = make_sv();
    EXPECT_NO_THROW(gates::apply_mcx(sv, {0, 1}, 2));
    EXPECT_NO_THROW(gates::apply_mcx(sv, {}, 0));  // zero controls == plain X
}

// =============================================================================
// apply_mcp — multi-controlled phase
// =============================================================================

TEST(R1191SvGateValidation, McpOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        Statevector sv = make_sv();
        EXPECT_THROW(gates::apply_mcp(sv, {0, bad}, 0.7), std::out_of_range);
    }
}

TEST(R1191SvGateValidation, McpNonDistinctThrows) {
    Statevector sv = make_sv();
    EXPECT_THROW(gates::apply_mcp(sv, {1, 1}, 0.7), std::invalid_argument);
}

TEST(R1191SvGateValidation, McpValidDoesNotThrow) {
    Statevector sv = make_sv();
    EXPECT_NO_THROW(gates::apply_mcp(sv, {0, 1, 2}, 0.7));
}

// =============================================================================
// apply_permutation — basis permutation on a target subspace
// =============================================================================

TEST(R1191SvGateValidation, PermutationOutOfRangeQubitThrows) {
    // A 2-qubit target subspace: identity permutation of [0,4).
    const std::vector<int> id_perm = {0, 1, 2, 3};
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { Statevector sv = make_sv(); EXPECT_THROW(gates::apply_permutation(sv, {bad, 1}, id_perm), std::out_of_range); }
        { Statevector sv = make_sv(); EXPECT_THROW(gates::apply_permutation(sv, {0, bad}, id_perm), std::out_of_range); }
    }
}

TEST(R1191SvGateValidation, PermutationNonDistinctQubitsThrows) {
    Statevector sv = make_sv();
    EXPECT_THROW(gates::apply_permutation(sv, {1, 1}, {0, 1, 2, 3}), std::invalid_argument);
}

TEST(R1191SvGateValidation, PermutationWrongSizeThrows) {
    Statevector sv = make_sv();
    // Two target qubits require a length-4 permutation.
    EXPECT_THROW(gates::apply_permutation(sv, {0, 1}, {0, 1, 2}), std::invalid_argument);
    EXPECT_THROW(gates::apply_permutation(sv, {0, 1}, {0, 1, 2, 3, 0}), std::invalid_argument);
}

TEST(R1191SvGateValidation, PermutationNonBijectionThrows) {
    Statevector sv = make_sv();
    // Out-of-range image (4 not in [0,4)) and a repeated image both break the
    // bijection contract; either would corrupt the strided relabel.
    EXPECT_THROW(gates::apply_permutation(sv, {0, 1}, {0, 1, 2, 4}), std::invalid_argument);
    EXPECT_THROW(gates::apply_permutation(sv, {0, 1}, {0, 1, 1, 3}), std::invalid_argument);
    EXPECT_THROW(gates::apply_permutation(sv, {0, 1}, {0, 1, 2, -1}), std::invalid_argument);
}

TEST(R1191SvGateValidation, PermutationValidDoesNotThrow) {
    Statevector sv = make_sv();
    EXPECT_NO_THROW(gates::apply_permutation(sv, {0, 1}, {1, 0, 3, 2}));  // a genuine swap
}

// =============================================================================
// Message format matches the circuit layer
// =============================================================================

TEST(R1191SvGateValidation, MessageFormatMatchesCircuitLayer) {
    // The kernel messages reuse the circuit layer's wording so a failure reads
    // the same no matter which layer raised it: "<ctx>: qubit index N out of
    // range [0, M)" for bounds, "<ctx>: qubits must be distinct" for structure.
    Statevector sv = make_sv();
    try {
        gates::apply_h(sv, 99);
        FAIL() << "expected std::out_of_range";
    } catch (const std::out_of_range& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("h"), std::string::npos);
        EXPECT_NE(msg.find("99"), std::string::npos);
        EXPECT_NE(msg.find("out of range [0, 3)"), std::string::npos);
    }
    try {
        gates::apply_cx(sv, 1, 1);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("cx"), std::string::npos);
        EXPECT_NE(msg.find("distinct"), std::string::npos);
    }
}
