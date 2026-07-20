// R.1.19.1 test wave — density-matrix primitive fail-loud validation.
//
// R.1.19.0 wired the shared operand checker into DensityMatrix::apply_gate,
// apply_kraus, apply_channel_superop, and apply_permutation. Before that these
// paths ran dm_build_tables with an unchecked qubit index (`1ULL << q` -> UB),
// trusted the Kraus operator sizes, and applied a permutation without proving
// it a bijection (an out-of-range image reads/writes out of the output buffer;
// a repeated image silently drops rows/cols).
//
// Coverage per primitive: out-of-range qubit index (below / at / above / far
// above range) -> std::out_of_range; non-distinct qubits, wrong gate-matrix,
// Kraus-operator, or superoperator size, and a non-bijection full-register
// permutation -> std::invalid_argument; plus a positive control per primitive.

#include <gtest/gtest.h>

#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

constexpr int kN = 3;                    // 3-qubit rho, dim 8
const std::vector<int> kBadIndices = {-1, kN, kN + 1, 64};

std::vector<Complex128> identity(size_t rows) {
    std::vector<Complex128> m(rows * rows, Complex128(0.0, 0.0));
    for (size_t i = 0; i < rows; ++i) m[i * rows + i] = Complex128(1.0, 0.0);
    return m;
}

DensityMatrix make_dm() { return DensityMatrix(kN); }

} // namespace

// =============================================================================
// apply_gate — rho -> U rho U†
// =============================================================================

TEST(R1191DmValidation, ApplyGateOutOfRangeThrows) {
    const auto U2 = identity(4);  // 2-qubit gate block
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { auto dm = make_dm(); EXPECT_THROW(dm.apply_gate(U2, {bad, 1}), std::out_of_range); }
        { auto dm = make_dm(); EXPECT_THROW(dm.apply_gate(U2, {0, bad}), std::out_of_range); }
    }
}

TEST(R1191DmValidation, ApplyGateNonDistinctThrows) {
    auto dm = make_dm();
    EXPECT_THROW(dm.apply_gate(identity(4), {1, 1}), std::invalid_argument);
}

TEST(R1191DmValidation, ApplyGateWrongMatrixSizeThrows) {
    auto dm = make_dm();
    // Two qubits require a 4x4 (16-entry) matrix.
    EXPECT_THROW(dm.apply_gate(identity(2), {0, 1}), std::invalid_argument);
    EXPECT_THROW(dm.apply_gate(std::vector<Complex128>(15, Complex128(0.0, 0.0)), {0, 1}),
                 std::invalid_argument);
}

TEST(R1191DmValidation, ApplyGateValidDoesNotThrow) {
    auto dm = make_dm();
    EXPECT_NO_THROW(dm.apply_gate(identity(2), {0}));
    EXPECT_NO_THROW(dm.apply_gate(identity(4), {0, 2}));
}

// =============================================================================
// apply_kraus — rho -> sum_k K rho K†
// =============================================================================

TEST(R1191DmValidation, ApplyKrausOutOfRangeThrows) {
    const std::vector<std::vector<Complex128>> id1 = {identity(2)};
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        auto dm = make_dm();
        EXPECT_THROW(dm.apply_kraus(id1, {bad}), std::out_of_range);
    }
}

TEST(R1191DmValidation, ApplyKrausNonDistinctThrows) {
    auto dm = make_dm();
    const std::vector<std::vector<Complex128>> id2 = {identity(4)};
    EXPECT_THROW(dm.apply_kraus(id2, {1, 1}), std::invalid_argument);
}

TEST(R1191DmValidation, ApplyKrausWrongOperatorSizeThrows) {
    auto dm = make_dm();
    // Two qubits: every Kraus operator must be 4x4 (16 entries).
    const std::vector<std::vector<Complex128>> bad_ops = {identity(2)};
    EXPECT_THROW(dm.apply_kraus(bad_ops, {0, 1}), std::invalid_argument);
    // A mixed list where only the second operator is the wrong size.
    const std::vector<std::vector<Complex128>> mixed = {identity(4), identity(2)};
    EXPECT_THROW(dm.apply_kraus(mixed, {0, 1}), std::invalid_argument);
}

TEST(R1191DmValidation, ApplyKrausValidDoesNotThrow) {
    auto dm = make_dm();
    const std::vector<std::vector<Complex128>> id1 = {identity(2)};
    EXPECT_NO_THROW(dm.apply_kraus(id1, {1}));
}

// =============================================================================
// apply_channel_superop — rho block <- S · vec(rho block)
// =============================================================================

TEST(R1191DmValidation, ApplyChannelSuperopOutOfRangeThrows) {
    const auto S1 = identity(4);  // 1-qubit superop is 4x4 (= 4^1 x 4^1)
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        auto dm = make_dm();
        EXPECT_THROW(dm.apply_channel_superop(S1, {bad}), std::out_of_range);
    }
}

TEST(R1191DmValidation, ApplyChannelSuperopNonDistinctThrows) {
    auto dm = make_dm();
    // Two-qubit superop is 16x16 (= 4^2 x 4^2); size passes, distinctness trips.
    EXPECT_THROW(dm.apply_channel_superop(identity(16), {1, 1}), std::invalid_argument);
}

TEST(R1191DmValidation, ApplyChannelSuperopWrongSizeThrows) {
    auto dm = make_dm();
    // One qubit requires a 4x4 superop; a 2x2 is rejected.
    EXPECT_THROW(dm.apply_channel_superop(identity(2), {0}), std::invalid_argument);
}

TEST(R1191DmValidation, ApplyChannelSuperopValidDoesNotThrow) {
    auto dm = make_dm();
    EXPECT_NO_THROW(dm.apply_channel_superop(identity(4), {0}));
}

// =============================================================================
// apply_permutation — full-register basis relabel (must be a bijection)
// =============================================================================

TEST(R1191DmValidation, ApplyPermutationWrongSizeThrows) {
    auto dm = make_dm();
    // dim = 8: the permutation must have exactly 8 entries.
    EXPECT_THROW(dm.apply_permutation({0, 1, 2, 3, 4, 5, 6}), std::invalid_argument);
    EXPECT_THROW(dm.apply_permutation({0, 1, 2, 3, 4, 5, 6, 7, 0}), std::invalid_argument);
}

TEST(R1191DmValidation, ApplyPermutationNonBijectionThrows) {
    // Out-of-range image, negative image, and a repeated image each break the
    // bijection contract.
    { auto dm = make_dm(); EXPECT_THROW(dm.apply_permutation({0, 1, 2, 3, 4, 5, 6, 8}), std::invalid_argument); }
    { auto dm = make_dm(); EXPECT_THROW(dm.apply_permutation({0, 1, 2, 3, 4, 5, 6, -1}), std::invalid_argument); }
    { auto dm = make_dm(); EXPECT_THROW(dm.apply_permutation({0, 1, 2, 3, 4, 5, 6, 6}), std::invalid_argument); }
}

TEST(R1191DmValidation, ApplyPermutationValidDoesNotThrow) {
    auto dm = make_dm();
    // A genuine bijection of [0, 8): reverse the basis order.
    EXPECT_NO_THROW(dm.apply_permutation({7, 6, 5, 4, 3, 2, 1, 0}));
}

// =============================================================================
// Message format
// =============================================================================

TEST(R1191DmValidation, MessageFormatMatchesCircuitLayer) {
    auto dm = make_dm();
    try {
        dm.apply_gate(identity(2), {42});
        FAIL() << "expected std::out_of_range";
    } catch (const std::out_of_range& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("DensityMatrix::apply_gate"), std::string::npos);
        EXPECT_NE(msg.find("42"), std::string::npos);
        EXPECT_NE(msg.find("out of range [0, 3)"), std::string::npos);
    }
}
