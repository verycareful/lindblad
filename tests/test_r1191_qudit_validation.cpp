// R.1.19.1 test wave — qudit-layer primitive fail-loud validation.
//
// R.1.19.0 wired the shared checker into every qudit backend. The qudit failure
// mode differs from the qubit one: an over-range digit q makes the stride
// ipow(d, q) exceed the state dimension, so the qudit statevector previously
// SILENTLY no-op'd (n_outer == 0) instead of corrupting memory, while a
// negative q cast to a huge exponent and diverged. Both are now loud bounds
// errors. The checker also unifies the qudit-MPS bounds path, which used to
// throw std::invalid_argument for an out-of-range digit and now throws
// std::out_of_range like every other layer (see also the pin in
// test_r1122_fill_frontends.cpp).
//
// Every apply-primitive across QuditStatevector, QuditDensityMatrix, QuditMPS,
// and QuditCliffordSimulator is driven through the full negative-path matrix:
// out-of-range digit (below / at / above / far above) -> std::out_of_range;
// non-distinct digits and wrong matrix / Kraus size -> std::invalid_argument;
// with positive controls per primitive.

#include <gtest/gtest.h>

#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/types.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

constexpr int kN = 3;   // 3 qudits
constexpr int kD = 3;   // qutrits: genuinely general-d (not the d=2 qubit case)
const std::vector<int> kBadIndices = {-1, kN, kN + 1, 64};

// d x d ... row-major identity with `rows` rows (rows*rows entries).
std::vector<Complex128> qidentity(size_t rows) {
    std::vector<Complex128> m(rows * rows, Complex128(0.0, 0.0));
    for (size_t i = 0; i < rows; ++i) m[i * rows + i] = Complex128(1.0, 0.0);
    return m;
}

const size_t kU1 = static_cast<size_t>(kD);          // 1-qudit matrix is d x d
const size_t kU2 = static_cast<size_t>(kD) * kD;     // 2-qudit matrix is d^2 x d^2
const size_t kU3 = static_cast<size_t>(kD) * kD * kD; // 3-qudit matrix is d^3 x d^3

} // namespace

// =============================================================================
// QuditStatevector
// =============================================================================

TEST(R1191QuditValidation, SvApply1QuditOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        QuditStatevector sv(kN, kD);
        EXPECT_THROW(sv.apply_1qudit(bad, qidentity(kU1)), std::out_of_range);
    }
}

TEST(R1191QuditValidation, SvApply1QuditWrongSizeThrows) {
    QuditStatevector sv(kN, kD);
    EXPECT_THROW(sv.apply_1qudit(0, qidentity(2)), std::invalid_argument);  // 4 != 9
}

TEST(R1191QuditValidation, SvApply2QuditOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { QuditStatevector sv(kN, kD); EXPECT_THROW(sv.apply_2qudit(bad, 1, qidentity(kU2)), std::out_of_range); }
        { QuditStatevector sv(kN, kD); EXPECT_THROW(sv.apply_2qudit(0, bad, qidentity(kU2)), std::out_of_range); }
    }
}

TEST(R1191QuditValidation, SvApply2QuditNonDistinctThrows) {
    QuditStatevector sv(kN, kD);
    EXPECT_THROW(sv.apply_2qudit(1, 1, qidentity(kU2)), std::invalid_argument);
}

TEST(R1191QuditValidation, SvApply2QuditWrongSizeThrows) {
    QuditStatevector sv(kN, kD);
    EXPECT_THROW(sv.apply_2qudit(0, 1, qidentity(kU1)), std::invalid_argument);  // 9 != 81
}

TEST(R1191QuditValidation, SvApplyKQuditOutOfRangeThrows) {
    // The k>=3 path validates the whole digit list; a bad digit in any position
    // throws before the matrix is touched.
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { QuditStatevector sv(kN, kD); EXPECT_THROW(sv.apply_kqudit({bad, 1, 2}, qidentity(kU3)), std::out_of_range); }
        { QuditStatevector sv(kN, kD); EXPECT_THROW(sv.apply_kqudit({0, bad, 2}, qidentity(kU3)), std::out_of_range); }
        { QuditStatevector sv(kN, kD); EXPECT_THROW(sv.apply_kqudit({0, 1, bad}, qidentity(kU3)), std::out_of_range); }
    }
}

TEST(R1191QuditValidation, SvApplyKQuditWrongSizeThrows) {
    QuditStatevector sv(kN, kD);
    EXPECT_THROW(sv.apply_kqudit({0, 1, 2}, qidentity(kU2)), std::invalid_argument);  // 81 != 729
}

TEST(R1191QuditValidation, SvApplyKQuditNonDistinctThrows) {
    // Size is checked before distinctness on the k>=3 path, so the matrix must
    // be correctly sized for the distinctness error to be the one that fires.
    QuditStatevector sv(kN, kD);
    EXPECT_THROW(sv.apply_kqudit({0, 1, 0}, qidentity(kU3)), std::invalid_argument);
}

TEST(R1191QuditValidation, SvValidDoesNotThrow) {
    QuditStatevector sv(kN, kD);
    EXPECT_NO_THROW(sv.apply_1qudit(0, qidentity(kU1)));
    EXPECT_NO_THROW(sv.apply_2qudit(0, 2, qidentity(kU2)));
    EXPECT_NO_THROW(sv.apply_kqudit({0, 1, 2}, qidentity(kU3)));
}

// =============================================================================
// QuditDensityMatrix
// =============================================================================

TEST(R1191QuditValidation, DmApply1QuditOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        QuditDensityMatrix dm(kN, kD);
        EXPECT_THROW(dm.apply_1qudit(bad, qidentity(kU1)), std::out_of_range);
    }
}

TEST(R1191QuditValidation, DmApply1QuditWrongSizeThrows) {
    QuditDensityMatrix dm(kN, kD);
    EXPECT_THROW(dm.apply_1qudit(0, qidentity(2)), std::invalid_argument);
}

TEST(R1191QuditValidation, DmApply2QuditOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { QuditDensityMatrix dm(kN, kD); EXPECT_THROW(dm.apply_2qudit(bad, 1, qidentity(kU2)), std::out_of_range); }
        { QuditDensityMatrix dm(kN, kD); EXPECT_THROW(dm.apply_2qudit(0, bad, qidentity(kU2)), std::out_of_range); }
    }
}

TEST(R1191QuditValidation, DmApply2QuditNonDistinctThrows) {
    QuditDensityMatrix dm(kN, kD);
    EXPECT_THROW(dm.apply_2qudit(1, 1, qidentity(kU2)), std::invalid_argument);
}

TEST(R1191QuditValidation, DmApply2QuditWrongSizeThrows) {
    QuditDensityMatrix dm(kN, kD);
    EXPECT_THROW(dm.apply_2qudit(0, 1, qidentity(kU1)), std::invalid_argument);
}

TEST(R1191QuditValidation, DmKraus1QuditOutOfRangeThrows) {
    const std::vector<std::vector<Complex128>> id1 = {qidentity(kU1)};
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        QuditDensityMatrix dm(kN, kD);
        EXPECT_THROW(dm.apply_kraus_1qudit(bad, id1), std::out_of_range);
    }
}

TEST(R1191QuditValidation, DmKraus1QuditWrongSizeThrows) {
    QuditDensityMatrix dm(kN, kD);
    const std::vector<std::vector<Complex128>> bad_ops = {qidentity(2)};
    EXPECT_THROW(dm.apply_kraus_1qudit(0, bad_ops), std::invalid_argument);
}

TEST(R1191QuditValidation, DmKraus2QuditOutOfRangeThrows) {
    const std::vector<std::vector<Complex128>> id2 = {qidentity(kU2)};
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { QuditDensityMatrix dm(kN, kD); EXPECT_THROW(dm.apply_kraus_2qudit(bad, 1, id2), std::out_of_range); }
        { QuditDensityMatrix dm(kN, kD); EXPECT_THROW(dm.apply_kraus_2qudit(0, bad, id2), std::out_of_range); }
    }
}

TEST(R1191QuditValidation, DmKraus2QuditNonDistinctThrows) {
    QuditDensityMatrix dm(kN, kD);
    const std::vector<std::vector<Complex128>> id2 = {qidentity(kU2)};
    EXPECT_THROW(dm.apply_kraus_2qudit(2, 2, id2), std::invalid_argument);
}

TEST(R1191QuditValidation, DmKraus2QuditWrongSizeThrows) {
    QuditDensityMatrix dm(kN, kD);
    const std::vector<std::vector<Complex128>> bad_ops = {qidentity(kU1)};  // d^2 != d^4
    EXPECT_THROW(dm.apply_kraus_2qudit(0, 1, bad_ops), std::invalid_argument);
}

TEST(R1191QuditValidation, DmValidDoesNotThrow) {
    QuditDensityMatrix dm(kN, kD);
    EXPECT_NO_THROW(dm.apply_1qudit(0, qidentity(kU1)));
    EXPECT_NO_THROW(dm.apply_2qudit(0, 2, qidentity(kU2)));
    EXPECT_NO_THROW(dm.apply_kraus_1qudit(1, {qidentity(kU1)}));
    EXPECT_NO_THROW(dm.apply_kraus_2qudit(0, 1, {qidentity(kU2)}));
}

// =============================================================================
// QuditMPS
// =============================================================================

TEST(R1191QuditValidation, MpsApply1QuditOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        QuditMPS mps(kN, kD, kD * kD * kD);
        EXPECT_THROW(mps.apply_1qudit(bad, qidentity(kU1)), std::out_of_range);
    }
}

TEST(R1191QuditValidation, MpsApply1QuditWrongSizeThrows) {
    QuditMPS mps(kN, kD, kD * kD * kD);
    EXPECT_THROW(mps.apply_1qudit(0, qidentity(2)), std::invalid_argument);
}

TEST(R1191QuditValidation, MpsApply2QuditAdjacentOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        QuditMPS mps(kN, kD, kD * kD * kD);
        EXPECT_THROW(mps.apply_2qudit_adjacent(bad, qidentity(kU2)), std::out_of_range);
    }
    // The last digit has no right neighbour: q = n-1 makes q+1 out of range.
    QuditMPS mps(kN, kD, kD * kD * kD);
    EXPECT_THROW(mps.apply_2qudit_adjacent(kN - 1, qidentity(kU2)), std::out_of_range);
}

TEST(R1191QuditValidation, MpsApply2QuditAdjacentWrongSizeThrows) {
    QuditMPS mps(kN, kD, kD * kD * kD);
    EXPECT_THROW(mps.apply_2qudit_adjacent(0, qidentity(kU1)), std::invalid_argument);
}

TEST(R1191QuditValidation, MpsApply2QuditOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { QuditMPS mps(kN, kD, kD * kD * kD); EXPECT_THROW(mps.apply_2qudit(bad, 1, qidentity(kU2)), std::out_of_range); }
        { QuditMPS mps(kN, kD, kD * kD * kD); EXPECT_THROW(mps.apply_2qudit(0, bad, qidentity(kU2)), std::out_of_range); }
    }
}

TEST(R1191QuditValidation, MpsApply2QuditNonDistinctThrows) {
    QuditMPS mps(kN, kD, kD * kD * kD);
    EXPECT_THROW(mps.apply_2qudit(1, 1, qidentity(kU2)), std::invalid_argument);
}

TEST(R1191QuditValidation, MpsApply2QuditWrongSizeThrows) {
    QuditMPS mps(kN, kD, kD * kD * kD);
    EXPECT_THROW(mps.apply_2qudit(0, 1, qidentity(kU1)), std::invalid_argument);
}

TEST(R1191QuditValidation, MpsValidDoesNotThrow) {
    QuditMPS mps(kN, kD, kD * kD * kD);
    EXPECT_NO_THROW(mps.apply_1qudit(0, qidentity(kU1)));
    EXPECT_NO_THROW(mps.apply_2qudit_adjacent(0, qidentity(kU2)));
    EXPECT_NO_THROW(mps.apply_2qudit(0, 2, qidentity(kU2)));
}

// =============================================================================
// QuditCliffordSimulator
// =============================================================================

TEST(R1191QuditValidation, CliffordSingleQuditOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_X(bad), std::out_of_range); }
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_Z(bad), std::out_of_range); }
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_H(bad), std::out_of_range); }
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_P(bad), std::out_of_range); }
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.measure_qudit(bad), std::out_of_range); }
    }
}

TEST(R1191QuditValidation, CliffordCSUMOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_CSUM(bad, 1), std::out_of_range); }
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_CSUM(0, bad), std::out_of_range); }
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_CSUM_dag(bad, 1), std::out_of_range); }
        { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_CSUM_dag(0, bad), std::out_of_range); }
    }
}

TEST(R1191QuditValidation, CliffordCSUMNonDistinctThrows) {
    { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_CSUM(1, 1), std::invalid_argument); }
    { QuditCliffordSimulator sim(kN, kD); EXPECT_THROW(sim.apply_CSUM_dag(2, 2), std::invalid_argument); }
}

TEST(R1191QuditValidation, CliffordValidDoesNotThrow) {
    QuditCliffordSimulator sim(kN, kD);
    EXPECT_NO_THROW(sim.apply_X(0));
    EXPECT_NO_THROW(sim.apply_Z(1));
    EXPECT_NO_THROW(sim.apply_H(2));
    EXPECT_NO_THROW(sim.apply_P(0));
    EXPECT_NO_THROW(sim.apply_CSUM(0, 1));
    EXPECT_NO_THROW(sim.apply_CSUM_dag(1, 2));
    EXPECT_NO_THROW(sim.measure_qudit(0));
}

// =============================================================================
// Message format (qudit wording + [0, n_qudits) bounds)
// =============================================================================

TEST(R1191QuditValidation, MessageFormatUsesQuditWording) {
    QuditStatevector sv(kN, kD);
    try {
        sv.apply_1qudit(55, qidentity(kU1));
        FAIL() << "expected std::out_of_range";
    } catch (const std::out_of_range& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("QuditStatevector::apply_1qudit"), std::string::npos);
        EXPECT_NE(msg.find("qudit"), std::string::npos);
        EXPECT_NE(msg.find("55"), std::string::npos);
        EXPECT_NE(msg.find("out of range [0, 3)"), std::string::npos);
    }
}
