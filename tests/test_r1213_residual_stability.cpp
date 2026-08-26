// R.1.21.3 - the Class C residuals are measurements, and must not depend on
// the compiler flags they were built with.
//
// THE LESSON THIS SUITE ENCODES, because the way the defect hid is the point.
//
// R.1.21.1 already tested the unitarity residual by comparing the library's
// upper-triangle walk against a full-walk reference in the test tree. That
// comparison passed at -march=native while BOTH sides were wrong: under
// -ffast-math the compiler transformed each into computing the same wrong
// value for a Hadamard. Two implementations agreeing proves only that they
// agree. It cannot detect a transformation applied to both.
//
// So the assertions here compare against KNOWN QUANTITIES rather than against
// another implementation. A quantity derived from the type's epsilon and the
// project's own INV_SQRT2 cannot be co-corrupted.
//
// Every expected value below is DERIVED, never transcribed: each is an exact
// multiple of the double epsilon, and the operands are built from INV_SQRT2 or
// from epsilon itself. Recording an observed output instead would pin whatever
// the build happens to produce, which is the failure mode this suite exists to
// prevent.
//
// WHAT MAKES THESE ASSERTIONS SOUND: the residuals live in
// src/validate_physical.cpp, compiled -fno-fast-math, and the project builds
// without LTO, so a call from this translation unit cannot be inlined into a
// fast-math context. If either of those changes, these tests are the ones that
// should fail first.

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

using namespace lindblad;

namespace {

constexpr const char* CTX = "R1213ResidualStability";

// The double's own spacing at 1.0, which is what every expectation here is a
// multiple of.
constexpr double kEps = std::numeric_limits<double>::epsilon();

// h*h where h = INV_SQRT2. h is not exactly 1/sqrt(2), so its square rounds to
// half an epsilon above 0.5 and the doubled sum overshoots 1.0 by exactly one
// epsilon. Folded at compile time from the project constant rather than
// written out, so the operand and the expectation share one source.
constexpr double kHalfPlus = INV_SQRT2 * INV_SQRT2;

// A Hadamard's unitarity residual: |2h^2 - 1| = one epsilon. This is a
// near-cancellation, which is the shape a permissive floating-point model
// perturbs hardest, and is why it is the operand these tests are built on.
constexpr double kHadamardResidual = kEps;

// diag(1 + eps, 1) squares to 1 + 2*eps, so its residual is twice the above.
// A second, independently derived expectation, so the suite does not rest on
// one number.
constexpr double kDiagResidual = 2.0 * kEps;

std::vector<Complex128> hadamard() {
    return {Complex128(INV_SQRT2, 0.0), Complex128(INV_SQRT2, 0.0),
            Complex128(INV_SQRT2, 0.0), Complex128(-INV_SQRT2, 0.0)};
}

// diag(1 + eps, 1)
std::vector<Complex128> nearly_identity() {
    return {Complex128(1.0 + kEps, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0),        Complex128(1.0, 0.0)};
}

const char* kFastMathHint =
    "the arithmetic is being evaluated under -ffast-math again: check that "
    "src/validate_physical.cpp still carries -fno-fast-math, and that no "
    "link-time optimisation is inlining it into a caller";

} // namespace

// =============================================================================
// unitarity_deviation
// =============================================================================

TEST(R1213ResidualStability, HadamardUnitarityIsTheExactIeeeValue) {
    const auto H = hadamard();
    EXPECT_EQ(detail::unitarity_deviation(H.data(), 2), kHadamardResidual)
        << "the unitarity residual is no longer the IEEE value; " << kFastMathHint;
}

TEST(R1213ResidualStability, ADifferentNearCancellationIsAlsoExact) {
    const auto U = nearly_identity();
    EXPECT_EQ(detail::unitarity_deviation(U.data(), 2), kDiagResidual)
        << kFastMathHint;
}

TEST(R1213ResidualStability, ExactlyUnitaryOperandsGiveExactlyZero) {
    // The other end of the range. A residual that drifts off zero for an
    // exactly-representable unitary would be the same class of defect showing
    // up as a false positive rather than a false negative.
    const std::vector<Complex128> I{Complex128(1.0, 0.0), Complex128(0.0, 0.0),
                                    Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
    const std::vector<Complex128> Y{Complex128(0.0, 0.0), Complex128(0.0, -1.0),
                                    Complex128(0.0, 1.0), Complex128(0.0, 0.0)};

    EXPECT_EQ(detail::unitarity_deviation(I.data(), 2), 0.0);
    EXPECT_EQ(detail::unitarity_deviation(Y.data(), 2), 0.0);
}

// =============================================================================
// kraus_tp_deviation
// =============================================================================

TEST(R1213ResidualStability, KrausTracePreservationIsTheExactIeeeValue) {
    // A one-operator channel {H}: sum_k K†K is H†H, so the trace-preservation
    // residual is the same near-cancellation as the unitarity case. Pinned
    // separately because this is a different function with its own
    // accumulation, and the quarantine has to cover all three.
    const std::vector<std::vector<Complex128>> ops{hadamard()};
    EXPECT_EQ(detail::kraus_tp_deviation(ops, 2), kHadamardResidual)
        << kFastMathHint;
}

// =============================================================================
// superop_tp_deviation
// =============================================================================

TEST(R1213ResidualStability, SuperoperatorTracePreservationIsTheExactIeeeValue) {
    // The trace condition sums the output diagonal at each input index pair:
    // sum_r S[(r,r),(ri,ci)] must be 1 on the diagonal and 0 off it. Placing
    // h*h at both (0,0) and (1,1) for each diagonal input reproduces the same
    // 2h^2 overshoot, so this function is pinned to the same derived quantity
    // as the other two.
    //
    // S is assembled from the compile-time constant rather than computed here,
    // so the operand is bit-identical whatever flags THIS translation unit
    // carries. Only the library's summation is under test.
    constexpr std::size_t dim = 2;
    constexpr std::size_t side = dim * dim;

    std::vector<Complex128> S(side * side, Complex128(0.0, 0.0));
    S[0 * side + 0] = Complex128(kHalfPlus, 0.0);  // (ro,co)=(0,0) <- (ri,ci)=(0,0)
    S[3 * side + 0] = Complex128(kHalfPlus, 0.0);  // (ro,co)=(1,1) <- (ri,ci)=(0,0)
    S[0 * side + 3] = Complex128(kHalfPlus, 0.0);  // (ro,co)=(0,0) <- (ri,ci)=(1,1)
    S[3 * side + 3] = Complex128(kHalfPlus, 0.0);  // (ro,co)=(1,1) <- (ri,ci)=(1,1)

    EXPECT_EQ(detail::superop_tp_deviation(S.data(), dim), kHadamardResidual)
        << kFastMathHint;
}

// =============================================================================
// The consequence the numbers stand for
// =============================================================================

TEST(R1213ResidualStability, AVerdictAtTheToleranceBoundaryIsDeterministic) {
    // Why an epsilon-scale wobble is worth a quarantine: the residual is
    // compared against the caller's atol, so a matrix sitting near that line
    // has its verdict decided by whichever value the arithmetic produced. A
    // measurement that moves with the target is a verdict that moves with the
    // target, for every operand in that band.
    //
    // The operand measures exactly two epsilons, so a tolerance either side of
    // that must decide the same way on every build.
    const auto U = nearly_identity();

    EXPECT_THROW(
        detail::check_unitary(U.data(), 2, {Validation::Throw, 1.5 * kEps}, CTX),
        std::invalid_argument)
        << "a residual above atol must be rejected on every build";
    EXPECT_NO_THROW(
        detail::check_unitary(U.data(), 2, {Validation::Throw, 2.5 * kEps}, CTX))
        << "a residual below atol must be accepted on every build";
}
