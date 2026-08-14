// R.1.12.1 total-coverage suite, Batch 1: lindblad/types.hpp.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 1: foundations".
//
// Covers Complex128 (every operator and helper, pinned against algebraic
// identities and exact values), the numeric constants, and the aligned
// allocation helpers. Test-only release content (.1 slot).

#include <gtest/gtest.h>

#include "lindblad/types.hpp"

#include <cmath>
#include <cstdint>

using namespace lindblad;

static constexpr double kTol = 1e-15;

// =============================================================================
// Construction
// =============================================================================

TEST(R1121Types, DefaultConstructionIsZero) {
    Complex128 z;
    EXPECT_EQ(z.real, 0.0);
    EXPECT_EQ(z.imag, 0.0);
}

TEST(R1121Types, TwoArgAndExplicitRealConstruction) {
    Complex128 a(3.0, -4.0);
    EXPECT_EQ(a.real, 3.0);
    EXPECT_EQ(a.imag, -4.0);

    Complex128 r(2.5);  // explicit real constructor
    EXPECT_EQ(r.real, 2.5);
    EXPECT_EQ(r.imag, 0.0);
}

TEST(R1121Types, AlignmentOfTypeIs16) {
    EXPECT_EQ(alignof(Complex128), 16u);
    EXPECT_EQ(sizeof(Complex128), 16u);
}

// =============================================================================
// Arithmetic operators: exact values
// =============================================================================

TEST(R1121Types, AdditionSubtraction) {
    Complex128 a(1.0, 2.0), b(3.0, -5.0);
    Complex128 s = a + b;
    EXPECT_EQ(s.real, 4.0);
    EXPECT_EQ(s.imag, -3.0);
    Complex128 d = a - b;
    EXPECT_EQ(d.real, -2.0);
    EXPECT_EQ(d.imag, 7.0);
}

TEST(R1121Types, MultiplicationKnownValue) {
    // (1 + 2i)(3 - 5i) = 3 - 5i + 6i + 10 = 13 + i
    Complex128 a(1.0, 2.0), b(3.0, -5.0);
    Complex128 p = a * b;
    EXPECT_EQ(p.real, 13.0);
    EXPECT_EQ(p.imag, 1.0);
}

TEST(R1121Types, ImaginaryUnitSquaresToMinusOne) {
    Complex128 i(0.0, 1.0);
    Complex128 m = i * i;
    EXPECT_EQ(m.real, -1.0);
    EXPECT_EQ(m.imag, 0.0);
}

TEST(R1121Types, ScalarMultiplyBothSidesAndDivide) {
    Complex128 a(1.5, -2.0);
    Complex128 l = a * 4.0;
    Complex128 r = 4.0 * a;
    EXPECT_EQ(l.real, 6.0);
    EXPECT_EQ(l.imag, -8.0);
    EXPECT_EQ(r.real, l.real);
    EXPECT_EQ(r.imag, l.imag);

    Complex128 q = l / 4.0;
    EXPECT_EQ(q.real, a.real);
    EXPECT_EQ(q.imag, a.imag);
}

TEST(R1121Types, CompoundAssignmentChain) {
    Complex128 a(1.0, 1.0);
    a += Complex128(2.0, -3.0);   // (3, -2)
    EXPECT_EQ(a.real, 3.0);
    EXPECT_EQ(a.imag, -2.0);
    a -= Complex128(1.0, 1.0);    // (2, -3)
    EXPECT_EQ(a.real, 2.0);
    EXPECT_EQ(a.imag, -3.0);
    a *= Complex128(0.0, 1.0);    // multiply by i: (3, 2)
    EXPECT_EQ(a.real, 3.0);
    EXPECT_EQ(a.imag, 2.0);
    a *= 2.0;                     // (6, 4)
    EXPECT_EQ(a.real, 6.0);
    EXPECT_EQ(a.imag, 4.0);
}

TEST(R1121Types, UnaryMinusAndEquality) {
    Complex128 a(1.0, -2.0);
    Complex128 n = -a;
    EXPECT_EQ(n.real, -1.0);
    EXPECT_EQ(n.imag, 2.0);
    EXPECT_TRUE(a == Complex128(1.0, -2.0));
    EXPECT_TRUE(a != n);
    EXPECT_FALSE(a == n);
}

// =============================================================================
// Algebraic identities (sampled)
// =============================================================================

TEST(R1121Types, ConjugationDistributesOverProduct) {
    Complex128 a(0.3, -1.7), b(-2.2, 0.9);
    Complex128 lhs = (a * b).conj();
    Complex128 rhs = a.conj() * b.conj();
    EXPECT_NEAR(lhs.real, rhs.real, kTol);
    EXPECT_NEAR(lhs.imag, rhs.imag, kTol);
}

TEST(R1121Types, NormSqIsMultiplicative) {
    Complex128 a(0.6, 0.8), b(1.0, -2.0);
    EXPECT_NEAR((a * b).norm_sq(), a.norm_sq() * b.norm_sq(), 1e-14);
}

TEST(R1121Types, DistributivitySample) {
    Complex128 a(1.1, -0.4), b(0.5, 2.0), c(-3.0, 0.25);
    Complex128 lhs = a * (b + c);
    Complex128 rhs = a * b + a * c;
    EXPECT_NEAR(lhs.real, rhs.real, 1e-14);
    EXPECT_NEAR(lhs.imag, rhs.imag, 1e-14);
}

// =============================================================================
// conj / norm / arg / polar / exp_i
// =============================================================================

TEST(R1121Types, ConjNormArgOn345Triangle) {
    Complex128 a(3.0, 4.0);
    Complex128 c = a.conj();
    EXPECT_EQ(c.real, 3.0);
    EXPECT_EQ(c.imag, -4.0);
    EXPECT_EQ(a.norm_sq(), 25.0);
    EXPECT_NEAR(a.norm(), 5.0, kTol);
    EXPECT_NEAR(a.arg(), std::atan2(4.0, 3.0), kTol);
}

TEST(R1121Types, PolarRoundTrip) {
    const double r = 2.0, theta = PI / 3.0;
    Complex128 p = Complex128::polar(r, theta);
    EXPECT_NEAR(p.norm(), r, kTol);
    EXPECT_NEAR(p.arg(), theta, kTol);
    EXPECT_NEAR(p.real, 1.0, 1e-14);              // 2*cos(pi/3) = 1
    EXPECT_NEAR(p.imag, std::sqrt(3.0), 1e-14);   // 2*sin(pi/3) = sqrt(3)
}

TEST(R1121Types, ExpIKnownAngles) {
    Complex128 e0 = Complex128::exp_i(0.0);
    EXPECT_EQ(e0.real, 1.0);
    EXPECT_EQ(e0.imag, 0.0);

    Complex128 epi = Complex128::exp_i(PI);
    EXPECT_NEAR(epi.real, -1.0, kTol);
    EXPECT_NEAR(epi.imag, 0.0, 1e-15);

    Complex128 ehalf = Complex128::exp_i(PI_2);
    EXPECT_NEAR(ehalf.real, 0.0, 1e-15);
    EXPECT_NEAR(ehalf.imag, 1.0, kTol);

    // exp_i is a unit complex for arbitrary angles
    EXPECT_NEAR(Complex128::exp_i(0.7321).norm_sq(), 1.0, kTol);
}

TEST(R1121Types, ExpIAdditionTheorem) {
    // exp(i a) * exp(i b) == exp(i (a + b))
    const double a = 0.9, b = -2.3;
    Complex128 lhs = Complex128::exp_i(a) * Complex128::exp_i(b);
    Complex128 rhs = Complex128::exp_i(a + b);
    EXPECT_NEAR(lhs.real, rhs.real, 1e-14);
    EXPECT_NEAR(lhs.imag, rhs.imag, 1e-14);
}

// =============================================================================
// Constants
// =============================================================================

TEST(R1121Types, ConstantsAreConsistent) {
    EXPECT_NEAR(2.0 * INV_SQRT2 * INV_SQRT2, 1.0, 1e-15);
    // INV_SQRT2 is DERIVED as SQRT2 / 2.0. Division by a power of two is exact,
    // so this relationship is bit-exact and EXPECT_EQ is the honest assertion.
    //
    // The reference used here previously was `1.0 / std::sqrt(2.0)` with a 1e-16
    // tolerance, and both halves of that were wrong. The expression rounds twice
    // (sqrt, then divide) and lands one ULP BELOW correctly-rounded 1/√2
    // (0x3FE6A09E667F3BCC against 0x3FE6A09E667F3BCD). The tolerance is smaller
    // than one ULP at this magnitude (ULP ≈ 1.11e-16), so EXPECT_NEAR was
    // already an exact-equality check — against the wrong value. It passed only
    // while the library carried the same one-ULP-low literal the reference does.
    EXPECT_EQ(INV_SQRT2, SQRT2 / 2.0);
    // PI_2 and PI_4 are exact binary scalings of PI
    EXPECT_EQ(2.0 * PI_2, PI);
    EXPECT_EQ(4.0 * PI_4, PI);
    EXPECT_NEAR(PI, std::acos(-1.0), 1e-15);
    EXPECT_EQ(CACHE_LINE_SIZE, 64u);
}

// =============================================================================
// Aligned allocation helpers
// =============================================================================

TEST(R1121Types, AlignedAllocReturns64ByteAlignedPointers) {
    for (size_t count : {size_t(1), size_t(7), size_t(8), size_t(1000)}) {
        double* p = aligned_alloc_doubles(count);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % CACHE_LINE_SIZE, 0u)
            << "allocation of " << count << " doubles is not 64-byte aligned";
        // The memory is writable across the requested range.
        for (size_t i = 0; i < count; ++i) p[i] = static_cast<double>(i);
        for (size_t i = 0; i < count; ++i)
            EXPECT_EQ(p[i], static_cast<double>(i));
        aligned_free(p);
    }
}

TEST(R1121Types, AlignedAllocNonMultipleSizesRoundUp) {
    // Sizes that are not a multiple of the cache line must still succeed
    // (the helper rounds the byte count up to a 64-byte multiple).
    for (size_t count : {size_t(1), size_t(3), size_t(9), size_t(17)}) {
        double* p = aligned_alloc_doubles(count);
        ASSERT_NE(p, nullptr);
        p[count - 1] = 42.0;
        EXPECT_EQ(p[count - 1], 42.0);
        aligned_free(p);
    }
}

TEST(R1121Types, AlignedFreeNullptrIsSafe) {
    aligned_free(nullptr);  // must not crash (free semantics)
    SUCCEED();
}
