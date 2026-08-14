// R.1.20.1 test suite — include/lindblad/constants.hpp.
//
// R.1.20.0 gave the library one home for every mathematical constant and, in
// doing so, CORRECTED the most-used value in the codebase: INV_SQRT2 had been
// hand-typed as 0.7071067811865475 — one unit in the last place below
// correctly-rounded 1/√2 — and copied to seven further sites, while the tests
// spelled the same amplitude correctly. Library and tests therefore disagreed
// by one ULP on the Hadamard amplitude, and nothing anywhere said so.
//
// This suite is the assertion form of that fix. It pins four separate
// properties, because the defect needed all four to go unnoticed:
//
//   VALUE     every constant is the correctly-rounded double for its
//             mathematical value, checked bit-for-bit against <numbers> and,
//             for the derived ones, against an exact power-of-two relationship.
//   IDENTITY  the specific wrong value is named and excluded by bit pattern, so
//             a regression to it fails HERE, loudly, instead of shifting every
//             amplitude in the library by an invisible ULP.
//   LINKAGE   `inline constexpr` really does give one shared object, which is
//             the property the header's comment claims and the reason the
//             spelling was chosen. Asserted across two translation units.
//   REACH     the library's own gates produce that same amplitude, so "library
//             and tests agree" is verified end to end rather than assumed.
//
// Deliberate convention in this file: no hand-typed decimal literal is ever
// used as a reference. References are either <numbers>, an exact arithmetic
// derivation, or an explicit integer BIT PATTERN — the last written as an
// integer precisely so it cannot be misread as a computed quantity.

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>

using namespace lindblad;

namespace {

// Bit-level identity. Two doubles are the same value iff their bit patterns
// match; == would also call +0.0 and -0.0 equal, and would tell us nothing
// about which of two adjacent representable values we actually hold.
std::uint64_t bits_of(double x) { return std::bit_cast<std::uint64_t>(x); }

// The two values at the heart of #70, named by bit pattern.
//
// Correctly-rounded 1/√2 and the value that sat in the library instead. They
// are adjacent representable doubles: for positive normals the encoding is
// monotonic, so "one ULP low" is literally "one less as an integer".
constexpr std::uint64_t kInvSqrt2Correct  = 0x3FE6A09E667F3BCDULL;
constexpr std::uint64_t kInvSqrt2OneUlpLo = 0x3FE6A09E667F3BCCULL;

static_assert(kInvSqrt2Correct == kInvSqrt2OneUlpLo + 1,
              "the two values must be adjacent doubles, or the premise of "
              "this file's ULP reasoning is wrong");

// Is `d` the nearest double to the long-double value `wide`? Only meaningful
// where long double is actually wider than double (x86-64: 80-bit; on targets
// where the two coincide the check degenerates to a tautology, so callers
// guard on kLongDoubleIsWider).
constexpr bool kLongDoubleIsWider =
    std::numeric_limits<long double>::digits >
    std::numeric_limits<double>::digits;

bool is_nearest_double(double d, long double wide) {
    const long double err = std::abs(static_cast<long double>(d) - wide);
    const double below = std::nextafter(d, -std::numeric_limits<double>::max());
    const double above = std::nextafter(d, std::numeric_limits<double>::max());
    return err <= std::abs(static_cast<long double>(below) - wide) &&
           err <= std::abs(static_cast<long double>(above) - wide);
}

} // namespace

// =============================================================================
// VALUE — sourced from <numbers>, bit for bit
// =============================================================================

// The standard requires std::numbers::X to BE the correctly-rounded value for
// the type, so equality here is not a numerical approximation but a statement
// that the library did not paraphrase its source.
TEST(R1201Constants, SourcedFromNumbersBitExact) {
    EXPECT_EQ(bits_of(PI),     bits_of(std::numbers::pi));
    EXPECT_EQ(bits_of(INV_PI), bits_of(std::numbers::inv_pi));
    EXPECT_EQ(bits_of(SQRT2),  bits_of(std::numbers::sqrt2));
    EXPECT_EQ(bits_of(SQRT3),  bits_of(std::numbers::sqrt3));
    EXPECT_EQ(bits_of(E),      bits_of(std::numbers::e));
    EXPECT_EQ(bits_of(LN2),    bits_of(std::numbers::ln2));
    EXPECT_EQ(bits_of(LN10),   bits_of(std::numbers::ln10));
    EXPECT_EQ(bits_of(LOG2E),  bits_of(std::numbers::log2e));
    EXPECT_EQ(bits_of(LOG10E), bits_of(std::numbers::log10e));
}

// Multiplying or dividing by a power of two only changes the exponent field, so
// every one of these is exact and the derived constant inherits its source's
// correct rounding. That is the whole reason the header derives rather than
// types them.
TEST(R1201Constants, PowerOfTwoDerivationsAreExact) {
    EXPECT_EQ(bits_of(TWO_PI),    bits_of(2.0 * PI));
    EXPECT_EQ(bits_of(PI_2),      bits_of(PI / 2.0));
    EXPECT_EQ(bits_of(PI_4),      bits_of(PI / 4.0));
    EXPECT_EQ(bits_of(INV_SQRT2), bits_of(SQRT2 / 2.0));

    // Exactness means the derivations invert with no residue at all.
    EXPECT_EQ(bits_of(TWO_PI / 2.0), bits_of(PI));
    EXPECT_EQ(bits_of(PI_2 * 2.0),   bits_of(PI));
    EXPECT_EQ(bits_of(PI_4 * 4.0),   bits_of(PI));
    EXPECT_EQ(bits_of(PI_2 / 2.0),   bits_of(PI_4));
}

// The π-derived inverses exist so there is never a reason to reach for the
// POSIX macros M_2_PI / M_2_SQRTPI, which are not standard C++ and simply do
// not exist on Microsoft's standard library without _USE_MATH_DEFINES.
TEST(R1201Constants, PiDerivedInversesMatchTheirSources) {
    EXPECT_EQ(bits_of(TWO_INV_PI), bits_of(2.0 * std::numbers::inv_pi));
    EXPECT_EQ(bits_of(TWO_INV_SQRTPI),
              bits_of(2.0 * std::numbers::inv_sqrtpi));
}

// =============================================================================
// IDENTITY — the #70 regression pin
// =============================================================================

// This is the test that would have caught the original defect. It does not ask
// whether INV_SQRT2 is "close to" 1/√2 — every wrong value in this story was
// close to 1/√2 — it asks which of two adjacent doubles the library holds.
TEST(R1201Constants, InvSqrt2IsCorrectlyRoundedNotOneUlpLow) {
    EXPECT_EQ(bits_of(INV_SQRT2), kInvSqrt2Correct)
        << "INV_SQRT2 is not the correctly-rounded value";
    EXPECT_NE(bits_of(INV_SQRT2), kInvSqrt2OneUlpLo)
        << "INV_SQRT2 has regressed to the hand-typed literal that #70 removed";

    // Stated the other way round: the old value is exactly one ULP below, and
    // stepping up from it lands on the constant the library now carries.
    const double one_ulp_low = std::bit_cast<double>(kInvSqrt2OneUlpLo);
    EXPECT_LT(one_ulp_low, INV_SQRT2);
    EXPECT_EQ(bits_of(std::nextafter(one_ulp_low, 1.0)), bits_of(INV_SQRT2));
}

// The trap the header warns about, asserted rather than left as a comment:
// 1.0 / SQRT2 rounds twice (the quotient is not exactly representable) and
// lands on the WRONG one of the two adjacent doubles. SQRT2 / 2.0 does not,
// because division by a power of two is exact. Both spellings look equally
// reasonable in review, which is precisely why this needs a test.
TEST(R1201Constants, ReciprocalSpellingIsTheOneThatRoundsWrong) {
    EXPECT_NE(bits_of(1.0 / SQRT2), bits_of(INV_SQRT2))
        << "if these now agree, the platform's division changed and the "
           "header's rationale needs rechecking";
    EXPECT_EQ(bits_of(1.0 / SQRT2), kInvSqrt2OneUlpLo);

    // std::sqrt is a further independent route to the same wrong value, and it
    // is the one the test suite itself used as a reference before this release.
    EXPECT_EQ(bits_of(1.0 / std::sqrt(2.0)), kInvSqrt2OneUlpLo);
}

// =============================================================================
// VALUE — independent confirmation at higher precision
// =============================================================================

// <numbers> agreeing with itself proves only that the library copied its source
// faithfully. This checks the source: at 80-bit precision, is the double the
// library holds the NEAREST double to the true constant? Skipped where long
// double buys no extra bits, since the comparison would then be circular.
TEST(R1201Constants, ConstantsAreNearestDoubleAtHigherPrecision) {
    if constexpr (!kLongDoubleIsWider) {
        GTEST_SKIP() << "long double is no wider than double on this target; "
                        "an independent-precision check is not available";
    } else {
        EXPECT_TRUE(is_nearest_double(PI,     std::numbers::pi_v<long double>));
        EXPECT_TRUE(is_nearest_double(E,      std::numbers::e_v<long double>));
        EXPECT_TRUE(is_nearest_double(SQRT2,  std::numbers::sqrt2_v<long double>));
        EXPECT_TRUE(is_nearest_double(SQRT3,  std::numbers::sqrt3_v<long double>));
        EXPECT_TRUE(is_nearest_double(LN2,    std::numbers::ln2_v<long double>));
        EXPECT_TRUE(is_nearest_double(LN10,   std::numbers::ln10_v<long double>));
        EXPECT_TRUE(is_nearest_double(INV_PI, std::numbers::inv_pi_v<long double>));

        // And the derived one, against the true 1/√2 rather than against √2.
        EXPECT_TRUE(is_nearest_double(
            INV_SQRT2, 1.0L / std::numbers::sqrt2_v<long double>));
    }
}

// =============================================================================
// Usability as constant expressions
// =============================================================================

// `inline constexpr` has to survive use in a constant expression — array
// bounds, template arguments, other constexpr initialisers. A static_assert
// that compiles IS the test; these fire at build time, not at run time.
TEST(R1201Constants, UsableInConstantExpressions) {
    static_assert(PI > 3.0 && PI < 4.0);
    static_assert(PI_2 * 2.0 == PI);
    static_assert(PI_4 * 4.0 == PI);
    static_assert(TWO_PI == 2.0 * PI);
    static_assert(INV_SQRT2 == SQRT2 / 2.0);
    static_assert(INV_SQRT2 > 0.0 && INV_SQRT2 < 1.0);
    static_assert(E > 2.0 && E < 3.0);

    constexpr double derived_at_compile_time = INV_SQRT2 * INV_SQRT2 * 2.0;
    static_assert(derived_at_compile_time > 0.99 &&
                  derived_at_compile_time < 1.01);

    // Present so the case is not reported as empty; the assertions above have
    // already been evaluated by the compiler.
    SUCCEED();
}

// =============================================================================
// Algebraic sanity
// =============================================================================

// Bit-exactness pins WHICH double each constant is; these pin that the value is
// the right constant at all. A transposed digit that stayed correctly rounded
// for some other number would pass every check above and fail here.
//
// Tolerance is expressed in ULPs OF THE EXPECTED VALUE, never as an absolute
// epsilon. An absolute figure is a different number of ULPs at every magnitude:
// 1e-15 is about 4.5 ULP at 1.0 but only 0.56 ULP at 10.0 — BELOW the spacing
// between representable doubles, which makes it an exact-equality demand
// wearing a tolerance's clothes. That is precisely the defect this release
// removes from R1121Types.ConstantsAreConsistent, and it would be poor form to
// reintroduce it in the suite written to cover the fix.
//
// The budget is deliberately generous. These identities exist to catch "this is
// the WRONG constant", and a wrong constant is off by orders of magnitude, not
// by a few ULP. Being loose here costs nothing and buys immunity to how any
// given libm rounds a transcendental — especially under -ffast-math, where the
// compiler may substitute a lower-accuracy implementation outright.
namespace {

constexpr int kUlpBudget = 16;

::testing::AssertionResult CloseInUlps(const char* what, double actual,
                                       double expected) {
    const double magnitude = std::abs(expected);
    const double spacing =
        std::nextafter(magnitude, std::numeric_limits<double>::max()) -
        magnitude;
    const double tolerance = kUlpBudget * spacing;
    const double error = std::abs(actual - expected);

    if (error <= tolerance) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << what << ": got " << actual << ", expected " << expected
           << ", off by " << error << " = " << (error / spacing)
           << " ULP (budget " << kUlpBudget << " ULP = " << tolerance << ")";
}

} // namespace

#define EXPECT_CLOSE_IN_ULPS(actual, expected) \
    EXPECT_TRUE(CloseInUlps(#actual, (actual), (expected)))

TEST(R1201Constants, AlgebraicIdentitiesHold) {
    EXPECT_CLOSE_IN_ULPS(2.0 * INV_SQRT2 * INV_SQRT2, 1.0);
    EXPECT_CLOSE_IN_ULPS(SQRT2 * SQRT2, 2.0);
    EXPECT_CLOSE_IN_ULPS(SQRT3 * SQRT3, 3.0);
    EXPECT_CLOSE_IN_ULPS(PI * INV_PI, 1.0);
    EXPECT_CLOSE_IN_ULPS(std::log(E), 1.0);
    EXPECT_CLOSE_IN_ULPS(std::exp(LN2), 2.0);
    EXPECT_CLOSE_IN_ULPS(std::exp(LN10), 10.0);
    EXPECT_CLOSE_IN_ULPS(LOG2E * LN2, 1.0);
    EXPECT_CLOSE_IN_ULPS(LOG10E * LN10, 1.0);
    EXPECT_CLOSE_IN_ULPS(std::cos(PI), -1.0);
    EXPECT_CLOSE_IN_ULPS(std::sin(PI_2), 1.0);
    EXPECT_CLOSE_IN_ULPS(std::sin(PI_4), INV_SQRT2);
    EXPECT_CLOSE_IN_ULPS(std::cos(PI_4), INV_SQRT2);
    EXPECT_CLOSE_IN_ULPS(TWO_INV_PI * PI, 2.0);
}

// =============================================================================
// REACH — the library's gates carry the corrected amplitude
// =============================================================================

// The end-to-end form of #70. A shared constant that the kernels do not
// actually use would satisfy every test above while leaving the divergence in
// place, so this asks the library for the amplitude directly.
//
// H|0> = (|0> + |1>)/√2, and the kernel computes INV_SQRT2 * (r0 + r1) with
// r0 = 1, r1 = 0. Multiplication by exactly 1.0 is exact, so the amplitude that
// comes back must be the constant itself, bit for bit — no tolerance needed
// and none wanted, since a one-ULP tolerance is what hid the original defect.
TEST(R1201Constants, HadamardAmplitudeIsTheSharedConstant) {
    Statevector sv(1);
    sv.initialize_basis(0);
    gates::apply_h(sv, 0);

    const auto amp = sv.amplitudes();
    ASSERT_EQ(amp.size(), 2u);

    EXPECT_EQ(bits_of(amp[0].real), bits_of(INV_SQRT2))
        << "H|0> amplitude differs from the shared constant; the gate kernel "
           "is not sourcing the value from constants.hpp";
    EXPECT_EQ(bits_of(amp[1].real), bits_of(INV_SQRT2));
    EXPECT_EQ(amp[0].imag, 0.0);
    EXPECT_EQ(amp[1].imag, 0.0);

    // Specifically not the value #70 removed.
    EXPECT_NE(bits_of(amp[0].real), kInvSqrt2OneUlpLo);
}

// H|1> = (|0> - |1>)/√2 — the negative branch of the same kernel expression,
// so the sign path is pinned too rather than assumed symmetric.
TEST(R1201Constants, HadamardNegativeBranchIsTheSharedConstant) {
    Statevector sv(1);
    sv.initialize_basis(1);
    gates::apply_h(sv, 0);

    const auto amp = sv.amplitudes();
    ASSERT_EQ(amp.size(), 2u);
    EXPECT_EQ(bits_of(amp[0].real), bits_of(INV_SQRT2));
    EXPECT_EQ(bits_of(amp[1].real), bits_of(-INV_SQRT2));
}

// =============================================================================
// LINKAGE — one shared object, not one per translation unit
// =============================================================================

// Declared in test_r1201_constants_tu2.cpp. A SECOND translation unit is the
// entire point: the property under test cannot be observed from inside one.
namespace r1201_tu2 {
const double* pi_address();
const double* inv_sqrt2_address();
double        pi_value();
double        inv_sqrt2_value();
} // namespace r1201_tu2

// At namespace scope `constexpr` implies `const`, which implies INTERNAL
// linkage — so a header constant without `inline` gives every including
// translation unit its own object at its own address. That is invisible while
// the constant is only read by value, and becomes an ODR violation the moment
// an inline function or template in a header binds a reference to it or takes
// its address: each TU then sees a different address for the same name.
//
// `inline constexpr` is the spelling that gives one shared entity, and this is
// the assertion that it is doing so.
TEST(R1201Constants, InlineConstexprGivesOneObjectAcrossTranslationUnits) {
    EXPECT_EQ(&PI, r1201_tu2::pi_address())
        << "PI has a different address in another TU: the constants are not "
           "inline, and any header that ODR-uses one is ill-formed";
    EXPECT_EQ(&INV_SQRT2, r1201_tu2::inv_sqrt2_address());

    // Values agree too, which is the weaker property that held even before.
    EXPECT_EQ(bits_of(r1201_tu2::pi_value()), bits_of(PI));
    EXPECT_EQ(bits_of(r1201_tu2::inv_sqrt2_value()), bits_of(INV_SQRT2));
}
