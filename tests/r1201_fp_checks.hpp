#pragma once

// R.1.20.1 — shared assertion body for the non-finite sentinel suite.
//
// Header-only ON PURPOSE, mirroring the diag_r1160_matrices.hpp precedent: each
// including translation unit compiles these checks under ITS OWN floating-point
// flags. test_r1201_strict_fp.cpp includes them under the project-wide flags;
// test_r1201_strict_fp_nofast.cpp includes them under -fno-fast-math
// (per-source CMake override). Identical source, identical inputs, and the FP
// model is the only variable.
//
// EVERYTHING BELOW HAS INTERNAL LINKAGE, which is load-bearing rather than
// stylistic. An inline function with external linkage defined in both including
// TUs collapses to a single merged definition at link time, and which TU's
// codegen survives is the linker's choice; the two legs would then run the same
// machine code under two labels and the comparison would establish nothing.
//
// What the pair checks: is_finite_strict and quiet_nan_strict answer the same
// way under both floating-point models the tree is built with. A leg that
// diverges names exactly which property that model changed.

#include <gtest/gtest.h>

#include "lindblad/types.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace r1201_fp {
namespace {

// IEEE-754 binary64 layout, by bit pattern rather than by <limits>, because
// <limits> is one of the things under test here.
inline constexpr std::uint64_t kQuietNaNBits = 0x7FF8000000000000ULL;
inline constexpr std::uint64_t kPosInfBits   = 0x7FF0000000000000ULL;
inline constexpr std::uint64_t kNegInfBits   = 0xFFF0000000000000ULL;
inline constexpr std::uint64_t kExponentMask = 0x7FF0000000000000ULL;
inline constexpr std::uint64_t kQuietBit     = 0x0008000000000000ULL;
inline constexpr std::uint64_t kSignBit      = 0x8000000000000000ULL;

inline std::uint64_t bits_of(double x) {
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

inline double from_bits(std::uint64_t b) {
    double x;
    std::memcpy(&x, &b, sizeof x);
    return x;
}

// Forces the value across a call boundary the optimiser cannot see through, so
// the marker cannot stay in a register the compiler has already drawn
// conclusions about. noinline here rather than a definition in one of the two
// TUs, so each leg exercises a copy compiled under its own flags.
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
inline double round_trip(double x) {
    volatile double sink = x;
    return sink;
}

// =============================================================================
// The checks
// =============================================================================
//
// `leg` names the floating-point model so a failure says which one broke.
// EXPECT_*, never ASSERT_*: a failure on one leg must not stop the other
// properties from being reported, since the interesting output is WHICH
// properties survive each model.

inline void check_quiet_nan_strict_is_non_finite(const std::string& leg) {
    const double nan_marker = lindblad::quiet_nan_strict();

    EXPECT_EQ(bits_of(nan_marker), kQuietNaNBits)
        << leg << ": quiet_nan_strict() did not produce the canonical quiet "
                  "NaN bit pattern";
    EXPECT_EQ(bits_of(nan_marker) & kExponentMask, kExponentMask)
        << leg << ": exponent field is not all-ones, so this is not a NaN";
    EXPECT_NE(bits_of(nan_marker) & kQuietBit, 0ULL)
        << leg << ": the quiet bit is clear — this is a SIGNALLING NaN, which "
                  "may trap instead of propagating";
    EXPECT_EQ(bits_of(nan_marker) & kSignBit, 0ULL)
        << leg << ": sign bit set on the canonical marker";

    EXPECT_FALSE(lindblad::is_finite_strict(nan_marker))
        << leg << ": is_finite_strict called the NaN marker finite; every "
                  "guard built on this pair is inoperative";
}

inline void check_is_finite_strict_accepts_finite_values(
    const std::string& leg) {
    // Ordinary values, the extremes of the normal range, and a subnormal —
    // is_finite_strict tests only the exponent field, so a subnormal (exponent
    // field all zeroes) must read as finite.
    const double finite_cases[] = {
        0.0, -0.0, 1.0, -1.0, 0.5, -12345.678,
        1.7976931348623157e308,   // DBL_MAX
        2.2250738585072014e-308,  // DBL_MIN, smallest normal
        from_bits(1ULL),          // smallest positive subnormal
        from_bits(kSignBit | 1ULL),
    };

    for (double x : finite_cases) {
        // Bit pattern printed as an unsigned integer, not via std::hex: gtest's
        // Message takes std::ostream&(*)(std::ostream&) manipulators only, and
        // std::hex is std::ios_base&(*)(std::ios_base&), so it does not bind.
        EXPECT_TRUE(lindblad::is_finite_strict(x))
            << leg << ": is_finite_strict rejected the finite value " << x
            << " (bits " << bits_of(x) << ")";
    }
}

inline void check_is_finite_strict_rejects_infinities(const std::string& leg) {
    // Built from bit patterns, not from std::numeric_limits::infinity(), for
    // the same reason quiet_nan_strict exists: under -ffinite-math-only the
    // library constant need not be materialised, and a test whose input
    // quietly became finite would report a library defect that is not there.
    const double pos_inf = from_bits(kPosInfBits);
    const double neg_inf = from_bits(kNegInfBits);

    EXPECT_FALSE(lindblad::is_finite_strict(pos_inf))
        << leg << ": +infinity read as finite";
    EXPECT_FALSE(lindblad::is_finite_strict(neg_inf))
        << leg << ": -infinity read as finite";
}

inline void check_marker_survives_storage_and_calls(const std::string& leg) {
    // The failure mode this guards is not "NaN compares oddly" — it is the
    // compiler deciding the value cannot be non-finite and propagating that
    // conclusion through a store, a container, or a call boundary.
    const double marker = lindblad::quiet_nan_strict();

    double via_local = marker;
    EXPECT_FALSE(lindblad::is_finite_strict(via_local))
        << leg << ": marker became finite through a local copy";

    std::vector<double> via_container(4, marker);
    for (size_t i = 0; i < via_container.size(); ++i) {
        EXPECT_FALSE(lindblad::is_finite_strict(via_container[i]))
            << leg << ": marker became finite through a container at index "
            << i;
    }

    EXPECT_FALSE(lindblad::is_finite_strict(round_trip(marker)))
        << leg << ": marker became finite across a call boundary";

    struct Holder { double value; bool written; };
    Holder h{marker, false};
    EXPECT_FALSE(lindblad::is_finite_strict(h.value))
        << leg << ": marker became finite through aggregate storage";
    EXPECT_FALSE(h.written);
}

inline void check_validity_flag_pattern_ranks_correctly(
    const std::string& leg) {
    // The positive statement behind the #68 remedy: a best-so-far guarded by an
    // explicit bool selects the true minimum regardless of FP model, because a
    // bool carries no floating-point meaning for the optimiser to reason about.
    //
    // The rejected alternative is deliberately NOT asserted here. A +infinity
    // seed is PERMITTED to work — whether the first comparison actually gets
    // folded depends on compiler and version — so asserting that it fails would
    // pin compiler behaviour rather than library behaviour, and would break the
    // day the compiler changed its mind. What matters is that the replacement
    // is correct unconditionally, which is what this checks.
    const double costs[] = {3.5, -2.0, 7.25, -2.0, 0.0};

    double best = 0.0;
    int    best_index = -1;
    bool   have_best = false;
    for (int i = 0; i < 5; ++i) {
        if (!have_best || costs[i] < best) {
            best = costs[i];
            best_index = i;
            have_best = true;
        }
    }

    EXPECT_TRUE(have_best) << leg << ": ranking loop never wrote a result";
    EXPECT_EQ(best_index, 1) << leg << ": wrong minimum selected";
    EXPECT_EQ(best, -2.0) << leg << ": wrong minimum value";

    // The empty-input case, which is where a seed-based loop silently returns
    // its seed and a flag-based loop reports that it has no answer.
    bool have_any = false;
    for (int i = 0; i < 0; ++i) have_any = true;
    EXPECT_FALSE(have_any)
        << leg << ": an empty ranking loop must report no result, not a seed";
}

// Not an assertion — a RECORD. Whether std::isfinite still works under
// -ffinite-math-only is a compiler question, and pinning an answer either way
// would make this suite fail on a compiler change that broke nothing. Printing
// it puts the divergence between the two legs in the test output, which is the
// evidence that motivated the strict helpers in the first place.
inline void report_std_isfinite_behaviour(const std::string& leg) {
    const double marker = lindblad::quiet_nan_strict();
    const double pos_inf = from_bits(kPosInfBits);

    std::cout << "[fp-model] " << leg << ": std::isfinite(quiet NaN) = "
              << std::isfinite(marker) << ", std::isnan(quiet NaN) = "
              << std::isnan(marker) << ", std::isfinite(+inf) = "
              << std::isfinite(pos_inf)
              << "  |  is_finite_strict(quiet NaN) = "
              << lindblad::is_finite_strict(marker)
              << ", is_finite_strict(+inf) = "
              << lindblad::is_finite_strict(pos_inf) << "\n";
}

} // namespace
} // namespace r1201_fp
