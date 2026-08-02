#pragma once

#include <numbers>

namespace lindblad {

// =============================================================================
// Mathematical constants — the single source of truth for the whole library
// =============================================================================
//
// Every mathematical constant used anywhere in Lindblad is declared here, once.
// Nothing else in the codebase should spell a constant's VALUE as a numeric
// literal or a preprocessor macro. A short local alias for readability inside a
// dense matrix literal (`constexpr double inv_sqrt2 = INV_SQRT2;`) is fine: the
// value still has exactly one definition, which is the property that matters.
//
// Why this file exists
// -------------------
// The POSIX `M_*` macros (M_PI, M_SQRT1_2, ...) are NOT standard C++. They come
// from <cmath> only on platforms that choose to define them; Microsoft's
// standard library gates them behind _USE_MATH_DEFINES, so `M_PI` simply does
// not exist there. Because a macro is only visible to translation units that
// happen to include whatever defined it, the resulting breakage is per-file and
// silent — code compiles for years on one platform and fails on another with no
// warning in between. Named constants have none of that behaviour: they are
// ordinary declarations, subject to normal lookup, and either in scope or a
// compile error.
//
// Duplication is the second failure mode, and it is the one that actually bit.
// INV_SQRT2 was hand-typed as 0.7071067811865475 and copied to seven further
// sites; that literal is one unit in the last place BELOW correctly-rounded
// 1/sqrt(2) (0x3FE6A09E667F3BCC against the correct 0x3FE6A09E667F3BCD), while
// the tests spelled the same amplitude via the POSIX macro and therefore got
// the correct value. Library and tests disagreed by one ULP on the Hadamard
// amplitude, the most-used constant in the codebase. One definition, derived
// rather than typed, removes both the divergence and the class of bug.
//
// Values
// ------
// Everything below derives from <numbers>, which the standard requires to be
// the correctly-rounded value of each constant for the type. Do not replace
// these with hand-written decimal literals: a literal is only as good as the
// digits someone typed, and a wrong one is invisible at a glance. Note that
// writing MORE digits does not buy accuracy — `double` carries 53 mantissa bits
// (≈ 15.95 decimal digits), so 3.14159265358979323846 and a forty-digit π
// produce the same bit pattern. Extra digits only matter for wider types, where
// std::numbers::pi_v<T> is the correct tool.
//
// Divisions and multiplications by powers of two below (PI_2, PI_4, TWO_PI,
// INV_SQRT2) are exact in binary floating point, so those are correctly
// rounded too.
//
// `inline constexpr`, not plain `constexpr`: at namespace scope `constexpr`
// implies `const`, which implies internal linkage, so every translation unit
// including this header would get its own object at its own address. Harmless
// while a constant is only read by value, but ODR-using one (taking its
// address, or binding it to a reference) inside an inline function or template
// in a header would then see a different address per TU. `inline` gives a
// single shared entity and is the correct spelling for a header constant.

// --- π and friends ----------------------------------------------------------
inline constexpr double PI     = std::numbers::pi;      // π
inline constexpr double TWO_PI = 2.0 * PI;              // 2π — full turn
inline constexpr double PI_2   = PI / 2.0;              // π/2
inline constexpr double PI_4   = PI / 4.0;              // π/4
inline constexpr double INV_PI = std::numbers::inv_pi;  // 1/π

// 2/π and 2/√π — the remaining π-derived POSIX macros (M_2_PI, M_2_SQRTPI),
// present so there is never a reason to reach for the macro form.
inline constexpr double TWO_INV_PI     = 2.0 * std::numbers::inv_pi;
inline constexpr double TWO_INV_SQRTPI = 2.0 * std::numbers::inv_sqrtpi;

// --- Roots ------------------------------------------------------------------
inline constexpr double SQRT2 = std::numbers::sqrt2;  // √2
inline constexpr double SQRT3 = std::numbers::sqrt3;  // √3

// 1/√2 — the Hadamard amplitude, and by a wide margin the most-used constant in
// this library. Derived as SQRT2 / 2.0 rather than written out: division by a
// power of two is exact, so this is the correctly-rounded value by
// construction. (1.0 / SQRT2 is NOT equivalent — it lands one ULP low.)
inline constexpr double INV_SQRT2 = SQRT2 / 2.0;

// --- Exponentials and logarithms --------------------------------------------
inline constexpr double E      = std::numbers::e;       // e
inline constexpr double LN2    = std::numbers::ln2;     // ln 2
inline constexpr double LN10   = std::numbers::ln10;    // ln 10
inline constexpr double LOG2E  = std::numbers::log2e;   // log₂ e
inline constexpr double LOG10E = std::numbers::log10e;  // log₁₀ e

} // namespace lindblad
