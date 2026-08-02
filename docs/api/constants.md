# Mathematical Constants

Every mathematical constant used anywhere in Lindblad is declared once, in one
header. Nothing else in the codebase spells a constant's value as a numeric
literal or a preprocessor macro.

## Header

```cpp
#include "lindblad/constants.hpp"
```

`lindblad/types.hpp` includes this header, so any translation unit that already
uses `Complex128` or `is_finite_strict` has the constants in scope without
naming the file.

## Namespace

`lindblad`

## Constants

All are `inline constexpr double`.

- `PI` — π
- `TWO_PI` — 2π, a full turn
- `PI_2` — π/2
- `PI_4` — π/4
- `INV_PI` — 1/π
- `TWO_INV_PI` — 2/π
- `TWO_INV_SQRTPI` — 2/√π
- `SQRT2` — √2
- `SQRT3` — √3
- `INV_SQRT2` — 1/√2, the Hadamard amplitude
- `E` — e
- `LN2` — ln 2
- `LN10` — ln 10
- `LOG2E` — log₂ e
- `LOG10E` — log₁₀ e

## Why a shared header

The POSIX `M_*` macros (`M_PI`, `M_SQRT1_2`, and the rest) are not standard C++.
They come from `<cmath>` only on platforms that choose to define them, and
Microsoft's standard library gates them behind `_USE_MATH_DEFINES`, so `M_PI`
simply does not exist there. A macro is visible only to translation units that
include whatever defined it, so the resulting breakage is per-file and silent:
code compiles for years on one platform and fails on another with no warning in
between. Named constants have none of that behaviour. They are ordinary
declarations subject to normal lookup, so a missing one is a compile error.

Duplication is the second failure mode. A constant written out by hand in eight
places is eight chances to type it differently, and a wrong digit deep in the
mantissa is invisible at a glance. Deriving one definition from `<numbers>`
removes both problems at once.

## Values

Every value derives from `<numbers>`, which the standard requires to be the
correctly-rounded value of the constant for the type. Do not replace these with
hand-written decimal literals: a literal is only as good as the digits someone
typed.

Writing more digits does not buy accuracy. `double` carries 53 mantissa bits,
about 15.95 decimal digits, so `3.14159265358979323846` and a forty-digit π
produce the same bit pattern. Extra digits matter only for wider types, where
`std::numbers::pi_v<T>` is the correct tool.

The derived values divide or multiply by powers of two (`PI_2`, `PI_4`,
`TWO_PI`, `INV_SQRT2`), which is exact in binary floating point, so those are
correctly rounded too. `INV_SQRT2` is defined as `SQRT2 / 2.0` rather than
`1.0 / SQRT2`; the two are not equivalent, and the reciprocal form lands one
unit in the last place low.

## Linkage

The constants are `inline constexpr`, not plain `constexpr`. At namespace scope
`constexpr` implies `const`, which implies internal linkage, so every
translation unit including the header would otherwise get its own object at its
own address. That is harmless while a constant is only read by value, but
ODR-using one (taking its address, or binding it to a reference) inside an
inline function or template in a header would then see a different address in
each translation unit. `inline` gives a single shared entity and is the correct
spelling for a header constant.

## Local aliases

A short local alias is fine where it reads better than the shared name, which in
practice means inside a dense matrix literal:

```cpp
constexpr double inv_sqrt2 = INV_SQRT2;
U[0] = U[1] = U[2] = Complex128(inv_sqrt2, 0);
U[3] = Complex128(-inv_sqrt2, 0);
```

The value still has exactly one definition, which is the property that matters.
What is not fine is writing the digits again.

## Related pages

- [docs/api/simulators.md](simulators.md)
- [docs/api/gates.md](gates.md)
