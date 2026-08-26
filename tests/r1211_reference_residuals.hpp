// Independent reference residuals for the R1211 validation suites.
//
// DECLARED here and DEFINED in r1211_reference_residuals_strictfp.cpp, which is
// the whole point of the arrangement: that translation unit is compiled with
// -fno-fast-math, so the reference is evaluated under IEEE semantics while the
// rest of the suite keeps the project-wide flags.
//
// A reference exists to be an oracle. Compiled under -ffast-math it is not one:
// the compiler may reassociate and contract it freely, and at -march=x86-64-v3
// Clang 18 transformed this computation into one returning a value that no
// faithful evaluation order produces (1.2386923780787705e-16 on a Hadamard,
// where every ordering gives either 2.2204460492503131e-16 or exactly zero).
// Comparing the library against a number like that tests nothing.
//
// Only the reference is quarantined, not the tests that call it. Putting the
// flag on the whole test file would also apply it to the library's own
// header-inlined residuals, and inline functions are merged across the final
// link, so the strict copy could win and silently change which machine code the
// other suites in this binary exercise. The declaration below keeps that
// blast radius to one function.
//
// Mirrors the precedent set by test_diag_r1160_strictfp.cpp and
// test_r1201_strict_fp_nofast.cpp.
#pragma once

#include "lindblad/types.hpp"

#include <cstddef>
#include <vector>

namespace r1211ref {

// max |(U†U - I)_ij| over EVERY entry, against the library's upper-triangle
// walk. The library's shortcut is sound because U†U is Hermitian, so comparing
// the two is a direct test of that argument rather than of the arithmetic they
// share.
//
// The running maximum here is NaN-sticky: once a NaN is seen it is never
// displaced by a later finite value. Distinguishing this from the library's
// behaviour is the subject of the R1211NonFinite suite.
double reference_unitarity_deviation(const std::vector<lindblad::Complex128>& U,
                                     std::size_t rows);

} // namespace r1211ref
