#pragma once

// =============================================================================
// autonne_backend - the SVDMethod::AutonneJacobi entry point
// =============================================================================
//
// Declared unconditionally and defined in its own translation unit, which is in
// the build whether or not autonne is linked. Two consequences, both deliberate.
//
// The enum value exists in every build, so a program compiles the same way
// against a library configured with the backend and one without, and the
// difference surfaces where the kernel is actually requested rather than as a
// missing symbol at link time or a missing enumerator at compile time.
//
// The translation unit is its own, not a branch inside eigen_backend.cpp. One
// decomposition per translation unit is what lets a floating-point flag mean
// anything: a file built under -ffast-math cannot host a routine whose contract
// depends on IEEE semantics, and the two kernels do not have to agree about
// which flags they were built with.
//
// The signature mirrors detail::svd_thin rather than autonne's own, because
// autonne's svd_thin deliberately takes no method parameter. Translating
// between the two is this adapter's job, and doing it here keeps the
// discrepancy in one place instead of at every call site.

// eigen_backend.hpp owns MatrixOrder, and this adapter takes it rather than
// declaring a parallel one: two orders that mean the same thing is one more
// place for a transpose to hide.
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/types.hpp"

#include <complex>

namespace lindblad {
namespace detail {

// Thin SVD through autonne. Returns false when the factorisation fails, in the
// same way detail::svd_thin does, so the SELECT rung treats both alike.
//
// Throws std::runtime_error when the build did not link autonne. That is a
// caller error rather than a factorisation failure: returning false would send
// it down the Gram rescue path and produce a valid answer from a kernel the
// caller did not ask for.
bool autonne_svd_thin(const std::complex<double>* data, int rows, int cols,
                      MatrixOrder order,
                      std::complex<double>* U_out, double* S_out,
                      std::complex<double>* V_out);

} // namespace detail
} // namespace lindblad
