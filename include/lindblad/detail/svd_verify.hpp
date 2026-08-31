#pragma once

#include "lindblad/detail/eigen_backend.hpp"  // MatrixOrder

#include <complex>

// =============================================================================
// detail::svd_reconstruction_residual_sq - the rung the ladder is judged on
// =============================================================================
// The truncation ladder accepts a factorisation by rebuilding the block from
// the kept slice and comparing against the input. That comparison is what makes
// a finite-but-wrong factorisation detectable: it does not inspect the spectrum
// or look for non-finite entries, it checks that the factors multiply back to
// the matrix they came from.
//
// It lives in its own translation unit, compiled -fno-fast-math, because
// ‖M - U S V†‖_F² subtracts two nearly identical matrices. That is a
// near-cancellation, the shape a permissive floating-point model perturbs
// hardest, and it fails in the dangerous direction: a residual computed too
// small admits a factorisation the ladder exists to reject. Every other rung
// can be released to the project-wide flags; this one guards the rest, so it
// cannot be judged by arithmetic it is meant to be checking.
//
// It is deliberately NOT part of the Eigen backend. That file's job is owning
// the Eigen dependency, and this residual is not Eigen's; filing it there
// because the two want the same compile flag would organise by flag rather
// than by purpose.

namespace lindblad {
namespace detail {

// ‖M - U diag(S) V†‖_F², where M is rows x cols in `order`, U is rows x k and
// V is cols x k both COLUMN-MAJOR, and S holds k singular values.
//
// Returns the squared Frobenius norm. A non-finite factorisation propagates
// into a non-finite result rather than being reported separately, so the caller
// makes one finiteness decision on the value it is about to compare.
double svd_reconstruction_residual_sq(const std::complex<double>* mat, int rows,
                                      int cols, MatrixOrder order,
                                      const std::complex<double>* U,
                                      const double* S,
                                      const std::complex<double>* V, int k);

}  // namespace detail
}  // namespace lindblad
