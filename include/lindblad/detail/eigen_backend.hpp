#pragma once

#include "lindblad/types.hpp"  // SVDMethod

#include <complex>

// =============================================================================
// detail::eigen_backend - the library's only Eigen decomposition site
// =============================================================================
// Eigen is header-only, so every translation unit that names a decomposition
// instantiates it into itself. Those instantiations carry vague linkage: the
// compiler emits each as a weak symbol in its own COMDAT group and the linker
// keeps exactly ONE definition per symbol for the whole binary, chosen by
// mangled name. Compile flags are not part of that name and are gone by link
// time, so the surviving copy may come from a translation unit built under a
// different floating-point model than the one calling it.
//
// That matters because -ffast-math breaks Eigen outright. JacobiSVD guards its
// entry with `if (!(numext::isfinite)(scale)) { m_info = InvalidInput; ... }`,
// and under -ffast-math the compiler may assume no infinity or NaN exists, so
// the predicate folds to a constant and the guard becomes dead code. Clang
// reports it at that line as undefined behaviour. A per-file -fno-fast-math
// cannot repair this on its own: it governs which variant a translation unit
// EMITS, never which variant survives the link.
//
// Routing every decomposition through this one strict translation unit leaves a
// single emitter, so there is nothing to merge and the flag means what it says.
//
// The interface passes raw buffers and names no Eigen type, so including this
// header instantiates nothing. Callers keep whatever matrix type they already
// hold; only the factorisation crosses.

namespace lindblad {
namespace detail {

// -----------------------------------------------------------------------------
// MatrixOrder - how the caller's buffer is laid out in memory
// -----------------------------------------------------------------------------
// Both orders are accepted so no caller has to transpose or copy before handing
// a block over: the qubit layer builds row-major blocks, the qudit layer holds
// column-major Eigen matrices, and each is mapped in place.
//
// It is a parameter rather than an assumption because getting it wrong is
// silent. A Hermitian matrix read in the opposite order is its own conjugate,
// so the decomposition succeeds and returns a wrong answer rather than failing.

enum class MatrixOrder { RowMajor, ColMajor };

// -----------------------------------------------------------------------------
// svd_thin
// -----------------------------------------------------------------------------
// Thin SVD of a rows x cols complex matrix. With k = min(rows, cols):
//
//   U_out  rows x k, column-major
//   S_out  k singular values, descending
//   V_out  cols x k, column-major, as V rather than V-dagger
//
// Every output buffer must hold its full size before the call. Returns false
// when the backend reports failure, in which case the outputs are unspecified
// and the caller is expected to take its fallback route rather than read them.
bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order, SVDMethod method,
              std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out);

// -----------------------------------------------------------------------------
// eigh
// -----------------------------------------------------------------------------
// Self-adjoint eigendecomposition of an n x n Hermitian matrix.
//
//   evals_out  n eigenvalues, ASCENDING
//   evecs_out  n x n eigenvectors, column-major, column i paired with
//              evals_out[i]. May be null when only the spectrum is wanted.
//
// Ascending order is the solver's own and is preserved rather than normalised,
// so a caller that wants sigmas descending reverses explicitly and the reversal
// is visible where it happens.
bool eigh(const std::complex<double>* data, int n, MatrixOrder order,
          double* evals_out, std::complex<double>* evecs_out);

}  // namespace detail
}  // namespace lindblad
