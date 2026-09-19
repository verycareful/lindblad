// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once
// =============================================================================
// autonne_backend - the SVDMethod::Jacobi and SVDMethod::BDC entry point
// =============================================================================
//
// autonne (shek014/autonne) supplies the two default kernels of the MPS bond
// split: one-sided Jacobi (SVDMethod::Jacobi) and Householder bidiagonalisation
// with Gu-Eisenstat divide and conquer (SVDMethod::BDC). This adapter is their
// only entry point in the library.
//
// It is its own translation unit, not a branch inside eigen_backend.cpp. One
// decomposition provider per translation unit is what lets a floating-point
// flag mean anything: eigen_backend.cpp is compiled strict because Eigen's
// entry guards die under -ffast-math, while autonne's kernels are verified
// under both models by their own suite and take the project flags as they are.
// The two providers do not have to agree about how they were built.
//
// The signature mirrors detail::svd_thin, method parameter included, because
// autonne exposes one algorithm per entry point (svd_thin, svd_thin_bdc) rather
// than a switch. Translating the enum into the entry point is this adapter's
// job, and doing it here keeps the discrepancy in one place instead of at
// every call site. eigen_backend.hpp owns MatrixOrder, and this adapter takes
// it rather than declaring a parallel one: two orders that mean the same thing
// is one more place for a transpose to hide.

#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/types.hpp"

#include <complex>

namespace lindblad {
namespace detail {

// Thin SVD through autonne. `method` must be SVDMethod::Jacobi or
// SVDMethod::BDC; an Eigen method here is a routing error in the caller and
// throws std::logic_error rather than silently running a kernel the caller did
// not name. Returns false when the factorisation fails, in the same way
// detail::svd_thin does, so the SELECT rung treats both providers alike.
bool autonne_svd_thin(const std::complex<double>* data, int rows, int cols,
                      MatrixOrder order, SVDMethod method,
                      std::complex<double>* U_out, double* S_out,
                      std::complex<double>* V_out);

} // namespace detail
} // namespace lindblad
