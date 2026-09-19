// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// =============================================================================
// autonne_backend - the SVDMethod::Jacobi and SVDMethod::BDC entry point
// =============================================================================
// Why this is its own translation unit is in
// include/lindblad/detail/autonne_backend.hpp.

#include "lindblad/detail/autonne_backend.hpp"

#include <stdexcept>

#include <autonne/autonne.hpp>

namespace lindblad {
namespace detail {

namespace {

// autonne's own MatrixOrder is a distinct type with the same meaning, so the
// two are mapped explicitly. Casting between them because the enumerators
// happen to line up would survive exactly until one of them gained a value.
autonne::MatrixOrder to_autonne_order(MatrixOrder order) {
    return order == MatrixOrder::RowMajor ? autonne::MatrixOrder::RowMajor
                                          : autonne::MatrixOrder::ColMajor;
}

}  // namespace

bool autonne_svd_thin(const std::complex<double>* data, int rows, int cols,
                      MatrixOrder order, SVDMethod method,
                      std::complex<double>* U_out, double* S_out,
                      std::complex<double>* V_out) {
    if (rows <= 0 || cols <= 0) return false;
    switch (method) {
        case SVDMethod::Jacobi:
            return autonne::svd_thin(data, rows, cols, to_autonne_order(order),
                                     U_out, S_out, V_out);
        case SVDMethod::BDC:
            return autonne::svd_thin_bdc(data, rows, cols, to_autonne_order(order),
                                         U_out, S_out, V_out);
        case SVDMethod::EigenJacobi:
        case SVDMethod::EigenBDC:
            break;
    }
    throw std::logic_error(
        "autonne_svd_thin: an Eigen SVDMethod reached the autonne adapter; "
        "detail::svd_thin routes by provider and should not have sent it here");
}

} // namespace detail
} // namespace lindblad
