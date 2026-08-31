// svd_verify.cpp - the truncation ladder's reconstruction residual
//
// Compiled -fno-fast-math (see the top-level CMakeLists.txt). Why this one
// expression is quarantined while the rest of the ladder is not is set out in
// detail/svd_verify.hpp.

#include "lindblad/detail/svd_verify.hpp"

#include <Eigen/Dense>

namespace lindblad {
namespace detail {

namespace {

using ColMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::ColMajor>;
using RowMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::RowMajor>;

// The reconstruction is materialised into the INPUT's storage order before
// subtracting. Eigen's coefficient-wise binary operations require both operands
// to agree on row- versus column-major, and the kept slice is always built
// column-major while a caller's block may be either.
template <typename PlainT>
double residual_sq(const std::complex<double>* mat, int rows, int cols,
                   const std::complex<double>* U, const double* S,
                   const std::complex<double>* V, int k) {
    const Eigen::Map<const PlainT> M(mat, rows, cols);
    const Eigen::Map<const ColMajorC> U_k(U, rows, k);
    const Eigen::Map<const ColMajorC> V_k(V, cols, k);
    const Eigen::Map<const Eigen::VectorXd> S_k(S, k);

    const PlainT recon = U_k * S_k.asDiagonal() * V_k.adjoint();
    return (M - recon).squaredNorm();
}

}  // namespace

double svd_reconstruction_residual_sq(const std::complex<double>* mat, int rows,
                                      int cols, MatrixOrder order,
                                      const std::complex<double>* U,
                                      const double* S,
                                      const std::complex<double>* V, int k) {
    return (order == MatrixOrder::RowMajor)
               ? residual_sq<RowMajorC>(mat, rows, cols, U, S, V, k)
               : residual_sq<ColMajorC>(mat, rows, cols, U, S, V, k);
}

}  // namespace detail
}  // namespace lindblad
