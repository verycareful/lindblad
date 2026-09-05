// eigen_backend.cpp - every Eigen decomposition in the library, in one TU
//
// Compiled -fno-fast-math (see the top-level CMakeLists.txt). The reason this
// file exists at all, and why the flag on it is load-bearing where a flag on a
// calling TU is not, is set out in detail/eigen_backend.hpp.
//
// Nothing here decides anything. Selection, truncation, verification and
// rescue routing all live with their callers; this file computes a
// factorisation and reports whether the backend succeeded. Keeping it that
// narrow is what lets the whole library's Eigen exposure be audited by reading
// one file.

#include "lindblad/detail/eigen_backend.hpp"

// For the AutonneJacobi route only. The definition lives in its own
// translation unit, so nothing of autonne's is compiled into this one.
#include "lindblad/detail/autonne_backend.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>

namespace lindblad {
namespace detail {

namespace {

using ColMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::ColMajor>;
using RowMajorC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::RowMajor>;

// The factorisation itself, templated on the INPUT's storage order so the
// caller's buffer is mapped in place rather than transposed into a temporary.
// Outputs are written column-major whichever way the input was stored, because
// a factor is a new object and giving it one fixed layout removes a question
// from every call site.
template <typename PlainT>
bool svd_run(const std::complex<double>* data, int rows, int cols,
             SVDMethod method, std::complex<double>* U_out, double* S_out,
             std::complex<double>* V_out) {
    const Eigen::Map<const PlainT> mat(data, rows, cols);
    const int k = std::min(rows, cols);

    // The two backends are separate instantiations rather than one call behind
    // a pointer: they are different types, and dispatching on the method here
    // keeps that the only place either name appears.
    if (method == SVDMethod::BDC) {
        Eigen::BDCSVD<PlainT> svd(mat,
                                  Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.info() != Eigen::Success) return false;
        Eigen::Map<ColMajorC>(U_out, rows, k) = svd.matrixU();
        Eigen::Map<ColMajorC>(V_out, cols, k) = svd.matrixV();
        Eigen::Map<Eigen::VectorXd>(S_out, k) = svd.singularValues();
        return true;
    }

    Eigen::JacobiSVD<PlainT> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.info() != Eigen::Success) return false;
    Eigen::Map<ColMajorC>(U_out, rows, k) = svd.matrixU();
    Eigen::Map<ColMajorC>(V_out, cols, k) = svd.matrixV();
    Eigen::Map<Eigen::VectorXd>(S_out, k) = svd.singularValues();
    return true;
}

template <typename PlainT>
bool eigh_run(const std::complex<double>* data, int n, double* evals_out,
              std::complex<double>* evecs_out) {
    const Eigen::Map<const PlainT> mat(data, n, n);
    // Asking for eigenvectors only when the caller wants them: the solver skips
    // accumulating them otherwise, and two of the three callers here need the
    // spectrum alone.
    Eigen::SelfAdjointEigenSolver<PlainT> es(
        mat, evecs_out ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) return false;
    Eigen::Map<Eigen::VectorXd>(evals_out, n) = es.eigenvalues();
    if (evecs_out) Eigen::Map<ColMajorC>(evecs_out, n, n) = es.eigenvectors();
    return true;
}

}  // namespace

bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order, SVDMethod method,
              std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out) {
    if (rows <= 0 || cols <= 0) return false;

    // Routed out BEFORE the Eigen instantiations below, and by an explicit
    // test rather than by falling off the end of them. svd_run treats anything
    // that is not BDC as Jacobi, so a method it does not know silently becomes
    // a different algorithm than the caller asked for, which is the one
    // outcome the SVD ladder exists to prevent.
    if (method == SVDMethod::AutonneJacobi) {
        return autonne_svd_thin(data, rows, cols, order, U_out, S_out, V_out);
    }

    return (order == MatrixOrder::RowMajor)
               ? svd_run<RowMajorC>(data, rows, cols, method, U_out, S_out,
                                    V_out)
               : svd_run<ColMajorC>(data, rows, cols, method, U_out, S_out,
                                    V_out);
}

bool eigh(const std::complex<double>* data, int n, MatrixOrder order,
          double* evals_out, std::complex<double>* evecs_out) {
    if (n <= 0) return false;
    return (order == MatrixOrder::RowMajor)
               ? eigh_run<RowMajorC>(data, n, evals_out, evecs_out)
               : eigh_run<ColMajorC>(data, n, evals_out, evecs_out);
}

}  // namespace detail
}  // namespace lindblad
