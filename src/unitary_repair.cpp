// unitary_repair.cpp - the unitary polar projection behind Repair::Attempt
//
// Contract and the reason the projection is the right repair:
// include/lindblad/detail/unitary_repair.hpp.

#include "lindblad/detail/unitary_repair.hpp"

#include "lindblad/detail/eigen_backend.hpp"

#include <complex>
#include <vector>

namespace lindblad {
namespace detail {

bool project_to_unitary(Complex128* U, std::size_t rows) {
    if (rows == 0) return false;
    const int n = static_cast<int>(rows);
    const std::size_t cells = rows * rows;

    // Complex128 and std::complex<double> hold the same two doubles in the same
    // order, but they are distinct types, so the conversion is written out
    // rather than cast through. These matrices are a single gate's operand,
    // 2x2 up to d^2 x d^2, so the copy is not where the time goes.
    std::vector<std::complex<double>> m(cells);
    for (std::size_t i = 0; i < cells; ++i)
        m[i] = std::complex<double>(U[i].real, U[i].imag);

    std::vector<std::complex<double>> W(cells), V(cells);
    std::vector<double> S(rows);

    // Row-major: an operand is stored with its row index major throughout the
    // library, and the seam maps it in place rather than transposing.
    //
    // Jacobi rather than BDC. The input here is by definition close to unitary,
    // so its spectrum is a cluster of values near 1, and a near-degenerate
    // spectrum is the case Jacobi resolves and divide-and-conquer does not.
    if (!svd_thin(m.data(), n, n, MatrixOrder::RowMajor, SVDMethod::Jacobi,
                  W.data(), S.data(), V.data())) {
        return false;
    }

    // W V†, built directly into the output layout. W and V come back
    // column-major from the seam, so W(i,k) is W[k*n + i] and the conjugated
    // V(j,k) is V[k*n + j]; the result is written row-major to match the input.
    std::vector<std::complex<double>> out(cells);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::complex<double> acc(0.0, 0.0);
            for (int k = 0; k < n; ++k)
                acc += W[static_cast<std::size_t>(k) * n + i] *
                       std::conj(V[static_cast<std::size_t>(k) * n + j]);
            out[static_cast<std::size_t>(i) * n + j] = acc;
        }
    }

    // Nothing is written back until the whole result is known finite, so a
    // failed repair leaves the caller holding exactly what it passed in.
    for (std::size_t i = 0; i < cells; ++i) {
        if (!is_finite_strict(out[i].real()) || !is_finite_strict(out[i].imag()))
            return false;
    }
    for (std::size_t i = 0; i < cells; ++i)
        U[i] = Complex128(out[i].real(), out[i].imag());

    return true;
}

}  // namespace detail
}  // namespace lindblad
