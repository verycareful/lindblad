#pragma once

#include <complex>
#include <cstddef>
#include <vector>

// =============================================================================
// detail::DenseMatrix, detail::RealVector - the library's own dense storage
// =============================================================================
// Owning, column-major, complex double. It exists so that no type crossing a
// Lindblad interface is a third-party type: a caller of the truncation ladder
// or of the qudit tensor accessors should not have to include Eigen, and a
// backend swap should not change a single signature.
//
// Storage is COLUMN-MAJOR because that is what every producer already hands
// back. eigen_backend.cpp writes its factors column-major whichever order the
// input arrived in, so a factor built here needs no transpose to be read there,
// and `data()` can be passed straight to the seam.
//
// Element access is `m(row, col)`, matching what the call sites already spell,
// so replacing the storage type does not rewrite the arithmetic that reads it.
//
// This is STORAGE, not a linear algebra library. There is no operator*, no
// adjoint, no determinant. Code that needs those maps this buffer with whatever
// backend it uses and works there, which keeps the decision about which library
// performs an operation at the site that performs it rather than baked into the
// type every caller holds.

namespace lindblad {
namespace detail {

class DenseMatrix {
    int rows_ = 0;
    int cols_ = 0;
    std::vector<std::complex<double>> d_;

public:
    DenseMatrix() = default;

    // Zero-initialised, matching the guarantee callers already rely on when
    // they fill only the columns above a validity floor and leave the rest.
    DenseMatrix(int rows, int cols)
        : rows_(rows), cols_(cols),
          d_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
             std::complex<double>(0.0, 0.0)) {}

    int rows() const noexcept { return rows_; }
    int cols() const noexcept { return cols_; }
    bool empty() const noexcept { return d_.empty(); }
    std::size_t size() const noexcept { return d_.size(); }

    std::complex<double>& operator()(int r, int c) {
        return d_[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows_) +
                  static_cast<std::size_t>(r)];
    }
    const std::complex<double>& operator()(int r, int c) const {
        return d_[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows_) +
                  static_cast<std::size_t>(r)];
    }

    std::complex<double>* data() noexcept { return d_.data(); }
    const std::complex<double>* data() const noexcept { return d_.data(); }

    // Reallocates and zeroes. Callers resize before filling, never to preserve
    // content, so growing and shrinking behave the same way and neither leaves
    // a stale value behind.
    void resize(int rows, int cols) {
        rows_ = rows;
        cols_ = cols;
        d_.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
                  std::complex<double>(0.0, 0.0));
    }

    void set_zero() noexcept {
        for (auto& z : d_) z = std::complex<double>(0.0, 0.0);
    }

    // Column base pointer. Column-major storage makes a column contiguous, so a
    // caller copying or scaling one column addresses it directly.
    std::complex<double>* col(int c) noexcept {
        return d_.data() + static_cast<std::size_t>(c) *
                               static_cast<std::size_t>(rows_);
    }
    const std::complex<double>* col(int c) const noexcept {
        return d_.data() + static_cast<std::size_t>(c) *
                               static_cast<std::size_t>(rows_);
    }
};

// Real-valued vector, for spectra. Separate from DenseMatrix rather than a
// one-column case of it because singular values are real and giving them a
// complex slot would invite a caller to write an imaginary part into one.
class RealVector {
    std::vector<double> d_;

public:
    RealVector() = default;
    explicit RealVector(int n) : d_(static_cast<std::size_t>(n), 0.0) {}

    int size() const noexcept { return static_cast<int>(d_.size()); }
    bool empty() const noexcept { return d_.empty(); }

    double& operator()(int i) { return d_[static_cast<std::size_t>(i)]; }
    double operator()(int i) const { return d_[static_cast<std::size_t>(i)]; }

    double* data() noexcept { return d_.data(); }
    const double* data() const noexcept { return d_.data(); }

    void resize(int n) { d_.assign(static_cast<std::size_t>(n), 0.0); }
};

}  // namespace detail
}  // namespace lindblad
