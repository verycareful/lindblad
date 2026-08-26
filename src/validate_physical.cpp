// validate_physical.cpp - the Class C residual arithmetic
//
// Compiled with -fno-fast-math (see the top-level CMakeLists.txt). Everything
// here is a MEASUREMENT, and a measurement that changes with the compiler flags
// is not one. Each residual evaluates a near-cancellation, which is the shape a
// permissive floating-point model perturbs hardest: a Hadamard measured
// 1.2386923780787705e-16 at -march=native under -ffast-math against the correct
// 2.2204460492503131e-16, and a residual near the caller's atol could be
// accepted on one build and rejected on another for no reason but codegen.
//
// The policy dispatch stays inline in detail/validate_physical.hpp under the
// project-wide flags. Only the arithmetic is quarantined, matching the
// mps_sim.cpp and NLopt precedents.

#include "lindblad/detail/validate_physical.hpp"

#include <cmath>

namespace lindblad {
namespace detail {

namespace {

// A running maximum that a non-finite entry, once seen, is never displaced
// from.
//
// Spelling a single comparison as !(d <= worst) makes a NaN a failure, since a
// NaN compares false against everything. That is not sufficient for a RUNNING
// maximum: once worst holds a NaN, !(d <= worst) is true for every later d as
// well, so the next finite entry overwrites it. Only the last pair visited
// would survive, and that pair is always the final diagonal one, which is why
// position within the operand would decide whether a NaN is reported at all.
//
// Latching is done on the bit pattern rather than on comparison, so the
// residual reports the non-finite operand under any floating-point model.
void accumulate_worst(double& worst_sq, double d) {
    if (!is_finite_strict(worst_sq)) return;
    if (!is_finite_strict(d) || d > worst_sq) worst_sq = d;
}

} // namespace

// max |(U†U - I)_ij|. U†U is Hermitian, so entry (j,i) is the conjugate of
// (i,j) and carries the same magnitude; walking the upper triangle halves the
// work without changing the maximum.
double unitarity_deviation(const Complex128* U, std::size_t rows) {
    double worst_sq = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = i; j < rows; ++j) {
            Complex128 acc(0.0, 0.0);
            for (std::size_t m = 0; m < rows; ++m)
                acc += U[m * rows + i].conj() * U[m * rows + j];
            const Complex128 diff = acc - Complex128(i == j ? 1.0 : 0.0, 0.0);
            const double d = diff.norm_sq();
            accumulate_worst(worst_sq, d);
        }
    }
    return std::sqrt(worst_sq);
}

// max |(Σ_k K_k†K_k - I)_ij|, the trace-preservation residual of a Kraus set.
double kraus_tp_deviation(
    const std::vector<std::vector<Complex128>>& ops, std::size_t dim) {
    std::vector<Complex128> sum(dim * dim, Complex128(0.0, 0.0));
    for (const auto& K : ops) {
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = i; j < dim; ++j) {
                Complex128 acc(0.0, 0.0);
                for (std::size_t m = 0; m < dim; ++m)
                    acc += K[m * dim + i].conj() * K[m * dim + j];
                sum[i * dim + j] += acc;
            }
        }
    }

    double worst_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i; j < dim; ++j) {
            const Complex128 diff =
                sum[i * dim + j] - Complex128(i == j ? 1.0 : 0.0, 0.0);
            const double d = diff.norm_sq();
            accumulate_worst(worst_sq, d);
        }
    }
    return std::sqrt(worst_sq);
}

// A superoperator maps rho'[ro,co] = Σ_(ri,ci) S[(ro,co),(ri,ci)]·rho[ri,ci],
// so preserving the trace of every rho means Σ_ro S[(ro,ro),(ri,ci)] = δ_ri,ci
// at every input index pair. That sum over the output diagonal is the entire
// condition, and it reads dim^3 of the 4^k x 4^k array once.
double superop_tp_deviation(const Complex128* S, std::size_t dim) {
    const std::size_t side = dim * dim;  // 4^k for a k-qubit channel
    double worst_sq = 0.0;
    for (std::size_t ri = 0; ri < dim; ++ri) {
        for (std::size_t ci = 0; ci < dim; ++ci) {
            const std::size_t in = ri * dim + ci;
            Complex128 acc(0.0, 0.0);
            for (std::size_t r = 0; r < dim; ++r)
                acc += S[(r * dim + r) * side + in];
            const Complex128 diff = acc - Complex128(ri == ci ? 1.0 : 0.0, 0.0);
            const double d = diff.norm_sq();
            accumulate_worst(worst_sq, d);
        }
    }
    return std::sqrt(worst_sq);
}

} // namespace detail
} // namespace lindblad
