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

// Pairwise summation, for the residuals that reduce over the whole state.
//
// Normalization is the only physical property here whose residual comes from a
// LONG sum. Every other one reduces over a handful of entries: a dot product
// down a 4x4, where a running total is already exact far inside the tolerance.
// A state's norm reduces over 2^n terms, all of them positive, so nothing
// cancels and the dropped low bits simply accrue.
//
// A single running total drifts like sqrt(N)·eps, which is 2.3e-13 at 20 qubits
// but 7.3e-12 at 30, and the tolerance it is compared against is 1e-12. The
// project's tolerances are ABSOLUTE with no relative counterpart anywhere, so
// one number has to mean the same thing at every register size. A correct
// 30-qubit state reported as unnormalized would be the measurement inventing
// the violation it reports.
//
// Summing as a balanced binary tree puts each term through about log2(N)
// additions instead of N, which bounds the error at roughly
// (log2(N/kBlock) + kBlock)·eps: about 3e-14 at 30 qubits, thirty times inside
// the tolerance, for the SAME number of additions as the naive loop. Only the
// order changes.
//
// The shape is fixed by the length alone, so this is also fully deterministic:
// no reduction tree chosen by a thread count, no dependence on vector width. A
// parallel reduction would be faster and would let the verdict move with how
// many cores happened to be free, which is exactly what a check must not do.
//
// kBlock stops the recursion where call overhead would start to dominate; the
// block itself is summed straight, and its length bounds that part of the error.
constexpr std::size_t kPairwiseBlock = 128;

template <typename Term>
double pairwise_sum(Term term, std::size_t lo, std::size_t hi) {
    const std::size_t n = hi - lo;
    if (n <= kPairwiseBlock) {
        double s = 0.0;
        for (std::size_t i = lo; i < hi; ++i) s += term(i);
        return s;
    }
    const std::size_t mid = lo + n / 2;
    return pairwise_sum(term, lo, mid) + pairwise_sum(term, mid, hi);
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

// Σ |a_i|² over separate real and imaginary arrays (the SoA statevector).
// Summed as a tree; see pairwise_sum for why the order is load-bearing. The
// kernels that evolve the state keep their vectorised sweeps, and this is the
// measurement, which has to answer the same on every build and at every width.
double state_norm_sq(const double* real, const double* imag, std::size_t n) {
    return pairwise_sum(
        [real, imag](std::size_t i) {
            return real[i] * real[i] + imag[i] * imag[i];
        },
        0, n);
}

// Σ |a_i|² over interleaved amplitudes. Same contract as the overload above.
double state_norm_sq(const Complex128* amps, std::size_t n) {
    return pairwise_sum(
        [amps](std::size_t i) {
            return amps[i].real * amps[i].real + amps[i].imag * amps[i].imag;
        },
        0, n);
}

// Re Tr(ρ) = Σ_i ρ[i][i].real, reading the diagonal of a row-major dim x dim
// matrix. Summed as a tree for the same reason as the norms above: the diagonal
// of a 30-qubit ρ is itself 2^30 entries.
double density_trace_real(const Complex128* rho, std::size_t dim) {
    return pairwise_sum(
        [rho, dim](std::size_t i) { return rho[i * dim + i].real; }, 0, dim);
}

} // namespace detail
} // namespace lindblad
