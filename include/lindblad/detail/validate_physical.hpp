#pragma once

#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// detail::validate_physical - Class C checks for caller-supplied operators
// =============================================================================
// Class A (index bounds) and Class B (operand structure) in detail/validate.hpp
// are integer comparisons guarding undefined behaviour, so they are
// unconditional. The checks here guard physical validity instead: unitarity of
// a supplied matrix, trace preservation of a supplied channel. Violating one
// costs a well-defined but unphysical result rather than memory safety, and
// the check itself is floating-point work of the same order as a dense
// multiply, so the caller's ValidationOptions decides whether it runs and what
// happens when it fails.
//
// Cost, for a k-operand matrix of side N = 2^k: the residual is N^3/2 complex
// multiplies, against a kernel that sweeps 2^n·2^k amplitudes. The check is
// therefore 4^k / 2^n of the work it guards, a fraction of a percent at any
// register size worth simulating, and it approaches the kernel's own cost only
// for wide gates on tiny registers.
//
// A matrix carrying a non-finite entry is exactly what these checks exist to
// catch, and two separate mechanisms are needed to catch one. The tolerance
// comparison below is written as !(dev <= tol) rather than (dev > tol), so a
// NaN residual, which compares false against everything, is a failure rather
// than a silent pass. Accumulating the residual needs more than that spelling,
// and that half lives with the arithmetic in src/validate_physical.cpp.

namespace lindblad {
namespace detail {

// -----------------------------------------------------------------------------
// The properties a primitive can be asked to enforce
// -----------------------------------------------------------------------------
// subject names the failure, residual names the measured quantity so a caller
// can see what the number refers to, and noun names the property in the
// message Fix produces where no repair exists.

struct PhysicalProperty {
    const char* subject;
    const char* residual;
    const char* noun;
};

inline constexpr PhysicalProperty UNITARITY{
    "matrix is not unitary", "U†U - I", "unitarity"};
inline constexpr PhysicalProperty KRAUS_TRACE_PRESERVING{
    "channel is not trace preserving", "Σ K†K - I", "trace preservation"};
inline constexpr PhysicalProperty SUPEROP_TRACE_PRESERVING{
    "superoperator is not trace preserving", "Σ_r S[(r,r),(i,j)] - δ_ij",
    "trace preservation"};

// -----------------------------------------------------------------------------
// Message formatting
// -----------------------------------------------------------------------------

// A residual is read in order to choose a tolerance, so it needs significant
// digits rather than the six decimal places std::to_string gives, which prints
// every deviation below 1e-6 as "0.000000".
inline std::string format_residual(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", x);
    return std::string(buf);
}

inline std::string physical_message(const char* ctx, const PhysicalProperty& p,
                                    double deviation, double atol) {
    return std::string(ctx) + ": " + p.subject + " (max |" + p.residual +
           "| = " + format_residual(deviation) + ", atol = " +
           format_residual(atol) + ")";
}

[[noreturn]] inline void throw_not_physical(const char* ctx,
                                            const PhysicalProperty& p,
                                            double deviation, double atol) {
    throw std::invalid_argument(physical_message(ctx, p, deviation, atol));
}

[[noreturn]] inline void throw_no_repair(const char* ctx,
                                         const PhysicalProperty& p) {
    throw std::invalid_argument(std::string(ctx) +
                                ": Validation::Fix has no repair defined for " +
                                p.noun);
}

// -----------------------------------------------------------------------------
// Policy dispatch
// -----------------------------------------------------------------------------

// Applies the caller's policy to a measured residual. Ignore never arrives
// here: the check entry points return before measuring anything.
inline void enforce_physical(double deviation, const ValidationOptions& v,
                             const char* ctx, const PhysicalProperty& p) {
    if (deviation <= v.atol) return;  // a NaN deviation falls through to fail

    switch (v.policy) {
        case Validation::Warn:
            emit_warning(physical_message(ctx, p, deviation, v.atol));
            return;
        case Validation::Fix:
            // Repair is defined per property. Where none is, saying so is the
            // only honest option: a Fix that quietly did nothing would return
            // an unphysical result under a policy that promised to correct it.
            throw_no_repair(ctx, p);
        case Validation::Throw:
        case Validation::Ignore:
            break;
    }
    throw_not_physical(ctx, p, deviation, v.atol);
}

// -----------------------------------------------------------------------------
// Residuals
// -----------------------------------------------------------------------------
// DEFINED in src/validate_physical.cpp, which is compiled with -fno-fast-math.
// These are measurements, and a measurement has to give the same answer on
// every target or it is not one. They evaluate U†U - I, a near-cancellation, and
// that is exactly the shape a permissive floating-point model perturbs most:
// under -ffast-math the same Hadamard yielded 1.2386923780787705e-16 at
// -march=native against the correct 2.2204460492503131e-16, a 44% swing decided
// by a compiler flag. A residual sitting near the caller's atol could then be
// accepted on one build and rejected on another.
//
// The policy dispatch below stays inline and stays under the project-wide
// flags; only the arithmetic is quarantined. Same pattern as the mps_sim.cpp
// and NLopt quarantines: numerical ALGORITHMS need IEEE semantics, kernel
// arithmetic keeps fast-math.

// max |(U†U - I)_ij|. U†U is Hermitian, so entry (j,i) is the conjugate of
// (i,j) and carries the same magnitude; walking the upper triangle halves the
// work without changing the maximum.
double unitarity_deviation(const Complex128* U, std::size_t rows);

// max |(Σ_k K_k†K_k - I)_ij|, the trace-preservation residual of a Kraus set.
double kraus_tp_deviation(const std::vector<std::vector<Complex128>>& ops,
                          std::size_t dim);

// A superoperator maps rho'[ro,co] = Σ_(ri,ci) S[(ro,co),(ri,ci)]·rho[ri,ci],
// so preserving the trace of every rho means Σ_ro S[(ro,ro),(ri,ci)] = δ_ri,ci
// at every input index pair. That sum over the output diagonal is the entire
// condition, and it reads dim^3 of the 4^k x 4^k array once.
double superop_tp_deviation(const Complex128* S, std::size_t dim);

// -----------------------------------------------------------------------------
// Check entry points
// -----------------------------------------------------------------------------
// Each returns before measuring anything under Validation::Ignore, so opting
// out costs one predictable branch and nothing else.

inline void check_unitary(const Complex128* U, std::size_t rows,
                          const ValidationOptions& v, const char* ctx) {
    if (v.policy == Validation::Ignore || rows == 0) return;
    enforce_physical(unitarity_deviation(U, rows), v, ctx, UNITARITY);
}

inline void check_unitary(const std::vector<Complex128>& U, std::size_t rows,
                          const ValidationOptions& v, const char* ctx) {
    check_unitary(U.data(), rows, v, ctx);
}

// An empty operator list returns without measuring, because rejecting it
// belongs to detail::check_kraus_nonempty and has already happened at every
// entry point that fuses a channel. Measuring it here as well would report a
// residual of exactly 1 against the trace-preservation tolerance, describing a
// malformed argument as physics the caller might opt out of.
inline void check_kraus_tp(const std::vector<std::vector<Complex128>>& ops,
                           std::size_t dim, const ValidationOptions& v,
                           const char* ctx) {
    if (v.policy == Validation::Ignore || dim == 0 || ops.empty()) return;
    enforce_physical(kraus_tp_deviation(ops, dim), v, ctx,
                     KRAUS_TRACE_PRESERVING);
}

inline void check_superop_tp(const std::vector<Complex128>& S, std::size_t dim,
                             const ValidationOptions& v, const char* ctx) {
    if (v.policy == Validation::Ignore || dim == 0) return;
    enforce_physical(superop_tp_deviation(S.data(), dim), v, ctx,
                     SUPEROP_TRACE_PRESERVING);
}

} // namespace detail
} // namespace lindblad
