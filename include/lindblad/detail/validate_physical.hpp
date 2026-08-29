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
//
// `residual` carries the COMPLETE expression, brackets and any maximum
// included, because not every property reduces over something. A matrix
// property reports the worst entry and says so; a state's normalization is one
// number and would be misdescribed by a maximum it does not take.

struct PhysicalProperty {
    const char* subject;
    const char* residual;
    const char* noun;
};

inline constexpr PhysicalProperty UNITARITY{
    "matrix is not unitary", "max |U†U - I|", "unitarity"};
inline constexpr PhysicalProperty KRAUS_TRACE_PRESERVING{
    "channel is not trace preserving", "max |Σ K†K - I|",
    "trace preservation"};
inline constexpr PhysicalProperty SUPEROP_TRACE_PRESERVING{
    "superoperator is not trace preserving",
    "max |Σ_r S[(r,r),(i,j)] - δ_ij|", "trace preservation"};

// A state's normalization and a density matrix's are separate properties
// because they are separate residuals, measured over different objects. They
// share a noun because a caller asking for the repair is asking for the same
// thing either way: divide the object by what it currently sums to.
inline constexpr PhysicalProperty STATE_NORMALIZATION{
    "state is not normalized", "|⟨ψ|ψ⟩ - 1|", "normalization"};
inline constexpr PhysicalProperty DENSITY_NORMALIZATION{
    "density matrix is not normalized", "|Tr(ρ) - 1|", "normalization"};

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
    return std::string(ctx) + ": " + p.subject + " (" + p.residual + " = " +
           format_residual(deviation) + ", atol = " + format_residual(atol) +
           ")";
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

// The same dispatch for a property that HAS a repair, which the three above do
// not. It cannot perform the repair itself: it sees a residual, not the object,
// and the repair belongs to whatever owns the data. So it reports back instead.
//
// Returns true when the caller must repair, which happens under Fix and only
// under Fix. Every other outcome is settled here:
//   - within tolerance: false, and the four policies are indistinguishable
//   - Warn: reported, then false. Warn describes, it does not repair, so the
//     caller proceeds with the object still violating the property.
//   - Throw: does not return.
//   - Ignore: never arrives, the entry points return before measuring.
inline bool enforce_physical_repairable(double deviation,
                                        const ValidationOptions& v,
                                        const char* ctx,
                                        const PhysicalProperty& p) {
    if (deviation <= v.atol) return false;  // a NaN deviation falls through

    switch (v.policy) {
        case Validation::Warn:
            emit_warning(physical_message(ctx, p, deviation, v.atol));
            return false;
        case Validation::Fix:
            return true;
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

// Σ |a_i|², summed as a balanced tree. Quarantined for the reason above and one
// more specific to it: this sum runs over 2^n positive terms, so nothing
// cancels and the dropped low bits accrue. A single running total drifts like
// sqrt(N)·eps, which at 30 qubits is seven times the tolerance it is compared
// against, and the project's tolerances are absolute with no relative
// counterpart, so one number must mean the same thing at every register size.
// Tree order costs no extra arithmetic and holds the error near 3e-14.
//
// Two overloads because there are two state layouts: the qubit statevector
// keeps real and imaginary parts in separate aligned arrays for SIMD, and
// everything else stores interleaved Complex128.
double state_norm_sq(const double* real, const double* imag, std::size_t n);
double state_norm_sq(const Complex128* amps, std::size_t n);

// Re Tr(ρ) over a dim x dim row-major matrix. The imaginary part is not
// summed: a Hermitian ρ has a real trace by construction, and a caller whose ρ
// is not Hermitian has a problem that normalization is not the repair for.
double density_trace_real(const Complex128* rho, std::size_t dim);

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

// -----------------------------------------------------------------------------
// Normalization checks
// -----------------------------------------------------------------------------
// These RETURN whether the caller must repair, rather than acting, because the
// repair rescales the object and only the object's owner can do that. `norm_sq`
// is ⟨ψ|ψ⟩ and `trace` is Re Tr(ρ), both measured through the quarantined sums
// above, or through a layer's own contraction where the state is not a flat
// array. Ignore returns before anything is judged.
//
// A non-finite measurement yields a non-finite deviation, which fails the
// tolerance comparison and is therefore reported rather than passed over.

inline bool check_normalized(double norm_sq, const ValidationOptions& v,
                             const char* ctx) {
    if (v.policy == Validation::Ignore) return false;
    return enforce_physical_repairable(std::abs(norm_sq - 1.0), v, ctx,
                                       STATE_NORMALIZATION);
}

inline bool check_trace_normalized(double trace, const ValidationOptions& v,
                                   const char* ctx) {
    if (v.policy == Validation::Ignore) return false;
    return enforce_physical_repairable(std::abs(trace - 1.0), v, ctx,
                                       DENSITY_NORMALIZATION);
}

} // namespace detail
} // namespace lindblad
