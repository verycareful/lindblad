#pragma once

#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "lindblad/detail/unitary_repair.hpp"

#include <array>
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
// message a repair request produces where no repair exists.
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
                                ": Repair::Attempt has no repair defined for " +
                                p.noun);
}

// -----------------------------------------------------------------------------
// Policy dispatch
// -----------------------------------------------------------------------------

// Whether the residual would be measured only to be discarded. Ignore with no
// repair asked for is the one combination where nothing consumes the number,
// so the check entry points return before taking it and opting out costs one
// predictable branch. Every other combination needs it: a repair has to know
// whether one is owed, and Warn has to report what it saw.
inline bool measurement_unused(const ValidationOptions& v) {
    return v.policy == Validation::Ignore && v.repair == Repair::None;
}

// Applies the caller's policy to a measured residual, for the properties that
// define no repair. Ignore arrives here only alongside Repair::Attempt, since
// the entry points return before measuring in the other case.
inline void enforce_physical(double deviation, const ValidationOptions& v,
                             const char* ctx, const PhysicalProperty& p) {
    if (deviation <= v.atol) return;  // a NaN deviation falls through to fail

    // Asking for a repair this property does not define is an error whatever
    // the response says, because the response governs an operand that is still
    // invalid and this is a mistake in the calling code. A repair that quietly
    // did nothing would be worse: it returns an unphysical operand under a
    // request to correct it.
    if (v.repair == Repair::Attempt) throw_no_repair(ctx, p);

    switch (v.policy) {
        case Validation::Warn:
            emit_warning(physical_message(ctx, p, deviation, v.atol));
            return;
        case Validation::Ignore:
            return;
        case Validation::Throw:
            break;
    }
    throw_not_physical(ctx, p, deviation, v.atol);
}

// The same dispatch for a property that HAS a repair. It cannot perform the
// repair itself: it sees a residual, not the object, and the repair belongs to
// whatever owns the data. So it reports back instead.
//
// Returns true when the caller must repair, which happens under Repair::Attempt
// and only under it, whatever the response says. The response knob does not
// decide whether to try, only what to do with an operand that is still invalid
// after trying, so it is consulted here only when nothing will be tried:
//   - within tolerance: false, and all six policies are indistinguishable
//   - Warn: reported, then false. Warn describes, it does not repair, so the
//     caller proceeds with the object still violating the property.
//   - Throw: does not return.
//   - Ignore: false, having reported nothing. Reached only under
//     Repair::Attempt, and that returns above.
inline bool enforce_physical_repairable(double deviation,
                                        const ValidationOptions& v,
                                        const char* ctx,
                                        const PhysicalProperty& p) {
    if (deviation <= v.atol) return false;  // a NaN deviation falls through
    if (v.repair == Repair::Attempt) return true;

    switch (v.policy) {
        case Validation::Warn:
            emit_warning(physical_message(ctx, p, deviation, v.atol));
            return false;
        case Validation::Ignore:
            return false;
        case Validation::Throw:
            break;
    }
    throw_not_physical(ctx, p, deviation, v.atol);
}

// Applies the response knob to an operand a repair was asked for, attempted on,
// and could not correct. This is the case the split exists to make sayable: the
// caller asked for a repair and also said what should happen when one is not
// enough, and those are different answers for a batch run and for an advisory
// check.
//
// `why` names what the repair did, because "still not unitary" alone cannot
// distinguish a projection that failed to factorise from one that ran and
// landed outside tolerance, and those want different responses from a caller.
inline void respond_unrepaired(const ValidationOptions& v, const char* ctx,
                               const PhysicalProperty& p,
                               const std::string& why) {
    const std::string msg =
        std::string(ctx) + ": could not repair " + p.noun + ", " + why;
    switch (v.policy) {
        case Validation::Warn:
            emit_warning(msg);
            return;
        case Validation::Ignore:
            return;
        case Validation::Throw:
            break;
    }
    throw std::invalid_argument(msg);
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
// flags; only the arithmetic is quarantined. Same pattern as the vendored NLopt
// and the decomposition backend: a numerical ALGORITHM whose result decides an
// accept or a reject needs IEEE semantics, while kernel arithmetic keeps
// fast-math.

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
// Each returns before measuring anything when the residual would go unused, so
// opting out costs one predictable branch and nothing else.

inline void check_unitary(const Complex128* U, std::size_t rows,
                          const ValidationOptions& v, const char* ctx) {
    if (measurement_unused(v) || rows == 0) return;
    enforce_physical(unitarity_deviation(U, rows), v, ctx, UNITARITY);
}

inline void check_unitary(const std::vector<Complex128>& U, std::size_t rows,
                          const ValidationOptions& v, const char* ctx) {
    check_unitary(U.data(), rows, v, ctx);
}

// The repairing form, split in two because the caller owns the storage.
//
// Unitarity is one of the three properties here with a repair defined, the
// other two being the normalizations. It follows the same shape as they do:
// this half measures and reports back whether a repair is owed, and the caller
// performs it, because the operand may live behind a type that decides for
// itself how it is written (a gate matrix is shared copy-on-write, so a repair
// rebinds a fresh buffer rather than mutating one that other instructions are
// reading).
//
// Splitting it this way also keeps the measurement free of an allocation on
// every path that does NOT repair: Repair::None never materialises a mutable
// copy under any response.
//
// Returns true under Repair::Attempt and only under it, when the operand is
// outside tolerance. Every other outcome is settled here exactly as it is for
// the normalization properties.
inline bool unitary_needs_repair(const Complex128* U, std::size_t rows,
                                 const ValidationOptions& v, const char* ctx) {
    if (measurement_unused(v) || rows == 0) return false;
    return enforce_physical_repairable(unitarity_deviation(U, rows), v, ctx,
                                       UNITARITY);
}

// Replaces U with its unitary polar factor and VERIFIES the result against the
// same tolerance that rejected the input.
//
// The verification is the whole reason this repair is safe to offer. A
// projection that silently failed to converge would hand back an operand as
// unphysical as the one it replaced, under a request to correct it, which is
// the exact failure mode Repair::Attempt exists to avoid. Since the
// postcondition is unitarity and unitarity_deviation measures precisely that,
// the check costs one more residual and leaves nothing asserted on trust.
//
// Returns true when U now satisfies unitarity. On false the response has
// already been delivered and U holds whatever the projection left behind, so
// the caller must fall back to the operand it copied from rather than install
// this buffer.
inline bool repair_unitary(Complex128* U, std::size_t rows,
                           const ValidationOptions& v, const char* ctx) {
    const double before = unitarity_deviation(U, rows);
    if (!project_to_unitary(U, rows)) {
        respond_unrepaired(v, ctx, UNITARITY,
                           "the polar projection failed to factorise the "
                           "operand (measured " + format_residual(before) + ")");
        return false;
    }
    const double after = unitarity_deviation(U, rows);
    if (!(after <= v.atol)) {
        respond_unrepaired(v, ctx, UNITARITY,
                           "the projection ran but its result is still outside "
                           "tolerance (" + format_residual(before) +
                           " before, " + format_residual(after) +
                           " after, atol " + format_residual(v.atol) + ")");
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// check_unitary_fixing - the repairing form for an entry point that BORROWS
// -----------------------------------------------------------------------------
// An entry point that stores its operand repairs once and keeps the result, so
// every later use sees the repaired matrix. QuantumCircuit::unitary is the only
// one that does. Everything else takes a const reference, applies it and lets it
// go, so a repair there fixes one call and evaporates: the caller still holds
// the operand it handed over, and a loop pays a projection every iteration
// without ever being told why.
//
// Hence the note, which is emitted here and NOT inside repair_unitary. The
// projection is shared with the storing path, where the cost is paid once and a
// note would be wrong; what deserves reporting is borrowing, which is a property
// of the entry point rather than of the arithmetic.
//
// Once-only delivery comes from the warning channel, which already suppresses a
// message it has already delivered until the next flush, rather than from a
// latch here. One text for every entry point, so the channel collapses them to
// one report however many sites raise it, and a caller that flushes to start a
// fresh accounting hears it again, which is what flushing is for. A local latch
// would defeat that and could not be reset at all.
inline void warn_unitary_repaired_once() {
    emit_warning(
        "note: Repair::Attempt repaired a matrix that was not unitary. The "
        "repair applies to this call and the matrix you passed is unchanged, so "
        "a loop over it, or a circuit run more than once, pays one projection "
        "every time. QuantumCircuit::unitary repairs the matrix it stores when "
        "the instruction is built, so correcting the operand at its source "
        "avoids the repeat entirely.");
}

// Measures unitarity and, under Repair::Attempt on an operand outside
// tolerance, repairs a COPY into `storage` and returns a reference to it.
// Repair::None behaves exactly as check_unitary does and returns the caller's
// operand untouched.
//
// A repair that does not converge returns the caller's operand as well, after
// respond_unrepaired has delivered the response the caller asked for. That is
// what makes Warn and Ignore usable alongside a repair: the call proceeds with
// the operand it was given rather than with a half-projected buffer.
//
// Storage is passed in rather than returned so the common path allocates
// nothing: an operand already inside tolerance, and Repair::None under every
// response, never materialises a copy. That matters because these entry points
// are the per-gate kernels.
inline const Complex128* check_unitary_fixing(const Complex128* U,
                                              std::size_t rows,
                                              const ValidationOptions& v,
                                              const char* ctx,
                                              std::vector<Complex128>& storage) {
    if (!unitary_needs_repair(U, rows, v, ctx)) return U;
    storage.assign(U, U + rows * rows);
    if (!repair_unitary(storage.data(), rows, v, ctx)) return U;
    warn_unitary_repaired_once();
    return storage.data();
}

inline const std::vector<Complex128>& check_unitary_fixing(
    const std::vector<Complex128>& U, std::size_t rows,
    const ValidationOptions& v, const char* ctx,
    std::vector<Complex128>& storage) {
    if (!unitary_needs_repair(U.data(), rows, v, ctx)) return U;
    storage = U;
    if (!repair_unitary(storage.data(), rows, v, ctx)) return U;
    warn_unitary_repaired_once();
    return storage;
}

// The fixed-size form, for the MPS gate entry points whose operands are arrays.
template <std::size_t N>
inline const std::array<Complex128, N>& check_unitary_fixing(
    const std::array<Complex128, N>& U, std::size_t rows,
    const ValidationOptions& v, const char* ctx,
    std::array<Complex128, N>& storage) {
    if (!unitary_needs_repair(U.data(), rows, v, ctx)) return U;
    storage = U;
    if (!repair_unitary(storage.data(), rows, v, ctx)) return U;
    warn_unitary_repaired_once();
    return storage;
}

// An empty operator list returns without measuring, because rejecting it
// belongs to detail::check_kraus_nonempty and has already happened at every
// entry point that fuses a channel. Measuring it here as well would report a
// residual of exactly 1 against the trace-preservation tolerance, describing a
// malformed argument as physics the caller might opt out of.
inline void check_kraus_tp(const std::vector<std::vector<Complex128>>& ops,
                           std::size_t dim, const ValidationOptions& v,
                           const char* ctx) {
    if (measurement_unused(v) || dim == 0 || ops.empty()) return;
    enforce_physical(kraus_tp_deviation(ops, dim), v, ctx,
                     KRAUS_TRACE_PRESERVING);
}

inline void check_superop_tp(const std::vector<Complex128>& S, std::size_t dim,
                             const ValidationOptions& v, const char* ctx) {
    if (measurement_unused(v) || dim == 0) return;
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
// array. Ignore with no repair asked for returns before anything is judged.
//
// A non-finite measurement yields a non-finite deviation, which fails the
// tolerance comparison and is therefore reported rather than passed over.

inline bool check_normalized(double norm_sq, const ValidationOptions& v,
                             const char* ctx) {
    if (measurement_unused(v)) return false;
    return enforce_physical_repairable(std::abs(norm_sq - 1.0), v, ctx,
                                       STATE_NORMALIZATION);
}

inline bool check_trace_normalized(double trace, const ValidationOptions& v,
                                   const char* ctx) {
    if (measurement_unused(v)) return false;
    return enforce_physical_repairable(std::abs(trace - 1.0), v, ctx,
                                       DENSITY_NORMALIZATION);
}

// The rescale these two properties define as their repair is impossible for
// exactly one input: an object with no norm to divide out, zero or non-finite.
// Routing that through the response knob is what lets a long run report one
// degenerate state and carry on rather than ending on it.
//
// `measure` is the same number that was judged, ⟨ψ|ψ⟩ or Re Tr(ρ), so the
// caller passes what it already has. Returns true when the rescale can be
// performed; on false the response has been delivered and the object must be
// left exactly as it is.
inline bool normalization_repairable(double measure, const ValidationOptions& v,
                                     const char* ctx,
                                     const PhysicalProperty& p) {
    if (measure > 0.0 && std::isfinite(measure)) return true;
    respond_unrepaired(v, ctx, p, "there is no norm to divide out (measured " +
                                      format_residual(measure) + ")");
    return false;
}

} // namespace detail
} // namespace lindblad
