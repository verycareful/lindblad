#pragma once

#include <functional>
#include <string>
#include <type_traits>

// =============================================================================
// Validation - physical-validity policy for caller-supplied operators
// =============================================================================
// Index bounds and operand structure (detail/validate.hpp) are always checked,
// because they are the difference between a loud error and undefined
// behaviour. This header covers the third class, physical validity: an operand
// that is the right size and memory-safe but not physically meaningful, such
// as a supplied gate matrix that is not unitary or a Kraus set that is not
// trace preserving. Those produce a well-defined but unphysical number rather
// than a crash, so the caller decides what should happen.
//
// The policy is per-primitive rather than a single global switch. Each
// primitive that can violate exactly one physical property takes a
// ValidationOptions, so a caller reaching a primitive directly (with no
// simulator object in sight) still controls it, and a circuit carries the
// policy on the instruction whose matrix it describes.

namespace lindblad {

// -----------------------------------------------------------------------------
// The two knobs
// -----------------------------------------------------------------------------
// A caller-supplied operator that fails its check raises two separate
// questions: may the library try to make it hold, and what happens if it still
// does not. They are separate enums because fusing them is what makes a policy
// unsayable. One enumerator meaning "repair, and throw when the operand is
// still invalid" cannot also mean "repair, and warn"; a caller wanting the
// second has no way to ask for it.
//
// Split, the six reachable policies are the product of the two:
//
//                    Validation::Throw   Validation::Warn   Validation::Ignore
//   Repair::None     reject              report, proceed    proceed
//   Repair::Attempt  repair, else reject repair, else warn  repair, else proceed
//
// Repair::None with Validation::Ignore is the one combination that measures
// nothing: with nothing to repair and nothing to report, the residual would be
// computed only to be discarded, so the check entry points return before
// taking it. Every other combination needs the number.

// What happens when the property does not hold and the operand was not, or
// could not be, made to hold it.
enum class Validation {
    Throw,   // reject the call with std::invalid_argument
    Warn,    // report through the warning handler, then proceed
    Ignore   // proceed silently
};

// Whether the library may rewrite the operand before judging it. An enum
// rather than a bool because a property can have more than one repair: the
// unitary polar projection is one choice among several, and naming a specific
// one later must not require changing the type.
enum class Repair {
    None,     // judge the operand exactly as it was handed over
    Attempt   // apply the repair defined for the property first

    // Asking for a repair a property does not define is an error under every
    // response, because it is a mistake in the calling code rather than a
    // property of the operand, and the response knob governs operands. Only
    // unitarity and the two normalizations define one; trace preservation
    // does not.
};

// -----------------------------------------------------------------------------
// ValidationOptions - the per-primitive knobs
// -----------------------------------------------------------------------------

// The tolerance every physical-validity check defaults to, and the value a
// predicate over the same property should default to as well, so that asking
// "is this valid?" and letting a policy judge it agree by construction.
inline constexpr double DEFAULT_PHYSICAL_ATOL = 1e-12;

struct ValidationOptions {
    Validation policy = Validation::Throw;

    // Absolute tolerance on the worst entry of the residual, e.g.
    // max |U†U - I| for unitarity. A correct gate over irrational amplitudes
    // is never bit-exactly unitary, so the check has to carry a tolerance at
    // all; 1e-12 is loose enough to accept a deeply composed or fused matrix,
    // which holds unitarity to around 1e-13, and tight enough to reject
    // anything that drifted for a reason.
    double atol = DEFAULT_PHYSICAL_ATOL;

    // Last, and the position is load-bearing. ValidationOptions is written
    // positionally as `{policy, atol}` at more than eighty call sites across
    // the library, its tests and its docs; a knob ahead of atol would bind a
    // tolerance to the wrong field at every one of them.
    Repair repair = Repair::None;
};

// The fusion pre-pass copies an Instruction per fused block, and an
// Instruction carries a ValidationOptions, so this type staying trivial is
// what keeps that copy free.
static_assert(std::is_trivially_copyable<ValidationOptions>::value,
              "ValidationOptions is copied per fused block and must stay trivial");

// -----------------------------------------------------------------------------
// Warning channel
// -----------------------------------------------------------------------------
// Every warning the library emits goes through one handler, so a caller can
// capture, redirect, or silence all of them in one place. The default writes
// to stderr.
//
// Warn is reachable from inside a shots loop, where the same violation recurs
// once per gate per shot on a matrix that has not changed. Identical messages
// are therefore emitted once and counted; the repeat count is reported at the
// next flush point, so nothing is dropped and the output stays readable.

// The handler is invoked with the warning lock held, which is what keeps a
// sink from being entered by two OpenMP threads at once. It must therefore
// not call back into set_warning_handler, emit_warning, or flush_warnings:
// the lock is not recursive.

using WarningHandler = std::function<void(const std::string&)>;

// Install a handler. An empty handler restores the stderr default. Replacing
// the handler flushes any pending repeat counts to the outgoing one, so a
// count is never attributed to a sink that did not see the first occurrence.
void set_warning_handler(WarningHandler handler);

// Report a warning. The first occurrence of a given message reaches the
// handler immediately; later identical ones increment its repeat count.
void emit_warning(const std::string& message);

// Emit "[repeated N times]" for every message seen more than once since the
// last flush, and clear the counts. run() flushes on completion; a caller
// driving primitives directly can flush whenever it wants a coherent report.
void flush_warnings();

// Flushes pending repeat counts when the scope ends, on every exit path
// including an exception. Each backend run() holds one, so a repeat count is
// always attributed to the run that produced it rather than accumulating into
// whatever runs next.
struct ScopedWarningFlush {
    ScopedWarningFlush() = default;
    ~ScopedWarningFlush() { flush_warnings(); }
    ScopedWarningFlush(const ScopedWarningFlush&) = delete;
    ScopedWarningFlush& operator=(const ScopedWarningFlush&) = delete;
};

} // namespace lindblad
