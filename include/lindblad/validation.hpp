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
// Validation - what to do when a physical property does not hold
// -----------------------------------------------------------------------------

enum class Validation {
    Throw,   // reject the call with std::invalid_argument
    Warn,    // report through the warning handler, then proceed
    Fix,     // repair and continue where a repair is defined, otherwise throw
    Ignore   // skip the check entirely; the only policy that costs nothing
};

// -----------------------------------------------------------------------------
// ValidationOptions - the per-primitive knobs
// -----------------------------------------------------------------------------

struct ValidationOptions {
    Validation policy = Validation::Throw;

    // Absolute tolerance on the worst entry of the residual, e.g.
    // max |U†U - I| for unitarity. A correct gate over irrational amplitudes
    // is never bit-exactly unitary, so the check has to carry a tolerance at
    // all; 1e-12 is loose enough to accept a deeply composed or fused matrix,
    // which holds unitarity to around 1e-13, and tight enough to reject
    // anything that drifted for a reason.
    double atol = 1e-12;
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
