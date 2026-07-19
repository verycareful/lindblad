#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// detail::validate — shared fail-loud checks for backend apply-primitives
// =============================================================================
// The low-level public primitives (gates::apply_*, DensityMatrix::apply_gate,
// StabilizerState gates, MPS gates, the qudit apply-primitives) touch raw memory
// via a caller-supplied index: `1ULL << q` for statevector kernels, `ipow(d, q)`
// strides for the qudit layer, direct tableau columns for Clifford. An
// out-of-range index there is undefined behaviour (shift-width overflow,
// out-of-bounds strided writes), not a loud error.
//
// These helpers make Class A (index bounds) and Class B (operand structure)
// fail-loud at every primitive, matching the circuit layer's contract
// (QuantumCircuit::validate_qubit): std::out_of_range for bounds,
// std::invalid_argument for structure, same message wording.
//
// Cost: each check is a handful of integer comparisons at the O(2^n) kernel
// entry, NOT per-amplitude. Next to the amplitude sweep it is a rounding error.
// The throw itself lives in a cold [[noreturn]] helper so the hot path is a
// single predict-not-taken branch.
//
// The checks throw, so any primitive that calls them must NOT be `noexcept`
// (a throw from a noexcept function calls std::terminate).

namespace lindblad {
namespace detail {

// -----------------------------------------------------------------------------
// Cold throw helpers (kept out-of-line-ish via [[noreturn]] so the check sites
// stay branch-only on the common path)
// -----------------------------------------------------------------------------

[[noreturn]] inline void throw_index_oor(const char* ctx, const char* noun,
                                         long long idx, long long n) {
    throw std::out_of_range(std::string(ctx) + ": " + noun + " index " +
                            std::to_string(idx) + " out of range [0, " +
                            std::to_string(n) + ")");
}

[[noreturn]] inline void throw_not_distinct(const char* ctx, const char* noun) {
    throw std::invalid_argument(std::string(ctx) + ": " + noun +
                                " must be distinct");
}

[[noreturn]] inline void throw_bad_size(const char* ctx, const char* what,
                                        long long expected, long long got) {
    throw std::invalid_argument(std::string(ctx) + ": " + what + " must have " +
                                std::to_string(expected) + " entries, got " +
                                std::to_string(got));
}

[[noreturn]] inline void throw_structure(const char* ctx, const std::string& msg) {
    throw std::invalid_argument(std::string(ctx) + ": " + msg);
}

// -----------------------------------------------------------------------------
// Class A — index bounds (0 <= idx < n)
// -----------------------------------------------------------------------------

// Qubit index into an n-qubit register. `ctx` is the primitive name (a string
// literal), used to prefix the message, e.g. "h: qubit index 99 out of range
// [0, 3)".
inline void check_qubit(int q, int n, const char* ctx) {
    if (q < 0 || q >= n) throw_index_oor(ctx, "qubit", q, n);
}

// Qudit index into an n-qudit register.
inline void check_qudit(int q, int n, const char* ctx) {
    if (q < 0 || q >= n) throw_index_oor(ctx, "qudit", q, n);
}

// Every entry of a qubit list must be in range.
inline void check_qubits(const std::vector<int>& qs, int n, const char* ctx) {
    for (int q : qs)
        if (q < 0 || q >= n) throw_index_oor(ctx, "qubit", q, n);
}

// Every entry of a qudit list must be in range.
inline void check_qudits(const std::vector<int>& qs, int n, const char* ctx) {
    for (int q : qs)
        if (q < 0 || q >= n) throw_index_oor(ctx, "qudit", q, n);
}

// -----------------------------------------------------------------------------
// Class B — operand structure (distinctness, operand count, dimension match)
// -----------------------------------------------------------------------------

// Two operands that must not alias (a strided kernel over an aliased pair
// silently corrupts state).
inline void check_distinct2(int a, int b, const char* ctx,
                            const char* noun = "qubits") {
    if (a == b) throw_not_distinct(ctx, noun);
}

// Pairwise distinctness over a small operand list (O(k^2), k is the gate arity).
inline void check_all_distinct(const std::vector<int>& qs, const char* ctx,
                               const char* noun = "qubits") {
    for (size_t i = 0; i < qs.size(); ++i)
        for (size_t j = i + 1; j < qs.size(); ++j)
            if (qs[i] == qs[j]) throw_not_distinct(ctx, noun);
}

// A supplied dense matrix (row-major) acting on `k` qubits/qudits of local
// dimension `local_dim` must hold exactly (local_dim^k)^2 entries.
inline void check_matrix_dim(size_t got, size_t rows, const char* ctx,
                             const char* what = "matrix") {
    const size_t expected = rows * rows;
    if (got != expected)
        throw_bad_size(ctx, what, static_cast<long long>(expected),
                       static_cast<long long>(got));
}

// A supplied vector/list must hold exactly `expected` entries.
inline void check_size(size_t got, size_t expected, const char* ctx,
                       const char* what) {
    if (got != expected)
        throw_bad_size(ctx, what, static_cast<long long>(expected),
                       static_cast<long long>(got));
}

// Generic structural predicate with a custom message tail.
inline void check_require(bool ok, const char* ctx, const std::string& msg) {
    if (!ok) throw_structure(ctx, msg);
}

} // namespace detail
} // namespace lindblad
