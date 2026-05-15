#pragma once

#include "lindblad/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace lindblad {

// =============================================================================
// QuditStatevector — dense state vector for n qudits each of dimension d.
//
// Indexing (little-endian mixed-radix, same convention as qubit Statevector):
//   index = x_0 * d^0 + x_1 * d^1 + ... + x_{n-1} * d^{n-1}
//   Stride of qudit q = d^q.
//
// dim = d^n_qudits. Amplitudes stored as a dense std::vector<Complex128>.
// =============================================================================

class QuditStatevector {
public:
    int n_qudits;
    int d;       // local Hilbert space dimension per qudit
    size_t dim;  // d^n_qudits

    std::vector<Complex128> amplitudes;  // length dim; amplitude[idx] = <idx|psi>

    // Construct and initialise to |0...0>
    QuditStatevector(int n_qudits, int d);

    // Set to |0...0> (amplitude[0] = 1, rest = 0)
    void initialize();

    // Renormalise so that norm_sq() == 1.0
    void normalize();

    // Sum of |amplitude[i]|^2 — should be 1.0 for a valid state
    double norm_sq() const;

    // Apply a d×d unitary U to qudit q.
    // U is stored row-major: U[row*d + col] is the (row,col) entry.
    void apply_1qudit(int q, const std::vector<Complex128>& U);

    // Apply a d²×d² unitary U to qudits (q0, q1), q0 != q1.
    // Row index r = new_q0*d + new_q1; col index c = old_q0*d + old_q1.
    // q0 is the "first" qudit (control) and q1 the "second" (target) in the
    // tensor-product ordering used to construct U.
    void apply_2qudit(int q0, int q1, const std::vector<Complex128>& U);

    // Apply a d^k × d^k unitary U to k distinct qudits in the order given by
    // `qudits`. Row index in U is r = sum_i new_x_{qudits[i]} * d^(k-1-i)
    // (i.e., the i-th qudit in the list is the most significant when i is
    // smallest). U is row-major and must have size d^(2k).
    // qudits must contain distinct indices in [0, n_qudits).
    void apply_kqudit(const std::vector<int>& qudits,
                      const std::vector<Complex128>& U);

    // Apply an oracle function f: Z_d^{n_query} -> Z_d^{n_output} as
    //   |x_0..x_{n_query-1}> |y_0..y_{n_output-1}>
    //     -> |x>|(y + f(x)) mod d>  (componentwise)
    // The query qudits are at indices [0, n_query) and the output qudits are
    // at indices [n_query, n_query + n_output).
    // f must return a vector of exactly n_output digits, each in [0, d).
    void apply_function_oracle(
        int n_query, int n_output,
        const std::function<std::vector<int>(const std::vector<int>&)>& f);

    // Apply a per-basis-state phase: amplitude[idx] *= phase_fn(digits(idx))
    // for every basis state.  The caller is responsible for `phase_fn` returning
    // a unit-modulus complex number so the operation is unitary.
    //
    // Used by Grover (target marker is -1; non-target is +1) and by Grover's
    // diffusion R_0 = 2|0><0| - I (phase +1 on |0..0>, -1 elsewhere).
    void apply_phase_oracle(
        const std::function<Complex128(const std::vector<int>&)>& phase_fn);

    // Sample one measurement outcome from the probability distribution |amplitude[i]|^2.
    // Returns a vector of n_qudits symbols, each in {0..d-1}, in qudit-index order
    // (result[q] is the measured value of qudit q).
    std::vector<int> measure(uint64_t seed = 0) const;

    // Decode a flat state index into per-qudit digit values.
    // result[q] = (idx / d^q) % d
    static std::vector<int> index_to_digits(size_t idx, int d, int n_qudits);

    // Encode per-qudit digit values into a flat state index.
    // idx = sum_q digits[q] * d^q
    static size_t digits_to_index(const std::vector<int>& digits, int d);

    // d^exp, computed iteratively (exact integer arithmetic)
    static size_t ipow(size_t base, int exp) noexcept;
};

} // namespace lindblad
