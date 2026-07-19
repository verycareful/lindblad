#include "lindblad/qudit/qudit_statevector.hpp"

#include "lindblad/detail/validate.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace lindblad {

// ---------------------------------------------------------------------------
// ipow — exact integer power, avoids floating-point rounding
// ---------------------------------------------------------------------------

size_t QuditStatevector::ipow(size_t base, int exp) noexcept {
    size_t result = 1;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

// ---------------------------------------------------------------------------
// Constructor / initialisation
// ---------------------------------------------------------------------------

QuditStatevector::QuditStatevector(int n_qudits_, int d_)
    : n_qudits(n_qudits_)
    , d(d_)
    , dim(ipow(static_cast<size_t>(d_), n_qudits_))
{
    if (d_ < 2)
        throw std::invalid_argument("QuditStatevector: d must be >= 2");
    if (n_qudits_ < 1)
        throw std::invalid_argument("QuditStatevector: n_qudits must be >= 1");
    amplitudes.resize(dim, Complex128(0.0, 0.0));
    amplitudes[0] = Complex128(1.0, 0.0);
}

void QuditStatevector::initialize() {
    std::fill(amplitudes.begin(), amplitudes.end(), Complex128(0.0, 0.0));
    amplitudes[0] = Complex128(1.0, 0.0);
}

void QuditStatevector::normalize() {
    double n = std::sqrt(norm_sq());
    if (n < 1e-15) return;
    double inv = 1.0 / n;
    for (auto& a : amplitudes) {
        a.real *= inv;
        a.imag *= inv;
    }
}

double QuditStatevector::norm_sq() const {
    double s = 0.0;
    for (const auto& a : amplitudes)
        s += a.real * a.real + a.imag * a.imag;
    return s;
}

// ---------------------------------------------------------------------------
// apply_1qudit — apply d×d unitary U to qudit q
// ---------------------------------------------------------------------------
// stride = d^q is the gap between consecutive values of qudit q in the flat array.
// We iterate over all independent "groups" of d amplitudes that share the same
// values for every qudit except q, then multiply each group by U in-place.
//
// outer steps by d^(q+1) = stride*d  (one full qudit-q cycle per outer block)
// inner steps by 1                   (all sub-qudit positions)
// base = outer + inner               (first element of this group, qudit q = 0)
// amplitudes[base + k*stride]        (element where qudit q = k)

void QuditStatevector::apply_1qudit(int q, const std::vector<Complex128>& U) {
    detail::check_qudit(q, n_qudits, "QuditStatevector::apply_1qudit");
    detail::check_size(U.size(), static_cast<size_t>(d) * static_cast<size_t>(d),
                       "QuditStatevector::apply_1qudit", "matrix");
    const size_t stride = ipow(static_cast<size_t>(d), q);
    const size_t block  = stride * static_cast<size_t>(d);  // = d^(q+1)
    const long long n_outer = static_cast<long long>(dim / block);

    // Parallelise over outer blocks (audit F-17); scratch is per-thread.
    #pragma omp parallel if(dim >= (1u << 12))
    {
        std::vector<Complex128> old_amp(static_cast<size_t>(d));
        std::vector<Complex128> new_amp(static_cast<size_t>(d));
        #pragma omp for schedule(static)
        for (long long ob = 0; ob < n_outer; ++ob) {
            const size_t outer = static_cast<size_t>(ob) * block;
            for (size_t inner = 0; inner < stride; ++inner) {
                const size_t base = outer + inner;
                for (int k = 0; k < d; ++k)
                    old_amp[static_cast<size_t>(k)] =
                        amplitudes[base + static_cast<size_t>(k) * stride];
                for (int k = 0; k < d; ++k) {
                    Complex128 acc(0.0, 0.0);
                    for (int j = 0; j < d; ++j)
                        acc += U[static_cast<size_t>(k * d + j)]
                             * old_amp[static_cast<size_t>(j)];
                    new_amp[static_cast<size_t>(k)] = acc;
                }
                for (int k = 0; k < d; ++k)
                    amplitudes[base + static_cast<size_t>(k) * stride] =
                        new_amp[static_cast<size_t>(k)];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// apply_2qudit — apply d²×d² unitary U to qudits (q0, q1)
// ---------------------------------------------------------------------------
// Convention (project LSB-first, docs/Architecture.md "Conventions"): the
// FIRST argument is the LEAST significant digit of the matrix index:
//   row r = new_q1 * d + new_q0;  col c = old_q1 * d + old_q0.
// This mirrors the qubit-layer apply_unitary contract (index bit i =
// qubits[i]) at d = 2.
//
// Base indices (both target digits 0) are enumerated directly over the
// d^(n-2) combinations of the remaining digits instead of scanning all dim
// indices with divisions.

void QuditStatevector::apply_2qudit(int q0, int q1,
                                    const std::vector<Complex128>& U) {
    detail::check_qudit(q0, n_qudits, "QuditStatevector::apply_2qudit");
    detail::check_qudit(q1, n_qudits, "QuditStatevector::apply_2qudit");
    detail::check_distinct2(q0, q1, "QuditStatevector::apply_2qudit", "qudits");
    detail::check_size(U.size(),
                       static_cast<size_t>(d) * d * d * d,
                       "QuditStatevector::apply_2qudit", "matrix");

    const size_t stride0 = ipow(static_cast<size_t>(d), q0);
    const size_t stride1 = ipow(static_cast<size_t>(d), q1);
    const int dd = d * d;

    // Strides of the non-target qudits, for direct base enumeration.
    std::vector<size_t> other_strides;
    other_strides.reserve(static_cast<size_t>(n_qudits));
    for (int q = 0; q < n_qudits; ++q)
        if (q != q0 && q != q1)
            other_strides.push_back(ipow(static_cast<size_t>(d), q));
    const long long n_bases = static_cast<long long>(dim / (static_cast<size_t>(dd)));

    #pragma omp parallel if(dim >= (1u << 12))
    {
        std::vector<Complex128> old_amp(static_cast<size_t>(dd));
        std::vector<Complex128> new_amp(static_cast<size_t>(dd));
        #pragma omp for schedule(static)
        for (long long b = 0; b < n_bases; ++b) {
            size_t idx = 0;
            size_t rem = static_cast<size_t>(b);
            for (size_t i = 0; i < other_strides.size(); ++i) {
                idx += (rem % static_cast<size_t>(d)) * other_strides[i];
                rem /= static_cast<size_t>(d);
            }

            // Extract d² amplitudes: old_amp[x1*d + x0] for (x0,x1) in {0..d-1}^2
            for (int x0 = 0; x0 < d; ++x0)
                for (int x1 = 0; x1 < d; ++x1)
                    old_amp[static_cast<size_t>(x1 * d + x0)] =
                        amplitudes[idx
                            + static_cast<size_t>(x0) * stride0
                            + static_cast<size_t>(x1) * stride1];

            // Matrix multiply: new = U * old  (U is d²×d², row-major)
            for (int r = 0; r < dd; ++r) {
                Complex128 acc(0.0, 0.0);
                for (int c = 0; c < dd; ++c)
                    acc += U[static_cast<size_t>(r * dd + c)]
                         * old_amp[static_cast<size_t>(c)];
                new_amp[static_cast<size_t>(r)] = acc;
            }

            // Write back
            for (int x0 = 0; x0 < d; ++x0)
                for (int x1 = 0; x1 < d; ++x1)
                    amplitudes[idx
                        + static_cast<size_t>(x0) * stride0
                        + static_cast<size_t>(x1) * stride1] =
                            new_amp[static_cast<size_t>(x1 * d + x0)];
        }
    }
}

// ---------------------------------------------------------------------------
// apply_kqudit — apply d^k × d^k unitary U to k distinct qudits
// ---------------------------------------------------------------------------
// Convention (project LSB-first, docs/Architecture.md "Conventions"):
// U[row, col] where row/col are k-digit base-d indices with qudits[0] as the
// LEAST significant digit and qudits[k-1] as the most significant. This is
// the d-level generalisation of the qubit-layer apply_unitary contract
// (index bit i = qubits[i]).
//
// For each "base" index (where every qudit in `qudits` has value 0):
//   - extract d^k amplitudes from the joint subspace
//   - multiply by U
//   - write back

void QuditStatevector::apply_kqudit(const std::vector<int>& qudits,
                                    const std::vector<Complex128>& U)
{
    const int k = static_cast<int>(qudits.size());
    if (k < 1) return;
    if (k == 1) { apply_1qudit(qudits[0], U); return; }
    if (k == 2) { apply_2qudit(qudits[0], qudits[1], U); return; }

    detail::check_qudits(qudits, n_qudits, "QuditStatevector::apply_kqudit");
    {
        size_t dk = 1;
        for (int i = 0; i < k; ++i) dk *= static_cast<size_t>(d);
        detail::check_size(U.size(), dk * dk,
                           "QuditStatevector::apply_kqudit", "matrix");
    }

    // Validate distinctness
    for (int i = 0; i < k; ++i)
        for (int j = i + 1; j < k; ++j)
            if (qudits[static_cast<size_t>(i)] == qudits[static_cast<size_t>(j)])
                throw std::invalid_argument(
                    "apply_kqudit: duplicate qudit index");

    // Pre-compute strides for each qudit in the list
    std::vector<size_t> strides(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i)
        strides[static_cast<size_t>(i)] =
            ipow(static_cast<size_t>(d),
                 qudits[static_cast<size_t>(i)]);

    // dk = d^k = total subspace dimension
    size_t dk = 1;
    for (int i = 0; i < k; ++i) dk *= static_cast<size_t>(d);

    // Build a list of (subspace_index, flat_offset) pairs:
    // subspace_index ranges over [0, dk); qudits[0] is the LEAST significant
    // digit of v (LSB-first), so digit_i = (v / d^i) % d maps to qudits[i].
    // offset = sum_i digit_i * strides[i]. Precompute these offsets once.
    std::vector<size_t> sub_offset(dk);
    for (size_t v = 0; v < dk; ++v) {
        size_t off = 0;
        size_t v_rem = v;
        for (int i = 0; i < k; ++i) {
            const size_t digit = v_rem % static_cast<size_t>(d);
            v_rem /= static_cast<size_t>(d);
            off += digit * strides[static_cast<size_t>(i)];
        }
        sub_offset[v] = off;
    }

    // Enumerate base indices (all qudits in `qudits` at digit 0) directly
    // over the d^(n-k) combinations of the remaining digits.
    std::vector<size_t> other_strides;
    other_strides.reserve(static_cast<size_t>(n_qudits));
    for (int q = 0; q < n_qudits; ++q) {
        bool is_target = false;
        for (int i = 0; i < k; ++i)
            if (qudits[static_cast<size_t>(i)] == q) { is_target = true; break; }
        if (!is_target)
            other_strides.push_back(ipow(static_cast<size_t>(d), q));
    }
    const long long n_bases = static_cast<long long>(dim / dk);

    #pragma omp parallel if(dim >= (1u << 12))
    {
        std::vector<Complex128> old_amp(dk);
        std::vector<Complex128> new_amp(dk);
        #pragma omp for schedule(static)
        for (long long b = 0; b < n_bases; ++b) {
            size_t idx = 0;
            size_t rem = static_cast<size_t>(b);
            for (size_t i = 0; i < other_strides.size(); ++i) {
                idx += (rem % static_cast<size_t>(d)) * other_strides[i];
                rem /= static_cast<size_t>(d);
            }

            for (size_t v = 0; v < dk; ++v)
                old_amp[v] = amplitudes[idx + sub_offset[v]];

            for (size_t r = 0; r < dk; ++r) {
                Complex128 acc(0.0, 0.0);
                for (size_t c = 0; c < dk; ++c)
                    acc += U[r * dk + c] * old_amp[c];
                new_amp[r] = acc;
            }

            for (size_t v = 0; v < dk; ++v)
                amplitudes[idx + sub_offset[v]] = new_amp[v];
        }
    }
}

// ---------------------------------------------------------------------------
// apply_function_oracle — apply U_f: |x>|y> -> |x>|(y + f(x)) mod d>
// ---------------------------------------------------------------------------
// f maps query digits (size n_query) to output digits (size n_output).
// Query qudits are at indices [0, n_query); output qudits at
// [n_query, n_query + n_output). This is a permutation of basis indices
// (provided every (x, y) has a unique image), so the result is unitary
// regardless of f.

void QuditStatevector::apply_function_oracle(
    int n_query, int n_output,
    const std::function<std::vector<int>(const std::vector<int>&)>& f)
{
    if (n_query < 1 || n_output < 0)
        throw std::invalid_argument(
            "apply_function_oracle: n_query >= 1 and n_output >= 0 required");
    if (n_query + n_output > n_qudits)
        throw std::invalid_argument(
            "apply_function_oracle: n_query + n_output exceeds n_qudits");

    std::vector<Complex128> new_amps(dim, Complex128(0.0, 0.0));

    for (size_t idx = 0; idx < dim; ++idx) {
        auto digits = index_to_digits(idx, d, n_qudits);
        std::vector<int> x(
            digits.begin(),
            digits.begin() + n_query);
        const auto fx = f(x);
        if (static_cast<int>(fx.size()) != n_output)
            throw std::invalid_argument(
                "apply_function_oracle: f returned wrong number of digits");

        // y' = (y + f(x)) mod d, componentwise on output qudits
        for (int i = 0; i < n_output; ++i) {
            const int v = fx[static_cast<size_t>(i)];
            if (v < 0 || v >= d)
                throw std::invalid_argument(
                    "apply_function_oracle: f returned value out of [0, d)");
            const size_t qi = static_cast<size_t>(n_query + i);
            digits[qi] = (digits[qi] + v) % d;
        }

        const size_t new_idx = digits_to_index(digits, d);
        new_amps[new_idx] += amplitudes[idx];
    }

    amplitudes = std::move(new_amps);
}

// ---------------------------------------------------------------------------
// apply_phase_oracle — apply per-basis phase: amplitude[idx] *= phase(digits)
// ---------------------------------------------------------------------------
// Used by Grover (target marker, diffusion R_0). The caller guarantees
// |phase| == 1, so this is unitary.

void QuditStatevector::apply_phase_oracle(
    const std::function<Complex128(const std::vector<int>&)>& phase_fn)
{
    for (size_t idx = 0; idx < dim; ++idx) {
        auto digits = index_to_digits(idx, d, n_qudits);
        const Complex128 phase = phase_fn(digits);
        amplitudes[idx] = amplitudes[idx] * phase;
    }
}

// ---------------------------------------------------------------------------
// measure — sample one outcome from |amplitude[i]|^2
// ---------------------------------------------------------------------------

std::vector<int> QuditStatevector::measure(uint64_t seed) const {
    std::mt19937_64 rng(seed == 0
        ? static_cast<uint64_t>(std::random_device{}())
        : seed);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    const double roll = udist(rng);
    double cumulative = 0.0;
    size_t chosen = dim - 1;  // fallback for floating-point rounding at tail
    for (size_t i = 0; i < dim; ++i) {
        cumulative += amplitudes[i].real * amplitudes[i].real
                    + amplitudes[i].imag * amplitudes[i].imag;
        if (roll <= cumulative) {
            chosen = i;
            break;
        }
    }

    return index_to_digits(chosen, d, n_qudits);
}

// ---------------------------------------------------------------------------
// Index / digit conversion
// ---------------------------------------------------------------------------

std::vector<int> QuditStatevector::index_to_digits(size_t idx, int d,
                                                   int n_qudits) {
    std::vector<int> digits(static_cast<size_t>(n_qudits));
    size_t rem = idx;
    for (int q = 0; q < n_qudits; ++q) {
        digits[static_cast<size_t>(q)] = static_cast<int>(
            rem % static_cast<size_t>(d));
        rem /= static_cast<size_t>(d);
    }
    return digits;
}

size_t QuditStatevector::digits_to_index(const std::vector<int>& digits, int d) {
    size_t idx = 0;
    size_t stride = 1;
    for (int x : digits) {
        idx += static_cast<size_t>(x) * stride;
        stride *= static_cast<size_t>(d);
    }
    return idx;
}

} // namespace lindblad
