// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/operators.hpp"
#include "lindblad/detail/memory_budget.hpp"
#include "lindblad/detail/pauli_rules.hpp"
#include "lindblad/detail/report.hpp"
#include "lindblad/detail/sv_runs.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace lindblad {

// =============================================================================
// PauliString
// =============================================================================

PauliString::PauliString(const std::string& p, Complex128 c) : pauli(p), coeff(c) {
    detail::check_pauli_label(pauli, "PauliString");
}

PauliString PauliString::compose(const PauliString& other) const {
    if (pauli.size() != other.pauli.size()) {
        throw std::invalid_argument("Pauli strings must have same length");
    }

    std::string result_pauli = pauli;
    Complex128 result_coeff = coeff * other.coeff;

    for (size_t i = 0; i < pauli.size(); ++i) {
        char a = pauli[i];
        char b = other.pauli[i];

        if (a == 'I') { result_pauli[i] = b; continue; }
        if (b == 'I') { result_pauli[i] = a; continue; }
        if (a == b) { result_pauli[i] = 'I'; continue; }

        // Non-trivial Pauli products
        if ((a == 'X' && b == 'Y') || (a == 'Y' && b == 'Z') || (a == 'Z' && b == 'X')) {
            result_coeff *= Complex128(0.0, 1.0);
            if (a == 'X' && b == 'Y') result_pauli[i] = 'Z';
            else if (a == 'Y' && b == 'Z') result_pauli[i] = 'X';
            else result_pauli[i] = 'Y';
        } else {
            result_coeff *= Complex128(0.0, -1.0);
            if (a == 'Y' && b == 'X') result_pauli[i] = 'Z';
            else if (a == 'Z' && b == 'Y') result_pauli[i] = 'X';
            else result_pauli[i] = 'Y';
        }
    }

    return {result_pauli, result_coeff};
}

PauliString PauliString::adjoint() const {
    return {pauli, coeff.conj()};
}

bool PauliString::commutes_with(const PauliString& other) const {
    if (pauli.size() != other.pauli.size()) return false;

    int anticommute_count = 0;
    for (size_t i = 0; i < pauli.size(); ++i) {
        if (pauli[i] != 'I' && other.pauli[i] != 'I' && pauli[i] != other.pauli[i]) {
            anticommute_count++;
        }
    }
    return (anticommute_count % 2 == 0);
}

// =============================================================================
// Pauli rules (detail/pauli_rules.hpp)
// =============================================================================

namespace detail {

namespace {

// Raises the refusal for `where`, the requester. When a run() asked through an
// observer, `entry_point` names it and the message starts with it; otherwise
// the requester is the entry point and the message reads as it always has.
[[noreturn]] void refuse(std::string_view entry_point, const char* where,
                         const std::string& body) {
    if (entry_point.empty()) raise<InvalidArgument>(where, body);
    raise<InvalidArgument>(entry_point, std::string(where) + ": " + body);
}

}  // namespace

void check_pauli_label(const std::string& label, const char* where,
                       std::string_view entry_point) {
    for (std::size_t q = 0; q < label.size(); ++q) {
        const char c = label[q];
        if (c != 'I' && c != 'X' && c != 'Y' && c != 'Z') {
            refuse(entry_point, where,
                "'" + std::string(1, c) + "' at position " +
                std::to_string(q) + " of '" + label + "' is not a Pauli; a Pauli "
                "string is written with I, X, Y and Z, uppercase");
        }
    }
}

int uniform_pauli_width(const std::vector<PauliString>& terms, const char* where,
                        std::string_view entry_point) {
    for (const PauliString& term : terms) check_pauli_label(term.pauli, where, entry_point);
    if (terms.empty()) return -1;
    const int width = terms[0].n_qubits();
    for (std::size_t i = 1; i < terms.size(); ++i) {
        if (terms[i].n_qubits() != width) {
            refuse(entry_point, where,
                "term " + std::to_string(i) + " ('" +
                terms[i].pauli + "') is " + std::to_string(terms[i].n_qubits()) +
                " qubits wide but term 0 ('" + terms[0].pauli + "') is " +
                std::to_string(width) + "; every term of one operator covers the "
                "same qubits");
        }
    }
    return width;
}

int required_pauli_width(const std::vector<PauliString>& terms, const char* where,
                         std::string_view entry_point) {
    const int width = uniform_pauli_width(terms, where, entry_point);
    if (width < 0) {
        refuse(entry_point, where,
            "the operator has no terms, so it has no width; "
            "the zero operator on n qubits is SparsePauliOp::zero(n)");
    }
    return width;
}

void check_hermitian(const SparsePauliOp& op, const char* where,
                     std::string_view entry_point) {
    // Real term by term is the common case and needs no merging.
    bool all_real = true;
    for (const PauliString& term : op.terms) {
        if (std::abs(term.coeff.imag) > DEFAULT_PHYSICAL_ATOL) {
            all_real = false;
            break;
        }
    }
    if (all_real) return;

    // Imaginary parts on repeated labels can cancel, so only the merged sum
    // decides.
    std::unordered_map<std::string, Complex128> merged;
    for (const PauliString& term : op.terms) merged[term.pauli] += term.coeff;
    for (const PauliString& term : op.terms) {
        const Complex128 total = merged[term.pauli];
        if (std::abs(total.imag) > DEFAULT_PHYSICAL_ATOL) {
            refuse(entry_point, where,
                "the operator is not Hermitian: label '" +
                term.pauli + "' has coefficient " + std::to_string(total.real) +
                (total.imag < 0.0 ? " - " : " + ") +
                std::to_string(std::abs(total.imag)) +
                "i once repeated labels are merged. Its expectation value is "
                "complex, and this returns a real number");
        }
    }
}

void check_observable(const SparsePauliOp& op, int n_qubits, const char* where,
                      const char* against, std::string_view entry_point) {
    (void)required_pauli_width(op.terms, where, entry_point);
    for (std::size_t i = 0; i < op.terms.size(); ++i) {
        const int width = op.terms[i].n_qubits();
        if (width != n_qubits) {
            refuse(entry_point, where,
                "term " + std::to_string(i) + " ('" +
                op.terms[i].pauli + "') is " + std::to_string(width) +
                " qubits wide, which does not match the " + std::to_string(n_qubits) +
                " qubit " + against + "; a term names exactly one Pauli per qubit");
        }
    }
    check_hermitian(op, where, entry_point);
}

}  // namespace detail

// =============================================================================
// SparsePauliOp
// =============================================================================

SparsePauliOp::SparsePauliOp(const std::vector<PauliString>& terms) : terms(terms) {
    (void)detail::uniform_pauli_width(this->terms, "SparsePauliOp");
}

SparsePauliOp SparsePauliOp::simplify(double atol) const {
    const int width = detail::uniform_pauli_width(terms, "SparsePauliOp::simplify");

    std::unordered_map<std::string, Complex128> merged;
    for (const auto& term : terms) {
        merged[term.pauli] += term.coeff;
    }

    SparsePauliOp result;
    for (const auto& [label, coeff] : merged) {
        if (coeff.norm_sq() > atol * atol) {
            result.terms.push_back({label, coeff});
        }
    }

    // Everything cancelled: the result is the zero operator, which keeps the
    // width the terms had. Returning no terms would lose it, and an operator
    // with no terms is refused wherever a width is needed, so H - H must stay
    // evaluable as 0. An operator that had no terms to begin with has no width
    // to keep and stays empty.
    if (result.terms.empty() && width >= 0) {
        result.terms.push_back({std::string(static_cast<std::size_t>(width), 'I'),
                                Complex128(0.0, 0.0)});
    }
    return result;
}

SparsePauliOp SparsePauliOp::compose(const SparsePauliOp& other) const {
    SparsePauliOp result;
    for (const auto& a : terms) {
        for (const auto& b : other.terms) {
            result.terms.push_back(a.compose(b));
        }
    }
    return result.simplify();
}

SparsePauliOp SparsePauliOp::adjoint() const {
    SparsePauliOp result;
    for (const auto& term : terms) {
        result.terms.push_back(term.adjoint());
    }
    return result;
}

SparsePauliOp SparsePauliOp::tensor(const SparsePauliOp& other) const {
    SparsePauliOp result;
    for (const auto& a : terms) {
        for (const auto& b : other.terms) {
            result.terms.push_back({a.pauli + b.pauli, a.coeff * b.coeff});
        }
    }
    return result;
}

SparsePauliOp SparsePauliOp::operator+(const SparsePauliOp& other) const {
    // An operand with no terms has no width and adds nothing, so only two
    // operands that both have terms can disagree.
    const int lhs = detail::uniform_pauli_width(terms, "SparsePauliOp::operator+");
    const int rhs = detail::uniform_pauli_width(other.terms, "SparsePauliOp::operator+");
    if (lhs >= 0 && rhs >= 0 && lhs != rhs) {
        throw std::invalid_argument(
            "SparsePauliOp::operator+: the left operand is " + std::to_string(lhs) +
            " qubits wide and the right is " + std::to_string(rhs) +
            "; a sum acts on one register");
    }

    SparsePauliOp result;
    result.terms = terms;
    result.terms.insert(result.terms.end(), other.terms.begin(), other.terms.end());
    return result.simplify();
}

SparsePauliOp SparsePauliOp::operator*(double scalar) const {
    SparsePauliOp result;
    for (const auto& term : terms) {
        result.terms.push_back({term.pauli, term.coeff * scalar});
    }
    return result;
}

std::vector<Complex128> SparsePauliOp::to_matrix() const {
    // Every term's mask indexes a dim x dim matrix, so a term wider than the
    // first would write past it.
    const int nq = detail::required_pauli_width(terms, "SparsePauliOp::to_matrix");
    size_t dim = 1ULL << nq;
    std::vector<Complex128> matrix(dim * dim, Complex128(0.0, 0.0));

    // For each Pauli term P, P|j⟩ = phase(j) * |j XOR x_mask⟩.
    // This fills one non-zero entry per column in O(2^n) per term,
    // vs O(n * 4^n) for the tensor-product construction.
    for (const auto& term : terms) {
        const int n = term.n_qubits();
        uint64_t x_mask = 0, z_mask = 0, y_mask = 0;
        for (int q = 0; q < n; ++q) {
            char c = term.pauli[q];
            if (c == 'X') { x_mask |= (1ULL << q); }
            else if (c == 'Z') { z_mask |= (1ULL << q); }
            else if (c == 'Y') {
                x_mask |= (1ULL << q);
                y_mask |= (1ULL << q);
                z_mask |= (1ULL << q);
            }
        }

        // P = i^{#Y} * X^{x_mask} * Z^{z_mask} (because Y = i*X*Z): the i
        // factor is a CONSTANT per term, one i for every Y in the string.
        // Applying i^popcount(j & y_mask) per column instead drops the factor
        // whenever the Y-position input bit is 0, which makes every
        // Y-containing matrix non-Hermitian (issue #30). Fold the constant into
        // the coefficient once, before the column loop.
        const int y_count = LINDBLAD_POPCOUNT64(y_mask) & 3;
        double ty_r = 1.0, ty_i = 0.0;
        switch (y_count) {
            case 1: ty_r =  0.0; ty_i =  1.0; break;
            case 2: ty_r = -1.0; ty_i =  0.0; break;
            case 3: ty_r =  0.0; ty_i = -1.0; break;
            default: break;
        }
        const Complex128 tcoeff(
            ty_r * term.coeff.real - ty_i * term.coeff.imag,
            ty_r * term.coeff.imag + ty_i * term.coeff.real
        );

        for (size_t j = 0; j < dim; ++j) {
            const size_t row = j ^ x_mask;
            const int z_parity = LINDBLAD_POPCOUNT64(j & z_mask) & 1;
            // P|j> = i^{#Y} * (-1)^{popcount(j & z_mask)} |j XOR x_mask>
            // (z_mask includes the Y positions: the Z factor of Y acts there).
            if (z_parity) {
                matrix[row * dim + j] += Complex128(-tcoeff.real, -tcoeff.imag);
            } else {
                matrix[row * dim + j] += tcoeff;
            }
        }
    }

    return matrix;
}

// =============================================================================
// Expectation values - one sweep per group of terms sharing an X mask
// =============================================================================
// A Pauli string is P = i^{#Y} X^x Z^z (Y = iXZ, so z holds the Y positions
// too), and P|j⟩ = i^{#Y} (-1)^{popcount(j & z)} |j XOR x⟩. So
//
//   ⟨ψ|P|ψ⟩ = i^{#Y} sum_k conj(ψ_k) ψ_j (-1)^{popcount(j & z)},  j = k XOR x:
//
// only the pair (k, k XOR x) meets. Terms with the same x pair the same
// amplitudes, so one sweep serves all of them: each pair's product
// conj(ψ_k) ψ_j is formed once, and each term adds it with its own sign. The
// factor i^{#Y} is one constant per term, folded into its coefficient once;
// applying it per amplitude instead drops it wherever a Y position of the
// input bit is 0, which makes every Y-containing term wrong (issue #30).
namespace {

struct PauliGroup {
    std::uint64_t x = 0;
    std::vector<std::uint64_t> z;  // per term
    std::vector<double> cr, ci;    // per term: coeff * i^{#Y}
};

// The terms grouped by X mask, groups in order of first appearance and terms
// in their own order within a group.
std::vector<PauliGroup> group_by_x(const std::vector<PauliString>& terms) {
    std::vector<PauliGroup> groups;
    std::unordered_map<std::uint64_t, std::size_t> index;
    for (const auto& term : terms) {
        std::uint64_t x = 0, z = 0, y = 0;
        for (int q = 0; q < term.n_qubits(); ++q) {
            const std::uint64_t bit = std::uint64_t{1} << q;
            const char c = term.pauli[static_cast<std::size_t>(q)];
            if (c == 'X') x |= bit;
            else if (c == 'Z') z |= bit;
            else if (c == 'Y') { x |= bit; z |= bit; y |= bit; }
        }
        // i^{#Y}: 0 -> 1, 1 -> i, 2 -> -1, 3 -> -i
        double pr = 1.0, pi = 0.0;
        switch (LINDBLAD_POPCOUNT64(y) & 3) {
            case 1: pr = 0.0; pi = 1.0; break;
            case 2: pr = -1.0; break;
            case 3: pr = 0.0; pi = -1.0; break;
            default: break;
        }
        const auto [at, fresh] = index.try_emplace(x, groups.size());
        if (fresh) groups.emplace_back().x = x;
        PauliGroup& g = groups[at->second];
        g.z.push_back(z);
        g.cr.push_back(pr * term.coeff.real - pi * term.coeff.imag);
        g.ci.push_back(pr * term.coeff.imag + pi * term.coeff.real);
    }
    return groups;
}

// Amplitudes per tile: a tile's pair products, two arrays of this many
// doubles (8 KiB), stay in L1 while every term of the group passes over them.
constexpr std::size_t kExpectationTile = 512;

// sum over the group's terms of Re(c_t * i^{#Y} * the term's sum over k), for
// k in [b, e). Each term's pass over a tile is a plain reduction the compiler
// vectorises: a sign from a popcount, two multiply-adds.
double group_share(const double* rp, const double* ip, const PauliGroup& g,
                   std::size_t b, std::size_t e) {
    const std::uint64_t x = g.x;
    // One term shares its products with nothing, so it takes them straight
    // into its sums: storing them to the tile and reading them back is
    // traffic with nothing to gain, and on a state larger than the caches,
    // where the pass waits on memory, it is a few percent of the pass.
    if (g.z.size() == 1) {
        const std::uint64_t z = g.z[0];
        double sr = 0.0, si = 0.0;
#if !defined(_MSC_VER)
        #pragma omp simd reduction(+:sr, si)
#endif
        for (std::size_t k = b; k < e; ++k) {
            const std::size_t j = k ^ x;
            const double sign =
                1.0 - 2.0 * static_cast<double>(LINDBLAD_POPCOUNT64(j & z) & 1);
            sr += sign * (rp[k] * rp[j] + ip[k] * ip[j]);
            si += sign * (rp[k] * ip[j] - ip[k] * rp[j]);
        }
        return g.cr[0] * sr - g.ci[0] * si;
    }
    double dr[kExpectationTile];
    double di[kExpectationTile];
    double total = 0.0;
    for (std::size_t s = b; s < e; s += kExpectationTile) {
        const std::size_t len = std::min(kExpectationTile, e - s);
        // conj(ψ_k) ψ_j = (r_k r_j + i_k i_j) + i (r_k i_j - i_k r_j)
        for (std::size_t i = 0; i < len; ++i) {
            const std::size_t k = s + i;
            const std::size_t j = k ^ x;
            dr[i] = rp[k] * rp[j] + ip[k] * ip[j];
            di[i] = rp[k] * ip[j] - ip[k] * rp[j];
        }
        for (std::size_t t = 0; t < g.z.size(); ++t) {
            const std::uint64_t z = g.z[t];
            double sr = 0.0, si = 0.0;
#if !defined(_MSC_VER)
            #pragma omp simd reduction(+:sr, si)
#endif
            for (std::size_t i = 0; i < len; ++i) {
                const std::uint64_t j = (s + i) ^ x;
                const double sign =
                    1.0 - 2.0 * static_cast<double>(LINDBLAD_POPCOUNT64(j & z) & 1);
                sr += sign * dr[i];
                si += sign * di[i];
            }
            total += g.cr[t] * sr - g.ci[t] * si;
        }
    }
    return total;
}

// ⟨ψ|H|ψ⟩ for H given as its groups. parallel = whether each group's sweep
// may use a team: its shares are summed in thread order (detail::reduce_shares),
// so the last bits depend on the team size and on nothing else. A group's
// work counts one pass per term.
double expectation_of_groups(const double* rp, const double* ip, std::size_t dim,
                             const std::vector<PauliGroup>& groups, bool parallel) {
    double result = 0.0;
    for (const PauliGroup& g : groups) {
        if (!parallel) {
            result += group_share(rp, ip, g, 0, dim);
            continue;
        }
        result += detail::reduce_shares(
            dim, detail::saturating_mul(dim, g.z.size()), 0.0,
            [rp, ip, &g](std::size_t b, std::size_t e) { return group_share(rp, ip, g, b, e); },
            [](double acc, double part) { return acc + part; });
    }
    return result;
}

}  // namespace

double SparsePauliOp::expectation_value(const Statevector& sv) const {
    // ⟨ψ|H|ψ⟩ = sum_t coeff_t ⟨ψ|P_t|ψ⟩, without copying the state: one sweep
    // per distinct X mask (see Expectation values above).
    //
    // The masks come from each term's own string, so the width check is what
    // keeps j = k ^ x_mask inside the amplitude arrays.
    detail::check_observable(*this, sv.n_qubits, "SparsePauliOp::expectation_value");
    return expectation_of_groups(sv.real_parts, sv.imag_parts, sv.dim, group_by_x(terms),
                                 true);
}

std::vector<double> SparsePauliOp::expectation_value_batch(
    const std::vector<const Statevector*>& states
) const {
    // Checked before the batch size, so an operator with no terms is refused
    // even when there is nothing to evaluate it on, and every state is checked
    // before any of them is evaluated.
    (void)detail::required_pauli_width(terms, "SparsePauliOp::expectation_value_batch");
    for (const Statevector* state : states) {
        detail::check_observable(*this, state->n_qubits,
                                       "SparsePauliOp::expectation_value_batch");
    }

    const size_t M = states.size();
    std::vector<double> results(M, 0.0);
    if (M == 0) return results;

    // The groups are built once for the whole batch. With at least one state
    // per thread, each thread takes whole states and sweeps them alone; with
    // fewer, the states go one at a time, each swept by the team, so no
    // thread waits while a batch smaller than the team runs.
    const std::vector<PauliGroup> groups = group_by_x(terms);
    if (M >= hw::available_threads()) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int sii = 0; sii < static_cast<int>(M); ++sii) {
            const Statevector* sv = states[static_cast<size_t>(sii)];
            results[static_cast<size_t>(sii)] =
                expectation_of_groups(sv->real_parts, sv->imag_parts, sv->dim, groups, false);
        }
    } else {
        for (size_t si = 0; si < M; ++si) {
            const Statevector* sv = states[si];
            results[si] =
                expectation_of_groups(sv->real_parts, sv->imag_parts, sv->dim, groups, true);
        }
    }
    return results;
}

int SparsePauliOp::n_qubits() const {
    if (terms.empty()) return 0;
    return terms[0].n_qubits();
}

SparsePauliOp SparsePauliOp::from_list(
    const std::vector<std::pair<std::string, Complex128>>& label_coeff
) {
    SparsePauliOp result;
    for (const auto& [label, coeff] : label_coeff) {
        result.terms.push_back({label, coeff});
    }
    (void)detail::uniform_pauli_width(result.terms, "SparsePauliOp::from_list");
    return result;
}

SparsePauliOp SparsePauliOp::identity(int n_qubits) {
    return SparsePauliOp({{std::string(n_qubits, 'I'), Complex128(1.0, 0.0)}});
}

SparsePauliOp SparsePauliOp::zero(int n_qubits) {
    return SparsePauliOp({{std::string(n_qubits, 'I'), Complex128(0.0, 0.0)}});
}

} // namespace lindblad

