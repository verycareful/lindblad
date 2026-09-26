// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/operators.hpp"
#include "lindblad/detail/pauli_rules.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/gates.hpp"
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

void check_pauli_label(const std::string& label, const char* where) {
    for (std::size_t q = 0; q < label.size(); ++q) {
        const char c = label[q];
        if (c != 'I' && c != 'X' && c != 'Y' && c != 'Z') {
            throw std::invalid_argument(
                std::string(where) + ": '" + std::string(1, c) + "' at position " +
                std::to_string(q) + " of '" + label + "' is not a Pauli; a Pauli "
                "string is written with I, X, Y and Z, uppercase");
        }
    }
}

int uniform_pauli_width(const std::vector<PauliString>& terms, const char* where) {
    for (const PauliString& term : terms) check_pauli_label(term.pauli, where);
    if (terms.empty()) return -1;
    const int width = terms[0].n_qubits();
    for (std::size_t i = 1; i < terms.size(); ++i) {
        if (terms[i].n_qubits() != width) {
            throw std::invalid_argument(
                std::string(where) + ": term " + std::to_string(i) + " ('" +
                terms[i].pauli + "') is " + std::to_string(terms[i].n_qubits()) +
                " qubits wide but term 0 ('" + terms[0].pauli + "') is " +
                std::to_string(width) + "; every term of one operator covers the "
                "same qubits");
        }
    }
    return width;
}

int required_pauli_width(const std::vector<PauliString>& terms, const char* where) {
    const int width = uniform_pauli_width(terms, where);
    if (width < 0) {
        throw std::invalid_argument(
            std::string(where) + ": the operator has no terms, so it has no width; "
            "the zero operator on n qubits is SparsePauliOp::zero(n)");
    }
    return width;
}

void check_hermitian(const SparsePauliOp& op, const char* where) {
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
            throw std::invalid_argument(
                std::string(where) + ": the operator is not Hermitian: label '" +
                term.pauli + "' has coefficient " + std::to_string(total.real) +
                (total.imag < 0.0 ? " - " : " + ") +
                std::to_string(std::abs(total.imag)) +
                "i once repeated labels are merged. Its expectation value is "
                "complex, and this returns a real number");
        }
    }
}

void check_observable(const SparsePauliOp& op, int n_qubits, const char* where,
                      const char* against) {
    (void)required_pauli_width(op.terms, where);
    for (std::size_t i = 0; i < op.terms.size(); ++i) {
        const int width = op.terms[i].n_qubits();
        if (width != n_qubits) {
            throw std::invalid_argument(
                std::string(where) + ": term " + std::to_string(i) + " ('" +
                op.terms[i].pauli + "') is " + std::to_string(width) +
                " qubits wide, which does not match the " + std::to_string(n_qubits) +
                " qubit " + against + "; a term names exactly one Pauli per qubit");
        }
    }
    check_hermitian(op, where);
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

double SparsePauliOp::expectation_value(const Statevector& sv) const {
    // ⟨ψ|H|ψ⟩ = sum_term coeff * ⟨ψ|P|ψ⟩
    //
    // Computed without cloning the statevector. For a Pauli string
    // P = ⊗ P_q, the matrix element ⟨k|P|j⟩ is non-zero for exactly
    // one j per k: j = k XOR x_mask (where x_mask has bits set at X and Y
    // positions). The phase is determined by Z and Y parities.
    //
    // This traverses the statevector once per term with no heap allocation.
    //
    // The masks come from each term's own string, so the width check is what
    // keeps j = k ^ x_mask inside the amplitude arrays.
    detail::check_observable(*this, sv.n_qubits, "SparsePauliOp::expectation_value");

    double result = 0.0;
    const double* rp = sv.real_parts;
    const double* ip = sv.imag_parts;
    const size_t dim = sv.dim;

    for (const auto& term : terms) {
        const int n = term.n_qubits();
        uint64_t x_mask = 0, z_mask = 0, y_mask = 0;
        for (int q = 0; q < n; ++q) {
            char c = term.pauli[q];
            if (c == 'X') {
                x_mask |= (1ULL << q);
            } else if (c == 'Z') {
                z_mask |= (1ULL << q);
            } else if (c == 'Y') {
                x_mask |= (1ULL << q);
                y_mask |= (1ULL << q);
                z_mask |= (1ULL << q);
            }
        }

        double re = 0.0, im = 0.0;

        #pragma omp parallel for reduction(+:re,im) schedule(static) if(dim >= (1<<20))
        for (int kk = 0; kk < static_cast<int>(dim); ++kk) {
            size_t k = kk;
            const size_t j = k ^ x_mask;

            // Phase from Z parity: Z acts on the input (column) state |j⟩, not |k⟩.
            // Y = iXZ: Z contributes (-1)^{j_bit} for each Y/Z qubit in j = k^x_mask.
            const int z_parity = LINDBLAD_POPCOUNT64(j & z_mask) & 1;
            // i^{#Y} factor is constant per Pauli term (independent of k).
            const int y_count  = LINDBLAD_POPCOUNT64(y_mask) & 3;

            // i^y_count: 0->1+0i, 1->0+1i, 2->-1+0i, 3->0-1i
            double phase_r = 1.0, phase_i = 0.0;
            switch (y_count) {
                case 1: phase_r =  0.0; phase_i =  1.0; break;
                case 2: phase_r = -1.0; phase_i =  0.0; break;
                case 3: phase_r =  0.0; phase_i = -1.0; break;
                default: break;
            }
            if (z_parity) { phase_r = -phase_r; phase_i = -phase_i; }

            // conj(ψ_k) * ψ_j = (r_k*r_j + i_k*i_j) + i*(r_k*i_j - i_k*r_j)
            const double r_k = rp[k], i_k = ip[k];
            const double r_j = rp[j], i_j = ip[j];
            const double dot_r = r_k * r_j + i_k * i_j;
            const double dot_i = r_k * i_j - i_k * r_j;

            re += phase_r * dot_r - phase_i * dot_i;
            im += phase_r * dot_i + phase_i * dot_r;
        }

        // Accumulate: Re(coeff * (re + i*im))
        result += term.coeff.real * re - term.coeff.imag * im;
    }

    return result;
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

    // Precompute masks once for all terms — shared across states.
    struct TermMasks { uint64_t x_mask, z_mask, y_mask; Complex128 coeff; };
    std::vector<TermMasks> masks(terms.size());
    for (size_t t = 0; t < terms.size(); ++t) {
        const int n = terms[t].n_qubits();
        uint64_t xm = 0, zm = 0, ym = 0;
        for (int q = 0; q < n; ++q) {
            char c = terms[t].pauli[q];
            if (c == 'X') { xm |= (1ULL << q); }
            else if (c == 'Z') { zm |= (1ULL << q); }
            else if (c == 'Y') { xm |= (1ULL << q); ym |= (1ULL << q); zm |= (1ULL << q); }
        }
        masks[t] = {xm, zm, ym, terms[t].coeff};
    }

    // Parallelise over states; each state is independent.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int sii = 0; sii < static_cast<int>(M); ++sii) {
        size_t si = sii;
        const Statevector* sv = states[si];
        const double* rp = sv->real_parts;
        const double* ip = sv->imag_parts;
        const size_t dim = sv->dim;
        double state_result = 0.0;

        for (const auto& m : masks) {
            double re = 0.0, im = 0.0;
            const uint64_t xm = m.x_mask, zm = m.z_mask, ym = m.y_mask;

#if !defined(_MSC_VER)
            #pragma omp simd reduction(+:re,im)
#endif
            for (size_t k = 0; k < dim; ++k) {
                const size_t j = k ^ xm;
                const int z_parity = LINDBLAD_POPCOUNT64(j & zm) & 1;
                const int y_count  = LINDBLAD_POPCOUNT64(ym) & 3;
                double phase_r = 1.0, phase_i = 0.0;
                switch (y_count) {
                    case 1: phase_r =  0.0; phase_i =  1.0; break;
                    case 2: phase_r = -1.0; phase_i =  0.0; break;
                    case 3: phase_r =  0.0; phase_i = -1.0; break;
                    default: break;
                }
                if (z_parity) { phase_r = -phase_r; phase_i = -phase_i; }
                const double r_k = rp[k], i_k = ip[k];
                const double r_j = rp[j], i_j = ip[j];
                const double dot_r = r_k * r_j + i_k * i_j;
                const double dot_i = r_k * i_j - i_k * r_j;
                re += phase_r * dot_r - phase_i * dot_i;
                im += phase_r * dot_i + phase_i * dot_r;
            }
            state_result += m.coeff.real * re - m.coeff.imag * im;
        }
        results[si] = state_result;
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

