#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/gates.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace lindblad {

// =============================================================================
// PauliString
// =============================================================================

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
// SparsePauliOp
// =============================================================================

SparsePauliOp SparsePauliOp::simplify(double atol) const {
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
    int nq = n_qubits();
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

        for (size_t j = 0; j < dim; ++j) {
            const size_t row = j ^ x_mask;
            const int z_parity = LINDBLAD_POPCOUNT64(j & z_mask) & 1;
            const int y_count  = LINDBLAD_POPCOUNT64(j & y_mask) & 3;

            double phase_r = 1.0, phase_i = 0.0;
            switch (y_count) {
                case 1: phase_r =  0.0; phase_i =  1.0; break;
                case 2: phase_r = -1.0; phase_i =  0.0; break;
                case 3: phase_r =  0.0; phase_i = -1.0; break;
                default: break;
            }
            if (z_parity) { phase_r = -phase_r; phase_i = -phase_i; }

            Complex128 entry(
                phase_r * term.coeff.real - phase_i * term.coeff.imag,
                phase_r * term.coeff.imag + phase_i * term.coeff.real
            );
            matrix[row * dim + j] += entry;
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

            // Phase from Z parity (Z_mask includes Y positions via Y=iXZ)
            const int z_parity = LINDBLAD_POPCOUNT64(k & z_mask) & 1;
            // Additional i^y_count factor from Y = iXZ decomposition
            const int y_count  = LINDBLAD_POPCOUNT64(k & y_mask) & 3;

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
    const size_t M = states.size();
    std::vector<double> results(M, 0.0);
    if (M == 0 || terms.empty()) return results;

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
                const int z_parity = LINDBLAD_POPCOUNT64(k & zm) & 1;
                const int y_count  = LINDBLAD_POPCOUNT64(k & ym) & 3;
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
    return result;
}

SparsePauliOp SparsePauliOp::identity(int n_qubits) {
    return SparsePauliOp({{std::string(n_qubits, 'I'), Complex128(1.0, 0.0)}});
}

SparsePauliOp SparsePauliOp::zero(int n_qubits) {
    return SparsePauliOp({{std::string(n_qubits, 'I'), Complex128(0.0, 0.0)}});
}

} // namespace lindblad

