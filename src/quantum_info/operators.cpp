#include "qpp/operators.hpp"
#include "qpp/statevector.hpp"
#include "qpp/simulators/density_matrix_sim.hpp"
#include "qpp/gates.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace qpp {

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

    for (const auto& term : terms) {
        // Build the matrix for this Pauli string
        // Start with a 1x1 identity and tensor product from right to left
        std::vector<Complex128> pauli_mat = {Complex128(1.0, 0.0)};
        int current_dim = 1;

        for (int q = nq - 1; q >= 0; --q) {
            std::vector<Complex128> single(4);
            char c = term.pauli[q];
            if (c == 'I') {
                single = {Complex128(1,0), Complex128(0,0), Complex128(0,0), Complex128(1,0)};
            } else if (c == 'X') {
                single = {Complex128(0,0), Complex128(1,0), Complex128(1,0), Complex128(0,0)};
            } else if (c == 'Y') {
                single = {Complex128(0,0), Complex128(0,-1), Complex128(0,1), Complex128(0,0)};
            } else if (c == 'Z') {
                single = {Complex128(1,0), Complex128(0,0), Complex128(0,0), Complex128(-1,0)};
            }

            // Tensor product: pauli_mat ⊗ single
            int new_dim = current_dim * 2;
            std::vector<Complex128> new_mat(new_dim * new_dim, Complex128(0.0, 0.0));

            for (int i = 0; i < current_dim; ++i) {
                for (int j = 0; j < current_dim; ++j) {
                    for (int si = 0; si < 2; ++si) {
                        for (int sj = 0; sj < 2; ++sj) {
                            new_mat[(i*2+si) * new_dim + (j*2+sj)] =
                                pauli_mat[i * current_dim + j] * single[si * 2 + sj];
                        }
                    }
                }
            }

            pauli_mat = new_mat;
            current_dim = new_dim;
        }

        // Accumulate with coefficient
        for (size_t i = 0; i < dim * dim; ++i) {
            matrix[i] += pauli_mat[i] * term.coeff;
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

        #pragma omp parallel for reduction(+:re,im) schedule(static) if(dim > (1<<20))
        for (size_t k = 0; k < dim; ++k) {
            const size_t j = k ^ x_mask;

            // Phase from Z parity (Z_mask includes Y positions via Y=iXZ)
            const int z_parity = __builtin_popcountll(k & z_mask) & 1;
            // Additional i^y_count factor from Y = iXZ decomposition
            const int y_count  = __builtin_popcountll(k & y_mask) & 3;

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

} // namespace qpp
