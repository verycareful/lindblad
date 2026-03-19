#include "qpp/operators.hpp"
#include "qpp/statevector.hpp"
#include "qpp/simulators/density_matrix_sim.hpp"

#include <algorithm>
#include <cmath>
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
    // For each Pauli string P, ⟨ψ|P|ψ⟩ can be computed efficiently
    double result = 0.0;

    for (const auto& term : terms) {
        // Apply the Pauli string to |ψ⟩ and compute inner product
        Statevector temp = sv.clone();

        for (int q = 0; q < term.n_qubits(); ++q) {
            char c = term.pauli[q];
            switch (c) {
                case 'X':
                    gates::apply_x(temp, q);
                    break;
                case 'Y':
                    gates::apply_y(temp, q);
                    break;
                case 'Z':
                    gates::apply_z(temp, q);
                    break;
                default:
                    break;  // 'I' — do nothing
            }
        }

        Complex128 inner = sv.inner_product(temp);
        Complex128 ev = term.coeff * inner;
        result += ev.real;
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
