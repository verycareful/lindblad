#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <optional>
#include <stdexcept>

namespace lindblad {

// =============================================================================
// Operator
// =============================================================================

Operator Operator::from_circuit(const QuantumCircuit& circuit_in) {
    // Judge the circuit's supplied matrices once, here, rather than once per
    // basis column below: the columns re-apply the same instructions 2^n times
    // and the verdict cannot differ between them.
    // Under Repair::Attempt a repaired copy is executed and the caller's
    // circuit is left exactly as it was handed over; Repair::None binds
    // straight to it and nothing is copied.
    std::optional<QuantumCircuit> repaired_storage =
        circuit_in.validated_physical();
    const QuantumCircuit& circuit =
        repaired_storage ? *repaired_storage : circuit_in;


    size_t dim = 1ULL << circuit.n_qubits;
    std::vector<Complex128> mat(dim * dim, Complex128(0.0, 0.0));

    StatevectorSimulator sim;

    for (size_t col = 0; col < dim; ++col) {
        Statevector sv(circuit.n_qubits);
        sv.initialize_basis(col);

        // Apply all gates
        for (const auto& inst : circuit.instructions) {
            sim.apply_instruction(sv, inst, {Validation::Ignore});
        }

        // Extract column
        for (size_t row = 0; row < dim; ++row) {
            mat[row * dim + col] = sv.amplitude(row);
        }
    }

    return Operator(mat, circuit.n_qubits);
}

Operator Operator::from_pauli(const SparsePauliOp& op) {
    auto mat = op.to_matrix();
    return Operator(mat, op.n_qubits());
}

Operator Operator::compose(const Operator& other) const {
    if (n_qubits != other.n_qubits) {
        throw std::invalid_argument("Cannot compose operators with different qubit counts");
    }

    size_t d = dim();
    std::vector<Complex128> result(d * d, Complex128(0.0, 0.0));

    // this * other (matrix multiplication)
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            Complex128 sum(0.0, 0.0);
            for (size_t k = 0; k < d; ++k) {
                sum += data[i * d + k] * other.data[k * d + j];
            }
            result[i * d + j] = sum;
        }
    }

    return Operator(result, n_qubits);
}

Operator Operator::tensor(const Operator& other) const {
    int new_nq = n_qubits + other.n_qubits;
    size_t d1 = dim();
    size_t d2 = other.dim();
    size_t d = d1 * d2;
    std::vector<Complex128> result(d * d, Complex128(0.0, 0.0));

    for (size_t i1 = 0; i1 < d1; ++i1) {
        for (size_t j1 = 0; j1 < d1; ++j1) {
            for (size_t i2 = 0; i2 < d2; ++i2) {
                for (size_t j2 = 0; j2 < d2; ++j2) {
                    result[(i1*d2+i2) * d + (j1*d2+j2)] =
                        data[i1*d1+j1] * other.data[i2*d2+j2];
                }
            }
        }
    }

    return Operator(result, new_nq);
}

Operator Operator::adjoint() const {
    size_t d = dim();
    std::vector<Complex128> result(d * d);

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[i * d + j] = data[j * d + i].conj();
        }
    }

    return Operator(result, n_qubits);
}

Operator Operator::power(int n) const {
    if (n < 0) throw std::invalid_argument("Operator::power requires n >= 0");

    size_t d = dim();
    std::vector<Complex128> id(d * d, Complex128(0.0, 0.0));
    for (size_t i = 0; i < d; ++i) id[i * d + i] = Complex128(1.0, 0.0);
    if (n == 0) return Operator(id, n_qubits);

    // Binary exponentiation: O(log n) multiplications instead of O(n).
    Operator result(id, n_qubits);
    Operator base = *this;
    while (n > 0) {
        if (n & 1) result = result.compose(base);
        base = base.compose(base);
        n >>= 1;
    }
    return result;
}

bool Operator::is_unitary(double atol) const {
    auto adj = adjoint();
    auto product = compose(adj);
    size_t d = dim();

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            auto diff = product.data[i * d + j] - Complex128(expected, 0.0);
            if (diff.norm_sq() > atol * atol) return false;
        }
    }
    return true;
}

bool Operator::is_hermitian(double atol) const {
    size_t d = dim();
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i; j < d; ++j) {
            auto diff = data[i * d + j] - data[j * d + i].conj();
            if (diff.norm_sq() > atol * atol) return false;
        }
    }
    return true;
}

Complex128 Operator::trace() const {
    Complex128 tr(0.0, 0.0);
    size_t d = dim();
    for (size_t i = 0; i < d; ++i) {
        tr += data[i * d + i];
    }
    return tr;
}

} // namespace lindblad
