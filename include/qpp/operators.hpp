#pragma once

#include "qpp/types.hpp"

#include <string>
#include <vector>

namespace qpp {

class Statevector;
class DensityMatrix;
class QuantumCircuit;

// =============================================================================
// PauliString — tensor product of single-qubit Paulis
// =============================================================================

struct PauliString {
    std::string pauli;    // e.g., "XYZII"
    Complex128 coeff;

    PauliString() : coeff(1.0, 0.0) {}
    PauliString(const std::string& p, Complex128 c = Complex128(1.0, 0.0))
        : pauli(p), coeff(c) {}

    int n_qubits() const { return static_cast<int>(pauli.size()); }
    PauliString compose(const PauliString& other) const;
    PauliString adjoint() const;
    bool commutes_with(const PauliString& other) const;
};

// =============================================================================
// SparsePauliOp — sum of Pauli strings
// =============================================================================

class SparsePauliOp {
public:
    std::vector<PauliString> terms;

    SparsePauliOp() = default;
    explicit SparsePauliOp(const std::vector<PauliString>& terms) : terms(terms) {}

    SparsePauliOp simplify(double atol = 1e-8) const;
    SparsePauliOp compose(const SparsePauliOp& other) const;
    SparsePauliOp adjoint() const;
    SparsePauliOp tensor(const SparsePauliOp& other) const;
    SparsePauliOp operator+(const SparsePauliOp& other) const;
    SparsePauliOp operator*(double scalar) const;

    std::vector<Complex128> to_matrix() const;
    double expectation_value(const Statevector& sv) const;
    int n_qubits() const;
    size_t size() const { return terms.size(); }

    static SparsePauliOp from_list(
        const std::vector<std::pair<std::string, Complex128>>& label_coeff
    );
    static SparsePauliOp identity(int n_qubits);
    static SparsePauliOp zero(int n_qubits);
};

// =============================================================================
// Operator — general matrix operator
// =============================================================================

class Operator {
public:
    std::vector<Complex128> data;
    int n_qubits;

    Operator() : n_qubits(0) {}
    Operator(std::vector<Complex128> d, int nq) : data(std::move(d)), n_qubits(nq) {}

    static Operator from_circuit(const QuantumCircuit& circuit);
    static Operator from_pauli(const SparsePauliOp& op);

    Operator compose(const Operator& other) const;
    Operator tensor(const Operator& other) const;
    Operator adjoint() const;
    Operator power(int n) const;

    bool is_unitary(double atol = 1e-8) const;
    bool is_hermitian(double atol = 1e-8) const;

    Complex128 trace() const;
    size_t dim() const { return 1ULL << n_qubits; }
};

// =============================================================================
// Quantum information metrics
// =============================================================================

namespace QuantumInfo {
    double state_fidelity(const Statevector& sv1, const Statevector& sv2);
    double state_fidelity(const DensityMatrix& rho1, const DensityMatrix& rho2);
    double process_fidelity(const Operator& channel1, const Operator& channel2);
    double average_gate_fidelity(const Operator& channel, const Operator& target);
    double entropy(const DensityMatrix& rho, double base = 2.0);
    double entanglement_entropy(const Statevector& sv, const std::vector<int>& subsystem);
    double concurrence(const DensityMatrix& rho);
    DensityMatrix partial_trace(const DensityMatrix& rho, const std::vector<int>& qargs);
    DensityMatrix partial_trace(const Statevector& sv, const std::vector<int>& qargs);

    std::vector<double> pauli_expectation_values(
        const Statevector& sv,
        const std::vector<std::string>& paulis
    );
}

} // namespace qpp
