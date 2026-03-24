#include "qpp/noise.hpp"

#include <algorithm>
#include <array>

namespace qpp {

// =============================================================================
// ReadoutError
// =============================================================================

std::array<std::array<double, 2>, 2> ReadoutError::assignment_matrix() const {
    return {{
        {{1.0 - prob_meas_1_prep_0, prob_meas_0_prep_1}},
        {{prob_meas_1_prep_0, 1.0 - prob_meas_0_prep_1}}
    }};
}

// =============================================================================
// NoiseModel
// =============================================================================

void NoiseModel::add_quantum_error(
    const KrausChannel& error,
    const std::string& gate_name,
    const std::vector<int>& qubits
) {
    GateError ge;
    ge.channel = error;
    ge.qubits = qubits;
    ge.after_gate = true;
    basis_gate_errors[gate_name].push_back(ge);

    if (std::find(noisy_gates.begin(), noisy_gates.end(), gate_name) == noisy_gates.end()) {
        noisy_gates.push_back(gate_name);
    }
}

void NoiseModel::add_readout_error(const ReadoutError& error, int qubit) {
    readout_errors[qubit] = error;
}

void NoiseModel::add_all_qubit_quantum_error(
    const KrausChannel& error,
    const std::string& gate_name
) {
    add_quantum_error(error, gate_name, {});
}

std::vector<NoiseModel::GateError> NoiseModel::errors_for_gate(
    const std::string& gate_name,
    const std::vector<int>& qubits
) const {
    std::vector<GateError> result;

    auto it = basis_gate_errors.find(gate_name);
    if (it == basis_gate_errors.end()) return result;

    for (const auto& ge : it->second) {
        if (ge.qubits.empty()) {
            // Applies to all qubits
            result.push_back(ge);
        } else if (ge.qubits == qubits) {
            result.push_back(ge);
        }
    }

    return result;
}

bool NoiseModel::is_ideal() const {
    return basis_gate_errors.empty() && readout_errors.empty();
}

} // namespace qpp
