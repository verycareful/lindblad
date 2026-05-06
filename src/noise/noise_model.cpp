#include "lindblad/noise.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace lindblad {

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
    // before_gate application is not yet implemented in DensityMatrixSimulator;
    // all errors are applied after the gate.
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

NoiseModel NoiseModel::from_t1_t2(
    const std::vector<double>& t1,
    const std::vector<double>& t2,
    const std::unordered_map<std::string, double>& gate_times,
    const std::unordered_map<std::string, std::vector<int>>& gate_qubits
) {
    int n = static_cast<int>(t1.size());
    if (static_cast<int>(t2.size()) != n) {
        throw std::invalid_argument("t1 and t2 must have the same length");
    }
    for (int q = 0; q < n; ++q) {
        if (t1[q] <= 0.0 || t2[q] <= 0.0) {
            throw std::invalid_argument("T1 and T2 must be positive");
        }
        if (t2[q] > 2.0 * t1[q]) {
            throw std::invalid_argument("T2 must be <= 2*T1 for qubit " + std::to_string(q));
        }
    }

    NoiseModel model;

    for (const auto& [gate_name, gate_time] : gate_times) {
        // Determine which qubits to apply this gate's noise to
        std::vector<int> qubits_for_gate;
        auto it = gate_qubits.find(gate_name);
        if (it != gate_qubits.end() && !it->second.empty()) {
            qubits_for_gate = it->second;
        } else {
            // Apply to all qubits
            qubits_for_gate.resize(n);
            for (int q = 0; q < n; ++q) qubits_for_gate[q] = q;
        }

        for (int q : qubits_for_gate) {
            if (q < 0 || q >= n) continue;
            auto channel = NoiseChannels::thermal_relaxation(t1[q], t2[q], gate_time);
            model.add_quantum_error(channel, gate_name, {q});
        }
    }

    return model;
}

} // namespace lindblad
