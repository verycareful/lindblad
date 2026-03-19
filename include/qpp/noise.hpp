#pragma once

#include "qpp/types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace qpp {

// Forward declarations
class BackendProperties;

// =============================================================================
// KrausChannel — Kraus representation of a quantum channel
// =============================================================================

struct KrausChannel {
    std::vector<std::vector<Complex128>> operators;  // list of Kraus matrices
    int n_qubits = 1;

    bool is_valid(double atol = 1e-8) const;  // Check sum_k K_k†K_k = I
    double trace_preserving_error() const;
};

// =============================================================================
// Standard noise channels
// =============================================================================

namespace NoiseChannels {
    KrausChannel depolarizing(double p, int n_qubits = 1);
    KrausChannel amplitude_damping(double gamma);
    KrausChannel phase_damping(double lambda);
    KrausChannel thermal_relaxation(
        double T1, double T2, double gate_time,
        double excited_state_population = 0.0
    );
    KrausChannel pauli(double px, double py, double pz);
    KrausChannel bit_flip(double p);
    KrausChannel phase_flip(double p);
    KrausChannel bit_phase_flip(double p);
    KrausChannel reset(double p0, double p1);
    KrausChannel coherent_unitary(double theta, double phi, double lambda);
}

// =============================================================================
// ReadoutError
// =============================================================================

struct ReadoutError {
    double prob_meas_0_prep_1;  // P(measure 0 | prepared 1)
    double prob_meas_1_prep_0;  // P(measure 1 | prepared 0)

    std::array<std::array<double, 2>, 2> assignment_matrix() const;
};

// =============================================================================
// NoiseModel — attaches errors to gates
// =============================================================================

class NoiseModel {
public:
    struct GateError {
        KrausChannel channel;
        std::vector<int> qubits;  // empty = all qubits
        bool after_gate = true;
    };

    void add_quantum_error(
        const KrausChannel& error,
        const std::string& gate_name,
        const std::vector<int>& qubits = {}
    );

    void add_readout_error(
        const ReadoutError& error,
        int qubit
    );

    void add_all_qubit_quantum_error(
        const KrausChannel& error,
        const std::string& gate_name
    );

    std::vector<GateError> errors_for_gate(
        const std::string& gate_name,
        const std::vector<int>& qubits
    ) const;

    bool is_ideal() const;

    std::unordered_map<std::string, std::vector<GateError>> basis_gate_errors;
    std::unordered_map<int, ReadoutError> readout_errors;
    std::vector<std::string> noisy_gates;
};

} // namespace qpp
