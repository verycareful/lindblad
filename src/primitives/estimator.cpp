#include "qpp/primitives.hpp"
#include "qpp/simulators/statevector_sim.hpp"

namespace qpp {

// =============================================================================
// Estimator
// =============================================================================

std::vector<double> Estimator::run_batch(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    const std::vector<std::vector<double>>& parameter_values
) {
    std::vector<double> results;
    results.reserve(parameter_values.size());

    for (const auto& params : parameter_values) {
        results.push_back(run_single(circuit, observable, params));
    }

    return results;
}

double Estimator::run_single(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    const std::vector<double>& parameters
) {
    // Bind parameters if provided
    QuantumCircuit bound_circuit = circuit;
    if (!parameters.empty()) {
        std::unordered_map<std::string, double> bindings;
        for (size_t i = 0; i < std::min(parameters.size(), circuit.parameter_names.size()); ++i) {
            bindings[circuit.parameter_names[i]] = parameters[i];
        }
        bound_circuit = circuit.assign_parameters(bindings);
    }

    // Simulate
    StatevectorSimulator sim;
    auto result = sim.run(bound_circuit);

    if (!result.success) {
        throw std::runtime_error("Simulation failed: " + result.error_message);
    }

    // Compute expectation value
    return observable.expectation_value(result.final_state);
}

} // namespace qpp
