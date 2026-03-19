#include "qpp/primitives.hpp"
#include "qpp/simulators/statevector_sim.hpp"

namespace qpp {

// =============================================================================
// Sampler
// =============================================================================

std::vector<std::unordered_map<std::string, int>> Sampler::run(
    const std::vector<QuantumCircuit>& circuits,
    const std::vector<std::vector<double>>& parameter_values
) {
    std::vector<std::unordered_map<std::string, int>> results;
    results.reserve(circuits.size());

    for (size_t i = 0; i < circuits.size(); ++i) {
        std::vector<double> params;
        if (i < parameter_values.size()) {
            params = parameter_values[i];
        }
        results.push_back(run_single(circuits[i], params));
    }

    return results;
}

std::unordered_map<std::string, int> Sampler::run_single(
    const QuantumCircuit& circuit,
    const std::vector<double>& parameters
) {
    QuantumCircuit bound_circuit = circuit;
    if (!parameters.empty()) {
        std::unordered_map<std::string, double> bindings;
        for (size_t i = 0; i < std::min(parameters.size(), circuit.parameter_names.size()); ++i) {
            bindings[circuit.parameter_names[i]] = parameters[i];
        }
        bound_circuit = circuit.assign_parameters(bindings);
    }

    StatevectorSimulator sim;
    auto result = sim.run(bound_circuit, options.shots, options.seed);

    if (!result.success) {
        throw std::runtime_error("Simulation failed: " + result.error_message);
    }

    return result.counts;
}

} // namespace qpp
