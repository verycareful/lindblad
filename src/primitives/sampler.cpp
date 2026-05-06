#include "lindblad/primitives.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

namespace lindblad {

// =============================================================================
// Sampler
// =============================================================================

std::vector<std::unordered_map<std::string, int>> Sampler::run(
    const std::vector<QuantumCircuit>& circuits,
    const std::vector<std::vector<double>>& parameter_values
) {
    std::vector<std::unordered_map<std::string, int>> results;
    results.reserve(circuits.size());

    const uint64_t base_seed = options.seed;
    for (size_t i = 0; i < circuits.size(); ++i) {
        std::vector<double> params;
        if (i < parameter_values.size()) {
            params = parameter_values[i];
        }
        // Give each circuit a distinct seed so batch results are independent.
        // seed==0 means "random"; preserve that by not offsetting.
        if (base_seed != 0) options.seed = base_seed + static_cast<uint64_t>(i);
        results.push_back(run_single(circuits[i], params));
    }
    options.seed = base_seed;

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

    if (!options.noise_model.is_ideal()) {
        DensityMatrixSimulator dm_sim;
        auto result = dm_sim.run(bound_circuit, options.noise_model,
                                 options.shots, options.seed);
        if (!result.success)
            throw std::runtime_error("Noisy simulation failed: " + result.error_message);
        return result.counts;
    }

    StatevectorSimulator sim;
    auto result = sim.run(bound_circuit, options.shots, options.seed);

    if (!result.success) {
        throw std::runtime_error("Simulation failed: " + result.error_message);
    }

    return result.counts;
}

} // namespace lindblad
