#include "qpp/backends/local_backend.hpp"
#include "qpp/simulators/clifford_sim.hpp"
#include "qpp/simulators/mps_sim.hpp"

namespace qpp {
namespace backends {

BackendResult LocalBackend::run(
    const QuantumCircuit& circuit,
    int shots,
    uint64_t seed
) {
    BackendResult result;
    result.backend_name = name();
    result.shots = shots;

    try {
        // Auto-select simulator
        SimType sim_type = config.simulator;
        if (sim_type == SimType::AUTO) {
            if (!noise_model.is_ideal()) {
                sim_type = SimType::DENSITY_MATRIX;
            } else if (CliffordSimulator::is_clifford(circuit)) {
                sim_type = SimType::CLIFFORD;
            } else if (circuit.n_qubits > 20) {
                sim_type = SimType::MPS;
            } else {
                sim_type = SimType::STATEVECTOR;
            }
        }

        switch (sim_type) {
            case SimType::STATEVECTOR: {
                StatevectorSimulator sim;
                auto sv_result = sim.run(circuit, shots, seed);
                result.counts = sv_result.counts;
                result.simulation_time_seconds = sv_result.simulation_time_seconds;
                result.success = sv_result.success;
                result.error_message = sv_result.error_message;
                break;
            }
            case SimType::DENSITY_MATRIX: {
                DensityMatrixSimulator sim;
                auto dm_result = sim.run(circuit, noise_model, shots, seed);
                result.counts = dm_result.counts;
                result.simulation_time_seconds = dm_result.simulation_time_seconds;
                result.success = dm_result.success;
                result.error_message = dm_result.error_message;
                break;
            }
            case SimType::CLIFFORD: {
                CliffordSimulator sim;
                auto cliff_result = sim.run(circuit, shots, seed);
                result.counts = cliff_result.counts;
                result.success = true;
                break;
            }
            case SimType::MPS: {
                MPSSimulator sim;
                auto mps_result = sim.run(circuit, config.mps_bond_dim, shots, seed);
                result.counts = mps_result.counts;
                result.simulation_time_seconds = mps_result.simulation_time_seconds;
                break;
            }
            default:
                result.success = false;
                result.error_message = "Unknown simulator type";
                break;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

std::vector<BackendResult> LocalBackend::run_batch(
    const std::vector<QuantumCircuit>& circuits,
    int shots,
    uint64_t seed
) {
    std::vector<BackendResult> results;
    results.reserve(circuits.size());

    for (const auto& circuit : circuits) {
        results.push_back(run(circuit, shots, seed));
    }

    return results;
}

} // namespace backends
} // namespace qpp
