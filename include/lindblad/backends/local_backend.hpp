#pragma once

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {
namespace backends {

// =============================================================================
// BackendResult — unified result type
// =============================================================================

struct BackendResult {
    std::unordered_map<std::string, int> counts;
    double simulation_time_seconds = 0.0;
    bool success = true;
    std::string error_message;
    std::string backend_name;
    int shots = 0;
};

// =============================================================================
// LocalBackend — wraps the simulators
// =============================================================================

class LocalBackend {
public:
    enum class SimType {
        STATEVECTOR,
        DENSITY_MATRIX,
        CLIFFORD,
        MPS,
        AUTO  // automatically pick best
    };

    struct Config {
        SimType simulator = SimType::AUTO;
        int max_parallel_threads = 0;
        uint64_t max_memory_mb = 0;
        int mps_bond_dim = 64;
    };

    Config config;
    NoiseModel noise_model;

    LocalBackend() = default;
    explicit LocalBackend(const Config& cfg) : config(cfg) {}

    BackendResult run(
        const QuantumCircuit& circuit,
        int shots = 1024,
        uint64_t seed = 0
    );

    std::vector<BackendResult> run_batch(
        const std::vector<QuantumCircuit>& circuits,
        int shots = 1024,
        uint64_t seed = 0
    );

    std::string name() const { return "lindblad_local_simulator"; }
    std::string version() const { return "1.9.6-alpha"; }
    int max_qubits() const { return 30; }
};

} // namespace backends
} // namespace lindblad
