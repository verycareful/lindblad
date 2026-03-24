#pragma once

#include "qpp/circuit.hpp"
#include "qpp/statevector.hpp"
#include "qpp/types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
namespace qpp {
class SparsePauliOp;
class NoiseModel;
}

namespace qpp {

// =============================================================================
// StatevectorSimulator
// =============================================================================

class StatevectorSimulator {
public:
    struct Options {
        int max_parallel_threads = 0;  // 0 = auto (all cores)
        uint64_t max_memory_mb = 0;    // 0 = auto
        int precision = 64;            // 32 or 64 bit
        bool zero_threshold = true;
        double threshold = 1e-10;
    };

    struct Result {
        Statevector final_state;
        std::unordered_map<std::string, int> counts;  // if measured
        std::vector<double> expectation_values;        // if observables requested
        double simulation_time_seconds = 0.0;
        bool success = true;
        std::string error_message;

        Result() : final_state(0) {}
        Result(Result&&) = default;
        Result& operator=(Result&&) = default;
    };

    Options options;

    StatevectorSimulator() = default;
    explicit StatevectorSimulator(const Options& opts) : options(opts) {}

    Result run(
        const QuantumCircuit& circuit,
        int shots = 0,                              // 0 = no measurement sampling
        uint64_t seed = 0
    );

    void simulate_circuit(
        Statevector& sv,
        const QuantumCircuit& circuit
    );

    void apply_instruction(Statevector& sv, const Instruction& inst);
};

} // namespace qpp
