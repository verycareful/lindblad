#pragma once

#include "qpp/circuit.hpp"
#include "qpp/operators.hpp"
#include "qpp/noise.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace qpp {

// =============================================================================
// Estimator primitive — computes expectation values
// =============================================================================

class Estimator {
public:
    struct Options {
        int shots = 0;          // 0 = exact (statevector)
        uint64_t seed = 0;
        NoiseModel noise_model;
        int optimization_level = 1;
    };

    Options options;

    Estimator() = default;
    explicit Estimator(const Options& opts) : options(opts) {}

    // Core: single circuit, single observable, multiple parameter sets
    std::vector<double> run_batch(
        const QuantumCircuit& circuit,
        const SparsePauliOp& observable,
        const std::vector<std::vector<double>>& parameter_values
    );

    // Single evaluation
    double run_single(
        const QuantumCircuit& circuit,
        const SparsePauliOp& observable,
        const std::vector<double>& parameters = {}
    );
};

// =============================================================================
// Sampler primitive — samples bitstring distributions
// =============================================================================

class Sampler {
public:
    struct Options {
        int shots = 1024;
        uint64_t seed = 0;
        NoiseModel noise_model;
    };

    Options options;

    Sampler() = default;
    explicit Sampler(const Options& opts) : options(opts) {}

    std::vector<std::unordered_map<std::string, int>> run(
        const std::vector<QuantumCircuit>& circuits,
        const std::vector<std::vector<double>>& parameter_values = {}
    );

    std::unordered_map<std::string, int> run_single(
        const QuantumCircuit& circuit,
        const std::vector<double>& parameters = {}
    );
};

} // namespace qpp
