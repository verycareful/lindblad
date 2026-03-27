#pragma once

#include "qpp/circuit.hpp"
#include "qpp/operators.hpp"
#include "qpp/noise.hpp"

#include <mutex>
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
        int optimization_level = 0;
    };

    Options options;

    Estimator() = default;
    explicit Estimator(const Options& opts) : options(opts) {}

    // Clear the transpilation cache (e.g. when options change)
    void clear_cache() {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        transpile_cache_.clear();
    }

private:
    // Transpiled-circuit cache: structure key → transpiled circuit.
    // Key = hash of (gate type, qubits) sequence — independent of parameters.
    // Protected by a mutex so run_batch threads can share the cache safely.
    mutable std::unordered_map<std::string, QuantumCircuit> transpile_cache_;
    mutable std::mutex cache_mutex_;

public:

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

    // Parameter-shift gradient: dE/dθ_i = (E(θ + π/2 eᵢ) − E(θ − π/2 eᵢ)) / 2
    // Evaluations are parallelised across parameters via run_batch.
    // Returns a vector of length parameters.size().
    std::vector<double> gradient(
        const QuantumCircuit& circuit,
        const SparsePauliOp& observable,
        const std::vector<double>& parameters
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
