#include "qpp/primitives.hpp"
#include "qpp/simulators/statevector_sim.hpp"
#include "qpp/transpiler.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <sstream>

namespace qpp {

// =============================================================================
// Estimator
// =============================================================================

std::vector<double> Estimator::run_batch(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    const std::vector<std::vector<double>>& parameter_values
) {
    const size_t n = parameter_values.size();
    std::vector<double> results(n);

    // run_single is thread-safe: all state is local (bound_circuit, sim, result).
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        results[i] = run_single(circuit, observable, parameter_values[i]);
    }

    return results;
}

// =============================================================================
// Structure key: hash of (gate_type_int, sorted_qubits) for each instruction,
// ignoring numeric parameters. Two circuits with the same gate structure but
// different parameter values map to the same key.
// =============================================================================

static std::string circuit_structure_key(const QuantumCircuit& qc) {
    std::ostringstream oss;
    oss << qc.n_qubits << ':';
    for (const auto& inst : qc.instructions) {
        oss << static_cast<int>(inst.type) << '(';
        for (int q : inst.qubits) oss << q << ',';
        oss << ')';
    }
    return oss.str();
}

double Estimator::run_single(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    const std::vector<double>& parameters
) {
    // Transpile with caching. The cache key is based on the UNBOUND circuit
    // structure (gate types + qubit indices, ignoring parameter values), so the
    // same transpiled layout is reused across all parameter evaluations.
    // Cache stores the transpiled-but-unbound circuit; parameters are bound
    // to the transpiled version, bypassing repeated SABRE/ZYZ work.
    QuantumCircuit to_simulate = circuit;

    if (options.optimization_level > 0) {
        std::string key = circuit_structure_key(circuit);

        {
            std::lock_guard<std::mutex> lk(cache_mutex_);
            auto it = transpile_cache_.find(key);
            if (it != transpile_cache_.end()) {
                to_simulate = it->second;  // cached transpiled (unbound) circuit
            } else {
                // Cache miss: transpile the unbound circuit
                to_simulate = qpp::transpile(circuit, CouplingMap(circuit.n_qubits),
                                             {}, options.optimization_level);
                transpile_cache_.emplace(key, to_simulate);
            }
        }
    }

    // Bind parameters to the (possibly transpiled) circuit
    if (!parameters.empty()) {
        std::unordered_map<std::string, double> bindings;
        const auto& names = to_simulate.parameter_names.empty()
                            ? circuit.parameter_names
                            : to_simulate.parameter_names;
        for (size_t i = 0; i < std::min(parameters.size(), names.size()); ++i) {
            bindings[names[i]] = parameters[i];
        }
        to_simulate = to_simulate.assign_parameters(bindings);
    }

    // Simulate
    StatevectorSimulator sim;
    auto result = sim.run(to_simulate);

    if (!result.success) {
        throw std::runtime_error("Simulation failed: " + result.error_message);
    }

    // Compute expectation value
    return observable.expectation_value(result.final_state);
}

// =============================================================================
// Estimator::gradient — parameter-shift rule
//
// dE/dθ_i = ( E(θ + π/2 eᵢ) − E(θ − π/2 eᵢ) ) / 2
//
// All 2P evaluations are packed into a single run_batch call so they are
// computed in parallel across threads.
// =============================================================================

std::vector<double> Estimator::gradient(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    const std::vector<double>& parameters
) {
    const size_t P = parameters.size();
    if (P == 0) return {};

    constexpr double shift = 3.14159265358979323846 / 2.0;

    // Build 2P shifted parameter sets: [θ+π/2 e₀, θ-π/2 e₀, θ+π/2 e₁, ...]
    std::vector<std::vector<double>> shifted(2 * P, parameters);
    for (size_t i = 0; i < P; ++i) {
        shifted[2 * i    ][i] += shift;
        shifted[2 * i + 1][i] -= shift;
    }

    auto evals = run_batch(circuit, observable, shifted);

    std::vector<double> grad(P);
    for (size_t i = 0; i < P; ++i) {
        grad[i] = (evals[2 * i] - evals[2 * i + 1]) / 2.0;
    }
    return grad;
}

} // namespace qpp
