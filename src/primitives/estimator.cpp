#include "lindblad/primitives.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/transpiler.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace lindblad {

namespace {

// Sample ⟨P⟩ for a single Pauli string `pauli` (e.g. "XYZII", index 0 = MSB
// per PauliString convention) on the prepared circuit. Appends basis-rotation
// gates so each non-I Pauli is measured in the Z basis, takes `shots`
// measurements, and returns the mean of (+1 for even-parity outcomes, -1 for
// odd parity) over the qubits where the Pauli is non-I.
//
// Backend choice: noise-aware (DM) when noise_model is non-ideal, otherwise
// SV. Shot count is the per-term budget; total work scales linearly with the
// number of non-identity Pauli terms in the observable.
double sample_pauli_expectation(
    const QuantumCircuit& circuit,
    const std::string& pauli,
    int shots,
    uint64_t seed,
    const NoiseModel& noise_model
) {
    const int n = circuit.n_qubits;
    if (static_cast<int>(pauli.size()) != n) {
        throw std::invalid_argument(
            "Estimator: Pauli string length " + std::to_string(pauli.size()) +
            " does not match circuit n_qubits=" + std::to_string(n));
    }

    // Build measurement circuit: copy + basis rotation + measure_all.
    QuantumCircuit meas = circuit;
    std::vector<int> non_identity_qubits;
    non_identity_qubits.reserve(n);
    for (int i = 0; i < n; ++i) {
        const int qubit = n - 1 - i;   // pauli[0] = MSB = qubit (n-1)
        const char p = pauli[i];
        if (p == 'I' || p == 'i') continue;
        non_identity_qubits.push_back(qubit);
        if (p == 'X' || p == 'x') {
            meas.h(qubit);
        } else if (p == 'Y' || p == 'y') {
            meas.sdg(qubit).h(qubit);   // S†H rotates Y eigenbasis → Z eigenbasis
        } else if (p == 'Z' || p == 'z') {
            // already Z basis
        } else {
            throw std::invalid_argument(
                std::string("Estimator: unknown Pauli character '") + p +
                "' in pauli string \"" + pauli + "\"");
        }
    }
    meas.measure_all();

    // If the Pauli is the identity, expectation is trivially +1 and we should
    // never have routed here, but guard regardless.
    if (non_identity_qubits.empty()) return 1.0;

    // Run shots through the appropriate backend. Each simulator returns its
    // own nested Result type, so we extract counts inside each branch into a
    // common local map and then process uniformly.
    std::unordered_map<std::string, int> counts;
    if (!noise_model.is_ideal()) {
        DensityMatrixSimulator dm_sim;
        auto res = dm_sim.run(meas, noise_model, shots, seed);
        if (!res.success) {
            throw std::runtime_error("Estimator sampling failed: " + res.error_message);
        }
        counts = std::move(res.counts);
    } else {
        StatevectorSimulator sv_sim;
        auto res = sv_sim.run(meas, shots, seed);
        if (!res.success) {
            throw std::runtime_error("Estimator sampling failed: " + res.error_message);
        }
        counts = std::move(res.counts);
    }

    // Accumulate parity-weighted estimate. Bitstring layout per statevector_sim:
    // clbit c is at string position (n - 1 - c); measure_all maps qubit q to
    // clbit q. So qubit q is at bits[n - 1 - q] (rightmost = qubit 0).
    long long pos_count = 0, total = 0;
    for (const auto& [bits, count] : counts) {
        int parity = 0;
        for (int q : non_identity_qubits) {
            const int pos = n - 1 - q;
            if (pos >= 0 && pos < static_cast<int>(bits.size()) && bits[pos] == '1') {
                parity ^= 1;
            }
        }
        total += count;
        pos_count += (parity == 0 ? +1 : -1) * count;
    }
    if (total == 0) return 0.0;
    return static_cast<double>(pos_count) / static_cast<double>(total);
}

// Sampling expectation value of a SparsePauliOp. Each term is measured with
// `shots` shots; the contributions are summed with their coefficients.
double sampled_expectation_value(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    int shots,
    uint64_t seed,
    const NoiseModel& noise_model
) {
    double total = 0.0;
    uint64_t term_seed = seed == 0 ? 0 : seed;
    for (const auto& term : observable.terms) {
        // Identity term: contributes coeff.real with no sampling noise.
        bool is_identity = true;
        for (char c : term.pauli) {
            if (c != 'I' && c != 'i') { is_identity = false; break; }
        }
        if (is_identity) {
            total += term.coeff.real;
            continue;
        }
        const double term_exp = sample_pauli_expectation(
            circuit, term.pauli, shots, term_seed, noise_model
        );
        total += term.coeff.real * term_exp;
        if (term_seed != 0) ++term_seed;   // decorrelate per-term sampling
    }
    return total;
}

}  // namespace

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
    oss << qc.n_qubits << ':' << qc.n_clbits << ':';
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

        // First check under lock (fast path for cache hit).
        {
            std::lock_guard<std::mutex> lk(cache_mutex_);
            auto it = transpile_cache_.find(key);
            if (it != transpile_cache_.end()) {
                to_simulate = it->second;
                goto cache_done;
            }
        }

        // Cache miss: transpile outside the lock so threads don't serialize.
        {
            QuantumCircuit transpiled = lindblad::transpile(circuit, CouplingMap(circuit.n_qubits),
                                                            {}, options.optimization_level);
            std::lock_guard<std::mutex> lk(cache_mutex_);
            // Another thread may have inserted while we compiled; use try_emplace
            // so the first writer wins and we always use the cached value.
            auto [it, inserted] = transpile_cache_.try_emplace(key, std::move(transpiled));
            to_simulate = it->second;
        }

        cache_done:;
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

    // Three modes:
    //   1. shots > 0          → real shot-noise sampling per Pauli term. Uses
    //                            the DM backend if noise_model is non-ideal,
    //                            otherwise SV. Variance ~ 1/sqrt(shots) per term.
    //   2. shots == 0 + noise → noisy DM evolution, exact expectation value
    //                            read off the final mixed state.
    //   3. shots == 0, ideal  → SV evolution, exact analytical expectation.
    if (options.shots > 0) {
        return sampled_expectation_value(
            to_simulate, observable, options.shots, options.seed, options.noise_model
        );
    }

    if (!options.noise_model.is_ideal()) {
        DensityMatrixSimulator dm_sim;
        auto dm_result = dm_sim.run(to_simulate, options.noise_model, /*shots=*/0, options.seed);
        if (!dm_result.success) {
            throw std::runtime_error("Simulation failed: " + dm_result.error_message);
        }
        return dm_result.final_state.expectation_value_sparse(observable);
    }

    StatevectorSimulator sim;
    return sim.eval_expectation(to_simulate, observable);
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

} // namespace lindblad
