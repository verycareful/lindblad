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

// Run `circuit` in the Z-measurement basis defined by `basis` (basis[q] in
// {I,X,Y,Z}, uppercase): X -> prepend H, Y -> prepend S†H, Z/I -> nothing.
// measure_all maps qubit q -> clbit q. Returns the shot counts. Backend is
// noise-aware DM when the model is non-ideal, otherwise statevector.
std::unordered_map<std::string, int> run_in_basis(
    const QuantumCircuit& circuit,
    const std::vector<char>& basis,
    int shots, uint64_t seed, const NoiseModel& noise_model
) {
    QuantumCircuit meas = circuit;
    for (int q = 0; q < circuit.n_qubits; ++q) {
        if (basis[static_cast<size_t>(q)] == 'X') meas.h(q);
        else if (basis[static_cast<size_t>(q)] == 'Y') meas.sdg(q).h(q);
    }
    meas.measure_all();

    if (!noise_model.is_ideal()) {
        DensityMatrixSimulator dm_sim;
        auto res = dm_sim.run(meas, noise_model, shots, seed);
        if (!res.success)
            throw std::runtime_error("Estimator sampling failed: " + res.error_message);
        return std::move(res.counts);
    }
    StatevectorSimulator sv_sim;
    auto res = sv_sim.run(meas, shots, seed);
    if (!res.success)
        throw std::runtime_error("Estimator sampling failed: " + res.error_message);
    return std::move(res.counts);
}

// Parity-weighted expectation of a single Pauli string over shot counts.
// Qubit q sits at bits[n - 1 - q] (clbit c at string position n-1-c;
// measure_all maps qubit q -> clbit q). Mean of +1 (even parity) / -1 (odd)
// over the qubits where `pauli` is non-I.
double parity_expectation(
    const std::unordered_map<std::string, int>& counts,
    const std::string& pauli, int n
) {
    long long pos_count = 0, total = 0;
    for (const auto& [bits, count] : counts) {
        int parity = 0;
        for (int q = 0; q < n; ++q) {
            const char p = pauli[static_cast<size_t>(q)];
            if (p == 'I' || p == 'i') continue;
            const int pos = n - 1 - q;
            if (pos >= 0 && pos < static_cast<int>(bits.size()) && bits[pos] == '1')
                parity ^= 1;
        }
        total += count;
        pos_count += (parity == 0 ? +1 : -1) * count;
    }
    return total == 0 ? 0.0 : static_cast<double>(pos_count) / static_cast<double>(total);
}

// Uppercase Pauli, mapping I/i->I etc.; throws on unknown characters.
char norm_pauli(char c, const std::string& ctx) {
    switch (c) {
        case 'I': case 'i': return 'I';
        case 'X': case 'x': return 'X';
        case 'Y': case 'y': return 'Y';
        case 'Z': case 'z': return 'Z';
        default:
            throw std::invalid_argument(
                std::string("Estimator: unknown Pauli character '") + c +
                "' in \"" + ctx + "\"");
    }
}

// Sampling expectation value of a SparsePauliOp (audit F-10). When
// `group_terms` is true, qubit-wise-commuting terms share one measurement run
// (all Z/I terms collapse into a single Z-basis run); otherwise each term is
// measured independently (pre-R.1.13 behaviour, byte-identical seeded stream).
double sampled_expectation_value(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    int shots, uint64_t seed,
    const NoiseModel& noise_model, bool group_terms
) {
    const int n = circuit.n_qubits;
    double total = 0.0;

    // Split off identity terms (exact, no sampling) and normalise the rest.
    std::vector<size_t> nonid;
    nonid.reserve(observable.terms.size());
    for (size_t i = 0; i < observable.terms.size(); ++i) {
        const auto& term = observable.terms[i];
        if (static_cast<int>(term.pauli.size()) != n)
            throw std::invalid_argument(
                "Estimator: Pauli string length " + std::to_string(term.pauli.size()) +
                " does not match circuit n_qubits=" + std::to_string(n));
        bool is_id = true;
        for (char c : term.pauli) { if (norm_pauli(c, term.pauli) != 'I') { is_id = false; break; } }
        if (is_id) total += term.coeff.real;
        else nonid.push_back(i);
    }
    if (nonid.empty()) return total;

    uint64_t grp_seed = seed;

    if (!group_terms) {
        for (size_t i : nonid) {
            const auto& term = observable.terms[i];
            std::vector<char> basis(static_cast<size_t>(n), 'I');
            for (int q = 0; q < n; ++q) basis[static_cast<size_t>(q)] = norm_pauli(term.pauli[static_cast<size_t>(q)], term.pauli);
            auto counts = run_in_basis(circuit, basis, shots, grp_seed, noise_model);
            total += term.coeff.real * parity_expectation(counts, term.pauli, n);
            if (grp_seed != 0) ++grp_seed;
        }
        return total;
    }

    // Greedy qubit-wise-commutativity grouping: a term joins the first group
    // whose merged basis agrees with it on every non-I qubit.
    struct Group { std::vector<char> basis; std::vector<size_t> terms; };
    std::vector<Group> groups;
    for (size_t i : nonid) {
        const auto& p = observable.terms[i].pauli;
        int placed = -1;
        for (size_t g = 0; g < groups.size() && placed < 0; ++g) {
            bool ok = true;
            for (int q = 0; q < n && ok; ++q) {
                const char pc = norm_pauli(p[static_cast<size_t>(q)], p);
                const char gb = groups[g].basis[static_cast<size_t>(q)];
                if (pc != 'I' && gb != 'I' && pc != gb) ok = false;
            }
            if (ok) placed = static_cast<int>(g);
        }
        if (placed < 0) {
            groups.push_back({std::vector<char>(static_cast<size_t>(n), 'I'), {}});
            placed = static_cast<int>(groups.size()) - 1;
        }
        Group& g = groups[static_cast<size_t>(placed)];
        for (int q = 0; q < n; ++q) {
            const char pc = norm_pauli(p[static_cast<size_t>(q)], p);
            if (pc != 'I') g.basis[static_cast<size_t>(q)] = pc;
        }
        g.terms.push_back(i);
    }

    for (const auto& g : groups) {
        auto counts = run_in_basis(circuit, g.basis, shots, grp_seed, noise_model);
        for (size_t ti : g.terms) {
            const auto& term = observable.terms[ti];
            total += term.coeff.real * parity_expectation(counts, term.pauli, n);
        }
        if (grp_seed != 0) ++grp_seed;
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
        // CouplingMap() (n = 0) declares "no connectivity constraint": routing
        // passes are skipped, only the optimisation passes run. An edgeless
        // CouplingMap(n) would be a literal no-edge graph and unroutable.
        {
            QuantumCircuit transpiled = lindblad::transpile(circuit, CouplingMap(),
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

    // Strict exact-evaluation rule (docs/api/estimator.md): shots == 0 means
    // "exact expectation value", which is undefined for stochastic evolution.
    // Circuits with measurement or classical conditioning must be estimated
    // from sampled counts instead of silently evaluating one trajectory.
    if (options.shots <= 0) {
        for (const auto& inst : to_simulate.instructions) {
            if (inst.type == Instruction::GateType::MEASURE ||
                inst.condition_clbit >= 0) {
                throw std::invalid_argument(
                    "Estimator: shots == 0 requests an exact expectation "
                    "value, which is undefined for circuits containing "
                    "measurement or classically-conditioned instructions. "
                    "Set options.shots > 0 to estimate from sampled counts.");
            }
        }
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
            to_simulate, observable, options.shots, options.seed,
            options.noise_model, options.group_pauli_terms
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
