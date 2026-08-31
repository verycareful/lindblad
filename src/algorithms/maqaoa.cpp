#include "lindblad/algorithms.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <complex>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <iostream>

#include <nlopt.h>

namespace lindblad {
namespace algorithms {

// =============================================================================
// Orbit utility — orbits_by_power (Change 3)
// =============================================================================

std::vector<int> orbits_by_power(
    const std::vector<double>& powers,
    double tolerance
) {
    const int n = static_cast<int>(powers.size());
    std::vector<int> result(n, -1);
    std::vector<double> orbit_centers;
    orbit_centers.reserve(n);

    // Sort by value so that elements within tolerance are always adjacent.
    // This prevents insertion-order artifacts where two values within tolerance
    // are assigned different orbits because a third (out-of-range) value snuck
    // between their two appearances.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
        [&](int a, int b){ return powers[a] < powers[b]; });

    for (int idx : order) {
        int assigned = -1;
        for (int k = 0; k < static_cast<int>(orbit_centers.size()); ++k) {
            if (std::abs(powers[idx] - orbit_centers[k]) <= tolerance) {
                assigned = k;
                break;
            }
        }
        if (assigned == -1) {
            assigned = static_cast<int>(orbit_centers.size());
            orbit_centers.push_back(powers[idx]);
        }
        result[idx] = assigned;
    }
    return result;
}

// =============================================================================
// MA-QAOA: Multi-Angle QAOA
// =============================================================================

// Returns the number of distinct cost-term orbit groups for a Hamiltonian.
// Two terms are in the same orbit if their sorted tuple of active-qubit orbit
// indices is identical (orbit-equivalent support).
static int count_cost_orbits(
    const SparsePauliOp& cost_hamiltonian,
    const std::vector<int>& qubit_orbits
) {
    int nq = cost_hamiltonian.n_qubits();
    std::map<std::vector<int>, int> seen;
    int count = 0;
    for (const auto& term : cost_hamiltonian.terms) {
        std::vector<int> key;
        for (int q = 0; q < nq; ++q) {
            if (term.pauli[q] != 'I') {
                key.push_back(qubit_orbits[q]);
            }
        }
        std::sort(key.begin(), key.end());
        if (!seen.count(key)) { seen[key] = count++; }
    }
    return count;
}

// Maps each cost term to its orbit index (position in the sorted unique list).
static std::vector<int> cost_term_orbit_map(
    const SparsePauliOp& cost_hamiltonian,
    const std::vector<int>& qubit_orbits
) {
    int nq = cost_hamiltonian.n_qubits();
    std::map<std::vector<int>, int> seen;
    int count = 0;
    std::vector<int> result;
    for (const auto& term : cost_hamiltonian.terms) {
        std::vector<int> key;
        for (int q = 0; q < nq; ++q) {
            if (term.pauli[q] != 'I') key.push_back(qubit_orbits[q]);
        }
        std::sort(key.begin(), key.end());
        if (!seen.count(key)) seen[key] = count++;
        result.push_back(seen[key]);
    }
    return result;
}

static std::vector<std::vector<int>> active_qubits_by_term(const SparsePauliOp& hamiltonian) {
    const int nq = hamiltonian.n_qubits();
    const int term_count = static_cast<int>(hamiltonian.terms.size());
    std::vector<std::vector<int>> active_qubits(term_count);
    for (int t = 0; t < term_count; ++t) {
        const auto& term = hamiltonian.terms[t];
        for (int q = 0; q < nq; ++q) {
            if (term.pauli[q] != 'I') active_qubits[t].push_back(q);
        }
    }
    return active_qubits;
}

static void validate_real_mixer_coeffs(const SparsePauliOp& mixer, const char* caller) {
    for (const auto& term : mixer.terms) {
        if (term.coeff.imag != 0.0) {
            throw std::invalid_argument(
                std::string("MAQAOA::") + caller +
                ": custom mixer Hamiltonian must be Hermitian (real coefficients required)");
        }
    }
}

// Classical energy of a computational-basis bitstring under the diagonal
// (I/Z-only) part of the Hamiltonian.
// PRECONDITION: bitstring.size() == cost_hamiltonian.n_qubits(). Callers filter
// mismatched keys out; this helper does not signal failure in-band. Returning
// +infinity for a size mismatch would put a sentinel into a value the ranking
// loop at the end of optimize() compares as ordinary cost data.
static double computational_basis_cost(
    const SparsePauliOp& cost_hamiltonian,
    const std::string& bitstring
) {
    const int nq = cost_hamiltonian.n_qubits();

    double energy = 0.0;
    for (const auto& term : cost_hamiltonian.terms) {
        double eigenvalue = 1.0;
        bool diagonal = true;

        for (int q = 0; q < nq; ++q) {
            const char p = term.pauli[q];
            if (p == 'I') continue;
            if (p == 'Z') {
                const char bit = bitstring[nq - 1 - q];
                eigenvalue *= (bit == '1') ? -1.0 : 1.0;
            } else {
                diagonal = false;
                break;
            }
        }

        if (diagonal) {
            energy += term.coeff.real * eigenvalue;
        }
    }
    return energy;
}

// =============================================================================
// Direct statevector evolution (Change 1)
//
// Bypasses QuantumCircuit construction, parameter binding, transpile-cache
// key computation, and instruction dispatch on every COBYLA evaluation.
// sv is reset to |0...0> then |+...+>, then all p layers are applied in-place.
//
// Preconditions:
//   term_orbit_map  — empty when orbit mode is off
//   orbit_assignments — empty when orbit mode is off
//   n_cost_params_per_layer and n_mixer_orbits precomputed at optimize() entry
// =============================================================================

static void evolve_into(
    Statevector& sv,
    const SparsePauliOp& cost,
    const SparsePauliOp& mixer,
    const std::vector<double>& params,
    int p,
    const std::vector<int>& term_orbit_map,
    int n_cost_params_per_layer,
    int n_mixer_orbits,
    const std::vector<int>& orbit_assignments,
    const std::vector<std::vector<int>>& precomp_aq,  // precomputed per-term active qubits
    const std::vector<std::vector<int>>& precomp_mixer_aq,
    const std::vector<double>& initial_thetas = {}    // QSP: empty = standard H init
) {
    const int nq          = cost.n_qubits();
    const int cost_terms  = static_cast<int>(cost.terms.size());
    const bool use_orbits = (static_cast<int>(orbit_assignments.size()) == nq);
    const bool has_custom_mixer = !mixer.terms.empty();

    sv.initialize();
    if (!initial_thetas.empty() &&
        static_cast<int>(initial_thetas.size()) == nq) {
        for (int q = 0; q < nq; ++q)
            gates::apply_ry(sv, q, initial_thetas[q]);
    } else {
        for (int q = 0; q < nq; ++q) gates::apply_h(sv, q);
    }

    int param_idx = 0;
    std::vector<double> layer_gammas(n_cost_params_per_layer);
    std::vector<double> layer_betas(n_mixer_orbits);

    for (int layer = 0; layer < p; ++layer) {
        // Cost unitary: one gamma per (orbit-)term
        for (int i = 0; i < n_cost_params_per_layer; ++i) {
            layer_gammas[i] = (param_idx < static_cast<int>(params.size()))
                              ? params[param_idx++] : 0.0;
        }

        for (int t = 0; t < cost_terms; ++t) {
            const auto& term = cost.terms[t];

            // Use precomputed active qubits — eliminates hot-path allocation.
            const auto& aq = precomp_aq[t];
            if (aq.empty()) continue;

            // Gamma dispatch: orbit-indexed → orbit map; term-indexed → t;
            // qubit-indexed (default) → aq[0] (lowest active qubit)
            int gamma_idx;
            if (use_orbits) {
                gamma_idx = term_orbit_map[t];
            } else if (n_cost_params_per_layer == cost_terms) {
                gamma_idx = t;
            } else {
                gamma_idx = aq[0];
            }

            const double gamma = layer_gammas[gamma_idx];
            const double angle = 2.0 * gamma * term.coeff.real;

            if (aq.size() == 1 && term.pauli[aq[0]] == 'Z') {
                gates::apply_rz(sv, aq[0], angle);
            } else if (aq.size() == 2 &&
                       term.pauli[aq[0]] == 'Z' &&
                       term.pauli[aq[1]] == 'Z') {
                gates::apply_cx(sv, aq[0], aq[1]);
                gates::apply_rz(sv, aq[1], angle);
                gates::apply_cx(sv, aq[0], aq[1]);
            } else {
                // General Pauli rotation: basis-change -> CNOT chain -> Rz -> uncompute
                for (int q : aq) {
                    if      (term.pauli[q] == 'X') gates::apply_h(sv, q);
                    else if (term.pauli[q] == 'Y') {
                        gates::apply_sdg(sv, q);
                        gates::apply_h(sv, q);
                    }
                }
                for (size_t i = 0; i + 1 < aq.size(); ++i)
                    gates::apply_cx(sv, aq[i], aq[i + 1]);
                gates::apply_rz(sv, aq.back(), angle);
                for (int i = static_cast<int>(aq.size()) - 2; i >= 0; --i)
                    gates::apply_cx(sv, aq[i], aq[i + 1]);
                for (int q : aq) {
                    if      (term.pauli[q] == 'X') gates::apply_h(sv, q);
                    else if (term.pauli[q] == 'Y') {
                        gates::apply_h(sv, q);
                        gates::apply_s(sv, q);
                    }
                }
            }
        }

        // Mixer unitary:
        // - default (empty mixer): fixed per-qubit transverse-field RX
        //   U_B(beta_l) = prod_i RX(2*beta_l_i) (MA-QAOA paper ansatz)
        // - custom mixer: ordered product of per-term rotations
        //   exp(-i*beta*c_k*P_k), exact when mixer terms commute and a
        //   first-order Trotter step otherwise.
        //
        // Beta dispatch for custom terms follows the default MA-QAOA sharing:
        // the lowest active qubit of each mixer term selects its beta index
        // (or that qubit's orbit index when orbit sharing is enabled).
        for (int i = 0; i < n_mixer_orbits; ++i) {
            layer_betas[i] = (param_idx < static_cast<int>(params.size()))
                             ? params[param_idx++] : 0.0;
        }

        if (!has_custom_mixer) {
            for (int q = 0; q < nq; ++q) {
                const int beta_idx = use_orbits ? orbit_assignments[q] : q;
                gates::apply_rx(sv, q, 2.0 * layer_betas[beta_idx]);
            }
            continue;
        }

        const int mixer_terms = static_cast<int>(mixer.terms.size());
        for (int t = 0; t < mixer_terms; ++t) {
            const auto& term = mixer.terms[t];
            const auto& aq = precomp_mixer_aq[t];
            if (aq.empty()) continue;

            const int beta_idx = use_orbits ? orbit_assignments[aq[0]] : aq[0];
            const double beta  = layer_betas[beta_idx];
            const double angle = 2.0 * beta * term.coeff.real;

            if (aq.size() == 1) {
                const int q = aq[0];
                if (term.pauli[q] == 'X') gates::apply_rx(sv, q, angle);
                else if (term.pauli[q] == 'Y') gates::apply_ry(sv, q, angle);
                else gates::apply_rz(sv, q, angle);
            } else {
                for (int q : aq) {
                    if      (term.pauli[q] == 'X') gates::apply_h(sv, q);
                    else if (term.pauli[q] == 'Y') {
                        gates::apply_sdg(sv, q);
                        gates::apply_h(sv, q);
                    }
                }
                for (size_t i = 0; i + 1 < aq.size(); ++i)
                    gates::apply_cx(sv, aq[i], aq[i + 1]);
                gates::apply_rz(sv, aq.back(), angle);
                for (int i = static_cast<int>(aq.size()) - 2; i >= 0; --i)
                    gates::apply_cx(sv, aq[i], aq[i + 1]);
                for (int q : aq) {
                    if      (term.pauli[q] == 'X') gates::apply_h(sv, q);
                    else if (term.pauli[q] == 'Y') {
                        gates::apply_h(sv, q);
                        gates::apply_s(sv, q);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Non-layerwise callback
// =============================================================================

struct MAQAOACallbackData {
    const SparsePauliOp* cost_hamiltonian;
    const SparsePauliOp* mixer_hamiltonian;
    const std::vector<int>* term_orbit_map;
    int n_cost_params_per_layer;
    int n_mixer_orbits;
    const MAQAOA* maqaoa;
    Statevector* sv;
    std::vector<double> params_buf;
    int nfev;
    // Best objective seen so far, guarded by best_val_valid rather than seeded
    // with +infinity. best_val reaches the caller (per_layer_costs), so a seed
    // that was never overwritten is a wrong result rather than a missing
    // diagnostic. A bool states "nothing selected yet" directly and carries no
    // floating-point meaning for the optimiser to reason about. Same pattern in
    // LayerCBData below.
    double best_val;
    bool best_val_valid;
    std::vector<double> initial_thetas;
    const std::vector<std::vector<int>>* active_qubits;
    const std::vector<std::vector<int>>* mixer_active_qubits;
};

static double maqaoa_objective(unsigned n, const double* x, double* /*grad*/, void* data) {
    auto* cb = static_cast<MAQAOACallbackData*>(data);
    if (cb->params_buf.size() != n) {
        cb->params_buf.resize(n);
    }
    std::copy(x, x + n, cb->params_buf.begin());
    if (!cb->maqaoa->estimator.options.noise_model.is_ideal()) {
        auto circuit = cb->maqaoa->build_circuit(
            *cb->cost_hamiltonian, *cb->mixer_hamiltonian, cb->params_buf);
        lindblad::DensityMatrixSimulator dm_sim;
        auto dm_result = dm_sim.run(
            circuit, cb->maqaoa->estimator.options.noise_model, 0, 0);
        const double value = dm_result.success
            ? dm_result.final_state.expectation_value_sparse(*cb->cost_hamiltonian)
            : 1e12;
        const double v = is_finite_strict(value) ? value : 1e12;
        ++cb->nfev;
        if (!cb->best_val_valid || v < cb->best_val) {
            cb->best_val       = v;
            cb->best_val_valid = true;
        }
        return v;
    }
    evolve_into(*cb->sv, *cb->cost_hamiltonian, *cb->mixer_hamiltonian, cb->params_buf,
                cb->maqaoa->options.p,
                *cb->term_orbit_map, cb->n_cost_params_per_layer,
                cb->n_mixer_orbits,
                cb->maqaoa->options.orbit_assignments,
                *cb->active_qubits,
                *cb->mixer_active_qubits,
                cb->initial_thetas);
    const double value = cb->cost_hamiltonian->expectation_value(*cb->sv);
    const double v     = is_finite_strict(value) ? value : 1e12;
    ++cb->nfev;
    if (!cb->best_val_valid || v < cb->best_val) {
        cb->best_val       = v;
        cb->best_val_valid = true;
    }
    return v;
}

// =============================================================================
// Layerwise callback (Change 7: all_params pre-allocated, free portion
// updated in-place — no per-evaluation vector copy)
// =============================================================================

struct LayerCBData {
    const MAQAOA*        maqaoa;
    const SparsePauliOp* cost_hamiltonian;
    const SparsePauliOp* mixer_hamiltonian;
    std::vector<double>  all_params;         // [frozen | free] — full-run parameter vector
    int                  free_start;         // index of first free parameter
    int                  p_total;            // active layers in this optimisation stage
    const std::vector<int>* term_orbit_map;
    int                  n_cost_params_per_layer;
    int                  n_mixer_orbits;
    const std::vector<int>* orbit_assignments;
    Statevector*         sv;
    int                  p_current;
    int                  nfev;
    double               best_val;        // guarded by best_val_valid, not +inf
    bool                 best_val_valid;  // see MAQAOACallbackData above
    std::vector<double>  initial_thetas;
    const std::vector<std::vector<int>>* active_qubits;
    const std::vector<std::vector<int>>* mixer_active_qubits;
};

static double layer_objective(unsigned n, const double* x, double* /*grad*/, void* raw) {
    auto* d = static_cast<LayerCBData*>(raw);
    // Update free portion in-place (Change 7: no allocation, no copy)
    std::copy(x, x + n, d->all_params.begin() + d->free_start);
    if (!d->maqaoa->estimator.options.noise_model.is_ideal()) {
        auto circuit = d->maqaoa->build_circuit(
            *d->cost_hamiltonian, *d->mixer_hamiltonian, d->all_params);
        lindblad::DensityMatrixSimulator dm_sim;
        auto dm_result = dm_sim.run(
            circuit, d->maqaoa->estimator.options.noise_model, 0, 0);
        const double value = dm_result.success
            ? dm_result.final_state.expectation_value_sparse(*d->cost_hamiltonian)
            : 1e12;
        const double v     = is_finite_strict(value) ? value : 1e12;
        ++d->nfev;
        if (!d->best_val_valid || v < d->best_val) {
            d->best_val       = v;
            d->best_val_valid = true;
        }
        if (d->nfev % 50 == 0) {
            std::cout << "[MAQAOA] layer=" << d->p_current
                      << " eval=" << d->nfev
                      << " best=" << d->best_val
                      << std::endl;
        }
        return v;
    }
    evolve_into(*d->sv, *d->cost_hamiltonian, *d->mixer_hamiltonian,
                d->all_params, d->p_total,
                *d->term_orbit_map, d->n_cost_params_per_layer,
                d->n_mixer_orbits, *d->orbit_assignments,
                *d->active_qubits,
                *d->mixer_active_qubits,
                d->initial_thetas);
    const double value = d->cost_hamiltonian->expectation_value(*d->sv);
    const double v     = is_finite_strict(value) ? value : 1e12;
    ++d->nfev;
    if (!d->best_val_valid || v < d->best_val) {
        d->best_val       = v;
        d->best_val_valid = true;
    }
    if (d->nfev % 50 == 0) {
        std::cout << "[MAQAOA] layer=" << d->p_current
                  << " eval=" << d->nfev
                  << " best=" << d->best_val
                  << std::endl;
    }
    return v;
}

// =============================================================================
// num_parameters
// =============================================================================

int MAQAOA::num_parameters(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian
) const {
    int nq = cost_hamiltonian.n_qubits();
    if (!mixer_hamiltonian.terms.empty() && mixer_hamiltonian.n_qubits() != nq) {
        throw std::invalid_argument(
            "MAQAOA::num_parameters: cost and mixer Hamiltonians must have the same number of qubits");
    }

    int cost_params, mixer_params;
    if (!options.orbit_assignments.empty() &&
        static_cast<int>(options.orbit_assignments.size()) == nq) {
        cost_params  = count_cost_orbits(cost_hamiltonian, options.orbit_assignments);
        mixer_params = *std::max_element(options.orbit_assignments.begin(),
                                          options.orbit_assignments.end()) + 1;
    } else if (options.term_indexed_gammas) {
        cost_params  = static_cast<int>(cost_hamiltonian.terms.size());
        mixer_params = nq;
    } else {
        cost_params  = nq;
        mixer_params = nq;
    }
    return options.p * (cost_params + mixer_params);
}

// =============================================================================
// optimize
// =============================================================================

MAQAOA::Result MAQAOA::optimize(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian_in
) {
    Result result;
    result.converged         = false;
    result.num_iterations    = 0;
    result.wall_time_seconds = 0.0;

    const int nq = cost_hamiltonian.n_qubits();
    SparsePauliOp mixer = mixer_hamiltonian_in;
    if (!mixer.terms.empty() && mixer.n_qubits() != nq) {
        throw std::invalid_argument(
            "MAQAOA::optimize: cost and mixer Hamiltonians must have the same number of qubits");
    }
    validate_real_mixer_coeffs(mixer, "optimize");

    // Precompute orbit data once for the entire run (Change 6)
    const bool use_orbits = (!options.orbit_assignments.empty() &&
                             static_cast<int>(options.orbit_assignments.size()) == nq);
    std::vector<int> term_orbit_map_cached;
    int n_cost_params_per_layer;
    int n_mixer_orbits;

    if (use_orbits) {
        term_orbit_map_cached   = cost_term_orbit_map(cost_hamiltonian, options.orbit_assignments);
        n_cost_params_per_layer = count_cost_orbits(cost_hamiltonian, options.orbit_assignments);
        n_mixer_orbits          = *std::max_element(options.orbit_assignments.begin(),
                                                     options.orbit_assignments.end()) + 1;
    } else if (options.term_indexed_gammas) {
        n_cost_params_per_layer = static_cast<int>(cost_hamiltonian.terms.size());
        n_mixer_orbits          = nq;
    } else {
        n_cost_params_per_layer = nq;
        n_mixer_orbits          = nq;
    }

    const int params_per_layer = n_cost_params_per_layer + n_mixer_orbits;
    const int n_params         = num_parameters(cost_hamiltonian, mixer);
    constexpr double kPi       = PI;
    constexpr double kBound    = 2.0 * kPi;

    // Precompute active qubits per cost term once — eliminates ~1M hot-path allocations
    // across 10K optimizer evaluations x 100 terms (P-8).
    std::vector<std::vector<int>> active_qubits_per_term = active_qubits_by_term(cost_hamiltonian);
    std::vector<std::vector<int>> active_qubits_per_mixer_term = active_qubits_by_term(mixer);

    // Single statevector allocation reused across all evaluations (Change 1)
    Statevector inner_sv(nq);

    // Validate PI-MA-QAOA mixer_weights size (Change 2)
    const bool has_mw = (!options.mixer_weights.empty() &&
                         static_cast<int>(options.mixer_weights.size()) == n_mixer_orbits);
    const double w_max = has_mw
        ? *std::max_element(options.mixer_weights.begin(), options.mixer_weights.end())
        : 0.0;

    const auto t_global_start = std::chrono::steady_clock::now();

    // Seeded RNG for initial-parameter perturbation
    std::mt19937_64 rng(options.seed != 0 ? static_cast<uint64_t>(options.seed)
                                          : static_cast<uint64_t>(std::random_device{}()));
    std::uniform_real_distribution<double> perturb(-0.05, 0.05);

    // -------------------------------------------------------------------------
    // Layerwise path
    // -------------------------------------------------------------------------
    if (options.layerwise) {
        std::vector<double> all_params;
        all_params.reserve(n_params);

        bool all_layers_converged = true;

        for (int layer = 0; layer < options.p; ++layer) {
            const auto t_layer_start = std::chrono::steady_clock::now();

            // Initialise this layer's parameters (Change 2: PI-MA-QAOA beta init)
            // Gammas: random perturbation seeded by options.seed
            for (int i = 0; i < n_cost_params_per_layer; ++i) {
                all_params.push_back(perturb(rng));
            }
            // Betas: PI-MA-QAOA when mixer_weights provided, else same alternating
            // pattern continuing from where gammas left off (identical to original)
            if (has_mw) {
                for (int i = 0; i < n_mixer_orbits; ++i) {
                    all_params.push_back(
                        options.beta_base * (options.mixer_weights[i] / w_max) + perturb(rng)
                    );
                }
            } else {
                for (int j = 0; j < n_mixer_orbits; ++j) {
                    all_params.push_back(perturb(rng));
                }
            }

            const int free_start = options.progressive ? 0 : layer * params_per_layer;
            const int n_free     = options.progressive ? (layer + 1) * params_per_layer
                                                       : params_per_layer;

            LayerCBData cb{
                this,
                &cost_hamiltonian,
                &mixer,
                all_params,       // copy: frozen prefix + this layer's init (Change 7)
                free_start,
                layer + 1,
                &term_orbit_map_cached,
                n_cost_params_per_layer,
                n_mixer_orbits,
                &options.orbit_assignments,
                &inner_sv,
                layer,
                0,
                0.0,     // best_val: meaningless until best_val_valid is set
                false,   // best_val_valid
                options.initial_thetas,
                &active_qubits_per_term,
                &active_qubits_per_mixer_term
            };

            std::cout << "[MAQAOA] layer=" << layer
                      << " starting, free_params=" << n_free
                      << " total_layers=" << (layer + 1)
                      << " budget=" << options.max_iterations
                      << std::endl;

            nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, n_free);
            nlopt_set_min_objective(opt, layer_objective, &cb);
            nlopt_set_maxeval(opt, options.max_iterations);
            nlopt_set_xtol_rel(opt, options.convergence_threshold);

            std::vector<double> lb(n_free, -kBound);
            std::vector<double> ub(n_free, kBound);
            nlopt_set_lower_bounds(opt, lb.data());
            nlopt_set_upper_bounds(opt, ub.data());
            std::vector<double> initial_step(n_free, 0.3);
            nlopt_set_initial_step(opt, initial_step.data());

            std::vector<double> x0(all_params.begin() + free_start, all_params.end());
            // Save initial guess before COBYLA modifies x0 (Change 4)
            result.initial_params.insert(result.initial_params.end(), x0.begin(), x0.end());

            // NLopt can return without writing min_val. The marker for that is
            // a detectably non-finite NaN (see quiet_nan_strict in types.hpp).
            // The integer nlopt_res is checked alongside it and carries no
            // floating-point meaning for the optimiser to reason about.
            double min_val              = quiet_nan_strict();
            const nlopt_result nlopt_res = nlopt_optimize(opt, x0.data(), &min_val);
            nlopt_destroy(opt);

            const auto   t_layer_end  = std::chrono::steady_clock::now();
            const double layer_wall   = std::chrono::duration<double>(
                t_layer_end - t_layer_start).count();

            std::cout << "[MAQAOA] layer=" << layer
                      << " done, nlopt_res=" << nlopt_res
                      << " nfev=" << cb.nfev
                      << " best=" << cb.best_val
                      << " wall_time=" << layer_wall << "s"
                      << std::endl;

            if (nlopt_res < 0 || nlopt_res == NLOPT_MAXEVAL_REACHED || !is_finite_strict(min_val)) {
                all_layers_converged = false;
            }

            // Write optimised params back into the outer all_params
            for (int i = 0; i < n_free; ++i) {
                all_params[free_start + i] = x0[i];
            }

            result.per_layer_costs.push_back(cb.best_val);    // Change 4
            result.layer_nfev.push_back(cb.nfev);              // Change 4
            result.wall_time_by_layer.push_back(layer_wall);   // Change 4
            result.num_iterations += cb.nfev;
        }

        result.optimal_params = all_params;

        // Final eval + sampling directly from the evolved statevector (Change 10)
        // No circuit rebuild, no estimator overhead, no second sampler run.
        if (!estimator.options.noise_model.is_ideal()) {
            auto circuit = build_circuit(cost_hamiltonian, mixer, all_params);
            lindblad::DensityMatrixSimulator dm_sim;
            auto dm_result = dm_sim.run(circuit, estimator.options.noise_model, 0, 0);
            result.optimal_value = dm_result.success
                ? dm_result.final_state.expectation_value_sparse(cost_hamiltonian)
                : 1e12;
        } else {
            evolve_into(inner_sv, cost_hamiltonian, mixer, all_params, options.p,
                        term_orbit_map_cached, n_cost_params_per_layer,
                        n_mixer_orbits, options.orbit_assignments,
                        active_qubits_per_term, active_qubits_per_mixer_term,
                        options.initial_thetas);
            result.optimal_value = cost_hamiltonian.expectation_value(inner_sv);
        }
        result.converged     = all_layers_converged && is_finite_strict(result.optimal_value);

        if (!is_finite_strict(result.optimal_value)) result.optimal_value = 1e12;

        if (!sampler.options.noise_model.is_ideal()) {
            auto circuit = build_circuit(cost_hamiltonian, mixer, all_params);
            lindblad::DensityMatrixSimulator dm_sim;
            result.counts = dm_sim.run(
                circuit, sampler.options.noise_model,
                sampler.options.shots, sampler.options.seed).counts;
        } else {
            result.counts = inner_sv.sample_counts(sampler.options.shots, sampler.options.seed);
        }

    // -------------------------------------------------------------------------
    // Standard path: all parameters optimised at once
    // -------------------------------------------------------------------------
    } else {
        std::vector<double> params;
        params.reserve(n_params);

        for (int layer = 0; layer < options.p; ++layer) {
            for (int i = 0; i < n_cost_params_per_layer; ++i) {
                params.push_back(perturb(rng));
            }
            if (has_mw) {
                for (int i = 0; i < n_mixer_orbits; ++i) {
                    params.push_back(
                        options.beta_base * (options.mixer_weights[i] / w_max) + perturb(rng)
                    );
                }
            } else {
                for (int j = 0; j < n_mixer_orbits; ++j) {
                    params.push_back(perturb(rng));
                }
            }
        }

        result.initial_params = params;   // Change 4

        MAQAOACallbackData cb_data{
            &cost_hamiltonian, &mixer,
            &term_orbit_map_cached,
            n_cost_params_per_layer, n_mixer_orbits,
            this, &inner_sv, {}, 0,
            0.0,     // best_val: meaningless until best_val_valid is set
            false,   // best_val_valid
            options.initial_thetas,
            &active_qubits_per_term,
            &active_qubits_per_mixer_term
        };
        cb_data.params_buf.resize(n_params);

        nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, n_params);
        nlopt_set_min_objective(opt, maqaoa_objective, &cb_data);
        nlopt_set_maxeval(opt, options.max_iterations);
        nlopt_set_xtol_rel(opt, options.convergence_threshold);

        std::vector<double> lb(n_params, -kBound);
        std::vector<double> ub(n_params, kBound);
        nlopt_set_lower_bounds(opt, lb.data());
        nlopt_set_upper_bounds(opt, ub.data());
        std::vector<double> initial_step(n_params, 0.3);
        nlopt_set_initial_step(opt, initial_step.data());

        // See the layerwise path above for why the marker is bit-built.
        double min_val               = quiet_nan_strict();
        const nlopt_result nlopt_res  = nlopt_optimize(opt, params.data(), &min_val);
        nlopt_destroy(opt);

        result.optimal_value  = is_finite_strict(min_val) ? min_val : 1e12;
        result.optimal_params = params;
        result.converged      = (nlopt_res > 0) &&
                    (nlopt_res != NLOPT_MAXEVAL_REACHED) &&
                    is_finite_strict(min_val);
        result.num_iterations = cb_data.nfev;

        // Sampling directly from the evolved statevector (Change 10)
        if (!sampler.options.noise_model.is_ideal()) {
            auto circuit = build_circuit(cost_hamiltonian, mixer, params);
            lindblad::DensityMatrixSimulator dm_sim;
            result.counts = dm_sim.run(
                circuit, sampler.options.noise_model,
                sampler.options.shots, sampler.options.seed).counts;
        } else {
            evolve_into(inner_sv, cost_hamiltonian, mixer, params, options.p,
                        term_orbit_map_cached, n_cost_params_per_layer,
                        n_mixer_orbits, options.orbit_assignments,
                        active_qubits_per_term, active_qubits_per_mixer_term,
                        options.initial_thetas);
            result.counts = inner_sv.sample_counts(sampler.options.shots, sampler.options.seed);
        }
    }

    const auto t_global_end     = std::chrono::steady_clock::now();
    result.wall_time_seconds     = std::chrono::duration<double>(
        t_global_end - t_global_start).count();

    // Rank by computational-basis objective; break ties by sample count.
    // have_best rather than a +infinity seed: see MAQAOACallbackData::best_val.
    double best_cost = 0.0;
    int best_count = -1;
    bool have_best = false;
    for (const auto& [bits, count] : result.counts) {
        if (static_cast<int>(bits.size()) != nq) continue;
        const double cost = computational_basis_cost(cost_hamiltonian, bits);
        if (!have_best || cost < best_cost ||
            (cost == best_cost && count > best_count)) {
            best_cost             = cost;
            best_count            = count;
            have_best             = true;
            result.best_bitstring = bits;
        }
    }

    return result;
}

// =============================================================================
// build_circuit — public entry point for offline inspection. Not on the hot
// path: optimize() uses evolve_into directly. Orbit maps are recomputed
// internally here since this path is cold.
// =============================================================================

QuantumCircuit MAQAOA::build_circuit(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian,
    const std::vector<double>& params
) const {
    int nq = cost_hamiltonian.n_qubits();
    if (!mixer_hamiltonian.terms.empty() && mixer_hamiltonian.n_qubits() != nq) {
        throw std::invalid_argument(
            "MAQAOA::build_circuit: cost and mixer Hamiltonians must have the same number of qubits");
    }
    validate_real_mixer_coeffs(mixer_hamiltonian, "build_circuit");
    QuantumCircuit qc(nq);

    if (!options.initial_thetas.empty() &&
        static_cast<int>(options.initial_thetas.size()) == nq) {
        for (int q = 0; q < nq; ++q)
            qc.ry(options.initial_thetas[q], q);
    } else {
        for (int q = 0; q < nq; ++q)
            qc.h(q);
    }

    int param_idx = 0;
    int cost_terms = static_cast<int>(cost_hamiltonian.terms.size());

    // Determine orbit mode
    const bool use_orbits = (!options.orbit_assignments.empty() &&
                             static_cast<int>(options.orbit_assignments.size()) == nq);
    std::vector<int> term_orbit_map;
    int n_mixer_orbits = nq;
    if (use_orbits) {
        term_orbit_map = cost_term_orbit_map(cost_hamiltonian, options.orbit_assignments);
        n_mixer_orbits = *std::max_element(options.orbit_assignments.begin(),
                                            options.orbit_assignments.end()) + 1;
    }

    for (int layer = 0; layer < options.p; ++layer) {
        // Cost unitary
        int n_cost_params;
        if (use_orbits) {
            n_cost_params = count_cost_orbits(cost_hamiltonian, options.orbit_assignments);
        } else if (options.term_indexed_gammas) {
            n_cost_params = cost_terms;
        } else {
            n_cost_params = nq;
        }

        std::vector<double> layer_gammas(n_cost_params);
        for (int i = 0; i < n_cost_params; ++i) {
            layer_gammas[i] = (param_idx < static_cast<int>(params.size())) ?
                              params[param_idx++] : 0.0;
        }

        for (int t = 0; t < cost_terms; ++t) {
            const auto& term = cost_hamiltonian.terms[t];

            std::vector<int> active_qubits;
            for (int q = 0; q < nq; ++q) {
                if (term.pauli[q] != 'I') active_qubits.push_back(q);
            }
            if (active_qubits.empty()) continue;

            int gamma_idx;
            if (use_orbits) {
                gamma_idx = term_orbit_map[t];
            } else if (n_cost_params == cost_terms) {
                gamma_idx = t;
            } else {
                gamma_idx = active_qubits[0];
            }

            double gamma = layer_gammas[gamma_idx];
            double angle = 2.0 * gamma * term.coeff.real;

            if (active_qubits.size() == 1 && term.pauli[active_qubits[0]] == 'Z') {
                qc.rz(angle, active_qubits[0]);
            } else if (active_qubits.size() == 2 &&
                       term.pauli[active_qubits[0]] == 'Z' &&
                       term.pauli[active_qubits[1]] == 'Z') {
                qc.cx(active_qubits[0], active_qubits[1]);
                qc.rz(angle, active_qubits[1]);
                qc.cx(active_qubits[0], active_qubits[1]);
            } else {
                for (int q : active_qubits) {
                    if (term.pauli[q] == 'X') qc.h(q);
                    else if (term.pauli[q] == 'Y') { qc.sdg(q); qc.h(q); }
                }
                for (size_t i = 0; i + 1 < active_qubits.size(); ++i) {
                    qc.cx(active_qubits[i], active_qubits[i + 1]);
                }
                qc.rz(angle, active_qubits.back());
                for (int i = static_cast<int>(active_qubits.size()) - 2; i >= 0; --i) {
                    qc.cx(active_qubits[i], active_qubits[i + 1]);
                }
                for (int q : active_qubits) {
                    if (term.pauli[q] == 'X') qc.h(q);
                    else if (term.pauli[q] == 'Y') { qc.h(q); qc.s(q); }
                }
            }
        }

        // Mixer unitary
        std::vector<double> layer_betas(n_mixer_orbits);
        for (int i = 0; i < n_mixer_orbits; ++i) {
            layer_betas[i] = (param_idx < static_cast<int>(params.size())) ?
                             params[param_idx++] : 0.0;
        }

        if (mixer_hamiltonian.terms.empty()) {
            for (int q = 0; q < nq; ++q) {
                int beta_idx = use_orbits ? options.orbit_assignments[q] : q;
                qc.rx(2.0 * layer_betas[beta_idx], q);
            }
            continue;
        }

        for (const auto& term : mixer_hamiltonian.terms) {
            std::vector<int> active_qubits;
            for (int q = 0; q < nq; ++q) {
                if (term.pauli[q] != 'I') active_qubits.push_back(q);
            }
            if (active_qubits.empty()) continue;

            const int beta_idx = use_orbits
                ? options.orbit_assignments[active_qubits[0]]
                : active_qubits[0];
            const double beta = layer_betas[beta_idx];
            const double angle = 2.0 * beta * term.coeff.real;

            if (active_qubits.size() == 1) {
                const int q = active_qubits[0];
                if (term.pauli[q] == 'X') qc.rx(angle, q);
                else if (term.pauli[q] == 'Y') qc.ry(angle, q);
                else qc.rz(angle, q);
            } else {
                for (int q : active_qubits) {
                    if (term.pauli[q] == 'X') qc.h(q);
                    else if (term.pauli[q] == 'Y') { qc.sdg(q); qc.h(q); }
                }
                for (size_t i = 0; i + 1 < active_qubits.size(); ++i) {
                    qc.cx(active_qubits[i], active_qubits[i + 1]);
                }
                qc.rz(angle, active_qubits.back());
                for (int i = static_cast<int>(active_qubits.size()) - 2; i >= 0; --i) {
                    qc.cx(active_qubits[i], active_qubits[i + 1]);
                }
                for (int q : active_qubits) {
                    if (term.pauli[q] == 'X') qc.h(q);
                    else if (term.pauli[q] == 'Y') { qc.h(q); qc.s(q); }
                }
            }
        }
    }

    return qc;
}

} // namespace algorithms
} // namespace lindblad
