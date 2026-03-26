#include "qpp/algorithms.hpp"
#include "qpp/gates.hpp"
#include "qpp/simulators/statevector_sim.hpp"

#include <algorithm>
#include <complex>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <iostream>

#include <nlopt.h>

namespace qpp {
namespace algorithms {

// =============================================================================
// MA-QAOA: Multi-Angle QAOA
// Each gate in the cost/mixer layers gets its own independent angle
// =============================================================================

struct MAQAOACallbackData {
    Estimator* estimator;
    const SparsePauliOp* cost_hamiltonian;
    const SparsePauliOp* mixer_hamiltonian;
    const MAQAOA* maqaoa;
};

static double maqaoa_objective(unsigned n, const double* x, double* /*grad*/, void* data) {
    auto* cb = static_cast<MAQAOACallbackData*>(data);
    std::vector<double> params(x, x + n);
    auto circuit = cb->maqaoa->build_circuit(*cb->cost_hamiltonian, *cb->mixer_hamiltonian, params);
    const double value = cb->estimator->run_single(circuit, *cb->cost_hamiltonian);
    return std::isfinite(value) ? value : 1e12;
}

// =============================================================================
// Orbit-QAOA helpers
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

int MAQAOA::num_parameters(const SparsePauliOp& cost_hamiltonian) const {
    int nq = cost_hamiltonian.n_qubits();

    int cost_params, mixer_params;
    if (!options.orbit_assignments.empty() &&
        static_cast<int>(options.orbit_assignments.size()) == nq) {
        // Orbit-reduced counts
        cost_params = count_cost_orbits(cost_hamiltonian, options.orbit_assignments);
        int n_orbits = *std::max_element(options.orbit_assignments.begin(),
                                         options.orbit_assignments.end()) + 1;
        mixer_params = n_orbits;
    } else {
        // Standard MA-QAOA: one angle per term / per qubit
        cost_params = static_cast<int>(cost_hamiltonian.terms.size());
        mixer_params = nq;
    }
    return options.p * (cost_params + mixer_params);
}

MAQAOA::Result MAQAOA::optimize(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian_in
) {
    Result result;
    result.converged = false;

    int nq = cost_hamiltonian.n_qubits();

    SparsePauliOp mixer = mixer_hamiltonian_in;
    if (mixer.terms.empty()) {
        for (int q = 0; q < nq; ++q) {
            std::string pauli(nq, 'I');
            pauli[q] = 'X';
            mixer.terms.push_back({pauli, Complex128(1.0, 0.0)});
        }
    }

    int n_params = num_parameters(cost_hamiltonian);
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kBound = 2.0 * kPi;  // Allow [0, 2π]

    if (options.layerwise) {
        // Layerwise training: optimise one layer at a time.
        // Previous layer parameters are FROZEN during optimisation of a new layer.
        std::vector<double> all_params;
        int cost_terms = static_cast<int>(cost_hamiltonian.terms.size());
        int mixer_terms = nq;
        int params_per_layer = cost_terms + mixer_terms;

        bool all_layers_converged = true;

        for (int layer = 0; layer < options.p; ++layer) {
            // Add new layer parameters (initialised near 0 to start from |+⟩ regime)
            for (int i = 0; i < params_per_layer; ++i) {
                all_params.push_back(0.1 * (i % 2 == 0 ? 1.0 : -1.0));
            }

            // Create a p=(layer+1) MAQAOA that uses a lambda objective
            // where only the last `params_per_layer` params are free;
            // the rest are fixed from previous rounds.
            int offset = layer * params_per_layer;
            const std::vector<double> frozen_prefix(all_params.begin(),
                                                     all_params.begin() + offset);

            // Custom callback struct for this layer
            struct LayerCBData {
                Estimator* estimator;
                const SparsePauliOp* cost_hamiltonian;
                const SparsePauliOp* mixer_hamiltonian;
                const MAQAOA* maqaoa;
                std::vector<double> frozen;
                int p_current;
                int nfev;
                double best_val;
            };

            static auto layer_objective = [](unsigned n, const double* x,
                                              double* /*grad*/, void* raw) -> double {
                auto* d = static_cast<LayerCBData*>(raw);
                std::vector<double> all = d->frozen;
                all.insert(all.end(), x, x + n);
                auto circuit = d->maqaoa->build_circuit(
                    *d->cost_hamiltonian, *d->mixer_hamiltonian, all);
                const double value = d->estimator->run_single(circuit, *d->cost_hamiltonian);
                const double v = std::isfinite(value) ? value : 1e12;
                ++d->nfev;
                if (v < d->best_val) d->best_val = v;
                if (d->nfev % 50 == 0) {
                    std::cout << "[MAQAOA] layer=" << d->p_current
                              << " eval=" << d->nfev
                              << " best=" << d->best_val
                              << std::endl;
                }
                return v;
            };

            std::cout << "[MAQAOA] layer=" << layer
                      << " starting, free_params=" << params_per_layer
                      << " total_layers=" << options.p
                      << " budget=" << options.max_iterations
                      << std::endl;

            LayerCBData cb{&estimator, &cost_hamiltonian, &mixer, this,
                           frozen_prefix, layer, 0,
                           std::numeric_limits<double>::infinity()};

            nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, params_per_layer);
            nlopt_set_min_objective(opt, layer_objective, &cb);
            nlopt_set_maxeval(opt, options.max_iterations);
            nlopt_set_xtol_rel(opt, options.convergence_threshold);

            // Set bounds to prevent parameters from blowing up
            std::vector<double> lb(params_per_layer, -kBound);
            std::vector<double> ub(params_per_layer, kBound);
            nlopt_set_lower_bounds(opt, lb.data());
            nlopt_set_upper_bounds(opt, ub.data());

            // Initial guess: the current layer portion of all_params
            std::vector<double> x0(all_params.begin() + offset, all_params.end());
            double min_val = std::numeric_limits<double>::infinity();
            nlopt_result nlopt_res = nlopt_optimize(opt, x0.data(), &min_val);
            nlopt_destroy(opt);

            std::cout << "[MAQAOA] layer=" << layer
                      << " done, nlopt_res=" << nlopt_res
                      << " nfev=" << cb.nfev
                      << " best=" << cb.best_val
                      << std::endl;

            if (nlopt_res < 0 || !std::isfinite(min_val)) {
                all_layers_converged = false;
            }

            // Update the free layer portion in all_params
            for (int i = 0; i < params_per_layer; ++i) {
                all_params[offset + i] = x0[i];
            }
        }

        result.optimal_params = all_params;
        auto circuit = build_circuit(cost_hamiltonian, mixer, all_params);
        result.optimal_value = estimator.run_single(circuit, cost_hamiltonian);
        result.converged = all_layers_converged && std::isfinite(result.optimal_value);

        if (!std::isfinite(result.optimal_value)) {
            result.optimal_value = 1e12;
        }

    } else {
        // Standard: optimise all parameters at once
        std::vector<double> params(n_params, 0.5);

        nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, n_params);
        MAQAOACallbackData cb_data{&estimator, &cost_hamiltonian, &mixer, this};
        nlopt_set_min_objective(opt, maqaoa_objective, &cb_data);
        nlopt_set_maxeval(opt, options.max_iterations);
        nlopt_set_xtol_rel(opt, options.convergence_threshold);

        // Set bounds to prevent parameters from blowing up
        std::vector<double> lb(n_params, -kBound);
        std::vector<double> ub(n_params, kBound);
        nlopt_set_lower_bounds(opt, lb.data());
        nlopt_set_upper_bounds(opt, ub.data());

        double min_val = std::numeric_limits<double>::infinity();
        nlopt_result nlopt_res = nlopt_optimize(opt, params.data(), &min_val);
        nlopt_destroy(opt);

        result.optimal_value = std::isfinite(min_val) ? min_val : 1e12;
        result.optimal_params = params;
        result.converged = (nlopt_res > 0) && std::isfinite(min_val);
    }

    // Sample
    auto circuit = build_circuit(cost_hamiltonian, mixer, result.optimal_params);
    result.counts = sampler.run_single(circuit);

    int max_count = 0;
    for (const auto& [bits, count] : result.counts) {
        if (count > max_count) {
            max_count = count;
            result.best_bitstring = bits;
        }
    }

    return result;
}

QuantumCircuit MAQAOA::build_circuit(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian,
    const std::vector<double>& params
) const {
    int nq = cost_hamiltonian.n_qubits();
    QuantumCircuit qc(nq);

    // Initial state: |+...+⟩
    for (int q = 0; q < nq; ++q) {
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
        // Orbit mode: one gamma per orbit group (shared across symmetry-equivalent terms).
        // Standard mode: one gamma per term.
        int n_cost_params = use_orbits
            ? count_cost_orbits(cost_hamiltonian, options.orbit_assignments)
            : cost_terms;

        // Read this layer's cost gammas
        std::vector<double> layer_gammas(n_cost_params);
        for (int i = 0; i < n_cost_params; ++i) {
            layer_gammas[i] = (param_idx < static_cast<int>(params.size())) ?
                              params[param_idx++] : 0.0;
        }

        for (int t = 0; t < cost_terms; ++t) {
            int gamma_idx = use_orbits ? term_orbit_map[t] : t;
            double gamma = layer_gammas[gamma_idx];

            const auto& term = cost_hamiltonian.terms[t];
            double angle = 2.0 * gamma * term.coeff.real;

            std::vector<int> active_qubits;
            for (int q = 0; q < nq; ++q) {
                if (term.pauli[q] != 'I') active_qubits.push_back(q);
            }
            if (active_qubits.empty()) continue;

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
        // Orbit mode: one beta per orbit, all qubits in orbit share it.
        // Standard mode: one beta per qubit.
        std::vector<double> layer_betas(n_mixer_orbits);
        for (int i = 0; i < n_mixer_orbits; ++i) {
            layer_betas[i] = (param_idx < static_cast<int>(params.size())) ?
                             params[param_idx++] : 0.0;
        }

        for (int q = 0; q < nq; ++q) {
            int beta_idx = use_orbits ? options.orbit_assignments[q] : q;
            qc.rx(2.0 * layer_betas[beta_idx], q);
        }
    }

    return qc;
}

// =============================================================================
// QPE
// =============================================================================

QuantumCircuit QPE::build_circuit(
    const QuantumCircuit& unitary,
    int num_eval_qubits
) {
    int total_qubits = num_eval_qubits + unitary.n_qubits;
    QuantumCircuit qc(total_qubits);

    // Hadamard on evaluation qubits
    for (int i = 0; i < num_eval_qubits; ++i) {
        qc.h(i);
    }

    // Controlled-U^(2^k) applications
    // Each controlled-U gate: control qubit k, target qubits num_eval_qubits..N-1
    // We use UNITARY gate with the full matrix repeated 2^k times, then wrap each
    // unitary with a controlled version via CU gate.
    // For circuits with a custom Instruction::GateType::UNITARY type:
    //   build the U^(2^k) matrix explicitly by repeated matrix multiplication.
    for (int k = 0; k < num_eval_qubits; ++k) {
        int power = 1 << k;

        // Get the U matrix from the unitary circuit (apply power times)
        // First, simulate the unitary on a fresh statevector to extract matrix columns
        int nu = unitary.n_qubits;
        size_t ud = 1ULL << nu;

        // Build U matrix
        std::vector<std::vector<std::complex<double>>> U_cols(ud,
            std::vector<std::complex<double>>(ud, 0.0));
        for (size_t col = 0; col < ud; ++col) {
            Statevector basis(nu);
            basis.initialize_basis(col);
            StatevectorSimulator sv_sim;
            sv_sim.simulate_circuit(basis, unitary);
            for (size_t row = 0; row < ud; ++row) {
                U_cols[row][col] = {basis.real_parts[row], basis.imag_parts[row]};
            }
        }

        // Compute U^power by repeated matrix multiply
        std::vector<std::vector<std::complex<double>>> Up = U_cols;
        for (int rep = 1; rep < power; ++rep) {
            std::vector<std::vector<std::complex<double>>> Unew(
                ud, std::vector<std::complex<double>>(ud, 0.0));
            for (size_t r = 0; r < ud; ++r)
                for (size_t c = 0; c < ud; ++c)
                    for (size_t m = 0; m < ud; ++m)
                        Unew[r][c] += Up[r][m] * U_cols[m][c];
            Up = Unew;
        }

        // Pack into flat Complex128 vector
        std::vector<Complex128> Upow_flat(ud * ud);
        for (size_t r = 0; r < ud; ++r)
            for (size_t c = 0; c < ud; ++c)
                Upow_flat[r * ud + c] = Complex128(Up[r][c].real(), Up[r][c].imag());

        // Build the controlled-U^pow instruction
        // The control qubit is k, target qubits are num_eval_qubits + 0..nu-1
        // We construct the 2^(1+nu) x 2^(1+nu) controlled unitary matrix
        size_t full_dim = 1ULL << (1 + nu);
        std::vector<Complex128> CU_matrix(full_dim * full_dim, Complex128(0.0, 0.0));
        // Block structure: |0><0| ⊗ I  +  |1><1| ⊗ U^pow
        // |0><0| block (control=0): identity on target
        for (size_t t = 0; t < ud; ++t)
            CU_matrix[t * full_dim + t] = Complex128(1.0, 0.0);
        // |1><1| block (control=1): apply U^pow on target
        for (size_t r = 0; r < ud; ++r)
            for (size_t c = 0; c < ud; ++c)
                CU_matrix[(ud + r) * full_dim + (ud + c)] = Upow_flat[r * ud + c];

        Instruction ctrl_u;
        ctrl_u.type = Instruction::GateType::UNITARY;
        ctrl_u.qubits = {k};
        for (int tq = 0; tq < nu; ++tq)
            ctrl_u.qubits.push_back(num_eval_qubits + tq);
        ctrl_u.matrix = CU_matrix;
        qc.instructions.push_back(ctrl_u);
    }

    // Inverse QFT on evaluation qubits
    for (int i = num_eval_qubits - 1; i >= 0; --i) {
        for (int j = num_eval_qubits - 1; j > i; --j) {
            double angle = -PI / (1 << (j - i));
            qc.cp(angle, j, i);
        }
        qc.h(i);
    }

    return qc;
}

double QPE::estimate_phase(
    const QuantumCircuit& unitary,
    int num_eval_qubits,
    int shots,
    uint64_t seed
) {
    auto circuit = build_circuit(unitary, num_eval_qubits);
    circuit.measure_all();

    StatevectorSimulator sim;
    auto result = sim.run(circuit, shots, seed);

    // Find most frequent measurement
    std::string best_bits;
    int max_count = 0;
    for (const auto& [bits, count] : result.counts) {
        if (count > max_count) {
            max_count = count;
            best_bits = bits;
        }
    }

    // Convert eval qubits to phase
    int measured = 0;
    for (int i = 0; i < num_eval_qubits; ++i) {
        if (best_bits[i] == '1') {
            measured |= (1 << (num_eval_qubits - 1 - i));
        }
    }

    return static_cast<double>(measured) / (1 << num_eval_qubits);
}

// =============================================================================
// Grover
// =============================================================================

QuantumCircuit Grover::build_circuit(
    const QuantumCircuit& oracle,
    int num_iterations
) {
    int nq = oracle.n_qubits;

    if (num_iterations < 0) {
        // Optimal number: pi/4 * sqrt(2^n)
        num_iterations = static_cast<int>(
            std::round(PI / 4.0 * std::sqrt(static_cast<double>(1 << nq)))
        );
        if (num_iterations < 1) num_iterations = 1;
    }

    QuantumCircuit qc(nq);

    // Initial superposition
    for (int q = 0; q < nq; ++q) {
        qc.h(q);
    }

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Oracle
        for (const auto& inst : oracle.instructions) {
            qc.instructions.push_back(inst);
        }

        // Diffusion operator: 2|s⟩⟨s| - I
        for (int q = 0; q < nq; ++q) qc.h(q);
        for (int q = 0; q < nq; ++q) qc.x(q);

        // Multi-controlled Z (on all qubits): 2|s><s| - I applied as H*MCX*H on last qubit
        if (nq >= 2) {
            qc.h(nq - 1);
            if (nq == 2) {
                qc.cx(0, 1);
            } else if (nq == 3) {
                qc.ccx(0, 1, 2);
            } else {
                // Ancilla-free multi-controlled X via recursive C^n-1X decomposition.
                // We use the relative-phase Toffoli (RCCX) ladder approach:
                // C^nX = RCCX(c0,c1,ancilla-free chain) ... This requires ancilla qubits.
                // Without ancilla, use the general decomposition:
                // C^nX = product of at most 2*(n-2) Toffoli gates using scratch qubits.
                //
                // For the diffusion operator, we only need an NCZ (no ancilla).
                // Use the identity: C^nZ = H on last qubit + C^nX + H on last qubit,
                // C^nX for n > 3 without ancilla:
                //   = product of (CX chain using linear CCNOT telescoping)
                // Standard n-qubit MCX without ancilla: O(n^2) CX decomposition
                // via Gray-code based phase kickback.
                //
                // Here we use a simple O(n^2) approach: repeated CCX telescoping
                // with temporary qubit state preservation.
                //
                // Note: This is correct but may have non-trivial depth for large n.
                // For the Grover diffusion, correctness > depth.
                //
                // C^n-1 CNOT implemented via ancilla telescoping:
                // Use qubits 0..n-2 as controls, qubit n-1 as target.
                // Auxiliary scratch in reverse order.

                // Build C^n X using the Lemma 7.2 construction (n-2 ancilla-free CCXs)
                // Actually we do Gray-code based phase kickback which is more complex.
                // Simplest correct no-ancilla approach for Grover's diffusion:
                // Use the multicontrolled-Z via phase-kickback trick.
                // C^n Z = (H on all) * C^n X * (H on all) — this is just what we're building.

                // Simple but correct: use n-qubit phase oracle as multi-Toffoli chain.
                // C^n X = CCX(0,1,n-1) when n=3; for n>3, recurse:
                // C^n X = CCX chain telescoped through intermediate qubits
                //
                // We'll use the standard decomposition:
                // For qubits [c0, c1, ..., c_{n-2}, target]:
                // Step 1: CCX c0, c1 → scratch qubit (but we have no scratch)
                // Alternative: use RZ-based multi-controlled approach
                //
                // For correctness without ancilla, use the U1-ladder formula:
                // C^n Z = prod of doubly-controlled phases
                // This is the Gray code diagonal:
                // C^n Z diagonal phase = (-1) for |1...1⟩ state
                // Implemented as: Pauli X on target, H, C^n X, H, Pauli X on target
                //
                // Since we need only correct MCZ for Grover, and the circuit is
                // simulated exactly, use the following iterative approach:
                // Use CCX chain on the controls, using the last qubit as scratch.
                // This is NOT ancilla-free in general, but works for Grover's diffusion
                // where target = nq-1 which is the last measurement qubit.

                // Recursive MCX without ancilla (depth 2n-3):
                // We recurse via C^(n-1) X and a single CX.
                // Base case n=3: CCX(0,1,2).
                // For n>3: decompose C^n-1 X using the qubit just before the target as temp.
                // temp is NOT reused as control → no state corruption.

                // Direct implementation: two-step telescoping
                // Step 1: C^{n-1} X with controls 0..n-3 and temp target = n-2
                // Step 2: CCX n-3, n-2, n-1
                // Step 3: undo Step 1

                // For the Grover diffusion specifically:
                // Use the RCCX (relative-phase Toffoli) ladder:
                // Controls: 0..n-2, Target: n-1
                // This costs (n-2) RCCX gates + 1 CNOT, total ~4(n-2) elementary gates.

                // RCCX ladder: accumulate controls into chain
                // temp[k] = AND(c_0, ..., c_k), implemented via RCCX using chain qubit
                // Since we don't have ancilla, use a CCX cascade into the target directly:

                // Most practical correct approach: decompose into smaller gates via:
                // Full MCX = CCX decomposition with no ancilla using CU1 phases (Selinger 2012)
                // or: use the direct C^n-1 Z phase gadget via Gray code

                // For now: implement as CCX telescoping using nq-1 as target (correct for n>3):
                // This is (n-3) intermediate CCX gates assuming qubit n-2 is scratch.
                // WARNING: This is only correct if qubit n-2 is NOT also a control!
                // But in Grover's diffusion, ALL qubits 0..n-2 are controls.

                // Correct ancilla-free MCX for Grover (n controls on 1 target):
                // Use repeated CX + phase decomposition.
                // The safe approach is to use up to n^2/2 CX gates via the formula:
                // MCX = product of Givens rotations.

                // Given simulation context (not real hardware), use UNITARY gate:
                size_t mcu = 1ULL << nq;
                std::vector<Complex128> mcx_mat(mcu * mcu, Complex128(0.0, 0.0));
                for (size_t idx = 0; idx < mcu; ++idx) {
                    if (idx == mcu - 2) mcx_mat[idx * mcu + (mcu - 1)] = Complex128(1.0, 0.0);
                    else if (idx == mcu - 1) mcx_mat[idx * mcu + (mcu - 2)] = Complex128(1.0, 0.0);
                    else mcx_mat[idx * mcu + idx] = Complex128(1.0, 0.0);
                }
                Instruction mcx_inst;
                mcx_inst.type = Instruction::GateType::UNITARY;
                for (int mq = 0; mq < nq; ++mq) mcx_inst.qubits.push_back(mq);
                mcx_inst.matrix = mcx_mat;
                qc.instructions.push_back(mcx_inst);
            }
            qc.h(nq - 1);
        }

        for (int q = 0; q < nq; ++q) qc.x(q);
        for (int q = 0; q < nq; ++q) qc.h(q);
    }

    return qc;
}

Grover::Result Grover::search(
    const QuantumCircuit& oracle,
    int num_iterations,
    int shots,
    uint64_t seed
) {
    auto circuit = build_circuit(oracle, num_iterations);
    circuit.measure_all();

    StatevectorSimulator sim;
    auto sim_result = sim.run(circuit, shots, seed);

    Result result;
    int max_count = 0;
    for (const auto& [bits, count] : sim_result.counts) {
        if (count > max_count) {
            max_count = count;
            result.solution = bits;
        }
    }
    result.probability = static_cast<double>(max_count) / shots;

    if (num_iterations < 0) {
        result.num_iterations = static_cast<int>(
            std::round(PI / 4.0 * std::sqrt(static_cast<double>(1 << oracle.n_qubits)))
        );
    } else {
        result.num_iterations = num_iterations;
    }

    return result;
}

} // namespace algorithms
} // namespace qpp
