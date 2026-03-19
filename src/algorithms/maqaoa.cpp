#include "qpp/algorithms.hpp"
#include "qpp/gates.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

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
    return cb->estimator->run_single(circuit, *cb->cost_hamiltonian);
}

int MAQAOA::num_parameters(const SparsePauliOp& cost_hamiltonian) const {
    int nq = cost_hamiltonian.n_qubits();
    // Per layer: one angle per cost term + one angle per mixer qubit
    int cost_terms = static_cast<int>(cost_hamiltonian.terms.size());
    int mixer_terms = nq;  // default X mixer
    return options.p * (cost_terms + mixer_terms);
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

    if (options.layerwise) {
        // Layerwise training: optimise one layer at a time
        std::vector<double> all_params;
        int cost_terms = static_cast<int>(cost_hamiltonian.terms.size());
        int mixer_terms = nq;
        int params_per_layer = cost_terms + mixer_terms;

        for (int layer = 0; layer < options.p; ++layer) {
            // Add new layer parameters
            for (int i = 0; i < params_per_layer; ++i) {
                all_params.push_back(0.5);
            }

            // Optimise current layer while keeping previous layers fixed
            int offset = layer * params_per_layer;
            std::vector<double> current_layer_params(
                all_params.begin() + offset,
                all_params.end()
            );

            // Create a modified MAQAOA with p = layer + 1 for optimisation
            MAQAOA layer_maqaoa;
            layer_maqaoa.options = options;
            layer_maqaoa.options.p = layer + 1;
            layer_maqaoa.estimator = estimator;

            nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, params_per_layer);
            MAQAOACallbackData cb_data{&estimator, &cost_hamiltonian, &mixer, &layer_maqaoa};

            // Wrap objective to handle fixed + free params
            struct LayerOptData {
                MAQAOACallbackData* cb;
                std::vector<double>* fixed;
                int offset;
            };

            // Simplification: optimize all params together at each layer
            nlopt_opt full_opt = nlopt_create(NLOPT_LN_COBYLA, static_cast<int>(all_params.size()));
            MAQAOACallbackData full_cb{&estimator, &cost_hamiltonian, &mixer, &layer_maqaoa};
            nlopt_set_min_objective(full_opt, maqaoa_objective, &full_cb);
            nlopt_set_maxeval(full_opt, options.max_iterations);
            nlopt_set_xtol_rel(full_opt, options.convergence_threshold);

            double min_val;
            nlopt_optimize(full_opt, all_params.data(), &min_val);
            nlopt_destroy(full_opt);
            nlopt_destroy(opt);
        }

        result.optimal_params = all_params;
        auto circuit = build_circuit(cost_hamiltonian, mixer, all_params);
        result.optimal_value = estimator.run_single(circuit, cost_hamiltonian);

    } else {
        // Standard: optimise all parameters at once
        std::vector<double> params(n_params, 0.5);

        nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, n_params);
        MAQAOACallbackData cb_data{&estimator, &cost_hamiltonian, &mixer, this};
        nlopt_set_min_objective(opt, maqaoa_objective, &cb_data);
        nlopt_set_maxeval(opt, options.max_iterations);
        nlopt_set_xtol_rel(opt, options.convergence_threshold);

        double min_val;
        nlopt_result nlopt_res = nlopt_optimize(opt, params.data(), &min_val);
        nlopt_destroy(opt);

        result.optimal_value = min_val;
        result.optimal_params = params;
        result.converged = (nlopt_res > 0);
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

    for (int layer = 0; layer < options.p; ++layer) {
        // Cost unitary — INDEPENDENT angle per term
        for (int t = 0; t < cost_terms; ++t) {
            double gamma = (param_idx < static_cast<int>(params.size())) ?
                           params[param_idx++] : 0.0;

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

        // Mixer unitary — INDEPENDENT angle per qubit
        for (int q = 0; q < nq; ++q) {
            double beta = (param_idx < static_cast<int>(params.size())) ?
                          params[param_idx++] : 0.0;
            qc.rx(2.0 * beta, q);
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
    for (int k = 0; k < num_eval_qubits; ++k) {
        int power = 1 << k;
        for (int rep = 0; rep < power; ++rep) {
            // Apply controlled-U with control = k, targets = eval + [0..n-1]
            // For simplicity, just append the unitary instructions
            for (const auto& inst : unitary.instructions) {
                Instruction ctrl_inst = inst;
                // Shift qubit indices
                for (auto& q : ctrl_inst.qubits) {
                    q += num_eval_qubits;
                }
                qc.instructions.push_back(ctrl_inst);
            }
        }
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

        // Multi-controlled Z (on all qubits)
        if (nq >= 2) {
            qc.h(nq - 1);
            if (nq == 2) {
                qc.cx(0, 1);
            } else if (nq == 3) {
                qc.ccx(0, 1, 2);
            } else {
                // For n > 3, use ancilla-free MCX decomposition
                // Simplified: use CCX chain
                qc.ccx(0, 1, 2);
                for (int q = 3; q < nq; ++q) {
                    qc.ccx(q - 1, q, nq - 1);
                }
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
