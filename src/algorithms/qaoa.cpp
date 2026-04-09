#include "qpp/algorithms.hpp"
#include "qpp/gates.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#include <nlopt.h>

namespace qpp {
namespace algorithms {

// =============================================================================
// QAOA NLopt callback
// =============================================================================

struct QAOACallbackData {
    Estimator* estimator;
    const SparsePauliOp* cost_hamiltonian;
    const SparsePauliOp* mixer_hamiltonian;
    const QAOA* qaoa;
};

static double qaoa_objective(unsigned n, const double* x, double* /*grad*/, void* data) {
    auto* cb = static_cast<QAOACallbackData*>(data);
    std::vector<double> params(x, x + n);
    auto circuit = cb->qaoa->build_circuit(*cb->cost_hamiltonian, *cb->mixer_hamiltonian, params);
    return cb->estimator->run_single(circuit, *cb->cost_hamiltonian);
}

// =============================================================================
// QAOA
// =============================================================================

QAOA::Result QAOA::optimize(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian_in
) {
    Result result;
    result.converged = false;

    int nq = cost_hamiltonian.n_qubits();

    // Default mixer: sum of X_i
    SparsePauliOp mixer = mixer_hamiltonian_in;
    if (mixer.terms.empty()) {
        for (int q = 0; q < nq; ++q) {
            std::string pauli(nq, 'I');
            pauli[q] = 'X';
            mixer.terms.push_back({pauli, Complex128(1.0, 0.0)});
        }
    }

    int n_params = 2 * options.p;  // gamma_i, beta_i for each layer
    std::vector<double> params(n_params, 0.5);

    // NLopt
    nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, n_params);
    QAOACallbackData cb_data{&estimator, &cost_hamiltonian, &mixer, this};
    nlopt_set_min_objective(opt, qaoa_objective, &cb_data);
    nlopt_set_maxeval(opt, options.max_iterations);
    nlopt_set_xtol_rel(opt, options.convergence_threshold);

    double min_val;
    nlopt_result nlopt_res = nlopt_optimize(opt, params.data(), &min_val);
    nlopt_destroy(opt);

    result.optimal_value = min_val;
    result.optimal_params = params;
    result.converged = (nlopt_res > 0);

    // Sample to get best bitstring
    auto circuit = build_circuit(cost_hamiltonian, mixer, params);
    result.counts = sampler.run_single(circuit);

    // Find most frequent bitstring
    int max_count = 0;
    for (const auto& [bits, count] : result.counts) {
        if (count > max_count) {
            max_count = count;
            result.best_bitstring = bits;
        }
    }

    return result;
}

QuantumCircuit QAOA::build_circuit(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian,
    const std::vector<double>& params
) const {
    int nq = cost_hamiltonian.n_qubits();
    QuantumCircuit qc(nq);

    // Initial state: |+...+⟩ or QSP Ry initialisation
    if (!options.initial_thetas.empty() &&
        static_cast<int>(options.initial_thetas.size()) == nq) {
        for (int q = 0; q < nq; ++q)
            qc.ry(options.initial_thetas[q], q);
    } else {
        for (int q = 0; q < nq; ++q)
            qc.h(q);
    }

    for (int layer = 0; layer < options.p; ++layer) {
        double gamma = params[2 * layer];
        double beta = params[2 * layer + 1];

        // Cost unitary: e^{-i * gamma * C}
        // Decompose into Pauli rotations
        for (const auto& term : cost_hamiltonian.terms) {
            double angle = 2.0 * gamma * term.coeff.real;

            // Count non-I Paulis
            std::vector<int> active_qubits;
            for (int q = 0; q < nq; ++q) {
                if (term.pauli[q] != 'I') active_qubits.push_back(q);
            }

            if (active_qubits.empty()) continue;

            if (active_qubits.size() == 1 && term.pauli[active_qubits[0]] == 'Z') {
                // Single Z rotation
                qc.rz(angle, active_qubits[0]);
            } else if (active_qubits.size() == 2 &&
                       term.pauli[active_qubits[0]] == 'Z' &&
                       term.pauli[active_qubits[1]] == 'Z') {
                // ZZ interaction: CNOT - RZ - CNOT
                qc.cx(active_qubits[0], active_qubits[1]);
                qc.rz(angle, active_qubits[1]);
                qc.cx(active_qubits[0], active_qubits[1]);
            } else {
                // General Pauli rotation: basis change + CX chain + RZ + reverse
                // Change basis
                for (int q : active_qubits) {
                    if (term.pauli[q] == 'X') qc.h(q);
                    else if (term.pauli[q] == 'Y') { qc.sdg(q); qc.h(q); }
                }

                // CX chain
                for (size_t i = 0; i + 1 < active_qubits.size(); ++i) {
                    qc.cx(active_qubits[i], active_qubits[i + 1]);
                }

                // RZ on last qubit
                qc.rz(angle, active_qubits.back());

                // Reverse CX chain
                for (int i = static_cast<int>(active_qubits.size()) - 2; i >= 0; --i) {
                    qc.cx(active_qubits[i], active_qubits[i + 1]);
                }

                // Undo basis change
                for (int q : active_qubits) {
                    if (term.pauli[q] == 'X') qc.h(q);
                    else if (term.pauli[q] == 'Y') { qc.h(q); qc.s(q); }
                }
            }
        }

        // Mixer unitary: e^{-i * beta * B}
        for (const auto& term : mixer_hamiltonian.terms) {
            double angle = 2.0 * beta * term.coeff.real;

            for (int q = 0; q < nq; ++q) {
                if (term.pauli[q] == 'X') qc.rx(angle, q);
                else if (term.pauli[q] == 'Y') qc.ry(angle, q);
                else if (term.pauli[q] == 'Z') qc.rz(angle, q);
            }
        }
    }

    return qc;
}

} // namespace algorithms
} // namespace qpp
