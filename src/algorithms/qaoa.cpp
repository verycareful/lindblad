#include "lindblad/algorithms.hpp"
#include "lindblad/gates.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include <nlopt.h>

namespace lindblad {
namespace algorithms {

static constexpr double kBound = 2.0 * PI;

static double computational_basis_cost(
    const SparsePauliOp& cost_hamiltonian,
    const std::string& bitstring
) {
    const int nq = cost_hamiltonian.n_qubits();
    if (static_cast<int>(bitstring.size()) != nq) {
        return std::numeric_limits<double>::infinity();
    }

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

    std::mt19937_64 rng(options.seed != 0
        ? static_cast<uint64_t>(options.seed)
        : static_cast<uint64_t>(std::random_device{}()));
    std::uniform_real_distribution<double> perturb(-0.05, 0.05);

    std::vector<double> params(n_params);
    for (auto& p : params) p = perturb(rng);
    result.initial_params = params;

    // NLopt — optimizer selected by options.optimizer
    nlopt_algorithm algo = NLOPT_LN_COBYLA;
    if (options.optimizer == "NELDER_MEAD") algo = NLOPT_LN_NELDERMEAD;
    else if (options.optimizer == "POWELL")  algo = NLOPT_LN_BOBYQA;
    nlopt_opt opt = nlopt_create(algo, n_params);
    QAOACallbackData cb_data{&estimator, &cost_hamiltonian, &mixer, this};
    nlopt_set_min_objective(opt, qaoa_objective, &cb_data);
    nlopt_set_maxeval(opt, options.max_iterations);
    nlopt_set_xtol_rel(opt, options.convergence_threshold);
    std::vector<double> lb(n_params, -kBound);
    std::vector<double> ub(n_params, kBound);
    nlopt_set_lower_bounds(opt, lb.data());
    nlopt_set_upper_bounds(opt, ub.data());
    std::vector<double> initial_step(n_params, 0.3);
    nlopt_set_initial_step(opt, initial_step.data());

    double min_val;
    nlopt_result nlopt_res = nlopt_optimize(opt, params.data(), &min_val);
    nlopt_destroy(opt);

    result.optimal_value = min_val;
    result.optimal_params = params;
    result.converged = (nlopt_res > 0 && nlopt_res != NLOPT_MAXEVAL_REACHED);

    // Sample to get best bitstring
    sampler.options.seed = options.seed;
    auto circuit = build_circuit(cost_hamiltonian, mixer, params);
    result.counts = sampler.run_single(circuit);

    // Rank by computational-basis objective; break ties by sample count.
    double best_cost = std::numeric_limits<double>::infinity();
    int best_count = -1;
    for (const auto& [bits, count] : result.counts) {
        const double cost = computational_basis_cost(cost_hamiltonian, bits);
        if ((cost < best_cost) ||
            (cost == best_cost && count > best_count)) {
            best_cost = cost;
            best_count = count;
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

        // Mixer unitary: e^{-i * beta * B}, applied as the ordered product of
        // per-term rotations exp(-i*beta*c_k*P_k). Exact when the terms
        // commute (the default X mixer); a first-order Trotter step
        // otherwise. Multi-qubit terms use the same basis-change + CX-chain
        // recipe as the cost unitary: until R.1.12.2 they were FACTORISED
        // into independent per-qubit rotations (RX(x)RY instead of
        // exp(-i*beta*XY)), a wrong ansatz for entangling mixers such as the
        // constraint-preserving XY family.
        for (const auto& term : mixer_hamiltonian.terms) {
            double angle = 2.0 * beta * term.coeff.real;

            std::vector<int> active_qubits;
            for (int q = 0; q < nq; ++q) {
                if (term.pauli[q] != 'I') active_qubits.push_back(q);
            }
            if (active_qubits.empty()) continue;

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
