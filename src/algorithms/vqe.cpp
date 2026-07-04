#include "lindblad/algorithms.hpp"
#include "lindblad/gates.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

// NLopt for optimisation
#include <nlopt.h>

namespace lindblad {
namespace algorithms {

// =============================================================================
// NLopt callback helper
// =============================================================================

struct VQECallbackData {
    Estimator* estimator;
    const QuantumCircuit* ansatz;
    const SparsePauliOp* hamiltonian;
    std::vector<double>* energy_history;
};

static double vqe_objective(unsigned n, const double* x, double* /*grad*/, void* data) {
    auto* cb = static_cast<VQECallbackData*>(data);
    std::vector<double> params(x, x + n);
    double energy = cb->estimator->run_single(*cb->ansatz, *cb->hamiltonian, params);
    cb->energy_history->push_back(energy);
    return energy;
}

// =============================================================================
// VQE
// =============================================================================

VQE::Result VQE::compute_minimum_eigenvalue(
    const SparsePauliOp& hamiltonian,
    const QuantumCircuit& ansatz,
    const std::vector<double>& initial_params
) {
    Result result;
    result.converged = false;

    int n_params = ansatz.num_parameters();
    if (n_params == 0 && !initial_params.empty()) {
        n_params = static_cast<int>(initial_params.size());
    }

    // Initial parameters
    std::vector<double> params = initial_params;
    if (params.empty()) {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-M_PI, M_PI);
        params.resize(n_params);
        for (auto& p : params) p = dist(rng);
    }

    // Set up NLopt optimiser
    nlopt_algorithm algo = NLOPT_LN_COBYLA;
    if (options.optimizer == "NELDER_MEAD") algo = NLOPT_LN_NELDERMEAD;
    else if (options.optimizer == "POWELL") algo = NLOPT_LN_BOBYQA;

    nlopt_opt opt = nlopt_create(algo, n_params);
    nlopt_set_maxeval(opt, options.max_iterations);
    nlopt_set_xtol_rel(opt, options.convergence_threshold);

    VQECallbackData cb_data;
    cb_data.estimator = &estimator;
    cb_data.ansatz = &ansatz;
    cb_data.hamiltonian = &hamiltonian;
    cb_data.energy_history = &result.energy_history;
    nlopt_set_min_objective(opt, vqe_objective, &cb_data);

    // NLopt failure codes (< 0) can return without writing min_val, which
    // previously surfaced an indeterminate stack value as Result::eigenvalue.
    // Initialise to NaN, recover the best objective actually evaluated from
    // the recorded history, and fail loudly if the objective never ran.
    // The integer status check comes FIRST: it is immune to -ffast-math,
    // under which std::isfinite may fold away (hence is_finite_strict).
    double min_val = std::numeric_limits<double>::quiet_NaN();
    nlopt_result nlopt_res = nlopt_optimize(opt, params.data(), &min_val);

    if (nlopt_res < 0 || !is_finite_strict(min_val)) {
        if (result.energy_history.empty()) {
            nlopt_destroy(opt);
            throw std::runtime_error(
                "VQE::compute_minimum_eigenvalue: optimiser returned no finite "
                "energy (nlopt code " +
                std::to_string(static_cast<int>(nlopt_res)) +
                ") and never evaluated the objective");
        }
        min_val = *std::min_element(result.energy_history.begin(),
                                    result.energy_history.end());
    }

    result.eigenvalue = min_val;
    result.optimal_parameters = params;
    result.num_iterations = static_cast<int>(result.energy_history.size());
    result.converged = (nlopt_res > 0);

    nlopt_destroy(opt);

    return result;
}

// =============================================================================
// Ansatz generators
// =============================================================================

QuantumCircuit VQE::efficient_su2(int n_qubits, int reps) {
    QuantumCircuit qc(n_qubits);
    int param_idx = 0;

    for (int r = 0; r <= reps; ++r) {
        // Rotation layer: RY + RZ on each qubit
        for (int q = 0; q < n_qubits; ++q) {
            qc.ry("theta_" + std::to_string(param_idx++), q);
            qc.rz("theta_" + std::to_string(param_idx++), q);
        }

        // Entanglement layer (skip on last rep)
        if (r < reps) {
            for (int q = 0; q < n_qubits - 1; ++q) {
                qc.cx(q, q + 1);
            }
        }
    }

    return qc;
}

QuantumCircuit VQE::real_amplitudes(int n_qubits, int reps) {
    QuantumCircuit qc(n_qubits);
    int param_idx = 0;

    for (int r = 0; r <= reps; ++r) {
        for (int q = 0; q < n_qubits; ++q) {
            qc.ry("theta_" + std::to_string(param_idx++), q);
        }

        if (r < reps) {
            for (int q = 0; q < n_qubits - 1; ++q) {
                qc.cx(q, q + 1);
            }
        }
    }

    return qc;
}

QuantumCircuit VQE::two_local(
    int n_qubits,
    const std::vector<std::string>& rotation_blocks,
    const std::vector<std::string>& entanglement_blocks,
    int reps,
    const std::string& entanglement
) {
    QuantumCircuit qc(n_qubits);
    int param_idx = 0;

    for (int r = 0; r <= reps; ++r) {
        // Rotation layer
        for (int q = 0; q < n_qubits; ++q) {
            for (const auto& gate : rotation_blocks) {
                std::string pname = "theta_" + std::to_string(param_idx++);
                if (gate == "ry") qc.ry(pname, q);
                else if (gate == "rz") qc.rz(pname, q);
                else if (gate == "rx") qc.rx(pname, q);
            }
        }

        // Entanglement layer
        if (r < reps) {
            if (entanglement == "linear") {
                for (int q = 0; q < n_qubits - 1; ++q) {
                    for (const auto& gate : entanglement_blocks) {
                        if (gate == "cx") qc.cx(q, q + 1);
                        else if (gate == "cz") qc.cz(q, q + 1);
                    }
                }
            } else if (entanglement == "full") {
                for (int q1 = 0; q1 < n_qubits; ++q1) {
                    for (int q2 = q1 + 1; q2 < n_qubits; ++q2) {
                        for (const auto& gate : entanglement_blocks) {
                            if (gate == "cx") qc.cx(q1, q2);
                            else if (gate == "cz") qc.cz(q1, q2);
                        }
                    }
                }
            } else if (entanglement == "circular") {
                for (int q = 0; q < n_qubits; ++q) {
                    int next = (q + 1) % n_qubits;
                    for (const auto& gate : entanglement_blocks) {
                        if (gate == "cx") qc.cx(q, next);
                        else if (gate == "cz") qc.cz(q, next);
                    }
                }
            }
        }
    }

    return qc;
}

} // namespace algorithms
} // namespace lindblad
