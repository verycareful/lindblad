// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/algorithms.hpp"
#include "lindblad/detail/optimizer.hpp"
#include "lindblad/gates.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lindblad {
namespace algorithms {

// =============================================================================
// VQE
// =============================================================================

VQE::Result VQE::compute_minimum_eigenvalue(
    const SparsePauliOp& hamiltonian,
    const QuantumCircuit& ansatz,
    const std::vector<double>& initial_params
) {
    static constexpr const char* kWhere = "VQE::compute_minimum_eigenvalue";

    Result result;
    result.converged = false;

    int n_params = ansatz.num_parameters();
    if (n_params == 0 && !initial_params.empty()) {
        n_params = static_cast<int>(initial_params.size());
    }

    // Initial parameters: the caller's, or a seeded draw over one full turn
    // per angle. The draw is bit-exact across compilers (see uniform_in).
    std::vector<double> params = initial_params;
    if (params.empty()) {
        auto rng = detail::seeded_rng(options.seed);
        params.resize(n_params);
        for (auto& p : params) p = detail::uniform_in(rng, -PI, PI);
    }

    detail::OptimizerSpec spec;
    spec.backend = detail::resolve_optimizer(options.optimizer, kWhere);
    spec.max_evaluations = options.max_iterations;
    spec.xtol_rel = options.convergence_threshold;
    spec.initial_step = options.initial_step;

    // Every evaluation lands in energy_history, so the history is the full
    // trajectory including the point the optimiser finally returns.
    auto objective = [&](std::span<const double> x) -> double {
        std::vector<double> point(x.begin(), x.end());
        const double energy = estimator.run_single(ansatz, hamiltonian, point);
        result.energy_history.push_back(energy);
        return energy;
    };

    const detail::OptimizerOutcome out = detail::minimize(spec, objective, params, kWhere);

    result.eigenvalue = out.f;
    result.optimal_parameters = out.x;
    result.num_iterations = out.evaluations;
    result.converged = out.converged;

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
