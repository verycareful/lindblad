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

#include <cmath>
#include <limits>
#include <stdexcept>

namespace lindblad {
namespace algorithms {

static constexpr double kBound = 2.0 * PI;    // box on every gamma and beta
static constexpr double kPerturb = 0.05;      // half-width of the initial draw

// PRECONDITION: bitstring.size() == cost_hamiltonian.n_qubits(). Callers filter
// mismatched keys out; this helper does not signal failure in-band. Returning
// +infinity for a size mismatch would put a sentinel into a value the ranking
// loop below compares as ordinary cost data.
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
// QAOA
// =============================================================================

QAOA::Result QAOA::optimize(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian_in
) {
    static constexpr const char* kWhere = "QAOA::optimize";

    Result result;

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

    // Small seeded perturbation about zero; bit-exact across compilers (see
    // uniform_in).
    auto rng = detail::seeded_rng(options.seed);
    std::vector<double> params(n_params);
    for (auto& p : params) p = detail::uniform_in(rng, -kPerturb, kPerturb);
    result.initial_params = params;

    detail::OptimizerSpec spec;
    spec.backend = detail::resolve_optimizer(options.optimizer, kWhere);
    spec.max_evaluations = options.max_iterations;
    spec.xtol_rel = options.convergence_threshold;
    spec.initial_step = options.initial_step;
    spec.lower.assign(n_params, -kBound);
    spec.upper.assign(n_params, kBound);

    auto objective = [&](std::span<const double> x) -> double {
        std::vector<double> point(x.begin(), x.end());
        auto circuit = build_circuit(cost_hamiltonian, mixer, point);
        return estimator.run_single(circuit, cost_hamiltonian);
    };

    const detail::OptimizerOutcome out = detail::minimize(spec, objective, params, kWhere);

    result.optimal_value = out.f;
    result.optimal_params = out.x;
    result.num_iterations = out.evaluations;
    result.converged = out.converged;
    params = out.x;

    // Sample to get best bitstring
    sampler.options.seed = options.seed;
    auto circuit = build_circuit(cost_hamiltonian, mixer, params);
    result.counts = sampler.run_single(circuit);

    // Rank by computational-basis objective; break ties by sample count.
    //
    // best_cost is guarded by an explicit have_best flag rather than seeded
    // with +infinity. A seed exists to lose its first comparison, and with no
    // eligible sample at all a seeded loop returns that seed as though it were
    // a result, leaving best_bitstring empty with no error reported. A bool
    // states "nothing selected yet" directly and carries no floating-point
    // meaning for the optimiser to reason about.
    double best_cost = 0.0;
    int best_count = -1;
    bool have_best = false;
    for (const auto& [bits, count] : result.counts) {
        if (static_cast<int>(bits.size()) != nq) continue;
        const double cost = computational_basis_cost(cost_hamiltonian, bits);
        if (!have_best || cost < best_cost ||
            (cost == best_cost && count > best_count)) {
            best_cost = cost;
            best_count = count;
            have_best = true;
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
        // recipe as the cost unitary, NOT factorised into independent
        // per-qubit rotations: RX(x)RY instead of exp(-i*beta*XY) is a wrong
        // ansatz for entangling mixers such as the constraint-preserving XY
        // family.
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
