#pragma once

#include "qpp/circuit.hpp"
#include "qpp/operators.hpp"
#include "qpp/primitives.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace qpp {
namespace algorithms {

// =============================================================================
// VQE — Variational Quantum Eigensolver
// =============================================================================

class VQE {
public:
    struct Options {
        int max_iterations = 100;
        double convergence_threshold = 1e-6;
        std::string optimizer = "COBYLA";  // COBYLA, NELDER_MEAD, POWELL
        uint64_t seed = 0;
    };

    struct Result {
        double eigenvalue;
        std::vector<double> optimal_parameters;
        int num_iterations;
        std::vector<double> energy_history;
        bool converged;
    };

    Options options;
    Estimator estimator;

    VQE() = default;
    VQE(const Options& opts, const Estimator& est)
        : options(opts), estimator(est) {}

    Result compute_minimum_eigenvalue(
        const SparsePauliOp& hamiltonian,
        const QuantumCircuit& ansatz,
        const std::vector<double>& initial_params = {}
    );

    // Ansatz generation
    static QuantumCircuit efficient_su2(int n_qubits, int reps = 3);
    static QuantumCircuit real_amplitudes(int n_qubits, int reps = 3);
    static QuantumCircuit two_local(
        int n_qubits,
        const std::vector<std::string>& rotation_blocks = {"ry", "rz"},
        const std::vector<std::string>& entanglement_blocks = {"cx"},
        int reps = 3,
        const std::string& entanglement = "full"
    );
};

// =============================================================================
// QAOA — Quantum Approximate Optimisation Algorithm
// =============================================================================

class QAOA {
public:
    struct Options {
        int p = 1;                     // number of QAOA layers
        int max_iterations = 100;
        double convergence_threshold = 1e-6;
        std::string optimizer = "COBYLA";
        uint64_t seed = 0;
    };

    struct Result {
        double optimal_value;
        std::vector<double> optimal_params;  // [gamma_1, beta_1, ..., gamma_p, beta_p]
        std::unordered_map<std::string, int> counts;
        std::string best_bitstring;
        int num_iterations;
        bool converged;
    };

    Options options;
    Estimator estimator;
    Sampler sampler;

    QAOA() = default;
    QAOA(const Options& opts, const Estimator& est, const Sampler& sam)
        : options(opts), estimator(est), sampler(sam) {}

    Result optimize(
        const SparsePauliOp& cost_hamiltonian,
        const SparsePauliOp& mixer_hamiltonian = {}
    );

    QuantumCircuit build_circuit(
        const SparsePauliOp& cost_hamiltonian,
        const SparsePauliOp& mixer_hamiltonian,
        const std::vector<double>& params
    ) const;
};

// =============================================================================
// MA-QAOA — Multi-Angle QAOA (independent angle per gate)
// =============================================================================

class MAQAOA {
public:
    struct Options {
        int p = 1;
        int max_iterations = 200;
        double convergence_threshold = 1e-6;
        std::string optimizer = "COBYLA";
        bool layerwise = false;        // iteratively optimise layer by layer
        uint64_t seed = 0;

        // Orbit-QAOA: qubits in the same orbit share a single mixer parameter.
        // orbit_assignments[q] = orbit index (0-based). Empty = no symmetry reduction.
        // Orbits reduce the mixer parameter count from n_qubits to n_distinct_orbits.
        // Cost-term orbit sharing is automatic: terms with the same sorted tuple of
        // qubit orbits share a single gamma parameter.
        std::vector<int> orbit_assignments;
    };

    struct Result {
        double optimal_value;
        std::vector<double> optimal_params;
        std::unordered_map<std::string, int> counts;
        std::string best_bitstring;
        int num_iterations;
        bool converged;
    };

    Options options;
    Estimator estimator;
    Sampler sampler;

    MAQAOA() = default;
    MAQAOA(const Options& opts, const Estimator& est, const Sampler& sam)
        : options(opts), estimator(est), sampler(sam) {}

    Result optimize(
        const SparsePauliOp& cost_hamiltonian,
        const SparsePauliOp& mixer_hamiltonian = {}
    );

    QuantumCircuit build_circuit(
        const SparsePauliOp& cost_hamiltonian,
        const SparsePauliOp& mixer_hamiltonian,
        const std::vector<double>& params
    ) const;

    int num_parameters(const SparsePauliOp& cost_hamiltonian) const;
};

// =============================================================================
// QPE — Quantum Phase Estimation
// =============================================================================

class QPE {
public:
    static QuantumCircuit build_circuit(
        const QuantumCircuit& unitary,
        int num_eval_qubits
    );

    static double estimate_phase(
        const QuantumCircuit& unitary,
        int num_eval_qubits,
        int shots = 1024,
        uint64_t seed = 0
    );
};

// =============================================================================
// Grover — Grover's search
// =============================================================================

class Grover {
public:
    static QuantumCircuit build_circuit(
        const QuantumCircuit& oracle,
        int num_iterations = -1  // -1 = auto (pi/4 * sqrt(N))
    );

    struct Result {
        std::string solution;
        int num_iterations;
        double probability;
    };

    static Result search(
        const QuantumCircuit& oracle,
        int num_iterations = -1,
        int shots = 1024,
        uint64_t seed = 0
    );
};

} // namespace algorithms
} // namespace qpp
