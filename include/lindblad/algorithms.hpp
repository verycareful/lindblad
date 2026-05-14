#pragma once

#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/backends/local_backend.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {
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

        // QSP-QAOA: per-qubit initial state preparation angles.
        // When non-empty, replaces the standard H|0> initialisation with Ry(theta[q])|0>
        // for each qubit q. Encodes a domain prior P(qubit q = 1) = sin²(theta[q]/2)
        // directly into the quantum initial state before any QAOA layers are applied.
        // Size must equal n_qubits. Empty = standard H initialisation (default behaviour).
        // Compute theta[q] = 2 * arcsin(sqrt(p_on[q])) from a domain prior p_on[q] ∈ [0,1].
        std::vector<double> initial_thetas;
    };

    struct Result {
        double optimal_value;
        std::vector<double> initial_params;
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

        // Progressive training: layerwise schedule without parameter freezing.
        // When true, all previously trained parameters remain free at each layer step.
        // Ignored when layerwise=false (joint optimisation already has no freezing).
        bool progressive = false;
        uint64_t seed = 0;

        // Orbit-QAOA: qubits in the same orbit share a single mixer parameter.
        // orbit_assignments[q] = orbit index (0-based). Empty = no symmetry reduction.
        // Orbits reduce the mixer parameter count from n_qubits to n_distinct_orbits.
        // Cost-term orbit sharing is automatic: terms with the same sorted tuple of
        // qubit orbits share a single gamma parameter.
        std::vector<int> orbit_assignments;

        // Gamma parameterisation convention.
        // false (default): qubit-indexed — N gammas per layer, gamma[i] drives
        //   all cost terms where qubit i is the lowest active qubit.
        //   Matches the Python/Qiskit baseline; 2N params per layer at N=20.
        // true: term-indexed — one gamma per Hamiltonian term per layer.
        //   More expressive but O(N^2) params (230/layer at N=20); use for ablation.
        bool term_indexed_gammas = false;

        // PI-MA-QAOA: per-orbit mixer weight vector (e.g. augmented cost per MW). (Might be published as a separate algorithm in future if it performs well.)
        // size must equal n_mixer_orbits (n_qubits in standard mode, n_orbits in
        // orbit mode). When non-empty, beta_i = beta_base * (mixer_weights[i] / w_max)
        // so expensive generators (large w_i) receive a large initial beta and
        // cheap generators (small w_i) receive a small initial beta.
        // To invert this (large beta for cheap generators), pass inverse weights:
        //   ipi_weights[i] = 1.0 / pi_weights[i]
        // This can be done entirely in the calling code without any lindblad changes.
        // Empty = standard random perturbation initialisation.
        std::vector<double> mixer_weights;

        // QSP-MA-QAOA: per-qubit initial state preparation angles.
        // When non-empty, replaces the standard H|0> initialisation with Ry(theta[q])|0>
        // for each qubit q. This encodes a prior probability P(qubit q = 1) = sin²(theta[q]/2)
        // directly into the quantum initial state before any QAOA layers are applied.
        // Size must equal n_qubits. Empty = standard H initialisation (default behaviour).
        // Compute theta[q] = 2 * arcsin(sqrt(p_on[q])) from a domain prior p_on[q] ∈ [0,1].
        std::vector<double> initial_thetas;

        double beta_base   = M_PI / 4.0;   // base angle scale for PI-MA-QAOA init
        double lambda_co2  = 0.0;          // carbon/other weighting factor (0 = pure economic) - Can be anything but normalised to a "cost"
    };

    struct Result {
        double optimal_value;
        std::vector<double> optimal_params;
        std::vector<double> initial_params;       // per-layer concatenated initial guess
        std::unordered_map<std::string, int> counts;
        std::string best_bitstring;
        int num_iterations = 0;                   // total evaluations across all layers
        bool converged;
        std::vector<double> per_layer_costs;      // best energy at end of each layer
        std::vector<int>    layer_nfev;           // evaluations per layer
        std::vector<double> wall_time_by_layer;   // wall seconds per layer
        double wall_time_seconds = 0.0;           // total wall time
    };

    Options options;
    Estimator estimator;
    Sampler sampler;

    MAQAOA() = default;

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
// Orbit utility — assign qubit orbit indices by power tier (Change 3)
// Generators within `tolerance` MW of each other share an orbit.
// Returns a vector of size n where result[i] = orbit index of generator i.
// =============================================================================

std::vector<int> orbits_by_power(
    const std::vector<double>& powers,
    double tolerance = 0.5
);

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

// =============================================================================
// Deutsch-Jozsa — determines constant vs balanced function in one query
// =============================================================================

class DeutschJozsa {
public:
    struct Result {
        enum Type { CONSTANT, BALANCED } type;
    };

    // oracle: circuit on (n+1) qubits — first n are query, last is ancilla
    static QuantumCircuit build_circuit(const QuantumCircuit& oracle, int n);
    static Result solve(const QuantumCircuit& oracle, int n,
                        int shots = 1, uint64_t seed = 0);
};

// =============================================================================
// Bernstein-Vazirani — recovers hidden string s from f(x)=s·x mod 2 in one query
// =============================================================================

class BernsteinVazirani {
public:
    struct Result {
        std::string secret;
    };

    static QuantumCircuit build_circuit(const QuantumCircuit& oracle, int n);
    static Result solve(const QuantumCircuit& oracle, int n,
                        int shots = 1, uint64_t seed = 0);
};

// =============================================================================
// RecursiveBernsteinVazirani — depth-d BV; d oracle calls vs classical d×n queries
//
// Each level has its own oracle encoding an independent n-bit secret.
// Quantum: 1 oracle call per level (single-shot BV); total = d oracle calls.
// Classical: n deterministic queries per level; total = d×n queries.
// The original BV paper uses a recursive oracle construction to prove a
// superpolynomial BQP vs BPP separation: QTM O(n), PTM Ω(n^{log n}).
// =============================================================================

class RecursiveBernsteinVazirani {
public:
    struct Result {
        std::vector<std::string> secrets;   // secrets[i] = secret recovered at depth i
        int depth;
        int total_oracle_calls;
    };

    // oracles: one per depth level, each a standard BV oracle on (n+1) qubits.
    // seed is incremented by 1 for each level to keep runs independent.
    static Result solve(const std::vector<QuantumCircuit>& oracles, int n,
                        int shots = 1, uint64_t seed = 0);
};

// =============================================================================
// ProbabilisticBernsteinVazirani — multi-key probabilistic oracle
//   (Shukla & Vedula 2023, arXiv:2301.10014)
//
// The oracle probabilistically encodes one of K secret keys per invocation.
// Quantum: recovers ANY key with certainty in 1 shot (phase kickback is exact
//          regardless of which key the oracle selected that run).
//          Recovers ALL K keys in ~K·ln(K) shots (coupon-collector bound).
// Classical: cannot determine even a single bit of any key with certainty
//            in the general case (oracle is probabilistic, no deterministic handle).
// =============================================================================

class ProbabilisticBernsteinVazirani {
public:
    struct Result {
        std::vector<std::string> discovered_keys;         // unique keys found, sorted
        std::unordered_map<std::string, int> key_counts;  // frequency per key
        int shots_used;
    };

    // oracle_pool: K circuits, each encoding a distinct secret key (standard BV oracle).
    // weights: sampling probability for each oracle; uniform if empty.
    // Each shot draws one oracle at random, runs BV once, records the recovered key.
    static Result solve(const std::vector<QuantumCircuit>& oracle_pool, int n,
                        const std::vector<double>& weights = {},
                        int shots = 50, uint64_t seed = 0);
};

// =============================================================================
// DistributedBernsteinVazirani — recovers s = s0||s1||...||s_{t-1} where
//   party j holds n_j bits and a local BV oracle on (n_j + 1) qubits.
//
// Secret s is an n-bit string partitioned across t parties:
//   s = S_{n_0} || S_{n_1} || ... || S_{n_{t-1}},  Σ n_j = n
// Party j's oracle: f_j(m_j) = ⟨S_{n_j} · m_j⟩ mod 2  (n_j input qubits + 1 ancilla)
//
// Quantum: O(1) communication rounds (all local oracles applied in one shot).
// Classical: O(t) rounds (must query each party independently).
// Circuit depth: 2^max(n_j) + 3  vs  2^n + 3 for monolithic BV.
// No auxiliary qubits or EPR pairs required beyond the shared ancilla.
//
// Reference: Distributed Quantum Computing literature (ScienceDirect 2024).
// =============================================================================

class DistributedBernsteinVazirani {
public:
    // One entry per party. local_oracle acts on (n_bits + 1) qubits:
    //   qubits 0..n_bits-1 = party's query register
    //   qubit  n_bits       = shared ancilla (last qubit in the global circuit)
    struct Party {
        QuantumCircuit local_oracle;
        int n_bits;               // n_j — number of bits this party holds
    };

    struct Result {
        std::string full_secret;                  // complete n-bit recovered secret
        std::vector<std::string> party_secrets;   // per-party slice, length n_j each
        int num_parties;
        int total_bits;
        int quantum_rounds = 1;   // always 1 — quantum advantage
        int classical_rounds;     // equals num_parties — what classical would cost
    };

    // Build the combined circuit: H+X prep → local oracles (qubit-remapped) → H → measure.
    static QuantumCircuit build_circuit(const std::vector<Party>& parties);

    // Build, simulate, and decode the full secret.
    static Result solve(const std::vector<Party>& parties,
                        int shots = 1, uint64_t seed = 0);
};

// =============================================================================
// QFT — Quantum Fourier Transform family
//
// Standard QFT on n qubits:
//   for j = 0..n-1:  H(j);  for k = j+1..n-1:  CP(π/2^{k-j}, k, j)
//   then SWAP pairs to reverse bit order (if do_swaps = true)
//   Total: n(n-1)/2 CP gates + n H gates + ⌊n/2⌋ SWAPs = O(n²)
//
// Approximate QFT (AQFT, Kitaev/Coppersmith):
//   Omit CP(θ) with |θ| < π/2^m.  Gate count reduces to O(n·m).
//   m = 0  →  exact QFT  (all CP retained)
//   m = 1  →  keep only CP(π/2) = CS;  Clifford-simulable for all n
//   m = n-1→  equivalent to exact QFT
//
// Inverse QFT (IQFT):
//   Reverse gate order and negate all CP angles.
//   Used as the final stage of QPE, Shor's algorithm, and QSP.
//
// Clifford-simulability:
//   CP(π/2^k) is Clifford iff k = 0 (I), 1 (CS = controlled-S), or the gate
//   reduces to X/Y/Z.  For k ≥ 2, CP(π/2^k) involves T-equivalent rotations
//   and is non-Clifford.  Exact QFT therefore requires Statevector, DM, or MPS
//   for n ≥ 3 (n=1: just H; n=2: H + CS + SWAP — both Clifford).
//   AQFT(m=1) retains only CS gates and is Clifford-simulable for all n.
//
// Future work — Clifford+T simulation:
//   Extending CliffordSimulator via stabilizer-rank decomposition
//   (Bravyi & Gosset 2016; Bravyi, Browne et al. 2019) would allow
//   simulation of circuits with a small number of T-equivalent gates (t ≤ ~30)
//   in O(2^t · n²) time.  This would enable AQFT(m≥2) on the Clifford backend
//   but would NOT be practical for exact QFT (t = O(n²)).
//   Tracked as a separate architectural feature for a future minor release.
//
// Semi-classical (iterative) QFT — Griffiths & Niu 1996:
//   Processes qubits from n-1 down to 0.  For each qubit j, previously measured
//   outcomes (stored in classical registers) drive classical feedforward rotations
//   P(π/2^{k-j}) conditioned on c[k]==1, replacing the quantum CP gates.
//   Requires only single-qubit gates + mid-circuit MEASURE + feedforward.
//   Output: n classical bits holding the QFT Fourier coefficients.
//   Clifford-compatible for n ≤ 2 (angles reduce to S/Z/SDG gates).
//
// References:
//   Coppersmith (1994), Cleve & Watrous (2000), Barenco et al. (1996 AQFT),
//   Nielsen & Chuang §5.1, Griffiths & Niu 1996 (iterative QPE/QFT)
// =============================================================================

class QFT {
public:
    // Options for circuit construction.
    struct Options {
        // Append bit-reversal SWAP layer at the end of QFT/IQFT (standard convention).
        // Set false when composing QFT as a subroutine where the caller handles ordering.
        bool do_swaps;
        int approximation_degree;
        bool inverse;

        Options(bool swaps = true, int approx_deg = 0, bool inv = false)
            : do_swaps(swaps), approximation_degree(approx_deg), inverse(inv) {}
    };

    // Result returned by run().
    struct Result {
        backends::BackendResult backend_result;  // counts, timing, success flag
        int n_qubits;
        // true iff the built circuit contains only Clifford-group gates.
        // (exact QFT: true only for n ≤ 2; AQFT(m=1): always true)
        bool clifford_compatible;
    };

    // -------------------------------------------------------------------------
    // Circuit builders — pure circuit constructors, no simulator dependency.
    // These are the primary composable subroutine interface.
    // -------------------------------------------------------------------------

    // Build a QFT or IQFT subcircuit on n qubits.
    // Returns a QuantumCircuit with no classical bits (measurements not included).
    static QuantumCircuit build_circuit(int n, const Options& opts = Options{});

    // Convenience: exact IQFT subcircuit, optionally with/without bit-reversal SWAPs.
    // Equivalent to build_circuit(n, {do_swaps, 0, true}).
    static QuantumCircuit build_inverse_circuit(int n, bool do_swaps = true);

    // AQFT: exact QFT dropping all CP(θ) with |θ| < π/2^m.
    // Equivalent to build_circuit(n, {true, m, false}).
    static QuantumCircuit build_approximate_circuit(int n, int m);

    // Append the QFT subcircuit to an existing circuit (in-place composition).
    // The input circuit is copied; the QFT subcircuit is appended on all qubits.
    // Returns the composed circuit (no measurements added).
    static QuantumCircuit apply(const QuantumCircuit& qc, const Options& opts = Options{});

    // -------------------------------------------------------------------------
    // run() — terminal entry points: build circuit, execute, return result.
    // input_state: circuit that prepares the input |ψ⟩ from |0...0⟩.
    // shots = 0: state inspection (no measure_all appended — use with SV/DM).
    // shots > 0: measure_all appended, backend samples the distribution.
    // -------------------------------------------------------------------------

    // Run using an explicit LocalBackend (any SimType: STATEVECTOR, DENSITY_MATRIX,
    // MPS, AUTO).  CLIFFORD will work only for Clifford-compatible circuits (n ≤ 2
    // or AQFT m=1); the backend will error otherwise.
    static Result run(
        const QuantumCircuit& input_state,
        backends::LocalBackend& backend,
        const Options& opts = Options{},
        int shots = 0,
        uint64_t seed = 0
    );

    // Convenience overload: run with a default Statevector backend.
    static Result run(
        const QuantumCircuit& input_state,
        const Options& opts = Options{},
        int shots = 0,
        uint64_t seed = 0
    );

    // -------------------------------------------------------------------------
    // Semi-classical (iterative) QFT — Griffiths & Niu 1996.
    // Builds a circuit with n qubits AND n classical bits.
    // Uses feedforward: P rotations conditioned on prior measurement outcomes.
    // The circuit always includes final measurements (no shots=0 mode).
    // Output bitstring c[n-1]...c[0] holds the QFT Fourier coefficients.
    // -------------------------------------------------------------------------

    // Build the semi-classical forward QFT circuit (n qubits, n classical bits).
    // Processing order: qubit n-1 first (MSB), qubit 0 last (LSB).
    static QuantumCircuit build_iterative_circuit(int n);

    // Build the semi-classical inverse QFT circuit (n qubits, n classical bits).
    // Processing order: qubit 0 first, qubit n-1 last.
    static QuantumCircuit build_iterative_inverse_circuit(int n);

    // Run semi-classical QFT on input_state using the given backend.
    // shots: number of samples (must be > 0; feedforward requires per-shot execution).
    static Result run_iterative(
        const QuantumCircuit& input_state,
        backends::LocalBackend& backend,
        int shots = 1024,
        uint64_t seed = 0
    );

    // Convenience overload: run semi-classical QFT with a default Statevector backend.
    static Result run_iterative(
        const QuantumCircuit& input_state,
        int shots = 1024,
        uint64_t seed = 0
    );
};

// =============================================================================
// Simon's Algorithm — finds period s where f(x)=f(x XOR s), O(n) queries
// =============================================================================

class Simon {
public:
    struct Result {
        std::string period;
        std::vector<std::string> equations;
    };

    // oracle: circuit on 2n qubits (n query + n output)
    static QuantumCircuit build_circuit(const QuantumCircuit& oracle, int n);
    static Result solve(const QuantumCircuit& oracle, int n,
                        uint64_t seed = 0, int extra_samples = 2);

private:
    static std::string gaussian_eliminate(
        const std::vector<std::string>& equations, int n);
};

} // namespace algorithms
} // namespace lindblad
