#include "lindblad/algorithms.hpp"
#include "lindblad/backends/local_backend.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace lindblad {
namespace algorithms {

// -----------------------------------------------------------------------------
// Clifford-compatibility check:
//   The circuit is Clifford iff it contains no CP(θ) with |θ| mod 2π ≠ 0, π/2, π, 3π/2.
//   In practice for QFT: Clifford iff n ≤ 2 (exact) or approximation_degree == 1.
// -----------------------------------------------------------------------------
static bool is_clifford_compatible(int n, const QFT::Options& opts) {
    if (opts.approximation_degree == 1) return true;  // only CS (π/2) retained
    if (n <= 2) return true;                            // n=1: H only; n=2: H+CS+SWAP
    return false;
}

// -----------------------------------------------------------------------------
// build_circuit — core circuit builder.
//
// Implements the QFT in the project-wide LSB-at-qubit-0 (little-endian)
// convention (see CLAUDE.md "Project Conventions"):
//   |x⟩ at amp[x] → (1/√N) Σ_y exp(2πi xy/N) |y⟩ at amp[y]
//
// Implementation detail. The textbook H+CP gate sequence (qc.h(j),
// qc.cp(π/2^(k-j), k, j) for k>j) implements a QFT that takes input in
// qubit-0=MSB convention and produces output in qubit-0=LSB convention. To
// expose a uniformly little-endian interface, do_swaps=true wraps the H+CP
// sequence with bit-reversal SWAPs that translate at the input side of the
// forward QFT (and the output side of the inverse QFT):
//
//   Forward QFT-with-swaps: SWAPs → H+CP        (LSB input → MSB internal → LSB output)
//   Inverse QFT-with-swaps: reverse-H+CP → SWAPs (LSB input → MSB internal → LSB output)
//
// do_swaps=false exposes the raw H+CP sequence and is provided for users who
// need to compose with a pre-bit-reversed register. Algorithms that prepare
// QPE-style Fourier states (controlled-U^(2^k) on qubit k, which naturally
// yields amp[K] = exp(2πi K φ)/√N with K in LSB convention) MUST use
// do_swaps=true so the IQFT operates in matching LSB convention.
// -----------------------------------------------------------------------------
QuantumCircuit QFT::build_circuit(int n, const Options& opts) {
    if (n <= 0)
        throw std::invalid_argument("QFT::build_circuit: n must be >= 1");

    QuantumCircuit qc(n);

    const double pi = PI;

    // Threshold angle below which CP gates are dropped (AQFT approximation).
    // approximation_degree = 0  → threshold = 0 (keep all).
    // approximation_degree = m  → threshold = π / 2^m.
    const double threshold = (opts.approximation_degree > 0)
        ? pi / static_cast<double>(1 << opts.approximation_degree)
        : 0.0;

    if (!opts.inverse) {
        // ---- Forward QFT ----
        // LSB convention: pre-swap converts qubit-0=LSB input → qubit-0=MSB
        // internal representation expected by the textbook H+CP sequence.
        if (opts.do_swaps) {
            for (int i = 0; i < n / 2; ++i)
                qc.swap(i, n - 1 - i);
        }
        for (int j = 0; j < n; ++j) {
            qc.h(j);
            for (int k = j + 1; k < n; ++k) {
                double angle = pi / static_cast<double>(1 << (k - j));
                if (angle >= threshold - 1e-12)  // small epsilon for float safety
                    qc.cp(angle, k, j);
            }
        }
    } else {
        // ---- Inverse QFT ----
        // Reverse-H+CP sequence first, then post-swap converts the
        // qubit-0=MSB internal output → qubit-0=LSB output, maintaining the
        // LSB convention end-to-end.
        for (int j = n - 1; j >= 0; --j) {
            for (int k = n - 1; k > j; --k) {
                double angle = -pi / static_cast<double>(1 << (k - j));
                if (std::abs(angle) >= threshold - 1e-12)
                    qc.cp(angle, k, j);
            }
            qc.h(j);
        }
        if (opts.do_swaps) {
            for (int i = 0; i < n / 2; ++i)
                qc.swap(i, n - 1 - i);
        }
    }

    return qc;
}

// -----------------------------------------------------------------------------
// build_inverse_circuit — convenience wrapper for exact IQFT.
// -----------------------------------------------------------------------------
QuantumCircuit QFT::build_inverse_circuit(int n, bool do_swaps) {
    return build_circuit(n, {do_swaps, 0, true});
}

// -----------------------------------------------------------------------------
// build_approximate_circuit — AQFT with Kitaev approximation degree m.
// -----------------------------------------------------------------------------
QuantumCircuit QFT::build_approximate_circuit(int n, int m) {
    if (m < 0)
        throw std::invalid_argument("QFT::build_approximate_circuit: m must be >= 0");
    return build_circuit(n, {true, m, false});
}

// -----------------------------------------------------------------------------
// apply — append QFT subcircuit to an existing circuit.
// The input circuit's n_qubits determines the register size.
// No measurements are added; the composed circuit is returned.
// -----------------------------------------------------------------------------
QuantumCircuit QFT::apply(const QuantumCircuit& qc, const Options& opts) {
    QuantumCircuit result = qc;  // copy
    auto subcircuit = build_circuit(qc.n_qubits, opts);
    for (const auto& inst : subcircuit.instructions)
        result.instructions.push_back(inst);
    return result;
}

// -----------------------------------------------------------------------------
// run() — build, optionally measure, execute, return Result.
// -----------------------------------------------------------------------------
QFT::Result QFT::run(
    const QuantumCircuit& input_state,
    backends::LocalBackend& backend,
    const Options& opts,
    int shots,
    uint64_t seed
) {
    int n = input_state.n_qubits;

    // Compose: input state → QFT subcircuit → (optional) measure_all.
    QuantumCircuit circuit = apply(input_state, opts);
    if (shots > 0)
        circuit.measure_all();

    backends::BackendResult br = backend.run(circuit, shots > 0 ? shots : 1, seed);

    return Result{
        br,
        n,
        is_clifford_compatible(n, opts)
    };
}

// -----------------------------------------------------------------------------
// run() — convenience overload with default Statevector backend.
// -----------------------------------------------------------------------------
QFT::Result QFT::run(
    const QuantumCircuit& input_state,
    const Options& opts,
    int shots,
    uint64_t seed
) {
    backends::LocalBackend::Config cfg;
    cfg.simulator = backends::LocalBackend::SimType::STATEVECTOR;
    backends::LocalBackend sv_backend(cfg);
    return run(input_state, sv_backend, opts, shots, seed);
}

// -----------------------------------------------------------------------------
// build_iterative_circuit — semi-classical (Griffiths-Niu) forward QFT.
//
// Processes qubits from j = n-1 (MSB of the QFT output) down to j = 0 (LSB).
// For each qubit j:
//   1. For each already-measured qubit k > j:
//        IF c[k] == 1: P(π / 2^{k-j}) on qubit j   (classical feedforward)
//   2. H(j)
//   3. MEASURE(j, j) → c[j]
//
// Output: classical bits c[n-1..0] hold the QFT Fourier coefficients,
// with c[n-1] = MSB.  No bit-reversal SWAPs needed (qubit order is already
// the natural Fourier-coefficient order by construction).
// -----------------------------------------------------------------------------
QuantumCircuit QFT::build_iterative_circuit(int n) {
    if (n <= 0)
        throw std::invalid_argument("QFT::build_iterative_circuit: n must be >= 1");

    QuantumCircuit qc(n, n);
    const double pi = PI;

    for (int j = n - 1; j >= 0; --j) {
        // Phase corrections from previously measured qubits (k > j, already done).
        for (int k = n - 1; k > j; --k) {
            double angle = pi / static_cast<double>(1 << (k - j));
            qc.p_if(angle, j, k, 1);
        }
        qc.h(j);
        qc.measure(j, j);
    }
    return qc;
}

// -----------------------------------------------------------------------------
// build_iterative_inverse_circuit — semi-classical inverse QFT.
//
// Processes qubits from j = 0 up to j = n-1.
// For each qubit j:
//   1. For each already-measured qubit k < j:
//        IF c[k] == 1: P(-π / 2^{j-k}) on qubit j   (classical feedforward)
//   2. H(j)
//   3. MEASURE(j, j) → c[j]
// -----------------------------------------------------------------------------
QuantumCircuit QFT::build_iterative_inverse_circuit(int n) {
    if (n <= 0)
        throw std::invalid_argument("QFT::build_iterative_inverse_circuit: n must be >= 1");

    QuantumCircuit qc(n, n);
    const double pi = PI;

    for (int j = 0; j < n; ++j) {
        // Phase corrections from previously measured qubits (k < j, already done).
        for (int k = 0; k < j; ++k) {
            double angle = -pi / static_cast<double>(1 << (j - k));
            qc.p_if(angle, j, k, 1);
        }
        qc.h(j);
        qc.measure(j, j);
    }
    return qc;
}

// -----------------------------------------------------------------------------
// run_iterative() — compose input_state + iterative QFT circuit, execute.
//
// The composed circuit has n qubits and n classical bits.
// Feedforward requires per-shot execution; shots must be > 0.
// All four simulators (SV, DM, Clifford for n≤2, MPS) support feedforward.
// -----------------------------------------------------------------------------
QFT::Result QFT::run_iterative(
    const QuantumCircuit& input_state,
    backends::LocalBackend& backend,
    int shots,
    uint64_t seed
) {
    const int n = input_state.n_qubits;
    if (shots <= 0)
        throw std::invalid_argument("QFT::run_iterative: shots must be > 0 (feedforward requires per-shot execution)");

    // Build composed circuit: n qubits, n classical bits.
    // Copy input state's gate instructions, then append iterative QFT.
    QuantumCircuit composed(n, n);
    for (const auto& inst : input_state.instructions)
        composed.instructions.push_back(inst);

    auto iter_circuit = build_iterative_circuit(n);
    for (const auto& inst : iter_circuit.instructions)
        composed.instructions.push_back(inst);

    backends::BackendResult br = backend.run(composed, shots, seed);

    return Result{
        br,
        n,
        false  // iterative QFT uses P(π/2^k) for k≥2 — not Clifford in general
    };
}

// -----------------------------------------------------------------------------
// run_iterative() — convenience overload with default Statevector backend.
// -----------------------------------------------------------------------------
QFT::Result QFT::run_iterative(
    const QuantumCircuit& input_state,
    int shots,
    uint64_t seed
) {
    backends::LocalBackend::Config cfg;
    cfg.simulator = backends::LocalBackend::SimType::STATEVECTOR;
    backends::LocalBackend sv_backend(cfg);
    return run_iterative(input_state, sv_backend, shots, seed);
}

} // namespace algorithms
} // namespace lindblad
