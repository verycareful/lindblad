#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"

#include <string>

namespace lindblad {
namespace algorithms {

QuantumCircuit DeutschJozsa::build_circuit(const QuantumCircuit& oracle, int n) {
    QuantumCircuit qc(n + 1, n);
    qc.x(n);
    for (int i = 0; i <= n; ++i) qc.h(i);
    for (const auto& inst : oracle.instructions) qc.instructions.push_back(inst);
    for (int i = 0; i < n; ++i) qc.h(i);
    for (int i = 0; i < n; ++i) qc.measure(i, i);
    return qc;
}

DeutschJozsa::Result DeutschJozsa::solve(const QuantumCircuit& oracle, int n,
                                          int shots, uint64_t seed) {
    auto qc = build_circuit(oracle, n);
    StatevectorSimulator sim;
    auto res = sim.run(qc, shots, seed);
    // Per-shot execution measures only the n query qubits into the classical register,
    // so bitstrings have length n. Constant oracle → all-zero query bits.
    std::string query_zero(n, '0');
    bool constant = false;
    for (const auto& [bits, cnt] : res.counts) {
        if (bits == query_zero) {
            constant = true;
            break;
        }
    }
    return { constant ? Result::CONSTANT : Result::BALANCED };
}

// =============================================================================
// QuditDeutschJozsa
// =============================================================================

QuditDeutschJozsa::Result QuditDeutschJozsa::solve(
    int n, int d,
    const std::function<int(const std::vector<int>&)>& f,
    uint64_t seed,
    QuditBackend backend,
    const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument("QuditDeutschJozsa::solve: d must be >= 2");
    if (n < 1)
        throw std::invalid_argument("QuditDeutschJozsa::solve: n must be >= 1");

    const auto Fd  = qudit_gates::qft_matrix(d);
    const auto Fdi = qudit_gates::iqft_matrix(d);
    const auto Xm  = qudit_gates::shift_matrix(d, d - 1);

    auto oracle_fn = [&](const std::vector<int>& x) -> std::vector<int> {
        const int val = f(x);
        if (val < 0 || val >= d)
            throw std::invalid_argument(
                "QuditDeutschJozsa::solve: f returned value out of [0, d)");
        return {val};
    };

    // ── DENSITY_MATRIX path ──────────────────────────────────────────────────
    if (backend == QuditBackend::DENSITY_MATRIX) {
        QuditDensityMatrix dm(n + 1, d);
        dm.apply_1qudit(n, Xm);
        dm.apply_1qudit(n, Fd);
        for (int i = 0; i < n; ++i) dm.apply_1qudit(i, Fd);
        dm.apply_function_oracle(n, 1, oracle_fn);
        for (int i = 0; i < n; ++i) dm.apply_1qudit(i, Fdi);
        if (noise) dm.apply_noise(*noise);
        const auto outcome = dm.measure(seed);
        for (int i = 0; i < n; ++i)
            if (outcome[static_cast<size_t>(i)] != 0)
                return {Verdict::BALANCED, d, n};
        return {Verdict::CONSTANT, d, n};
    }

    // ── MPS path ─────────────────────────────────────────────────────────────
    if (backend == QuditBackend::MPS) {
        QuditMPS mps(n + 1, d);
        mps.apply_1qudit(n, Xm);
        mps.apply_1qudit(n, Fd);
        for (int i = 0; i < n; ++i) mps.apply_1qudit(i, Fd);
        // MPS oracle: f takes flat query index, returns flat output index (addend)
        mps.apply_function_oracle(n, 1,
            [&](int x_flat) -> int {
                auto digits = QuditStatevector::index_to_digits(
                    static_cast<size_t>(x_flat), d, n);
                return f(digits);
            });
        for (int i = 0; i < n; ++i) mps.apply_1qudit(i, Fdi);
        const auto outcome = mps.measure(seed);
        for (int i = 0; i < n; ++i)
            if (outcome[static_cast<size_t>(i)] != 0)
                return {Verdict::BALANCED, d, n};
        return {Verdict::CONSTANT, d, n};
    }

    // ── CLIFFORD path — falls back to statevector (general oracle may not be
    // Clifford-decomposable from a black-box function; TODO: structured oracle API)
    // ── STATEVECTOR path (default + CLIFFORD fallback) ───────────────────────
    (void)backend; // STATEVECTOR and CLIFFORD both use this path
    QuditStatevector sv(n + 1, d);
    sv.apply_1qudit(n, Xm);
    sv.apply_1qudit(n, Fd);
    for (int i = 0; i < n; ++i) sv.apply_1qudit(i, Fd);
    sv.apply_function_oracle(n, 1, oracle_fn);
    for (int i = 0; i < n; ++i) sv.apply_1qudit(i, Fdi);
    const auto outcome = sv.measure(seed);
    for (int i = 0; i < n; ++i)
        if (outcome[static_cast<size_t>(i)] != 0)
            return {Verdict::BALANCED, d, n};
    return {Verdict::CONSTANT, d, n};
}

} // namespace algorithms
} // namespace lindblad
