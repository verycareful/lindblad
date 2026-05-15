#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

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
    uint64_t seed)
{
    if (d < 2)
        throw std::invalid_argument(
            "QuditDeutschJozsa::solve: d must be >= 2");
    if (n < 1)
        throw std::invalid_argument(
            "QuditDeutschJozsa::solve: n must be >= 1");

    // Qudit layout: 0..n-1 = query register, n = ancilla
    QuditStatevector sv(n + 1, d);

    const auto Fd  = qudit_gates::qft_matrix(d);
    const auto Fdi = qudit_gates::iqft_matrix(d);
    const auto Xm  = qudit_gates::shift_matrix(d, d - 1);  // X^{d-1}: |0⟩→|d-1⟩

    // Steps 1–2: ancilla → |d-1⟩ then F_d → |−⟩_d
    sv.apply_1qudit(n, Xm);
    sv.apply_1qudit(n, Fd);

    // Step 3: uniform superposition on query register
    for (int i = 0; i < n; ++i) sv.apply_1qudit(i, Fd);

    // Step 4: oracle U_f — phase kickback via |x⟩|−⟩_d → ω^{f(x)}|x⟩|−⟩_d
    sv.apply_function_oracle(n, 1,
        [&](const std::vector<int>& x) -> std::vector<int> {
            const int val = f(x);
            if (val < 0 || val >= d)
                throw std::invalid_argument(
                    "QuditDeutschJozsa::solve: f returned value out of [0, d)");
            return {val};
        });

    // Step 5: inverse QFT on query register
    for (int i = 0; i < n; ++i) sv.apply_1qudit(i, Fdi);

    // Step 6: measure — all-zero query ↔ constant; any nonzero ↔ balanced
    const auto outcome = sv.measure(seed);
    for (int i = 0; i < n; ++i)
        if (outcome[static_cast<size_t>(i)] != 0)
            return {Verdict::BALANCED, d, n};
    return {Verdict::CONSTANT, d, n};
}

} // namespace algorithms
} // namespace lindblad
