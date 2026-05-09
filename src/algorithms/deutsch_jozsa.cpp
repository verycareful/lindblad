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
    // Bitstring has length n (only query qubits 0..n-1 measured).
    // Constant oracle → all query qubits measure 0 → all-zero bitstring.
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

} // namespace algorithms
} // namespace lindblad
