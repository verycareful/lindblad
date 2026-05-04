#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <vector>

namespace lindblad {
namespace algorithms {

QuantumCircuit Grover::build_circuit(
    const QuantumCircuit& oracle,
    int num_iterations
) {
    int nq = oracle.n_qubits;

    if (num_iterations < 0) {
        num_iterations = static_cast<int>(
            std::round(PI / 4.0 * std::sqrt(static_cast<double>(1 << nq)))
        );
        if (num_iterations < 1) num_iterations = 1;
    }

    QuantumCircuit qc(nq);

    for (int q = 0; q < nq; ++q) qc.h(q);

    for (int iter = 0; iter < num_iterations; ++iter) {
        for (const auto& inst : oracle.instructions)
            qc.instructions.push_back(inst);

        // Diffusion operator: 2|s><s| - I
        for (int q = 0; q < nq; ++q) qc.h(q);
        for (int q = 0; q < nq; ++q) qc.x(q);

        if (nq >= 2) {
            qc.h(nq - 1);
            if (nq == 2) {
                qc.cx(0, 1);
            } else if (nq == 3) {
                qc.ccx(0, 1, 2);
            } else {
                size_t mcu = 1ULL << nq;
                std::vector<Complex128> mcx_mat(mcu * mcu, Complex128(0.0, 0.0));
                for (size_t idx = 0; idx < mcu; ++idx) {
                    if (idx == mcu - 2)      mcx_mat[idx * mcu + (mcu - 1)] = Complex128(1.0, 0.0);
                    else if (idx == mcu - 1) mcx_mat[idx * mcu + (mcu - 2)] = Complex128(1.0, 0.0);
                    else                     mcx_mat[idx * mcu + idx]        = Complex128(1.0, 0.0);
                }
                Instruction mcx_inst;
                mcx_inst.type = Instruction::GateType::UNITARY;
                for (int mq = 0; mq < nq; ++mq) mcx_inst.qubits.push_back(mq);
                mcx_inst.matrix = mcx_mat;
                qc.instructions.push_back(mcx_inst);
            }
            qc.h(nq - 1);
        }

        for (int q = 0; q < nq; ++q) qc.x(q);
        for (int q = 0; q < nq; ++q) qc.h(q);
    }

    return qc;
}

Grover::Result Grover::search(
    const QuantumCircuit& oracle,
    int num_iterations,
    int shots,
    uint64_t seed
) {
    auto circuit = build_circuit(oracle, num_iterations);
    circuit.measure_all();

    StatevectorSimulator sim;
    auto sim_result = sim.run(circuit, shots, seed);

    Result result;
    int max_count = 0;
    for (const auto& [bits, count] : sim_result.counts) {
        if (count > max_count) {
            max_count = count;
            result.solution = bits;
        }
    }
    result.probability = static_cast<double>(max_count) / shots;

    result.num_iterations = (num_iterations < 0)
        ? static_cast<int>(std::round(PI / 4.0 * std::sqrt(static_cast<double>(1 << oracle.n_qubits))))
        : num_iterations;

    return result;
}

} // namespace algorithms
} // namespace lindblad
