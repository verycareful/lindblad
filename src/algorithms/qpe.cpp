#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <complex>
#include <string>
#include <vector>

namespace lindblad {
namespace algorithms {

QuantumCircuit QPE::build_circuit(
    const QuantumCircuit& unitary,
    int num_eval_qubits
) {
    int total_qubits = num_eval_qubits + unitary.n_qubits;
    QuantumCircuit qc(total_qubits);

    for (int i = 0; i < num_eval_qubits; ++i) {
        qc.h(i);
    }

    for (int k = 0; k < num_eval_qubits; ++k) {
        int power = 1 << k;

        int nu = unitary.n_qubits;
        size_t ud = 1ULL << nu;

        std::vector<std::vector<std::complex<double>>> U_cols(ud,
            std::vector<std::complex<double>>(ud, 0.0));
        for (size_t col = 0; col < ud; ++col) {
            Statevector basis(nu);
            basis.initialize_basis(col);
            StatevectorSimulator sv_sim;
            sv_sim.simulate_circuit(basis, unitary);
            for (size_t row = 0; row < ud; ++row) {
                U_cols[row][col] = {basis.real_parts[row], basis.imag_parts[row]};
            }
        }

        std::vector<std::vector<std::complex<double>>> Up = U_cols;
        for (int rep = 1; rep < power; ++rep) {
            std::vector<std::vector<std::complex<double>>> Unew(
                ud, std::vector<std::complex<double>>(ud, 0.0));
            for (size_t r = 0; r < ud; ++r)
                for (size_t c = 0; c < ud; ++c)
                    for (size_t m = 0; m < ud; ++m)
                        Unew[r][c] += Up[r][m] * U_cols[m][c];
            Up = Unew;
        }

        std::vector<Complex128> Upow_flat(ud * ud);
        for (size_t r = 0; r < ud; ++r)
            for (size_t c = 0; c < ud; ++c)
                Upow_flat[r * ud + c] = Complex128(Up[r][c].real(), Up[r][c].imag());

        // apply_unitary maps targets[0] (= ctrl qubit k) to bit 0 (LSB) of the
        // subspace index.  Even subspace indices have ctrl=0; odd have ctrl=1.
        size_t full_dim = 1ULL << (1 + nu);
        std::vector<Complex128> CU_matrix(full_dim * full_dim, Complex128(0.0, 0.0));
        // ctrl=0 block: even rows/cols (bit 0 = 0) → identity
        for (size_t t = 0; t < ud; ++t)
            CU_matrix[(2 * t) * full_dim + (2 * t)] = Complex128(1.0, 0.0);
        // ctrl=1 block: odd rows/cols (bit 0 = 1) → U^power
        for (size_t r = 0; r < ud; ++r)
            for (size_t c = 0; c < ud; ++c)
                CU_matrix[(2 * r + 1) * full_dim + (2 * c + 1)] = Upow_flat[r * ud + c];

        Instruction ctrl_u;
        ctrl_u.type = Instruction::GateType::UNITARY;
        ctrl_u.qubits = {k};
        for (int tq = 0; tq < nu; ++tq)
            ctrl_u.qubits.push_back(num_eval_qubits + tq);
        ctrl_u.matrix = CU_matrix;
        qc.instructions.push_back(ctrl_u);
    }

    // Inverse QFT on the evaluation register.
    // Delegated to QFT::build_inverse_circuit — canonical IQFT implementation.
    // do_swaps=true: output bit-reversal included so estimate_phase reads MSB-first.
    auto iqft = QFT::build_inverse_circuit(num_eval_qubits, /*do_swaps=*/true);
    for (const auto& inst : iqft.instructions)
        qc.instructions.push_back(inst);

    return qc;
}

double QPE::estimate_phase(
    const QuantumCircuit& unitary,
    int num_eval_qubits,
    int shots,
    uint64_t seed
) {
    auto circuit = build_circuit(unitary, num_eval_qubits);
    circuit.measure_all();

    StatevectorSimulator sim;
    auto result = sim.run(circuit, shots, seed);

    std::string best_bits;
    int max_count = 0;
    for (const auto& [bits, count] : result.counts) {
        if (count > max_count) {
            max_count = count;
            best_bits = bits;
        }
    }

    int measured = 0;
    for (int i = 0; i < num_eval_qubits; ++i) {
        if (best_bits[i] == '1') {
            measured |= (1 << (num_eval_qubits - 1 - i));
        }
    }

    return static_cast<double>(measured) / (1 << num_eval_qubits);
}

} // namespace algorithms
} // namespace lindblad
