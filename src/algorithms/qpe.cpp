#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"

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

    // Extract the unitary's matrix ONCE (2^nu basis-column simulations); the
    // per-k powers are then produced by repeated squaring below, so the whole
    // loop costs m-1 matrix products instead of 2^m - 1.
    const int nu = unitary.n_qubits;
    const size_t ud = 1ULL << nu;

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

    // Up holds U^(2^k) for the current evaluation qubit k.
    std::vector<std::vector<std::complex<double>>> Up = U_cols;

    for (int k = 0; k < num_eval_qubits; ++k) {
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

        // Advance to U^(2^(k+1)) by one squaring (repeated squaring).
        if (k + 1 < num_eval_qubits) {
            std::vector<std::vector<std::complex<double>>> Unew(
                ud, std::vector<std::complex<double>>(ud, 0.0));
            for (size_t r = 0; r < ud; ++r)
                for (size_t c = 0; c < ud; ++c)
                    for (size_t m = 0; m < ud; ++m)
                        Unew[r][c] += Up[r][m] * Up[m][c];
            Up = std::move(Unew);
        }
    }

    // Inverse QFT on the evaluation register.
    // Delegated to QFT::build_inverse_circuit — canonical LSB-LSB IQFT.
    // do_swaps=true is required: the controlled-U^(2^k) sequence above prepares
    // the eval register in qubit-0=LSB Fourier convention (qubit k carries the
    // 2^k weight of the phase), which the LSB-LSB IQFT inverts cleanly. See
    // CLAUDE.md "Project Conventions".
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

    // Standard QPE assumes the target register holds a single eigenstate of
    // the unitary, so the QPE Fourier state has a single peak and the most-
    // frequent bitstring is the best estimate of m. Callers passing a
    // superposition of eigenstates (such as Shor's |1⟩ = (1/√r) Σ |u_s⟩) get
    // a distribution over s that biases the most-frequent bitstring toward
    // exact-m peaks — those callers should iterate over result.counts in
    // descending frequency, as Shor::find_order does.
    std::string best_bits;
    int max_count = 0;
    for (const auto& [bits, count] : result.counts) {
        if (count > max_count) {
            max_count = count;
            best_bits = bits;
        }
    }

    // Extract m from eval register in project LSB-at-qubit-0 convention.
    //
    // measure_all() maps qubit q → clbit q for all (num_eval_qubits +
    // unitary.n_qubits) qubits. The statevector simulator builds the bitstring
    // with clbit 0 at the rightmost character, so qubit q lives at position
    // (total - 1 - q). The eval register occupies qubits [0, num_eval_qubits),
    // i.e. the rightmost num_eval_qubits characters of best_bits. After the
    // LSB-LSB IQFT, eval qubit q carries bit q of the QPE integer m.
    const int total = static_cast<int>(best_bits.size());
    uint64_t measured = 0;
    for (int q = 0; q < num_eval_qubits && q < total; ++q) {
        if (best_bits[total - 1 - q] == '1') measured |= (1ULL << q);
    }

    return static_cast<double>(measured) /
           static_cast<double>(1ULL << num_eval_qubits);
}

// =============================================================================
// QuditPhaseEstimation
// =============================================================================

QuditPhaseEstimation::Result QuditPhaseEstimation::estimate(
    int m, int d,
    const std::vector<Complex128>& U,
    const std::vector<Complex128>& eigenstate,
    uint64_t seed,
    QuditBackend backend,
    const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument("QuditPhaseEstimation::estimate: d must be >= 2");
    if (m < 1)
        throw std::invalid_argument("QuditPhaseEstimation::estimate: m must be >= 1");
    if (static_cast<int>(U.size()) != d * d)
        throw std::invalid_argument("QuditPhaseEstimation::estimate: U must be a d×d matrix (size d*d)");
    if (static_cast<int>(eigenstate.size()) != d)
        throw std::invalid_argument("QuditPhaseEstimation::estimate: eigenstate must have d elements");

    if (backend == QuditBackend::CLIFFORD)
        throw std::invalid_argument(
            "QuditPhaseEstimation::estimate: CLIFFORD backend is not supported "
            "(controlled-U^k operations are not Clifford in general)");

    const auto F  = qudit_gates::qft_matrix(d);
    const auto Fd = qudit_gates::iqft_matrix(d);

    // ── DENSITY_MATRIX path ──────────────────────────────────────────────────
    if (backend == QuditBackend::DENSITY_MATRIX) {
        // Build a QuditStatevector first, then construct DM from it
        QuditStatevector sv_tmp(m + 1, d);
        sv_tmp.amplitudes.assign(sv_tmp.dim, Complex128(0.0, 0.0));
        for (int v = 0; v < d; ++v) {
            std::vector<int> digs(static_cast<size_t>(m + 1), 0);
            digs[static_cast<size_t>(m)] = v;
            sv_tmp.amplitudes[QuditStatevector::digits_to_index(digs, d)] =
                eigenstate[static_cast<size_t>(v)];
        }

        QuditDensityMatrix dm(sv_tmp);
        for (int j = 0; j < m; ++j) dm.apply_1qudit(j, F);
        for (int j = 0; j < m; ++j) {
            const int power = static_cast<int>(QuditStatevector::ipow(static_cast<size_t>(d), j));
            const auto ctrl_U = qudit_gates::controlled_power_matrix(d, U, power);
            dm.apply_2qudit(j, m, ctrl_U);
        }
        for (int j = 0; j < m; ++j) dm.apply_1qudit(j, Fd);
        if (noise) dm.apply_noise(*noise);
        const auto outcome = dm.measure(seed);

        std::vector<int> phase_digits(outcome.begin(), outcome.begin() + m);
        double phase = 0.0;
        double power_inv = 1.0 / static_cast<double>(d);
        for (int j = 0; j < m; ++j) {
            phase += static_cast<double>(outcome[static_cast<size_t>(j)]) * power_inv;
            power_inv /= static_cast<double>(d);
        }
        return Result{phase_digits, phase, m, d};
    }

    // ── MPS path ─────────────────────────────────────────────────────────────
    if (backend == QuditBackend::MPS) {
        QuditStatevector sv_tmp(m + 1, d);
        sv_tmp.amplitudes.assign(sv_tmp.dim, Complex128(0.0, 0.0));
        for (int v = 0; v < d; ++v) {
            std::vector<int> digs(static_cast<size_t>(m + 1), 0);
            digs[static_cast<size_t>(m)] = v;
            sv_tmp.amplitudes[QuditStatevector::digits_to_index(digs, d)] =
                eigenstate[static_cast<size_t>(v)];
        }

        QuditMPS mps(sv_tmp);
        for (int j = 0; j < m; ++j) mps.apply_1qudit(j, F);
        for (int j = 0; j < m; ++j) {
            const int power = static_cast<int>(QuditStatevector::ipow(static_cast<size_t>(d), j));
            const auto ctrl_U = qudit_gates::controlled_power_matrix(d, U, power);
            mps.apply_2qudit(j, m, ctrl_U);
        }
        for (int j = 0; j < m; ++j) mps.apply_1qudit(j, Fd);
        const auto outcome = mps.measure(seed);

        std::vector<int> phase_digits(outcome.begin(), outcome.begin() + m);
        double phase = 0.0;
        double power_inv = 1.0 / static_cast<double>(d);
        for (int j = 0; j < m; ++j) {
            phase += static_cast<double>(outcome[static_cast<size_t>(j)]) * power_inv;
            power_inv /= static_cast<double>(d);
        }
        return Result{phase_digits, phase, m, d};
    }

    // ── STATEVECTOR path (default) ───────────────────────────────────────────
    QuditStatevector sv(m + 1, d);
    sv.amplitudes.assign(sv.dim, Complex128(0.0, 0.0));
    for (int v = 0; v < d; ++v) {
        std::vector<int> digs(static_cast<size_t>(m + 1), 0);
        digs[static_cast<size_t>(m)] = v;
        sv.amplitudes[QuditStatevector::digits_to_index(digs, d)] =
            eigenstate[static_cast<size_t>(v)];
    }
    for (int j = 0; j < m; ++j) sv.apply_1qudit(j, F);
    for (int j = 0; j < m; ++j) {
        const int power = static_cast<int>(QuditStatevector::ipow(static_cast<size_t>(d), j));
        const auto ctrl_U = qudit_gates::controlled_power_matrix(d, U, power);
        sv.apply_2qudit(j, m, ctrl_U);
    }
    for (int j = 0; j < m; ++j) sv.apply_1qudit(j, Fd);
    const auto outcome = sv.measure(seed);

    std::vector<int> phase_digits(outcome.begin(), outcome.begin() + m);
    double phase = 0.0;
    double power_inv = 1.0 / static_cast<double>(d);
    for (int j = 0; j < m; ++j) {
        phase += static_cast<double>(outcome[static_cast<size_t>(j)]) * power_inv;
        power_inv /= static_cast<double>(d);
    }
    return Result{phase_digits, phase, m, d};
}

} // namespace algorithms
} // namespace lindblad
