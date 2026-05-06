#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/gates.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>

namespace lindblad {

// =============================================================================
// apply_instruction — dispatch to the appropriate gate function
// =============================================================================

void StatevectorSimulator::apply_instruction(Statevector& sv, const Instruction& inst) {
    using GT = Instruction::GateType;
    const auto& q = inst.qubits;
    const auto& p = inst.params;

    switch (inst.type) {
        // Single-qubit gates
        case GT::H:    gates::apply_h(sv, q[0]); break;
        case GT::X:    gates::apply_x(sv, q[0]); break;
        case GT::Y:    gates::apply_y(sv, q[0]); break;
        case GT::Z:    gates::apply_z(sv, q[0]); break;
        case GT::S:    gates::apply_s(sv, q[0]); break;
        case GT::SDG:  gates::apply_sdg(sv, q[0]); break;
        case GT::T:    gates::apply_t(sv, q[0]); break;
        case GT::TDG:  gates::apply_tdg(sv, q[0]); break;
        case GT::SX:   gates::apply_sx(sv, q[0]); break;
        case GT::SXDG: gates::apply_sxdg(sv, q[0]); break;
        case GT::RX:   gates::apply_rx(sv, q[0], p[0]); break;
        case GT::RY:   gates::apply_ry(sv, q[0], p[0]); break;
        case GT::RZ:   gates::apply_rz(sv, q[0], p[0]); break;
        case GT::P:    gates::apply_p(sv, q[0], p[0]); break;
        case GT::U:    gates::apply_u(sv, q[0], p[0], p[1], p[2]); break;
        case GT::U1:   gates::apply_u1(sv, q[0], p[0]); break;
        case GT::U2:   gates::apply_u2(sv, q[0], p[0], p[1]); break;
        case GT::U3:   gates::apply_u3(sv, q[0], p[0], p[1], p[2]); break;

        // Two-qubit gates
        case GT::CX:   gates::apply_cx(sv, q[0], q[1]); break;
        case GT::CY:   gates::apply_cy(sv, q[0], q[1]); break;
        case GT::CZ:   gates::apply_cz(sv, q[0], q[1]); break;
        case GT::CH:   gates::apply_ch(sv, q[0], q[1]); break;
        case GT::SWAP: gates::apply_swap(sv, q[0], q[1]); break;
        case GT::ISWAP: gates::apply_iswap(sv, q[0], q[1]); break;
        case GT::CRX:  gates::apply_crx(sv, q[0], q[1], p[0]); break;
        case GT::CRY:  gates::apply_cry(sv, q[0], q[1], p[0]); break;
        case GT::CRZ:  gates::apply_crz(sv, q[0], q[1], p[0]); break;
        case GT::CP:   gates::apply_cp(sv, q[0], q[1], p[0]); break;
        case GT::CU:   gates::apply_cu(sv, q[0], q[1], p[0], p[1], p[2], p[3]); break;
        case GT::ECR:  gates::apply_ecr(sv, q[0], q[1]); break;
        case GT::RZX:  gates::apply_rzx(sv, q[0], q[1], p[0]); break;
        case GT::RXX:  gates::apply_rxx(sv, q[0], q[1], p[0]); break;
        case GT::RYY:  gates::apply_ryy(sv, q[0], q[1], p[0]); break;
        case GT::RZZ:  gates::apply_rzz(sv, q[0], q[1], p[0]); break;

        // Three-qubit gates
        case GT::CCX:   gates::apply_ccx(sv, q[0], q[1], q[2]); break;
        case GT::CCZ:   gates::apply_ccz(sv, q[0], q[1], q[2]); break;
        case GT::CSWAP: gates::apply_cswap(sv, q[0], q[1], q[2]); break;
        case GT::RCCX:  gates::apply_rccx(sv, q[0], q[1], q[2]); break;

        // Custom unitary
        case GT::UNITARY:
            gates::apply_unitary(sv, q, inst.matrix);
            break;

        // Special — handled at circuit level
        case GT::MEASURE:
        case GT::RESET:
        case GT::BARRIER:
            break;  // No effect on statevector

        // Parameterised — should have been resolved
        case GT::PARAM_RX:
        case GT::PARAM_RY:
        case GT::PARAM_RZ:
        case GT::PARAM_P:
        case GT::PARAM_U:
            throw std::runtime_error(
                "Unresolved parameterised gate: " + inst.gate_name() +
                ". Call assign_parameters() first."
            );
    }
}

// =============================================================================
// simulate_circuit
// =============================================================================

void StatevectorSimulator::simulate_circuit(
    Statevector& sv,
    const QuantumCircuit& circuit
) {
    for (const auto& inst : circuit.instructions) {
        apply_instruction(sv, inst);
    }
}

// =============================================================================
// run
// =============================================================================

StatevectorSimulator::Result StatevectorSimulator::run(
    const QuantumCircuit& circuit,
    int shots,
    uint64_t seed
) {
    Result result;

    try {
        if (circuit.n_qubits < 1) {
            throw std::invalid_argument("Circuit must have at least 1 qubit");
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // Reuse a thread-local working buffer to avoid repeated aligned alloc/free
        // on variational hot paths (VQE, QAOA) that call run() thousands of times.
        thread_local std::unique_ptr<Statevector> sv_work;
        if (!sv_work || sv_work->n_qubits != circuit.n_qubits) {
            sv_work = std::make_unique<Statevector>(circuit.n_qubits);
        } else {
            sv_work->initialize();
        }
        simulate_circuit(*sv_work, circuit);

        if (shots > 0) {
            result.counts = sv_work->sample_counts(shots, seed);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        result.simulation_time_seconds =
            std::chrono::duration<double>(t_end - t_start).count();

        // Copy simulated state to result (callers own result.final_state;
        // sv_work stays in the cache for the next call).
        result.final_state = Statevector(circuit.n_qubits);
        result.final_state.set_amplitudes(sv_work->real_parts, sv_work->imag_parts, sv_work->dim);
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

} // namespace lindblad
