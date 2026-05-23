#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/operators.hpp"
#include <cmath>
#include <chrono>
#include <memory>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lindblad {

// Thread-local RNG for mid-circuit measurement collapse.
// Seeded from run() for reproducibility; default-initialised from random_device otherwise.
thread_local std::mt19937_64 sv_sim_rng{std::random_device{}()};

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

        // BARRIER — no effect on statevector
        case GT::BARRIER:
            break;

        // MEASURE — project qubit to a random outcome, collapse and renormalise
        case GT::MEASURE: {
            int qubit = q[0];
            size_t step = 1ULL << qubit;
            double prob0 = 0.0;
            for (size_t i = 0; i < sv.dim; i += 2 * step)
                for (size_t j = i; j < i + step; ++j)
                    prob0 += sv.real_parts[j] * sv.real_parts[j]
                           + sv.imag_parts[j] * sv.imag_parts[j];
            std::uniform_real_distribution<double> udist(0.0, 1.0);
            int outcome = (udist(sv_sim_rng) < prob0) ? 0 : 1;
            double p_out = (outcome == 0) ? prob0 : (1.0 - prob0);
            double inv_norm = (p_out > 1e-15) ? 1.0 / std::sqrt(p_out) : 1.0;
            for (size_t i = 0; i < sv.dim; ++i) {
                if (static_cast<int>((i >> qubit) & 1) != outcome) {
                    sv.real_parts[i] = 0.0;
                    sv.imag_parts[i] = 0.0;
                } else {
                    sv.real_parts[i] *= inv_norm;
                    sv.imag_parts[i] *= inv_norm;
                }
            }
            break;
        }

        // RESET — measure qubit, then apply X if outcome was |1⟩ to restore |0⟩
        case GT::RESET: {
            int qubit = q[0];
            size_t step = 1ULL << qubit;
            double prob0 = 0.0;
            for (size_t i = 0; i < sv.dim; i += 2 * step)
                for (size_t j = i; j < i + step; ++j)
                    prob0 += sv.real_parts[j] * sv.real_parts[j]
                           + sv.imag_parts[j] * sv.imag_parts[j];
            std::uniform_real_distribution<double> udist(0.0, 1.0);
            int outcome = (udist(sv_sim_rng) < prob0) ? 0 : 1;
            double p_out = (outcome == 0) ? prob0 : (1.0 - prob0);
            double inv_norm = (p_out > 1e-15) ? 1.0 / std::sqrt(p_out) : 1.0;
            for (size_t i = 0; i < sv.dim; ++i) {
                if (static_cast<int>((i >> qubit) & 1) != outcome) {
                    sv.real_parts[i] = 0.0;
                    sv.imag_parts[i] = 0.0;
                } else {
                    sv.real_parts[i] *= inv_norm;
                    sv.imag_parts[i] *= inv_norm;
                }
            }
            if (outcome == 1) gates::apply_x(sv, qubit);
            break;
        }

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
// eval_expectation
// =============================================================================

double StatevectorSimulator::eval_expectation(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable
) {
    if (circuit.n_qubits < 1)
        throw std::invalid_argument(
            "StatevectorSimulator::eval_expectation: circuit must have at least 1 qubit");

    thread_local std::unique_ptr<Statevector> sv_work;
    if (!sv_work || sv_work->n_qubits != circuit.n_qubits) {
        sv_work = std::make_unique<Statevector>(circuit.n_qubits);
    } else {
        sv_work->initialize();
    }

    simulate_circuit(*sv_work, circuit);
    return observable.expectation_value(*sv_work);
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
        // Derive a per-thread seed so parallel batches (e.g. Estimator::run_batch)
        // produce statistically independent RNG streams rather than every thread
        // reseeding to the same Mersenne Twister state. The single-threaded path
        // is unaffected: outside a parallel region omp_get_thread_num() returns 0
        // and the same `seed` always produces the same RNG stream.
        const uint64_t base_seed = (seed == 0)
            ? static_cast<uint64_t>(std::random_device{}())
            : seed;
#ifdef _OPENMP
        const uint32_t tid = static_cast<uint32_t>(omp_get_thread_num());
#else
        const uint32_t tid = 0;
#endif
        std::seed_seq seq{
            static_cast<uint32_t>(base_seed),
            static_cast<uint32_t>(base_seed >> 32),
            tid
        };
        sv_sim_rng.seed(seq);

        // Determine whether the circuit contains any mid-circuit MEASURE gates.
        // If so, each shot must re-run the full circuit from |0...0⟩ so that the
        // stochastic collapse is sampled independently per shot.  If there are no
        // MEASURE instructions the state is deterministic after one forward pass
        // and we fall back to the fast sample_counts path.
        bool has_measure = false;
        int n_clbits = circuit.n_clbits > 0 ? circuit.n_clbits : circuit.n_qubits;
        for (const auto& inst : circuit.instructions) {
            if (inst.type == Instruction::GateType::MEASURE) {
                has_measure = true;
                break;
            }
        }

        if (shots > 0 && has_measure) {
            // Per-shot execution: re-initialise and re-simulate for every shot so
            // that each MEASURE collapses the state independently.
            result.counts.clear();
            std::vector<int> clreg(n_clbits, 0);  // classical register for one shot

            for (int shot = 0; shot < shots; ++shot) {
                sv_work->initialize();
                clreg.assign(n_clbits, 0);

                for (const auto& inst : circuit.instructions) {
                    if (inst.type == Instruction::GateType::MEASURE) {
                        // Collapse the qubit and record the outcome in the classical register.
                        int qubit = inst.qubits[0];
                        int clbit = inst.clbits[0];
                        size_t step = 1ULL << qubit;
                        double prob0 = 0.0;
                        for (size_t i = 0; i < sv_work->dim; i += 2 * step)
                            for (size_t j = i; j < i + step; ++j)
                                prob0 += sv_work->real_parts[j] * sv_work->real_parts[j]
                                       + sv_work->imag_parts[j] * sv_work->imag_parts[j];
                        std::uniform_real_distribution<double> udist(0.0, 1.0);
                        int outcome = (udist(sv_sim_rng) < prob0) ? 0 : 1;
                        double p_out = (outcome == 0) ? prob0 : (1.0 - prob0);
                        double inv_norm = (p_out > 1e-15) ? 1.0 / std::sqrt(p_out) : 1.0;
                        for (size_t i = 0; i < sv_work->dim; ++i) {
                            if (static_cast<int>((i >> qubit) & 1) != outcome) {
                                sv_work->real_parts[i] = 0.0;
                                sv_work->imag_parts[i] = 0.0;
                            } else {
                                sv_work->real_parts[i] *= inv_norm;
                                sv_work->imag_parts[i] *= inv_norm;
                            }
                        }
                        if (clbit >= 0 && clbit < n_clbits) {
                            clreg[clbit] = outcome;
                        }
                    } else {
                        if (inst.condition_clbit >= 0) {
                            int cv = (inst.condition_clbit < n_clbits)
                                     ? clreg[inst.condition_clbit] : 0;
                            if (cv != inst.condition_value) continue;
                        }
                        apply_instruction(*sv_work, inst);
                    }
                }

                // Build bitstring: clbit 0 is LSB (rightmost), highest clbit is MSB.
                std::string bits(n_clbits, '0');
                for (int c = 0; c < n_clbits; ++c) {
                    if (clreg[c]) bits[n_clbits - 1 - c] = '1';
                }
                result.counts[bits]++;
            }
        } else {
            simulate_circuit(*sv_work, circuit);
            if (shots > 0) {
                result.counts = sv_work->sample_counts(shots, seed);
            }
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
