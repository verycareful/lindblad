#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/operators.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lindblad {

// Thread-local RNG for mid-circuit measurement collapse.
// Seeded from run() for reproducibility; default-initialised from random_device otherwise.
thread_local std::mt19937_64 sv_sim_rng{std::random_device{}()};

// Forward declaration: shared Born-rule collapse helper (defined with the
// trajectory helpers below; used by the MEASURE/RESET dispatch cases).
static int sv_collapse_qubit(Statevector& sv, int qubit, std::mt19937_64& rng);

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

        // Multi-controlled X: qubits = [controls..., target].
        case GT::MCX: {
            std::vector<int> controls(q.begin(), q.end() - 1);
            gates::apply_mcx(sv, controls, q.back());
            break;
        }
        // Multi-controlled phase (symmetric controls).
        case GT::MCP:
            gates::apply_mcp(sv, q, p[0]);
            break;
        // Basis permutation on the target subspace.
        case GT::PERMUTATION:
            gates::apply_permutation(sv, q, inst.permutation);
            break;

        // BARRIER — no effect on statevector
        case GT::BARRIER:
            break;

        // MEASURE: project the qubit to a random outcome, collapse and
        // renormalise via the shared Born-rule helper (outcome recording
        // happens in the trajectory runner, which calls the helper directly).
        case GT::MEASURE:
            sv_collapse_qubit(sv, q[0], sv_sim_rng);
            break;

        // RESET: measure the qubit, then apply X if the outcome was |1⟩ so
        // the post-state is |0⟩.
        case GT::RESET:
            if (sv_collapse_qubit(sv, q[0], sv_sim_rng) == 1)
                gates::apply_x(sv, q[0]);
            break;

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
// Trajectory helpers
// =============================================================================

// Collapse `qubit` to a sampled outcome (Born rule), renormalise, return it.
// Shared by the MEASURE/RESET dispatch cases and the trajectory runner
// (pre-R.1.13 the same O(dim) loops were triplicated and always serial;
// audit F-11). Both passes iterate the two qubit-value halves as strided
// blocks with branch-free inner loops and are OMP-gated at the same
// dim >= 2^20 threshold as the gate kernels.
static int sv_collapse_qubit(Statevector& sv, int qubit, std::mt19937_64& rng) {
    const size_t step = 1ULL << qubit;
    const size_t dim = sv.dim;
    const int n_blocks = static_cast<int>(dim / (2 * step));
    double* __restrict__ rp = sv.real_parts;
    double* __restrict__ ip = sv.imag_parts;

    double prob0 = 0.0;
    #pragma omp parallel for reduction(+:prob0) schedule(static) if(dim >= (1<<20))
    for (int bi = 0; bi < n_blocks; ++bi) {
        const size_t base = static_cast<size_t>(bi) * 2 * step;
        double acc = 0.0;
        #pragma omp simd reduction(+:acc)
        for (size_t j = 0; j < step; ++j)
            acc += rp[base + j] * rp[base + j] + ip[base + j] * ip[base + j];
        prob0 += acc;
    }

    std::uniform_real_distribution<double> udist(0.0, 1.0);
    const int outcome = (udist(rng) < prob0) ? 0 : 1;
    const double p_out = (outcome == 0) ? prob0 : (1.0 - prob0);
    const double inv_norm = (p_out > 1e-15) ? 1.0 / std::sqrt(p_out) : 1.0;
    const size_t keep_off = (outcome == 0) ? 0 : step;   // half that survives
    const size_t zero_off = step - keep_off;             // half that vanishes

    #pragma omp parallel for schedule(static) if(dim >= (1<<20))
    for (int bi = 0; bi < n_blocks; ++bi) {
        const size_t base = static_cast<size_t>(bi) * 2 * step;
        double* __restrict__ kr = rp + base + keep_off;
        double* __restrict__ ki = ip + base + keep_off;
        double* __restrict__ zr = rp + base + zero_off;
        double* __restrict__ zi = ip + base + zero_off;
        #pragma omp simd
        for (size_t j = 0; j < step; ++j) {
            kr[j] *= inv_norm;
            ki[j] *= inv_norm;
            zr[j] = 0.0;
            zi[j] = 0.0;
        }
    }
    return outcome;
}

// Execute one trajectory: classical conditions are honoured against `clreg`,
// MEASURE collapses the state and records its outcome into `clreg`.
static void sv_run_trajectory(StatevectorSimulator& sim, Statevector& sv,
                              const QuantumCircuit& circuit,
                              std::vector<int>& clreg, int n_clbits,
                              std::mt19937_64& rng) {
    using GT = Instruction::GateType;
    for (const auto& inst : circuit.instructions) {
        if (inst.type == GT::BARRIER) continue;
        if (inst.condition_clbit >= 0) {
            const int cv = (inst.condition_clbit < n_clbits)
                           ? clreg[inst.condition_clbit] : 0;
            if (cv != inst.condition_value) continue;
        }
        if (inst.type == GT::MEASURE) {
            const int qubit = inst.qubits[0];
            const int clbit = inst.clbits.empty() ? -1 : inst.clbits[0];
            const int outcome = sv_collapse_qubit(sv, qubit, rng);
            if (clbit >= 0 && clbit < n_clbits) clreg[clbit] = outcome;
            continue;
        }
        sim.apply_instruction(sv, inst);
    }
}

// True when no instruction (other than BARRIER) acts on a qubit after that
// qubit has been measured: the pre-measurement state is then deterministic and
// outcomes can be sampled from one forward pass instead of per-shot reruns.
static bool sv_measures_are_terminal(const QuantumCircuit& circuit) {
    std::vector<bool> measured(static_cast<size_t>(circuit.n_qubits), false);
    for (const auto& inst : circuit.instructions) {
        if (inst.type == Instruction::GateType::BARRIER) continue;
        for (int q : inst.qubits)
            if (q >= 0 && q < circuit.n_qubits &&
                measured[static_cast<size_t>(q)])
                return false;
        if (inst.type == Instruction::GateType::MEASURE)
            measured[static_cast<size_t>(inst.qubits[0])] = true;
    }
    return true;
}

// =============================================================================
// simulate_circuit
// =============================================================================

void StatevectorSimulator::simulate_circuit(
    Statevector& sv,
    const QuantumCircuit& circuit
) {
    // Trajectory semantics (docs/api/simulators.md, Execution semantics):
    // classical conditions are honoured against a local register and MEASURE
    // outcomes are recorded into it (collapse drawn from the thread-local
    // RNG). For circuits without measurement or conditioning this reduces to
    // the plain forward pass.
    const int n_clbits =
        circuit.n_clbits > 0 ? circuit.n_clbits : circuit.n_qubits;
    std::vector<int> clreg(static_cast<size_t>(n_clbits), 0);
    sv_run_trajectory(*this, sv, circuit, clreg, n_clbits, sv_sim_rng);
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

    // Strict exact-evaluation rule (docs/api/simulators.md): an exact
    // expectation value is undefined for stochastic evolution. Circuits with
    // measurement or classical conditioning must be estimated from counts
    // (run() with shots > 0) instead of silently evaluating one trajectory.
    for (const auto& inst : circuit.instructions) {
        if (inst.type == Instruction::GateType::MEASURE ||
            inst.condition_clbit >= 0) {
            throw std::invalid_argument(
                "StatevectorSimulator::eval_expectation: circuit contains "
                "measurement or classically-conditioned instructions; the "
                "exact expectation of a stochastic trajectory is undefined. "
                "Use run() with shots > 0 and estimate from counts.");
        }
    }

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

        // Execution strategy (docs/api/simulators.md, Execution semantics):
        //   1. Terminal-only measurements (no feedforward, nothing acting on
        //      a qubit after it was measured): ONE forward pass, then sample
        //      outcomes from the final state with the qubit -> clbit map.
        //      O(gates + shots) instead of O(shots * gates).
        //   2. Mid-circuit measurement or feedforward with shots > 0:
        //      per-shot trajectories (each collapse drawn independently).
        //   3. Otherwise (shots == 0, or no measurements): a single seeded
        //      trajectory; conditions honoured, MEASURE outcomes recorded.
        bool has_measure = false;
        bool has_condition = false;
        int n_clbits = circuit.n_clbits > 0 ? circuit.n_clbits : circuit.n_qubits;
        for (const auto& inst : circuit.instructions) {
            if (inst.type == Instruction::GateType::MEASURE) has_measure = true;
            if (inst.condition_clbit >= 0) has_condition = true;
        }
        const bool terminal_only =
            has_measure && !has_condition && sv_measures_are_terminal(circuit);

        if (shots > 0 && has_measure && !terminal_only) {
            // Per-shot trajectories: re-initialise and re-simulate for every
            // shot so each MEASURE collapses the state independently.
            result.counts.clear();
            std::vector<int> clreg(n_clbits, 0);

            for (int shot = 0; shot < shots; ++shot) {
                sv_work->initialize();
                clreg.assign(n_clbits, 0);
                sv_run_trajectory(*this, *sv_work, circuit, clreg, n_clbits,
                                  sv_sim_rng);

                // Build bitstring: clbit 0 is LSB (rightmost), highest clbit is MSB.
                std::string bits(n_clbits, '0');
                for (int c = 0; c < n_clbits; ++c) {
                    if (clreg[c]) bits[n_clbits - 1 - c] = '1';
                }
                result.counts[bits]++;
            }
        } else if (shots > 0 && terminal_only) {
            // Terminal-measurement fast path: a single evolution with MEASURE
            // skipped (the pre-measurement state is deterministic), then
            // multinomial sampling keyed by the qubit -> clbit mapping.
            for (const auto& inst : circuit.instructions) {
                if (inst.type == Instruction::GateType::MEASURE ||
                    inst.type == Instruction::GateType::BARRIER) continue;
                apply_instruction(*sv_work, inst);
            }

            std::vector<std::pair<int, int>> meas;  // (qubit, clbit)
            for (const auto& inst : circuit.instructions)
                if (inst.type == Instruction::GateType::MEASURE)
                    meas.emplace_back(inst.qubits[0],
                                      inst.clbits.empty() ? inst.qubits[0]
                                                          : inst.clbits[0]);

            // Cumulative probabilities + binary search per shot (same scheme
            // as Statevector::sample_counts, with clbit-mapped keys).
            std::vector<double> cum(sv_work->dim);
            double acc = 0.0;
            for (size_t i = 0; i < sv_work->dim; ++i) {
                acc += sv_work->real_parts[i] * sv_work->real_parts[i]
                     + sv_work->imag_parts[i] * sv_work->imag_parts[i];
                cum[i] = acc;
            }
            // Sample with the per-call RNG seeded above from {seed, tid}:
            // identical seeds on DIFFERENT threads stay statistically
            // independent (B9_StatevectorRngParallelIndependence, relied on
            // by Estimator::run_batch), while single-threaded runs remain
            // reproducible (sv_sim_rng is reseeded deterministically per
            // call). A plain mt19937_64(seed) here would give every thread
            // the same stream.
            std::uniform_real_distribution<double> udist(0.0, 1.0);
            for (int s = 0; s < shots; ++s) {
                const double r = udist(sv_sim_rng);
                auto it = std::lower_bound(cum.begin(), cum.end(), r);
                size_t outcome = static_cast<size_t>(
                    std::distance(cum.begin(), it));
                if (outcome >= sv_work->dim) outcome = sv_work->dim - 1;

                std::string bits(n_clbits, '0');
                for (const auto& [q, c] : meas) {
                    if (c < 0 || c >= n_clbits) continue;
                    if ((outcome >> q) & 1) bits[n_clbits - 1 - c] = '1';
                }
                result.counts[bits]++;
            }
        } else {
            // Single seeded trajectory (shots == 0 semantics; also the plain
            // forward pass for measurement-free circuits with shots > 0).
            std::vector<int> clreg(n_clbits, 0);
            sv_run_trajectory(*this, *sv_work, circuit, clreg, n_clbits,
                              sv_sim_rng);
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
