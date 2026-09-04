#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/operators.hpp"
#include <optional>
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

void StatevectorSimulator::apply_instruction(Statevector& sv,
                                            const Instruction& inst) {
    apply_instruction(sv, inst, inst.validation);
}

void StatevectorSimulator::apply_instruction(Statevector& sv,
                                            const Instruction& inst,
                                            ValidationOptions physical) {
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
            gates::apply_unitary(sv, q, inst.matrix, physical);
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
// rather than triplicating the same O(dim) loops at each call site. Both
// passes iterate the two qubit-value halves as strided
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
// runner = the run's observation harness, or null when nothing is watching.
// An instruction that is skipped (a barrier, or a conditioned gate whose
// condition does not hold) still fires its anchors: the anchor names a point
// the run reached, and a caller watching every instruction wants to see the
// state at that point whether or not it changed.
static void sv_run_trajectory(StatevectorSimulator& sim, Statevector& sv,
                              const QuantumCircuit& circuit,
                              std::vector<int>& clreg, int n_clbits,
                              std::mt19937_64& rng,
                              detail::ObservationRunner* runner = nullptr) {
    using GT = Instruction::GateType;

    const StateView view(StateForm::Statevector, &sv, sv.n_qubits);
    if (runner) runner->at_start(view);

    int index = -1;
    for (const auto& inst : circuit.instructions) {
        ++index;
        if (runner) runner->before_instruction(index, inst, view);

        bool skip = inst.type == GT::BARRIER;
        if (!skip && inst.condition_clbit >= 0) {
            const int cv = (inst.condition_clbit < n_clbits)
                           ? clreg[inst.condition_clbit] : 0;
            if (cv != inst.condition_value) skip = true;
        }

        if (!skip) {
            if (inst.type == GT::MEASURE) {
                const int qubit = inst.qubits[0];
                const int clbit = inst.clbits.empty() ? -1 : inst.clbits[0];
                const int outcome = sv_collapse_qubit(sv, qubit, rng);
                if (clbit >= 0 && clbit < n_clbits) clreg[clbit] = outcome;
            } else {
                sim.apply_instruction(sv, inst, {Validation::Ignore});
            }
        }

        if (runner) runner->after_instruction(index, inst, view);
    }

    if (runner) runner->at_end(view, index);
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
    const QuantumCircuit& circuit_in
) {
    // Trajectory semantics (docs/api/simulators.md, Execution semantics):
    // classical conditions are honoured against a local register and MEASURE
    // outcomes are recorded into it (collapse drawn from the thread-local
    // RNG). For circuits without measurement or conditioning this reduces to
    // the plain forward pass.
    // A public entry that does not pass through run(), so it owns the physical
    // pre-flight itself. The trajectory below then applies under Ignore, the
    // same division run() uses.
    // Under Repair::Attempt a repaired copy is executed and the caller's
    // circuit is left exactly as it was handed over; Repair::None binds
    // straight to it and nothing is copied.
    std::optional<QuantumCircuit> repaired_storage =
        circuit_in.validated_physical();
    const QuantumCircuit& circuit =
        repaired_storage ? *repaired_storage : circuit_in;

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

// =============================================================================
// Gate fusion (R.1.17) — fuse runs of small gates into dense k-qubit blocks
//
// At large n every gate application is a full sweep of 2^n amplitudes and
// the simulation is memory-bound: k gates on overlapping supports cost k
// sweeps but could cost one. This pre-pass greedily merges consecutive
// FUSABLE gates while the union of their supports stays within
// Options::fusion_max_qubit, composes each block into a dense 2^k x 2^k unitary
// (by applying the member gates to basis columns through the simulator's own
// dispatch, so every gate type composes exactly), and emits it as an
// ordinary UNITARY instruction handled by the existing gates::apply_unitary
// stride kernel. Blocks with a single member emit the original instruction.
//
// Semantics are untouched by construction: MEASURE / RESET / BARRIER,
// classically-conditioned instructions, unresolved parameterised gates, and
// the structured ops (MCX / MCP / PERMUTATION, which already have fast
// native paths) are never fused — they flush the current block and pass
// through verbatim. The result is a plain QuantumCircuit, so all execution
// paths (including per-shot trajectories, which then reuse the fused plan
// across every shot) consume it unchanged.
//
// Fusion engages when the statevector stops fitting in one last-level
// cache instance (16·2^n bytes vs hw::llc_bytes(), per-instance by design;
// 32 MiB assumed when detection fails). Cache-resident states are
// compute-bound -- the specialised per-gate kernels win there and a dense
// 32x32 block would ADD arithmetic -- while DRAM-resident states are
// bandwidth-bound and the sweep count is all that matters. Measured on a
// 32 MiB-L3 part: fusing a 32 MiB state ran 3.3x SLOWER, fusing a 64 MiB
// state ran 2.7x faster, and cache-resident sizes regressed up to 15x when
// fused -- so the boundary is "strictly larger than one LLC instance".
// Options::fusion_enable / fusion_threshold / fusion_max_qubit (Aer-parity
// names) override the automatics; below the engagement point execution is
// bit-identical to the unfused path by construction.
// =============================================================================

namespace {

constexpr int SV_FUSION_MAX_QUBIT_LIMIT = 6;  // block scratch is 2^k x 2^k; 6 -> 64x64
constexpr std::size_t SV_BYTES_PER_AMP = 16;  // SoA double re + im per amplitude

// Bounds for the auto engagement point. The fallback applies when the OS
// reports no L3 (containers, exotic platforms); the clamp guards the
// formula against garbage detection values, and both err toward engaging
// LATER (a missed fusion win costs far less than a mid-range regression).
constexpr std::size_t SV_FUSION_LLC_FALLBACK = std::size_t(32) << 20;  // 32 MiB
constexpr std::size_t SV_FUSION_LLC_MIN = std::size_t(1) << 20;        // 1 MiB
constexpr std::size_t SV_FUSION_LLC_MAX = std::size_t(1) << 30;        // 1 GiB

// Smallest n whose statevector exceeds one LLC instance -- the point where
// simulation turns bandwidth-bound and fusion starts paying (see the block
// comment above). Detected once, cached thread-safely.
int sv_fusion_auto_threshold() {
    static const int cached = [] {
        std::size_t llc = hw::llc_bytes();
        if (llc == 0) llc = SV_FUSION_LLC_FALLBACK;
        llc = std::clamp(llc, SV_FUSION_LLC_MIN, SV_FUSION_LLC_MAX);
        int n = 1;
        while ((SV_BYTES_PER_AMP << n) <= llc) ++n;  // first n: 16·2^n > llc
        return n;
    }();
    return cached;
}

bool sv_gate_is_fusable(const Instruction& inst, int max_qubit) {
    using GT = Instruction::GateType;
    if (inst.condition_clbit >= 0) return false;
    switch (inst.type) {
        case GT::MEASURE: case GT::RESET: case GT::BARRIER:
        case GT::MCX: case GT::MCP: case GT::PERMUTATION:
        case GT::PARAM_RX: case GT::PARAM_RY: case GT::PARAM_RZ:
        case GT::PARAM_P: case GT::PARAM_U:
            return false;
        default:
            return static_cast<int>(inst.qubits.size()) <= max_qubit;
    }
}

QuantumCircuit sv_fuse_circuit(StatevectorSimulator& sim,
                               const QuantumCircuit& circuit,
                               int max_qubit) {
    QuantumCircuit out(circuit.n_qubits, circuit.n_clbits);
    out.name = circuit.name;

    std::vector<int> support;               // global qubit of each local slot
    std::vector<Complex128> colM;           // block unitary, COLUMN-major
    std::vector<Instruction> members;       // originals, for 1-gate blocks

    auto flush = [&]() {
        if (members.empty()) return;
        if (members.size() == 1) {
            out.instructions.push_back(members.front());
        } else {
            const size_t d = size_t(1) << support.size();
            Instruction fused;
            fused.type = Instruction::GateType::UNITARY;
            fused.qubits = support;
            // The block is a product the library formed from matrices the
            // pre-flight already accepted, so it carries no caller declaration
            // to check. Its drift away from exact unitarity is the accumulated
            // rounding of that product, which is the library's own arithmetic.
            fused.validation = {Validation::Ignore, 0.0};
            std::vector<Complex128> rowM(d * d);
            for (size_t r = 0; r < d; ++r)          // row-major = transpose of
                for (size_t c = 0; c < d; ++c)      // the column-major store
                    rowM[r * d + c] = colM[c * d + r];
            fused.matrix = std::move(rowM);
            out.instructions.push_back(std::move(fused));
        }
        support.clear();
        colM.clear();
        members.clear();
    };

    for (const auto& inst : circuit.instructions) {
        if (!sv_gate_is_fusable(inst, max_qubit)) {
            flush();
            out.instructions.push_back(inst);
            continue;
        }

        // Local slots for this gate's qubits; grow the support (and embed the
        // existing block as identity on the new slots) as needed. LSB-first
        // throughout: local slot b of the block addresses support[b].
        std::vector<int> new_qubits;
        for (int q : inst.qubits) {
            if (std::find(support.begin(), support.end(), q) == support.end())
                new_qubits.push_back(q);
        }
        if (static_cast<int>(support.size() + new_qubits.size()) > max_qubit) {
            flush();
            new_qubits = inst.qubits;  // fresh block: all qubits are new
        }
        if (support.empty()) {
            colM.assign(1, Complex128(1.0, 0.0));  // 1x1 identity seed
        }
        for (int q : new_qubits) {
            // Tensor an identity slot at the TOP local index: the new
            // column-major matrix is block-diagonal with the old one
            // replicated on the new qubit's 0/1 branches.
            const size_t d_old = size_t(1) << support.size();
            const size_t d_new = d_old << 1;
            std::vector<Complex128> grown(d_new * d_new, Complex128(0.0, 0.0));
            for (size_t h = 0; h < 2; ++h)
                for (size_t c = 0; c < d_old; ++c)
                    for (size_t r = 0; r < d_old; ++r)
                        grown[(h * d_old + c) * d_new + (h * d_old + r)] =
                            colM[c * d_old + r];
            colM = std::move(grown);
            support.push_back(q);
        }

        // Compose the gate into the block: apply it (qubits remapped to
        // local slots) to every column of the block unitary through the
        // simulator's own dispatch, so all gate types compose exactly.
        Instruction local = inst;
        for (auto& q : local.qubits) {
            q = static_cast<int>(
                std::find(support.begin(), support.end(), q) - support.begin());
        }
        const int k = static_cast<int>(support.size());
        const size_t d = size_t(1) << k;
        Statevector col(k);
        for (size_t c = 0; c < d; ++c) {
            for (size_t r = 0; r < d; ++r) {
                col.real_parts[r] = colM[c * d + r].real;
                col.imag_parts[r] = colM[c * d + r].imag;
            }
            sim.apply_instruction(col, local, {Validation::Ignore});
            for (size_t r = 0; r < d; ++r)
                colM[c * d + r] = Complex128(col.real_parts[r],
                                             col.imag_parts[r]);
        }
        members.push_back(inst);
    }
    flush();
    return out;
}

}  // namespace

StatevectorSimulator::Result StatevectorSimulator::run(
    const QuantumCircuit& circuit_in,
    int shots,
    uint64_t seed,
    const RunPlan& plan
) {
    ScopedWarningFlush flush_on_exit;
    Result result;

    try {
        if (circuit_in.n_qubits < 1) {
            throw std::invalid_argument("Circuit must have at least 1 qubit");
        }
        // Pre-flight: reject any out-of-range operand index up front so the
        // failure surfaces through Result rather than reaching a kernel.
        circuit_in.validate_operands();
        // Under Repair::Attempt a repaired copy is executed and the caller's
        // circuit is left exactly as it was handed over; Repair::None binds
        // straight to it and nothing is copied. Same shape as the fused-circuit
        // swap below, and it runs first so fusion consumes repaired matrices.
        std::optional<QuantumCircuit> repaired_storage =
            circuit_in.validated_physical();
        const QuantumCircuit& circuit =
            repaired_storage ? *repaired_storage : circuit_in;
        if (options.fusion_max_qubit < 2 ||
            options.fusion_max_qubit > SV_FUSION_MAX_QUBIT_LIMIT) {
            throw std::invalid_argument(
                "Options::fusion_max_qubit must be in [2, 6]");
        }
        if (options.fusion_threshold < 0) {
            throw std::invalid_argument(
                "Options::fusion_threshold must be >= 0 (0 = auto)");
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // Reuse a thread-local working buffer to avoid repeated aligned alloc/free
        // on variational hot paths (VQE, QAOA) that call run() thousands of times.
        thread_local std::unique_ptr<Statevector> sv_work;
        if (!sv_work || sv_work->n_qubits != circuit.n_qubits) {
            sv_work = std::make_unique<Statevector>(circuit.n_qubits);
        }
        // Writes |0...0> for the default plan, so a reused buffer is cleared on
        // the same call that seeds a supplied initial state.
        detail::apply_initial_state(plan, *sv_work);
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

        // Gate fusion (R.1.17): at bandwidth-bound sizes, execute a fused
        // equivalent of the circuit (see sv_fuse_circuit above). The
        // engagement point is Options::fusion_threshold when set, else the
        // hardware-derived auto value (statevector > one LLC instance).
        // All three strategy paths below consume `exec`; the per-shot
        // trajectory path therefore builds the fused plan ONCE and reuses
        // it every shot. Strategy detection stays on the original circuit —
        // fusion preserves measures, conditions, and their ordering by
        // construction, so the classification is identical.
        QuantumCircuit fused_storage;
        const QuantumCircuit* exec = &circuit;
        // An observed run suppresses fusion by default, so that anchors keep
        // naming the instructions the caller wrote. Options::Fusion::Keep opts
        // back in and accepts that anchors then bind to the fused circuit.
        const bool observed = !plan.observations.empty();
        const bool fuse = options.fusion_enable &&
                          (!observed ||
                           plan.options.fusion == RunPlan::Options::Fusion::Keep);
        if (fuse) {
            const int fusion_min = options.fusion_threshold > 0
                ? options.fusion_threshold
                : sv_fusion_auto_threshold();
            if (circuit.n_qubits >= fusion_min) {
                fused_storage =
                    sv_fuse_circuit(*this, circuit, options.fusion_max_qubit);
                exec = &fused_storage;
            }
        }

        // Anchors resolve against the circuit that actually executes, which is
        // the fused one when fusion survived the check above. This is where an
        // anchor that cannot fire stops the run, before any state is touched: a
        // plan that does not match its circuit is a mistake in the caller's
        // code, not a capability the backend lacks, so no response knob softens
        // it.
        detail::ObservationRunner runner(plan, *exec, StateForm::Statevector);
        runner.set_bundle(&result.observations);

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
            runner.begin_run(circuit.n_qubits, shots);

            for (int shot = 0; shot < shots; ++shot) {
                detail::apply_initial_state(plan, *sv_work);
                clreg.assign(n_clbits, 0);
                runner.begin_shot(shot, clreg);
                sv_run_trajectory(*this, *sv_work, *exec, clreg, n_clbits,
                                  sv_sim_rng, runner.active() ? &runner : nullptr);

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
            // One evolution serves every shot here, because nothing before the
            // terminal measurements is stochastic, so the observers fire once
            // and that single firing describes all of them.
            const std::vector<int> no_clreg;
            const StateView view(StateForm::Statevector, sv_work.get(),
                                 sv_work->n_qubits);
            runner.begin_run(circuit.n_qubits, 1);
            runner.begin_shot(0, no_clreg);
            if (runner.active()) runner.at_start(view);

            int index = -1;
            for (const auto& inst : exec->instructions) {
                ++index;
                if (runner.active()) runner.before_instruction(index, inst, view);
                if (inst.type != Instruction::GateType::MEASURE &&
                    inst.type != Instruction::GateType::BARRIER) {
                    apply_instruction(*sv_work, inst, {Validation::Ignore});
                }
                if (runner.active()) runner.after_instruction(index, inst, view);
            }
            if (runner.active()) runner.at_end(view, index);

            std::vector<std::pair<int, int>> meas;  // (qubit, clbit)
            for (const auto& inst : exec->instructions)
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
            runner.begin_run(circuit.n_qubits, 1);
            runner.begin_shot(0, clreg);
            sv_run_trajectory(*this, *sv_work, *exec, clreg, n_clbits,
                              sv_sim_rng, runner.active() ? &runner : nullptr);
            if (shots > 0) {
                result.counts = sv_work->sample_counts(shots, seed);
            }
        }

        // Flushes every labelled observer into result.observations. Runs before
        // the timer stops, since collecting what was observed is part of the
        // work the run was asked to do.
        runner.end_run();

        auto t_end = std::chrono::high_resolution_clock::now();
        result.simulation_time_seconds =
            std::chrono::duration<double>(t_end - t_start).count();

        // Copy simulated state to result (callers own result.final_state;
        // sv_work stays in the cache for the next call).
        //
        // Ignore, because this is not a hand-over: these amplitudes are the
        // library's own output, and their norm carries whatever rounding the
        // circuit accumulated. Checking here would judge a circuit's arithmetic
        // against a caller's tolerance and could make run() throw on a
        // simulation that did nothing wrong.
        result.final_state = Statevector(circuit.n_qubits);
        result.final_state.set_amplitudes(sv_work->real_parts, sv_work->imag_parts,
                                          sv_work->dim, {Validation::Ignore});
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        // Observers write into the bundle as end_run walks them, so a failure
        // partway through that walk leaves entries from the ones ahead of it. A
        // caller checking the flag would be told the run failed while a caller
        // reading the bundle found real observations in it, and one of the two
        // would be acting on a run that did not happen.
        result.observations = ObservationBundle();
    }

    return result;
}

} // namespace lindblad
