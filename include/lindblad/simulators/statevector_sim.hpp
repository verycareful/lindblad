// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/circuit.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
namespace lindblad {
class SparsePauliOp;
class NoiseModel;
}

namespace lindblad {

// =============================================================================
// StatevectorSimulator
// =============================================================================

class StatevectorSimulator {
public:
    // What the fusion pre-pass weighs when it decides which gates to merge,
    // each in units of one single-qubit gate's pass over the state: a pass of
    // a dense k-qubit block (entry k - 1), of a diagonal block of any width,
    // and of one CZ, CP or CCZ on qubits at 3 or above, which touch a quarter
    // of the state's cache lines (the pre-pass doubles it for each of a
    // gate's qubits fewer up there, up to 1). Any other gate costs 1. All zero
    // means unset.
    struct FusionCosts {
        double dense[6] = {};
        double diagonal = 0.0;
        double controlled_phase = 0.0;

        bool unset() const noexcept {
            for (double c : dense)
                if (c != 0.0) return false;
            return diagonal == 0.0 && controlled_phase == 0.0;
        }
    };

    // How unset fusion costs are measured (see Options::fusion_costs).
    //   Hybrid      each dense block's arithmetic timed on the run's own
    //               threads, on a small state held in the cores' caches, and
    //               weighed against the gate passes, timed on those threads
    //               over the run's state. Under half a second, once per
    //               process.
    //   OneThread   every timing on one thread. Steady and quick, but with
    //               many threads on a state larger than the caches it rates
    //               dense blocks dearer than they are, so the plan fuses less
    //               than it could.
    //   RunThreads  every kernel timed on the run's own threads over the run's
    //               state, in longer stretches: a few seconds for the first
    //               fused run, and it varies from run to run where a state
    //               about the size of the caches is partly evicted.
    enum class FusionCalibration { Hybrid, OneThread, RunThreads };

    struct Options {
        // The most OpenMP threads any parallel region of run(),
        // simulate_circuit() or eval_expectation() may use; 0 leaves OpenMP's
        // own choice (OMP_NUM_THREADS, else every core). The caller's setting
        // is restored when the call returns. apply_instruction, a one-gate
        // primitive, runs under the caller's setting. Negative values are
        // refused.
        int max_parallel_threads = 0;
        // How many amplitudes of the state each thread must get before a
        // gate kernel runs in parallel; a gate then uses
        // min(threads available, 2^n / this) threads, and runs serially when
        // that is one. 0, the default, is DEFAULT_MIN_AMPLITUDES_PER_THREAD
        // (types.hpp). Larger values keep small registers serial; 1 threads
        // every gate of 2 or more amplitudes. Applies for the length of
        // run(), simulate_circuit() and eval_expectation().
        std::size_t min_amplitudes_per_thread = 0;
        // The most memory a run may use, in MiB (2^20 bytes). 0, the default,
        // is automatic: the memory this machine reports available, or
        // FALLBACK_MEMORY_CAP_MB when it gives no coherent reading.
        // NO_MEMORY_CAP means no cap. A run needs one state (the working
        // buffer, handed over as Result::final_state) and, when it samples,
        // the sampler's arrays: a run needing more is refused before anything
        // is allocated, and every copy an observer takes is charged against
        // it while the run goes on. The prefix snapshot a run sampled shot by
        // shot may take is taken only when it fits beside the state.
        uint64_t max_memory_mb = 0;
        // The widest register a run accepts: 30 qubits under Enforce, 59 under
        // Lift (see QubitLimit in types.hpp). Lifting never bypasses
        // max_memory_mb.
        QubitLimit qubit_limit = QubitLimit::Enforce;
        // Whether a per-shot run keeps a snapshot of the stretch before its
        // first MEASURE, RESET or conditioned instruction; see PrefixReuse in
        // types.hpp. The snapshot is released before the result's copy is
        // made, so here it never raises the two-state peak.
        PrefixReuse prefix_reuse = PrefixReuse::Hardware;
        int precision = 64;            // 32 or 64 bit
        bool zero_threshold = true;
        double threshold = 1e-10;

        // Gate fusion (R.1.17). Option names follow Qiskit Aer for API
        // parity; the default engagement point is hardware-derived rather
        // than Aer's flat qubit count (see docs/api/simulators.md).
        bool fusion_enable = true;  // master switch for the fusion pre-pass
        int fusion_threshold = 0;   // min qubits to engage fusion; 0 = auto:
                                    // first n whose statevector (16·2^n B)
                                    // exceeds one L3 instance (hw::llc_bytes;
                                    // 32 MiB assumed when undetectable)
        int fusion_max_qubit = 5;   // max fused-block width, 2..6 (Aer default 5)
        // Unset (the default): measured on this machine the first time a run
        // fuses, once per process for each thread count and state size. The
        // measurement follows the machine's speed, so two runs can choose
        // different fusions and differ in the last bits of their results.
        // Set (every entry the run uses positive), it fixes the fusion plan,
        // and a seeded run gives the same bits every time;
        // measured_fusion_costs gives the values to set.
        FusionCosts fusion_costs;
        // How unset fusion_costs are measured.
        FusionCalibration fusion_calibration = FusionCalibration::Hybrid;

        // Whether run() reports where its time went, in Result::phase_times
        // (PhaseTimes in types.hpp). Off reads no clock for it; on reads a few
        // per run, plus two per shot in a run sampled shot by shot.
        bool record_phase_times = false;
    };

    struct Result {
        Statevector final_state;
        std::unordered_map<std::string, int> counts;  // if measured
        std::vector<double> expectation_values;        // if observables requested
        double simulation_time_seconds = 0.0;
        // Set only when Options::record_phase_times asked for it.
        std::optional<PhaseTimes> phase_times;

        // Whatever the run's labelled observers collected. Empty unless the
        // RunPlan attached observers carrying labels.
        ObservationBundle observations;

        Result() : final_state(1) {}
        Result(Result&&) = default;
        Result& operator=(Result&&) = default;
    };

    Options options;

    StatevectorSimulator() = default;
    explicit StatevectorSimulator(const Options& opts) : options(opts) {}

    // The fusion costs a run on n_qubits would measure with the calling
    // thread's current thread limit and the given calibration: measured now
    // unless this process already measured them. Copy them into
    // Options::fusion_costs to fix the fusion plan of later runs.
    static FusionCosts measured_fusion_costs(
        int n_qubits, FusionCalibration calibration = FusionCalibration::Hybrid);

    // plan = the harness: where the run starts and what is watched while it
    // runs. Default constructed it starts at |0...0> and watches nothing, which
    // is the zero-overhead path. A plan carrying anchors suppresses gate
    // fusion, since an anchor names a position in the circuit the caller
    // handed over and fusion rewrites those positions away.
    Result run(
        const QuantumCircuit& circuit,
        int shots = 0,                              // 0 = no measurement sampling
        uint64_t seed = 0,
        const RunPlan& plan = {}
    );

    void simulate_circuit(
        Statevector& sv,
        const QuantumCircuit& circuit
    );

    // Run circuit into the thread-local working buffer and compute ⟨ψ|observable|ψ⟩
    // without allocating or populating Result::final_state. For the Estimator
    // ideal-path hot loop. Circuit must contain no MEASURE instructions.
    double eval_expectation(
        const QuantumCircuit& circuit,
        const SparsePauliOp& observable
    );

    // Dispatch one instruction. The two-argument form applies the
    // instruction's own ValidationOptions to its matrix, which is what a
    // caller driving instructions by hand wants. run() uses the three-argument
    // form with Validation::Ignore: its pre-flight has already checked every
    // matrix in the circuit once, and re-checking per shot would measure the
    // same unchanged matrix `shots` times.
    void apply_instruction(Statevector& sv, const Instruction& inst);
    void apply_instruction(Statevector& sv, const Instruction& inst,
                           ValidationOptions physical);
};

} // namespace lindblad
