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
    struct Options {
        // The most OpenMP threads any parallel region of run(),
        // simulate_circuit() or eval_expectation() may use; 0 leaves OpenMP's
        // own choice (OMP_NUM_THREADS, else every core). The caller's setting
        // is restored when the call returns. apply_instruction, a one-gate
        // primitive, runs under the caller's setting. Negative values are
        // refused.
        int max_parallel_threads = 0;
        // The most memory the caller gives a run, in MiB (2^20 bytes); 0 = no
        // limit. It caps the states a run holds at once, which for this
        // simulator is always two (the working buffer and the copy returned in
        // Result::final_state): a run needing more is refused up front,
        // through Result, before anything is allocated.
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
    };

    struct Result {
        Statevector final_state;
        std::unordered_map<std::string, int> counts;  // if measured
        std::vector<double> expectation_values;        // if observables requested
        double simulation_time_seconds = 0.0;
        bool success = true;
        std::string error_message;

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
