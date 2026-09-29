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
#include "lindblad/noise.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {
namespace backends {

// =============================================================================
// BackendResult — unified result type
// =============================================================================

struct BackendResult {
    std::unordered_map<std::string, int> counts;
    double simulation_time_seconds = 0.0;
    // true and empty on every result a run returns: a run that fails
    // throws instead, and leaves what it had computed in a FailedRun
    // (failed_run.hpp).
    bool success = true;
    std::string error_message;
    std::string backend_name;
    int shots = 0;
};

// =============================================================================
// LocalBackend — wraps the simulators
// =============================================================================

class LocalBackend {
public:
    enum class SimType {
        STATEVECTOR,
        DENSITY_MATRIX,
        CLIFFORD,
        MPS,
        AUTO  // automatically pick best
    };

    struct Config {
        SimType simulator = SimType::AUTO;
        // The most OpenMP threads the run may use on whichever backend it
        // reaches; 0 leaves OpenMP's own choice. Restored on return.
        int max_parallel_threads = 0;
        // The memory cap, passed to whichever backend the run reaches; see
        // StatevectorSimulator::Options::max_memory_mb. 0 is automatic.
        uint64_t max_memory_mb = 0;
        int mps_bond_dim = 64;
        // Passed to the backend the run reaches: the statevector limit (30
        // qubits under Enforce, 59 under Lift) and the MPS dense-fallback limit
        // (25 and 59). See QubitLimit in types.hpp.
        QubitLimit qubit_limit = QubitLimit::Enforce;
    };

    Config config;
    NoiseModel noise_model;

    LocalBackend() = default;
    explicit LocalBackend(const Config& cfg) : config(cfg) {}

    BackendResult run(
        const QuantumCircuit& circuit,
        int shots = 1024,
        uint64_t seed = 0
    );

    std::vector<BackendResult> run_batch(
        const std::vector<QuantumCircuit>& circuits,
        int shots = 1024,
        uint64_t seed = 0
    );

    std::string name() const { return "lindblad_local_simulator"; }
    std::string version() const { return LINDBLAD_VERSION_LABEL; }
    // The widest register the statevector backend accepts under
    // config.qubit_limit.
    int max_qubits() const { return max_statevector_qubits(config.qubit_limit); }
};

} // namespace backends
} // namespace lindblad
