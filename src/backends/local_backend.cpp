// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/detail/report.hpp"
#include "lindblad/detail/thread_cap.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

namespace lindblad {
namespace backends {

BackendResult LocalBackend::run(
    const QuantumCircuit& circuit,
    int shots,
    uint64_t seed
) {
    BackendResult result;
    result.backend_name = name();
    result.shots = shots;

    // No catch here: the backend a run reaches throws on failure with the
    // failed-run record in this thread's slot, and converting that into a flag
    // would hand the caller a result to read as an answer.

    // Every backend the run reaches works under Config::
    // max_parallel_threads; the caller's own setting is back on return.
    const detail::ScopedThreadCap threads(config.max_parallel_threads,
                                          "LocalBackend::run");
    // Auto-select simulator
    SimType sim_type = config.simulator;
    if (sim_type == SimType::AUTO) {
        if (!noise_model.is_ideal()) {
            sim_type = SimType::DENSITY_MATRIX;
        } else if (CliffordSimulator::is_clifford(circuit)) {
            sim_type = SimType::CLIFFORD;
        } else if (circuit.n_qubits > 20) {
            sim_type = SimType::MPS;
        } else {
            sim_type = SimType::STATEVECTOR;
        }
    }

    switch (sim_type) {
        case SimType::STATEVECTOR: {
            StatevectorSimulator sim;
            sim.options.max_memory_mb = config.max_memory_mb;
            sim.options.qubit_limit = config.qubit_limit;
            auto sv_result = sim.run(circuit, shots, seed);
            result.counts = std::move(sv_result.counts);
            result.simulation_time_seconds = sv_result.simulation_time_seconds;
            break;
        }
        case SimType::DENSITY_MATRIX: {
            DensityMatrixSimulator sim;
            sim.options.max_memory_mb = config.max_memory_mb;
            auto dm_result = sim.run(circuit, noise_model, shots, seed);
            result.counts = std::move(dm_result.counts);
            result.simulation_time_seconds = dm_result.simulation_time_seconds;
            break;
        }
        case SimType::CLIFFORD: {
            CliffordSimulator sim;
            sim.options.max_memory_mb = config.max_memory_mb;
            auto cliff_result = sim.run(circuit, shots, seed);
            result.counts = std::move(cliff_result.counts);
            break;
        }
        case SimType::MPS: {
            MPSSimulator sim;
            sim.qubit_limit = config.qubit_limit;
            sim.max_memory_mb = config.max_memory_mb;
            auto mps_result = sim.run(circuit, config.mps_bond_dim, shots, seed);
            result.counts = std::move(mps_result.counts);
            result.simulation_time_seconds = mps_result.simulation_time_seconds;
            break;
        }
        default:
            detail::raise_internal("LocalBackend::run",
                                   "the dispatcher reached an unknown simulator type");
    }

    result.success = true;
    return result;
}

std::vector<BackendResult> LocalBackend::run_batch(
    const std::vector<QuantumCircuit>& circuits,
    int shots,
    uint64_t seed
) {
    std::vector<BackendResult> results;
    results.reserve(circuits.size());

    for (const auto& circuit : circuits) {
        results.push_back(run(circuit, shots, seed));
    }

    return results;
}

} // namespace backends
} // namespace lindblad
