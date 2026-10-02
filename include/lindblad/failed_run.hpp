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
#include "lindblad/errors.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace lindblad {

// =============================================================================
// FailedRun - everything a run had computed when it failed
// =============================================================================
// A run that fails after its first gate leaves one of these behind: in this
// thread's slot (take_failed_run) and, unless saving is off, in a folder the
// exception names (load_failed_run). Once a part is written to disk it is
// released from memory, and the record in the slot keeps saved_to to find it;
// a part that could not be written stays here.
//
// A failure before the first gate leaves no record, since nothing has been
// computed: the exception alone says what to change.
struct FailedRun {
    using State = std::variant<std::monostate, Statevector, DensityMatrix,
                               MPSState, StabilizerState>;

    std::string library_version;
    std::string entry_point;          // e.g. "StatevectorSimulator::run"
    std::string backend;              // statevector | density_matrix | mps | clifford
    int n_qubits = 0;
    int shots_requested = 0;
    int shots_completed = 0;          // shots whose counts are in `counts`
    std::uint64_t seed = 0;           // the seed actually used, also when 0 was passed
    std::optional<FailurePoint> where;
    std::string exception_type;       // the dynamic type, demangled where possible
    std::string message;              // what(), as thrown
    std::vector<std::pair<std::string, std::string>> options;  // name, JSON value

    std::unordered_map<std::string, int> counts;   // finished shots only
    ObservationBundle observations;   // every observer flushed as the run failed
    State state;                      // moved out of the run, never copied
    std::optional<QuantumCircuit> circuit;         // the caller's circuit, as passed
    std::optional<NoiseModel> noise_model;         // density-matrix runs

    std::optional<std::filesystem::path> saved_to;
    std::string save_note;            // what was not saved and why; empty when all was
};

// This thread's most recent failed run, moved out, leaving the slot empty. A
// newer failure replaces an older one that was never taken; starting a run
// does not clear it.
std::optional<FailedRun> take_failed_run();

// Reads a saved folder back in full. Throws InvalidArgument naming the file
// when a file is missing, malformed, or fails its checksum.
FailedRun load_failed_run(const std::filesystem::path& folder);

}  // namespace lindblad
