// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/failed_run.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lindblad {
namespace detail {

// Friend of MPSState and StabilizerState: reads and rebuilds their private
// storage for the state file.
struct StateFileAccess;

// =============================================================================
// FailureCollector - one run's failure path
// =============================================================================
// Constructed before a run's try block, holding pointers to what the caller
// passed and nothing else, so an ordinary run pays a few stores for it. The
// run reports where it is as it goes (at_instruction, set_shot, shot_done);
// on failure, fail() turns what exists into a FailedRun, saves it, puts it in
// this thread's slot and rethrows.
//
// A failure before the first instruction has executed leaves no record, since
// nothing has been computed, and is rethrown untouched.
class FailureCollector {
public:
    FailureCollector(const char* entry_point, const char* backend,
                     const QuantumCircuit& circuit, int shots,
                     const RunPlan& plan) noexcept
        : entry_point_(entry_point), backend_(backend), circuit_(&circuit),
          shots_(shots), plan_(&plan) {}

    FailureCollector(const FailureCollector&) = delete;
    FailureCollector& operator=(const FailureCollector&) = delete;

    void set_seed(std::uint64_t seed) noexcept { seed_ = seed; }
    // Work has started, although no instruction is being named: a prefix run
    // once for every shot, or a pass over a circuit's instructions.
    void begin_work() noexcept { work_started_ = true; }
    // The instruction about to execute. Marks work as started.
    void at_instruction(int index, const Instruction* inst) noexcept {
        work_started_ = true;
        instruction_ = index;
        inst_ = inst;
    }
    // A pass over the instructions has ended: what fails next (sampling, the
    // end-of-run checks) belongs to no instruction.
    void leave_instructions() noexcept {
        instruction_ = -1;
        inst_ = nullptr;
    }
    bool work_started() const noexcept { return work_started_; }
    void set_shot(int shot) noexcept {
        shot_ = shot;
        instruction_ = -1;
        inst_ = nullptr;
    }
    void shot_done() noexcept { ++shots_completed_; }

    // The noise model a density-matrix run used. Held by pointer and copied
    // only if the run fails.
    void set_noise_model(const NoiseModel& model) noexcept { noise_model_ = &model; }

    // Recorded only on the failure path, so building the JSON costs nothing
    // on a run that succeeds. The run plan's own options are added by fail().
    void add_option(std::string name, std::string json_value);
    void note(std::string text);

    // Assembles the record (moving `counts`, `observations` and `state` into
    // it), saves it unless the plan's options say not to, stores it in this
    // thread's slot and rethrows `failure`. A lindblad::Error is rethrown as a
    // copy that names the saved folder and what it does not hold, and the
    // shot and instruction the run had reached when the original named no
    // position; anything else is rethrown unchanged. Nothing here replaces
    // the run's own failure: a failure to allocate while assembling leaves a
    // minimal record of the moved parts, and one while amending rethrows the
    // original.
    [[noreturn]] void fail(std::exception_ptr failure,
                           std::unordered_map<std::string, int>&& counts,
                           ObservationBundle&& observations,
                           FailedRun::State&& state);

private:
    const char* entry_point_;
    const char* backend_;
    const QuantumCircuit* circuit_;
    int shots_;
    const RunPlan* plan_;
    std::uint64_t seed_ = 0;
    bool work_started_ = false;
    int shot_ = -1;
    int instruction_ = -1;
    const Instruction* inst_ = nullptr;
    int shots_completed_ = 0;
    const NoiseModel* noise_model_ = nullptr;
    std::vector<std::pair<std::string, std::string>> options_;
    std::vector<std::string> notes_;
};

// Moves `record` into this thread's slot, replacing any record there. Can
// throw std::bad_alloc where moving a map allocates.
void store_failed_run(FailedRun&& record);

// Puts back into this thread's slot a record taken from it, as if it had never
// left: unlike store_failed_run, it records no new failure, so
// failed_run_stores() is unchanged. Can throw std::bad_alloc as
// store_failed_run can.
void restore_failed_run(FailedRun&& record);

// How many records have been stored on this thread. A caller that reads it
// before a run and again after the run throws knows whether that failure left
// a record, or whether the slot still holds an older one: a failure before the
// first gate stores nothing.
std::uint64_t failed_run_stores() noexcept;

// Writes `record` into a new folder under `dir` (the default location when
// empty), sets saved_to, and releases from memory every part it wrote. Never
// throws: what it could not write is left in memory and described in
// save_note.
void save_failed_run(FailedRun& record, const std::filesystem::path& dir) noexcept;

// CRC-32C (the Castagnoli polynomial) of `n` bytes, continuing from `crc`
// (0 to start). Every file a failed run saves carries one in its manifest, and
// load_failed_run refuses a file whose bytes no longer match it. Computed with
// the SSE4.2 instruction where the build targets it, and a table elsewhere;
// both give the same value.
std::uint32_t crc32c(std::uint32_t crc, const void* data, std::size_t n) noexcept;

// JSON values for the option record of a failed run.
std::string option_value(bool value);
std::string option_value(int value);
std::string option_value(std::uint64_t value);
std::string option_value(double value);
std::string option_value(const std::filesystem::path& value);
std::string option_value(QubitLimit value);
std::string option_value(PrefixReuse value);
std::string option_value(SVDMethod value);
std::string option_value(SvdRejection value);
std::string option_value(SvdReport value);
std::string option_value(CanonicalForm value);
std::string option_value(UncheckedGates value);

}  // namespace detail
}  // namespace lindblad
