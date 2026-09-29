// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/failed_run.hpp"
#include "lindblad/detail/failure_collector.hpp"
#include "lindblad/detail/json.hpp"
#include "lindblad/detail/report.hpp"

#include <cstdlib>
#include <exception>
#include <string>
#include <typeinfo>
#include <utility>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace lindblad {

// =============================================================================
// The slot - one failed run per thread, newest wins
// =============================================================================

namespace {

thread_local std::optional<FailedRun> slot;

// The dynamic type of an exception as a reader would write it:
// "lindblad::RuntimeFailure" rather than the mangled symbol.
std::string demangled_type_name(const std::type_info& type) {
#if defined(__GNUG__)
    int status = 0;
    char* name = abi::__cxa_demangle(type.name(), nullptr, nullptr, &status);
    if (status == 0 && name != nullptr) {
        std::string out(name);
        std::free(name);
        return out;
    }
    std::free(name);
#endif
    return type.name();
}

const char* name_of(Conversion value) {
    switch (value) {
        case Conversion::Convert: return "Convert";
        case Conversion::Never:   return "Never";
    }
    return "Conversion(?)";
}

const char* name_of(Cost value) {
    switch (value) {
        case Cost::Guarded:   return "Guarded";
        case Cost::Unlimited: return "Unlimited";
    }
    return "Cost(?)";
}

const char* name_of(Response value) {
    switch (value) {
        case Response::Throw:  return "Throw";
        case Response::Warn:   return "Warn";
        case Response::Ignore: return "Ignore";
        case Response::Auto:   return "Auto";
    }
    return "Response(?)";
}

const char* name_of(RunPlan::Options::Fusion value) {
    switch (value) {
        case RunPlan::Options::Fusion::Suppress: return "Suppress";
        case RunPlan::Options::Fusion::Keep:     return "Keep";
    }
    return "Fusion(?)";
}

const char* name_of(RunPlan::Options::SaveFailedRuns value) {
    switch (value) {
        case RunPlan::Options::SaveFailedRuns::Save:      return "Save";
        case RunPlan::Options::SaveFailedRuns::DoNotSave: return "DoNotSave";
    }
    return "SaveFailedRuns(?)";
}

}  // namespace

std::optional<FailedRun> take_failed_run() {
    std::optional<FailedRun> out = std::move(slot);
    slot.reset();
    return out;
}

namespace detail {

void store_failed_run(FailedRun&& record) noexcept {
    // Moving a FailedRun moves strings, maps and one state; none of those
    // moves allocates, so nothing here can throw.
    slot = std::move(record);
}

// =============================================================================
// Option values
// =============================================================================

std::string option_value(bool value) { return value ? "true" : "false"; }
std::string option_value(int value) { return std::to_string(value); }
std::string option_value(std::uint64_t value) { return std::to_string(value); }
std::string option_value(double value) { return json_number(value); }
std::string option_value(const std::filesystem::path& value) {
    return json_escape(value.string());
}
std::string option_value(QubitLimit value) {
    return json_escape(value == QubitLimit::Lift ? "Lift" : "Enforce");
}
std::string option_value(PrefixReuse value) { return json_escape(to_string(value)); }
std::string option_value(SVDMethod value) { return json_escape(to_string(value)); }
std::string option_value(CanonicalForm value) { return json_escape(to_string(value)); }
std::string option_value(UncheckedGates value) { return json_escape(to_string(value)); }

// =============================================================================
// FailureCollector
// =============================================================================

void FailureCollector::add_option(std::string name, std::string json_value) {
    options_.emplace_back(std::move(name), std::move(json_value));
}

void FailureCollector::note(std::string text) { notes_.push_back(std::move(text)); }

void FailureCollector::fail(std::exception_ptr failure,
                            std::unordered_map<std::string, int>&& counts,
                            ObservationBundle&& observations,
                            FailedRun::State&& state) {
    if (!work_started_) std::rethrow_exception(failure);

    std::optional<std::filesystem::path> saved;
    try {
        FailedRun r;
        r.library_version = LINDBLAD_VERSION_LABEL;
        r.entry_point = entry_point_;
        r.backend = backend_;
        r.n_qubits = circuit_->n_qubits;
        r.shots_requested = shots_;
        r.shots_completed = shots_completed_;
        r.seed = seed_;

        const RunPlan::Options& plan_options = plan_->options;
        r.options.emplace_back("plan.conversion", json_escape(name_of(plan_options.conversion)));
        r.options.emplace_back("plan.cost", json_escape(name_of(plan_options.cost)));
        r.options.emplace_back("plan.initial_cost",
                               json_escape(name_of(plan_options.initial_cost)));
        r.options.emplace_back("plan.response", json_escape(name_of(plan_options.response)));
        r.options.emplace_back("plan.fusion", json_escape(name_of(plan_options.fusion)));
        r.options.emplace_back("plan.guard_multiple", option_value(plan_options.guard_multiple));
        r.options.emplace_back("plan.save_failed_runs",
                               json_escape(name_of(plan_options.save_failed_runs)));
        r.options.emplace_back("plan.failed_run_dir", option_value(plan_options.failed_run_dir));
        for (auto& option : options_) r.options.push_back(std::move(option));

        r.counts = std::move(counts);
        r.observations = std::move(observations);
        r.state = std::move(state);
        r.circuit = *circuit_;
        if (noise_model_ != nullptr) r.noise_model = *noise_model_;

        // Where the run was: what the exception itself names, else the shot
        // and instruction the run had reached.
        FailurePoint reached;
        reached.shot = shot_;
        reached.instruction = instruction_;
        if (inst_ != nullptr) {
            reached.gate = inst_->gate_name();
            reached.qubits = inst_->qubits;
        }
        try {
            std::rethrow_exception(failure);
        } catch (const Error& e) {
            r.where = e.where() ? e.where() : std::optional<FailurePoint>(reached);
            r.exception_type = demangled_type_name(typeid(e));
            r.message = dynamic_cast<const std::exception&>(e).what();
        } catch (const std::exception& e) {
            r.where = reached;
            r.exception_type = demangled_type_name(typeid(e));
            r.message = e.what();
        } catch (...) {
            r.where = reached;
            r.exception_type = "(not derived from std::exception)";
        }

        for (auto& text : notes_) {
            if (!r.save_note.empty()) r.save_note += ' ';
            r.save_note += text;
        }

        if (plan_options.save_failed_runs == RunPlan::Options::SaveFailedRuns::Save) {
            save_failed_run(r, plan_options.failed_run_dir);
            saved = r.saved_to;
        }
        store_failed_run(std::move(r));
    } catch (...) {
        // Assembling allocates; under memory exhaustion it may not complete.
        // The run's own failure is what the caller must see.
    }

    if (saved) {
        try {
            std::rethrow_exception(failure);
        } catch (const Error& e) {
            FailurePathAccess::rethrow_saved(e, *saved);
        } catch (...) {
            // A foreign exception keeps its type and object; the folder is in
            // the record in the slot.
        }
    }
    std::rethrow_exception(failure);
}

// =============================================================================
// Saving
// =============================================================================

void save_failed_run(FailedRun&, const std::filesystem::path&) noexcept {}

}  // namespace detail
}  // namespace lindblad
