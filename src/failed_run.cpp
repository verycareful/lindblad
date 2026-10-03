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
#include <vector>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace lindblad {

// =============================================================================
// The slot - one failed run per thread, newest wins
// =============================================================================

namespace {

thread_local std::optional<FailedRun> slot;
thread_local std::uint64_t stores = 0;  // records stored on this thread

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

// What a saved folder does not hold, as two sentences for the rethrown message,
// or empty when it holds everything: "Not saved: the state, the observations.
// Take from memory with lindblad::take_failed_run()." The parts are listed
// in the order the record keeps them, with no verb or pronoun whose number
// would have to agree with theirs. The saver releases every part it writes,
// so after a save the parts still in the record are exactly the ones it could
// not write; save_note says why.
std::string unsaved_parts(const FailedRun& r) {
    std::vector<std::string> parts;
    if (r.state.index() != 0) parts.emplace_back("the state");
    if (r.observations.size() != 0) parts.emplace_back("the observations");
    if (!r.counts.empty()) parts.emplace_back("the counts");
    if (r.circuit) parts.emplace_back("the circuit");
    if (r.noise_model) parts.emplace_back("the noise model");
    if (parts.empty()) return {};

    std::string out = "Not saved: ";
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += ", ";
        out += parts[i];
    }
    out += ". Take from memory with lindblad::take_failed_run().";
    return out;
}

}  // namespace

std::optional<FailedRun> take_failed_run() {
    std::optional<FailedRun> out = std::move(slot);
    slot.reset();
    return out;
}

namespace detail {

void store_failed_run(FailedRun&& record) {
    // Moving a FailedRun moves strings, maps and one state. A standard library
    // may allocate to move a map, so this can throw, and every caller holds an
    // exception of its own that must not be replaced: each calls it inside a
    // try.
    slot = std::move(record);
    ++stores;
}

void restore_failed_run(FailedRun&& record) { slot = std::move(record); }

std::uint64_t failed_run_stores() noexcept { return stores; }

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
std::string option_value(SvdRejection value) { return json_escape(to_string(value)); }
std::string option_value(SvdReport value) { return json_escape(to_string(value)); }
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
    std::string unsaved;
    std::optional<FailurePoint> reached;  // the shot and instruction the run had reached
    bool parts_moved = false;
    bool stored = false;
    try {
        FailurePoint point;
        point.shot = shot_;
        point.instruction = instruction_;
        if (inst_ != nullptr) {
            point.gate = inst_->gate_name();
            point.qubits = inst_->qubits;
        }
        reached = std::move(point);

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

        // The copies allocate, so they come before the parts that are moved
        // in: an allocation that fails here leaves those parts where they
        // are, for the minimal record below.
        r.circuit = *circuit_;
        if (noise_model_ != nullptr) r.noise_model = *noise_model_;

        // Where the run was: what the exception itself names, else the shot
        // and instruction the run had reached.
        try {
            std::rethrow_exception(failure);
        } catch (const Error& e) {
            r.where = e.where() ? e.where() : reached;
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

        r.counts = std::move(counts);
        r.observations = std::move(observations);
        r.state = std::move(state);
        parts_moved = true;

        if (plan_options.save_failed_runs == RunPlan::Options::SaveFailedRuns::Save) {
            save_failed_run(r, plan_options.failed_run_dir);
            saved = r.saved_to;
            if (saved) {
                try {
                    unsaved = unsaved_parts(r);
                } catch (...) {
                    // The sentence is a convenience; save_note in the record
                    // already says what was not written.
                }
            }
        }
        store_failed_run(std::move(r));
        stored = true;
    } catch (...) {
        // Assembling allocates; under memory exhaustion it may not complete.
        // The run's own failure is what the caller must see.
    }

    // A record that could not be assembled still keeps what the run computed,
    // in a minimal record built from moves and plain values alone.
    if (!stored && !parts_moved) {
        try {
            FailedRun minimal;
            minimal.n_qubits = circuit_->n_qubits;
            minimal.shots_requested = shots_;
            minimal.shots_completed = shots_completed_;
            minimal.seed = seed_;
            minimal.counts = std::move(counts);
            minimal.observations = std::move(observations);
            minimal.state = std::move(state);
            store_failed_run(std::move(minimal));
        } catch (...) {
            // Nothing more can be kept.
        }
    }

    // A Lindblad exception is rethrown as a copy when there is something to
    // add to it: the folder a save wrote, or the position it did not name.
    try {
        std::rethrow_exception(failure);
    } catch (const Error& e) {
        Amendment amendment;
        bool amend = false;
        try {
            if (saved) {
                amendment.saved_to = saved;
                amendment.unsaved = unsaved;
                amend = true;
            }
            if (!e.where() && reached && (reached->instruction >= 0 || reached->shot >= 0)) {
                amendment.where = reached;
                amendment.position = failure_message("", "", reached);
                amend = true;
            }
        } catch (...) {
            amend = false;
        }
        if (amend) {
            try {
                FailurePathAccess::rethrow_amended(e, amendment);
            } catch (const Error&) {
                throw;
            } catch (...) {
                // Building the copy allocates; under memory exhaustion the
                // original is what the caller sees.
            }
        }
    } catch (...) {
        // A foreign exception keeps its type and object; the record in the
        // slot holds the folder and the position.
    }
    std::rethrow_exception(failure);
}

}  // namespace detail
}  // namespace lindblad
