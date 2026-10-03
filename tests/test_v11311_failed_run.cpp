// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - a run that fails after its first gate keeps what it computed.
//
// 1.1.31.0 gave every failed run a record: the finished shots' counts, the
// observations, the state as it stood, the circuit and noise model, the
// options, and where and why it failed. The record waits in the thread's slot
// and, unless saving is off, is written to a folder the exception names, from
// which load_failed_run reads it back. Not one test exercised any of it: no
// test in that release's suite failed a run after its first gate.
//
// This file pins the record and the slot on all four backends, the folder
// round trip byte for byte, and what a save does when it cannot write
// everything. The folder's defences against editing and crafting, the
// threads, and the batch primitive are in test_v11311_failed_run_io.cpp.
//
// The failure used throughout is the caller's own: an observer that throws in
// a chosen shot. It reaches every backend the same way, it fails after work
// has started, and the caller must receive it unchanged, which is the case
// with the least room for the library to dress the failure up.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/detail/failure_collector.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/types.hpp"
#include "v11311_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace lindblad;
namespace fs = std::filesystem;

namespace {

// =============================================================================
// The failure: the caller's observer throws in a chosen shot
// =============================================================================

struct CallerError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

constexpr const char* kCallerMessage = "the caller's observer fails";

class ThrowInShot : public Observer {
public:
    explicit ThrowInShot(int shot) : shot_(shot) {}
    void observe(const ObservationContext& ctx) override {
        if (ctx.shot == shot_) throw CallerError(kCallerMessage);
    }

private:
    int shot_;
};

class ThrowAtEndRun : public Observer {
public:
    void observe(const ObservationContext&) override {}
    void end_run() override { throw CallerError(kCallerMessage); }
};

// Writes one payload of every kind into the bundle at every firing, labelled
// by shot so no firing overwrites another. The values are the ones a text
// round trip most easily breaks: a value no decimal string holds exactly, a
// subnormal, a negative zero, infinities, the largest double, the extremes of
// int, and text that needs every escape.
class PayloadWriter : public Observer {
public:
    void observe(const ObservationContext& ctx) override {
        if (ctx.bundle == nullptr) return;
        const std::string s = std::to_string(ctx.shot);
        ctx.bundle->put("number" + s, 0.1 + ctx.shot);
        ctx.bundle->put("reals" + s,
                        std::vector<double>{std::numeric_limits<double>::denorm_min(), -0.0,
                                            std::numeric_limits<double>::infinity(),
                                            -std::numeric_limits<double>::infinity(),
                                            std::numeric_limits<double>::max(), 1.0 / 3.0});
        ctx.bundle->put("amplitudes" + s,
                        std::vector<Complex128>{Complex128(INV_SQRT2, -0.0), Complex128(-1e-300, PI)});
        ctx.bundle->put("integers" + s,
                        std::vector<int>{std::numeric_limits<int>::min(), -1, 0,
                                         std::numeric_limits<int>::max()});
        ctx.bundle->put("text" + s, std::string("quote \" back \\ tab \t line \n carriage \r pi π"));
    }
};

// A circuit every backend runs shot by shot (the X after the MEASURE acts on
// the measured qubit), all Clifford.
QuantumCircuit per_shot_circuit() {
    QuantumCircuit qc(2, 2);
    qc.h(0).measure(0, 0).x(0).cx(0, 1);
    return qc;
}

constexpr int kShots = 5;
constexpr int kFailingShot = 2;
constexpr std::uint64_t kSeed = 11;

NoiseModel small_noise() {
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.05), "h");
    noise.add_quantum_error(NoiseChannels::depolarizing(0.02, 2), "cx");
    return noise;
}

struct Backend {
    const char* entry_point;
    const char* name;  // as the record names it
    std::function<void(const QuantumCircuit&, int shots, std::uint64_t seed, const RunPlan&)> run;
};

std::vector<Backend> backends() {
    return {
        {"StatevectorSimulator::run", "statevector",
         [](const QuantumCircuit& qc, int shots, std::uint64_t seed, const RunPlan& plan) {
             (void)StatevectorSimulator().run(qc, shots, seed, plan);
         }},
        {"DensityMatrixSimulator::run", "density_matrix",
         [](const QuantumCircuit& qc, int shots, std::uint64_t seed, const RunPlan& plan) {
             (void)DensityMatrixSimulator().run(qc, small_noise(), shots, seed, plan);
         }},
        {"MPSSimulator::run", "mps",
         [](const QuantumCircuit& qc, int shots, std::uint64_t seed, const RunPlan& plan) {
             MPSSimulator sim;
             // Every setting away from its default, so the record and the
             // state file are shown to carry each one.
             sim.svd_method = SVDMethod::Jacobi;
             sim.svd_rejection = SvdRejection::Throw;
             sim.svd_accept_gram = true;
             sim.svd_report = SvdReport::Silent;
             sim.canonical_form = CanonicalForm::Auto;
             (void)sim.run(qc, 6, shots, seed, plan);
         }},
        {"CliffordSimulator::run", "clifford",
         [](const QuantumCircuit& qc, int shots, std::uint64_t seed, const RunPlan& plan) {
             (void)CliffordSimulator().run(qc, shots, seed, plan);
         }},
    };
}

// A plan that fails in kFailingShot after instruction 3, writes every payload
// kind at the start of each shot, and keeps a native state observation.
RunPlan failing_plan(const fs::path& dir, RunPlan::Options::SaveFailedRuns save) {
    RunPlan plan;
    plan.options.failed_run_dir = dir;
    plan.options.save_failed_runs = save;
    plan.observations.observe(Anchor::at_start(), std::make_shared<PayloadWriter>());
    plan.observations.observe(Anchor::after_instruction(1), std::make_shared<StateObserver>("state"));
    plan.observations.observe(Anchor::after_instruction(3), std::make_shared<ThrowInShot>(kFailingShot));
    return plan;
}

// Runs `b` into a failure and returns the record, after checking the caller
// received its own exception unchanged.
FailedRun fail(const Backend& b, const RunPlan& plan, std::uint64_t seed = kSeed) {
    (void)take_failed_run();
    const auto e = v11311::thrown<CallerError>([&] { b.run(per_shot_circuit(), kShots, seed, plan); });
    EXPECT_TRUE(e.has_value());
    if (e) EXPECT_STREQ(e->what(), kCallerMessage) << "the caller's exception was changed";
    auto record = take_failed_run();
    EXPECT_TRUE(record.has_value()) << "a failure after the first gate left no record";
    return record ? std::move(*record) : FailedRun{};
}

int total(const std::unordered_map<std::string, int>& counts) {
    int t = 0;
    for (const auto& [k, v] : counts) t += v;
    return t;
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// The state's bytes compared where the type exposes them, and by everything
// it determines where it does not (the tableau's storage is private; every
// Pauli expectation of two qubits pins the stabilizer state).
void expect_same_state(const FailedRun::State& a, const FailedRun::State& b) {
    ASSERT_EQ(a.index(), b.index());
    if (const auto* x = std::get_if<Statevector>(&a)) {
        const auto& y = std::get<Statevector>(b);
        ASSERT_EQ(x->dim, y.dim);
        EXPECT_EQ(std::memcmp(x->real_parts, y.real_parts, x->dim * sizeof(double)), 0);
        EXPECT_EQ(std::memcmp(x->imag_parts, y.imag_parts, x->dim * sizeof(double)), 0);
    } else if (const auto* x = std::get_if<DensityMatrix>(&a)) {
        const auto& y = std::get<DensityMatrix>(b);
        ASSERT_EQ(x->data.size(), y.data.size());
        EXPECT_EQ(std::memcmp(x->data.data(), y.data.data(), x->data.size() * sizeof(Complex128)), 0);
    } else if (const auto* x = std::get_if<MPSState>(&a)) {
        const auto& y = std::get<MPSState>(b);
        ASSERT_EQ(x->tensors().size(), y.tensors().size());
        for (std::size_t i = 0; i < x->tensors().size(); ++i) {
            const MPSTensor& s = x->tensors()[i];
            const MPSTensor& t = y.tensors()[i];
            EXPECT_EQ(s.bond_left, t.bond_left);
            EXPECT_EQ(s.bond_right, t.bond_right);
            ASSERT_EQ(s.data.size(), t.data.size());
            EXPECT_EQ(std::memcmp(s.data.data(), t.data.data(), s.data.size() * sizeof(Complex128)), 0);
        }
        EXPECT_EQ(x->max_bond_dim, y.max_bond_dim);
        EXPECT_TRUE(same_bits(x->cutoff, y.cutoff));
        EXPECT_EQ(x->svd_method, y.svd_method);
        EXPECT_EQ(x->svd_rejection, y.svd_rejection);
        EXPECT_EQ(x->svd_accept_gram, y.svd_accept_gram);
        EXPECT_EQ(x->svd_report, y.svd_report);
        EXPECT_EQ(x->canonical_form, y.canonical_form);
        EXPECT_EQ(x->unchecked_gates, y.unchecked_gates);
        EXPECT_EQ(x->qubit_limit, y.qubit_limit);
        EXPECT_EQ(x->open_span(), y.open_span());
        EXPECT_EQ(x->fidelity_estimate(), y.fidelity_estimate());
        EXPECT_EQ(x->fidelity_lower_bound(), y.fidelity_lower_bound());
        EXPECT_TRUE(same_bits(x->truncation_error(), y.truncation_error()));
        EXPECT_EQ(x->jacobi_rescue_count(), y.jacobi_rescue_count());
        EXPECT_EQ(x->gram_fallback_count(), y.gram_fallback_count());
        EXPECT_EQ(x->ignored_rejection_count(), y.ignored_rejection_count());
        EXPECT_TRUE(same_bits(x->floor_rejected_weight(), y.floor_rejected_weight()));
        EXPECT_EQ(x->svd_call_count(), y.svd_call_count());
        EXPECT_TRUE(same_bits(x->max_verify_residual_excess(), y.max_verify_residual_excess()));
    } else if (const auto* x = std::get_if<StabilizerState>(&a)) {
        const auto& y = std::get<StabilizerState>(b);
        ASSERT_EQ(x->n_qubits, y.n_qubits);
        const char paulis[] = {'I', 'X', 'Y', 'Z'};
        for (char p : paulis) {
            for (char q : paulis) {
                const std::string label{p, q};
                EXPECT_EQ(x->expectation_pauli(label), y.expectation_pauli(label)) << label;
            }
        }
    }
}

}  // namespace

// =============================================================================
// The record
// =============================================================================

TEST(V11311FailedRun, TheRecordHoldsTheFinishedShotsTheStateAndWhereItStopped) {
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        const v11311::TempDir dir("record");
        const FailedRun r = fail(b, failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::DoNotSave));

        EXPECT_EQ(r.library_version, LINDBLAD_VERSION_LABEL);
        EXPECT_EQ(r.entry_point, b.entry_point);
        EXPECT_EQ(r.backend, b.name);
        EXPECT_EQ(r.n_qubits, 2);
        EXPECT_EQ(r.shots_requested, kShots);
        EXPECT_EQ(r.shots_completed, kFailingShot);
        EXPECT_EQ(total(r.counts), kFailingShot) << "the counts are the finished shots' only";
        EXPECT_EQ(r.seed, kSeed);
        v11311::expect_point(r.where, kFailingShot, 3, "cx", {0, 1});
        EXPECT_NE(r.exception_type.find("CallerError"), std::string::npos) << r.exception_type;
        EXPECT_EQ(r.message, kCallerMessage);

        // Moved out of the run (its form is pinned per backend below).
        ASSERT_NE(r.state.index(), 0u) << "the record holds no state";
        ASSERT_TRUE(r.circuit.has_value());
        EXPECT_EQ(r.circuit->to_json(), per_shot_circuit().to_json());
        EXPECT_EQ(r.noise_model.has_value(), std::string(b.name) == "density_matrix");
        if (r.noise_model) EXPECT_EQ(r.noise_model->to_json(), small_noise().to_json());

        // The observations of the finished shots and of the failing one up to
        // its failure are kept: three shots' payloads and the state captures.
        EXPECT_TRUE(r.observations.contains("number0"));
        EXPECT_TRUE(r.observations.contains("number" + std::to_string(kFailingShot)));
        EXPECT_FALSE(r.observations.contains("number" + std::to_string(kFailingShot + 1)));
        EXPECT_FALSE(r.saved_to.has_value());
        EXPECT_EQ(r.save_note, "");
    }
}

TEST(V11311FailedRun, TheStateIsTheBackendsOwnForm) {
    const std::size_t forms[] = {1, 2, 3, 4};  // Statevector, DensityMatrix, MPSState, StabilizerState
    const auto list = backends();
    for (std::size_t i = 0; i < list.size(); ++i) {
        SCOPED_TRACE(list[i].name);
        const v11311::TempDir dir("form");
        const FailedRun r = fail(list[i], failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::DoNotSave));
        EXPECT_EQ(r.state.index(), forms[i]);
    }
}

TEST(V11311FailedRun, TheRecordNamesEveryOptionOfTheRunAndOfThePlan) {
    const std::vector<std::string> plan_names = {
        "plan.conversion", "plan.cost",           "plan.initial_cost",     "plan.response",
        "plan.fusion",     "plan.guard_multiple", "plan.save_failed_runs", "plan.failed_run_dir"};
    const std::vector<std::vector<std::string>> backend_names = {
        {"max_parallel_threads", "max_memory_mb", "qubit_limit", "prefix_reuse", "precision",
         "zero_threshold", "threshold", "fusion_enable", "fusion_threshold", "fusion_max_qubit"},
        {"max_memory_mb", "prefix_reuse"},
        {"max_bond_dim", "svd_method", "svd_rejection", "svd_accept_gram", "svd_report",
         "canonical_form", "unchecked_gates", "qubit_limit", "max_memory_mb"},
        {"sampling", "elimination", "max_memory_mb"},
    };
    const auto list = backends();
    for (std::size_t i = 0; i < list.size(); ++i) {
        SCOPED_TRACE(list[i].name);
        const v11311::TempDir dir("options");
        const FailedRun r = fail(list[i], failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::DoNotSave));
        std::vector<std::string> names;
        for (const auto& [name, value] : r.options) names.push_back(name);
        std::vector<std::string> expected = plan_names;
        expected.insert(expected.end(), backend_names[i].begin(), backend_names[i].end());
        std::sort(names.begin(), names.end());
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(names, expected);

        const auto value_of = [&](const std::string& name) {
            for (const auto& [n, v] : r.options)
                if (n == name) return v;
            return std::string("(absent)");
        };
        EXPECT_EQ(value_of("plan.response"), "\"Auto\"");
        EXPECT_EQ(value_of("plan.save_failed_runs"), "\"DoNotSave\"");
        if (std::string(list[i].name) == "mps") {
            // The factorisation settings, each away from its default.
            EXPECT_EQ(value_of("svd_method"), "\"Jacobi\"");
            EXPECT_EQ(value_of("svd_rejection"), "\"Throw\"");
            EXPECT_EQ(value_of("svd_accept_gram"), "true");
            EXPECT_EQ(value_of("svd_report"), "\"Silent\"");
            EXPECT_EQ(value_of("max_bond_dim"), "6");
        }
    }
}

TEST(V11311FailedRun, TheSeedRecordedIsTheOneUsedEvenWhenNoneWasGiven) {
    // A seed of 0 asks for a random one; the record holds the seed actually
    // drawn, and a rerun with it reproduces the finished shots exactly.
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        const v11311::TempDir dir("seed");
        const RunPlan plan = failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::DoNotSave);
        const FailedRun first = fail(b, plan, 0);
        EXPECT_NE(first.seed, 0u);
        const FailedRun again = fail(b, plan, first.seed);
        EXPECT_EQ(again.seed, first.seed);
        EXPECT_EQ(again.counts, first.counts);
    }
}

// =============================================================================
// The slot
// =============================================================================

TEST(V11311FailedRunSlot, TheNewestFailureWinsAndTakingEmptiesTheSlot) {
    const v11311::TempDir dir("slot");
    const Backend sv = backends()[0];
    RunPlan plan = failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::DoNotSave);
    (void)take_failed_run();
    EXPECT_THROW(sv.run(per_shot_circuit(), kShots, 1, plan), CallerError);
    EXPECT_THROW(sv.run(per_shot_circuit(), kShots, 2, plan), CallerError);
    auto record = take_failed_run();
    ASSERT_TRUE(record.has_value());
    EXPECT_EQ(record->seed, 2u) << "the older failure was not replaced";
    EXPECT_FALSE(take_failed_run().has_value()) << "taking left the record behind";
}

TEST(V11311FailedRunSlot, StartingARunDoesNotClearTheSlot) {
    const v11311::TempDir dir("slot-kept");
    const Backend sv = backends()[0];
    (void)take_failed_run();
    EXPECT_THROW(sv.run(per_shot_circuit(), kShots, 5,
                        failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::DoNotSave)),
                 CallerError);
    // A run that succeeds, and one refused before its first gate, leave it.
    ASSERT_NO_THROW(sv.run(per_shot_circuit(), kShots, 6, RunPlan{}));
    QuantumCircuit bad(1);
    bad.rx(std::numeric_limits<double>::quiet_NaN(), 0);
    EXPECT_THROW(sv.run(bad, 1, 7, RunPlan{}), InvalidArgument);
    const auto record = take_failed_run();
    ASSERT_TRUE(record.has_value());
    EXPECT_EQ(record->seed, 5u);
}

// =============================================================================
// Saving: what a save writes, where, and what it leaves in memory
// =============================================================================

TEST(V11311FailedRunSave, ASavedRunLeavesOnlyItsScalarsInMemory) {
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        const v11311::TempDir dir("saved");
        const FailedRun r = fail(b, failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::Save));
        ASSERT_TRUE(r.saved_to.has_value()) << r.save_note;
        EXPECT_EQ(r.save_note, "");
        // What is on disk leaves memory.
        EXPECT_TRUE(r.counts.empty());
        EXPECT_EQ(r.observations.size(), 0u);
        EXPECT_EQ(r.state.index(), 0u);
        EXPECT_FALSE(r.circuit.has_value());
        EXPECT_FALSE(r.noise_model.has_value());
        // The scalars stay, to say what happened and where to look.
        EXPECT_EQ(r.shots_completed, kFailingShot);
        v11311::expect_point(r.where, kFailingShot, 3, "cx", {0, 1});
    }
}

TEST(V11311FailedRunSave, TheFolderIsNamedUniquelyAndPrivate) {
    const v11311::TempDir dir("named");
    const FailedRun r = fail(backends()[0], failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::Save));
    ASSERT_TRUE(r.saved_to.has_value()) << r.save_note;
    EXPECT_EQ(r.saved_to->parent_path(), dir.path());
    // <YYYYmmdd-HHMMSS>-<pid>-<thread>-<counter>
    EXPECT_TRUE(std::regex_match(r.saved_to->filename().string(),
                                 std::regex(R"(\d{8}-\d{6}-\d+-[0-9a-f]+-\d+)")))
        << r.saved_to->filename();
    std::vector<std::string> expected = {"circuit.json",      "counts.json",       "manifest.crc32c",
                                         "manifest.json",     "observations",      "observations.json",
                                         "state.bin"};
    std::vector<std::string> present;
    for (const auto& entry : fs::directory_iterator(*r.saved_to)) present.push_back(entry.path().filename().string());
    std::sort(present.begin(), present.end());
    EXPECT_EQ(present, expected);
#if !defined(_WIN32)
    EXPECT_EQ(fs::status(*r.saved_to).permissions() & fs::perms::all, fs::perms::owner_all);
    for (const auto& entry : fs::recursive_directory_iterator(*r.saved_to)) {
        if (!entry.is_regular_file()) continue;
        EXPECT_EQ(entry.status().permissions() & fs::perms::all,
                  fs::perms::owner_read | fs::perms::owner_write)
            << entry.path();
    }
#endif
}

TEST(V11311FailedRunSave, AMissingParentIsCreated) {
    const v11311::TempDir dir("parent");
    const fs::path deep = dir.path() / "a" / "b" / "c";
    const FailedRun r = fail(backends()[0], failing_plan(deep, RunPlan::Options::SaveFailedRuns::Save));
    ASSERT_TRUE(r.saved_to.has_value()) << r.save_note;
    EXPECT_EQ(r.saved_to->parent_path(), deep);
}

TEST(V11311FailedRunSave, DoNotSaveWritesNothing) {
    const fs::path default_dir = v11311::default_failed_run_folder();
    const auto count = [&] {
        std::error_code ec;
        std::size_t n = 0;
        for (auto it = fs::directory_iterator(default_dir, ec); !ec && it != fs::directory_iterator(); ++it) ++n;
        return n;
    };
    const std::size_t before = count();
    const FailedRun r = fail(backends()[0], failing_plan({}, RunPlan::Options::SaveFailedRuns::DoNotSave));
    EXPECT_FALSE(r.saved_to.has_value());
    EXPECT_EQ(count(), before);
    // Everything stayed in memory instead.
    EXPECT_EQ(total(r.counts), kFailingShot);
    EXPECT_NE(r.state.index(), 0u);
    EXPECT_TRUE(r.circuit.has_value());
}

TEST(V11311FailedRunSave, WithNoFolderNamedTheDefaultFollowsTheEnvironment) {
#if defined(_WIN32)
    GTEST_SKIP() << "the POSIX state directory variables";
#else
    const v11311::TempDir home("home");
    {
        // XDG_STATE_HOME counts only when absolute; otherwise HOME decides.
        v11311::ScopedEnv xdg("XDG_STATE_HOME", std::string("relative/state"));
        v11311::ScopedEnv h("HOME", home.path().string());
        const FailedRun r = fail(backends()[0], failing_plan({}, RunPlan::Options::SaveFailedRuns::Save));
        ASSERT_TRUE(r.saved_to.has_value()) << r.save_note;
        EXPECT_EQ(r.saved_to->parent_path(),
                  home.path() / ".local" / "state" / "lindblad" / "failed-runs");
    }
    {
        // Neither set: nothing can be saved, the run still reports its own
        // failure, and the record keeps everything.
        v11311::ScopedEnv xdg("XDG_STATE_HOME", std::nullopt);
        v11311::ScopedEnv h("HOME", std::nullopt);
        const FailedRun r = fail(backends()[0], failing_plan({}, RunPlan::Options::SaveFailedRuns::Save));
        EXPECT_FALSE(r.saved_to.has_value());
        EXPECT_EQ(r.save_note,
                  "Nothing was saved: there is no folder to save into; set "
                  "RunPlan::Options::failed_run_dir, or HOME or XDG_STATE_HOME.");
        EXPECT_EQ(total(r.counts), kFailingShot);
        EXPECT_NE(r.state.index(), 0u);
    }
#endif
}

TEST(V11311FailedRunSave, AFolderThatCannotBeWrittenSavesNothingAndKeepsEverything) {
#if defined(_WIN32)
    GTEST_SKIP() << "POSIX permissions";
#else
    if (::geteuid() == 0) GTEST_SKIP() << "permissions do not bind the superuser";
    const v11311::TempDir dir("read-only");
    fs::permissions(dir.path(), fs::perms::owner_read | fs::perms::owner_exec, fs::perm_options::replace);
    const FailedRun r = fail(backends()[0], failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::Save));
    EXPECT_FALSE(r.saved_to.has_value());
    EXPECT_EQ(r.save_note.rfind("Nothing was saved: a folder cannot be created in " + dir.path().string(), 0), 0u)
        << r.save_note;
    EXPECT_EQ(total(r.counts), kFailingShot);
    EXPECT_NE(r.state.index(), 0u);
    EXPECT_TRUE(r.circuit.has_value());
    EXPECT_GT(r.observations.size(), 0u);
#endif
}

// =============================================================================
// A failed Lindblad run's exception names its folder
// =============================================================================

namespace {

// A normless state met at a MEASURE in shot 0 of a run walked shot by shot.
QuantumCircuit normless_trajectory(int n) {
    QuantumCircuit qc(n, n);
    qc.unitary(std::vector<Complex128>(4, Complex128(0.0, 0.0)), {0}, "zero", {Validation::Ignore});
    qc.measure(1, 1).x(1);
    return qc;
}

const std::string kNoNorm =
    "StatevectorSimulator::run: no norm to sample from; the state is zero or non-finite "
    "(instruction 1: measure on qubit 1) at shot 0";

}  // namespace

TEST(V11311FailedRunSave, TheRethrownCopyKeepsItsTypeAndNamesTheFolder) {
    const v11311::TempDir dir("copy");
    RunPlan plan;
    plan.options.failed_run_dir = dir.path();
    (void)take_failed_run();
    const auto e = v11311::thrown<RuntimeFailure>(
        [&] { (void)StatevectorSimulator().run(normless_trajectory(2), 3, 1, plan); });
    ASSERT_TRUE(e.has_value());
    ASSERT_TRUE(e->saved_to().has_value());
    EXPECT_EQ(e->saved_to()->parent_path(), dir.path());
    EXPECT_EQ(std::string(e->what()),
              kNoNorm + ". Partial results saved to " + e->saved_to()->string() + ".");
    v11311::expect_point(e->where(), 0, 1, "measure", {1});
    const auto record = take_failed_run();
    ASSERT_TRUE(record.has_value());
    EXPECT_EQ(record->saved_to, e->saved_to());
    EXPECT_EQ(record->exception_type, "lindblad::RuntimeFailure");
    // The record holds the exception as raised, before the folder was known.
    EXPECT_EQ(record->message,
              "StatevectorSimulator::run: no norm to sample from; the state is zero or non-finite");
}

TEST(V11311FailedRunSave, AFailureThatNamesNoPositionReachesTheCallerWithOne) {
    // The collapse that fails knows nothing of where the run was; the run
    // does, and the caller is told, with and without a save (M8).
    for (const auto save : {RunPlan::Options::SaveFailedRuns::Save, RunPlan::Options::SaveFailedRuns::DoNotSave}) {
        SCOPED_TRACE(save == RunPlan::Options::SaveFailedRuns::Save ? "saved" : "not saved");
        const v11311::TempDir dir("position");
        RunPlan plan;
        plan.options.failed_run_dir = dir.path();
        plan.options.save_failed_runs = save;
        (void)take_failed_run();
        const auto e = v11311::thrown<RuntimeFailure>(
            [&] { (void)StatevectorSimulator().run(normless_trajectory(2), 3, 1, plan); });
        ASSERT_TRUE(e.has_value());
        v11311::expect_point(e->where(), 0, 1, "measure", {1});
        if (save == RunPlan::Options::SaveFailedRuns::DoNotSave) {
            EXPECT_EQ(std::string(e->what()), kNoNorm);
            EXPECT_FALSE(e->saved_to().has_value());
        }
        const auto record = take_failed_run();
        ASSERT_TRUE(record.has_value());
        v11311::expect_point(record->where, 0, 1, "measure", {1});
    }
}

// =============================================================================
// The folder reads back exactly
// =============================================================================

TEST(V11311FailedRunLoad, ASavedRunLoadsBackAsTheRunLeftIt) {
    // Two identical runs, one kept in memory and one saved: what loads from
    // the folder is the record the first left in the slot.
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        const v11311::TempDir dir("load");
        const FailedRun kept = fail(b, failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::DoNotSave));
        const FailedRun saved = fail(b, failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::Save));
        ASSERT_TRUE(saved.saved_to.has_value()) << saved.save_note;
        const FailedRun loaded = load_failed_run(*saved.saved_to);

        EXPECT_EQ(loaded.saved_to, saved.saved_to);
        EXPECT_EQ(loaded.library_version, kept.library_version);
        EXPECT_EQ(loaded.entry_point, kept.entry_point);
        EXPECT_EQ(loaded.backend, kept.backend);
        EXPECT_EQ(loaded.n_qubits, kept.n_qubits);
        EXPECT_EQ(loaded.shots_requested, kept.shots_requested);
        EXPECT_EQ(loaded.shots_completed, kept.shots_completed);
        EXPECT_EQ(loaded.seed, kept.seed);
        v11311::expect_point(loaded.where, kFailingShot, 3, "cx", {0, 1});
        EXPECT_EQ(loaded.exception_type, kept.exception_type);
        EXPECT_EQ(loaded.message, kept.message);
        // The one option the two runs differ in is the save itself.
        auto expected_options = kept.options;
        for (auto& [name, value] : expected_options)
            if (name == "plan.save_failed_runs") value = "\"Save\"";
        EXPECT_EQ(loaded.options, expected_options);
        EXPECT_EQ(loaded.save_note, "");
        EXPECT_EQ(loaded.counts, kept.counts);
        ASSERT_TRUE(loaded.circuit.has_value());
        EXPECT_EQ(loaded.circuit->to_json(), kept.circuit->to_json());
        ASSERT_EQ(loaded.noise_model.has_value(), kept.noise_model.has_value());
        if (loaded.noise_model) EXPECT_EQ(loaded.noise_model->to_json(), kept.noise_model->to_json());
        expect_same_state(loaded.state, kept.state);

        ASSERT_EQ(loaded.observations.labels(), kept.observations.labels());
        for (const std::string& label : kept.observations.labels()) {
            SCOPED_TRACE(label);
            const auto& a = loaded.observations.payload(label);
            const auto& k = kept.observations.payload(label);
            ASSERT_EQ(a.index(), k.index());
            if (const auto* v = std::get_if<double>(&k)) {
                EXPECT_TRUE(same_bits(std::get<double>(a), *v));
            } else if (const auto* v = std::get_if<std::vector<double>>(&k)) {
                const auto& w = std::get<std::vector<double>>(a);
                ASSERT_EQ(w.size(), v->size());
                for (std::size_t i = 0; i < v->size(); ++i) EXPECT_TRUE(same_bits(w[i], (*v)[i])) << i;
            } else if (const auto* v = std::get_if<std::vector<Complex128>>(&k)) {
                const auto& w = std::get<std::vector<Complex128>>(a);
                ASSERT_EQ(w.size(), v->size());
                EXPECT_EQ(std::memcmp(w.data(), v->data(), v->size() * sizeof(Complex128)), 0);
            } else if (const auto* v = std::get_if<std::vector<int>>(&k)) {
                EXPECT_EQ(std::get<std::vector<int>>(a), *v);
            } else if (const auto* v = std::get_if<std::string>(&k)) {
                EXPECT_EQ(std::get<std::string>(a), *v);
            } else {
                EXPECT_EQ(loaded.observations.form(label), kept.observations.form(label));
            }
        }
    }
}

TEST(V11311FailedRunLoad, SavingWhatWasLoadedWritesTheSameFolderByteForByte) {
    // Every file the loader read, written again from what it returned, is the
    // file it read: nothing is lost or reformatted on the way through.
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        const v11311::TempDir dir("bytes");
        const FailedRun saved = fail(b, failing_plan(dir.path(), RunPlan::Options::SaveFailedRuns::Save));
        ASSERT_TRUE(saved.saved_to.has_value()) << saved.save_note;
        FailedRun loaded = load_failed_run(*saved.saved_to);
        const v11311::TempDir again("bytes-again");
        detail::save_failed_run(loaded, again.path());
        ASSERT_TRUE(loaded.saved_to.has_value()) << loaded.save_note;
        ASSERT_EQ(v11311::listing(*saved.saved_to), v11311::listing(*loaded.saved_to));
        for (const std::string& name : v11311::listing(*saved.saved_to)) {
            if (fs::is_directory(*saved.saved_to / name)) continue;
            EXPECT_EQ(v11311::read_file(*saved.saved_to / name), v11311::read_file(*loaded.saved_to / name))
                << name;
        }
    }
}

// =============================================================================
// A record whose final state is taken as the run ends (C1)
// =============================================================================

TEST(V11311FailedRunState, AnEndRunFailureKeepsTheFinalStateOnEveryBackend) {
    // The observer's end_run throws after the last instruction. The state the
    // run finished with is what the record and the folder hold: compared with
    // the same run completed without the failing observer.
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).s(1).h(1);
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        const v11311::TempDir dir("end-run");
        RunPlan failing;
        failing.options.failed_run_dir = dir.path();
        failing.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::DoNotSave;
        failing.observations.observe(Anchor::at_end(), std::make_shared<ThrowAtEndRun>());

        // The completed run's state, through a native state observation at
        // the end, so every backend hands it over the same way.
        auto state = std::make_shared<StateObserver>("final");
        RunPlan clean;
        clean.observations.observe(Anchor::at_end(), state);
        ASSERT_NO_THROW(b.run(qc, 0, kSeed, clean));

        (void)take_failed_run();
        const auto e = v11311::thrown<CallerError>([&] { b.run(qc, 0, kSeed, failing); });
        ASSERT_TRUE(e.has_value());
        auto record = take_failed_run();
        ASSERT_TRUE(record.has_value());
        ASSERT_NE(record->state.index(), 0u) << "the final state was lost";

        FailedRun::State expected;
        switch (record->state.index()) {
            case 1: expected = state->statevector().clone(); break;
            case 2: expected = DensityMatrix(state->density_matrix()); break;
            case 3: expected = MPSState(state->mps()); break;
            case 4: expected = StabilizerState(state->stabilizer()); break;
        }
        expect_same_state(record->state, expected);

        // And it saves to a folder that loads.
        failing.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::Save;
        EXPECT_THROW(b.run(qc, 0, kSeed, failing), CallerError);
        auto saved = take_failed_run();
        ASSERT_TRUE(saved.has_value());
        ASSERT_TRUE(saved->saved_to.has_value()) << saved->save_note;
        const FailedRun loaded = load_failed_run(*saved->saved_to);
        expect_same_state(loaded.state, expected);
    }
}
