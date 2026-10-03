// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// The suite's own state folder.
//
// A run that fails after its first gate saves a folder, by default under
// $XDG_STATE_HOME/lindblad/failed-runs (%LOCALAPPDATA% on Windows). The suite
// fails runs on purpose, so without this every run of it would leave folders
// in the state directory of whoever ran it. A global environment points that
// variable at a folder of the suite's own before any test runs, puts the old
// value back afterwards and removes the folder.
//
// The folder is created only when a save needs it, so a child process a death
// test starts, which never saves, leaves nothing behind.
//
// When the suite ends, no folder may still carry the .partial suffix: a save
// writes under that name and renames it once complete, and removes it when it
// cannot finish, so one left over is a save that broke its own contract.

#include <gtest/gtest.h>

#include "v11311_helpers.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace v11311 {

namespace {

#if defined(_WIN32)
constexpr const char* STATE_VARIABLE = "LOCALAPPDATA";
#else
constexpr const char* STATE_VARIABLE = "XDG_STATE_HOME";
#endif

fs::path& root_slot() {
    static fs::path root;
    return root;
}

class SuiteStateEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
#if defined(_WIN32)
        const long pid = static_cast<long>(_getpid());
#else
        const long pid = static_cast<long>(::getpid());
#endif
        root_slot() = fs::temp_directory_path() / ("lindblad-tests-" + std::to_string(pid));
        std::error_code ec;
        fs::remove_all(root_slot(), ec);
        env_.emplace(STATE_VARIABLE, (root_slot() / "state").string());
    }

    void TearDown() override {
        std::error_code ec;
        if (fs::exists(root_slot(), ec)) {
            for (const auto& entry : fs::recursive_directory_iterator(root_slot(), ec)) {
                const std::string name = entry.path().filename().string();
                if (name.size() > 8 && name.compare(name.size() - 8, 8, ".partial") == 0) {
                    ADD_FAILURE() << "a save left an incomplete folder behind: " << entry.path();
                }
            }
            fs::remove_all(root_slot(), ec);
        }
        env_.reset();
    }

private:
    std::optional<ScopedEnv> env_;
};

::testing::Environment* const suite_state_environment =
    ::testing::AddGlobalTestEnvironment(new SuiteStateEnvironment);

}  // namespace

fs::path suite_state_root() { return root_slot(); }

}  // namespace v11311

namespace {

using lindblad::Anchor;
using lindblad::CallbackObserver;
using lindblad::ObservationContext;
using lindblad::QuantumCircuit;
using lindblad::RunPlan;
using lindblad::StatevectorSimulator;

// The environment is what keeps the suite out of the user's state directory,
// so it is checked like anything else: a run that fails with no folder named
// saves under the suite's own root, and nowhere else.
TEST(V11311SuiteState, ADefaultSaveLandsInTheSuitesOwnFolder) {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1);
    qc.measure(0, 0);
    qc.measure(1, 1);
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<CallbackObserver>(
        [](const ObservationContext&) { throw std::runtime_error("the observer fails"); }));

    StatevectorSimulator sim;
    const auto e = v11311::thrown<std::runtime_error>([&] { sim.run(qc, 4, 11, plan); });
    ASSERT_TRUE(e.has_value());

    auto record = lindblad::take_failed_run();
    ASSERT_TRUE(record.has_value());
    ASSERT_TRUE(record->saved_to.has_value()) << record->save_note;
    EXPECT_EQ(record->saved_to->parent_path(), v11311::default_failed_run_folder());
    EXPECT_TRUE(std::filesystem::is_directory(*record->saved_to));
    std::filesystem::remove_all(*record->saved_to);
}

}  // namespace
