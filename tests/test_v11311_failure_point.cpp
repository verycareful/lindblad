// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - which shot a failure names, the same on every backend.
//
// A failure's position names the shot it happened in when the run walks the
// circuit shot by shot, and a shots == 0 run is one trajectory, shot 0, which
// is also what its observers are told. A walk that serves every shot at once
// (one forward pass, or the stretch before the first MEASURE that every shot
// shares) belongs to no single shot, and its failure names none (-1). The four
// backends must agree on each case: a caller handling a failure must not need
// to know which backend raised it.
//
// The failure is the caller's own observer throwing, which reaches every
// backend the same way; the shared prefix, which an observer would switch off,
// is failed instead by a two-qubit gate the MPS factorisation cannot split.
//
// KNOWN RED until 1.1.31.2: a shots == 0 run walked shot by shot names no shot
// on the statevector and MPS backends, where the density-matrix and Clifford
// backends name shot 0; and an MPS factorisation failure inside a run names an
// internal helper as its entry point instead of the run.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
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

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

struct CallerError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Throws at its anchor in the given shot.
class ThrowInShot : public Observer {
public:
    explicit ThrowInShot(int shot) : shot_(shot) {}
    void observe(const ObservationContext& ctx) override {
        seen_shot = ctx.shot;
        if (ctx.shot == shot_) throw CallerError("thrown by the caller's observer");
    }
    int seen_shot = -2;

private:
    int shot_;
};

struct Backend {
    const char* name;
    std::function<void(const QuantumCircuit&, int shots, const RunPlan&)> run;
};

std::vector<Backend> backends() {
    return {
        {"statevector", [](const QuantumCircuit& qc, int shots, const RunPlan& p) {
             (void)StatevectorSimulator().run(qc, shots, 5, p);
         }},
        {"density matrix", [](const QuantumCircuit& qc, int shots, const RunPlan& p) {
             (void)DensityMatrixSimulator().run(qc, NoiseModel{}, shots, 5, p);
         }},
        {"MPS", [](const QuantumCircuit& qc, int shots, const RunPlan& p) {
             (void)MPSSimulator().run(qc, 8, shots, 5, p);
         }},
        {"Clifford", [](const QuantumCircuit& qc, int shots, const RunPlan& p) {
             (void)CliffordSimulator().run(qc, shots, 5, p);
         }},
    };
}

// A circuit walked shot by shot: the X after the MEASURE acts on the measured
// qubit, so each shot's state depends on its own outcome.
QuantumCircuit per_shot() {
    QuantumCircuit qc(2, 2);
    qc.h(0).measure(0, 0).x(0).cx(0, 1);
    return qc;
}

// A circuit one forward pass serves: its only MEASUREs are at the end.
QuantumCircuit one_walk() {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).s(1).measure(0, 0).measure(1, 1);
    return qc;
}

// Fails `b` after `instruction` in `fail_in_shot` and checks the shot its
// observer was told (`told`) and the shot the record names (`expected_shot`).
void expect_failure_shot(const Backend& b, const QuantumCircuit& qc, int shots, int instruction,
                         int fail_in_shot, int told, int expected_shot) {
    const v11311::TempDir dir("shot");
    auto obs = std::make_shared<ThrowInShot>(fail_in_shot);
    RunPlan plan;
    plan.options.failed_run_dir = dir.path();
    plan.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::DoNotSave;
    plan.observations.observe(Anchor::after_instruction(instruction), obs);
    (void)take_failed_run();
    EXPECT_THROW(b.run(qc, shots, plan), CallerError);
    EXPECT_EQ(obs->seen_shot, told);
    const auto record = take_failed_run();
    ASSERT_TRUE(record.has_value());
    ASSERT_TRUE(record->where.has_value());
    EXPECT_EQ(record->where->instruction, instruction);
    EXPECT_EQ(record->where->shot, expected_shot);
}

}  // namespace

// Row A. KNOWN RED on the statevector and MPS backends.
TEST(V11311FailurePoint, AShotsZeroRunWalkedShotByShotFailsInShotZero) {
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        expect_failure_shot(b, per_shot(), 0, 2, 0, 0, 0);
    }
}

// Row B.
TEST(V11311FailurePoint, AShotsZeroRunOfOneWalkNamesNoShot) {
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        expect_failure_shot(b, one_walk(), 0, 1, 0, 0, -1);
    }
}

// Row C: a later shot, so the shot named is not 0 by accident.
TEST(V11311FailurePoint, ARunWalkedShotByShotNamesTheShotItFailedIn) {
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        expect_failure_shot(b, per_shot(), 3, 2, 1, 1, 1);
    }
}

// Row D.
TEST(V11311FailurePoint, OneWalkServingEveryShotNamesNoShot) {
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.name);
        expect_failure_shot(b, one_walk(), 3, 1, 0, 0, -1);
    }
}

namespace {

// A two-qubit UNITARY of NaN, let through by Validation::Ignore, which no rung
// of the MPS factorisation can split. It sits before the first MEASURE, in the
// stretch every shot shares, and the X after the MEASURE makes the rest of the
// run per shot.
QuantumCircuit unsplittable_prefix() {
    QuantumCircuit qc(2, 2);
    qc.unitary(std::vector<Complex128>(16, Complex128(std::numeric_limits<double>::quiet_NaN(), 0.0)),
               {0, 1}, "nan", {Validation::Ignore});
    qc.measure(1, 1).x(1);
    return qc;
}

// Runs the MPS backend into the factorisation failure and returns the record.
FailedRun fail_in_prefix(int shots, bool watched, std::optional<RuntimeFailure>& error) {
    const v11311::TempDir dir("prefix");
    RunPlan plan;
    plan.options.failed_run_dir = dir.path();
    plan.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::DoNotSave;
    // An observer, firing nowhere that matters, turns the shared prefix off.
    if (watched) plan.observations.observe(Anchor::after_instruction(2), std::make_shared<ThrowInShot>(-5));
    v11311::WarningCapture quiet;
    (void)take_failed_run();
    error = v11311::thrown<RuntimeFailure>([&] { (void)MPSSimulator().run(unsplittable_prefix(), 8, shots, 5, plan); });
    auto record = take_failed_run();
    EXPECT_TRUE(record.has_value());
    return record ? std::move(*record) : FailedRun{};
}

}  // namespace

// Row E: the same failure, in a prefix every shot shares and in one walked by
// a single shot.
TEST(V11311FailurePoint, AFailureInTheSharedPrefixNamesNoShotAndOneInAShotNamesIt) {
    std::optional<RuntimeFailure> e;
    {
        SCOPED_TRACE("four shots, the prefix shared");
        const FailedRun r = fail_in_prefix(4, false, e);
        ASSERT_TRUE(r.where.has_value());
        EXPECT_EQ(r.where->instruction, 0);
        EXPECT_EQ(r.where->shot, -1);
    }
    {
        SCOPED_TRACE("four shots, an observer attached, so each shot walks its own");
        const FailedRun r = fail_in_prefix(4, true, e);
        ASSERT_TRUE(r.where.has_value());
        EXPECT_EQ(r.where->instruction, 0);
        EXPECT_EQ(r.where->shot, 0);
    }
    {
        SCOPED_TRACE("one shot: no prefix to share");
        const FailedRun r = fail_in_prefix(1, false, e);
        ASSERT_TRUE(r.where.has_value());
        EXPECT_EQ(r.where->shot, 0);
    }
}

TEST(V11311FailurePoint, ACollapseAfterTheSharedPrefixNamesItsShotOnTheDenseBackends) {
    // Nothing in a dense prefix fails mid-walk; the first thing that can is
    // the collapse at the first MEASURE, which is the shot's own, whether or
    // not the prefix before it was shared.
    QuantumCircuit qc(2, 2);
    qc.unitary(std::vector<Complex128>(4, Complex128(0.0, 0.0)), {0}, "zero", {Validation::Ignore});
    qc.h(1).measure(0, 0).x(0);
    for (const PrefixReuse mode : {PrefixReuse::Off, PrefixReuse::Manual}) {
        SCOPED_TRACE(to_string(mode));
        const v11311::TempDir dir("collapse");
        RunPlan plan;
        plan.options.failed_run_dir = dir.path();
        plan.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::DoNotSave;
        {
            StatevectorSimulator sim;
            sim.options.prefix_reuse = mode;
            (void)take_failed_run();
            const auto e = v11311::thrown<RuntimeFailure>([&] { (void)sim.run(qc, 4, 1, plan); });
            ASSERT_TRUE(e.has_value());
            v11311::expect_point(e->where(), 0, 2, "measure", {0});
            (void)take_failed_run();
        }
        {
            DensityMatrixSimulator sim;
            sim.options.prefix_reuse = mode;
            const auto e = v11311::thrown<RuntimeFailure>([&] { (void)sim.run(qc, NoiseModel{}, 4, 1, plan); });
            ASSERT_TRUE(e.has_value());
            v11311::expect_point(e->where(), 0, 2, "measure", {0});
            (void)take_failed_run();
        }
    }
}

// =============================================================================
// The entry point of a factorisation failure inside a run
// =============================================================================

// KNOWN RED until 1.1.31.2. A split no rung can factorise ends the run with
// RuntimeFailure, but its entry point, and the start of its message, is the
// internal "MPS svd_truncate" rather than the run the caller called.
TEST(V11311FailurePoint, AFactorisationFailureInsideARunNamesTheRun) {
    std::optional<RuntimeFailure> e;
    const FailedRun r = fail_in_prefix(1, false, e);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "MPSSimulator::run");
    EXPECT_EQ(std::string(e->what()).rfind("MPSSimulator::run: ", 0), 0u) << e->what();
    EXPECT_EQ(r.entry_point, "MPSSimulator::run");
}
