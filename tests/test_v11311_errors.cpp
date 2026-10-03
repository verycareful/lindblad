// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - the typed errors, the message format, and the knob that decides a
// refusal by the run's phase.
//
// 1.1.31.0 made every exception Lindblad raises one of four types, each also
// its std counterpart, with a common mixin carrying the entry point, the
// position and the folder a failed run was saved to. It formats every message
// one way, adds a report request to a defect, and decides a knobbed refusal by
// whether any work would be lost: Response::Auto throws before the first gate
// and warns after it. None of that had a test of its own; this file is it.
//
// The phase tests use a caller-written observer that refuses through the same
// detail::refuse_observation every built-in observer uses, so the decision is
// the library's and the position of each refusal is chosen exactly: at_start
// of shot 0 is the one point before the first gate, and every other firing,
// at_start of a later shot included, is mid-run.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/detail/report.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/validation.hpp"
#include "v11311_helpers.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

constexpr const char* kReport =
    " This is a defect in Lindblad, not in your code. Please report it at "
    "https://github.com/verycareful/lindblad/issues with this message.";

FailurePoint point(int shot, int instruction, std::string gate, std::vector<int> qubits) {
    FailurePoint p;
    p.shot = shot;
    p.instruction = instruction;
    p.gate = std::move(gate);
    p.qubits = std::move(qubits);
    return p;
}

// Raises E through the reporting core and checks everything the error carries.
template <class E>
void expect_raised(const std::optional<FailurePoint>& where, const std::string& message) {
    const auto e = v11311::thrown<E>([&] { detail::raise<E>("Thing::call", "it broke", where); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(std::string(e->what()), message);
    EXPECT_EQ(e->entry_point(), "Thing::call");
    EXPECT_EQ(e->where().has_value(), where.has_value());
    if (where && e->where()) {
        EXPECT_EQ(e->where()->shot, where->shot);
        EXPECT_EQ(e->where()->instruction, where->instruction);
        EXPECT_EQ(e->where()->gate, where->gate);
        EXPECT_EQ(e->where()->qubits, where->qubits);
    }
    EXPECT_FALSE(e->saved_to().has_value());
}

// =============================================================================
// The phase, as a run decides it
// =============================================================================

struct Seen {
    int shot;
    int instruction;
    RunPhase phase;
};

// Records every firing and refuses the ones `refuse` picks, through the same
// call the built-in observers make.
class RefusingObserver : public Observer {
public:
    explicit RefusingObserver(std::function<bool(const ObservationContext&)> refuse)
        : refuse_(std::move(refuse)) {}

    void observe(const ObservationContext& ctx) override {
        seen.push_back({ctx.shot, ctx.instruction_index, ctx.phase});
        if (refuse_(ctx)) {
            ++refusals_returned;
            detail::refuse_observation(ctx, "RefusingObserver declines this firing.");
        }
    }

    std::vector<Seen> seen;
    int refusals_returned = 0;

private:
    std::function<bool(const ObservationContext&)> refuse_;
};

// A circuit every backend runs shot by shot: the X after the MEASURE acts on
// the measured qubit. Every gate is Clifford, so the tableau runs it too.
QuantumCircuit per_shot_circuit() {
    QuantumCircuit qc(2, 2);
    qc.h(0).measure(0, 0).x(0).cx(0, 1);
    return qc;
}

constexpr int kShots = 3;

struct Backend {
    const char* entry_point;
    std::function<void(const RunPlan&)> run;
};

std::vector<Backend> backends() {
    return {
        {"StatevectorSimulator::run",
         [](const RunPlan& plan) { StatevectorSimulator().run(per_shot_circuit(), kShots, 7, plan); }},
        {"DensityMatrixSimulator::run",
         [](const RunPlan& plan) {
             DensityMatrixSimulator().run(per_shot_circuit(), NoiseModel{}, kShots, 7, plan);
         }},
        {"MPSSimulator::run",
         [](const RunPlan& plan) { MPSSimulator().run(per_shot_circuit(), 8, kShots, 7, plan); }},
        {"CliffordSimulator::run",
         [](const RunPlan& plan) { CliffordSimulator().run(per_shot_circuit(), kShots, 7, plan); }},
    };
}

RunPlan plan_with(Anchor anchor, const std::shared_ptr<RefusingObserver>& obs, Response response,
                  const std::filesystem::path& dir = {}) {
    RunPlan plan;
    plan.options.response = response;
    plan.options.failed_run_dir = dir;
    plan.observations.observe(std::move(anchor), obs);
    return plan;
}

const std::string kRefusal = "RefusingObserver declines this firing.";

// How many deliveries are exactly `message`, and how many are its repeat tally.
std::pair<std::size_t, std::size_t> deliveries(v11311::WarningCapture& warnings,
                                               const std::string& message) {
    std::size_t first = 0, tallies = 0;
    for (const std::string& m : warnings.messages()) {
        if (m == message) ++first;
        if (m.rfind(message + " [repeated ", 0) == 0) ++tallies;
    }
    return {first, tallies};
}

}  // namespace

// =============================================================================
// The four types
// =============================================================================

TEST(V11311Errors, EachTypeIsItsStdCounterpartAndAnError) {
    static_assert(std::is_base_of_v<std::invalid_argument, InvalidArgument>);
    static_assert(std::is_base_of_v<std::out_of_range, OutOfRange>);
    static_assert(std::is_base_of_v<std::runtime_error, RuntimeFailure>);
    static_assert(std::is_base_of_v<std::logic_error, InternalError>);
    static_assert(std::is_base_of_v<Error, InvalidArgument>);
    static_assert(std::is_base_of_v<Error, OutOfRange>);
    static_assert(std::is_base_of_v<Error, RuntimeFailure>);
    static_assert(std::is_base_of_v<Error, InternalError>);
    // The mixin carries no std::exception of its own: a second one would make
    // catch (const std::exception&) ambiguous, and an ambiguous handler does
    // not match at all.
    static_assert(!std::is_base_of_v<std::exception, Error>);
    SUCCEED();
}

TEST(V11311Errors, AStdHandlerAndAnErrorHandlerBothCatchEveryType) {
    const auto through_std = [](const std::function<void()>& raise) {
        try {
            raise();
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string("not caught");
    };
    const auto through_error = [](const std::function<void()>& raise) {
        try {
            raise();
        } catch (const Error& e) {
            return e.entry_point();
        }
        return std::string("not caught");
    };
    const std::vector<std::function<void()>> raisers = {
        [] { detail::raise<InvalidArgument>("A::a", "one"); },
        [] { detail::raise<OutOfRange>("A::a", "one"); },
        [] { detail::raise<RuntimeFailure>("A::a", "one"); },
        [] { detail::raise_internal("A::a", "one"); },
    };
    for (std::size_t i = 0; i < raisers.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(through_std(raisers[i]).rfind("A::a: one", 0), 0u);
        EXPECT_EQ(through_error(raisers[i]), "A::a");
    }
}

// =============================================================================
// The message, one format everywhere
// =============================================================================

TEST(V11311Errors, TheMessageIsTheEntryPointTheTextAndThePosition) {
    expect_raised<InvalidArgument>(std::nullopt, "Thing::call: it broke");
    expect_raised<OutOfRange>(point(-1, 3, "rx", {2}),
                              "Thing::call: it broke (instruction 3: rx on qubit 2)");
    expect_raised<RuntimeFailure>(point(-1, 4, "cx", {0, 1}),
                                  "Thing::call: it broke (instruction 4: cx on qubits 0, 1)");
    expect_raised<RuntimeFailure>(point(7, 4, "cx", {1, 0}),
                                  "Thing::call: it broke (instruction 4: cx on qubits 1, 0) at shot 7");
    expect_raised<RuntimeFailure>(point(7, -1, "", {}), "Thing::call: it broke at shot 7");
    // An instruction known by index alone, and a position with no coordinate.
    expect_raised<RuntimeFailure>(point(-1, 2, "", {}), "Thing::call: it broke (instruction 2)");
    expect_raised<InvalidArgument>(point(-1, -1, "", {}), "Thing::call: it broke");
}

TEST(V11311Errors, AnEmptyEntryPointLeavesTheTextAlone) {
    EXPECT_EQ(detail::failure_message("", "already named", std::nullopt), "already named");
    EXPECT_EQ(detail::failure_message("", "already named", point(2, 1, "h", {0})),
              "already named (instruction 1: h on qubit 0) at shot 2");
    EXPECT_EQ(detail::failure_message("X::y", "z", std::nullopt), "X::y: z");
}

TEST(V11311Errors, ADefectAsksToBeReportedOnce) {
    for (const std::string text : {"the chain is not centred", "the chain is not centred."}) {
        SCOPED_TRACE(text);
        const auto e = v11311::thrown<InternalError>(
            [&] { detail::raise_internal("MPSState::sample", text, point(-1, 5, "cx", {0, 1})); });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(std::string(e->what()),
                  "MPSState::sample: the chain is not centred." + std::string(kReport) +
                      " (instruction 5: cx on qubits 0, 1)");
        EXPECT_EQ(e->entry_point(), "MPSState::sample");
        v11311::expect_point(e->where(), -1, 5, "cx", {0, 1});
    }
}

// =============================================================================
// respond: the one decision for a refusal that leaves the answer intact
// =============================================================================

TEST(V11311Respond, ResponseKeepsItsEnumeratorsAndAutoIsTheDefault) {
    // Auto was appended, so a value stored as an integer keeps its meaning.
    EXPECT_EQ(static_cast<int>(Response::Throw), 0);
    EXPECT_EQ(static_cast<int>(Response::Warn), 1);
    EXPECT_EQ(static_cast<int>(Response::Ignore), 2);
    EXPECT_EQ(static_cast<int>(Response::Auto), 3);
    EXPECT_EQ(RunPlan::Options{}.response, Response::Auto);
}

TEST(V11311Respond, ThrowRaisesInEveryPhase) {
    for (const RunPhase phase : {RunPhase::BeforeFirstGate, RunPhase::MidRun}) {
        v11311::WarningCapture warnings;
        const auto e = v11311::thrown<InvalidArgument>(
            [&] { detail::respond(Response::Throw, phase, "R::run", "no route"); });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(std::string(e->what()), "R::run: no route");
        EXPECT_EQ(e->entry_point(), "R::run");
        EXPECT_TRUE(warnings.messages().empty());
    }
}

TEST(V11311Respond, AutoThrowsBeforeTheFirstGateAndWarnsAfterIt) {
    {
        v11311::WarningCapture warnings;
        const auto e = v11311::thrown<InvalidArgument>([] {
            detail::respond(Response::Auto, RunPhase::BeforeFirstGate, "R::run", "no route",
                            point(-1, 2, "h", {1}));
        });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(std::string(e->what()), "R::run: no route (instruction 2: h on qubit 1)");
        EXPECT_TRUE(warnings.messages().empty());
    }
    {
        v11311::WarningCapture warnings;
        bool returned = true;
        EXPECT_NO_THROW(returned = detail::respond(Response::Auto, RunPhase::MidRun, "R::run",
                                                   "no route", point(4, 2, "h", {1})));
        EXPECT_FALSE(returned);
        ASSERT_EQ(warnings.messages().size(), 1u);
        EXPECT_EQ(warnings.messages()[0],
                  "note: R::run: no route (instruction 2: h on qubit 1) at shot 4. "
                  "The observation is omitted.");
    }
}

TEST(V11311Respond, WarnAndIgnoreNeverThrow) {
    for (const RunPhase phase : {RunPhase::BeforeFirstGate, RunPhase::MidRun}) {
        {
            v11311::WarningCapture warnings;
            bool returned = true;
            EXPECT_NO_THROW(returned = detail::respond(Response::Warn, phase, "R::run",
                                                       "no route ends here."));
            EXPECT_FALSE(returned);
            ASSERT_EQ(warnings.messages().size(), 1u);
            // A text ending in a full stop does not get a second.
            EXPECT_EQ(warnings.messages()[0],
                      "note: R::run: no route ends here. The observation is omitted.");
        }
        {
            v11311::WarningCapture warnings;
            bool returned = true;
            EXPECT_NO_THROW(returned = detail::respond(Response::Ignore, phase, "R::run", "x"));
            EXPECT_FALSE(returned);
            EXPECT_TRUE(warnings.messages().empty());
        }
    }
}

// =============================================================================
// Response::Auto on a real run, every backend
// =============================================================================

TEST(V11311AutoOnARun, OnlyTheStartOfTheFirstShotIsBeforeTheFirstGate) {
    // The phase every firing reports, on every backend: at_start of shot 0
    // precedes every instruction of every shot; nothing else does.
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.entry_point);
        auto obs = std::make_shared<RefusingObserver>([](const ObservationContext&) { return false; });
        RunPlan plan;
        plan.observations.observe(Anchor::at_start(), obs);
        plan.observations.observe(Anchor::every_instruction(), obs);
        ASSERT_NO_THROW(b.run(plan));
        ASSERT_FALSE(obs->seen.empty());
        for (const Seen& s : obs->seen) {
            const bool first = s.shot == 0 && s.instruction == -1;
            EXPECT_EQ(s.phase, first ? RunPhase::BeforeFirstGate : RunPhase::MidRun)
                << "shot " << s.shot << " instruction " << s.instruction;
        }
        // One at_start and four instructions in each of the three shots.
        EXPECT_EQ(obs->seen.size(), static_cast<std::size_t>(kShots * 5));
    }
}

TEST(V11311AutoOnARun, ARefusalBeforeTheFirstGateThrowsAndLeavesNoRecord) {
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.entry_point);
        auto obs = std::make_shared<RefusingObserver>([](const ObservationContext&) { return true; });
        const RunPlan plan = plan_with(Anchor::at_start(), obs, Response::Auto);
        const std::uint64_t stores = detail::failed_run_stores();
        const auto e = v11311::thrown<InvalidArgument>([&] { b.run(plan); });
        EXPECT_EQ(detail::failed_run_stores(), stores) << "nothing had run, so nothing is kept";
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(e->entry_point(), b.entry_point);
        EXPECT_EQ(std::string(e->what()), std::string(b.entry_point) + ": " + kRefusal);
        EXPECT_FALSE(e->saved_to().has_value());
        EXPECT_EQ(obs->seen.size(), 1u) << "the run went on past the refusal";
    }
}

TEST(V11311AutoOnARun, ARefusalMidRunWarnsOmitsAndTheRunCompletes) {
    // Two mid-run points: the start of a later shot, and after an instruction
    // of the first. Each refusal is delivered once and its repeats tallied.
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.entry_point);
        for (const bool at_later_start : {true, false}) {
            SCOPED_TRACE(at_later_start ? "at_start of shots 1 and 2" : "after instruction 0");
            auto obs = std::make_shared<RefusingObserver>([&](const ObservationContext& ctx) {
                return at_later_start ? ctx.shot >= 1 : true;
            });
            const RunPlan plan = plan_with(
                at_later_start ? Anchor::at_start() : Anchor::after_instruction(0), obs,
                Response::Auto);
            v11311::WarningCapture warnings;
            const std::uint64_t stores = detail::failed_run_stores();
            ASSERT_NO_THROW(b.run(plan));
            EXPECT_EQ(detail::failed_run_stores(), stores);
            EXPECT_EQ(obs->refusals_returned, at_later_start ? kShots - 1 : kShots);
            const std::string note = "note: " + std::string(b.entry_point) + ": " + kRefusal +
                                     " The observation is omitted.";
            const auto [first, tallies] = deliveries(warnings, note);
            EXPECT_EQ(first, 1u) << "delivered once, then tallied";
            EXPECT_EQ(tallies, 1u);
            EXPECT_EQ(warnings.messages().size(), 2u);
        }
    }
}

TEST(V11311AutoOnARun, ThrowMidRunFailsTheRunKeepingItsTypeAndItsWork) {
    // Under an explicit Throw a mid-run refusal ends the run. Work had begun,
    // so the run keeps a record and saves it, and the caller receives an
    // InvalidArgument still: the rethrown copy names the folder and the
    // position the run had reached, and is the same type.
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.entry_point);
        const v11311::TempDir dir("throw-mid-run");
        auto obs = std::make_shared<RefusingObserver>([](const ObservationContext&) { return true; });
        const RunPlan plan = plan_with(Anchor::after_instruction(1), obs, Response::Throw, dir.path());
        (void)take_failed_run();
        const auto e = v11311::thrown<InvalidArgument>([&] { b.run(plan); });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(e->entry_point(), b.entry_point);
        ASSERT_TRUE(e->saved_to().has_value());
        EXPECT_EQ(e->saved_to()->parent_path(), dir.path());
        EXPECT_EQ(std::string(e->what()),
                  std::string(b.entry_point) + ": RefusingObserver declines this firing "
                  "(instruction 1: measure on qubit 0) at shot 0. Partial results saved to " +
                      e->saved_to()->string() + ".");
        v11311::expect_point(e->where(), 0, 1, "measure", {0});

        const auto record = take_failed_run();
        ASSERT_TRUE(record.has_value());
        EXPECT_EQ(record->exception_type, "lindblad::InvalidArgument");
        EXPECT_EQ(record->saved_to, e->saved_to());
        EXPECT_EQ(record->entry_point, b.entry_point);
    }
}

TEST(V11311AutoOnARun, WarnAndIgnoreDecideTheSameInEveryPhase) {
    for (const Backend& b : backends()) {
        SCOPED_TRACE(b.entry_point);
        for (const Response response : {Response::Warn, Response::Ignore}) {
            SCOPED_TRACE(response == Response::Warn ? "Warn" : "Ignore");
            auto obs = std::make_shared<RefusingObserver>([](const ObservationContext&) { return true; });
            RunPlan plan = plan_with(Anchor::at_start(), obs, response);
            plan.observations.observe(Anchor::after_instruction(0), obs);
            v11311::WarningCapture warnings;
            ASSERT_NO_THROW(b.run(plan));
            EXPECT_EQ(obs->refusals_returned, kShots * 2);
            const std::string note = "note: " + std::string(b.entry_point) + ": " + kRefusal +
                                     " The observation is omitted.";
            const auto [first, tallies] = deliveries(warnings, note);
            EXPECT_EQ(first, response == Response::Warn ? 1u : 0u);
            EXPECT_EQ(tallies, response == Response::Warn ? 1u : 0u);
            EXPECT_EQ(warnings.messages().size(), response == Response::Warn ? 2u : 0u);
        }
    }
}

// =============================================================================
// A built-in observer's refusal names the run (M9)
// =============================================================================

TEST(V11311AutoOnARun, ABuiltInRefusalNamesTheRunThatAskedForIt) {
    // Each backend is asked for what it cannot produce, decided before the run:
    // bonds from a dense or tableau backend, a tableau from a chain.
    const auto bond_plan = [] {
        RunPlan plan;
        plan.observations.observe(Anchor::at_end(), std::make_shared<BondDimensionObserver>());
        return plan;
    };
    struct Case {
        const char* entry_point;
        std::string held;
        std::function<void(const RunPlan&)> run;
        RunPlan plan;
    };
    RunPlan tableau_plan;
    tableau_plan.observations.observe(Anchor::at_end(),
                                      std::make_shared<StateObserver>(StateForm::Stabilizer));
    const std::vector<Case> cases = {
        {"StatevectorSimulator::run", to_string(StateForm::Statevector),
         [](const RunPlan& p) { StatevectorSimulator().run(per_shot_circuit(), kShots, 7, p); },
         bond_plan()},
        {"DensityMatrixSimulator::run", to_string(StateForm::DensityMatrix),
         [](const RunPlan& p) {
             DensityMatrixSimulator().run(per_shot_circuit(), NoiseModel{}, kShots, 7, p);
         },
         bond_plan()},
        {"CliffordSimulator::run", to_string(StateForm::Stabilizer),
         [](const RunPlan& p) { CliffordSimulator().run(per_shot_circuit(), kShots, 7, p); },
         bond_plan()},
        {"MPSSimulator::run", "",
         [](const RunPlan& p) { MPSSimulator().run(per_shot_circuit(), 8, kShots, 7, p); },
         tableau_plan},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.entry_point);
        const std::uint64_t stores = detail::failed_run_stores();
        const auto e = v11311::thrown<InvalidArgument>([&] { c.run(c.plan); });
        EXPECT_EQ(detail::failed_run_stores(), stores);
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(e->entry_point(), c.entry_point);
        const std::string expected_text =
            c.held.empty()
                ? std::string(c.entry_point) + ": StateObserver asks for a " +
                      to_string(StateForm::Stabilizer) + " from a backend holding a " +
                      to_string(StateForm::MPS) + ", and no conversion between those exists at all."
                : std::string(c.entry_point) +
                      ": BondDimensionObserver asks for bond dimensions from a backend holding a " +
                      c.held + ", which has no bonds to report.";
        EXPECT_EQ(std::string(e->what()), expected_text);
    }
}
