// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.2 - max_parallel_threads caps the threads a call uses.
//
// StatevectorSimulator::Options::max_parallel_threads and
// LocalBackend::Config::max_parallel_threads were documented as OpenMP thread
// caps and read by nothing. They now hold for the length of one call: set on
// the calling thread when the call starts, the caller's own setting put back
// when it returns. An observer runs inside the call on the calling thread, so
// it reads the cap the call's parallel regions see. Each test sets its own
// baseline first, so the result does not depend on how many cores the machine
// has, and puts the process's setting back afterwards.

#include <gtest/gtest.h>

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/detail/thread_cap.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "v11311_helpers.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lindblad;

namespace {

QuantumCircuit small_circuit() {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2).measure_all();
    return qc;
}

#ifdef _OPENMP
class ThreadReading final : public Observer {
public:
    void observe(const ObservationContext&) override { seen.push_back(omp_get_max_threads()); }
    std::vector<int> seen;
};

// Sets a baseline for the test and restores the process's own on exit.
class Baseline {
public:
    explicit Baseline(int threads) : previous_(omp_get_max_threads()) {
        omp_set_num_threads(threads);
    }
    ~Baseline() { omp_set_num_threads(previous_); }

private:
    int previous_;
};
#endif

}  // namespace

#ifdef _OPENMP
TEST(V11302ThreadCap, TheCapHoldsInsideARunAndTheCallersSettingReturns) {
    const Baseline baseline(3);
    auto reading = std::make_shared<ThreadReading>();
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), reading);

    StatevectorSimulator capped;
    capped.options.max_parallel_threads = 2;
    ASSERT_NO_THROW((void)capped.run(small_circuit(), 16, 1, plan));
    ASSERT_FALSE(reading->seen.empty());
    for (int t : reading->seen) EXPECT_EQ(t, 2);
    EXPECT_EQ(omp_get_max_threads(), 3) << "the caller's setting was not put back";

    // 0 leaves the caller's setting in force inside the run.
    reading->seen.clear();
    StatevectorSimulator open;
    ASSERT_NO_THROW((void)open.run(small_circuit(), 16, 1, plan));
    for (int t : reading->seen) EXPECT_EQ(t, 3);
}

TEST(V11302ThreadCap, TheOtherEntryPointsPutTheCallersSettingBack) {
    const Baseline baseline(3);
    StatevectorSimulator sim;
    sim.options.max_parallel_threads = 2;

    QuantumCircuit gates(3);
    gates.h(0).cx(0, 1);
    Statevector sv(3);
    sim.simulate_circuit(sv, gates);
    EXPECT_EQ(omp_get_max_threads(), 3) << "simulate_circuit";

    const SparsePauliOp z(std::vector<PauliString>{PauliString("ZII")});
    sim.eval_expectation(gates, z);
    EXPECT_EQ(omp_get_max_threads(), 3) << "eval_expectation";

    backends::LocalBackend::Config cfg;
    cfg.max_parallel_threads = 2;
    backends::LocalBackend backend(cfg);
    EXPECT_NO_THROW((void)backend.run(small_circuit(), 16, 1));
    EXPECT_EQ(omp_get_max_threads(), 3) << "LocalBackend::run";
}

TEST(V11302ThreadCap, TheGuardRestoresEvenWhenTheCallThrows) {
    const Baseline baseline(3);
    try {
        const detail::ScopedThreadCap cap(2, "test");
        EXPECT_EQ(omp_get_max_threads(), 2);
        throw std::runtime_error("leaving early");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(omp_get_max_threads(), 3);
}
#endif

TEST(V11302ThreadCap, ANegativeCapIsRefused) {
    // Each refuses by its own name: the statevector run checks its option,
    // and LocalBackend checks its own before reaching any backend.
    const auto expect_refused = [](const std::string& entry, const std::function<void()>& run) {
        const std::uint64_t stores = detail::failed_run_stores();
        const auto e = v11311::thrown<InvalidArgument>(run);
        EXPECT_EQ(detail::failed_run_stores(), stores) << "a refusal left a failed-run record";
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(e->entry_point(), entry);
        EXPECT_EQ(std::string(e->what()),
                  entry + ": max_parallel_threads must be >= 0 (0 = the OpenMP default), got -1");
        EXPECT_FALSE(e->where().has_value());
    };
    StatevectorSimulator sim;
    sim.options.max_parallel_threads = -1;
    expect_refused("StatevectorSimulator::run", [&] { (void)sim.run(small_circuit(), 16, 1); });

    backends::LocalBackend::Config cfg;
    cfg.max_parallel_threads = -1;
    backends::LocalBackend backend(cfg);
    expect_refused("LocalBackend::run", [&] { (void)backend.run(small_circuit(), 16, 1); });
}
