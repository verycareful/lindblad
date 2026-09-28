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
    ASSERT_TRUE(capped.run(small_circuit(), 16, 1, plan).success);
    ASSERT_FALSE(reading->seen.empty());
    for (int t : reading->seen) EXPECT_EQ(t, 2);
    EXPECT_EQ(omp_get_max_threads(), 3) << "the caller's setting was not put back";

    // 0 leaves the caller's setting in force inside the run.
    reading->seen.clear();
    StatevectorSimulator open;
    ASSERT_TRUE(open.run(small_circuit(), 16, 1, plan).success);
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
    EXPECT_TRUE(backend.run(small_circuit(), 16, 1).success);
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
    StatevectorSimulator sim;
    sim.options.max_parallel_threads = -1;
    const auto r = sim.run(small_circuit(), 16, 1);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.error_message.rfind("StatevectorSimulator::run: ", 0), 0u) << r.error_message;
    EXPECT_NE(r.error_message.find("max_parallel_threads"), std::string::npos);

    backends::LocalBackend::Config cfg;
    cfg.max_parallel_threads = -1;
    backends::LocalBackend backend(cfg);
    EXPECT_FALSE(backend.run(small_circuit(), 16, 1).success);
}
