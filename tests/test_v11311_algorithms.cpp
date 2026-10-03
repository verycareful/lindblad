// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - an algorithm whose run is refused throws the refusal.
//
// Before 1.1.31.0 a run could fail and still return a result, and every
// algorithm built on the runs read that result as an answer: a register too
// wide to simulate gave Deutsch-Jozsa a verdict, Bernstein-Vazirani a secret
// and Shor a failed attempt, all computed from a run that never happened.
// MA-QAOA's noisy path scored such a run as a very poor energy and kept
// optimising. Every run now throws instead, and the algorithms let it through
// unchanged. Each algorithm is driven here one qubit past the statevector
// limit, where the run's refusal is the only thing that may come back.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/types.hpp"
#include "v11311_helpers.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

namespace {

constexpr int kOver = ENFORCED_MAX_QUBITS + 1;

// The statevector run's refusal of an n-qubit circuit, which is all the
// algorithm may let through: before the first gate, so with no record.
void expect_over_the_limit(int n, const std::function<void()>& solve) {
    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<InvalidArgument>(solve);
    EXPECT_EQ(detail::failed_run_stores(), stores);
    ASSERT_TRUE(e.has_value()) << "the algorithm returned an answer from a refused run";
    EXPECT_EQ(e->entry_point(), "StatevectorSimulator::run");
    EXPECT_EQ(std::string(e->what()),
              "StatevectorSimulator::run: the circuit has " + std::to_string(n) +
                  " qubits, over the statevector limit of " + std::to_string(ENFORCED_MAX_QUBITS) +
                  "; Options::qubit_limit = QubitLimit::Lift raises it to " +
                  std::to_string(LIFTED_MAX_QUBITS));
}

}  // namespace

TEST(V11311Algorithms, DeutschJozsaAtThirtyQueryQubitsThrows) {
    // n query qubits and one ancilla: n = 30 is a 31-qubit circuit.
    const int n = kOver - 1;
    QuantumCircuit oracle(n + 1);
    oracle.cx(0, n);  // balanced
    expect_over_the_limit(kOver, [&] { (void)DeutschJozsa::solve(oracle, n, 1, 1); });
}

TEST(V11311Algorithms, BernsteinVaziraniWithAnOversizedOracleThrows) {
    const int n = kOver - 1;
    QuantumCircuit oracle(n + 1);
    oracle.cx(0, n).cx(2, n);
    expect_over_the_limit(kOver, [&] { (void)BernsteinVazirani::solve(oracle, n, 1, 1); });
}

TEST(V11311Algorithms, SimonThrowsOnBothSamplingPaths) {
    // 2n qubits: n = 16 is a 32-qubit circuit.
    const int n = 16;
    QuantumCircuit oracle(2 * n);
    oracle.cx(0, n);
    for (const bool batch : {false, true}) {
        SCOPED_TRACE(batch ? "one batched simulation" : "one simulation per sample");
        expect_over_the_limit(2 * n, [&] { (void)Simon::solve(oracle, n, 1, 2, batch); });
    }
}

TEST(V11311Algorithms, PhaseEstimationThrows) {
    // One target qubit and thirty evaluation qubits.
    QuantumCircuit unitary(1);
    unitary.p(0.5, 0);
    expect_over_the_limit(kOver, [&] { (void)QPE::estimate_phase(unitary, kOver - 1, 16, 1); });
}

TEST(V11311Algorithms, GroverThrows) {
    QuantumCircuit oracle(kOver);
    oracle.z(0);
    expect_over_the_limit(kOver, [&] { (void)Grover::search(oracle, 1, 16, 1); });
}

TEST(V11311Algorithms, ShorThrowsRatherThanReportingAFailedAttempt) {
    // 323 = 17 * 19 passes every classical shortcut, so the quantum path runs:
    // a 9-qubit target register and 22 evaluation qubits make 31.
    Shor::Options options;
    options.n_eval_qubits = kOver - 9;
    options.max_attempts = 2;
    options.seed = 7;
    const Shor shor(options);
    expect_over_the_limit(kOver, [&] { (void)shor.factorize(323); });
}

TEST(V11311Algorithms, MaqaoaNoisyPathThrowsTheRunsRefusal) {
    // A one-qubit channel attached to CX, which the ansatz uses for its ZZ
    // terms: every density-matrix run refuses the width mismatch before its
    // first gate, and the optimisation stops there instead of scoring the
    // point as a poor energy.
    const SparsePauliOp cost(std::vector<PauliString>{PauliString("ZZ", Complex128(1.0, 0.0)),
                                                      PauliString("ZI", Complex128(0.5, 0.0))});
    MAQAOA maqaoa;
    maqaoa.options.p = 1;
    maqaoa.options.max_iterations = 3;
    maqaoa.options.seed = 3;
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.01), "cx");
    maqaoa.estimator.options.noise_model = noise;

    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)maqaoa.optimize(cost); });
    EXPECT_EQ(detail::failed_run_stores(), stores);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "DensityMatrixSimulator::run");
    v11311::expect_message(*e, {"DensityMatrixSimulator::run: the noise model's channel attached to "
                                "'cx' acts on 1 qubit(s), but it is being applied to 2."});
    ASSERT_TRUE(e->where().has_value());
    EXPECT_EQ(e->where()->gate, "cx");
}

// A non-finite initial theta (the QSP rotation that prepares each qubit) is
// refused by optimize() before any evaluation, on both paths, naming itself and
// the index, as QAOA's evaluations refuse the same input on both of its. The
// noiseless path evolves its own state, so without the check it would run the
// optimisation on NaN energies and fail only when the final state is sampled.
TEST(V11311Algorithms, MaqaoaRefusesANonFiniteInitialThetaUpFront) {
    const SparsePauliOp cost(std::vector<PauliString>{PauliString("ZZ", Complex128(1.0, 0.0)),
                                                      PauliString("ZI", Complex128(0.5, 0.0))});
    for (const bool noisy : {false, true}) {
        SCOPED_TRACE(noisy ? "noisy" : "noiseless");
        MAQAOA maqaoa;
        maqaoa.options.p = 1;
        maqaoa.options.max_iterations = 3;
        maqaoa.options.seed = 3;
        maqaoa.options.initial_thetas = {0.2, quiet_nan_strict()};
        if (noisy) {
            NoiseModel noise;
            noise.add_quantum_error(NoiseChannels::depolarizing(0.01), "h");
            maqaoa.estimator.options.noise_model = noise;
        }
        const std::uint64_t stores = detail::failed_run_stores();
        const auto e = v11311::thrown<InvalidArgument>([&] { (void)maqaoa.optimize(cost); });
        EXPECT_EQ(detail::failed_run_stores(), stores) << "a run was attempted";
        if (!e) continue;
        EXPECT_EQ(e->entry_point(), "MAQAOA::optimize");
        EXPECT_EQ(std::string(e->what()), "MAQAOA::optimize: initial_thetas[1] = " +
                                              std::to_string(quiet_nan_strict()) +
                                              "; every initial theta must be finite");
    }
}

TEST(V11312Algorithms, MaqaoaRefusesANonFiniteThetaInAVectorItWouldNotUse) {
    // Three thetas for two qubits: the vector's length leaves it unused, and
    // its infinity is still refused rather than ignored.
    const SparsePauliOp cost(std::vector<PauliString>{PauliString("ZZ", Complex128(1.0, 0.0))});
    MAQAOA maqaoa;
    maqaoa.options.p = 1;
    maqaoa.options.max_iterations = 3;
    maqaoa.options.initial_thetas = {0.1, 0.2, std::numeric_limits<double>::infinity()};
    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)maqaoa.optimize(cost); });
    EXPECT_EQ(detail::failed_run_stores(), stores) << "a run was attempted";
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(std::string(e->what()), "MAQAOA::optimize: initial_thetas[2] = " +
                                          std::to_string(std::numeric_limits<double>::infinity()) +
                                          "; every initial theta must be finite");
}

TEST(V11311Algorithms, MaqaoaRefusesANonFiniteStartingPoint) {
    // A starting point built from a non-finite angle is refused before the
    // optimiser evaluates anything, on either path.
    const SparsePauliOp cost(std::vector<PauliString>{PauliString("ZZ", Complex128(1.0, 0.0)),
                                                      PauliString("ZI", Complex128(0.5, 0.0))});
    for (const bool noisy : {false, true}) {
        SCOPED_TRACE(noisy ? "noisy" : "noiseless");
        MAQAOA maqaoa;
        maqaoa.options.p = 1;
        maqaoa.options.max_iterations = 3;
        maqaoa.options.seed = 3;
        maqaoa.options.mixer_weights = {1.0, 0.5};  // the starting betas scale beta_base
        maqaoa.options.beta_base = quiet_nan_strict();
        if (noisy) {
            NoiseModel noise;
            noise.add_quantum_error(NoiseChannels::depolarizing(0.01), "h");
            maqaoa.estimator.options.noise_model = noise;
        }
        const auto e = v11311::thrown<std::invalid_argument>([&] { (void)maqaoa.optimize(cost); });
        ASSERT_TRUE(e.has_value());
        EXPECT_STREQ(e->what(), "MAQAOA::optimize: the initial parameters contain a non-finite value");
    }
}
