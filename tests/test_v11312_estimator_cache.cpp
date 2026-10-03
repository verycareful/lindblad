// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.2 - the Estimator's transpile cache tells every circuit apart.
//
// At optimization_level >= 1 Estimator::run_single transpiles a circuit once
// and caches the result. The cache was keyed on the circuit's shape alone (its
// widths, gate types and qubit indices), while the circuit it held carried the
// first caller's numbers: a second circuit of the same shape, rx(0.7) after
// rx(0.5), was answered with the first one's expectation value, silently. The
// same went for a matrix, a classical condition, a measurement's target bit
// and the order of symbolic parameters. The key now holds everything the
// transpiled circuit depends on; symbolic parameter values, bound after the
// lookup, still share one entry.
//
// Each test primes an Estimator with one circuit and asks it about another
// that differs in a single part, then compares with a fresh Estimator that
// has seen only the second. Sampled runs use a fixed seed, so the two must
// agree exactly.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"

#include <cmath>
#include <functional>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

Estimator::Options cached_options(int shots) {
    Estimator::Options o;
    o.optimization_level = 1;
    o.shots = shots;
    o.seed = 7;
    return o;
}

SparsePauliOp z_on(int n, int qubit) {
    std::string pauli(static_cast<std::size_t>(n), 'I');
    pauli[static_cast<std::size_t>(qubit)] = 'Z';
    return SparsePauliOp(std::vector<PauliString>{PauliString(pauli)});
}

// Primes an Estimator with `first`, then asks it about `second`, and expects
// what an Estimator that has seen only `second` answers.
void expect_told_apart(const QuantumCircuit& first, const QuantumCircuit& second,
                       const SparsePauliOp& observable, int shots,
                       const std::vector<double>& parameters = {}) {
    Estimator primed(cached_options(shots));
    (void)primed.run_single(first, observable, parameters);
    const double got = primed.run_single(second, observable, parameters);
    Estimator fresh(cached_options(shots));
    const double want = fresh.run_single(second, observable, parameters);
    EXPECT_DOUBLE_EQ(got, want);
}

}  // namespace

TEST(V11312EstimatorCache, NumericCircuitsOfOneShapeEachGetTheirOwnValue) {
    Estimator est(cached_options(0));
    const SparsePauliOp z = z_on(1, 0);
    for (const double theta : {0.5, 0.7, 1.9}) {
        SCOPED_TRACE(theta);
        QuantumCircuit qc(1);
        qc.rx(theta, 0);
        EXPECT_NEAR(est.run_single(qc, z, {}), std::cos(theta), 1e-12);
    }
}

TEST(V11312EstimatorCache, AParameterValueKeysTheCache) {
    QuantumCircuit a(1), b(1);
    a.rx(0.5, 0);
    b.rx(0.9, 0);
    expect_told_apart(a, b, z_on(1, 0), 0);
}

TEST(V11312EstimatorCache, AMatrixKeysTheCache) {
    // X and H as one-qubit UNITARY instructions of the same label: <Z> is -1
    // after X and 0 after H.
    const std::vector<Complex128> x{Complex128(0.0, 0.0), Complex128(1.0, 0.0),
                                    Complex128(1.0, 0.0), Complex128(0.0, 0.0)};
    const std::vector<Complex128> h{Complex128(INV_SQRT2, 0.0), Complex128(INV_SQRT2, 0.0),
                                    Complex128(INV_SQRT2, 0.0), Complex128(-INV_SQRT2, 0.0)};
    QuantumCircuit a(1), b(1);
    a.unitary(x, {0}, "u");
    b.unitary(h, {0}, "u");
    expect_told_apart(a, b, z_on(1, 0), 0);
}

TEST(V11312EstimatorCache, AConditionKeysTheCache) {
    // Qubit 0 is set and measured into clbit 0; qubit 1 is flipped when clbit
    // 0 holds 1 in one circuit and when it holds 0 in the other.
    const auto build = [](int value) {
        QuantumCircuit qc(2, 1);
        qc.x(0).measure(0, 0);
        qc.add_if(0, value, GT::X, {1});
        return qc;
    };
    expect_told_apart(build(1), build(0), z_on(2, 1), 64);
}

TEST(V11312EstimatorCache, AMeasurementsTargetBitKeysTheCache) {
    // The condition reads clbit 0; one circuit measures into it, the other
    // into clbit 1.
    const auto build = [](int target) {
        QuantumCircuit qc(2, 2);
        qc.x(0).measure(0, target);
        qc.add_if(0, 1, GT::X, {1});
        return qc;
    };
    expect_told_apart(build(0), build(1), z_on(2, 1), 64);
}

TEST(V11312EstimatorCache, TheOrderOfSymbolicParametersKeysTheCache) {
    // The same gates, with the parameter list declared in the other order:
    // binding {0.3, 1.1} rotates qubit 0 by 0.3 in one circuit and by 1.1 in
    // the other. A transpiled circuit carries no parameter list, so the values
    // are bound by the caller's own; this holds whatever the cache shares.
    QuantumCircuit a(2), b(2);
    a.rx("a", 0).rx("b", 1);
    b.rx("a", 0).rx("b", 1);
    b.parameter_names = {"b", "a"};
    expect_told_apart(a, b, z_on(2, 0), 0, {0.3, 1.1});
}

TEST(V11312EstimatorCache, SymbolicValuesStillShareOneTranspile) {
    // One symbolic circuit, many bindings: every value is the circuit's own.
    Estimator est(cached_options(0));
    QuantumCircuit qc(1);
    qc.rx("theta", 0);
    const SparsePauliOp z = z_on(1, 0);
    for (const double theta : {0.4, 1.1, 2.6}) {
        SCOPED_TRACE(theta);
        EXPECT_NEAR(est.run_single(qc, z, {theta}), std::cos(theta), 1e-12);
    }
}
