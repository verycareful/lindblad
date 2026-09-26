// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.2 - a Pauli string's alphabet, and an observable's Hermiticity.
//
// Alphabet. The paths read Pauli characters differently: the exact statevector
// kernels recognised only uppercase X, Y and Z and read anything else as
// identity; the density-matrix and tableau paths also read lowercase and
// treated any other character as identity; the sampled estimator accepted both
// cases and refused the rest; QAOA's builder counted a lowercase 'i' as an
// active qubit. So SparsePauliOp("z") on |1> read +1 on one path and -1 on
// another. A Pauli string is now written with I, X, Y and Z, uppercase, and any
// other character is refused on construction and again at evaluation (the
// `pauli` member is public), as Qiskit's labels are.
//
// Hermiticity. Every evaluation returns a double. The expectation of a
// non-Hermitian operator is complex, and every path returned its real part
// silently, as did QAOA's and MA-QAOA's cost layers. An operator is now refused
// unless its coefficients are real once repeated labels are merged, which is
// exactly Hermiticity for a sum of Pauli strings, as Qiskit's estimators refuse
// it. to_matrix() returns the matrix itself and still takes any coefficients.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/validation.hpp"

#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

namespace {

constexpr double kTol = 64.0 * std::numeric_limits<double>::epsilon();

std::string refusal_of(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    ADD_FAILURE() << "no std::invalid_argument";
    return {};
}

// Built valid, then given a label no constructor would accept, which is how a
// caller holding the public members reaches an evaluation with one.
SparsePauliOp relabelled(const std::string& valid, const std::string& label) {
    SparsePauliOp op({PauliString(valid)});
    op.terms[0].pauli = label;
    return op;
}

// Z + iX: the i makes it anti-Hermitian in its X part.
SparsePauliOp z_plus_ix() {
    return SparsePauliOp::from_list({{"Z", Complex128(1.0, 0.0)}, {"X", Complex128(0.0, 1.0)}});
}

QuantumCircuit one_qubit_h() {
    QuantumCircuit qc(1);
    qc.h(0);
    return qc;
}

}  // namespace

// =============================================================================
// Alphabet
// =============================================================================

TEST(V11292PauliRules, ALabelOutsideUppercaseIXYZIsRefusedOnConstruction) {
    for (const char* label : {"z", "Zz", "xI", "Q", "I1", "i"}) {
        SCOPED_TRACE(label);
        EXPECT_THROW((void)PauliString(label), std::invalid_argument);
        EXPECT_THROW((void)SparsePauliOp::from_list({{label, Complex128(1.0, 0.0)}}),
                     std::invalid_argument);
    }
    EXPECT_NO_THROW((void)PauliString("IXYZ"));
}

TEST(V11292PauliRules, TheRefusalNamesTheCharacterAndItsPosition) {
    const std::string msg = refusal_of([] { (void)PauliString("XzI"); });
    EXPECT_NE(msg.find("'z'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("position 1"), std::string::npos) << msg;
}

TEST(V11292PauliRules, ALabelAssignedAfterConstructionIsRefusedAtEvaluation) {
    const SparsePauliOp lower = relabelled("ZZ", "zz");
    Statevector sv(2);
    EXPECT_THROW((void)lower.expectation_value(sv), std::invalid_argument);
    EXPECT_THROW((void)lower.expectation_value_batch({&sv}), std::invalid_argument);
    EXPECT_THROW((void)lower.to_matrix(), std::invalid_argument);
    EXPECT_THROW((void)DensityMatrix(2).expectation_value_sparse(lower), std::invalid_argument);

    QuantumCircuit qc(2);
    qc.h(0);
    for (int shots : {0, 64}) {
        SCOPED_TRACE(shots);
        Estimator est;
        est.options.shots = shots;
        EXPECT_THROW((void)est.run_single(qc, lower), std::invalid_argument);
    }
}

TEST(V11292PauliRules, TheTableauRefusesALabelOutsideTheAlphabet) {
    StabilizerState st(2);
    for (const char* label : {"zz", "ZQ", "iZ"}) {
        SCOPED_TRACE(label);
        EXPECT_THROW((void)st.expectation_pauli(label), std::invalid_argument);
    }
    EXPECT_EQ(st.expectation_pauli("ZZ"), 1);
}

TEST(V11292PauliRules, QaoaRefusesACostWithALabelOutsideTheAlphabet) {
    // The builder reads term.pauli[q] != 'I', which counts a lowercase 'i' as
    // an active qubit; the entry refuses it first.
    QAOA q;
    q.options.p = 1;
    EXPECT_THROW((void)q.build_circuit(relabelled("ZZ", "Zi"), SparsePauliOp(), {0.1, 0.2}),
                 std::invalid_argument);
}

// =============================================================================
// Hermiticity
// =============================================================================

TEST(V11292PauliRules, ANonHermitianOperatorIsRefusedByEveryEvaluation) {
    const SparsePauliOp op = z_plus_ix();
    Statevector plus = StatevectorSimulator().run(one_qubit_h(), 0, 0).final_state;

    const std::string msg = refusal_of([&] { (void)op.expectation_value(plus); });
    EXPECT_NE(msg.find("not Hermitian"), std::string::npos) << msg;
    EXPECT_NE(msg.find("'X'"), std::string::npos) << msg;

    EXPECT_THROW((void)op.expectation_value_batch({&plus}), std::invalid_argument);
    EXPECT_THROW((void)DensityMatrix(1).expectation_value_sparse(op), std::invalid_argument);

    for (int shots : {0, 64}) {
        SCOPED_TRACE(shots);
        Estimator est;
        est.options.shots = shots;
        EXPECT_THROW((void)est.run_single(one_qubit_h(), op), std::invalid_argument);
    }
    Estimator noisy;
    noisy.options.noise_model.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.01), "h");
    EXPECT_THROW((void)noisy.run_single(one_qubit_h(), op), std::invalid_argument);
}

TEST(V11292PauliRules, ImaginaryPartsThatCancelAcrossRepeatedLabelsAreHermitian) {
    // (1 + i)Z + (1 - i)Z is 2Z. Checking term by term would refuse a
    // Hermitian operator; merging first accepts it and evaluates it exactly.
    const SparsePauliOp two_z = SparsePauliOp::from_list(
        {{"Z", Complex128(1.0, 1.0)}, {"Z", Complex128(1.0, -1.0)}});
    const Statevector zero(1);
    EXPECT_NEAR(two_z.expectation_value(zero), 2.0, kTol);
    EXPECT_NEAR(DensityMatrix(1).expectation_value_sparse(two_z), 2.0, kTol);
}

TEST(V11292PauliRules, AnImaginaryPartInsideTheToleranceIsReal) {
    const SparsePauliOp nearly_real =
        SparsePauliOp::from_list({{"Z", Complex128(1.0, DEFAULT_PHYSICAL_ATOL / 2.0)}});
    EXPECT_NO_THROW((void)nearly_real.expectation_value(Statevector(1)));
}

TEST(V11292PauliRules, ToMatrixStillTakesAnyCoefficients) {
    // A matrix is not an expectation value: iY is a legitimate operator to
    // write down, it is only not an observable.
    const SparsePauliOp iy = SparsePauliOp::from_list({{"Y", Complex128(0.0, 1.0)}});
    const std::vector<Complex128> m = iy.to_matrix();
    ASSERT_EQ(m.size(), 4u);
    // iY = [[0, 1], [-1, 0]].
    EXPECT_NEAR(m[1].real, 1.0, kTol);
    EXPECT_NEAR(m[2].real, -1.0, kTol);
}

TEST(V11292PauliRules, TheDenseDensityMatrixExpectationChecksSizeAndHermiticity) {
    const DensityMatrix dm(1);
    // Too few entries would be read past.
    EXPECT_THROW((void)dm.expectation_value(std::vector<Complex128>(2, Complex128(1.0, 0.0))),
                 std::invalid_argument);
    // [[0, 1], [0, 0]] is not Hermitian.
    const std::vector<Complex128> raising = {Complex128(0.0, 0.0), Complex128(1.0, 0.0),
                                             Complex128(0.0, 0.0), Complex128(0.0, 0.0)};
    EXPECT_THROW((void)dm.expectation_value(raising), std::invalid_argument);
    // Z on |0> is 1.
    const std::vector<Complex128> z = {Complex128(1.0, 0.0), Complex128(0.0, 0.0),
                                       Complex128(0.0, 0.0), Complex128(-1.0, 0.0)};
    EXPECT_NEAR(dm.expectation_value(z), 1.0, kTol);
}

TEST(V11292PauliRules, TheObserverRefusesANonHermitianObservableBeforeTheRun) {
    v11261::RecorderPtr witness = v11261::recorder();
    RunPlan plan;
    plan.observations.observe(Anchor::every_instruction(), witness);
    plan.observations.observe(
        Anchor::at_end(),
        std::make_shared<ExpectationObserver>(SparsePauliOp::from_list(
            {{"ZIII", Complex128(1.0, 0.0)}, {"XIII", Complex128(0.0, 1.0)}})));
    StatevectorSimulator sim;
    const auto result = sim.run(v11261::layered_circuit(), 0, 11292, plan);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("not Hermitian"), std::string::npos)
        << result.error_message;
    EXPECT_EQ(witness->count(), 0u);
}

TEST(V11292PauliRules, QaoaAndMaqaoaRefuseANonHermitianCost) {
    // The cost layer rotates by each coefficient's real part, so the circuit
    // would apply a different cost than the one written.
    const SparsePauliOp cost = SparsePauliOp::from_list(
        {{"ZZ", Complex128(1.0, 0.0)}, {"XI", Complex128(0.0, 0.5)}});
    QAOA q;
    q.options.p = 1;
    EXPECT_THROW((void)q.build_circuit(cost, SparsePauliOp(), {0.1, 0.2}), std::invalid_argument);
    MAQAOA m;
    m.options.p = 1;
    EXPECT_THROW((void)m.num_parameters(cost), std::invalid_argument);
    EXPECT_THROW((void)m.build_circuit(cost, SparsePauliOp(), {0.1, 0.2, 0.3, 0.4}),
                 std::invalid_argument);
}
