// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.2 - one width rule for Pauli operators, held everywhere one is
// evaluated or built (issue #130).
//
// A Pauli string names one Pauli per qubit and carries no labels, so its length
// is the register it acts on. The exact statevector expectation took each
// term's masks from the string and never compared them with the state: a Z
// beyond the last qubit read that qubit as |0>, and an X or Y beyond it indexed
// past the amplitude arrays. The density-matrix path compared only the first
// term, to_matrix() wrote past its matrix for a term wider than the first, and
// the estimator's three modes answered the same mismatch three different ways.
//
// The rule, as it now stands: every term of one operator has one width
// (refused at construction and again at evaluation, since `terms` is public),
// an evaluation needs exactly the state's width, and an operator with no terms
// has no width and is refused. simplify() keeps the zero operator at its width
// when everything cancels, so H - H evaluates to 0 rather than being refused.
//
// The same rule reaches the callers that index term.pauli[q] themselves: QAOA's
// circuit builder (which never checked its mixer against its cost), MA-QAOA's
// entry points (which never checked the cost), the expectation observer (which
// found a mismatch only mid-run), and Estimator::run_batch, whose OpenMP loop
// turned any refusal from run_single into std::terminate.
//
// The X and Y cases below are safe to run because the check precedes the
// kernel; before it existed they were out-of-bounds reads.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/ising.hpp"
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

SparsePauliOp single(const std::string& label, Complex128 coeff = Complex128(1.0, 0.0)) {
    return SparsePauliOp({PauliString(label, coeff)});
}

// An operator assembled term by term, which is how a caller reaches a mixed
// width without passing through any constructor that checks.
SparsePauliOp built_by_hand(const std::vector<std::string>& labels) {
    SparsePauliOp op;
    for (const auto& label : labels) op.terms.push_back(PauliString(label));
    return op;
}

// q0 = 1, q1 = 0: basis index 1. Z on qubit 0 reads -1 and Z on qubit 1 reads
// +1, so a check that confused the two ends of the string would show.
Statevector q0_set() {
    QuantumCircuit qc(2);
    qc.x(0);
    return StatevectorSimulator().run(qc, 0, 0).final_state;
}

std::string refusal_of(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    ADD_FAILURE() << "no std::invalid_argument";
    return {};
}

// The estimator's three modes, set on a fresh estimator each time.
enum class Mode { ExactIdeal, ExactNoisy, Sampled };
const Mode kModes[] = {Mode::ExactIdeal, Mode::ExactNoisy, Mode::Sampled};

// Estimator holds a mutex, so it is configured in place rather than returned.
void configure(Estimator& est, Mode mode) {
    if (mode == Mode::ExactNoisy) {
        est.options.noise_model.add_all_qubit_quantum_error(
            NoiseChannels::depolarizing(0.01), "h");
    }
    if (mode == Mode::Sampled) est.options.shots = 64;
}

const char* name_of(Mode mode) {
    switch (mode) {
        case Mode::ExactIdeal: return "exact, ideal";
        case Mode::ExactNoisy: return "exact, noisy";
        case Mode::Sampled: return "sampled";
    }
    return "?";
}

QuantumCircuit two_qubit_circuit() {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    return qc;
}

}  // namespace

// =============================================================================
// The exact statevector expectation
// =============================================================================

TEST(V11292ObservableWidth, AnXOrYTermWiderThanTheStateIsRefused) {
    Statevector sv(2);
    for (const char* label : {"XXX", "IIY", "YYY", "IIX"}) {
        SCOPED_TRACE(label);
        EXPECT_THROW((void)single(label).expectation_value(sv), std::invalid_argument);
    }
}

TEST(V11292ObservableWidth, ATermNarrowerThanTheStateIsRefused) {
    // A shorter string would read as identity on the qubits it never names,
    // which is a guess about the caller's intent, not what the string says.
    Statevector sv(3);
    for (const char* label : {"Z", "ZZ", "X"}) {
        SCOPED_TRACE(label);
        EXPECT_THROW((void)single(label).expectation_value(sv), std::invalid_argument);
    }
}

TEST(V11292ObservableWidth, TheRefusalNamesTheTermAndBothWidths) {
    Statevector sv(2);
    const std::string msg = refusal_of([&] { (void)single("ZZZ").expectation_value(sv); });
    EXPECT_NE(msg.find("'ZZZ'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("3 qubits wide"), std::string::npos) << msg;
    EXPECT_NE(msg.find("2 qubit state"), std::string::npos) << msg;
}

TEST(V11292ObservableWidth, AMatchingOperatorStillEvaluatesOnTheRightQubit) {
    // pauli[q] acts on qubit q, and only qubit 0 is set.
    const Statevector sv = q0_set();
    EXPECT_NEAR(single("ZI").expectation_value(sv), -1.0, kTol);
    EXPECT_NEAR(single("IZ").expectation_value(sv), 1.0, kTol);
}

// =============================================================================
// The batch
// =============================================================================

TEST(V11292ObservableWidth, TheBatchChecksEveryStateBeforeEvaluatingAny) {
    const Statevector two = q0_set();
    const Statevector three(3);
    const std::vector<const Statevector*> states{&two, &three};
    EXPECT_THROW((void)single("ZI").expectation_value_batch(states), std::invalid_argument);
}

TEST(V11292ObservableWidth, AMatchingBatchAgreesWithTheSingleEvaluations) {
    const Statevector set = q0_set();
    const Statevector clear(2);
    const SparsePauliOp op = SparsePauliOp::from_list(
        {{"ZI", Complex128(0.75, 0.0)}, {"IZ", Complex128(-0.5, 0.0)}, {"XX", Complex128(0.25, 0.0)}});
    const std::vector<double> batch = op.expectation_value_batch({&set, &clear});
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_NEAR(batch[0], op.expectation_value(set), kTol);
    EXPECT_NEAR(batch[1], op.expectation_value(clear), kTol);
}

// =============================================================================
// An operator with no terms
// =============================================================================

TEST(V11292ObservableWidth, AnOperatorWithNoTermsIsRefusedByEveryEvaluation) {
    const SparsePauliOp none;
    const Statevector sv(2);
    const DensityMatrix dm(2);

    EXPECT_THROW((void)none.expectation_value(sv), std::invalid_argument);
    EXPECT_THROW((void)none.expectation_value_batch({&sv}), std::invalid_argument);
    EXPECT_THROW((void)none.expectation_value_batch({}), std::invalid_argument)
        << "no states to evaluate on does not give an operator with no terms a width";
    EXPECT_THROW((void)dm.expectation_value_sparse(none), std::invalid_argument);
    EXPECT_THROW((void)none.to_matrix(), std::invalid_argument);

    for (Mode mode : kModes) {
        SCOPED_TRACE(name_of(mode));
        Estimator est;
        configure(est, mode);
        const std::string msg =
            refusal_of([&] { (void)est.run_single(two_qubit_circuit(), none); });
        EXPECT_NE(msg.find("no terms"), std::string::npos) << msg;
    }
}

TEST(V11292ObservableWidth, TheZeroOperatorCarriesItsWidthAndEvaluatesToZero) {
    const SparsePauliOp zero = SparsePauliOp::zero(2);
    EXPECT_EQ(zero.expectation_value(q0_set()), 0.0);
    EXPECT_EQ(DensityMatrix(2).expectation_value_sparse(zero), 0.0);
    EXPECT_THROW((void)zero.expectation_value(Statevector(3)), std::invalid_argument);
}

// =============================================================================
// Construction and simplify
// =============================================================================

TEST(V11292ObservableWidth, ConstructionRefusesTermsOfDifferentWidths) {
    EXPECT_THROW((void)SparsePauliOp({PauliString("Z"), PauliString("ZZ")}),
                 std::invalid_argument);
    EXPECT_THROW((void)SparsePauliOp::from_list(
                     {{"Z", Complex128(1.0, 0.0)}, {"ZZ", Complex128(1.0, 0.0)}}),
                 std::invalid_argument);

    const std::string msg = refusal_of([] { (void)(single("Z") + single("ZZ")); });
    EXPECT_NE(msg.find("left operand is 1"), std::string::npos) << msg;
    EXPECT_NE(msg.find("right is 2"), std::string::npos) << msg;
}

TEST(V11292ObservableWidth, AddingAnOperatorWithNoTermsChangesNothing) {
    const SparsePauliOp h = SparsePauliOp::from_list(
        {{"ZI", Complex128(0.75, 0.0)}, {"XX", Complex128(0.25, 0.0)}});
    const Statevector sv = q0_set();
    EXPECT_NEAR((SparsePauliOp() + h).expectation_value(sv), h.expectation_value(sv), kTol);
    EXPECT_NEAR((h + SparsePauliOp()).expectation_value(sv), h.expectation_value(sv), kTol);
}

TEST(V11292ObservableWidth, EverythingCancellingLeavesTheZeroOperatorAtItsWidth) {
    const SparsePauliOp h = SparsePauliOp::from_list(
        {{"ZI", Complex128(0.75, 0.0)}, {"XX", Complex128(0.25, 0.0)}});
    const SparsePauliOp cancelled = h + h * -1.0;

    ASSERT_EQ(cancelled.size(), 1u);
    EXPECT_EQ(cancelled.terms[0].pauli, "II");
    EXPECT_EQ(cancelled.terms[0].coeff.real, 0.0);
    EXPECT_EQ(cancelled.terms[0].coeff.imag, 0.0);

    EXPECT_EQ(cancelled.expectation_value(q0_set()), 0.0);
    EXPECT_EQ(DensityMatrix(2).expectation_value_sparse(cancelled), 0.0);
    for (const Complex128& entry : cancelled.to_matrix()) {
        EXPECT_EQ(entry.real, 0.0);
        EXPECT_EQ(entry.imag, 0.0);
    }
}

TEST(V11292ObservableWidth, SimplifyingNoTermsStaysEmpty) {
    // There is no width to keep, so nothing is invented.
    EXPECT_EQ(SparsePauliOp().simplify().size(), 0u);
}

TEST(V11292ObservableWidth, MixedWidthsBuiltTermByTermAreRefusedWhereverTheyArrive) {
    // `terms` is public, so the evaluation check stands on its own.
    const SparsePauliOp wide_second = built_by_hand({"Z", "ZZ"});
    const SparsePauliOp narrow_second = built_by_hand({"ZZ", "Z"});

    EXPECT_THROW((void)wide_second.simplify(), std::invalid_argument);
    EXPECT_THROW((void)wide_second.to_matrix(), std::invalid_argument)
        << "the second term's mask would index past a 2x2 matrix";
    EXPECT_THROW((void)narrow_second.expectation_value(Statevector(2)), std::invalid_argument);
    EXPECT_THROW((void)DensityMatrix(2).expectation_value_sparse(narrow_second),
                 std::invalid_argument)
        << "the density-matrix path read pauli[1] of the one-character second term";
}

// =============================================================================
// The density matrix
// =============================================================================

TEST(V11292ObservableWidth, TheDensityMatrixRefusesWiderAndNarrowerTerms) {
    const DensityMatrix dm(2);
    EXPECT_THROW((void)dm.expectation_value_sparse(single("ZZZ")), std::invalid_argument);
    EXPECT_THROW((void)dm.expectation_value_sparse(single("Z")), std::invalid_argument);
}

// =============================================================================
// The estimator
// =============================================================================

TEST(V11292ObservableWidth, TheEstimatorRefusesAMismatchOnEveryMode) {
    for (Mode mode : kModes) {
        for (const char* label : {"ZZZ", "Z"}) {
            SCOPED_TRACE(std::string(name_of(mode)) + ", " + label);
            Estimator est;
            configure(est, mode);
            const std::string msg =
                refusal_of([&] { (void)est.run_single(two_qubit_circuit(), single(label)); });
            EXPECT_NE(msg.find("does not match"), std::string::npos) << msg;
            EXPECT_NE(msg.find("2 qubit circuit"), std::string::npos) << msg;
        }
    }
}

TEST(V11292ObservableWidth, RunBatchCarriesARefusalOutOfItsParallelLoop) {
    // An exception leaving an OpenMP region ends the process; this test reaching
    // its assertions at all is half of what it checks.
    Estimator est;
    EXPECT_THROW((void)est.run_batch(two_qubit_circuit(), single("ZZZ"), {{}, {}, {}, {}}),
                 std::invalid_argument);

    // Not only the width refusal: anything run_single raises comes out.
    QuantumCircuit measured(2, 2);
    measured.h(0).measure(0, 0);
    EXPECT_THROW((void)est.run_batch(measured, single("ZI"), {{}, {}}), std::invalid_argument);
}

TEST(V11292ObservableWidth, TheGradientCarriesTheRefusal) {
    const QuantumCircuit ansatz = VQE::real_amplitudes(2, 1);
    const std::vector<double> params(ansatz.parameter_names.size(), 0.0);
    Estimator est;
    EXPECT_THROW((void)est.gradient(ansatz, single("ZZZ"), params), std::invalid_argument);
}

// =============================================================================
// The Ising builder
// =============================================================================

TEST(V11292ObservableWidth, AnIsingModelWithNoCoefficientIsTheZeroOperator) {
    // An edgeless graph with no field: every coefficient zero.
    const std::vector<std::vector<double>> no_coupling(3, std::vector<double>(3, 0.0));
    const SparsePauliOp op =
        IsingHamiltonian::from_hJ({0.0, 0.0, 0.0}, no_coupling, 0.0).to_sparse_pauli_op();

    ASSERT_EQ(op.size(), 1u);
    EXPECT_EQ(op.terms[0].pauli, "III");
    EXPECT_EQ(op.terms[0].coeff.real, 0.0);
    EXPECT_EQ(op.expectation_value(Statevector(3)), 0.0);
}

// =============================================================================
// QAOA and MA-QAOA
// =============================================================================

namespace {

SparsePauliOp zz_cost() { return single("ZZ"); }

// QAOA owns an Estimator, so it too is configured in place.
void configure(QAOA& q) {
    q.options.p = 1;
    q.options.seed = 11292;
    q.options.max_iterations = 20;
}

}  // namespace

TEST(V11292ObservableWidth, QaoaRefusesAMixerOfAnotherWidth) {
    QAOA q;
    configure(q);
    for (const char* label : {"X", "XII"}) {
        SCOPED_TRACE(label);
        const SparsePauliOp mixer = single(label);
        const std::string msg = refusal_of([&] { (void)q.optimize(zz_cost(), mixer); });
        EXPECT_NE(msg.find("QAOA::optimize"), std::string::npos) << msg;
        EXPECT_NE(msg.find("cost Hamiltonian"), std::string::npos) << msg;
        EXPECT_THROW((void)q.build_circuit(zz_cost(), mixer, {0.1, 0.2}),
                     std::invalid_argument);
    }
}

TEST(V11292ObservableWidth, QaoaRefusesACostWithNoTerms) {
    QAOA q;
    configure(q);
    EXPECT_THROW((void)q.optimize(SparsePauliOp()), std::invalid_argument);
    EXPECT_THROW((void)q.build_circuit(SparsePauliOp(), SparsePauliOp(), {0.1, 0.2}),
                 std::invalid_argument);
}

TEST(V11292ObservableWidth, QaoaRefusesAMixerThatIsNotHermitian) {
    // The rotation uses the real part only, so accepting this would run a
    // different mixer under the caller's name. MA-QAOA refuses it already.
    QAOA q;
    configure(q);
    const SparsePauliOp complex_mixer = single("XI", Complex128(1.0, 0.5));
    EXPECT_THROW((void)q.optimize(zz_cost(), complex_mixer), std::invalid_argument);
    EXPECT_THROW((void)q.build_circuit(zz_cost(), complex_mixer, {0.1, 0.2}),
                 std::invalid_argument);

    // Real to within the tolerance is real, as in MA-QAOA.
    const SparsePauliOp nearly_real =
        single("XI", Complex128(1.0, DEFAULT_PHYSICAL_ATOL / 2.0));
    EXPECT_NO_THROW((void)q.build_circuit(zz_cost(), nearly_real, {0.1, 0.2}));
}

TEST(V11292ObservableWidth, QaoaStillReadsAnEmptyMixerAsTheDefault) {
    QAOA q;
    configure(q);
    EXPECT_NO_THROW((void)q.optimize(zz_cost()));
}

TEST(V11292ObservableWidth, MaqaoaRefusesACostWithNoTerms) {
    MAQAOA m;
    m.options.p = 1;
    EXPECT_THROW((void)m.num_parameters(SparsePauliOp(), SparsePauliOp()),
                 std::invalid_argument);
    EXPECT_THROW((void)m.build_circuit(SparsePauliOp(), SparsePauliOp(), {0.1, 0.2}),
                 std::invalid_argument);
    EXPECT_THROW((void)m.optimize(SparsePauliOp()), std::invalid_argument);
}

// =============================================================================
// The expectation observer
// =============================================================================

TEST(V11292ObservableWidth, TheObserverRefusesAWrongWidthBeforeTheRun) {
    // Decidable from the observable and the register alone, so the plan is
    // refused before any instruction runs (the preflight contract).
    for (const char* label : {"ZZZ", "ZZZZZ"}) {
        SCOPED_TRACE(label);
        v11261::RecorderPtr witness = v11261::recorder();
        RunPlan plan;
        plan.observations.observe(Anchor::every_instruction(), witness);
        plan.observations.observe(Anchor::at_end(),
                                  std::make_shared<ExpectationObserver>(single(label)));

        StatevectorSimulator sim;
        const auto result = sim.run(v11261::layered_circuit(), 0, 11292, plan);
        EXPECT_FALSE(result.success) << "a " << label << " observable ran on four qubits";
        EXPECT_NE(result.error_message.find("ExpectationObserver"), std::string::npos)
            << result.error_message;
        EXPECT_EQ(witness->count(), 0u) << "instructions ran before the refusal";
    }
}

TEST(V11292ObservableWidth, TheObserverRefusesAWrongWidthOnTheTableauToo) {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(),
                              std::make_shared<ExpectationObserver>(single("ZZZ")));
    CliffordSimulator clifford;
    EXPECT_THROW((void)clifford.run(v11261::layered_circuit(), 8, 11292, plan),
                 std::invalid_argument);
}

TEST(V11292ObservableWidth, TheObserverRefusesAnObservableWithNoTerms) {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(),
                              std::make_shared<ExpectationObserver>(SparsePauliOp()));
    StatevectorSimulator sim;
    const auto result = sim.run(v11261::layered_circuit(), 0, 11292, plan);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("no terms"), std::string::npos) << result.error_message;
}
