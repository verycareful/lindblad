// 1.1.26.1 test wave - entanglement entropy across a cut, by four routes.
//
// This is the highest-risk surface in the release. One quantity is computed
// four different ways: a GF(2) rank on the tableau, the spectrum of two
// environment Gram matrices on the MPS, and a reduced matrix plus an eigensolve
// on the statevector and the density matrix. Four implementations of one
// definition is four chances to be plausibly wrong, and a plausible wrong
// answer here is invisible to a smoke test: it is a positive real number of
// roughly the right size.
//
// So the design of this suite is cross-checking rather than fixture matching.
// Every value is either analytic (a Bell pair is exactly one bit) or is
// produced twice by routes that share no code and compared. The MPS comparison
// is the load-bearing one: the environment Gram construction has an ordering
// that must be G_R * conj(G_L), and the natural-looking G_L * G_R returns a
// wrong value of the same order of magnitude. Nothing but an independent route
// separates those two, which is why every MPS case here has a statevector
// twin computed on the same circuit.
//
// The states are built by GATES rather than handed over, so the MPS tensors are
// in whatever form gate application leaves them. The route explicitly does not
// assume a canonical form, and preparing a canonical state would test the
// assumption instead of the code.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace lindblad;

using v11261::sv_run_failure;

namespace {

constexpr double kTol = 1e-9;
constexpr double kLoose = 1e-7;  // the eigensolve and SVD routes

QuantumCircuit bell() {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    return qc;
}

QuantumCircuit ghz(int n) {
    QuantumCircuit qc(n);
    qc.h(0);
    for (int q = 1; q < n; ++q) qc.cx(q - 1, q);
    return qc;
}

// Two independent Bell pairs: 0 with 1, and 2 with 3.
QuantumCircuit two_bell_pairs() {
    QuantumCircuit qc(4);
    qc.h(0);
    qc.cx(0, 1);
    qc.h(2);
    qc.cx(2, 3);
    return qc;
}

// A state with an entropy that is not 0, 1 or 2, built by gates so no tensor is
// canonical. The value is not asserted directly anywhere: it is compared
// between routes, which is what makes it useful.
//
// Every CNOT target starts in |0> and not in |+>. A CNOT onto an X eigenstate
// creates no entanglement whatever, so a chain built on |+> targets leaves a
// product state with entropy zero at every cut, and every comparison below
// would then be zero against zero.
//
// RY splits the amplitude unevenly before the pair is formed, which is what
// makes the entropy non-integral: a maximally entangled pair gives exactly one
// bit at any cut, and one bit is a value the wrong Gram ordering can also
// produce.
QuantumCircuit lopsided(int n) {
    QuantumCircuit qc(n);
    qc.ry(PI / 3.0, 0);
    qc.cx(0, 1);
    qc.ry(PI / 5.0, 1);
    if (n > 2) {
        qc.cx(1, 2);
        qc.ry(PI / 7.0, 2);
    }
    if (n > 3) qc.cx(2, 3);
    qc.rz(PI / 4.0, 0);
    return qc;
}

double sv_entropy(const QuantumCircuit& qc, std::vector<int> region,
                  double order = 1.0) {
    auto obs = std::make_shared<EntropyObserver>(std::move(region), order);
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(qc, 0, 20261, plan);
    EXPECT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(obs->count(), 1u);
    return obs->count() == 1u ? obs->value() : -1.0;
}

double clifford_entropy(const QuantumCircuit& qc, std::vector<int> region,
                        double order = 1.0) {
    auto obs = std::make_shared<EntropyObserver>(std::move(region), order);
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::at_end(), obs);

    CliffordSimulator sim;
    sim.run(qc, 8, 20261, plan);
    EXPECT_EQ(obs->count(), 1u);
    return obs->count() == 1u ? obs->value() : -1.0;
}

double mps_entropy(const QuantumCircuit& qc, std::vector<int> region,
                   double order = 1.0) {
    auto obs = std::make_shared<EntropyObserver>(std::move(region), order);
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::at_end(), obs);

    MPSSimulator sim;
    sim.run(qc, 32, 0, 20261, plan);
    EXPECT_EQ(obs->count(), 1u);
    return obs->count() == 1u ? obs->value() : -1.0;
}

}  // namespace

// =============================================================================
// Clifford, against hand-checked values
// =============================================================================

TEST(V11261Entropy, AProductStateHasNoEntanglement) {
    QuantumCircuit qc(3);
    qc.x(0);
    qc.h(1);
    EXPECT_NEAR(clifford_entropy(qc, {0}), 0.0, kTol);
    EXPECT_NEAR(clifford_entropy(qc, {1}), 0.0, kTol);
    EXPECT_NEAR(sv_entropy(qc, {0}), 0.0, kLoose);
}

TEST(V11261Entropy, ABellPairIsExactlyOneBit) {
    EXPECT_NEAR(clifford_entropy(bell(), {0}), 1.0, kTol);
    EXPECT_NEAR(clifford_entropy(bell(), {1}), 1.0, kTol);
    EXPECT_NEAR(sv_entropy(bell(), {0}), 1.0, kLoose);
    EXPECT_NEAR(mps_entropy(bell(), {0}), 1.0, kLoose);
}

TEST(V11261Entropy, GhzIsOneBitAtAnySingleQubitCut) {
    // Every single-qubit marginal of a GHZ state is maximally mixed, so the
    // figure does not depend on which qubit is named. A route that had the cut
    // the wrong way round would still pass on a symmetric cut, which is why
    // the four-qubit asymmetric cases below matter.
    const QuantumCircuit qc = ghz(4);
    for (int q = 0; q < 4; ++q) {
        EXPECT_NEAR(clifford_entropy(qc, {q}), 1.0, kTol) << "qubit " << q;
    }
    EXPECT_NEAR(sv_entropy(qc, {0}), 1.0, kLoose);
}

TEST(V11261Entropy, TwoBellPairsCutAcrossBothGiveTwoBits) {
    // The cut {0, 2} takes one qubit from each pair, so it breaks two
    // independent bits of entanglement and the entropies add.
    EXPECT_NEAR(clifford_entropy(two_bell_pairs(), {0, 2}), 2.0, kTol);
    EXPECT_NEAR(sv_entropy(two_bell_pairs(), {0, 2}), 2.0, kLoose);

    // The cut {0, 1} takes a whole pair, so it breaks nothing.
    EXPECT_NEAR(clifford_entropy(two_bell_pairs(), {0, 1}), 0.0, kTol);
    EXPECT_NEAR(sv_entropy(two_bell_pairs(), {0, 1}), 0.0, kLoose);
}

TEST(V11261Entropy, CliffordEntropiesAreIntegersAndOrderIndependent) {
    // A stabilizer state's reduced state is maximally mixed on its support, so
    // the spectrum is flat: every Renyi order gives the same integer.
    const QuantumCircuit qc = two_bell_pairs();
    const double reference = clifford_entropy(qc, {0, 2});

    EXPECT_NEAR(reference, std::round(reference), kTol);
    for (const double order : {0.5, 1.0, 2.0, 3.0, 8.0}) {
        EXPECT_NEAR(clifford_entropy(qc, {0, 2}, order), reference, kTol)
            << "Renyi order " << order;
    }
}

TEST(V11261Entropy, TheStatevectorAgreesWithTheTableauOnAStabilizerCircuit) {
    // Same circuit, two backends, two entirely different algorithms: a GF(2)
    // rank against an eigensolve on a reduced matrix.
    const QuantumCircuit qc = two_bell_pairs();
    for (const std::vector<int>& region :
         {std::vector<int>{0}, std::vector<int>{0, 2}, std::vector<int>{1, 2, 3}}) {
        EXPECT_NEAR(clifford_entropy(qc, region), sv_entropy(qc, region), kLoose);
    }
}

// =============================================================================
// MPS against the statevector, which is what pins the Gram ordering
// =============================================================================

TEST(V11261Entropy, TheMpsRouteAgreesWithTheStatevectorAtAPrefixCut) {
    const QuantumCircuit qc = lopsided(4);
    const double reference = sv_entropy(qc, {0, 1});

    // Not 0, 1 or 2: a value the wrong Gram ordering would miss by an amount
    // too small to look like a bug and too large to be rounding.
    EXPECT_GT(reference, 0.05);
    EXPECT_NEAR(mps_entropy(qc, {0, 1}), reference, kLoose);
}

TEST(V11261Entropy, TheMpsRouteAgreesWithTheStatevectorAtASuffixCut) {
    const QuantumCircuit qc = lopsided(4);
    const double reference = sv_entropy(qc, {2, 3});
    EXPECT_NEAR(mps_entropy(qc, {2, 3}), reference, kLoose);
}

TEST(V11261Entropy, TheMpsRouteAgreesAtEverySingleQubitCut) {
    const QuantumCircuit qc = lopsided(4);
    for (int q = 0; q < 4; ++q) {
        EXPECT_NEAR(mps_entropy(qc, {q}), sv_entropy(qc, {q}), kLoose)
            << "qubit " << q;
    }
}

TEST(V11261Entropy, ANonContiguousMpsCutFallsBackToDenseAndStillAgrees) {
    // {0, 2} is not a prefix or a suffix, so the bond spectrum does not
    // describe it and the observer has to reduce the dense state instead. The
    // answer must not change because the route did.
    const QuantumCircuit qc = lopsided(4);
    EXPECT_NEAR(mps_entropy(qc, {0, 2}), sv_entropy(qc, {0, 2}), kLoose);
}

// =============================================================================
// The Renyi ladder
// =============================================================================

TEST(V11261Entropy, RenyiOrderTwoIsTheNegativeLogOfTheReducedPurity) {
    // Independent identity rather than a second call to the same code: on a
    // Bell pair the reduced state is maximally mixed on two levels, so its
    // purity is 1/2 and -log2(1/2) is exactly 1.
    const double expected = -std::log2(1.0 / 2.0);
    EXPECT_NEAR(sv_entropy(bell(), {0}, 2.0), expected, kLoose);
    EXPECT_NEAR(mps_entropy(bell(), {0}, 2.0), expected, kLoose);
}

TEST(V11261Entropy, TheOrderOneBranchIsSelectedByATolerance) {
    // Order 1 is von Neumann and is chosen within 1e-12, so an order just
    // inside that window takes the same branch and must give the same answer.
    const QuantumCircuit qc = lopsided(3);
    const double at_one = sv_entropy(qc, {0}, 1.0);
    EXPECT_NEAR(sv_entropy(qc, {0}, 1.0 + 1e-13), at_one, kTol);

    // Just outside it, the Renyi formula is used instead. On a spectrum that
    // is not flat the two genuinely differ, which is what makes the boundary
    // observable at all.
    const double just_outside = sv_entropy(qc, {0}, 1.0 + 1e-3);
    EXPECT_NEAR(just_outside, at_one, 1e-2);
}

TEST(V11261Entropy, HigherOrdersDoNotExceedTheVonNeumannValue) {
    // Renyi entropy is non-increasing in its order. Not a tautology of the
    // implementation: it is a property of the definition, and a spectrum
    // mishandled in normalisation breaks it.
    const QuantumCircuit qc = lopsided(4);
    const double s1 = sv_entropy(qc, {0, 1}, 1.0);
    const double s2 = sv_entropy(qc, {0, 1}, 2.0);
    const double s8 = sv_entropy(qc, {0, 1}, 8.0);

    EXPECT_LE(s2, s1 + kLoose);
    EXPECT_LE(s8, s2 + kLoose);
    EXPECT_GE(s8, -kLoose);
}

// =============================================================================
// The density matrix reports a different quantity, and says so
// =============================================================================

TEST(V11261Entropy, APureDensityMatrixAgreesWithTheStatevector) {
    auto obs = std::make_shared<EntropyObserver>(std::vector<int>{0});
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::at_end(), obs);

    DensityMatrixSimulator sim;
    const NoiseModel noise;
    auto r = sim.run(bell(), noise, 8, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    ASSERT_EQ(obs->count(), 1u);
    EXPECT_NEAR(obs->value(), 1.0, kLoose);
}

TEST(V11261Entropy, OnADensityMatrixTheFigureIsNotAnEntanglementEntropy) {
    // A mixed global state mixes classical correlation with entanglement, so
    // the reduced entropy is a different quantity. The observer reports WHICH
    // of the two it handed back rather than leaving it to be inferred from the
    // backend, and that flag is the whole point.
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.3), "h");

    auto obs = std::make_shared<EntropyObserver>(std::vector<int>{0});
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::at_end(), obs);

    DensityMatrixSimulator sim;
    auto r = sim.run(bell(), noise, 8, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    ASSERT_EQ(obs->count(), 1u);
    EXPECT_FALSE(obs->is_entanglement());
    EXPECT_GE(obs->value(), 0.0);
}

TEST(V11261Entropy, OnThePureBackendsTheFigureIsAnEntanglementEntropy) {
    auto obs = std::make_shared<EntropyObserver>(std::vector<int>{0});
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::at_end(), obs);

    StatevectorSimulator sim;
    auto r = sim.run(bell(), 0, 20261, plan);
    ASSERT_TRUE(r.success) << r.error_message;

    ASSERT_EQ(obs->count(), 1u);
    EXPECT_TRUE(obs->is_entanglement());
}

// =============================================================================
// Region validation, and WHERE each check happens
// =============================================================================

TEST(V11261Entropy, AnEmptyRegionIsRejectedAtConstruction) {
    EXPECT_THROW(EntropyObserver(std::vector<int>{}), std::invalid_argument);
}

TEST(V11261Entropy, ANonPositiveRenyiOrderIsRejectedAtConstruction) {
    EXPECT_THROW(EntropyObserver(std::vector<int>{0}, 0.0), std::invalid_argument);
    EXPECT_THROW(EntropyObserver(std::vector<int>{0}, -1.0), std::invalid_argument);
}

TEST(V11261Entropy, ARepeatedQubitInTheRegionFailsTheRun) {
    // A cut has each qubit on exactly one side of it.
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<EntropyObserver>(std::vector<int>{1, 1}));

    const std::string message = sv_run_failure(bell(), plan);
    EXPECT_NE(message.find("twice"), std::string::npos) << message;
}

TEST(V11261Entropy, AQubitOutsideTheRegisterFailsTheRun) {
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<EntropyObserver>(std::vector<int>{7}));

    const std::string message = sv_run_failure(bell(), plan);
    EXPECT_NE(message.find("outside"), std::string::npos) << message;
}

TEST(V11261Entropy, ACutNamingEveryQubitFailsTheRun) {
    RunPlan plan;
    plan.observations.observe(
        Anchor::at_end(), std::make_shared<EntropyObserver>(std::vector<int>{0, 1}));

    const std::string message = sv_run_failure(bell(), plan);
    EXPECT_FALSE(message.empty());
}
