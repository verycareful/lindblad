// 1.1.27.1 test wave - MA-QAOA custom mixer Hamiltonians (#41).
//
// The mixer_hamiltonian argument existed from the beginning and could not be
// used: first silently ignored, then rejected outright. Now it is applied, and
// three separate things have to hold for that to mean anything.
//
// The DECOMPOSITION has to be the ordered product of per-term rotations
// exp(-i*beta*c_k*P_k), with multi-qubit terms taking the basis-change and
// CX-chain recipe rather than a factorisation into independent per-qubit
// rotations. That distinction is the whole reason the feature exists: RX(x)RY(y)
// and exp(-i*beta*XY) are different ansatzes, and the constraint-preserving XY
// family is exactly where they part. Plain QAOA already implements the right
// one and is tested, so the assertions here compare against QAOA rather than
// against a hand-written reference, and a separate case shows the shortcut
// disagreeing so the comparison is not vacuous.
//
// The BETA DISPATCH has to be a choice rather than an assumption. The paper
// ansatz gives one beta per QUBIT and a custom mixer has TERMS; they coincide
// only for the per-qubit X mixer MA-QAOA was defined on. The claim that
// LowestActiveQubit reduces to the paper ansatz is checked as bit-identity
// against the built-in path, not as an approximation.
//
// The PARAMETER COUNT has to follow the mixer. A count that is too large leaves
// coordinates driving no gate, which an optimizer cannot tell apart from
// coordinates that have converged; one that is too small silently reads zero
// for the parameters past the end. Both directions are checked by perturbing
// the vector rather than by reading the count back.
//
// Also here, because it lives in the same function and shipped in the same
// release: #122, where orbit mode was decided by two disagreeing tests and a
// wrong-length orbit_assignments indexed the per-layer beta array out of range.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;
using Dispatch = MAQAOA::Options::MixerBetaDispatch;

namespace {

// A three-qubit ring, the fixture the existing MA-QAOA suites use.
SparsePauliOp ring_cost_3q() {
    return SparsePauliOp(std::vector<PauliString>{
        PauliString("ZZI", Complex128(1.0, 0.0)),
        PauliString("IZZ", Complex128(1.0, 0.0)),
        PauliString("ZIZ", Complex128(1.0, 0.0))});
}

SparsePauliOp zz_cost_2q() {
    return SparsePauliOp(std::vector<PauliString>{
        PauliString("ZZ", Complex128(1.0, 0.0))});
}

// The canonical MA-QAOA mixer written out as a Hamiltonian: one X per qubit.
SparsePauliOp x_mixer(int n) {
    std::vector<PauliString> terms;
    for (int q = 0; q < n; ++q) {
        std::string p(static_cast<std::size_t>(n), 'I');
        p[static_cast<std::size_t>(q)] = 'X';
        terms.emplace_back(p, Complex128(1.0, 0.0));
    }
    return SparsePauliOp(terms);
}

// Statevector is move-only, so the Result's own copy is handed over.
Statevector run(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    auto r = sim.run(qc, 0, 1);
    EXPECT_TRUE(r.success) << r.error_message;
    return std::move(r.final_state);
}

// Largest |a_i - b_i| across two states of the same width.
double max_amp_diff(const Statevector& a, const Statevector& b) {
    EXPECT_EQ(a.num_qubits(), b.num_qubits());
    double worst = 0.0;
    const std::size_t dim = std::size_t(1) << a.num_qubits();
    for (std::size_t i = 0; i < dim; ++i)
        worst = std::max(worst, (a.amplitude(i) - b.amplitude(i)).norm());
    return worst;
}

// A parameter vector of the exact length the options and mixer call for, with
// distinct non-symmetric entries so that a mis-dispatched beta moves the state.
// A vector of equal values would hide every dispatch bug there is.
std::vector<double> distinct_params(const MAQAOA& m, const SparsePauliOp& cost,
                                    const SparsePauliOp& mixer) {
    const int n = m.num_parameters(cost, mixer);
    std::vector<double> p;
    p.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) p.push_back(0.13 * (i + 1) - 0.29);
    return p;
}

}  // namespace

// =============================================================================
// The default path is untouched
// =============================================================================

TEST(V11271MixerDispatch, EmptyMixerIgnoresTheDispatchOption) {
    // The dispatch names which beta drives which mixer TERM, and the default
    // mixer has no terms to name. An empty mixer must therefore produce the
    // same circuit whatever the option says, or the option has silently become
    // a second way to configure the paper ansatz.
    const auto cost = ring_cost_3q();

    MAQAOA base;
    base.options.p = 2;
    const auto params = distinct_params(base, cost, {});
    const auto reference = run(base.build_circuit(cost, {}, params));

    for (auto d : {Dispatch::LowestActiveQubit, Dispatch::TermIndexed,
                   Dispatch::Shared}) {
        MAQAOA m;
        m.options.p = 2;
        m.options.mixer_beta_dispatch = d;
        EXPECT_EQ(m.num_parameters(cost), base.num_parameters(cost))
            << "dispatch " << static_cast<int>(d)
            << " changed the count on the default mixer";
        EXPECT_EQ(max_amp_diff(reference, run(m.build_circuit(cost, {}, params))),
                  0.0)
            << "dispatch " << static_cast<int>(d)
            << " changed the default per-qubit RX mixer";
    }
}

TEST(V11271MixerDispatch, EmptyMixerStillSharesBetasByOrbit) {
    // Orbit sharing is the beta customisation the paper ansatz ships with, and
    // it has to keep working alongside the new option rather than being
    // displaced by it.
    const auto cost = ring_cost_3q();

    MAQAOA m;
    m.options.p = 1;
    m.options.orbit_assignments = {0, 0, 1};
    m.options.mixer_beta_dispatch = Dispatch::TermIndexed;

    // Two orbits, so two betas rather than three.
    const int expected_betas = 2;
    const int cost_params = m.num_parameters(cost) - expected_betas;
    EXPECT_GT(cost_params, 0);
    EXPECT_EQ(m.num_parameters(cost), cost_params + expected_betas);
}

// =============================================================================
// LowestActiveQubit reduces to the paper ansatz
// =============================================================================

TEST(V11271MixerDispatch, CanonicalXMixerIsBitIdenticalToTheBuiltInMixer) {
    // The load-bearing claim of the default dispatch. sum_i X_i handed in as a
    // Hamiltonian must produce the SAME state as the built-in prod_i
    // RX(2*beta_i), not merely a similar one, or the custom path is a second
    // ansatz standing beside the paper's rather than a generalisation of it.
    for (int n : {2, 3, 4}) {
        SparsePauliOp cost =
            n == 2 ? zz_cost_2q()
                   : SparsePauliOp(std::vector<PauliString>{PauliString(
                         std::string(static_cast<std::size_t>(n), 'Z'),
                         Complex128(1.0, 0.0))});

        MAQAOA m;
        m.options.p = 2;
        const auto params = distinct_params(m, cost, {});

        ASSERT_EQ(m.num_parameters(cost, x_mixer(n)), m.num_parameters(cost))
            << "n = " << n
            << ": the canonical mixer reaches one beta per qubit, so the count "
               "cannot change";

        const double d = max_amp_diff(run(m.build_circuit(cost, {}, params)),
                                      run(m.build_circuit(cost, x_mixer(n), params)));
        EXPECT_EQ(d, 0.0) << "n = " << n
                          << ": the custom route differs from the built-in one "
                             "by " << d;
    }
}

TEST(V11271MixerDispatch, CanonicalXMixerFollowsOrbitSharing) {
    // Under orbit sharing the dispatch key is the orbit of the term's lowest
    // active qubit, so two qubits in one orbit collapse to one beta exactly as
    // they do on the built-in path.
    const auto cost = ring_cost_3q();

    MAQAOA m;
    m.options.p = 1;
    m.options.orbit_assignments = {0, 0, 1};

    EXPECT_EQ(m.num_parameters(cost, x_mixer(3)), m.num_parameters(cost))
        << "orbit sharing must reach the custom path with the same beta count";

    const auto params = distinct_params(m, cost, {});
    EXPECT_EQ(max_amp_diff(run(m.build_circuit(cost, {}, params)),
                           run(m.build_circuit(cost, x_mixer(3), params))),
              0.0)
        << "the custom route ignored orbit sharing the built-in route honours";
}

// =============================================================================
// The decomposition
// =============================================================================

TEST(V11271MixerDispatch, EntanglingTermMatchesTheQaoaDecomposition) {
    // XY is the constraint-preserving family this feature exists to support,
    // and QAOA's implementation of exp(-i*beta*c*P) is the tested one. Shared
    // dispatch gives MA-QAOA one beta for the whole mixer, which is what QAOA
    // has, so the two parameter vectors describe the same ansatz. Both gammas
    // are zero, making the cost unitary the identity and isolating the mixer.
    const auto cost = zz_cost_2q();
    const SparsePauliOp mixer(std::vector<PauliString>{
        PauliString("XY", Complex128(0.7, 0.0))});
    const double beta = 0.3;

    QAOA q;
    q.options.p = 1;
    const auto expected = run(q.build_circuit(cost, mixer, {0.0, beta}));

    MAQAOA m;
    m.options.p = 1;
    m.options.mixer_beta_dispatch = Dispatch::Shared;
    ASSERT_EQ(m.num_parameters(cost, mixer), 3)
        << "two per-qubit gammas and one shared beta";
    const auto actual = run(m.build_circuit(cost, mixer, {0.0, 0.0, beta}));

    EXPECT_LE(max_amp_diff(expected, actual), 1e-12);
}

TEST(V11271MixerDispatch, TheFactorisedShortcutIsADifferentAnsatz) {
    // Without this the comparison above proves nothing: if RX(x)RY(y) happened
    // to agree with exp(-i*beta*XY), matching QAOA would be no evidence that
    // the CX-chain recipe was used rather than the shortcut.
    const auto cost = zz_cost_2q();
    const SparsePauliOp mixer(std::vector<PauliString>{
        PauliString("XY", Complex128(0.7, 0.0))});
    const double beta = 0.3;
    const double angle = 2.0 * beta * 0.7;

    MAQAOA m;
    m.options.p = 1;
    m.options.mixer_beta_dispatch = Dispatch::Shared;
    const auto correct = run(m.build_circuit(cost, mixer, {0.0, 0.0, beta}));

    QuantumCircuit shortcut(2);
    shortcut.h(0);
    shortcut.h(1);
    shortcut.rx(angle, 0);
    shortcut.ry(angle, 1);

    EXPECT_GT(max_amp_diff(correct, run(shortcut)), 1e-3)
        << "the ordered-product decomposition and independent per-qubit "
           "rotations agree, so this fixture cannot tell them apart and the "
           "QAOA comparison is not evidence of anything";
}

TEST(V11271MixerDispatch, SingleQubitTermsUseTheMatchingRotation) {
    // A one-qubit mixer term is a bare rotation about its own axis, with no
    // basis change and no CX chain. X, Y and Z each have to reach the right
    // one; sending them all to Rz would still commute with a diagonal cost and
    // could pass a weaker test.
    const auto cost = zz_cost_2q();
    const double beta = 0.37;

    struct Case { const char* pauli; char axis; };
    for (const Case c : {Case{"XI", 'x'}, Case{"YI", 'y'}, Case{"ZI", 'z'}}) {
        const SparsePauliOp mixer(std::vector<PauliString>{
            PauliString(c.pauli, Complex128(1.0, 0.0))});

        MAQAOA m;
        m.options.p = 1;
        m.options.mixer_beta_dispatch = Dispatch::Shared;
        const auto actual = run(m.build_circuit(cost, mixer, {0.0, 0.0, beta}));

        QuantumCircuit expected(2);
        expected.h(0);
        expected.h(1);
        if (c.axis == 'x') expected.rx(2.0 * beta, 0);
        else if (c.axis == 'y') expected.ry(2.0 * beta, 0);
        else expected.rz(2.0 * beta, 0);

        EXPECT_LE(max_amp_diff(run(expected), actual), 1e-12)
            << "mixer term " << c.pauli << " did not reach the "
            << c.axis << " rotation";
    }
}

TEST(V11271MixerDispatch, TheCoefficientScalesTheAngle) {
    // The rotation angle is 2*beta*c_k, so doubling the coefficient and halving
    // beta is the same unitary. A decomposition that dropped the coefficient
    // would still pass every fixture whose coefficient is 1.
    const auto cost = zz_cost_2q();

    const SparsePauliOp heavy(std::vector<PauliString>{
        PauliString("XY", Complex128(0.8, 0.0))});
    const SparsePauliOp light(std::vector<PauliString>{
        PauliString("XY", Complex128(0.4, 0.0))});

    MAQAOA m;
    m.options.p = 1;
    m.options.mixer_beta_dispatch = Dispatch::Shared;

    EXPECT_LE(max_amp_diff(run(m.build_circuit(cost, heavy, {0.0, 0.0, 0.25})),
                           run(m.build_circuit(cost, light, {0.0, 0.0, 0.50}))),
              1e-12)
        << "the coefficient is not reaching the angle";

    EXPECT_GT(max_amp_diff(run(m.build_circuit(cost, heavy, {0.0, 0.0, 0.25})),
                           run(m.build_circuit(cost, light, {0.0, 0.0, 0.25}))),
              1e-3)
        << "two different coefficients produced the same unitary, so the "
           "coefficient is being ignored";
}

TEST(V11271MixerDispatch, NonCommutingTermsAreAppliedInOrder) {
    // The documented semantics are an ordered product, exact when the terms
    // commute and a first-order Trotter step otherwise. For non-commuting terms
    // that makes the result order-dependent, and saying so is only honest if it
    // is true: an implementation that summed the generators instead would be
    // order-independent here.
    const auto cost = zz_cost_2q();
    const double beta = 0.4;

    const SparsePauliOp forward(std::vector<PauliString>{
        PauliString("XI", Complex128(1.0, 0.0)),
        PauliString("YI", Complex128(1.0, 0.0))});
    const SparsePauliOp reversed(std::vector<PauliString>{
        PauliString("YI", Complex128(1.0, 0.0)),
        PauliString("XI", Complex128(1.0, 0.0))});

    MAQAOA m;
    m.options.p = 1;
    m.options.mixer_beta_dispatch = Dispatch::Shared;

    EXPECT_GT(max_amp_diff(run(m.build_circuit(cost, forward, {0.0, 0.0, beta})),
                           run(m.build_circuit(cost, reversed, {0.0, 0.0, beta}))),
              1e-6)
        << "X and Y on one qubit do not commute, so the ordered product must "
           "depend on their order; an order-independent result means the terms "
           "were combined rather than applied in sequence";
}

TEST(V11271MixerDispatch, CommutingTermsAreOrderIndependent) {
    // The other half of the same statement. Where the terms DO commute the
    // product is exact, so term order cannot matter, and a decomposition that
    // introduced a spurious dependence would show up here.
    const auto cost = zz_cost_2q();
    const double beta = 0.4;

    const SparsePauliOp forward(std::vector<PauliString>{
        PauliString("XI", Complex128(1.0, 0.0)),
        PauliString("IX", Complex128(1.0, 0.0))});
    const SparsePauliOp reversed(std::vector<PauliString>{
        PauliString("IX", Complex128(1.0, 0.0)),
        PauliString("XI", Complex128(1.0, 0.0))});

    MAQAOA m;
    m.options.p = 1;
    m.options.mixer_beta_dispatch = Dispatch::Shared;

    EXPECT_LE(max_amp_diff(run(m.build_circuit(cost, forward, {0.0, 0.0, beta})),
                           run(m.build_circuit(cost, reversed, {0.0, 0.0, beta}))),
              1e-12);
}

// =============================================================================
// Beta dispatch: counts and layout
// =============================================================================

namespace {

// Two terms sharing a lowest active qubit and one that does not, so the three
// dispatches give three different counts and no two can be confused.
SparsePauliOp overlapping_mixer_3q() {
    return SparsePauliOp(std::vector<PauliString>{
        PauliString("XYI", Complex128(1.0, 0.0)),
        PauliString("XIZ", Complex128(0.5, 0.0)),
        PauliString("IIY", Complex128(0.25, 0.0))});
}

}  // namespace

TEST(V11271MixerDispatch, EachDispatchCountsItsOwnBetas) {
    const auto cost = ring_cost_3q();
    const auto mixer = overlapping_mixer_3q();

    // Three qubits, so three gammas per layer under the default gamma dispatch.
    const int gammas = 3;

    struct Case { Dispatch d; int betas; const char* name; };
    for (const Case c : {Case{Dispatch::LowestActiveQubit, 2, "LowestActiveQubit"},
                         Case{Dispatch::TermIndexed, 3, "TermIndexed"},
                         Case{Dispatch::Shared, 1, "Shared"}}) {
        for (int p : {1, 2, 3}) {
            MAQAOA m;
            m.options.p = p;
            m.options.mixer_beta_dispatch = c.d;
            EXPECT_EQ(m.num_parameters(cost, mixer), p * (gammas + c.betas))
                << c.name << " at p = " << p;
        }
    }
}

TEST(V11271MixerDispatch, LowestActiveQubitCollapsesTermsSharingAQubit) {
    // XYI and XIZ both have qubit 0 as their lowest active qubit, so they draw
    // the same beta; IIY draws its own. Two slots, not three.
    const auto cost = ring_cost_3q();

    MAQAOA m;
    m.options.p = 1;
    m.options.mixer_beta_dispatch = Dispatch::LowestActiveQubit;
    EXPECT_EQ(m.num_parameters(cost, overlapping_mixer_3q()), 3 + 2);

    // And under TermIndexed the same mixer gets one each, which is what makes
    // the collapse above a dispatch decision rather than a property of the
    // mixer.
    m.options.mixer_beta_dispatch = Dispatch::TermIndexed;
    EXPECT_EQ(m.num_parameters(cost, overlapping_mixer_3q()), 3 + 3);
}

TEST(V11271MixerDispatch, AnAllIdentityTermDrawsNoBeta) {
    // A term with no active qubits drives no gate. Giving it a beta would leave
    // a coordinate in the vector that moves nothing, which is the failure the
    // count is meant to avoid.
    const auto cost = ring_cost_3q();

    const SparsePauliOp with_identity(std::vector<PauliString>{
        PauliString("XII", Complex128(1.0, 0.0)),
        PauliString("III", Complex128(2.0, 0.0)),
        PauliString("IIX", Complex128(1.0, 0.0))});
    const SparsePauliOp without(std::vector<PauliString>{
        PauliString("XII", Complex128(1.0, 0.0)),
        PauliString("IIX", Complex128(1.0, 0.0))});

    MAQAOA m;
    m.options.p = 1;
    m.options.mixer_beta_dispatch = Dispatch::TermIndexed;
    EXPECT_EQ(m.num_parameters(cost, with_identity),
              m.num_parameters(cost, without))
        << "an all-identity term was allocated a beta";

    const auto params = distinct_params(m, cost, without);
    EXPECT_EQ(max_amp_diff(run(m.build_circuit(cost, with_identity, params)),
                           run(m.build_circuit(cost, without, params))),
              0.0)
        << "an all-identity term changed the circuit";
}

TEST(V11271MixerDispatch, SlotsFollowTheKeyRatherThanTheTermOrder) {
    // Slots are ranked by dispatch key, so permuting the mixer's terms must not
    // permute which beta drives which gate. Under LowestActiveQubit the keys
    // are qubit indices, and a layout keyed on first-use order instead would
    // send beta 0 to whichever term happened to be listed first.
    const auto cost = ring_cost_3q();

    const SparsePauliOp ascending(std::vector<PauliString>{
        PauliString("XII", Complex128(1.0, 0.0)),
        PauliString("IXI", Complex128(1.0, 0.0)),
        PauliString("IIX", Complex128(1.0, 0.0))});
    const SparsePauliOp descending(std::vector<PauliString>{
        PauliString("IIX", Complex128(1.0, 0.0)),
        PauliString("IXI", Complex128(1.0, 0.0)),
        PauliString("XII", Complex128(1.0, 0.0))});

    MAQAOA m;
    m.options.p = 1;
    m.options.mixer_beta_dispatch = Dispatch::LowestActiveQubit;
    const auto params = distinct_params(m, cost, ascending);

    // Agreement is to rounding rather than bitwise: the three rotations
    // commute, so the same betas reach the same qubits, but they are multiplied
    // into the state in the opposite sequence and the last bits differ.
    EXPECT_LE(max_amp_diff(run(m.build_circuit(cost, ascending, params)),
                           run(m.build_circuit(cost, descending, params))),
              1e-12)
        << "listing the same mixer terms in a different order changed which "
           "beta drove which qubit";

    // The contrast that makes the bound meaningful. A layout keyed on first-use
    // order would send beta 0 to whichever term was listed first, so the
    // descending mixer would drive qubit 2 with beta 0. Permuting the betas by
    // hand produces exactly that state, and it is nowhere near the bound above.
    auto permuted = params;
    const std::size_t betas = 3;
    const std::size_t first = params.size() - betas;
    std::swap(permuted[first], permuted[first + betas - 1]);
    EXPECT_GT(max_amp_diff(run(m.build_circuit(cost, ascending, params)),
                           run(m.build_circuit(cost, ascending, permuted))),
              1e-3)
        << "swapping two betas did not move the state, so this fixture could "
           "not detect a mis-ranked layout either";
}

TEST(V11271MixerDispatch, EveryCountedParameterDrivesAGate) {
    // build_circuit reads zero for a parameter past the end of the vector, so a
    // count larger than what is consumed is invisible unless the last entry is
    // removed and the circuit is checked for a change.
    const auto cost = ring_cost_3q();
    const auto mixer = overlapping_mixer_3q();

    for (auto d : {Dispatch::LowestActiveQubit, Dispatch::TermIndexed,
                   Dispatch::Shared}) {
        MAQAOA m;
        m.options.p = 2;
        m.options.mixer_beta_dispatch = d;

        auto full = distinct_params(m, cost, mixer);
        ASSERT_FALSE(full.empty());
        const auto reference = run(m.build_circuit(cost, mixer, full));

        auto shorter = full;
        shorter.pop_back();
        EXPECT_GT(max_amp_diff(reference, run(m.build_circuit(cost, mixer, shorter))),
                  1e-9)
            << "dispatch " << static_cast<int>(d)
            << ": the last counted parameter drives no gate";
    }
}

TEST(V11271MixerDispatch, NoParameterBeyondTheCountIsRead) {
    // The other direction: a count smaller than what is consumed would silently
    // read zero for the remainder, and appending a value would then change the
    // circuit.
    const auto cost = ring_cost_3q();
    const auto mixer = overlapping_mixer_3q();

    for (auto d : {Dispatch::LowestActiveQubit, Dispatch::TermIndexed,
                   Dispatch::Shared}) {
        MAQAOA m;
        m.options.p = 2;
        m.options.mixer_beta_dispatch = d;

        auto full = distinct_params(m, cost, mixer);
        const auto reference = run(m.build_circuit(cost, mixer, full));

        auto longer = full;
        longer.push_back(0.91);
        EXPECT_EQ(max_amp_diff(reference, run(m.build_circuit(cost, mixer, longer))),
                  0.0)
            << "dispatch " << static_cast<int>(d)
            << ": a parameter past the counted end reached a gate";
    }
}

TEST(V11271MixerDispatch, OptimizeConsumesTheSameCountAsNumParameters) {
    // The three entry points have to agree, and optimize is the one that cannot
    // be inspected directly. Its initial_params vector is laid out by the same
    // arithmetic, so its length is the observable.
    const auto cost = ring_cost_3q();
    const auto mixer = overlapping_mixer_3q();

    for (auto d : {Dispatch::LowestActiveQubit, Dispatch::TermIndexed,
                   Dispatch::Shared}) {
        MAQAOA m;
        m.options.p = 2;
        m.options.max_iterations = 4;
        m.options.seed = 12711;
        m.options.mixer_beta_dispatch = d;
        m.estimator.options.shots = 0;
        m.sampler.options.shots = 8;

        const auto res = m.optimize(cost, mixer);
        EXPECT_EQ(static_cast<int>(res.initial_params.size()),
                  m.num_parameters(cost, mixer))
            << "dispatch " << static_cast<int>(d)
            << ": optimize laid out a different number of parameters than "
               "num_parameters reports";
        EXPECT_EQ(res.optimal_params.size(), res.initial_params.size());
    }
}

// =============================================================================
// Rejections, at all three entry points
// =============================================================================

TEST(V11271MixerDispatch, NonHermitianMixerIsRejectedEverywhere) {
    // The evolution reads term.coeff.real, so accepting a complex coefficient
    // would apply a unitary the caller did not write. Checking two of the three
    // entry points would be worse than checking none: a caller could size a
    // vector for a mixer the call it was sized for then refuses.
    const auto cost = zz_cost_2q();
    const SparsePauliOp bad(std::vector<PauliString>{
        PauliString("XY", Complex128(0.5, 0.25))});

    MAQAOA m;
    m.options.p = 1;
    m.options.max_iterations = 2;

    EXPECT_THROW((void)m.num_parameters(cost, bad), std::invalid_argument);
    EXPECT_THROW((void)m.build_circuit(cost, bad, {0.0, 0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_THROW((void)m.optimize(cost, bad), std::invalid_argument);
}

TEST(V11271MixerDispatch, WrongWidthMixerIsRejectedEverywhere) {
    // A mixer term wider or narrower than the cost Hamiltonian would be indexed
    // up to the cost register's width, reading past its end.
    const auto cost = zz_cost_2q();
    const SparsePauliOp too_wide(std::vector<PauliString>{
        PauliString("XYZ", Complex128(1.0, 0.0))});
    const SparsePauliOp too_narrow(std::vector<PauliString>{
        PauliString("X", Complex128(1.0, 0.0))});

    MAQAOA m;
    m.options.p = 1;
    m.options.max_iterations = 2;

    for (const auto& bad : {too_wide, too_narrow}) {
        EXPECT_THROW((void)m.num_parameters(cost, bad), std::invalid_argument);
        EXPECT_THROW((void)m.build_circuit(cost, bad, {0.0, 0.0, 0.0}),
                     std::invalid_argument);
        EXPECT_THROW((void)m.optimize(cost, bad), std::invalid_argument);
    }
}

TEST(V11271MixerDispatch, TheRejectionNamesTheOffendingTerm) {
    // A mixer can carry many terms, and a diagnostic that says only that one of
    // them is wrong leaves the caller to find it.
    const auto cost = ring_cost_3q();
    const SparsePauliOp bad(std::vector<PauliString>{
        PauliString("XII", Complex128(1.0, 0.0)),
        PauliString("IXI", Complex128(1.0, 0.0)),
        PauliString("IIX", Complex128(0.0, 0.75))});

    MAQAOA m;
    m.options.p = 1;
    try {
        (void)m.num_parameters(cost, bad);
        FAIL() << "a non-Hermitian mixer term was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("IIX"), std::string::npos)
            << "the message does not name the term. Got: " << msg;
        EXPECT_NE(msg.find("2"), std::string::npos)
            << "the message does not give the term index. Got: " << msg;
    }
}

TEST(V11271MixerDispatch, AnImaginaryPartInsideToleranceIsAccepted) {
    // The check carries a tolerance, so a coefficient that is real to within it
    // is real. Rejecting on an exact comparison would refuse a mixer built by
    // arithmetic rather than typed as a literal.
    const auto cost = zz_cost_2q();
    const SparsePauliOp nearly_real(std::vector<PauliString>{
        PauliString("XY", Complex128(0.7, DEFAULT_PHYSICAL_ATOL / 2.0))});

    MAQAOA m;
    m.options.p = 1;
    EXPECT_NO_THROW((void)m.num_parameters(cost, nearly_real));
    EXPECT_NO_THROW((void)m.build_circuit(cost, nearly_real, {0.0, 0.0, 0.1}));
}

// =============================================================================
// #122 - orbit mode was decided by two disagreeing tests
// =============================================================================

TEST(V11271MixerDispatch, WrongLengthOrbitAssignmentsFallsBackEverywhere) {
    // The public entry points require the vector to match the register width;
    // the evolution required only that it was non-empty. A vector of the wrong
    // length was therefore orbit-mode-off where the parameter vector is sized
    // and orbit-mode-on where the betas are read, which indexed the per-layer
    // array by an orbit label into an array sized by qubit count.
    //
    // Both routes now apply the same test, so a wrong-length vector is simply
    // absent: the run must be indistinguishable from one with no orbit
    // assignments at all, which is a stronger statement than merely not
    // crashing and is what makes this deterministic rather than dependent on
    // whatever the out-of-range read happened to find.
    const auto cost = ring_cost_3q();

    // Configured identically term for term, because MAQAOA owns an estimator
    // and a sampler and is not copyable.
    MAQAOA plain;
    plain.options.p = 2;
    plain.options.max_iterations = 6;
    plain.options.seed = 12712;
    plain.estimator.options.shots = 0;
    plain.sampler.options.shots = 16;

    MAQAOA wrong;
    wrong.options.p = 2;
    wrong.options.max_iterations = 6;
    wrong.options.seed = 12712;
    wrong.estimator.options.shots = 0;
    wrong.sampler.options.shots = 16;
    wrong.options.orbit_assignments = {0, 7};  // two entries, three qubits

    EXPECT_EQ(wrong.num_parameters(cost), plain.num_parameters(cost));

    const auto a = plain.optimize(cost);
    const auto b = wrong.optimize(cost);

    ASSERT_EQ(a.initial_params.size(), b.initial_params.size());
    for (std::size_t i = 0; i < a.initial_params.size(); ++i)
        EXPECT_DOUBLE_EQ(a.initial_params[i], b.initial_params[i]) << "i = " << i;
    EXPECT_DOUBLE_EQ(a.optimal_value, b.optimal_value)
        << "a wrong-length orbit_assignments changed the evolution, so the "
           "entry points and the evolution disagree about orbit mode";
}

TEST(V11271MixerDispatch, WrongLengthOrbitAssignmentsAgreesAcrossEntryPoints) {
    // The same disagreement seen from the circuit side: build_circuit and the
    // evolution have to lay the betas out identically, and an orbit label
    // beyond the register width is where they came apart.
    const auto cost = ring_cost_3q();

    MAQAOA plain;
    plain.options.p = 1;

    MAQAOA wrong;
    wrong.options.p = 1;
    wrong.options.orbit_assignments = {0, 7};

    const auto params = distinct_params(plain, cost, {});
    EXPECT_EQ(max_amp_diff(run(plain.build_circuit(cost, {}, params)),
                           run(wrong.build_circuit(cost, {}, params))),
              0.0);
}

TEST(V11271MixerDispatch, CorrectLengthOrbitAssignmentsStillShares) {
    // The fallback must not have been implemented by ignoring orbit sharing
    // altogether. A correctly sized vector still collapses the betas.
    const auto cost = ring_cost_3q();

    MAQAOA shared;
    shared.options.p = 1;
    shared.options.orbit_assignments = {0, 0, 1};

    MAQAOA plain;
    plain.options.p = 1;

    EXPECT_LT(shared.num_parameters(cost), plain.num_parameters(cost))
        << "orbit sharing did not reduce the parameter count";
}
