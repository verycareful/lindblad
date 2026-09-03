// 1.1.26.1 test wave - StabilizerState::to_statevector.
//
// The tableau had no route to dense amplitudes before this release. The one it
// gained applies the stabilizer projector to a support point taken from the
// outcome slab, holding intermediate terms sparsely so the working set is the
// state's own support rather than 2^n.
//
// Two things can go wrong in a way that survives a casual look. The phases: a
// stabilizer generator carrying Y contributes factors of i, and getting those
// wrong leaves the probabilities correct while the amplitudes are not, so every
// case here is checked as AMPLITUDES against the statevector simulator running
// the same circuit, not as a distribution. And the support: a projector applied
// from the wrong point produces a normalised state supported on the wrong
// coset, which again looks entirely reasonable on its own.
//
// The comparison is up to global phase, because the two backends have no reason
// to agree on one and neither claims to.

#include <gtest/gtest.h>

#include "v11261_observation_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 1e-9;

// The tableau's dense amplitudes and the statevector backend's, for the same
// circuit, compared up to one global phase.
void expect_same_state_up_to_phase(const QuantumCircuit& qc) {
    CliffordSimulator clifford;
    auto cliff = clifford.run(qc, 0, 20261);
    const Statevector from_tableau = cliff.final_state.to_statevector();

    StatevectorSimulator sv;
    auto dense = sv.run(qc, 0, 20261);
    ASSERT_TRUE(dense.success) << dense.error_message;

    const std::vector<Complex128> a = from_tableau.amplitudes();
    const std::vector<Complex128> b = dense.final_state.amplitudes();
    ASSERT_EQ(a.size(), b.size());

    // Fix the global phase on the largest entry, which is the numerically
    // safest place to divide. Complex128 carries no complex division, so the
    // quotient is written out as a * conj(b) / |b|^2.
    std::size_t pivot = 0;
    for (std::size_t i = 1; i < b.size(); ++i) {
        if (b[i].norm() > b[pivot].norm()) pivot = i;
    }
    ASSERT_GT(b[pivot].norm(), kTol);

    const Complex128 phase = (a[pivot] * b[pivot].conj()) / b[pivot].norm_sq();
    EXPECT_NEAR(phase.norm(), 1.0, kTol) << "the two states differ in norm";

    for (std::size_t i = 0; i < a.size(); ++i) {
        const Complex128 diff = a[i] - phase * b[i];
        EXPECT_NEAR(diff.norm(), 0.0, kTol) << "amplitude " << i;
    }
}

}  // namespace

TEST(V11261StabilizerDense, TheAllZeroStateIsRecovered) {
    expect_same_state_up_to_phase(QuantumCircuit(3));
}

TEST(V11261StabilizerDense, ABellPairIsRecovered) {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    expect_same_state_up_to_phase(qc);
}

TEST(V11261StabilizerDense, AGhzStateIsRecovered) {
    QuantumCircuit qc(4);
    qc.h(0);
    qc.cx(0, 1);
    qc.cx(1, 2);
    qc.cx(2, 3);
    expect_same_state_up_to_phase(qc);
}

TEST(V11261StabilizerDense, AYStabilizedQubitIsRecoveredWithItsFactorOfI) {
    // H then S is (|0> + i|1>)/sqrt(2), stabilized by Y. The imaginary unit is
    // the part a probabilities-only check would never see.
    QuantumCircuit qc(1);
    qc.h(0);
    qc.s(0);
    expect_same_state_up_to_phase(qc);
}

TEST(V11261StabilizerDense, SeveralYStabilizedQubitsAreRecovered) {
    QuantumCircuit qc(3);
    for (int q = 0; q < 3; ++q) {
        qc.h(q);
        qc.s(q);
    }
    expect_same_state_up_to_phase(qc);
}

TEST(V11261StabilizerDense, AnEntangledStateCarryingYIsRecovered) {
    QuantumCircuit qc(3);
    qc.h(0);
    qc.s(0);
    qc.cx(0, 1);
    qc.h(2);
    qc.cz(1, 2);
    qc.sdg(1);
    expect_same_state_up_to_phase(qc);
}

TEST(V11261StabilizerDense, AProductOfSingleQubitStatesIsRecovered) {
    QuantumCircuit qc(4);
    qc.x(0);
    qc.h(1);
    qc.h(2);
    qc.s(2);
    qc.z(3);
    expect_same_state_up_to_phase(qc);
}

TEST(V11261StabilizerDense, TheAmplitudesAreNormalised) {
    QuantumCircuit qc(4);
    qc.h(0);
    qc.cx(0, 1);
    qc.h(2);
    qc.s(2);
    qc.cx(2, 3);

    CliffordSimulator sim;
    auto r = sim.run(qc, 0, 20261);
    const Statevector dense = r.final_state.to_statevector();

    EXPECT_NEAR(dense.norm(), 1.0, kTol);
}

TEST(V11261StabilizerDense, ReadingTheTableauDoesNotChangeIt) {
    // The projector walks the generators, and a walk that consumed them would
    // leave the state usable-looking and wrong for every later reader.
    QuantumCircuit qc(3);
    qc.h(0);
    qc.s(0);
    qc.cx(0, 1);
    qc.cx(1, 2);

    CliffordSimulator sim;
    auto r = sim.run(qc, 0, 20261);

    const Statevector first = r.final_state.to_statevector();
    const double entropy_first = r.final_state.entanglement_entropy_bits({0});
    const Statevector second = r.final_state.to_statevector();
    const double entropy_second = r.final_state.entanglement_entropy_bits({0});

    const std::vector<Complex128> a = first.amplitudes();
    const std::vector<Complex128> b = second.amplitudes();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR((a[i] - b[i]).norm(), 0.0, kTol) << "amplitude " << i;
    }
    EXPECT_NEAR(entropy_first, entropy_second, kTol);
}

TEST(V11261StabilizerDense, ATooWideRegisterIsRefusedRatherThanAttempted) {
    // The amplitudes of a 63-qubit state cannot be indexed, let alone
    // allocated, so this refuses instead of trying and failing on the way.
    const StabilizerState wide(63);
    EXPECT_THROW(wide.to_statevector(), std::exception);
}

TEST(V11261StabilizerDense, TheDenseAmplitudesMatchWhatSamplingProduces) {
    // The other direction: the amplitudes' probabilities have to describe the
    // distribution the tableau's own sampling draws from, since both claim to
    // describe one state.
    QuantumCircuit qc(3, 3);
    qc.h(0);
    qc.cx(0, 1);
    qc.h(2);
    qc.s(2);

    CliffordSimulator sim;
    auto state_run = sim.run(qc, 0, 20261);
    const Statevector dense = state_run.final_state.to_statevector();

    QuantumCircuit measured = qc;
    measured.measure_all();
    const int shots = 40000;
    auto counts_run = sim.run(measured, shots, 20261);

    for (const auto& [bits, count] : counts_run.counts) {
        // The key reads as a big-endian integer, which IS the amplitude index.
        const std::size_t index = std::stoul(bits, nullptr, 2);
        const double observed = static_cast<double>(count) / shots;
        EXPECT_NEAR(observed, dense.probability(index), 0.02) << bits;
    }
}
