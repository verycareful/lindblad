// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.26.2 test wave - a channel and the gate it is attached to must agree.
//
// A one-qubit channel attached to a two-qubit gate was accepted without a word,
// and the density matrix backend then built its superoperator by reading each
// operator as a block sized by the GATE rather than by the channel. It read past
// the operators it was given and produced a state whose every entry was NaN,
// from a run that reported success. trace() and purity() both returned -nan and
// nothing anywhere said why.
//
// This is the first golden rule rather than an observation defect, and it was
// found by making the mistake rather than by looking for it, which is the
// argument for the check: the two arities are stated in different places and
// nothing compared them.
//
// Two checks, because they cover different cases and neither covers both.
// Naming the qubits fixes the width at the call site, so that mismatch is
// refused where it is written. An empty qubit list means whichever qubits the
// gate acts on, whose width is not known until an instruction is in hand, so
// that one is refused when the channel is resolved against the circuit.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "v11311_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 1e-9;

QuantumCircuit bell() {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    return qc;
}

// A channel whose width disagrees with its gate is a refusal before the first
// gate: exactly InvalidArgument from the run, naming the gate and where it
// sits, with no failed-run record, since no state was ever evolved.
void expect_width_refused(const NoiseModel& noise) {
    DensityMatrixSimulator sim;
    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)sim.run(bell(), noise, 8, 20261); });
    EXPECT_EQ(detail::failed_run_stores(), stores) << "a refusal before the first gate left a record";
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "DensityMatrixSimulator::run");
    v11311::expect_message(*e, {"DensityMatrixSimulator::run: the noise model's channel attached "
                                "to 'cx' acts on 1 qubit(s), but it is being applied to 2",
                                "(instruction 1: cx on qubits 0, 1)"});
    v11311::expect_point(e->where(), -1, 1, "cx", {0, 1});
}

}  // namespace

// =============================================================================
// Refused where the qubits are named
// =============================================================================

TEST(V11262NoiseWidth, AOneQubitChannelOnTwoNamedQubitsIsRefusedAtAttachment) {
    NoiseModel noise;
    EXPECT_THROW(
        noise.add_quantum_error(NoiseChannels::depolarizing(0.1), "cx", {0, 1}),
        std::invalid_argument);
}

TEST(V11262NoiseWidth, ATwoQubitChannelOnOneNamedQubitIsRefusedAtAttachment) {
    // The other direction. A channel wider than the qubits it is given is the
    // same mistake and is equally silent without a check.
    NoiseModel noise;
    EXPECT_THROW(
        noise.add_quantum_error(NoiseChannels::depolarizing(0.1, 2), "h", {0}),
        std::invalid_argument);
}

TEST(V11262NoiseWidth, TheRefusalNamesTheGateAndBothWidths) {
    NoiseModel noise;
    try {
        noise.add_quantum_error(NoiseChannels::depolarizing(0.1), "cx", {0, 1});
        FAIL() << "a one-qubit channel was attached to two qubits";
    } catch (const std::invalid_argument& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cx"), std::string::npos) << message;
        EXPECT_NE(message.find("1"), std::string::npos) << message;
        EXPECT_NE(message.find("2"), std::string::npos) << message;
    }
}

TEST(V11262NoiseWidth, MatchingWidthsAttachAtEveryArity) {
    // The check must not refuse the ordinary case at any width.
    NoiseModel noise;
    EXPECT_NO_THROW(noise.add_quantum_error(NoiseChannels::depolarizing(0.1), "h", {0}));
    EXPECT_NO_THROW(
        noise.add_quantum_error(NoiseChannels::depolarizing(0.1, 2), "cx", {0, 1}));
    EXPECT_NO_THROW(noise.add_quantum_error(NoiseChannels::amplitude_damping(0.05), "x", {1}));
    EXPECT_NO_THROW(noise.add_quantum_error(NoiseChannels::phase_damping(0.05), "z", {0}));
}

TEST(V11262NoiseWidth, AnUnqualifiedAttachmentIsStillAccepted) {
    // An empty qubit list is not a mismatch, it is a deferral: the width comes
    // from whichever instruction the channel meets.
    NoiseModel noise;
    EXPECT_NO_THROW(noise.add_quantum_error(NoiseChannels::depolarizing(0.1), "h"));
    EXPECT_NO_THROW(noise.add_quantum_error(NoiseChannels::depolarizing(0.1, 2), "cx"));
}

// =============================================================================
// Refused when the deferred width turns out to disagree
// =============================================================================

TEST(V11262NoiseWidth, AnUnqualifiedOneQubitChannelOnATwoQubitGateFailsTheRun) {
    // The case that produced the NaN. Attachment cannot catch it, because the
    // arity is not knowable from the gate name alone for every gate.
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.2), "h");
    noise.add_quantum_error(NoiseChannels::depolarizing(0.1), "cx");

    expect_width_refused(noise);
}

TEST(V11262NoiseWidth, TheRunNeverReturnsANonFiniteState) {
    // What the defect looked like from outside was a NaN state handed back as
    // an answer. The run now refuses before any state exists, so there is no
    // state to return and none in a failed-run record either.
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.2), "h");
    noise.add_quantum_error(NoiseChannels::depolarizing(0.1), "cx");

    expect_width_refused(noise);
}

TEST(V11262NoiseWidth, TheCorrectlySizedChannelRunsAndMixesTheState) {
    // The same circuit with the arity the caller meant. One argument apart from
    // the test above, and the difference between a result and NaN.
    NoiseModel noise;
    noise.add_quantum_error(NoiseChannels::depolarizing(0.2), "h");
    noise.add_quantum_error(NoiseChannels::depolarizing(0.1, 2), "cx");

    DensityMatrixSimulator sim;
    auto r = sim.run(bell(), noise, 8, 20261);

    EXPECT_NEAR(r.final_state.trace(), 1.0, kTol);
    EXPECT_TRUE(std::isfinite(r.final_state.purity()));
    // Noise mixes it, so this is strictly below a pure state's 1.
    EXPECT_LT(r.final_state.purity(), 1.0);
    EXPECT_GT(r.final_state.purity(), 0.0);
}

TEST(V11262NoiseWidth, AnIdealRunIsUnaffectedByTheCheck) {
    const NoiseModel noise;
    DensityMatrixSimulator sim;
    auto r = sim.run(bell(), noise, 8, 20261);

    EXPECT_NEAR(r.final_state.trace(), 1.0, kTol);
    EXPECT_NEAR(r.final_state.purity(), 1.0, kTol);
}
