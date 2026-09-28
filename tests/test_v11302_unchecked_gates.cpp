// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.2 - an MPS keeps its records straight after a gate that is not
// unitary.
//
// Both MPS layers read the norm, the marginals and every measurement at the
// orthogonality centre, trusting every site outside the open span to be
// orthonormal, and derive their fidelity bound for unitary gates between
// splits. A single-site matrix that is not unitary, applied to a site outside
// the span, broke the first while the span went on claiming it; any gate that
// is not unitary broke the second while the figures went on reporting. Both
// arose only from a caller's matrix: under Ignore, under Warn, or inside a wide
// atol.
//
// The chain now learns whether each gate it applies is unitary to
// DEFAULT_PHYSICAL_ATOL and keeps its records by the answer: a non-unitary
// single-site gate off the centre widens the span over its site, and any
// non-unitary gate empties the fidelity figures. The matrix is applied as given
// and nothing is reported. Under Throw, Warn or Repair::Attempt the policy's own
// measurement answers; under Ignore with Repair::None the chain's
// UncheckedGates setting decides: Track (the default) measures for the
// records, AssumeUnitary measures nothing and accepts wrong reads.
//
// Every expectation is derived from the construction: the dense contraction,
// which ignores the span, is the reference for what the chain holds.

#include <gtest/gtest.h>

#include "v11301_mps_oracle.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;
using v11301::Amplitudes;
using v11301::kEps;
using v11301::kSlack;

namespace {

using Span = std::pair<int, int>;

// Off unitarity by far more than the default tolerance and far less than a
// caller's widened one, both read from the canonical default.
constexpr double kOffUnitary = 1e3 * DEFAULT_PHYSICAL_ATOL;
constexpr double kWideAtol = 1e6 * DEFAULT_PHYSICAL_ATOL;

// diag(a, b): unitary only when |a| = |b| = 1.
std::array<Complex128, 4> diag2(double a, double b) {
    return {Complex128(a, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(b, 0.0)};
}

std::array<Complex128, 4> hadamard() {
    return {Complex128(INV_SQRT2, 0.0), Complex128(INV_SQRT2, 0.0),
            Complex128(INV_SQRT2, 0.0), Complex128(-INV_SQRT2, 0.0)};
}

// Keeps basis state |00> and deletes the others: plainly not unitary.
std::array<Complex128, 16> projector_00() {
    std::array<Complex128, 16> u{};
    u[0] = Complex128(1.0, 0.0);
    return u;
}

// A chain with H on `site` and the centre left on site 0, so `site` sits
// outside the open span.
MPSState h_on(int n, int site) {
    MPSState s(n, 8);
    s.apply_single_qubit_gate(hadamard(), site);
    return s;
}

class WarningCapture {
public:
    WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
        set_warning_handler([this](const std::string& m) { lines_.push_back(m); });
    }
    ~WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
    }
    std::size_t count() const { return lines_.size(); }

private:
    std::vector<std::string> lines_;
};

double norm_tol(double norm_sq) { return kSlack * kEps * norm_sq; }

}  // namespace

// =============================================================================
// Defaults
// =============================================================================

TEST(V11302UncheckedGates, EveryChainAndTheSimulatorDefaultToTrack) {
    EXPECT_EQ(MPSState(3).unchecked_gates, UncheckedGates::Track);
    EXPECT_EQ(QuditMPS(3, 3).unchecked_gates, UncheckedGates::Track);
    EXPECT_EQ(MPSSimulator().unchecked_gates, UncheckedGates::Track);
    EXPECT_STREQ(to_string(UncheckedGates::Track), "Track");
    EXPECT_STREQ(to_string(UncheckedGates::AssumeUnitary), "AssumeUnitary");
}

// =============================================================================
// Track: the chain reads what it holds
// =============================================================================

TEST(V11302UncheckedGates, ANonUnitaryGateOffTheCentreIsReadAtItsTrueNorm) {
    // H on qubit 2 then diag(2, 1) on it under Ignore: qubit 2 holds
    // (2|0> + |1>) / sqrt(2), so the raw marginals are {2, 1/2} and the norm²
    // is their sum, 5/2. The centre is on 0 throughout, so without the span
    // widening over site 2 every read below would see norm² 1.
    const int n = 3;
    const double a0 = 2.0, a1 = 1.0;
    MPSState s = h_on(n, 2);
    ASSERT_EQ(s.open_span(), (Span{0, 0}));
    s.apply_single_qubit_gate(diag2(a0, a1), 2, {Validation::Ignore});
    EXPECT_EQ(s.open_span(), (Span{0, 2})) << "the span must take in site 2";

    const double p0 = a0 * a0 * INV_SQRT2 * INV_SQRT2;
    const double p1 = a1 * a1 * INV_SQRT2 * INV_SQRT2;
    const Amplitudes dense = v11301::dense(s);
    ASSERT_NEAR(v11301::norm_sq(dense), p0 + p1, norm_tol(p0 + p1))
        << "the dense reference itself disagrees with the construction";
    EXPECT_NEAR(s.norm_sq(), p0 + p1, norm_tol(p0 + p1));
    const auto marg = s.probabilities_single(2);
    EXPECT_NEAR(marg[0], p0, norm_tol(p0 + p1));
    EXPECT_NEAR(marg[1], p1, norm_tol(p0 + p1));
}

TEST(V11302UncheckedGates, ANonUnitaryGateOnTheCentreLeavesTheSpan) {
    // Site 0 is the centre, already inside the span, so nothing widens; the
    // fidelity figures still empty.
    MPSState s(3, 8);
    s.apply_single_qubit_gate(diag2(2.0, 1.0), 0, {Validation::Ignore});
    EXPECT_EQ(s.open_span(), (Span{0, 0}));
    EXPECT_FALSE(s.fidelity_estimate().has_value());
    EXPECT_FALSE(s.fidelity_lower_bound().has_value());
}

TEST(V11302UncheckedGates, TheFiguresEmptyAfterAnyNonUnitaryGate) {
    {
        MPSState s = h_on(3, 2);
        s.apply_single_qubit_gate(diag2(2.0, 1.0), 2, {Validation::Ignore});
        EXPECT_FALSE(s.fidelity_estimate().has_value()) << "single-site";
    }
    {
        MPSState s = h_on(3, 0);
        s.apply_two_qubit_gate(projector_00(), 0, 1, {Validation::Ignore});
        EXPECT_FALSE(s.fidelity_estimate().has_value()) << "two-site";
        EXPECT_FALSE(s.fidelity_lower_bound().has_value()) << "two-site";
    }
}

TEST(V11302UncheckedGates, UnitaryGatesKeepTheSpanAndTheFigures) {
    // The same routes with unitary matrices, under Ignore so that Track is the
    // one measuring: nothing widens and the figures survive.
    MPSState s = h_on(3, 2);
    s.apply_single_qubit_gate(hadamard(), 1, {Validation::Ignore});
    EXPECT_EQ(s.open_span(), (Span{0, 0}));
    s.apply_two_qubit_gate(v11301::as_gate4(v11301::random_unitary(4, 7301)), 0, 1,
                           {Validation::Ignore});
    EXPECT_TRUE(s.fidelity_estimate().has_value());
    EXPECT_TRUE(s.fidelity_lower_bound().has_value());
}

// =============================================================================
// AssumeUnitary: nothing is measured
// =============================================================================

TEST(V11302UncheckedGates, AssumeUnitaryAppliesTheGateAndKeepsTheRecords) {
    // The matrix lands exactly as under Track, and the records are left as
    // they were: the span does not widen and the figures stay. What the reads
    // then say is the caller's to accept, so it is not pinned here.
    MPSState tracked = h_on(3, 2);
    MPSState assumed = h_on(3, 2);
    assumed.unchecked_gates = UncheckedGates::AssumeUnitary;
    tracked.apply_single_qubit_gate(diag2(2.0, 1.0), 2, {Validation::Ignore});
    assumed.apply_single_qubit_gate(diag2(2.0, 1.0), 2, {Validation::Ignore});

    EXPECT_EQ(assumed.open_span(), (Span{0, 0}));
    EXPECT_TRUE(assumed.fidelity_estimate().has_value());
    const Amplitudes want = v11301::dense(tracked);
    EXPECT_LE(v11301::max_abs_diff(v11301::dense(assumed), want),
              v11301::amplitude_tol(4, want.size(), std::sqrt(v11301::norm_sq(want))))
        << "AssumeUnitary must apply the same matrix";

    MPSState pair = h_on(3, 0);
    pair.unchecked_gates = UncheckedGates::AssumeUnitary;
    pair.apply_two_qubit_gate(projector_00(), 0, 1, {Validation::Ignore});
    EXPECT_TRUE(pair.fidelity_estimate().has_value());
}

// =============================================================================
// Policies that measure answer for the records whatever the setting
// =============================================================================

TEST(V11302UncheckedGates, WarnAndAWideAtolKeepTheRecordsUnderEitherSetting) {
    for (const UncheckedGates setting : {UncheckedGates::Track, UncheckedGates::AssumeUnitary}) {
        SCOPED_TRACE(to_string(setting));
        {
            MPSState s = h_on(3, 2);
            s.unchecked_gates = setting;
            WarningCapture cap;
            s.apply_single_qubit_gate(diag2(2.0, 1.0), 2, {Validation::Warn});
            EXPECT_EQ(cap.count(), 1u) << "Warn reports the operand once";
            EXPECT_EQ(s.open_span(), (Span{0, 2}));
            EXPECT_FALSE(s.fidelity_estimate().has_value());
        }
        {
            // Inside the caller's atol, so accepted, yet off by more than the
            // default: the chain must still stop trusting the site.
            MPSState s = h_on(3, 2);
            s.unchecked_gates = setting;
            EXPECT_NO_THROW(s.apply_single_qubit_gate(diag2(1.0 + kOffUnitary, 1.0), 2,
                                                      {Validation::Throw, kWideAtol}));
            EXPECT_EQ(s.open_span(), (Span{0, 2}));
            EXPECT_FALSE(s.fidelity_estimate().has_value());
        }
        {
            // Repair::Attempt applies the polar projection, which is unitary,
            // so there is nothing to record.
            MPSState s = h_on(3, 2);
            s.unchecked_gates = setting;
            s.apply_single_qubit_gate(diag2(1.0 + kOffUnitary, 1.0), 2,
                                      {Validation::Ignore, DEFAULT_PHYSICAL_ATOL,
                                       Repair::Attempt});
            EXPECT_EQ(s.open_span(), (Span{0, 0}));
            EXPECT_TRUE(s.fidelity_estimate().has_value());
        }
    }
}

// =============================================================================
// A collapse is not a gate
// =============================================================================

TEST(V11302UncheckedGates, ACollapseEmptiesTheFiguresUnderEitherSetting) {
    for (const UncheckedGates setting : {UncheckedGates::Track, UncheckedGates::AssumeUnitary}) {
        SCOPED_TRACE(to_string(setting));
        MPSState s = h_on(3, 1);
        s.unchecked_gates = setting;
        ASSERT_TRUE(s.fidelity_estimate().has_value());
        std::mt19937_64 rng(11);
        s.measure_qubit(1, rng);
        EXPECT_FALSE(s.fidelity_estimate().has_value());

        QuditMPS q(3, 3);
        q.unchecked_gates = setting;
        std::mt19937_64 rq(12);
        q.measure_qudit(1, rq);
        EXPECT_FALSE(q.fidelity_estimate().has_value());
    }
}

// =============================================================================
// Through MPSSimulator::run
// =============================================================================

namespace {

// H on qubit 2, then the caller's diag(2, 1) on it with `policy`, and nothing
// that moves the centre off site 0: the dense result has norm² 5/2.
QuantumCircuit scaled_qubit_circuit(ValidationOptions policy) {
    QuantumCircuit qc(3);
    qc.h(2);
    const auto m = diag2(2.0, 1.0);
    qc.unitary(std::vector<Complex128>(m.begin(), m.end()), {2}, "scale", policy);
    return qc;
}

}  // namespace

TEST(V11302UncheckedGates, ARunJudgesACircuitMatrixForTheRecordsUnderItsOwnPolicy) {
    const double want = (4.0 + 1.0) * INV_SQRT2 * INV_SQRT2;
    MPSSimulator sim;
    const auto r = sim.run(scaled_qubit_circuit({Validation::Ignore}), 8, 0, 1);
    EXPECT_NEAR(r.final_state.norm_sq(), want, norm_tol(want))
        << "Track is the default on the simulator's chains";
    EXPECT_FALSE(r.final_state.fidelity_estimate().has_value());

    MPSSimulator trusting;
    trusting.unchecked_gates = UncheckedGates::AssumeUnitary;
    const auto t = trusting.run(scaled_qubit_circuit({Validation::Ignore}), 8, 0, 1);
    EXPECT_EQ(t.final_state.unchecked_gates, UncheckedGates::AssumeUnitary);
    EXPECT_TRUE(t.final_state.fidelity_estimate().has_value())
        << "AssumeUnitary measures an Ignore'd matrix in a run no more than outside one";

    // Both the builder and the pre-flight report a Warn'd matrix; what matters
    // here is the chain's records, so the reports are only kept off the log.
    WarningCapture quiet;
    const auto w = trusting.run(scaled_qubit_circuit({Validation::Warn}), 8, 0, 1);
    EXPECT_GE(quiet.count(), 1u);
    EXPECT_NEAR(w.final_state.norm_sq(), want, norm_tol(want))
        << "a Warn'd matrix answers for the records whatever the setting";
    EXPECT_FALSE(w.final_state.fidelity_estimate().has_value());
}

TEST(V11302UncheckedGates, LibraryGatesInARunNeverTouchTheRecords) {
    // A brickwork of named gates: every gate is unitary by construction, so
    // nothing in the run empties the figures.
    QuantumCircuit qc(5);
    for (int layer = 0; layer < 3; ++layer) {
        for (int q = 0; q < 5; ++q) qc.ry(0.3 + 0.1 * q + layer, q).t(q);
        for (int q = layer % 2; q + 1 < 5; q += 2) qc.cx(q, q + 1);
    }
    qc.ccx(0, 2, 4);
    MPSSimulator sim;
    const auto r = sim.run(qc, 64, 0, 1);
    EXPECT_TRUE(r.final_state.fidelity_estimate().has_value());
    EXPECT_TRUE(r.final_state.fidelity_lower_bound().has_value());
}

// =============================================================================
// The qudit chain
// =============================================================================

namespace {

std::vector<Complex128> diag3(double a, double b, double c) {
    std::vector<Complex128> m(9, Complex128(0.0, 0.0));
    m[0] = Complex128(a, 0.0);
    m[4] = Complex128(b, 0.0);
    m[8] = Complex128(c, 0.0);
    return m;
}

// The qudit Fourier gate F|x> = sum_k w^(xk) |k> / sqrt(3): spreads |0> evenly.
std::vector<Complex128> fourier3() {
    const double r = 1.0 / SQRT3;
    std::vector<Complex128> m(9);
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            const double phase = TWO_PI * j * k / 3.0;
            m[static_cast<std::size_t>(j * 3 + k)] =
                Complex128(r * std::cos(phase), r * std::sin(phase));
        }
    return m;
}

}  // namespace

TEST(V11302UncheckedGates, TheQuditChainKeepsTheSameRecords) {
    // F on qudit 2 (weights 1/3 each), then diag(2, 1, 1) under Ignore: the
    // norm² is (4 + 1 + 1) / 3 = 2, read at the centre on site 0 only if the
    // span took in site 2.
    const int n = 3, d = 3;
    const double want = (4.0 + 1.0 + 1.0) / 3.0;
    QuditMPS s(n, d, 9);
    s.apply_1qudit(2, fourier3());
    ASSERT_EQ(s.open_span(), (Span{0, 0}));
    s.apply_1qudit(2, diag3(2.0, 1.0, 1.0), {Validation::Ignore});
    EXPECT_EQ(s.open_span(), (Span{0, 2}));
    EXPECT_NEAR(v11301::norm_sq(v11301::dense(s)), want, norm_tol(want));
    EXPECT_NEAR(s.norm_sq(), want, norm_tol(want));
    EXPECT_FALSE(s.fidelity_estimate().has_value());

    QuditMPS assumed(n, d, 9);
    assumed.unchecked_gates = UncheckedGates::AssumeUnitary;
    assumed.apply_1qudit(2, fourier3());
    assumed.apply_1qudit(2, diag3(2.0, 1.0, 1.0), {Validation::Ignore});
    EXPECT_EQ(assumed.open_span(), (Span{0, 0}));
    EXPECT_TRUE(assumed.fidelity_estimate().has_value());

    // Two-site gates empty the figures under Track and not under AssumeUnitary.
    std::vector<Complex128> keep00(81, Complex128(0.0, 0.0));
    keep00[0] = Complex128(1.0, 0.0);
    QuditMPS pair(n, d, 9);
    pair.apply_2qudit(2, 0, keep00, {Validation::Ignore});
    EXPECT_FALSE(pair.fidelity_estimate().has_value());
    QuditMPS adjacent(n, d, 9);
    adjacent.apply_2qudit_adjacent(1, keep00, {Validation::Ignore});
    EXPECT_FALSE(adjacent.fidelity_estimate().has_value());
    QuditMPS trusted(n, d, 9);
    trusted.unchecked_gates = UncheckedGates::AssumeUnitary;
    trusted.apply_2qudit(2, 0, keep00, {Validation::Ignore});
    EXPECT_TRUE(trusted.fidelity_estimate().has_value());
}
