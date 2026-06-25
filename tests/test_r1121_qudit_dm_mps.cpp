// R.1.12.1 total-coverage suite, Batch 4: QuditDensityMatrix and QuditMPS.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 4".
//
// Density matrix: |0><0| init, statevector lift, unitary purity/trace
// preservation, Kraus trace preservation, partial trace, measurement. MPS:
// to_statevector vs the dense reference, dense round-trip, adjacent and
// non-adjacent two-qudit gates, norm, measurement. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace lindblad;

namespace {

void expect_states_close(const QuditStatevector& a, const QuditStatevector& b,
                         double tol = 1e-7) {
    ASSERT_EQ(a.dim, b.dim);
    for (size_t i = 0; i < a.dim; ++i) {
        EXPECT_NEAR(a.amplitudes[i].real, b.amplitudes[i].real, tol) << "re @ " << i;
        EXPECT_NEAR(a.amplitudes[i].imag, b.amplitudes[i].imag, tol) << "im @ " << i;
    }
}

QuditStatevector entangled_state(int n, int d) {
    QuditStatevector sv(n, d);
    sv.apply_1qudit(0, qudit_gates::qft_matrix(d));
    for (int q = 0; q + 1 < n; ++q)
        sv.apply_2qudit(q, q + 1, qudit_gates::cadd_matrix(d, 1));
    return sv;
}

}  // namespace

// =============================================================================
// QuditDensityMatrix
// =============================================================================

TEST(R1121QuditDM, InitialiseIsPureGroundState) {
    QuditDensityMatrix rho(2, 3);
    EXPECT_NEAR(rho.trace(), 1.0, 1e-12);
    EXPECT_NEAR(rho.purity(), 1.0, 1e-12);
    EXPECT_NEAR(rho.rho[0].real, 1.0, 1e-12);  // |00><00|
}

TEST(R1121QuditDM, FromStatevectorHasDiagonalProbabilities) {
    auto sv = entangled_state(2, 3);
    QuditDensityMatrix rho(sv);
    EXPECT_NEAR(rho.trace(), 1.0, 1e-9);
    for (size_t i = 0; i < sv.dim; ++i) {
        double p = sv.amplitudes[i].real * sv.amplitudes[i].real +
                   sv.amplitudes[i].imag * sv.amplitudes[i].imag;
        EXPECT_NEAR(rho.rho[i * sv.dim + i].real, p, 1e-9);
    }
}

TEST(R1121QuditDM, UnitaryEvolutionPreservesTraceAndPurity) {
    QuditDensityMatrix rho(1, 3);
    rho.apply_1qudit(0, qudit_gates::qft_matrix(3));
    EXPECT_NEAR(rho.trace(), 1.0, 1e-9);
    EXPECT_NEAR(rho.purity(), 1.0, 1e-9) << "unitary keeps the state pure";
}

TEST(R1121QuditDM, KrausChannelPreservesTraceReducesPurity) {
    QuditDensityMatrix rho(1, 3);
    rho.apply_1qudit(0, qudit_gates::qft_matrix(3));  // pure superposition
    auto chan = QuditNoiseModel::depolarizing_channel(3, 0.3);
    rho.apply_kraus_1qudit(0, chan.ops);
    EXPECT_NEAR(rho.trace(), 1.0, 1e-7) << "CPTP channel is trace preserving";
    EXPECT_LT(rho.purity(), 1.0) << "depolarising decreases purity";
}

TEST(R1121QuditDM, PartialTraceKeepsSubsystem) {
    auto sv = entangled_state(2, 3);
    QuditDensityMatrix rho(sv);
    auto reduced = rho.partial_trace({0});
    EXPECT_EQ(reduced.n_qudits, 1);
    EXPECT_EQ(reduced.d, 3);
    EXPECT_NEAR(reduced.trace(), 1.0, 1e-9);
    EXPECT_THROW(rho.partial_trace({}), std::invalid_argument);
}

TEST(R1121QuditDM, MeasurementIsSeedDeterministic) {
    QuditDensityMatrix rho(2, 3);
    rho.apply_1qudit(0, qudit_gates::qft_matrix(3));
    QuditDensityMatrix copy = rho;  // measure collapses, so use independent copies
    EXPECT_EQ(rho.measure(77), copy.measure(77));
}

TEST(R1121QuditDM, LindbladStepPreservesTraceApproximately) {
    QuditDensityMatrix rho(1, 3);
    rho.apply_1qudit(0, qudit_gates::shift_matrix(3, 1));  // |1>
    std::vector<QuditLindbladOp> ops = {
        QuditNoiseModel::amplitude_damping_lindblad(3, 0.5)};
    rho.apply_lindblad_step(0, ops, 0.01);
    EXPECT_NEAR(rho.trace(), 1.0, 1e-3) << "Lindblad step is trace preserving (Euler O(dt^2))";
}

TEST(R1121QuditDM, EvolutionMatchesStatevectorProbabilities) {
    // Evolving the density matrix with apply_1qudit/apply_2qudit must track the
    // statevector probabilities exactly (no noise, so populations = |amp|^2).
    const int d = 3;
    QuditDensityMatrix rho(2, d);
    QuditStatevector sv(2, d);
    auto F = qudit_gates::qft_matrix(d);
    auto cadd = qudit_gates::cadd_matrix(d, 1);
    rho.apply_1qudit(0, F);     sv.apply_1qudit(0, F);
    rho.apply_2qudit(0, 1, cadd); sv.apply_2qudit(0, 1, cadd);
    rho.apply_1qudit(1, qudit_gates::shift_matrix(d, 1));
    sv.apply_1qudit(1, qudit_gates::shift_matrix(d, 1));
    for (size_t i = 0; i < sv.dim; ++i) {
        double p = sv.amplitudes[i].real * sv.amplitudes[i].real +
                   sv.amplitudes[i].imag * sv.amplitudes[i].imag;
        EXPECT_NEAR(rho.rho[i * sv.dim + i].real, p, 1e-9) << "population " << i;
    }
    EXPECT_NEAR(rho.purity(), 1.0, 1e-7) << "unitary evolution stays pure";
}

TEST(R1121QuditDM, AmplitudeDampingLindbladDrivesPopulationDown) {
    // Amplitude damping lowers |1> toward |0>: P(|1>) decreases monotonically
    // and P(|0>) grows, while the trace is preserved across many small steps.
    const int d = 3;
    QuditDensityMatrix rho(1, d);
    rho.apply_1qudit(0, qudit_gates::shift_matrix(d, 1));  // |1>
    std::vector<QuditLindbladOp> ops = {
        QuditNoiseModel::amplitude_damping_lindblad(d, 0.8)};
    double prev_excited = rho.rho[1 * d + 1].real;  // = 1 initially
    for (int step = 0; step < 40; ++step) {
        rho.apply_lindblad_step(0, ops, 0.01);
        double excited = rho.rho[1 * d + 1].real;
        EXPECT_LE(excited, prev_excited + 1e-9) << "population decays monotonically";
        prev_excited = excited;
    }
    EXPECT_LT(rho.rho[1 * d + 1].real, 0.9) << "|1> has partially decayed";
    EXPECT_GT(rho.rho[0].real, 0.1) << "|0> has gained population";
    EXPECT_NEAR(rho.trace(), 1.0, 1e-2);
}

// =============================================================================
// QuditMPS
// =============================================================================

TEST(R1121QuditMps, GroundStateToStatevector) {
    QuditMPS mps(3, 3);
    auto sv = mps.to_statevector();
    EXPECT_NEAR(sv.amplitudes[0].real, 1.0, 1e-12);
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}

TEST(R1121QuditMps, SingleQuditGateMatchesStatevector) {
    QuditMPS mps(2, 3);
    QuditStatevector ref(2, 3);
    auto F = qudit_gates::qft_matrix(3);
    mps.apply_1qudit(0, F);
    ref.apply_1qudit(0, F);
    expect_states_close(mps.to_statevector(), ref);
}

TEST(R1121QuditMps, TwoQuditAdjacentAndNonAdjacentMatchStatevector) {
    for (int d : {2, 3}) {
        QuditMPS mps(3, d);
        QuditStatevector ref(3, d);
        auto F = qudit_gates::qft_matrix(d);
        auto cadd = qudit_gates::cadd_matrix(d, 1);
        mps.apply_1qudit(0, F);          ref.apply_1qudit(0, F);
        mps.apply_2qudit(0, 1, cadd);    ref.apply_2qudit(0, 1, cadd);  // adjacent
        mps.apply_2qudit(0, 2, cadd);    ref.apply_2qudit(0, 2, cadd);  // non-adjacent
        SCOPED_TRACE("d = " + std::to_string(d));
        expect_states_close(mps.to_statevector(), ref, 1e-6);
    }
}

TEST(R1121QuditMps, DenseRoundTrip) {
    for (int d : {2, 3, 4, 5}) {
        auto sv = entangled_state(3, d);
        QuditMPS mps(sv);
        SCOPED_TRACE("d = " + std::to_string(d));
        EXPECT_NEAR(mps.norm_sq(), 1.0, 1e-7);
        expect_states_close(mps.to_statevector(), sv, 1e-6);
    }
}

TEST(R1121QuditMps, MeasurementIsSeedDeterministic) {
    auto sv = entangled_state(3, 3);
    QuditMPS mps(sv);
    QuditMPS copy = mps;
    EXPECT_EQ(mps.measure(33), copy.measure(33));
}
