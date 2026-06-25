// R.1.12.1 total-coverage suite, Batch 2: lindblad/noise.hpp standard channels.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 2: engines".
//
// Each channel is checked for CPTP validity (sum K†K = I) and for its exact
// action on a density matrix against the closed-form prediction. Throw paths
// for invalid parameters are exercised. The KrausChannel LSB convention is
// pinned with an asymmetric 2-qubit custom channel. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/noise.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 1e-9;

DensityMatrix basis_dm(int n, size_t k) {
    Statevector sv(n);
    sv.initialize_basis(k);
    return DensityMatrix::from_statevector(sv);
}

DensityMatrix plus_dm() {
    Statevector sv(1);
    std::vector<Complex128> amps = {Complex128(INV_SQRT2, 0.0),
                                    Complex128(INV_SQRT2, 0.0)};
    sv.set_amplitudes(amps);
    return DensityMatrix::from_statevector(sv);
}

}  // namespace

// =============================================================================
// CPTP validity across every channel
// =============================================================================

TEST(R1121NoiseChannels, AllChannelsAreTracePreserving) {
    using namespace NoiseChannels;
    std::vector<KrausChannel> chans = {
        depolarizing(0.2, 1), depolarizing(0.1, 2), depolarizing(0.05, 3),
        amplitude_damping(0.3), phase_damping(0.4),
        thermal_relaxation(100.0, 80.0, 10.0, 0.0),
        pauli(0.1, 0.2, 0.3), bit_flip(0.25), phase_flip(0.25),
        bit_phase_flip(0.25), reset(0.6, 0.4),
        coherent_unitary(0.7, 0.3, -0.4),
    };
    for (size_t i = 0; i < chans.size(); ++i) {
        SCOPED_TRACE("channel #" + std::to_string(i));
        EXPECT_TRUE(chans[i].is_valid());
        EXPECT_NEAR(chans[i].trace_preserving_error(), 0.0, 1e-8);
    }
}

// =============================================================================
// Depolarizing: edge probabilities + throws
// =============================================================================

TEST(R1121NoiseChannels, DepolarizingZeroIsIdentity) {
    auto ch = NoiseChannels::depolarizing(0.0, 1);
    auto rho = plus_dm();
    rho.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(rho.purity(), 1.0, kTol) << "p=0 must not decohere";
    EXPECT_NEAR(rho(0, 1).real, 0.5, kTol);
}

TEST(R1121NoiseChannels, DepolarizingThrowsOnInvalidArgs) {
    using namespace NoiseChannels;
    EXPECT_THROW(depolarizing(0.1, 0), std::invalid_argument);
    EXPECT_THROW(depolarizing(0.1, 7), std::invalid_argument);
    EXPECT_THROW(depolarizing(1.5, 1), std::invalid_argument);
    EXPECT_THROW(depolarizing(-0.1, 1), std::invalid_argument);
    EXPECT_NO_THROW(depolarizing(0.5, 6));
}

// =============================================================================
// Amplitude / phase damping: exact closed-form DM evolution
// =============================================================================

TEST(R1121NoiseChannels, AmplitudeDampingTransfersPopulation) {
    const double g = 0.3;
    auto ch = NoiseChannels::amplitude_damping(g);
    auto rho = basis_dm(1, 1);  // |1><1|
    rho.apply_kraus(ch.operators, {0});
    // |1> decays to |0> with probability gamma.
    EXPECT_NEAR(rho(0, 0).real, g, kTol);
    EXPECT_NEAR(rho(1, 1).real, 1.0 - g, kTol);
    EXPECT_NEAR(rho(0, 1).real, 0.0, kTol);
}

TEST(R1121NoiseChannels, PhaseDampingShrinksCoherenceKeepsPopulation) {
    const double lam = 0.36;
    auto ch = NoiseChannels::phase_damping(lam);
    auto rho = plus_dm();  // 0.5 [[1,1],[1,1]]
    rho.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(rho(0, 0).real, 0.5, kTol);
    EXPECT_NEAR(rho(1, 1).real, 0.5, kTol);
    EXPECT_NEAR(rho(0, 1).real, 0.5 * std::sqrt(1.0 - lam), kTol);  // 0.5*0.8
}

// =============================================================================
// Thermal relaxation: coherence decays exactly exp(-t/T2) (the R.1.12.0 fix)
// =============================================================================

TEST(R1121NoiseChannels, ThermalRelaxationCoherenceAndPopulation) {
    const double T1 = 100.0, T2 = 80.0, t = 10.0;
    auto ch = NoiseChannels::thermal_relaxation(T1, T2, t, 0.0);

    auto coh = plus_dm();
    coh.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(coh(0, 1).real, 0.5 * std::exp(-t / T2), 1e-6)
        << "coherence must decay exactly exp(-t/T2)";

    auto pop = basis_dm(1, 1);  // |1>
    pop.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(pop(1, 1).real, std::exp(-t / T1), 1e-6)
        << "excited population must decay exp(-t/T1) with zero excited eq.";
}

TEST(R1121NoiseChannels, ThermalRelaxationThrowsWhenT2ExceedsTwiceT1) {
    EXPECT_THROW(NoiseChannels::thermal_relaxation(100.0, 250.0, 10.0, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(NoiseChannels::thermal_relaxation(-1.0, 1.0, 10.0, 0.0),
                 std::invalid_argument);
}

// =============================================================================
// Pauli-type channels: fixed points and throws
// =============================================================================

TEST(R1121NoiseChannels, BitFlipFixesXEigenstate) {
    // |+> is a +1 eigenstate of X, so bit_flip leaves it invariant.
    auto ch = NoiseChannels::bit_flip(0.3);
    auto rho = plus_dm();
    rho.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(rho(0, 0).real, 0.5, kTol);
    EXPECT_NEAR(rho(0, 1).real, 0.5, kTol) << "X-eigenstate is a bit_flip fixed point";
}

TEST(R1121NoiseChannels, PhaseFlipDampsCoherenceOfComputationalSuperposition) {
    auto ch = NoiseChannels::phase_flip(0.25);
    auto rho = plus_dm();
    rho.apply_kraus(ch.operators, {0});
    // phase_flip(p): off-diagonal scales by (1 - 2p) = 0.5.
    EXPECT_NEAR(rho(0, 1).real, 0.5 * 0.5, kTol);
    EXPECT_NEAR(rho(0, 0).real, 0.5, kTol);
}

TEST(R1121NoiseChannels, PauliChannelThrowsWhenProbabilitiesExceedOne) {
    EXPECT_THROW(NoiseChannels::pauli(0.5, 0.4, 0.3), std::invalid_argument);
    EXPECT_NO_THROW(NoiseChannels::pauli(0.1, 0.2, 0.3));
}

TEST(R1121NoiseChannels, ResetChannelDrivesTowardRequestedPopulations) {
    // reset(p0, p1): drive the qubit toward p0|0><0| + p1|1><1|.
    auto ch = NoiseChannels::reset(1.0, 0.0);  // full reset to |0>
    auto rho = basis_dm(1, 1);                 // start in |1>
    rho.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(rho(0, 0).real, 1.0, kTol);
    EXPECT_NEAR(rho(1, 1).real, 0.0, kTol);
}

// =============================================================================
// coherent_unitary: a single unitary Kraus operator
// =============================================================================

TEST(R1121NoiseChannels, CoherentUnitaryIsSinglePureRotation) {
    auto ch = NoiseChannels::coherent_unitary(0.7, 0.3, -0.4);
    ASSERT_EQ(ch.operators.size(), 1u) << "a coherent error is a single Kraus op";
    EXPECT_TRUE(ch.is_valid());
    // Purity is preserved by a unitary channel.
    auto rho = plus_dm();
    rho.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(rho.purity(), 1.0, kTol);
}

// =============================================================================
// KrausChannel LSB convention: a 2-qubit channel acting as X on qubit 0 only
// =============================================================================

TEST(R1121NoiseChannels, TwoQubitKrausUsesQubitZeroAsLsb) {
    // 4x4 matrix for X on qubit 0 (LSB): swaps indices (0<->1) and (2<->3).
    std::vector<Complex128> Xq0(16, Complex128(0, 0));
    Xq0[0 * 4 + 1] = {1, 0};
    Xq0[1 * 4 + 0] = {1, 0};
    Xq0[2 * 4 + 3] = {1, 0};
    Xq0[3 * 4 + 2] = {1, 0};
    KrausChannel ch;
    ch.n_qubits = 2;
    ch.operators = {Xq0};
    EXPECT_TRUE(ch.is_valid());

    auto rho = basis_dm(2, 0);  // |00>
    rho.apply_kraus(ch.operators, {0, 1});
    // X on qubit 0 sends |00> (index 0) to |01> = q0=1,q1=0 (index 1).
    EXPECT_NEAR(rho(1, 1).real, 1.0, kTol);
    EXPECT_NEAR(rho(0, 0).real, 0.0, kTol);
}

// =============================================================================
// Thermal relaxation: full {T2 = 2T1, T2 = T1, T2 << T1} x {p1 = 0, 0.3} grid
// =============================================================================
//
// Closed form for the implemented channel (derived from the Kraus set in
// src/noise/channels.cpp, with e1 = exp(-t/T1)):
//   coherence:          r01 -> exp(-t/T2) * r01          (any p1)
//   excited population: r11 -> (p0*e1 + p1) * r11 + p1*(1-e1) * r00
// In particular |1><1| -> r11 = (1-p1)*exp(-t/T1) + p1, and the thermal
// equilibrium diag(p0, p1) is a fixed point. The coherence law exp(-t/T2)
// holds for EVERY grid cell -- the R.1.12.0 dephasing-rate fix.

TEST(R1121NoiseChannels, ThermalRelaxationGridPinsCoherenceAndPopulation) {
    const double T1 = 100.0, t = 12.0;
    struct Cell { double T2; const char* name; };
    const Cell cells[] = {
        {2.0 * T1, "T2 = 2T1 (no pure dephasing)"},
        {T1,       "T2 = T1"},
        {5.0,      "T2 << T1"},
    };
    for (const Cell& c : cells) {
        for (double p1 : {0.0, 0.3}) {
            SCOPED_TRACE(std::string(c.name) + ", p1 = " + std::to_string(p1));
            auto ch = NoiseChannels::thermal_relaxation(T1, c.T2, t, p1);
            const double e1 = std::exp(-t / T1);
            const double p0 = 1.0 - p1;

            // Coherence law: exactly exp(-t/T2), independent of p1.
            auto coh = plus_dm();
            coh.apply_kraus(ch.operators, {0});
            EXPECT_NEAR(coh(0, 1).real, 0.5 * std::exp(-t / c.T2), 1e-6);
            EXPECT_NEAR(coh(0, 1).imag, 0.0, 1e-6);

            // Excited population from |1>.
            auto from1 = basis_dm(1, 1);
            from1.apply_kraus(ch.operators, {0});
            EXPECT_NEAR(from1(1, 1).real, p0 * e1 + p1, 1e-6);

            // Excitation from |0> (thermal absorption).
            auto from0 = basis_dm(1, 0);
            from0.apply_kraus(ch.operators, {0});
            EXPECT_NEAR(from0(1, 1).real, p1 * (1.0 - e1), 1e-6);

            // Thermal equilibrium diag(p0, p1) is a fixed point.
            DensityMatrix eq(1);
            eq(0, 0) = Complex128(p0, 0.0);
            eq(1, 1) = Complex128(p1, 0.0);
            eq(0, 1) = Complex128(0.0, 0.0);
            eq(1, 0) = Complex128(0.0, 0.0);
            eq.apply_kraus(ch.operators, {0});
            EXPECT_NEAR(eq(0, 0).real, p0, 1e-6);
            EXPECT_NEAR(eq(1, 1).real, p1, 1e-6);

            EXPECT_TRUE(ch.is_valid());
            EXPECT_NEAR(ch.trace_preserving_error(), 0.0, 1e-8);
        }
    }
}

// =============================================================================
// Depolarizing: n = 1..3 validity + the exact n=1 Bloch contraction
// =============================================================================

TEST(R1121NoiseChannels, DepolarizingKrausCountAndValidityByWidth) {
    using namespace NoiseChannels;
    // 4^n Kraus operators (K0 + the 4^n-1 error terms) at generic p.
    EXPECT_EQ(depolarizing(0.2, 1).operators.size(), 4u);
    EXPECT_EQ(depolarizing(0.2, 2).operators.size(), 16u);
    EXPECT_EQ(depolarizing(0.2, 3).operators.size(), 64u);
    for (int n = 1; n <= 3; ++n) {
        SCOPED_TRACE("n = " + std::to_string(n));
        EXPECT_TRUE(depolarizing(0.37, n).is_valid());
        // p = 1 drops K0 (4^n - 1 error operators only); still trace-preserving.
        auto full = depolarizing(1.0, n);
        EXPECT_EQ(full.operators.size(), (1u << (2 * n)) - 1u);
        EXPECT_TRUE(full.is_valid());
    }
}

TEST(R1121NoiseChannels, DepolarizingOneQubitContractsBlochByExactFactor) {
    // Single-qubit depolarizing: coherence of |+> scales by (1 - 4p/3), the
    // Z-populations of |0> become 1 - 2p/3, and I/2 is the fixed point.
    for (double p : {0.15, 0.5, 0.9}) {
        SCOPED_TRACE("p = " + std::to_string(p));
        auto ch = NoiseChannels::depolarizing(p, 1);

        auto plus = plus_dm();
        plus.apply_kraus(ch.operators, {0});
        EXPECT_NEAR(plus(0, 1).real, 0.5 * (1.0 - 4.0 * p / 3.0), kTol);
        EXPECT_NEAR(plus(0, 0).real, 0.5, kTol) << "|+> populations unchanged";

        auto zero = basis_dm(1, 0);
        zero.apply_kraus(ch.operators, {0});
        EXPECT_NEAR(zero(0, 0).real, 1.0 - 2.0 * p / 3.0, kTol);
        EXPECT_NEAR(zero(1, 1).real, 2.0 * p / 3.0, kTol);

        // Maximally mixed state is the unique fixed point.
        DensityMatrix mm(1);
        mm(0, 0) = Complex128(0.5, 0.0);
        mm(1, 1) = Complex128(0.5, 0.0);
        mm(0, 1) = Complex128(0.0, 0.0);
        mm(1, 0) = Complex128(0.0, 0.0);
        mm.apply_kraus(ch.operators, {0});
        EXPECT_NEAR(mm(0, 0).real, 0.5, kTol);
        EXPECT_NEAR(mm(0, 1).real, 0.0, kTol);
    }
}

// =============================================================================
// Channel fixed points and full-mixing limits
// =============================================================================

TEST(R1121NoiseChannels, AmplitudeDampingGroundStateIsFixedPointFullDecayCollapses) {
    // gamma = 1 fully collapses any state to |0>; |0> is always a fixed point.
    auto full = NoiseChannels::amplitude_damping(1.0);
    auto rho = plus_dm();
    rho.apply_kraus(full.operators, {0});
    EXPECT_NEAR(rho(0, 0).real, 1.0, kTol);
    EXPECT_NEAR(rho(1, 1).real, 0.0, kTol);
    EXPECT_NEAR(rho(0, 1).real, 0.0, kTol) << "coherence vanishes once |1> is empty";

    auto ground = basis_dm(1, 0);
    ground.apply_kraus(NoiseChannels::amplitude_damping(0.42).operators, {0});
    EXPECT_NEAR(ground(0, 0).real, 1.0, kTol) << "|0> is an amplitude-damping fixed point";
}

TEST(R1121NoiseChannels, PhaseDampingFullyDecoheresAtLambdaOne) {
    auto ch = NoiseChannels::phase_damping(1.0);
    auto rho = plus_dm();
    rho.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(rho(0, 1).real, 0.0, kTol) << "lambda = 1 kills all coherence";
    EXPECT_NEAR(rho(0, 0).real, 0.5, kTol) << "populations preserved";
}

TEST(R1121NoiseChannels, PauliChannelOffDiagonalScaleMatchesAnalyticForm) {
    // For a general Pauli channel, the |+> coherence scales by
    // (1 - 2*px - 2*py): X preserves it, Y and Z each flip its sign weighted.
    const double px = 0.1, py = 0.15, pz = 0.2;
    auto ch = NoiseChannels::pauli(px, py, pz);
    auto rho = plus_dm();
    rho.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(rho(0, 1).real, 0.5 * (1.0 - 2.0 * py - 2.0 * pz), kTol);
    EXPECT_NEAR(rho(0, 0).real, 0.5, kTol);
}

TEST(R1121NoiseChannels, ResetChannelDrivesTowardMixtureFromAnyInput) {
    // reset(0.6, 0.4) is full reset (p0+p1 = 1): output is p0|0><0| + p1|1><1|
    // regardless of the input state, with no surviving coherence.
    auto ch = NoiseChannels::reset(0.6, 0.4);
    auto rho = plus_dm();
    rho.apply_kraus(ch.operators, {0});
    EXPECT_NEAR(rho(0, 0).real, 0.6, kTol);
    EXPECT_NEAR(rho(1, 1).real, 0.4, kTol);
    EXPECT_NEAR(rho(0, 1).real, 0.0, kTol);
}
