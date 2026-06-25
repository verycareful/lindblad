// R.1.12.1 total-coverage suite, Batch 4: qudit noise channels and the qudit
// Clifford tableau. Plan: docs (R.1.12.1 coverage plan), section "Batch 4".
//
// Noise: depolarising / amplitude-damping / phase-damping channels are trace
// preserving on a density matrix; the noise model apply_noise path preserves
// trace. Clifford: prime-d constructor guard, is_prime, symplectic invariants,
// seeded measurement, and agreement with the statevector on deterministic
// Clifford circuits (X, CSUM). Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/types.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {
QuditDensityMatrix superposed_dm(int d) {
    QuditDensityMatrix rho(1, d);
    rho.apply_1qudit(0, qudit_gates::qft_matrix(d));
    return rho;
}
}  // namespace

// =============================================================================
// Qudit noise channels — trace preservation
// =============================================================================

TEST(R1121QuditNoise, AmplitudeAndPhaseDampingPreserveTrace) {
    auto ad = QuditNoiseModel::amplitude_damping_channel(3, 0.3);  // (d-1)*g=0.6<=1
    auto pd = QuditNoiseModel::phase_damping_channel(3, 0.4);

    auto r1 = superposed_dm(3);
    r1.apply_kraus_1qudit(0, ad.ops);
    EXPECT_NEAR(r1.trace(), 1.0, 1e-7);

    auto r2 = superposed_dm(3);
    r2.apply_kraus_1qudit(0, pd.ops);
    EXPECT_NEAR(r2.trace(), 1.0, 1e-7);
}

TEST(R1121QuditNoise, NoiseModelApplyNoisePreservesTrace) {
    QuditNoiseModel model;
    model.add_depolarizing(0, 3, 0.2);
    auto rho = superposed_dm(3);
    rho.apply_noise(model);
    EXPECT_NEAR(rho.trace(), 1.0, 1e-7);
}

TEST(R1121QuditNoise, ChannelsAreTracePreservingAcrossDimensions) {
    for (int d : {2, 3, 5, 7}) {
        SCOPED_TRACE("d = " + std::to_string(d));
        const double g = 1.0 / (2.0 * (d - 1));  // keeps (d-1)*g <= 1
        auto r_ad = superposed_dm(d);
        r_ad.apply_kraus_1qudit(0, QuditNoiseModel::amplitude_damping_channel(d, g).ops);
        EXPECT_NEAR(r_ad.trace(), 1.0, 1e-6);

        auto r_pd = superposed_dm(d);
        r_pd.apply_kraus_1qudit(0, QuditNoiseModel::phase_damping_channel(d, 0.3).ops);
        EXPECT_NEAR(r_pd.trace(), 1.0, 1e-6);

        auto r_dp = superposed_dm(d);
        r_dp.apply_kraus_1qudit(0, QuditNoiseModel::depolarizing_channel(d, 0.25).ops);
        EXPECT_NEAR(r_dp.trace(), 1.0, 1e-6);
        EXPECT_LE(r_dp.purity(), 1.0 + 1e-9);
    }
}

TEST(R1121QuditNoise, LindbladBuildersHaveExpectedShape) {
    auto ad = QuditNoiseModel::amplitude_damping_lindblad(3, 0.5);
    EXPECT_EQ(ad.L.size(), 9u);   // 3x3
    EXPECT_GE(ad.rate, 0.0);
    auto deph = QuditNoiseModel::dephasing_lindblad(3, 0.5);
    EXPECT_EQ(deph.size(), 2u);   // d-1 operators
}

// =============================================================================
// Qudit Clifford — constructor guard, is_prime, symplectic invariants
// =============================================================================

TEST(R1121QuditClifford, ConstructorRequiresPrimeDimension) {
    EXPECT_NO_THROW(QuditCliffordSimulator(2, 3));
    EXPECT_THROW(QuditCliffordSimulator(2, 4), std::invalid_argument);  // 4 not prime
}

TEST(R1121QuditClifford, IsPrimeTable) {
    EXPECT_TRUE(QuditCliffordSimulator::is_prime(2));
    EXPECT_TRUE(QuditCliffordSimulator::is_prime(3));
    EXPECT_FALSE(QuditCliffordSimulator::is_prime(4));
    EXPECT_TRUE(QuditCliffordSimulator::is_prime(5));
    EXPECT_FALSE(QuditCliffordSimulator::is_prime(6));
    EXPECT_TRUE(QuditCliffordSimulator::is_prime(7));
    EXPECT_FALSE(QuditCliffordSimulator::is_prime(8));
    EXPECT_FALSE(QuditCliffordSimulator::is_prime(9));
    EXPECT_TRUE(QuditCliffordSimulator::is_prime(11));
    EXPECT_TRUE(QuditCliffordSimulator::is_prime(13));
}

TEST(R1121QuditClifford, MeasureQuditDeterministicAfterShift) {
    // On a basis state, measure_qudit is deterministic: ground -> 0; X^m -> m.
    const int d = 5;
    QuditCliffordSimulator s(2, d);
    EXPECT_EQ(s.measure_qudit(0, 3), 0);
    QuditCliffordSimulator s2(2, d);
    s2.apply_X(1, 3);                       // qudit 1 -> |3>
    EXPECT_EQ(s2.measure_qudit(1, 3), 3);
    EXPECT_EQ(s2.measure_qudit(0, 3), 0);   // untouched qudit stays 0
}

TEST(R1121QuditClifford, SymplecticProductOfConjugatePairs) {
    QuditCliffordSimulator s(2, 3);  // D_j = X_j (rows 0,1), S_j = Z_j (rows 2,3)
    EXPECT_EQ(s.symplectic_product(0, 2), 1);  // <X_0, Z_0> = 1
    EXPECT_EQ(s.symplectic_product(1, 3), 1);  // <X_1, Z_1> = 1
    EXPECT_EQ(s.symplectic_product(0, 0), 0);  // antisymmetric -> self is 0
    EXPECT_EQ(s.symplectic_product(0, 1), 0);  // X_0, X_1 commute
    EXPECT_EQ(s.symplectic_product(0, 3), 0);  // X_0, Z_1 commute
}

// =============================================================================
// Qudit Clifford — measurement and statevector agreement
// =============================================================================

TEST(R1121QuditClifford, GroundStateMeasuresZero) {
    QuditCliffordSimulator s(2, 3);
    auto m = s.measure(7);
    EXPECT_EQ(m[0], 0);
    EXPECT_EQ(m[1], 0);
}

TEST(R1121QuditClifford, DeterministicCircuitMatchesStatevector) {
    const int d = 3;
    // Clifford: X on qudit 0, then CSUM(0->1): |0,0> -> |1,1>.
    QuditCliffordSimulator cl(2, d);
    cl.apply_X(0, 1);
    cl.apply_CSUM(0, 1);
    auto cm = cl.measure(1);

    // Statevector: shift on qudit 0, then cadd(s=1) on (0,1).
    QuditStatevector sv(2, d);
    sv.apply_1qudit(0, qudit_gates::shift_matrix(d, 1));
    sv.apply_2qudit(0, 1, qudit_gates::cadd_matrix(d, 1));
    auto sm = sv.measure(1);

    EXPECT_EQ(cm, sm);
    EXPECT_EQ(cm[0], 1);
    EXPECT_EQ(cm[1], 1);
}

TEST(R1121QuditClifford, CsumThenInverseRestoresState) {
    const int d = 5;
    QuditCliffordSimulator cl(2, d);
    cl.apply_X(0, 2);          // qudit 0 -> |2>
    cl.apply_CSUM(0, 1);       // qudit 1 -> 0 + 2 = 2
    cl.apply_CSUM_dag(0, 1);   // qudit 1 -> 2 - 2 = 0
    auto m = cl.measure(9);
    EXPECT_EQ(m[0], 2);
    EXPECT_EQ(m[1], 0);
}
