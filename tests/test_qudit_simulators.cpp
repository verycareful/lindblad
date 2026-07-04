// R.1.7.1 test suite — qudit backend simulators
// Covers: QuditDensityMatrix, QuditMPS, QuditCliffordSimulator, QuditNoiseModel,
//         and backend dispatch across all 5 qudit algorithms.

#include "lindblad/algorithms.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"
#include "lindblad/qudit/qudit_simulator.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;
using namespace lindblad::qudit_gates;

static constexpr double kTol = 1e-9;
static constexpr double kLooseTol = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static double dm_trace(const QuditDensityMatrix& dm) {
    return dm.trace();
}

// Compare amplitudes of two statevectors element-wise.
static bool sv_equal(const QuditStatevector& a, const QuditStatevector& b, double tol = kTol) {
    if (a.dim != b.dim) return false;
    for (size_t i = 0; i < a.dim; ++i) {
        if (std::abs(a.amplitudes[i].real - b.amplitudes[i].real) > tol) return false;
        if (std::abs(a.amplitudes[i].imag - b.amplitudes[i].imag) > tol) return false;
    }
    return true;
}

// DM diagonal probability at index i.
static double dm_prob(const QuditDensityMatrix& dm, size_t i) {
    return dm.rho[i * dm.dim + i].real;
}

// SV probability at index i.
static double sv_prob(const QuditStatevector& sv, size_t i) {
    const auto& a = sv.amplitudes[i];
    return a.real * a.real + a.imag * a.imag;
}

// Build the d×d Hadamard (= QFT for d=2) gate manually for verification.
static std::vector<Complex128> hadamard_d2() {
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    return {
        Complex128( inv_sqrt2, 0.0), Complex128( inv_sqrt2, 0.0),
        Complex128( inv_sqrt2, 0.0), Complex128(-inv_sqrt2, 0.0)
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditDensityMatrix — basic construction and invariants
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditDensityMatrix, InitializesToZeroState_d2_n2) {
    QuditDensityMatrix dm(2, 2);
    EXPECT_EQ(dm.n_qudits, 2);
    EXPECT_EQ(dm.d, 2);
    EXPECT_EQ(dm.dim, 4u);
    // rho[0][0] = 1, all others 0
    EXPECT_NEAR(dm_prob(dm, 0), 1.0, kTol);
    for (size_t i = 1; i < dm.dim; ++i)
        EXPECT_NEAR(dm_prob(dm, i), 0.0, kTol) << "at i=" << i;
}

TEST(QuditDensityMatrix, InitializesToZeroState_d3_n1) {
    QuditDensityMatrix dm(1, 3);
    EXPECT_EQ(dm.dim, 3u);
    EXPECT_NEAR(dm_prob(dm, 0), 1.0, kTol);
    EXPECT_NEAR(dm_prob(dm, 1), 0.0, kTol);
    EXPECT_NEAR(dm_prob(dm, 2), 0.0, kTol);
}

TEST(QuditDensityMatrix, TraceIsOne_AfterInit) {
    QuditDensityMatrix dm(3, 2);
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
}

TEST(QuditDensityMatrix, PurityIsOne_PureState) {
    QuditDensityMatrix dm(2, 2);
    EXPECT_NEAR(dm.purity(), 1.0, kTol);
}

TEST(QuditDensityMatrix, ConstructFromStatevector_d2_n1_Superposition) {
    // |+> = H|0>
    QuditStatevector sv(1, 2);
    const auto H = hadamard_d2();
    sv.apply_1qudit(0, H);

    QuditDensityMatrix dm(sv);
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
    EXPECT_NEAR(dm.purity(), 1.0, kTol);
    // Diagonal: 0.5 each
    EXPECT_NEAR(dm_prob(dm, 0), 0.5, kTol);
    EXPECT_NEAR(dm_prob(dm, 1), 0.5, kTol);
    // Off-diagonal: rho[0][1] = 0.5 (real)
    EXPECT_NEAR(dm.rho[0 * 2 + 1].real, 0.5, kTol);
    EXPECT_NEAR(dm.rho[1 * 2 + 0].real, 0.5, kTol);
}

TEST(QuditDensityMatrix, Apply1QuditGate_MatchesSV_d2) {
    const auto H = hadamard_d2();

    QuditStatevector sv(2, 2);
    sv.apply_1qudit(0, H);

    QuditDensityMatrix dm(2, 2);
    dm.apply_1qudit(0, H);

    for (size_t i = 0; i < sv.dim; ++i)
        EXPECT_NEAR(dm_prob(dm, i), sv_prob(sv, i), kTol) << "i=" << i;
}

TEST(QuditDensityMatrix, Apply1QuditGate_MatchesSV_d3) {
    const auto F = qft_matrix(3);

    QuditStatevector sv(2, 3);
    sv.apply_1qudit(1, F);

    QuditDensityMatrix dm(2, 3);
    dm.apply_1qudit(1, F);

    for (size_t i = 0; i < sv.dim; ++i)
        EXPECT_NEAR(dm_prob(dm, i), sv_prob(sv, i), kTol) << "i=" << i;
}

TEST(QuditDensityMatrix, Apply2QuditGate_MatchesSV_d2) {
    // Build H|0> ⊗ |0>, then apply CADD(1) to entangle
    const auto H    = hadamard_d2();
    const auto CADD = cadd_matrix(2, 1);  // CNOT for d=2

    QuditStatevector sv(2, 2);
    sv.apply_1qudit(0, H);
    sv.apply_2qudit(0, 1, CADD);

    QuditDensityMatrix dm(2, 2);
    dm.apply_1qudit(0, H);
    dm.apply_2qudit(0, 1, CADD);

    for (size_t i = 0; i < sv.dim; ++i)
        EXPECT_NEAR(dm_prob(dm, i), sv_prob(sv, i), kTol) << "i=" << i;
}

TEST(QuditDensityMatrix, TracePreservation_AfterGates_d3) {
    const auto F    = qft_matrix(3);
    const auto Finv = iqft_matrix(3);
    QuditDensityMatrix dm(2, 3);
    dm.apply_1qudit(0, F);
    dm.apply_1qudit(1, F);
    dm.apply_1qudit(0, Finv);
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditDensityMatrix — Kraus channels and noise
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditDensityMatrix, KrausTracePreservation_Depolarizing_d2) {
    QuditDensityMatrix dm(1, 2);
    // Apply H to make a superposition, then apply noise
    dm.apply_1qudit(0, hadamard_d2());
    auto ch = QuditNoiseModel::depolarizing_channel(2, 0.1);
    dm.apply_kraus_1qudit(0, ch.ops);
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
}

TEST(QuditDensityMatrix, PurityDecays_Depolarizing_d2) {
    QuditDensityMatrix dm(1, 2);
    EXPECT_NEAR(dm.purity(), 1.0, kTol);
    auto ch = QuditNoiseModel::depolarizing_channel(2, 0.3);
    dm.apply_kraus_1qudit(0, ch.ops);
    EXPECT_LT(dm.purity(), 1.0 - kTol);
}

TEST(QuditDensityMatrix, PurityDecays_Depolarizing_d3) {
    QuditDensityMatrix dm(1, 3);
    auto ch = QuditNoiseModel::depolarizing_channel(3, 0.3);
    dm.apply_kraus_1qudit(0, ch.ops);
    EXPECT_LT(dm.purity(), 1.0 - kTol);
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
}

TEST(QuditDensityMatrix, AmplitudeDamping_PopulationDecay_d2) {
    // Start in |1>: apply X to flip from |0>
    QuditDensityMatrix dm(1, 2);
    const auto X = shift_matrix(2, 1);
    dm.apply_1qudit(0, X);
    EXPECT_NEAR(dm_prob(dm, 1), 1.0, kTol);  // in |1>

    const auto ch = QuditNoiseModel::amplitude_damping_channel(2, 0.5);
    dm.apply_kraus_1qudit(0, ch.ops);
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
    // |0> population must have increased
    EXPECT_GT(dm_prob(dm, 0), kTol);
}

TEST(QuditDensityMatrix, ApplyNoise_UsesModel_d2) {
    QuditNoiseModel model;
    model.add_depolarizing(0, 2, 0.2);

    QuditDensityMatrix dm(2, 2);
    double purity_before = dm.purity();
    dm.apply_noise(model);
    EXPECT_LT(dm.purity(), purity_before - kTol);
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
}

TEST(QuditDensityMatrix, LindbladStep_AmplitudeDamping_d2) {
    // |1> should decay toward |0> under Lindblad amplitude damping
    QuditDensityMatrix dm(1, 2);
    const auto X = shift_matrix(2, 1);
    dm.apply_1qudit(0, X);  // now in |1>

    const double gamma = 1.0;
    const double dt = 0.1;
    auto lop = QuditNoiseModel::amplitude_damping_lindblad(2, gamma);
    dm.apply_lindblad_step(0, {lop}, dt);

    // After dt Lindblad evolution, |1> population should have decreased
    EXPECT_LT(dm_prob(dm, 1), 1.0 - kTol);
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditDensityMatrix — measurement and partial trace
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditDensityMatrix, Measure_ZeroState_AlwaysReturnsZero_d2) {
    QuditDensityMatrix dm(2, 2);
    for (int trial = 0; trial < 5; ++trial) {
        auto outcome = dm.measure(static_cast<uint64_t>(trial));
        for (int v : outcome) EXPECT_EQ(v, 0);
    }
}

TEST(QuditDensityMatrix, Measure_Collapses_d2) {
    // |+0>: after measure, must be 0 or 1 on qudit 0 with prob 0.5 each
    QuditDensityMatrix dm(2, 2);
    dm.apply_1qudit(0, hadamard_d2());
    auto outcome = dm.measure(42);
    // After measurement, dm diagonal should have sum 1 at the measured index
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
}

TEST(QuditDensityMatrix, PartialTrace_ZeroState_d2) {
    QuditDensityMatrix dm(2, 2);
    // Trace out qudit 1 (index 1), keep qudit 0 (index 0)
    auto reduced = dm.partial_trace({0});
    EXPECT_EQ(reduced.n_qudits, 1);
    EXPECT_NEAR(dm_prob(reduced, 0), 1.0, kTol);
}

TEST(QuditDensityMatrix, PartialTrace_EntangledState_d2) {
    // Bell state (|00> + |11>)/√2 — each qudit should be maximally mixed
    QuditDensityMatrix dm(2, 2);
    dm.apply_1qudit(0, hadamard_d2());
    dm.apply_2qudit(0, 1, cadd_matrix(2, 1));
    // Trace out qudit 1, keep qudit 0
    auto reduced = dm.partial_trace({0});
    EXPECT_NEAR(dm_prob(reduced, 0), 0.5, kTol);
    EXPECT_NEAR(dm_prob(reduced, 1), 0.5, kTol);
    // Off-diagonal should be 0 (maximally mixed)
    EXPECT_NEAR(reduced.rho[0 * 2 + 1].real, 0.0, kTol);
    EXPECT_NEAR(reduced.rho[1 * 2 + 0].real, 0.0, kTol);
}

TEST(QuditDensityMatrix, PartialTrace_EmptyKeep_Throws) {
    QuditDensityMatrix dm(2, 2);
    EXPECT_THROW(dm.partial_trace({}), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditMPS — construction and round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditMPS, InitializesToZeroState_d2_n3) {
    QuditMPS mps(3, 2);
    auto sv = mps.to_statevector();
    EXPECT_EQ(sv.dim, 8u);
    EXPECT_NEAR(sv.amplitudes[0].real, 1.0, kTol);
    for (size_t i = 1; i < sv.dim; ++i)
        EXPECT_NEAR(sv_prob(sv, i), 0.0, kTol) << "i=" << i;
}

TEST(QuditMPS, InitializesToZeroState_d3_n2) {
    QuditMPS mps(2, 3);
    auto sv = mps.to_statevector();
    EXPECT_EQ(sv.dim, 9u);
    EXPECT_NEAR(sv.amplitudes[0].real, 1.0, kTol);
    EXPECT_NEAR(sv_prob(sv, 0), 1.0, kTol);
}

TEST(QuditMPS, SVRoundTrip_d2_n3_UniformSuperposition) {
    // Create uniform superposition via gates on SV, then convert to MPS and back
    QuditStatevector sv(3, 2);
    const auto H = hadamard_d2();
    sv.apply_1qudit(0, H);
    sv.apply_1qudit(1, H);
    sv.apply_1qudit(2, H);

    QuditMPS mps(sv);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

TEST(QuditMPS, SVRoundTrip_d3_n2_QFT) {
    QuditStatevector sv(2, 3);
    const auto F = qft_matrix(3);
    sv.apply_1qudit(0, F);
    sv.apply_1qudit(1, F);

    QuditMPS mps(sv);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

TEST(QuditMPS, SVRoundTrip_d5_n2) {
    QuditStatevector sv(2, 5);
    const auto F = qft_matrix(5);
    sv.apply_1qudit(0, F);

    QuditMPS mps(sv);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

TEST(QuditMPS, NormSq_UnityAfterGates_d2) {
    QuditMPS mps(3, 2);
    const auto H = hadamard_d2();
    mps.apply_1qudit(0, H);
    mps.apply_1qudit(1, H);
    EXPECT_NEAR(mps.norm_sq(), 1.0, kLooseTol);
}

TEST(QuditMPS, Apply1QuditGate_MatchesSV_d2) {
    const auto H = hadamard_d2();

    QuditStatevector sv(3, 2);
    sv.apply_1qudit(1, H);

    QuditMPS mps(3, 2);
    mps.apply_1qudit(1, H);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

TEST(QuditMPS, Apply1QuditGate_MatchesSV_d3) {
    const auto F = qft_matrix(3);

    QuditStatevector sv(2, 3);
    sv.apply_1qudit(0, F);

    QuditMPS mps(2, 3);
    mps.apply_1qudit(0, F);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

TEST(QuditMPS, Apply2QuditGate_Adjacent_MatchesSV_d2) {
    const auto H    = hadamard_d2();
    const auto CNOT = cadd_matrix(2, 1);

    QuditStatevector sv(2, 2);
    sv.apply_1qudit(0, H);
    sv.apply_2qudit(0, 1, CNOT);

    QuditMPS mps(2, 2);
    mps.apply_1qudit(0, H);
    mps.apply_2qudit_adjacent(0, CNOT);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

TEST(QuditMPS, Apply2QuditGate_NonAdjacent_d2_n3) {
    // Apply CNOT(0,2) on 3-qubit MPS — requires SWAP chain
    const auto H    = hadamard_d2();
    const auto CNOT = cadd_matrix(2, 1);

    QuditStatevector sv(3, 2);
    sv.apply_1qudit(0, H);
    sv.apply_2qudit(0, 2, CNOT);

    QuditMPS mps(3, 2);
    mps.apply_1qudit(0, H);
    mps.apply_2qudit(0, 2, CNOT);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

TEST(QuditMPS, Measure_ZeroState_AlwaysReturnsZero_d2) {
    QuditMPS mps(2, 2);
    for (int trial = 0; trial < 5; ++trial) {
        auto outcome = mps.measure(static_cast<uint64_t>(trial));
        for (int v : outcome) EXPECT_EQ(v, 0);
    }
}

TEST(QuditMPS, LeftCanonicalize_NormPreserved_d2) {
    const auto H = hadamard_d2();
    const auto CNOT = cadd_matrix(2, 1);

    QuditMPS mps(3, 2);
    mps.apply_1qudit(0, H);
    mps.apply_2qudit_adjacent(0, CNOT);
    mps.apply_2qudit_adjacent(1, CNOT);
    double ns_before = mps.norm_sq();
    mps.left_canonicalize();
    EXPECT_NEAR(mps.norm_sq(), ns_before, kLooseTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditCliffordSimulator — tableau structure and gate correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditClifford, InitialTableau_d2_n3) {
    QuditCliffordSimulator c(3, 2);
    EXPECT_EQ(c.n_qudits, 3);
    EXPECT_EQ(c.d, 2);
    // Destabilizers (rows 0,1,2): X_j — xbits[j][j]=1, zbits=0, phase=0
    for (int j = 0; j < 3; ++j) {
        EXPECT_EQ(c.xbits[j][static_cast<size_t>(j)], 1) << "destab j=" << j;
        for (int q = 0; q < 3; ++q) {
            if (q != j) {
                EXPECT_EQ(c.xbits[j][static_cast<size_t>(q)], 0);
            }
        }
        EXPECT_EQ(c.phase[j], 0);
    }
    // Stabilizers (rows 3,4,5): Z_j — zbits[n+j][j]=1, xbits=0, phase=0
    for (int j = 0; j < 3; ++j) {
        EXPECT_EQ(c.zbits[3 + j][static_cast<size_t>(j)], 1) << "stab j=" << j;
        for (int q = 0; q < 3; ++q) {
            if (q != j) {
                EXPECT_EQ(c.zbits[3 + j][static_cast<size_t>(q)], 0);
            }
        }
        EXPECT_EQ(c.phase[3 + j], 0);
    }
}

TEST(QuditClifford, InitialTableau_d3_n2) {
    QuditCliffordSimulator c(2, 3);
    // Destabilizers
    for (int j = 0; j < 2; ++j)
        EXPECT_EQ(c.xbits[j][static_cast<size_t>(j)], 1);
    // Stabilizers
    for (int j = 0; j < 2; ++j)
        EXPECT_EQ(c.zbits[2 + j][static_cast<size_t>(j)], 1);
}

TEST(QuditClifford, ApplyX_UpdatesStabilizerPhase_d2) {
    // X on q=0 conjugates Z_0 → -Z_0 (phase += 2 in Z_4)
    QuditCliffordSimulator c(1, 2);
    // Stabilizer row 1 = Z_0: phase=0, x=0, z=1
    c.apply_X(0);
    // Expected: phase[1] = -2*z[0] = -2 mod 4 = 2
    EXPECT_EQ(c.phase[1], 2);
    EXPECT_EQ(c.zbits[1][0], 1);
    EXPECT_EQ(c.xbits[1][0], 0);
}

TEST(QuditClifford, ApplyH_SwapsXZ_d2) {
    // H on q=0: stabilizer Z_0 → X_0
    QuditCliffordSimulator c(2, 2);
    c.apply_H(0);
    // Row n+0 = row 2: stabilizer for q=0
    // After H: x[2][0] should be 1, z[2][0] should be 0
    EXPECT_EQ(c.xbits[2][0], 1);
    EXPECT_EQ(c.zbits[2][0], 0);
}

TEST(QuditClifford, ApplyH_TwiceIsIdentity_d2) {
    QuditCliffordSimulator c(2, 2);
    c.apply_H(0);
    c.apply_H(0);
    // Should be back to initial Z_0 stabilizer
    EXPECT_EQ(c.xbits[2][0], 0);
    EXPECT_EQ(c.zbits[2][0], 1);
}

TEST(QuditClifford, ApplyCSum_UpdatesTargetX_d2) {
    // CSUM(0,1): X_c → X_c X_t. Destabilizer X_0 (row 0) should gain x[1]=1
    QuditCliffordSimulator c(2, 2);
    c.apply_CSUM(0, 1);
    // Destabilizer row 0 = X_0: after CSUM → X_0 X_1
    EXPECT_EQ(c.xbits[0][0], 1);
    EXPECT_EQ(c.xbits[0][1], 1);
}

TEST(QuditClifford, ApplyCSum_UpdatesControlZ_d2) {
    // CSUM(0,1): Z_t → Z_c Z_t. Stabilizer Z_1 (row n+1=3) should gain zbits[3][0]=1
    QuditCliffordSimulator c(2, 2);
    c.apply_CSUM(0, 1);
    // Stabilizer row 3 = Z_1: after CSUM → Z_0^{d-1} Z_1 (d-1 = 1 for d=2)
    EXPECT_EQ(c.zbits[3][1], 1);
    EXPECT_EQ(c.zbits[3][0], 1);  // d-1 = 1 for d=2
}

TEST(QuditClifford, Measure_ZeroState_AlwaysZero_d2) {
    QuditCliffordSimulator c(3, 2);
    auto outcome = c.measure(42);
    for (int v : outcome) EXPECT_EQ(v, 0);
}

TEST(QuditClifford, Measure_ZeroState_AlwaysZero_d3) {
    QuditCliffordSimulator c(2, 3);
    auto outcome = c.measure(123);
    for (int v : outcome) EXPECT_EQ(v, 0);
}

TEST(QuditClifford, ApplyH_ThenMeasure_UniformBit_d2) {
    // H|0> → |+>: measuring in Z basis should give 0 or 1 (determinism lost)
    // Run many shots and verify both outcomes appear
    int count0 = 0, count1 = 0;
    for (int seed = 0; seed < 100; ++seed) {
        QuditCliffordSimulator c(1, 2);
        c.apply_H(0);
        auto outcome = c.measure(static_cast<uint64_t>(seed));
        if (outcome[0] == 0) ++count0;
        else ++count1;
    }
    EXPECT_GT(count0, 20);
    EXPECT_GT(count1, 20);
}

TEST(QuditClifford, NonPrime_Throws_d4) {
    EXPECT_THROW(QuditCliffordSimulator(2, 4), std::invalid_argument);
}

TEST(QuditClifford, NonPrime_Throws_d6) {
    EXPECT_THROW(QuditCliffordSimulator(2, 6), std::invalid_argument);
}

TEST(QuditClifford, Prime_d5_ConstructOk) {
    EXPECT_NO_THROW(QuditCliffordSimulator(2, 5));
}

// ─────────────────────────────────────────────────────────────────────────────
// QuditNoiseModel — channel construction
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditNoiseModel, DepolarizingChannel_d2_NumKrausOps) {
    // d^2 = 4 Kraus operators for d=2
    auto ch = QuditNoiseModel::depolarizing_channel(2, 0.1);
    EXPECT_EQ(ch.ops.size(), 4u);
}

TEST(QuditNoiseModel, DepolarizingChannel_d3_NumKrausOps) {
    // d^2 = 9 Kraus operators for d=3
    auto ch = QuditNoiseModel::depolarizing_channel(3, 0.1);
    EXPECT_EQ(ch.ops.size(), 9u);
}

TEST(QuditNoiseModel, AmplitudeDampingChannel_d2_NumOps) {
    // d Kraus operators (K_0 + d-1 lowering ops)
    auto ch = QuditNoiseModel::amplitude_damping_channel(2, 0.3);
    EXPECT_EQ(ch.ops.size(), 2u);
}

TEST(QuditNoiseModel, AmplitudeDampingChannel_d3_NumOps) {
    auto ch = QuditNoiseModel::amplitude_damping_channel(3, 0.1);
    EXPECT_EQ(ch.ops.size(), 3u);
}

TEST(QuditNoiseModel, PhaseDampingChannel_d2_NumOps) {
    // 1 + d = 3 Kraus operators for d=2
    auto ch = QuditNoiseModel::phase_damping_channel(2, 0.2);
    EXPECT_EQ(ch.ops.size(), 3u);
}

TEST(QuditNoiseModel, PhaseDampingChannel_d3_NumOps) {
    auto ch = QuditNoiseModel::phase_damping_channel(3, 0.1);
    EXPECT_EQ(ch.ops.size(), 4u);
}

TEST(QuditNoiseModel, AddDepolarizing_RegistersQudit) {
    QuditNoiseModel model;
    model.add_depolarizing(0, 2, 0.1);
    EXPECT_EQ(model.per_qudit.count(0), 1u);
    EXPECT_EQ(model.per_qudit.at(0).kraus.ops.size(), 4u);
}

TEST(QuditNoiseModel, AmplitudeDampingLindblad_d2_RateOne) {
    auto lop = QuditNoiseModel::amplitude_damping_lindblad(2, 1.0);
    EXPECT_EQ(lop.L.size(), 4u);  // 2x2 matrix
    EXPECT_NEAR(lop.rate, 1.0, kTol);
    // L[0,1] = sqrt(1) = 1 (lower diagonal)
    EXPECT_NEAR(lop.L[0 * 2 + 1].real, 1.0, kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend dispatch — QuditBernsteinVazirani
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditBV_Backend, Statevector_Recovers_d2) {
    const std::vector<int> secret{1, 0, 1};
    auto r = QuditBernsteinVazirani::solve(secret, 2, 1, 42, QuditBackend::STATEVECTOR);
    EXPECT_EQ(r.secret, secret);
}

TEST(QuditBV_Backend, DensityMatrix_Recovers_d2) {
    const std::vector<int> secret{1, 0, 1};
    auto r = QuditBernsteinVazirani::solve(secret, 2, 1, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_EQ(r.secret, secret);
}

TEST(QuditBV_Backend, MPS_Recovers_d2) {
    const std::vector<int> secret{1, 0, 1};
    auto r = QuditBernsteinVazirani::solve(secret, 2, 1, 42, QuditBackend::MPS);
    EXPECT_EQ(r.secret, secret);
}

TEST(QuditBV_Backend, Clifford_Recovers_d2) {
    const std::vector<int> secret{1, 0, 1};
    auto r = QuditBernsteinVazirani::solve(secret, 2, 1, 42, QuditBackend::CLIFFORD);
    EXPECT_EQ(r.secret, secret);
}

TEST(QuditBV_Backend, Clifford_Recovers_d3) {
    const std::vector<int> secret{2, 0, 1};
    auto r = QuditBernsteinVazirani::solve(secret, 3, 5, 42, QuditBackend::CLIFFORD);
    EXPECT_EQ(r.secret, secret);
}

TEST(QuditBV_Backend, Clifford_Recovers_d5) {
    const std::vector<int> secret{3, 1, 4};
    auto r = QuditBernsteinVazirani::solve(secret, 5, 5, 42, QuditBackend::CLIFFORD);
    EXPECT_EQ(r.secret, secret);
}

TEST(QuditBV_Backend, AllBackends_AgreeOn_d2) {
    const std::vector<int> secret{1, 1, 0, 1};
    auto sv  = QuditBernsteinVazirani::solve(secret, 2, 3, 42, QuditBackend::STATEVECTOR);
    auto dm  = QuditBernsteinVazirani::solve(secret, 2, 3, 42, QuditBackend::DENSITY_MATRIX);
    auto mps = QuditBernsteinVazirani::solve(secret, 2, 3, 42, QuditBackend::MPS);
    auto clf = QuditBernsteinVazirani::solve(secret, 2, 3, 42, QuditBackend::CLIFFORD);
    EXPECT_EQ(sv.secret,  secret);
    EXPECT_EQ(dm.secret,  secret);
    EXPECT_EQ(mps.secret, secret);
    EXPECT_EQ(clf.secret, secret);
}

TEST(QuditBV_Backend, DM_WithNoise_StillRecovers_LowNoise_d2) {
    QuditNoiseModel model;
    model.add_depolarizing(0, 2, 0.01);
    const std::vector<int> secret{1, 0};
    // With very low noise and many shots, should still recover
    auto r = QuditBernsteinVazirani::solve(secret, 2, 20, 42,
        QuditBackend::DENSITY_MATRIX, &model);
    EXPECT_EQ(r.secret, secret);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend dispatch — QuditDeutschJozsa
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditDJ_Backend, DensityMatrix_Constant_d2) {
    auto f = [](const std::vector<int>&) -> int { return 1; };
    auto r = QuditDeutschJozsa::solve(2, 2, f, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_EQ(r.verdict, QuditDeutschJozsa::Verdict::CONSTANT);
}

TEST(QuditDJ_Backend, DensityMatrix_Balanced_d2) {
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    auto r = QuditDeutschJozsa::solve(1, 2, f, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_EQ(r.verdict, QuditDeutschJozsa::Verdict::BALANCED);
}

TEST(QuditDJ_Backend, MPS_Constant_d3) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    auto r = QuditDeutschJozsa::solve(2, 3, f, 42, QuditBackend::MPS);
    EXPECT_EQ(r.verdict, QuditDeutschJozsa::Verdict::CONSTANT);
}

TEST(QuditDJ_Backend, MPS_Balanced_d3) {
    // f(x) = x[0] is balanced for n=2, d=3
    auto f = [](const std::vector<int>& x) -> int { return x[0]; };
    auto r = QuditDeutschJozsa::solve(2, 3, f, 42, QuditBackend::MPS);
    EXPECT_EQ(r.verdict, QuditDeutschJozsa::Verdict::BALANCED);
}

TEST(QuditDJ_Backend, Clifford_OpaqueThrows_AffineWorks_d2) {
    // R.1.11.0: an opaque std::function oracle has no Clifford decomposition,
    // so the CLIFFORD backend now throws instead of silently falling back to SV.
    auto f = [](const std::vector<int>&) -> int { return 0; };
    EXPECT_THROW(QuditDeutschJozsa::solve(2, 2, f, 42, QuditBackend::CLIFFORD),
                 std::invalid_argument);
    // The structured affine oracle IS Clifford-decomposable and runs on the tableau.
    QuditAffineOracle constant{ {{0, 0}}, {0} };   // f(x) = 0  → CONSTANT
    EXPECT_EQ(QuditDeutschJozsa::solve(constant, 2, 42, QuditBackend::CLIFFORD).verdict,
              QuditDeutschJozsa::Verdict::CONSTANT);
    QuditAffineOracle balanced{ {{1, 0}}, {0} };   // f(x) = x_0 → BALANCED
    EXPECT_EQ(QuditDeutschJozsa::solve(balanced, 2, 42, QuditBackend::CLIFFORD).verdict,
              QuditDeutschJozsa::Verdict::BALANCED);
}

TEST(QuditDJ_Backend, AllBackends_AgreeOn_Constant_d2) {
    auto f = [](const std::vector<int>&) -> int { return 2; };
    auto sv  = QuditDeutschJozsa::solve(2, 3, f, 42, QuditBackend::STATEVECTOR);
    auto dm  = QuditDeutschJozsa::solve(2, 3, f, 42, QuditBackend::DENSITY_MATRIX);
    auto mps = QuditDeutschJozsa::solve(2, 3, f, 42, QuditBackend::MPS);
    EXPECT_EQ(sv.verdict,  QuditDeutschJozsa::Verdict::CONSTANT);
    EXPECT_EQ(dm.verdict,  QuditDeutschJozsa::Verdict::CONSTANT);
    EXPECT_EQ(mps.verdict, QuditDeutschJozsa::Verdict::CONSTANT);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend dispatch — QuditGrover (CLIFFORD must throw)
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditGrover_Backend, Clifford_Throws) {
    EXPECT_THROW(
        QuditGrover::search(1, 2, {0}, -1, 10, 42, QuditBackend::CLIFFORD),
        std::invalid_argument);
}

TEST(QuditGrover_Backend, DM_FindsTarget_d2_n2) {
    const std::vector<int> target{1, 0};
    auto r = QuditGrover::search(2, 2, target, -1, 50, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_EQ(r.solution, target);
}

TEST(QuditGrover_Backend, MPS_FindsTarget_d2_n2) {
    const std::vector<int> target{0, 1};
    auto r = QuditGrover::search(2, 2, target, -1, 50, 42, QuditBackend::MPS);
    EXPECT_EQ(r.solution, target);
}

TEST(QuditGrover_Backend, SV_vs_DM_AgreesOnTarget_d2) {
    const std::vector<int> target{1, 1};
    auto sv = QuditGrover::search(2, 2, target, 1, 50, 42, QuditBackend::STATEVECTOR);
    auto dm = QuditGrover::search(2, 2, target, 1, 50, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_EQ(sv.solution, dm.solution);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend dispatch — QuditPhaseEstimation (CLIFFORD must throw)
// ─────────────────────────────────────────────────────────────────────────────

// Eigenstate of shift_matrix(d,1) for eigenvalue exp(2*pi*i*k/d)
static std::vector<Complex128> shift_eigenstate_local(int d, int k) {
    std::vector<Complex128> psi(static_cast<size_t>(d));
    const double norm = 1.0 / std::sqrt(static_cast<double>(d));
    const double two_pi_over_d = 2.0 * M_PI / static_cast<double>(d);
    for (int j = 0; j < d; ++j)
        psi[static_cast<size_t>(j)] =
            Complex128::exp_i(-two_pi_over_d * static_cast<double>(k * j)) * norm;
    return psi;
}

TEST(QuditQPE_Backend, Clifford_Throws) {
    const auto U = shift_matrix(2, 1);
    const auto psi = shift_eigenstate_local(2, 1);
    EXPECT_THROW(
        QuditPhaseEstimation::estimate(2, 2, U, psi, 42, QuditBackend::CLIFFORD),
        std::invalid_argument);
}

TEST(QuditQPE_Backend, DM_EstimatesPhase_d2) {
    // Shift gate X: eigenvalue for k=1 is exp(2*pi*i*1/2) = -1 → phase = 0.5
    const auto U = shift_matrix(2, 1);
    const auto psi = shift_eigenstate_local(2, 1);
    auto r = QuditPhaseEstimation::estimate(3, 2, U, psi, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_NEAR(r.phase_estimate, 0.5, 0.25);  // loose tolerance: within 1 qubit
}

TEST(QuditQPE_Backend, MPS_EstimatesPhase_d2) {
    const auto U = shift_matrix(2, 1);
    const auto psi = shift_eigenstate_local(2, 1);
    auto r = QuditPhaseEstimation::estimate(3, 2, U, psi, 42, QuditBackend::MPS);
    EXPECT_NEAR(r.phase_estimate, 0.5, 0.25);
}

TEST(QuditQPE_Backend, SV_vs_DM_AgreeOnPhaseDigits_d3) {
    const auto U = shift_matrix(3, 1);
    const auto psi = shift_eigenstate_local(3, 1);
    auto sv = QuditPhaseEstimation::estimate(2, 3, U, psi, 42, QuditBackend::STATEVECTOR);
    auto dm = QuditPhaseEstimation::estimate(2, 3, U, psi, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_EQ(sv.phase_digits, dm.phase_digits);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend dispatch — QuditSimon (CLIFFORD must throw)
// ─────────────────────────────────────────────────────────────────────────────

// Coset-canonical Simon oracle: f(x) = min of {x + k*s mod d}
static std::function<std::vector<int>(const std::vector<int>&)>
make_simon_oracle(const std::vector<int>& s, int n, int d) {
    return [s, n, d](const std::vector<int>& x) -> std::vector<int> {
        std::vector<int> best = x;
        std::vector<int> cur(static_cast<size_t>(n));
        for (int k = 1; k < d; ++k) {
            for (int i = 0; i < n; ++i)
                cur[static_cast<size_t>(i)] =
                    (x[static_cast<size_t>(i)] + k * s[static_cast<size_t>(i)]) % d;
            if (cur < best) best = cur;
        }
        return best;
    };
}

static bool is_simon_period(const std::vector<int>& p, const std::vector<int>& s, int d) {
    for (int k = 1; k < d; ++k) {
        bool match = true;
        for (size_t i = 0; i < p.size() && match; ++i)
            match = ((k * s[i]) % d == p[i]);
        if (match) return true;
    }
    return false;
}

TEST(QuditSimon_Backend, Clifford_Throws) {
    const std::vector<int> s{1, 0};
    auto f = make_simon_oracle(s, 2, 2);
    EXPECT_THROW(
        QuditSimon::solve(2, 2, f, 3, 42, QuditBackend::CLIFFORD),
        std::invalid_argument);
}

TEST(QuditSimon_Backend, DM_FindsPeriod_d2) {
    const std::vector<int> s{1, 0};
    auto f = make_simon_oracle(s, 2, 2);
    auto r = QuditSimon::solve(2, 2, f, 3, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_FALSE(r.is_trivial);
    EXPECT_TRUE(is_simon_period(r.period, s, 2));
}

TEST(QuditSimon_Backend, MPS_FindsPeriod_d2) {
    const std::vector<int> s{1, 0};
    auto f = make_simon_oracle(s, 2, 2);
    auto r = QuditSimon::solve(2, 2, f, 3, 42, QuditBackend::MPS);
    EXPECT_FALSE(r.is_trivial);
    EXPECT_TRUE(is_simon_period(r.period, s, 2));
}

TEST(QuditSimon_Backend, DM_FindsPeriod_d3) {
    const std::vector<int> s{1, 2};
    auto f = make_simon_oracle(s, 2, 3);
    auto r = QuditSimon::solve(2, 3, f, 3, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_FALSE(r.is_trivial);
    EXPECT_TRUE(is_simon_period(r.period, s, 3));
}

TEST(QuditSimon_Backend, SV_vs_DM_BothFindPeriod_d2) {
    const std::vector<int> s{0, 1};
    auto f = make_simon_oracle(s, 2, 2);
    auto sv = QuditSimon::solve(2, 2, f, 3, 42, QuditBackend::STATEVECTOR);
    auto dm = QuditSimon::solve(2, 2, f, 3, 42, QuditBackend::DENSITY_MATRIX);
    EXPECT_FALSE(sv.is_trivial);
    EXPECT_FALSE(dm.is_trivial);
    EXPECT_TRUE(is_simon_period(sv.period, s, 2));
    EXPECT_TRUE(is_simon_period(dm.period, s, 2));
}

TEST(QuditSimon_Backend, CompositeD_FindsPeriod_d4) {
    // R.1.11.0: composite d is supported via the integer-SNF ring kernel
    // (formerly threw std::invalid_argument).
    const std::vector<int> s{2, 0};
    auto f = make_simon_oracle(s, 2, 4);
    auto r = QuditSimon::solve(2, 4, f, 6, 42);
    EXPECT_FALSE(r.is_trivial);
    EXPECT_TRUE(is_simon_period(r.period, s, 4));
}

// ─────────────────────────────────────────────────────────────────────────────
// DM vs SV agreement — comprehensive pure-state evolution checks
// ─────────────────────────────────────────────────────────────────────────────

TEST(DM_SV_Agreement, MultiGate_d2_n3) {
    const auto H    = hadamard_d2();
    const auto CNOT = cadd_matrix(2, 1);
    const auto X    = shift_matrix(2, 1);

    QuditStatevector sv(3, 2);
    sv.apply_1qudit(0, H);
    sv.apply_2qudit(0, 1, CNOT);
    sv.apply_1qudit(2, X);

    QuditDensityMatrix dm(3, 2);
    dm.apply_1qudit(0, H);
    dm.apply_2qudit(0, 1, CNOT);
    dm.apply_1qudit(2, X);

    for (size_t i = 0; i < sv.dim; ++i)
        EXPECT_NEAR(dm_prob(dm, i), sv_prob(sv, i), kTol) << "i=" << i;
    EXPECT_NEAR(dm_trace(dm), 1.0, kTol);
}

TEST(DM_SV_Agreement, MultiGate_d3_n2) {
    const auto F    = qft_matrix(3);
    const auto Finv = iqft_matrix(3);
    const auto X    = shift_matrix(3, 1);

    QuditStatevector sv(2, 3);
    sv.apply_1qudit(0, F);
    sv.apply_1qudit(1, X);
    sv.apply_1qudit(0, Finv);

    QuditDensityMatrix dm(2, 3);
    dm.apply_1qudit(0, F);
    dm.apply_1qudit(1, X);
    dm.apply_1qudit(0, Finv);

    for (size_t i = 0; i < sv.dim; ++i)
        EXPECT_NEAR(dm_prob(dm, i), sv_prob(sv, i), kTol) << "i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// MPS vs SV agreement — multi-gate pure-state evolution
// ─────────────────────────────────────────────────────────────────────────────

TEST(MPS_SV_Agreement, MultiGate_d2_n4) {
    const auto H    = hadamard_d2();
    const auto CNOT = cadd_matrix(2, 1);

    QuditStatevector sv(4, 2);
    sv.apply_1qudit(0, H);
    sv.apply_2qudit(0, 1, CNOT);
    sv.apply_1qudit(2, H);
    sv.apply_2qudit(2, 3, CNOT);

    QuditMPS mps(4, 2);
    mps.apply_1qudit(0, H);
    mps.apply_2qudit_adjacent(0, CNOT);
    mps.apply_1qudit(2, H);
    mps.apply_2qudit_adjacent(2, CNOT);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

TEST(MPS_SV_Agreement, MultiGate_d3_n2) {
    const auto F    = qft_matrix(3);
    const auto CADD = cadd_matrix(3, 1);

    QuditStatevector sv(2, 3);
    sv.apply_1qudit(0, F);
    sv.apply_2qudit(0, 1, CADD);

    QuditMPS mps(2, 3);
    mps.apply_1qudit(0, F);
    mps.apply_2qudit_adjacent(0, CADD);
    auto sv2 = mps.to_statevector();

    EXPECT_TRUE(sv_equal(sv, sv2, kLooseTol));
}

// ─────────────────────────────────────────────────────────────────────────────
// Error handling
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuditBV_Errors, EmptySecret_Throws) {
    EXPECT_THROW(QuditBernsteinVazirani::solve({}, 2), std::invalid_argument);
}

TEST(QuditBV_Errors, InvalidDimension_Throws) {
    EXPECT_THROW(QuditBernsteinVazirani::solve({1}, 1), std::invalid_argument);
}

TEST(QuditDJ_Errors, InvalidDimension_Throws) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    EXPECT_THROW(QuditDeutschJozsa::solve(1, 1, f), std::invalid_argument);
}

TEST(QuditQPE_Errors, WrongUSize_Throws) {
    const std::vector<Complex128> U_bad(3, Complex128(1.0, 0.0));
    const std::vector<Complex128> psi(2, Complex128(1.0, 0.0));
    EXPECT_THROW(QuditPhaseEstimation::estimate(1, 2, U_bad, psi), std::invalid_argument);
}

TEST(QuditQPE_Errors, InvalidM_Throws) {
    const auto U = shift_matrix(2, 1);
    const auto psi = shift_eigenstate_local(2, 0);
    EXPECT_THROW(QuditPhaseEstimation::estimate(0, 2, U, psi), std::invalid_argument);
}
