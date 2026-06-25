// R.1.12.1 total-coverage suite, Batch 1: lindblad/statevector.hpp.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 1: foundations".
//
// Covers every Statevector method: construction bounds, initialisation,
// amplitude access and bounds checks, probabilities, norms, inner products,
// sampling (determinism + distribution sanity + key conventions), cloning,
// move semantics, and the debug printer.

#include <gtest/gtest.h>

#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

static constexpr double kTol = 1e-12;

// =============================================================================
// Construction and bounds
// =============================================================================

TEST(R1121Statevector, ConstructorInitialisesToZeroKet) {
    Statevector sv(3);
    EXPECT_EQ(sv.num_qubits(), 3);
    EXPECT_EQ(sv.dimension(), 8u);
    EXPECT_EQ(sv.n_qubits, 3);
    EXPECT_EQ(sv.dim, 8u);
    EXPECT_NEAR(sv.probability(0), 1.0, kTol);
    for (size_t i = 1; i < sv.dim; ++i)
        EXPECT_NEAR(sv.probability(i), 0.0, kTol);
}

TEST(R1121Statevector, ConstructorBoundsThrow) {
    EXPECT_THROW(Statevector(0), std::invalid_argument);
    EXPECT_THROW(Statevector(-2), std::invalid_argument);
    EXPECT_THROW(Statevector(31), std::invalid_argument);
    EXPECT_NO_THROW(Statevector(1));
    EXPECT_NO_THROW(Statevector(12));
}

// =============================================================================
// Initialisation
// =============================================================================

TEST(R1121Statevector, InitializeResetsState) {
    Statevector sv(2);
    sv.initialize_basis(3);
    sv.initialize();
    EXPECT_NEAR(sv.probability(0), 1.0, kTol);
    EXPECT_NEAR(sv.probability(3), 0.0, kTol);
}

TEST(R1121Statevector, InitializeBasisSetsExactlyOneAmplitude) {
    Statevector sv(3);
    sv.initialize_basis(5);
    EXPECT_NEAR(sv.amplitude(5).real, 1.0, kTol);
    EXPECT_NEAR(sv.amplitude(5).imag, 0.0, kTol);
    EXPECT_NEAR(sv.norm_sq(), 1.0, kTol);
    EXPECT_NEAR(sv.probability(0), 0.0, kTol);
}

TEST(R1121Statevector, InitializeBasisOutOfRangeThrows) {
    Statevector sv(2);
    EXPECT_THROW(sv.initialize_basis(4), std::out_of_range);
    EXPECT_NO_THROW(sv.initialize_basis(3));
}

// =============================================================================
// set_amplitudes (both overloads)
// =============================================================================

TEST(R1121Statevector, SetAmplitudesFromArrays) {
    Statevector sv(1);
    const double re[2] = {INV_SQRT2, 0.0};
    const double im[2] = {0.0, INV_SQRT2};
    sv.set_amplitudes(re, im, 2);
    EXPECT_NEAR(sv.amplitude(0).real, INV_SQRT2, kTol);
    EXPECT_NEAR(sv.amplitude(1).imag, INV_SQRT2, kTol);
    EXPECT_NEAR(sv.norm_sq(), 1.0, kTol);
}

TEST(R1121Statevector, SetAmplitudesArrayCountMismatchThrows) {
    Statevector sv(2);
    const double re[2] = {1.0, 0.0};
    const double im[2] = {0.0, 0.0};
    EXPECT_THROW(sv.set_amplitudes(re, im, 2), std::invalid_argument);
}

TEST(R1121Statevector, SetAmplitudesFromVector) {
    Statevector sv(2);
    std::vector<Complex128> amps = {
        Complex128(0.5, 0.0), Complex128(0.0, 0.5),
        Complex128(-0.5, 0.0), Complex128(0.0, -0.5)
    };
    sv.set_amplitudes(amps);
    EXPECT_NEAR(sv.norm_sq(), 1.0, kTol);
    EXPECT_NEAR(sv.amplitude(2).real, -0.5, kTol);
    EXPECT_NEAR(sv.amplitude(3).imag, -0.5, kTol);

    std::vector<Complex128> wrong(3, Complex128(1.0, 0.0));
    EXPECT_THROW(sv.set_amplitudes(wrong), std::invalid_argument);
}

// =============================================================================
// Amplitude access, probabilities, bounds
// =============================================================================

TEST(R1121Statevector, AmplitudeAndProbabilityBoundsThrow) {
    Statevector sv(2);
    EXPECT_THROW(sv.amplitude(4), std::out_of_range);
    EXPECT_THROW(sv.probability(4), std::out_of_range);
}

TEST(R1121Statevector, AmplitudesVectorMatchesElementAccess) {
    Statevector sv(2);
    sv.initialize_basis(2);
    auto amps = sv.amplitudes();
    ASSERT_EQ(amps.size(), 4u);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(amps[i].real, sv.amplitude(i).real);
        EXPECT_EQ(amps[i].imag, sv.amplitude(i).imag);
    }
}

TEST(R1121Statevector, ProbabilitiesSumToOne) {
    Statevector sv(3);
    std::vector<Complex128> amps(8, Complex128(0.0, 0.0));
    // Asymmetric normalised state
    amps[1] = Complex128(0.6, 0.0);
    amps[6] = Complex128(0.0, 0.8);
    sv.set_amplitudes(amps);
    auto probs = sv.probabilities();
    double total = 0.0;
    for (double p : probs) total += p;
    EXPECT_NEAR(total, 1.0, kTol);
    EXPECT_NEAR(probs[1], 0.36, kTol);
    EXPECT_NEAR(probs[6], 0.64, kTol);
}

// =============================================================================
// Norms and normalisation
// =============================================================================

TEST(R1121Statevector, NormalizeRescalesToUnitNorm) {
    Statevector sv(1);
    sv.real_parts[0] = 3.0;
    sv.imag_parts[1] = 4.0;
    EXPECT_NEAR(sv.norm_sq(), 25.0, kTol);
    EXPECT_NEAR(sv.norm(), 5.0, kTol);
    sv.normalize();
    EXPECT_NEAR(sv.norm_sq(), 1.0, kTol);
    EXPECT_NEAR(sv.probability(0), 0.36, kTol);
    EXPECT_NEAR(sv.probability(1), 0.64, kTol);
}

TEST(R1121Statevector, NormalizeZeroStateThrows) {
    Statevector sv(1);
    sv.real_parts[0] = 0.0;  // wipe the |0> amplitude
    EXPECT_THROW(sv.normalize(), std::runtime_error);
}

// =============================================================================
// Inner product
// =============================================================================

TEST(R1121Statevector, InnerProductOrthonormalBasis) {
    Statevector a(2), b(2);
    a.initialize_basis(1);
    b.initialize_basis(1);
    Complex128 same = a.inner_product(b);
    EXPECT_NEAR(same.real, 1.0, kTol);
    EXPECT_NEAR(same.imag, 0.0, kTol);

    b.initialize_basis(2);
    Complex128 ortho = a.inner_product(b);
    EXPECT_NEAR(ortho.real, 0.0, kTol);
    EXPECT_NEAR(ortho.imag, 0.0, kTol);
}

TEST(R1121Statevector, InnerProductConjugateSymmetry) {
    Statevector a(1), b(1);
    std::vector<Complex128> av = {Complex128(0.6, 0.0), Complex128(0.0, 0.8)};
    std::vector<Complex128> bv = {Complex128(INV_SQRT2, 0.0),
                                  Complex128(0.5, 0.5)};
    a.set_amplitudes(av);
    b.set_amplitudes(bv);
    Complex128 ab = a.inner_product(b);
    Complex128 ba = b.inner_product(a);
    EXPECT_NEAR(ab.real, ba.real, kTol);
    EXPECT_NEAR(ab.imag, -ba.imag, kTol);  // <a|b> = conj(<b|a>)
}

TEST(R1121Statevector, InnerProductDimensionMismatchThrows) {
    Statevector a(1), b(2);
    EXPECT_THROW(a.inner_product(b), std::invalid_argument);
}

// =============================================================================
// Sampling: determinism, totals, key convention, distribution sanity
// =============================================================================

TEST(R1121Statevector, SampleCountsSeedDeterminism) {
    Statevector sv(2);
    std::vector<Complex128> amps = {
        Complex128(0.5, 0.0), Complex128(0.5, 0.0),
        Complex128(0.5, 0.0), Complex128(0.5, 0.0)
    };
    sv.set_amplitudes(amps);
    auto c1 = sv.sample_counts(256, 12345);
    auto c2 = sv.sample_counts(256, 12345);
    EXPECT_EQ(c1, c2);
}

TEST(R1121Statevector, SampleCountsTotalAndKeyWidth) {
    Statevector sv(3);
    sv.initialize_basis(0);
    auto counts = sv.sample_counts(100, 7);
    int total = 0;
    for (const auto& [bits, cnt] : counts) {
        EXPECT_EQ(bits.size(), 3u);
        total += cnt;
    }
    EXPECT_EQ(total, 100);
}

TEST(R1121Statevector, SampleCountsKeyConventionQubitZeroRightmost) {
    // Deterministic state |q1=0, q0=1> = amp index 1 must sample as "01".
    Statevector sv(2);
    sv.initialize_basis(1);
    auto counts = sv.sample_counts(32, 99);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.begin()->first, "01");
    EXPECT_EQ(counts.begin()->second, 32);

    // And the asymmetric partner: amp index 2 = |q1=1, q0=0> -> "10".
    sv.initialize_basis(2);
    auto counts2 = sv.sample_counts(32, 99);
    ASSERT_EQ(counts2.size(), 1u);
    EXPECT_EQ(counts2.begin()->first, "10");
}

TEST(R1121Statevector, SampleCountsDistributionSanity) {
    // 64/36 split: with 4096 shots, expect roughly 2621/1475 within ~5 sigma.
    Statevector sv(1);
    std::vector<Complex128> amps = {Complex128(0.8, 0.0), Complex128(0.0, 0.6)};
    sv.set_amplitudes(amps);
    auto counts = sv.sample_counts(4096, 4242);
    int n0 = counts.count("0") ? counts.at("0") : 0;
    int n1 = counts.count("1") ? counts.at("1") : 0;
    EXPECT_EQ(n0 + n1, 4096);
    EXPECT_GT(n0, 2400);
    EXPECT_LT(n0, 2850);
    EXPECT_GT(n1, 1250);
    EXPECT_LT(n1, 1700);
}

TEST(R1121Statevector, MeasureOnceSeededAndDeterministicState) {
    Statevector sv(3);
    sv.initialize_basis(5);  // q0 = 1, q2 = 1 -> "101"
    EXPECT_EQ(sv.measure_once(11), "101");
    EXPECT_EQ(sv.measure_once(12), "101");  // deterministic regardless of seed

    // Seeded reproducibility on a stochastic state.
    Statevector h(1);
    std::vector<Complex128> amps = {Complex128(INV_SQRT2, 0.0),
                                    Complex128(INV_SQRT2, 0.0)};
    h.set_amplitudes(amps);
    EXPECT_EQ(h.measure_once(777), h.measure_once(777));
}

// =============================================================================
// Clone and move semantics
// =============================================================================

TEST(R1121Statevector, CloneIsDeepAndIndependent) {
    Statevector sv(2);
    sv.initialize_basis(3);
    Statevector copy = sv.clone();
    EXPECT_NEAR(copy.probability(3), 1.0, kTol);

    copy.real_parts[3] = 0.0;
    copy.real_parts[0] = 1.0;
    EXPECT_NEAR(sv.probability(3), 1.0, kTol)
        << "mutating the clone must not affect the original";
}

TEST(R1121Statevector, MoveConstructionTransfersOwnership) {
    Statevector src(2);
    src.initialize_basis(2);
    Statevector dst(std::move(src));
    EXPECT_EQ(dst.num_qubits(), 2);
    EXPECT_NEAR(dst.probability(2), 1.0, kTol);
    // Moved-from object is the documented empty state.
    EXPECT_EQ(src.dim, 0u);
    EXPECT_EQ(src.real_parts, nullptr);
    EXPECT_EQ(src.imag_parts, nullptr);
}

TEST(R1121Statevector, MoveAssignmentTransfersOwnership) {
    Statevector a(1), b(2);
    b.initialize_basis(3);
    a = std::move(b);
    EXPECT_EQ(a.num_qubits(), 2);
    EXPECT_NEAR(a.probability(3), 1.0, kTol);
    EXPECT_EQ(b.dim, 0u);
}

// =============================================================================
// to_string
// =============================================================================

TEST(R1121Statevector, ToStringShowsNonZeroEntriesAndHeader) {
    Statevector sv(2);
    sv.initialize_basis(1);
    std::string s = sv.to_string();
    EXPECT_NE(s.find("Statevector(2 qubits"), std::string::npos);
    EXPECT_NE(s.find("|01"), std::string::npos)
        << "basis label for amp index 1 must render as 01 (q0 rightmost)";
    EXPECT_EQ(s.find("|11"), std::string::npos)
        << "zero-probability entries must not be listed";
}

TEST(R1121Statevector, ToStringTruncatesAfter32Entries) {
    Statevector sv(6);  // dim = 64 > 32
    std::vector<Complex128> amps(64, Complex128(0.125, 0.0));  // uniform
    sv.set_amplitudes(amps);
    std::string s = sv.to_string(3);
    EXPECT_NE(s.find("more entries"), std::string::npos);
}
