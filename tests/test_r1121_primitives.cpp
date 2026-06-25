// R.1.12.1 total-coverage suite, Batch 3: lindblad/primitives.hpp
// (Estimator, Sampler). Plan: docs (R.1.12.1 coverage plan), "Batch 3".
//
// Estimator expectations and gradients are pinned to analytic values
// (E(Rx(t)) = cos t, dE/dt = -sin t); exact vs sampled agree within
// statistical tolerance; the cache is transparent; shots==0 throws on
// measure circuits. Sampler determinism and distribution are checked.
// Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <unordered_map>
#include <vector>

using namespace lindblad;

namespace {
SparsePauliOp Z1() { return SparsePauliOp::from_list({{"Z", Complex128(1, 0)}}); }
SparsePauliOp X1() { return SparsePauliOp::from_list({{"X", Complex128(1, 0)}}); }
}

// =============================================================================
// Estimator — exact expectation values
// =============================================================================

TEST(R1121Primitives, EstimatorExactExpectation) {
    QuantumCircuit qc(1);
    qc.h(0);
    Estimator est;  // shots = 0 -> exact
    EXPECT_NEAR(est.run_single(qc, X1()), 1.0, 1e-9);   // <+|X|+> = 1
    EXPECT_NEAR(est.run_single(qc, Z1()), 0.0, 1e-9);   // <+|Z|+> = 0
}

TEST(R1121Primitives, EstimatorParameterisedMatchesCosine) {
    QuantumCircuit qc(1);
    qc.rx("theta", 0);
    Estimator est;
    EXPECT_NEAR(est.run_single(qc, Z1(), {0.7}), std::cos(0.7), 1e-9);
    EXPECT_NEAR(est.run_single(qc, Z1(), {0.0}), 1.0, 1e-9);
}

TEST(R1121Primitives, EstimatorRunBatchMatchesPerPoint) {
    QuantumCircuit qc(1);
    qc.rx("theta", 0);
    Estimator est;
    auto vals = est.run_batch(qc, Z1(), {{0.0}, {PI_2}, {PI}});
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_NEAR(vals[0], 1.0, 1e-9);
    EXPECT_NEAR(vals[1], 0.0, 1e-9);
    EXPECT_NEAR(vals[2], -1.0, 1e-9);
}

TEST(R1121Primitives, EstimatorGradientMatchesParameterShift) {
    QuantumCircuit qc(1);
    qc.rx("theta", 0);
    Estimator est;
    auto g = est.gradient(qc, Z1(), {0.7});
    ASSERT_EQ(g.size(), 1u);
    EXPECT_NEAR(g[0], -std::sin(0.7), 1e-7);  // d/dt cos t = -sin t
}

TEST(R1121Primitives, EstimatorCacheIsTransparent) {
    QuantumCircuit qc(1);
    qc.rx("theta", 0);
    Estimator est;
    double a = est.run_single(qc, Z1(), {0.3});
    double b = est.run_single(qc, Z1(), {0.3});  // served from cache
    est.clear_cache();
    double c = est.run_single(qc, Z1(), {0.3});  // recomputed
    EXPECT_NEAR(a, b, 1e-12);
    EXPECT_NEAR(a, c, 1e-12);
}

TEST(R1121Primitives, EstimatorSampledApproximatesExact) {
    QuantumCircuit qc(1);
    qc.ry("theta", 0);  // E(Ry(t)) for Z is cos t
    Estimator::Options opt;
    opt.shots = 8192;
    opt.seed = 7;
    Estimator est(opt);
    double sampled = est.run_single(qc, Z1(), {0.9});
    EXPECT_NEAR(sampled, std::cos(0.9), 0.05);
}

TEST(R1121Primitives, EstimatorShotsZeroThrowsOnMeasurement) {
    QuantumCircuit qc(1, 1);
    qc.h(0).measure(0, 0);
    Estimator est;  // shots = 0
    EXPECT_ANY_THROW(est.run_single(qc, Z1()));
}

// The four Estimator modes on the SAME instance: circuit X|0> = |1>, observable
// Z. Ideal <Z> = -1. With amplitude_damping(gamma) after X, rho = diag(gamma,
// 1-gamma) so <Z> = 2*gamma - 1. The exact and sampled variants of each must
// agree.
TEST(R1121Primitives, EstimatorFourModeMatrixAgrees) {
    const double gamma = 0.25;
    const double noisy = 2.0 * gamma - 1.0;  // = -0.5

    NoiseModel noise;
    noise.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(gamma), "x");

    auto make = [&](int shots, bool with_noise) {
        Estimator::Options o;
        o.shots = shots;
        o.seed = 20;
        if (with_noise) o.noise_model = noise;
        return Estimator(o);
    };
    QuantumCircuit qc(1);
    qc.x(0);

    // 1. exact (shots 0, ideal)
    EXPECT_NEAR(make(0, false).run_single(qc, Z1()), -1.0, 1e-9);
    // 2. sampled (shots > 0, ideal): |1> in Z basis is deterministic.
    EXPECT_NEAR(make(8192, false).run_single(qc, Z1()), -1.0, 1e-9);
    // 3. noisy-exact (shots 0 + noise): exact DM expectation.
    EXPECT_NEAR(make(0, true).run_single(qc, Z1()), noisy, 1e-9);
    // 4. noisy-sampled (shots > 0 + noise): within shot noise.
    EXPECT_NEAR(make(16384, true).run_single(qc, Z1()), noisy, 0.05);
}

TEST(R1121Primitives, EstimatorMultiTermSampledDecorrelatesPerTerm) {
    // Observable X + Z on |+>: <X> = 1, <Z> = 0 -> exact 1.0. Two non-identity
    // terms exercise the per-term seed increment; same seed is reproducible.
    auto obs = SparsePauliOp::from_list(
        {{"X", Complex128(1, 0)}, {"Z", Complex128(1, 0)}});
    QuantumCircuit qc(1);
    qc.h(0);
    Estimator::Options o;
    o.shots = 16384;
    o.seed = 9;
    Estimator est(o);
    double a = est.run_single(qc, obs);
    double b = est.run_single(qc, obs);
    EXPECT_NEAR(a, 1.0, 0.05) << "<X>+<Z> on |+> = 1";
    EXPECT_NEAR(a, b, 1e-12) << "same seed -> reproducible multi-term estimate";
}

// =============================================================================
// Sampler
// =============================================================================

TEST(R1121Primitives, SamplerSeedDeterminism) {
    QuantumCircuit qc(1, 1);
    qc.h(0).measure(0, 0);
    Sampler::Options opt;
    opt.shots = 512;
    opt.seed = 42;
    Sampler s(opt);
    auto c1 = s.run_single(qc);
    auto c2 = s.run_single(qc);
    EXPECT_EQ(c1, c2) << "same seed -> identical counts";
}

TEST(R1121Primitives, SamplerDistributionAndTotals) {
    QuantumCircuit qc(1, 1);
    qc.h(0).measure(0, 0);
    Sampler::Options opt;
    opt.shots = 4096;
    opt.seed = 3;
    Sampler s(opt);
    auto counts = s.run_single(qc);
    int n0 = counts.count("0") ? counts.at("0") : 0;
    int n1 = counts.count("1") ? counts.at("1") : 0;
    EXPECT_EQ(n0 + n1, 4096);
    EXPECT_GT(n0, 1700);
    EXPECT_GT(n1, 1700);
}

TEST(R1121Primitives, SamplerRunBatchReturnsPerCircuit) {
    QuantumCircuit bell(2, 2);
    bell.h(0).cx(0, 1).measure_all();
    QuantumCircuit one(1, 1);
    one.x(0).measure(0, 0);
    Sampler::Options opt;
    opt.shots = 256;
    opt.seed = 11;
    Sampler s(opt);
    auto results = s.run({bell, one});
    ASSERT_EQ(results.size(), 2u);
    // Bell: only correlated keys.
    for (const auto& [bits, n] : results[0])
        EXPECT_TRUE(bits == "00" || bits == "11");
    // x(0) measured -> always "1".
    ASSERT_EQ(results[1].size(), 1u);
    EXPECT_EQ(results[1].begin()->first, "1");
}

TEST(R1121Primitives, SamplerBatchSeedSchemeIsReproducibleAndDecorrelated) {
    // run() threads a distinct per-circuit seed; the whole batch is reproducible
    // across calls, and two identical stochastic circuits in one batch receive
    // DECORRELATED seeds (their exact counts differ).
    QuantumCircuit qc(3, 3);
    qc.h(0).h(1).h(2).measure_all();
    Sampler::Options opt;
    opt.shots = 2000;
    opt.seed = 77;
    Sampler s(opt);
    auto r1 = s.run({qc, qc});
    auto r2 = s.run({qc, qc});
    ASSERT_EQ(r1.size(), 2u);
    EXPECT_EQ(r1, r2) << "the batch is fully reproducible for a fixed seed";
    EXPECT_NE(r1[0], r1[1])
        << "identical circuits in one batch get decorrelated seeds";
}

TEST(R1121Primitives, SamplerNoisyPathLeaksProbability) {
    // x(0) then measure is ideally "1"; amplitude_damping(0.3) after x sends
    // P("0") ~ 0.3 via the density-matrix path.
    NoiseModel noise;
    noise.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.3), "x");
    QuantumCircuit qc(1, 1);
    qc.x(0).measure(0, 0);
    Sampler::Options opt;
    opt.shots = 8192;
    opt.seed = 4;
    opt.noise_model = noise;
    Sampler s(opt);
    auto counts = s.run_single(qc);
    int n0 = counts.count("0") ? counts.at("0") : 0;
    double f0 = double(n0) / 8192.0;
    EXPECT_GT(f0, 0.24) << "amplitude damping must leak |0> outcomes";
    EXPECT_LT(f0, 0.36);
}
