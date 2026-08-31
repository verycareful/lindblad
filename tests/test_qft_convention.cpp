// Regression guard for the gate-based QFT/IQFT under the project's LSB-at-
// qubit-0 (little-endian) convention. See CLAUDE.md "Project Conventions".
//
// History (R.1.10.5): pre-fix, no test verified the input/output convention
// or inverse property of QFT::build_circuit / build_inverse_circuit. The only
// existing QFT correctness tests were for the iterative variant in
// test_feedforward_qft.cpp. QPE tests only used phase=0 inputs which are
// symmetric under any bit-ordering, and the pre-PR-#7 Shor tests passed by
// continued-fraction lottery on garbage m values. PR #7 fixed Shor's slice
// but the underlying gate-based QFT was using MSB-at-qubit-0 internally,
// while QPE prepares LSB-convention Fourier states from controlled-U^(2^k) on
// qubit k. R.1.10.5 reworked QFT::build_circuit so do_swaps=true delivers a
// uniformly LSB-LSB QFT/IQFT; this test pins the convention down so a future
// regression is caught immediately.

#include <gtest/gtest.h>
#include "lindblad/circuit.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

namespace {

constexpr double kTol = 1e-9;

Statevector run_from_basis(int n_qubits, size_t basis_index,
                           const QuantumCircuit& circuit) {
    Statevector sv(n_qubits);
    sv.initialize_basis(basis_index);
    StatevectorSimulator sim;
    sim.simulate_circuit(sv, circuit);
    return sv;
}

Statevector run_from_amplitudes(int n_qubits,
                                const std::vector<Complex128>& amplitudes,
                                const QuantumCircuit& circuit) {
    Statevector sv(n_qubits);
    sv.set_amplitudes(amplitudes);
    StatevectorSimulator sim;
    sim.simulate_circuit(sv, circuit);
    return sv;
}

std::pair<size_t, double> find_peak_amp(const Statevector& sv) {
    size_t peak = 0;
    double max_p = -1.0;
    for (size_t i = 0; i < sv.dim; ++i) {
        const double p = sv.probability(i);
        if (p > max_p) { max_p = p; peak = i; }
    }
    return {peak, max_p};
}

void append_qft(QuantumCircuit& qc, int n, bool do_swaps) {
    auto sub = QFT::build_circuit(n, {do_swaps, /*approximation_degree=*/0,
                                       /*inverse=*/false});
    for (const auto& inst : sub.instructions)
        qc.instructions.push_back(inst);
}

void append_iqft(QuantumCircuit& qc, int n, bool do_swaps) {
    auto sub = QFT::build_inverse_circuit(n, do_swaps);
    for (const auto& inst : sub.instructions)
        qc.instructions.push_back(inst);
}

} // namespace

// =============================================================================
// Suite 1 — Roundtrip identity. QFT ∘ IQFT = identity for both do_swaps values.
// Convention-independent: if IQFT is the proper algebraic inverse of QFT, the
// roundtrip returns the input regardless of the convention either side uses.
// =============================================================================

TEST(QFTConvention, Roundtrip_N2_DoSwapsTrue) {
    const int n = 2;
    for (size_t k = 0; k < (1ULL << n); ++k) {
        QuantumCircuit qc(n);
        append_qft(qc, n, /*do_swaps=*/true);
        append_iqft(qc, n, /*do_swaps=*/true);
        auto sv = run_from_basis(n, k, qc);
        EXPECT_NEAR(sv.probability(k), 1.0, kTol)
            << "QFT∘IQFT on |amp[" << k << "]⟩ did not return |amp[" << k << "]⟩";
    }
}

TEST(QFTConvention, Roundtrip_N3_DoSwapsTrue) {
    const int n = 3;
    for (size_t k = 0; k < (1ULL << n); ++k) {
        QuantumCircuit qc(n);
        append_qft(qc, n, /*do_swaps=*/true);
        append_iqft(qc, n, /*do_swaps=*/true);
        auto sv = run_from_basis(n, k, qc);
        EXPECT_NEAR(sv.probability(k), 1.0, kTol)
            << "QFT∘IQFT on |amp[" << k << "]⟩ did not return |amp[" << k << "]⟩";
    }
}

TEST(QFTConvention, Roundtrip_N3_DoSwapsFalse) {
    const int n = 3;
    for (size_t k = 0; k < (1ULL << n); ++k) {
        QuantumCircuit qc(n);
        append_qft(qc, n, /*do_swaps=*/false);
        append_iqft(qc, n, /*do_swaps=*/false);
        auto sv = run_from_basis(n, k, qc);
        EXPECT_NEAR(sv.probability(k), 1.0, kTol)
            << "QFT∘IQFT (no swaps) on |amp[" << k << "]⟩ did not return |amp[" << k << "]⟩";
    }
}

// =============================================================================
// Suite 2 — IQFT on uniform superposition gives |amp[0]⟩.
// Convention-independent: the phase-0 Fourier state is uniform; its inverse
// QFT must concentrate on |amp[0]⟩ in any consistent convention.
// =============================================================================

TEST(QFTConvention, IQFT_Uniform_GivesAmpZero_N3_DoSwapsTrue) {
    const int n = 3;
    QuantumCircuit qc(n);
    for (int q = 0; q < n; ++q) qc.h(q);
    append_iqft(qc, n, /*do_swaps=*/true);
    auto sv = run_from_basis(n, 0, qc);
    EXPECT_NEAR(sv.probability(0), 1.0, kTol)
        << "IQFT|+⟩^3 (do_swaps=true) did not concentrate on amp[0]";
}

TEST(QFTConvention, IQFT_Uniform_GivesAmpZero_N3_DoSwapsFalse) {
    const int n = 3;
    QuantumCircuit qc(n);
    for (int q = 0; q < n; ++q) qc.h(q);
    append_iqft(qc, n, /*do_swaps=*/false);
    auto sv = run_from_basis(n, 0, qc);
    EXPECT_NEAR(sv.probability(0), 1.0, kTol)
        << "IQFT|+⟩^3 (do_swaps=false) did not concentrate on amp[0]";
}

// =============================================================================
// Suite 3 — Forward QFT|amp[1]⟩ for n=2: direct amplitude check pins down the
// LSB convention.
//
// Standard math: QFT|x⟩ = (1/√N) Σ_y exp(2πi xy/N) |y⟩.
// For n=2 (N=4):
//   QFT|0⟩ = (1/2)[1, 1, 1, 1]
//   QFT|1⟩ = (1/2)[1, i, -1, -i]
//   QFT|2⟩ = (1/2)[1, -1, 1, -1]
//   QFT|3⟩ = (1/2)[1, -i, -1, i]
//
// Initializing the statevector to |amp[1]⟩ means qubit 0 = 1, qubit 1 = 0.
// Under LSB convention this is integer x=1, so the output must equal QFT|1⟩.
// =============================================================================

TEST(QFTConvention, ForwardQFT_N2_Amp1_DoSwapsTrue_Matches_QFT1) {
    const int n = 2;
    QuantumCircuit qc(n);
    append_qft(qc, n, /*do_swaps=*/true);
    auto sv = run_from_basis(n, 1, qc);
    // Expected: QFT|1⟩ = (1/2)[1, i, -1, -i]
    EXPECT_NEAR(sv.amplitude(0).real,  0.5, kTol);
    EXPECT_NEAR(sv.amplitude(0).imag,  0.0, kTol);
    EXPECT_NEAR(sv.amplitude(1).real,  0.0, kTol);
    EXPECT_NEAR(sv.amplitude(1).imag,  0.5, kTol);
    EXPECT_NEAR(sv.amplitude(2).real, -0.5, kTol);
    EXPECT_NEAR(sv.amplitude(2).imag,  0.0, kTol);
    EXPECT_NEAR(sv.amplitude(3).real,  0.0, kTol);
    EXPECT_NEAR(sv.amplitude(3).imag, -0.5, kTol);
}

TEST(QFTConvention, ForwardQFT_N3_Amp3_DoSwapsTrue_Matches_QFT3) {
    // Independent confirmation at n=3. amp[3] = qubit 0 = qubit 1 = 1, qubit 2 = 0.
    // LSB → x = 3. QFT|3⟩ has coefficient (1/√8) exp(2πi 3y/8) at |y⟩.
    const int n = 3;
    const size_t N = 1ULL << n;
    QuantumCircuit qc(n);
    append_qft(qc, n, /*do_swaps=*/true);
    auto sv = run_from_basis(n, 3, qc);
    const double inv_sqrt_N = 1.0 / std::sqrt(static_cast<double>(N));
    for (size_t y = 0; y < N; ++y) {
        const double angle = 2.0 * PI * 3.0 * static_cast<double>(y) /
                             static_cast<double>(N);
        EXPECT_NEAR(sv.amplitude(y).real, std::cos(angle) * inv_sqrt_N, kTol)
            << "amp[" << y << "].real mismatch (LSB convention QFT|3⟩)";
        EXPECT_NEAR(sv.amplitude(y).imag, std::sin(angle) * inv_sqrt_N, kTol)
            << "amp[" << y << "].imag mismatch (LSB convention QFT|3⟩)";
    }
}

// =============================================================================
// Suite 4 — IQFT applied to a QPE-style Fourier state for known phase φ.
// Under the LSB convention, the peak amp index must equal m_expected = φ · 2^n.
// =============================================================================

TEST(QFTConvention, IQFT_QPE_FourierState_N4_PhaseQuarter_PeakAt_M4) {
    const int n = 4;
    const size_t N = 1ULL << n;
    const double phi = 0.25;
    const size_t expected_m = 4;  // φ · N

    std::vector<Complex128> amps(N);
    const double inv_sqrt_N = 1.0 / std::sqrt(static_cast<double>(N));
    for (size_t K = 0; K < N; ++K) {
        const double angle = 2.0 * PI * static_cast<double>(K) * phi;
        amps[K] = {std::cos(angle) * inv_sqrt_N, std::sin(angle) * inv_sqrt_N};
    }

    QuantumCircuit qc(n);
    append_iqft(qc, n, /*do_swaps=*/true);
    auto sv = run_from_amplitudes(n, amps, qc);

    auto [peak, prob] = find_peak_amp(sv);
    EXPECT_EQ(peak, expected_m)
        << "Peak at amp[" << peak << "], expected amp[" << expected_m
        << "]. Indicates LSB convention is broken in the IQFT.";
    EXPECT_GT(prob, 0.99)
        << "Fourier state for exact φ=1/4 should fully concentrate; got "
        << prob;
}

TEST(QFTConvention, IQFT_QPE_FourierState_N4_PhaseEighth_PeakAt_M2) {
    const int n = 4;
    const size_t N = 1ULL << n;
    const double phi = 0.125;
    const size_t expected_m = 2;

    std::vector<Complex128> amps(N);
    const double inv_sqrt_N = 1.0 / std::sqrt(static_cast<double>(N));
    for (size_t K = 0; K < N; ++K) {
        const double angle = 2.0 * PI * static_cast<double>(K) * phi;
        amps[K] = {std::cos(angle) * inv_sqrt_N, std::sin(angle) * inv_sqrt_N};
    }

    QuantumCircuit qc(n);
    append_iqft(qc, n, /*do_swaps=*/true);
    auto sv = run_from_amplitudes(n, amps, qc);

    auto [peak, prob] = find_peak_amp(sv);
    EXPECT_EQ(peak, expected_m)
        << "Peak at amp[" << peak << "], expected amp[" << expected_m << "]";
    EXPECT_GT(prob, 0.99)
        << "Fourier state for exact φ=1/8 should fully concentrate; got "
        << prob;
}

TEST(QFTConvention, IQFT_QPE_FourierState_N5_Phase_3_8ths_PeakAt_M12) {
    // Wider register to rule out small-n accidental symmetries.
    const int n = 5;
    const size_t N = 1ULL << n;
    const double phi = 3.0 / 8.0;
    const size_t expected_m = 12;  // φ · N = 12

    std::vector<Complex128> amps(N);
    const double inv_sqrt_N = 1.0 / std::sqrt(static_cast<double>(N));
    for (size_t K = 0; K < N; ++K) {
        const double angle = 2.0 * PI * static_cast<double>(K) * phi;
        amps[K] = {std::cos(angle) * inv_sqrt_N, std::sin(angle) * inv_sqrt_N};
    }

    QuantumCircuit qc(n);
    append_iqft(qc, n, /*do_swaps=*/true);
    auto sv = run_from_amplitudes(n, amps, qc);

    auto [peak, prob] = find_peak_amp(sv);
    EXPECT_EQ(peak, expected_m)
        << "Peak at amp[" << peak << "], expected amp[" << expected_m << "]";
    EXPECT_GT(prob, 0.99);
}

// =============================================================================
// Suite 4b — Scale check at n=11 (matches Shor n_eval for N=21). Verifies the
// LSB-LSB QFT/IQFT works for a wider register than Suite 4 covers.
// =============================================================================

TEST(QFTConvention, Roundtrip_N11_DoSwapsTrue_SeveralBasis) {
    const int n = 11;
    for (size_t k : {0u, 1u, 7u, 256u, 1024u, 2047u}) {
        QuantumCircuit qc(n);
        append_qft(qc, n, true);
        append_iqft(qc, n, true);
        auto sv = run_from_basis(n, k, qc);
        EXPECT_NEAR(sv.probability(k), 1.0, kTol)
            << "n=11 QFT∘IQFT on |amp[" << k << "]⟩ did not return |amp[" << k << "]⟩";
    }
}

TEST(QFTConvention, IQFT_QPE_FourierState_N11_PhaseQuarter_PeakAt_M512) {
    // Exact phase 1/4 at n=11: m = 512. IQFT must concentrate fully on amp[512].
    const int n = 11;
    const size_t N = 1ULL << n;
    const double phi = 0.25;
    const size_t expected_m = 512;

    std::vector<Complex128> amps(N);
    const double inv_sqrt_N = 1.0 / std::sqrt(static_cast<double>(N));
    for (size_t K = 0; K < N; ++K) {
        const double angle = 2.0 * PI * static_cast<double>(K) * phi;
        amps[K] = {std::cos(angle) * inv_sqrt_N, std::sin(angle) * inv_sqrt_N};
    }

    QuantumCircuit qc(n);
    append_iqft(qc, n, /*do_swaps=*/true);
    auto sv = run_from_amplitudes(n, amps, qc);

    auto [peak, prob] = find_peak_amp(sv);
    EXPECT_EQ(peak, expected_m)
        << "n=11 exact-phase IQFT peak at amp[" << peak
        << "], expected amp[" << expected_m << "]";
    EXPECT_GT(prob, 0.99);
}

TEST(QFTConvention, IQFT_QPE_FourierState_N11_PhaseSixth_Spread_PeakNear_M341) {
    // Non-exact phase 1/6 at n=11: expected m = round(2048/6) = 341. QPE has
    // sinc spread; standard bound P(closest m) >= 4/π² ≈ 0.405. The peak must
    // land at m=341 (or 342 on a near-equal split) with prob >= 0.35. This is
    // the case Shor N=21 exercises via s=1 collapse.
    const int n = 11;
    const size_t N = 1ULL << n;
    const double phi = 1.0 / 6.0;

    std::vector<Complex128> amps(N);
    const double inv_sqrt_N = 1.0 / std::sqrt(static_cast<double>(N));
    for (size_t K = 0; K < N; ++K) {
        const double angle = 2.0 * PI * static_cast<double>(K) * phi;
        amps[K] = {std::cos(angle) * inv_sqrt_N, std::sin(angle) * inv_sqrt_N};
    }

    QuantumCircuit qc(n);
    append_iqft(qc, n, /*do_swaps=*/true);
    auto sv = run_from_amplitudes(n, amps, qc);

    auto [peak, prob] = find_peak_amp(sv);
    std::cout << "\n[DIAGNOSTIC] IQFT n=11 φ=1/6 peak at amp[" << peak
              << "] prob " << prob << " (expected near 341)\n";
    // Top 5 amps for inspection.
    std::vector<std::pair<size_t, double>> ranked;
    for (size_t i = 0; i < N; ++i) ranked.emplace_back(i, sv.probability(i));
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "    amp[" << ranked[i].first << "] prob "
                  << ranked[i].second << "\n";
    }

    EXPECT_TRUE(peak == 341 || peak == 342)
        << "Peak should be at amp[341] or amp[342] for φ=1/6 at n=11";
    EXPECT_GE(prob, 0.35) << "QPE lower-bound 4/π² ≈ 0.405";
}

// =============================================================================
// Suite 5 — End-to-end QPE smoke test with non-zero phase.
// S gate on |1⟩: S|1⟩ = i|1⟩, phase = 1/4. With n_eval=2 the QPE is exact
// (S^4 = I, so n_eval > 2 wastes precision and spreads probability). Expected
// m = 1 (φ · 2^2 = 1), bitstring "101" (target=1, eval q_1=0, eval q_0=1).
// =============================================================================

TEST(QFTConvention, QPE_SGate_PhaseQuarter_N2eval_TopBitstringIs_101) {
    const int n_eval = 2;
    const int n_target = 1;
    const int total = n_eval + n_target;

    QuantumCircuit qc(total);

    // Eval register in uniform superposition.
    for (int q = 0; q < n_eval; ++q) qc.h(q);

    // Target = |1⟩ (eigenstate of S with phase 1/4).
    qc.x(n_eval);

    // Controlled-S^(2^k) on (ctrl=qubit k, target=qubit n_eval). S = diag(1, i),
    // so S^(2^k) on |1⟩ adds phase i^(2^k) = exp(2πi 2^k / 4).
    for (int k = 0; k < n_eval; ++k) {
        const double lambda = (PI / 2.0) * static_cast<double>(1ULL << k);
        qc.cp(lambda, k, n_eval);
    }

    // LSB-LSB IQFT on eval register.
    auto iqft = QFT::build_inverse_circuit(n_eval, /*do_swaps=*/true);
    for (const auto& inst : iqft.instructions)
        qc.instructions.push_back(inst);

    qc.measure_all();

    StatevectorSimulator sim;
    auto result = sim.run(qc, 1024, 42);
    ASSERT_TRUE(result.success);

    // Sort counts descending.
    std::vector<std::pair<std::string, int>> sorted(result.counts.begin(),
                                                     result.counts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    ASSERT_FALSE(sorted.empty());

    // Bitstring layout: 3 chars = target (qubit 2, leftmost) | eval q_1 | eval q_0.
    // Expected m=1: eval q_0=1, q_1=0 → eval part "01"; target=1 (X applied) → "1".
    // Full bitstring: "101".
    EXPECT_EQ(sorted[0].first, "101")
        << "Top bitstring was \"" << sorted[0].first
        << "\" with count " << sorted[0].second
        << "; expected \"101\" (m=1, target=1) under LSB convention";
    EXPECT_EQ(sorted[0].second, 1024)
        << "Exact QPE (φ=1/4, n_eval=2) should give all 1024 shots on \"101\"";
}

// =============================================================================
// Suite 6 — Shor full-pipeline diagnostic for N=21 (order 6, n_eval=11).
//
// Builds the period-finding circuit directly and prints, for several seeds, the
// most-frequent bitstring and the m value the LSB-convention extraction
// derives. Lets us see whether (a) the IQFT scales to n_eval=11, (b) the m
// values cluster near the expected QPE peaks (s/6 · 2048 for s∈{0..5}), or (c)
// something downstream of the IQFT (controlled-U for n_target=5, target
// preparation, etc.) is producing the wrong distribution.
//
// Useful s values for r=6 are s ∈ {1, 5} (gcd(s, 6) = 1). Other s either give
// m=0 (rejected) or m peaks whose cf convergents have denominators that fail
// `mod_pow(2, r, 21) == 1`.
// =============================================================================

TEST(QFTConvention, Shor_N21_a2_PeaksLandWhereTheOrderPredicts) {
    const uint64_t a = 2, N = 21;
    const int n_target = 5;
    const int n_eval = 2 * n_target + 1;  // = 11
    const int total = n_eval + n_target;  // = 16
    const size_t Nm = 1ULL << n_eval;     // 2048

    // The order is derived rather than asserted from memory: r is the least
    // exponent with a^r == 1 (mod N), and every expectation below is a function
    // of it, so this test states the theory and not a previous run's output.
    uint64_t r = 0;
    for (uint64_t e = 1; e < N; ++e) {
        uint64_t acc = 1;
        for (uint64_t i = 0; i < e; ++i) acc = (acc * a) % N;
        if (acc == 1) { r = e; break; }
    }
    ASSERT_GT(r, 1u) << "a=" << a << " has no order mod N=" << N;

    backends::LocalBackend::Config cfg;
    cfg.simulator = backends::LocalBackend::SimType::STATEVECTOR;
    backends::LocalBackend backend(cfg);

    auto circuit = Shor::build_period_finding_circuit(a, N, n_eval, n_target);
    circuit.measure_all();

    int valid_cf_count = 0;
    for (uint64_t seed = 1; seed <= 10; ++seed) {
        auto br = backend.run(circuit, 128, seed);
        std::string best;
        int best_count = 0;
        for (const auto& [bits, count] : br.counts) {
            if (count > best_count) { best_count = count; best = bits; }
        }
        ASSERT_EQ(best.size(), static_cast<size_t>(total))
            << "seed " << seed << ": counts key is not the full register";

        // Apply Shor's LSB-convention extraction.
        uint64_t m = 0;
        for (int q = 0; q < n_eval && q < total; ++q) {
            if (best[total - 1 - q] == '1') m |= (1ULL << q);
        }

        // The eigenvalue phases are k/r, so the ideal peaks sit at
        // round(k * 2^n_eval / r). The mode over a shot batch must land on one
        // of those integers or its immediate neighbour: the sinc envelope puts
        // almost all of a peak's weight on the nearest bin, and the neighbour is
        // reachable only when k * 2^n_eval / r falls near a half-integer.
        uint64_t nearest = 0;
        for (uint64_t k = 0; k < r; ++k) {
            const uint64_t ideal =
                static_cast<uint64_t>(std::llround(
                    static_cast<double>(k) * static_cast<double>(Nm) /
                    static_cast<double>(r)));
            const uint64_t dist = (m > ideal) ? (m - ideal) : (ideal - m);
            if (k == 0 || dist < nearest) nearest = dist;
        }
        EXPECT_LE(nearest, 1u)
            << "seed " << seed << ": the peak landed at m=" << m
            << ", which is " << nearest << " away from every multiple of "
            << Nm << "/" << r << ". A peak off the predicted lattice means the "
            << "evaluation register is not holding the phase the theory says, "
            << "which is a convention or a controlled-U defect rather than "
            << "sampling noise.";

        double phi_est = static_cast<double>(m) / static_cast<double>(Nm);
        auto convs = Shor::cf_convergents(phi_est, N);
        uint64_t found_r = 0;
        for (const auto& [sv, rv] : convs) {
            if (rv == 0 || rv >= N) continue;
            uint64_t base = a % N, exp = rv, result = 1;
            while (exp > 0) {
                if (exp & 1) result = (result * base) % N;
                base = (base * base) % N;
                exp >>= 1;
            }
            if (result == 1) { found_r = rv; break; }
        }
        if (found_r > 0) {
            ++valid_cf_count;
            EXPECT_EQ(r % found_r, 0u)
                << "seed " << seed << ": recovered " << found_r
                << ", which is not a divisor of the true order " << r;
        }
    }

    // Recovery cannot succeed on every seed by construction: the k=0 peak
    // carries phase 0, whose convergents yield no usable denominator, and it is
    // a legitimate outcome of the measurement. What must hold is that the
    // continued-fraction stage recovers the order at all, since a pipeline that
    // never does is indistinguishable from one that produces noise.
    EXPECT_GE(valid_cf_count, 1)
        << "no seed recovered the order across 10 batches of 128 shots, so "
           "either the peaks are wrong or the convergent search is";
}
