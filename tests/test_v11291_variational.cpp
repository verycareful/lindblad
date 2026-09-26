// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.1 test wave - VQE, QAOA and MA-QAOA through the optimizer seam, and
// the starting point they draw.
//
// 1.1.29.0 fixed three defects that each made an option a no-op or a read
// undefined: VQE ignored its seed, QAOA never wrote num_iterations, and both
// MA-QAOA paths ran COBYLA whatever the optimizer option said. It also added
// initial_step and a fourth optimizer name. Every one of those is observable
// only through what a run returns, so every test here compares runs: the same
// options give the same bits, and changing one option changes the run. A test
// that looked at one run in isolation could not tell an honoured option from
// an ignored one.
//
// The initial draw is documented as bit-identical across compilers and
// floating-point models for a given seed. One binary cannot run under two
// compilers, so the claim is pinned against a reference computed in integer
// arithmetic from the engine's output words, which the standard fixes for
// std::mt19937_64. The reference rounds the exact value of
// lo + (hi - lo) * k * 2^-53 once, to nearest even, with no floating-point
// operation that a compiler flag could change. Every leg of the suite compares
// against the same numbers, so the legs agree with each other exactly when they
// all pass.
//
// FOUR TESTS SHIP RED in this release:
//   V11291Variational.AnExactExpectationRefusesATermWiderThanTheState
//   V11291Variational.TheExactEstimatorRefusesATermWiderThanTheCircuit
//     The exact statevector expectation takes each Pauli term's width from
//     the string and never compares it with the state: a Z beyond the last
//     qubit is evaluated as if that qubit were |0>, and an X or Y beyond it
//     reads past the amplitude arrays. The estimator's default (exact) path
//     inherits it, so a Hamiltonian wider than the ansatz produces a number.
//   V11291Variational.QaoaBobyqaWithAStepTooWideForTheBoxIsAnInvalidArgument
//     NLopt's BOBYQA refuses a box narrower than twice the initial step before
//     evaluating anything, and the refusal surfaces as a runtime_error saying
//     the objective produced no finite value, which names the wrong cause.
//   V11291UniformDraw.TheHighestWordStaysBelowHiOnEveryOrderedRange
//     The draw is documented as [lo, hi), and on a range where hi is large
//     against the width the top engine word rounds onto hi. No algorithm draws
//     from such a range; every range they use is symmetric about zero, where
//     the top draw stays at least one spacing below hi.
// The library changes that turn them green belong to the next patch release.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/optimizer.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/validation.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

namespace {

const char* const kOptimizers[] = {"COBYLA", "NLOPT_COBYLA", "NELDER_MEAD", "BOBYQA"};
// unsigned long long, like the literal seeds it is listed beside, so a braced
// list of seeds deduces one element type.
constexpr unsigned long long kSeed = 11291ULL;

std::vector<std::string> capture_warnings(const std::function<void()>& fn) {
    flush_warnings();
    std::vector<std::string> captured;
    set_warning_handler([&captured](const std::string& m) { captured.push_back(m); });
    fn();
    set_warning_handler(nullptr);
    return captured;
}

bool any_contains(const std::vector<std::string>& msgs, const std::string& needle) {
    for (const auto& m : msgs)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

// =============================================================================
// Exact reference for the documented draw
// =============================================================================

// Unsigned 128-bit value as two limbs. Only what the reference needs.
struct U128 {
    std::uint64_t hi = 0, lo = 0;
};

U128 mul64(std::uint64_t a, std::uint64_t b) {
    const std::uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32;
    const std::uint64_t b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
    const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const std::uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFu) + (p10 & 0xFFFFFFFFu);
    U128 r;
    r.lo = (mid << 32) | (p00 & 0xFFFFFFFFu);
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}

U128 shl(U128 v, int n) {
    if (n == 0) return v;
    if (n >= 64) return {v.lo << (n - 64), 0};
    return {(v.hi << n) | (v.lo >> (64 - n)), v.lo << n};
}

bool less(U128 a, U128 b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; }

U128 add(U128 a, U128 b) {
    U128 r{a.hi + b.hi, a.lo + b.lo};
    if (r.lo < a.lo) ++r.hi;
    return r;
}

U128 sub(U128 a, U128 b) {  // requires a >= b
    U128 r{a.hi - b.hi, a.lo - b.lo};
    if (a.lo < b.lo) --r.hi;
    return r;
}

int bit_length(U128 v) {
    return v.hi ? 128 - std::countl_zero(v.hi) : 64 - std::countl_zero(v.lo);
}

// |x| = m * 2^e with m an integer below 2^53. frexp and ldexp are exact.
void decompose(double x, std::uint64_t& m, int& e) {
    int ex = 0;
    const double f = std::frexp(std::abs(x), &ex);
    m = static_cast<std::uint64_t>(std::ldexp(f, 53));
    e = ex - 53;
}

// v * 2^e, rounded once to the nearest double, ties to even.
double round_once(U128 v, int e, bool negative) {
    if (v.hi == 0 && v.lo == 0) return 0.0;
    const int len = bit_length(v);
    std::uint64_t q;
    if (len > 53) {
        const int sh = len - 53;
        // q = v >> sh, and the discarded bits compared against half an ulp.
        const U128 shifted = sh >= 64 ? U128{0, v.hi >> (sh - 64)}
                                      : U128{v.hi >> sh, (v.lo >> sh) | (sh ? v.hi << (64 - sh) : 0)};
        q = shifted.lo;
        const U128 kept = shl(shifted, sh);
        const U128 rem = sub(v, kept);
        const U128 half = shl(U128{0, 1}, sh - 1);
        if (less(half, rem) || (!less(rem, half) && (q & 1u))) ++q;
        e += sh;
    } else {
        q = v.lo;
    }
    const double r = std::ldexp(static_cast<double>(q), e);
    return negative ? -r : r;
}

// The documented draw: k = word >> 11, then lo + (hi - lo) * k * 2^-53 rounded
// once. (hi - lo) is the double the rule multiplies by, so it is formed the same
// way here; every step after it is integer arithmetic.
double exact_draw(std::uint64_t word, double lo, double hi) {
    const double width = hi - lo;
    const std::uint64_t k = word >> 11;

    std::uint64_t wm = 0, lm = 0;
    int we = 0, le = 0;
    decompose(width, wm, we);
    const U128 prod = mul64(wm, k);   // width * k, exponent we - 53
    const int pe = we - 53;

    if (lo == 0.0) return round_once(prod, pe, false);
    decompose(lo, lm, le);

    const int e = std::min(pe, le);
    const U128 p = shl(prod, pe - e);
    const U128 l = shl(U128{0, lm}, le - e);
    EXPECT_LT(bit_length(p), 126) << "reference range too wide for two limbs";
    EXPECT_LT(bit_length(l), 126) << "reference range too wide for two limbs";

    if (lo > 0.0) return round_once(add(p, l), e, false);
    // lo < 0: the sum is prod - |lo|.
    if (!less(p, l)) return round_once(sub(p, l), e, false);
    return round_once(sub(l, p), e, true);
}

// The words the next `count` draws consume, from an engine in the given state.
std::vector<std::uint64_t> words(std::mt19937_64 rng, int count) {
    std::vector<std::uint64_t> w(static_cast<std::size_t>(count));
    for (auto& x : w) x = rng();
    return w;
}

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

// =============================================================================
// An engine whose next output is a chosen word
// =============================================================================

// Inverse of mt19937_64's output tempering, one xorshift at a time. Each step
// y ^= (y >> s) & m (or << s) is undone by iterating it until the shifted term
// has run off the word.
std::uint64_t untemper(std::uint64_t y) {
    auto undo_right = [](std::uint64_t v, int s, std::uint64_t m) {
        std::uint64_t x = v;
        for (int i = 0; i * s < 64; ++i) x = v ^ ((x >> s) & m);
        return x;
    };
    auto undo_left = [](std::uint64_t v, int s, std::uint64_t m) {
        std::uint64_t x = v;
        for (int i = 0; i * s < 64; ++i) x = v ^ ((x << s) & m);
        return x;
    };
    y = undo_right(y, 43, ~0ULL);
    y = undo_left(y, 37, 0xFFF7EEE000000000ULL);
    y = undo_left(y, 17, 0x71D67FFFEDA60000ULL);
    y = undo_right(y, 29, 0x5555555555555555ULL);
    return y;
}

// Loads a state from which the next output is `target`. The standard's text
// form is the 312 state words, and the next call regenerates; with words 0 and
// 1 zero, regeneration makes the first new word equal word m = 156, so that
// slot carries the untempered target. libstdc++ appends its position index to
// the text form; writing 312 there asks for the same regeneration.
std::mt19937_64 engine_emitting(std::uint64_t target) {
    constexpr std::size_t n = 312, m = 156;
    std::ostringstream probe;
    probe << std::mt19937_64{};
    std::istringstream count_in(probe.str());
    std::size_t tokens = 0;
    for (std::string t; count_in >> t;) ++tokens;

    std::ostringstream text;
    for (std::size_t i = 0; i < n; ++i) text << (i == m ? untemper(target) : 0u) << ' ';
    if (tokens == n + 1) text << n;

    std::mt19937_64 rng;
    std::istringstream in(text.str());
    in >> rng;
    EXPECT_FALSE(in.fail()) << "the crafted engine state did not load";
    return rng;
}

// =============================================================================
// Problems small enough to run many times
// =============================================================================

SparsePauliOp vqe_hamiltonian() {
    return SparsePauliOp({PauliString("ZZ", Complex128(1.0, 0.0)),
                          PauliString("XI", Complex128(0.5, 0.0)),
                          PauliString("IZ", Complex128(-0.3, 0.0))});
}

// The three algorithms own an Estimator, whose cache mutex makes them neither
// copyable nor movable, so each is configured where it is declared.
void configure(VQE& vqe, const std::string& optimizer, std::uint64_t seed = kSeed,
               int cap = 60) {
    vqe.options.optimizer = optimizer;
    vqe.options.seed = seed;
    vqe.options.max_iterations = cap;
}

// Three-qubit MaxCut on a path with an asymmetric weight, so the optimum is not
// symmetric under qubit reversal.
SparsePauliOp cut_hamiltonian() {
    return SparsePauliOp({PauliString("ZZI", Complex128(1.0, 0.0)),
                          PauliString("IZZ", Complex128(0.6, 0.0))});
}

void configure(QAOA& q, const std::string& optimizer, std::uint64_t seed = kSeed, int p = 2,
               int cap = 50) {
    q.options.optimizer = optimizer;
    q.options.seed = seed;
    q.options.p = p;
    q.options.max_iterations = cap;
    q.sampler.options.shots = 256;
}

void configure(MAQAOA& m, const std::string& optimizer, bool layerwise,
               std::uint64_t seed = kSeed, int cap = 40) {
    m.options.optimizer = optimizer;
    m.options.layerwise = layerwise;
    m.options.seed = seed;
    m.options.p = 2;
    m.options.max_iterations = cap;
    m.sampler.options.shots = 256;
    m.sampler.options.seed = 7;
}

// FLOP's COBYLA and NLopt's are two implementations of one algorithm, so the
// pair may legitimately walk the same path. Every other pair is a different
// method, and two of them producing the same run means one name was ignored.
bool comparable(std::size_t i, std::size_t j) { return !(i == 0 && j == 1); }

bool same_run(const MAQAOA::Result& a, const MAQAOA::Result& b) {
    return a.initial_params == b.initial_params && a.optimal_params == b.optimal_params &&
           same_bits(a.optimal_value, b.optimal_value) &&
           a.num_iterations == b.num_iterations && a.layer_nfev == b.layer_nfev;
}

}  // namespace

// =============================================================================
// Defaults
// =============================================================================

TEST(V11291Variational, TheDocumentedOptionDefaults) {
    const VQE::Options v{};
    EXPECT_EQ(v.optimizer, "COBYLA");
    EXPECT_EQ(v.initial_step, 0.3);
    EXPECT_EQ(v.seed, 0u);
    EXPECT_EQ(v.max_iterations, 100);

    const QAOA::Options q{};
    EXPECT_EQ(q.optimizer, "COBYLA");
    EXPECT_EQ(q.initial_step, 0.3);

    const MAQAOA::Options m{};
    EXPECT_EQ(m.optimizer, "COBYLA");
    EXPECT_EQ(m.initial_step, 0.3);
    EXPECT_EQ(m.max_iterations, 200);
}

// =============================================================================
// VQE
// =============================================================================

TEST(V11291Variational, VqeReportsTheLowestEnergyItEvaluated) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        VQE vqe;
        configure(vqe, opt);
        const auto r = vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);

        ASSERT_FALSE(r.energy_history.empty());
        EXPECT_EQ(r.eigenvalue,
                  *std::min_element(r.energy_history.begin(), r.energy_history.end()));
        // The parameters reported are the ones that energy came from: the
        // estimator is exact at shots == 0, so re-evaluating them reproduces it.
        Estimator est;
        EXPECT_TRUE(same_bits(est.run_single(ansatz, vqe_hamiltonian(), r.optimal_parameters),
                              r.eigenvalue));
    }
}

TEST(V11291Variational, VqeNumIterationsIsTheHistoryLength) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        VQE vqe;
        configure(vqe, opt);
        const auto r = vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
        EXPECT_EQ(static_cast<std::size_t>(r.num_iterations), r.energy_history.size());
        EXPECT_LE(r.num_iterations, 60);
    }
}

TEST(V11291Variational, VqeTheSameSeedGivesTheSameRunBitForBit) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        VQE a, b;
        configure(a, opt);
        configure(b, opt);
        const auto ra = a.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
        const auto rb = b.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
        EXPECT_EQ(ra.energy_history, rb.energy_history);
        EXPECT_EQ(ra.optimal_parameters, rb.optimal_parameters);
    }
}

TEST(V11291Variational, VqeTheSeedDecidesTheStart) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    VQE a, b;
    configure(a, "COBYLA", 1);
    configure(b, "COBYLA", 2);
    const auto ra = a.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
    const auto rb = b.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
    EXPECT_NE(ra.energy_history.front(), rb.energy_history.front())
        << "two seeds evaluated the same starting energy";
}

TEST(V11291Variational, VqeSeedZeroDrawsAFreshStartEachRun) {
    // Two draws from std::random_device landing on the same four angles has
    // probability of order 2^-200.
    const auto ansatz = VQE::real_amplitudes(2, 1);
    VQE a, b;
    configure(a, "COBYLA", 0, 5);
    configure(b, "COBYLA", 0, 5);
    const auto ra = a.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
    const auto rb = b.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
    EXPECT_NE(ra.energy_history.front(), rb.energy_history.front());
}

TEST(V11291Variational, VqeStartsFromTheDocumentedDraw) {
    // No initial_params: the start is one draw per parameter in [-pi, pi)
    // under options.seed, and the first energy evaluated is the energy there.
    const auto ansatz = VQE::real_amplitudes(2, 1);
    const int n = ansatz.num_parameters();
    for (std::uint64_t seed : {1ULL, 42ULL, kSeed, 0xFFFFFFFFFFFFULL}) {
        SCOPED_TRACE(seed);
        std::vector<double> x0;
        for (std::uint64_t w : words(std::mt19937_64(seed), n))
            x0.push_back(exact_draw(w, -PI, PI));

        VQE vqe;

        configure(vqe, "COBYLA", seed, 3);
        const auto r = vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
        Estimator est;
        EXPECT_TRUE(same_bits(r.energy_history.front(),
                              est.run_single(ansatz, vqe_hamiltonian(), x0)));
    }
}

TEST(V11291Variational, VqeSuppliedParametersAreTheStartWhateverTheSeed) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    const std::vector<double> x0 = {0.1, -0.2, 0.3, -0.4};
    Estimator est;
    const double e0 = est.run_single(ansatz, vqe_hamiltonian(), x0);
    for (std::uint64_t seed : {1ULL, 2ULL}) {
        VQE vqe;
        configure(vqe, "COBYLA", seed, 5);
        const auto r = vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz, x0);
        EXPECT_TRUE(same_bits(r.energy_history.front(), e0));
    }
}

TEST(V11291Variational, VqeRunsTheNamedMinimiserWithItsOptions) {
    // VQE's objective is documented as the estimator on the ansatz and the
    // Hamiltonian, so the whole run can be replayed through the seam directly:
    // same backend, cap, tolerance and step, same start. Every energy must
    // match in order, which is only possible if each option reached the
    // backend it names.
    const auto ansatz = VQE::real_amplitudes(2, 1);
    const int n = ansatz.num_parameters();
    std::vector<double> x0;
    for (std::uint64_t w : words(std::mt19937_64(kSeed), n)) x0.push_back(exact_draw(w, -PI, PI));

    for (const char* opt : kOptimizers) {
        for (double step : {0.3, 0.07}) {
            SCOPED_TRACE(std::string(opt) + " step " + std::to_string(step));
            VQE vqe;
            configure(vqe, opt, kSeed, 45);
            vqe.options.initial_step = step;
            vqe.options.convergence_threshold = 1e-5;
            const auto r = vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);

            detail::OptimizerSpec spec;
            ASSERT_TRUE(detail::parse_optimizer_name(opt, spec.backend));
            spec.max_evaluations = 45;
            spec.xtol_rel = 1e-5;
            spec.initial_step = step;
            Estimator est;
            std::vector<double> replay;
            (void)detail::minimize(
                spec,
                [&](std::span<const double> x) {
                    const double e = est.run_single(ansatz, vqe_hamiltonian(),
                                                    std::vector<double>(x.begin(), x.end()));
                    replay.push_back(e);
                    return e;
                },
                x0, "replay");

            EXPECT_EQ(r.energy_history, replay);
        }
    }
}

TEST(V11291Variational, VqeEachOptimizerNameRunsADifferentMinimiser) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    std::vector<std::vector<double>> histories;
    for (const char* opt : kOptimizers) {
        VQE vqe;
        configure(vqe, opt);
        histories.push_back(vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz)
                                .energy_history);
    }
    for (std::size_t i = 0; i < histories.size(); ++i)
        for (std::size_t j = i + 1; j < histories.size(); ++j)
            if (comparable(i, j))
                EXPECT_NE(histories[i], histories[j])
                    << kOptimizers[i] << " and " << kOptimizers[j] << " ran identically";
}

TEST(V11291Variational, VqeAnUnknownOptimizerWarnsAndRunsTheDefault) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    VQE reference;
    configure(reference, "COBYLA");
    const auto want = reference.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);

    VQE vqe;

    configure(vqe, "POWELL");
    VQE::Result got;
    const auto msgs = capture_warnings(
        [&] { got = vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz); });

    EXPECT_TRUE(any_contains(msgs, "POWELL"));
    EXPECT_EQ(got.energy_history, want.energy_history);
}

TEST(V11291Variational, VqeTheInitialStepIsHonouredByEveryOptimizer) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        VQE wide, narrow;
        configure(wide, opt);
        configure(narrow, opt);
        narrow.options.initial_step = 0.05;
        const auto rw = wide.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
        const auto rn = narrow.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
        ASSERT_GE(rw.energy_history.size(), 2u);
        ASSERT_GE(rn.energy_history.size(), 2u);
        EXPECT_TRUE(same_bits(rw.energy_history[0], rn.energy_history[0]))
            << "the step moved the start";
        EXPECT_NE(rw.energy_history[1], rn.energy_history[1])
            << "the first trial point did not move with the step";
    }
}

TEST(V11291Variational, VqeTheCapBoundsTheRunAndIsNotConvergence) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        VQE vqe;
        configure(vqe, opt, kSeed, 6);
        vqe.options.convergence_threshold = 1e-14;
        const auto r = vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz);
        EXPECT_LE(r.num_iterations, 6);
        EXPECT_FALSE(r.converged);
    }
}

TEST(V11291Variational, VqeAMismatchedHamiltonianSurfacesTheEstimatorsError) {
    // The sampling estimator throws on the first evaluation when a Pauli term
    // is longer than the circuit. On the NLopt backends that exception is
    // raised inside a C callback, so this is the public face of the seam
    // re-throwing after NLopt returns.
    const auto ansatz = VQE::real_amplitudes(2, 1);
    const SparsePauliOp three({PauliString("ZZZ", Complex128(1.0, 0.0))});
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        VQE vqe;
        configure(vqe, opt);
        vqe.estimator.options.shots = 64;
        try {
            (void)vqe.compute_minimum_eigenvalue(three, ansatz);
            ADD_FAILURE() << "a 3-qubit Hamiltonian on a 2-qubit ansatz ran";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("does not match"), std::string::npos)
                << e.what();
        }
    }
}

// ---- A Pauli term wider than the state -------------------------------------
//
// Both use a Z-only term. Without the width check, a term with X or Y beyond
// the last qubit makes the exact expectation index past the amplitude arrays,
// and undefined behaviour inside the suite could take the whole run down with
// it; a Z beyond the state reaches the same check without reading out of
// bounds.

TEST(V11291Variational, AnExactExpectationRefusesATermWiderThanTheState) {
    Statevector sv(2);
    const SparsePauliOp wide({PauliString("ZZZ", Complex128(1.0, 0.0))});
    EXPECT_THROW((void)wide.expectation_value(sv), std::invalid_argument);
}

TEST(V11291Variational, TheExactEstimatorRefusesATermWiderThanTheCircuit) {
    // shots == 0 is the default and the exact path. The sampled path and the
    // density-matrix path already refuse this; the exact one must agree.
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    const SparsePauliOp wide({PauliString("ZZZ", Complex128(1.0, 0.0))});
    Estimator est;
    ASSERT_EQ(est.options.shots, 0);
    EXPECT_THROW((void)est.run_single(qc, wide), std::invalid_argument);
}

TEST(V11291Variational, VqeMalformedOptionsAreRefused) {
    const auto ansatz = VQE::real_amplitudes(2, 1);
    {
        VQE vqe;
        configure(vqe, "COBYLA");
        vqe.options.max_iterations = 0;
        EXPECT_THROW((void)vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz),
                     std::invalid_argument);
    }
    {
        VQE vqe;
        configure(vqe, "NELDER_MEAD");
        vqe.options.initial_step = 0.0;
        EXPECT_THROW((void)vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz),
                     std::invalid_argument);
    }
    {
        VQE vqe;
        configure(vqe, "BOBYQA");
        vqe.options.convergence_threshold = -1.0;
        EXPECT_THROW((void)vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), ansatz),
                     std::invalid_argument);
    }
}

TEST(V11291Variational, VqeAnAnsatzWithNothingToOptimiseIsRefused) {
    QuantumCircuit fixed(2);
    fixed.h(0);
    fixed.cx(0, 1);
    VQE vqe;
    configure(vqe, "COBYLA");
    EXPECT_THROW((void)vqe.compute_minimum_eigenvalue(vqe_hamiltonian(), fixed),
                 std::invalid_argument);
}

// =============================================================================
// QAOA
// =============================================================================

TEST(V11291Variational, QaoaNumIterationsIsWrittenAndBounded) {
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        QAOA q;
        configure(q, opt);
        const auto r = q.optimize(cut_hamiltonian());
        EXPECT_GE(r.num_iterations, 1);
        EXPECT_LE(r.num_iterations, 50);
    }
}

TEST(V11291Variational, QaoaStartsFromTheDocumentedDraw) {
    for (std::uint64_t seed : {1ULL, 42ULL, kSeed}) {
        for (int p : {1, 3}) {
            SCOPED_TRACE("seed " + std::to_string(seed) + " p " + std::to_string(p));
            QAOA q;
            configure(q, "COBYLA", seed, p, 3);
            const auto r = q.optimize(cut_hamiltonian());
            ASSERT_EQ(r.initial_params.size(), static_cast<std::size_t>(2 * p));
            const auto w = words(std::mt19937_64(seed), 2 * p);
            for (std::size_t i = 0; i < w.size(); ++i) {
                const double want = exact_draw(w[i], -0.05, 0.05);
                EXPECT_TRUE(same_bits(r.initial_params[i], want))
                    << "param " << i << ": " << r.initial_params[i] << " vs " << want;
                EXPECT_GE(r.initial_params[i], -0.05);
                EXPECT_LE(r.initial_params[i], 0.05);
            }
        }
    }
}

TEST(V11291Variational, QaoaTheSameSeedGivesTheSameRunBitForBit) {
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        QAOA a, b;
        configure(a, opt);
        configure(b, opt);
        const auto ra = a.optimize(cut_hamiltonian());
        const auto rb = b.optimize(cut_hamiltonian());
        EXPECT_EQ(ra.optimal_params, rb.optimal_params);
        EXPECT_TRUE(same_bits(ra.optimal_value, rb.optimal_value));
        EXPECT_EQ(ra.num_iterations, rb.num_iterations);
        EXPECT_EQ(ra.counts, rb.counts);
    }
}

TEST(V11291Variational, QaoaEachOptimizerNameRunsADifferentMinimiser) {
    std::vector<std::vector<double>> finals;
    for (const char* opt : kOptimizers) {
        QAOA q;
        configure(q, opt);
        finals.push_back(q.optimize(cut_hamiltonian()).optimal_params);
    }
    for (std::size_t i = 0; i < finals.size(); ++i)
        for (std::size_t j = i + 1; j < finals.size(); ++j)
            if (comparable(i, j))
                EXPECT_NE(finals[i], finals[j])
                    << kOptimizers[i] << " and " << kOptimizers[j] << " ran identically";
}

TEST(V11291Variational, QaoaAnUnknownOptimizerWarnsAndRunsTheDefault) {
    QAOA reference;
    configure(reference, "COBYLA");
    const auto want = reference.optimize(cut_hamiltonian());

    QAOA q;

    configure(q, "POWELL");
    QAOA::Result got;
    const auto msgs = capture_warnings([&] { got = q.optimize(cut_hamiltonian()); });

    EXPECT_TRUE(any_contains(msgs, "POWELL"));
    EXPECT_EQ(got.optimal_params, want.optimal_params);
    EXPECT_EQ(got.num_iterations, want.num_iterations);
}

TEST(V11291Variational, QaoaTheInitialStepIsHonouredByEveryOptimizer) {
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        QAOA wide, narrow;
        configure(wide, opt);
        configure(narrow, opt);
        narrow.options.initial_step = 0.05;
        const auto rw = wide.optimize(cut_hamiltonian());
        const auto rn = narrow.optimize(cut_hamiltonian());
        EXPECT_EQ(rw.initial_params, rn.initial_params) << "the step moved the start";
        EXPECT_NE(rw.optimal_params, rn.optimal_params) << "the step changed nothing";
    }
}

// =============================================================================
// MA-QAOA, on both paths
// =============================================================================

TEST(V11291Variational, MaqaoaEachOptimizerNameRunsADifferentMinimiser) {
    for (bool layerwise : {false, true}) {
        SCOPED_TRACE(layerwise ? "layerwise" : "joint");
        std::vector<MAQAOA::Result> runs;
        for (const char* opt : kOptimizers) {
            MAQAOA m;
            configure(m, opt, layerwise);
            runs.push_back(m.optimize(cut_hamiltonian()));
        }
        for (std::size_t i = 0; i < runs.size(); ++i)
            for (std::size_t j = i + 1; j < runs.size(); ++j)
                if (comparable(i, j))
                    EXPECT_FALSE(same_run(runs[i], runs[j]))
                        << kOptimizers[i] << " and " << kOptimizers[j] << " ran identically";
    }
}

TEST(V11291Variational, MaqaoaAnUnknownOptimizerWarnsAndRunsTheDefault) {
    for (bool layerwise : {false, true}) {
        SCOPED_TRACE(layerwise ? "layerwise" : "joint");
        MAQAOA reference;
        configure(reference, "COBYLA", layerwise);
        const auto want = reference.optimize(cut_hamiltonian());

        MAQAOA m;

        configure(m, "POWELL", layerwise);
        MAQAOA::Result got;
        const auto msgs = capture_warnings([&] { got = m.optimize(cut_hamiltonian()); });

        EXPECT_TRUE(any_contains(msgs, "POWELL"));
        EXPECT_TRUE(same_run(got, want));
    }
}

TEST(V11291Variational, MaqaoaTheSameSeedGivesTheSameRunBitForBit) {
    for (bool layerwise : {false, true}) {
        for (const char* opt : kOptimizers) {
            SCOPED_TRACE(std::string(layerwise ? "layerwise " : "joint ") + opt);
            MAQAOA a, b;
            configure(a, opt, layerwise);
            configure(b, opt, layerwise);
            EXPECT_TRUE(same_run(a.optimize(cut_hamiltonian()), b.optimize(cut_hamiltonian())));
        }
    }
}

TEST(V11291Variational, MaqaoaTheInitialStepIsHonouredOnBothPaths) {
    for (bool layerwise : {false, true}) {
        for (const char* opt : kOptimizers) {
            SCOPED_TRACE(std::string(layerwise ? "layerwise " : "joint ") + opt);
            MAQAOA wide, narrow;
            configure(wide, opt, layerwise);
            configure(narrow, opt, layerwise);
            narrow.options.initial_step = 0.05;
            const auto rw = wide.optimize(cut_hamiltonian());
            const auto rn = narrow.optimize(cut_hamiltonian());
            EXPECT_EQ(rw.initial_params, rn.initial_params) << "the step moved the start";
            EXPECT_NE(rw.optimal_params, rn.optimal_params) << "the step changed nothing";
        }
    }
}

TEST(V11291Variational, MaqaoaStartsFromTheDocumentedDrawOnBothPaths) {
    // No mixer weights, so every parameter is one draw in [-0.05, 0.05], taken
    // in layer order: each layer's gammas, then its betas. Non-progressive
    // layerwise records each layer's own start, which is the same sequence.
    for (bool layerwise : {false, true}) {
        SCOPED_TRACE(layerwise ? "layerwise" : "joint");
        MAQAOA m;
        configure(m, "COBYLA", layerwise, kSeed, 3);
        const auto r = m.optimize(cut_hamiltonian());
        const int n = m.num_parameters(cut_hamiltonian());
        ASSERT_EQ(r.initial_params.size(), static_cast<std::size_t>(n));
        const auto w = words(std::mt19937_64(kSeed), n);
        for (std::size_t i = 0; i < w.size(); ++i)
            EXPECT_TRUE(same_bits(r.initial_params[i], exact_draw(w[i], -0.05, 0.05)))
                << "param " << i;
    }
}

TEST(V11291Variational, MaqaoaLayerwiseCountsAddUp) {
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        MAQAOA m;
        configure(m, opt, true);
        const auto r = m.optimize(cut_hamiltonian());
        ASSERT_EQ(r.layer_nfev.size(), 2u);
        ASSERT_EQ(r.per_layer_costs.size(), 2u);
        int total = 0;
        for (int nfev : r.layer_nfev) {
            EXPECT_GE(nfev, 1);
            EXPECT_LE(nfev, 40) << "a layer went past the per-layer cap";
            total += nfev;
        }
        EXPECT_EQ(total, r.num_iterations);
    }
}

TEST(V11291Variational, MaqaoaJointRunStaysWithinItsCap) {
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        MAQAOA m;
        configure(m, opt, false, kSeed, 9);
        m.options.convergence_threshold = 1e-14;
        const auto r = m.optimize(cut_hamiltonian());
        EXPECT_GE(r.num_iterations, 1);
        EXPECT_LE(r.num_iterations, 9);
        EXPECT_FALSE(r.converged);
    }
}

// =============================================================================
// The documented box, [-2pi, 2pi] on every parameter
// =============================================================================

namespace {
::testing::AssertionResult inside_box(const std::vector<double>& params) {
    for (std::size_t i = 0; i < params.size(); ++i)
        if (!(params[i] >= -TWO_PI && params[i] <= TWO_PI))
            return ::testing::AssertionFailure()
                   << "parameter " << i << " = " << params[i] << " is outside [-2pi, 2pi]";
    return ::testing::AssertionSuccess();
}
}  // namespace

TEST(V11291Variational, QaoaParametersStayInsideTheDocumentedBox) {
    // A wide first step (3 rad, under half the 4pi box so BOBYQA accepts it)
    // sends every backend's opening design towards the walls.
    for (const char* opt : kOptimizers) {
        SCOPED_TRACE(opt);
        QAOA q;
        configure(q, opt);
        q.options.initial_step = 3.0;
        const auto r = q.optimize(cut_hamiltonian());
        EXPECT_TRUE(inside_box(r.optimal_params));
    }
}

TEST(V11291Variational, MaqaoaParametersStayInsideTheDocumentedBoxOnBothPaths) {
    for (bool layerwise : {false, true}) {
        for (const char* opt : kOptimizers) {
            SCOPED_TRACE(std::string(layerwise ? "layerwise " : "joint ") + opt);
            MAQAOA m;
            configure(m, opt, layerwise);
            m.options.initial_step = 3.0;
            const auto r = m.optimize(cut_hamiltonian());
            EXPECT_TRUE(inside_box(r.optimal_params));
        }
    }
}

// ---- A BOBYQA step too wide for the box ------------------------------------

TEST(V11291Variational, QaoaBobyqaWithAStepTooWideForTheBoxIsAnInvalidArgument) {
    // BOBYQA opens with points one step either side of the start, so it needs
    // each interval at least twice the step wide; the QAOA box is 4pi wide and
    // a 7 rad step does not fit. The caller set that step, so the refusal is an
    // argument error naming the entry point and the method, not a report that
    // the objective produced nothing.
    QAOA q;
    configure(q, "BOBYQA");
    q.options.initial_step = 7.0;
    try {
        (void)q.optimize(cut_hamiltonian());
        FAIL() << "BOBYQA ran with a step wider than half its box";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("QAOA::optimize"), std::string::npos) << msg;
        EXPECT_NE(msg.find("BOBYQA"), std::string::npos) << msg;
    }
}

// =============================================================================
// The draw itself
// =============================================================================

TEST(V11291UniformDraw, MatchesTheExactReferenceOverManyWords) {
    struct Range { double lo, hi; };
    const Range ranges[] = {{-PI, PI}, {-0.05, 0.05}, {0.0, 1.0}, {-3.0, 7.5},
                            {1e-3, 2e-3}, {-1e-9, 1e-9}};
    for (const Range& rg : ranges) {
        SCOPED_TRACE(std::to_string(rg.lo) + " .. " + std::to_string(rg.hi));
        std::mt19937_64 rng(kSeed), shadow(kSeed);
        for (int i = 0; i < 20000; ++i) {
            const double got = detail::uniform_in(rng, rg.lo, rg.hi);
            const double want = exact_draw(shadow(), rg.lo, rg.hi);
            ASSERT_TRUE(same_bits(got, want)) << "draw " << i << ": " << got << " vs " << want;
        }
    }
}

TEST(V11291UniformDraw, TheCraftedEngineEmitsTheChosenWord) {
    // Guards the two extreme-draw tests below: if the state did not load as
    // intended, they would be testing some other word.
    for (std::uint64_t target : {0ULL, ~0ULL, 0x0123456789ABCDEFULL}) {
        std::mt19937_64 rng = engine_emitting(target);
        EXPECT_EQ(rng(), target);
    }
}

TEST(V11291UniformDraw, TheLowestWordGivesExactlyLo) {
    for (double lo : {-PI, -0.05}) {
        std::mt19937_64 rng = engine_emitting(0);
        EXPECT_TRUE(same_bits(detail::uniform_in(rng, lo, -lo), lo));
    }
}

TEST(V11291UniformDraw, TheHighestWordStaysBelowHiOnTheRangesTheAlgorithmsUse) {
    // [-pi, pi) is VQE's range and [-0.05, 0.05] the QAOA and MA-QAOA
    // perturbation. The top word, whose 53 kept bits are all ones, is the one
    // draw that can round up to hi.
    for (double hi : {PI, 0.05}) {
        SCOPED_TRACE(hi);
        std::mt19937_64 rng = engine_emitting(~0ULL);
        const double got = detail::uniform_in(rng, -hi, hi);
        EXPECT_LT(got, hi);
        EXPECT_TRUE(same_bits(got, exact_draw(~0ULL, -hi, hi)));
    }
}

TEST(V11291UniformDraw, TheHighestWordStaysBelowHiOnEveryOrderedRange) {
    // On these ranges hi is large against the width, so the exact value of
    // the top draw lies within half a spacing of hi and the
    // single correct rounding lands ON hi: the premise below shows it with the
    // integer reference. The draw is documented as [lo, hi), so the top word
    // must give a value below hi, and never one below the next word down, so
    // the draw stays monotone in the word.
    struct Range { double lo, hi; };
    const Range ranges[] = {{1.0, 2.0}, {0.5, 1.0}, {PI, TWO_PI}};
    const std::uint64_t top = ~0ULL;
    const std::uint64_t next_down = top - (1ULL << 11);  // one step in the kept 53 bits
    for (const Range& rg : ranges) {
        SCOPED_TRACE(std::to_string(rg.lo) + " .. " + std::to_string(rg.hi));
        ASSERT_TRUE(same_bits(exact_draw(top, rg.lo, rg.hi), rg.hi))
            << "premise: the correctly rounded top draw is hi on this range";

        std::mt19937_64 a = engine_emitting(top), b = engine_emitting(next_down);
        const double got = detail::uniform_in(a, rg.lo, rg.hi);
        const double below = detail::uniform_in(b, rg.lo, rg.hi);
        EXPECT_LT(got, rg.hi);
        EXPECT_GE(got, below) << "the top word drew less than the word below it";
        EXPECT_GE(got, rg.lo);
    }
}

TEST(V11291UniformDraw, SeededRngIsTheStandardEngineForANonZeroSeed) {
    for (std::uint64_t seed : {1ULL, kSeed, ~0ULL}) {
        std::mt19937_64 a = detail::seeded_rng(seed), b(seed);
        for (int i = 0; i < 16; ++i) EXPECT_EQ(a(), b());
    }
}
