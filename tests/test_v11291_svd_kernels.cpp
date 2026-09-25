// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.1 test wave - the four SVD kernels, the rescue ladder's reporting,
// and the counters both MPS layers expose.
//
// 1.1.29.0 made autonne's divide and conquer the default kernel, kept Eigen's
// two as selections, turned the ladder into kernel -> Jacobi -> Gram -> throw
// with a warning per rung, and added svd_rescue = false, jacobi_rescue_count,
// floor_rejected_weight and the qudit svd_time_ns. Every kernel is held to the
// same claim here: in the exact regime, the chain it factorises reproduces the
// dense state it stands for, on qubit chains and on d = 3 qudit chains.
//
// Two rungs are deliberately not driven. A Jacobi or Gram rescue SUCCEEDING
// needs a block the selected kernel rejects and a later rung accepts, and no
// input produces that on demand. The qudit chain below is the one the default
// kernel declines blocks of on current builds, so the rescue tests assert what
// must hold whether or not a decline happens on a given compiler: every
// rescue the counter records is one warning naming its block, and the state
// is still exact. The THROW rung is reached with a non-finite block, which
// every rung rejects.
//
// The Jacobi selection note is latched once per layer for the whole process,
// so an earlier suite selecting Jacobi would have consumed it. The tests that
// count it run in a re-executed child process, which starts with the latch
// clear.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/detail/svd_truncate.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

constexpr SVDMethod kKernels[] = {SVDMethod::BDC, SVDMethod::Jacobi, SVDMethod::EigenBDC,
                                  SVDMethod::EigenJacobi};

std::vector<std::string> capture_warnings(const std::function<void()>& fn) {
    flush_warnings();
    std::vector<std::string> captured;
    set_warning_handler([&captured](const std::string& m) { captured.push_back(m); });
    fn();
    flush_warnings();
    set_warning_handler(nullptr);
    return captured;
}

// How many times a message containing `needle` was emitted. The warning channel
// delivers the first occurrence of a message and collapses later identical ones
// into one "[repeated N more times]" line at the flush, so a first delivery
// counts once and a tally line counts N.
std::size_t count_containing(const std::vector<std::string>& msgs, const std::string& needle) {
    static const std::string kTally = "[repeated ";
    std::size_t n = 0;
    for (const auto& m : msgs) {
        if (m.find(needle) == std::string::npos) continue;
        const std::size_t at = m.find(kTally);
        n += (at == std::string::npos) ? 1 : std::stoul(m.substr(at + kTally.size()));
    }
    return n;
}

double nan_bits() { return quiet_nan_strict(); }

// |<a|b>|^2 over two amplitude lists of equal length.
template <class A, class B>
double overlap_sq(std::size_t dim, A a, B b) {
    Complex128 acc(0.0, 0.0);
    for (std::size_t i = 0; i < dim; ++i) acc += a(i).conj() * b(i);
    return acc.real * acc.real + acc.imag * acc.imag;
}

// Seven qubits, three brickwork rounds of arbitrary rotations and CX on
// neighbours and on qubits two apart, so the chain routes through swaps. The
// largest bond is 2^3 = 8, far under the cap, so the MPS is exact.
QuantumCircuit qubit_chain(std::uint64_t seed) {
    constexpr int n = 7;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(-PI, PI);
    QuantumCircuit qc(n);
    for (int round = 0; round < 3; ++round) {
        for (int q = 0; q < n; ++q) {
            qc.ry(angle(rng), q);
            qc.rz(angle(rng), q);
        }
        for (int q = round % 2; q + 1 < n; q += 2) qc.cx(q, q + 1);
        for (int q = 0; q + 2 < n; q += 3) qc.cx(q + 2, q);
    }
    return qc;
}

double mps_fidelity(const QuantumCircuit& qc, SVDMethod method, bool rescue = true) {
    MPSSimulator mps;
    mps.svd_method = method;
    mps.svd_rescue = rescue;
    auto mr = mps.run(qc, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/1);
    StatevectorSimulator sv;
    auto sr = sv.run(qc, /*shots=*/0, /*seed=*/1);
    const Statevector got = mr.final_state.to_statevector();
    return overlap_sq(got.dim, [&](std::size_t i) { return got.amplitude(i); },
                      [&](std::size_t i) { return sr.final_state.amplitude(i); });
}

// ---- d = 3 qudit chain ----------------------------------------------------

// exp(2 pi i a b / d) on |a, b>: diagonal, and symmetric in the two digits, so
// its layout does not depend on which operand is the low digit.
std::vector<Complex128> cphase(int d) {
    const std::size_t d2 = static_cast<std::size_t>(d * d);
    std::vector<Complex128> u(d2 * d2, Complex128(0.0, 0.0));
    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
            const std::size_t k = static_cast<std::size_t>(a + d * b);
            const double phi = 2.0 * PI * a * b / d;
            u[k * d2 + k] = Complex128(std::cos(phi), std::sin(phi));
        }
    return u;
}

struct QuditOp {
    bool two;
    int a, b;
    const std::vector<Complex128>* u;
};

// The gate chain of R1161QuditStress.LongSeededChainExact: seven qutrits, 168
// gates, every pair distance. The default kernel declines several of its blocks
// on current builds, which is what makes it the ladder's live test.
std::vector<QuditOp> qutrit_chain(const std::vector<Complex128>& F,
                                  const std::vector<Complex128>& SUM,
                                  const std::vector<Complex128>& CP) {
    constexpr int n = 7;
    std::vector<QuditOp> ops;
    for (int step = 0; step < 56; ++step) {
        ops.push_back({false, (3 * step + 1) % n, 0, &F});
        const int a = (5 * step) % n;
        int b = (2 * step + 3) % n;
        if (b == a) b = (b + 1) % n;
        ops.push_back({true, a, b, (step % 2 == 0) ? &SUM : &CP});
        const int c0 = (step * 4 + 2) % n;
        int c1 = (n - 1) - (step % n);
        if (c1 == c0) c1 = (c1 + 1) % n;
        ops.push_back({true, c0, c1, (step % 3 == 0) ? &CP : &SUM});
    }
    return ops;
}

struct QuditRun {
    double fidelity = 0.0;
    std::size_t svd_calls = 0, jacobi = 0, gram = 0;
    double floor_rejected = 0.0;
    std::uint64_t svd_ns = 0;
    std::vector<std::string> warnings;
};

QuditRun run_qutrit_chain(SVDMethod method) {
    constexpr int d = 3, n = 7;
    const auto F = qudit_gates::qft_matrix(d);
    const auto SUM = qudit_gates::cadd_matrix(d, 1);
    const auto CP = cphase(d);
    const auto ops = qutrit_chain(F, SUM, CP);

    QuditRun out;
    QuditMPS mps(n, d, /*max_bond_dim=*/64);
    mps.svd_method = method;
    QuditStatevector ref(n, d);
    out.warnings = capture_warnings([&] {
        for (const QuditOp& op : ops) {
            if (op.two) {
                mps.apply_2qudit(op.a, op.b, *op.u);
                ref.apply_2qudit(op.a, op.b, *op.u);
            } else {
                mps.apply_1qudit(op.a, *op.u);
                ref.apply_1qudit(op.a, *op.u);
            }
        }
    });
    const QuditStatevector got = mps.to_statevector();
    out.fidelity = overlap_sq(got.dim, [&](std::size_t i) { return got.amplitudes[i]; },
                              [&](std::size_t i) { return ref.amplitudes[i]; });
    out.svd_calls = mps.svd_call_count();
    out.jacobi = mps.jacobi_rescue_count();
    out.gram = mps.gram_fallback_count();
    out.floor_rejected = mps.floor_rejected_weight();
    out.svd_ns = mps.svd_time_ns();
    return out;
}

// A two-qubit "gate" whose every entry is NaN. Validation is told to ignore
// it, so the only thing left to object is the factorisation it poisons.
std::array<Complex128, 16> nan_gate() {
    std::array<Complex128, 16> u;
    for (auto& z : u) z = Complex128(nan_bits(), 0.0);
    return u;
}

}  // namespace

// =============================================================================
// Selection surface
// =============================================================================

TEST(V11291SvdKernels, BdcAndRescueAreTheDefaultOnEverySurface) {
    const MPSState state(4);
    EXPECT_EQ(state.svd_method, SVDMethod::BDC);
    EXPECT_TRUE(state.svd_rescue);

    const MPSSimulator sim{};
    EXPECT_EQ(sim.svd_method, SVDMethod::BDC);
    EXPECT_TRUE(sim.svd_rescue);

    const QuditMPS qudit(3, 3);
    EXPECT_EQ(qudit.svd_method, SVDMethod::BDC);
    EXPECT_TRUE(qudit.svd_rescue);
}

TEST(V11291SvdKernels, EachKernelHasTheNameTheDocumentationUses) {
    EXPECT_STREQ(to_string(SVDMethod::BDC), "BDC");
    EXPECT_STREQ(to_string(SVDMethod::Jacobi), "Jacobi");
    EXPECT_STREQ(to_string(SVDMethod::EigenBDC), "EigenBDC");
    EXPECT_STREQ(to_string(SVDMethod::EigenJacobi), "EigenJacobi");
}

TEST(V11291SvdKernels, TheSimulatorHandsItsSelectionToTheChainItBuilds) {
    const QuantumCircuit qc = qubit_chain(3);
    for (SVDMethod m : kKernels) {
        for (bool rescue : {true, false}) {
            SCOPED_TRACE(std::string(to_string(m)) + (rescue ? " rescue" : " no rescue"));
            MPSSimulator sim;
            sim.svd_method = m;
            sim.svd_rescue = rescue;
            const auto r = sim.run(qc, 64, 0, 1);
            EXPECT_EQ(r.final_state.svd_method, m);
            EXPECT_EQ(r.final_state.svd_rescue, rescue);
        }
    }
}

// =============================================================================
// Every kernel is exact in the exact regime
// =============================================================================

TEST(V11291SvdKernels, EveryKernelReproducesTheStatevectorOnAQubitChain) {
    for (SVDMethod m : kKernels) {
        for (std::uint64_t seed : {1ULL, 2ULL, 3ULL}) {
            SCOPED_TRACE(std::string(to_string(m)) + " seed " + std::to_string(seed));
            EXPECT_NEAR(mps_fidelity(qubit_chain(seed), m), 1.0, 1e-10);
        }
    }
}

TEST(V11291SvdKernels, EveryKernelReproducesTheStatevectorWithRescueOff) {
    // A kernel that is accepted on every split has nothing to rescue, so
    // forbidding the rescue changes nothing about a clean run.
    for (SVDMethod m : kKernels) {
        SCOPED_TRACE(to_string(m));
        EXPECT_NEAR(mps_fidelity(qubit_chain(5), m, /*rescue=*/false), 1.0, 1e-10);
    }
}

TEST(V11291SvdKernels, EveryKernelReproducesTheStatevectorOnAQutritChain) {
    for (SVDMethod m : kKernels) {
        SCOPED_TRACE(to_string(m));
        const QuditRun r = run_qutrit_chain(m);
        EXPECT_NEAR(r.fidelity, 1.0, 1e-9);
        EXPECT_GT(r.svd_calls, 0u);
    }
}

// =============================================================================
// The ladder's reporting
// =============================================================================

TEST(V11291SvdKernels, EveryRescueTheCounterRecordsIsOneWarningNamingItsBlock) {
    // Holds with any number of declines, including none: the counter and the
    // warning channel must describe the same events.
    for (SVDMethod m : kKernels) {
        SCOPED_TRACE(to_string(m));
        const QuditRun r = run_qutrit_chain(m);

        // No split threw, so every descent ended in a rescue. A split rescued by
        // Gram was first announced as a retry with Jacobi, unless Jacobi was
        // the kernel that failed; every descent to Gram is announced once.
        const std::size_t retries = count_containing(r.warnings, "retrying with SVDMethod::Jacobi");
        const std::size_t to_gram = count_containing(r.warnings, "recomputing through the Gram route");
        const std::size_t via_jacobi = (m == SVDMethod::Jacobi) ? 0 : r.gram;
        EXPECT_EQ(retries, r.jacobi + via_jacobi) << "Jacobi retries vs rescues";
        EXPECT_EQ(to_gram, r.gram) << "Gram descents vs rescues";

        const std::regex shape(R"(\d+x\d+ block)");
        for (const std::string& w : r.warnings) {
            if (w.find("retrying with SVDMethod::Jacobi") == std::string::npos) continue;
            // Layer, block shape, the kernel that failed, and why.
            EXPECT_NE(w.find("QuditMPS"), std::string::npos) << w;
            EXPECT_NE(w.find(std::string("the ") + to_string(m) + " factorisation"),
                      std::string::npos) << w;
            EXPECT_TRUE(std::regex_search(w, shape)) << w;
            EXPECT_NE(w.find("failed verification ("), std::string::npos) << w;
        }
        // A rescued split still yields the exact state.
        EXPECT_NEAR(r.fidelity, 1.0, 1e-9);
    }
}

TEST(V11291SvdKernels, JacobiSelectedIsNeverRescuedByJacobi) {
    // The Jacobi rung is skipped when Jacobi was the kernel that failed.
    const QuditRun r = run_qutrit_chain(SVDMethod::Jacobi);
    EXPECT_EQ(r.jacobi, 0u);
    EXPECT_EQ(count_containing(r.warnings, "retrying with SVDMethod::Jacobi"), 0u);
}

TEST(V11291SvdKernels, FloorRejectedWeightIsZeroUnlessTheGramRungRan) {
    for (SVDMethod m : kKernels) {
        SCOPED_TRACE(to_string(m));
        const QuditRun r = run_qutrit_chain(m);
        if (r.gram == 0) EXPECT_EQ(r.floor_rejected, 0.0);
        EXPECT_GE(r.floor_rejected, 0.0);
    }
}

TEST(V11291SvdKernels, TruncationIsNotBookedAsFloorRejectedWeight) {
    // A bond cap of 2 truncates heavily. All of that weight belongs to
    // truncation_error; floor_rejected_weight is the Gram route's own cost and
    // stays zero on every run the Gram route did not serve.
    for (SVDMethod m : kKernels) {
        SCOPED_TRACE(to_string(m));
        MPSSimulator sim;
        sim.svd_method = m;
        const auto r = sim.run(qubit_chain(9), /*max_bond_dim=*/2, 0, 1);
        EXPECT_GT(r.final_state.truncation_error(), 0.0);
        if (r.final_state.gram_fallback_count() == 0)
            EXPECT_EQ(r.final_state.floor_rejected_weight(), 0.0);
    }
}

TEST(V11291SvdKernels, TheQuditLayerTimesItsSplits) {
    QuditMPS mps(4, 3);
    const auto F = qudit_gates::qft_matrix(3);
    mps.apply_1qudit(0, F);
    mps.apply_1qudit(1, F);
    EXPECT_EQ(mps.svd_call_count(), 0u);
    EXPECT_EQ(mps.svd_time_ns(), 0u) << "time was booked with no split";

    mps.apply_2qudit_adjacent(0, qudit_gates::cadd_matrix(3, 1));
    const std::uint64_t after_one = mps.svd_time_ns();
    EXPECT_GT(mps.svd_call_count(), 0u);
    EXPECT_GT(after_one, 0u);

    mps.apply_1qudit(2, F);
    EXPECT_EQ(mps.svd_time_ns(), after_one) << "a one-qudit gate booked split time";

    mps.apply_2qudit_adjacent(1, qudit_gates::cadd_matrix(3, 1));
    EXPECT_GT(mps.svd_time_ns(), after_one);
}

// =============================================================================
// A block every rung rejects
// =============================================================================

TEST(V11291SvdKernels, WithRescueOffTheFirstRejectionThrowsWithoutDescending) {
    std::vector<Complex128> block(16, Complex128(0.5, 0.0));
    block[5] = Complex128(nan_bits(), 0.0);
    for (SVDMethod m : kKernels) {
        for (auto order : {detail::MatrixOrder::RowMajor, detail::MatrixOrder::ColMajor}) {
            SCOPED_TRACE(to_string(m));
            bool threw = false;
            const auto msgs = capture_warnings([&] {
                try {
                    (void)detail::svd_truncate_verified(block.data(), 4, 4, order, 4, 1e-16, m,
                                                        /*rescue=*/false, "V11291Ladder");
                } catch (const std::runtime_error&) {
                    threw = true;
                }
            });
            EXPECT_TRUE(threw);
            EXPECT_TRUE(msgs.empty()) << "the ladder descended with rescue off: " << msgs.front();
        }
    }
}

TEST(V11291SvdKernels, WithRescueOnEveryRungIsReportedBeforeTheThrow) {
    std::vector<Complex128> block(24, Complex128(0.25, -0.25));
    block[7] = Complex128(0.0, nan_bits());
    for (SVDMethod m : kKernels) {
        SCOPED_TRACE(to_string(m));
        bool threw = false;
        const auto msgs = capture_warnings([&] {
            try {
                (void)detail::svd_truncate_verified(block.data(), 4, 6,
                                                    detail::MatrixOrder::ColMajor, 4, 1e-16, m,
                                                    /*rescue=*/true, "V11291Ladder");
            } catch (const std::runtime_error&) {
                threw = true;
            }
        });
        EXPECT_TRUE(threw) << "a non-finite block produced a tensor";
        // One warning per rung descended: to Jacobi and on to Gram, or
        // straight to Gram when Jacobi was the kernel that failed.
        const std::size_t expected = (m == SVDMethod::Jacobi) ? 1u : 2u;
        EXPECT_EQ(msgs.size(), expected);
        for (const std::string& w : msgs) {
            EXPECT_NE(w.find("V11291Ladder"), std::string::npos) << w;
            EXPECT_NE(w.find("4x6"), std::string::npos) << w;
            EXPECT_NE(w.find(to_string(m)), std::string::npos) << w;
        }
    }
}

TEST(V11291SvdKernels, RescueOffOnTheQubitStateThrowsWithoutDescending) {
    for (bool rescue : {false, true}) {
        SCOPED_TRACE(rescue ? "rescue on" : "rescue off");
        MPSState state(3);
        state.svd_rescue = rescue;
        bool threw = false;
        const auto msgs = capture_warnings([&] {
            try {
                state.apply_two_qubit_gate(nan_gate(), 0, 1, {Validation::Ignore});
            } catch (const std::runtime_error&) {
                threw = true;
            }
        });
        EXPECT_TRUE(threw);
        EXPECT_EQ(msgs.empty(), !rescue);
        EXPECT_EQ(state.jacobi_rescue_count(), 0u);
        EXPECT_EQ(state.gram_fallback_count(), 0u);
    }
}

TEST(V11291SvdKernels, RescueOffOnTheQuditStateThrowsWithoutDescending) {
    std::vector<Complex128> bad(81, Complex128(nan_bits(), 0.0));
    for (bool rescue : {false, true}) {
        SCOPED_TRACE(rescue ? "rescue on" : "rescue off");
        QuditMPS mps(3, 3);
        mps.svd_rescue = rescue;
        bool threw = false;
        const auto msgs = capture_warnings([&] {
            try {
                mps.apply_2qudit_adjacent(0, bad, {Validation::Ignore});
            } catch (const std::runtime_error&) {
                threw = true;
            }
        });
        EXPECT_TRUE(threw);
        EXPECT_EQ(msgs.empty(), !rescue);
    }
}

// =============================================================================
// The Jacobi selection note, in a fresh process
// =============================================================================

namespace {

// Runs splits under the given kernels on both layers and exits with
// 10 * (qubit notes) + (qudit notes).
[[noreturn]] void exit_with_note_counts(std::vector<SVDMethod> kernels) {
    std::size_t qubit = 0, qudit = 0;
    set_warning_handler([&](const std::string& m) {
        if (m.find("[repeated ") != std::string::npos) return;
        if (m.find("selected for the qubit MPS") != std::string::npos) ++qubit;
        if (m.find("selected for the qudit MPS") != std::string::npos) ++qudit;
    });
    for (SVDMethod m : kernels) {
        for (int rep = 0; rep < 2; ++rep) {
            MPSSimulator sim;
            sim.svd_method = m;
            (void)sim.run(qubit_chain(4), 64, 0, 1);

            QuditMPS mps(4, 3);
            mps.svd_method = m;
            mps.apply_1qudit(0, qudit_gates::qft_matrix(3));
            mps.apply_2qudit_adjacent(0, qudit_gates::cadd_matrix(3, 1));
            mps.apply_2qudit_adjacent(1, qudit_gates::cadd_matrix(3, 1));
        }
    }
    flush_warnings();
    set_warning_handler(nullptr);
    std::exit(static_cast<int>(10 * qubit + qudit));
}

}  // namespace

TEST(V11291SvdKernels, SelectingJacobiNotesItOncePerLayerPerProcess) {
    // Both Jacobi kernels, each selected twice on each layer: one note per
    // layer for the whole process, whichever Jacobi came first.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(exit_with_note_counts({SVDMethod::Jacobi, SVDMethod::EigenJacobi}),
                ::testing::ExitedWithCode(11), "");
}

TEST(V11291SvdKernels, SelectingEigenJacobiAloneAlsoNotesIt) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(exit_with_note_counts({SVDMethod::EigenJacobi}),
                ::testing::ExitedWithCode(11), "");
}

TEST(V11291SvdKernels, TheBdcKernelsAreSilent) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(exit_with_note_counts({SVDMethod::BDC, SVDMethod::EigenBDC}),
                ::testing::ExitedWithCode(0), "");
}
