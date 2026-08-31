// test_v11241_truncation.cpp - the truncation ladder's accounting, and the
// statevector reconstruction that now goes through it (#93).
//
// ACCOUNTING. SvdTruncation grew a second weight field, and the split between
// the two is the whole of a fix that cleared three regressions. Forming the
// Gram matrix squares the condition number, so a singular value that is exactly
// zero in the input returns at the scale of sqrt(eps); the validity floor
// refuses to keep those, and folding their weight into the truncation error
// made a bond that discarded nothing report a loss. Reported loss now covers
// only what the budget or the bond cap dropped. Nothing in the tree referenced
// either field.
//
// The fixtures are diagonal with singular values at exact powers of two, so
// every expected weight is a dyadic rational, the totals are exact in binary
// floating point, and each expectation is computed from the chosen spectrum
// rather than read off a run.
//
// One rung stays out of reach and is deliberately not faked here. The Gram
// rescue runs only when the primary factorisation fails verification, and a
// backend that fails on demand is precisely what this project cannot produce;
// reaching it needs the ladder to accept a supplied factorisation instead of
// computing one. What can be asserted from outside is the other half of the
// same contract: on the primary route the floor is never applied, so its weight
// field is exactly zero, and a nonzero value there is therefore unambiguous
// evidence of a rescue.
//
// #93. mps_from_sv selected a rank from singular values it had not verified. A
// non-finite value made the running comparison false at every step, so the loop
// ran to rank one and returned finite tensors carrying no marker of what had
// happened. The existing coverage cannot see this: its only test routes a
// 3-qubit CCX whose bond dimension never approaches the cap, so the truncating
// path it now shares with every other split is never entered. These cases enter
// it, in both regimes: a cap generous enough that the reconstruction must be
// exact, and a cap tight enough that it must truncate and must still not
// collapse.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/svd_truncate.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using lindblad::Complex128;
using lindblad::MPSSimulator;
using lindblad::QuantumCircuit;
using lindblad::SVDMethod;
using lindblad::Statevector;
using lindblad::StatevectorSimulator;
using lindblad::detail::MatrixOrder;
using lindblad::detail::SvdTruncation;
using lindblad::detail::svd_truncate_verified;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// Singular values at exact powers of two, descending. Every weight and every
// partial sum is then a dyadic rational and exact in double, so the expected
// values below carry no rounding of their own.
const std::vector<double>& dyadic_sigmas() {
    static const std::vector<double> s{1.0, 0.5, 0.25, 0.125};
    return s;
}

// A square diagonal block with the given spectrum. Diagonal because the
// singular values are then the entries themselves, so the expectation is a
// property of the fixture rather than of whatever the backend returns.
std::vector<Complex128> diagonal_block(const std::vector<double>& sigmas) {
    const int n = static_cast<int>(sigmas.size());
    std::vector<Complex128> m(static_cast<size_t>(n) * n, Complex128(0.0, 0.0));
    for (int i = 0; i < n; ++i)
        m[static_cast<size_t>(i) * n + i] = Complex128(sigmas[static_cast<size_t>(i)], 0.0);
    return m;
}

double total_weight(const std::vector<double>& sigmas) {
    double t = 0.0;
    for (double s : sigmas) t += s * s;
    return t;
}

// Weight carried by everything below rank k.
double tail_weight(const std::vector<double>& sigmas, int k) {
    double t = 0.0;
    for (size_t i = static_cast<size_t>(k); i < sigmas.size(); ++i)
        t += sigmas[i] * sigmas[i];
    return t;
}

SvdTruncation truncate(const std::vector<Complex128>& block, int n,
                       int max_bond_dim, double cutoff,
                       SVDMethod method = SVDMethod::BDC) {
    return svd_truncate_verified(block.data(), n, n, MatrixOrder::RowMajor,
                                 max_bond_dim, cutoff, method, "v11241");
}

// A circuit whose state needs more bond dimension than a tight cap allows, ending
// in a multi-control gate so the MPS engine must fall back to a statevector and
// rebuild through mps_from_sv. The angles are irrational multiples so no
// amplitude lands on a symmetric value that could mask a convention error.
QuantumCircuit entangling_then_fallback(int n) {
    QuantumCircuit qc(n);
    for (int q = 0; q < n; ++q) qc.h(q);
    for (int q = 0; q + 1 < n; ++q)
        qc.rzz(lindblad::PI / (3.0 + q), q, q + 1);
    for (int q = 0; q < n; ++q) qc.ry(lindblad::PI / (5.0 + q), q);
    std::vector<int> controls;
    for (int q = 0; q + 1 < n; ++q) controls.push_back(q);
    qc.mcx(controls, n - 1);  // forces to_statevector -> mps_from_sv
    return qc;
}

// Statevector holds real and imaginary parts in separate aligned arrays, so a
// comparison against another engine's output reads them into one interleaved
// vector rather than indexing two.
std::vector<Complex128> to_vec(const Statevector& sv) {
    std::vector<Complex128> a(sv.dim);
    for (size_t i = 0; i < sv.dim; ++i)
        a[i] = Complex128(sv.real_parts[i], sv.imag_parts[i]);
    return a;
}

std::vector<Complex128> sv_amplitudes(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    auto res = sim.run(qc, /*shots=*/0);
    return to_vec(res.final_state);
}

std::vector<Complex128> mps_amplitudes(const QuantumCircuit& qc, int chi) {
    MPSSimulator sim;
    auto res = sim.run(qc, chi, /*shots=*/0, /*seed=*/7);
    const Statevector sv = res.final_state.to_statevector();
    return to_vec(sv);
}

double overlap_magnitude(const std::vector<Complex128>& a,
                         const std::vector<Complex128>& b) {
    double re = 0.0, im = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        re += a[i].real * b[i].real + a[i].imag * b[i].imag;
        im += a[i].real * b[i].imag - a[i].imag * b[i].real;
    }
    return std::sqrt(re * re + im * im);
}

}  // namespace

// =============================================================================
// The accounting split
// =============================================================================

TEST(V11241Truncation, TheFloorFieldIsExactlyZeroOnThePrimaryRoute) {
    // The primary route applies no validity floor, so this field is not merely
    // small there: it is untouched. That is what makes a nonzero value elsewhere
    // mean "this split was rescued" rather than "this split was noisy".
    const auto sigmas = dyadic_sigmas();
    const int n = static_cast<int>(sigmas.size());
    const auto block = diagonal_block(sigmas);

    for (auto method : {SVDMethod::Jacobi, SVDMethod::BDC}) {
        for (double cutoff : {0.0, 0.02, 0.07, 0.5}) {
            const auto r = truncate(block, n, n, cutoff, method);
            EXPECT_EQ(r.floor_rejected_weight, 0.0)
                << "cutoff " << cutoff << " reported floor-rejected weight on "
                << "the primary route";
            EXPECT_FALSE(r.used_gram_fallback)
                << "a healthy block was rescued at cutoff " << cutoff;
        }
    }
}

TEST(V11241Truncation, ABondThatDiscardsNothingReportsNoLoss) {
    // The regression the accounting fix repaired: weight that was never in the
    // matrix was being reported as truncation error, so a split that kept
    // everything still claimed a loss.
    const auto sigmas = dyadic_sigmas();
    const int n = static_cast<int>(sigmas.size());
    const auto r = truncate(diagonal_block(sigmas), n, /*max_bond_dim=*/n,
                            /*cutoff=*/0.0);

    EXPECT_EQ(r.rank, n) << "a zero budget and a generous cap dropped a direction";
    EXPECT_EQ(r.discarded_weight, 0.0)
        << "a split that kept every direction reported having lost weight";
    EXPECT_EQ(r.floor_rejected_weight, 0.0);
}

TEST(V11241Truncation, DiscardedWeightIsExactlyWhatTheBudgetDropped) {
    // The budget is a fraction of total weight. For each rank the fixture
    // admits, the smallest cutoff that reaches it is the tail weight below it
    // over the total, so both the expected rank and the expected loss come from
    // the chosen spectrum.
    const auto sigmas = dyadic_sigmas();
    const int n = static_cast<int>(sigmas.size());
    const double total = total_weight(sigmas);
    const auto block = diagonal_block(sigmas);

    for (int keep = 1; keep <= n; ++keep) {
        const double dropped = tail_weight(sigmas, keep);
        // Sit strictly between the budget that reaches this rank and the one
        // that would reach the next, so the choice is unambiguous.
        const double next = tail_weight(sigmas, keep - 1);
        const double cutoff = (dropped + next) / (2.0 * total);

        const auto r = truncate(block, n, n, cutoff);
        EXPECT_EQ(r.rank, keep) << "cutoff " << cutoff << " chose rank " << r.rank;
        EXPECT_NEAR(r.discarded_weight, dropped, 64.0 * n * kEps * total)
            << "at rank " << keep << " the reported loss and the dropped "
            << "weight parted";
        EXPECT_EQ(r.floor_rejected_weight, 0.0);
    }
}

TEST(V11241Truncation, TheBondCapCountsAsDiscardedWeightToo) {
    // A direction dropped by the cap is as absent from the kept slice as one
    // dropped by the budget, so it belongs in the same field.
    const auto sigmas = dyadic_sigmas();
    const int n = static_cast<int>(sigmas.size());
    const double total = total_weight(sigmas);
    const auto block = diagonal_block(sigmas);

    for (int cap = 1; cap < n; ++cap) {
        const auto r = truncate(block, n, cap, /*cutoff=*/0.0);
        EXPECT_EQ(r.rank, cap) << "the cap was not the binding constraint";
        EXPECT_NEAR(r.discarded_weight, tail_weight(sigmas, cap),
                    64.0 * n * kEps * total)
            << "cap " << cap << " did not report what it dropped";
        EXPECT_EQ(r.floor_rejected_weight, 0.0);
    }
}

TEST(V11241Truncation, ACleanSplitSitsAtMachineEpsilonAboveTheIdeal) {
    // residual_excess reports how far the accepted factorisation sat above a
    // perfect one, as a fraction of the block's weight. A truncated SVD
    // satisfies the Frobenius identity with equality, so a clean run has
    // nothing left over.
    const auto sigmas = dyadic_sigmas();
    const int n = static_cast<int>(sigmas.size());
    const auto block = diagonal_block(sigmas);

    for (int cap = 1; cap <= n; ++cap) {
        const auto r = truncate(block, n, cap, /*cutoff=*/0.0);
        EXPECT_GE(r.residual_excess, 0.0) << "excess is clamped at zero";
        EXPECT_LT(r.residual_excess, 64.0 * n * kEps)
            << "cap " << cap << " accepted a factorisation carrying "
            << r.residual_excess << " of unexplained residual";
    }
}

TEST(V11241Truncation, KeptSingularValuesDescendAndMatchTheFixture) {
    const auto sigmas = dyadic_sigmas();
    const int n = static_cast<int>(sigmas.size());
    const auto r = truncate(diagonal_block(sigmas), n, n, /*cutoff=*/0.0);
    ASSERT_EQ(r.rank, n);
    for (int i = 0; i < r.rank; ++i)
        EXPECT_NEAR(r.S(i), sigmas[static_cast<size_t>(i)], 64.0 * n * kEps)
            << "sigma[" << i << "]";
}

// =============================================================================
// #93 - mps_from_sv through the verified ladder
// =============================================================================

TEST(V11241MpsFromSv, AGenerousCapReconstructsTheStateExactly) {
    // The reconstruction is only a fallback, so when nothing forces truncation
    // it must be lossless. This is the half the existing coverage does test,
    // stated here at a bond dimension the state genuinely needs so the path is
    // exercised rather than skirted.
    const int n = 6;
    const auto qc = entangling_then_fallback(n);
    const auto want = sv_amplitudes(qc);
    const auto got = mps_amplitudes(qc, /*chi=*/64);

    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i].real, want[i].real, 1e-9) << "amplitude " << i;
        EXPECT_NEAR(got[i].imag, want[i].imag, 1e-9) << "amplitude " << i;
    }
}

TEST(V11241MpsFromSv, ATightCapTruncatesWithoutCollapsingToRankOne) {
    // The #93 signature. A rank chosen from unverified singular values ran the
    // selection loop to rank one and returned finite tensors with no marker, so
    // the assertion that matters is not finiteness but that the cap, rather
    // than a failed comparison, is what bounded the bond dimension.
    const int n = 6;
    const int chi = 4;
    const auto qc = entangling_then_fallback(n);

    MPSSimulator sim;
    auto res = sim.run(qc, chi, /*shots=*/0, /*seed=*/7);
    const auto& tensors = res.final_state.tensors;
    ASSERT_EQ(static_cast<int>(tensors.size()), n);

    int widest = 0;
    for (const auto& t : tensors) widest = std::max(widest, t.bond_right);
    EXPECT_GT(widest, 1)
        << "every bond came back at dimension 1, which is what a selection "
           "loop driven by a non-finite comparison produces";
    EXPECT_LE(widest, chi) << "a bond exceeded the cap it was given";

    for (const auto& t : tensors)
        for (const auto& z : t.data) {
            ASSERT_TRUE(std::isfinite(z.real)) << "non-finite amplitude survived";
            ASSERT_TRUE(std::isfinite(z.imag));
        }
}

TEST(V11241MpsFromSv, TheExactCapIsTheOneTheRegisterPredicts) {
    // Where the reconstruction stops truncating is a property of the register,
    // not of a measurement: the widest bond of an n-qubit chain carries Schmidt
    // rank at most 2^(n/2), so a cap of that size is lossless and anything well
    // below it is not.
    //
    // Deliberately NOT an ordering across caps. Fidelity rising with the cap
    // holds for one truncation of a fixed state; a sequential chain at two
    // different caps diverges after the first split and approximates different
    // trajectories thereafter, so deep in the truncated regime the value is
    // numerical noise and its ordering is decided by floating-point
    // reassociation rather than by the cap. The two claims below survive a
    // compiler change because one is exact and the other is a wide margin.
    const int n = 6;
    const int exact_chi = 1 << (n / 2);  // 8
    const auto qc = entangling_then_fallback(n);
    const auto exact = sv_amplitudes(qc);

    const auto at_exact = mps_amplitudes(qc, exact_chi);
    ASSERT_EQ(at_exact.size(), exact.size());
    EXPECT_NEAR(overlap_magnitude(exact, at_exact), 1.0, 1e-9)
        << "a cap of " << exact_chi << " covers every Schmidt direction a "
        << n << "-qubit state can carry, so the reconstruction must be exact";

    // And the case must genuinely truncate below that, or the assertion above
    // is measuring a path that never engages.
    const auto at_one = mps_amplitudes(qc, 1);
    ASSERT_EQ(at_one.size(), exact.size());
    for (const auto& z : at_one) {
        ASSERT_TRUE(std::isfinite(z.real));
        ASSERT_TRUE(std::isfinite(z.imag));
    }
    EXPECT_LT(overlap_magnitude(exact, at_one), 0.9)
        << "a product state reproduced an entangled one, so this circuit is "
           "not entangled enough for the reconstruction path to be under test";
}

TEST(V11241MpsFromSv, TheFallbackAgreesAcrossBothMultiQubitEntryPoints) {
    // Two instruction shapes reach the reconstruction: a multi-control gate and
    // a caller-supplied multi-qubit UNITARY. They must land on the same state,
    // since the second is only the first written out as a matrix.
    const int n = 5;

    QuantumCircuit viamcx(n);
    for (int q = 0; q < n; ++q) viamcx.h(q);
    viamcx.rzz(lindblad::PI / 3.0, 0, 1);
    viamcx.mcx({0, 1}, 2);

    QuantumCircuit viaunitary(n);
    for (int q = 0; q < n; ++q) viaunitary.h(q);
    viaunitary.rzz(lindblad::PI / 3.0, 0, 1);
    {
        // CCX over qubits {0, 1, 2} with qubits[0] the least significant, so
        // the flipped pair is the two indices whose low two bits are both set.
        std::vector<Complex128> ccx(64, Complex128(0.0, 0.0));
        for (int i = 0; i < 8; ++i) {
            const int j = ((i & 0b011) == 0b011) ? (i ^ 0b100) : i;
            ccx[static_cast<size_t>(j) * 8 + i] = Complex128(1.0, 0.0);
        }
        viaunitary.unitary(ccx, {0, 1, 2}, "ccx");
    }

    const auto a = mps_amplitudes(viamcx, 64);
    const auto b = mps_amplitudes(viaunitary, 64);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, 1e-9) << "amplitude " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, 1e-9) << "amplitude " << i;
    }
}
