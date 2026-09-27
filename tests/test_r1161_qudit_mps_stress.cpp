// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// R.1.16.1 qudit MPS stress suite (gap 3 of the #44 follow-up).
//
// Why this exists: the R.1.16.0 investigation showed every existing qudit
// MPS test sits at n <= 6 sites with dense-ctor round-trips or SHORT gate
// sequences, while qudit_mps.cpp carries the same unguarded SVD-truncation
// pattern that broke the qubit MPS. The d=2 canary
// (R1161MpsShor.QuditTwinStaysExact) covers exactly one scenario. This file
// adds the missing coverage class — flat degenerate spectra at scale and
// LONG gate chains through the two-site SVD path — in two tiers:
//
//   R1161QuditStress   — asserted, exact-bond-regime scenarios (chi_max of
//                        the state <= max_bond_dim, so fidelity against the
//                        dense reference MUST be 1 and any deviation is a
//                        defect of the R.1.16.0 class);
//   R1161QuditFrontier — cases at and beyond the exact regime. Values are
//                        still printed, because the fidelity curve is worth
//                        reading, but each is also asserted: inside the exact
//                        regime against 1, and beyond it against the chain's
//                        own fidelity floor, which the dense reference checks,
//                        so no expected number is carried over from a
//                        previous run.
//
// All gates go through the GATE path (apply_1qudit / apply_2qudit), not the
// dense constructor, because the two-site SVD split is the code under test.
// Matrix conventions: apply_2qudit's FIRST operand is the LEAST significant
// digit (project LSB-first rule); the SUM and CPHASE matrices below are
// built directly in that convention.

#include <gtest/gtest.h>

#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// Short local alias, library-sourced value.
constexpr double kPi = PI;

constexpr double kEps = std::numeric_limits<double>::epsilon();

// Backward-error allowance per operation, the figure the SVD ladder's verify
// rung grants, so these bounds and the ladder agree on what rounding is.
constexpr double kSlack = 64.0;

bool fp_bad(double x) {
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return ((b >> 52) & 0x7FFu) == 0x7FFu;
}

// SUM gate |a>|b> -> |a>|a+b mod d> with the FIRST operand (a) as control.
// apply_2qudit convention: first operand = LSB digit of the matrix index,
// so column c = a + d*b, row r = a + d*((a + b) % d).
std::vector<Complex128> sum_gate(int d) {
    std::vector<Complex128> U(static_cast<size_t>(d) * d * d * d,
                              Complex128(0, 0));
    const int dim = d * d;
    for (int a = 0; a < d; ++a) {
        for (int b = 0; b < d; ++b) {
            const int c = a + d * b;
            const int r = a + d * ((a + b) % d);
            U[static_cast<size_t>(r) * dim + c] = Complex128(1, 0);
        }
    }
    return U;
}

// CPHASE gate diag(omega^{a*b}), omega = exp(2*pi*i/d) — symmetric in its
// operands, so the LSB-first convention cannot be misapplied.
std::vector<Complex128> cphase_gate(int d) {
    std::vector<Complex128> U(static_cast<size_t>(d) * d * d * d,
                              Complex128(0, 0));
    const int dim = d * d;
    for (int a = 0; a < d; ++a) {
        for (int b = 0; b < d; ++b) {
            const int idx = a + d * b;
            const double phi = 2.0 * kPi * a * b / d;
            U[static_cast<size_t>(idx) * dim + idx] =
                Complex128(std::cos(phi), std::sin(phi));
        }
    }
    return U;
}

// Drive the SAME gate list through QuditMPS (gate path) and the dense
// QuditStatevector; return |<dense|mps>|^2. Asserts finiteness on the way.
struct GateOp {
    bool two;
    int q0, q1;
    const std::vector<Complex128>* U;
};

double gate_path_fidelity(int n, int d, const std::vector<GateOp>& ops,
                          int max_bond, bool* corrupt_out = nullptr) {
    QuditMPS mps(n, d, max_bond);
    QuditStatevector dense(n, d);
    for (const auto& op : ops) {
        if (op.two) {
            mps.apply_2qudit(op.q0, op.q1, *op.U);
            dense.apply_2qudit(op.q0, op.q1, *op.U);
        } else {
            mps.apply_1qudit(op.q0, *op.U);
            dense.apply_1qudit(op.q0, *op.U);
        }
    }
    const QuditStatevector out = mps.to_statevector();
    bool corrupt = false;
    std::complex<double> ov(0, 0);
    for (size_t i = 0; i < dense.amplitudes.size(); ++i) {
        const auto& m = out.amplitudes[i];
        corrupt = corrupt || fp_bad(m.real) || fp_bad(m.imag);
        ov += std::conj(std::complex<double>(dense.amplitudes[i].real,
                                             dense.amplitudes[i].imag)) *
              std::complex<double>(m.real, m.imag);
    }
    if (corrupt_out) *corrupt_out = corrupt;
    return std::norm(ov);
}

// One run of the same gate list through the chain and the dense reference,
// with every figure the frontier probe reads taken from that one chain.
struct CappedRun {
    double raw_overlap_sq = 0.0;  // |<dense|mps>|^2 for the chain as it stands
    double fidelity = 0.0;        // the same over both norms: normalised states
    double norm_sq = 0.0;         // <mps|mps>, which truncation lowers
    std::optional<double> estimate;
    std::optional<double> lower_bound;
    double discarded = 0.0;
    double dust_bound = 0.0;
    std::size_t splits = 0;
    int widest = 1;
    bool corrupt = false;
};

CappedRun run_capped(int n, int d, const std::vector<GateOp>& ops,
                     int max_bond) {
    QuditMPS mps(n, d, max_bond);
    QuditStatevector dense(n, d);
    for (const auto& op : ops) {
        if (op.two) {
            mps.apply_2qudit(op.q0, op.q1, *op.U);
            dense.apply_2qudit(op.q0, op.q1, *op.U);
        } else {
            mps.apply_1qudit(op.q0, *op.U);
            dense.apply_1qudit(op.q0, *op.U);
        }
    }
    CappedRun r;
    const QuditStatevector out = mps.to_statevector();
    std::complex<double> ov(0, 0);
    double dense_sq = 0.0;
    double mps_sq = 0.0;
    for (size_t i = 0; i < dense.amplitudes.size(); ++i) {
        const auto& m = out.amplitudes[i];
        const auto& a = dense.amplitudes[i];
        r.corrupt = r.corrupt || fp_bad(m.real) || fp_bad(m.imag);
        ov += std::conj(std::complex<double>(a.real, a.imag)) *
              std::complex<double>(m.real, m.imag);
        dense_sq += a.real * a.real + a.imag * a.imag;
        mps_sq += m.real * m.real + m.imag * m.imag;
    }
    r.raw_overlap_sq = std::norm(ov);
    r.fidelity = (dense_sq > 0.0 && mps_sq > 0.0)
                     ? r.raw_overlap_sq / (dense_sq * mps_sq)
                     : 0.0;
    r.norm_sq = mps.norm_sq();
    r.estimate = mps.fidelity_estimate();
    r.lower_bound = mps.fidelity_lower_bound();
    r.discarded = mps.truncation_error();
    r.splits = mps.svd_call_count();
    // The most a run can discard without the cap ever binding. svd_cutoff is
    // a FRACTION of a block's weight, a block in canonical gauge carries the
    // state's norm, which starts at 1 and only falls, and truncation_error()
    // sums one contribution per split. So the budget alone cannot account for
    // more than cutoff * splits, and a covering cap has nothing else to reject
    // with.
    r.dust_bound = mps.svd_cutoff * static_cast<double>(r.splits);
    // The widest bond the run reached: an integer, so unlike a fidelity it is
    // identical on every compiler and every flag setting, which is what makes
    // it usable as the guard on a truncating path.
    for (const auto& t : mps.tensors()) r.widest = std::max(r.widest, t.chi_R);
    return r;
}

}  // namespace

// =============================================================================
// R1161QuditStress — asserted, exact-bond regime
// =============================================================================

// GHZ_d via the gate path (F(d) then a SUM chain): flat rank-d spectrum
// with exact zeros at every interior cut — the Eigen trigger family — far
// beyond the existing GHZ_d6_n3 coverage, and swap-routed via long-range
// operands.
TEST(R1161QuditStress, GhzGatePathFlatSpectrumExact) {
    struct Case { int d, n; };
    const Case cases[] = {{3, 8}, {5, 7}, {6, 6}};
    for (const auto& c : cases) {
        SCOPED_TRACE("d=" + std::to_string(c.d) + " n=" + std::to_string(c.n));
        const auto F = qudit_gates::qft_matrix(c.d);
        const auto SUM = sum_gate(c.d);

        std::vector<GateOp> ops;
        ops.push_back({false, 0, 0, &F});
        // Long-range chain covering every site exactly once as target.
        std::vector<bool> seen(static_cast<size_t>(c.n), false);
        seen[0] = true;
        int prev = 0;
        for (int t = 1; t < c.n; ++t) {
            int tgt = (prev + (c.n / 2) + 1) % c.n;
            while (seen[static_cast<size_t>(tgt)]) tgt = (tgt + 1) % c.n;
            seen[static_cast<size_t>(tgt)] = true;
            ops.push_back({true, prev, tgt, &SUM});
            prev = tgt;
        }

        bool corrupt = false;
        const double fid = gate_path_fidelity(c.n, c.d, ops, 64, &corrupt);
        EXPECT_FALSE(corrupt) << "non-finite amplitude: the latent qudit "
                                 "pattern manifested — port the R.1.16.0 fix";
        EXPECT_NEAR(fid, 1.0, 1e-9)
            << "qudit MPS diverged on a flat-spectrum state (chi_max = d = "
            << c.d << " << 64: exact regime)";
    }
}

// Qudit QFT ladder (per-site F(d) + full CPHASE ladder), the qudit analogue
// of the circuit family that exposed #44. The d=2, n=12 case puts chi_max
// at 2^6 = 64 == the bond cap: the exact-regime EDGE, driven through the
// QUDIT code paths. (A d=4, n=6 edge would hit the same chi with 256-wide
// theta matrices whose JacobiSVDs cost tens of seconds — the d=2 edge
// exercises the identical truncation logic at 128-wide affordably; d=3
// covers the d > 2 arithmetic.)
TEST(R1161QuditStress, QftLadderExactIncludingBondEdge) {
    struct Case { int d, n; };
    const Case cases[] = {{3, 6}, {2, 12}};
    for (const auto& c : cases) {
        SCOPED_TRACE("d=" + std::to_string(c.d) + " n=" + std::to_string(c.n));
        const auto F = qudit_gates::qft_matrix(c.d);
        const auto CP = cphase_gate(c.d);

        std::vector<GateOp> ops;
        // Entangle first so the ladder acts on a correlated state: GHZ prep.
        const auto SUM = sum_gate(c.d);
        ops.push_back({false, 0, 0, &F});
        for (int t = 1; t < c.n; ++t) ops.push_back({true, t - 1, t, &SUM});
        // QFT-style ladder.
        for (int i = c.n - 1; i >= 0; --i) {
            ops.push_back({false, i, i, &F});
            for (int j = i - 1; j >= 0; --j) ops.push_back({true, j, i, &CP});
        }

        bool corrupt = false;
        const double fid = gate_path_fidelity(c.n, c.d, ops, 64, &corrupt);
        EXPECT_FALSE(corrupt);
        EXPECT_NEAR(fid, 1.0, 1e-9)
            << "qudit MPS diverged (chi_max = d^(n/2) = "
            << std::pow(c.d, c.n / 2) << " <= 64: exact regime)";
    }
}

// The missing coverage class outright: a LONG deterministic gate chain
// (168 gates) at d=3, n=7 (chi_max = 27 << 64), mixing F(d), SUM on
// long-range pairs, and CPHASE. Every two-site op exercises the qudit SVD
// split; the dense reference pins exactness after the full chain.
TEST(R1161QuditStress, LongSeededChainExact) {
    const int d = 3, n = 7;
    const auto F = qudit_gates::qft_matrix(d);
    const auto SUM = sum_gate(d);
    const auto CP = cphase_gate(d);

    std::vector<GateOp> ops;
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
    ASSERT_EQ(ops.size(), 168u);

    bool corrupt = false;
    const double fid = gate_path_fidelity(n, d, ops, 64, &corrupt);
    EXPECT_FALSE(corrupt) << "non-finite amplitude in the long-chain run";
    EXPECT_NEAR(fid, 1.0, 1e-9)
        << "qudit MPS drifted over a long gate chain in the exact regime";
}

// =============================================================================
// R1161QuditFrontier — at and beyond the exact regime; values printed, all asserted
// =============================================================================

// Simon-style state one size beyond the R.1.11.2 case (n=3 query digits at
// d=6): dense-ctor round trip through the reconstruction SVD chain.
TEST(R1161QuditFrontier, SimonD6N3RoundTripProbe) {
    const int d = 6;
    QuditStatevector psi(6, d);  // 3 query + 3 output
    const auto F = qudit_gates::qft_matrix(d);
    for (int q = 0; q < 3; ++q) psi.apply_1qudit(q, F);
    const std::vector<int> s = {2, 4, 3};
    psi.apply_function_oracle(3, 3, [&](const std::vector<int>& x) {
        std::vector<int> best = x;
        for (int k = 1; k < d; ++k) {
            std::vector<int> cand = {(x[0] + k * s[0]) % d,
                                     (x[1] + k * s[1]) % d,
                                     (x[2] + k * s[2]) % d};
            if (cand < best) best = cand;
        }
        return best;
    });

    QuditMPS mps(psi, /*max_bond_dim=*/64);
    const QuditStatevector back = mps.to_statevector();

    bool corrupt = false;
    std::complex<double> ov(0, 0);
    double norm_sq = 0.0;
    for (size_t i = 0; i < psi.amplitudes.size(); ++i) {
        const auto& m = back.amplitudes[i];
        corrupt = corrupt || fp_bad(m.real) || fp_bad(m.imag);
        ov += std::conj(std::complex<double>(psi.amplitudes[i].real,
                                             psi.amplitudes[i].imag)) *
              std::complex<double>(m.real, m.imag);
        norm_sq += m.real * m.real + m.imag * m.imag;
    }
    const double fid = std::norm(ov);

    std::cout << std::fixed << std::setprecision(12)
              << "[qudit-frontier] Simon d=6 n=3 round trip: fid=" << fid
              << "  norm^2=" << norm_sq
              << (corrupt ? "  <-- NON-FINITE" : "") << "\n";
    EXPECT_FALSE(corrupt);
    EXPECT_NEAR(norm_sq, 1.0, 1e-6);
    // This case is INSIDE the exact regime, so its fidelity is not
    // informational. Every bond of a 3-site chain carries Schmidt rank at most
    // d = 6, far under the cap of 64, so nothing truncates and the round trip
    // is lossless in exact arithmetic. A shortfall is a defect rather than a
    // cost of truncation, which is why it is asserted and not printed.
    EXPECT_NEAR(fid, 1.0, 1e-9)
        << "a dense round trip that truncates nothing lost overlap with the "
           "state it came from";
}

// Deliberately BEYOND the exact regime: d=3, n=8 (chi_max = 81 > cap 64),
// same long-chain generator. Truncation is expected and legitimate here;
// the probe reports the infidelity it costs and guards only against
// non-finite garbage (which truncation must never produce).
TEST(R1161QuditFrontier, BeyondExactRegimeTruncationProbe) {
    const int d = 3, n = 8;
    const auto F = qudit_gates::qft_matrix(d);
    const auto SUM = sum_gate(d);
    const auto CP = cphase_gate(d);

    std::vector<GateOp> ops;
    for (int step = 0; step < 40; ++step) {
        ops.push_back({false, (3 * step + 1) % n, 0, &F});
        const int a = (5 * step) % n;
        int b = (2 * step + 3) % n;
        if (b == a) b = (b + 1) % n;
        ops.push_back({true, a, b, (step % 2 == 0) ? &SUM : &CP});
    }

    // The only qudit case where truncation actually engages, so it is the only
    // guard the qudit truncation path has. A non-finite check cannot be that
    // guard: the failure this library keeps meeting is an answer that is
    // finite, self-consistent and wrong, and #91 retained a sixteenth of its
    // overlap with every amplitude finite.
    //
    // WHAT IS PROVEN HERE, and what is not, because the distinction decides
    // which assertions are worth anything.
    //
    // Every split runs in canonical gauge, so each one removes a fraction of
    // the STATE, and the chain carries two figures built from those fractions.
    // fidelity_lower_bound() is a floor on the fidelity between the chain and
    // the state an untruncated run would hold, and the dense statevector IS
    // that state, so the floor is checked here against an independent
    // reference at every cap. fidelity_estimate() is not a bound and is held
    // only to [0, 1] and to sitting on or above the floor, which follows from
    // the two definitions (detail/fidelity_ledger.hpp). And truncation_error()
    // is the weight the state lost, so on this run, with no collapse and no
    // normalisation, it equals how far norm_sq() fell below 1.
    //
    // What is NOT assertable is an ordering across caps. Two runs at different
    // caps diverge after the first truncation and approximate different
    // trajectories, so deep in the truncated regime the fidelity is not
    // monotone in the cap, and its value moves with codegen.
    //
    // At the exact cap the pairing holds as well: the library reports
    // discarding no more than the weight budget's dust AND reproduces the dense
    // state exactly. Either alone is weak, since a broken path could report
    // nothing while losing weight, or lose nothing while miscounting.
    int exact_chi = 1;
    for (int i = 0; i < n / 2; ++i) exact_chi *= d;  // 3^4 = 81

    std::size_t dim = 1;
    for (int i = 0; i < n; ++i) dim *= static_cast<std::size_t>(d);

    for (int cap : {2, 8, 32, 64, exact_chi}) {
        SCOPED_TRACE("chi=" + std::to_string(cap));
        const CappedRun r = run_capped(n, d, ops, cap);
        std::cout << std::fixed << std::setprecision(12)
                  << "[qudit-frontier] d=3 n=8 chi=" << cap
                  << " fid=" << r.fidelity << " floor="
                  << (r.lower_bound ? *r.lower_bound : -1.0) << " estimate="
                  << (r.estimate ? *r.estimate : -1.0)
                  << " widest bond=" << r.widest
                  << " discarded=" << r.discarded
                  << (r.corrupt ? "  <-- NON-FINITE" : "") << std::endl;

        ASSERT_FALSE(r.corrupt)
            << "truncation may lose fidelity but must NEVER produce garbage";
        EXPECT_GE(r.discarded, 0.0) << "negative discarded weight";

        // Rounding in either engine: every gate and every term of the overlap
        // sum can move a fidelity by a few eps, and the canonical-gauge
        // identity picks up a few eps per split and per centre step, of which
        // there are at most n per gate.
        const double fid_tol =
            kSlack * static_cast<double>(ops.size() + dim) * kEps;
        const double norm_tol =
            kSlack * static_cast<double>(r.splits + ops.size() * n) * kEps;

        EXPECT_GE(r.fidelity, 0.0);
        EXPECT_LE(r.fidelity, 1.0 + fid_tol)
            << "normalised overlap with the dense state exceeded unity";
        EXPECT_LE(r.raw_overlap_sq, 1.0 + fid_tol)
            << "the chain's overlap with a unit state exceeded unity";

        ASSERT_TRUE(r.lower_bound.has_value())
            << "no collapse happened, so the floor must be reported";
        ASSERT_TRUE(r.estimate.has_value())
            << "no collapse happened, so the estimate must be reported";
        EXPECT_GE(*r.lower_bound, 0.0);
        EXPECT_LE(*r.estimate, 1.0);
        EXPECT_LE(*r.lower_bound, *r.estimate + fid_tol)
            << "the floor sits above the estimate, which the two definitions "
               "rule out";
        EXPECT_LE(*r.lower_bound, r.fidelity + fid_tol)
            << "the chain's fidelity floor " << *r.lower_bound
            << " exceeds its true fidelity " << r.fidelity
            << " against the dense reference: the floor is not a bound";

        EXPECT_NEAR(1.0 - r.norm_sq, r.discarded, norm_tol)
            << "truncation_error() is not the weight the state lost: norm_sq() "
               "fell by " << (1.0 - r.norm_sq) << " while the splits report "
            << r.discarded;
        // From unit norm each split removes eps_k of what is left, so the
        // weights telescope to 1 - prod_k (1 - eps_k), the estimate's
        // complement.
        EXPECT_NEAR(r.discarded, 1.0 - *r.estimate, norm_tol)
            << "the discarded weight and the retained product disagree";

        if (cap >= exact_chi) {
            // A cap covering every Schmidt direction the state can carry has
            // nothing to reject beyond the weight budget's dust, and a path
            // that rejected only dust must reproduce the reference. Bounded
            // rather than compared to zero: the weight budget is a fraction
            // and remains in force at any cap, so a covering run may still
            // shed dust. The bound is what that budget can account for,
            // derived from the cutoff and the split count the run reports.
            EXPECT_LE(r.discarded, r.dust_bound)
                << "a cap of " << cap << " covers d^(n/2) = " << exact_chi
                << ", so the only thing left to reject with is the weight "
                << "budget, which cannot account for more than " << r.dust_bound
                << "; it discarded " << r.discarded << " instead, which means "
                << "the cap bound something it should not have";
            EXPECT_NEAR(r.raw_overlap_sq, 1.0, 1e-9)
                << "nothing was discarded and the state still moved, so the "
                   "loss is in the contraction or the factorisation rather "
                   "than in truncation";
        } else {
            // And below it the case must genuinely truncate, or the assertions
            // above are describing a path that never engages.
            EXPECT_GT(r.discarded, 0.0)
                << "chi=" << cap << " discarded nothing, so this case is not "
                << "beyond the exact regime and proves nothing about it";
            EXPECT_LT(*r.estimate, 1.0)
                << "chi=" << cap << " truncated, so the retained product must "
                   "fall below 1";
            EXPECT_EQ(r.widest, cap)
                << "chi=" << cap << ": the widest bond came back at " << r.widest
                << " rather than at the cap, so something other than the cap "
                << "bounded the selection. That is the shape of a rank chosen "
                << "from a comparison that failed rather than from the budget.";
        }
    }
}
