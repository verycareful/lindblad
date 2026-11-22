// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

// =============================================================================
// sv_timing - timing a statevector pass on this machine
// =============================================================================
// The measurement helpers and the kernels a calibration times, shared by the
// fusion costs (statevector_sim.cpp) and the threading costs
// (threading_model.cpp), so that both rate a pass by the same clock, the same
// kernels and the same rule: every time is the fastest of several alternating
// stretches, since interference (another process, a thread placed badly, a
// state about the size of the caches partly evicted) only ever slows a
// stretch down.

#include "lindblad/constants.hpp"
#include "lindblad/detail/sv_dense.hpp"
#include "lindblad/detail/sv_groups.hpp"
#include "lindblad/detail/sv_runs.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace lindblad {
namespace detail {

// Calls of f that last at least `seconds`, at least one. The first call warms
// the caches; when it alone lasts that long, it is the answer.
template <class F>
long calls_to_fill(F& f, double seconds) {
    using Clock = std::chrono::steady_clock;
    const auto w0 = Clock::now();
    f();
    if (std::chrono::duration<double>(Clock::now() - w0).count() >= seconds) return 1;
    long calls = 1;
    for (;;) {
        const auto t0 = Clock::now();
        for (long c = 0; c < calls; ++c) f();
        if (std::chrono::duration<double>(Clock::now() - t0).count() >= seconds) return calls;
        calls *= 2;
    }
}

template <class F>
double seconds_per_call(F& f, long calls) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    for (long c = 0; c < calls; ++c) f();
    return std::chrono::duration<double>(Clock::now() - t0).count() / static_cast<double>(calls);
}

// The fastest per-call time of each kernel over `rounds` rounds, each round
// timing every kernel once in turn, each stretch at least `seconds` long.
template <class... F>
std::array<double, sizeof...(F)> fastest(int rounds, double seconds, F&... f) {
    const std::array<long, sizeof...(F)> calls{calls_to_fill(f, seconds)...};
    std::array<double, sizeof...(F)> best;
    best.fill(std::numeric_limits<double>::infinity());
    for (int round = 0; round < rounds; ++round) {
        std::size_t i = 0;
        ((best[i] = std::min(best[i], seconds_per_call(f, calls[i])), ++i), ...);
    }
    return best;
}

// The kernels a measurement times, on re, im, 2^m amplitudes.
struct CalibrationKernels {
    double* re;
    double* im;
    int m;
    std::size_t dim() const { return std::size_t{1} << m; }

    // Every amplitude the same finite value, so no pass meets a denormal; every
    // pass below is unitary, so the values stay that size.
    void fill() const {
        const double v = 1.0 / std::sqrt(static_cast<double>(dim()));
        double* r = re;
        double* i = im;
        detail::for_each_share(dim(), detail::streaming_work(dim()), [=](std::size_t b, std::size_t e) {
            for (std::size_t k = b; k < e; ++k) {
                r[k] = v;
                i[k] = 0.0;
            }
        });
    }
    void h() const {
        const int q[1] = {m / 2};
        detail::for_each_group<1, 0b11>(re, im, dim(), q, [](auto* r, auto* i) {
            const auto r0 = r[0], i0 = i[0], r1 = r[1], i1 = i[1];
            r[0] = INV_SQRT2 * (r0 + r1);
            i[0] = INV_SQRT2 * (i0 + i1);
            r[1] = INV_SQRT2 * (r0 - r1);
            i[1] = INV_SQRT2 * (i0 - i1);
        });
    }
    // H passes for `seconds`, until the state has settled: the fill divides
    // the state among threads differently from a gate's sweep.
    void settle(double seconds) const {
        const auto t0 = std::chrono::steady_clock::now();
        do h();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds);
    }
    void cphase() const {
        const int q[2] = {detail::kLaneBits, m - 2};
        const double c = std::cos(0.3), s = std::sin(0.3);
        detail::for_each_diagonal<2, 0b1000>(re, im, dim(), q, [=](auto, auto& r, auto& i) {
            const auto r0 = r, i0 = i;
            r = r0 * c - i0 * s;
            i = r0 * s + i0 * c;
        });
    }
    // Targets alternate low and high qubits, the mix a fused block has.
    std::vector<int> targets(int k) const {
        const int pool[detail::kDenseMaxQubits] = {0, m - 1, 2, m / 2, 1, m - 3};
        return std::vector<int>(pool, pool + k);
    }
    // The dense matrix is the discrete Fourier transform: unitary, with no
    // zero entry, so no path takes a shortcut.
    static std::vector<Complex128> dft(int k) {
        const std::size_t n = std::size_t{1} << k;
        const double norm = 1.0 / std::sqrt(static_cast<double>(n));
        std::vector<Complex128> f(n * n);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c) {
                const double a = 2.0 * PI * static_cast<double>((r * c) % n) / static_cast<double>(n);
                f[r * n + c] = Complex128(norm * std::cos(a), norm * std::sin(a));
            }
        return f;
    }
    static std::vector<Complex128> phases(int k) {
        const std::size_t n = std::size_t{1} << k;
        std::vector<Complex128> d(n * n, Complex128(0.0, 0.0));
        for (std::size_t s = 0; s < n; ++s)
            d[s * n + s] = Complex128(std::cos(0.1 * static_cast<double>(s)),
                                      std::sin(0.1 * static_cast<double>(s)));
        return d;
    }
};

}  // namespace detail
}  // namespace lindblad
