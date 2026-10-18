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
// apply_dense - a dense k-qubit unitary, eight groups at a time
// =============================================================================
// A k-qubit gate acts on groups of N = 2^k amplitudes: member s of a group has
// bit j of s equal to the bit of targets[j], and the new member r is
// sum_s M[r][s] * old member s, M row-major. apply_dense runs that product on
// Lane8 values (sv_lanes.hpp), each lane holding a different group, so one
// pass over the matrix updates eight groups.
//
// A target below kLaneBits is a lane bit (a LOW target): the amplitudes it
// pairs sit in the same cache line. The others are line bits (HIGH targets).
// With t low targets, a cache line holds 2^t members of each of 2^(3-t)
// groups, and a BLOCK-GROUP (a line with every high-target bit clear, and the
// lines its high targets reach) holds 2^k members of each of those groups. A
// TILE is 2^t consecutive block-groups: 2^k lines of each array, eight groups.
//
//   load        line h of block-group u of the tile goes to slot h * 2^t + u.
//   split       for each low target i (ascending), deinterleave the slots
//               whose u differs in bit i by that target's lane bit: after the
//               t stages, slot h * 2^t + u holds one member of all eight
//               groups, its low-target bits the bits of u and its high-target
//               bits the bits of h (sv_lanes.hpp explains the lane order).
//   multiply    y[r] = sum_c W[r][c] x[c] over slots, W = M with rows and
//               columns renumbered from members to slots, split into real and
//               imaginary arrays once per call. Four rows at a time, so each
//               x[c] is loaded once per four rows.
//   join        interleave the stages back in reverse and store.
//
// With no low target there is nothing to split: each line already holds one
// member of eight consecutive groups.
//
// A diagonal M (every off-diagonal entry exactly zero, which a block fused
// from diagonal gates always is) scales each amplitude by its own entry: each
// line of a block-group is multiplied by a coefficient Lane8 whose lane l
// holds the entry of the member that line and lane select. One complex
// multiply per amplitude replaces 2^k.
//
// Tiles and block-groups are split across a team as for_each_share splits any
// flat pass, sized for the pass's work: dim * N / 2 one-qubit-gate amplitudes
// for the dense product (N complex multiply-adds per amplitude), dim for the
// diagonal. Every amplitude's value is computed the same way whatever the
// team, so the result does not depend on the thread count.
//
// apply_dense needs k <= kDenseMaxQubits (the slot arrays live on the stack)
// and one whole tile, dim >= kLanes * 2^k.
//
// For k = 1 and 2 the dense matrix is 2x2 or 4x4, too small for the tile
// machinery to pay for itself: the product is a group function for
// for_each_group (sv_groups.hpp), the sweep the one- and two-qubit gates use.
// A diagonal matrix of any k takes the diagonal product above, and
// apply_small_dense serves k = 1 and 2 on a state smaller than one tile.

#include "lindblad/detail/sv_groups.hpp"
#include "lindblad/detail/sv_lanes.hpp"
#include "lindblad/detail/sv_runs.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace lindblad {
namespace detail {

inline constexpr int kDenseMaxQubits = 6;

namespace sv_dense_impl {

inline constexpr std::size_t kMaxMembers = std::size_t{1} << kDenseMaxQubits;

// The lane bit by which split stage i deinterleaves, for low-target lane bits
// L: the i-th set bit of L (ascending), shifted right by i because the i lower
// lane bits split out before it are no longer lane bits of the slots.
template <unsigned L>
constexpr std::size_t stage_bit(int i) {
    int seen = 0;
    for (unsigned s = 1; s <= 4; s <<= 1) {
        if (L & s) {
            if (seen == i) return std::size_t{s} >> i;
            ++seen;
        }
    }
    return 0;
}

// Whether every off-diagonal entry of the n x n row-major m is exactly zero.
inline bool is_diagonal(const std::vector<Complex128>& m, std::size_t n) {
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c)
            if (r != c && (m[r * n + c].real != 0.0 || m[r * n + c].imag != 0.0)) return false;
    return true;
}

// Everything apply_dense derives from the targets and the matrix, once per
// call.
struct DensePlan {
    int k = 0;
    std::size_t n = 0;                // members, 2^k
    unsigned low = 0;                 // lane bits of the low targets
    int t = 0;                        // number of low targets
    int high[kDenseMaxQubits] = {};   // high targets as line-index bits, ascending
    int m_high = 0;
    std::size_t mask = 0;             // the high targets' line-index bits
    std::size_t line_off[kMaxMembers] = {};  // amplitude offset of line h
    bool diagonal = false;
    std::vector<double> wre, wim;     // dense: W in slot order, n x n row-major
    std::vector<double> dre, dimg;    // diagonal: coefficient lanes, one Lane8 per line h
};

inline DensePlan make_plan(const std::vector<int>& targets, const std::vector<Complex128>& m) {
    DensePlan p;
    p.k = static_cast<int>(targets.size());
    p.n = std::size_t{1} << p.k;
    int low_rank[kDenseMaxQubits];   // for each target j: rank among low targets, or -1
    int high_rank[kDenseMaxQubits];  // rank among high targets, or -1
    std::vector<int> highs;
    std::vector<int> lows;
    for (int q : targets) {
        if (q < kLaneBits) lows.push_back(q);
        else highs.push_back(q - kLaneBits);
    }
    std::sort(lows.begin(), lows.end());
    std::sort(highs.begin(), highs.end());
    for (int q : lows) p.low |= 1u << q;
    p.t = static_cast<int>(lows.size());
    p.m_high = static_cast<int>(highs.size());
    for (int j = 0; j < p.m_high; ++j) {
        p.high[j] = highs[static_cast<std::size_t>(j)];
        p.mask |= std::size_t{1} << p.high[j];
    }
    for (int j = 0; j < p.k; ++j) {
        const int q = targets[static_cast<std::size_t>(j)];
        low_rank[j] = -1;
        high_rank[j] = -1;
        if (q < kLaneBits)
            low_rank[j] = static_cast<int>(std::find(lows.begin(), lows.end(), q) - lows.begin());
        else
            high_rank[j] = static_cast<int>(std::find(highs.begin(), highs.end(), q - kLaneBits) -
                                            highs.begin());
    }
    const std::size_t lines = p.n >> p.t;
    for (std::size_t h = 0; h < lines; ++h) {
        std::size_t off = 0;
        for (int j = 0; j < p.m_high; ++j)
            if ((h >> j) & 1u) off += std::size_t{1} << (p.high[j] + kLaneBits);
        p.line_off[h] = off;
    }

    p.diagonal = is_diagonal(m, p.n);

    if (p.diagonal) {
        // Lane l of line h: the member whose high-target bits are h's and
        // whose low-target bits are those of lane l.
        p.dre.assign(lines * kLanes, 0.0);
        p.dimg.assign(lines * kLanes, 0.0);
        for (std::size_t h = 0; h < lines; ++h)
            for (std::size_t l = 0; l < kLanes; ++l) {
                std::size_t s = 0;
                for (int j = 0; j < p.k; ++j) {
                    const int q = targets[static_cast<std::size_t>(j)];
                    const std::size_t bit = q < kLaneBits ? (l >> q) & 1u
                                                          : (h >> high_rank[j]) & 1u;
                    s |= bit << j;
                }
                p.dre[h * kLanes + l] = m[s * p.n + s].real;
                p.dimg[h * kLanes + l] = m[s * p.n + s].imag;
            }
        return p;
    }

    // Slot h * 2^t + u holds the member whose low-target bits are u's and
    // whose high-target bits are h's.
    std::size_t member[kMaxMembers];
    for (std::size_t slot = 0; slot < p.n; ++slot) {
        const std::size_t u = slot & ((std::size_t{1} << p.t) - 1);
        const std::size_t h = slot >> p.t;
        std::size_t s = 0;
        for (int j = 0; j < p.k; ++j) {
            const std::size_t bit = low_rank[j] >= 0 ? (u >> low_rank[j]) & 1u
                                                     : (h >> high_rank[j]) & 1u;
            s |= bit << j;
        }
        member[slot] = s;
    }
    p.wre.resize(p.n * p.n);
    p.wim.resize(p.n * p.n);
    for (std::size_t r = 0; r < p.n; ++r)
        for (std::size_t c = 0; c < p.n; ++c) {
            const Complex128& e = m[member[r] * p.n + member[c]];
            p.wre[r * p.n + c] = e.real;
            p.wim[r * p.n + c] = e.imag;
        }
    return p;
}

// y = W x over n slots, RB rows at a time (each x[c] loaded once per RB rows).
template <std::size_t RB>
inline void multiply(const double* __restrict__ wre, const double* __restrict__ wim,
                     std::size_t n, const Lane8* xr, const Lane8* xi, Lane8* yr, Lane8* yi) {
    for (std::size_t r = 0; r < n; r += RB) {
        Lane8 ar[RB];
        Lane8 ai[RB];
        for (std::size_t q = 0; q < RB; ++q) {
            ar[q] = Lane8{};
            ai[q] = Lane8{};
        }
        for (std::size_t c = 0; c < n; ++c) {
            const Lane8 vr = xr[c];
            const Lane8 vi = xi[c];
            for (std::size_t q = 0; q < RB; ++q) {
                const double mr = wre[(r + q) * n + c];
                const double mi = wim[(r + q) * n + c];
                ar[q] = ar[q] + (mr * vr - mi * vi);
                ai[q] = ai[q] + (mr * vi + mi * vr);
            }
        }
        for (std::size_t q = 0; q < RB; ++q) {
            yr[r + q] = ar[q];
            yi[r + q] = ai[q];
        }
    }
}

// The dense product for low-target lane bits L.
template <unsigned L>
inline void dense_tiles(double* re, double* im, std::size_t dim, const DensePlan& p) {
    constexpr int t = std::popcount(L);
    constexpr std::size_t U = std::size_t{1} << t;
    const std::size_t lines = p.n >> t;
    const std::size_t block_groups = (dim >> kLaneBits) >> (p.k - t);
    const std::size_t tiles = block_groups >> t;
    const std::size_t work = dim > std::numeric_limits<std::size_t>::max() / (p.n / 2)
                                 ? std::numeric_limits<std::size_t>::max()
                                 : dim * (p.n / 2);

    for_each_share(tiles, work, [=, &p](std::size_t begin, std::size_t end) {
        // The plan's fields as this thread's own locals: a Lane8 store is a
        // memcpy, after which anything read through a reference is reloaded.
        const std::size_t n = p.n;
        const std::size_t mask = p.mask;
        std::size_t off[kMaxMembers];
        std::copy(p.line_off, p.line_off + lines, off);
        const double* __restrict__ wre = p.wre.data();
        const double* __restrict__ wim = p.wim.data();
        double* __restrict__ rp = re;
        double* __restrict__ ip = im;
        Lane8 xr[kMaxMembers], xi[kMaxMembers], yr[kMaxMembers], yi[kMaxMembers];
        std::size_t base[U];  // line index of each block-group of the tile

        // Split stage I on one array: slots whose u differs in bit I.
        auto split = [&](Lane8* x, auto stage) {
            constexpr int I = decltype(stage)::value;
            constexpr std::size_t S = stage_bit<L>(I);
            for (std::size_t h = 0; h < lines; ++h)
                for (std::size_t u = 0; u < U; ++u)
                    if (!((u >> I) & 1u)) {
                        const std::size_t a = h * U + u;
                        const std::size_t b = a + (std::size_t{1} << I);
                        const Lane8 va = x[a];
                        const Lane8 vb = x[b];
                        deinterleave<S>(va, vb, x[a], x[b]);
                    }
        };
        auto join = [&](Lane8* x, auto stage) {
            constexpr int I = decltype(stage)::value;
            constexpr std::size_t S = stage_bit<L>(I);
            for (std::size_t h = 0; h < lines; ++h)
                for (std::size_t u = 0; u < U; ++u)
                    if (!((u >> I) & 1u)) {
                        const std::size_t a = h * U + u;
                        const std::size_t b = a + (std::size_t{1} << I);
                        const Lane8 va = x[a];
                        const Lane8 vb = x[b];
                        interleave<S>(va, vb, x[a], x[b]);
                    }
        };

        std::size_t b = insert_zeros(begin * U, p.high, p.m_high);
        for (std::size_t tile = begin; tile < end; ++tile) {
            for (std::size_t u = 0; u < U; ++u) {
                base[u] = b << kLaneBits;
                b = next_clear(b, mask);
            }
            for (std::size_t h = 0; h < lines; ++h)
                for (std::size_t u = 0; u < U; ++u) {
                    lane_load(xr[h * U + u], rp + base[u] + off[h]);
                    lane_load(xi[h * U + u], ip + base[u] + off[h]);
                }
            [&]<int... I>(std::integer_sequence<int, I...>) {
                (split(xr, std::integral_constant<int, I>{}), ...);
                (split(xi, std::integral_constant<int, I>{}), ...);
            }(std::make_integer_sequence<int, t>{});

            if (n >= 4) multiply<4>(wre, wim, n, xr, xi, yr, yi);
            else        multiply<2>(wre, wim, n, xr, xi, yr, yi);

            [&]<int... I>(std::integer_sequence<int, I...>) {
                constexpr int last = t - 1;
                (join(yr, std::integral_constant<int, last - I>{}), ...);
                (join(yi, std::integral_constant<int, last - I>{}), ...);
            }(std::make_integer_sequence<int, t>{});
            for (std::size_t h = 0; h < lines; ++h)
                for (std::size_t u = 0; u < U; ++u) {
                    lane_store(rp + base[u] + off[h], yr[h * U + u]);
                    lane_store(ip + base[u] + off[h], yi[h * U + u]);
                }
        }
    });
}

// High targets whose lines lie less than this many line indices apart are
// NEAR: 64 lines of eight doubles span 4 KiB, the address bits a processor
// compares when it decides whether a load may pass an earlier store.
inline constexpr int kNearLineBits = 6;

// The diagonal product: every line of every block-group times its
// coefficient Lane8. The line pattern h splits into far bits (high targets at
// line bits kNearLineBits and up) and near bits (the others: the lower bits
// of h, as the high targets are sorted). The sweep makes
// one pass over the thread's share per far pattern, and within a pass visits
// each block-group's near lines in address order:
//
//   - a pass streams: block-groups come in runs of consecutive line indices
//     (2^(lowest high target's line bit) long, the whole share when every
//     target is a lane bit), and the next run follows by a masked increment;
//   - lines whose addresses agree in their low 12 bits (a far pattern apart)
//     are a whole pass apart, so a load never waits on a store it only
//     appears to depend on;
//   - each line belongs to one pattern, so the passes together touch every
//     line once.
// NL = the near lines per block-group, a compile-time count so the per-line
// loop unrolls and a pass with one near line keeps its coefficient in
// registers.
template <std::size_t NL>
inline void dense_diagonal_near(double* re, double* im, std::size_t dim, const DensePlan& p) {
    const std::size_t lines = p.n >> p.t;
    const std::size_t block_groups = (dim >> kLaneBits) >> (p.k - p.t);
    const std::size_t far_patterns = lines / NL;

    for_each_share(block_groups, dim, [=, &p](std::size_t begin, std::size_t end) {
        const std::size_t mask = p.mask;
        const std::size_t run = p.m_high > 0 ? std::size_t{1} << p.high[0] : end - begin;
        const std::size_t first = insert_zeros(begin, p.high, p.m_high);
        double* __restrict__ rp = re;
        double* __restrict__ ip = im;
        for (std::size_t hf = 0; hf < far_patterns; ++hf) {
            std::size_t off[NL];
            Lane8 cr[NL], ci[NL];
            for (std::size_t hn = 0; hn < NL; ++hn) {
                const std::size_t h = hf * NL + hn;
                off[hn] = p.line_off[h];
                lane_load(cr[hn], p.dre.data() + h * kLanes);
                lane_load(ci[hn], p.dimg.data() + h * kLanes);
            }
            std::size_t b = first;
            for (std::size_t g = begin; g < end;) {
                const std::size_t len = std::min(run - (b & (run - 1)), end - g);
                for (std::size_t k = 0; k < len; ++k) {
                    const std::size_t a = (b + k) << kLaneBits;
                    for (std::size_t hn = 0; hn < NL; ++hn) {
                        Lane8 vr, vi;
                        lane_load(vr, rp + a + off[hn]);
                        lane_load(vi, ip + a + off[hn]);
                        const Lane8 nr = vr * cr[hn] - vi * ci[hn];
                        const Lane8 ni = vr * ci[hn] + vi * cr[hn];
                        lane_store(rp + a + off[hn], nr);
                        lane_store(ip + a + off[hn], ni);
                    }
                }
                g += len;
                b = next_clear(b + len - 1, mask);
            }
        }
    });
}

inline void dense_diagonal(double* re, double* im, std::size_t dim, const DensePlan& p) {
    int m_near = 0;
    while (m_near < p.m_high && p.high[m_near] < kNearLineBits) ++m_near;
    switch (m_near) {
        case 0: dense_diagonal_near<1>(re, im, dim, p); break;
        case 1: dense_diagonal_near<2>(re, im, dim, p); break;
        case 2: dense_diagonal_near<4>(re, im, dim, p); break;
        case 3: dense_diagonal_near<8>(re, im, dim, p); break;
        case 4: dense_diagonal_near<16>(re, im, dim, p); break;
        case 5: dense_diagonal_near<32>(re, im, dim, p); break;
        default: dense_diagonal_near<64>(re, im, dim, p); break;
    }
}

template <int K>
inline void small_dense(double* re, double* im, std::size_t dim, const std::vector<int>& targets,
                        const std::vector<Complex128>& m) {
    constexpr std::size_t N = std::size_t{1} << K;
    constexpr unsigned all = (1u << N) - 1;
    int q[K];
    for (int j = 0; j < K; ++j) q[j] = targets[static_cast<std::size_t>(j)];
    struct Matrix {
        double re[N][N];
        double im[N][N];
    } w;
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c) {
            w.re[r][c] = m[r * N + c].real;
            w.im[r][c] = m[r * N + c].imag;
        }
    if (is_diagonal(m, N)) {
        for_each_diagonal<K, all>(re, im, dim, q, [w](auto x, auto& r, auto& i) {
            constexpr std::size_t X = decltype(x)::value;
            const auto r0 = r;
            const auto i0 = i;
            r = r0 * w.re[X][X] - i0 * w.im[X][X];
            i = r0 * w.im[X][X] + i0 * w.re[X][X];
        });
        return;
    }
    for_each_group<K, all>(re, im, dim, q, [=](auto* r, auto* i) {
        using T = std::remove_reference_t<decltype(r[0])>;
        T xr[N];
        T xi[N];
        for (std::size_t c = 0; c < N; ++c) {
            xr[c] = r[c];
            xi[c] = i[c];
        }
        for (std::size_t row = 0; row < N; ++row) {
            T ar = w.re[row][0] * xr[0] - w.im[row][0] * xi[0];
            T ai = w.re[row][0] * xi[0] + w.im[row][0] * xr[0];
            for (std::size_t c = 1; c < N; ++c) {
                ar = ar + (w.re[row][c] * xr[c] - w.im[row][c] * xi[c]);
                ai = ai + (w.re[row][c] * xi[c] + w.im[row][c] * xr[c]);
            }
            r[row] = ar;
            i[row] = ai;
        }
    });
}

}  // namespace sv_dense_impl

// The k = 1 and 2 products on any state.
inline void apply_small_dense(double* re, double* im, std::size_t dim,
                              const std::vector<int>& targets, const std::vector<Complex128>& m) {
    if (targets.size() == 1) sv_dense_impl::small_dense<1>(re, im, dim, targets, m);
    else                     sv_dense_impl::small_dense<2>(re, im, dim, targets, m);
}

// re, im  = the amplitude arrays, dim of each (dim >= kLanes * 2^k)
// targets = the gate's k distinct qubits, k <= kDenseMaxQubits; targets[j] is
//           bit j of a member index
// m       = the 2^k x 2^k matrix, row-major
inline void apply_dense(double* re, double* im, std::size_t dim, const std::vector<int>& targets,
                        const std::vector<Complex128>& m) {
    using namespace sv_dense_impl;
    if (targets.size() <= 2 && !is_diagonal(m, std::size_t{1} << targets.size())) {
        apply_small_dense(re, im, dim, targets, m);
        return;
    }
    const DensePlan p = make_plan(targets, m);
    if (p.diagonal) {
        dense_diagonal(re, im, dim, p);
        return;
    }
    switch (p.low) {
        case 0: dense_tiles<0>(re, im, dim, p); break;
        case 1: dense_tiles<1>(re, im, dim, p); break;
        case 2: dense_tiles<2>(re, im, dim, p); break;
        case 3: dense_tiles<3>(re, im, dim, p); break;
        case 4: dense_tiles<4>(re, im, dim, p); break;
        case 5: dense_tiles<5>(re, im, dim, p); break;
        case 6: dense_tiles<6>(re, im, dim, p); break;
        default: dense_tiles<7>(re, im, dim, p); break;
    }
}

}  // namespace detail
}  // namespace lindblad
