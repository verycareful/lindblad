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
// for_each_group - a one- or two-qubit gate as one formula over its groups
// =============================================================================
// A gate on M qubits acts on groups of 2^M amplitudes, one per setting of its
// qubits. A kernel states the gate ONCE, as a group function
//
//   g(T* r, T* i)    r[x], i[x] = real and imaginary part of member x
//
// that reads the members it needs and writes the new values back. Member x has
// bit j of x equal to the bit of qubit q[j], where q is the gate's qubits in
// the kernel's own order (control first, say). T is double or Lane8, so g is a
// generic lambda written with the ordinary operators and every path below
// evaluates the same formula. Which path runs depends on the qubits and the
// state size only, so an amplitude's new value does not depend on how many
// threads ran.
//
// Touched is the set of members g reads or writes, bit x for member x. The
// paths load and store only those, so a controlled gate moves only the memory
// its controls select. g must not read a member outside Touched.
//
// g runs eight groups at a time, on Lane8 values (sv_lanes.hpp) or in an omp
// simd loop, by one of two paths chosen by the lowest qubit q_min:
//
//   blocks  q_min >= kLaneBits: the groups' base indices come in runs of
//           2^q_min >= kLanes consecutive indices, so member x of eight
//           consecutive groups is one whole block (cache line), kLanes-aligned.
//           A block-group is a block of bases with every gate bit clear. A
//           thread sweeps its share as runs of consecutive block-groups, the
//           next run's first block following from the last by a masked
//           increment; each block-group loads its members' blocks as Lane8,
//           runs g once and stores them. (A diagonal gate's long runs are
//           one omp simd loop instead, see kStretchBlocks.)
//   lanes   q_min < kLaneBits: a run is shorter than one cache line, so the
//           groups are swept by TILES of whole cache lines instead. A tile is
//           2^M blocks of each array; deinterleave sorts their lanes by the
//           gate's lane bits into one Lane8 per member, with the eight lanes of
//           the member Lane8s holding eight groups, g runs once, and
//           interleave puts the lanes back. A qubit at or above kLaneBits picks
//           which blocks a tile loads: its partner block lies 2^q amplitudes
//           away.
//
// Both paths need kLanes * 2^M amplitudes at least (one tile); a smaller state
// runs g with T = double, one group per iteration of a loop over
// for_each_run's runs. Both paths split their block-groups or tiles across a
// team as for_each_share splits any flat pass, sized for dim amplitudes of work
// (sv_team); a share is whole cache lines, so no two threads write one line.
// Each thread copies g and the sweep's other invariants into its own locals
// once for its share, so the compiler keeps them in registers: a Lane8 store
// is a memcpy, which may alias anything reached through a reference.

#include "lindblad/detail/sv_lanes.hpp"
#include "lindblad/detail/sv_runs.hpp"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace lindblad {
namespace detail {

// The run length, in blocks, from which a diagonal gate's blocks path sweeps a
// run as one omp simd loop over its amplitudes (the member function on
// doubles) rather than block by block (on Lane8). A loop pays its entry once
// per run, which a run of one or two blocks cannot amortise, while a long
// run's simd loop is the compiler's own vectorisation of a contiguous
// stretch. Measured with RZ at n = 18 on one thread: blocks are 1.5x faster
// for runs of one block and level at four, the simd loop 1 to 4 percent faster
// from eight. A whole-group function always runs on Lane8: the compiler does
// not reliably vectorise a heavier one inside a simd loop (a dense 4x4 block
// on high qubits ran 3.5x to 5.8x slower that way, measured at n = 22).
inline constexpr std::size_t kStretchBlocks = 8;

namespace sv_groups_impl {

// Calls f(std::integral_constant<std::size_t, x>) for x = 0 .. N-1.
template <std::size_t N, class F>
inline void for_each_member(F&& f) {
    [&]<std::size_t... X>(std::index_sequence<X...>) {
        (f(std::integral_constant<std::size_t, X>{}), ...);
    }(std::make_index_sequence<N>{});
}

template <unsigned Touched, std::size_t X>
inline constexpr bool touches = ((Touched >> X) & 1u) != 0;

// Member x's index minus its group's base: the sum of 2^q[j] over the bits j
// set in x. Member 0 is the base itself, which offset() states as a constant.
template <int M>
struct Offsets {
    std::size_t of[std::size_t{1} << M];

    explicit Offsets(const int* q) noexcept {
        for (std::size_t x = 0; x < (std::size_t{1} << M); ++x) {
            of[x] = 0;
            for (int j = 0; j < M; ++j)
                if ((x >> j) & 1u) of[x] += std::size_t{1} << q[j];
        }
    }

    template <std::size_t X>
    std::size_t offset() const noexcept { return X == 0 ? 0 : of[X]; }
};

// Member index in the kernel's order for sorted index y (bit 0 = the lower
// qubit): the two bits trade places when the kernel lists the higher qubit
// first.
template <bool Swapped>
inline constexpr std::size_t member(std::size_t y) {
    return Swapped ? (((y & 1u) << 1) | (y >> 1)) : y;
}

// ---- doubles: states smaller than one tile -----------------------------------

// A state this small (under kLanes * 2^M amplitudes) is a few dozen
// amplitudes, so the loop is plain: there is nothing for vectorising to win.
template <int M, unsigned Touched, class G>
inline void by_runs(double* re, double* im, std::size_t dim, const int* q, const G& g) {
    constexpr std::size_t N = std::size_t{1} << M;
    const Offsets<M> off(q);
    int sorted[M];
    std::copy(q, q + M, sorted);
    std::sort(sorted, sorted + M);

    for_each_run(dim, sorted, M, [=](std::size_t base, auto len) {
        double* __restrict__ rp = re;
        double* __restrict__ ip = im;
        const Offsets<M> o = off;
        for (std::size_t j = base; j < base + len; ++j) {
            double r[N];
            double i[N];
            for_each_member<N>([&](auto x) {
                constexpr std::size_t X = decltype(x)::value;
                if constexpr (touches<Touched, X>) {
                    r[X] = rp[j + o.template offset<X>()];
                    i[X] = ip[j + o.template offset<X>()];
                }
            });
            g(r, i);
            for_each_member<N>([&](auto x) {
                constexpr std::size_t X = decltype(x)::value;
                if constexpr (touches<Touched, X>) {
                    rp[j + o.template offset<X>()] = r[X];
                    ip[j + o.template offset<X>()] = i[X];
                }
            });
        }
    });
}

// ---- blocks path ------------------------------------------------------------


// The gate's qubits as bit positions in the block index (sorted), their mask,
// and the length of a run of consecutive block-groups: 2^(q_min - kLaneBits).
template <int M>
struct BlockGates {
    int bq[M];
    std::size_t mask = 0;
    std::size_t run = 0;

    explicit BlockGates(const int* q) noexcept {
        for (int j = 0; j < M; ++j) {
            bq[j] = q[j] - kLaneBits;
            mask |= std::size_t{1} << bq[j];
        }
        std::sort(bq, bq + M);
        run = std::size_t{1} << bq[0];
    }
};

// Calls body(first, len) for each run of len consecutive block indices
// covering block-groups [begin, end): within a run the next block-group is the
// next block, so the body's loop is a plain stride the compiler unrolls, and
// next_clear is taken only between runs.
template <int M, class Body>
inline void for_each_block_run(std::size_t begin, std::size_t end, const BlockGates<M>& bg,
                               Body&& body) {
    const std::size_t mask = bg.mask;
    const std::size_t run = bg.run;
    std::size_t b = insert_zeros(begin, bg.bq, M);
    for (std::size_t t = begin; t < end;) {
        const std::size_t len = std::min(run - (b & (run - 1)), end - t);
        body(b, len);
        t += len;
        b = next_clear(b + len - 1, mask);
    }
}

// Each thread sweeps its share of block-groups (a block with every gate bit
// clear, and its members' blocks) in runs (for_each_block_run). Everything the
// loop reads besides the amplitudes (g's coefficients, the offsets, the mask)
// is a value copied into the thread's own locals: a Lane8 load or store is a
// memcpy, which may alias any memory, so an invariant reached through a
// reference would be reloaded after every store.
template <int M, unsigned Touched, class G>
inline void by_blocks(double* re, double* im, std::size_t dim, const int* q, const G& g) {
    constexpr std::size_t N = std::size_t{1} << M;
    const Offsets<M> off(q);
    const BlockGates<M> gates(q);
    const std::size_t groups = (dim >> kLaneBits) >> M;

    for_each_share(groups, dim, [=, &g](std::size_t begin, std::size_t end) {
        const G gl = g;
        const Offsets<M> o = off;
        double* __restrict__ rp = re;
        double* __restrict__ ip = im;
        for_each_block_run(begin, end, gates, [&](std::size_t first, std::size_t len) {
            for (std::size_t k = 0; k < len; ++k) {
                const std::size_t a = (first + k) << kLaneBits;
                Lane8 r[N];
                Lane8 i[N];
                for_each_member<N>([&](auto x) {
                    constexpr std::size_t X = decltype(x)::value;
                    if constexpr (touches<Touched, X>) {
                        lane_load(r[X], rp + a + o.template offset<X>());
                        lane_load(i[X], ip + a + o.template offset<X>());
                    }
                });
                gl(r, i);
                for_each_member<N>([&](auto x) {
                    constexpr std::size_t X = decltype(x)::value;
                    if constexpr (touches<Touched, X>) {
                        lane_store(rp + a + o.template offset<X>(), r[X]);
                        lane_store(ip + a + o.template offset<X>(), i[X]);
                    }
                });
            }
        });
    });
}

// A diagonal gate's blocks path: d(x, r, i) scales member x on its own, so a
// run is swept one member at a time, streaming through one member's blocks
// before the next member's.
template <int M, unsigned Touched, class D>
inline void by_blocks_diagonal(double* re, double* im, std::size_t dim, const int* q,
                               const D& d) {
    constexpr std::size_t N = std::size_t{1} << M;
    const Offsets<M> off(q);
    const BlockGates<M> gates(q);
    const std::size_t groups = (dim >> kLaneBits) >> M;

    for_each_share(groups, dim, [=, &d](std::size_t begin, std::size_t end) {
        const D dl = d;
        const Offsets<M> o = off;
        double* __restrict__ rp = re;
        double* __restrict__ ip = im;
        for_each_block_run(begin, end, gates, [&](std::size_t first, std::size_t len) {
            for_each_member<N>([&](auto x) {
                constexpr std::size_t X = decltype(x)::value;
                if constexpr (touches<Touched, X>) {
                    const std::size_t a = (first << kLaneBits) + o.template offset<X>();
                    if (len >= kStretchBlocks) {
                        #pragma omp simd
                        for (std::size_t j = a; j < a + (len << kLaneBits); ++j) {
                            double r = rp[j];
                            double i = ip[j];
                            dl(x, r, i);
                            rp[j] = r;
                            ip[j] = i;
                        }
                        return;
                    }
                    for (std::size_t k = 0; k < len; ++k) {
                        const std::size_t b = a + (k << kLaneBits);
                        Lane8 r;
                        Lane8 i;
                        lane_load(r, rp + b);
                        lane_load(i, ip + b);
                        dl(x, r, i);
                        lane_store(rp + b, r);
                        lane_store(ip + b, i);
                    }
                }
            });
        });
    });
}

// ---- lanes path -------------------------------------------------------------

// One qubit on lane bit S: a tile is two consecutive blocks.
template <std::size_t S, class G>
inline void one_qubit_lanes(double* re, double* im, std::size_t dim, const G& g) {
    constexpr std::size_t tile = 2 * kLanes;
    for_each_share(dim / tile, dim, [=, &g](std::size_t begin, std::size_t end) {
        const G gl = g;
        double* __restrict__ rp = re;
        double* __restrict__ ip = im;
        for (std::size_t t = begin; t < end; ++t) {
            const std::size_t a = t * tile;
            Lane8 x, y, r[2], i[2];
            lane_load(x, rp + a);
            lane_load(y, rp + a + kLanes);
            deinterleave<S>(x, y, r[0], r[1]);
            lane_load(x, ip + a);
            lane_load(y, ip + a + kLanes);
            deinterleave<S>(x, y, i[0], i[1]);
            gl(r, i);
            interleave<S>(r[0], r[1], x, y);
            lane_store(rp + a, x);
            lane_store(rp + a + kLanes, y);
            interleave<S>(i[0], i[1], x, y);
            lane_store(ip + a, x);
            lane_store(ip + a + kLanes, y);
        }
    });
}

// Two qubits, the lower on lane bit S and the higher, `high`, at or above
// kLaneBits. A block-group is a block with the high bit clear and its partner
// 2^high above; a tile is two block-groups (consecutive in the order of
// block-group indices), so each deinterleave fills a Lane8 with the eight
// groups of the tile. The blocks of one setting of the high bit are left
// unread and unwritten when g touches none of their members.
template <std::size_t S, bool Swapped, unsigned Touched, class G>
inline void low_high_lanes(double* re, double* im, std::size_t dim, int high, const G& g) {
    constexpr std::size_t x00 = member<Swapped>(0);  // x{low bit}{high bit}
    constexpr std::size_t x10 = member<Swapped>(1);
    constexpr std::size_t x01 = member<Swapped>(2);
    constexpr std::size_t x11 = member<Swapped>(3);
    constexpr bool high_clear = touches<Touched, x00> || touches<Touched, x10>;
    constexpr bool high_set = touches<Touched, x01> || touches<Touched, x11>;
    const int block_high = high - kLaneBits;  // the high qubit in block indices
    const std::size_t bmask = std::size_t{1} << block_high;
    const std::size_t hb = std::size_t{1} << high;
    constexpr std::size_t tile = 4 * kLanes;

    for_each_share(dim / tile, dim, [=, &g](std::size_t begin, std::size_t end) {
        const G gl = g;
        double* __restrict__ rp = re;
        double* __restrict__ ip = im;
        std::size_t b = insert_zeros(2 * begin, &block_high, 1);
        for (std::size_t t = begin; t < end; ++t) {
            const std::size_t a0 = b << kLaneBits;
            b = next_clear(b, bmask);
            const std::size_t a1 = b << kLaneBits;
            b = next_clear(b, bmask);
            Lane8 x, y, r[4], i[4];
            if constexpr (high_clear) {
                lane_load(x, rp + a0);
                lane_load(y, rp + a1);
                deinterleave<S>(x, y, r[x00], r[x10]);
                lane_load(x, ip + a0);
                lane_load(y, ip + a1);
                deinterleave<S>(x, y, i[x00], i[x10]);
            }
            if constexpr (high_set) {
                lane_load(x, rp + a0 + hb);
                lane_load(y, rp + a1 + hb);
                deinterleave<S>(x, y, r[x01], r[x11]);
                lane_load(x, ip + a0 + hb);
                lane_load(y, ip + a1 + hb);
                deinterleave<S>(x, y, i[x01], i[x11]);
            }
            gl(r, i);
            if constexpr (high_clear) {
                interleave<S>(r[x00], r[x10], x, y);
                lane_store(rp + a0, x);
                lane_store(rp + a1, y);
                interleave<S>(i[x00], i[x10], x, y);
                lane_store(ip + a0, x);
                lane_store(ip + a1, y);
            }
            if constexpr (high_set) {
                interleave<S>(r[x01], r[x11], x, y);
                lane_store(rp + a0 + hb, x);
                lane_store(rp + a1 + hb, y);
                interleave<S>(i[x01], i[x11], x, y);
                lane_store(ip + a0 + hb, x);
                lane_store(ip + a1 + hb, y);
            }
        }
    });
}

// Sorts the four blocks at p into one Lane8 per member: first by lane bit Sa
// (the lower qubit), which leaves the higher qubit's lane bit Sb at Sb / 2,
// then by that bit.
template <std::size_t Sa, std::size_t Sb, bool Swapped>
inline void split_low_low(const double* p, Lane8* m) {
    Lane8 x0, x1, x2, x3, c0, d0, c1, d1;
    lane_load(x0, p);
    lane_load(x1, p + kLanes);
    lane_load(x2, p + 2 * kLanes);
    lane_load(x3, p + 3 * kLanes);
    deinterleave<Sa>(x0, x1, c0, d0);
    deinterleave<Sa>(x2, x3, c1, d1);
    deinterleave<Sb / 2>(c0, c1, m[member<Swapped>(0)], m[member<Swapped>(2)]);
    deinterleave<Sb / 2>(d0, d1, m[member<Swapped>(1)], m[member<Swapped>(3)]);
}

template <std::size_t Sa, std::size_t Sb, bool Swapped>
inline void join_low_low(const Lane8* m, double* p) {
    Lane8 x0, x1, x2, x3, c0, d0, c1, d1;
    interleave<Sb / 2>(m[member<Swapped>(0)], m[member<Swapped>(2)], c0, c1);
    interleave<Sb / 2>(m[member<Swapped>(1)], m[member<Swapped>(3)], d0, d1);
    interleave<Sa>(c0, d0, x0, x1);
    interleave<Sa>(c1, d1, x2, x3);
    lane_store(p, x0);
    lane_store(p + kLanes, x1);
    lane_store(p + 2 * kLanes, x2);
    lane_store(p + 3 * kLanes, x3);
}

// Two qubits both below kLaneBits, on lane bits Sa < Sb: every block holds two
// whole groups, and a tile is four consecutive blocks.
template <std::size_t Sa, std::size_t Sb, bool Swapped, class G>
inline void low_low_lanes(double* re, double* im, std::size_t dim, const G& g) {
    static_assert(Sa < Sb, "Sa is the lower qubit's lane bit");
    constexpr std::size_t tile = 4 * kLanes;
    for_each_share(dim / tile, dim, [=, &g](std::size_t begin, std::size_t end) {
        const G gl = g;
        double* __restrict__ rp = re;
        double* __restrict__ ip = im;
        for (std::size_t t = begin; t < end; ++t) {
            const std::size_t a = t * tile;
            Lane8 r[4], i[4];
            split_low_low<Sa, Sb, Swapped>(rp + a, r);
            split_low_low<Sa, Sb, Swapped>(ip + a, i);
            gl(r, i);
            join_low_low<Sa, Sb, Swapped>(r, rp + a);
            join_low_low<Sa, Sb, Swapped>(i, ip + a);
        }
    });
}

template <unsigned Touched, class G>
inline void two_qubit_lanes(double* re, double* im, std::size_t dim, const int* q, const G& g) {
    const int lo = std::min(q[0], q[1]);
    const int hi = std::max(q[0], q[1]);
    const bool swapped = q[0] > q[1];
    if (hi >= kLaneBits) {
        auto run = [&]<std::size_t S>(std::integral_constant<std::size_t, S>) {
            if (swapped) low_high_lanes<S, true, Touched>(re, im, dim, hi, g);
            else         low_high_lanes<S, false, Touched>(re, im, dim, hi, g);
        };
        if (lo == 0)      run(std::integral_constant<std::size_t, 1>{});
        else if (lo == 1) run(std::integral_constant<std::size_t, 2>{});
        else              run(std::integral_constant<std::size_t, 4>{});
        return;
    }
    auto run = [&]<std::size_t Sa, std::size_t Sb>(std::integral_constant<std::size_t, Sa>,
                                                   std::integral_constant<std::size_t, Sb>) {
        if (swapped) low_low_lanes<Sa, Sb, true>(re, im, dim, g);
        else         low_low_lanes<Sa, Sb, false>(re, im, dim, g);
    };
    using c1 = std::integral_constant<std::size_t, 1>;
    using c2 = std::integral_constant<std::size_t, 2>;
    using c4 = std::integral_constant<std::size_t, 4>;
    if (lo == 0 && hi == 1)      run(c1{}, c2{});
    else if (lo == 0)            run(c1{}, c4{});
    else                         run(c2{}, c4{});
}

}  // namespace sv_groups_impl

namespace sv_groups_impl {

// Blocks is the blocks path's callable (g, or a diagonal gate's d); Whole is
// the group function every other path runs.
template <int M, unsigned Touched, bool Diagonal, class Blocks, class Whole>
inline void dispatch(double* re, double* im, std::size_t dim, const int (&q)[M],
                     const Blocks& blocks, const Whole& whole) {
    static_assert(M == 1 || M == 2, "one- and two-qubit gates");
    static_assert(Touched != 0 && Touched < (1u << (1u << M)), "Touched names members");
    if (dim < (kLanes << M)) {
        by_runs<M, Touched>(re, im, dim, q, whole);
        return;
    }
    if (std::min(q[0], q[M - 1]) >= kLaneBits) {
        if constexpr (Diagonal) by_blocks_diagonal<M, Touched>(re, im, dim, q, blocks);
        else                    by_blocks<M, Touched>(re, im, dim, q, blocks);
        return;
    }
    if constexpr (M == 1) {
        if (q[0] == 0)      one_qubit_lanes<1>(re, im, dim, whole);
        else if (q[0] == 1) one_qubit_lanes<2>(re, im, dim, whole);
        else                one_qubit_lanes<4>(re, im, dim, whole);
    } else {
        two_qubit_lanes<Touched>(re, im, dim, q, whole);
    }
}

}  // namespace sv_groups_impl

// re, im = the amplitude arrays, dim of each
// q      = the gate's M qubits in the kernel's member order, all distinct
// g      = the group function; Touched = the members it reads or writes
template <int M, unsigned Touched, class G>
inline void for_each_group(double* re, double* im, std::size_t dim, const int (&q)[M],
                           const G& g) {
    sv_groups_impl::dispatch<M, Touched, false>(re, im, dim, q, g, g);
}

// A diagonal gate: each member is scaled on its own, so the kernel states the
// gate as a member function d(x, r, i), x a std::integral_constant naming the
// member and r, i that member's real and imaginary parts (double or Lane8),
// updated in place. The blocks path sweeps a run one member at a time, which
// streams through half (or a quarter) of the memory regions at once; the other
// paths run d on every touched member of a group.
template <int M, unsigned Touched, class D>
inline void for_each_diagonal(double* re, double* im, std::size_t dim, const int (&q)[M],
                              const D& d) {
    const auto whole = [d](auto* r, auto* i) {
        sv_groups_impl::for_each_member<(std::size_t{1} << M)>([&](auto x) {
            if constexpr (sv_groups_impl::touches<Touched, decltype(x)::value>)
                d(x, r[decltype(x)::value], i[decltype(x)::value]);
        });
    };
    sv_groups_impl::dispatch<M, Touched, true>(re, im, dim, q, d, whole);
}

}  // namespace detail
}  // namespace lindblad
