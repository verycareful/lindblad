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
// Lane8 - the eight amplitudes of one cache line, as one value
// =============================================================================
// A block is kLanes = 8 consecutive amplitudes of one array: one 64-byte cache
// line of doubles. Lane8 holds a block, and arithmetic on it acts lane by lane,
// so a formula written for a double is the same formula for a Lane8: a * x + y
// with a a double and x, y Lane8 multiplies and adds every lane.
//
// Under GCC and Clang, Lane8 is a vector extension type. The compiler lowers it
// to whatever the target has: one AVX-512 register, two AVX registers, four SSE
// or NEON registers, and the lane shuffles below to that target's permutes, so
// the code is the same for every target and none of them is assumed. Elsewhere
// Lane8 is a plain array of eight doubles with lane-by-lane operators, which
// computes the same values.
//
// Lane8 values never cross a function boundary by value (loads, stores and
// shuffles take references), because the calling convention for a 64-byte
// vector differs between targets with AVX-512 and without, and compilers warn
// about every such signature.
//
// deinterleave<S> and interleave<S> sort lanes by one lane bit S (1, 2 or 4):
//
//   deinterleave<S>(x, y, c, d)   c = the lanes of x, then of y, whose bit S
//                                 is clear, in lane order; d = those whose bit
//                                 S is set. Lane j of c and lane j of d are
//                                 partners: they differ in bit S only.
//   interleave<S>(c, d, x, y)     the inverse: puts every lane back.
//
// Example, S = 1: x = (x0 .. x7), y = (y0 .. y7) give
//   c = (x0 x2 x4 x6 y0 y2 y4 y6),  d = (x1 x3 x5 x7 y1 y3 y5 y7).
//
// In c and d, lane bit 4 says which of x and y a lane came from, and lane bits
// 1 and 2 are the source lane's other two bits, in order: a lane bit T > S of
// the source is lane bit T / 2 of c and d.

#include <cstddef>
#include <cstring>

namespace lindblad {
namespace detail {

inline constexpr int kLaneBits = 3;
inline constexpr std::size_t kLanes = std::size_t{1} << kLaneBits;

#if defined(__GNUC__) || defined(__clang__)

typedef double Lane8 __attribute__((vector_size(kLanes * sizeof(double))));

template <std::size_t S>
inline void deinterleave(const Lane8& x, const Lane8& y, Lane8& c, Lane8& d) {
    static_assert(S == 1 || S == 2 || S == 4, "S is one lane bit");
    if constexpr (S == 1) {
        c = __builtin_shufflevector(x, y, 0, 2, 4, 6, 8, 10, 12, 14);
        d = __builtin_shufflevector(x, y, 1, 3, 5, 7, 9, 11, 13, 15);
    } else if constexpr (S == 2) {
        c = __builtin_shufflevector(x, y, 0, 1, 4, 5, 8, 9, 12, 13);
        d = __builtin_shufflevector(x, y, 2, 3, 6, 7, 10, 11, 14, 15);
    } else {
        c = __builtin_shufflevector(x, y, 0, 1, 2, 3, 8, 9, 10, 11);
        d = __builtin_shufflevector(x, y, 4, 5, 6, 7, 12, 13, 14, 15);
    }
}

template <std::size_t S>
inline void interleave(const Lane8& c, const Lane8& d, Lane8& x, Lane8& y) {
    static_assert(S == 1 || S == 2 || S == 4, "S is one lane bit");
    if constexpr (S == 1) {
        x = __builtin_shufflevector(c, d, 0, 8, 1, 9, 2, 10, 3, 11);
        y = __builtin_shufflevector(c, d, 4, 12, 5, 13, 6, 14, 7, 15);
    } else if constexpr (S == 2) {
        x = __builtin_shufflevector(c, d, 0, 1, 8, 9, 2, 3, 10, 11);
        y = __builtin_shufflevector(c, d, 4, 5, 12, 13, 6, 7, 14, 15);
    } else {
        x = __builtin_shufflevector(c, d, 0, 1, 2, 3, 8, 9, 10, 11);
        y = __builtin_shufflevector(c, d, 4, 5, 6, 7, 12, 13, 14, 15);
    }
}

#else

struct Lane8 {
    double v[kLanes];
};

inline Lane8 operator+(const Lane8& a, const Lane8& b) {
    Lane8 o;
    for (std::size_t k = 0; k < kLanes; ++k) o.v[k] = a.v[k] + b.v[k];
    return o;
}
inline Lane8 operator-(const Lane8& a, const Lane8& b) {
    Lane8 o;
    for (std::size_t k = 0; k < kLanes; ++k) o.v[k] = a.v[k] - b.v[k];
    return o;
}
inline Lane8 operator*(const Lane8& a, const Lane8& b) {
    Lane8 o;
    for (std::size_t k = 0; k < kLanes; ++k) o.v[k] = a.v[k] * b.v[k];
    return o;
}
inline Lane8 operator-(const Lane8& a) {
    Lane8 o;
    for (std::size_t k = 0; k < kLanes; ++k) o.v[k] = -a.v[k];
    return o;
}
inline Lane8 operator*(double s, const Lane8& a) {
    Lane8 o;
    for (std::size_t k = 0; k < kLanes; ++k) o.v[k] = s * a.v[k];
    return o;
}
inline Lane8 operator*(const Lane8& a, double s) {
    Lane8 o;
    for (std::size_t k = 0; k < kLanes; ++k) o.v[k] = a.v[k] * s;
    return o;
}

// The source lane at position `rank` among the lanes whose bit S is `set`:
// rank with a bit inserted at S.
template <std::size_t S>
inline constexpr std::size_t lane_of_rank(std::size_t rank, bool set) {
    return ((rank & ~(S - 1)) << 1) | (rank & (S - 1)) | (set ? S : 0);
}

template <std::size_t S>
inline void deinterleave(const Lane8& x, const Lane8& y, Lane8& c, Lane8& d) {
    static_assert(S == 1 || S == 2 || S == 4, "S is one lane bit");
    constexpr std::size_t half = kLanes / 2;
    Lane8 oc, od;
    for (std::size_t r = 0; r < half; ++r) {
        oc.v[r] = x.v[lane_of_rank<S>(r, false)];
        od.v[r] = x.v[lane_of_rank<S>(r, true)];
        oc.v[half + r] = y.v[lane_of_rank<S>(r, false)];
        od.v[half + r] = y.v[lane_of_rank<S>(r, true)];
    }
    c = oc;
    d = od;
}

template <std::size_t S>
inline void interleave(const Lane8& c, const Lane8& d, Lane8& x, Lane8& y) {
    static_assert(S == 1 || S == 2 || S == 4, "S is one lane bit");
    constexpr std::size_t half = kLanes / 2;
    Lane8 ox, oy;
    for (std::size_t r = 0; r < half; ++r) {
        ox.v[lane_of_rank<S>(r, false)] = c.v[r];
        ox.v[lane_of_rank<S>(r, true)] = d.v[r];
        oy.v[lane_of_rank<S>(r, false)] = c.v[half + r];
        oy.v[lane_of_rank<S>(r, true)] = d.v[half + r];
    }
    x = ox;
    y = oy;
}

#endif

// The block at p, unaligned.
inline void lane_load(Lane8& v, const double* p) noexcept { std::memcpy(&v, p, sizeof v); }
inline void lane_store(double* p, const Lane8& v) noexcept { std::memcpy(p, &v, sizeof v); }

}  // namespace detail
}  // namespace lindblad
