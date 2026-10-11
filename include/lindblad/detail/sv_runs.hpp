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
// for_each_run - a statevector gate's work, split evenly across threads
// =============================================================================
// A gate on the sorted qubits q_0 < ... < q_{m-1} acts on dim / 2^m groups of
// 2^m amplitudes. Group g's base index is g with a zero bit inserted at every
// q_j (insert_zeros); its members are that base with the target bits set.
// Inserting the zeros moves only the bits of g at or above q_0, so groups
// g, g+1, ... have consecutive base indices until g crosses a multiple of
// 2^(q_0): they come in RUNS of 2^(q_0) groups, each a contiguous stretch of
// memory one SIMD loop can sweep.
//
// for_each_run calls body(base, len) for every run, where base is the base
// index of the first group swept and len how many groups the call sweeps. The
// groups are divided among the threads BY COUNT, in contiguous shares, so
// every thread has the same work whichever qubits the gate touches. Dividing
// by blocks of 2^(q_{m-1}+1) amplitudes instead would leave
// dim / 2^(q_{m-1}+1) blocks, which is 1 for a gate on the top qubit: one
// thread working and the rest idle.
//
// The team is min(hw::available_threads(), dim / parallel_grain()) threads
// (parallel_grain in thread_cap.hpp, DEFAULT_MIN_AMPLITUDES_PER_THREAD unless
// a simulator's option sets it), and a team of one runs serially.
//
// Two shapes, by how many runs there are:
//
//   whole runs  at least kRunsPerThread runs per thread: shares are cut on run
//               boundaries, so every call sweeps one complete run and len is
//               always 2^(q_0). For 2^(q_0) up to kConstRunMax, len is passed
//               as a std::integral_constant, so the body's loop has a trip
//               count the compiler knows: a run of one pair is straight-line
//               code with no loop entry to pay for. That matters most on the
//               lowest qubits, where a gate is millions of one-pair runs.
//               Longer runs take a run-time len, which the compiler
//               vectorises as an ordinary loop.
//   cut runs    fewer runs than that (q_0 is high, so the runs are long): each
//               share is a stretch of kRunAlign-aligned groups and may begin
//               or end inside a run, costing one extra call per thread.
//
// Within a share the next run's base is found from the last index swept, as
// the next index with every target bit clear: ((last | tmask) + 1) & ~tmask.
// Bodies take len as `auto` so both shapes instantiate them.
//
// The per-amplitude arithmetic is the body's alone and each amplitude is
// written by exactly one call, so the division of work never changes a result:
// a kernel computes the same bits on one thread or on many.

#include "lindblad/detail/thread_cap.hpp"
#include "lindblad/hw_info.hpp"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lindblad {
namespace detail {

// Groups per share unit in the cut-runs shape: one 64-byte cache line of
// doubles, so two threads do not write one line at a share boundary.
inline constexpr std::size_t kRunAlign = 8;

// Whole runs per thread needed before shares are cut on run boundaries; with
// fewer, a share of whole runs can be uneven by a whole run.
inline constexpr std::size_t kRunsPerThread = 8;

// The longest run passed with a compile-time length. Runs of 1 and 2 groups
// gain from it: no loop entry per run. Runs of 4 and 8 run faster with a
// run-time length: given those trip counts as constants, the compiler emits
// code for the body that takes about twice as long as its vectorised loop.
inline constexpr std::size_t kConstRunMax = 2;

// k with a zero bit inserted at each position of sorted_q, lowest first, so
// every later position is counted in the index already widened.
inline std::size_t insert_zeros(std::size_t k, const int* sorted_q, int m) noexcept {
    for (int j = 0; j < m; ++j) {
        const int q = sorted_q[j];
        const std::size_t low = k & ((std::size_t{1} << q) - 1);
        k = ((k >> q) << (q + 1)) | low;
    }
    return k;
}

namespace sv_runs_impl {

// Sweeps groups [begin, end) of one share. Len = std::integral_constant (whole
// runs of a known length) or std::size_t (anything else).
template <class Len, class Body>
inline void sweep_share(std::size_t begin, std::size_t end, const int* sorted_q, int m,
                        std::size_t run_cap, std::size_t tmask, Len whole, Body& body) {
    if (begin >= end) return;
    std::size_t base = insert_zeros(begin, sorted_q, m);
    std::size_t g = begin;
    while (g < end) {
        std::size_t len;
        if constexpr (std::is_same_v<Len, std::size_t>) {
            len = std::min(run_cap - (g & (run_cap - 1)), end - g);
            body(base, len);
        } else {
            len = Len::value;
            body(base, whole);
        }
        g += len;
        base = ((base + len - 1) | tmask) + 1;
        base &= ~tmask;
    }
}

}  // namespace sv_runs_impl

// sorted_q = the gate's qubits in ascending order, m of them
// body     = callable as body(std::size_t base, Len len) for every Len above
template <class Body>
inline void for_each_run(std::size_t dim, const int* sorted_q, int m, Body&& body) {
    const std::size_t n_groups = dim >> m;
    const std::size_t run_cap = std::size_t{1} << sorted_q[0];
    const std::size_t n_runs = n_groups / run_cap;
    std::size_t tmask = 0;
    for (int j = 0; j < m; ++j) tmask |= std::size_t{1} << sorted_q[j];

    // Each thread gets at least parallel_grain() amplitudes of the state, and
    // a sweep with only one thread's worth starts no region at all.
    const std::size_t team = std::min(hw::available_threads(), dim / parallel_grain());
    const int team_size = static_cast<int>(std::max<std::size_t>(team, 1));

    #pragma omp parallel num_threads(team_size) if(team_size > 1)
    {
        std::size_t threads = 1;
        std::size_t t = 0;
#ifdef _OPENMP
        threads = static_cast<std::size_t>(omp_get_num_threads());
        t = static_cast<std::size_t>(omp_get_thread_num());
#endif
        const bool whole = n_runs >= kRunsPerThread * threads;
        const std::size_t unit = whole ? run_cap
                               : (n_groups % kRunAlign == 0 ? kRunAlign : 1);
        const std::size_t n_units = n_groups / unit;
        const std::size_t share = n_units / threads;
        const std::size_t extra = n_units % threads;
        const std::size_t begin = (t * share + std::min(t, extra)) * unit;
        const std::size_t end = begin + (share + (t < extra ? 1 : 0)) * unit;

        using sv_runs_impl::sweep_share;
        if (whole && run_cap <= kConstRunMax) {
            if (run_cap == 1)
                sweep_share(begin, end, sorted_q, m, run_cap, tmask,
                            std::integral_constant<std::size_t, 1>{}, body);
            else
                sweep_share(begin, end, sorted_q, m, run_cap, tmask,
                            std::integral_constant<std::size_t, 2>{}, body);
        } else {
            sweep_share(begin, end, sorted_q, m, run_cap, tmask, std::size_t{0}, body);
        }
    }
}

} // namespace detail
} // namespace lindblad
