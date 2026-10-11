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
// The team is sv_team(dim) threads: min(hw::available_threads(),
// dim / parallel_grain()) (parallel_grain in thread_cap.hpp,
// DEFAULT_MIN_AMPLITUDES_PER_THREAD unless a simulator's option sets it), and a
// team of one runs serially.
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
#include <memory>
#include <new>
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

// The threads for a statevector pass worth `work` amplitudes: each thread gets
// at least parallel_grain() of them, the team never exceeds the threads a
// region may use, and a pass with only one thread's worth runs serially. Work
// is counted in one-qubit-gate amplitudes: a pass doing more per amplitude than
// a one-qubit gate passes more than its amplitude count and reaches a team
// sooner, one doing less (streaming_work) passes fewer and reaches it later.
inline int sv_team(std::size_t work) {
    const std::size_t team = std::min(hw::available_threads(), work / parallel_grain());
    return static_cast<int>(std::max<std::size_t>(team, 1));
}

// The work of a streaming pass over `amplitudes` amplitudes (a fill, a scale,
// a sum, a collapse), in one-qubit-gate amplitudes: half. Such a pass touches
// each amplitude once with one or two operations, where a gate reads and
// writes it through a 2x2 multiply, so its threads pay later: measured at
// 2^12 amplitudes, a team of two loses to one thread on every streaming pass
// and wins on a one-qubit gate.
inline std::size_t streaming_work(std::size_t amplitudes) noexcept { return amplitudes / 2; }

namespace sv_runs_impl {

// Share t of `threads` over n_units units of `unit` items: contiguous, in
// thread order, sizes differing by at most one unit.
struct Share {
    std::size_t begin;
    std::size_t end;
};

inline Share share_of(std::size_t n_units, std::size_t unit, std::size_t threads,
                      std::size_t t) noexcept {
    const std::size_t share = n_units / threads;
    const std::size_t extra = n_units % threads;
    const std::size_t begin = (t * share + std::min(t, extra)) * unit;
    return {begin, begin + (share + (t < extra ? 1 : 0)) * unit};
}

// Items per share unit: kRunAlign when it divides the count, so shares of an
// amplitude array begin on a cache line, else 1.
inline std::size_t share_unit(std::size_t count) noexcept {
    return count % kRunAlign == 0 ? kRunAlign : 1;
}

}  // namespace sv_runs_impl

// =============================================================================
// for_each_share / reduce_shares - a flat pass split across a team
// =============================================================================
// For passes with no gate structure: a fill, a scale, a sum over the state.
// The items [0, count) are cut into one contiguous share per thread of a team
// of sv_team(work), never more threads than share units.
//
// for_each_share calls body(begin, end) once per share. reduce_shares has each
// share return its partial, body(begin, end) -> T, and folds the partials with
// combine(acc, partial) starting from init, in THREAD ORDER after the region.
// The fold order is fixed by the code, not by which thread finished first, so a
// sum depends on the team size but is the same on every run with that team: an
// OpenMP reduction clause leaves the order to the runtime, and libomp combines
// small teams atomically, in arrival order. A team of one returns
// combine(init, body(0, count)). T must be default-constructible.
template <class Body>
inline void for_each_share(std::size_t count, std::size_t work, Body&& body) {
    if (count == 0) return;
    const std::size_t unit = sv_runs_impl::share_unit(count);
    const std::size_t n_units = count / unit;
    const int team = static_cast<int>(
        std::min<std::size_t>(static_cast<std::size_t>(sv_team(work)), n_units));
    if (team <= 1) {
        body(std::size_t{0}, count);
        return;
    }
    #pragma omp parallel num_threads(team)
    {
        std::size_t threads = 1;
        std::size_t t = 0;
#ifdef _OPENMP
        threads = static_cast<std::size_t>(omp_get_num_threads());
        t = static_cast<std::size_t>(omp_get_thread_num());
#endif
        const sv_runs_impl::Share s = sv_runs_impl::share_of(n_units, unit, threads, t);
        if (s.begin < s.end) body(s.begin, s.end);
    }
}

template <class T, class Body, class Combine>
inline T reduce_shares(std::size_t count, std::size_t work, T init, Body&& body,
                       Combine&& combine) {
    if (count == 0) return init;
    const std::size_t unit = sv_runs_impl::share_unit(count);
    const std::size_t n_units = count / unit;
    const int team = static_cast<int>(
        std::min<std::size_t>(static_cast<std::size_t>(sv_team(work)), n_units));
    // The partials are allocated without throwing, so a pass that cannot get
    // them runs serially and reduce_shares adds no exception of its own (the
    // finite check that calls it is noexcept).
    std::unique_ptr<T[]> partial;
    if (team > 1) partial.reset(new (std::nothrow) T[static_cast<std::size_t>(team)]);
    if (!partial) return combine(init, body(std::size_t{0}, count));
    for (int t = 0; t < team; ++t) partial[static_cast<std::size_t>(t)] = init;
    std::size_t used = 1;  // threads the region actually ran with
    #pragma omp parallel num_threads(team)
    {
        std::size_t threads = 1;
        std::size_t t = 0;
#ifdef _OPENMP
        threads = static_cast<std::size_t>(omp_get_num_threads());
        t = static_cast<std::size_t>(omp_get_thread_num());
#endif
        if (t == 0) used = threads;
        const sv_runs_impl::Share s = sv_runs_impl::share_of(n_units, unit, threads, t);
        if (s.begin < s.end) partial[t] = body(s.begin, s.end);
    }
    T acc = init;
    for (std::size_t t = 0; t < used; ++t) acc = combine(acc, partial[t]);
    return acc;
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

    const int team_size = sv_team(dim);

    #pragma omp parallel num_threads(team_size) if(team_size > 1)
    {
        std::size_t threads = 1;
        std::size_t t = 0;
#ifdef _OPENMP
        threads = static_cast<std::size_t>(omp_get_num_threads());
        t = static_cast<std::size_t>(omp_get_thread_num());
#endif
        const bool whole = n_runs >= kRunsPerThread * threads;
        const std::size_t unit = whole ? run_cap : sv_runs_impl::share_unit(n_groups);
        const sv_runs_impl::Share s =
            sv_runs_impl::share_of(n_groups / unit, unit, threads, t);
        const std::size_t begin = s.begin;
        const std::size_t end = s.end;

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
