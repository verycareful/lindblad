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
// shot_model - the shot team that costs the least on this machine
// =============================================================================
// The automatic shot team is the k, from 1 to the most threads the run may
// use, that minimises the run's time:
//
//   k = 1    work(1)
//   k >= 2   region(k) + work(k) * imbalance(k) + merge
//
//   region(k)     the cost of starting a team of k (ThreadingCosts::
//                 region_seconds, measured once per process; the cold figure
//                 when the costs hold one, since a run starts after whatever
//                 came before it).
//   work(k)       the run's work, in one-qubit-gate amplitudes (a backend's
//                 work_per_shot times the shots), at one thread's time per
//                 amplitude (ThreadingCosts::in_cache.streaming at team 1),
//                 divided by what k threads gain over one. For shots that
//                 evolve a state each (trajectories) the gain is that of
//                 independent work, measured once per process: every thread
//                 runs its own compute loop on its own data with no barrier
//                 inside (team_speedups). A gate pass that synchronises its
//                 team would read as far less at 24 threads (the barrier
//                 dominates) than such threads gain from their SMT siblings.
//                 For shots drawn from a held distribution (sampling) the
//                 threads stream shared tables, and the gain is the streaming
//                 pass's (ThreadingCosts::in_cache.streaming at team k), which
//                 stops growing where memory does. Costs the caller set carry
//                 no independent figure and always use the streaming one.
//   imbalance(k)  ceil(chunks / k) * k / chunks: chunks are whole, so the
//                 last round of a team that does not divide them leaves
//                 threads idle.
//   merge         the in-order merge of the chunks' counts, which a run on one
//                 thread does not pay: the keys the chunks hold (at most the
//                 shots of a chunk, and at most 2^key_bits) times the time one
//                 key takes to merge (merge_seconds_per_key, measured once per
//                 process).
//
// A figure the costs lack leaves the answer to the grain rule (0 is returned).

#include "lindblad/detail/shot_loop.hpp"
#include "lindblad/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lindblad {
namespace detail {

struct ShotModelInput {
    std::uint64_t work_per_shot = 0;  // one-qubit-gate amplitudes
    ShotChunks chunks;
    int key_bits = 0;                 // the counts' key width in clbits; 0 = not known
    int max_team = 1;                 // the most threads the run may use
    bool sampling = false;            // shots drawn from a held distribution; see ShotTeamRequest
};

// The seconds one distinct key takes to merge into the run's counts, measured
// on this machine the first time it is asked and kept for the process.
double merge_seconds_per_key();

// What a team of each of `sizes` gains over one thread on independent compute
// work: one entry per size, 1 for a team of 1. Measured now, on the threads the
// calling thread may start; the caller keeps the result. Empty when OpenMP is
// not available.
std::vector<double> team_speedups(const std::vector<int>& sizes);

// The team (1 to max_team) of least modelled time for `costs`, or 0 when the
// costs hold no figure for it. `merge_seconds_per_key` is the per-key merge
// time: 0 prices the merge as free (what a run on costs set by its caller
// gets, since nothing of it is measured).
// `speedups` is team_speedups(costs.team_sizes), or empty to read the team's
// gain from costs.in_cache.streaming.
int best_shot_team(const ThreadingCosts& costs, double merge_seconds_per_key,
                   const std::vector<double>& speedups, const ShotModelInput& in);

// best_shot_team over `pinned` when it is set, else over the costs of this
// machine (region and in-cache figures, measured once per process) and the
// measured merge time.
int modelled_shot_team(const ShotModelInput& in, const ThreadingCosts* pinned);

}  // namespace detail
}  // namespace lindblad
