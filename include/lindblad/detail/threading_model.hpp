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
// threading_model - measuring what parallelism costs on this machine
// =============================================================================
// ThreadingCosts (types.hpp) holds the numbers; this header measures them and
// checks a caller's own. A measurement is made on demand, one part at a time,
// and kept for the rest of the process for each (threads the process may use,
// processors its team may be placed on): the numbers describe that set and no
// other. The parts differ in what they cost to measure, which is why a run
// asks only for those it needs:
//
//   Region        the cost of a warm parallel region at each team size: a few
//                 milliseconds.
//   InCache       the streaming and diagonal passes timed with each
//                 thread's share in its core's cache: about ten milliseconds.
//   Dense         the dense k-qubit blocks, k = 1 to 6, timed the same way:
//                 about a hundred milliseconds, since a block's arithmetic
//                 grows with 2^k.
//   Shared        the streaming and diagonal passes with the target changing
//                 from pass to pass, as a circuit's gates change it: about
//                 ten milliseconds, since it times what InCache times with
//                 the targets cycling. It prices amplitudes moving between
//                 cores, and between L3 instances once a team spans two.
//   MemoryBound   the passes over a scratch state larger than every L3
//                 instance together: it allocates and touches that state, so
//                 it costs as much as writing it once on top of the timing,
//                 and it needs that much memory to spare (see the room
//                 argument of measure_threading_costs).
//   ColdRegions   a region after the team has slept: each sample waits out the
//                 block time (plus a margin), so it costs team sizes x samples
//                 x block time. Quick under a passive wait policy, seconds
//                 under libomp's default block time of 200 ms, and impossible
//                 when the block time is unknown or infinite.
//
// The wait policy of the process is part of what is measured, and is recorded
// with it (ThreadingCosts::block_time_ms): a warm region is one that follows
// another within the block time, so under a passive policy (block time 0) a
// "warm" region is the cold one and the record says so.

#include "lindblad/types.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace lindblad {
namespace detail {

enum class ThreadingParts : unsigned {
    Region = 1,       // also reads the block time
    InCache = 2,
    Dense = 4,
    MemoryBound = 8,
    ColdRegions = 16,
    Shared = 32,
    All = 47,         // every part but Cold
};

constexpr ThreadingParts operator|(ThreadingParts a, ThreadingParts b) {
    return static_cast<ThreadingParts>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

// The team sizes a measurement samples for the calling thread: 1, 2 and 4,
// the cores of one L3 instance and its threads, the physical cores, and every
// thread the thread may start, each only if it is at most the threads
// available, sorted and distinct. Where the topology is unknown the machine is
// one domain of every thread available, and the sizes are 1, 2, 4 and that.
std::vector<int> threading_team_sizes();

// The bytes MemoryBound's scratch state takes, or 0 when it cannot be measured
// on this machine (no L3 size detected).
std::uint64_t threading_memory_scratch_bytes();

// The costs of `parts` for the calling thread's reach, measured now unless
// this process already measured them (a second call asks for nothing it holds).
// Parts not asked for, and not measured before, are empty. Measuring holds a
// lock, so a second thread asking waits and then reads the first one's result
// instead of timing beside it.
//
// `scratch_room_bytes` is the memory the caller can spare for MemoryBound's
// scratch state, on top of what it holds already (a run passes what is left of
// its memory cap after its own buffers). When the scratch state does not fit in
// it, MemoryBound is left empty and ThreadingCosts::memory_bound_note says so;
// a later call with more room measures it then. Under no cap, leave it at its
// default.
inline constexpr std::uint64_t kNoScratchLimit = std::numeric_limits<std::uint64_t>::max();
ThreadingCosts measure_threading_costs(ThreadingParts parts,
                                       std::uint64_t scratch_room_bytes = kNoScratchLimit);

// Whether measuring ColdRegions takes little time for the calling thread's
// block time: a known, finite block time for which every team size's samples
// together wait no longer than a tenth of a second. A passive wait policy
// qualifies; libomp's default block time of 200 ms does not.
bool threading_cold_regions_quick();

// Why `costs` is not a usable set of ThreadingCosts, or an empty string when it
// is: the team sizes ascending from 1, every vector one entry per team size,
// every entry finite and positive, `memory_bound` either whole or empty.
std::string threading_costs_problem(const ThreadingCosts& costs);

}  // namespace detail
}  // namespace lindblad
