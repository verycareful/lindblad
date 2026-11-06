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
// shot_loop - running a run's independent shots across a team of threads
// =============================================================================
// A run that walks its shots one by one (mid-circuit measurement, feedforward,
// a RESET, or a backend that samples shot by shot) gives the same answer
// whatever thread runs a shot, provided every shot draws from the same random
// numbers. This header makes that so:
//
//   1. The shots are cut into chunks that depend on the shot count only
//      (chunk_shots; the rule is in types.hpp beside SHOT_CHUNK_TARGET).
//   2. Chunk c draws from its own stream, seeded from the run's seed, a salt
//      and c (chunk_stream). A thread always runs a whole chunk.
//   3. Chunks are handed out in ascending order through one counter, and their
//      counts are merged in chunk order (OrderedCounts).
//   4. A failing shot is reduced to the lowest-numbered one (ChunkFailures):
//      every chunk below it still runs, every chunk above it is skipped or
//      discarded, so the record is the one a serial run would have left.
//
// A team of one runs the same chunks with the same streams on the calling
// thread, so a serial run and a parallel one agree bit for bit.
//
// Nothing leaves the parallel region by exception: every throw inside it is
// caught and handed to ChunkFailures, and the caller rethrows after the region.

#include "lindblad/failed_run.hpp"
#include "lindblad/shot_rng.hpp"
#include "lindblad/types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lindblad {

struct Instruction;

namespace detail {

// =============================================================================
// ShotChunks - how a run's shots are cut
// =============================================================================
// shots >= 1. A run of 0 shots walks one trajectory and is described as one
// chunk of one shot.
struct ShotChunks {
    int shots = 1;  // shots the chunks cover
    int chunk = 1;  // shots per chunk, except possibly the last
    int count = 1;  // number of chunks

    int first(int c) const noexcept { return c * chunk; }
    int size(int c) const noexcept {
        const int begin = first(c);
        return shots - begin < chunk ? shots - begin : chunk;
    }
};

// The chunks of a run of `shots` shots. Depends on `shots` only.
ShotChunks chunk_shots(int shots) noexcept;

// The random stream of chunk c. salt separates runs that share a seed and must
// still differ (the statevector simulator passes the calling thread's index,
// so concurrent runs with one seed draw independently); other backends pass 0.
// Work done once for every shot (a shared prefix) draws from prefix_stream,
// never from a chunk's stream.
ShotRng chunk_stream(std::uint64_t base_seed, std::uint32_t salt, int c);
ShotRng prefix_stream(std::uint64_t base_seed, std::uint32_t salt);

// =============================================================================
// shot_team - how many threads run a run's shots
// =============================================================================
// What a backend knows when it sizes the team. Memory is in bytes.
struct ShotTeamRequest {
    const char* entry_point = "";
    int max_parallel_shots = 0;          // the option: 0 automatic, 1 serial, k >= 2
    ShotMemory memory = ShotMemory::Shrink;
    std::uint64_t cap_bytes = 0;         // the run's resolved memory cap
    std::uint64_t held_bytes = 0;        // what the run holds with one working state
    std::uint64_t per_thread_bytes = 0;  // one more thread's working state and scratch
    std::size_t work_per_shot = 0;       // one shot's work, in one-qubit-gate amplitudes
    bool serial_only = false;            // the backend needs one thread for this run
};

struct ShotTeam {
    int threads = 1;               // >= 1, never more than the chunks
    std::uint64_t share_bytes = 0;  // ShotMemory::FixedShare: each thread's cap; else 0
};

// The team for a run, in this order:
//   - max_parallel_shots < 0, or ShotMemory::Exact without a team named, is
//     refused (InvalidArgument).
//   - serial_only, a single chunk, max_parallel_shots == 1, or a call from
//     inside a parallel region run on one thread. Exact asking for k >= 2
//     inside a parallel region is refused, since it cannot be honoured.
//   - The team wanted is max_parallel_shots when it is >= 2. Automatic runs on
//     one thread unless SHOT_TEAM_MIN_CACHED_STATES per-thread states fit one
//     last-level-cache instance (types.hpp), and is then
//     min(available threads, shots * work_per_shot / parallel_grain()), the
//     statevector kernels' rule applied to the whole run's work.
//   - Never more threads than chunks.
//   - The memory policy then sizes it against room = cap - held (see
//     ShotMemory in types.hpp).
ShotTeam shot_team(const ShotTeamRequest& req, const ShotChunks& chunks);

// =============================================================================
// OrderedCounts - chunk counts merged in chunk order
// =============================================================================
// Thread-safe. A chunk's counts are held until every chunk below it has been
// committed, then merged, so the merged map is always the counts of a
// contiguous run of chunks from chunk 0. After a failure in chunk f that is
// exactly chunks 0 .. f-1: every chunk below f finishes, and nothing above f
// is merged past the gap f leaves.
class OrderedCounts {
public:
    using Counts = std::unordered_map<std::string, int>;

    explicit OrderedCounts(int chunk_count);

    void commit(int c, Counts&& counts);
    // Every chunk at once, from a team of one that kept a single map.
    void commit_all(Counts&& counts);
    // The merged counts. Leaves this empty.
    Counts take();
    // Shots in the chunks merged so far.
    int shots_committed(const ShotChunks& chunks) const;

private:
    mutable std::mutex mutex_;
    std::vector<std::optional<Counts>> pending_;
    Counts merged_;
    int next_ = 0;  // the lowest chunk not yet merged
};

// =============================================================================
// ShotFailure / ChunkFailures - the lowest failing shot of a run
// =============================================================================
// What a worker hands over when one of its shots throws.
struct ShotFailure {
    int shot = -1;                     // the failing shot; -1 while none has failed
    int instruction = -1;              // the instruction it had reached, or -1
    const Instruction* inst = nullptr;
    bool started = false;              // its worker had reached an instruction
    std::exception_ptr error;
    FailedRun::State state;            // the failing shot's working state, if any
    std::unordered_map<std::string, int> partial;  // its chunk's shots before it
};

// Thread-safe. Keeps the failure with the lowest shot.
class ChunkFailures {
public:
    explicit ChunkFailures(int chunk_count) : limit_(chunk_count) {}

    // Keeps f when no failure is held or f.shot is lower than the one held.
    void offer(int c, ShotFailure&& f);
    // Chunks at or above this are not run: the chunk of the lowest failure so
    // far, or the chunk count when there is none.
    int limit() const noexcept { return limit_.load(std::memory_order_acquire); }
    std::optional<ShotFailure> take();

private:
    std::mutex mutex_;
    std::optional<ShotFailure> held_;
    std::atomic<int> limit_;
};

// =============================================================================
// run_shot_chunks - the driver
// =============================================================================
// Runs every chunk once, on team.threads threads (a team of one runs on the
// calling thread with no parallel region). make_worker(thread_index) is called
// once per thread, on that thread, and returns a Worker providing
//
//   void run_chunk(int c, int first_shot, int n, ShotRng& rng,
//                  std::unordered_map<std::string, int>& counts,
//                  ShotFailure& failure_out);
//
// run_chunk runs shots [first_shot, first_shot + n) drawing from rng and adds
// each shot's key to counts. When a shot throws, it fills failure_out (shot set,
// error captured, counts so far moved into partial) and returns; it never lets
// an exception out. A worker whose chunk failed takes no further chunk: chunks
// are handed out in ascending order, so every later one is above the failure.
//
// A throw from make_worker, or anything else that escapes, fails the chunk the
// thread holds (or the next it would take) at that chunk's first shot.
//
// A team of one keeps one counts map across all its chunks and hands it to
// `counts` at the end: a serial run pays no merge per chunk, and on a failure
// the failing chunk's partial is then every shot before the failing one.
//
// Returns the index of the thread that ran the last chunk: its working state is
// the run's final state.
template <class MakeWorker>
int run_shot_chunks(const ShotChunks& chunks, const ShotTeam& team,
                    std::uint64_t base_seed, std::uint32_t salt,
                    OrderedCounts& counts, ChunkFailures& failures,
                    MakeWorker&& make_worker) {
    std::atomic<int> next{0};
    std::atomic<int> owner{0};

    const auto fail_at = [&](int c) noexcept {
        if (c >= chunks.count) return;
        ShotFailure f;
        f.shot = chunks.first(c);
        f.error = std::current_exception();
        failures.offer(c, std::move(f));
    };

    const bool serial = team.threads <= 1;

    const auto run_thread = [&](int t) noexcept {
        int held = -1;  // the chunk this thread is working on, for fail_at
        try {
            auto worker = make_worker(t);
            std::unordered_map<std::string, int> chunk_counts;
            for (int c = next.fetch_add(1, std::memory_order_relaxed); c < chunks.count;
                 c = next.fetch_add(1, std::memory_order_relaxed)) {
                if (c >= failures.limit()) break;
                held = c;
                ShotFailure failure;
                ShotRng rng = chunk_stream(base_seed, salt, c);
                worker.run_chunk(c, chunks.first(c), chunks.size(c), rng, chunk_counts,
                                 failure);
                if (failure.shot >= 0) {
                    failures.offer(c, std::move(failure));
                    return;
                }
                if (c == chunks.count - 1) owner.store(t, std::memory_order_relaxed);
                if (!serial) {
                    counts.commit(c, std::move(chunk_counts));
                    chunk_counts.clear();
                }
                held = -1;
            }
            if (serial) counts.commit_all(std::move(chunk_counts));
        } catch (...) {
            fail_at(held >= 0 ? held : next.fetch_add(1, std::memory_order_relaxed));
        }
    };

    if (serial) {
        run_thread(0);
    } else {
        #pragma omp parallel num_threads(team.threads)
        {
            int t = 0;
#ifdef _OPENMP
            t = omp_get_thread_num();
#endif
            run_thread(t);
        }
    }
    return owner.load(std::memory_order_relaxed);
}

}  // namespace detail
}  // namespace lindblad
