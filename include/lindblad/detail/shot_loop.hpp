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
#include <string_view>
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

// Refuses (InvalidArgument, naming entry_point) a shot_counting_distinct_
// fraction that is not 0 and not in (0, 1], NaN included. Called before the
// run's first gate.
void check_shot_counting(const char* entry_point, double distinct_fraction);

// =============================================================================
// ShotCounts - the outcome counts of a chunk or a run
// =============================================================================
// A run's keys all have one width: its classical register, or the qubit count
// when the register is sampled from the end state. Clbit c is bit c of the key
// and the c-th character from the right of its string.
//
// A key of up to PACKED_MAX_WIDTH bits can be counted under its packed
// integer, so a repeated outcome costs one integer hash and no string; a wider
// key is always counted under its string. to_strings() puts the counts under
// their strings, one per distinct outcome; when and on which thread that
// happens is the run's ShotCounting (types.hpp), applied by OrderedCounts.
//
// merge() moves the other map's entries in as they are (node handles), so a
// key new to this map costs no allocation; only keys both maps hold are added.
class ShotCounts {
public:
    using Strings = std::unordered_map<std::string, int>;
    static constexpr int PACKED_MAX_WIDTH = 64;

    // strings = count under strings from the first shot.
    explicit ShotCounts(int width = 0, bool strings = false) noexcept
        : width_(width), packed_mode_(!strings && width <= PACKED_MAX_WIDTH) {}

    int width() const noexcept { return width_; }
    bool under_strings() const noexcept { return !packed_mode_; }
    bool empty() const noexcept { return packed_mode_ ? packed_.empty() : strings_.empty(); }
    // Distinct outcomes counted.
    std::size_t size() const noexcept {
        return packed_mode_ ? packed_.size() : strings_.size();
    }

    // One shot whose clbit c reads reg[c] (nonzero is 1). reg.size() == width().
    void add(const std::vector<int>& reg);
    // One shot whose key is the integer `outcome`, bit c being clbit c. Bits at
    // or above width() are ignored. For a key wider than 64 bits the clbits at
    // and above 64 read 0.
    void add_value(std::uint64_t outcome);

    // Puts every count under its string. Later shots are counted under their
    // strings too.
    void to_strings();

    // Adds every count of `other` and leaves `other` empty. `other` has this
    // width or is empty; any other width is a defect, reported as an
    // InternalError naming entry_point.
    void merge(ShotCounts&& other, std::string_view entry_point);

    // The counts under their strings. Leaves this empty.
    Strings strings();

private:
    int width_ = 0;
    bool packed_mode_ = true;                        // counting under packed_
    std::unordered_map<std::uint64_t, int> packed_;
    Strings strings_;
};

// =============================================================================
// OrderedCounts - chunk counts merged in chunk order
// =============================================================================
// Thread-safe. A chunk's counts are held until every chunk below it has been
// committed, then merged, so the merged map is always the counts of a
// contiguous run of chunks from chunk 0. After a failure in chunk f that is
// exactly chunks 0 .. f-1: every chunk below f finishes, and nothing above f
// is merged past the gap f leaves.
//
// The merge runs under the lock, one chunk after another, so its cost per
// chunk bounds the speedup of a run whose shots are cheap: it moves entries
// and adds the counts of repeated keys, and builds a string only for a packed
// chunk merged into counts already under strings. The run's ShotCounting
// decides, per chunk, which keys a worker counts under and whether it puts
// them under strings before committing (begin_strings, end_chunk).
class OrderedCounts {
public:
    // entry_point = the run's public name, for an InternalError; width = the
    // run's key width (see ShotCounts); mode and distinct_fraction = the run's
    // Options::shot_counting and shot_counting_distinct_fraction, already
    // checked (check_shot_counting); a fraction of 0 is
    // SHOT_COUNTING_DISTINCT_FRACTION.
    OrderedCounts(const char* entry_point, int chunk_count, int width,
                  ShotCounting mode, double distinct_fraction);

    // Whether a worker counts its first chunk under strings.
    bool begin_strings() const noexcept { return mode_ == ShotCounting::Strings; }
    // An empty map of the run's width, counting under strings or not.
    ShotCounts fresh(bool strings) const noexcept { return ShotCounts(width_, strings); }

    // Ends a worker's chunk of `shots` shots, which added `gained` distinct
    // outcomes to `counts`, and returns whether its next chunk is counted under
    // strings. team = the run has more than one shot thread. Puts `counts`
    // under strings, on the calling thread, when the mode says:
    // PackedPerChunk on a team; Auto when gained > distinct_fraction * shots.
    bool end_chunk(ShotCounts& counts, int shots, std::size_t gained, bool team) const;

    void commit(int c, ShotCounts&& counts);
    // Every chunk at once, from a team of one that kept a single map.
    void commit_all(ShotCounts&& counts);
    // The merged counts with `tail` added (a failing chunk's shots before the
    // failure; empty when no shot failed), under their strings. Leaves this
    // empty.
    ShotCounts::Strings take(ShotCounts&& tail = ShotCounts());
    // Shots in the chunks merged so far.
    int shots_committed(const ShotChunks& chunks) const;

private:
    mutable std::mutex mutex_;
    const char* entry_point_ = "";
    int width_ = 0;
    ShotCounting mode_ = ShotCounting::Auto;
    double distinct_fraction_ = SHOT_COUNTING_DISTINCT_FRACTION;
    std::vector<std::optional<ShotCounts>> pending_;
    ShotCounts merged_;
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
    ShotCounts partial;                // its chunk's shots before it
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
//                  ShotCounts& counts, ShotFailure& failure_out);
//
// run_chunk runs shots [first_shot, first_shot + n) drawing from rng and adds
// each shot's key to counts, whose width is the one `counts` (OrderedCounts)
// was built with. When a shot throws, it fills failure_out (shot set,
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
            bool strings = counts.begin_strings();
            ShotCounts chunk_counts = counts.fresh(strings);
            for (int c = next.fetch_add(1, std::memory_order_relaxed); c < chunks.count;
                 c = next.fetch_add(1, std::memory_order_relaxed)) {
                if (c >= failures.limit()) break;
                held = c;
                ShotFailure failure;
                ShotRng rng = chunk_stream(base_seed, salt, c);
                const std::size_t had = chunk_counts.size();
                worker.run_chunk(c, chunks.first(c), chunks.size(c), rng, chunk_counts,
                                 failure);
                if (failure.shot >= 0) {
                    failures.offer(c, std::move(failure));
                    return;
                }
                if (c == chunks.count - 1) owner.store(t, std::memory_order_relaxed);
                strings = counts.end_chunk(chunk_counts, chunks.size(c),
                                           chunk_counts.size() - had, !serial);
                if (!serial) {
                    counts.commit(c, std::move(chunk_counts));
                    chunk_counts = counts.fresh(strings);
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
