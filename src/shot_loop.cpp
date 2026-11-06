// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/detail/shot_loop.hpp"

#include "lindblad/detail/memory_budget.hpp"
#include "lindblad/detail/report.hpp"
#include "lindblad/detail/thread_cap.hpp"
#include "lindblad/hw_info.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace lindblad {
namespace detail {

namespace {

// Mixed into a stream's seed after the salt. The two values only keep the two
// stream families apart: a chunk stream and the prefix stream never start from
// the same seed, whatever the chunk index.
constexpr std::uint64_t kChunkTag = 0x5C4A11C5u;
constexpr std::uint64_t kPrefixTag = 0x9E3779B9u;

// The seed of one stream: the run's seed, the salt with the family tag, and
// the index, each folded in through a splitmix64 step, so streams whose inputs
// differ in any one of them start from unrelated states.
ShotRng seeded_stream(std::uint64_t base_seed, std::uint32_t salt, std::uint64_t index,
                      std::uint64_t tag) {
    std::uint64_t state = base_seed;
    std::uint64_t key = ShotRng::splitmix64(state);
    state = key ^ ((static_cast<std::uint64_t>(salt) << 32) | tag);
    key = ShotRng::splitmix64(state);
    state = key ^ index;
    return ShotRng(ShotRng::splitmix64(state));
}

// The cache assumed when the operating system reports none: the size of one
// L3 instance on current desktop and server parts. Too large an assumption
// sends a run's shots across threads where they run slower; the automatic
// rule's margin (SHOT_TEAM_MIN_CACHED_STATES) is what limits that cost.
constexpr std::size_t kLlcFallbackBytes = std::size_t(32) << 20;

// Whether enough per-thread states fit one cache instance for an automatic
// team to beat running every gate across threads (types.hpp).
bool states_fit_cache(std::uint64_t per_thread_bytes) {
    if (per_thread_bytes == 0) return true;
    std::size_t llc = hw::llc_bytes();
    if (llc == 0) llc = kLlcFallbackBytes;
    return static_cast<std::uint64_t>(llc) / per_thread_bytes >= SHOT_TEAM_MIN_CACHED_STATES;
}

bool in_parallel_region() noexcept {
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

}  // namespace

ShotChunks chunk_shots(int shots) noexcept {
    ShotChunks ch;
    if (shots <= 0) return ch;  // one chunk of one trajectory
    ch.shots = shots;
    ch.chunk = (shots + SHOT_CHUNK_TARGET - 1) / SHOT_CHUNK_TARGET;
    ch.count = (shots + ch.chunk - 1) / ch.chunk;
    return ch;
}

ShotRng chunk_stream(std::uint64_t base_seed, std::uint32_t salt, int c) {
    return seeded_stream(base_seed, salt, static_cast<std::uint64_t>(c), kChunkTag);
}

ShotRng prefix_stream(std::uint64_t base_seed, std::uint32_t salt) {
    return seeded_stream(base_seed, salt, 0, kPrefixTag);
}

ShotTeam shot_team(const ShotTeamRequest& req, const ShotChunks& chunks) {
    const int k = req.max_parallel_shots;
    if (k < 0) {
        raise<InvalidArgument>(req.entry_point,
            "max_parallel_shots must be >= 0 (0 = automatic, 1 = serial), got " +
                std::to_string(k));
    }
    const bool exact = req.memory == ShotMemory::Exact;
    if (exact && k == 0) {
        raise<InvalidArgument>(req.entry_point,
            "ShotMemory::Exact needs max_parallel_shots set to the team it should hold");
    }

    const std::uint64_t room =
        req.cap_bytes > req.held_bytes ? req.cap_bytes - req.held_bytes : 0;
    // A FixedShare team of one holds the whole room plus its own state.
    const auto serial = [&] {
        ShotTeam t;
        if (req.memory == ShotMemory::FixedShare)
            t.share_bytes = saturating_add(room, req.per_thread_bytes);
        return t;
    };

    if (in_parallel_region()) {
        if (exact && k >= 2) {
            raise<InvalidArgument>(req.entry_point,
                "ShotMemory::Exact asks for " + std::to_string(k) +
                    " shot threads, but the run was called from inside a parallel "
                    "region, where its shots run on one thread");
        }
        return serial();
    }
    if (req.serial_only || chunks.count <= 1 || k == 1) return serial();

    std::size_t wanted = 0;
    if (k >= 2) {
        wanted = static_cast<std::size_t>(k);
    } else {
        if (!states_fit_cache(req.per_thread_bytes)) return serial();
        const std::uint64_t work = saturating_mul(
            static_cast<std::uint64_t>(req.work_per_shot),
            static_cast<std::uint64_t>(chunks.shots));
        wanted = std::min<std::uint64_t>(hw::available_threads(), work / parallel_grain());
        wanted = std::max<std::size_t>(wanted, 1);
    }
    wanted = std::min(wanted, static_cast<std::size_t>(chunks.count));
    if (wanted <= 1) return serial();

    const std::uint64_t per = req.per_thread_bytes;
    // Bytes the extra threads' states need beside the one the run holds.
    const auto extra = [per](std::size_t team) {
        return saturating_mul(per, static_cast<std::uint64_t>(team - 1));
    };

    ShotTeam t;
    switch (req.memory) {
        case ShotMemory::Shrink: {
            const std::uint64_t fit = per == 0 ? wanted : 1 + room / per;
            t.threads = static_cast<int>(std::min<std::uint64_t>(wanted, fit));
            break;
        }
        case ShotMemory::SerialIfShort:
            t.threads = extra(wanted) <= room ? static_cast<int>(wanted) : 1;
            break;
        case ShotMemory::Exact:
            if (extra(wanted) > room) {
                raise<InvalidArgument>(req.entry_point,
                    "ShotMemory::Exact asks for " + std::to_string(wanted) +
                        " shot threads, whose working states need " +
                        bytes_text(extra(wanted)) + " beside the " +
                        bytes_text(req.held_bytes) + " the run holds, over the cap of " +
                        bytes_text(req.cap_bytes));
            }
            t.threads = static_cast<int>(wanted);
            break;
        case ShotMemory::FixedShare: {
            // An automatic team takes only as many shares as each hold one
            // working state; a team the caller named is held to its number.
            if (k == 0 && per > 0) wanted = std::min<std::uint64_t>(wanted, 1 + room / per);
            if (wanted <= 1) return serial();
            t.threads = static_cast<int>(wanted);
            t.share_bytes = saturating_add(room, per) / wanted;
            if (t.share_bytes < per) {
                raise<InvalidArgument>(req.entry_point,
                    "ShotMemory::FixedShare splits the " + bytes_text(room + per) +
                        " left under the cap across " + std::to_string(wanted) +
                        " shot threads, a share of " + bytes_text(t.share_bytes) +
                        ", below the " + bytes_text(per) +
                        " one thread's working state needs; max_parallel_shots = 1 "
                        "gives the shot the whole cap");
            }
            break;
        }
    }
    return t;
}

OrderedCounts::OrderedCounts(int chunk_count)
    : pending_(static_cast<std::size_t>(std::max(chunk_count, 0))) {}

void OrderedCounts::commit(int c, Counts&& counts) {
    const std::lock_guard<std::mutex> lock(mutex_);
    pending_[static_cast<std::size_t>(c)] = std::move(counts);
    while (next_ < static_cast<int>(pending_.size()) &&
           pending_[static_cast<std::size_t>(next_)]) {
        auto& ready = *pending_[static_cast<std::size_t>(next_)];
        if (merged_.empty()) {
            merged_ = std::move(ready);
        } else {
            for (auto& [key, n] : ready) merged_[key] += n;
        }
        pending_[static_cast<std::size_t>(next_)].reset();
        ++next_;
    }
}

void OrderedCounts::commit_all(Counts&& counts) {
    const std::lock_guard<std::mutex> lock(mutex_);
    merged_ = std::move(counts);
    next_ = static_cast<int>(pending_.size());
}

OrderedCounts::Counts OrderedCounts::take() {
    const std::lock_guard<std::mutex> lock(mutex_);
    Counts out = std::move(merged_);
    merged_.clear();
    return out;
}

int OrderedCounts::shots_committed(const ShotChunks& chunks) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return next_ >= chunks.count ? chunks.shots : chunks.first(next_);
}

void ChunkFailures::offer(int c, ShotFailure&& f) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (held_ && held_->shot <= f.shot) return;
    held_ = std::move(f);
    if (c < limit_.load(std::memory_order_relaxed))
        limit_.store(c, std::memory_order_release);
}

std::optional<ShotFailure> ChunkFailures::take() {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::optional<ShotFailure> out = std::move(held_);
    held_.reset();
    return out;
}

}  // namespace detail
}  // namespace lindblad
