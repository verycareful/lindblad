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
// rule's margin (SHOT_TEAM_MIN_CACHED_STATES, or the run's own) is what limits
// that cost.
constexpr std::size_t kLlcFallbackBytes = std::size_t(32) << 20;

// Whether enough per-thread states fit one cache instance for an automatic
// team to beat running every gate across threads (types.hpp). min_states is
// the run's margin, 0 being SHOT_TEAM_MIN_CACHED_STATES.
bool states_fit_cache(std::uint64_t per_thread_bytes, std::size_t min_states) {
    if (per_thread_bytes == 0) return true;
    if (min_states == 0) min_states = SHOT_TEAM_MIN_CACHED_STATES;
    std::size_t llc = hw::llc_bytes();
    if (llc == 0) llc = kLlcFallbackBytes;
    return static_cast<std::uint64_t>(llc) / per_thread_bytes >= min_states;
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
        if (req.cache_rule && !states_fit_cache(req.per_thread_bytes, req.min_cached_states))
            return serial();
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

// The string of a register: clbit c is the c-th character from the right.
// Every operand is a local: a char store may alias any object, so a bound or a
// pointer read through a member or a vector would be reloaded after each
// character, and the loop would run with no known trip count.
static std::string register_string(const int* reg, int width) {
    std::string bits(static_cast<std::size_t>(width), '0');
    char* const out = bits.data();
    for (int c = 0; c < width; ++c)
        if (reg[c]) out[width - 1 - c] = '1';
    return bits;
}

// The string of a packed key; clbits at and above 64 read 0.
static std::string packed_key_string(std::uint64_t key, int width) {
    std::string bits(static_cast<std::size_t>(width), '0');
    char* const out = bits.data();
    const int low = width < ShotCounts::PACKED_MAX_WIDTH ? width : ShotCounts::PACKED_MAX_WIDTH;
    for (int c = 0; c < low; ++c)
        if ((key >> c) & 1ULL) out[width - 1 - c] = '1';
    return bits;
}

void ShotCounts::add(const std::vector<int>& reg) {
    const int width = width_;
    const int* const r = reg.data();
    if (packed_mode_) {
        std::uint64_t key = 0;
        for (int c = 0; c < width; ++c)
            if (r[c]) key |= 1ULL << c;
        ++packed_[key];
        return;
    }
    ++strings_[register_string(r, width)];
}

void ShotCounts::add_value(std::uint64_t outcome) {
    const int width = width_;
    if (width < PACKED_MAX_WIDTH) outcome &= (1ULL << width) - 1;
    if (packed_mode_) {
        ++packed_[outcome];
        return;
    }
    ++strings_[packed_key_string(outcome, width)];
}

void ShotCounts::to_strings() {
    if (!packed_mode_) return;
    const int width = width_;
    strings_.reserve(strings_.size() + packed_.size());
    for (const auto& [key, n] : packed_) strings_[packed_key_string(key, width)] += n;
    packed_ = {};
    packed_mode_ = false;
}

// Adds `from` into `into` and leaves `from` empty. The larger map receives the
// smaller one's entries, so the work is the smaller map's size; the sums are
// integers, so which side receives does not change them. Entries whose key
// `into` lacks move over as nodes; the ones left in `from` are the keys both
// held.
template <class Map>
static void merge_counts(Map& into, Map& from) {
    if (from.size() > into.size()) into.swap(from);
    if (from.empty()) return;
    into.merge(from);
    for (auto& [key, n] : from) into.find(key)->second += n;
    from.clear();
}

void ShotCounts::merge(ShotCounts&& other, std::string_view entry_point) {
    if (other.empty()) return;
    if (other.width_ != width_) {
        raise_internal(entry_point,
            "shot counts of key width " + std::to_string(other.width_) +
                " cannot be merged into counts of width " + std::to_string(width_));
    }
    if (empty()) {
        std::swap(*this, other);
        other = ShotCounts(width_);
        return;
    }
    // One side already under its strings: both go under strings.
    if (packed_mode_ != other.packed_mode_) {
        to_strings();
        other.to_strings();
    }
    if (packed_mode_)
        merge_counts(packed_, other.packed_);
    else
        merge_counts(strings_, other.strings_);
    other = ShotCounts(width_);
}

ShotCounts::Strings ShotCounts::strings() {
    to_strings();
    Strings out = std::move(strings_);
    *this = ShotCounts(width_);
    return out;
}

void check_shot_counting(const char* entry_point, double distinct_fraction) {
    if (distinct_fraction == 0.0) return;
    if (!(distinct_fraction > 0.0 && distinct_fraction <= 1.0)) {
        raise<InvalidArgument>(entry_point,
            "Options::shot_counting_distinct_fraction must be 0 (the default, " +
                std::to_string(SHOT_COUNTING_DISTINCT_FRACTION) +
                ") or in (0, 1], got " + std::to_string(distinct_fraction));
    }
}

bool OrderedCounts::end_chunk(ShotCounts& counts, int shots, std::size_t gained,
                              bool team) const {
    switch (mode_) {
        case ShotCounting::Strings:
            return true;
        case ShotCounting::Packed:
            return false;
        case ShotCounting::PackedPerChunk:
            if (team) counts.to_strings();
            return false;
        case ShotCounting::Auto:
            break;
    }
    // A run on one thread keeps one map, so `gained` counts the outcomes new
    // to the run; on a team it counts the chunk's own distinct outcomes.
    if (static_cast<double>(gained) > distinct_fraction_ * static_cast<double>(shots)) {
        counts.to_strings();
        return true;
    }
    return counts.under_strings();
}

OrderedCounts::OrderedCounts(const char* entry_point, int chunk_count, int width,
                             ShotCounting mode, double distinct_fraction)
    : entry_point_(entry_point),
      width_(width),
      mode_(mode),
      distinct_fraction_(distinct_fraction == 0.0 ? SHOT_COUNTING_DISTINCT_FRACTION
                                                  : distinct_fraction),
      pending_(static_cast<std::size_t>(std::max(chunk_count, 0))),
      merged_(width) {}

void OrderedCounts::commit(int c, ShotCounts&& counts) {
    const std::lock_guard<std::mutex> lock(mutex_);
    pending_[static_cast<std::size_t>(c)] = std::move(counts);
    while (next_ < static_cast<int>(pending_.size()) &&
           pending_[static_cast<std::size_t>(next_)]) {
        merged_.merge(std::move(*pending_[static_cast<std::size_t>(next_)]), entry_point_);
        pending_[static_cast<std::size_t>(next_)].reset();
        ++next_;
    }
}

void OrderedCounts::commit_all(ShotCounts&& counts) {
    const std::lock_guard<std::mutex> lock(mutex_);
    merged_ = std::move(counts);
    next_ = static_cast<int>(pending_.size());
}

ShotCounts::Strings OrderedCounts::take(ShotCounts&& tail) {
    const std::lock_guard<std::mutex> lock(mutex_);
    merged_.merge(std::move(tail), entry_point_);
    ShotCounts::Strings out = merged_.strings();
    merged_ = ShotCounts(width_);
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
