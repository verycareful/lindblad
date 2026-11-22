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
// TeamModel - the threads a statevector pass deserves
// =============================================================================
// A pass over the state is asked how many threads it should start. The grain
// rule answers from the pass's size alone: every thread gets at least
// parallel_grain() amplitudes. The automatic team answers from what the
// machine does with that pass. The pass is described by SvWork:
//
//   cls          which measured pass it resembles: a one-qubit gate (Gate), a
//                diagonal gate (Diagonal), a dense k-qubit block (Dense1 to
//                Dense6). Passes with no figure of their own take Gate's, at
//                the weight their work per amplitude has against a one-qubit
//                gate (weight_log2, a power of two: a flat pass over the
//                state, which reads and writes each amplitude once with one or
//                two operations, is half a gate's work, so -1).
//   amplitudes   how many amplitudes the pass runs over, which with the class
//                and the weight is the cell the answer is kept under.
//   grain_work   the number the grain rule divides by the grain, kept so that
//                a grain set by the caller reproduces its teams exactly.
//
// Two sources fill the answers, both kept in a table of one cell per (class,
// weight, floor of log2 of the amplitudes) that a decision reads with one
// atomic load:
//
//   measured     the default. The first time a cell is asked, the pass itself
//                is timed on a scratch state of that size, once at each team
//                size the machine's topology suggests (threading_team_sizes),
//                with the target changing from pass to pass as a circuit's
//                gates change it, and the fastest team is kept for the rest of
//                the process. This sees what a model of the machine does not:
//                how far amplitudes move between cores at that state size, on
//                the placement this process got, whatever the wait policy. A
//                state larger than twice the L3 caches together is timed on a
//                window of that size, and its cells share the answer. The
//                scratch state is charged to the caller's memory room (what
//                the cap leaves beside the run's own buffers): a cell that
//                does not fit holds no team and is asked again by the next
//                decision, so a later call with room measures it.
//   costs        Options::threading_costs set: the team that minimises the
//                region's cost plus the pass's time, from those costs alone;
//                nothing is measured.
//
// A cell with no figure (no L3 size detected, a figure missing from costs the
// caller set, a pass the timing cannot run) takes the grain rule; the
// execution report counts those.

#include "lindblad/detail/thread_cap.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace lindblad {
namespace detail {

enum class SvClass : int {
    Gate = 0,
    Diagonal,
    Dense1,
    Dense2,
    Dense3,
    Dense4,
    Dense5,
    Dense6,
};
inline constexpr int kSvClasses = 8;

inline SvClass dense_class(int k) noexcept {
    return static_cast<SvClass>(static_cast<int>(SvClass::Dense1) + std::clamp(k, 1, 6) - 1);
}

const char* sv_class_name(SvClass cls) noexcept;

struct SvWork {
    std::size_t grain_work;
    std::size_t amplitudes;
    SvClass cls = SvClass::Gate;
    int weight_log2 = 0;
};

// Weights a Gate pass may carry: 2^-1 to 2^4, and heavier ones are rated as 2^4.
inline constexpr int kMinWeightLog2 = -1;
inline constexpr int kMaxWeightLog2 = 4;
inline constexpr int kWeightSlots = kMaxWeightLog2 - kMinWeightLog2 + 1;
// Buckets by floor of log2 of the amplitudes.
inline constexpr int kAmplitudeBuckets = 48;

// What a cell holds besides a team (1 and up).
inline constexpr std::int16_t kCellUnknown = 0;
inline constexpr std::int16_t kCellNoFigure = -1;   // the grain rule decides
inline constexpr std::int16_t kCellRetry = -2;      // no room to measure now; ask again

// =============================================================================
// TeamTable - the answers, shared by every run of the process that has the
// same thread limit and processors
// =============================================================================
class TeamTable {
public:
    // The table of measured cells for the calling thread's thread limit and
    // processors, made once per process.
    static std::shared_ptr<TeamTable> measured();

    // A table decided from `costs` alone (analytic: nothing is measured).
    static std::shared_ptr<TeamTable> from_costs(ThreadingCosts costs);

    // The team for `work` (1 and up), or 0 when the table has no figure for it
    // and the grain rule must decide. `room_bytes` is the memory a measurement
    // may use for its scratch state.
    int team(const SvWork& work, std::uint64_t room_bytes) {
        std::atomic<std::int16_t>& cell = cells_[index(work)];
        int value = cell.load(std::memory_order_relaxed);
        if (value == kCellUnknown || value == kCellRetry) {
            value = compute(work, room_bytes);
            cell.store(static_cast<std::int16_t>(value), std::memory_order_relaxed);
        }
        return value > 0 ? value : 0;
    }

    // The costs a table made from costs decides by; empty for a measured one.
    const ThreadingCosts* costs() const noexcept { return costs_.unset() ? nullptr : &costs_; }

    TeamTable(const TeamTable&) = delete;
    TeamTable& operator=(const TeamTable&) = delete;

private:
    TeamTable(ThreadingCosts costs, int limit);

    static std::size_t index(const SvWork& w) noexcept {
        const int widx = w.cls == SvClass::Gate
                             ? std::clamp(w.weight_log2, kMinWeightLog2, kMaxWeightLog2) - kMinWeightLog2
                             : 0;
        const int bucket = w.amplitudes == 0
                               ? 0
                               : std::min(static_cast<int>(std::bit_width(w.amplitudes)) - 1,
                                          kAmplitudeBuckets - 1);
        return (static_cast<std::size_t>(w.cls) * kWeightSlots + static_cast<std::size_t>(widx)) *
                   kAmplitudeBuckets +
               static_cast<std::size_t>(bucket);
    }

    int compute(const SvWork& work, std::uint64_t room_bytes);
    int compute_measured(const SvWork& work, std::uint64_t room_bytes);
    int compute_from_costs(const SvWork& work);
    double reachable_cache_bytes(int t) const noexcept;

    ThreadingCosts costs_;   // set for a table made from costs
    int limit_;
    double llc_bytes_ = 0.0;
    std::size_t llc_instances_ = 0;
    std::size_t threads_per_llc_ = 0;
    std::mutex mutex_;
    std::unique_ptr<std::atomic<std::int16_t>[]> cells_;
};

// =============================================================================
// TeamModel - one call's view of a table: its room and what it decided
// =============================================================================
class TeamModel {
public:
    TeamModel(std::shared_ptr<TeamTable> table, std::uint64_t room_bytes)
        : table_(std::move(table)), room_(room_bytes), limit_(static_cast<int>(hw::available_threads())) {}

    int team(const SvWork& work) { return table_->team(work, room_); }

    // One decision, for the execution report: the pass class, the team it got,
    // and whether the grain rule decided because the table had no figure.
    void note(SvClass cls, int team, bool fell_back) noexcept {
        Use& u = use_[static_cast<std::size_t>(cls)];
        ++u.decisions;
        if (fell_back) ++u.fallbacks;
        const std::size_t t = static_cast<std::size_t>(std::max(team, 1));
        if (u.histogram.size() <= t) u.histogram.resize(std::max<std::size_t>(t, static_cast<std::size_t>(limit_)) + 1, 0);
        ++u.histogram[t];
    }

    // The report's team rule and kernel teams from what has been noted, and the
    // costs the decisions were made from when the table was made from costs.
    void fill_report(ExecutionReport& report) const;

private:
    struct Use {
        std::uint64_t decisions = 0;
        std::uint64_t fallbacks = 0;
        std::vector<std::uint64_t> histogram;
    };

    std::shared_ptr<TeamTable> table_;
    std::uint64_t room_;
    int limit_;
    std::array<Use, kSvClasses> use_{};
};

// =============================================================================
// ScopedTeamModel - the model for the length of one simulator call
// =============================================================================
// With `automatic` false it installs nothing and every pass takes the grain
// rule. Otherwise it makes a model the calling thread's: over the table made
// from `pinned` when that is set, else over the process's measured table. It
// measures nothing itself. The previous model comes back when it ends.
class ScopedTeamModel {
public:
    ScopedTeamModel(bool automatic, const ThreadingCosts& pinned, std::uint64_t scratch_room_bytes);
    ~ScopedTeamModel();

    ScopedTeamModel(const ScopedTeamModel&) = delete;
    ScopedTeamModel& operator=(const ScopedTeamModel&) = delete;

    // Writes what the call decided into `report`.
    void fill_report(ExecutionReport& report) const;

private:
    std::unique_ptr<TeamModel> model_;
    TeamModel* previous_ = nullptr;
};

}  // namespace detail
}  // namespace lindblad
