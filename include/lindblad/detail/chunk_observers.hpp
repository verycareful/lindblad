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
// chunk_observers - observers on a shot team
// =============================================================================
// A run's observers fire from the shot loop. On a team the loop's chunks run on
// different threads, so each chunk fires into forks of the observers
// (ObservationRunner::fork) instead of the caller's own. The forks are kept by
// chunk and folded back, on the calling thread and in chunk order, after the
// team is done, which leaves the caller's observers holding the firings in the
// order a run on one thread would have given them.
//
// Nothing here is shared between threads while the team runs: chunk c's slot is
// written by the one thread that runs chunk c.

#include "lindblad/observation.hpp"
#include "lindblad/detail/memory_budget.hpp"
#include "lindblad/detail/shot_loop.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace lindblad {
namespace detail {

// =============================================================================
// ChunkObservers - the runner each chunk fires into, and the folding back
// =============================================================================
class ChunkObservers {
public:
    // parent = the run's runner, or null when the run is not watched; chunks =
    // the run's chunk count; fork = whether chunks run on a team (a team of one
    // fires into the parent directly, so a serial run pays nothing for this).
    ChunkObservers(ObservationRunner* parent, int chunks, bool fork)
        : parent_(parent != nullptr && parent->active() ? parent : nullptr),
          fork_(fork && parent_ != nullptr) {
        if (fork_) forks_.resize(static_cast<std::size_t>(chunks));
    }

    // The runner chunk c fires into: null when nothing is watched, the parent
    // when there is no team, else a fork over `worker_budget` (the budget of
    // the thread running the chunk), kept until merge_through. Called once per
    // chunk, on the thread that runs it.
    ObservationRunner* begin_chunk(int c, RunBudget* worker_budget) {
        if (!fork_) return parent_;
        auto& slot = forks_[static_cast<std::size_t>(c)];
        slot = parent_->fork(worker_budget);
        return slot.get();
    }

    // Folds the forks of chunks 0 through `last` into the parent, in chunk
    // order. A chunk whose thread never reached begin_chunk has no fork and
    // adds nothing. Chunks above `last` are dropped: after a failure they hold
    // firings of shots a run on one thread would never have reached.
    void merge_through(int last) {
        if (!fork_) return;
        const int stop = std::min(last + 1, static_cast<int>(forks_.size()));
        for (int c = 0; c < stop; ++c) {
            auto& slot = forks_[static_cast<std::size_t>(c)];
            if (slot) parent_->merge(*slot);
            slot.reset();
        }
        forks_.clear();
    }

private:
    ObservationRunner* parent_;
    bool fork_;
    std::vector<std::unique_ptr<ObservationRunner>> forks_;
};

// =============================================================================
// team_budget - what one team thread's observers charge
// =============================================================================
// The run's RunBudget belongs to the calling thread and is not thread-safe, so
// each team thread charges a budget of its own. Under ShotMemory::FixedShare
// that is the thread's share of the cap with its own state held; otherwise it
// is the run's cap with every thread's state held, which leaves the room that
// is left once all the states are there. A team of one uses the run's own
// budget and needs none of this. RunBudget cannot be moved, so the result is
// built in place and handed over as it is.
inline std::optional<RunBudget> team_budget(bool needed, const ShotTeam& team,
                                            std::uint64_t cap_bytes,
                                            std::uint64_t held_bytes,
                                            std::uint64_t per_thread_bytes,
                                            const char* entry_point) {
    if (!needed) return std::nullopt;
    if (team.share_bytes != 0) {
        return std::optional<RunBudget>(std::in_place, team.share_bytes, per_thread_bytes,
                                        entry_point);
    }
    const std::uint64_t others =
        saturating_mul(static_cast<std::uint64_t>(team.threads - 1), per_thread_bytes);
    return std::optional<RunBudget>(std::in_place, cap_bytes,
                                    saturating_add(held_bytes, others), entry_point);
}

}  // namespace detail
}  // namespace lindblad
