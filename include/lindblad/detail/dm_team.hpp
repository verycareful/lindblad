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
// DmTeamModel - the threads a density-matrix pass deserves
// =============================================================================
// Every parallel pass over ρ (dm_kernels.hpp) asks dm_team(DmWork) how many
// threads to start. Two rules answer:
//
//   fixed      a pass whose work reaches DM_PAR_THRESHOLD elements takes every
//              thread the region may use, a smaller one runs serially; or,
//              when a grain is in force (Options::min_amplitudes_per_thread >
//              0), min(threads, work / grain). This is the rule of a
//              DensityMatrix used outside a simulator, of a pass whose cell
//              holds no figure, and of a run that set the grain.
//   automatic  the default under a simulator: the team in which a sweep of
//              passes ran fastest on this machine, kept per register width in a
//              table read with one atomic load.
//
// A width's team is found the first time any pass of that width is asked, by
// timing a sweep of the passes a noisy circuit runs (a gate's ket and bra
// multiplies and its channel's superoperator pass, on one and on two qubits, the
// targets changing from call to call as a circuit's gates do) on a scratch
// matrix of the width, at each team size the machine's topology suggests
// (threading_team_sizes), and keeping the fastest. Every class of that width
// takes it: a circuit's passes follow one another on the same matrix, and a
// pass that finds it partitioned differently from the pass before it reads rows
// another core holds, so a team chosen per pass from that pass alone loses to
// the one team (a matrix that fits the cores' private caches at the full team
// runs several times slower once its passes alternate between team sizes). The
// scratch matrix is charged to the caller's memory room; a width that does not
// fit holds no team and is asked again by the next decision. A register wider
// than one last-level cache's worth of matrix shares the team of the widest
// one that fits, since timing it on the real size would take longer than the
// calls it speeds up.
//
// With Options::threading_costs set nothing is timed: the cell is decided from
// the costs by the statevector's analytic model, with each density-matrix pass
// read as the statevector pass of the same arithmetic per element (a ket or
// bra multiply on k qubits is a dense block of k qubits, a channel's
// superoperator on k qubits one of 2k qubits, a phase a diagonal, a permutation
// or a trace over the diagonal a one-qubit pass of twice and four times the
// work). That is an approximation, the one place a pinned density-matrix
// decision rests on a figure not timed on its own kernel.
//
// The model's table is separate from the statevector's (TeamTable): a
// statevector pass during a density-matrix call can never read a density-matrix
// cell. A table decided from costs has a cell per class; a measured one gives
// every class of a width the same team.

#include "lindblad/detail/team_model.hpp"
#include "lindblad/detail/thread_cap.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace lindblad {
namespace detail {

// A pass over ρ reaches the parallel region once its work, in elements of ρ
// touched, is this many: below it the region costs more than it saves on every
// machine this was timed on, and a DensityMatrix with no simulator around it
// keeps that rule.
inline constexpr std::size_t DM_PAR_THRESHOLD = std::size_t{1} << 16;

// Ket, bra and superoperator passes are classed by the qubits of the gate or
// channel, 1 to 3; a wider one is rated as 3.
enum class DmClass : int {
    Ket1 = 0,
    Ket2,
    Ket3,
    Bra1,
    Bra2,
    Bra3,
    Superop1,
    Superop2,
    Superop3,
    Permute,     // full-register basis permutation
    Phase,       // multi-controlled phase
    Trace,       // trace against a Pauli string (reads one row element per row)
};
inline constexpr int kDmClasses = 12;
inline constexpr int kDmMaxGateClass = 3;

// Registers of 0 to 15 qubits have a cell each; wider ones take the last.
inline constexpr int kDmQubitSlots = 16;

inline DmClass dm_ket_class(int k) noexcept {
    return static_cast<DmClass>(static_cast<int>(DmClass::Ket1) + std::clamp(k, 1, kDmMaxGateClass) - 1);
}
inline DmClass dm_bra_class(int k) noexcept {
    return static_cast<DmClass>(static_cast<int>(DmClass::Bra1) + std::clamp(k, 1, kDmMaxGateClass) - 1);
}
inline DmClass dm_superop_class(int k) noexcept {
    return static_cast<DmClass>(static_cast<int>(DmClass::Superop1) + std::clamp(k, 1, kDmMaxGateClass) - 1);
}

const char* dm_class_name(DmClass cls) noexcept;

// A pass, as the team rules see it.
//   grain_work  the elements of ρ the pass touches, which the fixed rule
//               compares with DM_PAR_THRESHOLD or divides by the grain
//   n_qubits    the register's width, which with cls names the cell
//   cls         the pass class
//   units       the parallel loop's trip count; a team never exceeds it
struct DmWork {
    std::size_t grain_work;
    int n_qubits;
    DmClass cls;
    std::size_t units;
};

// What a cell holds besides a team (1 and up): the statevector table's markers.
inline constexpr std::int16_t kDmCellUnknown = 0;
inline constexpr std::int16_t kDmCellNoFigure = -1;   // the fixed rule decides
inline constexpr std::int16_t kDmCellRetry = -2;      // no room to measure now; ask again

// =============================================================================
// DmTeamTable - the answers, shared by every run of the process that has the
// same thread limit and processors
// =============================================================================
class DmTeamTable {
public:
    // The table of measured cells for the calling thread's thread limit and
    // processors, made once per process.
    static std::shared_ptr<DmTeamTable> measured();

    // A table decided from `costs` alone: nothing is measured.
    static std::shared_ptr<DmTeamTable> from_costs(ThreadingCosts costs);

    // The team for `work` (1 and up), or 0 when the table has no figure for it
    // and the fixed rule must decide. `room_bytes` is the memory a measurement
    // may use for its scratch matrix. Inside a parallel region nothing is
    // measured (a nested region runs on one thread, so every candidate would
    // take the same time) and nothing is kept.
    int team(const DmWork& work, std::uint64_t room_bytes) {
        std::atomic<std::int16_t>& cell = cells_[index(work)];
        int value = cell.load(std::memory_order_relaxed);
        if (value == kDmCellUnknown || value == kDmCellRetry) {
            if (inside_parallel_region()) return 0;
            value = compute(work, room_bytes);
            cell.store(static_cast<std::int16_t>(value), std::memory_order_relaxed);
        }
        return value > 0 ? value : 0;
    }

    // The costs a table made from costs decides by; null for a measured one.
    const ThreadingCosts* costs() const noexcept { return costs_.unset() ? nullptr : &costs_; }

    DmTeamTable(const DmTeamTable&) = delete;
    DmTeamTable& operator=(const DmTeamTable&) = delete;

private:
    DmTeamTable(ThreadingCosts costs, int limit);

    static int slot(const DmWork& w) noexcept {
        return std::clamp(w.n_qubits, 0, kDmQubitSlots - 1);
    }
    static std::size_t index(const DmWork& w) noexcept {
        return static_cast<std::size_t>(w.cls) * kDmQubitSlots + static_cast<std::size_t>(slot(w));
    }

    int compute(const DmWork& work, std::uint64_t room_bytes);
    int compute_measured(const DmWork& work, std::uint64_t room_bytes);
    int compute_from_costs(const DmWork& work);

    ThreadingCosts costs_;                 // set for a table made from costs
    std::shared_ptr<TeamTable> analytic_;  // the statevector model over costs_
    int limit_;
    double llc_bytes_ = 0.0;
    std::size_t llc_instances_ = 0;
    std::mutex mutex_;
    // The team a register of each width was timed to run its passes on, shared
    // by every class (see compute_measured).
    std::array<std::int16_t, kDmQubitSlots> uniform_{};
    std::unique_ptr<std::atomic<std::int16_t>[]> cells_;
};

// =============================================================================
// DmTeamModel - one call's view of a table: its room and what it decided
// =============================================================================
class DmTeamModel {
public:
    DmTeamModel(std::shared_ptr<DmTeamTable> table, std::uint64_t room_bytes)
        : table_(std::move(table)), room_(room_bytes), limit_(static_cast<int>(hw::available_threads())) {}

    int team(const DmWork& work) { return table_->team(work, room_); }

    // One decision, for the execution report.
    void note(DmClass cls, int team, bool fell_back) noexcept {
        Use& u = use_[static_cast<std::size_t>(cls)];
        ++u.decisions;
        if (fell_back) ++u.fallbacks;
        const std::size_t t = static_cast<std::size_t>(std::max(team, 1));
        if (u.histogram.size() <= t)
            u.histogram.resize(std::max<std::size_t>(t, static_cast<std::size_t>(limit_)) + 1, 0);
        ++u.histogram[t];
    }

    void fill_report(ExecutionReport& report) const;

private:
    struct Use {
        std::uint64_t decisions = 0;
        std::uint64_t fallbacks = 0;
        std::vector<std::uint64_t> histogram;
    };

    std::shared_ptr<DmTeamTable> table_;
    std::uint64_t room_;
    int limit_;
    std::array<Use, kDmClasses> use_{};
};

// The calling thread's model and grain. A grain above 0 is the fixed rule's
// work per thread; the model is null where none is in force.
inline thread_local DmTeamModel* t_dm_team_model = nullptr;
inline thread_local std::size_t t_dm_grain = 0;

// The fixed rule: the threads for a pass worth `work` elements, never more than
// a region may use.
inline int dm_team_fixed(std::size_t work) {
    const std::size_t threads = std::max<std::size_t>(hw::available_threads(), 1);
    if (t_dm_grain > 0) return static_cast<int>(std::max<std::size_t>(std::min(threads, work / t_dm_grain), 1));
    return work >= DM_PAR_THRESHOLD ? static_cast<int>(threads) : 1;
}

// The threads for a pass over ρ: the model's team when one is in force and has
// a figure, else the fixed rule's; never above the pass's loop count nor the
// threads a region may use.
inline int dm_team(const DmWork& work) {
    int team;
    if (DmTeamModel* model = t_dm_team_model) {
        const int modelled = model->team(work);
        team = modelled > 0 ? static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(modelled),
                                                                    hw::available_threads()))
                            : dm_team_fixed(work.grain_work);
        model->note(work.cls, team, modelled <= 0);
    } else {
        team = dm_team_fixed(work.grain_work);
    }
    return static_cast<int>(std::max<std::size_t>(std::min<std::size_t>(static_cast<std::size_t>(std::max(team, 1)),
                                                                        work.units), 1));
}

// =============================================================================
// ScopedDmTeam - the density-matrix team rule for the length of one call
// =============================================================================
// grain > 0 puts the fixed rule in force with that grain and installs no
// model. grain = 0 installs a model over the table made from `pinned` when
// that is set, else over the process's measured table; the fixed rule with no
// grain stays the fallback for a cell without a figure. The previous settings
// come back when it ends. It measures nothing itself.
class ScopedDmTeam {
public:
    ScopedDmTeam(std::size_t grain, const ThreadingCosts& pinned, std::uint64_t scratch_room_bytes);
    ~ScopedDmTeam();

    ScopedDmTeam(const ScopedDmTeam&) = delete;
    ScopedDmTeam& operator=(const ScopedDmTeam&) = delete;

    // Writes what the call decided into `report`.
    void fill_report(ExecutionReport& report, std::size_t grain) const;

private:
    std::unique_ptr<DmTeamModel> model_;
    DmTeamModel* previous_model_ = nullptr;
    std::size_t previous_grain_ = 0;
};

// The fixed rule with a grain of one element, the model set aside: a team of
// exactly the threads the region may use whatever the pass's size. A
// measurement that forces its own teams holds this while it times a kernel.
class ScopedDmForcedTeam {
public:
    ScopedDmForcedTeam() : previous_model_(t_dm_team_model), previous_grain_(t_dm_grain) {
        t_dm_team_model = nullptr;
        t_dm_grain = 1;
    }
    ~ScopedDmForcedTeam() {
        t_dm_team_model = previous_model_;
        t_dm_grain = previous_grain_;
    }
    ScopedDmForcedTeam(const ScopedDmForcedTeam&) = delete;
    ScopedDmForcedTeam& operator=(const ScopedDmForcedTeam&) = delete;

private:
    DmTeamModel* previous_model_;
    std::size_t previous_grain_;
};

}  // namespace detail
}  // namespace lindblad
