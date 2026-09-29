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
// memory_budget - the memory cap, the checks against it, and the run budget
// =============================================================================
// max_memory_mb is the most memory a run may use, in MiB (2^20 bytes). The cap
// a run answers to is resolved from it once, before the run starts:
//
//   max_memory_mb > 0         that many MiB; NO_MEMORY_CAP means no cap
//   max_memory_mb == 0        automatic: the memory this machine reports
//                             available, or FALLBACK_MEMORY_CAP_MB when it
//                             gives no coherent answer
//
// Two checks use it. Before the first gate, a run whose unavoidable buffers
// exceed the cap is refused (the statevector simulator counts two states, the
// density-matrix simulator one matrix, the Clifford simulator its tableau and
// sampling slab). While the run goes on, a RunBudget charges everything the
// run allocates beyond that fixed footprint before it is allocated: an MPS
// chain's growth and its dense fallbacks, and the copies observers take.
// Going over is then a failure the run reports and can save, rather than a
// request the operating system may answer by killing the process, which no
// program can catch.
//
// What stays out of reach: another process allocating during the run (a
// reading is not a reservation), and the kernel's out-of-memory killer, which
// ends the process without an exception.

#include "lindblad/detail/report.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lindblad {
namespace detail {

// Every size below saturates at the largest uint64_t rather than wrapping, so
// a register too wide to hold at all reads as needing more than any cap.
inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) {
    return (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
               ? std::numeric_limits<std::uint64_t>::max()
               : a * b;
}

inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return b > std::numeric_limits<std::uint64_t>::max() - a
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

inline std::uint64_t pow2_saturating(int n) {
    return n >= std::numeric_limits<std::uint64_t>::digits
               ? std::numeric_limits<std::uint64_t>::max()
               : std::uint64_t{1} << n;
}

// Bytes in `count` complex amplitudes or entries.
inline std::uint64_t complex_bytes(std::uint64_t count) {
    return saturating_mul(count, sizeof(Complex128));
}

inline std::uint64_t mib_to_bytes(std::uint64_t mib) {
    return mib > (std::numeric_limits<std::uint64_t>::max() >> 20)
               ? std::numeric_limits<std::uint64_t>::max()
               : mib << 20;
}

// A byte count for a message: exact bytes, with the MiB beside it once there
// is at least one.
inline std::string bytes_text(std::uint64_t bytes) {
    std::string out = std::to_string(bytes) + " bytes";
    if (bytes >= (std::uint64_t{1} << 20)) {
        out += " (" + std::to_string(bytes >> 20) + " MiB)";
    }
    return out;
}

// -----------------------------------------------------------------------------
// The resolved cap
// -----------------------------------------------------------------------------

struct MemoryCap {
    enum class From { Caller, NoCap, Machine, Fallback };
    std::uint64_t bytes = 0;
    From from = From::Caller;
};

// The cap a run answers to. The machine is asked at most once per call, so the
// figure and where it came from always agree.
inline MemoryCap resolve_memory_cap(std::uint64_t max_memory_mb) {
    if (max_memory_mb == NO_MEMORY_CAP) {
        return {std::numeric_limits<std::uint64_t>::max(), MemoryCap::From::NoCap};
    }
    if (max_memory_mb != 0) return {mib_to_bytes(max_memory_mb), MemoryCap::From::Caller};
    const std::size_t probed = hw::recent_available_memory_bytes();
    if (probed != 0) return {static_cast<std::uint64_t>(probed), MemoryCap::From::Machine};
    return {mib_to_bytes(FALLBACK_MEMORY_CAP_MB), MemoryCap::From::Fallback};
}

inline std::uint64_t resolve_memory_cap_bytes(std::uint64_t max_memory_mb) {
    return resolve_memory_cap(max_memory_mb).bytes;
}

// Where a cap came from, for the message that reports it.
inline std::string memory_cap_source(const MemoryCap& cap, std::uint64_t max_memory_mb) {
    switch (cap.from) {
        case MemoryCap::From::NoCap:
            return "NO_MEMORY_CAP";
        case MemoryCap::From::Caller:
            return "max_memory_mb = " + std::to_string(max_memory_mb);
        case MemoryCap::From::Machine:
            return "the memory this machine reports available (max_memory_mb = 0)";
        case MemoryCap::From::Fallback:
            return "the " + std::to_string(FALLBACK_MEMORY_CAP_MB) +
                   " MiB fallback, since the machine gave no coherent reading "
                   "(max_memory_mb = 0)";
    }
    return {};
}

// -----------------------------------------------------------------------------
// Before the first gate
// -----------------------------------------------------------------------------

// Refuses a run whose unavoidable state buffers (`need` bytes) exceed the
// resolved cap, and returns that cap in bytes so the run can build its budget
// from the same reading.
inline std::uint64_t require_memory_budget(std::uint64_t need, std::uint64_t max_memory_mb,
                                           const char* ctx) {
    const MemoryCap cap = resolve_memory_cap(max_memory_mb);
    if (need > cap.bytes) {
        raise<InvalidArgument>(ctx, "the run needs " + bytes_text(need) +
                                        " for its state buffers, over the cap of " +
                                        bytes_text(cap.bytes) + " from " +
                                        memory_cap_source(cap, max_memory_mb));
    }
    return cap.bytes;
}

// Whether a run takes its prefix snapshot. `snapshot` = the snapshot's bytes;
// `peak_with` = the state bytes the run holds at once if it takes it. The
// caller has already ruled out observed runs and single shots.
inline bool take_prefix_snapshot(PrefixReuse mode, std::uint64_t snapshot,
                                 std::uint64_t peak_with,
                                 std::uint64_t max_memory_mb) {
    const bool within_cap = peak_with <= resolve_memory_cap_bytes(max_memory_mb);
    switch (mode) {
        case PrefixReuse::Off:
            return false;
        case PrefixReuse::Manual:
            return within_cap;
        case PrefixReuse::Hardware: {
            if (!within_cap) return false;
            const std::uint64_t available = hw::available_memory_bytes();
            return available != 0 && snapshot <= available / 2;
        }
    }
    return false;
}

// =============================================================================
// RunBudget - memory a run may still take while it runs
// =============================================================================
// Created once the checks before the first gate have passed, with the cap
// they used. What the run holds is kept in three parts, so updating one never
// erases another:
//
//   held      buffers of fixed size the run keeps: its states, a snapshot,
//             chains it is not evolving right now
//   state     the state being evolved, when its size moves (an MPS chain)
//   retained  copies the run keeps to its end (states an observer captured)
//
// Everything the run allocates beyond them is checked BEFORE it is
// allocated. Going over is a RuntimeFailure naming the allocation, its size
// and the cap. The cost is arithmetic, with no system call per charge.
class RunBudget {
public:
    // entry_point must outlive the budget; the run's own name, a literal.
    RunBudget(std::uint64_t cap_bytes, std::uint64_t held_bytes, const char* entry_point)
        : cap_(cap_bytes), held_(held_bytes), entry_point_(entry_point) {}

    RunBudget(const RunBudget&) = delete;
    RunBudget& operator=(const RunBudget&) = delete;

    std::uint64_t cap() const noexcept { return cap_; }
    std::uint64_t live() const noexcept {
        return saturating_add(saturating_add(held_, state_), retained_);
    }

    // Whether `extra_bytes` more would stay within the cap. The cheap test a
    // hot path makes before it builds the arguments of check_peak.
    bool fits(std::uint64_t extra_bytes) const noexcept {
        return saturating_add(live(), extra_bytes) <= cap_;
    }

    // A transient allocation of `extra_bytes` on top of what is live: refused
    // if the peak would exceed the cap. `what` names it for the message.
    void check_peak(std::uint64_t extra_bytes, std::string_view what,
                    const std::optional<FailurePoint>& where = {}) const {
        const std::uint64_t peak = saturating_add(live(), extra_bytes);
        if (peak <= cap_) return;
        raise<RuntimeFailure>(entry_point_,
                              std::string(what) + " needs " + bytes_text(extra_bytes) +
                                  " more, which would take the run to " + bytes_text(peak) +
                                  ", over its cap of " + bytes_text(cap_),
                              where);
    }

    // An allocation the run keeps to its end: checked as a peak, then held.
    void retain(std::uint64_t bytes, std::string_view what,
                const std::optional<FailurePoint>& where = {}) {
        check_peak(bytes, what, where);
        retained_ = saturating_add(retained_, bytes);
    }

    void set_held(std::uint64_t bytes) noexcept { held_ = bytes; }
    void set_state(std::uint64_t bytes) noexcept { state_ = bytes; }

private:
    std::uint64_t cap_;
    std::uint64_t held_;
    std::uint64_t state_ = 0;
    std::uint64_t retained_ = 0;
    const char* entry_point_;
};

}  // namespace detail
}  // namespace lindblad
