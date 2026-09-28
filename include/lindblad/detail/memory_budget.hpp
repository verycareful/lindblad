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
// memory_budget - max_memory_mb and the prefix snapshot
// =============================================================================
// max_memory_mb is the most memory the caller gives a dense run, in MiB
// (2^20 bytes), 0 meaning no limit. It caps the buffers whose size grows with
// the register, the states a run holds at the same time, which is where a
// dense run's memory goes; fixed-size bookkeeping is not counted. Each
// simulator counts the states ITS path holds (the statevector simulator two,
// the density-matrix simulator one, plus a prefix snapshot when one is
// taken), refuses up front a run whose unavoidable states exceed the cap, and
// takes a snapshot only when the run with it still fits.

#include "lindblad/detail/validate.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace lindblad {
namespace detail {

// Every size below saturates at the largest uint64_t rather than wrapping, so
// a register too wide to hold at all reads as needing more than any cap.
inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) {
    return (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
               ? std::numeric_limits<std::uint64_t>::max()
               : a * b;
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

// Refuses, through the structural guard, a run whose state buffers need more
// than the caller's cap. `need` = the bytes the run cannot do without.
inline void require_memory_budget(std::uint64_t need, std::uint64_t max_memory_mb,
                                  const char* ctx) {
    if (max_memory_mb == 0) return;
    const std::uint64_t cap = mib_to_bytes(max_memory_mb);
    check_require(need <= cap, ctx,
                  "the run needs " + std::to_string(need >> 20) +
                      " MiB for its state buffers, over max_memory_mb = " +
                      std::to_string(max_memory_mb));
}

// Whether a run takes its prefix snapshot. `snapshot` = the snapshot's bytes;
// `peak_with` = the state bytes the run holds at once if it takes it. The
// caller has already ruled out observed runs and single shots.
inline bool take_prefix_snapshot(PrefixReuse mode, std::uint64_t snapshot,
                                 std::uint64_t peak_with,
                                 std::uint64_t max_memory_mb) {
    const bool within_cap =
        max_memory_mb == 0 || peak_with <= mib_to_bytes(max_memory_mb);
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

}  // namespace detail
}  // namespace lindblad
