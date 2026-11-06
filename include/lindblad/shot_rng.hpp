// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include <cstdint>
#include <limits>

namespace lindblad {

// =============================================================================
// ShotRng - the random engine a run's shots draw from
// =============================================================================
// xoshiro256** (Blackman and Vigna), a 64-bit generator with 256 bits of state
// and period 2^256 - 1, which passes the BigCrush and PractRand batteries. It
// meets the standard UniformRandomBitGenerator requirements, so the standard
// distributions take it.
//
// A run gives every chunk of its shots its own engine, started from the run's
// seed and the chunk's index, so the counts do not depend on which thread ran
// which chunk. That makes starting an engine part of the per-chunk cost, and
// this one starts in four splitmix64 steps: std::mt19937_64 has 2.5 KiB of
// state to fill first, which costs more than the cheapest shots themselves.
//
// The four state words are successive splitmix64 outputs from `seed`.
// splitmix64 is a bijection on its counter, so four successive outputs are
// never all zero, the one state xoshiro cannot leave.
class ShotRng {
public:
    using result_type = std::uint64_t;

    static constexpr result_type min() noexcept { return 0; }
    static constexpr result_type max() noexcept {
        return std::numeric_limits<result_type>::max();
    }

    explicit ShotRng(std::uint64_t seed) noexcept {
        for (std::uint64_t& word : s_) word = splitmix64(seed);
    }

    result_type operator()() noexcept {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // One splitmix64 step: advances `state` and returns its mixed output.
    // Also the mixer that turns a run's seed and a chunk's index into the
    // seed of that chunk's engine.
    static constexpr std::uint64_t splitmix64(std::uint64_t& state) noexcept {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

private:
    static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    std::uint64_t s_[4];
};

}  // namespace lindblad
