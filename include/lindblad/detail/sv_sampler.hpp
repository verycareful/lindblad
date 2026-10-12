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
// StateSampler - many Born-rule draws from one statevector, in two levels
// =============================================================================
// Level one cuts the amplitudes into blocks of `block` consecutive indices and
// keeps the running sum of the block probabilities: prefix[b] is the weight of
// blocks 0..b, so prefix[nb - 1] is the total. Level two is the block itself.
// A draw scales a uniform u in [0, 1) to target = u * total, finds the first
// block whose running sum exceeds the target (binary search over nb entries),
// and walks that block from the running sum before it to the first amplitude
// whose running sum exceeds the target, strictly. This is the rule of
// born_draw.hpp, with the same two fallbacks: an amplitude of weight zero is
// never chosen, and where rounding leaves the walk short of the target the draw
// takes the last amplitude of positive weight in its block, or in the whole
// state when the target equals the total.
//
// The block width follows the number of draws: block = dim / shots rounded down
// to a power of two, between 1 and kMaxBlock. Few draws from a large state then
// keep a table of dim / kMaxBlock entries instead of one entry per amplitude,
// and each draw walks at most kMaxBlock amplitudes; with at least as many draws
// as amplitudes the block is one amplitude and the table is the running sum of
// every probability.
//
// The block sums are computed across a team, each block by one thread in a
// fixed order, and the running sum over the blocks by the calling thread, so
// every draw depends on the state, the block width and u, and not on how many
// threads ran. Draws are independent of one another, so sample_outcomes takes
// the uniforms in shot order from the caller's generator and resolves them
// across a team.
//
// draw() requires what every caller establishes first with
// require_norm_to_sample (validate_physical.hpp): a finite total above zero.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace lindblad {
namespace detail {

class StateSampler {
public:
    // The widest block: the most amplitudes one draw walks.
    static constexpr std::size_t kMaxBlock = 1024;

    // re, im = the amplitudes, dim of each; they must outlive the sampler and
    //          stay unchanged while it is used.
    // shots  = the draws the caller will make (0 counts as 1); sets the width.
    StateSampler(const double* re, const double* im, std::size_t dim, std::size_t shots);

    // The block width for `shots` draws from `dim` amplitudes.
    static std::size_t block_for(std::size_t dim, std::size_t shots) noexcept;

    // Whether sample_outcomes tallies `shots` draws from `dim` amplitudes in
    // an array indexed by outcome (else in a hash map).
    static bool dense_tally(std::size_t dim, std::size_t shots) noexcept;

    // The bytes of the arrays a sampler and sample_outcomes allocate for
    // `shots` draws from `dim` amplitudes: the block sums, one chunk of
    // uniforms and outcomes, and the dense tally when there is one. A run
    // charges this against its memory cap before the first gate.
    static std::uint64_t scratch_bytes(std::size_t dim, std::size_t shots) noexcept;

    double total() const noexcept { return total_; }
    std::size_t dim() const noexcept { return dim_; }

    // The outcome for one uniform u in [0, 1).
    std::size_t draw(double u) const noexcept;

    // Resolves out[i] = draw(u[i]) for i < n across a team.
    void draw_all(const double* u, std::size_t* out, std::size_t n) const;

private:
    const double* re_;
    const double* im_;
    std::size_t dim_;
    std::size_t block_;           // amplitudes per block, a power of two
    std::size_t nb_;              // dim_ / block_
    std::unique_ptr<double[]> prefix_;
    double total_ = 0.0;
    std::size_t last_positive_ = 0;  // the last amplitude of positive weight
};

// Draws `shots` outcomes from `sampler`, the uniforms taken from uniform() in
// shot order, then calls emit(outcome, count) once for every outcome that came
// up, in no particular order. Uniforms are drawn and resolved in chunks of
// kSampleChunk, so memory stays bounded at any shot count. The tally is an
// array indexed by outcome when the state has at most kDenseTallyPerShot
// amplitudes per shot (small states, where most shots are distinct and a
// hash per shot would cost more than the draw), else a hash map.
inline constexpr std::size_t kSampleChunk = std::size_t{1} << 16;
inline constexpr std::size_t kDenseTallyPerShot = 4;

template <class Uniform, class Emit>
void sample_outcomes(const StateSampler& sampler, std::size_t shots, Uniform&& uniform,
                     Emit&& emit) {
    if (shots == 0) return;
    const std::size_t chunk = shots < kSampleChunk ? shots : kSampleChunk;
    std::unique_ptr<double[]> u(new double[chunk]);
    std::unique_ptr<std::size_t[]> out(new std::size_t[chunk]);
    const bool dense = StateSampler::dense_tally(sampler.dim(), shots);
    std::unique_ptr<int[]> counts(dense ? new int[sampler.dim()]() : nullptr);
    std::unordered_map<std::uint64_t, int> tally;
    for (std::size_t done = 0; done < shots;) {
        const std::size_t n = shots - done < chunk ? shots - done : chunk;
        for (std::size_t i = 0; i < n; ++i) u[i] = uniform();
        sampler.draw_all(u.get(), out.get(), n);
        if (dense)
            for (std::size_t i = 0; i < n; ++i) ++counts[out[i]];
        else
            for (std::size_t i = 0; i < n; ++i) ++tally[out[i]];
        done += n;
    }
    if (dense) {
        for (std::size_t k = 0; k < sampler.dim(); ++k)
            if (counts[k] > 0) emit(static_cast<std::uint64_t>(k), counts[k]);
    } else {
        for (const auto& [outcome, count] : tally) emit(outcome, count);
    }
}

}  // namespace detail
}  // namespace lindblad
