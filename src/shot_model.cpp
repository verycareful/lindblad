// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/detail/shot_model.hpp"

#include "lindblad/detail/threading_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lindblad {
namespace detail {

namespace {

// The merge is timed on two maps of this many distinct keys, half of them in
// both, so the timing covers both the nodes that move and the counts that add.
constexpr std::uint64_t kMergeTimingKeys = 4096;
constexpr int kMergeTimingRounds = 5;
constexpr int kMergeTimingBits = 20;

// The independent work a team's gain is timed on: each thread sweeps its own
// buffer of kSpeedupDoubles (it stays in the core's L1) kSpeedupSweeps times
// with a multiply-add per element, which lasts a few hundred microseconds on
// one thread. A team is timed kSpeedupRounds times and its fastest round kept.
constexpr int kSpeedupDoubles = 512;
constexpr int kSpeedupSweeps = 4000;
constexpr int kSpeedupRounds = 5;

bool whole(const std::vector<double>& v, std::size_t n) { return v.size() == n; }

#ifdef _OPENMP
// The longest any thread of a team of k took for its own sweeps, in seconds.
// The clock starts after the thread has its buffer, so what is timed is the
// work and not the team's start or the allocation.
double slowest_sweeps(int k) {
    double slowest = 0.0;
#pragma omp parallel num_threads(k) reduction(max : slowest)
    {
        std::vector<double> buffer(kSpeedupDoubles);
        for (int i = 0; i < kSpeedupDoubles; ++i) buffer[static_cast<std::size_t>(i)] = 1.0 + 1e-3 * i;
        double* const x = buffer.data();
        const double begin = omp_get_wtime();
        for (int sweep = 0; sweep < kSpeedupSweeps; ++sweep)
            for (int i = 0; i < kSpeedupDoubles; ++i) x[i] = x[i] * 0.999999 + 1e-6;
        const double end = omp_get_wtime();
        // The buffer is read once so that the sweeps are not removed.
        volatile double keep = x[kSpeedupDoubles / 2];
        (void)keep;
        slowest = end - begin;
    }
    return slowest;
}
#endif

// The seconds one key takes to merge: the fastest of several rounds, since a
// round that a preemption lengthens says nothing about the merge.
double measure_merge_seconds_per_key() {
    using clock = std::chrono::steady_clock;
    double best = std::numeric_limits<double>::infinity();
    for (int round = 0; round < kMergeTimingRounds; ++round) {
        ShotCounts into(kMergeTimingBits);
        ShotCounts from(kMergeTimingBits);
        for (std::uint64_t key = 0; key < kMergeTimingKeys; ++key) {
            into.add_value(key);
            from.add_value(key + kMergeTimingKeys / 2);
        }
        const std::size_t keys = from.size();
        const auto begin = clock::now();
        into.merge(std::move(from), "shot team model");
        const std::chrono::duration<double> took = clock::now() - begin;
        best = std::min(best, took.count() / static_cast<double>(keys));
    }
    return best;
}

// The distinct keys a chunk may hand to the merge: at most its shots, and at
// most the keys its register has.
double keys_per_chunk(const ShotModelInput& in) {
    const double shots = static_cast<double>(in.chunks.chunk);
    if (in.key_bits <= 0 || in.key_bits >= std::numeric_limits<double>::digits) return shots;
    return std::min(shots, std::ldexp(1.0, in.key_bits));
}

}  // namespace

double merge_seconds_per_key() {
    static const double seconds = measure_merge_seconds_per_key();
    return seconds;
}

std::vector<double> team_speedups(const std::vector<int>& sizes) {
#ifdef _OPENMP
    std::vector<double> best(sizes.size(), std::numeric_limits<double>::infinity());
    for (int round = 0; round < kSpeedupRounds; ++round)
        for (std::size_t i = 0; i < sizes.size(); ++i)
            best[i] = std::min(best[i], slowest_sweeps(sizes[i]));
    std::vector<double> gains(sizes.size(), 1.0);
    for (std::size_t i = 0; i < sizes.size(); ++i)
        gains[i] = static_cast<double>(sizes[i]) * best.front() / best[i];
    return gains;
#else
    (void)sizes;
    return {};
#endif
}

int best_shot_team(const ThreadingCosts& costs, double merge_seconds_per_key,
                   const std::vector<double>& speedups, const ShotModelInput& in) {
    const std::size_t n = costs.team_sizes.size();
    if (n == 0 || costs.team_sizes.front() != 1) return 0;
    if (!whole(costs.region_seconds, n) || !whole(costs.in_cache.streaming, n)) return 0;
    if (in.max_team <= 1) return 1;
    const bool independent = !in.sampling && whole(speedups, n);

    const bool cold_known = whole(costs.cold_region_seconds, n);
    const std::vector<double>& region = cold_known ? costs.cold_region_seconds : costs.region_seconds;
    const double units = static_cast<double>(in.work_per_shot) * static_cast<double>(in.chunks.shots);
    const double chunks = static_cast<double>(in.chunks.count);
    const double merge = chunks * keys_per_chunk(in) * merge_seconds_per_key;

    const double one_thread = units * threading_interpolate(costs.team_sizes, costs.in_cache.streaming, 1);
    // The time of the work on a team of k, before the chunks it cannot divide.
    const auto team_work = [&](int k) {
        if (independent) return one_thread / threading_interpolate(costs.team_sizes, speedups, k);
        return units * threading_interpolate(costs.team_sizes, costs.in_cache.streaming, k);
    };

    int best_team = 1;
    double best_cost = one_thread;
    for (int k = 2; k <= in.max_team; ++k) {
        const double rounds = std::ceil(chunks / static_cast<double>(k));
        const double imbalance = rounds * static_cast<double>(k) / chunks;
        const double cost = threading_interpolate(costs.team_sizes, region, k) +
                            team_work(k) * imbalance + merge;
        if (cost < best_cost) {
            best_cost = cost;
            best_team = k;
        }
    }
    return best_team;
}

int modelled_shot_team(const ShotModelInput& in, const ThreadingCosts* pinned) {
    if (in.max_team <= 1) return 1;
    if (pinned != nullptr && !pinned->unset()) return best_shot_team(*pinned, 0.0, {}, in);
    const ThreadingCosts measured =
        measure_threading_costs(ThreadingParts::Region | ThreadingParts::InCache);
    // One measurement per set of team sizes: the sizes follow the threads the
    // calling thread may start.
    static std::mutex mutex;
    static std::map<std::vector<int>, std::vector<double>> kept;
    std::vector<double> speedups;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        auto it = kept.find(measured.team_sizes);
        if (it == kept.end()) it = kept.emplace(measured.team_sizes, team_speedups(measured.team_sizes)).first;
        speedups = it->second;
    }
    return best_shot_team(measured, merge_seconds_per_key(), speedups, in);
}

}  // namespace detail
}  // namespace lindblad
