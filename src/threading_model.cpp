// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/detail/threading_model.hpp"

#include "lindblad/detail/hw_reach.hpp"
#include "lindblad/detail/sv_timing.hpp"
#include "lindblad/detail/thread_cap.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/statevector.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lindblad {
namespace detail {

namespace {

// =============================================================================
// Constants - how long and how often each part is timed
// =============================================================================
// Every time is the fastest of kRounds alternating stretches (sv_timing.hpp).
// A pass over a state held in cache is short, so a stretch is filled to
// kInCacheStretchSeconds; a pass over memory is long enough that a stretch is
// one or two passes.
constexpr int kRounds = 5;
constexpr int kMemoryRounds = 3;
constexpr double kInCacheStretchSeconds = 200e-6;
constexpr double kMemoryStretchSeconds = 500e-6;
constexpr double kRegionStretchSeconds = 100e-6;
constexpr int kRegionSettleRegions = 16;     // regions run before a stretch: the team has just changed size
constexpr int kRegionWarmUpRegions = 64;     // regions run before a stretch's length is found

// A cold sample waits the block time and this margin for every worker to be
// asleep, then times one region; a cold region's cost is the median of
// kColdSamples of them. Measuring it is quick when all the waiting together
// is under kColdQuickSeconds.
constexpr int kColdMarginMs = 1;
constexpr int kColdSamples = 7;
constexpr double kColdQuickSeconds = 0.1;

// 2^13 amplitudes per thread are 128 KiB: in a core's L2 on every part with
// one, so the pass is arithmetic and L1/L2 traffic only.
constexpr int kInCacheQubitsPerThread = 13;

// The memory-bound scratch state holds this many times the L3 caches' total,
// so that no pass finds its amplitudes in cache, and a pass runs over one
// window of it, 1/kMemoryWindows of the state, the windows taken in turn so
// that the pass before the one timed has not left the window's lines behind.
constexpr std::size_t kMemoryScratchOverLlc = 2;
constexpr int kMemoryWindowBits = 3;  // 8 windows

// The scratch state must fit in what the operating system says is available
// with this much to spare, or the memory-bound part is left unmeasured.
constexpr std::size_t kMemoryHeadroom = 2;

// One int per thread, a cache line apart, so the threads of a region do not
// write to a shared line.
constexpr std::size_t kSinkStride = CACHE_LINE_SIZE / sizeof(int);

// Bytes of one amplitude: a double each for the real and imaginary parts.
constexpr std::size_t kBytesPerAmplitude = 16;

constexpr const char* kContext = "threading model";

// What is left of a timed pass once the region that started its team is taken
// out: the pass's own time, which the model adds to the region's cost itself.
// A team of one starts no region. The region is measured on its own and a
// collision with another task can make it longer than the same region was
// inside the pass, so no more than half of a pass's time is taken out: a pass
// is never rated at less than kMarginalFloor of what it was timed at.
constexpr double kMarginalFloor = 0.5;
double marginal_seconds(double pass_seconds, double region_seconds, int team) {
    if (team <= 1) return pass_seconds;
    return std::max(pass_seconds - region_seconds, kMarginalFloor * pass_seconds);
}

// =============================================================================
// Team sizes
// =============================================================================

// Sizes 1, 2 and 4 are the small teams every part has; the rest come from the
// machine (threads_per_llc, smt_width, physical_cores, available_threads).
std::vector<int> sample_sizes() {
    const int all = static_cast<int>(hw::available_threads());
    int domain = static_cast<int>(hw::threads_per_llc());
    if (domain <= 0) domain = all;  // topology unknown: one domain of every thread
    const int smt = static_cast<int>(hw::smt_width());
    std::vector<int> sizes{1, 2, 4, smt > 0 ? domain / smt : 0, domain,
                           static_cast<int>(hw::physical_cores()), all};
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [&](int s) { return s < 1 || s > all; }),
                sizes.end());
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

// =============================================================================
// Region cost
// =============================================================================

// One parallel region of `team` threads, each doing the least a thread can do
// that the compiler cannot remove: one write to its own cache line.
void run_region(int team, int* sink) {
#ifdef _OPENMP
#pragma omp parallel num_threads(team)
    { sink[omp_get_thread_num() * kSinkStride] += 1; }
#else
    (void)team;
    sink[0] += 1;
#endif
}

// A team that has just changed size pays to re-form, so each stretch is
// preceded by a few regions of its size, and a size's stretches alternate with
// the others' across rounds.
std::vector<double> measure_regions(const std::vector<int>& sizes) {
    using Clock = std::chrono::steady_clock;
    const std::size_t slots = static_cast<std::size_t>(sizes.back()) * kSinkStride;
    std::vector<int> sink(slots, 0);
    std::vector<long> calls(sizes.size(), 1);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        for (int w = 0; w < kRegionWarmUpRegions; ++w) run_region(sizes[i], sink.data());
        for (;;) {
            const auto t0 = Clock::now();
            for (long c = 0; c < calls[i]; ++c) run_region(sizes[i], sink.data());
            if (std::chrono::duration<double>(Clock::now() - t0).count() >= kRegionStretchSeconds) break;
            calls[i] *= 2;
        }
    }
    std::vector<double> best(sizes.size(), std::numeric_limits<double>::infinity());
    for (int round = 0; round < kRounds; ++round) {
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            for (int w = 0; w < kRegionSettleRegions; ++w) run_region(sizes[i], sink.data());
            const auto t0 = Clock::now();
            for (long c = 0; c < calls[i]; ++c) run_region(sizes[i], sink.data());
            best[i] = std::min(best[i], std::chrono::duration<double>(Clock::now() - t0).count() /
                                            static_cast<double>(calls[i]));
        }
    }
    return best;
}

// The wait policy's block time in milliseconds, where the runtime reports it:
// only LLVM's libomp does, and its omp.h is the one that defines
// KMP_VERSION_MAJOR.
std::optional<int> read_block_time_ms() {
#if defined(_OPENMP) && defined(KMP_VERSION_MAJOR)
    return kmp_get_blocktime();
#else
    return std::nullopt;
#endif
}

// A finite block time we can wait out.
bool block_time_is_finite(const std::optional<int>& ms) {
    return ms.has_value() && *ms >= 0 && *ms < std::numeric_limits<int>::max();
}

// The wait one cold sample needs, in seconds.
double cold_wait_seconds(int block_ms) { return (block_ms + kColdMarginMs) * 1e-3; }

bool cold_is_quick(const std::vector<int>& sizes, const std::optional<int>& block_ms) {
    if (!block_time_is_finite(block_ms)) return false;
    return static_cast<double>(sizes.size()) * kColdSamples * cold_wait_seconds(*block_ms) <= kColdQuickSeconds;
}

// One region after the team has slept, per team size: the median of
// kColdSamples samples, each timed after waiting out the block time. The
// median, not the fastest: the cost of waking varies from wake to wake, and a
// run meets the typical one.
std::vector<double> measure_cold_regions(const std::vector<int>& sizes, int block_ms) {
    using Clock = std::chrono::steady_clock;
    std::vector<int> sink(static_cast<std::size_t>(sizes.back()) * kSinkStride, 0);
    const auto wait = std::chrono::duration<double>(cold_wait_seconds(block_ms));
    std::vector<double> costs;
    for (const int team : sizes) {
        run_region(team, sink.data());  // the team exists before it is left to sleep
        std::vector<double> samples;
        for (int i = 0; i < kColdSamples; ++i) {
            std::this_thread::sleep_for(wait);
            const auto t0 = Clock::now();
            run_region(team, sink.data());
            samples.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
        }
        std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());
        costs.push_back(samples[samples.size() / 2]);
    }
    return costs;
}

// =============================================================================
// Passes over a state
// =============================================================================

struct DenseMatrices {
    std::array<std::vector<Complex128>, 6> m;
    DenseMatrices() {
        for (int k = 1; k <= 6; ++k) m[k - 1] = CalibrationKernels::dft(k);
    }
};

// What the in-cache parts time on: the first 2^q amplitudes of a scratch state
// of 2^widest, q growing with the team so that every thread has the same
// share.
struct InCacheScratch {
    int widest;
    Statevector state;
    explicit InCacheScratch(int largest_team)
        : widest(kInCacheQubitsPerThread +
                 static_cast<int>(std::bit_width(static_cast<unsigned>(largest_team - 1)))),
          state(widest) {
        CalibrationKernels{state.real_parts, state.imag_parts, widest}.fill();
    }
    CalibrationKernels kernels(int team) {
        const int q = std::min(widest, kInCacheQubitsPerThread +
                                           static_cast<int>(std::bit_width(static_cast<unsigned>(team - 1))));
        return CalibrationKernels{state.real_parts, state.imag_parts, q};
    }
};

void measure_light(const std::vector<int>& sizes, const std::vector<double>& region, ThreadingCosts::Passes& out) {
    InCacheScratch scratch(sizes.back());
    out.streaming.clear();
    out.diagonal.clear();
    for (const int team : sizes) {
        // The cap sets the most threads and a grain of one amplitude lets a
        // pass take them all, whatever its size.
        ScopedThreadCap cap(team, kContext);
        ScopedParallelGrain grain(1);
        const CalibrationKernels k = scratch.kernels(team);
        auto streaming = [&] { k.h(); };
        auto diagonal = [&] { k.cphase(); };
        const auto t = fastest(kRounds, kInCacheStretchSeconds, streaming, diagonal);
        const double n = static_cast<double>(k.dim());
        const std::size_t i = out.streaming.size();
        out.streaming.push_back(marginal_seconds(t[0], region[i], team) / n);
        out.diagonal.push_back(marginal_seconds(t[1], region[i], team) / n);
    }
}

// The same streaming and diagonal passes as measure_light, with the target
// changing from pass to pass: a one-qubit pass on each qubit in turn, a
// controlled phase on each pair of qubits in turn. A circuit's gates do this;
// it is what moves a thread's amplitudes to another thread, and through
// another L3 once the team spans two.
void measure_shared(const std::vector<int>& sizes, const std::vector<double>& region, ThreadingCosts::Passes& out) {
    InCacheScratch scratch(sizes.back());
    out.streaming.clear();
    out.diagonal.clear();
    for (const int team : sizes) {
        ScopedThreadCap cap(team, kContext);
        ScopedParallelGrain grain(1);
        const CalibrationKernels k = scratch.kernels(team);
        unsigned h_calls = 0, cp_calls = 0;
        auto streaming = [&] { k.h_at(static_cast<int>(h_calls++ % static_cast<unsigned>(k.m))); };
        auto diagonal = [&] {
            const unsigned c = cp_calls++;
            const int lo = static_cast<int>(c % static_cast<unsigned>(k.m - 1));
            const int span = k.m - 1 - lo;
            k.cphase_at(lo, lo + 1 + static_cast<int>((c / static_cast<unsigned>(k.m - 1)) % static_cast<unsigned>(span)));
        };
        const auto t = fastest(kRounds, kInCacheStretchSeconds, streaming, diagonal);
        const double n = static_cast<double>(k.dim());
        const std::size_t i = out.streaming.size();
        out.streaming.push_back(marginal_seconds(t[0], region[i], team) / n);
        out.diagonal.push_back(marginal_seconds(t[1], region[i], team) / n);
    }
}

// The dense k-qubit blocks, k = 1 to 6, each timed twice: on one fixed set of
// targets (in_cache.dense, the arithmetic and the traffic of a block whose
// shares stay with their threads) and on a set that changes from block to
// block (shared.dense, what a circuit's blocks do).
void measure_dense(const std::vector<int>& sizes, const std::vector<double>& region,
                   ThreadingCosts::Passes& in_cache, ThreadingCosts::Passes& shared) {
    InCacheScratch scratch(sizes.back());
    const DenseMatrices dft;
    for (auto& row : in_cache.dense) row.clear();
    for (auto& row : shared.dense) row.clear();
    for (const int team : sizes) {
        ScopedThreadCap cap(team, kContext);
        ScopedParallelGrain grain(1);
        const CalibrationKernels k = scratch.kernels(team);
        const std::array<std::vector<int>, 6> targets{k.targets(1), k.targets(2), k.targets(3),
                                                      k.targets(4), k.targets(5), k.targets(6)};
        auto dense = [&](int kk) {
            return [&, kk] { apply_dense(k.re, k.im, k.dim(), targets[kk], dft.m[kk]); };
        };
        // The pool of targets shifted by the call count, so no two consecutive
        // blocks share a target set.
        std::array<unsigned, 6> calls{};
        auto moving = [&](int kk) {
            return [&, kk] {
                std::vector<int> t = targets[kk];
                const int shift = static_cast<int>(calls[kk]++ % static_cast<unsigned>(k.m));
                for (int& q : t) q = (q + shift) % k.m;
                apply_dense(k.re, k.im, k.dim(), t, dft.m[kk]);
            };
        };
        auto d1 = dense(0), d2 = dense(1), d3 = dense(2), d4 = dense(3), d5 = dense(4), d6 = dense(5);
        auto m1 = moving(0), m2 = moving(1), m3 = moving(2), m4 = moving(3), m5 = moving(4), m6 = moving(5);
        const auto t = fastest(kRounds, kInCacheStretchSeconds, d1, d2, d3, d4, d5, d6, m1, m2, m3, m4, m5, m6);
        const double n = static_cast<double>(k.dim());
        const std::size_t j = in_cache.dense[0].size();
        for (int i = 0; i < 6; ++i) {
            in_cache.dense[i].push_back(marginal_seconds(t[i], region[j], team) / n);
            shared.dense[i].push_back(marginal_seconds(t[6 + i], region[j], team) / n);
        }
    }
}

// The scratch state's qubits for the memory-bound part, or 0 when no L3 size
// is detected: the state is kMemoryScratchOverLlc times every L3 instance
// together.
int memory_scratch_qubits() {
    const std::size_t llc = hw::llc_bytes();
    const std::size_t instances = hw::llc_instances();
    if (llc == 0 || instances == 0) return 0;
    const std::size_t want = kMemoryScratchOverLlc * llc * instances;
    int m = 1;
    while ((kBytesPerAmplitude << m) < want) ++m;
    return m;
}

// Why the memory-bound part was not measured, or empty when it can be: the L3
// is not detected, the caller's room is too small, or the operating system
// cannot sure-spare the state. `cap_limited` is true when the caller's room is
// the reason, the one reason a later call with more room can lift.
std::string memory_bound_obstacle(std::uint64_t room_bytes, bool& cap_limited) {
    cap_limited = false;
    const int m = memory_scratch_qubits();
    if (m == 0) return "no L3 size detected";
    const std::uint64_t bytes = static_cast<std::uint64_t>(kBytesPerAmplitude) << m;
    if (room_bytes < bytes) {
        cap_limited = true;
        return "the memory cap leaves " + std::to_string(room_bytes >> 20) + " MiB, the scratch state needs " +
               std::to_string(bytes >> 20) + " MiB";
    }
    const std::size_t room = hw::recent_available_memory_bytes();
    if (room == 0 || room / kMemoryHeadroom < bytes)
        return "the operating system does not report " + std::to_string((bytes * kMemoryHeadroom) >> 20) +
               " MiB available";
    return {};
}

// Streaming and diagonal passes over one window of a state larger than the
// caches. Only these two are rated: a dense block's memory-bound pass is the
// longer of its in-cache pass and the streaming one. Returns false, with `out`
// empty, when the scratch state could not be had.
bool measure_memory_bound(const std::vector<int>& sizes, const std::vector<double>& region,
                          ThreadingCosts::Passes& out) {
    out = {};
    const int m = memory_scratch_qubits();
    if (m == 0) return false;
    try {
        Statevector scratch(m);
        const CalibrationKernels whole{scratch.real_parts, scratch.imag_parts, m};
        whole.fill();
        const int window_qubits = m - kMemoryWindowBits;
        const std::size_t window = std::size_t{1} << window_qubits;
        const std::size_t windows = std::size_t{1} << kMemoryWindowBits;
        for (const int team : sizes) {
            ScopedThreadCap cap(team, kContext);
            ScopedParallelGrain grain(1);
            std::size_t next = 0;
            auto at = [&] {
                const std::size_t w = next;
                next = (next + 1) % windows;
                return CalibrationKernels{scratch.real_parts + w * window, scratch.imag_parts + w * window,
                                          window_qubits};
            };
            auto streaming = [&] { at().h(); };
            auto diagonal = [&] { at().cphase(); };
            const auto t = fastest(kMemoryRounds, kMemoryStretchSeconds, streaming, diagonal);
            const std::size_t i = out.streaming.size();
            out.streaming.push_back(marginal_seconds(t[0], region[i], team) / static_cast<double>(window));
            out.diagonal.push_back(marginal_seconds(t[1], region[i], team) / static_cast<double>(window));
        }
        return true;
    } catch (const std::bad_alloc&) {
        out = {};
        return false;
    }
}

// =============================================================================
// The cache - one measurement per (threads available, processors reachable)
// =============================================================================

struct Entry {
    ThreadingCosts costs;
    unsigned have = 0;       // ThreadingParts measured
    unsigned attempted = 0;  // ThreadingParts tried, including those that came back empty
    // The most room a MemoryBound attempt was refused for lack of, in bytes; a
    // call with more room tries again. Zero when no attempt was refused so.
    std::uint64_t memory_refused_room = 0;
};

// (threads available, processors the team may use, block time in ms or -1)
using Key = std::tuple<std::size_t, std::vector<bool>, int>;

std::mutex& cache_mutex() {
    static std::mutex m;
    return m;
}

std::map<Key, Entry>& cache() {
    static std::map<Key, Entry> c;
    return c;
}

Key current_key() {
    Key key;
    std::get<0>(key) = hw::available_threads();
    team_processors(std::get<1>(key));
    const std::optional<int> block = read_block_time_ms();
    std::get<2>(key) = block.has_value() ? *block : -1;
    return key;
}

bool wants(ThreadingParts parts, ThreadingParts one) {
    return (static_cast<unsigned>(parts) & static_cast<unsigned>(one)) != 0;
}

std::string bad_entries(const char* name, const std::vector<double>& v, std::size_t n) {
    if (v.size() != n) return std::string(" ") + name + " (" + std::to_string(v.size()) + " entries for " +
                              std::to_string(n) + " team sizes)";
    for (const double x : v)
        if (!(x > 0.0) || !std::isfinite(x)) return std::string(" ") + name + " (an entry is not finite and positive)";
    return {};
}

}  // namespace

std::vector<int> threading_team_sizes() { return sample_sizes(); }

std::uint64_t threading_memory_scratch_bytes() {
    const int m = memory_scratch_qubits();
    return m == 0 ? 0 : static_cast<std::uint64_t>(kBytesPerAmplitude) << m;
}

ThreadingCosts measure_threading_costs(ThreadingParts parts, std::uint64_t scratch_room_bytes) {
    std::lock_guard<std::mutex> lock(cache_mutex());
    // Every pass below runs on the grain rule, the caller's team model set
    // aside: a model asked for the costs it is being given would wait on
    // itself, and the scratch states are filled before any part's own scope.
    ScopedParallelGrain grain_rule(1);
    Entry& entry = cache()[current_key()];
    const std::vector<int> sizes = sample_sizes();
    if (entry.costs.team_sizes != sizes) {
        entry = Entry{};
        entry.costs.team_sizes = sizes;
    }
    // The passes are rated with their team's region taken out, so the regions
    // are measured before any of them.
    if (wants(parts, ThreadingParts::InCache | ThreadingParts::Shared | ThreadingParts::Dense |
                         ThreadingParts::MemoryBound))
        parts = parts | ThreadingParts::Region;
    auto todo = [&](ThreadingParts one) {
        return wants(parts, one) && !(entry.attempted & static_cast<unsigned>(one));
    };
    if (todo(ThreadingParts::Region)) {
        entry.costs.block_time_ms = read_block_time_ms();
        entry.costs.region_seconds = measure_regions(sizes);
        entry.attempted |= static_cast<unsigned>(ThreadingParts::Region);
    }
    if (todo(ThreadingParts::ColdRegions)) {
        const std::optional<int> block = read_block_time_ms();
        if (block_time_is_finite(block)) entry.costs.cold_region_seconds = measure_cold_regions(sizes, *block);
        entry.attempted |= static_cast<unsigned>(ThreadingParts::ColdRegions);
    }
    if (todo(ThreadingParts::InCache)) {
        measure_light(sizes, entry.costs.region_seconds, entry.costs.in_cache);
        entry.attempted |= static_cast<unsigned>(ThreadingParts::InCache);
    }
    if (todo(ThreadingParts::Shared)) {
        measure_shared(sizes, entry.costs.region_seconds, entry.costs.shared);
        entry.attempted |= static_cast<unsigned>(ThreadingParts::Shared);
    }
    if (todo(ThreadingParts::Dense)) {
        measure_dense(sizes, entry.costs.region_seconds, entry.costs.in_cache, entry.costs.shared);
        entry.attempted |= static_cast<unsigned>(ThreadingParts::Dense);
    }
    // MemoryBound is tried again when an earlier attempt was refused for lack
    // of room and this call has more of it.
    const unsigned memory_bit = static_cast<unsigned>(ThreadingParts::MemoryBound);
    const bool memory_retry = (entry.attempted & memory_bit) == 0 ||
                              (entry.memory_refused_room != 0 && scratch_room_bytes > entry.memory_refused_room);
    if (wants(parts, ThreadingParts::MemoryBound) && memory_retry) {
        bool cap_limited = false;
        const std::string obstacle = memory_bound_obstacle(scratch_room_bytes, cap_limited);
        entry.costs.memory_bound = {};
        entry.costs.memory_bound_note = obstacle;
        if (obstacle.empty() && !measure_memory_bound(sizes, entry.costs.region_seconds, entry.costs.memory_bound))
            entry.costs.memory_bound_note = "the scratch state could not be allocated";
        if (cap_limited) {
            entry.memory_refused_room = scratch_room_bytes;
        } else {
            entry.memory_refused_room = 0;
            entry.attempted |= memory_bit;
        }
    }
    return entry.costs;
}

bool threading_cold_regions_quick() { return cold_is_quick(sample_sizes(), read_block_time_ms()); }

double threading_interpolate(const std::vector<int>& sizes, const std::vector<double>& values, int t) {
    if (t <= sizes.front()) return values.front();
    if (t >= sizes.back()) return values.back();
    std::size_t hi = 1;
    while (sizes[hi] < t) ++hi;
    const double lo_t = sizes[hi - 1], hi_t = sizes[hi];
    const double f = (static_cast<double>(t) - lo_t) / (hi_t - lo_t);
    return values[hi - 1] + f * (values[hi] - values[hi - 1]);
}

std::string threading_costs_problem(const ThreadingCosts& c) {
    const std::size_t n = c.team_sizes.size();
    if (n == 0) return "team_sizes is empty";
    std::string bad;
    if (c.team_sizes.front() != 1) bad += " team_sizes (the first must be 1)";
    for (std::size_t i = 1; i < n; ++i)
        if (c.team_sizes[i] <= c.team_sizes[i - 1]) {
            bad += " team_sizes (not strictly ascending)";
            break;
        }
    bad += bad_entries("region_seconds", c.region_seconds, n);
    if (c.block_time_ms.has_value() && *c.block_time_ms < 0) bad += " block_time_ms (negative)";
    if (!c.cold_region_seconds.empty()) bad += bad_entries("cold_region_seconds", c.cold_region_seconds, n);
    bad += bad_entries("in_cache.streaming", c.in_cache.streaming, n);
    bad += bad_entries("in_cache.diagonal", c.in_cache.diagonal, n);
    for (std::size_t k = 0; k < c.in_cache.dense.size(); ++k)
        bad += bad_entries(("in_cache.dense[" + std::to_string(k) + "]").c_str(), c.in_cache.dense[k], n);
    const bool shared_empty = c.shared.streaming.empty() && c.shared.diagonal.empty();
    if (!shared_empty) {
        bad += bad_entries("shared.streaming", c.shared.streaming, n);
        bad += bad_entries("shared.diagonal", c.shared.diagonal, n);
    }
    for (std::size_t k = 0; k < c.shared.dense.size(); ++k)
        if (!c.shared.dense[k].empty())
            bad += bad_entries(("shared.dense[" + std::to_string(k) + "]").c_str(), c.shared.dense[k], n);
    const bool memory_empty = c.memory_bound.streaming.empty() && c.memory_bound.diagonal.empty();
    if (!memory_empty) {
        bad += bad_entries("memory_bound.streaming", c.memory_bound.streaming, n);
        bad += bad_entries("memory_bound.diagonal", c.memory_bound.diagonal, n);
    }
    for (const auto& d : c.memory_bound.dense)
        if (!d.empty()) {
            bad += " memory_bound.dense (must stay empty)";
            break;
        }
    return bad;
}

}  // namespace detail
}  // namespace lindblad
