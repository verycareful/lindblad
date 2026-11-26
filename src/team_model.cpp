// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/detail/team_model.hpp"

#include "lindblad/detail/hw_reach.hpp"
#include "lindblad/detail/sv_timing.hpp"
#include "lindblad/detail/threading_model.hpp"
#include "lindblad/statevector.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <utility>

namespace lindblad {
namespace detail {

namespace {

// Bytes of one amplitude: a double each for the real and imaginary parts.
constexpr std::size_t kBytesPerAmplitude = 16;

// A state larger than this many times every L3 instance together is timed on
// a window of that size: no pass finds its amplitudes in cache either way.
constexpr std::size_t kWindowOverLlc = 2;

// Where the largest window ends when no L3 size is detected, in qubits.
constexpr int kFallbackWindowQubits = 22;

// Rounds and stretch length of one candidate team's timing. A pass over a
// state held in cache is short, so a stretch is filled to kStretchSeconds; a
// pass over memory lasts longer than that and a stretch is one call.
constexpr int kRounds = 3;
constexpr double kStretchSeconds = 200e-6;

constexpr const char* kContext = "team model";

bool whole(const std::vector<double>& v, std::size_t n) { return v.size() == n; }

// The qubits a window may have at most, from the L3 caches together.
int widest_window_qubits(double llc_total_bytes) {
    if (llc_total_bytes <= 0.0) return kFallbackWindowQubits;
    const double want = static_cast<double>(kWindowOverLlc) * llc_total_bytes;
    int m = 1;
    while (static_cast<double>(kBytesPerAmplitude << m) < want) ++m;
    return m;
}

}  // namespace

const char* sv_class_name(SvClass cls) noexcept {
    switch (cls) {
        case SvClass::Gate: return "Gate";
        case SvClass::Diagonal: return "Diagonal";
        case SvClass::Dense1: return "Dense1";
        case SvClass::Dense2: return "Dense2";
        case SvClass::Dense3: return "Dense3";
        case SvClass::Dense4: return "Dense4";
        case SvClass::Dense5: return "Dense5";
        case SvClass::Dense6: return "Dense6";
    }
    return "?";
}

// =============================================================================
// TeamTable
// =============================================================================

TeamTable::TeamTable(ThreadingCosts costs, int limit)
    : costs_(std::move(costs)),
      limit_(limit),
      cells_(new std::atomic<std::int16_t>[static_cast<std::size_t>(kSvClasses) * kWeightSlots * kAmplitudeBuckets]) {
    llc_bytes_ = static_cast<double>(hw::llc_bytes());
    llc_instances_ = hw::llc_instances();
    threads_per_llc_ = hw::threads_per_llc();
    const std::size_t cells = static_cast<std::size_t>(kSvClasses) * kWeightSlots * kAmplitudeBuckets;
    for (std::size_t i = 0; i < cells; ++i) cells_[i].store(kCellUnknown, std::memory_order_relaxed);
}

std::shared_ptr<TeamTable> TeamTable::measured() {
    using Key = std::pair<std::size_t, std::vector<bool>>;
    static std::mutex mutex;
    static std::map<Key, std::shared_ptr<TeamTable>> tables;
    Key key;
    key.first = hw::available_threads();
    team_processors(key.second);
    std::lock_guard<std::mutex> lock(mutex);
    std::shared_ptr<TeamTable>& slot = tables[key];
    if (!slot)
        slot = std::shared_ptr<TeamTable>(
            new TeamTable(ThreadingCosts{}, static_cast<int>(std::max<std::size_t>(key.first, 1))));
    return slot;
}

std::shared_ptr<TeamTable> TeamTable::from_costs(ThreadingCosts costs) {
    return std::shared_ptr<TeamTable>(
        new TeamTable(std::move(costs), static_cast<int>(std::max<std::size_t>(hw::available_threads(), 1))));
}

double TeamTable::reachable_cache_bytes(int t) const noexcept {
    const std::size_t needed =
        threads_per_llc_ > 0 ? (static_cast<std::size_t>(t) + threads_per_llc_ - 1) / threads_per_llc_
                             : llc_instances_;
    return llc_bytes_ * static_cast<double>(std::min(llc_instances_, std::max<std::size_t>(needed, 1)));
}

int TeamTable::compute(const SvWork& work, std::uint64_t room_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    return costs_.unset() ? compute_measured(work, room_bytes) : compute_from_costs(work);
}

// The cell by timing: the pass on a scratch state of the cell's size at each
// team size, the fastest kept. Called with the table's lock held.
int TeamTable::compute_measured(const SvWork& work, std::uint64_t room_bytes) {
    // Everything below runs on the grain rule with the model set aside: the
    // scratch state is initialised through the team rule too, and a model asked
    // for the cell it is measuring would wait on its own lock.
    ScopedParallelGrain grain_rule(1);
    const int widest = widest_window_qubits(llc_bytes_ * static_cast<double>(llc_instances_));
    const int bucket = work.amplitudes == 0 ? 0 : static_cast<int>(std::bit_width(work.amplitudes)) - 1;
    const int m = std::min(bucket, widest);
    const bool dense = work.cls != SvClass::Gate && work.cls != SvClass::Diagonal;
    const int k = dense ? static_cast<int>(work.cls) - static_cast<int>(SvClass::Dense1) + 1 : 0;
    // A pass the timing cannot run on a state this small has no figure.
    if (m < 1 || (work.cls == SvClass::Diagonal && m < 2) || (dense && m < k)) return kCellNoFigure;
    const std::size_t dim = std::size_t{1} << m;
    if (dense) {
        const bool tiles = k <= kDenseMaxQubits && dim >= (static_cast<std::size_t>(kLanes) << k);
        if (!tiles && k > 2) return kCellNoFigure;
    }
    if ((static_cast<std::uint64_t>(kBytesPerAmplitude) << m) > room_bytes) return kCellRetry;

    std::optional<Statevector> scratch;
    try {
        scratch.emplace(m);
    } catch (const std::bad_alloc&) {
        return kCellNoFigure;
    }
    const CalibrationKernels kernels{scratch->real_parts, scratch->imag_parts, m};
    kernels.fill();

    const std::vector<int> sizes = threading_team_sizes();
    const std::vector<Complex128> matrix = dense ? CalibrationKernels::dft(k) : std::vector<Complex128>{};
    const int weight =
        work.cls == SvClass::Gate ? std::clamp(work.weight_log2, kMinWeightLog2, kMaxWeightLog2) : 0;

    int best_team = 1;
    double best_seconds = std::numeric_limits<double>::infinity();
    for (const int team : sizes) {
        if (team > limit_) break;
        ScopedThreadCap cap(team, kContext);
        unsigned calls = 0;
        auto pass = [&] {
            const unsigned c = calls++;
            switch (work.cls) {
                case SvClass::Gate:
                    if (weight < 0) {
                        kernels.scale();
                    } else {
                        // 2^weight one-qubit gates, on consecutive targets.
                        for (unsigned r = 0; r < (1u << weight); ++r)
                            kernels.h_at(static_cast<int>((c * (1u << weight) + r) % static_cast<unsigned>(m)));
                    }
                    break;
                case SvClass::Diagonal: {
                    const int lo = static_cast<int>(c % static_cast<unsigned>(m - 1));
                    const int span = m - 1 - lo;
                    kernels.cphase_at(lo, lo + 1 +
                                              static_cast<int>((c / static_cast<unsigned>(m - 1)) %
                                                               static_cast<unsigned>(span)));
                    break;
                }
                default: {
                    const std::vector<int> t = moving_targets(k, m, c % static_cast<unsigned>(m));
                    if (dim >= (static_cast<std::size_t>(kLanes) << k) && k <= kDenseMaxQubits)
                        apply_dense(kernels.re, kernels.im, dim, t, matrix);
                    else
                        apply_small_dense(kernels.re, kernels.im, dim, t, matrix);
                    break;
                }
            }
        };
        const double seconds = fastest(kRounds, kStretchSeconds, pass)[0];
        if (seconds < best_seconds) {
            best_seconds = seconds;
            best_team = team;
        }
    }
    return best_team;
}

// The cell from costs alone: the team size with the least region cost plus
// pass time. Called with the table's lock held.
int TeamTable::compute_from_costs(const SvWork& work) {
    const std::size_t n = costs_.team_sizes.size();
    if (n == 0 || llc_bytes_ == 0.0 || llc_instances_ == 0) return kCellNoFigure;
    if (!whole(costs_.region_seconds, n) || !whole(costs_.in_cache.streaming, n)) return kCellNoFigure;
    const double bytes = static_cast<double>(work.amplitudes) * static_cast<double>(kBytesPerAmplitude);

    const bool diagonal = work.cls == SvClass::Diagonal;
    const bool dense = work.cls != SvClass::Gate && !diagonal;
    const int k = dense ? static_cast<int>(work.cls) - static_cast<int>(SvClass::Dense1) : 0;
    const double weight =
        work.cls == SvClass::Gate
            ? std::ldexp(1.0, std::clamp(work.weight_log2, kMinWeightLog2, kMaxWeightLog2))
            : 1.0;
    const double work_amps = static_cast<double>(work.amplitudes) * weight;
    const bool cold_known = whole(costs_.cold_region_seconds, n) && costs_.block_time_ms.has_value();
    const double block_seconds = cold_known ? static_cast<double>(*costs_.block_time_ms) * 1e-3 : 0.0;

    // Seconds per amplitude of this pass class at team size t, or nothing when
    // the costs hold no figure for it.
    auto per_amp = [&](int t) -> std::optional<double> {
        const bool in_cache = bytes <= reachable_cache_bytes(t);
        const std::vector<double>& light_cache =
            whole(costs_.shared.streaming, n) ? costs_.shared.streaming : costs_.in_cache.streaming;
        const std::vector<double>& diag_cache =
            whole(costs_.shared.diagonal, n) ? costs_.shared.diagonal : costs_.in_cache.diagonal;
        const std::vector<double>& light = in_cache ? light_cache : costs_.memory_bound.streaming;
        const std::vector<double>& diag = in_cache ? diag_cache : costs_.memory_bound.diagonal;
        if (dense) {
            const std::vector<double>& moving = costs_.shared.dense[static_cast<std::size_t>(k)];
            const std::vector<double>& arith =
                whole(moving, n) ? moving : costs_.in_cache.dense[static_cast<std::size_t>(k)];
            if (!whole(arith, n) || !whole(light, n)) return std::nullopt;
            return std::max(threading_interpolate(costs_.team_sizes, arith, t), threading_interpolate(costs_.team_sizes, light, t));
        }
        const std::vector<double>& v = diagonal ? diag : light;
        if (!whole(v, n)) return std::nullopt;
        return threading_interpolate(costs_.team_sizes, v, t);
    };

    int best_team = 1;
    double best_cost = std::numeric_limits<double>::infinity();
    const int top = std::min(limit_, costs_.team_sizes.back());
    for (int t = 1; t <= top; ++t) {
        const std::optional<double> a = per_amp(t);
        if (!a) return kCellNoFigure;
        const double pass = work_amps * *a;
        double region = 0.0;
        if (t > 1) {
            // A region that follows a pass longer than the block time finds the
            // team asleep.
            const bool cold = cold_known && pass > block_seconds;
            region = threading_interpolate(costs_.team_sizes, cold ? costs_.cold_region_seconds : costs_.region_seconds, t);
        }
        const double cost = region + pass;
        if (cost < best_cost) {
            best_cost = cost;
            best_team = t;
        }
    }
    return best_team;
}

// =============================================================================
// TeamModel and ScopedTeamModel
// =============================================================================

void TeamModel::fill_report(ExecutionReport& report) const {
    report.team_rule = TeamRule::Auto;
    report.kernel_teams.clear();
    bool modelled = false;
    for (int c = 0; c < kSvClasses; ++c) {
        const Use& u = use_[static_cast<std::size_t>(c)];
        if (u.decisions == 0) continue;
        KernelTeamUse out;
        out.pass_class = sv_class_name(static_cast<SvClass>(c));
        out.decisions = u.decisions;
        out.histogram = u.histogram;
        out.grain_fallbacks = u.fallbacks;
        for (std::size_t t = 0; t < u.histogram.size(); ++t) {
            if (u.histogram[t] == 0) continue;
            if (out.min_team == 0) out.min_team = static_cast<int>(t);
            out.max_team = static_cast<int>(t);
        }
        if (u.fallbacks < u.decisions) modelled = true;
        report.kernel_teams.push_back(std::move(out));
    }
    if (modelled && table_->costs() != nullptr) report.threading_costs = *table_->costs();
}

ScopedTeamModel::ScopedTeamModel(bool automatic, const ThreadingCosts& pinned,
                                 std::uint64_t scratch_room_bytes) {
    if (!automatic) return;
    std::shared_ptr<TeamTable> table =
        pinned.unset() ? TeamTable::measured() : TeamTable::from_costs(pinned);
    model_ = std::make_unique<TeamModel>(std::move(table), scratch_room_bytes);
    previous_ = t_team_model;
    t_team_model = model_.get();
}

ScopedTeamModel::~ScopedTeamModel() {
    if (model_) t_team_model = previous_;
}

void ScopedTeamModel::fill_report(ExecutionReport& report) const {
    if (model_) model_->fill_report(report);
}

}  // namespace detail
}  // namespace lindblad
