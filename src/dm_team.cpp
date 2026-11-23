// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/detail/dm_team.hpp"

#include "lindblad/detail/dm_kernels.hpp"
#include "lindblad/detail/hw_reach.hpp"
#include "lindblad/detail/sv_timing.hpp"
#include "lindblad/detail/threading_model.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <utility>

namespace lindblad {
namespace detail {

namespace {

// Bytes of one entry of ρ.
constexpr std::size_t kBytesPerEntry = 16;

// A matrix larger than this many times every L3 instance together is timed on
// the widest one that fits in it: a pass over a matrix beyond the caches takes
// tens to hundreds of milliseconds, so timing it at each team size would cost
// seconds per cell, and its cell is read from the widest matrix the caches hold.
constexpr std::size_t kWindowOverLlc = 1;

// The widest register timed when no L3 size is detected.
constexpr int kFallbackWindowQubits = 10;

// Rounds and stretch length of one candidate team's timing: a pass over a
// matrix held in cache is short, so a stretch is filled to kStretchSeconds; a
// pass over memory lasts longer than that and a stretch is one call.
constexpr int kRounds = 3;
constexpr double kStretchSeconds = 500e-6;

// The scan over team sizes (ascending) stops after this many in a row that are
// slower than the fastest so far: a pass too small for any team to pay for its
// region is seen to be slower at 2 and 4 threads, and timing 6, 12 and 24 would
// only repeat that at the cost of the longest regions. A pass that scales
// improves at each size until its best, so one slower size (a team that
// straddles two L3 caches) does not end the scan.
constexpr int kStopAfterSlower = 2;

// The registers a timed pass's targets are spread over before they repeat.
constexpr int kTargetVariants = 8;

constexpr const char* kContext = "density-matrix team model";

enum class Kind { Ket, Bra, Superop, Permute, Phase, Trace };

struct ClassInfo {
    Kind kind;
    int k;   // the gate's or channel's qubits, 0 where the pass has none
};

ClassInfo describe(DmClass cls) noexcept {
    const int c = static_cast<int>(cls);
    if (c <= static_cast<int>(DmClass::Ket3)) return {Kind::Ket, c - static_cast<int>(DmClass::Ket1) + 1};
    if (c <= static_cast<int>(DmClass::Bra3)) return {Kind::Bra, c - static_cast<int>(DmClass::Bra1) + 1};
    if (c <= static_cast<int>(DmClass::Superop3))
        return {Kind::Superop, c - static_cast<int>(DmClass::Superop1) + 1};
    if (cls == DmClass::Permute) return {Kind::Permute, 0};
    if (cls == DmClass::Phase) return {Kind::Phase, 0};
    return {Kind::Trace, 0};
}

// The widest register whose matrix the L3 caches together hold.
int widest_window_qubits(double llc_total_bytes) {
    if (llc_total_bytes <= 0.0) return kFallbackWindowQubits;
    const double room = static_cast<double>(kWindowOverLlc) * llc_total_bytes;
    int m = 1;
    while (static_cast<double>(kBytesPerEntry) * std::ldexp(1.0, 2 * (m + 1)) <= room) ++m;
    return m;
}

// The channel U·ρ·U† as a superoperator in the kernels' internal sub-index
// convention: S[(ro·sd + co)·sd² + (ri·sd + ci)] = U[ro,ri] · conj(U[co,ci]).
std::vector<Complex128> superop_of(const std::vector<Complex128>& U, std::size_t sd) {
    const std::size_t sd2 = sd * sd;
    std::vector<Complex128> S(sd2 * sd2);
    for (std::size_t ro = 0; ro < sd; ++ro)
        for (std::size_t co = 0; co < sd; ++co)
            for (std::size_t ri = 0; ri < sd; ++ri)
                for (std::size_t ci = 0; ci < sd; ++ci)
                    S[(ro * sd + co) * sd2 + (ri * sd + ci)] = U[ro * sd + ri] * U[co * sd + ci].conj();
    return S;
}

}  // namespace

const char* dm_class_name(DmClass cls) noexcept {
    switch (cls) {
        case DmClass::Ket1: return "Ket1";
        case DmClass::Ket2: return "Ket2";
        case DmClass::Ket3: return "Ket3";
        case DmClass::Bra1: return "Bra1";
        case DmClass::Bra2: return "Bra2";
        case DmClass::Bra3: return "Bra3";
        case DmClass::Superop1: return "Superop1";
        case DmClass::Superop2: return "Superop2";
        case DmClass::Superop3: return "Superop3";
        case DmClass::Permute: return "Permute";
        case DmClass::Phase: return "Phase";
        case DmClass::Trace: return "Trace";
    }
    return "?";
}

// =============================================================================
// DmTeamTable
// =============================================================================

DmTeamTable::DmTeamTable(ThreadingCosts costs, int limit)
    : costs_(std::move(costs)),
      limit_(limit),
      cells_(new std::atomic<std::int16_t>[static_cast<std::size_t>(kDmClasses) * kDmQubitSlots]) {
    llc_bytes_ = static_cast<double>(hw::llc_bytes());
    llc_instances_ = hw::llc_instances();
    if (!costs_.unset()) analytic_ = TeamTable::from_costs(costs_);
    const std::size_t cells = static_cast<std::size_t>(kDmClasses) * kDmQubitSlots;
    for (std::size_t i = 0; i < cells; ++i) cells_[i].store(kDmCellUnknown, std::memory_order_relaxed);
    uniform_.fill(kDmCellUnknown);
}

std::shared_ptr<DmTeamTable> DmTeamTable::measured() {
    using Key = std::pair<std::size_t, std::vector<bool>>;
    static std::mutex mutex;
    static std::map<Key, std::shared_ptr<DmTeamTable>> tables;
    Key key;
    key.first = hw::available_threads();
    team_processors(key.second);
    std::lock_guard<std::mutex> lock(mutex);
    std::shared_ptr<DmTeamTable>& slot_ = tables[key];
    if (!slot_)
        slot_ = std::shared_ptr<DmTeamTable>(new DmTeamTable(
            ThreadingCosts{}, static_cast<int>(std::max<std::size_t>(key.first, 1))));
    return slot_;
}

std::shared_ptr<DmTeamTable> DmTeamTable::from_costs(ThreadingCosts costs) {
    return std::shared_ptr<DmTeamTable>(new DmTeamTable(
        std::move(costs), static_cast<int>(std::max<std::size_t>(hw::available_threads(), 1))));
}

int DmTeamTable::compute(const DmWork& work, std::uint64_t room_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    return costs_.unset() ? compute_measured(work, room_bytes) : compute_from_costs(work);
}

// The cell from costs alone, through the statevector's analytic model (see
// dm_team.hpp for how a density-matrix pass is read as a statevector pass).
// Called with the table's lock held.
int DmTeamTable::compute_from_costs(const DmWork& work) {
    const ClassInfo info = describe(work.cls);
    const int n = slot(work);
    if (info.k > 0 && n < info.k) return kDmCellNoFigure;
    const std::size_t dim = std::size_t{1} << n;
    const std::size_t elements = dim * dim;
    SvWork sv{elements, elements, SvClass::Gate, 0};
    switch (info.kind) {
        case Kind::Ket:
        case Kind::Bra: sv.cls = dense_class(info.k); break;
        case Kind::Superop: sv.cls = dense_class(2 * info.k); break;
        case Kind::Phase: sv.cls = SvClass::Diagonal; break;
        case Kind::Permute: sv.weight_log2 = 1; break;
        case Kind::Trace:
            sv = SvWork{dim, dim, SvClass::Gate, 2};
            break;
    }
    const int team = analytic_->team(sv, 0);
    return team > 0 ? team : kDmCellNoFigure;
}

// The cell by timing. Every class of a register width shares one team: the
// team under which a representative sweep of passes (a one-qubit gate's ket and
// bra multiplies and its channel's superoperator pass, and the same for two
// qubits) ran fastest on a scratch matrix of that width, at each team size the
// topology suggests. One team for the whole sweep, not one per pass, because a
// circuit's passes follow one another on the same matrix: a pass that finds the
// matrix partitioned differently from the pass before it reads rows another
// core holds, and a matrix that fits the cores' private caches at the full team
// runs several times slower once its passes alternate between team sizes. A
// team chosen per pass from that pass repeated on its own cannot see that cost.
// Called with the table's lock held.
int DmTeamTable::compute_measured(const DmWork& work, std::uint64_t room_bytes) {
    const int n = slot(work);
    if (uniform_[static_cast<std::size_t>(n)] != kDmCellUnknown) return uniform_[static_cast<std::size_t>(n)];

    // Everything below forces its own team through the fixed rule, with the
    // model set aside: a model asked for the cell it is measuring would wait
    // on its own lock.
    ScopedDmForcedTeam forced;
    const int widest = widest_window_qubits(llc_bytes_ * static_cast<double>(llc_instances_));
    const int m = std::min(n, widest);
    if (m < 1) return kDmCellNoFigure;
    const std::size_t dim = std::size_t{1} << m;
    const std::uint64_t bytes = static_cast<std::uint64_t>(kBytesPerEntry) * dim * dim;
    if (bytes > room_bytes) return kDmCellRetry;

    std::optional<DensityMatrix> scratch;
    try {
        scratch.emplace(m);
    } catch (const std::bad_alloc&) {
        return kDmCellNoFigure;
    }
    // Filled serially, as a run's matrix is (DensityMatrix zero-initialises it
    // on the calling thread), with finite values of one size, so no pass meets a
    // denormal; every pass below is unitary, so they stay that size.
    Complex128* const data = scratch->data.data();
    {
        const double v = 1.0 / static_cast<double>(dim);
        for (std::size_t i = 0; i < dim * dim; ++i)
            data[i] = Complex128(v * (1.0 + 0.25 * static_cast<double>(i % 3)),
                                 v * (0.1 * static_cast<double>(i % 5) - 0.2));
    }

    // The gates of the sweep: one and two qubits (two only where the register
    // has them), each with the target sets it cycles through so that
    // consecutive sweeps differ as a circuit's gates do.
    struct Gate {
        std::vector<Complex128> U, S;
        std::size_t sd;
        std::vector<std::vector<std::size_t>> sub_off, bg;
    };
    const int variants = std::min(m, kTargetVariants);
    std::vector<Gate> gates;
    for (int k = 1; k <= std::min(2, m); ++k) {
        Gate g;
        g.sd = std::size_t{1} << k;
        g.U = CalibrationKernels::dft(k);
        g.S = superop_of(g.U, g.sd);
        g.sub_off.resize(static_cast<std::size_t>(variants));
        g.bg.resize(static_cast<std::size_t>(variants));
        for (int v = 0; v < variants; ++v)
            dm_build_tables(m, dim, moving_targets(k, m, static_cast<unsigned>(v)),
                            g.sub_off[static_cast<std::size_t>(v)], g.bg[static_cast<std::size_t>(v)]);
        gates.push_back(std::move(g));
    }
    unsigned calls = 0;
    auto sweep = [&] {
        const std::size_t v = calls++ % static_cast<unsigned>(variants);
        for (const Gate& g : gates) {
            dm_ket_apply_inplace(data, g.U, g.sub_off[v], g.bg[v], g.sd, dim);
            dm_bra_apply_inplace(data, g.U, g.sub_off[v], g.bg[v], g.sd, dim);
            dm_superop_apply_inplace(data, g.S, g.sub_off[v], g.bg[v], g.sd, dim);
        }
    };

    const std::vector<int> sizes = threading_team_sizes();
    int best_team = 1;
    double best_seconds = std::numeric_limits<double>::infinity();
    int slower_in_a_row = 0;
    for (const int team : sizes) {
        if (team > limit_) break;
        ScopedThreadCap cap(team, kContext);
        const double seconds = fastest(kRounds, kStretchSeconds, sweep)[0];
        if (seconds < best_seconds) {
            best_seconds = seconds;
            best_team = team;
            slower_in_a_row = 0;
        } else if (++slower_in_a_row >= kStopAfterSlower) {
            break;
        }
    }
    uniform_[static_cast<std::size_t>(n)] = static_cast<std::int16_t>(best_team);
    return best_team;
}

// =============================================================================
// DmTeamModel and ScopedDmTeam
// =============================================================================

void DmTeamModel::fill_report(ExecutionReport& report) const {
    report.team_rule = TeamRule::Auto;
    report.kernel_teams.clear();
    bool modelled = false;
    for (int c = 0; c < kDmClasses; ++c) {
        const Use& u = use_[static_cast<std::size_t>(c)];
        if (u.decisions == 0) continue;
        KernelTeamUse out;
        out.pass_class = dm_class_name(static_cast<DmClass>(c));
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

ScopedDmTeam::ScopedDmTeam(std::size_t grain, const ThreadingCosts& pinned,
                           std::uint64_t scratch_room_bytes)
    : previous_model_(t_dm_team_model), previous_grain_(t_dm_grain) {
    if (grain > 0) {
        t_dm_grain = grain;
        t_dm_team_model = nullptr;
        return;
    }
    std::shared_ptr<DmTeamTable> table =
        pinned.unset() ? DmTeamTable::measured() : DmTeamTable::from_costs(pinned);
    model_ = std::make_unique<DmTeamModel>(std::move(table), scratch_room_bytes);
    t_dm_grain = 0;
    t_dm_team_model = model_.get();
}

ScopedDmTeam::~ScopedDmTeam() {
    t_dm_team_model = previous_model_;
    t_dm_grain = previous_grain_;
}

void ScopedDmTeam::fill_report(ExecutionReport& report, std::size_t grain) const {
    if (grain > 0) {
        report.team_rule = TeamRule::Grain;
        report.kernel_teams.clear();
        return;
    }
    if (model_) model_->fill_report(report);
}

}  // namespace detail
}  // namespace lindblad
