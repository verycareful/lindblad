// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.30.2 - a RESET is a collapse in every shot, and the shared start of a
// per-shot run is computed once.
//
// The statevector and MPS simulators treated a RESET on a qubit nothing had
// measured as deterministic and sampled every shot from one forward pass, so
// one collapse served all of them (V11301ResetSampling pins that). A RESET
// that can change the state now sends a run shot by shot on the statevector,
// MPS and Clifford backends; the density-matrix backend applies it as a
// channel, which one pass describes exactly. A RESET on a qubit known to be
// |0> cannot change the state and keeps the one-pass path, by one rule
// (detail::trivial_resets) that the transpiler's RemoveResetInZeroState now
// shares: that pass once treated a conditioned RESET as leaving its qubit
// known |0>, and no preset level runs it any more, since it cannot know the
// state a run starts from.
//
// A circuit with no MEASURE samples the whole register on every backend, the
// Clifford one included, and a RESET-only circuit does so once per shot.
//
// A per-shot run repeats in every shot everything before its first MEASURE,
// RESET or conditioned instruction. That stretch draws nothing, so an
// unobserved run of more than one shot computes it once and starts every
// shot from a copy: the counts for a seed are exactly those of a rerun, which
// is what the reuse tests compare. The dense backends take the copy under
// PrefixReuse and max_memory_mb, the budget both now honour.

#include <gtest/gtest.h>

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/trivial_resets.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/transpiler.hpp"
#include "lindblad/types.hpp"
#include "v11311_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

using Counts = std::unordered_map<std::string, int>;
using GT = Instruction::GateType;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kSlack = 64.0;

// A seeded run's chance of failing a statistical bound when the library is
// right.
constexpr double kFailureProbability = 1e-9;

std::size_t index_of(const std::string& key) {
    std::size_t i = 0;
    for (char ch : key) i = (i << 1) | static_cast<std::size_t>(ch == '1');
    return i;
}

double tvd(const Counts& counts, const std::vector<double>& p) {
    double shots = 0.0;
    for (const auto& kv : counts) shots += kv.second;
    std::vector<bool> seen(p.size(), false);
    double sum = 0.0;
    for (const auto& [key, n] : counts) {
        const std::size_t i = index_of(key);
        seen[i] = true;
        sum += std::abs(static_cast<double>(n) / shots - p[i]);
    }
    for (std::size_t i = 0; i < p.size(); ++i)
        if (!seen[i]) sum += p[i];
    return 0.5 * sum;
}

// E|p_hat - p| <= sqrt(p (1 - p) / N) per outcome, plus McDiarmid's
// sqrt(ln(1 / delta) / (2N)) since one shot moves the distance by at most 1/N.
double tvd_bound(const std::vector<double>& p, int shots) {
    const double N = static_cast<double>(shots);
    double expected = 0.0;
    for (double x : p) expected += std::sqrt(std::max(0.0, x * (1.0 - x)) / N);
    return 0.5 * expected + std::sqrt(std::log(1.0 / kFailureProbability) / (2.0 * N));
}

class ShotLog final : public Observer {
public:
    void observe(const ObservationContext& ctx) override { shots.push_back(ctx.shot); }
    std::vector<int> shots;
};

void add_condition(QuantumCircuit& qc, int clbit, int value) {
    qc.instructions.back().condition_clbit = clbit;
    qc.instructions.back().condition_value = value;
}

}  // namespace

// =============================================================================
// Which RESETs cannot change the state
// =============================================================================

TEST(V11302TrivialResets, AResetIsTrivialExactlyWhileItsQubitIsKnownZero) {
    QuantumCircuit qc(2, 1);
    qc.reset(0);        // 0: fresh qubit, trivial
    qc.h(0);            // 1
    qc.reset(0);        // 2: after a gate, not trivial
    qc.reset(0);        // 3: after an unconditioned reset, trivial
    qc.measure(0, 0);   // 4
    qc.reset(0);        // 5: a measurement ends the knowledge, not trivial
    qc.barrier({1});    // 6
    qc.reset(1);        // 7: a barrier does not, trivial
    const std::vector<bool> want = {true, false, false, true, false, false, false, true};
    EXPECT_EQ(detail::trivial_resets(qc, detail::zero_at_start(InitialState(), 2)), want);
}

TEST(V11302TrivialResets, AConditionedResetKeepsKnowledgeOnlyIfItHadIt) {
    // It may not run, so it cannot make a qubit known |0>; on a qubit already
    // known |0> it changes nothing either way.
    QuantumCircuit dirty(1, 1);
    dirty.x(0).reset(0);
    add_condition(dirty, 0, 1);
    dirty.reset(0);
    EXPECT_EQ(detail::trivial_resets(dirty, detail::zero_at_start(InitialState(), 1)),
              (std::vector<bool>{false, false, false}));

    QuantumCircuit clean(1, 1);
    clean.reset(0);
    add_condition(clean, 0, 1);
    clean.reset(0);
    EXPECT_EQ(detail::trivial_resets(clean, detail::zero_at_start(InitialState(), 1)),
              (std::vector<bool>{true, true}));
}

TEST(V11302TrivialResets, TheInitialStateDecidesWhatStartsAtZero) {
    EXPECT_EQ(detail::zero_at_start(InitialState(), 3), (std::vector<bool>{true, true, true}));
    // Basis index 0b101: qubits 0 and 2 start at |1>.
    EXPECT_EQ(detail::zero_at_start(InitialState::basis(5), 3),
              (std::vector<bool>{false, true, false}));
    const auto supplied = std::make_shared<const Statevector>(3);
    EXPECT_EQ(detail::zero_at_start(InitialState::from(supplied), 3),
              (std::vector<bool>{false, false, false}));
}

// =============================================================================
// Which path a RESET sends a run down
// =============================================================================

TEST(V11302ResetRouting, ALeadingResetKeepsTheSinglePass) {
    // reset, then H, then a terminal MEASURE. The single pass skips the
    // MEASURE, so the returned state is |+>; a per-shot run returns a
    // collapsed trajectory. After an H the RESET is no longer trivial, and the
    // same check shows the per-shot path.
    const int shots = 64;
    QuantumCircuit leading(1, 1);
    leading.reset(0).h(0).measure(0, 0);
    QuantumCircuit late(1, 1);
    late.h(0).reset(0).h(0).measure(0, 0);

    StatevectorSimulator sv;
    const auto a = sv.run(leading, shots, 3);
    EXPECT_NEAR(a.final_state.amplitudes()[1].norm_sq(), 0.5, kSlack * kEps)
        << "the returned state was collapsed, so the run went shot by shot";
    const auto b = sv.run(late, shots, 3);
    const double w1 = b.final_state.amplitudes()[1].norm_sq();
    EXPECT_TRUE(w1 < kSlack * kEps || w1 > 1.0 - kSlack * kEps)
        << "a RESET after an H must send the run shot by shot";

    MPSSimulator mps;
    const auto m = mps.run(leading, 4, shots, 3);
    EXPECT_NEAR(m.final_state.probabilities_single(0)[1], 0.5, kSlack * kEps);

    CliffordSimulator clifford;
    EXPECT_EQ(clifford.run(leading, shots, 3).final_state.expectation_pauli("X"), 1)
        << "the terminal path returns the unmeasured |+>";
    EXPECT_EQ(clifford.run(late, shots, 3).final_state.expectation_pauli("X"), 0)
        << "the general path returns a collapsed trajectory";
}

TEST(V11302ResetRouting, AResetWithNoMeasureIsACollapseInEveryShot) {
    // ry(theta) on qubit 0 puts it in |1> with probability sin^2(theta/2), the
    // CX copies it onto qubit 1, and the RESET of qubit 0 collapses qubit 1
    // onto that outcome, afresh in every shot. With no MEASURE the whole
    // register is sampled, qubit 0 rightmost: "00" with 1 - p, "10" with p.
    // The density-matrix backend, where RESET is a channel, is the control.
    const int shots = 20000;
    const double theta = PI / 3.0;
    const double p1 = std::pow(std::sin(theta / 2.0), 2);
    const std::vector<double> want = {1.0 - p1, 0.0, p1, 0.0};
    QuantumCircuit qc(2);
    qc.ry(theta, 0).cx(0, 1).reset(0);

    StatevectorSimulator sv;
    const auto s = sv.run(qc, shots, 11);
    EXPECT_LT(tvd(s.counts, want), tvd_bound(want, shots)) << "statevector";

    MPSSimulator mps;
    EXPECT_LT(tvd(mps.run(qc, 4, shots, 11).counts, want), tvd_bound(want, shots)) << "MPS";

    DensityMatrixSimulator dm;
    const auto d = dm.run(qc, NoiseModel{}, shots, 11);
    EXPECT_LT(tvd(d.counts, want), tvd_bound(want, shots)) << "density matrix";

    // Clifford takes H for the rotation, a stabilizer gate: p = 1/2.
    QuantumCircuit cq(2);
    cq.h(0).cx(0, 1).reset(0);
    const std::vector<double> half = {0.5, 0.0, 0.5, 0.0};
    CliffordSimulator clifford;
    EXPECT_LT(tvd(clifford.run(cq, shots, 11).counts, half), tvd_bound(half, shots))
        << "Clifford";
}

// =============================================================================
// The shared start of a per-shot run
// =============================================================================

namespace {

// A prefix of real work, then a mid-circuit measurement with gates after it,
// so every backend runs shot by shot. Only stabilizer gates, so the Clifford
// backend runs it too.
QuantumCircuit prefix_circuit() {
    QuantumCircuit qc(3, 3);
    qc.h(0).cx(0, 1).s(2).h(2).cx(1, 2);
    qc.measure(0, 0);
    qc.h(0).cx(0, 2);
    qc.measure(0, 1).measure(2, 2);
    return qc;
}

RunPlan watched(std::shared_ptr<ShotLog> log, Anchor anchor) {
    RunPlan plan;
    plan.observations.observe(std::move(anchor), std::move(log));
    return plan;
}

}  // namespace

TEST(V11302PrefixReuse, TheSharedStartLeavesTheSeededCountsOfARerun) {
    const int shots = 300;
    const QuantumCircuit qc = prefix_circuit();

    for (const PrefixReuse mode : {PrefixReuse::Manual, PrefixReuse::Hardware}) {
        SCOPED_TRACE(to_string(mode));
        StatevectorSimulator reuse, rerun;
        reuse.options.prefix_reuse = mode;
        rerun.options.prefix_reuse = PrefixReuse::Off;
        const auto a = reuse.run(qc, shots, 17);
        const auto b = rerun.run(qc, shots, 17);
        EXPECT_EQ(a.counts, b.counts) << "statevector";

        DensityMatrixSimulator dreuse, drerun;
        dreuse.options.prefix_reuse = mode;
        drerun.options.prefix_reuse = PrefixReuse::Off;
        const auto c = dreuse.run(qc, NoiseModel{}, shots, 17);
        const auto d = drerun.run(qc, NoiseModel{}, shots, 17);
        EXPECT_EQ(c.counts, d.counts) << "density matrix";
    }

    // The MPS and Clifford backends reuse whenever no observer is attached;
    // an observer makes the same run rerun every shot.
    MPSSimulator mps;
    const auto m_reuse = mps.run(qc, 8, shots, 17);
    const auto m_rerun =
        mps.run(qc, 8, shots, 17, watched(std::make_shared<ShotLog>(), Anchor::at_end()));
    EXPECT_EQ(m_reuse.counts, m_rerun.counts) << "MPS";

    CliffordSimulator clifford;
    const auto c_reuse = clifford.run(qc, shots, 17);
    const auto c_rerun =
        clifford.run(qc, shots, 17, watched(std::make_shared<ShotLog>(), Anchor::at_end()));
    EXPECT_EQ(c_reuse.counts, c_rerun.counts) << "Clifford";
}

TEST(V11302PrefixReuse, AnObservedRunStillFiresOncePerShot) {
    // An anchor inside the shared stretch fires in every shot with that
    // shot's index, on every backend.
    const int shots = 5;
    const QuantumCircuit qc = prefix_circuit();
    std::vector<int> every(shots);
    for (int s = 0; s < shots; ++s) every[static_cast<std::size_t>(s)] = s;

    auto sv_log = std::make_shared<ShotLog>();
    StatevectorSimulator().run(qc, shots, 5, watched(sv_log, Anchor::after_instruction(1)));
    EXPECT_EQ(sv_log->shots, every) << "statevector";

    auto dm_log = std::make_shared<ShotLog>();
    DensityMatrixSimulator().run(qc, NoiseModel{}, shots, 5,
                                 watched(dm_log, Anchor::after_instruction(1)));
    EXPECT_EQ(dm_log->shots, every) << "density matrix";

    auto mps_log = std::make_shared<ShotLog>();
    MPSSimulator().run(qc, 8, shots, 5, watched(mps_log, Anchor::after_instruction(1)));
    EXPECT_EQ(mps_log->shots, every) << "MPS";

    auto cl_log = std::make_shared<ShotLog>();
    CliffordSimulator().run(qc, shots, 5, watched(cl_log, Anchor::after_instruction(1)));
    EXPECT_EQ(cl_log->shots, every) << "Clifford";
}

// =============================================================================
// max_memory_mb and the snapshot setting
// =============================================================================

TEST(V11302MemoryBudget, TheDefaultsAndNames) {
    EXPECT_EQ(StatevectorSimulator().options.prefix_reuse, PrefixReuse::Hardware);
    EXPECT_EQ(DensityMatrixSimulator().options.prefix_reuse, PrefixReuse::Hardware);
    EXPECT_EQ(StatevectorSimulator().options.max_memory_mb, 0u);
    EXPECT_EQ(DensityMatrixSimulator().options.max_memory_mb, 0u);
    EXPECT_STREQ(to_string(PrefixReuse::Hardware), "Hardware");
    EXPECT_STREQ(to_string(PrefixReuse::Manual), "Manual");
    EXPECT_STREQ(to_string(PrefixReuse::Off), "Off");
}

namespace {

// Bytes in a dense array of 2^n complex amplitudes.
std::uint64_t state_bytes(int n) { return (std::uint64_t{1} << n) * sizeof(Complex128); }

// A byte count as the refusal words it: exact, with the MiB beside it.
std::string bytes_text(std::uint64_t bytes) {
    return std::to_string(bytes) + " bytes (" + std::to_string(bytes >> 20) + " MiB)";
}

// A run whose state buffers exceed the cap is refused before the first gate:
// exactly InvalidArgument from the run, naming both figures and where the cap
// came from, with no position and no failed-run record.
void expect_over_cap(const std::string& entry_point, std::uint64_t need, std::uint64_t cap_mb,
                     const std::function<void()>& run) {
    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<InvalidArgument>(run);
    EXPECT_EQ(detail::failed_run_stores(), stores) << "a refusal left a failed-run record";
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), entry_point);
    EXPECT_EQ(std::string(e->what()),
              entry_point + ": the run needs " + bytes_text(need) +
                  " for its state buffers, over the cap of " + bytes_text(cap_mb << 20) +
                  " from max_memory_mb = " + std::to_string(cap_mb));
    EXPECT_FALSE(e->where().has_value());
}

}  // namespace

TEST(V11302MemoryBudget, AStatevectorRunCountsTwoStates) {
    // 16 qubits: 2^16 amplitudes of 16 bytes, one MiB per state, two held.
    const int n = 16;
    QuantumCircuit qc(n);
    qc.h(0).measure_all();
    StatevectorSimulator tight;
    tight.options.max_memory_mb = 1;
    expect_over_cap("StatevectorSimulator::run", 2 * state_bytes(n), 1,
                    [&] { (void)tight.run(qc, 8, 1); });

    StatevectorSimulator enough;
    enough.options.max_memory_mb = 2;
    EXPECT_NO_THROW((void)enough.run(qc, 8, 1));
}

TEST(V11302MemoryBudget, ADensityMatrixRunCountsOneMatrix) {
    // 9 qubits: 4^9 entries of 16 bytes, four MiB.
    QuantumCircuit qc(9);
    qc.h(0).measure_all();
    DensityMatrixSimulator tight;
    tight.options.max_memory_mb = 3;
    expect_over_cap("DensityMatrixSimulator::run", state_bytes(2 * 9), 3,
                    [&] { (void)tight.run(qc, NoiseModel{}, 8, 1); });
    DensityMatrixSimulator enough;
    enough.options.max_memory_mb = 4;
    EXPECT_NO_THROW((void)enough.run(qc, NoiseModel{}, 8, 1));
}

TEST(V11302MemoryBudget, ASnapshotThatWouldNotFitIsNotTaken) {
    // 8 qubits: one MiB per density matrix. Under a one-MiB cap the run fits
    // and the snapshot, a second matrix, does not: the run goes ahead without
    // it, and its counts are a rerun's.
    QuantumCircuit qc(8, 8);
    qc.h(0).cx(0, 1).h(2);
    qc.measure(0, 0).x(0).measure(0, 1).measure(2, 2);
    DensityMatrixSimulator capped, rerun;
    capped.options.max_memory_mb = 1;
    capped.options.prefix_reuse = PrefixReuse::Manual;
    rerun.options.prefix_reuse = PrefixReuse::Off;
    const auto a = capped.run(qc, NoiseModel{}, 64, 19);
    const auto b = rerun.run(qc, NoiseModel{}, 64, 19);
    EXPECT_EQ(a.counts, b.counts);
}

TEST(V11302MemoryBudget, TheLocalBackendPassesTheCapThrough) {
    backends::LocalBackend::Config cfg;
    cfg.simulator = backends::LocalBackend::SimType::STATEVECTOR;
    cfg.max_memory_mb = 1;
    backends::LocalBackend backend(cfg);
    QuantumCircuit qc(16);
    qc.h(0).measure_all();
    // The backend the run reaches refuses it, by its own name.
    expect_over_cap("StatevectorSimulator::run", 2 * state_bytes(16), 1,
                    [&] { (void)backend.run(qc, 8, 1); });
}

#if defined(__linux__) || defined(_WIN32)
TEST(V11302MemoryBudget, TheMachineReportsItsAvailableMemory) {
    EXPECT_GT(hw::available_memory_bytes(), 0u);
}
#endif

// =============================================================================
// The transpiler's RemoveResetInZeroState
// =============================================================================

TEST(V11302ResetPass, AConditionedResetDoesNotLicenseRemovingTheNext) {
    // Qubit 1 is |0>, so clbit 0 reads 0 and the conditioned reset does not
    // run: qubit 0 is still |1> when the unconditioned reset clears it, and
    // clbit 1 must read 0 in every shot. Removing that second reset would leave
    // it reading 1.
    QuantumCircuit qc(2, 2);
    qc.x(0).measure(1, 0).reset(0);
    add_condition(qc, 0, 1);
    qc.reset(0).measure(0, 1);

    const QuantumCircuit out =
        RemoveResetInZeroState().run(DAGCircuit::from_circuit(qc), TranspilationContext{})
            .to_circuit();
    int resets = 0;
    for (const auto& inst : out.instructions) resets += inst.type == GT::RESET;
    EXPECT_EQ(resets, 2);

    StatevectorSimulator sv;
    const auto r = sv.run(out, 64, 23);
    ASSERT_EQ(r.counts.size(), 1u);
    EXPECT_EQ(r.counts.begin()->first, "00");
}

TEST(V11302ResetPass, NoPresetLevelRemovesAReset) {
    // The pass assumes every qubit starts at |0>, which a run from a supplied
    // initial state does not, so no preset runs it.
    QuantumCircuit qc(1, 1);
    qc.reset(0).h(0).measure(0, 0);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        const QuantumCircuit out = transpile(qc, CouplingMap(), {}, level);
        int resets = 0;
        for (const auto& inst : out.instructions) resets += inst.type == GT::RESET;
        EXPECT_EQ(resets, 1);
    }
}
