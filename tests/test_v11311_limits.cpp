// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - how wide a dense state may be, and how much memory a run may use.
//
// 1.1.31.0 added QubitLimit (Enforce keeps the ceilings sized for a
// workstation, Lift raises them to what a 64-bit byte count can address), a
// memory cap every backend answers to (a caller's figure, the machine's
// reading when that is 0, nothing under NO_MEMORY_CAP), and a run budget that
// charges an MPS chain's growth and its dense fallbacks before they are
// allocated. Lifting a limit never skips the memory check, and that is what
// lets every test here reach a register far too large to allocate: a small cap
// refuses it first, so nothing is ever asked of the operating system.

#include <gtest/gtest.h>

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/detail/memory_budget.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/hw_info.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "v11311_helpers.hpp"

#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <variant>

using namespace lindblad;

namespace {

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;

std::string bytes_text(std::uint64_t bytes) {
    std::string out = std::to_string(bytes) + " bytes";
    if (bytes >= kMiB) out += " (" + std::to_string(bytes >> 20) + " MiB)";
    return out;
}

// Bytes in 2^n complex amplitudes.
std::uint64_t state_bytes(int n) { return (std::uint64_t{1} << n) * sizeof(Complex128); }

QuantumCircuit wide(int n) {
    QuantumCircuit qc(n);
    qc.h(0);
    return qc;
}

// A refusal before the first gate: exactly InvalidArgument from `entry_point`,
// whose message is `entry_point: what`, with no position and no record.
void expect_refused(const std::string& entry_point, const std::string& what,
                    const std::function<void()>& run) {
    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<InvalidArgument>(run);
    EXPECT_EQ(detail::failed_run_stores(), stores) << "a refusal left a failed-run record";
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), entry_point);
    EXPECT_EQ(std::string(e->what()), entry_point + ": " + what);
    EXPECT_FALSE(e->where().has_value());
}

std::string over_cap(std::uint64_t need, std::uint64_t cap_mb) {
    return "the run needs " + bytes_text(need) + " for its state buffers, over the cap of " +
           bytes_text(cap_mb * kMiB) + " from max_memory_mb = " + std::to_string(cap_mb);
}

}  // namespace

// =============================================================================
// QubitLimit on the statevector
// =============================================================================

TEST(V11311QubitLimit, TheDefaultsAndTheLiftedCeilings) {
    EXPECT_EQ(StatevectorSimulator::Options{}.qubit_limit, QubitLimit::Enforce);
    EXPECT_EQ(MPSSimulator{}.qubit_limit, QubitLimit::Enforce);
    EXPECT_EQ(backends::LocalBackend::Config{}.qubit_limit, QubitLimit::Enforce);
    EXPECT_EQ(max_statevector_qubits(QubitLimit::Enforce), ENFORCED_MAX_QUBITS);
    EXPECT_EQ(max_statevector_qubits(QubitLimit::Lift), LIFTED_MAX_QUBITS);
    EXPECT_EQ(max_mps_dense_qubits(QubitLimit::Enforce), ENFORCED_MPS_DENSE_MAX_QUBITS);
    EXPECT_EQ(max_mps_dense_qubits(QubitLimit::Lift), LIFTED_MPS_DENSE_MAX_QUBITS);
    // The lifted statevector ceiling is the widest whose byte count, 16 * 2^n,
    // a 64-bit count holds: one more qubit and it would not.
    EXPECT_EQ(LIFTED_MAX_QUBITS + 1 + std::bit_width(sizeof(Complex128)) - 1,
              std::numeric_limits<std::uint64_t>::digits);
}

TEST(V11311QubitLimit, OverTheEnforcedLimitTheRunIsToldWhatLiftsIt) {
    const int n = ENFORCED_MAX_QUBITS + 1;
    expect_refused("StatevectorSimulator::run",
                   "the circuit has " + std::to_string(n) + " qubits, over the statevector limit of " +
                       std::to_string(ENFORCED_MAX_QUBITS) +
                       "; Options::qubit_limit = QubitLimit::Lift raises it to " +
                       std::to_string(LIFTED_MAX_QUBITS),
                   [&] { (void)StatevectorSimulator().run(wide(n), 0, 1); });
}

TEST(V11311QubitLimit, LiftingAdmitsTheWidthAndTheMemoryCheckStillRefuses) {
    // Never a real 64 GiB allocation: the cap refuses the run first.
    const int n = ENFORCED_MAX_QUBITS + 1;
    StatevectorSimulator sim;
    sim.options.qubit_limit = QubitLimit::Lift;
    sim.options.max_memory_mb = 1;
    expect_refused("StatevectorSimulator::run", over_cap(2 * state_bytes(n), 1),
                   [&] { (void)sim.run(wide(n), 0, 1); });
}

TEST(V11311QubitLimit, NothingLiftsTheLiftedCeiling) {
    const int n = LIFTED_MAX_QUBITS + 1;
    StatevectorSimulator sim;
    sim.options.qubit_limit = QubitLimit::Lift;
    sim.options.max_memory_mb = 1;
    // No hint this time: no setting raises it.
    expect_refused("StatevectorSimulator::run",
                   "the circuit has " + std::to_string(n) + " qubits, over the statevector limit of " +
                       std::to_string(LIFTED_MAX_QUBITS),
                   [&] { (void)sim.run(wide(n), 0, 1); });

    const auto e = v11311::thrown<InvalidArgument>([&] { Statevector sv(n, QubitLimit::Lift); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(std::string(e->what()), "Statevector: n_qubits must be in [1, " +
                                          std::to_string(LIFTED_MAX_QUBITS) + "], got " +
                                          std::to_string(n));
}

TEST(V11311QubitLimit, TheLocalBackendFollowsAndPassesItsLimitAndCap) {
    using backends::LocalBackend;
    EXPECT_EQ(LocalBackend().max_qubits(), ENFORCED_MAX_QUBITS);
    LocalBackend::Config cfg;
    cfg.simulator = LocalBackend::SimType::STATEVECTOR;
    cfg.qubit_limit = QubitLimit::Lift;
    cfg.max_memory_mb = 1;
    LocalBackend backend(cfg);
    EXPECT_EQ(backend.max_qubits(), LIFTED_MAX_QUBITS);
    // Both settings reach the backend: the width is admitted and the cap
    // refuses it, by the backend's own name.
    const int n = ENFORCED_MAX_QUBITS + 1;
    expect_refused("StatevectorSimulator::run", over_cap(2 * state_bytes(n), 1),
                   [&] { (void)backend.run(wide(n), 4, 1); });
}

// =============================================================================
// QubitLimit on the MPS dense fallback
// =============================================================================

namespace {

// A four-operand MCX is applied through the dense fallback on the MPS
// backend; CCX and the other three-qubit gates are decomposed instead.
QuantumCircuit dense_route(int n) {
    QuantumCircuit qc(n);
    qc.mcx({0, 1, 2}, 3);
    return qc;
}

const std::string kDenseRefusal =
    ". Otherwise decompose it to 1- and 2-qubit gates or use the statevector or "
    "density-matrix backend (instruction 0: mcx on qubits 0, 1, 2, 3)";

}  // namespace

TEST(V11311QubitLimit, TheMpsDenseFallbackIsRefusedOverItsLimitBeforeTheFirstGate) {
    const int n = ENFORCED_MPS_DENSE_MAX_QUBITS + 1;
    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)MPSSimulator().run(dense_route(n), 8, 0, 1); });
    EXPECT_EQ(detail::failed_run_stores(), stores);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(std::string(e->what()),
              "MPSSimulator::run: mcx is applied through the dense fallback, and " +
                  std::to_string(n) + " qubits exceed the dense-fallback limit (" +
                  std::to_string(ENFORCED_MPS_DENSE_MAX_QUBITS) +
                  "); qubit_limit = QubitLimit::Lift raises it to " +
                  std::to_string(LIFTED_MPS_DENSE_MAX_QUBITS) + kDenseRefusal);
    v11311::expect_point(e->where(), -1, 0, "mcx", {0, 1, 2, 3});
}

TEST(V11311QubitLimit, LiftedTheDenseFallbackIsChargedToTheRunBudget) {
    // Lift admits the width, so the run starts; the dense array is charged to
    // the budget before it is allocated, and a 1 MiB cap refuses it there, as
    // a failure mid-run with its position and a saved record.
    const int n = ENFORCED_MPS_DENSE_MAX_QUBITS + 1;
    const v11311::TempDir dir("dense-lift");
    RunPlan plan;
    plan.options.failed_run_dir = dir.path();
    MPSSimulator sim;
    sim.qubit_limit = QubitLimit::Lift;
    sim.max_memory_mb = 1;
    (void)take_failed_run();
    const auto e = v11311::thrown<RuntimeFailure>([&] { (void)sim.run(dense_route(n), 8, 0, 1, plan); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "MPSSimulator::run");
    v11311::expect_message(*e, {"MPSSimulator::run: the dense fallback for mcx needs ",
                                "over its cap of " + bytes_text(kMiB) +
                                    " (instruction 0: mcx on qubits 0, 1, 2, 3)",
                                "Partial results saved to "});
    v11311::expect_point(e->where(), -1, 0, "mcx", {0, 1, 2, 3});
    const auto record = take_failed_run();
    ASSERT_TRUE(record.has_value());
    EXPECT_TRUE(record->saved_to.has_value()) << record->save_note;
}

TEST(V11311QubitLimit, NothingLiftsTheMpsDenseFallbackPastItsCeiling) {
    const int n = LIFTED_MPS_DENSE_MAX_QUBITS + 1;
    MPSSimulator sim;
    sim.qubit_limit = QubitLimit::Lift;
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)sim.run(dense_route(n), 8, 0, 1); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(std::string(e->what()),
              "MPSSimulator::run: mcx is applied through the dense fallback, and " +
                  std::to_string(n) + " qubits exceed the dense-fallback limit (" +
                  std::to_string(LIFTED_MPS_DENSE_MAX_QUBITS) + "), which no setting raises past " +
                  std::to_string(LIFTED_MPS_DENSE_MAX_QUBITS) +
                  ": the chain is rebuilt by factorising a 2 x 2^(n-1) block, and that is the "
                  "widest the factorisation can address" + kDenseRefusal);

    // The chain refuses the same width when asked for its amplitudes.
    MPSState chain(n);
    chain.qubit_limit = QubitLimit::Lift;
    const auto s = v11311::thrown<InvalidArgument>([&] { (void)chain.to_statevector(); });
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(s->entry_point(), "MPSState::to_statevector");
    v11311::expect_message(*s, {"which no setting raises past " +
                                std::to_string(LIFTED_MPS_DENSE_MAX_QUBITS)});
}

TEST(V11311QubitLimit, TheLocalBackendPassesItsLimitAndCapToTheMpsBackend) {
    using backends::LocalBackend;
    LocalBackend::Config cfg;
    cfg.simulator = LocalBackend::SimType::MPS;
    cfg.qubit_limit = QubitLimit::Lift;
    cfg.max_memory_mb = 1;
    const auto e = v11311::thrown<RuntimeFailure>(
        [&] { (void)LocalBackend(cfg).run(dense_route(ENFORCED_MPS_DENSE_MAX_QUBITS + 1), 4, 1); });
    (void)take_failed_run();
    ASSERT_TRUE(e.has_value()) << "the width was not lifted, or the cap did not arrive";
    EXPECT_EQ(e->entry_point(), "MPSSimulator::run");
    v11311::expect_message(*e, {"over its cap of " + bytes_text(kMiB)});
}

// =============================================================================
// The memory cap and where it comes from
// =============================================================================

TEST(V11311MemoryCap, ACallerFigureIsTheCapInMebibytes) {
    const auto cap = detail::resolve_memory_cap(5);
    EXPECT_EQ(cap.from, detail::MemoryCap::From::Caller);
    EXPECT_EQ(cap.bytes, 5 * kMiB);
    EXPECT_EQ(detail::memory_cap_source(cap, 5), "max_memory_mb = 5");
    // A figure too large to express in bytes saturates instead of wrapping.
    const std::uint64_t huge = std::numeric_limits<std::uint64_t>::max() - 1;
    EXPECT_EQ(detail::resolve_memory_cap(huge).bytes, std::numeric_limits<std::uint64_t>::max());
}

TEST(V11311MemoryCap, NoMemoryCapMeansNone) {
    const auto cap = detail::resolve_memory_cap(NO_MEMORY_CAP);
    EXPECT_EQ(cap.from, detail::MemoryCap::From::NoCap);
    EXPECT_EQ(cap.bytes, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(detail::memory_cap_source(cap, NO_MEMORY_CAP), "NO_MEMORY_CAP");
    EXPECT_NO_THROW(
        (void)detail::require_memory_budget(std::numeric_limits<std::uint64_t>::max(), NO_MEMORY_CAP, "x"));
    StatevectorSimulator sim;
    sim.options.max_memory_mb = NO_MEMORY_CAP;
    EXPECT_NO_THROW((void)sim.run(wide(3), 8, 1));
}

TEST(V11311MemoryCap, ZeroIsTheMachinesReadingAndARunOverItIsRefused) {
    // On every host the suite runs on the machine answers, so the fallback
    // figure is out of reach here (a known gap: no seam is added for it).
    const std::size_t reading = hw::recent_available_memory_bytes();
    ASSERT_GT(reading, 0u) << "this host gives no memory reading";
    const auto cap = detail::resolve_memory_cap(0);
    EXPECT_EQ(cap.from, detail::MemoryCap::From::Machine);
    EXPECT_EQ(detail::memory_cap_source(cap, 0),
              "the memory this machine reports available (max_memory_mb = 0)");
    EXPECT_EQ(StatevectorSimulator::Options{}.max_memory_mb, 0u);

    // 2^45 amplitudes twice over is 1 PiB, over any machine's reading: the
    // automatic cap refuses it before anything is allocated.
    const int n = 45;
    StatevectorSimulator sim;
    sim.options.qubit_limit = QubitLimit::Lift;
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)sim.run(wide(n), 0, 1); });
    ASSERT_TRUE(e.has_value());
    v11311::expect_message(*e, {"StatevectorSimulator::run: the run needs " +
                                    bytes_text(2 * state_bytes(n)) +
                                    " for its state buffers, over the cap of ",
                                " from the memory this machine reports available (max_memory_mb = 0)"});
}

TEST(V11311MemoryCap, TheReadingIsSharedForASecond) {
    const std::size_t a = hw::recent_available_memory_bytes();
    const std::size_t b = hw::recent_available_memory_bytes();
    EXPECT_EQ(a, b) << "two reads in the same second gave two readings";
}

TEST(V11311MemoryCap, TheTableauAnswersToTheCapToo) {
    // Three tableau-sized buffers: the run's own, the one its result already
    // holds, and the sampling slab; each is 2n rows of ceil(2n / 64) + 1 words
    // plus a phase byte per row.
    const int n = 1000;
    const std::uint64_t rows = 2 * n;
    const std::uint64_t words = (rows + 63) / 64 + 1;
    const std::uint64_t need = 3 * (rows * words * sizeof(std::uint64_t) + rows);
    QuantumCircuit qc = wide(n);
    CliffordSimulator::Options tight;
    tight.max_memory_mb = 1;
    expect_refused("CliffordSimulator::run", over_cap(need, 1),
                   [&] { (void)CliffordSimulator(tight).run(qc, 4, 1); });
    CliffordSimulator::Options enough;
    enough.max_memory_mb = 2;
    EXPECT_NO_THROW((void)CliffordSimulator(enough).run(qc, 4, 1));
}

// =============================================================================
// The MPS run budget
// =============================================================================

TEST(V11311RunBudget, AChainThatOutgrowsTheCapFailsMidRunAndKeepsItsWork) {
    // Twenty qubits entangled layer by layer: the chain grows past 1 MiB long
    // before a bond cap of 64 binds. The split that would take it over is
    // refused before its tensors are allocated, as a failure mid-run.
    const int n = 20;
    QuantumCircuit qc(n);
    for (int layer = 0; layer < 12; ++layer) {
        for (int q = 0; q < n; ++q) qc.ry(0.3 + 0.1 * q + layer, q);
        for (int q = layer % 2; q + 1 < n; q += 2) qc.cx(q, q + 1);
    }
    const v11311::TempDir dir("budget");
    RunPlan plan;
    plan.options.failed_run_dir = dir.path();
    MPSSimulator sim;
    sim.max_memory_mb = 1;
    (void)take_failed_run();
    const auto e = v11311::thrown<RuntimeFailure>([&] { (void)sim.run(qc, 64, 0, 1, plan); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "MPSSimulator::run");
    v11311::expect_message(*e, {"MPSSimulator::run: a two-site update needs ",
                                "over its cap of " + bytes_text(kMiB)});
    ASSERT_TRUE(e->where().has_value());
    EXPECT_EQ(e->where()->shot, -1) << "one walk serves the whole run";
    EXPECT_GE(e->where()->instruction, 0);
    EXPECT_EQ(e->where()->gate, "cx");

    const auto record = take_failed_run();
    ASSERT_TRUE(record.has_value());
    ASSERT_TRUE(record->saved_to.has_value()) << record->save_note;
    const FailedRun loaded = load_failed_run(*record->saved_to);
    ASSERT_TRUE(std::holds_alternative<MPSState>(loaded.state));
    EXPECT_EQ(std::get<MPSState>(loaded.state).n_qubits, n) << "the chain as it stood was kept";
}
