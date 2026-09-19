// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.28.2 test wave - the rebuild from dense amplitudes uses the selected
// kernel (#127).
//
// MPSState::rebuild_from_statevector is the ladder's second call site: every
// dense fallback and every dense seeding factorises through it. It runs the
// kernel the chain's svd_method selects, like the gate path does, so a caller
// who chose one gets it on every split and a caller who chose none gets BDC
// on every split. This file pins that the selection reaches the rebuild, which
// needs the kernels to actually differ.
//
// The instrument is the same as for the gate path: the default BDC and
// EigenJacobi are different algorithms from different libraries and do not
// agree bit for bit on blocks of any size, so two rebuilds of one state under
// the two kernels that agree exactly mean the selection never reached the
// split. The difference is bounded above as well, because at a cap the state
// cannot exceed both rebuilds are exact and two exact factorisations of one
// state agree to rounding. A random 8-qubit state at a cap of 16 gives both
// halves: its site-0 block is 2 x 128, wide enough for the two kernels to take
// different routes, and no cut of 8 qubits has rank above 16.
//
// The last group holds autonne's Jacobi against the default: the rebuild
// under it passes the ladder without a rescue and agrees with BDC to rounding.
// The one-time note that Jacobi is the slower kernel also fires from the
// rebuild, but its latch is per process and any earlier test may have tripped
// it, so that is not pinned here.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace lindblad;

using Z = std::complex<double>;

namespace {

constexpr int kShots = 0;
constexpr std::uint64_t kSeed = 42;
constexpr double kEps = std::numeric_limits<double>::epsilon();

// Slack a backward-stable decomposition is entitled to, in units of
// n * eps: the shape the truncation ladder's verify rung allows, so this suite
// and the ladder cannot disagree about what "correct" means.
constexpr double kSlack = 64.0;

double tol(int n) {
    return kSlack * static_cast<double>(n) * kEps;
}

// The register the whole file works on, and the cap no cut of it can exceed.
constexpr int kQubits = 8;
constexpr int kCap = 1 << (kQubits / 2);

// Widest block side the sequential rebuild of an n-qubit state forms. Site 0
// splits a 2 x 2^(n-1) block; every later block has fewer columns and at most
// 2 * cap rows, and 2 * cap <= 2^(n-1) whenever n >= 3.
int widest_rebuild_side(int n) {
    return 1 << (n - 1);
}

// How far two exact rebuilds of one state may drift apart through rounding
// alone: each of the n - 1 splits is entitled to the verify rung's slack on
// its block, the state norm is 1, and both rebuilds contribute.
double rebuild_tol(int n) {
    return 2.0 * static_cast<double>(n - 1) * tol(widest_rebuild_side(n));
}

// A normalised state with every amplitude drawn independently, so every cut
// has full rank and every split of the rebuild has something to resolve.
std::shared_ptr<const Statevector> random_state(int n, std::uint64_t seed) {
    const std::size_t dim = std::size_t{1} << n;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::vector<Complex128> amps(dim);
    double sum_sq = 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
        const double re = unit(rng);
        const double im = unit(rng);
        amps[k] = Complex128(re, im);
        sum_sq += re * re + im * im;
    }
    const double scale = 1.0 / std::sqrt(sum_sq);
    for (auto& a : amps) a = Complex128(a.real * scale, a.imag * scale);
    auto sv = std::make_shared<Statevector>(n);
    sv->set_amplitudes(amps);
    return sv;
}

// Largest amplitude difference between two states of one width. A global
// phase shows up here, which is correct: both chains factorise the same
// amplitudes, so neither may pick up a phase the other did not.
double max_amplitude_diff(const Statevector& a, const Statevector& b) {
    const auto va = a.amplitudes();
    const auto vb = b.amplitudes();
    if (va.size() != vb.size()) {
        throw std::runtime_error("states cover different register widths");
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < va.size(); ++i) {
        const Z za(va[i].real, va[i].imag);
        const Z zb(vb[i].real, vb[i].imag);
        worst = std::max(worst, std::abs(za - zb));
    }
    return worst;
}

MPSState rebuilt_under(const Statevector& sv, SVDMethod method) {
    MPSState chain(sv.n_qubits, kCap);
    chain.svd_method = method;
    chain.rebuild_from_statevector(sv);
    return chain;
}

// The state a run seeded with `sv` and given nothing to do hands back: the
// rebuild through MPSSimulator::run's own seeding path, and nothing else.
MPSState seeded_under(std::shared_ptr<const Statevector> sv, SVDMethod method) {
    RunPlan plan;
    plan.initial = InitialState::from(sv);
    MPSSimulator sim;
    sim.svd_method = method;
    return sim.run(QuantumCircuit(sv->n_qubits), kCap, kShots, kSeed, plan).final_state;
}

}  // namespace

// =============================================================================
// The selection reaches the rebuild
// =============================================================================

TEST(V11282RebuildKernel, BothKernelsReconstructTheInput) {
    // The anchor for everything below: each rebuild is exact at this cap, so
    // the chain contracts back to the amplitudes it was built from, to the
    // rounding of n - 1 verified splits.
    const auto sv = random_state(kQubits, kSeed);
    for (const SVDMethod method : {SVDMethod::BDC, SVDMethod::EigenJacobi}) {
        const MPSState chain = rebuilt_under(*sv, method);
        EXPECT_EQ(chain.svd_call_count(), static_cast<std::size_t>(kQubits - 1));
        EXPECT_EQ(chain.current_max_bond_dim(), kCap)
            << "a random state saturates the widest cut";
        EXPECT_LT(max_amplitude_diff(chain.to_statevector(), *sv), rebuild_tol(kQubits) / 2.0);
    }
}

TEST(V11282RebuildKernel, RebuildUsesTheSelectedKernel) {
    // Nonzero, or the selection never reached the split and one kernel ran
    // twice. Below rounding, or one of the two is not an exact factorisation.
    const auto sv = random_state(kQubits, kSeed);
    const MPSState bdc = rebuilt_under(*sv, SVDMethod::BDC);
    const MPSState jacobi = rebuilt_under(*sv, SVDMethod::EigenJacobi);
    ASSERT_EQ(bdc.svd_call_count(), jacobi.svd_call_count());

    const double diff = max_amplitude_diff(bdc.to_statevector(), jacobi.to_statevector());
    ::testing::Test::RecordProperty("rebuild_bdc_vs_jacobi_max_amplitude_diff",
                                    std::to_string(diff));
    EXPECT_GT(diff, 0.0)
        << "BDC and Jacobi rebuilt a bit-identical chain: the rebuild is not "
           "reading svd_method, so the same kernel ran twice (#127)";
    EXPECT_LT(diff, rebuild_tol(kQubits))
        << "two exact factorisations of one state disagree beyond rounding";
}

TEST(V11282RebuildKernel, RebuildDefaultsToBdc) {
    // A chain that selected nothing rebuilds exactly as one that selected
    // BDC, and not as one that selected Jacobi. Bit-for-bit, because the
    // default is a value of the same field and not a third route.
    const auto sv = random_state(kQubits, kSeed);
    MPSState untouched(kQubits, kCap);
    ASSERT_EQ(untouched.svd_method, SVDMethod::BDC);
    untouched.rebuild_from_statevector(*sv);
    const Statevector by_default = untouched.to_statevector();

    EXPECT_EQ(max_amplitude_diff(by_default, rebuilt_under(*sv, SVDMethod::BDC).to_statevector()), 0.0);
    EXPECT_GT(max_amplitude_diff(by_default, rebuilt_under(*sv, SVDMethod::EigenJacobi).to_statevector()), 0.0);
}

TEST(V11282RebuildKernel, SeedingReachesTheRebuildKernel) {
    // A dense seed through MPSSimulator::run is the rebuild and nothing else
    // when the circuit is empty, so the returned chain matches a direct
    // rebuild under the simulator's kernel bit for bit, under each kernel, and
    // the two kernels' results differ from each other.
    const auto sv = random_state(kQubits, kSeed);
    const Statevector via_bdc = seeded_under(sv, SVDMethod::BDC).to_statevector();
    const Statevector via_jacobi = seeded_under(sv, SVDMethod::EigenJacobi).to_statevector();

    EXPECT_EQ(max_amplitude_diff(via_bdc, rebuilt_under(*sv, SVDMethod::BDC).to_statevector()), 0.0);
    EXPECT_EQ(max_amplitude_diff(via_jacobi, rebuilt_under(*sv, SVDMethod::EigenJacobi).to_statevector()), 0.0);
    EXPECT_GT(max_amplitude_diff(via_bdc, via_jacobi), 0.0)
        << "the seeding path rebuilt under one kernel whichever was selected";
    EXPECT_EQ(seeded_under(sv, SVDMethod::EigenJacobi).svd_call_count(),
              static_cast<std::size_t>(kQubits - 1));
}

TEST(V11282RebuildKernel, SelectionAppliesToEverySplitOfTheRebuild) {
    // Selecting Jacobi on a chain that already rebuilt under BDC, then
    // rebuilding again, gives the Jacobi chain and not a mixture: the field
    // is read at each split of the sweep, so the sweep that runs after the
    // change is wholly the new kernel's.
    const auto sv = random_state(kQubits, kSeed);
    MPSState chain = rebuilt_under(*sv, SVDMethod::BDC);
    chain.svd_method = SVDMethod::EigenJacobi;
    chain.rebuild_from_statevector(*sv);
    EXPECT_EQ(max_amplitude_diff(chain.to_statevector(),
                                 rebuilt_under(*sv, SVDMethod::EigenJacobi).to_statevector()),
              0.0);
    EXPECT_EQ(chain.svd_call_count(), 2u * static_cast<std::size_t>(kQubits - 1))
        << "the second rebuild's splits accumulate onto the first's";
}

// =============================================================================
// autonne's Jacobi against the default
// =============================================================================

TEST(V11282RebuildKernel, AutonneRebuildPassesTheLadderWithoutRescue) {
    // The control (BDC against Jacobi) is asserted nonzero in the same test,
    // so a selection that never reaches the split cannot pass as agreement,
    // and the rescue counter is asserted zero, so a broken adapter cannot pass
    // by being rescued into correctness.
    const auto sv = random_state(kQubits, kSeed);
    const MPSState bdc = rebuilt_under(*sv, SVDMethod::BDC);
    const MPSState jacobi = rebuilt_under(*sv, SVDMethod::EigenJacobi);
    const MPSState autonne = rebuilt_under(*sv, SVDMethod::Jacobi);

    ASSERT_EQ(autonne.svd_method, SVDMethod::Jacobi);
    ASSERT_EQ(autonne.svd_call_count(), bdc.svd_call_count());
    EXPECT_EQ(autonne.gram_fallback_count(), 0u)
        << "the autonne route was rescued through Gram on some split of the "
           "rebuild, so this comparison would be between BDC and the Gram route";

    const Statevector sv_bdc = bdc.to_statevector();
    const double control = max_amplitude_diff(sv_bdc, jacobi.to_statevector());
    const double subject = max_amplitude_diff(sv_bdc, autonne.to_statevector());
    ::testing::Test::RecordProperty("rebuild_control_bdc_vs_jacobi", std::to_string(control));
    ::testing::Test::RecordProperty("rebuild_subject_bdc_vs_autonne", std::to_string(subject));

    ASSERT_GT(control, 0.0) << "INCONCLUSIVE: BDC and Jacobi are bit-identical, so "
                               "svd_method is not reaching the rebuild's splits";
    EXPECT_GT(subject, 0.0) << "INCONCLUSIVE: the control differs but the subject "
                               "is bit-identical to BDC, so the autonne route is "
                               "not being taken";
    EXPECT_LT(subject, rebuild_tol(kQubits))
        << "the adapter hands the kernel the wrong matrix or reads the factors "
           "back in the wrong layout";
    EXPECT_LT(max_amplitude_diff(autonne.to_statevector(), *sv), rebuild_tol(kQubits) / 2.0);
}

TEST(V11282RebuildKernel, AutonneSeedingReachesTheRebuild) {
    const auto sv = random_state(kQubits, kSeed);
    const MPSState seeded = seeded_under(sv, SVDMethod::Jacobi);
    EXPECT_EQ(seeded.svd_method, SVDMethod::Jacobi);
    EXPECT_EQ(seeded.gram_fallback_count(), 0u);
    EXPECT_EQ(max_amplitude_diff(seeded.to_statevector(),
                                 rebuilt_under(*sv, SVDMethod::Jacobi).to_statevector()),
              0.0);
}
