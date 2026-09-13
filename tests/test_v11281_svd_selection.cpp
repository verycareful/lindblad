// 1.1.28.1 test wave - which SVD kernel a run actually uses.
//
// 1.1.28.0 added a third SVDMethod, AutonneJacobi, and a public
// MPSSimulator::svd_method so a caller can choose the kernel through the main
// entry point rather than by driving MPSState directly. It also fixed the
// reason that knob could never have worked: seeding the initial state
// rebuilds the chain, the constructor carries the bond cap and the weight
// cutoff across but not the factorisation choice, and every path through
// apply_initial_state silently reverted to the default before the first gate.
// Selecting Jacobi through the simulator had the same defect in every release
// before that one.
//
// Three groups here.
//
// The knob survives. One test per construction site, because each site is a
// separate place the choice was lost: the default seeding, a basis seeding, a
// statevector seeding, the per-shot loop and the terminal-only pass. The
// method is read back off the returned chain, and then shown to have reached
// the factorisation, because a field that carries and a kernel that runs are
// two different claims. Two kernels that produce a bit-identical state are the
// same code having run twice, which is exactly what the pre-fix simulator did.
//
// The absent backend fails loud. Without LINDBLAD_WITH_AUTONNE the enumerator
// still exists and selecting it throws where the kernel is requested. It does
// NOT return false, because false means "attempted and did not converge",
// which sends the ladder into its Gram rescue and hands the caller a valid
// answer from a kernel they did not ask for. These tests are compiled out of a
// build that links autonne, since a superset build cannot exercise the branch,
// and run on the default configuration.
//
// The linked backend agrees with the one it replaces. Skipped unless the build
// links autonne. Two layers: the seam (svd_thin) on blocks of the shapes a
// saturating chain forms, where the factors are checked directly, and the
// simulator on qv_n8 at a cap that provably cannot bind, where two exact
// factorisations of one state must agree to rounding. The ladder's verify
// rung would rescue a transposed or swapped factor through the Gram route and
// the state comparison would then pass for the wrong reason, so the number of
// rescues is asserted zero alongside it.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/detail/svd_truncate.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "compare_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;
using lindblad::detail::MatrixOrder;
using lindblad::detail::svd_thin;
using lindblad_bench::load_corpus_circuit;

using Z = std::complex<double>;

namespace {

constexpr int kShots = 0;
constexpr std::uint64_t kSeed = 42;
constexpr double kEps = std::numeric_limits<double>::epsilon();

// Slack a backward-stable decomposition is entitled to, in units of
// n * eps * scale: the shape the truncation ladder's verify rung allows, so
// this suite and the ladder cannot disagree about what "correct" means.
constexpr double kSlack = 64.0;

double tol(int n, double scale = 1.0) {
    return kSlack * static_cast<double>(n) * kEps * scale;
}

// The smallest circuit that splits once.
QuantumCircuit one_split_circuit() {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    return qc;
}

// Mid-circuit measurement with a gate on the measured qubit afterwards, so the
// run takes the per-shot trajectory path rather than the terminal-only pass.
QuantumCircuit per_shot_circuit() {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1);
    qc.measure(0, 0);
    qc.x(0);
    qc.measure(1, 1);
    return qc;
}

// A normalised, non-symmetric 3-qubit state: amplitude k+1 at index k, scaled
// by the exact sum of squares 1 + 4 + ... + 64 = 204.
std::shared_ptr<const Statevector> ramp_state() {
    const int n = 3;
    const std::size_t dim = std::size_t{1} << n;
    double sum_sq = 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
        const double a = static_cast<double>(k + 1);
        sum_sq += a * a;
    }
    std::vector<Complex128> amps(dim);
    for (std::size_t k = 0; k < dim; ++k) {
        amps[k] = Complex128(static_cast<double>(k + 1) / std::sqrt(sum_sq), 0.0);
    }
    auto sv = std::make_shared<Statevector>(n);
    sv->set_amplitudes(amps);
    return sv;
}

Statevector state_under(const QuantumCircuit& qc, int cap, SVDMethod method) {
    MPSSimulator sim;
    sim.svd_method = method;
    return sim.run(qc, cap, kShots, kSeed).final_state.to_statevector();
}

// Largest amplitude difference between two states of one width. A global
// phase shows up here, which is correct: both runs start from the same state
// and apply the same gates, so neither may pick up a phase the other did not.
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

// How far two exact runs of the same circuit may drift apart through
// rounding alone. Each split is entitled to the verify rung's slack on a block
// of at most 2*cap on a side, the state norm is 1, unitary evolution does not
// amplify what an earlier split left behind, and both runs contribute.
double state_tol(std::size_t splits, int cap) {
    return 2.0 * static_cast<double>(splits) * tol(2 * cap, 1.0);
}

}  // namespace

// =============================================================================
// The knob and its default
// =============================================================================

TEST(V11281SvdSelection, DefaultMethodIsBdcEverywhere) {
    // The documented default, at the simulator, on a bare chain, and on the
    // chain a default run returns. A change to any one of the three without
    // the others is the kind of drift the survival tests below would then
    // misread.
    EXPECT_EQ(MPSSimulator{}.svd_method, SVDMethod::BDC);
    EXPECT_EQ(MPSState(2).svd_method, SVDMethod::BDC);
    MPSSimulator sim;
    const auto r = sim.run(one_split_circuit(), 4, kShots, kSeed);
    EXPECT_EQ(r.final_state.svd_method, SVDMethod::BDC);
}

TEST(V11281SvdSelection, MethodSurvivesDefaultSeeding) {
    MPSSimulator sim;
    sim.svd_method = SVDMethod::Jacobi;
    const auto r = sim.run(one_split_circuit(), 4, kShots, kSeed);
    EXPECT_EQ(r.final_state.svd_method, SVDMethod::Jacobi)
        << "the default |0...0> seeding rebuilt the chain and dropped the kernel";
}

TEST(V11281SvdSelection, MethodSurvivesBasisSeeding) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);
    RunPlan plan;
    plan.initial = InitialState::basis(5);
    MPSSimulator sim;
    sim.svd_method = SVDMethod::Jacobi;
    const auto r = sim.run(qc, 4, kShots, kSeed, plan);
    EXPECT_EQ(r.final_state.svd_method, SVDMethod::Jacobi)
        << "the basis seeding rebuilt the chain and dropped the kernel";
}

TEST(V11281SvdSelection, MethodSurvivesStatevectorSeeding) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);
    RunPlan plan;
    plan.initial = InitialState::from(ramp_state());
    MPSSimulator sim;
    sim.svd_method = SVDMethod::Jacobi;
    const auto r = sim.run(qc, 4, kShots, kSeed, plan);
    EXPECT_EQ(r.final_state.svd_method, SVDMethod::Jacobi)
        << "the dense seeding rebuilt the chain and dropped the kernel";
}

TEST(V11281SvdSelection, MethodSurvivesThePerShotPath) {
    // The per-shot loop constructs a fresh chain for every trajectory. That is
    // a construction site of its own, distinct from the seeding branches, and
    // the returned chain is the last trajectory's.
    MPSSimulator sim;
    sim.svd_method = SVDMethod::Jacobi;
    const auto r = sim.run(per_shot_circuit(), 4, 4, kSeed);
    EXPECT_EQ(r.final_state.svd_method, SVDMethod::Jacobi)
        << "the per-shot rebuild dropped the kernel";
}

TEST(V11281SvdSelection, MethodSurvivesTheTerminalOnlyPath) {
    QuantumCircuit qc = one_split_circuit();
    qc.measure_all();
    MPSSimulator sim;
    sim.svd_method = SVDMethod::Jacobi;
    const auto r = sim.run(qc, 4, 16, kSeed);
    EXPECT_EQ(r.final_state.svd_method, SVDMethod::Jacobi);
}

TEST(V11281SvdSelection, SuppliedChainKeepsItsOwnMethod) {
    // The one seeding branch that must NOT copy the simulator's choice: a
    // chain handed in already answers the question, and overriding it would
    // change the kernel under a caller who configured the chain themselves.
    auto chain = std::make_shared<MPSState>(3, 4);
    chain->svd_method = SVDMethod::Jacobi;
    RunPlan plan;
    plan.initial = InitialState::from(std::shared_ptr<const MPSState>(chain));
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1);
    MPSSimulator sim;
    sim.svd_method = SVDMethod::BDC;
    const auto r = sim.run(qc, 4, kShots, kSeed, plan);
    EXPECT_EQ(r.final_state.svd_method, SVDMethod::Jacobi)
        << "a supplied chain's own kernel was overridden by the simulator's";
}

TEST(V11281SvdSelection, SelectorReachesTheFactorisation) {
    // The field carrying is necessary and not sufficient. BDC and Jacobi are
    // different algorithms and do not agree bit for bit on the block sizes
    // qv_n8 forms at a cap of 16, so two runs that differ by exactly zero mean
    // the selector never reached the split. The difference is bounded above
    // as well, because at a cap of 2^(n/2) neither run truncates and two
    // exact factorisations of one state agree to rounding.
    const auto qc = load_corpus_circuit("qv_n8.qasm", false);
    const int cap = 1 << (qc.n_qubits / 2);

    MPSSimulator sim;
    sim.svd_method = SVDMethod::BDC;
    const auto bdc = sim.run(qc, cap, kShots, kSeed);
    sim.svd_method = SVDMethod::Jacobi;
    const auto jacobi = sim.run(qc, cap, kShots, kSeed);
    ASSERT_EQ(bdc.final_state.svd_call_count(), jacobi.final_state.svd_call_count());

    const double diff = max_amplitude_diff(bdc.final_state.to_statevector(),
                                           jacobi.final_state.to_statevector());
    ::testing::Test::RecordProperty("bdc_vs_jacobi_max_amplitude_diff",
                                    std::to_string(diff));
    EXPECT_GT(diff, 0.0)
        << "BDC and Jacobi produced a bit-identical state: the selector is not "
           "reaching the factorisation, so the same kernel ran twice";
    EXPECT_LT(diff, state_tol(bdc.final_state.svd_call_count(), cap))
        << "two exact factorisations of one state disagree beyond rounding";
}

// =============================================================================
// The absent backend. Compiled only where autonne is NOT linked.
// =============================================================================

#ifndef LINDBLAD_WITH_AUTONNE

namespace {

// Single-qubit gates only, so no bond split and the kernel is never asked.
QuantumCircuit no_split_circuit() {
    QuantumCircuit qc(2);
    qc.h(0).x(1).h(1);
    return qc;
}

// CX as MPSState::apply_two_qubit_gate takes it: rows and columns index
// (q1, q2) with q1 the high bit, the layout the adjacent kernel documents, so
// with q1 as control the gate swaps |10> and |11>.
std::array<Complex128, 16> cx_matrix() {
    std::array<Complex128, 16> u{};
    u[0 * 4 + 0] = Complex128(1.0, 0.0);
    u[1 * 4 + 1] = Complex128(1.0, 0.0);
    u[2 * 4 + 3] = Complex128(1.0, 0.0);
    u[3 * 4 + 2] = Complex128(1.0, 0.0);
    return u;
}

}  // namespace

TEST(V11281SvdSelection, AutonneWithoutTheLibraryThrowsAtTheKernel) {
    const int n = 2;
    std::vector<Z> a = {Z(1.0, 0.0), Z(0.0, 0.5), Z(0.25, 0.0), Z(0.0, -1.0)};
    std::vector<Z> u(static_cast<std::size_t>(n) * n), v(static_cast<std::size_t>(n) * n);
    std::vector<double> s(static_cast<std::size_t>(n));
    try {
        svd_thin(a.data(), n, n, MatrixOrder::RowMajor, SVDMethod::AutonneJacobi,
                 u.data(), s.data(), v.data());
        FAIL() << "svd_thin returned instead of throwing for a backend this "
                  "build did not link";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("AutonneJacobi"), std::string::npos) << what;
        EXPECT_NE(what.find("LINDBLAD_WITH_AUTONNE"), std::string::npos)
            << "the message does not say how to get the backend: " << what;
    }
}

TEST(V11281SvdSelection, AutonneWithoutTheLibraryThrowsFromRun) {
    // Through the main entry point. run() does not catch, so the throw reaches
    // the caller as a throw, and the fact that it is a throw at all is the
    // contract: a false return would have been rescued through the Gram route
    // and the caller would hold a valid state from a kernel they did not ask
    // for, with nothing to say so.
    MPSSimulator sim;
    sim.svd_method = SVDMethod::AutonneJacobi;
    EXPECT_THROW(sim.run(one_split_circuit(), 4, kShots, kSeed), std::runtime_error);
}

TEST(V11281SvdSelection, AutonneWithoutTheLibraryIsHarmlessUntilASplit) {
    // The throw is at the point of use. A circuit that never splits never asks
    // for the kernel, so selecting the absent backend costs nothing until the
    // first two-qubit gate. The enumerator exists in every build for exactly
    // this reason: code compiles the same way either way.
    MPSSimulator sim;
    sim.svd_method = SVDMethod::AutonneJacobi;
    EXPECT_NO_THROW(sim.run(no_split_circuit(), 4, kShots, kSeed));
}

TEST(V11281SvdSelection, AutonneWithoutTheLibraryThrowsFromTheChainDirectly) {
    MPSState chain(2, 4);
    chain.svd_method = SVDMethod::AutonneJacobi;
    EXPECT_THROW(chain.apply_two_qubit_gate(cx_matrix(), 0, 1), std::runtime_error);
}

TEST(V11281SvdSelection, AutonneWithoutTheLibraryThrowsFromTheRebuildPath) {
    // The rebuild from dense amplitudes is the second call site of the ladder
    // and factorises through the same kernel selector.
    MPSState chain(3, 4);
    chain.svd_method = SVDMethod::AutonneJacobi;
    EXPECT_THROW(chain.rebuild_from_statevector(*ramp_state()), std::runtime_error);
}

#endif  // !LINDBLAD_WITH_AUTONNE

// =============================================================================
// The linked backend. Skipped unless the build links autonne.
// =============================================================================

namespace {

// One factorisation with the buffers sized as the seam requires.
struct Svd {
    std::vector<Z> U, V;
    std::vector<double> S;
    int rows = 0, cols = 0, k = 0;
    bool ok = false;

    Z u(int r, int j) const { return U[static_cast<std::size_t>(j) * rows + r]; }
    Z v(int c, int j) const { return V[static_cast<std::size_t>(j) * cols + c]; }

    // M(r, c) = sum_j U(r,j) S(j) conj(V(c,j)). V is returned as V and not as
    // V-dagger, so the conjugate belongs here.
    Z recon(int r, int c) const {
        Z acc(0.0, 0.0);
        for (int j = 0; j < k; ++j) {
            acc += u(r, j) * S[static_cast<std::size_t>(j)] * std::conj(v(c, j));
        }
        return acc;
    }
};

Svd run_svd(const std::vector<Z>& buf, int rows, int cols, MatrixOrder order,
            SVDMethod method) {
    Svd f;
    f.rows = rows;
    f.cols = cols;
    f.k = std::min(rows, cols);
    f.U.assign(static_cast<std::size_t>(rows) * f.k, Z(0.0, 0.0));
    f.V.assign(static_cast<std::size_t>(cols) * f.k, Z(0.0, 0.0));
    f.S.assign(static_cast<std::size_t>(f.k), 0.0);
    f.ok = svd_thin(buf.data(), rows, cols, order, method, f.U.data(),
                    f.S.data(), f.V.data());
    return f;
}

// Full-rank random block, entries uniform on the unit square in the complex
// plane, from a seeded stream so the fixture is the same on every platform.
std::vector<Z> random_block(int rows, int cols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::vector<Z> a(static_cast<std::size_t>(rows) * cols);
    for (auto& z : a) z = Z(unit(rng), unit(rng));
    return a;
}

// A rank-2 block: every entry is affine in its row and column index, so the
// spectrum has k - 2 exact zeros. Where a one-sided Jacobi sweep has to
// resolve a null space rather than merely order a spectrum.
std::vector<Z> rank_two_block(int rows, int cols) {
    std::vector<Z> a(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            a[static_cast<std::size_t>(r) * cols + c] =
                Z(1.0 + r + 2.0 * c, 0.5 + 3.0 * r - c);
        }
    }
    return a;
}

Z at(const std::vector<Z>& buf, int rows, int cols, MatrixOrder order, int r, int c) {
    return order == MatrixOrder::RowMajor
               ? buf[static_cast<std::size_t>(r) * cols + c]
               : buf[static_cast<std::size_t>(c) * rows + r];
}

double frob(const std::vector<Z>& a) {
    double t = 0.0;
    for (const Z& z : a) t += std::norm(z);
    return std::sqrt(t);
}

// The shapes a saturating brickwork chain at chi = 64 forms, squares
// dominating and the asymmetric shapes from the chain's ends.
const std::vector<std::pair<int, int>> kShapes = {
    {2, 2}, {4, 4}, {8, 8}, {16, 16}, {32, 32}, {64, 64}, {128, 128},
    {2, 4}, {4, 2}, {4, 8}, {8, 4}, {16, 32}, {32, 16}, {64, 128}, {128, 64},
};

#ifdef LINDBLAD_WITH_AUTONNE
constexpr bool kAutonneLinked = true;
#else
constexpr bool kAutonneLinked = false;
#endif

#define SKIP_UNLESS_AUTONNE()                                                  \
    do {                                                                       \
        if (!kAutonneLinked) {                                                 \
            GTEST_SKIP() << "configure with -DLINDBLAD_WITH_AUTONNE=ON to run"; \
        }                                                                      \
    } while (0)

}  // namespace

TEST(V11281SvdSelection, AutonneReturnsTrueOnEveryShape) {
    SKIP_UNLESS_AUTONNE();
    for (const auto& [rows, cols] : kShapes) {
        const auto a = random_block(rows, cols, kSeed + rows * 1000 + cols);
        const auto f = run_svd(a, rows, cols, MatrixOrder::RowMajor,
                               SVDMethod::AutonneJacobi);
        EXPECT_TRUE(f.ok) << rows << "x" << cols << ": the kernel reported failure";
    }
}

TEST(V11281SvdSelection, AutonneSpectrumMatchesBdc) {
    SKIP_UNLESS_AUTONNE();
    for (const auto& [rows, cols] : kShapes) {
        const auto a = random_block(rows, cols, kSeed + rows * 1000 + cols);
        const int n = std::max(rows, cols);
        const double scale = frob(a);
        const auto b = run_svd(a, rows, cols, MatrixOrder::RowMajor, SVDMethod::BDC);
        const auto j = run_svd(a, rows, cols, MatrixOrder::RowMajor,
                               SVDMethod::AutonneJacobi);
        ASSERT_TRUE(b.ok);
        ASSERT_TRUE(j.ok);
        for (int i = 0; i < b.k; ++i) {
            EXPECT_NEAR(j.S[static_cast<std::size_t>(i)], b.S[static_cast<std::size_t>(i)],
                        tol(n, scale))
                << rows << "x" << cols << ": sigma[" << i << "] disagrees";
        }
        for (int i = 1; i < j.k; ++i) {
            EXPECT_LE(j.S[static_cast<std::size_t>(i)], j.S[static_cast<std::size_t>(i - 1)])
                << rows << "x" << cols << ": spectrum not descending at " << i;
        }
    }
}

TEST(V11281SvdSelection, AutonneFactorsReconstructTheInputInBothOrders) {
    // The spectrum is blind to a transposed read: a transpose carries the same
    // singular values. Reconstruction is not, and it is checked under both
    // layouts because the adapter maps MatrixOrder explicitly and a wrong
    // mapping factorises the transpose cleanly.
    SKIP_UNLESS_AUTONNE();
    for (const auto& [rows, cols] : kShapes) {
        const auto a = random_block(rows, cols, kSeed + rows * 7 + cols);
        const int n = std::max(rows, cols);
        const double scale = frob(a);
        for (MatrixOrder order : {MatrixOrder::RowMajor, MatrixOrder::ColMajor}) {
            const auto f = run_svd(a, rows, cols, order, SVDMethod::AutonneJacobi);
            ASSERT_TRUE(f.ok);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    EXPECT_NEAR(std::abs(f.recon(r, c) - at(a, rows, cols, order, r, c)),
                                0.0, tol(n, scale))
                        << rows << "x" << cols << " "
                        << (order == MatrixOrder::RowMajor ? "row" : "col")
                        << "-major: M[" << r << "," << c << "] not reconstructed";
                }
            }
        }
    }
}

TEST(V11281SvdSelection, AutonneFactorsAreIsometries) {
    SKIP_UNLESS_AUTONNE();
    for (const auto& [rows, cols] : kShapes) {
        const auto a = random_block(rows, cols, kSeed + rows + cols * 7);
        const int n = std::max(rows, cols);
        const auto f = run_svd(a, rows, cols, MatrixOrder::RowMajor,
                               SVDMethod::AutonneJacobi);
        ASSERT_TRUE(f.ok);
        for (int i = 0; i < f.k; ++i) {
            for (int j = 0; j < f.k; ++j) {
                Z uu(0.0, 0.0), vv(0.0, 0.0);
                for (int r = 0; r < rows; ++r) uu += std::conj(f.u(r, i)) * f.u(r, j);
                for (int c = 0; c < cols; ++c) vv += std::conj(f.v(c, i)) * f.v(c, j);
                const double want = (i == j) ? 1.0 : 0.0;
                EXPECT_NEAR(std::abs(uu - want), 0.0, tol(n))
                    << rows << "x" << cols << ": U not an isometry at " << i << "," << j;
                EXPECT_NEAR(std::abs(vv - want), 0.0, tol(n))
                    << rows << "x" << cols << ": V not an isometry at " << i << "," << j;
            }
        }
    }
}

TEST(V11281SvdSelection, AutonneNullSpaceTailLandsAtEpsilon) {
    SKIP_UNLESS_AUTONNE();
    for (int n : {4, 8, 16, 32}) {
        const auto a = rank_two_block(n, n);
        const double scale = frob(a);
        const auto f = run_svd(a, n, n, MatrixOrder::RowMajor, SVDMethod::AutonneJacobi);
        ASSERT_TRUE(f.ok);
        for (int i = 2; i < n; ++i) {
            EXPECT_NEAR(f.S[static_cast<std::size_t>(i)], 0.0, tol(n, scale))
                << n << "x" << n << ": null-space sigma[" << i << "] not at epsilon";
        }
        EXPECT_GT(f.S[1], tol(n, scale)) << "the second direction is real";
    }
}

TEST(V11281SvdSelection, AutonnePassesTheLadderWithoutRescue) {
    // The ladder is SELECT -> VERIFY -> FALLBACK -> THROW. A kernel whose
    // factors fail verification is rescued through the Gram route and the
    // caller gets a valid slice with only a counter to say the primary route
    // was not taken. This asserts the primary route IS taken for autonne, on
    // every shape, which is the check the state comparison below cannot make.
    SKIP_UNLESS_AUTONNE();
    const double cutoff = MPSState(1).cutoff;
    for (const auto& [rows, cols] : kShapes) {
        const auto z = random_block(rows, cols, kSeed + rows * 3 + cols * 5);
        std::vector<Complex128> a(z.size());
        for (std::size_t i = 0; i < z.size(); ++i) a[i] = Complex128(z[i].real(), z[i].imag());
        const int cap = std::min(rows, cols);
        const auto split = lindblad::detail::svd_truncate_verified(
            a.data(), rows, cols, MatrixOrder::RowMajor, cap, cutoff,
            SVDMethod::AutonneJacobi, "V11281SvdSelection");
        EXPECT_FALSE(split.used_gram_fallback)
            << rows << "x" << cols << ": the autonne factors failed verification "
               "and the Gram route produced the slice";
        EXPECT_EQ(split.rank, cap) << rows << "x" << cols << ": a full-rank block "
                                      "lost directions at a cap equal to its rank";
        EXPECT_LE(split.residual_excess, tol(std::max(rows, cols)))
            << rows << "x" << cols << ": the accepted factorisation sits far "
               "above a perfect one";
    }
}

TEST(V11281SvdSelection, AutonneAgreesWithBdcAtANonBindingCap) {
    // qv_n8 at 2^(n/2): the widest cut splits the chain 4 | 4, the rank cannot
    // exceed 16, and neither kernel discards anything. Two exact
    // factorisations of one state must then agree to rounding. The control
    // (BDC against Jacobi) is asserted nonzero in the same test, so a selector
    // that never reaches the split cannot pass as agreement, and the rescue
    // counter is asserted zero, so a broken adapter cannot pass by being
    // rescued into correctness.
    SKIP_UNLESS_AUTONNE();
    const auto qc = load_corpus_circuit("qv_n8.qasm", false);
    const int cap = 1 << (qc.n_qubits / 2);

    MPSSimulator sim;
    sim.svd_method = SVDMethod::BDC;
    const auto bdc = sim.run(qc, cap, kShots, kSeed);
    sim.svd_method = SVDMethod::Jacobi;
    const auto jacobi = sim.run(qc, cap, kShots, kSeed);
    sim.svd_method = SVDMethod::AutonneJacobi;
    const auto autonne = sim.run(qc, cap, kShots, kSeed);

    ASSERT_EQ(autonne.final_state.svd_method, SVDMethod::AutonneJacobi);
    ASSERT_EQ(autonne.final_state.svd_call_count(), bdc.final_state.svd_call_count());
    EXPECT_EQ(autonne.final_state.gram_fallback_count(), 0u)
        << "the autonne route was rescued through Gram on some split, so this "
           "comparison would be between BDC and the Gram route";

    const auto sv_bdc = bdc.final_state.to_statevector();
    const double control = max_amplitude_diff(sv_bdc, jacobi.final_state.to_statevector());
    const double subject = max_amplitude_diff(sv_bdc, autonne.final_state.to_statevector());
    ::testing::Test::RecordProperty("control_bdc_vs_jacobi", std::to_string(control));
    ::testing::Test::RecordProperty("subject_bdc_vs_autonne", std::to_string(subject));

    ASSERT_GT(control, 0.0) << "INCONCLUSIVE: BDC and Jacobi are bit-identical, so "
                               "svd_method is not reaching the factorisation";
    EXPECT_GT(subject, 0.0) << "INCONCLUSIVE: the control differs but the subject "
                               "is bit-identical to BDC, so the autonne route is "
                               "not being taken";
    EXPECT_LT(subject, state_tol(bdc.final_state.svd_call_count(), cap))
        << "the adapter hands the kernel the wrong matrix or reads the factors "
           "back in the wrong layout";
}

TEST(V11281SvdSelection, AutonneAtABindingCapIsReportedNotAsserted) {
    // Where the cap binds the two kernels are entitled to differ: they may
    // select different directions to keep at a near-degenerate split. The
    // figure is recorded for the release notes and not judged.
    SKIP_UNLESS_AUTONNE();
    const auto qc = load_corpus_circuit("qv_n8.qasm", false);
    const int cap = 4;
    const double diff = max_amplitude_diff(state_under(qc, cap, SVDMethod::BDC),
                                           state_under(qc, cap, SVDMethod::AutonneJacobi));
    ::testing::Test::RecordProperty("binding_cap_bdc_vs_autonne", std::to_string(diff));
    SUCCEED() << "chi = " << cap << " max amplitude difference " << diff;
}
