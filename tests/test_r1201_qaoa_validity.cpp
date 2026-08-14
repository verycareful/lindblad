// R.1.20.1 test suite — QAOA / MA-QAOA result validity (#68).
//
// R.1.20.0 fixed a defect that produced a WRONG ANSWER rather than a missing
// diagnostic, which is why it needs assertions of its own rather than a note in
// the changelog.
//
// Both optimisers seeded their best-so-far with +infinity. A seed like that
// exists for exactly one purpose: to lose its first comparison. That is also
// exactly the comparison -ffinite-math-only (implied by the project-wide
// -ffast-math) is licensed to fold away, on the premise that infinities do not
// occur. Folded, the branch never runs, and the caller receives an EMPTY best
// bitstring alongside an unwritten best cost — with no exception, no warning,
// and a Result that looks structurally complete. MAQAOA::Result::per_layer_costs
// reached callers through the same fold, so it was a wrong result, not a
// diagnostic. The remedy was an explicit bool flag at eight sites, which no
// floating-point model is entitled to reason about.
//
// Separately, QAOA::optimize read an UNINITIALISED min_val straight into
// Result::optimal_value: NLopt failure codes can return without writing it, so
// the field could carry an indeterminate stack value. It is now seeded with a
// bit-built quiet NaN and `converged` additionally requires the result to be
// finite, so a failed run reports something detectable instead of garbage.
//
// What this suite pins, therefore:
//   - the best bitstring is written, at the right width, and is genuinely the
//     minimum-cost sample rather than merely non-empty
//   - per-layer costs are written and finite under the layerwise schedule
//   - `converged` is never claimed alongside a non-finite value
//   - results are reproducible under a fixed seed
//
// Finiteness is always checked with is_finite_strict, never std::isfinite: this
// TU compiles under -ffast-math, where the latter may fold to a constant true
// and turn the check into an assertion that cannot fail.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/ising.hpp"
#include "lindblad/types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

namespace {

// A deliberately ASYMMETRIC instance. A symmetric cost landscape would let a
// broken ranking loop land on a correct-looking answer by accident — every
// bitstring being equally good hides exactly the bug under test here.
IsingHamiltonian asymmetric_instance() {
    const std::vector<std::vector<double>> Q = {
        {1.0, -2.0, 0.5},
        {0.0, -1.0, 1.5},
        {0.0, 0.0, 2.0}};
    return IsingHamiltonian::from_qubo(Q);
}

// Every bitstring the sampler returned, at the width the problem expects.
// Anything narrower or wider is skipped by the library's own ranking loop, so
// the reference must skip it too.
std::vector<std::string> sampled_keys(
    const std::unordered_map<std::string, int>& counts, size_t width) {
    std::vector<std::string> keys;
    for (const auto& [bits, count] : counts) {
        (void)count;
        if (bits.size() == width) keys.push_back(bits);
    }
    return keys;
}

// The ranking contract, asserted against an INDEPENDENT cost evaluation.
//
// IsingHamiltonian::evaluate and the library's internal
// computational_basis_cost differ at most by the constant offset, and a
// constant cannot reorder anything — so "no sample is cheaper than the one
// chosen" is checkable without reaching into the implementation. Ties are
// broken by sample count inside the library, which is why this is <= over the
// whole sample set rather than a claim about a unique winner.
void expect_best_is_minimal(const IsingHamiltonian& ising,
                            const std::string& best,
                            const std::unordered_map<std::string, int>& counts,
                            const char* what) {
    const auto keys = sampled_keys(counts, best.size());
    ASSERT_FALSE(keys.empty()) << what << ": no samples at the expected width";

    const double best_cost = ising.evaluate(best);
    for (const auto& bits : keys) {
        EXPECT_LE(best_cost, ising.evaluate(bits))
            << what << ": ranking returned " << best << " (cost " << best_cost
            << ") while " << bits << " (cost " << ising.evaluate(bits)
            << ") was cheaper and was also sampled — the ranking loop is not "
               "selecting the minimum";
    }
}

} // namespace

// =============================================================================
// QAOA
// =============================================================================

// The headline symptom of #68. An empty best bitstring was the user-visible
// form of the folded comparison, and it arrived with converged == true and no
// error of any kind.
TEST(R1201QaoaValidity, QaoaWritesABestBitstringOfTheRightWidth) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();
    ASSERT_EQ(cost.n_qubits(), 3);

    QAOA qaoa;
    qaoa.options.p = 2;
    qaoa.options.max_iterations = 120;
    qaoa.options.seed = 11;
    auto res = qaoa.optimize(cost);

    EXPECT_FALSE(res.best_bitstring.empty())
        << "best_bitstring is empty: the ranking loop never wrote a result, "
           "which is the #68 symptom";
    EXPECT_EQ(res.best_bitstring.size(), 3u);
    EXPECT_FALSE(res.counts.empty()) << "no samples were drawn at all";
}

// Non-empty is necessary but nowhere near sufficient: a loop that wrote only
// its FIRST candidate and never compared again would also return a full-width
// bitstring. This checks that the value returned is the minimum.
TEST(R1201QaoaValidity, QaoaBestBitstringIsTheCheapestSample) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    QAOA qaoa;
    qaoa.options.p = 2;
    qaoa.options.max_iterations = 120;
    qaoa.options.seed = 11;
    auto res = qaoa.optimize(cost);

    ASSERT_EQ(res.best_bitstring.size(), 3u);
    expect_best_is_minimal(ising, res.best_bitstring, res.counts, "QAOA");
}

// The uninitialised-min_val defect. `converged` is defined to require a finite
// result, so this pairing can never be observed unless that guard is gone —
// and an indeterminate stack value is finite far more often than not, which is
// precisely why the failure was invisible.
TEST(R1201QaoaValidity, QaoaNeverClaimsConvergenceWithANonFiniteValue) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    // Several budgets, including ones too small to converge, so both the
    // success and the give-up path are exercised.
    for (int max_iter : {1, 5, 40, 200}) {
        QAOA qaoa;
        qaoa.options.p = 1;
        qaoa.options.max_iterations = max_iter;
        qaoa.options.seed = 3;
        auto res = qaoa.optimize(cost);

        if (res.converged) {
            EXPECT_TRUE(is_finite_strict(res.optimal_value))
                << "max_iterations = " << max_iter
                << ": converged reported alongside a non-finite optimal_value ("
                << res.optimal_value << ")";
        }

        // Whatever the outcome, the bitstring comes from sampling rather than
        // from the optimiser, so it is written either way.
        EXPECT_EQ(res.best_bitstring.size(), 3u)
            << "max_iterations = " << max_iter
            << ": sampling and ranking do not depend on optimiser success";
    }
}

// A run that does not converge must not silently look like one that did.
TEST(R1201QaoaValidity, QaoaOptimalParamsAreAlwaysWritten) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    QAOA qaoa;
    qaoa.options.p = 2;
    qaoa.options.max_iterations = 60;
    qaoa.options.seed = 21;
    auto res = qaoa.optimize(cost);

    EXPECT_EQ(res.optimal_params.size(), 4u) << "2 params per layer at p = 2";
    for (size_t i = 0; i < res.optimal_params.size(); ++i) {
        EXPECT_TRUE(is_finite_strict(res.optimal_params[i]))
            << "optimal_params[" << i << "] is non-finite";
    }
}

// Reproducibility. A ranking loop reading uninitialised or folded state can
// return different answers on identical inputs, so seeded determinism is part
// of the same story rather than a separate concern.
TEST(R1201QaoaValidity, QaoaIsReproducibleUnderAFixedSeed) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    auto run = [&]() {
        QAOA qaoa;
        qaoa.options.p = 2;
        qaoa.options.max_iterations = 80;
        qaoa.options.seed = 1234;
        return qaoa.optimize(cost);
    };

    const auto a = run();
    const auto b = run();

    EXPECT_EQ(a.best_bitstring, b.best_bitstring);
    EXPECT_EQ(a.converged, b.converged);
    EXPECT_EQ(a.counts, b.counts);
    EXPECT_EQ(a.optimal_value, b.optimal_value);
}

// =============================================================================
// MA-QAOA
// =============================================================================

TEST(R1201QaoaValidity, MaqaoaWritesABestBitstringOfTheRightWidth) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    MAQAOA maqaoa;
    maqaoa.options.p = 2;
    maqaoa.options.max_iterations = 150;
    maqaoa.options.seed = 5;
    auto res = maqaoa.optimize(cost);

    EXPECT_FALSE(res.best_bitstring.empty());
    EXPECT_EQ(res.best_bitstring.size(), 3u);
    EXPECT_FALSE(res.counts.empty());
}

TEST(R1201QaoaValidity, MaqaoaBestBitstringIsTheCheapestSample) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    MAQAOA maqaoa;
    maqaoa.options.p = 2;
    maqaoa.options.max_iterations = 150;
    maqaoa.options.seed = 5;
    auto res = maqaoa.optimize(cost);

    ASSERT_EQ(res.best_bitstring.size(), 3u);
    expect_best_is_minimal(ising, res.best_bitstring, res.counts, "MA-QAOA");
}

// per_layer_costs is the field that made #68 a WRONG RESULT rather than a
// missing diagnostic: it reaches callers directly, and it was fed by the same
// folded comparison. Under the layerwise schedule every layer must contribute
// one finite entry.
TEST(R1201QaoaValidity, MaqaoaLayerwiseWritesFinitePerLayerCosts) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    for (int p : {1, 2, 3}) {
        MAQAOA maqaoa;
        maqaoa.options.p = p;
        maqaoa.options.layerwise = true;
        maqaoa.options.max_iterations = 40;
        maqaoa.options.seed = 9;
        auto res = maqaoa.optimize(cost);

        ASSERT_EQ(res.per_layer_costs.size(), static_cast<size_t>(p))
            << "p = " << p
            << ": one cost per layer is the layerwise contract; a short vector "
               "means a layer's best-so-far was never written";

        for (int layer = 0; layer < p; ++layer) {
            EXPECT_TRUE(is_finite_strict(res.per_layer_costs[layer]))
                << "p = " << p << ", layer " << layer
                << ": per-layer cost is non-finite ("
                << res.per_layer_costs[layer] << ")";
        }

        EXPECT_EQ(res.best_bitstring.size(), 3u) << "p = " << p;
    }
}

// The layer bookkeeping travels with per_layer_costs and is fed by the same
// loop, so a short vector in either is the same class of defect.
TEST(R1201QaoaValidity, MaqaoaLayerwiseBookkeepingIsConsistent) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    const int p = 3;
    MAQAOA maqaoa;
    maqaoa.options.p = p;
    maqaoa.options.layerwise = true;
    maqaoa.options.max_iterations = 40;
    maqaoa.options.seed = 17;
    auto res = maqaoa.optimize(cost);

    EXPECT_EQ(res.per_layer_costs.size(), static_cast<size_t>(p));
    EXPECT_EQ(res.layer_nfev.size(), static_cast<size_t>(p));
    EXPECT_EQ(res.wall_time_by_layer.size(), static_cast<size_t>(p));

    int summed = 0;
    for (int nfev : res.layer_nfev) {
        EXPECT_GT(nfev, 0) << "a layer that ran must record evaluations";
        summed += nfev;
    }
    EXPECT_EQ(res.num_iterations, summed)
        << "num_iterations is documented as the total across all layers";

    for (double seconds : res.wall_time_by_layer) {
        EXPECT_GE(seconds, 0.0);
        EXPECT_TRUE(is_finite_strict(seconds));
    }
}

// The joint (non-layerwise) schedule has no per-layer stage to report, so the
// vector stays empty. Pinned because "empty" and "never written" look identical
// from the outside, and only one of them is correct here.
TEST(R1201QaoaValidity, MaqaoaJointScheduleLeavesPerLayerCostsEmpty) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    MAQAOA maqaoa;
    maqaoa.options.p = 2;
    maqaoa.options.layerwise = false;
    maqaoa.options.max_iterations = 80;
    maqaoa.options.seed = 4;
    auto res = maqaoa.optimize(cost);

    EXPECT_TRUE(res.per_layer_costs.empty())
        << "the joint schedule optimises all layers at once and has no "
           "per-layer best to report";
    EXPECT_EQ(res.best_bitstring.size(), 3u)
        << "ranking still runs on the joint path";
}

// MA-QAOA's contract differs from QAOA's here: a non-finite optimiser result is
// replaced by a large finite sentinel rather than surfaced as a NaN, so
// optimal_value is finite on every path. `converged` still carries the truth.
TEST(R1201QaoaValidity, MaqaoaOptimalValueIsFiniteOnEveryPath) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    for (bool layerwise : {false, true}) {
        for (int max_iter : {1, 10, 100}) {
            MAQAOA maqaoa;
            maqaoa.options.p = 2;
            maqaoa.options.layerwise = layerwise;
            maqaoa.options.max_iterations = max_iter;
            maqaoa.options.seed = 33;
            auto res = maqaoa.optimize(cost);

            EXPECT_TRUE(is_finite_strict(res.optimal_value))
                << "layerwise = " << layerwise
                << ", max_iterations = " << max_iter
                << ": optimal_value is non-finite (" << res.optimal_value
                << "); a failed optimisation is reported through converged, "
                   "with the value clamped to a finite sentinel";

            if (res.converged) {
                EXPECT_TRUE(is_finite_strict(res.optimal_value));
            }
        }
    }
}

TEST(R1201QaoaValidity, MaqaoaIsReproducibleUnderAFixedSeed) {
    auto ising = asymmetric_instance();
    auto cost = ising.to_sparse_pauli_op();

    auto run = [&]() {
        MAQAOA maqaoa;
        maqaoa.options.p = 2;
        maqaoa.options.layerwise = true;
        maqaoa.options.max_iterations = 50;
        maqaoa.options.seed = 777;
        return maqaoa.optimize(cost);
    };

    const auto a = run();
    const auto b = run();

    EXPECT_EQ(a.best_bitstring, b.best_bitstring);
    EXPECT_EQ(a.converged, b.converged);
    EXPECT_EQ(a.optimal_value, b.optimal_value);
    ASSERT_EQ(a.per_layer_costs.size(), b.per_layer_costs.size());
    for (size_t i = 0; i < a.per_layer_costs.size(); ++i) {
        EXPECT_EQ(a.per_layer_costs[i], b.per_layer_costs[i])
            << "per_layer_costs[" << i << "] differs between identical runs";
    }
}

// =============================================================================
// VQE — the same defect class, fixed earlier and pinned here alongside its
// siblings so the three optimisers are covered by one story.
// =============================================================================

TEST(R1201QaoaValidity, VqeNeverClaimsConvergenceWithANonFiniteEigenvalue) {
    SparsePauliOp h(std::vector<PauliString>{PauliString("Z")});
    QuantumCircuit ansatz(1);
    ansatz.ry("t", 0);

    for (int max_iter : {1, 10, 80}) {
        VQE vqe;
        vqe.options.max_iterations = max_iter;
        vqe.options.optimizer = "COBYLA";
        auto res = vqe.compute_minimum_eigenvalue(h, ansatz, {});

        EXPECT_TRUE(is_finite_strict(res.eigenvalue))
            << "max_iterations = " << max_iter
            << ": eigenvalue is non-finite (" << res.eigenvalue
            << "); the recorded history minimum must be returned when the "
               "optimiser bails without writing a value";
        EXPECT_GE(res.eigenvalue, -1.0 - 1e-9)
            << "<Z> is bounded below by -1";
    }
}
