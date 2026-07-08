// R.1.13.1 test patch — Estimator::Options::group_pauli_terms (audit F-10).
// Qubit-wise-commuting (QWC) grouping shares measurement runs across commuting
// Pauli terms. It must be statistically equivalent to the ungrouped per-term
// sampling (both converge to the exact statevector expectation), and each path
// must remain seed-deterministic. group=false additionally preserves the
// pre-R.1.13 one-run-per-term behaviour.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"

#include <cmath>

using namespace lindblad;

namespace {

// A non-trivial 3-qubit state with nonzero X, Y and Z expectations per qubit.
QuantumCircuit make_ansatz() {
    QuantumCircuit qc(3);
    qc.h(0);
    qc.ry(0.7, 1);
    qc.rx(0.4, 2);
    qc.cx(0, 1);
    qc.t(2);
    qc.ry(0.9, 0);
    return qc;
}

// Mixed observable: several qubit-wise-commuting Z/I terms (grouped into one
// run) plus non-commuting X / Y terms (separate groups).
SparsePauliOp make_hamiltonian() {
    return SparsePauliOp::from_list({
        {"ZII", Complex128(0.5, 0.0)},
        {"IZI", Complex128(0.3, 0.0)},
        {"ZZI", Complex128(0.7, 0.0)},
        {"IIZ", Complex128(-0.4, 0.0)},
        {"XII", Complex128(0.4, 0.0)},
        {"IXX", Complex128(0.2, 0.0)},
        {"YII", Complex128(0.25, 0.0)},
    });
}

double run_with(bool group, int shots, uint64_t seed,
                const QuantumCircuit& qc, const SparsePauliOp& H) {
    Estimator::Options opts;
    opts.shots = shots;
    opts.seed = seed;
    opts.group_pauli_terms = group;
    Estimator est(opts);
    return est.run_single(qc, H);
}

} // namespace

TEST(R1131EstimatorGrouping, GroupingDoesNotChangeExactValue) {
    // shots == 0 is the exact statevector path; the grouping flag must not
    // affect it (it only reorders sampling).
    auto qc = make_ansatz();
    auto H = make_hamiltonian();
    double exact_grouped = run_with(true, 0, 0, qc, H);
    double exact_ungrouped = run_with(false, 0, 0, qc, H);
    EXPECT_NEAR(exact_grouped, exact_ungrouped, 1e-12);
}

TEST(R1131EstimatorGrouping, BothPathsConvergeToExact) {
    auto qc = make_ansatz();
    auto H = make_hamiltonian();
    const double exact = run_with(true, 0, 0, qc, H);

    const int shots = 200000;
    const double grouped = run_with(true, shots, 12345, qc, H);
    const double ungrouped = run_with(false, shots, 12345, qc, H);

    // ~sum|coeff| / sqrt(shots) ≈ 0.006 std error; 0.05 is a safe band.
    EXPECT_NEAR(grouped, exact, 0.05);
    EXPECT_NEAR(ungrouped, exact, 0.05);
    // The two sampling strategies must also agree with each other.
    EXPECT_NEAR(grouped, ungrouped, 0.05);
}

TEST(R1131EstimatorGrouping, EachPathIsSeedDeterministic) {
    auto qc = make_ansatz();
    auto H = make_hamiltonian();
    const int shots = 4096;
    const uint64_t seed = 987654321ULL;

    EXPECT_DOUBLE_EQ(run_with(true, shots, seed, qc, H),
                     run_with(true, shots, seed, qc, H));
    EXPECT_DOUBLE_EQ(run_with(false, shots, seed, qc, H),
                     run_with(false, shots, seed, qc, H));
}

TEST(R1131EstimatorGrouping, AllZTermObservableIsExactUnderGrouping) {
    // A pure Z/I observable is diagonal: sampling in the computational basis is
    // exact regardless of shots, and every term shares ONE grouped run.
    QuantumCircuit qc(2);
    qc.x(0);                 // |...01>  -> Z0 = -1, Z1 = +1
    auto H = SparsePauliOp::from_list({
        {"ZI", Complex128(1.0, 0.0)},   // Z on qubit 0
        {"IZ", Complex128(1.0, 0.0)},   // Z on qubit 1
    });
    double v = run_with(true, 1024, 7, qc, H);
    EXPECT_NEAR(v, 0.0, 1e-9);          // (-1) + (+1) = 0, no sampling noise
}
