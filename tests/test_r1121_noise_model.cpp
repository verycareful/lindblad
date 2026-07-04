// R.1.12.1 total-coverage suite, Batch 2: lindblad/noise.hpp NoiseModel /
// ReadoutError. Plan: docs (R.1.12.1 coverage plan), section "Batch 2: engines".
//
// Covers gate-error attachment and matching (exact-qubit vs all-qubit, order
// sensitivity), before/after-gate ordering as an observable DM difference,
// is_ideal, from_t1_t2 (filtering + throws), and the ReadoutError assignment
// matrix. Includes one EXPECTED-RED test asserting that readout errors affect
// counts (the DM simulator does not yet apply them; ships red + issue per the
// new-bug policy). Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {
constexpr double kTol = 1e-9;
}

// =============================================================================
// is_ideal / attachment / matching
// =============================================================================

TEST(R1121NoiseModel, IsIdealUntilAnErrorIsAdded) {
    NoiseModel nm;
    EXPECT_TRUE(nm.is_ideal());
    nm.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.1, 1), "h");
    EXPECT_FALSE(nm.is_ideal());

    NoiseModel nr;
    nr.add_readout_error(ReadoutError{0.1, 0.05}, 0);
    EXPECT_FALSE(nr.is_ideal()) << "a readout error also breaks ideality";
}

TEST(R1121NoiseModel, ExactQubitErrorMatchesQubitListInOrder) {
    NoiseModel nm;
    nm.add_quantum_error(NoiseChannels::depolarizing(0.1, 2), "cx", {0, 1});
    EXPECT_EQ(nm.errors_for_gate("cx", {0, 1}).size(), 1u);
    EXPECT_EQ(nm.errors_for_gate("cx", {1, 0}).size(), 0u)
        << "qubit-list matching is order-sensitive";
    EXPECT_EQ(nm.errors_for_gate("cx", {2, 3}).size(), 0u);
    EXPECT_EQ(nm.errors_for_gate("cz", {0, 1}).size(), 0u) << "gate name must match";
}

TEST(R1121NoiseModel, AllQubitErrorMatchesAnyQubits) {
    NoiseModel nm;
    nm.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.2), "h");
    EXPECT_EQ(nm.errors_for_gate("h", {0}).size(), 1u);
    EXPECT_EQ(nm.errors_for_gate("h", {5}).size(), 1u);
}

// =============================================================================
// before/after gate ordering — observable DM difference
// =============================================================================

TEST(R1121NoiseModel, BeforeVsAfterGateOrderingIsObservable) {
    QuantumCircuit qc(1);
    qc.x(0);

    NoiseModel after;
    after.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.5), "x", true);
    NoiseModel before;
    before.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.5), "x", false);

    DensityMatrixSimulator sim;
    auto ra = sim.run(qc, after, 1, 42);
    auto rb = sim.run(qc, before, 1, 42);
    ASSERT_TRUE(ra.success);
    ASSERT_TRUE(rb.success);

    // after:  |0> -X-> |1> -damp-> population 0.5 in |1>.
    // before: |0> -damp-> |0> (fixed point) -X-> |1> with population 1.
    EXPECT_NEAR(ra.final_state(1, 1).real, 0.5, kTol);
    EXPECT_NEAR(rb.final_state(1, 1).real, 1.0, kTol);
}

// =============================================================================
// from_t1_t2 — gate_qubits filtering and validation throws
// =============================================================================

TEST(R1121NoiseModel, FromT1T2FiltersByGateQubits) {
    NoiseModel nm = NoiseModel::from_t1_t2(
        {100.0, 100.0}, {80.0, 80.0}, {{"x", 10.0}}, {{"x", {0}}});
    EXPECT_EQ(nm.errors_for_gate("x", {0}).size(), 1u);
    EXPECT_EQ(nm.errors_for_gate("x", {1}).size(), 0u)
        << "gate_qubits restricts the error to qubit 0 only";
    EXPECT_FALSE(nm.is_ideal());
}

TEST(R1121NoiseModel, FromT1T2ValidationThrows) {
    EXPECT_THROW(NoiseModel::from_t1_t2({100.0}, {250.0}, {{"x", 10.0}}),
                 std::invalid_argument);  // T2 > 2*T1
    EXPECT_THROW(NoiseModel::from_t1_t2({-1.0}, {1.0}, {{"x", 10.0}}),
                 std::invalid_argument);  // non-positive T1
    EXPECT_THROW(NoiseModel::from_t1_t2({100.0, 100.0}, {80.0}, {{"x", 10.0}}),
                 std::invalid_argument);  // length mismatch
}

// =============================================================================
// ReadoutError assignment matrix
// =============================================================================

TEST(R1121NoiseModel, AssignmentMatrixColumnsAreStochastic) {
    ReadoutError ro{/*prob_meas_0_prep_1=*/0.1, /*prob_meas_1_prep_0=*/0.05};
    auto m = ro.assignment_matrix();
    // M[measured][prepared]: each prepared-state column sums to 1.
    EXPECT_NEAR(m[0][0] + m[1][0], 1.0, kTol);  // prepared 0
    EXPECT_NEAR(m[0][1] + m[1][1], 1.0, kTol);  // prepared 1
    EXPECT_NEAR(m[0][0], 1.0 - 0.05, kTol);     // P(meas0|prep0)
    EXPECT_NEAR(m[1][0], 0.05, kTol);           // P(meas1|prep0)
    EXPECT_NEAR(m[0][1], 0.1, kTol);            // P(meas0|prep1)
    EXPECT_NEAR(m[1][1], 1.0 - 0.1, kTol);      // P(meas1|prep1)
}

// =============================================================================
// REGRESSION (shipped red in R.1.12.1, fixed in R.1.12.2): the
// DensityMatrixSimulator applies per-qubit readout errors to every sampled
// MEASURE outcome through the confusion matrix; the state itself stays
// collapsed to the true outcome.
// =============================================================================

TEST(R1121NoiseModel, ReadoutErrorPerturbsCounts) {
    QuantumCircuit qc(1, 1);
    qc.measure(0, 0);  // prepared |0>, measured ideally always "0"

    NoiseModel nm;
    nm.add_readout_error(ReadoutError{/*p0|1=*/0.0, /*p1|0=*/0.5}, 0);  // 50% 0->1 flips

    DensityMatrixSimulator sim;
    auto res = sim.run(qc, nm, 4000, 7);
    int ones = res.counts.count("1") ? res.counts.at("1") : 0;
    EXPECT_GT(ones, 1000)
        << "readout error P(meas1|prep0)=0.5 must produce ~half '1' outcomes";
}
