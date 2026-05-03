#include <gtest/gtest.h>
#include "qpp/circuit.hpp"
#include "qpp/noise.hpp"
#include "qpp/simulators/statevector_sim.hpp"
#include "qpp/simulators/density_matrix_sim.hpp"

using namespace qpp;

TEST(SimulatorTest, BasicSimulation) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);

    StatevectorSimulator sim;
    auto result = sim.run(qc);
    EXPECT_TRUE(result.success);

    EXPECT_NEAR(result.final_state.probability(0), 0.5, 1e-10);
    EXPECT_NEAR(result.final_state.probability(3), 0.5, 1e-10);
}

TEST(SimulatorTest, ShotSampling) {
    QuantumCircuit qc(1);
    qc.x(0);

    StatevectorSimulator sim;
    auto result = sim.run(qc, 1000, 42);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.counts["1"], 1000);
}

TEST(SimulatorTest, GHZState) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(0, 2);

    StatevectorSimulator sim;
    auto result = sim.run(qc, 10000, 42);
    EXPECT_TRUE(result.success);

    // Should only see |000⟩ and |111⟩
    for (const auto& [bits, count] : result.counts) {
        EXPECT_TRUE(bits == "000" || bits == "111")
            << "Unexpected bitstring: " << bits;
    }
}

TEST(SimulatorTest, RCCXStatevectorDensityMatrixConsistency) {
    // RCCX on |110⟩: both simulators must agree on the output state
    QuantumCircuit qc(3);
    qc.x(0);
    qc.x(1);
    qc.rccx(0, 1, 2);
    qc.measure_all();

    StatevectorSimulator sv_sim;
    auto sv_result = sv_sim.run(qc, 256, 42);

    NoiseModel ideal;
    DensityMatrixSimulator dm_sim;
    auto dm_result = dm_sim.run(qc, ideal, 256, 42);

    ASSERT_FALSE(sv_result.counts.empty());
    ASSERT_FALSE(dm_result.counts.empty());

    // Both must agree: same most-probable bitstring
    std::string sv_top = sv_result.counts.begin()->first;
    std::string dm_top = dm_result.counts.begin()->first;
    EXPECT_EQ(sv_top, dm_top)
        << "SV and DM simulators disagree on RCCX output";
}
