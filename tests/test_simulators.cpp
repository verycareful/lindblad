#include <gtest/gtest.h>
#include "qpp/circuit.hpp"
#include "qpp/simulators/statevector_sim.hpp"

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
