#include <gtest/gtest.h>
#include "lindblad/noise.hpp"

using namespace lindblad;

TEST(NoiseTest, DepolarizingChannelValid) {
    auto ch = NoiseChannels::depolarizing(0.1);
    EXPECT_TRUE(ch.is_valid());
    EXPECT_EQ(ch.operators.size(), 4u);
}

TEST(NoiseTest, AmplitudeDampingValid) {
    auto ch = NoiseChannels::amplitude_damping(0.5);
    EXPECT_TRUE(ch.is_valid());
    EXPECT_EQ(ch.operators.size(), 2u);
}

TEST(NoiseTest, PhaseDampingValid) {
    auto ch = NoiseChannels::phase_damping(0.3);
    EXPECT_TRUE(ch.is_valid());
}

TEST(NoiseTest, PauliChannelValid) {
    auto ch = NoiseChannels::pauli(0.1, 0.2, 0.3);
    EXPECT_TRUE(ch.is_valid());
}

TEST(NoiseTest, BitFlipValid) {
    auto ch = NoiseChannels::bit_flip(0.1);
    EXPECT_TRUE(ch.is_valid());
}

TEST(NoiseTest, NoiseModelSetup) {
    NoiseModel nm;
    EXPECT_TRUE(nm.is_ideal());

    nm.add_quantum_error(NoiseChannels::depolarizing(0.01), "cx");
    EXPECT_FALSE(nm.is_ideal());

    auto errors = nm.errors_for_gate("cx", {0, 1});
    EXPECT_EQ(errors.size(), 1u);
}
