// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

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
