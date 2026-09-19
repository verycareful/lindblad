// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include <gtest/gtest.h>
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

TEST(MAQAOATest, CircuitBuild) {
    // Simple 2-qubit cost Hamiltonian: ZZ
    SparsePauliOp cost({PauliString("ZZ", Complex128(1.0, 0.0))});

    MAQAOA maqaoa;
    maqaoa.options.p = 1;

    int n_params = maqaoa.num_parameters(cost);
    // qubit-indexed default: N gammas + N betas = 2*N = 4 params per layer
    EXPECT_EQ(n_params, 4);

    std::vector<double> params(n_params, 0.5);
    auto circuit = maqaoa.build_circuit(cost, {}, params);

    EXPECT_EQ(circuit.n_qubits, 2);
    EXPECT_GT(circuit.size(), 0);
}

TEST(MAQAOATest, MoreParams_MoreLayers) {
    SparsePauliOp cost({
        PauliString("ZI", Complex128(0.5, 0.0)),
        PauliString("IZ", Complex128(0.5, 0.0)),
        PauliString("ZZ", Complex128(1.0, 0.0))
    });

    MAQAOA maqaoa;
    maqaoa.options.p = 2;
    int n_params = maqaoa.num_parameters(cost);
    // qubit-indexed default: (N gammas + N betas) * p = (2+2)*2 = 8
    EXPECT_EQ(n_params, 8);
}

TEST(MAQAOATest, TermIndexedGammas) {
    // Verify term-indexed path still accessible via opt-in flag
    SparsePauliOp cost({PauliString("ZZ", Complex128(1.0, 0.0))});
    MAQAOA maqaoa;
    maqaoa.options.p = 1;
    maqaoa.options.term_indexed_gammas = true;
    // 1 cost term + 2 mixer qubits = 3 params per layer
    EXPECT_EQ(maqaoa.num_parameters(cost), 3);

    // Multi-layer
    maqaoa.options.p = 2;
    EXPECT_EQ(maqaoa.num_parameters(cost), 6);
}
