#include <gtest/gtest.h>
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

TEST(MAQAOATest, CircuitBuild) {
    // Simple 2-qubit cost Hamiltonian: ZZ
    SparsePauliOp cost({PauliString("ZZ", Complex128(1.0, 0.0))});

    MAQAOA maqaoa;
    maqaoa.options.p = 1;

    int n_params = maqaoa.num_parameters(cost);
    // 1 cost term + 2 mixer qubits = 3 params per layer
    EXPECT_EQ(n_params, 3);

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
    // (3 cost terms + 2 mixer) * 2 layers = 10
    EXPECT_EQ(n_params, 10);
}
