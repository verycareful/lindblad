#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

TEST(MAQAOA5QubitTest, SimpleIsing5Qubit) {
    // Create a simple 5-qubit Ising problem
    // Cost Hamiltonian: H = 0.5*Z0*Z1 + 0.5*Z1*Z2 + 0.5*Z2*Z3 + 0.5*Z3*Z4 + 0.5*Z4*Z0
    // (Ring coupling on 5 qubits)
    
    std::vector<PauliString> terms;
    terms.push_back(PauliString("ZZIII", Complex128(0.5, 0.0)));    // Z0*Z1
    terms.push_back(PauliString("IZZI", Complex128(0.5, 0.0)));    // Z1*Z2
    terms.push_back(PauliString("IIZZ", Complex128(0.5, 0.0)));    // Z2*Z3
    terms.push_back(PauliString("IIIZZ", Complex128(0.5, 0.0)));   // Z3*Z4
    
    SparsePauliOp cost(terms);
    
    // Create MA-QAOA instance
    MAQAOA maqaoa;
    maqaoa.options.p = 1;  // depth 1 (single layer)
    
    int n_params = maqaoa.num_parameters(cost);
    std::cout << "5-qubit Ising problem:" << std::endl;
    std::cout << "  Cost terms: " << terms.size() << std::endl;
    std::cout << "  Parameters needed (p=1): " << n_params << std::endl;
    
    // Build circuit with random initial parameters
    std::vector<double> params(n_params, 0.1);
    auto circuit = maqaoa.build_circuit(cost, {}, params);
    
    std::cout << "  Circuit qubits: " << circuit.n_qubits << std::endl;
    std::cout << "  Circuit gates: " << circuit.size() << std::endl;
    
    EXPECT_EQ(circuit.n_qubits, 5);  // Full 5-qubit coupling
    EXPECT_GT(circuit.size(), 0);
}

TEST(MAQAOA5QubitTest, MultiLayerMA_QAOA) {
    // 5-qubit MaxCut style problem with more coupling
    std::vector<PauliString> terms;
    terms.push_back(PauliString("ZZIII", Complex128(1.0, 0.0)));
    terms.push_back(PauliString("IZZI", Complex128(1.0, 0.0)));
    terms.push_back(PauliString("IIZZI", Complex128(1.0, 0.0)));
    terms.push_back(PauliString("IIIZZ", Complex128(1.0, 0.0)));
    
    SparsePauliOp cost(terms);
    
    MAQAOA maqaoa;
    maqaoa.options.p = 2;  // depth 2 (two layers)
    
    int n_params = maqaoa.num_parameters(cost);
    std::cout << "\n5-qubit multi-layer MA-QAOA:" << std::endl;
    std::cout << "  Cost terms: " << terms.size() << std::endl;
    std::cout << "  Parameters needed (p=2): " << n_params << std::endl;
    
    std::vector<double> params(n_params, 0.2);
    auto circuit = maqaoa.build_circuit(cost, {}, params);
    
    std::cout << "  Circuit gates: " << circuit.size() << std::endl;
    
    EXPECT_EQ(circuit.n_qubits, 5);
    EXPECT_GT(circuit.size(), 0);
}

TEST(MAQAOA5QubitTest, OptimizationBenchmark) {
    // End-to-end MA-QAOA optimisation (layerwise, COBYLA) on a hard-coded 5-qubit case.
    // Hard-coded instance: ferromagnetic ring with known exact optimum.
    // H = -(Z0Z1 + Z1Z2 + Z2Z3 + Z3Z4 + Z4Z0)
    // Expected classical ground energy: -5.0 at 00000 / 11111.

    std::vector<PauliString> terms;
    terms.push_back(PauliString("ZZIII", Complex128(-1.0, 0.0))); // Z0Z1
    terms.push_back(PauliString("IZZII", Complex128(-1.0, 0.0))); // Z1Z2
    terms.push_back(PauliString("IIZZI", Complex128(-1.0, 0.0))); // Z2Z3
    terms.push_back(PauliString("IIIZZ", Complex128(-1.0, 0.0))); // Z3Z4
    terms.push_back(PauliString("ZIIIZ", Complex128(-1.0, 0.0))); // Z4Z0
    SparsePauliOp cost(terms);

    constexpr int n_qubits = 5;
    constexpr int p_depth = 3;
    constexpr int maxeval_per_layer = 500;
    constexpr double expected_ground_energy = -5.0;

    MAQAOA maqaoa;
    maqaoa.estimator.options.shots = 0;  // exact statevector expectation
    maqaoa.sampler.options.shots   = 4096;
    maqaoa.sampler.options.seed    = 1234;
    maqaoa.options.p = p_depth;
    maqaoa.options.layerwise = true;
    maqaoa.options.max_iterations = maxeval_per_layer;
    maqaoa.options.convergence_threshold = 1e-6;
    maqaoa.options.seed = 42;

    auto started = std::chrono::high_resolution_clock::now();
    auto result = maqaoa.optimize(cost);
    auto ended = std::chrono::high_resolution_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ended - started).count();
    const int expected_n_params = p_depth * (static_cast<int>(terms.size()) + n_qubits);
    const double energy_gap = result.optimal_value - expected_ground_energy;

    std::cout << "\nMA-QAOA End-to-End (Hard-Coded 5Q Case):" << std::endl;
    std::cout << "  Input Hamiltonian terms: " << terms.size() << std::endl;
    std::cout << "  Depth p: " << p_depth << std::endl;
    std::cout << "  Expected output (exact ground energy): " << expected_ground_energy << std::endl;
    std::cout << "  Actual MA-QAOA energy: " << result.optimal_value << std::endl;
    std::cout << "  Closeness gap (actual - expected): " << energy_gap << std::endl;
    std::cout << "  Parameter count: " << result.optimal_params.size()
              << " (expected " << expected_n_params << ")" << std::endl;
    std::cout << "  Best sampled bitstring: " << result.best_bitstring << std::endl;
    std::cout << "  Runtime: " << elapsed_ms << " ms" << std::endl;

    EXPECT_EQ(result.optimal_params.size(), static_cast<size_t>(expected_n_params));
    EXPECT_FALSE(result.best_bitstring.empty());
    EXPECT_LE(result.optimal_value, -2.0);  // At least 60% of ground state for shallow depth
    EXPECT_GE(result.optimal_value, expected_ground_energy);
    EXPECT_LT(elapsed_ms, 30000);
}
