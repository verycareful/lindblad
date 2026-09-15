
#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/dispatch.hpp"
#include "lindblad/ising.hpp"
#include "lindblad/operators.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "r1211_policy_probe.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

namespace {

SparsePauliOp vqe_hamiltonian() {
    return SparsePauliOp({PauliString("Z")});
}

QuantumCircuit vqe_ansatz() {
    QuantumCircuit ansatz(1);
    ansatz.ry("theta", 0);
    return ansatz;
}

SparsePauliOp qaoa_cost_hamiltonian() {
    return SparsePauliOp({PauliString("Z")});
}

SparsePauliOp qaoa_mixer_hamiltonian() {
    return SparsePauliOp({PauliString("X")});
}

} // namespace

// VQE no longer says converged when evaluation limit is reached

TEST(VQE, ConvergenceFlag) {
    VQE vqe;
    vqe.options.max_iterations = 1;  // limit NLopt to one objective evaluation

    auto hamiltonian = vqe_hamiltonian();
    auto ansatz = vqe_ansatz();

    auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(is_finite_strict(result.eigenvalue));
    EXPECT_FALSE(result.energy_history.empty());
}

// VQE optimizer selection and unknown-optimizer warning behavior.

TEST(VQE, UnknownOptimizer) {
    
    VQE vqe;
    vqe.options.optimizer = "UNKNOWN_OPTIMIZER";

    auto hamiltonian = vqe_hamiltonian();
    auto ansatz = vqe_ansatz();

    EXPECT_NO_THROW({
        auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);
        EXPECT_TRUE(is_finite_strict(result.eigenvalue));
        EXPECT_FALSE(result.energy_history.empty());
    });

}

TEST(VQE, ValidOptimizer_neldermead) {
    
    VQE vqe;
    vqe.options.optimizer = "NELDER_MEAD";

    auto hamiltonian = vqe_hamiltonian();
    auto ansatz = vqe_ansatz();

    EXPECT_NO_THROW({
        auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);
        EXPECT_TRUE(is_finite_strict(result.eigenvalue));
        EXPECT_FALSE(result.energy_history.empty());
    });

}

TEST(VQE, ValidOptimizer_cobyla) {
    
    VQE vqe;
    vqe.options.optimizer = "COBYLA";

    auto hamiltonian = vqe_hamiltonian();
    auto ansatz = vqe_ansatz();

    EXPECT_NO_THROW({
        auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);
        EXPECT_TRUE(is_finite_strict(result.eigenvalue));
        EXPECT_FALSE(result.energy_history.empty());
    });

}

TEST(VQE, ValidOptimizer_bobyqa) {
    
    VQE vqe;
    vqe.options.optimizer = "BOBYQA";

    auto hamiltonian = vqe_hamiltonian();
    auto ansatz = vqe_ansatz();

    EXPECT_NO_THROW({
        auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);
        EXPECT_TRUE(is_finite_strict(result.eigenvalue));
        EXPECT_FALSE(result.energy_history.empty());
    });

}

TEST(VQE, DefaultOptimizer) {
    
    VQE vqe;

    auto hamiltonian = vqe_hamiltonian();
    auto ansatz = vqe_ansatz();

    EXPECT_NO_THROW({
        auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);
        EXPECT_TRUE(is_finite_strict(result.eigenvalue));
        EXPECT_FALSE(result.energy_history.empty());
    });

}

TEST(VQE, UnknownOptimizerWarnsBeforeDefaultingToCobyla) {
    VQE vqe;
    vqe.options.optimizer = "UNKNOWN_OPTIMIZER";

    auto hamiltonian = vqe_hamiltonian();
    auto ansatz = vqe_ansatz();

    r1211::WarningProbe probe;

    EXPECT_NO_THROW({
        const auto result =
            vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);

        EXPECT_FALSE(result.energy_history.empty());
    });

    EXPECT_TRUE(probe.any_contains("unknown optimizer"));
    EXPECT_TRUE(probe.any_contains("defaulting to COBYLA"));
}

// QAOA optimizer selection and unknown-optimizer warning behavior.

TEST(QAOA, UnknownOptimizer) {
    
    QAOA qaoa;
    qaoa.options.optimizer = "UNKNOWN_OPTIMIZER";

    auto cost_hamiltonian = qaoa_cost_hamiltonian();
    auto mixer = qaoa_mixer_hamiltonian();

    EXPECT_NO_THROW({
        auto result = qaoa.optimize(cost_hamiltonian, mixer);
        EXPECT_TRUE(is_finite_strict(result.optimal_value));
        EXPECT_FALSE(result.optimal_params.empty());
    });

}

TEST(QAOA, ValidOptimizer_neldermead) {
    
    QAOA qaoa;
    qaoa.options.optimizer = "NELDER_MEAD";

    auto cost_hamiltonian = qaoa_cost_hamiltonian();
    auto mixer = qaoa_mixer_hamiltonian();

    EXPECT_NO_THROW({
        auto result = qaoa.optimize(cost_hamiltonian, mixer);
        EXPECT_TRUE(is_finite_strict(result.optimal_value));
        EXPECT_FALSE(result.optimal_params.empty());
    });

}

TEST(QAOA, ValidOptimizer_cobyla) {
    
    QAOA qaoa;
    qaoa.options.optimizer = "COBYLA";

    auto cost_hamiltonian = qaoa_cost_hamiltonian();
    auto mixer = qaoa_mixer_hamiltonian();

    EXPECT_NO_THROW({
        auto result = qaoa.optimize(cost_hamiltonian, mixer);
        EXPECT_TRUE(is_finite_strict(result.optimal_value));
        EXPECT_FALSE(result.optimal_params.empty());
    });

}

TEST(QAOA, ValidOptimizer_bobyqa) {
    
    QAOA qaoa;
    qaoa.options.optimizer = "BOBYQA";

    auto cost_hamiltonian = qaoa_cost_hamiltonian();
    auto mixer = qaoa_mixer_hamiltonian();

    EXPECT_NO_THROW({
        auto result = qaoa.optimize(cost_hamiltonian, mixer);
        EXPECT_TRUE(is_finite_strict(result.optimal_value));
        EXPECT_FALSE(result.optimal_params.empty());
    });

}

TEST(QAOA, DefaultOptimizer) {
    
    QAOA qaoa;

    auto cost_hamiltonian = qaoa_cost_hamiltonian();
    auto mixer = qaoa_mixer_hamiltonian();

    EXPECT_NO_THROW({
        auto result = qaoa.optimize(cost_hamiltonian, mixer);
        EXPECT_TRUE(is_finite_strict(result.optimal_value));
        EXPECT_FALSE(result.optimal_params.empty());
    });

}

TEST(QAOA, UnknownOptimizerWarnsBeforeDefaultingToCobyla) {
    QAOA qaoa;
    qaoa.options.optimizer = "UNKNOWN_OPTIMIZER";

    auto cost_hamiltonian = qaoa_cost_hamiltonian();
    auto mixer = qaoa_mixer_hamiltonian();

    r1211::WarningProbe probe;

    EXPECT_NO_THROW({
        const auto result = qaoa.optimize(cost_hamiltonian, mixer);
        EXPECT_FALSE(result.optimal_params.empty());
    });

    EXPECT_TRUE(probe.any_contains("unknown optimizer"));
    EXPECT_TRUE(probe.any_contains("defaulting to COBYLA"));
}
