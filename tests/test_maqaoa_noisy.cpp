#include <cmath>

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/noise.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

TEST(MAQAOANoisyTest, LayerwiseRunCompletes) {
    SparsePauliOp cost({
        PauliString("ZIIII", Complex128(1.0, 0.0)),
        PauliString("IZIII", Complex128(1.0, 0.0)),
        PauliString("IIZII", Complex128(1.0, 0.0)),
        PauliString("IIIZI", Complex128(1.0, 0.0)),
        PauliString("IIIIZ", Complex128(1.0, 0.0))
    });

    NoiseModel noise_model;
    noise_model.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.002), "h");
    noise_model.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.002), "rx");
    noise_model.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.01, 2), "cx");

    MAQAOA maqaoa;
    maqaoa.options.p = 1;
    maqaoa.options.layerwise = true;
    maqaoa.options.max_iterations = 6;
    maqaoa.options.convergence_threshold = 1e-3;
    maqaoa.options.seed = 7;
    maqaoa.options.initial_thetas = {0.2, 0.4, 0.6, 0.8, 1.0};
    maqaoa.estimator.options.noise_model = noise_model;
    maqaoa.sampler.options.shots = 64;
    maqaoa.sampler.options.seed = 13;
    maqaoa.sampler.options.noise_model = noise_model;

    auto result = maqaoa.optimize(cost);

    EXPECT_EQ(result.best_bitstring.size(), 5u);
    EXPECT_TRUE(std::isfinite(result.optimal_value));
    EXPECT_FALSE(result.counts.empty());
}
