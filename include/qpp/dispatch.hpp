#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace qpp {

// =============================================================================
// SoftDispatchResult — extract fractional and integer dispatch schedules from
// MA-QAOA / QAOA bitstring distributions.
//
// The "soft dispatch" is the probability-weighted average spin assignment:
//   x_i_soft = sum_{bitstrings b} P(b) * b_i
// It represents the fractional commitment of each generator/variable.
//
// From this, an integer feasible solution is obtained by:
//   - Threshold rounding: x_i = 1 if x_i_soft > threshold, else 0
//   - Greedy rounding: sort by x_i_soft descending, commit generators until
//     demand constraint is met (suitable for QUBO-encoded dispatch problems).
// =============================================================================

struct SoftDispatchResult {
    // Probability distribution over bitstrings (from sampler counts)
    std::unordered_map<std::string, int> counts;
    int total_shots = 0;

    // Computed fields (filled by compute())
    std::vector<double> soft_assignment;   // fractional x_i ∈ [0,1]
    std::string best_bitstring;            // highest-probability bitstring
    double best_probability = 0.0;

    // Construct from raw sampler counts
    explicit SoftDispatchResult(const std::unordered_map<std::string, int>& counts_in);

    // Fill soft_assignment and best_bitstring from counts
    void compute();

    // Round soft_assignment to a binary solution by thresholding.
    // Returns a bitstring (MSB first).
    std::string threshold_round(double threshold = 0.5) const;

    // Greedy rounding for demand-constrained dispatch.
    // generator_capacities[i] = capacity of generator i (MW, units, etc.).
    // demand = total required capacity.
    // Selects generators in descending order of soft_assignment until demand is met.
    // Returns selected generator indices (0-indexed).
    std::vector<int> greedy_dispatch(
        const std::vector<double>& generator_capacities,
        double demand
    ) const;

    // Expected cost under the soft assignment, given an Ising cost function
    // evaluated at each sampled bitstring.
    // cost_fn: bitstring → cost (e.g., IsingHamiltonian::evaluate)
    double expected_cost(
        const std::function<double(const std::string&)>& cost_fn
    ) const;

    // Top-k most probable bitstrings with their probabilities
    std::vector<std::pair<std::string, double>> top_k(int k) const;
};

} // namespace qpp
