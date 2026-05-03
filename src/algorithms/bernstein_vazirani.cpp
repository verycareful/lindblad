#include "qpp/algorithms.hpp"
#include "qpp/simulators/statevector_sim.hpp"

#include <algorithm>
#include <random>
#include <string>

namespace qpp {
namespace algorithms {

// =============================================================================
// BernsteinVazirani
// =============================================================================

QuantumCircuit BernsteinVazirani::build_circuit(const QuantumCircuit& oracle, int n) {
    QuantumCircuit qc(n + 1, n);
    qc.x(n);
    for (int i = 0; i <= n; ++i) qc.h(i);
    for (const auto& inst : oracle.instructions) qc.instructions.push_back(inst);
    for (int i = 0; i < n; ++i) qc.h(i);
    for (int i = 0; i < n; ++i) qc.measure(i, i);
    return qc;
}

BernsteinVazirani::Result BernsteinVazirani::solve(const QuantumCircuit& oracle, int n,
                                                    int shots, uint64_t seed) {
    auto qc = build_circuit(oracle, n);
    StatevectorSimulator sim;
    auto res = sim.run(qc, shots, seed);
    std::string bits = res.counts.begin()->first;
    std::string secret = bits.substr(1);          // drop ancilla at pos 0
    std::reverse(secret.begin(), secret.end());   // MSB-first → index order
    return { secret };
}

// =============================================================================
// RecursiveBernsteinVazirani
//
// Applies standard BV independently at each depth level.
// Quantum cost: d oracle calls (one per level).
// Classical cost: d × n deterministic queries.
// =============================================================================

RecursiveBernsteinVazirani::Result RecursiveBernsteinVazirani::solve(
    const std::vector<QuantumCircuit>& oracles, int n, int shots, uint64_t seed)
{
    Result result;
    result.depth = static_cast<int>(oracles.size());
    result.total_oracle_calls = result.depth;
    result.secrets.reserve(result.depth);

    for (int level = 0; level < result.depth; ++level) {
        auto bv = BernsteinVazirani::solve(oracles[level], n, shots, seed + static_cast<uint64_t>(level));
        result.secrets.push_back(std::move(bv.secret));
    }
    return result;
}

// =============================================================================
// ProbabilisticBernsteinVazirani  (Shukla & Vedula 2023, arXiv:2301.10014)
//
// Each shot draws one oracle from the pool according to `weights` (uniform if
// empty), runs BV once, and records which secret was returned.
//
// Quantum guarantee: every shot recovers the drawn secret with certainty —
// the phase kickback is exact regardless of which oracle was selected.
// Expected shots to see all K keys: K·H_K ≈ K·ln(K) (coupon collector).
// =============================================================================

ProbabilisticBernsteinVazirani::Result ProbabilisticBernsteinVazirani::solve(
    const std::vector<QuantumCircuit>& oracle_pool, int n,
    const std::vector<double>& weights, int shots, uint64_t seed)
{
    std::mt19937_64 rng(seed);

    std::vector<double> probs = weights;
    if (probs.empty())
        probs.assign(oracle_pool.size(), 1.0);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    Result result;
    result.shots_used = shots;

    for (int s = 0; s < shots; ++s) {
        int idx = dist(rng);
        auto bv = BernsteinVazirani::solve(oracle_pool[idx], n, 1, rng());
        result.key_counts[bv.secret]++;
    }

    result.discovered_keys.reserve(result.key_counts.size());
    for (const auto& [key, _] : result.key_counts)
        result.discovered_keys.push_back(key);
    std::sort(result.discovered_keys.begin(), result.discovered_keys.end());

    return result;
}

} // namespace algorithms
} // namespace qpp
