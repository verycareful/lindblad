#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <algorithm>
#include <random>
#include <string>

namespace lindblad {
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
    auto best = std::max_element(res.counts.begin(), res.counts.end(),
        [](const auto& a, const auto& b){ return a.second < b.second; });
    // Per-shot execution measures only the n query qubits into the classical register,
    // so best->first has length n in MSB-first order. Reverse to get index order s[0]..s[n-1].
    std::string secret = best->first;
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

// =============================================================================
// DistributedBernsteinVazirani
//
// Circuit structure (combined register, n_total = Σ n_j):
//
//   Qubits: [0..n_0-1] party 0 query | ... | [n_total-1] | n_total = ancilla
//
//   1. X(ancilla)                         → ancilla = |1⟩
//   2. H(0..n_total)                      → query |+⟩^n, ancilla |−⟩
//   3. For each party j:
//        remap local oracle qubit k:
//          k < n_j   →  offset_j + k   (maps to this party's query slice)
//          k == n_j  →  n_total         (shared ancilla)
//        append remapped instructions
//   4. H(0..n_total-1)                    → decode phase kickback
//   5. measure(0..n_total-1)              → full secret (MSB-first)
//
// Quantum advantage: all t party oracles applied in a single circuit pass
// (1 communication round). Classical requires t separate rounds.
// Circuit depth: 2^max(n_j) + 3  vs  2^n + 3 for monolithic BV.
// =============================================================================

QuantumCircuit DistributedBernsteinVazirani::build_circuit(
    const std::vector<Party>& parties
) {
    int n_total = 0;
    for (const auto& p : parties) n_total += p.n_bits;
    const int ancilla = n_total;

    QuantumCircuit qc(n_total + 1, n_total);

    // Step 1-2: prepare |−⟩ on ancilla, |+⟩^n on query register
    qc.x(ancilla);
    for (int q = 0; q <= n_total; ++q) qc.h(q);

    // Step 3: remap and append each party's local oracle
    int offset = 0;
    for (const auto& p : parties) {
        for (auto inst : p.local_oracle.instructions) {
            // Translate qubit indices from local (0..n_j, n_j=ancilla)
            // to global (offset..offset+n_j-1, n_total=ancilla)
            for (int& q : inst.qubits) {
                q = (q < p.n_bits) ? (offset + q) : ancilla;
            }
            qc.instructions.push_back(inst);
        }
        offset += p.n_bits;
    }

    // Step 4: Hadamard on query register to decode phase kickback
    for (int q = 0; q < n_total; ++q) qc.h(q);

    // Step 5: measure the full query register
    for (int q = 0; q < n_total; ++q) qc.measure(q, q);

    return qc;
}

DistributedBernsteinVazirani::Result DistributedBernsteinVazirani::solve(
    const std::vector<Party>& parties,
    int shots, uint64_t seed
) {
    auto qc = build_circuit(parties);
    StatevectorSimulator sim;
    auto res = sim.run(qc, shots, seed);

    // Pick the most-frequent measurement outcome
    auto best = std::max_element(res.counts.begin(), res.counts.end(),
        [](const auto& a, const auto& b){ return a.second < b.second; });

    // MSB-first → index order (same convention as standard BV)
    std::string full = best->first;
    std::reverse(full.begin(), full.end());

    // Assemble result and slice full_secret into per-party portions
    Result result;
    result.full_secret      = full;
    result.num_parties      = static_cast<int>(parties.size());
    result.total_bits       = static_cast<int>(full.size());
    result.classical_rounds = result.num_parties;

    int offset = 0;
    for (const auto& p : parties) {
        result.party_secrets.push_back(full.substr(offset, p.n_bits));
        offset += p.n_bits;
    }

    return result;
}

} // namespace algorithms
} // namespace lindblad
