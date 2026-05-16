#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"

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

// =============================================================================
// QuditBernsteinVazirani — recovers s in Z_d^n from f(x) = s·x mod d in 1 query
// =============================================================================

std::vector<Complex128> QuditBernsteinVazirani::oracle_gate(int d, int s_i) {
    return qudit_gates::cadd_matrix(d, s_i);
}

QuditBernsteinVazirani::Result QuditBernsteinVazirani::solve(
    const std::vector<int>& secret, int d, int shots, uint64_t seed,
    QuditBackend backend, const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument("QuditBernsteinVazirani::solve: d must be >= 2");
    if (secret.empty())
        throw std::invalid_argument("QuditBernsteinVazirani::solve: secret must not be empty");
    for (int x : secret)
        if (x < 0 || x >= d)
            throw std::invalid_argument("QuditBernsteinVazirani::solve: all secret values must be in [0, d)");
    if (shots < 1) shots = 1;

    const int n = static_cast<int>(secret.size());

    const auto Fd  = qudit_gates::qft_matrix(d);
    const auto Fdi = qudit_gates::iqft_matrix(d);
    const auto Xm  = qudit_gates::shift_matrix(d, d - 1);

    std::vector<std::vector<Complex128>> cadd(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        if (secret[static_cast<size_t>(i)] != 0)
            cadd[static_cast<size_t>(i)] = qudit_gates::cadd_matrix(d, secret[static_cast<size_t>(i)]);

    std::vector<std::vector<int>> votes(static_cast<size_t>(n), std::vector<int>(static_cast<size_t>(d), 0));

    // ── CLIFFORD path ────────────────────────────────────────────────────────
    if (backend == QuditBackend::CLIFFORD) {
        if (!QuditCliffordSimulator::is_prime(d))
            throw std::invalid_argument(
                "QuditBernsteinVazirani::solve: CLIFFORD backend requires prime d");

        for (int shot = 0; shot < shots; ++shot) {
            QuditCliffordSimulator c(n + 1, d);

            // Step 1: X^{d-1} on ancilla qudit n → |d-1>
            c.apply_X(n, d - 1);
            // Step 2: H (QFT) on ancilla → |->_d
            c.apply_H(n);
            // Step 3: H on each query qudit
            for (int i = 0; i < n; ++i) c.apply_H(i);
            // Step 4: CADD(s_i) from query_i to ancilla  (CADD(k) = CSUM applied k times)
            for (int i = 0; i < n; ++i)
                for (int rep = 0; rep < secret[static_cast<size_t>(i)]; ++rep)
                    c.apply_CSUM(i, n);
            // Step 5: H† = H^3 (since H^4 = I in the qudit Clifford group) on each query qudit
            for (int i = 0; i < n; ++i) {
                c.apply_H(i); c.apply_H(i); c.apply_H(i);
            }
            // Step 6: measure
            auto outcome = c.measure(seed + static_cast<uint64_t>(shot));
            for (int i = 0; i < n; ++i)
                votes[static_cast<size_t>(i)][static_cast<size_t>(outcome[static_cast<size_t>(i)])]++;
        }

        std::vector<int> recovered(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const auto& v = votes[static_cast<size_t>(i)];
            recovered[static_cast<size_t>(i)] = static_cast<int>(
                std::max_element(v.begin(), v.end()) - v.begin());
        }
        return Result{recovered, d, n};
    }

    // ── DENSITY_MATRIX path ──────────────────────────────────────────────────
    if (backend == QuditBackend::DENSITY_MATRIX) {
        for (int shot = 0; shot < shots; ++shot) {
            QuditDensityMatrix dm(n + 1, d);

            dm.apply_1qudit(n, Xm);
            dm.apply_1qudit(n, Fd);
            for (int i = 0; i < n; ++i) dm.apply_1qudit(i, Fd);
            for (int i = 0; i < n; ++i)
                if (secret[static_cast<size_t>(i)] != 0)
                    dm.apply_2qudit(i, n, cadd[static_cast<size_t>(i)]);
            for (int i = 0; i < n; ++i) dm.apply_1qudit(i, Fdi);
            if (noise) dm.apply_noise(*noise);

            auto outcome = dm.measure(seed + static_cast<uint64_t>(shot));
            for (int i = 0; i < n; ++i)
                votes[static_cast<size_t>(i)][static_cast<size_t>(outcome[static_cast<size_t>(i)])]++;
        }

        std::vector<int> recovered(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const auto& v = votes[static_cast<size_t>(i)];
            recovered[static_cast<size_t>(i)] = static_cast<int>(
                std::max_element(v.begin(), v.end()) - v.begin());
        }
        return Result{recovered, d, n};
    }

    // ── MPS path ─────────────────────────────────────────────────────────────
    if (backend == QuditBackend::MPS) {
        for (int shot = 0; shot < shots; ++shot) {
            QuditMPS mps(n + 1, d);

            mps.apply_1qudit(n, Xm);
            mps.apply_1qudit(n, Fd);
            for (int i = 0; i < n; ++i) mps.apply_1qudit(i, Fd);
            for (int i = 0; i < n; ++i)
                if (secret[static_cast<size_t>(i)] != 0)
                    mps.apply_2qudit(i, n, cadd[static_cast<size_t>(i)]);
            for (int i = 0; i < n; ++i) mps.apply_1qudit(i, Fdi);

            auto outcome = mps.measure(seed + static_cast<uint64_t>(shot));
            for (int i = 0; i < n; ++i)
                votes[static_cast<size_t>(i)][static_cast<size_t>(outcome[static_cast<size_t>(i)])]++;
        }

        std::vector<int> recovered(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const auto& v = votes[static_cast<size_t>(i)];
            recovered[static_cast<size_t>(i)] = static_cast<int>(
                std::max_element(v.begin(), v.end()) - v.begin());
        }
        return Result{recovered, d, n};
    }

    // ── STATEVECTOR path (default) ───────────────────────────────────────────
    for (int shot = 0; shot < shots; ++shot) {
        QuditStatevector sv(n + 1, d);

        std::vector<QuditGateOp> ops;
        ops.reserve(static_cast<size_t>(2 + 3 * n));

        ops.push_back({QuditGateOp::Type::SINGLE, n, -1, Xm});
        ops.push_back({QuditGateOp::Type::SINGLE, n, -1, Fd});
        for (int i = 0; i < n; ++i)
            ops.push_back({QuditGateOp::Type::SINGLE, i, -1, Fd});
        for (int i = 0; i < n; ++i)
            if (secret[static_cast<size_t>(i)] != 0)
                ops.push_back({QuditGateOp::Type::TWO, i, n, cadd[static_cast<size_t>(i)]});
        for (int i = 0; i < n; ++i)
            ops.push_back({QuditGateOp::Type::SINGLE, i, -1, Fdi});

        auto res = QuditSimulator::run(sv, ops, seed + static_cast<uint64_t>(shot));
        for (int i = 0; i < n; ++i)
            votes[static_cast<size_t>(i)][static_cast<size_t>(res.outcome[static_cast<size_t>(i)])]++;
    }

    std::vector<int> recovered(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& v = votes[static_cast<size_t>(i)];
        recovered[static_cast<size_t>(i)] = static_cast<int>(
            std::max_element(v.begin(), v.end()) - v.begin());
    }
    return Result{recovered, d, n};
}

} // namespace algorithms
} // namespace lindblad
