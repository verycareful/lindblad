// R.1.14.1 test patch -- comparison-benchmark corpus semantics (behavioural).
// The structural suite proves the corpus parses; this one proves the circuits
// COMPUTE what the generator documents, against independent references:
//   - grover_n8 concentrates on the documented NON-symmetric marked state with
//     the v-chain ancillas returned to |0> -- the same anchor the cross-engine
//     parity gate leans on, pinned here against the exact expected bitstring
//     so a qubit-ordering or ancilla regression fails loudly in ctest
//   - qft_n8 matches its closed-form product state, computed from first
//     principles in this file. NOTE: the corpus QFT is the ascending-loop
//     variant (each qubit i acquires phase 2*pi*0.k_i k_{i+1}...k_{n-1}, then
//     bit-reversal swaps); it is deliberately anchored to its own closed form,
//     NOT to QFT::build_circuit, which implements the canonical LSB-LSB DFT
//     (a different operator). Both engines run the same file, so the
//     benchmark comparison is unaffected.
//   - clifford_n8 is accepted by the Clifford backend and its sampled
//     distribution agrees with the statevector backend on the same file
// Non-symmetric inputs throughout, per the CLAUDE.md convention rule.

#include <gtest/gtest.h>

#include "compare_common.hpp"

#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <complex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;
using namespace lindblad_bench;

namespace {

constexpr int kShotsLocal = 8192;

int total(const std::unordered_map<std::string, int>& counts) {
    int t = 0;
    for (const auto& [k, v] : counts) t += v;
    return t;
}

} // namespace

// =============================================================================
// Grover: non-symmetric marked state, ancillas restored
// =============================================================================

TEST(R1141CorpusSemantics, GroverN8PeaksOnMarkedStateWithAncillasInZero) {
    // gen_circuits.py marks bit i = 0 iff i % 3 == 1 (s = 8):
    // b0..b7 = 1,0,1,1,0,1,1,0. Counts keys are q12..q0 (qubit 0 rightmost);
    // qubits 8..12 are the v-chain ancillas and must read 0.
    const std::string marked_key = "00000" "01101101";
    {
        // Cross-check the hardcoded key against the generator's rule.
        std::string derived(13, '0');
        for (int i = 0; i < 8; ++i) {
            if (i % 3 != 1) derived[12 - i] = '1';
        }
        ASSERT_EQ(derived, marked_key);
    }

    StatevectorSimulator sim;
    const auto qc = load_corpus_circuit("grover_n8.qasm", /*add_measure=*/true);
    ASSERT_EQ(qc.n_qubits, 13);
    const auto counts = sim.run(qc, kShotsLocal, kSeed).counts;

    ASSERT_EQ(total(counts), kShotsLocal);
    for (const auto& [key, c] : counts) {
        ASSERT_EQ(key.size(), 13u);
        EXPECT_EQ(key.substr(0, 5), "00000")
            << "ancilla not restored to |0> in outcome " << key;
    }
    // 12 iterations at s = 8 leave the marked state at ~99.99% probability.
    const auto it = counts.find(marked_key);
    ASSERT_NE(it, counts.end()) << "marked state never sampled";
    EXPECT_GT(it->second, static_cast<int>(0.9 * kShotsLocal))
        << "marked state " << marked_key << " not dominant";
}

// =============================================================================
// QFT: closed-form product-state reference on a non-symmetric input
// =============================================================================

TEST(R1141CorpusSemantics, QftN8MatchesClosedFormOnNonSymmetricInput) {
    constexpr int n = 8;
    constexpr int k = 5;  // 00000101: non-symmetric (docs worked-example value)

    // Prepare |k> in front of the corpus QFT gates.
    QuantumCircuit qc(n);
    for (int q = 0; q < n; ++q) {
        if ((k >> q) & 1) qc.x(q);
    }
    const auto corpus = load_corpus_circuit("qft_n8.qasm", /*add_measure=*/false);
    for (const auto& ins : corpus.instructions) qc.instructions.push_back(ins);

    StatevectorSimulator sim;
    const auto amps = sim.run(qc, 0, kSeed).final_state.amplitudes();
    ASSERT_EQ(amps.size(), static_cast<size_t>(1 << n));

    // Closed form of the generator's construction: before the swap layer,
    // qubit i carries phase phi_i = 2*pi * (0.k_i k_{i+1} ... k_{n-1}); the
    // final swaps relabel i <-> n-1-i, so output qubit q carries phi_{n-1-q}.
    std::vector<double> phi(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double frac = 0.0;
        for (int j = i; j < n; ++j) {
            frac += ((k >> j) & 1) ? std::pow(0.5, j - i + 1) : 0.0;
        }
        phi[i] = 2.0 * PI * frac;
    }
    const double norm = std::pow(0.5, n / 2.0);
    for (int m = 0; m < (1 << n); ++m) {
        double phase = 0.0;
        for (int q = 0; q < n; ++q) {
            if ((m >> q) & 1) phase += phi[n - 1 - q];
        }
        // Complex128 is the project's SIMD struct (public .real/.imag data
        // members), not std::complex, so build e^{i*phase} by components.
        const Complex128 expected(norm * std::cos(phase), norm * std::sin(phase));
        const Complex128 diff = amps[m] - expected;
        EXPECT_NEAR(std::sqrt(diff.real * diff.real + diff.imag * diff.imag),
                    0.0, 1e-9)
            << "amplitude mismatch at m=" << m;
    }
}

// =============================================================================
// Clifford: backend acceptance + agreement with statevector on the same file
// =============================================================================

TEST(R1141CorpusSemantics, CliffordN8RunsOnTableauAndAgreesWithStatevector) {
    const auto qc = load_corpus_circuit("clifford_n8.qasm", /*add_measure=*/true);

    CliffordSimulator cliff;
    StatevectorSimulator sv;
    std::unordered_map<std::string, int> tab_counts, sv_counts;
    ASSERT_NO_THROW(tab_counts = cliff.run(qc, kShotsLocal, kSeed).counts)
        << "corpus Clifford circuit rejected by the tableau backend";
    sv_counts = sv.run(qc, kShotsLocal, kSeed).counts;

    ASSERT_EQ(total(tab_counts), kShotsLocal);
    ASSERT_EQ(total(sv_counts), kShotsLocal);

    // Same distribution within sampling noise: TVD between two independent
    // N-shot samples of one K-outcome distribution scales as sqrt(K/(pi*N))
    // (the bench_report.py parity-gate threshold; 3x guards against flakes
    // while a backend divergence produces TVD near 1).
    std::unordered_map<std::string, double> p;
    for (const auto& [key, c] : tab_counts) p[key] += c / double(kShotsLocal);
    for (const auto& [key, c] : sv_counts) p[key] -= c / double(kShotsLocal);
    double tvd = 0.0;
    for (const auto& [key, d] : p) tvd += std::abs(d);
    tvd *= 0.5;

    const double support = static_cast<double>(p.size());
    const double noise_scale = std::sqrt(support / (PI * kShotsLocal));
    EXPECT_LT(tvd, 3.0 * noise_scale + 0.01)
        << "tableau vs statevector distributions diverge (TVD " << tvd << ")";
}
