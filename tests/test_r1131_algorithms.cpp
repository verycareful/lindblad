// R.1.13.1 test patch — algorithm end-to-end guards for R.1.13.0 changes.
// Simon batch_shots (audit F-21), Grover diffusion lowered to MCX (F-7), and
// Shor's controlled modular multiplication emitted as a PERMUTATION oracle
// (F-9). These verify the optimised paths still produce correct answers.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

namespace {

// Standard textbook Simon oracle for period s (matches test_classic_algorithms).
QuantumCircuit simon_oracle(const std::string& s) {
    const int n = static_cast<int>(s.size());
    QuantumCircuit qc(2 * n);
    for (int i = 0; i < n; ++i) qc.cx(i, n + i);
    int k = -1;
    for (int i = 0; i < n; ++i) if (s[i] == '1') { k = i; break; }
    if (k >= 0)
        for (int i = 0; i < n; ++i)
            if (s[i] == '1') qc.cx(k, n + i);
    return qc;
}

// Diagonal phase oracle marking a single basis state with -1.
QuantumCircuit grover_oracle(int n, size_t target) {
    const size_t dim = size_t(1) << n;
    std::vector<Complex128> m(dim * dim, Complex128(0, 0));
    for (size_t i = 0; i < dim; ++i)
        m[i * dim + i] = Complex128(i == target ? -1.0 : 1.0, 0.0);
    QuantumCircuit oracle(n);
    std::vector<int> qs;
    for (int q = 0; q < n; ++q) qs.push_back(q);
    oracle.unitary(m, qs);
    return oracle;
}

} // namespace

// F-21: both batch and per-sample equation harvesting recover the secret.
TEST(R1131Algorithms, SimonBatchAndPerSampleRecoverSecret) {
    for (const std::string& s : {std::string("101"), std::string("110"),
                                 std::string("1010")}) {
        const int n = static_cast<int>(s.size());
        auto batched = Simon::solve(simon_oracle(s), n, /*seed=*/42,
                                    /*extra_samples=*/2, /*batch_shots=*/true);
        auto per_sample = Simon::solve(simon_oracle(s), n, /*seed=*/42,
                                       /*extra_samples=*/2, /*batch_shots=*/false);
        EXPECT_EQ(batched.period, s) << "batched Simon failed for s=" << s;
        EXPECT_EQ(per_sample.period, s) << "per-sample Simon failed for s=" << s;
    }
}

// Determinism: each Simon path reproduces its result under a fixed seed.
TEST(R1131Algorithms, SimonPathsAreSeedDeterministic) {
    const std::string s = "1010";
    const int n = 4;
    auto a = Simon::solve(simon_oracle(s), n, 7, 2, true);
    auto b = Simon::solve(simon_oracle(s), n, 7, 2, true);
    EXPECT_EQ(a.period, b.period);
    auto c = Simon::solve(simon_oracle(s), n, 7, 2, false);
    auto d = Simon::solve(simon_oracle(s), n, 7, 2, false);
    EXPECT_EQ(c.period, d.period);
}

// F-7: Grover's diffusion (now built from qc.mcx) still amplifies the marked
// item to a dominant probability.
TEST(R1131Algorithms, GroverDiffusionViaMcxFindsMarked) {
    const int n = 4;
    const size_t target = 5;   // |0101>
    auto result = Grover::search(grover_oracle(n, target), -1, 2048, 42);
    EXPECT_EQ(result.solution, "0101");
    EXPECT_GT(result.probability, 0.5);
}

// F-9 + R.1.13.0 MPS fallback: Shor's controlled-modmult PERMUTATION oracle runs
// on the MPS backend via the bounded statevector fallback (the regression was a
// hard throw). At 8 qubits the entanglement bond (<= 16) fits well within the
// bond dim (64), so the MPS state must match the exact statevector amplitude for
// amplitude — a rigorous correctness check of the oracle fallback on the REAL
// Shor circuit.
//
// NOTE: this deliberately does NOT assert order recovery on MPS. Full N=15
// order finding needs the 13-qubit high-entanglement QPE circuit, where MPS
// bond-dim truncation makes the result approximate; exact order recovery is a
// statevector claim covered by ShorTest.FindOrderA2N15.
TEST(R1131Algorithms, ShorPermutationOracleMpsFallbackMatchesStatevector) {
    auto qc = Shor::build_period_finding_circuit(2, 15, /*n_eval=*/4, /*n_target=*/4);
    ASSERT_EQ(qc.n_qubits, 8);

    StatevectorSimulator svsim;
    auto sv = svsim.run(qc, 0, 0).final_state.amplitudes();

    MPSSimulator mpssim;
    auto mps = mpssim.run(qc, /*max_bond_dim=*/64, 0, 0)
                   .final_state.to_statevector().amplitudes();

    ASSERT_EQ(sv.size(), mps.size());
    for (size_t i = 0; i < sv.size(); ++i) {
        EXPECT_NEAR(mps[i].real, sv[i].real, 1e-9) << "re @ " << i;
        EXPECT_NEAR(mps[i].imag, sv[i].imag, 1e-9) << "im @ " << i;
    }
}

// Full end-to-end: factorize 15 via the quantum path (PERMUTATION oracle).
TEST(R1131Algorithms, ShorFactorsFifteen) {
    Shor::Options opts;
    opts.seed = 12345;
    opts.max_attempts = 12;
    Shor shor(opts);
    auto r = shor.factorize(15);

    ASSERT_TRUE(r.success) << "Shor failed to factor 15";
    EXPECT_EQ(r.factor * r.cofactor, 15u);
    EXPECT_TRUE(r.factor == 3u || r.factor == 5u) << "factor = " << r.factor;
}
