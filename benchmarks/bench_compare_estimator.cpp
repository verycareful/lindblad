// =============================================================================
// bench_compare_estimator.cpp -- Estimator vs Qiskit Aer EstimatorV2
// =============================================================================
// Heisenberg-chain expectation (3(n-1) + n Pauli terms) on the measure-free
// hardware-efficient ansatz from the corpus:
//   - exact:     shots = 0 (Lindblad) vs exact statevector expectation (Aer)
//   - shots4096: sampled, seed 42; Aer runs EstimatorV2 at the matching
//                precision 1/sqrt(4096). Lindblad's qubit-wise-commuting term
//                grouping (R.1.13, group_pauli_terms) stays at its default,
//                and Aer keeps its own grouping: out-of-box on both sides.
// The Estimator reuses one instance across iterations, so its transpile cache
// is warm after the first pass; Qiskit primitives cache similarly. Documented
// in docs/Benchmarks.md.

#include <benchmark/benchmark.h>

#include "lindblad/primitives.hpp"

#include "compare_common.hpp"

using namespace lindblad;
using namespace lindblad_bench;

static void BM_CmpEst(benchmark::State& state, const char* ansatz_file,
                      const char* obs_file, int shots) {
    const auto qc = load_corpus_circuit(ansatz_file, /*add_measure=*/false);
    const auto obs = load_observable(obs_file);
    Estimator est;
    est.options.shots = shots;
    est.options.seed = kSeed;
    for (auto _ : state) {
        double value = est.run_single(qc, obs);
        benchmark::DoNotOptimize(value);
    }
}

// Exact (shots = 0).
BENCHMARK_CAPTURE(BM_CmpEst, est__heisenberg__n12__exact, "ansatz_n12.qasm", "obs_heisenberg_n12.txt", 0)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpEst, est__heisenberg__n16__exact, "ansatz_n16.qasm", "obs_heisenberg_n16.txt", 0)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpEst, est__heisenberg__n20__exact, "ansatz_n20.qasm", "obs_heisenberg_n20.txt", 0)->Unit(benchmark::kMillisecond);

// Sampled (4096 shots).
BENCHMARK_CAPTURE(BM_CmpEst, est__heisenberg__n12__shots4096, "ansatz_n12.qasm", "obs_heisenberg_n12.txt", 4096)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpEst, est__heisenberg__n16__shots4096, "ansatz_n16.qasm", "obs_heisenberg_n16.txt", 4096)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpEst, est__heisenberg__n20__shots4096, "ansatz_n20.qasm", "obs_heisenberg_n20.txt", 4096)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
