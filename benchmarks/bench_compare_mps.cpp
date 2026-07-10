// =============================================================================
// bench_compare_mps.cpp -- MPS backend vs Qiskit Aer 'matrix_product_state'
// =============================================================================
// Two sweeps on the layered scaling circuit (low, ladder-local entanglement,
// the regime MPS exists for):
//   - qubit sweep at fixed bond dimension chi = 32 (n = 16..40)
//   - bond sweep at fixed n = 24 (chi = 8..64)
// aer_bench.py mirrors both, setting matrix_product_state_max_bond_dimension
// to the same chi. Lindblad's qubit-MPS default SVD is Jacobi (R.1.13,
// accuracy-first); Aer uses its own SVD. Documented in docs/Benchmarks.md.

#include <benchmark/benchmark.h>

#include "lindblad/simulators/mps_sim.hpp"

#include "compare_common.hpp"

using namespace lindblad;
using namespace lindblad_bench;

static void BM_CmpMPS(benchmark::State& state, const char* file, int bond_dim) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/true);
    MPSSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, bond_dim, kShots, kSeed);
        benchmark::DoNotOptimize(r.counts);
    }
}

// Qubit sweep at chi = 32.
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n16__chi32, "scaling_n16.qasm", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n24__chi32, "scaling_n24.qasm", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n32__chi32, "scaling_n32.qasm", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n40__chi32, "scaling_n40.qasm", 32)->Unit(benchmark::kMillisecond);

// Bond sweep at n = 24 (chi = 32 point provided by the sweep above).
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n24__chi8,  "scaling_n24.qasm",  8)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n24__chi16, "scaling_n24.qasm", 16)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n24__chi64, "scaling_n24.qasm", 64)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
