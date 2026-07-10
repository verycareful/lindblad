// =============================================================================
// bench_compare_clifford.cpp -- Clifford backend vs Qiskit Aer 'stabilizer'
// =============================================================================
// H + CX ladder + S layered circuits (10 layers): pure stabilizer content, so
// both engines take their tableau paths. Sizes extend well past statevector
// reach (n = 160) since tableau cost is polynomial in n.

#include <benchmark/benchmark.h>

#include "lindblad/simulators/clifford_sim.hpp"

#include "compare_common.hpp"

using namespace lindblad;
using namespace lindblad_bench;

static void BM_CmpClifford(benchmark::State& state, const char* file) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/true);
    CliffordSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, kShots, kSeed);
        benchmark::DoNotOptimize(r.counts);
    }
}

BENCHMARK_CAPTURE(BM_CmpClifford, clifford__ladder__n20,  "clifford_n20.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpClifford, clifford__ladder__n40,  "clifford_n40.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpClifford, clifford__ladder__n80,  "clifford_n80.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpClifford, clifford__ladder__n160, "clifford_n160.qasm")->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
