// =============================================================================
// bench_compare_sv.cpp -- statevector backend vs Qiskit Aer 'statevector'
// =============================================================================
// Timed region: simulator run only (parse + measure_all happen in setup, and
// aer_bench.py likewise excludes circuit loading/transpilation from timing).
// Benchmark labels ("sv__<workload>__<param>") are the pairing keys consumed
// by tools/bench_report.py; aer_bench.py emits the identical strings.

#include <benchmark/benchmark.h>

#include "lindblad/simulators/statevector_sim.hpp"

#include "compare_common.hpp"

using namespace lindblad;
using namespace lindblad_bench;

static void BM_CmpSV(benchmark::State& state, const char* file) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/true);
    StatevectorSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, kShots, kSeed);
        benchmark::DoNotOptimize(r.counts);
    }
}

// Layered H + CX ladder + RZ scaling circuit.
BENCHMARK_CAPTURE(BM_CmpSV, sv__scaling__n10, "scaling_n10.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__scaling__n14, "scaling_n14.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__scaling__n18, "scaling_n18.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__scaling__n22, "scaling_n22.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__scaling__n26, "scaling_n26.qasm")->Unit(benchmark::kMillisecond);

// QFT with final swaps.
BENCHMARK_CAPTURE(BM_CmpSV, sv__qft__n10, "qft_n10.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__qft__n14, "qft_n14.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__qft__n18, "qft_n18.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__qft__n22, "qft_n22.qasm")->Unit(benchmark::kMillisecond);

// Quantum-volume-style random u3/cx circuit, depth = n.
BENCHMARK_CAPTURE(BM_CmpSV, sv__qv__n10, "qv_n10.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__qv__n14, "qv_n14.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__qv__n18, "qv_n18.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__qv__n22, "qv_n22.qasm")->Unit(benchmark::kMillisecond);

// Grover with ccx-lowered oracle/diffusion; s search qubits, 2s-3 total.
BENCHMARK_CAPTURE(BM_CmpSV, sv__grover__s8,  "grover_n8.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__grover__s10, "grover_n10.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpSV, sv__grover__s12, "grover_n12.qasm")->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
