// =============================================================================
// bench_compare_transpiler.cpp -- Lindblad transpile() vs qiskit.transpile()
// =============================================================================
// Full pipeline comparison on the SAME input circuits and the SAME coupling
// graphs: layout + routing + optimisation + basis translation. Both sides
// consume the .edges files from the corpus, so neither engine gets a
// friendlier topology, and both translate into the same {cx, u3} basis, so
// twoq/depth quality counters compare like for like (CX against CX).
//
// R.1.15.0 note: this file previously carried two workarounds (circuits sized
// EXACTLY to their map, and NO basis translation) for the frozen-slot and
// silent-basis_gates defects fixed in R.1.15.0. Both are reverted: circuits
// are now SMALLER than their maps (n = 22 on 25/27-slot devices), which
// exercises the layout-expansion path end-to-end, exactly like Qiskit's
// ancilla allocation on real devices.
//
// Wall time is the benchmark; output QUALITY is reported via counters:
//   twoq_out  = 2-qubit gates in the transpiled circuit (routing overhead)
//   depth_out = transpiled circuit depth
// Counters come from one representative call outside the timed loop (the
// pipeline is deterministic for a fixed input; SABRE seeding is internal and
// fixed).

#include <benchmark/benchmark.h>

#include "compare_common.hpp"

using namespace lindblad;
using namespace lindblad_bench;

static void BM_CmpTrans(benchmark::State& state, const char* qasm_file,
                        const char* edges_file, int opt_level) {
    const auto qc = load_corpus_circuit(qasm_file, /*add_measure=*/false);
    const auto cmap = load_coupling(edges_file);
    // Real target basis on both sides (aer_bench.py passes the same list):
    // the library's equivalence set. Output is verified by BasisTranslator.
    const std::vector<std::string> basis = {"cx", "u3"};

    const auto sample = transpile(qc, cmap, basis, opt_level);

    for (auto _ : state) {
        auto out = transpile(qc, cmap, basis, opt_level);
        benchmark::DoNotOptimize(out.instructions);
    }

    // Counters must be set AFTER the timed loop: Google Benchmark clears
    // state.counters when the loop starts, so setup-time values report as 0
    // (observed on the first recorded run). The pipeline is deterministic for
    // a fixed input, so the pre-loop sample's metrics are the run's metrics.
    state.counters["twoq_out"] = two_qubit_count(sample);
    state.counters["depth_out"] = sample.depth();
}

// QV: dense random 2q structure, the routing stress case (n = 22 circuits on
// 25/27-slot maps: layout expansion + idle-wire routing in play).
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv22__linear27__o2,   "qv_n22.qasm", "coupling_linear27.edges",   2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv22__linear27__o3,   "qv_n22.qasm", "coupling_linear27.edges",   3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv22__grid25__o2,     "qv_n22.qasm", "coupling_grid25.edges",     2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv22__grid25__o3,     "qv_n22.qasm", "coupling_grid25.edges",     3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv22__heavyhex27__o2, "qv_n22.qasm", "coupling_heavyhex27.edges", 2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv22__heavyhex27__o3, "qv_n22.qasm", "coupling_heavyhex27.edges", 3)->Unit(benchmark::kMillisecond);

// QFT: structured long-range cp ladder, the classic routing benchmark.
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft22__linear27__o2,   "qft_n22.qasm", "coupling_linear27.edges",   2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft22__linear27__o3,   "qft_n22.qasm", "coupling_linear27.edges",   3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft22__grid25__o2,     "qft_n22.qasm", "coupling_grid25.edges",     2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft22__grid25__o3,     "qft_n22.qasm", "coupling_grid25.edges",     3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft22__heavyhex27__o2, "qft_n22.qasm", "coupling_heavyhex27.edges", 2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft22__heavyhex27__o3, "qft_n22.qasm", "coupling_heavyhex27.edges", 3)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
