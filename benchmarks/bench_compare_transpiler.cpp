// =============================================================================
// bench_compare_transpiler.cpp -- Lindblad transpile() vs qiskit.transpile()
// =============================================================================
// Layout + routing + optimization on the SAME input circuits and the SAME
// coupling graphs: both sides consume the .edges files from the corpus, so
// neither engine gets a friendlier topology.
//
// Two deliberate constraints (see local/plans/TODO.md "Transpiler" for the
// tracked defects behind them):
//   - circuits are sized EXACTLY to their coupling map (n_circuit ==
//     n_physical): routing smaller circuits trips a SABRE frozen-slot defect
//     (swap candidates never enter unoccupied physical slots).
//   - NO basis translation on either side (Qiskit runs basis_gates=None):
//     transpile() currently ignores basis_gates (BasisTranslator is absent
//     from the preset pipeline), so a routing-only comparison is the only
//     apples-to-apples one. twoq/depth quality metrics stay comparable
//     because both engines keep the input gate vocabulary and add SWAPs.
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
    const std::vector<std::string> basis = {};  // routing-only; see header

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

// QV: dense random 2q structure, the routing stress case (n == map size).
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv27__linear27__o2,   "qv_n27.qasm", "coupling_linear27.edges",   2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv27__linear27__o3,   "qv_n27.qasm", "coupling_linear27.edges",   3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv25__grid25__o2,     "qv_n25.qasm", "coupling_grid25.edges",     2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv25__grid25__o3,     "qv_n25.qasm", "coupling_grid25.edges",     3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv27__heavyhex27__o2, "qv_n27.qasm", "coupling_heavyhex27.edges", 2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qv27__heavyhex27__o3, "qv_n27.qasm", "coupling_heavyhex27.edges", 3)->Unit(benchmark::kMillisecond);

// QFT: structured long-range cp ladder, the classic routing benchmark.
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft27__linear27__o2,   "qft_n27.qasm", "coupling_linear27.edges",   2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft27__linear27__o3,   "qft_n27.qasm", "coupling_linear27.edges",   3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft25__grid25__o2,     "qft_n25.qasm", "coupling_grid25.edges",     2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft25__grid25__o3,     "qft_n25.qasm", "coupling_grid25.edges",     3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft27__heavyhex27__o2, "qft_n27.qasm", "coupling_heavyhex27.edges", 2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpTrans, trans__qft27__heavyhex27__o3, "qft_n27.qasm", "coupling_heavyhex27.edges", 3)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
