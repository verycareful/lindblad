// =============================================================================
// bench_clifford.cpp - tableau backend internals, gate pass against sampling
// =============================================================================
// The cross-engine suite times one number per circuit: gates followed by
// sampling, with no way to tell which of the two it is reporting. The two costs
// have different orders in the qubit count, so a single number cannot be
// attributed to either. Gate work is one pass per instruction over 2N tableau
// rows, so it grows as n^2 on the layered corpus, whose instruction count is
// linear in n. Sampling runs `shots` measurement passes, each of which walks up
// to 2N rows per measured qubit and multiplies rows costing O(N), so it grows
// as n^3.
//
// The two families here are the same circuits measured with sampling switched
// off and on. Their difference is the sampling cost, and their ratio says which
// term dominates at each size. Both are in one binary so the subtraction is
// taken within a single run, at one machine state, rather than across two.
//
// GATES: the corpus circuit with no terminal measurement, one shot. Loading and
// validation are inside the timed region because they are per-call costs the
// caller pays; the gate pass dominates them by orders of magnitude at every
// size here. One shot still copies the tableau once, which is the constant the
// gate figure carries.
//
// SAMPLE: the same circuit with measure_all() appended at kShots, matching the
// cross-engine protocol exactly, so this family is directly comparable to the
// published clifford__ladder__* rows.
//
// This is an internal micro-benchmark and deliberately not named
// bench_compare_*: there is no Qiskit half to pair it against, and the report
// generator merges the comparison suite by name.

#include <benchmark/benchmark.h>

#include "lindblad/simulators/clifford_sim.hpp"

#include "compare_common.hpp"

using namespace lindblad;
using namespace lindblad_bench;

// =============================================================================
// Gate pass only
// =============================================================================

static void BM_CliffordGates(benchmark::State& state, const char* file) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/false);
    CliffordSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, 1, kSeed);
        benchmark::DoNotOptimize(r.final_state);
    }
}

BENCHMARK_CAPTURE(BM_CliffordGates, clifford__gates__n20,  "clifford_n20.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CliffordGates, clifford__gates__n40,  "clifford_n40.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CliffordGates, clifford__gates__n80,  "clifford_n80.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CliffordGates, clifford__gates__n160, "clifford_n160.qasm")->Unit(benchmark::kMillisecond);

// =============================================================================
// Gate pass plus sampling
// =============================================================================

static void BM_CliffordSample(benchmark::State& state, const char* file) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/true);
    CliffordSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, kShots, kSeed);
        benchmark::DoNotOptimize(r.counts);
    }
}

BENCHMARK_CAPTURE(BM_CliffordSample, clifford__sample__n20,  "clifford_n20.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CliffordSample, clifford__sample__n40,  "clifford_n40.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CliffordSample, clifford__sample__n80,  "clifford_n80.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CliffordSample, clifford__sample__n160, "clifford_n160.qasm")->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
