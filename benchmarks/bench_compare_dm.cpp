// =============================================================================
// bench_compare_dm.cpp -- density-matrix backend vs Qiskit Aer 'density_matrix'
// =============================================================================
// Noise cannot be expressed in QASM2, so the model is twin-built: the channels
// and attachment points below are replicated gate-for-gate in aer_bench.py
// (make_noise_model()). Change one side only and the comparison is invalid.
//   - 2-qubit depolarizing, p = 0.01, after every cx
//   - amplitude damping,   gamma = 0.005, after every h

#include <benchmark/benchmark.h>

#include "lindblad/noise.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"

#include "compare_common.hpp"

using namespace lindblad;
using namespace lindblad_bench;

static NoiseModel make_dm_noise() {
    NoiseModel nm;
    nm.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.01, 2), "cx");
    nm.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.005), "h");
    return nm;
}

static void BM_CmpDM(benchmark::State& state, const char* file) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/true);
    const NoiseModel noise = make_dm_noise();
    DensityMatrixSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, noise, kShots, kSeed);
        benchmark::DoNotOptimize(r.counts);
    }
}

// 2-layer scaling circuit under the twin noise model (4^n storage: n <= 10).
BENCHMARK_CAPTURE(BM_CmpDM, dm__scaling__n4,  "dmscaling_n4.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpDM, dm__scaling__n6,  "dmscaling_n6.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpDM, dm__scaling__n8,  "dmscaling_n8.qasm")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpDM, dm__scaling__n10, "dmscaling_n10.qasm")->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
