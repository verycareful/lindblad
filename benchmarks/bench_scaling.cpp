// bench_scaling.cpp — standard-circuit scaling across the four backends
// (R.1.13 benchmarks). Each backend runs the same layered circuit over the
// qubit range where it is viable: statevector (n<=26), density matrix (n<=10,
// 4^n storage), MPS (n<=28, bond-dim truncated), Clifford (n<=40).

#include <benchmark/benchmark.h>

#include "lindblad/circuit.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/noise.hpp"

using namespace lindblad;

// H layer + CX ladder + RZ layer, repeated, then terminal measurement.
static QuantumCircuit standard_circuit(int n, int layers) {
    QuantumCircuit qc(n);
    for (int l = 0; l < layers; ++l) {
        for (int q = 0; q < n; ++q) qc.h(q);
        for (int q = 0; q + 1 < n; ++q) qc.cx(q, q + 1);
        for (int q = 0; q < n; ++q) qc.rz(0.3, q);
    }
    qc.measure_all();
    return qc;
}

// Clifford variant (S instead of RZ) so the stabilizer backend accepts it.
static QuantumCircuit clifford_circuit(int n, int layers) {
    QuantumCircuit qc(n);
    for (int l = 0; l < layers; ++l) {
        for (int q = 0; q < n; ++q) qc.h(q);
        for (int q = 0; q + 1 < n; ++q) qc.cx(q, q + 1);
        for (int q = 0; q < n; ++q) qc.s(q);
    }
    qc.measure_all();
    return qc;
}

static void BM_Scaling_Statevector(benchmark::State& state) {
    const auto qc = standard_circuit(static_cast<int>(state.range(0)), 3);
    StatevectorSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, 256, 42);
        benchmark::DoNotOptimize(r.counts);
    }
}
BENCHMARK(BM_Scaling_Statevector)->DenseRange(10, 26, 2)->Unit(benchmark::kMillisecond);

static void BM_Scaling_DensityMatrix(benchmark::State& state) {
    const auto qc = standard_circuit(static_cast<int>(state.range(0)), 2);
    DensityMatrixSimulator sim;
    const NoiseModel ideal;
    for (auto _ : state) {
        auto r = sim.run(qc, ideal, 256, 42);
        benchmark::DoNotOptimize(r.counts);
    }
}
BENCHMARK(BM_Scaling_DensityMatrix)->DenseRange(4, 10, 1)->Unit(benchmark::kMillisecond);

static void BM_Scaling_MPS(benchmark::State& state) {
    const auto qc = standard_circuit(static_cast<int>(state.range(0)), 3);
    MPSSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, 64, 256, 42);
        benchmark::DoNotOptimize(r.counts);
    }
}
BENCHMARK(BM_Scaling_MPS)->DenseRange(10, 28, 2)->Unit(benchmark::kMillisecond);

static void BM_Scaling_Clifford(benchmark::State& state) {
    const auto qc = clifford_circuit(static_cast<int>(state.range(0)), 3);
    CliffordSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, 256, 42);
        benchmark::DoNotOptimize(r.counts);
    }
}
BENCHMARK(BM_Scaling_Clifford)->DenseRange(10, 40, 5)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
