#include <benchmark/benchmark.h>
#include "lindblad/statevector.hpp"
#include "lindblad/gates.hpp"

using namespace lindblad;

static void BM_StatevectorInit(benchmark::State& state) {
    int n = state.range(0);
    for (auto _ : state) {
        Statevector sv(n);
        benchmark::DoNotOptimize(sv.real_parts);
    }
}
BENCHMARK(BM_StatevectorInit)->RangeMultiplier(2)->Range(8, 24);

static void BM_StatevectorNorm(benchmark::State& state) {
    int n = state.range(0);
    Statevector sv(n);
    gates::apply_h(sv, 0);
    for (auto _ : state) {
        double norm = sv.norm();
        benchmark::DoNotOptimize(norm);
    }
}
BENCHMARK(BM_StatevectorNorm)->RangeMultiplier(2)->Range(8, 24);

static void BM_StatevectorSample(benchmark::State& state) {
    int n = state.range(0);
    Statevector sv(n);
    for (int q = 0; q < n; ++q) gates::apply_h(sv, q);
    for (auto _ : state) {
        auto counts = sv.sample_counts(1000, 42);
        benchmark::DoNotOptimize(counts);
    }
}
BENCHMARK(BM_StatevectorSample)->RangeMultiplier(2)->Range(4, 16);

static void BM_RandomCircuit(benchmark::State& state) {
    int n = state.range(0);
    for (auto _ : state) {
        Statevector sv(n);
        for (int q = 0; q < n; ++q) gates::apply_h(sv, q);
        for (int q = 0; q + 1 < n; ++q) gates::apply_cx(sv, q, q + 1);
        for (int q = 0; q < n; ++q) gates::apply_rz(sv, q, 1.0);
        benchmark::DoNotOptimize(sv.real_parts);
    }
}
BENCHMARK(BM_RandomCircuit)->RangeMultiplier(2)->Range(8, 24);

BENCHMARK_MAIN();
