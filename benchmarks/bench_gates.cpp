#include <benchmark/benchmark.h>
#include "qpp/statevector.hpp"
#include "qpp/gates.hpp"

using namespace qpp;

static void BM_H_Gate(benchmark::State& state) {
    int n = state.range(0);
    Statevector sv(n);
    for (auto _ : state) {
        gates::apply_h(sv, 0);
    }
}
BENCHMARK(BM_H_Gate)->RangeMultiplier(2)->Range(8, 24);

static void BM_CX_Gate(benchmark::State& state) {
    int n = state.range(0);
    Statevector sv(n);
    gates::apply_h(sv, 0);
    for (auto _ : state) {
        gates::apply_cx(sv, 0, 1);
    }
}
BENCHMARK(BM_CX_Gate)->RangeMultiplier(2)->Range(8, 24);

static void BM_RX_Gate(benchmark::State& state) {
    int n = state.range(0);
    Statevector sv(n);
    for (auto _ : state) {
        gates::apply_rx(sv, 0, 1.234);
    }
}
BENCHMARK(BM_RX_Gate)->RangeMultiplier(2)->Range(8, 24);

static void BM_CCX_Gate(benchmark::State& state) {
    int n = state.range(0);
    Statevector sv(n);
    gates::apply_x(sv, 0);
    gates::apply_x(sv, 1);
    for (auto _ : state) {
        gates::apply_ccx(sv, 0, 1, 2);
    }
}
BENCHMARK(BM_CCX_Gate)->RangeMultiplier(2)->Range(8, 24);

BENCHMARK_MAIN();
