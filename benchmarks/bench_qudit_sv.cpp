// bench_qudit_sv.cpp — qudit statevector kernels (audit F-17 / R.1.13 benchmarks)
//
// Exercises the OpenMP-parallelised apply_1qudit / apply_2qudit paths across
// d = 2..7 and n = 2..5. Args are {d, n}.

#include <benchmark/benchmark.h>

#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_gates.hpp"

using namespace lindblad;

// Apply a single-qudit QFT to every qudit (one full sweep per iteration).
static void BM_Qudit_Apply1(benchmark::State& state) {
    const int d = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    const auto F = qudit_gates::qft_matrix(d);
    QuditStatevector sv(n, d);
    for (auto _ : state) {
        for (int q = 0; q < n; ++q) sv.apply_1qudit(q, F);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Qudit_Apply1)
    ->Args({2, 5})->Args({3, 4})->Args({4, 4})->Args({5, 3})->Args({7, 3});

// Apply a controlled-add (d^2 x d^2) gate to each adjacent qudit pair.
static void BM_Qudit_Apply2(benchmark::State& state) {
    const int d = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    const auto CADD = qudit_gates::cadd_matrix(d, 1);
    QuditStatevector sv(n, d);
    for (auto _ : state) {
        for (int q = 0; q + 1 < n; ++q) sv.apply_2qudit(q, q + 1, CADD);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Qudit_Apply2)
    ->Args({2, 5})->Args({3, 4})->Args({4, 3})->Args({5, 3})->Args({7, 2});

// A short mixed circuit: QFT sweep + CADD ladder (representative of the qudit
// algorithm inner loops).
static void BM_Qudit_Circuit(benchmark::State& state) {
    const int d = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    const auto F = qudit_gates::qft_matrix(d);
    const auto CADD = qudit_gates::cadd_matrix(d, 1);
    for (auto _ : state) {
        QuditStatevector sv(n, d);
        for (int q = 0; q < n; ++q) sv.apply_1qudit(q, F);
        for (int q = 0; q + 1 < n; ++q) sv.apply_2qudit(q, q + 1, CADD);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Qudit_Circuit)
    ->Args({2, 5})->Args({3, 4})->Args({4, 3})->Args({5, 3})->Args({7, 2});

BENCHMARK_MAIN();
