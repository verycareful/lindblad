// bench_qudit_dm.cpp — qudit density-matrix kernels (R.1.13 benchmarks)
//
// Exercises apply_1qudit / apply_2qudit and the Kraus / noise paths on the
// qudit density matrix. Storage is d^(2n) complex, so the qudit/register
// ranges are kept small. Args are {d, n}.

#include <benchmark/benchmark.h>

#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"
#include "lindblad/qudit/qudit_gates.hpp"

using namespace lindblad;

// Single-qudit QFT sweep on the density matrix (rho -> U rho U^dagger).
static void BM_QuditDM_Apply1(benchmark::State& state) {
    const int d = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    const auto F = qudit_gates::qft_matrix(d);
    QuditDensityMatrix dm(n, d);
    for (auto _ : state) {
        for (int q = 0; q < n; ++q) dm.apply_1qudit(q, F);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_QuditDM_Apply1)
    ->Args({2, 4})->Args({3, 3})->Args({4, 2})->Args({5, 2})->Args({7, 2});

// Controlled-add on adjacent pairs.
static void BM_QuditDM_Apply2(benchmark::State& state) {
    const int d = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    const auto CADD = qudit_gates::cadd_matrix(d, 1);
    QuditDensityMatrix dm(n, d);
    for (auto _ : state) {
        for (int q = 0; q + 1 < n; ++q) dm.apply_2qudit(q, q + 1, CADD);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_QuditDM_Apply2)
    ->Args({2, 4})->Args({3, 3})->Args({4, 2})->Args({5, 2});

// Depolarising Kraus channel on every qudit (the noisy-simulation hot path).
static void BM_QuditDM_Kraus(benchmark::State& state) {
    const int d = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    const auto ch = QuditNoiseModel::depolarizing_channel(d, 0.05);
    QuditDensityMatrix dm(n, d);
    for (auto _ : state) {
        for (int q = 0; q < n; ++q) dm.apply_kraus_1qudit(q, ch.ops);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_QuditDM_Kraus)
    ->Args({2, 4})->Args({3, 3})->Args({4, 2})->Args({5, 2});

BENCHMARK_MAIN();
