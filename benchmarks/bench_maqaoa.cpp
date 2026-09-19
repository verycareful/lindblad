// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include <benchmark/benchmark.h>
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

static void BM_MAQAOA_CircuitBuild(benchmark::State& state) {
    int n = state.range(0);
    SparsePauliOp cost;
    for (int q = 0; q + 1 < n; ++q) {
        std::string pauli(n, 'I');
        pauli[q] = 'Z';
        pauli[q + 1] = 'Z';
        cost.terms.push_back({pauli, Complex128(1.0, 0.0)});
    }

    MAQAOA maqaoa;
    maqaoa.options.p = 1;
    int n_params = maqaoa.num_parameters(cost);
    std::vector<double> params(n_params, 0.5);

    for (auto _ : state) {
        auto circuit = maqaoa.build_circuit(cost, {}, params);
        benchmark::DoNotOptimize(circuit.instructions);
    }
}
BENCHMARK(BM_MAQAOA_CircuitBuild)->Range(4, 16);

static void BM_QAOA_CircuitBuild(benchmark::State& state) {
    int n = state.range(0);
    SparsePauliOp cost;
    for (int q = 0; q + 1 < n; ++q) {
        std::string pauli(n, 'I');
        pauli[q] = 'Z';
        pauli[q + 1] = 'Z';
        cost.terms.push_back({pauli, Complex128(1.0, 0.0)});
    }

    QAOA qaoa;
    qaoa.options.p = 2;
    std::vector<double> params(4, 0.5);

    for (auto _ : state) {
        auto circuit = qaoa.build_circuit(cost, {}, params);
        benchmark::DoNotOptimize(circuit.instructions);
    }
}
BENCHMARK(BM_QAOA_CircuitBuild)->Range(4, 16);

BENCHMARK_MAIN();
