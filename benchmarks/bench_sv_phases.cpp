// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// =============================================================================
// bench_sv_phases.cpp - where a statevector run's time goes, by phase
// =============================================================================
// The cross-engine suite reports one number per circuit. A run is five
// phases (PhaseTimes in types.hpp: setup, fusion, evolution, sampling,
// final_copy), and a change to the sampler or the final copy moves only its
// own phase, which a whole-run number can bury. A cycles profile cannot stand
// in: it counts every thread, so a serial phase shows up to the thread count
// smaller than its share of wall time.
//
// PHASES/<row>: the comparison corpus rows of bench_compare_sv (same files,
// shots, seed and measurement), run with Options::record_phase_times. Each
// phase is a counter averaged per run, in milliseconds. The run is also timed
// from outside; "unaccounted_ms" is that outside time minus the phase total,
// which is the call's own entry and return and should stay near zero. A phase
// total above the outside time cannot happen if the phases tile the run, so it
// fails the case instead of reporting it.
//
// NOSAMPLE/<row>: the same circuit with no measurement and no shots, which
// takes the single-trajectory path and never samples. PHASES minus NOSAMPLE is
// an estimate of the sampling cost made without the phase clock at all, the
// way bench_clifford separates its two halves. It is coarse: the two runs take
// different execution paths (the single forward pass against the trajectory
// runner), and at large n the run-to-run variation of the evolution exceeds
// the whole sampling cost, so the difference can come out negative. It is a
// sanity check on the order of magnitude of the sampling counter, not a
// second measurement of it.
//
// This is an internal micro-benchmark and deliberately not named
// bench_compare_*: there is no Qiskit half to pair it against, and the report
// generator merges the comparison suite by name.

#include <benchmark/benchmark.h>

#include "lindblad/simulators/statevector_sim.hpp"

#include "compare_common.hpp"

#include <chrono>
#include <string>

using namespace lindblad;
using namespace lindblad_bench;

namespace {

struct Row {
    const char* key;  // the bench_compare_sv / aer_bench.py pairing key
    const char* file;
};

constexpr Row kRows[] = {
    {"sv__scaling__n10", "scaling_n10.qasm"}, {"sv__scaling__n14", "scaling_n14.qasm"},
    {"sv__scaling__n18", "scaling_n18.qasm"}, {"sv__scaling__n22", "scaling_n22.qasm"},
    {"sv__scaling__n26", "scaling_n26.qasm"}, {"sv__qft__n10", "qft_n10.qasm"},
    {"sv__qft__n14", "qft_n14.qasm"},         {"sv__qft__n18", "qft_n18.qasm"},
    {"sv__qft__n22", "qft_n22.qasm"},         {"sv__qv__n10", "qv_n10.qasm"},
    {"sv__qv__n14", "qv_n14.qasm"},           {"sv__qv__n18", "qv_n18.qasm"},
    {"sv__qv__n22", "qv_n22.qasm"},           {"sv__grover__s8", "grover_n8.qasm"},
    {"sv__grover__s10", "grover_n10.qasm"},   {"sv__grover__s12", "grover_n12.qasm"},
};

constexpr double kMsPerSecond = 1e3;

benchmark::Counter per_run_ms(double total_seconds) {
    return benchmark::Counter(total_seconds * kMsPerSecond, benchmark::Counter::kAvgIterations);
}

void BM_Phases(benchmark::State& state, const char* file) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/true);
    StatevectorSimulator::Options opts;
    opts.record_phase_times = true;
    StatevectorSimulator sim(opts);

    PhaseTimes sum;
    double outside = 0.0;
    for (auto _ : state) {
        const auto t0 = std::chrono::steady_clock::now();
        auto r = sim.run(qc, kShots, kSeed);
        const auto t1 = std::chrono::steady_clock::now();
        benchmark::DoNotOptimize(r.counts);

        if (!r.phase_times) {
            state.SkipWithError("record_phase_times was set and phase_times is empty");
            return;
        }
        const PhaseTimes& p = *r.phase_times;
        const double run_seconds = std::chrono::duration<double>(t1 - t0).count();
        if (p.total() > run_seconds) {
            state.SkipWithError("the phases add up to more than the run itself");
            return;
        }
        sum.setup += p.setup;
        sum.fusion += p.fusion;
        sum.evolution += p.evolution;
        sum.sampling += p.sampling;
        sum.final_copy += p.final_copy;
        outside += run_seconds;
    }
    state.counters["setup_ms"] = per_run_ms(sum.setup);
    state.counters["fusion_ms"] = per_run_ms(sum.fusion);
    state.counters["evolution_ms"] = per_run_ms(sum.evolution);
    state.counters["sampling_ms"] = per_run_ms(sum.sampling);
    state.counters["final_copy_ms"] = per_run_ms(sum.final_copy);
    state.counters["unaccounted_ms"] = per_run_ms(outside - sum.total());
}

void BM_NoSample(benchmark::State& state, const char* file) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/false);
    StatevectorSimulator sim;
    for (auto _ : state) {
        auto r = sim.run(qc, /*shots=*/0, kSeed);
        benchmark::DoNotOptimize(r.final_state.real_parts);
    }
}

}  // namespace

int main(int argc, char** argv) {
    for (const Row& row : kRows) {
        const char* file = row.file;
        benchmark::RegisterBenchmark((std::string("phases/") + row.key).c_str(),
                                     [file](benchmark::State& st) { BM_Phases(st, file); })
            ->Unit(benchmark::kMillisecond)
            ->UseRealTime();
        benchmark::RegisterBenchmark((std::string("nosample/") + row.key).c_str(),
                                     [file](benchmark::State& st) { BM_NoSample(st, file); })
            ->Unit(benchmark::kMillisecond)
            ->UseRealTime();
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
