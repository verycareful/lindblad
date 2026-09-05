// =============================================================================
// bench_compare_mps.cpp -- MPS backend vs Qiskit Aer 'matrix_product_state'
// =============================================================================
// Three sweeps, across two circuit families chosen for opposite entanglement
// behaviour:
//   - qubit sweep at fixed bond dimension chi = 32 (n = 16..40), scaling family
//   - bond sweep at fixed n = 24 (chi = 8..64), brickwork family
//   - the same bond sweep on the scaling family, as a non-binding control
//
// The two families answer different questions, and only one of them can carry
// a bond sweep.
//
// The scaling family is a layered CX ladder: one CX crosses each cut per layer,
// three layers, so the Schmidt rank it produces peaks at 2. Every cap in the
// sweep sits above that, so none of them truncates and the four points are one
// measurement under four labels. That is worth publishing as a control, because
// a workload staying far under its cap is the regime MPS exists for and is the
// context the binding rows are read against. It is not worth publishing as bond
// scaling.
//
// The brickwork family applies random SU(4) blocks to alternating bond
// parities, so a cut is crossed once per two layers and the rank multiplies by
// up to 4 each time. At its generated depth the rank passes 256, so all four
// caps discard weight and the four points are four distinct truncation
// regimes. The depth is measured rather than derived; see BRICKWORK_LAYERS in
// benchmarks/compare/gen_circuits.py.
//
// aer_bench.py mirrors all three, setting matrix_product_state_max_bond_dimension
// to the same chi. Lindblad splits bonds with BDC, its default; Jacobi is
// available and is what the dense-rebuild fallback uses for accuracy. Aer uses
// its own SVD. Documented in docs/Benchmarks.md.
//
// Each row also reports `svd_ms`, the time that run spent in the bond-split
// ladder, which the report turns into a share of the row's own median. The
// share is an upper bound on what a faster SVD kernel could remove rather than
// an estimate of it: the interval includes the verification each split is held
// to, and verification survives a change of kernel. See MPSState::svd_time_ns.

#include <benchmark/benchmark.h>

#include "lindblad/simulators/mps_sim.hpp"

#include "compare_common.hpp"

#include <chrono>

using namespace lindblad;
using namespace lindblad_bench;

static void BM_CmpMPS(benchmark::State& state, const char* file, int bond_dim) {
    const auto qc = load_corpus_circuit(file, /*add_measure=*/true);
    MPSSimulator sim;

    // Held across the loop so the counter below comes from a run the timer
    // actually measured, rather than from an extra one taken under a different
    // cache state. Every run() starts a fresh chain, so this carries one
    // iteration's splits and not the sum over them.
    //
    // Result has no default constructor, only Result(int), so it is built on
    // the register width and then move-assigned by each iteration.
    MPSSimulator::Result last(qc.n_qubits);
    for (auto _ : state) {
        last = sim.run(qc, bond_dim, kShots, kSeed);
        benchmark::DoNotOptimize(last.counts);
    }

    // Set after the loop: Google Benchmark clears counters when the loop
    // starts, so a value assigned before it reports as zero.
    //
    // The corpus circuits carry terminal-only measurement, so a run is one
    // forward pass and these figures cover all of it. On a mid-circuit
    // measurement circuit the simulator re-simulates per shot and the counters
    // would describe the last shot alone, which is why this benchmark's
    // circuits are the gate-only corpus members.
    const std::chrono::nanoseconds svd_ns(
        static_cast<std::chrono::nanoseconds::rep>(
            last.final_state.svd_time_ns()));
    state.counters["svd_ms"] =
        std::chrono::duration<double, std::milli>(svd_ns).count();
    state.counters["svd_calls"] =
        static_cast<double>(last.final_state.svd_call_count());
}

// Qubit sweep at chi = 32.
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n16__chi32, "scaling_n16.qasm", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n24__chi32, "scaling_n24.qasm", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n32__chi32, "scaling_n32.qasm", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n40__chi32, "scaling_n40.qasm", 32)->Unit(benchmark::kMillisecond);

// Bond sweep at n = 24: every cap truncates, so the four points differ.
BENCHMARK_CAPTURE(BM_CmpMPS, mps__brickwork__n24__chi8,  "brickwork_n24.qasm",  8)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__brickwork__n24__chi16, "brickwork_n24.qasm", 16)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__brickwork__n24__chi32, "brickwork_n24.qasm", 32)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__brickwork__n24__chi64, "brickwork_n24.qasm", 64)->Unit(benchmark::kMillisecond);

// Non-binding control at n = 24 (chi = 32 point provided by the qubit sweep).
// The rank here peaks at 2, so these rows should agree with each other and with
// the chi = 32 point to within run-to-run noise. Divergence among them means
// the cap is reaching a workload that cannot produce enough rank to feel it,
// which is a defect in the harness rather than a bond-dimension effect.
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n24__chi8,  "scaling_n24.qasm",  8)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n24__chi16, "scaling_n24.qasm", 16)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_CmpMPS, mps__scaling__n24__chi64, "scaling_n24.qasm", 64)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
