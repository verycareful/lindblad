// What the per-call unitarity check actually costs.
//
// R.1.21.0 put a unitarity check inside gates::apply_unitary. The backends
// disagree about whether to pay it on every call: the statevector simulator
// forwards the instruction's policy and so pays it per application, while the
// MPS backend passes Validation::Ignore at one and two qubits and pays it only
// on the three-or-more-qubit fallback path.
//
// Deciding which is right from operation counts alone is unreliable, because
// the two costs scale in different variables: the check is O(rows^3) in the
// GATE width k alone, and applying the gate is O(2^n * 2^k) in the REGISTER
// size n as well. Whether the check is negligible therefore depends entirely on
// where a given call sits in (k, n), and the crossover is not obvious by
// inspection. This measures both halves instead of arguing about them.
//
// Three benchmarks, so the check can be read two independent ways: as the
// difference between a checked and an unchecked application, and directly on
// its own.
//
//   BM_ApplyUnitary_Checked     apply with the default policy (Throw, 1e-12)
//   BM_ApplyUnitary_Unchecked   apply with Validation::Ignore
//   BM_UnitarityCheckAlone      the check with no application at all
//
// Argument pairs are (k, n). The NARROW set holds k small and sweeps n, which
// is where ordinary circuits live. The WIDE set holds k == n, where the check
// is asymptotically the dominant term.

#include <benchmark/benchmark.h>

#include "lindblad/circuit.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/validation.hpp"

#include <vector>

using namespace lindblad;

namespace {

// A genuinely unitary k-qubit operand, built from library gates so it holds
// unitarity to about 1e-15 and passes the default 1e-12 check. The check's cost
// is data-independent, but a matrix assembled by hand could carry denormals,
// which are not.
std::vector<Complex128> dense_unitary(int k) {
    QuantumCircuit qc(k);
    for (int q = 0; q < k; ++q)
        qc.u(0.3 + 0.1 * q, 0.5 - 0.07 * q, 0.9 + 0.03 * q, q);
    for (int q = 0; q + 1 < k; ++q) qc.cx(q, q + 1);
    for (int q = 0; q < k; ++q)
        qc.u(1.1 - 0.05 * q, 0.2 + 0.11 * q, -0.4 + 0.02 * q, q);
    return Operator::from_circuit(qc).data;
}

std::vector<int> first_k(int k) {
    std::vector<int> t(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) t[static_cast<size_t>(i)] = i;
    return t;
}

// k small, n swept: where ordinary circuits live.
void NarrowArgs(benchmark::internal::Benchmark* b) {
    for (int k : {1, 2, 3, 4})
        for (int n : {12, 16, 20}) b->Args({k, n});
}

// k == n, the full-width operand. Capped at 8 because the check is O(8^k):
// at k = 10 a single iteration already runs into fractions of a second.
void WideArgs(benchmark::internal::Benchmark* b) {
    for (int k : {4, 6, 8}) b->Args({k, k});
}

}  // namespace

static void BM_ApplyUnitary_Checked(benchmark::State& state) {
    const int k = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    Statevector sv(n);
    const auto U = dense_unitary(k);
    const auto targets = first_k(k);
    for (auto _ : state) {
        gates::apply_unitary(sv, targets, U);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_ApplyUnitary_Checked)->Apply(NarrowArgs);
BENCHMARK(BM_ApplyUnitary_Checked)->Apply(WideArgs);

static void BM_ApplyUnitary_Unchecked(benchmark::State& state) {
    const int k = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    Statevector sv(n);
    const auto U = dense_unitary(k);
    const auto targets = first_k(k);
    const ValidationOptions unchecked{Validation::Ignore};
    for (auto _ : state) {
        gates::apply_unitary(sv, targets, U, unchecked);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_ApplyUnitary_Unchecked)->Apply(NarrowArgs);
BENCHMARK(BM_ApplyUnitary_Unchecked)->Apply(WideArgs);

// The check with no application under it. Independent of n by construction, so
// this is run once per k; the n argument is carried only to keep the row labels
// lined up with the two benchmarks above.
static void BM_UnitarityCheckAlone(benchmark::State& state) {
    const int k = static_cast<int>(state.range(0));
    const auto U = dense_unitary(k);
    const size_t rows = size_t(1) << k;
    const ValidationOptions strict{};
    for (auto _ : state) {
        detail::check_unitary(U, rows, strict, "bench");
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_UnitarityCheckAlone)->Apply(NarrowArgs);
BENCHMARK(BM_UnitarityCheckAlone)->Apply(WideArgs);

BENCHMARK_MAIN();
