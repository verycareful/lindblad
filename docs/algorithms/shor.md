# Shor's Algorithm

This page documents `lindblad::algorithms::Shor`.

## Purpose

Shor's algorithm factors a composite integer N into two non-trivial factors using quantum order finding. It is the canonical demonstration of exponential quantum speedup over classical algorithms for a practical problem (integer factorisation).

In lindblad, `Shor` assembles the existing `QFT`/`IQFT` primitives and a `PERMUTATION` modular-multiplication oracle into a complete factoring pipeline with classical preprocessing and continued-fraction period recovery.

## Theory Summary

Shor's algorithm factors N = p × q through these steps:

1. **Classical pre-screening**: reject N < 4 or prime N; handle even N, perfect powers, and small trial GCDs without any quantum computation.
2. **Random base selection**: pick a random a in [2, N−2], coprime to N.
3. **Quantum period finding**: build a QPE-based circuit that implements the modular exponentiation oracle |x⟩|1⟩ → |x⟩|aˣ mod N⟩, apply the inverse QFT on the evaluation register, and measure.
4. **Continued-fraction recovery**: interpret the measured bitstring as a phase m/2^n_eval ≈ s/r. Extract candidate periods r via continued-fraction convergents.
5. **Classical post-processing**: if r is even and a^(r/2) ≢ −1 (mod N), compute p = gcd(a^(r/2) − 1, N) and q = gcd(a^(r/2) + 1, N).
6. **Retry**: if the period is odd or the GCD yields a trivial factor, repeat with a new random base (up to max_attempts).

The quantum circuit uses O(n²) controlled-unitary steps where n = ⌈log₂N⌉. Each controlled-U^(2^k) encodes the map |x⟩ → |a^(2^k)·x mod N⟩ on the target register, with identity on the control-0 subspace. As of R.1.13 (audit F-9) this is emitted as a native `PERMUTATION` instruction over (control, target) — a `2^(n_target+1)`-entry index map — rather than a dense `(2·2^n × 2·2^n)` matrix. Memory drops from `O(4^n_target)` to `O(2^n_target)` per step and the application is `O(dim)` instead of `O(dim·2^n_target)`.

## Required Inputs

- `N` — the composite integer to factor (must be ≥ 4, not prime)

Optional configuration via `Shor::Options`:
- `n_eval_qubits` — number of evaluation (phase) qubits. Default 0 = auto: 2·⌈log₂N⌉ + 1
- `max_attempts` — maximum random bases to try before giving up. Default 10
- `seed` — RNG seed for reproducibility. Default 0 = random
- `simulator` — backend simulator type. Default `STATEVECTOR`

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Factor a number:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

Shor shor;
auto result = shor.factorize(15);
// result.factor = 3 (or 5)
// result.cofactor = 5 (or 3)
// result.success = true
// result.method = "trivial_gcd" or "quantum"
```

With explicit options:

```cpp
Shor::Options opts;
opts.max_attempts = 20;
opts.seed = 42;
opts.simulator = backends::LocalBackend::SimType::STATEVECTOR;

Shor shor(opts);
auto result = shor.factorize(21);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator Dependencies

Shor uses `backends::LocalBackend` internally. The backend selection is configured via `Options::simulator`.

The modular exponentiation oracle is implemented as `Instruction::GateType::PERMUTATION` gates (R.1.13, audit F-9) — a basis-index map over (control, target), applied as an amplitude gather rather than a dense matrix multiply. This means:

| Backend | Supported | Notes |
|---|---|---|
| `STATEVECTOR` | ✓ | Default. Exact statevector simulation; PERMUTATION applied natively as an O(dim) gather. |
| `DENSITY_MATRIX` | ✓ | Full mixed-state; useful for studying noise effects on factoring. PERMUTATION applied as a native row/column relabel. |
| `MPS` | ✓ | PERMUTATION uses the bounded statevector fallback (`to_statevector` → apply → `rebuild_from_statevector`, capped at `MPS_SV_MAX_QUBITS`), as the dense UNITARY did before R.1.13. |
| `CLIFFORD` | ✗ | **Not supported.** The modular exponentiation map is an arbitrary permutation — it cannot be decomposed into the Clifford gate set {H, S, CX, X, Y, Z}. This is a fundamental mathematical constraint, not an implementation limitation. |

The `QFT::build_inverse_circuit` is used for the IQFT stage of the evaluation register.

## Circuit Architecture

The period-finding circuit has two registers:

- **Evaluation register**: qubits 0 to n_eval−1 — initialised with Hadamards, receives the IQFT, and is measured to extract the phase.
- **Target register**: qubits n_eval to n_eval+n_target−1 — initialised to |1⟩ via X on the first target qubit.

For each evaluation qubit k (k = 0, 1, ..., n_eval−1):
- Build the permutation index map over the (control, target) sub-state of size 2^(n_target+1), using the LSB convention (control qubit k maps to bit 0): control-0 sub-states map to themselves; control-1 sub-states map |x⟩ → |a^(2^k)·x mod N⟩
- Append as a single `PERMUTATION` instruction on qubits {k, n_eval, n_eval+1, ..., n_eval+n_target−1} (a 2^(n_target+1)-entry map, not a dense matrix)

Total qubit count: n_eval + n_target.

## Example Code

```cpp
#include "lindblad/algorithms.hpp"
#include <iostream>

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    // Factor 15 = 3 × 5
    Shor::Options opts;
    opts.seed = 42;
    Shor shor(opts);

    auto result = shor.factorize(15);
    if (result.success) {
        std::cout << "15 = " << result.factor << " × " << result.cofactor << "\n";
        std::cout << "Method: " << result.method << "\n";
        std::cout << "Attempts: " << result.attempts << "\n";
    }

    // Inspect the period-finding circuit directly
    int n_target = 4;  // ⌈log₂(15+1)⌉
    int n_eval = 9;    // 2*4 + 1
    auto circuit = Shor::build_period_finding_circuit(2, 15, n_eval, n_target);
    std::cout << "Circuit qubits: " << circuit.n_qubits << "\n";
    std::cout << "Circuit gates: " << circuit.instructions.size() << "\n";
}
```

## Return Values and Outputs

`Shor::factorize` returns a `Shor::Result`:

- `factor` — a non-trivial factor of N (0 on failure)
- `cofactor` — N / factor (0 on failure)
- `success` — true if a non-trivial factorisation was found
- `attempts` — number of random bases tried (0 for classical pre-screening results)
- `method` — one of `"trivial_gcd"`, `"perfect_power"`, or `"quantum"`

`Shor::build_period_finding_circuit` returns a `QuantumCircuit` with n_eval + n_target qubits and no measurements.

`Shor::find_order` returns the multiplicative order r = ord_N(a), or 0 on failure.

## Exceptions and Failure Modes

- `std::invalid_argument` if N < 4
- `std::invalid_argument` if N is prime (determined via deterministic Miller-Rabin with witnesses {2, 3, 5, 7, 11, 13})
- `Result::success = false` if all max_attempts are exhausted without finding a non-trivial factor

Failure modes that lead to retries (not exceptions):
- The measured phase yields r = 0 (zero measurement outcome)
- The recovered period r is odd
- a^(r/2) ≡ −1 (mod N) — trivial square root

## Common Pitfalls

- **Memory**: since R.1.13 each controlled-U step stores a `PERMUTATION` index map of 2^(n_target+1) integers (not a 2^(n_target+1) × 2^(n_target+1) dense matrix). For N = 15 (n_target = 4) that is 32 entries; for N = 100 (n_target = 7), 256 entries. The dominant cost is the statevector itself (2^(n_eval + n_target) amplitudes).
- **Practical range**: N ≤ ~100 is feasible on a desktop. Larger N requires exponentially more statevector memory and simulation time.
- **Seed sensitivity**: the quantum measurement outcome is probabilistic. Different seeds may require different numbers of attempts.
- **Classical shortcuts**: for small N, the algorithm often returns via `"trivial_gcd"` (even numbers, numbers with small prime factors) before reaching the quantum path. This is by design — classical pre-screening avoids expensive QPE when possible.
- **CLIFFORD backend**: will fail at runtime because the controlled modular-exponentiation gates are non-Clifford.

## Testing Notes

A dedicated test suite (`tests/test_shor.cpp`) ships with R.1.8.1 (30 tests) and is extended in R.1.8.2 (8 additional tests covering `cf_convergents` directly, unitarity invariants, and stricter order-finding assertions).

## Related Source Files

- [docs/api/shor.md](../api/shor.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/shor.cpp](../../src/algorithms/shor.cpp)
- [docs/algorithms/qpe.md](qpe.md)
- [docs/algorithms/qft.md](qft.md)
