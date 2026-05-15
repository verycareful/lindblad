# Simon API Deep Dive

This page documents the public `lindblad::algorithms::Simon` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Class Overview

`Simon` builds the standard Simon query circuit and recovers a hidden period by
sampling equations and solving a GF(2) system.

## `Result`

Fields:

- `period`: recovered hidden period string
- `equations`: sampled non-zero equations used for elimination

## `build_circuit`

Signature:

```cpp
static QuantumCircuit build_circuit(const QuantumCircuit& oracle, int n);
```

Behavior (verified against `src/algorithms/simon.cpp`):

- Allocates `2n` qubits and `n` classical bits
- Applies H to the query register
- Appends the oracle instructions
- Applies H to the query register again
- Measures the query register into classical bits 0..n-1

## `solve`

Signature:

```cpp
static Result solve(
    const QuantumCircuit& oracle,
    int n,
    uint64_t seed = 0,
    int extra_samples = 2
);
```

Behavior:

- Builds the circuit via `build_circuit`
- Repeatedly runs the circuit for single-shot samples
- Reverses the sampled bitstring directly to obtain index-order query bits
- Skips the all-zero equation and ignores duplicates
- Stops after `n - 1 + extra_samples` equations or `4n` attempts
- Solves the system with `gaussian_eliminate`

Bitstring handling detail:

- The classical register contains only the `n` query bits (qubits 0..n-1);
  the output register is not measured. The sampled bitstring has length `n`
  and is reversed directly to index order — no `substr` is needed.

## `gaussian_eliminate`

Private helper used by `solve`:

- Performs row-reduction over GF(2)
- Uses a free-variable fallback (sets free vars to 1)
- Returns the reconstructed period string

## Example

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit oracle(6); // 3 query + 3 output
    auto result = Simon::solve(oracle, 3, 42);
    return result.period.empty() ? 1 : 0;
}
```

## Notes and Preconditions

- The oracle must act on `2n` qubits with the query register in the first half
- Recovery quality depends on the number of independent equations

## QuditSimon

Declared in `include/lindblad/algorithms.hpp`. Implemented in `src/algorithms/simon.cpp`.

### Result struct

```cpp
struct Result {
    std::vector<int> period;   // s in Z_d^n, each element in {0..d-1}
    bool is_trivial;           // true iff s = 0...0 (f is injective)
    int d;
    int n;
    int quantum_queries;       // number of quantum circuit executions performed
};
```

### solve

```cpp
static Result solve(
    int n, int d,
    const std::function<std::vector<int>(const std::vector<int>&)>& f,
    int extra_samples = 3,
    uint64_t seed = 0
);
```

Runs O(n+extra_samples) quantum circuit queries, then performs Gaussian elimination over GF(d) to recover s.

- `extra_samples`: additional queries beyond the minimum (n-1). Increase if incorrect results are obtained.
- `d` must be **prime**. Throws `std::invalid_argument` if d is composite.

**Throws** `std::invalid_argument` if `d < 2`, `d` is not prime, `n < 1`, or `f` returns a vector of wrong size or with out-of-range digits.

## Related Pages

- [docs/algorithms/simon.md](../algorithms/simon.md)
- [docs/APIOverview.md](../APIOverview.md)
