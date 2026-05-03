# Simon API Deep Dive

This page documents the public `qpp::algorithms::Simon` API in detail.

## Header and Namespace

- Header: `include/qpp/algorithms.hpp`
- Namespace: `qpp::algorithms`

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
- Extracts the query-register bits by taking `raw.substr(n)` and reversing
- Skips the all-zero equation and ignores duplicates
- Stops after `n - 1 + extra_samples` equations or `4n` attempts
- Solves the system with `gaussian_eliminate`

Bitstring handling detail:

- The sampled bitstring has length `2n`; the query bits are expected in the
  second half (`raw.substr(n)`), then reversed to index order

## `gaussian_eliminate`

Private helper used by `solve`:

- Performs row-reduction over GF(2)
- Uses a free-variable fallback (sets free vars to 1)
- Returns the reconstructed period string

## Example

```cpp
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

int main() {
    QuantumCircuit oracle(6); // 3 query + 3 output
    auto result = Simon::solve(oracle, 3, 42);
    return result.period.empty() ? 1 : 0;
}
```

## Notes and Preconditions

- The oracle must act on `2n` qubits with the query register in the first half
- Recovery quality depends on the number of independent equations

## Related Pages

- [docs/algorithms/simon.md](../algorithms/simon.md)
- [docs/APIOverview.md](../APIOverview.md)
