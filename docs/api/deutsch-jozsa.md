# Deutsch-Jozsa API Deep Dive

This page documents the public `qpp::algorithms::DeutschJozsa` API in detail.

## Header and Namespace

- Header: `include/qpp/algorithms.hpp`
- Namespace: `qpp::algorithms`

## Class Overview

`DeutschJozsa` builds the standard Deutsch-Jozsa circuit around a provided oracle
and classifies the oracle as constant or balanced based on sampled outcomes.

## `Result`

Fields:

- `type`: `CONSTANT` or `BALANCED`

## `build_circuit`

Signature:

```cpp
static QuantumCircuit build_circuit(const QuantumCircuit& oracle, int n);
```

Behavior (verified against `src/algorithms/deutsch_jozsa.cpp`):

- Allocates `n + 1` qubits and `n` classical bits
- Prepares ancilla as |1> then applies H to all qubits
- Appends the oracle instructions
- Applies H to the query register only
- Measures the query register into classical bits 0..n-1

## `solve`

Signature:

```cpp
static Result solve(
    const QuantumCircuit& oracle,
    int n,
    int shots = 1,
    uint64_t seed = 0
);
```

Behavior:

- Builds the circuit via `build_circuit`
- Runs `StatevectorSimulator::run` with the provided `shots` and `seed`
- Scans the sampled counts for a query-register outcome of all zeros
- Returns `CONSTANT` if any sampled bitstring has all-zero query bits, else `BALANCED`

Bitstring handling detail:

- The implementation checks `bits.substr(1)` against an all-zero query string,
  so it expects the query bits to follow the ancilla bit at position 0

## Example

```cpp
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

int main() {
    QuantumCircuit oracle(3); // 2 query + 1 ancilla
    auto result = DeutschJozsa::solve(oracle, 2, 10, 42);
    return (result.type == DeutschJozsa::Result::CONSTANT) ? 0 : 1;
}
```

## Notes and Preconditions

- The oracle must act on `n + 1` qubits with the ancilla as the last qubit
- The solver uses the sampled counts; `shots` should be > 1 if you want repeated sampling

## Related Pages

- [docs/algorithms/deutsch-jozsa.md](../algorithms/deutsch-jozsa.md)
- [docs/APIOverview.md](../APIOverview.md)
