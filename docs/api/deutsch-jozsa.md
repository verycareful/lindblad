# Deutsch-Jozsa API Deep Dive

This page documents the public `lindblad::algorithms::DeutschJozsa` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

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

- The classical register contains only the `n` query bits (qubits 0..n-1);
  the ancilla is not measured. The sampled bitstring has length `n` and is
  compared directly against `std::string(n, '0')` — no `substr` stripping is needed.

## Example

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit oracle(3); // 2 query + 1 ancilla
    auto result = DeutschJozsa::solve(oracle, 2, 10, 42);
    return (result.type == DeutschJozsa::Result::CONSTANT) ? 0 : 1;
}
```

## Notes and Preconditions

- The oracle must act on `n + 1` qubits with the ancilla as the last qubit
- The solver uses the sampled counts; `shots` should be > 1 if you want repeated sampling

## QuditDeutschJozsa

Declared in `include/lindblad/algorithms.hpp`. Implemented in `src/algorithms/deutsch_jozsa.cpp`.

### Verdict enum

```cpp
enum class Verdict { CONSTANT, BALANCED };
```

### Result struct

```cpp
struct Result {
    QuditDeutschJozsa::Verdict verdict;  // CONSTANT or BALANCED
    int d;                               // qudit dimension
    int n;                               // number of query qudits
};
```

### solve

```cpp
static Result solve(
    int n, int d,
    const std::function<int(const std::vector<int>&)>& f,
    uint64_t seed = 0
);
```

Runs the qudit Deutsch-Jozsa circuit (n+1 qudits) and measures once. Single-shot — deterministic under the promise. `seed` controls measurement sampling (result is identical for any seed under exact simulation).

**Throws** `std::invalid_argument` if `d < 2`, `n < 1`, or `f` returns a value outside `[0, d)`.

## Related Pages

- [docs/algorithms/deutsch-jozsa.md](../algorithms/deutsch-jozsa.md)
- [docs/APIOverview.md](../APIOverview.md)
