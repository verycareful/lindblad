# Simon

This page documents `lindblad::algorithms::Simon`.

## Purpose

Simon’s algorithm finds the hidden period `s` of a 2-to-1 function where `f(x) = f(x ⊕ s)`.

In lindblad, the helper is built around the standard Simon oracle layout and returns both the recovered period and the sampled linear equations.

## Theory Summary

Simon’s algorithm uses the following flow:

1. Prepare the query register in superposition.
2. Apply the oracle.
3. Apply Hadamard gates to the query register again.
4. Measure the query register repeatedly.
5. Use the resulting equations to solve for the hidden period over GF(2).

Each measured string `y` satisfies `y · s = 0 mod 2`.

## Required Inputs

- A `QuantumCircuit` oracle on `2n` qubits
- `n`, the query register size
- Optional seed and extra sampling count

The oracle should encode a valid Simon promise instance.

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Build the oracle and solve:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

QuantumCircuit oracle(6); // 3 query qubits + 3 output qubits
// oracle construction goes here

auto result = Simon::solve(oracle, 3, 42);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator Dependencies

Simon uses `StatevectorSimulator` internally to sample repeated query-register outcomes.

The implementation stores the raw measured equations and then runs a GF(2) elimination step to recover the period.

## Public API Details

### `Simon::build_circuit`

- Takes a Simon oracle and the query size
- Builds the standard Simon query circuit
- Returns a `QuantumCircuit`

### `Simon::Result`

- `period` stores the recovered hidden string
- `equations` stores the non-zero sampled equation strings used in the solve step

### `Simon::solve`

- Repeatedly samples the Simon circuit
- Collects independent equations from measurement results
- Solves for the hidden period using Gaussian elimination over GF(2)

### `Simon::gaussian_eliminate`

- Private helper used by `solve`
- Reduces the sampled equations and reconstructs the final bitstring

## Example Code

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit oracle(6);
    // build a 3-qubit Simon oracle here

    auto result = Simon::solve(oracle, 3, 7);
    std::cout << "period: " << result.period << '\n';
}
```

## Return Values and Outputs

- `period` is the recovered hidden period string
- `equations` lists the sampled linear equations used during recovery

## Exceptions and Failure Modes

Common issues include:

- an oracle that does not satisfy Simon’s promise
- a query/output register layout that does not match the documented `2n` qubit structure
- too few independent equations to recover the full period reliably

In those cases, the solver may return an incomplete or incorrect period.

## Common Pitfalls

- Simon’s algorithm usually needs multiple samples, not just one run.
- The oracle must be a proper 2-to-1 promise instance.
- The returned equations are useful for debugging and should not be ignored when diagnosing incorrect recovery.

## Testing Notes

Relevant tests live in:

- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)

## Related Source Files

- [docs/api/simon.md](../api/simon.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/simon.cpp](../../src/algorithms/simon.cpp)
- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)
