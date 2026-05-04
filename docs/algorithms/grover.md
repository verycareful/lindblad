# Grover

This page documents `lindblad::algorithms::Grover`.

## Purpose

Grover's algorithm amplifies the probability of marked states in an unstructured search space.

In lindblad, the Grover helper builds a circuit around a provided oracle and either uses an explicit iteration count or computes the usual near-optimal default.

## Theory Summary

Grover search alternates between:

1. applying an oracle that marks the desired state
2. applying the diffusion operator
3. repeating for a chosen number of iterations

The diffusion operator in lindblad is implemented with the standard Hadamard/X pattern and a multi-controlled phase flip on the all-ones state.

## Required Inputs

- A `QuantumCircuit` oracle acting on the search register
- An optional iteration count
- Optional shot count and seed when calling `search`

The oracle should mark the solution state by phase inversion or equivalent amplitude-marking behavior.

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Construct or reuse an oracle and run Grover search:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

QuantumCircuit oracle(3);
// oracle construction goes here

auto circuit = Grover::build_circuit(oracle);
auto result = Grover::search(oracle, -1, 1024, 42);
```

If you want to override the iteration count explicitly:

```cpp
auto circuit = Grover::build_circuit(oracle, 2);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator Dependencies

Grover uses `StatevectorSimulator` internally for exact circuit execution and sampled counts.

The implementation constructs the diffusion operator directly in the circuit and measures the final state after all iterations.

## Public API Details

### `Grover::build_circuit`

- Takes the oracle circuit and an optional iteration count
- If the iteration count is negative, the implementation uses the usual `pi/4 * sqrt(N)` style estimate
- Returns a `QuantumCircuit` containing the full Grover iterate sequence

### `Grover::Result`

- `solution` stores the most frequent measured bitstring
- `num_iterations` stores the iteration count used by the solver
- `probability` stores the measured frequency of the chosen solution bitstring

### `Grover::search`

- Builds the Grover circuit
- Measures all qubits
- Returns the most likely observed solution and its empirical probability

## Example Code

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit oracle(2);
    oracle.cz(0, 1);

    auto result = Grover::search(oracle, -1, 1024, 7);
    std::cout << "solution: " << result.solution << '\n';
}
```

## Return Values and Outputs

- `build_circuit` returns a Grover search circuit
- `search` returns a `Grover::Result` with the best observed bitstring and its measured probability

## Exceptions and Failure Modes

Common issues include:

- an oracle that does not mark the intended solution state
- an iteration count that is too small or too large for the target search space
- a search register size that does not match the oracle structure

## Common Pitfalls

- Grover returns the most frequently observed bitstring, not a guaranteed exact answer for every shot.
- The default iteration heuristic is only an estimate.
- The oracle must be built so that the marked state is the one you expect the algorithm to amplify.

## Testing Notes

Grover behavior is exercised indirectly through the simulator and algorithm coverage already present in the repository.

## Related Source Files

- [docs/api/grover.md](../api/grover.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/grover.cpp](../../src/algorithms/grover.cpp)
- [tests/test_simulators.cpp](../../tests/test_simulators.cpp)
