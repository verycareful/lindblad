# Deutsch-Jozsa

This page documents `qpp::algorithms::DeutschJozsa`.

## Purpose

Deutsch-Jozsa determines whether a Boolean oracle is constant or balanced using a single quantum query.

In q++, the helper is designed for the standard textbook oracle layout:

- the first `n` qubits form the query register
- the final qubit is the ancilla

## Theory Summary

The algorithm prepares the query register in superposition, initializes the ancilla to `|1⟩`, and then uses the oracle once.

After the oracle:

1. Apply Hadamard gates to the query register again.
2. Measure the query register.
3. If the measured string is all zeros, the oracle is constant.
4. Otherwise, the oracle is balanced.

## Required Inputs

- A `QuantumCircuit` oracle on `n + 1` qubits
- `n`, the number of query qubits
- Optional shot count and seed

The oracle should follow the standard Deutsch-Jozsa form and act on the ancilla as the Boolean output bit.

## How to Invoke

Include the header:

```cpp
#include "qpp/algorithms.hpp"
```

Build the oracle and run the solver:

```cpp
using namespace qpp;
using namespace qpp::algorithms;

QuantumCircuit oracle(4); // 3 query qubits + 1 ancilla
// constant oracle: leave it empty

auto result = DeutschJozsa::solve(oracle, 3);
```

A balanced example that flips the ancilla based on one query qubit:

```cpp
QuantumCircuit balanced(4);
balanced.cx(0, 3);
auto result = DeutschJozsa::solve(balanced, 3);
```

## Header Include Instructions

Use:

```cpp
#include "qpp/algorithms.hpp"
```

## Simulator Dependencies

Deutsch-Jozsa uses `StatevectorSimulator` internally for exact execution and sampled measurement counts.

The implementation measures the query register after the oracle and the second Hadamard layer.

## Public API Details

### `DeutschJozsa::build_circuit`

- Takes the oracle circuit and the query register size
- Builds the full Deutsch-Jozsa circuit around the provided oracle
- Returns a `QuantumCircuit` ready for measurement or simulation

### `DeutschJozsa::Result`

- `type` is either `CONSTANT` or `BALANCED`

### `DeutschJozsa::solve`

- Runs the Deutsch-Jozsa circuit
- Returns the classification result
- Uses the measured query register to decide the answer

## Example Code

```cpp
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

int main() {
    QuantumCircuit oracle(2);
    auto result = DeutschJozsa::solve(oracle, 1);
    if (result.type == DeutschJozsa::Result::CONSTANT) {
        std::cout << "constant\n";
    }
}
```

## Return Values and Outputs

- `Result::type` reports whether the oracle is constant or balanced

## Exceptions and Failure Modes

Common issues include:

- the oracle does not use the last qubit as the ancilla
- the oracle has the wrong qubit count
- the oracle encodes a function that is neither constant nor balanced in the expected format

In those cases, the solver may return the wrong classification rather than throwing.

## Common Pitfalls

- The ancilla is part of the circuit and must be included when building the oracle.
- Only the query register is used for the final classification decision.
- This algorithm is only valid for the textbook Deutsch-Jozsa oracle structure.

## Testing Notes

Relevant tests live in:

- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)

## Related Source Files

- [docs/api/deutsch-jozsa.md](../api/deutsch-jozsa.md)
- [include/qpp/algorithms.hpp](../../include/qpp/algorithms.hpp)
- [src/algorithms/deutsch_jozsa.cpp](../../src/algorithms/deutsch_jozsa.cpp)
- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)
