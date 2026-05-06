# QPE

This page documents `lindblad::algorithms::QPE`.

## Purpose

QPE (Quantum Phase Estimation) estimates the phase of a unitary by applying controlled powers of that unitary to an evaluation register and then using the inverse QFT to decode the phase.

In lindblad, QPE is useful when you already have a unitary circuit and want a direct phase estimate from a measured output circuit.

## Theory Summary

QPE uses two registers:

- an evaluation register of `m` qubits
- a target register holding the unitary being estimated

The usual steps are:

1. Prepare the evaluation register in a uniform superposition.
2. Apply controlled powers of the unitary.
3. Apply the inverse QFT to the evaluation register.
4. Measure the evaluation register.

The measured bitstring approximates the phase of the unitary.

## Required Inputs

- A target `QuantumCircuit` representing the unitary to estimate
- The number of evaluation qubits
- Optional shot count and seed when calling `estimate_phase`

The target circuit must represent a unitary operation on `unitary.n_qubits` qubits.

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Build a unitary circuit and estimate its phase:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

QuantumCircuit unitary(1);
unitary.rz(0.25 * M_PI, 0);

auto circuit = QPE::build_circuit(unitary, 3);
auto phase = QPE::estimate_phase(unitary, 3, 1024, 42);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator Dependencies

QPE uses `StatevectorSimulator` internally to evaluate the controlled-unitary circuit and sample measurement counts.

The implementation in lindblad builds the controlled powers of the unitary explicitly and then measures the evaluation register.

## Public API Details

### `QPE::build_circuit`

- Takes the target unitary and the number of evaluation qubits
- Returns a `QuantumCircuit` containing the controlled-unitary ladder and inverse QFT
- The resulting circuit includes the evaluation register plus the original target register
- Each CU instruction uses the **LSB convention**: the control qubit occupies `targets[0]`
  and maps to bit 0 of the subspace index; even indices apply identity, odd apply $U^{2^k}$

### `QPE::estimate_phase`

- Builds the QPE circuit
- Measures the evaluation register
- Returns the estimated phase as a floating-point value in the range `[0, 1)`

## Example Code

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit unitary(1);
    unitary.rz(M_PI / 2.0, 0);

    double phase = QPE::estimate_phase(unitary, 4, 2048, 7);
    std::cout << "estimated phase: " << phase << '\n';
}
```

## Return Values and Outputs

- `build_circuit` returns a circuit with the evaluation and target registers combined
- `estimate_phase` returns a normalized phase estimate as a `double`

## Exceptions and Failure Modes

Common problems include:

- passing a non-unitary or mismatched target circuit
- using too few evaluation qubits for the desired precision
- expecting the estimate to be exact when the phase is not representable at the chosen resolution

## Common Pitfalls

- QPE estimates a phase, not a raw eigenvalue.
- Precision depends directly on the number of evaluation qubits.
- The target circuit must be a unitary operation that can be powered and controlled in the way the implementation expects.

## Testing Notes

At the moment the primary validation is implementation-level coverage through the algorithm source and the general simulator tests used elsewhere in the repo.

## Related Source Files

- [docs/api/qpe.md](../api/qpe.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/qpe.cpp](../../src/algorithms/qpe.cpp)
- [tests/test_simulators.cpp](../../tests/test_simulators.cpp)
