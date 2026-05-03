# QPE API Deep Dive

This page documents the public `qpp::algorithms::QPE` API in detail.

## Header and Namespace

- Header: `include/qpp/algorithms.hpp`
- Namespace: `qpp::algorithms`

## Class Overview

`QPE` builds a phase-estimation circuit from a provided unitary circuit and returns a phase estimate derived from sampled measurements.

## `build_circuit`

Signature:

```cpp
static QuantumCircuit build_circuit(
    const QuantumCircuit& unitary,
    int num_eval_qubits
);
```

Behavior (verified against `src/algorithms/qpe.cpp`):

- Allocates `num_eval_qubits + unitary.n_qubits` total qubits
- Applies Hadamards to the evaluation register
- Builds controlled $U^{2^k}$ by explicitly constructing the unitary matrix
- Applies the inverse QFT on the evaluation qubits (controlled phase rotations + H)

Implementation details:

- The unitary matrix is built by simulating `unitary` on each basis vector
- $U^{2^k}$ is formed by repeated matrix multiplication
- Controlled unitary is embedded into a `UNITARY` instruction on control + target

## `estimate_phase`

Signature:

```cpp
tatic double estimate_phase(
    const QuantumCircuit& unitary,
    int num_eval_qubits,
    int shots = 1024,
    uint64_t seed = 0
);
```

Behavior:

- Builds the QPE circuit
- Measures all qubits
- Selects the most frequent measurement bitstring
- Interprets the first `num_eval_qubits` bits as a binary fraction in MSB-first order
- Returns $\frac{measured}{2^{num\_eval\_qubits}}$

## Example

```cpp
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

int main() {
    QuantumCircuit unitary(1);
    unitary.rz(M_PI / 2.0, 0);

    double phase = QPE::estimate_phase(unitary, 4, 2048, 7);
    return (phase > 0.0) ? 0 : 1;
}
```

## Notes and Preconditions

- `unitary` must be a valid unitary circuit; the matrix is reconstructed via simulation
- The full circuit measures all qubits, but only the evaluation register bits are used to extract the phase
- Precision depends on `num_eval_qubits`

## Related Pages

- [docs/algorithms/qpe.md](../algorithms/qpe.md)
- [docs/APIOverview.md](../APIOverview.md)
