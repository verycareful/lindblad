# QPE API Deep Dive

This page documents the public `lindblad::algorithms::QPE` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

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
- Controlled unitary is embedded into a `UNITARY` instruction on `{k} ∪ target_qubits`
- The CU matrix uses the **LSB convention**: `targets[0]` (= control qubit `k`)
  maps to bit 0 of the subspace index. Even indices have ctrl=0 (identity block);
  odd indices have ctrl=1 (U^power block)

## `estimate_phase`

Signature:

```cpp
static double estimate_phase(
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
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

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

## QuditPhaseEstimation

Declared in `include/lindblad/algorithms.hpp`. Implemented in `src/algorithms/qpe.cpp`.

### Result struct

```cpp
struct Result {
    std::vector<int> phase_digits;   // m clock digits in little-endian base d
    double phase_estimate;            // φ = Σ_j digit_j / d^{j+1} ∈ [0, 1)
    int m;                            // number of clock qudits
    int d;                            // qudit dimension
};
```

### estimate

```cpp
static Result estimate(
    int m, int d,
    const std::vector<Complex128>& U,          // d×d row-major unitary
    const std::vector<Complex128>& eigenstate,  // d-element normalised amplitude vector
    uint64_t seed = 0
);
```

Runs the qudit QPE circuit and measures once. Single-shot — deterministic when `eigenstate` is an exact eigenstate of `U`.

**Throws** `std::invalid_argument` if `d < 2`, `m < 1`, `U.size() != d*d`, or `eigenstate.size() != d`.

## Related Pages

- [docs/algorithms/qpe.md](../algorithms/qpe.md)
- [docs/APIOverview.md](../APIOverview.md)
