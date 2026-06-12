# Estimator API Deep Dive

This page documents the public `lindblad::Estimator` primitive.

## Header and Namespace

- Header: `include/lindblad/primitives.hpp`
- Namespace: `lindblad`

## Class Overview

`Estimator` computes expectation values of a `SparsePauliOp` against a circuit.
It supports three execution modes — exact statevector, exact density-matrix
(for noisy circuits), and per-Pauli-term shot sampling — selected automatically
from `Options`. It also supports a cached transpilation path for repeated
parameter sweeps.

## `Options`

Fields and defaults:

- `shots = 0`: shot budget **per non-identity Pauli term** in the observable.
  - `shots == 0`: exact analytic expectation (zero variance, reproducible).
    THROWS for circuits containing measurement or classically-conditioned
    instructions (frozen in R.1.12): the exact expectation of a stochastic
    trajectory is undefined; estimate such circuits from counts with
    `shots > 0`.
  - `shots  > 0`: real shot-noise sampling — each non-identity term is rotated
    into the Z measurement basis and measured `shots` times. Variance scales
    as `1/√shots` per term.
  - **Changed in R.1.10.7:** `shots > 0` now performs real sampling. Prior
    releases used `shots` as a hidden DM-backend toggle and returned exact
    values with zero variance regardless of `shots`.
  - Pauli strings follow the project LSB-first convention (`pauli[q]` acts on
    qubit q); the exact and sampling paths agree on it (frozen in R.1.12).
- `seed = 0`: RNG seed forwarded to the simulator. Each Pauli term gets a
  decorrelated derived seed so the shot streams across terms are independent.
- `noise_model`: when non-ideal (`!is_ideal()`), the sampling/exact path uses
  `DensityMatrixSimulator` so Kraus channels are applied. Otherwise
  `StatevectorSimulator` is used.
- `optimization_level = 0`: enables transpilation and caching when > 0.

## `clear_cache`

Clears the internal transpilation cache. This is useful when you change
transpilation settings or need to release cached circuits.

## `run_batch`

Signature:

```cpp
std::vector<double> run_batch(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    const std::vector<std::vector<double>>& parameter_values
);
```

Behavior:

- Evaluates each parameter vector with `run_single`
- Parallelized with OpenMP when enabled
- Returns one expectation value per parameter vector

## `run_single`

Signature:

```cpp
double run_single(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    const std::vector<double>& parameters = {}
);
```

Behavior (verified against the implementation):

- When `options.optimization_level > 0`, transpiles the unbound circuit and
  caches the result using a structure key (gate types, qubit indices, and `n_clbits`)
- Transpilation runs outside the cache mutex (double-checked locking); insertion is guarded
- On cache hit, reuses the cached unbound circuit
- Binds parameters by name using the transpiled circuit parameter names when
  available, otherwise the original circuit parameter names
- Binds up to `min(parameters.size(), names.size())` parameters
- Three execution modes, selected from `Options`:
  - **`shots > 0` — real sampling**: decomposes the observable into Pauli terms;
    for each non-identity term, rotates the prepared circuit into that term's
    Z measurement basis (H for X, S†H for Y, identity for Z), runs `options.shots`
    measurements through `StatevectorSimulator` (or `DensityMatrixSimulator` when
    the noise model is non-ideal), and accumulates a parity-weighted estimate.
    Identity terms contribute their coefficient analytically (no sampling).
    Variance scales as `1/√shots` per term.
  - **`shots == 0`, noise non-ideal**: runs `DensityMatrixSimulator` with zero
    shots and reads the exact expectation off the final mixed state via
    `expectation_value_sparse(observable)`.
  - **`shots == 0`, ideal**: runs `StatevectorSimulator::eval_expectation(circuit, observable)`
    which evaluates the observable in-place, avoiding a $O(2^n)$ allocation for
    the final state vector.
- Throws `std::runtime_error` if simulation fails on any path

Preconditions:

- The circuit and observable must target the same number of qubits
- Parameter ordering must match the circuit parameter name order

## `gradient`

Signature:

```cpp
std::vector<double> gradient(
    const QuantumCircuit& circuit,
    const SparsePauliOp& observable,
    const std::vector<double>& parameters
);
```

Behavior:

- Uses the parameter-shift rule with a shift of `pi/2`
- Builds `2 * parameters.size()` shifted vectors
- Evaluates them with a single `run_batch` call
- Returns a gradient vector of the same length as `parameters`

## Example

```cpp
#include "lindblad/primitives.hpp"

using namespace lindblad;

int main() {
    QuantumCircuit circuit(2);
    circuit.h(0).cx(0, 1);

    SparsePauliOp observable({PauliString("ZZ", Complex128(1.0, 0.0))});

    Estimator estimator;
    estimator.options.optimization_level = 1;

    double value = estimator.run_single(circuit, observable);
    return (value > -2.0) ? 0 : 1;
}
```

## Related Pages

- [docs/APIOverview.md](../APIOverview.md)
- [docs/algorithms/vqe.md](../algorithms/vqe.md)
- [docs/algorithms/qaoa.md](../algorithms/qaoa.md)
- [docs/algorithms/maqaoa.md](../algorithms/maqaoa.md)
