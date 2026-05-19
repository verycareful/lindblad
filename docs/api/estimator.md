# Estimator API Deep Dive

This page documents the public `lindblad::Estimator` primitive.

## Header and Namespace

- Header: `include/lindblad/primitives.hpp`
- Namespace: `lindblad`

## Class Overview

`Estimator` computes expectation values of a `SparsePauliOp` against a circuit.
It supports two execution paths — exact statevector (default, ideal circuits) and
density-matrix simulation (noisy circuits or shot-based sampling) — selected
automatically from `Options`. It also supports a cached transpilation path for
repeated parameter sweeps.

## `Options`

Fields and defaults:

- `shots = 0`: when > 0, routes execution through `DensityMatrixSimulator` with this many shots
- `seed = 0`: RNG seed forwarded to `DensityMatrixSimulator::run` when the noisy path is active
- `noise_model`: when non-ideal (i.e. `!noise_model.is_ideal()`), routes execution through `DensityMatrixSimulator`; when ideal and `shots == 0`, the exact statevector path is used
- `optimization_level = 0`: enables transpilation and caching when > 0

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
- **Noisy / shot-based path** (when `!options.noise_model.is_ideal()` or `options.shots > 0`):
  routes through `DensityMatrixSimulator`; uses `options.shots` shots (defaults to 8192 if
  `shots == 0` but noise model is non-ideal); returns
  `dm_result.final_state.expectation_value_sparse(observable)`
- **Exact path** (when noise model is ideal and `shots == 0`): runs `StatevectorSimulator`
  and returns `observable.expectation_value(result.final_state)`
- Throws `std::runtime_error` if simulation fails on either path

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
