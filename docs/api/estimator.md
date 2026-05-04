# Estimator API Deep Dive

This page documents the public `lindblad::Estimator` primitive.

## Header and Namespace

- Header: `include/lindblad/primitives.hpp`
- Namespace: `lindblad`

## Class Overview

`Estimator` computes expectation values of a `SparsePauliOp` against a circuit.
The current implementation always evaluates with a statevector simulation and
supports a cached transpilation path for repeated parameter sweeps.

## `Options`

Fields and defaults:

- `shots = 0`: declared but not used in the current implementation
- `seed = 0`: declared but not used in the current implementation
- `noise_model`: declared but not used in the current implementation
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
  caches the result using a structure key (gate types and qubit indices only)
- On cache hit, reuses the cached unbound circuit
- Binds parameters by name using the transpiled circuit parameter names when
  available, otherwise the original circuit parameter names
- Binds up to `min(parameters.size(), names.size())` parameters
- Runs a statevector simulation and returns `observable.expectation_value(state)`
- Throws `std::runtime_error` if simulation fails

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
