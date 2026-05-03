# QAOA

This page documents `qpp::algorithms::QAOA`.

## Purpose

QAOA (Quantum Approximate Optimization Algorithm) solves discrete optimization problems by alternating between a cost Hamiltonian and a mixer Hamiltonian.

In q++, QAOA is the lighter-weight variational solver compared with MAQAOA. It is the right choice when you want a compact parameterization with one gamma and one beta per layer.

## Theory Summary

For each layer `l`:

1. Apply the cost Hamiltonian with angle `gamma_l`.
2. Apply the mixer Hamiltonian with angle `beta_l`.
3. Repeat for `p` layers.
4. Measure or evaluate the resulting circuit with the configured primitives.

The default implementation follows the familiar alternating-operator pattern and returns the best sampled bitstring and the optimized parameter vector.

## Required Inputs

- A cost Hamiltonian as `qpp::SparsePauliOp`
- Optionally, a mixer Hamiltonian as `qpp::SparsePauliOp`
- `QAOA::Options` settings:
  - `p`: number of layers
  - `max_iterations`
  - `convergence_threshold`
  - `optimizer`
  - `seed`
  - `initial_thetas` for QSP-style initial state preparation

## How to Invoke

Include the public header:

```cpp
#include "qpp/algorithms.hpp"
```

Construct the Hamiltonian and configure the solver:

```cpp
using namespace qpp;
using namespace qpp::algorithms;

QAOA qaoa;
qaoa.options.p = 2;
qaoa.options.max_iterations = 100;
qaoa.options.optimizer = "COBYLA";
qaoa.estimator.options.shots = 0;
qaoa.sampler.options.shots = 1024;

SparsePauliOp cost({
    PauliString("ZZ", Complex128(1.0, 0.0))
});

auto result = qaoa.optimize(cost);
```

To inspect the circuit directly:

```cpp
auto circuit = qaoa.build_circuit(cost, {}, result.optimal_params);
```

## Header Include Instructions

Use:

```cpp
#include "qpp/algorithms.hpp"
```

## Simulator and Primitive Dependencies

QAOA uses both `Estimator` and `Sampler`.

- `Estimator` evaluates the objective during optimization
- `Sampler` collects bitstring outcomes from the final circuit
- The solver can use exact or sampled primitives depending on the configured shot counts

## Public API Details

### `QAOA::Options`

- `p` sets the number of layers
- `max_iterations` controls the optimizer budget
- `convergence_threshold` controls stop tolerance
- `optimizer` is currently not wired; the implementation always uses COBYLA
- `seed` drives reproducible initialization
- `initial_thetas` optionally replaces the default H-state preparation with per-qubit `Ry(theta)` initialization

### `QAOA::Result`

- `optimal_value` is the best energy found
- `initial_params` stores the initial guess used by the optimizer
- `optimal_params` stores the best parameters in `[gamma_1, beta_1, ..., gamma_p, beta_p]` order
- `counts` stores measurement counts from the final sampled circuit
- `best_bitstring` stores the highest-priority sampled bitstring
- `num_iterations` records optimizer iterations
- `converged` indicates whether the optimizer stopped due to convergence rather than iteration exhaustion

### `QAOA::build_circuit`

- Builds the layerwise alternating circuit for a given cost and mixer Hamiltonian
- Takes an explicit parameter vector in layer order
- Returns a `QuantumCircuit` ready for simulation or measurement

### `QAOA::optimize`

- Runs the optimizer and returns the best circuit parameters found
- Uses the configured `Estimator` and `Sampler`
- Returns both continuous optimization output and sampled bitstring data
- `best_bitstring` is chosen by minimum computational-basis cost (tie-break by count)

## Example Code

```cpp
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

int main() {
    SparsePauliOp cost({
        PauliString("ZZ", Complex128(1.0, 0.0))
    });

    QAOA qaoa;
    qaoa.options.p = 1;
    qaoa.options.optimizer = "COBYLA";
    qaoa.estimator.options.shots = 0;
    qaoa.sampler.options.shots = 1024;

    auto result = qaoa.optimize(cost);
    std::cout << "best bitstring: " << result.best_bitstring << '\n';
}
```

## Return Values and Outputs

- `optimal_value` is the cost-energy objective value at the best parameters
- `optimal_params` contains the layer parameters chosen by the optimizer
- `counts` contains final measurement statistics
- `best_bitstring` is the best sampled solution candidate

## Exceptions and Failure Modes

Common issues include:

- the cost Hamiltonian is empty or malformed
- the mixer Hamiltonian does not match the expected qubit count
- the optimizer name is not supported by NLopt
- the circuit is configured with inconsistent qubit/register dimensions

## Common Pitfalls

- `optimal_params` are stored in layer order, not grouped by Hamiltonian term.
- `initial_thetas` changes the state preparation before the QAOA layers start.
- `best_bitstring` is selected from sampled outcomes, so it reflects the final measurement stage, not only the continuous optimizer's objective value.

## Testing Notes

There is currently no standalone QAOA test file. The closest coverage lives in the MAQAOA tests, which exercise the same circuit-building and optimization plumbing:

- [tests/test_maqaoa.cpp](../../tests/test_maqaoa.cpp)
- [tests/test_maqaoa_5qubit.cpp](../../tests/test_maqaoa_5qubit.cpp)
- [tests/test_maqaoa_microgrid.cpp](../../tests/test_maqaoa_microgrid.cpp)

## Related Source Files

- [docs/api/qaoa.md](../api/qaoa.md)
- [include/qpp/algorithms.hpp](../../include/qpp/algorithms.hpp)
- [src/algorithms/qaoa.cpp](../../src/algorithms/qaoa.cpp)
- [tests/test_maqaoa.cpp](../../tests/test_maqaoa.cpp)
