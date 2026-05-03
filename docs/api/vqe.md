# VQE API Deep Dive

This page documents the public `qpp::algorithms::VQE` API in more detail.

## Header and Namespace

- Header: `include/qpp/algorithms.hpp`
- Namespace: `qpp::algorithms`

## Class Overview

`VQE` combines a configurable optimizer with an `Estimator` backend to compute the minimum eigenvalue of a Hamiltonian using a parameterized ansatz.

The class exposes one optimization entry point and a small set of ansatz generators.

## `Options`

Fields:

- `max_iterations`: maximum NLopt iterations
- `convergence_threshold`: relative stopping tolerance
- `optimizer`: NLopt backend name
- `seed`: reproducibility seed for workflows that use seeded components

Defaults:

- `max_iterations = 100`
- `convergence_threshold = 1e-6`
- `optimizer = "COBYLA"`
- `seed = 0`

## `Result`

Fields:

- `eigenvalue`: minimum energy found
- `optimal_parameters`: final optimized ansatz parameters
- `num_iterations`: number of recorded optimizer steps
- `energy_history`: objective trace over the optimization run
- `converged`: whether NLopt reported success

## `compute_minimum_eigenvalue`

Signature:

```cpp
Result compute_minimum_eigenvalue(
    const SparsePauliOp& hamiltonian,
    const QuantumCircuit& ansatz,
    const std::vector<double>& initial_params = {}
);
```

Behavior:

- Evaluates the supplied ansatz against the supplied Hamiltonian
- Uses the `Estimator` member on the `VQE` instance
- Uses `initial_params` if provided, otherwise starts from a small default vector

Preconditions:

- The ansatz must have the same qubit count as the Hamiltonian
- The ansatz must expose the expected number of parameters if you rely on automatic initialization

## Ansatz Generators

### `efficient_su2(int n_qubits, int reps = 3)`

Builds a layered RY/RZ ansatz with linear CX entanglement.

### `real_amplitudes(int n_qubits, int reps = 3)`

Builds a layered RY-only ansatz with linear CX entanglement.

### `two_local(...)`

Builds a configurable ansatz with selectable rotation blocks, entanglement blocks, repetition count, and entanglement topology.

Supported entanglement modes:

- `linear`
- `full`
- `circular`

## Example

```cpp
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

int main() {
    auto ansatz = VQE::real_amplitudes(2, 1);
    SparsePauliOp hamiltonian({PauliString("ZZ", Complex128(1.0, 0.0))});

    VQE vqe;
    vqe.options.optimizer = "COBYLA";
    auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);
}
```

## Related Pages

- [docs/algorithms/vqe.md](../algorithms/vqe.md)
- [docs/APIOverview.md](../APIOverview.md)