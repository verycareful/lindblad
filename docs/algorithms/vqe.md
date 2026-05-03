# VQE

This page documents `qpp::algorithms::VQE`.

## Purpose

VQE, the Variational Quantum Eigensolver, estimates the minimum eigenvalue of a Hamiltonian by optimizing a parameterized ansatz against the configured primitive stack.

In q++, VQE is the most general variational solver. QAOA and MAQAOA are specialized descendants of the same optimization pattern.

## Theory Summary

VQE alternates between two components:

1. a parameterized ansatz circuit
2. a Hamiltonian expectation-value objective

The optimizer adjusts the ansatz parameters until the measured energy is minimized.

The public API in q++ exposes both the optimizer entry point and a small set of ansatz generators for common use cases.

## Required Inputs

- A cost Hamiltonian as `qpp::SparsePauliOp`
- A parameterized ansatz circuit
- Optional initial parameters
- `VQE::Options` settings such as optimizer choice, iteration limit, and convergence threshold

## How to Invoke

Include the header:

```cpp
#include "qpp/algorithms.hpp"
```

Create an ansatz and optimize a Hamiltonian:

```cpp
using namespace qpp;
using namespace qpp::algorithms;

auto ansatz = VQE::real_amplitudes(4, 2);

SparsePauliOp hamiltonian({
    PauliString("ZZII", Complex128(1.0, 0.0)),
    PauliString("IIZZ", Complex128(0.5, 0.0))
});

VQE vqe;
vqe.options.max_iterations = 100;
vqe.options.optimizer = "COBYLA";

auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);
```

## Header Include Instructions

Use:

```cpp
#include "qpp/algorithms.hpp"
```

That header provides the VQE class and the common ansatz helper methods.

## Simulator and Primitive Dependencies

VQE uses `Estimator` internally to evaluate the ansatz energy against the target Hamiltonian.

The optimizer itself is driven by NLopt, while energy evaluations are handled through the estimator path configured on the VQE instance.

## Public API Details

### `VQE::Options`

- `max_iterations` limits the NLopt budget
- `convergence_threshold` sets the relative tolerance
- `optimizer` selects the local optimizer backend
- `seed` is available for reproducible workflows that use seeded components

### `VQE::Result`

- `eigenvalue` is the minimum energy found
- `optimal_parameters` stores the optimized ansatz parameters
- `num_iterations` counts the recorded optimization steps
- `energy_history` stores the objective value after each callback invocation
- `converged` reports whether NLopt returned success

### `compute_minimum_eigenvalue`

- Optimizes the supplied ansatz against the supplied Hamiltonian
- Uses the estimator attached to the `VQE` instance
- Returns the best energy and parameter set found by the optimizer

### Ansatz generators

- `efficient_su2(int n_qubits, int reps = 3)` builds a layered RY/RZ ansatz with linear entanglement
- `real_amplitudes(int n_qubits, int reps = 3)` builds an RY-only layered ansatz
- `two_local(...)` builds a configurable ansatz with custom rotation and entanglement blocks

## Example Code

```cpp
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

int main() {
    auto ansatz = VQE::efficient_su2(3, 1);

    SparsePauliOp hamiltonian({
        PauliString("ZZZ", Complex128(1.0, 0.0))
    });

    VQE vqe;
    vqe.options.max_iterations = 50;
    vqe.options.optimizer = "COBYLA";

    auto result = vqe.compute_minimum_eigenvalue(hamiltonian, ansatz);
    std::cout << "energy: " << result.eigenvalue << '\n';
}
```

## Return Values and Outputs

- `eigenvalue` is the best objective value found during optimization
- `optimal_parameters` contains the final ansatz parameters used by the optimizer
- `energy_history` records the optimizer trace
- `converged` reports whether the optimizer reached its stopping criterion

## Exceptions and Failure Modes

Common issues include:

- a Hamiltonian that does not match the ansatz qubit count
- an ansatz with zero trainable parameters
- an unsupported optimizer name
- estimator or primitive configuration that cannot evaluate the provided circuit

## Common Pitfalls

- VQE is generic; it does not assume a QAOA-style cost/mixer structure.
- The ansatz must already be parameterized before calling `compute_minimum_eigenvalue`.
- If `initial_params` is empty, the optimizer starts from a default small positive vector.

## Testing Notes

There is currently no dedicated VQE-only test file in the repository. VQE behavior is exercised indirectly through the variational stack and shared primitive coverage.

## Related Source Files

- [include/qpp/algorithms.hpp](../../include/qpp/algorithms.hpp)
- [src/algorithms/vqe.cpp](../../src/algorithms/vqe.cpp)
- [docs/algorithms/qaoa.md](qaoa.md)
- [docs/algorithms/maqaoa.md](maqaoa.md)
- [docs/api/vqe.md](../api/vqe.md)