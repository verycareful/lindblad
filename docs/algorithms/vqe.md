# VQE

This page documents `lindblad::algorithms::VQE`.

## Purpose

VQE, the Variational Quantum Eigensolver, estimates the minimum eigenvalue of a Hamiltonian by optimizing a parameterized ansatz against the configured primitive stack.

In lindblad, VQE is the most general variational solver. QAOA and MAQAOA are specialized descendants of the same optimization pattern.

## Theory Summary

VQE alternates between two components:

1. a parameterized ansatz circuit
2. a Hamiltonian expectation-value objective

The optimizer adjusts the ansatz parameters until the measured energy is minimized.

The public API in lindblad exposes both the optimizer entry point and a small set of ansatz generators for common use cases.

## Required Inputs

- A cost Hamiltonian as `lindblad::SparsePauliOp`
- A parameterized ansatz circuit
- Optional initial parameters
- `VQE::Options` settings such as optimizer choice, iteration limit, and convergence threshold

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Create an ansatz and optimize a Hamiltonian:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

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
#include "lindblad/algorithms.hpp"
```

That header provides the VQE class and the common ansatz helper methods.

## Simulator and Primitive Dependencies

VQE uses `Estimator` internally to evaluate the ansatz energy against the target Hamiltonian.

The classical loop runs through one internal seam over two minimiser libraries. `options.optimizer` selects the algorithm: `"COBYLA"` (default, FLOP's implementation), or NLopt's `"NLOPT_COBYLA"`, `"NELDER_MEAD"` and `"BOBYQA"`. Energy evaluations go through the estimator attached to the `VQE` instance.

Two things about the numbers a run produces. A fixed `seed` gives the same starting point on every compiler and floating-point model, so a trajectory is reproducible across builds. Above `2^20` amplitudes the statevector kernels and the expectation-value reduction run in parallel, and the summation order then depends on the thread count; the energy differs in its last bits between thread counts, and a derivative-free minimiser can turn that into a different trajectory and a different final energy. To reproduce a run at that size, pin `OMP_NUM_THREADS` as well as the seed.

## Public API Details

### `VQE::Options`

- `max_iterations` caps objective evaluations (one per iteration for every method here); reaching it reports `converged = false`
- `convergence_threshold` sets the relative `x` tolerance
- `optimizer` selects the minimiser: `"COBYLA"` (default, FLOP), `"NLOPT_COBYLA"`, `"NELDER_MEAD"`, `"BOBYQA"` (NLopt)
- `initial_step` sets the first trial displacement along each parameter axis (default `0.3`)
- `seed` seeds the initial parameter draw; `0` draws from `std::random_device`

### `VQE::Result`

- `eigenvalue` is the minimum energy found
- `optimal_parameters` stores the optimized ansatz parameters
- `num_iterations` counts the objective evaluations made
- `energy_history` stores every energy evaluated, in order
- `converged` reports whether the minimiser stopped on its tolerance with a finite value; the evaluation cap reports `false`

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
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

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
- an unsupported optimizer name (defaults to COBYLA with a warning)
- estimator or primitive configuration that cannot evaluate the provided circuit

## Common Pitfalls

- VQE is generic; it does not assume a QAOA-style cost/mixer structure.
- The ansatz must already be parameterized before calling `compute_minimum_eigenvalue`.
- If `initial_params` is empty, the start point is drawn uniformly in `[-pi, pi)` per parameter from `options.seed`.

## Testing Notes

The dedicated optimizer selection tests live in [tests/test_V11283_algos.cpp](../../tests/test_V11283_algos.cpp). VQE behavior is also exercised through the variational stack and shared primitive coverage.

## Related Source Files

- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/vqe.cpp](../../src/algorithms/vqe.cpp)
- [include/lindblad/detail/optimizer.hpp](../../include/lindblad/detail/optimizer.hpp): the minimiser seam (FLOP and NLopt behind one interface)
- [src/algorithms/optimizer.cpp](../../src/algorithms/optimizer.cpp)
- [docs/algorithms/qaoa.md](qaoa.md)
- [docs/algorithms/maqaoa.md](maqaoa.md)
- [docs/api/vqe.md](../api/vqe.md)
