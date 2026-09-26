# VQE API Deep Dive

This page documents the public `lindblad::algorithms::VQE` API in more detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Class Overview

`VQE` combines a configurable optimizer with an `Estimator` backend to compute the minimum eigenvalue of a Hamiltonian using a parameterized ansatz.

The class exposes one optimization entry point and a small set of ansatz generators.

## `Options`

Fields:

- `max_iterations`: cap on objective evaluations. Each evaluation is one
  circuit run through the estimator, and every derivative-free method here
  makes one evaluation per iteration, so the two counts coincide. Reaching
  the cap ends the run with `converged = false`; raise it and rerun
- `convergence_threshold`: relative `x` tolerance at which the minimiser
  declares convergence. FLOP measures it against `initial_step`, NLopt
  against the parameter magnitudes
- `optimizer`: which minimiser runs the classical loop. `"COBYLA"` is FLOP's
  implementation and the default; `"NLOPT_COBYLA"`, `"NELDER_MEAD"` and
  `"BOBYQA"` are NLopt's. An unknown name emits a warning through the warning
  channel and runs the default
- `initial_step`: displacement of the first trial points from the start along
  each parameter axis, in radians. A derivative-free method spends its first
  `n + 1` evaluations building this simplex, so on a small budget the step
  decides how much of it is left for descent
- `seed`: seed for the initial parameter draw when `initial_params` is empty.
  `0` draws the seed from `std::random_device`, as every simulator does. The
  draw is bit-identical across compilers and floating-point models for a
  given seed

Defaults:

- `max_iterations = 100`
- `convergence_threshold = 1e-6`
- `optimizer = "COBYLA"`
- `initial_step = 0.3`
- `seed = 0`

## `Result`

Fields:

- `eigenvalue`: minimum energy found. Always finite: when the minimiser
  returns without a finite value, the best finite energy actually evaluated
  is returned instead, and if the objective never produced one the call
  throws `std::runtime_error`. An NLopt method that refuses the request before evaluating anything throws `std::invalid_argument` naming the method, with NLopt's own reason after the colon
- `optimal_parameters`: the parameters at that energy
- `num_iterations`: objective evaluations made, which is also
  `energy_history.size()`
- `energy_history`: every energy evaluated, in order
- `converged`: whether the minimiser stopped on its tolerance with a finite
  value. Reaching `max_iterations`, a non-finite energy or a minimiser failure
  all report `false`

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
- Uses `initial_params` if provided, otherwise draws each parameter uniformly
  from `[-pi, pi)` under `options.seed`

Preconditions:

- Every term of the Hamiltonian is exactly the ansatz's qubit count wide, and the Hamiltonian has at least one term. Otherwise the estimator throws `std::invalid_argument` on the first evaluation (the width rule in [operators.md](operators.md#width-rule))
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
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

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
