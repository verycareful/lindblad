# QAOA API Deep Dive

This page documents the public `lindblad::algorithms::QAOA` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Class Overview

`QAOA` optimizes a cost Hamiltonian by alternating cost and mixer unitaries for `p` layers and using NLopt for parameter search. It exposes an `optimize` entry point and a `build_circuit` helper.

## `Options`

Fields and defaults (from the header):

- `p = 1`: number of QAOA layers
- `max_iterations = 100`: optimizer budget
- `convergence_threshold = 1e-6`: relative tolerance
- `optimizer = "COBYLA"`: intended NLopt backend selector
- `seed = 0`: RNG seed for parameter initialization
- `initial_thetas`: optional per-qubit `Ry(theta)` initialization (empty uses H)

Important implementation detail:

- The current implementation always constructs a COBYLA optimizer and ignores `options.optimizer`.

## `Result`

Fields:

- `optimal_value`: best energy found
- `initial_params`: initial parameter vector (random perturbation)
- `optimal_params`: final parameter vector, ordered `[gamma_1, beta_1, ..., gamma_p, beta_p]`
- `counts`: final sampler counts
- `best_bitstring`: selected bitstring after post-processing
- `num_iterations`: total optimizer iterations
- `converged`: true when NLopt reports success (excluding max-eval termination)

## `optimize`

Signature:

```cpp
Result optimize(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian = {}
);
```

Behavior (verified against `src/algorithms/qaoa.cpp`):

- If `mixer_hamiltonian` is empty, constructs a default mixer $\sum_i X_i$
- Parameter count is `2 * p`
- Initializes parameters in `[-0.05, 0.05]` with RNG seeded by `options.seed`
- Uses COBYLA with bounds `[-2*pi, 2*pi]` and initial step size `0.3`
- Evaluates the objective with `Estimator::run_single`
- Samples the final circuit with `Sampler::run_single`
- Chooses `best_bitstring` by minimum computational-basis cost (tie-break by count)

Bitstring note:

- The computational-basis scoring treats bitstrings as MSB-first

## `build_circuit`

Signature:

```cpp
QuantumCircuit build_circuit(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian,
    const std::vector<double>& params
) const;
```

Behavior:

- Initial state is `H|0>` on each qubit, unless `initial_thetas` is set
- Each layer applies cost then mixer rotations
- Cost terms are implemented as Pauli rotations (including basis changes for X/Y)
- Mixer terms apply `Rx`, `Ry`, or `Rz` depending on the Pauli operator

## Example

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    SparsePauliOp cost({
        PauliString("ZZ", Complex128(1.0, 0.0))
    });

    QAOA qaoa;
    qaoa.options.p = 2;
    qaoa.options.max_iterations = 100;
    qaoa.estimator.options.shots = 0;
    qaoa.sampler.options.shots = 1024;

    auto result = qaoa.optimize(cost);
    return result.converged ? 0 : 1;
}
```

## Related Pages

- [docs/algorithms/qaoa.md](../algorithms/qaoa.md)
- [docs/APIOverview.md](../APIOverview.md)
