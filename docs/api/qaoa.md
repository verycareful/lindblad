# QAOA API Deep Dive

This page documents the public `lindblad::algorithms::QAOA` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Class Overview

`QAOA` optimizes a cost Hamiltonian by alternating cost and mixer unitaries for `p` layers, with a derivative-free minimiser (FLOP's COBYLA by default, NLopt's methods selectable) for the parameter search. It exposes an `optimize` entry point and a `build_circuit` helper.

## `Options`

Fields and defaults (from the header):

- `p = 1`: number of QAOA layers
- `max_iterations = 100`: cap on objective evaluations (one circuit run through the estimator each; one per iteration for every method here). Reaching it ends the run with `converged = false`
- `convergence_threshold = 1e-6`: relative `x` tolerance at which the minimiser declares convergence
- `optimizer = "COBYLA"`: which minimiser runs the classical loop. `"COBYLA"` is FLOP's; `"NLOPT_COBYLA"`, `"NELDER_MEAD"` and `"BOBYQA"` are NLopt's. An unknown name warns and runs the default
- `initial_step = 0.3`: first trial displacement along each parameter axis, in radians. BOBYQA needs every parameter interval at least twice the step wide, so with the `[-2*pi, 2*pi]` box a step above `2*pi` makes it refuse the request before evaluating anything; that throws `std::invalid_argument` naming `QAOA::optimize` and the method, with NLopt's own reason after the colon
- `seed = 0`: seed for the initial parameter perturbation and the final sampling; `0` draws from `std::random_device`. The draw is bit-identical across compilers for a given seed
- `initial_thetas`: optional per-qubit `Ry(theta)` initialization (empty uses H)

## `Result`

Fields:

- `optimal_value`: best energy found
- `initial_params`: initial parameter vector (random perturbation)
- `optimal_params`: final parameter vector, ordered `[gamma_1, beta_1, ..., gamma_p, beta_p]`
- `counts`: final sampler counts
- `best_bitstring`: selected bitstring after post-processing
- `num_iterations`: objective evaluations made
- `converged`: true when the minimiser stopped on its tolerance with a finite value; the evaluation cap, a non-finite value and a minimiser failure all report false

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
- Throws `std::invalid_argument`, here and in `build_circuit` alike, for a cost Hamiltonian with no terms, with terms of different widths, or that is not Hermitian (the cost layer, like the mixer's, rotates by each coefficient's real part), for a mixer term whose width is not the cost Hamiltonian's, and for a mixer term with a non-zero imaginary coefficient: the mixer rotation uses the real part only, and a mixer that is not Hermitian has no unitary `exp(-i*beta*B)`, so the real part alone would run a different mixer under the caller's name
- Parameter count is `2 * p`
- Initializes parameters in `[-0.05, 0.05]` with RNG seeded by `options.seed`
- Uses the minimiser `options.optimizer` names (default FLOP COBYLA); bounds `[-2*pi, 2*pi]` on every parameter, first step `options.initial_step`
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
- Mixer terms are evolved as the ordered product of per-term rotations
  `exp(-i·β·c_k·P_k)`: single-qubit terms as `Rx`/`Ry`/`Rz`, multi-qubit
  terms via the same basis-change + CX-chain recipe as the cost unitary.
  Exact when the terms commute (the default X mixer); a first-order Trotter
  step otherwise

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
