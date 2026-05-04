# MAQAOA

This page documents `lindblad::algorithms::MAQAOA`.

## Purpose

MAQAOA extends QAOA by allowing multiple angle assignments per layer rather than a single gamma and beta pair.

This is the right page when you want:

- one gamma per cost term or qubit family
- orbit-reduced parameterization
- per-qubit mixer weighting
- QSP-style initial state preparation
- layerwise or progressive optimization

## Theory Summary

MAQAOA generalizes the alternating-operator idea used by QAOA.

Instead of one global angle per layer, MAQAOA can use:

- qubit-indexed gammas
- term-indexed gammas
- orbit assignments to share parameters across symmetric structures
- independent beta parameters per qubit or orbit

This makes MAQAOA more expressive, but also more expensive to tune.

## Required Inputs

- A cost Hamiltonian as `lindblad::SparsePauliOp`
- Optionally, a mixer Hamiltonian as `lindblad::SparsePauliOp`
- `MAQAOA::Options` settings
- `Estimator` and `Sampler` configuration

Important options include:

- `p`: number of layers
- `max_iterations`
- `convergence_threshold`
- `optimizer`
- `layerwise`
- `progressive`
- `seed`
- `orbit_assignments`
- `term_indexed_gammas`
- `mixer_weights`
- `initial_thetas`
- `beta_base`
- `lambda_co2`

## How to Invoke

Include the public header:

```cpp
#include "lindblad/algorithms.hpp"
```

Basic optimization:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

SparsePauliOp cost({
    PauliString("ZZ", Complex128(1.0, 0.0))
});

MAQAOA maqaoa;
maqaoa.options.p = 2;
maqaoa.options.layerwise = true;
maqaoa.estimator.options.shots = 0;
maqaoa.sampler.options.shots = 1024;

auto result = maqaoa.optimize(cost);
```

To build a circuit directly:

```cpp
auto circuit = maqaoa.build_circuit(cost, {}, result.optimal_params);
```

To inspect the parameter count for a Hamiltonian:

```cpp
int n_params = maqaoa.num_parameters(cost);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator and Primitive Dependencies

MAQAOA uses the same primitive interfaces as QAOA:

- `Estimator` for objective evaluation
- `Sampler` for final bitstring statistics

The implementation supports exact or sampled execution depending on primitive settings and the noise model assigned to the primitives.

## Public API Details

### `MAQAOA::Options`

The most important fields are:

- `p`: number of layers
- `layerwise`: enable layer-by-layer optimization
- `progressive`: keep earlier parameters free during layerwise refinement
- `orbit_assignments`: symmetry-reduction map for orbit-based parameter sharing; `orbits_by_power` can generate these indices from power tiers
- `term_indexed_gammas`: choose between qubit-indexed and term-indexed gamma layout
- `mixer_weights`: physics-informed mixer scaling
- `initial_thetas`: per-qubit initial state preparation angles
- `beta_base`: baseline beta scale for PI-MA-QAOA
- `lambda_co2`: scalar objective weighting term
- `optimizer` is currently not wired; the implementation always uses COBYLA

### `MAQAOA::Result`

- `optimal_value` is the best energy found
- `optimal_params` stores the final parameter vector
- `initial_params` stores the initial guess used by the optimizer
- `counts` stores sampled outcome frequencies
- `best_bitstring` stores the best measured string
- `num_iterations` counts total evaluations across all layers
- `converged` indicates optimization success
- `per_layer_costs` stores best energy per layer
- `layer_nfev` stores evaluations per layer
- `wall_time_by_layer` stores elapsed wall time per layer
- `wall_time_seconds` stores the total runtime

### `MAQAOA::build_circuit`

- Builds the circuit for a supplied Hamiltonian and parameter vector
- Supports the richer parameterization described above
- Returns a `QuantumCircuit`

### `MAQAOA::optimize`

- Runs the full optimizer
- Can operate layerwise or jointly
- Returns the final parameter vector and sampled outputs
- `best_bitstring` is chosen by minimum computational-basis cost (tie-break by count)

### `MAQAOA::num_parameters`

- Computes the total parameter count for the current Hamiltonian and option set
- The result depends on layer count, gamma layout, orbit sharing, and mixer structure

## Example Code

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    SparsePauliOp cost({
        PauliString("ZI", Complex128(0.5, 0.0)),
        PauliString("IZ", Complex128(0.5, 0.0)),
        PauliString("ZZ", Complex128(1.0, 0.0))
    });

    MAQAOA maqaoa;
    maqaoa.options.p = 1;
    maqaoa.options.layerwise = true;
    maqaoa.options.term_indexed_gammas = true;
    maqaoa.estimator.options.shots = 0;
    maqaoa.sampler.options.shots = 1024;

    auto result = maqaoa.optimize(cost);
    std::cout << "best bitstring: " << result.best_bitstring << '\n';
}
```

## Return Values and Outputs

- `optimal_value` is the best objective value found
- `optimal_params` is the optimized MAQAOA parameter vector
- `counts` is the sampled measurement distribution
- `best_bitstring` is the most useful candidate solution from the final samples
- `per_layer_costs`, `layer_nfev`, and wall-time fields are intended for benchmarking and research analysis

## Exceptions and Failure Modes

Common issues include:

- inconsistent parameter count for the chosen layout
- orbit assignments that do not match the Hamiltonian structure
- `term_indexed_gammas` enabled with a parameter vector built for qubit-indexed gamma layout
- missing or unsupported optimizer selection

## Common Pitfalls

- MAQAOA is not just QAOA with a different name; its parameter shape can change substantially.
- Layerwise optimization changes how the optimizer state evolves across layers.
- `term_indexed_gammas` should be documented carefully in examples because it changes the expected parameter count.
- If you set orbit assignments, the parameter count can drop significantly.

## Testing Notes

Relevant tests live in:

- [tests/test_maqaoa.cpp](../../tests/test_maqaoa.cpp)
- [tests/test_maqaoa_5qubit.cpp](../../tests/test_maqaoa_5qubit.cpp)
- [tests/test_maqaoa_microgrid.cpp](../../tests/test_maqaoa_microgrid.cpp)
- [tests/test_maqaoa_noisy.cpp](../../tests/test_maqaoa_noisy.cpp)
- [tests/test_maqaoa_20qubit.cpp](../../tests/test_maqaoa_20qubit.cpp)

## Related Source Files

- [docs/api/maqaoa.md](../api/maqaoa.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/maqaoa.cpp](../../src/algorithms/maqaoa.cpp)
- [tests/test_maqaoa.cpp](../../tests/test_maqaoa.cpp)
