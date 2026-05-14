# MAQAOA API Deep Dive

This page documents the public `lindblad::algorithms::MAQAOA` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Class Overview

`MAQAOA` generalizes QAOA by assigning multiple angles per layer (per qubit, per term, or per orbit) and supports layerwise or joint optimization. The implementation uses direct statevector evolution when noise is ideal and falls back to density-matrix simulation when noise is configured.

## `Options`

Fields and defaults (from the header):

- `p = 1`: number of layers
- `max_iterations = 200`: optimizer budget
- `convergence_threshold = 1e-6`: relative tolerance
- `optimizer = "COBYLA"`: declared but not wired (implementation uses COBYLA unconditionally)
- `layerwise = false`: enable layer-by-layer optimization
- `progressive = false`: keep earlier layers free when layerwise
- `seed = 0`: RNG seed for parameter initialization
- `orbit_assignments`: optional orbit indices per qubit
- `term_indexed_gammas = false`: use term-indexed gammas when true
- `mixer_weights`: optional PI-MA-QAOA weights
- `initial_thetas`: optional per-qubit `Ry(theta)` initialization
- `beta_base = pi / 4`: base scale for PI-MA-QAOA betas
- `lambda_co2 = 0.0`: stored but not referenced in the current implementation

Orbit behavior:

- Orbit mode is only active if `orbit_assignments.size() == n_qubits`
- Mixer orbit count is `max(orbit_assignments) + 1`

Orbit assignment helper: `orbits_by_power`

Signature:

```cpp
std::vector<int> orbits_by_power(
    const std::vector<double>& powers,
    double tolerance = 0.5
);
```

Behavior (from the implementation):

- Sorts `powers` by value (ascending) before processing, making the result order-independent
- Assigns each sorted power to the first orbit whose center is within `tolerance`
- Creates a new orbit when no existing center matches; the center is the current power
- Returns a vector of orbit indices in input order (same length as `powers`)
- Empty input returns an empty vector

Example:

```cpp
std::vector<double> powers = {1.0, 1.2, 2.8, 2.9};
auto orbits = orbits_by_power(powers, 0.25); // {0, 0, 1, 1}
```

PI-MA-QAOA behavior:

- PI-MA-QAOA beta initialization is active only if `mixer_weights.size() == n_mixer_orbits`
- Betas initialize as `beta_base * (mixer_weights[i] / w_max) + U(-0.05, 0.05)`

## `Result`

Fields:

- `optimal_value`: best energy found
- `optimal_params`: final parameter vector
- `initial_params`: initial parameter vector (concatenated per layer)
- `counts`: sampled measurement counts
- `best_bitstring`: selected bitstring after post-processing
- `num_iterations`: total evaluations across all layers
- `converged`: true if all layers converged and the final value is finite
- `per_layer_costs`: best energy per layer (layerwise only)
- `layer_nfev`: evaluations per layer (layerwise only)
- `wall_time_by_layer`: per-layer wall time (layerwise only)
- `wall_time_seconds`: total wall time

## `num_parameters`

Signature:

```cpp
int num_parameters(const SparsePauliOp& cost_hamiltonian) const;
```

Behavior:

- Orbit mode: `p * (n_cost_orbits + n_mixer_orbits)`
- Term-indexed mode: `p * (n_terms + n_qubits)`
- Default mode: `p * (n_qubits + n_qubits)`

## `optimize`

Signature:

```cpp
Result optimize(
    const SparsePauliOp& cost_hamiltonian,
    const SparsePauliOp& mixer_hamiltonian = {}
);
```

Behavior (verified against `src/algorithms/maqaoa.cpp`):

- If `mixer_hamiltonian` is empty, constructs a default mixer $\sum_i X_i$
- Parameter layout per layer: `[gammas..., betas...]`
- Uses COBYLA with bounds `[-2*pi, 2*pi]` and initial step size `0.3`
- If `estimator.options.noise_model` is non-ideal, evaluates with `DensityMatrixSimulator`
- If ideal, uses direct statevector evolution (`evolve_into`) instead of rebuilding circuits
- Sampling uses `sampler.options.noise_model`; ideal sampling uses `Statevector::sample_counts`
- `best_bitstring` is chosen by minimum computational-basis cost (tie-break by count)
- The cost ranking ignores non-diagonal Pauli terms (X/Y) in the Hamiltonian

Layerwise details:

- Each layer is optimized with its own COBYLA run
- `progressive = true` keeps all prior parameters free
- `progressive = false` freezes earlier layers

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

- Initializes with `H|0>` on each qubit unless `initial_thetas` is provided
- Uses orbit or term-indexed gamma dispatch depending on options
- Applies per-orbit or per-qubit mixer `Rx` rotations
- Kept for API compatibility and offline inspection; hot path uses direct evolution

## Example

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    SparsePauliOp cost({
        PauliString("ZZ", Complex128(1.0, 0.0))
    });

    MAQAOA maqaoa;
    maqaoa.options.p = 2;
    maqaoa.options.layerwise = true;
    maqaoa.sampler.options.shots = 1024;

    auto result = maqaoa.optimize(cost);
    return result.converged ? 0 : 1;
}
```

## Related Pages

- [docs/algorithms/maqaoa.md](../algorithms/maqaoa.md)
- [docs/APIOverview.md](../APIOverview.md)
