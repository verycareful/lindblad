# Sampler API Deep Dive

This page documents the public `qpp::Sampler` primitive.

## Header and Namespace

- Header: `include/qpp/primitives.hpp`
- Namespace: `qpp`

## Class Overview

`Sampler` executes circuits and returns sampled bitstring counts. It selects
statevector simulation for ideal noise and density-matrix simulation when a
noise model is provided.

## `Options`

Fields and defaults:

- `shots = 1024`: number of samples to draw per circuit
- `seed = 0`: seed passed to the simulator
- `noise_model`: when non-ideal, enables density-matrix sampling

## `run`

Signature:

```cpp
std::vector<std::unordered_map<std::string, int>> run(
    const std::vector<QuantumCircuit>& circuits,
    const std::vector<std::vector<double>>& parameter_values = {}
);
```

Behavior:

- Evaluates each circuit with `run_single`
- If `parameter_values` has fewer entries than `circuits`, missing entries are
  treated as empty parameter lists
- Returns one counts map per circuit

## `run_single`

Signature:

```cpp
std::unordered_map<std::string, int> run_single(
    const QuantumCircuit& circuit,
    const std::vector<double>& parameters = {}
);
```

Behavior (verified against the implementation):

- Binds parameters by name using `circuit.parameter_names`
- Binds up to `min(parameters.size(), circuit.parameter_names.size())` parameters
- If `options.noise_model` is non-ideal, uses `DensityMatrixSimulator`
- If `options.noise_model` is ideal, uses `StatevectorSimulator`
- Passes `options.shots` and `options.seed` into the simulator run
- Throws `std::runtime_error` if simulation fails

## Example

```cpp
#include "qpp/primitives.hpp"

using namespace qpp;

int main() {
    QuantumCircuit circuit(2, 2);
    circuit.h(0).cx(0, 1).measure(0, 0).measure(1, 1);

    Sampler sampler;
    sampler.options.shots = 256;

    auto counts = sampler.run_single(circuit);
    return counts.empty() ? 1 : 0;
}
```

## Related Pages

- [docs/APIOverview.md](../APIOverview.md)
- [docs/algorithms/qaoa.md](../algorithms/qaoa.md)
- [docs/algorithms/maqaoa.md](../algorithms/maqaoa.md)
