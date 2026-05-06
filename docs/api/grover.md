# Grover API Deep Dive

This page documents the public `lindblad::algorithms::Grover` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Class Overview

`Grover` builds a Grover search circuit from an oracle and optionally chooses an
iteration count. The `search` helper runs the circuit with a
`StatevectorSimulator` and returns the most frequent measured bitstring.

## `build_circuit`

Signature:

```cpp
static QuantumCircuit build_circuit(
        const QuantumCircuit& oracle,
        int num_iterations = -1
);
```

Behavior (verified against `src/algorithms/grover.cpp`):

- `num_iterations < 0` uses `round(π/4 · √(2^n))` with a minimum of 1; this
  formula assumes **one marked item**
- Starts with Hadamards on all qubits
- Each iteration:
    - appends the oracle instructions
    - applies the diffusion operator via H/X, then a multi-controlled X on the
        last qubit wrapped by `H` on the last qubit, then uncomputes with X/H
- For `n == 2` uses `cx`, for `n == 3` uses `ccx`, and for `n > 3` uses a dense
    `UNITARY` matrix for the multi-controlled X

## `Result`

Fields:

- `solution`: most frequent measured bitstring
- `num_iterations`: iteration count used by the solver
- `probability`: frequency of the `solution` bitstring

## `search`

Signature:

```cpp
static Result search(
        const QuantumCircuit& oracle,
        int num_iterations = -1,
        int shots = 1024,
        uint64_t seed = 0
);
```

Behavior:

- Resolves `num_iterations` (if negative) **before** calling `build_circuit`, so
  `result.num_iterations` is always consistent with the circuit that was executed
- Builds the circuit, applies `measure_all`, and runs `StatevectorSimulator::run`
- Chooses the bitstring with the highest count
- Computes `probability` as `max_count / shots`

**Note**: The default iteration formula `round(π/4 · √(2^n))` assumes **one marked item**.
For $k > 1$ solutions pass `num_iterations` explicitly:
$\lceil \frac{\pi}{4} \sqrt{\frac{2^n}{k}} \rceil$

## Example

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
        QuantumCircuit oracle(2);
        oracle.cz(0, 1);

        auto result = Grover::search(oracle, -1, 1024, 7);
        return result.solution.empty() ? 1 : 0;
}
```

## Notes and Preconditions

- The oracle is assumed to act on the full search register
- The diffusion operator uses a dense `UNITARY` for the multi-controlled X when
    `n > 3`

## Related Pages

- [docs/algorithms/grover.md](../algorithms/grover.md)
- [docs/APIOverview.md](../APIOverview.md)
