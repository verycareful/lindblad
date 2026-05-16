# Grover

This page documents `lindblad::algorithms::Grover`.

## Purpose

Grover's algorithm amplifies the probability of marked states in an unstructured search space.

In lindblad, the Grover helper builds a circuit around a provided oracle and either uses an explicit iteration count or computes the usual near-optimal default.

## Theory Summary

Grover search alternates between:

1. applying an oracle that marks the desired state
2. applying the diffusion operator
3. repeating for a chosen number of iterations

The diffusion operator in lindblad is implemented with the standard Hadamard/X pattern and a multi-controlled phase flip on the all-ones state.

## Required Inputs

- A `QuantumCircuit` oracle acting on the search register
- An optional iteration count
- Optional shot count and seed when calling `search`

The oracle should mark the solution state by phase inversion or equivalent amplitude-marking behavior.

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Construct or reuse an oracle and run Grover search:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

QuantumCircuit oracle(3);
// oracle construction goes here

auto circuit = Grover::build_circuit(oracle);
auto result = Grover::search(oracle, -1, 1024, 42);
```

If you want to override the iteration count explicitly:

```cpp
auto circuit = Grover::build_circuit(oracle, 2);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator Dependencies

Grover uses `StatevectorSimulator` internally for exact circuit execution and sampled counts.

The implementation constructs the diffusion operator directly in the circuit and measures the final state after all iterations.

## Public API Details

### `Grover::build_circuit`

- Takes the oracle circuit and an optional iteration count
- If the iteration count is negative, uses `round(π/4 · √(2^n))` with a minimum
  of 1; this formula assumes **one marked item**
- Returns a `QuantumCircuit` containing the full Grover iterate sequence

### `Grover::Result`

- `solution` stores the most frequent measured bitstring
- `num_iterations` stores the iteration count used by the solver
- `probability` stores the measured frequency of the chosen solution bitstring

### `Grover::search`

- Resolves `num_iterations` before calling `build_circuit` so that
  `result.num_iterations` always matches the circuit that was run
- Builds the Grover circuit, applies `measure_all`, and runs statevector simulation
- Returns the most likely observed solution and its empirical probability

## Example Code

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit oracle(2);
    oracle.cz(0, 1);

    auto result = Grover::search(oracle, -1, 1024, 7);
    std::cout << "solution: " << result.solution << '\n';
}
```

## Return Values and Outputs

- `build_circuit` returns a Grover search circuit
- `search` returns a `Grover::Result` with the best observed bitstring and its measured probability

## Exceptions and Failure Modes

Common issues include:

- an oracle that does not mark the intended solution state
- an iteration count that is too small or too large for the target search space
- a search register size that does not match the oracle structure

## Common Pitfalls

- Grover returns the most frequently observed bitstring, not a guaranteed exact answer for every shot.
- The default iteration heuristic is only an estimate.
- The oracle must be built so that the marked state is the one you expect the algorithm to amplify.

## Testing Notes

Grover behavior is exercised indirectly through the simulator and algorithm coverage already present in the repository.

## QuditGrover

**Purpose:** Search a d^n-element space for marked state(s) in O(√(d^n)) oracle queries.

**d-ary generalization:** Extends Grover's algorithm from binary (d=2) to d-dimensional quantum systems. Works for any d ≥ 2. When d=2, identical to standard Grover.

**Circuit per Grover iteration** (n qudits, dimension d):

| Step | Gate | Operation |
|------|------|-----------|
| 1 | Oracle | apply_phase_oracle: −1 for marked states, +1 otherwise |
| 2 | F_d†^n | IQFT on all qudits |
| 3 | R_0 | apply_phase_oracle: −1 for all non-\|0...0⟩ states |
| 4 | F_d^n | QFT on all qudits (completes diffusion) |

Initial state: F_d^n\|0...0⟩ = uniform superposition (1/√(d^n)) Σ_x \|x⟩.

**Optimal iterations:** R ≈ round(π/4 · √(d^n)) (assumes 1 marked item). Pass `num_iterations` explicitly when the number of marked items ≠ 1.

**Quantum advantage:** O(√(d^n)) vs O(d^n) classical.

**Required Inputs:**
- `n` — number of qudits (≥ 1)
- `d` — qudit dimension (≥ 2)
- `target` — explicit target state (for `search`), or `is_marked` predicate (for `search_with_oracle`)
- `shots` — number of independent circuit executions (default 100)
- `backend` (optional): `QuditBackend` — simulator to use (default `STATEVECTOR`); `CLIFFORD` throws
- `noise` (optional): `const QuditNoiseModel*` — noise model; only applied with `DENSITY_MATRIX` (default `nullptr`)

**How to Invoke:**
```cpp
#include "lindblad/algorithms.hpp"
using namespace lindblad::algorithms;

// Single target (explicit)
auto r = QuditGrover::search(/*n=*/2, /*d=*/3, {1, 2}, /*shots=*/200);
// r.solution == {1, 2},  r.probability > 0.5

// Arbitrary predicate
auto r2 = QuditGrover::search_with_oracle(2, 3,
    [](const std::vector<int>& x) { return x[0] == 1 && x[1] == 2; },
    /*num_iterations=*/-1, 200);
```

**Supported d:** Any d ≥ 2. For d=2 reproduces qubit Grover exactly.

**Result:**
```cpp
struct Result {
    std::vector<int> solution;  // most probable marked state across shots
    double probability;          // fraction of shots returning solution
    int num_iterations;
    int d;
    int n;
};
```

**Exceptions:**
- `std::invalid_argument` if `d < 2`, `n < 1`, `target.size() != n`, or any `target[i]` outside `[0, d)`
- `std::invalid_argument` if `backend == QuditBackend::CLIFFORD` (Grover is not Clifford-simulable)

**Backend:**

| Backend | Supported | Notes |
|---|---|---|
| `STATEVECTOR` | ✓ | Default. Exact dense statevector; each shot creates a fresh state vector. |
| `DENSITY_MATRIX` | ✓ | Full mixed-state; applies `noise` model if provided. |
| `MPS` | ✓ | Tensor-network. |
| `CLIFFORD` | ✗ | Throws `std::invalid_argument`. Grover's diffusion operator is not Clifford-simulable. |

The `noise` argument is applied only in the `DENSITY_MATRIX` path. See [docs/api/qudit-simulators.md](../api/qudit-simulators.md) for the full backend API reference.

## Related Source Files

- [docs/api/grover.md](../api/grover.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/grover.cpp](../../src/algorithms/grover.cpp)
- [tests/test_simulators.cpp](../../tests/test_simulators.cpp)
