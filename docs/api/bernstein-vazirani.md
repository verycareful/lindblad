# Bernstein-Vazirani API Deep Dive

This page documents the public Bernstein-Vazirani family APIs in `lindblad::algorithms`.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Family Overview

The family includes three solvers:

- `BernsteinVazirani`
- `RecursiveBernsteinVazirani`
- `ProbabilisticBernsteinVazirani`

All three build circuits with `n` query qubits and one ancilla, then interpret
sampled outputs to return secrets.

## `BernsteinVazirani`

### `build_circuit`

Signature:

```cpp
static QuantumCircuit build_circuit(const QuantumCircuit& oracle, int n);
```

Behavior (verified against `src/algorithms/bernstein_vazirani.cpp`):

- Allocates `n + 1` qubits and `n` classical bits
- Prepares the ancilla as |1> then applies H to all qubits
- Appends the oracle instructions
- Applies H to the query register
- Measures the query register into classical bits 0..n-1

### `solve`

Signature:

```cpp
static Result solve(
    const QuantumCircuit& oracle,
    int n,
    int shots = 1,
    uint64_t seed = 0
);
```

Behavior:

- Builds the circuit via `build_circuit`
- Runs `StatevectorSimulator::run`
- Takes the first sampled bitstring from the counts map
- Drops the leading ancilla bit, then reverses the remaining bits
- Returns the reversed string as the secret

Bitstring handling detail:

- The implementation assumes the ancilla bit is at index 0 of the measured string
- The query bits are reversed to convert MSB-first ordering into index order

## `RecursiveBernsteinVazirani`

### `solve`

Signature:

```cpp
static Result solve(
    const std::vector<QuantumCircuit>& oracles,
    int n,
    int shots = 1,
    uint64_t seed = 0
);
```

Behavior:

- Executes `BernsteinVazirani::solve` once per oracle
- Increments `seed` by `level` to keep runs independent
- Returns `depth`, `total_oracle_calls`, and a `secrets` vector

## `ProbabilisticBernsteinVazirani`

### `solve`

Signature:

```cpp
static Result solve(
    const std::vector<QuantumCircuit>& oracle_pool,
    int n,
    const std::vector<double>& weights = {},
    int shots = 50,
    uint64_t seed = 0
);
```

Behavior:

- Builds a discrete distribution from `weights` (uniform if empty)
- For each shot:
  - samples an oracle index
  - runs `BernsteinVazirani::solve` with `shots = 1`
  - increments the key count
- Populates `discovered_keys` as sorted unique keys

## Result Types

### `BernsteinVazirani::Result`

- `secret`: recovered key (index-order string)

### `RecursiveBernsteinVazirani::Result`

- `secrets`: recovered keys per depth level
- `depth`: number of oracle levels
- `total_oracle_calls`: total oracle invocations (one per level)

### `ProbabilisticBernsteinVazirani::Result`

- `discovered_keys`: sorted unique keys found
- `key_counts`: map from key to count
- `shots_used`: total number of shots requested

## Example

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

QuantumCircuit bv_oracle(const std::string& secret) {
    QuantumCircuit qc(secret.size() + 1);
    for (int i = 0; i < static_cast<int>(secret.size()); ++i) {
        if (secret[i] == '1') qc.cx(i, static_cast<int>(secret.size()));
    }
    return qc;
}

int main() {
    auto oracle = bv_oracle("101");
    auto result = BernsteinVazirani::solve(oracle, 3);
    return result.secret.empty() ? 1 : 0;
}
```

## Related Pages

- [docs/algorithms/bernstein-vazirani.md](../algorithms/bernstein-vazirani.md)
- [docs/APIOverview.md](../APIOverview.md)
