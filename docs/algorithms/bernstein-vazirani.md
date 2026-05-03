# Bernstein-Vazirani Family

This page covers the Bernstein-Vazirani family of algorithms in q++:

- `BernsteinVazirani`
- `RecursiveBernsteinVazirani`
- `ProbabilisticBernsteinVazirani`

All three live in the `qpp::algorithms` namespace and are declared in [include/qpp/algorithms.hpp](../../include/qpp/algorithms.hpp).

## Purpose

Bernstein-Vazirani recovers a hidden bit string `s` from an oracle implementing $f(x) = s \cdot x \pmod 2$ using a single quantum query.

The recursive and probabilistic variants extend the same core idea:

- `RecursiveBernsteinVazirani` runs BV independently across multiple depth levels, each with its own oracle.
- `ProbabilisticBernsteinVazirani` samples from a pool of BV oracles and records the recovered keys.

## Theory Summary

The standard BV circuit uses phase kickback:

1. Prepare the query register in a uniform superposition.
2. Prepare the ancilla in $|1\rangle$ and apply Hadamard gates.
3. Apply the oracle once.
4. Apply Hadamards to the query register.
5. Measure the query register.

The measured bitstring is the hidden secret string `s`.

For the recursive variant, each oracle encodes an independent secret. For the probabilistic variant, the oracle pool represents a set of candidate secrets drawn by a discrete distribution.

## Required Inputs

### Standard BV

- A `QuantumCircuit` oracle with `n + 1` qubits
- The first `n` qubits are the query register
- The last qubit is the ancilla

### Recursive BV

- A vector of `QuantumCircuit` objects
- One standard BV oracle per depth level
- The same `n` for each level

### Probabilistic BV

- A vector of `QuantumCircuit` objects
- Each circuit must encode one secret key using the standard BV oracle pattern
- Optional weight vector for sampling the oracle pool

## How to Invoke

### Standard BV

Include the header:

```cpp
#include "qpp/algorithms.hpp"
```

Build the oracle so that each `cx(i, n)` toggles the ancilla when bit `i` of the secret is `1`.

```cpp
using namespace qpp;
using namespace qpp::algorithms;

QuantumCircuit oracle(4); // 3 query qubits + 1 ancilla
oracle.cx(0, 3);
oracle.cx(2, 3);

auto result = BernsteinVazirani::solve(oracle, 3);
```

### Recursive BV

```cpp
std::vector<QuantumCircuit> oracles = {oracle0, oracle1, oracle2};
auto result = RecursiveBernsteinVazirani::solve(oracles, 3, 1, 42);
```

### Probabilistic BV

```cpp
std::vector<QuantumCircuit> pool = {oracle0, oracle1, oracle2};
std::vector<double> weights = {0.2, 0.3, 0.5};
auto result = ProbabilisticBernsteinVazirani::solve(pool, 3, weights, 50, 1234);
```

## Header Include Instructions

Use:

```cpp
#include "qpp/algorithms.hpp"
```

That header brings in the public BV family declarations and their result types.

## Simulator Dependencies

The BV implementations use `StatevectorSimulator` internally for exact evaluation.

That means:

- no noise model is required for the standard usage shown here
- the circuit is measured after the BV pattern is executed
- the returned bitstring comes from the most frequent sampled measurement result

## Public API Details

### `BernsteinVazirani`

Declared in [include/qpp/algorithms.hpp](../../include/qpp/algorithms.hpp).

- `build_circuit(const QuantumCircuit& oracle, int n)` constructs the BV query circuit
- `solve(const QuantumCircuit& oracle, int n, int shots = 1, uint64_t seed = 0)` executes the circuit and returns the recovered secret
- `Result::secret` stores the recovered bitstring

The recovered secret is returned in the same bit order used by the tests and implementation: the measured bitstring is converted back into index order before returning.

### `RecursiveBernsteinVazirani`

- `solve(const std::vector<QuantumCircuit>& oracles, int n, int shots = 1, uint64_t seed = 0)` runs the BV solve step independently for each oracle
- `Result::secrets` stores one recovered secret per depth level
- `Result::depth` records how many oracle levels were processed
- `Result::total_oracle_calls` reports the number of quantum oracle invocations used by the solver

### `ProbabilisticBernsteinVazirani`

- `solve(const std::vector<QuantumCircuit>& oracle_pool, int n, const std::vector<double>& weights = {}, int shots = 50, uint64_t seed = 0)` samples oracles from the pool and records the keys returned by BV
- `Result::discovered_keys` contains the unique recovered keys in sorted order
- `Result::key_counts` stores the number of times each key was observed
- `Result::shots_used` records the total number of shots requested

## Example Code

```cpp
#include "qpp/algorithms.hpp"

using namespace qpp;
using namespace qpp::algorithms;

QuantumCircuit bv_oracle(const std::string& secret) {
    QuantumCircuit qc(secret.size() + 1);
    for (int i = 0; i < static_cast<int>(secret.size()); ++i) {
        if (secret[i] == '1') {
            qc.cx(i, static_cast<int>(secret.size()));
        }
    }
    return qc;
}

int main() {
    auto oracle = bv_oracle("101");
    auto result = BernsteinVazirani::solve(oracle, 3);
    std::cout << result.secret << '\n';
}
```

## Return Values and Outputs

- `BernsteinVazirani::Result::secret` is the recovered hidden string
- `RecursiveBernsteinVazirani::Result::secrets` is the recovered string for each recursive level
- `ProbabilisticBernsteinVazirani::Result::discovered_keys` is the sorted set of unique keys observed across shots
- `ProbabilisticBernsteinVazirani::Result::key_counts` is a frequency map keyed by recovered bitstring

## Exceptions and Failure Modes

The BV solvers expect oracles that match the documented register layout.

Common failure cases:

- The oracle does not have `n + 1` qubits
- The ancilla is not the last qubit
- The oracle does not encode a linear parity function in the standard BV form
- The oracle pool is empty for probabilistic BV
- The weight vector length does not match the oracle pool length

When the oracle layout is wrong, the algorithm may return an incorrect secret rather than throwing.

## Common Pitfalls

- The returned string is not just the raw measurement result; the implementation converts the bitstring back into index order.
- Oracle qubit ordering matters. The ancilla must be the final qubit.
- Recursive and probabilistic BV are variants of the BV family, not separate unrelated algorithms.
- The probabilistic solver sorts the discovered keys before returning them.

## Testing Notes

The BV family is covered by [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp).

The tests cover:

- constant and balanced Deutsch-Jozsa style oracle behavior for the same oracle pattern
- BV secret recovery for multiple secrets
- recursive BV across multiple depths
- probabilistic BV discovery, counts, and ordering

## Related Source Files

- [docs/api/bernstein-vazirani.md](../api/bernstein-vazirani.md)
- [include/qpp/algorithms.hpp](../../include/qpp/algorithms.hpp)
- [src/algorithms/bernstein_vazirani.cpp](../../src/algorithms/bernstein_vazirani.cpp)
- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)
- [docs/MasterDocumentation.md](../MasterDocumentation.md)
