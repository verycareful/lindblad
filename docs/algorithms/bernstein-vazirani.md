# Bernstein-Vazirani Family

This page covers the Bernstein-Vazirani family of algorithms in lindblad:

- `BernsteinVazirani`
- `RecursiveBernsteinVazirani`
- `ProbabilisticBernsteinVazirani`
- `DistributedBernsteinVazirani`

All four live in the `lindblad::algorithms` namespace and are declared in [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp).

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
#include "lindblad/algorithms.hpp"
```

Build the oracle so that each `cx(i, n)` toggles the ancilla when bit `i` of the secret is `1`.

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

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
#include "lindblad/algorithms.hpp"
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

Declared in [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp).

- `build_circuit(const QuantumCircuit& oracle, int n)` constructs the BV query circuit
- `solve(const QuantumCircuit& oracle, int n, int shots = 1, uint64_t seed = 0)` executes the circuit and returns the recovered secret
- `Result::secret` stores the recovered bitstring

The most frequent bitstring is selected via `std::max_element` with a count comparator. Per-shot execution records only the `n` query qubits into the `n`-bit classical register, so bitstrings already have length `n` (the ancilla is never written to the classical register). The `n`-bit MSB-first string is reversed so that `secret[i] == '1'` means bit `i` of the hidden string is set (index order).

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
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

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
- For `DistributedBernsteinVazirani`, each party's local oracle must use qubit index `n_bits` for its ancilla — the last qubit in the local circuit. The combined circuit remaps it to the shared global ancilla automatically.

## Testing Notes

The BV family is covered by [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp).

The tests cover:

- constant and balanced Deutsch-Jozsa style oracle behavior for the same oracle pattern
- BV secret recovery for multiple secrets
- recursive BV across multiple depths
- probabilistic BV discovery, counts, and ordering
- distributed BV tests are planned for R.1.4.1 (test-suite release)

## DistributedBernsteinVazirani

### Purpose

`DistributedBernsteinVazirani` (DBVA) solves the BV problem when the secret is split across
multiple independent parties. Each party holds a local oracle over their portion of the secret.
The quantum protocol recovers the full secret in a single joint circuit execution — one
communication round — whereas a classical approach must query each party separately, requiring
as many rounds as there are parties.

### Theory Summary

The n-bit secret `s` is partitioned as:

```
s = S_{n_0} || S_{n_1} || ... || S_{n_{t-1}},   Σ n_j = n
```

Party `j` holds `n_j` bits and a local BV oracle:

```
f_j(m_j) = ⟨S_{n_j} · m_j⟩ mod 2
```

The combined circuit uses `n + 1` qubits: `n_total = Σ n_j` query qubits laid out as
`[party 0 slice | party 1 slice | ...]`, plus one shared ancilla at index `n_total`.

**Circuit steps:**

1. `X(ancilla)` — prepare ancilla in |1⟩
2. `H` on all qubits — query in |+⟩ⁿ, ancilla in |−⟩
3. For each party j: apply their local oracle, with qubits remapped from local indices
   `[0..n_j-1, n_j]` to global indices `[offset_j..offset_j+n_j-1, n_total]`
4. `H` on query register — decode phase kickback
5. Measure query register — full secret (MSB-first, reversed to index order)

**Complexity:**

| Metric | Quantum | Classical |
|---|---|---|
| Communication rounds | 1 | t (one per party) |
| Query complexity | O(1) | O(n) |
| Circuit depth | 2^max(n_j) + 3 | 2^n + 3 (monolithic) |

### Required Inputs

- A vector of `Party` objects, one per node
- Each `Party::local_oracle` is a standard BV oracle on `(n_bits + 1)` qubits:
  - qubits `0..n_bits-1` = party's query register
  - qubit `n_bits` = ancilla (will be remapped to the shared global ancilla)
- `Party::n_bits` = number of bits this party holds

### How to Invoke

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

// Helper: standard BV oracle for a given secret slice
static QuantumCircuit local_oracle(const std::string& secret) {
    int n = static_cast<int>(secret.size());
    QuantumCircuit qc(n + 1);
    for (int i = 0; i < n; ++i)
        if (secret[i] == '1') qc.cx(i, n);
    return qc;
}

int main() {
    // Two parties: party 0 holds "101", party 1 holds "010"
    std::vector<DistributedBernsteinVazirani::Party> parties = {
        { local_oracle("101"), 3 },
        { local_oracle("010"), 3 },
    };

    auto result = DistributedBernsteinVazirani::solve(parties);
    // result.full_secret    == "101010"
    // result.party_secrets  == {"101", "010"}
    // result.quantum_rounds == 1
    // result.classical_rounds == 2
}
```

### Public API Details

#### `DistributedBernsteinVazirani::Party`

- `local_oracle` — `QuantumCircuit` on `(n_bits + 1)` qubits encoding the party's BV oracle
- `n_bits` — number of secret bits this party holds

#### `DistributedBernsteinVazirani::build_circuit`

```cpp
static QuantumCircuit build_circuit(const std::vector<Party>& parties);
```

Builds and returns the combined `(n_total + 1)`-qubit circuit with all oracles remapped.

#### `DistributedBernsteinVazirani::solve`

```cpp
static Result solve(const std::vector<Party>& parties,
                    int shots = 1, uint64_t seed = 0);
```

Builds the circuit, runs the statevector simulator, and decodes the result.

#### `DistributedBernsteinVazirani::Result`

- `full_secret` — complete n-bit recovered secret (index order)
- `party_secrets` — per-party slices of `full_secret`, length `n_j` each
- `num_parties` — total number of parties t
- `total_bits` — total n = Σ n_j
- `quantum_rounds` — always 1
- `classical_rounds` — equals `num_parties`

---

## Future Work

### Qudit BV (d-dimensional generalisation)

The BV algorithm generalises to d-dimensional quantum systems (qudits) where each
"qudit" has d basis states and the inner product is computed mod d. The quantum
speedup (1 query vs O(n) classical) transfers directly to the qudit setting.

This variant is **not yet implemented** in lindblad. It requires architectural
changes to support d-dimensional gate matrices, d^n-dimensional statevectors, and
qudit-aware circuit and simulator backends throughout the framework. Once qudit
support is added as a core feature, the BV qudit extension will be a straightforward
addition.

Reference: Bernstein-Vazirani generalisation to qudit systems (Springer Quantum Studies, 2023).

---

## Related Source Files

- [docs/api/bernstein-vazirani.md](../api/bernstein-vazirani.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/bernstein_vazirani.cpp](../../src/algorithms/bernstein_vazirani.cpp)
- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)
- [docs/MasterDocumentation.md](../MasterDocumentation.md)
