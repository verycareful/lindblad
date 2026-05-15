# Bernstein-Vazirani API Deep Dive

This page documents the public Bernstein-Vazirani family APIs in `lindblad::algorithms`.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Family Overview

The family includes five solvers:

- `BernsteinVazirani`
- `RecursiveBernsteinVazirani`
- `ProbabilisticBernsteinVazirani`
- `DistributedBernsteinVazirani`
- `QuditBernsteinVazirani` (d-dimensional generalisation, d ≥ 2)

The first four build circuits over qubits with one shared ancilla and interpret
sampled bitstrings as secrets. `QuditBernsteinVazirani` runs on the qudit layer
(`QuditStatevector` / `QuditGates` / `QuditSimulator` — documented in
[docs/api/qudit.md](qudit.md)) and recovers an integer-valued secret in Z_d^n.

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
- Selects the most-frequent bitstring via `std::max_element` with a count comparator
- Reverses the string to convert MSB-first ordering into index order
- Returns the reversed string as the secret

Bitstring handling detail:

- Per-shot execution records only the `n` query qubits (0..n-1) into the `n`-bit classical register; bitstrings have length `n`
- No `substr` stripping is needed: the ancilla is never written to the classical register
- The `n`-bit MSB-first string is reversed so that `secret[i] == '1'` means bit `i` of the hidden string is set

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

## `DistributedBernsteinVazirani`

### `Party`

```cpp
struct Party {
    QuantumCircuit local_oracle;  // (n_bits + 1) qubits: 0..n_bits-1 query, n_bits ancilla
    int n_bits;                   // number of bits this party holds
};
```

### `build_circuit`

Signature:

```cpp
static QuantumCircuit build_circuit(const std::vector<Party>& parties);
```

Behavior:

- Computes `n_total = Σ party.n_bits`; allocates `(n_total + 1)` qubits, `n_total` classical bits
- Prepares ancilla as |1⟩ then applies H to all qubits
- For each party, remaps oracle qubit indices:
  - local `k < n_bits` → `offset + k` (query slice)
  - local `k == n_bits` → `n_total` (shared ancilla)
- Appends H to the full query register, then measures it

### `solve`

Signature:

```cpp
static Result solve(const std::vector<Party>& parties,
                    int shots = 1, uint64_t seed = 0);
```

Behavior:

- Builds the circuit via `build_circuit`
- Runs `StatevectorSimulator::run`
- Selects the most-frequent bitstring, reverses it to index order
- Slices the full secret into per-party portions matching each `party.n_bits`

### `DistributedBernsteinVazirani::Result`

- `full_secret`: complete n-bit recovered secret (index order)
- `party_secrets`: per-party slices of `full_secret`, length `n_j` each
- `num_parties`: number of parties t
- `total_bits`: total n = Σ n_j
- `quantum_rounds`: always 1
- `classical_rounds`: equals `num_parties`

## `QuditBernsteinVazirani`

The d-dimensional BV variant runs on qudits rather than qubits. The secret is
an integer vector `s ∈ Z_d^n` and the oracle computes `f(x) = s·x mod d`.
Operates on the qudit layer documented in [docs/api/qudit.md](qudit.md).

### `Result`

```cpp
struct Result {
    std::vector<int> secret;  // recovered s, each element in {0..d-1}
    int d;                    // qudit dimension used
    int n;                    // number of query qudits
};
```

### `solve`

Signature:

```cpp
static Result solve(
    const std::vector<int>& secret,
    int d,
    int shots = 1,
    uint64_t seed = 0
);
```

Behavior (verified against `src/algorithms/qudit_bv.cpp`):

- Validates inputs (`d >= 2`, `secret` non-empty, all `secret[i]` in `[0, d)`)
- Constructs a `QuditStatevector` of `(n + 1)` qudits and dimension `d`
- Prepares the ancilla in `|−⟩_d` via `X_d^{d-1}` then `F_d`
- Applies `F_d` to each query qudit
- For every `i` with `secret[i] != 0`, applies the controlled-ADD gate
  `CADD_{s_i}` from query qudit `i` to the ancilla
- Applies `F_d†` to each query qudit
- Calls `QuditSimulator::run`, which measures the full register once
- Accumulates per-position votes across `shots` (only the first `n` qudits;
  the ancilla is ignored)
- Returns the per-position argmax as the recovered secret

For exact statevector simulation, `shots = 1` is sufficient — the protocol is
deterministic. Higher shot counts exist for future noise-aware experiments.

### `oracle_gate`

Signature:

```cpp
static std::vector<Complex128> oracle_gate(int d, int s_i);
```

Returns the d²×d² CADD gate matrix for secret component `s_i`. Equivalent to
`qudit_gates::cadd_matrix(d, s_i)`. Exposed for testing and inspection.

### `QuditBernsteinVazirani::Result`

- `secret`: recovered `s ∈ Z_d^n`, length n
- `d`: qudit dimension used
- `n`: number of query qudits

### Exceptions

`solve()` throws `std::invalid_argument` if:
- `d < 2`
- `secret` is empty
- any `secret[i]` is outside `[0, d)`

## Related Pages

- [docs/algorithms/bernstein-vazirani.md](../algorithms/bernstein-vazirani.md)
- [docs/api/qudit.md](qudit.md) — qudit layer API (state vector, gates, simulator)
- [docs/APIOverview.md](../APIOverview.md)
