# IsingHamiltonian API Deep Dive

This page documents the public `qpp::IsingHamiltonian` API.

## Header and Namespace

- Header: `include/qpp/ising.hpp`
- Namespace: `qpp`

## Type Overview

`IsingHamiltonian` stores the coefficients of an Ising cost Hamiltonian:

$$H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{offset}$$

It supports three main workflows:

- converting a QUBO matrix into Ising coefficients
- evaluating energy for a binary or spin assignment
- exporting to `SparsePauliOp` for use with primitives and algorithms

## Fields

- `h`: linear coefficients, size `n` (one per qubit)
- `J`: quadratic couplings, size `n x n` (only `i < j` entries are used)
- `offset`: constant energy shift

## Construction

### `from_qubo`

Signature:

```cpp
static IsingHamiltonian from_qubo(
    const std::vector<std::vector<double>>& Q,
    double penalty_A = 1.0
);
```

Requirements:

- `Q` must be a square matrix
- `penalty_A` rescales all coefficients before conversion

### `from_hJ`

Signature:

```cpp
static IsingHamiltonian from_hJ(
    const std::vector<double>& h,
    const std::vector<std::vector<double>>& J,
    double offset = 0.0
);
```

Requirements:

- `J.size()` must match `h.size()`
- only the upper triangle (`i < j`) is used by evaluation and export

## Methods

### `n_qubits()`

Returns the number of qubits implied by `h.size()`.

### `to_sparse_pauli_op()`

- Returns a `SparsePauliOp` containing the Ising Hamiltonian terms
- Includes the constant offset as an identity term when `offset != 0.0`
- Uses the MSB-first Pauli string convention: qubit `i` maps to position `n - 1 - i`

### `evaluate(const std::string& bitstring)`

- `bitstring` must be length `n`, MSB-first
- Uses the spin mapping $s_i = 1 - 2 x_i$ where $x_i \in \{0,1\}$

### `evaluate_spins(const std::vector<int>& spins)`

- `spins` must be length `n`
- Each entry should be `-1` or `+1`

## Exceptions and Preconditions

- `from_qubo` throws `std::invalid_argument` if `Q` is not square
- `from_hJ` throws `std::invalid_argument` if `J` does not match `h` size
- `evaluate` throws `std::invalid_argument` if the bitstring length does not match `n_qubits()`

## Example

```cpp
#include "qpp/ising.hpp"

using namespace qpp;

int main() {
    std::vector<std::vector<double>> Q = {
        {1.0, 2.0},
        {0.0, 3.0}
    };

    auto ham = IsingHamiltonian::from_qubo(Q);
    double energy = ham.evaluate("10");
    auto op = ham.to_sparse_pauli_op();
    (void)op;
    return 0;
}
```

## Related Pages

- [docs/algorithms/ising.md](../algorithms/ising.md)
- [docs/APIOverview.md](../APIOverview.md)
