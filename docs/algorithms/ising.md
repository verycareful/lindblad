# Ising Hamiltonian

This page documents `lindblad::IsingHamiltonian`.

## Purpose

`IsingHamiltonian` is the native C++ representation of an Ising cost Hamiltonian.

It is designed to make QUBO-to-Ising conversion and energy evaluation easy without needing an external preprocessing layer.

The model represents:

$$H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{offset}$$

## Theory Summary

QUBO problems are usually written over binary variables `x_i ∈ {0,1}`.

The Ising form uses spins `s_i ∈ {-1,+1}` and the standard substitution:

$$x_i = \frac{1 - s_i}{2}$$

lindblad stores the linear terms `h`, the quadratic couplings `J`, and a constant offset so the Hamiltonian can be converted to `SparsePauliOp` or evaluated directly.

## Required Inputs

- Either a QUBO matrix `Q`
- Or explicit `h` and `J` coefficients
- Optional constant offset

## How to Invoke

Include the header:

```cpp
#include "lindblad/ising.hpp"
```

Create an Ising model from a QUBO matrix:

```cpp
using namespace lindblad;

std::vector<std::vector<double>> Q = {
    {1.0, 2.0},
    {0.0, 3.0}
};

auto ham = IsingHamiltonian::from_qubo(Q);
```

Evaluate a bitstring directly:

```cpp
double energy = ham.evaluate("10");
```

Convert to a sparse Pauli operator for use with `Estimator`, `VQE`, or `QAOA`:

```cpp
SparsePauliOp op = ham.to_sparse_pauli_op();
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/ising.hpp"
```

## Simulator and Primitive Dependencies

`IsingHamiltonian` is not a simulator itself. It is an intermediate problem representation used by algorithms and primitives.

Typical consumers include:

- `QAOA`
- `MAQAOA`
- `VQE`
- `Estimator`
- `SoftDispatchResult`

## Public API Details

### Fields

- `h` stores linear Z terms
- `J` stores quadratic ZZ couplings
- `offset` stores the constant shift

### `n_qubits()`

- Returns the number of qubits implied by the model

### `to_sparse_pauli_op()`

- Converts the Ising model into a `SparsePauliOp`
- Uses the project’s MSB-first Pauli string convention

### `evaluate(const std::string& bitstring)`

- Evaluates the energy for a binary assignment
- Expects an MSB-first bitstring

### `evaluate_spins(const std::vector<int>& spins)`

- Evaluates the energy for a spin assignment in `{-1,+1}` form

### `from_qubo(const std::vector<std::vector<double>>& Q, double penalty_A = 1.0)`

- Builds an Ising Hamiltonian from a QUBO matrix
- `penalty_A` rescales the conversion if needed

### `from_hJ(const std::vector<double>& h, const std::vector<std::vector<double>>& J, double offset = 0.0)`

- Builds an Ising Hamiltonian directly from coefficients

## Example Code

```cpp
#include "lindblad/ising.hpp"

using namespace lindblad;

int main() {
    std::vector<std::vector<double>> Q = {
        {0.0, 1.5, 0.0},
        {0.0, 0.0, 2.0},
        {0.0, 0.0, 0.0}
    };

    auto ham = IsingHamiltonian::from_qubo(Q);
    auto op = ham.to_sparse_pauli_op();
    std::cout << "n_qubits: " << ham.n_qubits() << '\n';
}
```

## Return Values and Outputs

- `from_qubo` returns a fully populated `IsingHamiltonian`
- `evaluate` and `evaluate_spins` return the scalar energy for the provided assignment
- `to_sparse_pauli_op` returns a Hamiltonian suitable for quantum primitives

## Exceptions and Failure Modes

Common issues include:

- a non-square or malformed QUBO matrix
- inconsistent vector and matrix dimensions when building directly from `h` and `J`
- bitstrings whose length does not match the Hamiltonian qubit count

## Common Pitfalls

- The bitstring convention is MSB-first.
- The `J` matrix is treated as upper triangular in the normal construction path.
- `offset` matters when you compare Ising energies to the original QUBO objective.

## Testing Notes

The Ising machinery is exercised indirectly through the MAQAOA and microgrid tests, especially:

- [tests/test_maqaoa_microgrid.cpp](../../tests/test_maqaoa_microgrid.cpp)

## Related Source Files

- [docs/api/ising.md](../api/ising.md)
- [include/lindblad/ising.hpp](../../include/lindblad/ising.hpp)
- [src/algorithms/ising.cpp](../../src/algorithms/ising.cpp)
- [tests/test_maqaoa_microgrid.cpp](../../tests/test_maqaoa_microgrid.cpp)
