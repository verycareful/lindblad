# Operators API Deep Dive

This page documents the public operator APIs in `lindblad/operators.hpp`, including
`PauliString`, `SparsePauliOp`, `Operator`, and `lindblad::QuantumInfo` helpers.

## Header and Namespace

- Header: `include/lindblad/operators.hpp`
- Namespace: `lindblad`
- Quantum information helpers live under `lindblad::QuantumInfo`

## `PauliString`

Represents a tensor product of single-qubit Paulis with a complex coefficient.

Ordering (project LSB-first convention, frozen in R.1.12; see
`docs/Architecture.md`, Conventions):

- `pauli[q]` acts on qubit q: `pauli[0]` is qubit 0 (least significant), `pauli[n-1]` is qubit n-1
- Example: `"XIZ"` means X on qubit 0, I on qubit 1, Z on qubit 2
- This is a deliberate deviation from Qiskit's label order (which puts qubit n-1 first)
- Footgun: Pauli strings read in the OPPOSITE direction from measurement bitstrings, whose rightmost character is qubit 0; `"XI"` (X on qubit 0) marks the basis state counted under the key `"01"`
- `tensor(other)`: `this` occupies the low qubits of the result, `other` the high qubits

Key API:

- `compose(other)`: multiplies two strings (same length required); throws on
  length mismatch
- `adjoint()`: conjugates the coefficient
- `commutes_with(other)`: counts anticommuting positions; even count means
  they commute

## `SparsePauliOp`

Represents a sum of Pauli strings.

Key API:

- `simplify(atol)`: merges identical Pauli labels and drops tiny coefficients
  (`coeff.norm_sq() > atol * atol`)
- `compose(other)`: pairwise composition followed by `simplify`
- `adjoint()`: adjoint of each term
- `tensor(other)`: concatenates Pauli labels and multiplies coefficients
- `operator+`: concatenates terms then `simplify`
- `operator*`: scales coefficients
- `to_matrix()`: builds the dense $2^n \times 2^n$ matrix; uses the action
  $P|j\rangle = i^{\#Y} \cdot (-1)^{\text{popcount}(j \wedge z\_\text{mask})}
  \cdot |j \oplus x\_\text{mask}\rangle$ to fill each column in $O(2^n)$ per
  term (not $O(4^n)$ tensor-product chains); the $i^{\#Y}$ factor is a
  per-term constant folded into the coefficient ($Y = iXZ$), so Hermitian
  operators yield Hermitian matrices and `to_matrix` agrees with
  `expectation_value` and composes homomorphically
- `expectation_value(statevector)`: computes ⟨psi|H|psi⟩ without cloning
- `expectation_value_batch(states)`: batch version with shared mask precompute
- `n_qubits()`: number of qubits in the first term (0 if empty)
- `from_list(label_coeff)` / `identity(n)` / `zero(n)` helpers

Notes:

- `n_qubits()` does not validate that all terms have equal length
- `expectation_value_batch` expects pointers to `Statevector` instances

## `Operator`

Represents a dense operator as a flat complex matrix.

Key API:

- `from_circuit(circuit)`: simulates each basis column with
  `StatevectorSimulator::apply_instruction`
- `from_pauli(op)`: uses `SparsePauliOp::to_matrix`
- `compose(other)`: matrix multiplication (throws on mismatched qubit counts)
- `tensor(other)`: Kronecker product
- `adjoint()`: conjugate transpose
- `power(n)`: repeated composition (`n == 0` returns identity)
- `is_unitary(atol)` / `is_hermitian(atol)`
- `trace()`

## `lindblad::QuantumInfo`

Quantum information metrics and helpers:

- `state_fidelity(statevector)`: $|<psi_1|psi_2>|^2$
- `state_fidelity(density_matrix)`: Uhlmann-Jozsa fidelity
- `process_fidelity`: squared Hilbert-Schmidt overlap
- `average_gate_fidelity`: derived from process fidelity
- `entropy(rho, base)`: von Neumann entropy via eigendecomposition
- `entanglement_entropy(statevector, subsystem)`: entropy of reduced state
- `concurrence(rho)`: Wootters concurrence for 2-qubit density matrices
- `partial_trace(rho, trace_out_qubits)` / `partial_trace(statevector, trace_out_qubits)`
- `pauli_expectation_values(statevector, paulis)`

Important behaviors (from the implementation):

- `process_fidelity` returns the squared quantity; take sqrt for the unsquared
  fidelity
- `concurrence` throws if `rho.n_qubits != 2`
- `partial_trace` takes the set of qubits to **trace out**; the complement defines the output subsystem

## Example

```cpp
#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"

using namespace lindblad;

int main() {
    PauliString p("ZZ", Complex128(1.0, 0.0));
    SparsePauliOp op({p});

    Statevector sv(2);
    double ev = op.expectation_value(sv);
    return (ev > 0.9) ? 0 : 1;
}
```

## Testing Notes

Relevant tests live in:

- [tests/test_operators.cpp](../../tests/test_operators.cpp)

## Related Pages

- [docs/APIOverview.md](../APIOverview.md)
