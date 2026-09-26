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

### Width rule

A Pauli string names one Pauli per qubit and carries no qubit labels, so its
length is the register it acts on. Two things follow, and both are enforced:

- Every term of one operator has the same length. The vector constructor,
  `from_list`, `operator+` and `simplify` refuse terms of different lengths.
- An operator evaluated against a state has exactly the state's qubit count.
  A longer term would name qubits the state does not have, and a shorter one
  would leave qubits it never mentions, so both are refused.
  `expectation_value`, `expectation_value_batch`, `to_matrix`,
  `DensityMatrix::expectation_value_sparse`, `Estimator` and
  `ExpectationObserver` all apply the rule.

`terms` is a public vector, so an operator can still be assembled term by term
into a mixed width. The evaluation checks do not rely on construction having
caught it.

An operator with no terms has no width, and every evaluation refuses it. The
zero operator on n qubits is `zero(n)`, a single all-identity term with
coefficient 0. `simplify` returns that when every term cancels, so `H - H`
(written `H + H * -1.0`) evaluates to 0 rather than being refused. A
default-constructed operator remains the way to build one term by term, and
QAOA and MA-QAOA read a mixer with no terms as "use the default mixer".

### Alphabet and Hermiticity

A Pauli string is written with `I`, `X`, `Y` and `Z`, uppercase. Any other
character, a lowercase letter included, is refused by the `PauliString`
constructor and again at every evaluation, since `pauli` is a public member.
`StabilizerState::expectation_pauli` applies the same rule to the string it is
given.

Every evaluation returns a real number, so it refuses an operator that is not
Hermitian. A sum of Pauli strings is Hermitian exactly when every coefficient
is real once repeated labels are merged, so `(1 + i)Z + (1 - i)Z` is accepted
(it is `2Z`) and `Z + iX` is refused. Imaginary parts within
`DEFAULT_PHYSICAL_ATOL` count as real. `to_matrix` returns the matrix itself
and accepts any coefficients.

Every refusal is `std::invalid_argument` and names the call that refused.

### Key API

- `SparsePauliOp()`: an operator with no terms
- `SparsePauliOp(terms)`: throws when the terms differ in length
- `simplify(atol)`: merges identical Pauli labels and drops coefficients with
  magnitude at most `atol`. When nothing survives, the result is the zero
  operator at the input's width, one all-identity term with coefficient 0.
  An operator with no terms simplifies to one with no terms.
- `compose(other)`: pairwise composition followed by `simplify`
- `adjoint()`: adjoint of each term
- `tensor(other)`: concatenates Pauli labels and multiplies coefficients
- `operator+`: concatenates terms then `simplify`; throws when both operands
  have terms of different widths. An operand with no terms adds nothing.
- `operator*`: scales coefficients
- `to_matrix()`: builds the dense $2^n \times 2^n$ matrix; uses the action
  $P|j\rangle = i^{\#Y} \cdot (-1)^{\text{popcount}(j \wedge z\_\text{mask})}
  \cdot |j \oplus x\_\text{mask}\rangle$ to fill each column in $O(2^n)$ per
  term (not $O(4^n)$ tensor-product chains); the $i^{\#Y}$ factor is a
  per-term constant folded into the coefficient ($Y = iXZ$), so Hermitian
  operators yield Hermitian matrices and `to_matrix` agrees with
  `expectation_value` and composes homomorphically. Throws for an operator
  with no terms or with terms of different widths.
- `expectation_value(statevector)`: computes ⟨psi|H|psi⟩ without cloning.
  Throws unless every term is exactly `statevector.n_qubits` wide.
- `expectation_value_batch(states)`: batch version with shared mask
  precompute. Takes pointers to `Statevector` instances, checks every state
  against the width rule before evaluating any, and refuses an operator with
  no terms even when `states` is empty.
- `n_qubits()`: number of qubits in the first term, and 0 for an operator with
  no terms (which has no width)
- `from_list(label_coeff)`: throws when the labels differ in length
- `identity(n)` / `zero(n)`: the identity and the zero operator on `n` qubits,
  each a single all-identity term with coefficient 1 or 0

### Exceptions

- `std::invalid_argument` from the constructor, `from_list`, `operator+` and
  `simplify` for terms of different widths
- `std::invalid_argument` from `PauliString` for a character other than `I`,
  `X`, `Y` or `Z`
- `std::invalid_argument` from every evaluation (`expectation_value`,
  `expectation_value_batch`, `to_matrix`) for an operator with no terms, for
  terms of different widths, for a term whose width is not the state's, and for
  a character outside the alphabet
- `std::invalid_argument` from `expectation_value` and
  `expectation_value_batch` for an operator that is not Hermitian

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
- [tests/test_v11292_observable_width.cpp](../../tests/test_v11292_observable_width.cpp):
  the width rule on every path that evaluates or builds an operator
- [tests/test_v11292_pauli_rules.cpp](../../tests/test_v11292_pauli_rules.cpp):
  the alphabet and Hermiticity rules

## Related Pages

- [docs/APIOverview.md](../APIOverview.md)
