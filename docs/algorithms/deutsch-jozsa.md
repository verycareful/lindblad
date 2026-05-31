# Deutsch-Jozsa

This page documents `lindblad::algorithms::DeutschJozsa`.

## Purpose

Deutsch-Jozsa determines whether a Boolean oracle is constant or balanced using a single quantum query.

In lindblad, the helper is designed for the standard textbook oracle layout:

- the first `n` qubits form the query register
- the final qubit is the ancilla

## Theory Summary

The algorithm prepares the query register in superposition, initializes the ancilla to `|1⟩`, and then uses the oracle once.

After the oracle:

1. Apply Hadamard gates to the query register again.
2. Measure the query register.
3. If the measured string is all zeros, the oracle is constant.
4. Otherwise, the oracle is balanced.

## Required Inputs

- A `QuantumCircuit` oracle on `n + 1` qubits
- `n`, the number of query qubits
- Optional shot count and seed

The oracle should follow the standard Deutsch-Jozsa form and act on the ancilla as the Boolean output bit.

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Build the oracle and run the solver:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

QuantumCircuit oracle(4); // 3 query qubits + 1 ancilla
// constant oracle: leave it empty

auto result = DeutschJozsa::solve(oracle, 3);
```

A balanced example that flips the ancilla based on one query qubit:

```cpp
QuantumCircuit balanced(4);
balanced.cx(0, 3);
auto result = DeutschJozsa::solve(balanced, 3);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator Dependencies

Deutsch-Jozsa uses `StatevectorSimulator` internally for exact execution and sampled measurement counts.

The implementation measures the query register after the oracle and the second Hadamard layer.

## Public API Details

### `DeutschJozsa::build_circuit`

- Takes the oracle circuit and the query register size
- Builds the full Deutsch-Jozsa circuit around the provided oracle
- Returns a `QuantumCircuit` ready for measurement or simulation

### `DeutschJozsa::Result`

- `type` is either `CONSTANT` or `BALANCED`

### `DeutschJozsa::solve`

- Runs the Deutsch-Jozsa circuit
- Returns the classification result
- Per-shot execution records only the `n` query qubits (0..n-1) into the classical register, so bitstrings have length `n`
- Compares each bitstring directly against `std::string(n, '0')`; if any match, the oracle is constant; otherwise balanced
- No `substr` stripping is needed: the classical register already excludes the ancilla

## Example Code

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit oracle(2);
    auto result = DeutschJozsa::solve(oracle, 1);
    if (result.type == DeutschJozsa::Result::CONSTANT) {
        std::cout << "constant\n";
    }
}
```

## Return Values and Outputs

- `Result::type` reports whether the oracle is constant or balanced

## Exceptions and Failure Modes

Common issues include:

- the oracle does not use the last qubit as the ancilla
- the oracle has the wrong qubit count
- the oracle encodes a function that is neither constant nor balanced in the expected format

In those cases, the solver may return the wrong classification rather than throwing.

## Common Pitfalls

- The ancilla is part of the circuit and must be included when building the oracle.
- Only the query register is used for the final classification decision.
- This algorithm is only valid for the textbook Deutsch-Jozsa oracle structure.

## Testing Notes

Relevant tests live in:

- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)

## QuditDeutschJozsa

**Purpose:** Determine in one quantum query whether f: Z_d^n → Z_d is constant or balanced.

**d-ary generalization:** Extends Deutsch-Jozsa from binary (d=2) to d-dimensional quantum systems. Works for any d ≥ 2, prime or composite. When d=2, identical to standard D-J.

**Promise:** f is either constant (same value for all inputs) or balanced (each value in Z_d appears exactly d^{n-1} times).

**Circuit** (n+1 qudits, dimension d; qudits 0..n-1 = query, qudit n = ancilla):

| Step | Gate | Target | State after |
|------|------|--------|-------------|
| 1 | X_d^{d-1} | ancilla n | \|d-1⟩ |
| 2 | F_d | ancilla n | \|−⟩_d (phase-kickback receiver) |
| 3 | F_d | each query qudit | \|+⟩_d^n (uniform superposition) |
| 4 | U_f | query→ancilla | phase ω^{f(x)} on each \|x⟩ |
| 5 | F_d† | each query qudit | decode phase kickback |
| 6 | Measure | query register | all-zero ↔ constant |

**Phase kickback:** After the oracle, each query basis state |x⟩ acquires phase ω^{f(x)} = exp(2πi·f(x)/d). For constant f=c, every state has the same phase; after F_d†, amplitude collapses to |0...0⟩. For balanced f, the sum Σ_x ω^{f(x)} vanishes at frequency k=0, so |0...0⟩ has zero amplitude after F_d†.

**Quantum advantage:** 1 oracle query vs 2·d^{n-1}+1 classical queries.

**Required Inputs:**
- `n` — number of query qudits (≥ 1)
- `d` — qudit dimension (≥ 2)
- `f` — `std::function<int(const std::vector<int>&)>` mapping n digits to one digit in [0, d)
- `backend` (optional): `QuditBackend` — simulator to use (default `STATEVECTOR`)
- `noise` (optional): `const QuditNoiseModel*` — noise model; only applied with `DENSITY_MATRIX` (default `nullptr`)

**How to Invoke:**
```cpp
#include "lindblad/algorithms.hpp"
using namespace lindblad::algorithms;

// f: Z_d^n → Z_d
auto f = [](const std::vector<int>& x) -> int { return x[0]; };  // balanced
auto result = QuditDeutschJozsa::solve(/*n=*/2, /*d=*/3, f);
if (result.verdict == QuditDeutschJozsa::Verdict::CONSTANT)
    std::cout << "Constant\n";
else
    std::cout << "Balanced\n";
```

**Supported d:** Any d ≥ 2 (prime or composite). For d=2 reproduces qubit D-J exactly.

**Result:**
```cpp
struct Result {
    QuditDeutschJozsa::Verdict verdict;  // CONSTANT or BALANCED
    int d;
    int n;
};
```

**Structured (affine) oracle overload:** `QuditDeutschJozsa::solve(const QuditAffineOracle& oracle, int d, ...)` accepts `f(x) = a·x + b (mod d)` (single output row). Affine maps are Clifford-decomposable, so this overload additionally supports the `CLIFFORD` backend on prime d. An affine `f` is constant iff `a = 0` and balanced iff `a ≠ 0` (for prime d every nonzero `a` is exactly balanced).

**Exceptions:**
- `std::invalid_argument` if `d < 2` or `n < 1`
- `std::invalid_argument` if `f` returns a value outside `[0, d)`
- `std::invalid_argument` if `backend == QuditBackend::CLIFFORD` for the opaque `f` overload (use the affine-oracle overload instead)

**Backend:**

| Backend | Supported | Notes |
|---|---|---|
| `STATEVECTOR` | ✓ | Default. Exact dense statevector, O(d^{n+1}) amplitudes. |
| `DENSITY_MATRIX` | ✓ | Full mixed-state; applies `noise` model if provided. |
| `MPS` | ✓ | Tensor-network. |
| `CLIFFORD` | affine only | A black-box `f` has no Clifford decomposition, so the opaque overload throws (no silent fallback). The affine-oracle overload runs on the stabilizer tableau for prime d. |

The `noise` argument is applied only in the `DENSITY_MATRIX` path. See [docs/api/qudit-simulators.md](../api/qudit-simulators.md) for the full backend API reference.

## Related Source Files

- [docs/api/deutsch-jozsa.md](../api/deutsch-jozsa.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/deutsch_jozsa.cpp](../../src/algorithms/deutsch_jozsa.cpp)
- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)
