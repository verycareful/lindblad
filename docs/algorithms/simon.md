# Simon

This page documents `lindblad::algorithms::Simon`.

## Purpose

Simon’s algorithm finds the hidden period `s` of a 2-to-1 function where `f(x) = f(x ⊕ s)`.

In lindblad, the helper is built around the standard Simon oracle layout and returns both the recovered period and the sampled linear equations.

## Theory Summary

Simon’s algorithm uses the following flow:

1. Prepare the query register in superposition.
2. Apply the oracle.
3. Apply Hadamard gates to the query register again.
4. Measure the query register repeatedly.
5. Use the resulting equations to solve for the hidden period over GF(2).

Each measured string `y` satisfies `y · s = 0 mod 2`.

## Required Inputs

- A `QuantumCircuit` oracle on `2n` qubits
- `n`, the query register size
- Optional seed and extra sampling count

The oracle should encode a valid Simon promise instance.

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Build the oracle and solve:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

QuantumCircuit oracle(6); // 3 query qubits + 3 output qubits
// oracle construction goes here

auto result = Simon::solve(oracle, 3, 42);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator Dependencies

Simon uses `StatevectorSimulator` internally to sample repeated query-register outcomes.

The implementation stores the raw measured equations and then runs a GF(2) elimination step to recover the period.

## Public API Details

### `Simon::build_circuit`

- Takes a Simon oracle and the query size
- Builds the standard Simon query circuit
- Returns a `QuantumCircuit`

### `Simon::Result`

- `period` stores the recovered hidden string
- `equations` stores the non-zero sampled equation strings used in the solve step

### `Simon::solve`

```cpp
static Result solve(const QuantumCircuit& oracle, int n,
                    uint64_t seed = 0, int extra_samples = 2,
                    bool batch_shots = true);
```

- Samples the Simon circuit, collects independent equations from the
  measurement outcomes, and solves for the hidden period using Gaussian
  elimination over GF(2).
- `batch_shots = true` (default, added in R.1.13, audit F-21): draws all
  equation samples from ONE batched simulation (the circuit's measurements are
  terminal) and harvests the distinct non-zero outcomes, using a `std::set` so
  the equation order is deterministic and a given `seed` is reproducible. Set
  `false` to restore the pre-R.1.13 per-sample loop (one single-shot simulation
  per equation), which reproduces the old seeded equation stream byte-for-byte.
  Both paths are statistically equivalent; only the seeded byte output differs
  (seeds reproduce within a version, not across versions).

### `Simon::gaussian_eliminate`

- Private helper used by `solve`
- Reduces the sampled equations and reconstructs the final bitstring

## Example Code

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit oracle(6);
    // build a 3-qubit Simon oracle here

    auto result = Simon::solve(oracle, 3, 7);
    std::cout << "period: " << result.period << '\n';
}
```

## Return Values and Outputs

- `period` is the recovered hidden period string
- `equations` lists the sampled linear equations used during recovery

## Exceptions and Failure Modes

Common issues include:

- an oracle that does not satisfy Simon’s promise
- a query/output register layout that does not match the documented `2n` qubit structure
- too few independent equations to recover the full period reliably

In those cases, the solver may return an incomplete or incorrect period.

## Common Pitfalls

- Simon’s algorithm usually needs multiple samples, not just one run.
- The oracle must be a proper 2-to-1 promise instance.
- The returned equations are useful for debugging and should not be ignored when diagnosing incorrect recovery.

## Testing Notes

Relevant tests live in:

- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)

## QuditSimon

**Purpose:** Find hidden period s ∈ Z_d^n such that f(x+s) = f(x) for all x, using O(n) quantum queries.

**d-ary generalization:** Extends Simon's algorithm from binary (d=2) to d-dimensional quantum systems. Requires d to be **prime** for Gaussian elimination over GF(d). When d=2, identical to standard Simon's algorithm.

**Promise:** f: Z_d^n → Z_d^n satisfies f(x) = f(y) ⟺ x − y ≡ 0 or s (mod d, componentwise).

**Circuit per quantum query** (2n qudits: first n = query, last n = output):

| Step | Gate | Target | State after |
|------|------|--------|-------------|
| 1 | F_d | each query qudit | uniform superposition |
| 2 | U_f | query→output | \|x⟩\|f(x)⟩ via apply_function_oracle |
| 3 | F_d† | each query qudit | decode |
| 4 | Measure | query register | y with s·y ≡ 0 (mod d) |

**Classical post-processing:** recover the hidden subgroup `H = {s : y·s ≡ 0 (mod d) for every measured y}` from ~n+extra_samples nonzero measurement vectors y, then report a nonzero generator.
- **prime d** — Gaussian elimination over the field GF(d); the null space of the measurement matrix (mod d) is spanned by s.
- **composite d** — Z_d is only a ring, so field elimination breaks. The kernel mod d is computed via the integer Smith Normal Form of the measurement matrix: for `U·E·V = D` diagonal, `s = V·t` solves `E·s ≡ 0 (mod d)` iff each pivot `t_i` is a multiple of `d/gcd(D_ii, d)` and free `t_i` range over Z_d. This is uniform across all moduli (it reproduces the field result for prime d) and needs no separate CRT recombination. Candidate generators are then verified against the oracle (`f(x) = f(x+s)`) to select a true period.

**Quantum advantage:** O(n) quantum queries vs exponential classical queries.

**Required Inputs:**
- `n` — number of qudits (≥ 1)
- `d` — qudit dimension; any d ≥ 2 (prime or composite)
- `f` — `std::function<std::vector<int>(const std::vector<int>&)>` mapping n digits to n digits
- `backend` (optional): `QuditBackend` — simulator to use (default `STATEVECTOR`). The opaque `f` overload does not support `CLIFFORD` (a black-box function has no Clifford decomposition); use the affine-oracle overload for the `CLIFFORD` backend.
- `noise` (optional): `const QuditNoiseModel*` — noise model; only applied with `DENSITY_MATRIX` (default `nullptr`)

**Structured (affine) oracle overload:** `QuditSimon::solve(const QuditAffineOracle& oracle, int d, ...)` accepts `f(x) = A·x + b (mod d)` (square `A`). Affine maps are Clifford-decomposable, so this overload additionally supports the `CLIFFORD` backend on prime d; all other backends run for any d. The hidden subgroup is `ker_{Z_d}(A)`; `b` does not affect the period.

**How to Invoke:**
```cpp
#include "lindblad/algorithms.hpp"
using namespace lindblad::algorithms;

// Build Simon oracle with hidden period s = {1, 2} in Z_3^2
std::vector<int> s = {1, 2};
auto f = [&](const std::vector<int>& x) -> std::vector<int> {
    // Canonical coset representative (lexicographic min of {x, x+s, x+2s,...} mod 3)
    std::vector<int> best = x, cur = x;
    for (int k = 1; k < 3; ++k) {
        for (int i = 0; i < 2; ++i) cur[i] = (x[i] + k * s[i]) % 3;
        if (cur < best) best = cur;
    }
    return best;
};

auto r = QuditSimon::solve(/*n=*/2, /*d=*/3, f, /*extra_samples=*/3, /*seed=*/42);
// r.period == {1, 2},  r.is_trivial == false
```

**Supported d:** Any d ≥ 2 (prime or composite).

**Result:**
```cpp
struct Result {
    std::vector<int> period;   // s in Z_d^n, each element in {0..d-1}
    bool is_trivial;           // true iff s = 0...0 (f is injective)
    int d;
    int n;
    int quantum_queries;       // total number of quantum circuit executions
};
```

**Exceptions:**
- `std::invalid_argument` if `d < 2` or `n < 1`
- `std::invalid_argument` if `f` returns a vector of wrong size or with digits outside `[0, d)`
- `std::invalid_argument` if `backend == QuditBackend::CLIFFORD` for the opaque `f` overload (use the affine-oracle overload instead), or `CLIFFORD` with composite d for the affine overload

**Backend:**

| Backend | Supported | Notes |
|---|---|---|
| `STATEVECTOR` | ✓ | Default. Exact dense statevector; each query creates a fresh 2n-qudit state vector. Any d. |
| `DENSITY_MATRIX` | ✓ | Full mixed-state; applies `noise` model if provided. Any d. |
| `MPS` | ✓ | Tensor-network. Any d. |
| `CLIFFORD` | affine only | Opaque `f` overload throws (no Clifford decomposition of a black box). The affine-oracle overload runs on the stabilizer tableau for prime d. |

The `noise` argument is applied only in the `DENSITY_MATRIX` path. See [docs/api/qudit-simulators.md](../api/qudit-simulators.md) for the full backend API reference.

**Common pitfalls:**
- Composite d (4, 6, 8, 9, ...) is supported via the integer-SNF ring kernel. For composite d the hidden subgroup may be larger than `{0, s}`; the returned period is one nonzero generator and may need more `extra_samples` to pin down.
- The function `f` must satisfy the Simon promise (constant on cosets of the hidden subgroup `H`, distinct across cosets). If the promise is violated, the classical post-processing may return an incorrect period.
- For small n, extra_samples=3 is sufficient. For n ≥ 5, increase extra_samples if the algorithm fails.

## Related Source Files

- [docs/api/simon.md](../api/simon.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/simon.cpp](../../src/algorithms/simon.cpp)
- [tests/test_classic_algorithms.cpp](../../tests/test_classic_algorithms.cpp)
