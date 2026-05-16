# Qudit Layer API Deep Dive

This page documents the general-purpose qudit simulation layer in lindblad:
the d-dimensional state vector, the qudit gate matrix generators, and the
qudit simulator. The layer is independent of the existing qubit infrastructure
and provides a reusable foundation for any d-dimensional quantum algorithm.

## Backend simulators

This page covers the foundational statevector layer. Three additional backends
are available for mixed-state, tensor-network, and stabilizer simulation:

| Backend | Class | Header | Memory | Restrictions |
|---|---|---|---|---|
| Density matrix | `QuditDensityMatrix` | `qudit/qudit_density_matrix.hpp` | O(d^{2n}) | Supports noise; no prime-d restriction |
| Matrix Product State | `QuditMPS` | `qudit/qudit_mps.hpp` | O(n·χ²·d) | χ = bond dimension; oracle fallback is O(d^n) |
| Clifford stabilizer | `QuditCliffordSimulator` | `qudit/qudit_clifford.hpp` | O(n²) | Prime d only; Grover/QPE/Simon throw |

All algorithms accept a `QuditBackend` enum argument and an optional
`QuditNoiseModel*`. See [qudit-simulators.md](qudit-simulators.md) for the
complete API reference for all backends.

---

## Header and Namespace

- Headers:
  - `include/lindblad/qudit/qudit_statevector.hpp`
  - `include/lindblad/qudit/qudit_gates.hpp`
  - `include/lindblad/qudit/qudit_simulator.hpp`
- Namespace: `lindblad` (state vector, simulator) and `lindblad::qudit_gates`
  (gate matrix generators)
- All three headers are also pulled in transitively by
  `include/lindblad/algorithms.hpp`.

## What It Provides

A qudit of dimension d has d basis states {0, …, d−1}. A system of n qudits
has a state space of dimension d^n. The qudit layer provides:

- **`QuditStatevector`** — dense `d^n`-dimensional state vector with single-
  and two-qudit gate application and computational-basis measurement
- **`qudit_gates`** — generators for the most common qudit gates as row-major
  `std::vector<Complex128>` matrices: qudit QFT, inverse QFT, generalised
  shift gate, controlled-ADD gate
- **`QuditSimulator`** — thin executor that applies an ordered sequence of
  gate operations then samples one measurement

Works for **any d ≥ 2**, prime or composite. d = 2 reproduces the qubit case
exactly.

## State Indexing

Mixed-radix little-endian (same convention as the binary `Statevector`):

```
index = x_0 · d^0 + x_1 · d^1 + … + x_{n-1} · d^{n-1}
```

Qudit 0 is the least-significant digit; stride of qudit q is `d^q`. The static
helpers `QuditStatevector::index_to_digits` and `digits_to_index` convert
between the flat index and the per-qudit digit vector.

---

## `QuditStatevector`

### Public Fields

| Field | Type | Description |
|---|---|---|
| `n_qudits` | `int` | number of qudits |
| `d` | `int` | local Hilbert space dimension per qudit |
| `dim` | `size_t` | `d^n_qudits` |
| `amplitudes` | `std::vector<Complex128>` | length-`dim` amplitude vector, row-major |

### Constructor

```cpp
QuditStatevector(int n_qudits, int d);
```

Allocates `d^n` amplitudes and initialises to `|0…0⟩` (amplitude[0] = 1).
Throws `std::invalid_argument` if `d < 2` or `n_qudits < 1`.

### Methods

| Method | Description |
|---|---|
| `initialize()` | Reset to `\|0…0⟩` |
| `normalize()` | Renormalise to unit norm |
| `norm_sq()` | Sum of `\|amplitude[i]\|²` |
| `apply_1qudit(q, U)` | Apply a d × d row-major gate `U` to qudit `q` |
| `apply_2qudit(q0, q1, U)` | Apply a d² × d² row-major gate `U` to qudits `(q0, q1)`, `q0 != q1` |
| `measure(seed)` | Sample one outcome; returns a length-`n_qudits` vector of digits in `{0..d-1}` |

`apply_2qudit` uses the row index convention `r = new_q0 · d + new_q1` and
column index `c = old_q0 · d + old_q1`. Throws `std::invalid_argument` if
`q0 == q1`.

### Static helpers

```cpp
static std::vector<int> index_to_digits(size_t idx, int d, int n_qudits);
static size_t           digits_to_index(const std::vector<int>& digits, int d);
static size_t           ipow(size_t base, int exp) noexcept;
```

`ipow` is exact integer power; used internally for `dim` and gate strides.

---

## `qudit_gates` namespace

All functions return row-major `std::vector<Complex128>` matrices. `ω = exp(2πi/d)`.

| Function | Returns | Definition |
|---|---|---|
| `qft_matrix(d)` | d × d | `F[j,k] = ω^{jk} / √d` |
| `iqft_matrix(d)` | d × d | `F†[j,k] = ω^{-jk} / √d` |
| `shift_matrix(d, m = 1)` | d × d | `X^m[j,k] = 1` if `k == (j+m) mod d` else 0; `m` may be negative or ≥ d (normalised internally) |
| `cadd_matrix(d, s)` | d² × d² | `\|x⟩\|y⟩ → \|x⟩\|(y + s·x) mod d⟩`; `s` normalised mod d |

For d = 2 these reduce to `F_2 = Hadamard`, `X_2^1 = Pauli X`,
`CADD_1 = CNOT`.

---

## `QuditSimulator`

### `QuditGateOp`

```cpp
struct QuditGateOp {
    enum class Type { SINGLE, TWO } type;
    int q0;          // qudit index (SINGLE) or control qudit (TWO)
    int q1 = -1;     // target qudit (TWO only; ignored for SINGLE)
    std::vector<Complex128> matrix;  // d×d (SINGLE) or d²×d² (TWO), row-major
};
```

### `QuditSimulator::run`

```cpp
static Result run(QuditStatevector& sv,
                  const std::vector<QuditGateOp>& ops,
                  uint64_t seed = 0);
```

Applies `ops` in order (modifying `sv` in-place), then samples one
measurement. Throws `std::invalid_argument` if a TWO-qudit op has `q1 < 0`.
Matrix size validation is performed at the underlying
`QuditStatevector::apply_*qudit` call.

```cpp
struct Result {
    std::vector<int> outcome;       // measured symbol per qudit
    double simulation_time_seconds; // wall-clock duration
};
```

---

## Example: building a custom qudit algorithm

```cpp
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_simulator.hpp"

using namespace lindblad;

// Three qutrits prepared in |0,0,0>; apply F_3 to qudit 0, shift qudit 1
// twice, then measure.
QuditStatevector sv(/*n_qudits=*/3, /*d=*/3);

auto F  = qudit_gates::qft_matrix(3);
auto X  = qudit_gates::shift_matrix(3, 1);

std::vector<QuditGateOp> ops = {
    {QuditGateOp::Type::SINGLE, 0, -1, F},
    {QuditGateOp::Type::SINGLE, 1, -1, X},
    {QuditGateOp::Type::SINGLE, 1, -1, X},
};

auto result = QuditSimulator::run(sv, ops, /*seed=*/42);
// result.outcome[0] in {0,1,2}, result.outcome[1] == 2, result.outcome[2] == 0
```

---

## Algorithms and Use Cases

The qudit layer is general-purpose. The list below shows what can be built
on top of it.

### Implemented

- **Qudit Bernstein-Vazirani** (`QuditBernsteinVazirani`) — recovers `s ∈ Z_d^n` from `f(x) = s·x mod d` in 1 query. See [bernstein-vazirani.md](bernstein-vazirani.md#quditbernsteinvazirani) for the algorithm doc.
- **Qudit Deutsch-Jozsa** (`QuditDeutschJozsa`) — determine constant vs balanced `f: Z_d^n → Z_d` in 1 query. See [deutsch-jozsa.md](deutsch-jozsa.md#quditdeutschjozsa).
- **Qudit Grover** (`QuditGrover`) — d-ary amplitude amplification search in O(√(d^n)) queries. See [grover.md](grover.md#quditgrover).
- **Qudit Phase Estimation** (`QuditPhaseEstimation`) — estimate eigenphase of d×d unitary to d^{-m} precision using m clock qudits. See [qpe.md](qpe.md#quditphaseestimation).
- **Qudit Simon's algorithm** (`QuditSimon`) — find hidden period s ∈ Z_d^n in O(n) queries; requires prime d. See [simon.md](simon.md#quditsimone).

### Algorithms that need a small gate-set extension

These build on the layer with one or two extra gate generators:

- **Qudit Shor / period finding** — needs modular-multiplication gates over Z_d.
- **Higher-spin Hamiltonian simulation** — spin-1 (d = 3), spin-3/2 (d = 4),
  …; needs generalised spin operators `S_x, S_y, S_z`.
- **Qudit stabiliser / CSS-d error-correcting codes** — needs generalised
  Pauli-X (already present as `shift_matrix`) and Pauli-Z (clock operator).
- **High-dimensional quantum walks** — d-ary discrete-time walks; needs a
  coin operator and a controlled-shift.

### Use cases beyond pure algorithms

- **Native simulation of multi-level systems** — Rydberg atom arrays,
  trapped ions with multiple addressable states, and NV centres all have
  d > 2 natively; simulating them in their native dimension avoids embedding
  overhead.
- **Compact circuit encoding** — n qutrits encode 3^n states with O(n)
  physical units, often producing shallower circuits than the qubit
  equivalent for the same problem size.
- **High-dimensional QKD analysis** — qudit BB84 with d > 2 alphabets has
  higher information rate per quantum signal.
- **Discretised continuous-variable simulation** — high-d qudits approximate
  truncated harmonic oscillators for chemistry workflows.

---

## Exceptions Summary

| Source | Throws | When |
|---|---|---|
| `QuditStatevector` ctor | `std::invalid_argument` | `d < 2` or `n_qudits < 1` |
| `QuditStatevector::apply_2qudit` | `std::invalid_argument` | `q0 == q1` |
| `QuditSimulator::run` | `std::invalid_argument` | TWO-qudit op with `q1 < 0` |

---

## Performance Notes

- `QuditStatevector` stores amplitudes as a single dense
  `std::vector<Complex128>` (length `d^n`). For correctness-first
  simulation this is intentionally simple; the SoA (separate real/imag
  arrays) layout used by the qubit `Statevector` is not yet replicated here.
- `apply_1qudit` is `O(dim · d)`; `apply_2qudit` is `O(dim · d²)`. The
  two-qudit loop currently scans all `dim` indices with a stride filter
  rather than directly iterating only the `dim/d²` base indices — straight
  to optimise later if the inner kernel becomes a bottleneck.
- No multithreading is wired into the qudit kernels yet. The qubit
  `Statevector` uses OpenMP; the same pragma pattern could be applied here.

---

## Related Files

- [docs/api/bernstein-vazirani.md](bernstein-vazirani.md) — `QuditBernsteinVazirani` API
- [docs/algorithms/bernstein-vazirani.md](../algorithms/bernstein-vazirani.md) — Qudit BV theory
- [include/lindblad/qudit/](../../include/lindblad/qudit/) — public headers
- [src/qudit/](../../src/qudit/) — implementations
- [tests/test_qudit_bv.cpp](../../tests/test_qudit_bv.cpp) — coverage for statevector, gates, simulator, and BV
- [docs/APIOverview.md](../APIOverview.md)
- [docs/MasterDocumentation.md](../MasterDocumentation.md)
