# Qudit Backend Simulators

This page documents the four simulation backends available for qudit algorithms
in lindblad, together with the noise model that plugs into the density-matrix path.

---

## Overview

All five qudit algorithms (`QuditBernsteinVazirani`, `QuditDeutschJozsa`,
`QuditGrover`, `QuditPhaseEstimation`, `QuditSimon`) accept an optional
`backend` argument of type `QuditBackend` and an optional `noise` pointer of
type `const QuditNoiseModel*`:

```cpp
// Example: run BV with the density-matrix backend and depolarizing noise
QuditNoiseModel model;
model.add_depolarizing(0, d, 0.01);

auto result = QuditBernsteinVazirani::solve(
    secret, d, shots, seed,
    QuditBackend::DENSITY_MATRIX, &model);
```

The four backends are:

| Backend | Class | Description |
|---|---|---|
| `STATEVECTOR` | `QuditStatevector` | Default. Exact dense state vector, exponential memory. |
| `DENSITY_MATRIX` | `QuditDensityMatrix` | Mixed-state ρ, supports noise channels and Lindblad evolution. Memory: d^{2n}. |
| `MPS` | `QuditMPS` | Tensor-train approximation; polynomial memory for low-entanglement states. Truncates bond dimension. |
| `CLIFFORD` | `QuditCliffordSimulator` | Stabilizer tableau; polynomial time and memory. Restricted to prime d and Clifford-simulable algorithms only. |

### Algorithm × backend compatibility

| Algorithm | STATEVECTOR | DENSITY_MATRIX | MPS | CLIFFORD |
|---|---|---|---|---|
| `QuditBernsteinVazirani` | ✓ | ✓ | ✓ | ✓ (prime d) |
| `QuditDeutschJozsa` | ✓ | ✓ | ✓ | ✓ affine oracle, prime d (opaque `f` throws) |
| `QuditGrover` | ✓ | ✓ | ✓ | ✗ throws |
| `QuditPhaseEstimation` | ✓ | ✓ | ✓ | ✗ throws |
| `QuditSimon` | ✓ | ✓ | ✓ | ✓ affine oracle, prime d (opaque `f` throws) |

Noise (`QuditNoiseModel*`) is applied only when `backend == DENSITY_MATRIX`.
Passing a noise pointer with any other backend has no effect.

---

## `QuditBackend` enum

```cpp
#include "lindblad/qudit/qudit_backend.hpp"

enum class QuditBackend {
    STATEVECTOR,      // default
    DENSITY_MATRIX,
    MPS,
    CLIFFORD,
};
```

---

## `QuditNoiseModel`

```cpp
#include "lindblad/qudit/qudit_noise_model.hpp"
```

Collects per-qudit noise specifications. Used exclusively by the
`DENSITY_MATRIX` backend via `QuditDensityMatrix::apply_noise`.

### Factory methods (static)

| Method | Kraus ops | Description |
|---|---|---|
| `depolarizing_channel(d, p)` | d² | K₀ = √(1−p)·I; K_{a,b} = √(p/(d²−1))·X^a Z^b for (a,b)≠(0,0). |
| `amplitude_damping_channel(d, γ)` | d | K₀ = diag(√(1−k·γ)); K_j[j−1,j] = √(j·γ). Requires (d−1)·γ ≤ 1. |
| `phase_damping_channel(d, p)` | d+1 | K₀ = √(1−p)·I; K_j = √p·\|j⟩⟨j\|. |
| `amplitude_damping_lindblad(d, γ)` | — | Single `QuditLindbladOp`: L[j−1,j] = √(j·γ), rate = 1. |
| `dephasing_lindblad(d, γ)` | — | d−1 `QuditLindbladOp`s: L_k[j,j] = √γ·ω^{kj}, rate = 1. |

### Convenience methods

```cpp
void add_depolarizing(int q, int d, double p);
void add_amplitude_damping(int q, int d, double gamma);
void add_phase_damping(int q, int d, double p);
void add_lindblad_op(int q, std::vector<Complex128> L, double rate);
```

Each method registers noise for qudit index `q`. Multiple calls for the same
qudit accumulate.

### Data members

```cpp
std::map<int, QuditQuditNoise> per_qudit;  // qudit index → noise spec
```

`QuditQuditNoise` holds a `QuditKrausChannel` (list of Kraus operators) and a
`std::vector<QuditLindbladOp>` (Lindblad jump operators with rates).

---

## `QuditDensityMatrix`

```cpp
#include "lindblad/qudit/qudit_density_matrix.hpp"
```

Density operator ρ for n qudits each of dimension d. Stored as a flat
`dim×dim` row-major complex matrix: `rho[i*dim + j] = ⟨i|ρ|j⟩`.

### Constructors

```cpp
QuditDensityMatrix(int n_qudits, int d);          // ρ = |0…0⟩⟨0…0|
explicit QuditDensityMatrix(const QuditStatevector& sv);  // ρ = |ψ⟩⟨ψ|
```

### Public fields

| Field | Type | Description |
|---|---|---|
| `n_qudits` | `int` | number of qudits |
| `d` | `int` | local dimension |
| `dim` | `size_t` | d^n_qudits |
| `dim_sq` | `size_t` | dim² |
| `rho` | `std::vector<Complex128>` | length dim², row-major |

### Invariant maintenance

| Method | Description |
|---|---|
| `initialize()` | Reset ρ to \|0…0⟩⟨0…0\| |
| `symmetrize()` | Enforce Hermiticity: ρ ← (ρ + ρ†)/2 |
| `normalize()` | Rescale so Tr(ρ) = 1. Throws when there is no trace to divide out |
| `trace()` | Returns Tr(ρ) |
| `purity()` | Returns Tr(ρ²) ∈ (0, 1] |
| `is_normalized(atol)` | Predicate: is Tr(ρ) within `atol` of 1? Does not repair or throw |
| `check_normalized(validation)` | Apply a validation policy; `Fix` renormalizes in place |

`normalize()` refuses rather than returning quietly. The two cases it rejects,
a zero trace and a non-finite one, are exactly those where dividing by the trace
produces garbage instead of a density matrix, and returning the matrix unchanged
would tell a caller who asked for normalization nothing at all.

### Unitary evolution

```cpp
void apply_1qudit(int q, const std::vector<Complex128>& U);
void apply_2qudit(int q0, int q1, const std::vector<Complex128>& U);
```

`U` is a d×d (or d²×d²) row-major unitary. Applies ρ → U_q ρ U_q†.
The indexing convention for `apply_2qudit` matches `QuditStatevector::apply_2qudit`:
row `r = new_q0·d + new_q1`, column `c = old_q0·d + old_q1`.

### Noise channels

```cpp
void apply_kraus_1qudit(int q,
    const std::vector<std::vector<Complex128>>& K_ops);   // ρ → Σ_k K_k ρ K_k†
void apply_kraus_2qudit(int q0, int q1,
    const std::vector<std::vector<Complex128>>& K_ops);
void apply_lindblad_step(int q,
    const std::vector<QuditLindbladOp>& ops, double dt);  // first-order Euler
void apply_noise(const QuditNoiseModel& model, double dt = 0.0);  // all per-qudit noise
```

`apply_noise` always applies Kraus channels; it additionally applies Lindblad
steps if `dt > 0`.

### Oracle operations

```cpp
void apply_phase_oracle(
    const std::function<Complex128(const std::vector<int>&)>& phase_fn);

void apply_function_oracle(
    int n_query, int n_output,
    const std::function<std::vector<int>(const std::vector<int>&)>& f);
```

`apply_phase_oracle` multiplies `rho[i][j]` by `phase_fn(digits(i)) * conj(phase_fn(digits(j)))`.

`apply_function_oracle` implements `|x⟩|y⟩ → |x⟩|(y + f(x)) mod d⟩`. Query
qudits are `[0, n_query)`; output qudits are `[n_query, n_query+n_output)`.
The function `f` maps the digit vector of the query register to the digit vector
of the addend.

### Measurement and partial trace

```cpp
std::vector<int> measure(uint64_t seed = 0);
QuditDensityMatrix partial_trace(const std::vector<int>& keep_qudits) const;
```

`measure` samples from the diagonal probabilities, collapses ρ to the
post-measurement pure state, and returns per-qudit digits.

`partial_trace` traces out all qudits not in `keep_qudits`. Throws
`std::invalid_argument` if `keep_qudits` is empty.

---

## `QuditMPS`

```cpp
#include "lindblad/qudit/qudit_mps.hpp"
```

Matrix Product State representation. Each site tensor `A_q` has shape
`(d, χ_L, χ_R)` with data layout `data[σ·χ_L·χ_R + aL·χ_R + aR]`. Bond
dimension is truncated after each two-site gate by SVD to `max_bond_dim`,
discarding the smallest singular values while the discarded weight
$\sum \sigma^2$ stays within `svd_cutoff` of the total. Same rule and same
meaning as `MPSState::cutoff` in the qubit layer.

### Constructors

```cpp
QuditMPS(int n_qudits, int d,
         int max_bond_dim = 64, double svd_cutoff = 1e-16);   // |0…0⟩
explicit QuditMPS(const QuditStatevector& sv,
                  int max_bond_dim = 64, double svd_cutoff = 1e-16);
```

The statevector constructor performs a sequential left-to-right SVD
decomposition, building an exact MPS representation (up to the truncation
parameters).

### Public fields

| Field | Type | Description |
|---|---|---|
| `n_qudits` | `int` | number of sites |
| `d` | `int` | local dimension |
| `max_bond_dim` | `int` | maximum retained singular values per bond |
| `svd_cutoff` | `double` | max fraction of total weight truncation may discard |
| `svd_method` | `SVDMethod` | SVD backend: default `BDC`, which is divide-and-conquer and pulls away from Jacobi as the block grows. `Jacobi` is selectable and emits a one-time note that it is the slower algorithm. Below a 16x16 block the two run identical code. Declared in `lindblad/types.hpp`. |
| `tensors` | `std::vector<MPSSiteTensor>` | site tensors |

### Gate and oracle API

```cpp
void apply_1qudit(int q, const std::vector<Complex128>& U);
void apply_2qudit_adjacent(int q, const std::vector<Complex128>& U);  // acts on (q, q+1)
void apply_2qudit(int q0, int q1, const std::vector<Complex128>& U);  // any q0 ≠ q1
```

Non-adjacent two-qudit gates (`apply_2qudit`) are handled via a SWAP chain:
the sites are brought adjacent, the gate is applied, and the SWAPs are reversed.

```cpp
void apply_phase_oracle(
    const std::function<Complex128(const std::vector<int>&)>& phase_fn);
void apply_function_oracle(int n_query, int n_output,
    const std::function<int(int)>& f);
```

Both oracles fall back to an exact dense statevector (via `to_statevector()`)
and reconstruct the MPS. Use for small systems only; not efficient for large n.

The `apply_function_oracle` `f` maps the flat index of the query register to
the flat addend for the output register (both as integers, not digit vectors).

### Measurement, norm, and canonicalisation

```cpp
std::vector<int> measure(uint64_t seed = 0);  // sequential environment sampling
double norm_sq() const;
void normalize();
bool is_normalized(double atol = DEFAULT_PHYSICAL_ATOL) const;
void check_normalized(ValidationOptions validation = {});
void left_canonicalize();
void right_canonicalize();
```

`norm_sq()` is a transfer-matrix contraction along the chain, `O(n · χ³)`, since
there is no flat amplitude array to sweep. `normalize()` rescales the first site
tensor, which is exact because the norm is multilinear in the tensors, and
throws when there is no norm to divide out rather than returning the state
unchanged.

`is_normalized` answers without repairing or throwing. `check_normalized`
applies a policy, with `Fix` renormalizing in place. Under `Ignore` neither the
contraction nor anything else runs, which matters more here than on the dense
classes because the measurement is the most expensive of any state type in the
library.

`measure` (R.1.13, audit F-5): precomputes the right environments once
(`build_right_envs`, $O(n \cdot \chi^3)$) and samples left-to-right read-only,
carrying the left environment incrementally. This replaced the previous path
that contracted the whole MPS to a dense $d^n$ statevector before sampling, so
measurement is now $O(n \cdot \chi^3)$ with memory bounded by the bond dimension.
The phase/function oracles still use the dense `to_statevector()` fallback (a
separate, documented limitation).

### `MPSSiteTensor`

```cpp
struct MPSSiteTensor {
    int d, chi_L, chi_R;
    std::vector<Complex128> data;  // [sigma * chi_L * chi_R + aL * chi_R + aR]

    Complex128& at(int sigma, int aL, int aR);
    Eigen::MatrixXcd as_left_matrix() const;   // shape (d*chi_L, chi_R)
    Eigen::MatrixXcd as_right_matrix() const;  // shape (chi_L, d*chi_R)
    static MPSSiteTensor from_left_matrix(const Eigen::MatrixXcd&, int d, int chi_L);
    static MPSSiteTensor from_right_matrix(const Eigen::MatrixXcd&, int d, int chi_R);
};
```

---

## `QuditCliffordSimulator`

```cpp
#include "lindblad/qudit/qudit_clifford.hpp"
```

Heisenberg-picture stabilizer tableau for prime d. Runs in polynomial time and
memory. **d must be prime**; the constructor throws `std::invalid_argument`
otherwise.

### Tableau representation

2n rows (n destabilizers [0, n) + n stabilizers [n, 2n)). Each row encodes:

```
τ^{phase[r]} · ∏_q X_q^{xbits[r][q]} · Z_q^{zbits[r][q]}
```

where `τ = exp(iπ(d+1)/d)`, `ω = τ² = exp(2πi/d)`.

Initial state `|0…0⟩`:
- Destabilizer j (row j): `X_j` — `xbits[j][j] = 1`, others 0.
- Stabilizer j (row n+j): `Z_j` — `zbits[n+j][j] = 1`, others 0.
- All phases = 0.

### Constructor

```cpp
QuditCliffordSimulator(int n_qudits, int d);
```

Throws `std::invalid_argument` if `d` is not prime.

### Public fields

```cpp
int n_qudits, d;
std::vector<std::vector<int>> xbits;  // [row][qudit], mod d
std::vector<std::vector<int>> zbits;  // [row][qudit], mod d
std::vector<int> phase;               // [row], mod 2d
```

### Gate API

All gates update the tableau in-place (Heisenberg picture).

| Method | Gate | Conjugation rule |
|---|---|---|
| `apply_X(q, m=1)` | X^m | phase -= 2m·z_q |
| `apply_Z(q, m=1)` | Z^m | phase += 2m·x_q |
| `apply_H(q)` | QFT | x_q ← −z_q mod d; z_q ← x_q; phase −= 2·old_x_q·old_z_q |
| `apply_P(q)` | Phase | d=2: z_q += x_q; phase += x_q.  odd prime d: z_q += x_q; phase += x_q(x_q−1) |
| `apply_CSUM(c, t)` | SUM gate | x_t += x_c; z_c -= z_t; phase -= 2·x_c·z_t (before update) |
| `apply_CSUM_dag(c, t)` | SUM† | x_t -= x_c; z_c += z_t; phase += 2·x_c·z_t (before update) |

All arithmetic is mod d (bits) or mod 2d (phase).

H⁴ = I in the qudit Clifford group, so H† = H³.

### Measurement

```cpp
int measure_qudit(int q, uint64_t seed = 0);   // one qudit; collapses tableau
std::vector<int> measure(uint64_t seed = 0);   // all n qudits
```

Returns outcomes in `{0, …, d−1}`. When the outcome is **indeterminate**
(qudit q anti-commutes with some stabilizer), a uniformly random outcome is
chosen and the tableau is updated to the post-measurement state. When
**determined**, the implementation solves a linear system over Z_d to find the
stabilizer combination equal to Z_q, then reads the outcome from that
combination's phase via `phase + 2·m·k ≡ 0 (mod 2d)`. The tableau is
collapsed to the post-measurement state in both cases.

### Utilities

```cpp
static bool is_prime(int d);
int symplectic_product(int row_a, int row_b) const;
void row_multiply(int r, int s);  // G_r ← G_r · G_s
```

---

## Algorithmic restrictions

### CLIFFORD backend limitations

Only algorithms that can be expressed entirely in terms of Clifford gates support
the `CLIFFORD` backend:

- **`QuditBernsteinVazirani`**: fully Clifford for prime d. Each CADD(s_i) is
  implemented as `apply_CSUM` repeated s_i times.
- **`QuditDeutschJozsa`** and **`QuditSimon`**: support `CLIFFORD` (prime d) only
  through the structured `QuditAffineOracle` overload, which lowers `f(x) = A·x + b`
  to `X^b` + `apply_CSUM` powers on the tableau. The opaque `std::function` overload
  throws `std::invalid_argument` for `CLIFFORD` (a black-box function has no Clifford
  decomposition), rather than silently substituting another backend.
- **`QuditGrover`**, **`QuditPhaseEstimation`**: throw `std::invalid_argument`
  when called with `QuditBackend::CLIFFORD`.

### Noise is ignored outside DENSITY_MATRIX

Passing a non-null `noise` pointer to an algorithm with `backend != DENSITY_MATRIX`
is silently ignored. The noise model has no effect on statevector, MPS, or
Clifford simulations.

### MPS oracle fallback

The MPS backend implements `apply_phase_oracle` and `apply_function_oracle` via
a dense statevector round-trip. This is exact but costs O(d^n) memory and time.
For oracles on large systems use the STATEVECTOR or DENSITY_MATRIX backends instead.

### Operand validation

The qudit apply-primitives across all four backends (`QuditStatevector`,
`QuditDensityMatrix`, `QuditMPS`, `QuditCliffordSimulator`) fail loud on a
malformed operand before touching memory. A qudit index outside
`[0, n_qudits)` throws `std::out_of_range`; a structural violation (the two
operands of a two-qudit gate not distinct, or a gate/Kraus matrix that is not
`d^k × d^k`) throws `std::invalid_argument`. Messages carry the primitive name
and the offending value, matching the qubit-layer contract.

### Physical validity

The qudit layer mirrors the qubit layer here too. Every apply-primitive taking
a caller-supplied matrix checks that it is unitary, and the two
`QuditDensityMatrix` Kraus entry points check trace preservation, each under a
trailing `ValidationOptions` defaulting to `Throw` at `1e-12`. The qudit
backends have no circuit-level pre-flight, so the primitive call is where the
check happens. See [validation.md](validation.md).

---

## Usage examples

### Density-matrix BV with depolarizing noise

```cpp
#include "lindblad/algorithms.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"
#include "lindblad/qudit/qudit_backend.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

QuditNoiseModel model;
model.add_depolarizing(0, 3, 0.02);
model.add_depolarizing(1, 3, 0.02);

std::vector<int> secret{2, 1, 0};
auto result = QuditBernsteinVazirani::solve(
    secret, 3, 10, 42,
    QuditBackend::DENSITY_MATRIX, &model);
// result.recovered ≈ {2, 1, 0}  (with high probability at low noise)
```

### MPS QPE

```cpp
#include "lindblad/algorithms.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_gates.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

// Shift gate X in d=3; eigenvalue for k=1 is exp(2πi/3)
auto U   = qudit_gates::shift_matrix(3, 1);
auto psi = /* normalised eigenstate for k=1 */;

auto r = QuditPhaseEstimation::estimate(
    3, 3, U, psi, 42, QuditBackend::MPS);
// r.phase_estimate ≈ 1/3
```

### Clifford BV (prime d only)

```cpp
std::vector<int> secret{1, 2, 0, 1};
auto r = QuditBernsteinVazirani::solve(
    secret, 3, 1, 42, QuditBackend::CLIFFORD);
// r.recovered == {1, 2, 0, 1}  (exact, single shot)
```

---

## Related Algorithm Pages

Each qudit algorithm page documents how to pass `backend` and `noise` to that algorithm's `solve()` or `search()` method, backend-specific behaviour, and worked examples:

- [Qudit Bernstein-Vazirani](../algorithms/bernstein-vazirani.md#quditbernsteinvazirani)
- [Qudit Deutsch-Jozsa](../algorithms/deutsch-jozsa.md#quditdeutschjozsa)
- [Qudit Grover](../algorithms/grover.md#quditgrover)
- [Qudit Phase Estimation](../algorithms/qpe.md#quditphaseestimation)
- [Qudit Simon's Algorithm](../algorithms/simon.md#quditsimonsalgorithm)
- [tests/test_qudit_simulators.cpp](../../tests/test_qudit_simulators.cpp) — full coverage for all four backends and `QuditNoiseModel`
