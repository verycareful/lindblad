# QPE

This page documents `lindblad::algorithms::QPE`.

## Purpose

QPE (Quantum Phase Estimation) estimates the phase of a unitary by applying controlled powers of that unitary to an evaluation register and then using the inverse QFT to decode the phase.

In lindblad, QPE is useful when you already have a unitary circuit and want a direct phase estimate from a measured output circuit.

## Theory Summary

QPE uses two registers:

- an evaluation register of `m` qubits
- a target register holding the unitary being estimated

The usual steps are:

1. Prepare the evaluation register in a uniform superposition.
2. Apply controlled powers of the unitary.
3. Apply the inverse QFT to the evaluation register.
4. Measure the evaluation register.

The measured bitstring approximates the phase of the unitary.

## Required Inputs

- A target `QuantumCircuit` representing the unitary to estimate
- The number of evaluation qubits
- Optional shot count and seed when calling `estimate_phase`

The target circuit must represent a unitary operation on `unitary.n_qubits` qubits.

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

Build a unitary circuit and estimate its phase:

```cpp
using namespace lindblad;
using namespace lindblad::algorithms;

QuantumCircuit unitary(1);
unitary.rz(0.25 * M_PI, 0);

auto circuit = QPE::build_circuit(unitary, 3);
auto phase = QPE::estimate_phase(unitary, 3, 1024, 42);
```

## Header Include Instructions

Use:

```cpp
#include "lindblad/algorithms.hpp"
```

## Simulator Dependencies

QPE uses `StatevectorSimulator` internally to evaluate the controlled-unitary circuit and sample measurement counts.

The implementation in lindblad builds the controlled powers of the unitary explicitly and then calls `measure_all()`, measuring every qubit. Only the evaluation register bits are used to extract the phase.

## Public API Details

### `QPE::build_circuit`

- Takes the target unitary and the number of evaluation qubits
- Returns a `QuantumCircuit` containing the controlled-unitary ladder and inverse QFT
- The resulting circuit includes the evaluation register plus the original target register
- Each CU instruction uses the **LSB convention**: the control qubit occupies `targets[0]`
  and maps to bit 0 of the subspace index; even indices apply identity, odd apply $U^{2^k}$

### `QPE::estimate_phase`

- Builds the QPE circuit
- Calls `measure_all()` (measures all qubits); extracts the evaluation register bits to decode the phase
- Returns the estimated phase as a floating-point value in the range `[0, 1)`

## Example Code

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    QuantumCircuit unitary(1);
    unitary.rz(M_PI / 2.0, 0);

    double phase = QPE::estimate_phase(unitary, 4, 2048, 7);
    std::cout << "estimated phase: " << phase << '\n';
}
```

## Return Values and Outputs

- `build_circuit` returns a circuit with the evaluation and target registers combined
- `estimate_phase` returns a normalized phase estimate as a `double`

## Exceptions and Failure Modes

Common problems include:

- passing a non-unitary or mismatched target circuit
- using too few evaluation qubits for the desired precision
- expecting the estimate to be exact when the phase is not representable at the chosen resolution

## Common Pitfalls

- QPE estimates a phase, not a raw eigenvalue.
- Precision depends directly on the number of evaluation qubits.
- The target circuit must be a unitary operation that can be powered and controlled in the way the implementation expects.

## Testing Notes

At the moment the primary validation is implementation-level coverage through the algorithm source and the general simulator tests used elsewhere in the repo.

## QuditPhaseEstimation

**Purpose:** Estimate the eigenphase φ ∈ [0,1) of a d×d unitary U to precision d^{-m} using m clock qudits.

**d-ary generalization:** Extends QPE from binary (d=2) to d-dimensional quantum systems. Uses the d-ary QFT on the clock register. When d=2, equivalent to standard QPE.

**Setup:** For U\|ψ⟩ = exp(2πiφ)\|ψ⟩, the circuit estimates φ to m base-d digits.

**Circuit** (m+1 qudits: 0..m-1 = clock, m = target):

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | Initialize target | Set target qudit to eigenstate \|ψ⟩ (provided as amplitude vector) |
| 2 | F_d on each clock qudit | Uniform superposition on clock |
| 3 | Controlled-U^{d^j} | Clock qudit j (little-endian) controls U^{d^j} on target |
| 4 | F_d† on each clock qudit | Inverse d-ary QFT decodes clock |
| 5 | Measure clock | phase_digits in base d |

**Phase decoding (little-endian):** φ ≈ Σ_j digit_j · d^{-(j+1)}, where j=0 is the least-significant digit (stride d^0). Equivalently, reading the digits as a d-ary fraction.

**Controlled-U^{d^j} gate:** d²×d² matrix where clock value c applies U^{c·d^j} to the target. Built via `qudit_gates::controlled_power_matrix(d, U, d^j)`.

**Precision:** |φ_estimate − φ_true| < d^{-m}.

**Required Inputs:**
- `m` — number of clock qudits (≥ 1); precision = d^{-m}
- `d` — qudit dimension (≥ 2)
- `U` — d×d row-major unitary matrix (size d*d)
- `eigenstate` — d-element normalised amplitude vector, exact eigenstate of U
- `backend` (optional): `QuditBackend` — simulator to use (default `STATEVECTOR`); `CLIFFORD` throws
- `noise` (optional): `const QuditNoiseModel*` — noise model; only applied with `DENSITY_MATRIX` (default `nullptr`)

**How to Invoke:**
```cpp
#include "lindblad/algorithms.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
using namespace lindblad;
using namespace lindblad::algorithms;

// X = shift_matrix(d,1) has eigenstates with phases k/d
auto U   = qudit_gates::shift_matrix(3, 1);
// Eigenstate for k=1: psi[j] = exp(2*pi*i*j/3) / sqrt(3), phi = 1/3
std::vector<Complex128> psi(3);
for (int j = 0; j < 3; ++j)
    psi[j] = Complex128::exp_i(2 * M_PI * j / 3.0) / std::sqrt(3.0);

auto r = QuditPhaseEstimation::estimate(/*m=*/3, /*d=*/3, U, psi, /*seed=*/42);
// r.phase_estimate ≈ 1.0/3.0  (within 1/27)
```

**Supported d:** Any d ≥ 2. Only a 1-qudit target register is supported.

**Result:**
```cpp
struct Result {
    std::vector<int> phase_digits;   // m measured clock digits, little-endian base d
    double phase_estimate;            // decimal φ ∈ [0, 1)
    int m;
    int d;
};
```

**Exceptions:**
- `std::invalid_argument` if `d < 2`, `m < 1`, `U.size() != d*d`, or `eigenstate.size() != d`
- `std::invalid_argument` if `backend == QuditBackend::CLIFFORD`

**Backend:**

| Backend | Supported | Notes |
|---|---|---|
| `STATEVECTOR` | ✓ | Default. Exact dense statevector. |
| `DENSITY_MATRIX` | ✓ | Full mixed-state; applies `noise` model if provided. |
| `MPS` | ✓ | Tensor-network. |
| `CLIFFORD` | ✗ | Throws `std::invalid_argument`. The controlled-U^{d^j} power gates are not Clifford in general. |

The `noise` argument is applied only in the `DENSITY_MATRIX` path. Only a single-qudit target register is supported (multi-qudit target requires k-qudit controlled operations, a planned extension). See [docs/api/qudit-simulators.md](../api/qudit-simulators.md) for the full backend API reference.

**Common pitfalls:**
- The eigenstate must be normalised and must be an exact eigenstate of U; passing an approximate eigenstate gives probabilistic (not deterministic) results.
- Little-endian clock: phase_digits[0] is the least significant digit. Convert to decimal with Σ_j digit_j/d^{j+1}.

## Related Source Files

- [docs/api/qpe.md](../api/qpe.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/qpe.cpp](../../src/algorithms/qpe.cpp)
- [tests/test_simulators.cpp](../../tests/test_simulators.cpp)
