# Noise API Deep Dive

This page documents the public noise API in `lindblad/noise.hpp`.

## Header and Namespace

- Header: `include/lindblad/noise.hpp`
- Namespace: `lindblad`

## `KrausChannel`

Represents a channel as a list of Kraus operators.

Fields:

- `operators`: list of flattened matrices (dimension depends on `n_qubits`)
- `n_qubits`: number of qubits the channel acts on
- Multi-qubit operator matrices follow the project-wide qubits[0]-is-LSB convention (`docs/Architecture.md`, Conventions): bit i of the matrix index is the state of the i-th targeted qubit. The density-matrix backend bridges to its internal addressing automatically.

Helpers:

- `is_valid(atol)`: checks whether sum_k K_k^dagger K_k is identity
- `trace_preserving_error()`: Frobenius norm of the trace-preserving error

## `NoiseChannels`

Factory functions that build common `KrausChannel` instances:

- `depolarizing(p, n_qubits = 1)`
  - General n-qubit Pauli twirl: total error probability p is split uniformly over the 4^n - 1 non-identity Paulis
  - `n_qubits` must be in [1, 6] (the operator count is 4^n; wider registers should compose per-qubit channels); out-of-range sizes and p outside [0, 1] throw
- `amplitude_damping(gamma)`
- `phase_damping(lambda)`
- `thermal_relaxation(T1, T2, gate_time, excited_state_population = 0.0)`
  - Validates `T2 <= 2*T1` and `T1`, `T2` > 0; `gate_time` must be >= 0
  - Clamps excited-state population into [0, 1]
  - Coherences decay by exactly `exp(-t/T2)` (fixed in R.1.12: the pure-dephasing factor previously decayed at half the requested rate for T2 < 2*T1)
- `pauli(px, py, pz)`
  - Validates `px + py + pz <= 1`
- `bit_flip(p)`
- `phase_flip(p)`
- `bit_phase_flip(p)`
- `reset(p0, p1)`
- `coherent_unitary(theta, phi, lambda)`

Input validation notes:

- Most channels do not validate that probabilities are in [0, 1]
- Negative values can produce `sqrt` of a negative number
- `pauli` only checks the sum against 1; it does not clamp negative inputs

## `ReadoutError`

Stores classical assignment error probabilities:

- `prob_meas_0_prep_1` = P(measure 0 | prepared 1)
- `prob_meas_1_prep_0` = P(measure 1 | prepared 0)

Helper:

- `assignment_matrix()` returns the 2x2 assignment matrix

## `NoiseModel`

Attaches noise channels to gate names and qubit patterns.

### Gate errors

- `add_quantum_error(error, gate_name, qubits = {}, after_gate = true)`
  - Adds a `GateError` for the gate name
  - `qubits` empty means all-qubit application
  - `after_gate` defaults to `true` (Kraus channel applied **after** the gate
    unitary). Pass `false` to apply it **before**. Both orderings are honoured
    by `DensityMatrixSimulator`. (R.1.10.7 fixed a silent override that forced
    `after_gate = true` regardless of caller intent.)
- `add_all_qubit_quantum_error(error, gate_name, after_gate = true)` is a
  convenience wrapper forwarding to `add_quantum_error` with an empty qubit
  list. Same `after_gate` semantics.
- `errors_for_gate(gate_name, qubits)`
  - Returns errors with empty `qubits` or an exact qubit list match
  - The qubit list match is order-sensitive

### Readout errors

- `add_readout_error(error, qubit)` stores a readout error per qubit
- Readout errors are stored in the model but are not currently applied by the
  density-matrix simulator path

### Other helpers

- `is_ideal()` returns true when there are no gate or readout errors
- `from_t1_t2(t1, t2, gate_times, gate_qubits)` builds per-gate, per-qubit
  thermal relaxation errors
  - Validates matching lengths and `T2 <= 2*T1` per qubit
  - If a gate is missing from `gate_qubits`, it applies to all qubits
  - Adds one error per (gate, qubit) pair

## Example

```cpp
#include "lindblad/noise.hpp"

using namespace lindblad;

int main() {
    NoiseModel model;
    auto ch = NoiseChannels::depolarizing(0.01);
    model.add_quantum_error(ch, "cx");

    if (model.is_ideal()) return 1;
    return 0;
}
```

## Testing Notes

Relevant tests live in:

- [tests/test_noise.cpp](../../tests/test_noise.cpp)

## Related Pages

- [docs/APIOverview.md](../APIOverview.md)
