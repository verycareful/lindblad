# Circuit API Deep Dive

This page documents the public circuit construction API in `lindblad::QuantumCircuit`
and the `lindblad::Instruction` record type.

## Header and Namespace

- Header: `include/lindblad/circuit.hpp`
- Namespace: `lindblad`

## `Instruction`

`Instruction` is the stored gate record in a circuit.

Key fields:

- `type`: `Instruction::GateType` enum for the operation
- `qubits`: target qubit indices
- `clbits`: classical bit indices (measure only)
- `params`: numeric parameters for rotation and phase gates
- `param_names`: symbolic parameter names for param gates
- `matrix`: custom unitary storage for `UNITARY`
- `label`: custom label (used for unitary naming)
- `condition_clbit` / `condition_value`: classical condition metadata (not enforced by the circuit)
- `schedule_time`: scheduling metadata (set by passes)

Helpers:

- `gate_name()` returns the gate mnemonic (and uses `label` for `UNITARY`)
- `is_parameterised()` is true for `PARAM_RX/RY/RZ/P/U`
- `is_classical()` is true for `MEASURE` and `RESET`

## `QuantumCircuit`

### Constructors

```cpp
QuantumCircuit();
QuantumCircuit(int n_qubits, int n_clbits = 0);
QuantumCircuit(int n_qubits, int n_clbits, const std::string& name);
```

- Throws `std::invalid_argument` if `n_qubits` or `n_clbits` is negative

### Gate Construction

All gate builders append an `Instruction` and return `*this` for fluent usage.
The implementation validates qubit indices and enforces distinct qubits for
2-qubit and 3-qubit gates.

Examples:

```cpp
QuantumCircuit qc(2, 2);
qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
```

Parameterised (symbolic) gates store parameter names in `param_names` and add
them to the circuit `parameter_names` list:

```cpp
QuantumCircuit qc(1);
qc.rx("theta", 0);
```

### Special Operations

- `measure(qubit, clbit)` validates both indices and adds a MEASURE instruction
- `measure_all()` grows `n_clbits` to `n_qubits` if needed and measures `i -> i`
- `barrier()` with no argument applies to all qubits
- `reset(qubit)` appends a RESET instruction

### Parameter Binding

```cpp
QuantumCircuit assign_parameters(
    const std::unordered_map<std::string, double>& bindings
) const;
```

Behavior:

- Stores bindings in `parameter_bindings`
- Converts parameterised instructions into concrete ones only when all of their
  `param_names` are present in the bindings
- Removes any bound parameter names from `parameter_names`

### Composition and Transforms

- `compose(other, qubits = {})` appends `other` to the circuit
  - If `qubits` is empty, `n_qubits` and `n_clbits` grow to fit `other`
  - If `qubits` is provided, it must match `other.n_qubits` exactly; the indices
    are remapped and appended without additional validation
- `inverse()` reverses instruction order and applies gate-specific inverses
  - Skips `MEASURE`, `RESET`, and `BARRIER` (barriers are preserved)
  - Throws for gate types without an inverse implementation
- `repeat(n)` repeats the instruction list `n` times
- `control(num_ctrl_qubits)` prepends control qubits and builds a controlled
  circuit
  - For single control, maps common gates to their controlled variants
  - For unsupported gates or multiple controls, emits a `UNITARY` that is block
    diagonal; if the gate is not already a unitary, the fallback matrix is the
    identity

### Analysis Helpers

- `depth()` computes logical depth ignoring barriers
- `size()` counts all instructions except barriers
- `count_ops()` returns a map keyed by `gate_name()`
- `num_parameters()` returns the size of `parameter_names`

### Export and Import

- `to_qasm2()` and `to_qasm3()` emit OpenQASM strings
  - Parameters are emitted only when `inst.params` is populated
  - Param gates should be bound before export to avoid missing parameters
- `from_qasm2()` uses the internal QASM2 parser
- `from_qasm3()` throws (parser not yet implemented)

### JSON Serialization

- `to_json()` and `from_json()` provide a minimal, zero-dependency format
- Gate strings are stored using `gate_name()`; custom unitaries include matrix
- Conditioning metadata is serialized when present

### ASCII Visualization

`to_ascii()` returns a text diagram with special handling for:

- `BARRIER` (drawn with vertical separators)
- `MEASURE`
- single-qubit gates
- `CX`, `CZ`, and `SWAP`
- other multi-qubit gates shown as a boxed name on the first qubit

## Example

```cpp
#include "lindblad/circuit.hpp"

using namespace lindblad;

int main() {
    QuantumCircuit qc(2, 2, "bell");
    qc.h(0).cx(0, 1).measure_all();

    auto inv = qc.inverse();
    auto qasm = qc.to_qasm2();
    return qasm.empty() ? 1 : 0;
}
```

## Testing Notes

Relevant tests live in:

- [tests/test_circuit.cpp](../../tests/test_circuit.cpp)

## Related Pages

- [docs/APIOverview.md](../APIOverview.md)
