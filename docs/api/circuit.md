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
- `param_exprs`: symbolic parameter expression trees (populated by `from_qasm3()`
  when an angle references a named `input` parameter); resolved by
  `QuantumCircuit::bind_parameters()` into `params`
- `matrix`: custom unitary storage for `UNITARY`. In R.1.13 (audit F-18) this is
  a **copy-on-write `CowMatrix`**, not a `std::vector<Complex128>`: the matrix is
  logically immutable, so copying an `Instruction` (per estimator evaluation,
  per transpiler pass) shares one buffer instead of deep-copying $2^k \times 2^k$
  data. Read access mirrors `std::vector` (`size()`, `operator[]`, `data()`,
  iteration) and implicitly converts to `const std::vector<Complex128>&`, so
  existing read call sites are unchanged. To set it, assign a whole vector
  (`inst.matrix = some_vector;`); there is no in-place element mutation.
- `permutation`: basis-index map for `PERMUTATION` (size $2^k$, LSB = `qubits[0]`)
- `label`: custom label (used for unitary naming)
- `condition_clbit` / `condition_value`: classical condition metadata (not enforced by the circuit)
- `schedule_time`: scheduling metadata (set by passes)

Helpers:

- `gate_name()` returns the gate mnemonic (and uses `label` for `UNITARY` /
  `PERMUTATION`; `mcx` / `mcp` for the multi-controlled ops)
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

**Multi-controlled and permutation builders (R.1.13, audit F-7/F-9):**

```cpp
qc.mcx({0, 1, 2}, 3);          // flip qubit 3 when 0,1,2 are all |1> (any control count)
qc.mcp(M_PI / 4, {0, 1, 2});   // phase when 0,1,2 are all |1> (1 qubit == P, 2 == CP)
qc.permute(perm, {0, 1});      // |x> -> |perm[x]> on the target subspace
```

- `mcx(controls, target)` validates each qubit and that no control equals the
  target; stores `[controls..., target]`. `mcp(lambda, qubits)` requires at
  least one qubit. `permute(perm, qubits)` requires `perm.size() == 2^qubits.size()`
  and validates that `perm` is a bijection (a non-bijection is non-unitary and
  throws `std::invalid_argument`).
- These are native in the statevector and density-matrix backends (no dense
  $2^k$ matrix). The MPS backend reduces small MCX to X/CX/CCX and uses the
  bounded statevector fallback for wider MCX / MCP / PERMUTATION. The
  peripheral tooling covers them too: QASM 3 export emits `ctrl(k) @` forms
  (PERMUTATION lowers to gates at export), QASM 2 export throws unless
  `QasmExportOptions::decompose_unrepresentable` is set, JSON round-trips all
  three natively, and the stage-0 `HighLevelDecompose` transpiler pass lowers
  them for routing / basis translation. See
  [Gates API](gates.md), [QASM API](qasm.md), and
  [Transpiler API](transpiler.md).

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

```cpp
void bind_parameters(
    const std::unordered_map<std::string, double>& bindings
);
```

Behavior:

- Walks every instruction; for any with a non-empty `param_exprs`, evaluates
  each `ParamExpr` against the bindings, populates `params`, and clears
  `param_exprs`
- Used after `QuantumCircuit::from_qasm3()` to resolve symbolic angles from
  `input float` parameter declarations
- Throws `std::runtime_error` if a referenced parameter has no binding
- Merges `bindings` into `parameter_bindings` for subsequent queries
- Numeric instructions are left untouched (no copy, no allocation)

### Composition and Transforms

- `compose(other, qubits = {})` appends `other` to the circuit
  - If `qubits` is empty, `n_qubits` and `n_clbits` grow to fit `other`
  - If `qubits` is provided, it must match `other.n_qubits` exactly; `n_clbits`
    grows to fit `other.n_clbits` before instructions are appended
- `inverse()` reverses instruction order and applies gate-specific inverses
  - Skips `MEASURE`, `RESET`, and `BARRIER` (barriers are preserved)
  - `UNITARY` gates are inverted as conjugate-transpose
  - `PARAM_*` symbolic gates are skipped (not invertible until parameters are bound)
  - Throws for any other gate type without an inverse implementation
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

- `to_qasm2(opts)` and `to_qasm3()` emit OpenQASM strings
  - Parameters are emitted only when `inst.params` is populated
  - Param gates should be bound before export to avoid missing parameters
  - `UNITARY` and `PARAM_*` gates have no QASM 2.0 representation; they are
    emitted as `// gate '...' omitted` comments so the output remains valid QASM
  - `MCX` / `MCP` / `PERMUTATION`: QASM 3 emits `ctrl(k) @ x` /
    `ctrl(m-1) @ p(λ)` and always lowers `PERMUTATION` to gates; QASM 2 throws
    by default, or lowers all three when
    `QasmExportOptions::decompose_unrepresentable` is set (see
    [QASM API](qasm.md))
- `from_qasm2()` uses the internal QASM2 parser
- `from_qasm3()` uses the internal QASM3 parser (`QASM3Lexer` + `QASM3Parser`)
  - Covers multi-register declarations, gate modifiers (`ctrl @`, `ctrl(n) @`,
    `inv @`, `pow(n) @`, chained; wide control stacks resolve to first-class
    `MCX` / `MCP`), `stdgates.inc`, user-defined `gate` bodies, classical
    `if`/`else` conditioning, `measure`/`reset`/`barrier`, and symbolic
    `input float` parameters
  - Symbolic parameters are stored as `Instruction::param_exprs` and resolved
    by `QuantumCircuit::bind_parameters(bindings)`
  - Modifier stacks that don't map to a named gate (e.g. `pow(2) @ s`) fall
    back to explicit matrix composition and emit `UNITARY`
  - Parse-time peephole cancels self-inverse pairs (`h; h`, `cx; cx`, …) and
    drops `pow(0) @ <gate>`
  - Unknown gates and unsupported constructs (`for`, `while`, `def`, `delay`,
    `box`, `cal`) throw `std::runtime_error` naming the offending line

### JSON Serialization

- `to_json()` and `from_json()` provide a minimal, zero-dependency format
- Gate strings are stored using `gate_name()`; custom unitaries include matrix
- Conditioning metadata is serialized when present
- `MCX` / `MCP` / `PERMUTATION` round-trip natively (the `permutation`
  basis-index map is a first-class JSON field); JSON is the lossless
  structural format for these ops, whereas QASM 3 lowers `PERMUTATION` to
  gates at export

### Visualisation

`draw(DrawMode, DrawOptions)` returns a circuit diagram in one of four backends:

- `DrawMode::ASCII` : monospaced text grid (terminal-friendly)
- `DrawMode::SVG` : self-contained SVG with class names and `data-` attributes
- `DrawMode::LATEX` : a Quantikz environment for academic papers
- `DrawMode::HTML` : standalone HTML page embedding the SVG with hover styling

See [`visualisation.md`](visualisation.md) for the full reference, including
`DrawOptions` knobs (`fold_width`, `show_clbits`, `param_format`, `ascii_safe`,
`cell_width_px`, etc.).

`to_ascii()` is retained as a thin compatibility wrapper that calls
`draw(DrawMode::ASCII)`; new code should call `draw()` directly.

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
