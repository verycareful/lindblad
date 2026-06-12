# OpenQASM API Deep Dive

This page documents the OpenQASM import and export surface on
`lindblad::QuantumCircuit`. Two dialects are supported as of R.1.9.0:

- OpenQASM 2.0 — `to_qasm2()` and `from_qasm2()`
- OpenQASM 3.0 — `to_qasm3()` and `from_qasm3()`

The serialisers live in `src/circuit.cpp`. Each parser lives in its own
translation unit under `src/qasm/` and is exposed to `circuit.cpp` through a
single free function (`qasm2_parse_impl`, `qasm3_parse_impl`) so the header
surface stays minimal.

## Header and Namespace

- Header: `include/lindblad/circuit.hpp`
- Namespace: `lindblad`

The four methods are part of the `QuantumCircuit` class:

```cpp
std::string to_qasm2() const;
std::string to_qasm3() const;
static QuantumCircuit from_qasm2(const std::string& qasm);
static QuantumCircuit from_qasm3(const std::string& qasm);
```

## QASM 2.0 Round-Trip

### Export — `to_qasm2()`

- Emits `OPENQASM 2.0;` + `include "qelib1.inc";`
- Declares `qreg q[n_qubits];` and (when present) `creg c[n_clbits];`
- Each instruction is rendered as `name(params)? q[i] (, q[j])* ;`
- `UNITARY` and `PARAM_*` instructions are rendered as `gate '<label>' omitted`
  comments because QASM 2.0 has no native representation for them; the output
  remains valid QASM 2.0

### Import — `from_qasm2()`

- Implemented in `src/qasm/qasm2_parser.cpp` as a two-pass line-oriented parser
- Discovers all `qreg name[N];` and `creg name[N];` declarations in the first
  pass; assigns global qubit/classical indices in declaration order
- Supports `gate name(params) qargs { body }` user definitions with recursive
  inlining and parameter substitution (including `pi` expressions)
- Recognises every gate in the `qelib1.inc` standard library plus the same
  multi-qubit set supported by the circuit (`ccx`, `cswap`, `rxx`, `ryy`,
  `rzz`, `iswap`, etc.)
- Throws `std::runtime_error` when no `qreg` is found
- Throws `std::runtime_error("QASM2Parser: unknown gate '<name>' …")` on any
  gate call whose name is not a built-in and not defined by an in-scope
  `gate` block. (R.1.10.7 removed a legacy "silently skip on miss" concession
  — silent drops produced round-trip mismatches attributed to other
  components. Custom `gate` definitions are still respected.)
- Supports both measurement forms (R.1.12): indexed `measure q[i] -> c[j];`
  and whole-register `measure q -> c;`, which expands to one measurement per
  bit (register sizes must match or the parser throws). `reset q;` expands the
  same way. Unresolvable measure/reset operands throw instead of being
  silently dropped (pre-R.1.12 behaviour lost all measurements of
  register-form files)
- `barrier` honours its operand list (`barrier q[0], r;` mixes indexed bits
  and whole registers); a bare `barrier;` covers the full register
- `if (creg == n) ...` conditionals are NOT supported and surface as an
  unknown-gate error; import feedforward circuits through QASM 3, whose
  parser supports single-bit `if` conditions

## QASM 3.0 Round-Trip

### Export — `to_qasm3()`

- Emits `OPENQASM 3.0;` + `include "stdgates.inc";`
- Declares `qubit[n_qubits] q;` and (when present) `bit[n_clbits] c;`
- Each gate becomes `name(params)? q[i] (, q[j])* ;`
- Measurements are emitted in the `c[i] = measure q[j];` form
- `barrier` and `reset` are emitted with their natural syntax

### Import — `from_qasm3()`

`from_qasm3()` is backed by `QASM3Lexer` + `QASM3Parser` in
`src/qasm/qasm3_parser.cpp`. The parser is production-grade for the
Qiskit-exportable subset of QASM 3.

#### Tokenizer

- `QASM3Lexer::tokenize(string_view)` runs in a single pass
- Tokens hold `std::string_view` slices of the source — zero copies
- The output vector is pre-reserved at `source.size() / 5` (empirical token
  density for QASM 3)
- Identifier scanning accepts any byte ≥ `0x80`, so UTF-8 parameter names
  (`θ`, `φ`) round-trip without special casing
- Comments (`//`, `/* */`), string literals, multi-character operators
  (`->`, `==`), and exponent-form numeric literals are all recognised

#### Parser

- Recursive-descent over the token vector with a single cursor
- First pass discovers register sizes, `input` parameters, and `gate`
  definitions. Second pass emits instructions. This split is necessary
  because the `QuantumCircuit` constructor needs `n_qubits` and `n_clbits`
  up front
- The full standard-gate set is dispatched through a single static
  `unordered_map<string_view, BuiltinSpec>` for O(1) average lookup

#### Supported constructs

- Multiple named registers: `qubit[N] name;`, `bit[N] name;`. Legacy
  `qreg name[N];` and `creg name[N];` are also accepted
- Gate modifiers — `ctrl @`, `inv @`, `pow(n) @` — with arbitrary chaining
  before the base gate name
- The full `stdgates.inc` library plus QASM 3 native names (`U`, `CX`,
  `phase`, `cphase`, `toffoli`, `fredkin`, …)
- User-defined gates: `gate name(params) qargs { body }`. Bodies are stored
  as a vector of parameterised `PreCall` records and re-emitted at every
  call site with deep-copied parameter substitution
- Classical conditioning: `if (c[i] == V) ...` and `else ...`. Single-bit
  conditions stamp the `condition_clbit` and `condition_value` fields on the
  enclosed instructions. `else` is implemented as the complementary value
  (only meaningful for binary classical bits)
- Symbolic parameters: `input float[N] name;` declarations register the
  parameter on the circuit. Angles that reference these names are stored as
  `Instruction::param_exprs` (see below) and resolved later by
  `bind_parameters()`
- `measure` in both forms — `c[i] = measure q[j];` and
  `measure q[j] -> c[i];`
- `reset q[j];` and `barrier q[i], q[j];` (barrier with no arguments
  expands to every qubit in the circuit)

#### Modifier resolution

The parser prefers named fast paths before falling back to matrix synthesis.

Named fast paths:

- `ctrl @ x → cx`, `ctrl @ y → cy`, `ctrl @ z → cz`, `ctrl @ h → ch`,
  `ctrl @ swap → cswap`
- `ctrl @ ctrl @ x → ccx`, `ctrl @ ctrl @ z → ccz`
- `ctrl @ rx(θ) → crx(θ)`, `ctrl @ ry(θ) → cry(θ)`, `ctrl @ rz(θ) → crz(θ)`,
  `ctrl @ p(λ) → cp(λ)`
- `inv @ s → sdg`, `inv @ sdg → s`, similarly `t ↔ tdg`, `sx ↔ sxdg`
- `inv @ <self-inverse> → <self-inverse>` (H, X, Y, Z, CX, CZ, SWAP, …)
- `inv @ rx(θ) → rx(-θ)`, `inv @ ry(θ) → ry(-θ)`, `inv @ rz(θ) → rz(-θ)`
- `pow(n) @ rx(θ) → rx(n·θ)` (and similarly for other rotations)
- `pow(0) @ <anything>` is dropped during modifier resolution
- `pow(even) @ <self-inverse>` collapses to identity (dropped)
- `pow(odd) @ <self-inverse>` collapses to the bare gate

Matrix fallback path (used when no named gate fits):

- Required for combinations like `pow(2) @ s` (= Z) or `inv @ iswap`
- The base 1-qubit matrix is constructed from `build_1q_base(name, params)`
- `inv` becomes a conjugate-transpose (`mat_dagger`)
- `pow(n)` becomes binary exponentiation (`mat_pow` — `O(log n)` matmuls)
- Each `ctrl` extends the matrix block-diagonally (`mat_add_control`)
- The resulting matrix is emitted as a `UNITARY` instruction
- The fallback requires numeric parameters; mixing symbolic angles with the
  fallback throws — bind first

#### Parse-time peephole

The parser maintains a per-qubit history stack of live instruction indices
plus a parallel `cancelled` flag vector. After every emission it checks the
most-recent instruction on the same exact qubit list. If both instructions
have the same self-inverse type, no parameters, identical qubit ordering,
and no classical condition, both are marked cancelled and stripped in a
final sweep. This handles `h; h`, `cx; cx`, `ccx; ccx`, and so on without
mid-vector erases that would invalidate other indices.

#### Errors and unsupported constructs

`QASM3Parser` throws `std::runtime_error` with the offending token and line
number for:

- Unknown gate names
- Mismatched parameter or qubit counts
- Modifier combinations not handled by either fast path or matrix fallback
  (e.g. `ctrl @ rxx`)
- Multi-bit classical register conditioning (`if (c == 3) ...` where `c` is
  more than one bit)
- Timing and control-flow constructs declared out of scope for R.1.9.0:
  `for`, `while`, `def`, `delay`, `stretch`, `box`, `cal`, `defcal`,
  `duration`
- Source missing a qubit register declaration

## Symbolic Parameters — `ParamExpr` and `bind_parameters()`

`ParamExpr` is the expression tree used by `from_qasm3()` to keep symbolic
angles deferred until binding.

```cpp
struct ParamExpr {
    enum class Kind { Literal, Name, BinaryOp };
    Kind kind;
    double literal;
    std::string name;
    char op;                         // '+', '-', '*', '/'
    std::unique_ptr<ParamExpr> lhs;
    std::unique_ptr<ParamExpr> rhs;

    static ParamExpr make_literal(double v);
    static ParamExpr make_name(std::string n);
    static ParamExpr make_binary(char op_char, ParamExpr l, ParamExpr r);

    double eval(const std::unordered_map<std::string, double>& bindings) const;
};
```

- Copy and move constructors deep-clone the owned subtrees so two
  `Instruction`s never share the same `unique_ptr` payload
- `eval()` throws `std::runtime_error` if a referenced name has no binding
  or if a divisor evaluates to zero
- `Instruction::param_exprs` holds the trees alongside the numeric
  `Instruction::params` vector. When `param_exprs` is non-empty the angles
  are symbolic; when it is empty the gate is fully numeric

`QuantumCircuit::bind_parameters(bindings)`:

```cpp
void bind_parameters(
    const std::unordered_map<std::string, double>& bindings
);
```

- Walks every instruction once
- For any instruction with a non-empty `param_exprs`, evaluates each
  expression against `bindings`, populates `params`, then clears
  `param_exprs`
- Merges `bindings` into the circuit's `parameter_bindings` map for
  subsequent queries
- Throws `std::runtime_error` if any referenced parameter is missing
- Skips fully-numeric instructions — no copy, no allocation

## Example — Bell state round-trip

```cpp
#include "lindblad/circuit.hpp"

using namespace lindblad;

QuantumCircuit qc(2, 2, "bell");
qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);

const std::string qasm = qc.to_qasm3();
QuantumCircuit qc2 = QuantumCircuit::from_qasm3(qasm);
```

`qc2` is instruction-equivalent to `qc`.

## Example — VQE ansatz with symbolic parameters

```cpp
#include "lindblad/circuit.hpp"

using namespace lindblad;

const std::string qasm = R"(
    OPENQASM 3.0;
    include "stdgates.inc";
    input float[64] theta;
    qubit[2] q;
    rx(theta) q[0];
    ry(theta + pi/4) q[1];
    cx q[0], q[1];
)";

QuantumCircuit qc = QuantumCircuit::from_qasm3(qasm);
qc.bind_parameters({{"theta", 0.3}});
// qc.instructions now carry literal angles in `params`.
```

## Example — gate modifiers

```cpp
const std::string qasm = R"(
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    ctrl @ x q[0], q[1];                 // emitted as CX
    ctrl @ ctrl @ x q[0], q[1], q[2];    // emitted as CCX
    inv @ s q[0];                        // emitted as SDG
    pow(3) @ inv @ rx(0.2) q[0];         // emitted as RX(-0.6)
)";
QuantumCircuit qc = QuantumCircuit::from_qasm3(qasm);
```

## Common Pitfalls

- `from_qasm3()` only supports single-bit classical conditioning. If the
  source uses `if (c == 3)` with `c` wider than one bit, the parser throws
- The matrix fallback requires purely-numeric parameters. Mixing symbolic
  angles with an unhandled modifier stack throws; bind first
- Self-inverse cancellation requires identical qubit ordering. `cx q[0], q[1]`
  followed by `cx q[1], q[0]` is **not** cancelled (the gates differ)
- The parser preserves the order of instructions Qiskit emitted, including
  `barrier`. If you intend to compare circuits after a round trip, expect
  any peephole-cancelled pairs (e.g. `h; h`) to be absent in the parsed
  circuit
- Both QASM 2 and QASM 3 parsers throw `std::runtime_error` on unknown gate
  calls (no built-in match and no in-scope `gate` definition). The legacy
  QASM 2 silent-skip behaviour was removed in R.1.10.7.

## Related Source Files

- `include/lindblad/circuit.hpp` — `Instruction`, `ParamExpr`,
  `QuantumCircuit::{to_qasm2, to_qasm3, from_qasm2, from_qasm3, bind_parameters}`
- `src/circuit.cpp` — `to_qasm2()`, `to_qasm3()` serialisers and the bridge
  glue for the two parsers
- `src/qasm/qasm2_parser.cpp` — `QASM2Parser` line-oriented parser and
  `qasm2_parse_impl()` bridge
- `src/qasm/qasm3_parser.cpp` — `QASM3Lexer`, `QASM3Parser`, modifier
  resolution, peephole, and `qasm3_parse_impl()` bridge

## Related API Pages

- [docs/api/circuit.md](circuit.md) — full `QuantumCircuit` reference
  including the rest of the `Instruction` field set and
  `assign_parameters()` (the symbolic counterpart for `PARAM_*` gate
  variants used outside QASM)
