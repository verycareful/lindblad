# lindblad Architecture

## Design Goals

lindblad is designed around three constraints:

- High computational throughput for simulation kernels
- Clear public APIs in headers under `include/lindblad/`
- Modular growth for algorithms, transpiler passes, and backends

## Qubit Ordering Convention

lindblad uses **LSB-at-qubit-0** (little-endian) project-wide. This matches Qiskit (our API-parity reference) and the statevector amp-index layout natively.

### Definition

For an n-qubit register holding integer K, K is encoded as:

```text
K = Σ qubit_i · 2^i
```

Qubit 0 contributes the 1's place. Qubit (n-1) contributes the 2^(n-1) place.

### Statevector amplitude index

The simulator stores amplitudes such that bit i of the array index equals the state of qubit i. Therefore `amp[K]` directly represents the basis state for integer K — no bit-reverse needed.

### Measurement bitstring

`QuantumCircuit::measure_all()` maps qubit q to clbit q. The statevector simulator constructs the bitstring with clbit 0 at the rightmost character (string position `n_clbits - 1`). Reading the bitstring as a standard big-endian integer recovers K directly.

### Worked example: n=4, K=5

K = 5 in binary is 0101.

- Qubit states: q_0 = 1, q_1 = 0, q_2 = 1, q_3 = 0
- Statevector: `amp[5] = 1`, all other amplitudes zero
- Bitstring after `measure_all`: `"0101"`. Reading left-to-right: q_3 = 0, q_2 = 1, q_1 = 0, q_0 = 1 → decimal 5.

### Implications for QFT and QPE

`QFT::build_circuit(n, opts)` with `opts.do_swaps = true` implements the LSB-LSB QFT:

```text
QFT|x⟩ = (1/√N) Σ_y exp(2πi xy/N) |y⟩
```

where both x and y are encoded in LSB-at-qubit-0. Specifically, `amp[x] = 1` input yields `amp[y] = (1/√N) exp(2πi xy/N)` output. The underlying textbook H + CP gate sequence operates in qubit-0=MSB internally; `do_swaps=true` wraps it with bit-reversal SWAPs at the input side (forward) or output side (inverse) to expose a uniformly little-endian interface.

`do_swaps=false` exposes the raw H + CP sequence (qubit-0=MSB input / qubit-0=LSB output) for advanced users composing with pre-bit-reversed registers. Algorithms that prepare QPE-style Fourier states from controlled-U^(2^k) on qubit k (which naturally yields `amp[K] = exp(2πi K φ)/√N` with K in LSB convention) must use `do_swaps=true` for both `QFT::build_circuit` and `QFT::build_inverse_circuit`.

`QPE::estimate_phase` and `Shor::find_order` follow this convention end-to-end: the controlled-U sequence prepares the eval register in LSB, the LSB-LSB IQFT inverts to `|m⟩` in LSB, and the bit-extraction loop reads the eval-register slice of the bitstring as a standard little-endian integer (qubit q at bitstring position `total - 1 - q` contributes bit q of m).

### Rule for new algorithms

When you introduce or modify any algorithm that maps qubit states to integer values, add an end-to-end test with a non-symmetric input — a non-zero phase, a non-uniform input, an oracle that selects a non-palindromic marked state. Symmetric tests (phase = 0, uniform inputs, palindromic outputs) cannot distinguish LSB from MSB conventions and will mask convention bugs.

## Conventions

Frozen as of R.1.12 and binding for all code, docs, and tests. Every entry
extends the LSB-at-qubit-0 rule above; where lindblad deviates from Qiskit it
does so deliberately and the deviation is stated.

### Pauli strings (LSB-first)

`PauliString::pauli[q]` acts on qubit q: `pauli[0]` is qubit 0 (least
significant), `pauli[n-1]` is qubit n-1. `"XIZ"` means X on qubit 0, I on
qubit 1, Z on qubit 2.

- DELIBERATE DEVIATION from Qiskit, whose labels put qubit n-1 first.
- Footgun: Pauli strings read in the OPPOSITE direction from measurement bitstrings (whose rightmost character is qubit 0). `"XI"` (X on qubit 0) marks the basis state counted under the key `"01"`.
- `SparsePauliOp::tensor(other)`: `this` occupies the LOW qubits of the result, `other` the HIGH qubits.
- Consumers bound to this rule: `SparsePauliOp::expectation_value` / `to_matrix` / batch, `DensityMatrix::expectation_value_sparse`, `StabilizerState::expectation_pauli`, `IsingHamiltonian::to_sparse_pauli_op`, the Estimator sampling path, QAOA/MAQAOA term readers.

### Multi-qubit matrices (qubits[0] = LSB of the matrix index)

Every externally supplied matrix follows `gates::apply_unitary`: bit i of the
row/column index is the state of `qubits[i]`, so `qubits[0]` is the least
significant bit. This binds `QuantumCircuit::unitary()`, `Instruction::matrix`,
`KrausChannel::operators` (for multi-qubit channels), and `control()`'s
generated matrices. Backends whose internal sub-block addressing is MSB-first
(DensityMatrix, the MPS 2-qubit path) bridge by bit-reversal internally; named
built-in gate matrices inside the simulators are an internal detail and stay
in their builders' frame.

### Controlled-gate operands and the interleaved control layout

Controls come first in the argument list (`cx(control, target)`); in a
generated controlled matrix the controls occupy the LOW index bits, so the
gate block lives on the index slice whose control bits are all 1:
`M[(2^nc - 1) | (r << nc), (2^nc - 1) | (c << nc)] = U[r][c]`. This is the
structure used by `control()`, Shor, and QPE.

### ECR argument order

`ecr(a, b)` binds the FIRST argument to the high bit of the documented ECR
matrix. DELIBERATE DEVIATION from Qiskit: `lindblad ecr(a, b)` equals
`Qiskit ecr(b, a)` (equivalently SWAP * ECR_qiskit * SWAP). All three
simulators implement the same convention.

### Qudit layer (digits LSB-first, mirroring the qubit layer)

Basis index `K = Σ digit_q · d^q` (qudit 0 is the least significant digit).
Subspace matrices follow the qubit rule at general d: in
`apply_2qudit(q0, q1, U)` the FIRST argument is the LEAST significant digit
(`row = x1*d + x0`); `apply_kqudit` generalises with `qudits[i]` at digit
weight `d^i`. `qudit_gates::cadd_matrix` and `controlled_power_matrix` are
built in this convention.

### Coupling maps

A `CouplingMap`'s edge list is literal: an edgeless `CouplingMap(n)` declares
n qubits where NO pair may interact, and routing 2-qubit gates against it
throws. "Unconstrained" is expressed by the absence of a map:
`CouplingMap()` (n = 0), which routing passes skip. Matches Qiskit/tket
semantics.

### Exact execution (shots == 0) and feedforward

`run(qc, shots = 0)` on any simulator returns ONE seeded trajectory:
classical conditions are honoured and MEASURE outcomes are drawn from the
seed and recorded. `eval_expectation` and `Estimator` with `shots == 0`
THROW for circuits containing measurement or classically-conditioned
instructions, because a single trajectory's "exact" expectation would be
statistically misleading. Circuits whose measurements are all terminal are
sampled from a single forward pass (counts keyed by the qubit-to-clbit map).

### Noise channel parameters

`depolarizing(p, n)` distributes total error probability p uniformly over the
4^n - 1 non-identity Paulis. `thermal_relaxation` decays coherences by
exactly `exp(-t/T2)`.

## Layered View

```text
Applications / Integrations
    |
Primitives and Algorithms
    |
Transpiler + Circuit IR
    |
Simulators + Noise + Quantum Info
    |
Core Types and Numeric Kernels
```

## Core Modules

### 1. Core Types and State Data

- `include/lindblad/types.hpp`
- `include/lindblad/statevector.hpp`
- `src/statevector.cpp`

Responsibilities:

- Complex value aliases and numeric utility types
- Statevector storage and low-level operations
- Memory layout and performance-sensitive operations used by simulators

### 2. Gates and Circuit Representation

- `include/lindblad/gates.hpp`
- `src/gates/*.cpp`
- `include/lindblad/circuit.hpp`
- `src/circuit.cpp`
- `include/lindblad/dag.hpp`
- `src/dag.cpp`

Responsibilities:

- Single-, two-, and multi-qubit gate implementations
- `Instruction` representation and gate metadata
- Fluent `QuantumCircuit` API for circuit assembly
- DAG form for compilation and transformation workflows

### 3. Simulators

- `include/lindblad/simulators/*.hpp`
- `src/simulators/*.cpp`

Implementations:

- Statevector simulator
- Density matrix simulator
- Clifford simulator
- MPS simulator

Responsibilities:

- Execute circuits against selected state model
- Provide deterministic and sampled outputs depending on options
- Integrate with noise and measurement flows where applicable

### 4. Noise and Quantum Information

- `include/lindblad/noise.hpp`
- `src/noise/*.cpp`
- `include/lindblad/operators.hpp`
- `src/quantum_info/*.cpp`

Responsibilities:

- Kraus channels and predefined error channel families
- Noise model registration and lookup
- Operators, Pauli objects, and metrics used by primitives/algorithms

### 5. Transpiler

- `include/lindblad/transpiler.hpp`
- `src/transpiler/**/*.cpp`

Responsibilities:

- Basis translation and gate-level canonicalization
- Layout and routing over coupling constraints
- Local optimization passes and pass sequencing

### 6. Primitives and Algorithms

- `include/lindblad/primitives.hpp`
- `src/primitives/*.cpp`
- `include/lindblad/algorithms.hpp`
- `src/algorithms/*.cpp`

Responsibilities:

- Primitive interfaces (`Estimator`, `Sampler`) for reusable execution
- Variational and search algorithms built on primitives
- Optimization loop integration through NLopt

### 7. External Interface Layers

- `src/backends/local_backend.cpp`
- `include/lindblad/backends/local_backend.hpp`
- `bindings/python_bindings.cpp`
- `src/qasm/*.cpp`

Responsibilities:

- Backend selection strategy for local execution
- Optional Python module exposure
- QASM import/export pathway

## Runtime Flow

Typical runtime flow for an algorithm run:

1. User constructs or loads a `QuantumCircuit`.
2. Parameters are bound (if symbolic).
3. Optional transpilation transforms circuit structure.
4. Primitive executes using a selected simulator backend.
5. Results are post-processed by algorithm logic.

## Build and Dependency Model

The top-level CMake project produces:

- `lindblad_core` static library (always)
- `lindblad_tests` executable (always)
- `bench_*` executables (when benchmarks are enabled)
- `lindblad` Python extension module (when Python bindings are enabled)

Third-party dependencies are declared centrally in `CMakeLists.txt` and fetched at configure time.

## Extension Strategy

Recommended extension points:

- Add new gates in `src/gates/` and wire through circuit/simulator dispatch
- Add transpiler passes under `src/transpiler/` and register in pass manager
- Add algorithms under `src/algorithms/` using primitive interfaces
- Keep public API declarations in `include/lindblad/` minimal and stable