# q++ Architecture

## Design Goals

q++ is designed around three constraints:

- High computational throughput for simulation kernels
- Clear public APIs in headers under `include/qpp/`
- Modular growth for algorithms, transpiler passes, and backends

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

- `include/qpp/types.hpp`
- `include/qpp/statevector.hpp`
- `src/statevector.cpp`

Responsibilities:

- Complex value aliases and numeric utility types
- Statevector storage and low-level operations
- Memory layout and performance-sensitive operations used by simulators

### 2. Gates and Circuit Representation

- `include/qpp/gates.hpp`
- `src/gates/*.cpp`
- `include/qpp/circuit.hpp`
- `src/circuit.cpp`
- `include/qpp/dag.hpp`
- `src/dag.cpp`

Responsibilities:

- Single-, two-, and multi-qubit gate implementations
- `Instruction` representation and gate metadata
- Fluent `QuantumCircuit` API for circuit assembly
- DAG form for compilation and transformation workflows

### 3. Simulators

- `include/qpp/simulators/*.hpp`
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

- `include/qpp/noise.hpp`
- `src/noise/*.cpp`
- `include/qpp/operators.hpp`
- `src/quantum_info/*.cpp`

Responsibilities:

- Kraus channels and predefined error channel families
- Noise model registration and lookup
- Operators, Pauli objects, and metrics used by primitives/algorithms

### 5. Transpiler

- `include/qpp/transpiler.hpp`
- `src/transpiler/**/*.cpp`

Responsibilities:

- Basis translation and gate-level canonicalization
- Layout and routing over coupling constraints
- Local optimization passes and pass sequencing

### 6. Primitives and Algorithms

- `include/qpp/primitives.hpp`
- `src/primitives/*.cpp`
- `include/qpp/algorithms.hpp`
- `src/algorithms/*.cpp`

Responsibilities:

- Primitive interfaces (`Estimator`, `Sampler`) for reusable execution
- Variational and search algorithms built on primitives
- Optimization loop integration through NLopt

### 7. External Interface Layers

- `src/backends/local_backend.cpp`
- `include/qpp/backends/local_backend.hpp`
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

- `qpp_core` static library (always)
- `qpp_tests` executable (always)
- `bench_*` executables (when benchmarks are enabled)
- `qpp` Python extension module (when Python bindings are enabled)

Third-party dependencies are declared centrally in `CMakeLists.txt` and fetched at configure time.

## Extension Strategy

Recommended extension points:

- Add new gates in `src/gates/` and wire through circuit/simulator dispatch
- Add transpiler passes under `src/transpiler/` and register in pass manager
- Add algorithms under `src/algorithms/` using primitive interfaces
- Keep public API declarations in `include/qpp/` minimal and stable