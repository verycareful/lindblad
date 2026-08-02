# API Overview

This document summarizes the primary public interfaces declared under `include/lindblad/`.

## Namespaces and Header Conventions

- Primary namespace: `lindblad`
- Algorithm namespace: `lindblad::algorithms`
- Public headers are organized by subsystem and intended to be included selectively.

## Circuit Construction

Header: `include/lindblad/circuit.hpp`

Primary types:

- `Instruction`: normalized gate operation record
- `QuantumCircuit`: circuit container and fluent builder

Key capabilities:

- Add single-, two-, and three-qubit gates
- Add custom unitaries
- Add measure/reset/barrier operations
- Use symbolic parameters and later bind values with `assign_parameters`
- Export/import OpenQASM 2.0 and OpenQASM 3.0; symbolic QASM 3 angles via
  `ParamExpr` are resolved later with `bind_parameters`

Deep dives: [docs/api/circuit.md](api/circuit.md), [docs/api/qasm.md](api/qasm.md), [docs/api/visualisation.md](api/visualisation.md)

### Minimal example

```cpp
#include "lindblad/circuit.hpp"

lindblad::QuantumCircuit bell(2, 2, "bell");
bell.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
```

## Simulators

Headers:

- `include/lindblad/simulators/statevector_sim.hpp`
- `include/lindblad/simulators/density_matrix_sim.hpp`
- `include/lindblad/simulators/clifford_sim.hpp`
- `include/lindblad/simulators/mps_sim.hpp`

Simulator classes and state representations:

- **StatevectorSimulator**: Exact pure-state simulation, $O(2^n)$ space, optimal for 5–25 qubits
- **DensityMatrixSimulator** + **DensityMatrix**: Mixed-state with Kraus noise, $O(4^n)$ space, noisy circuits up to ~10 qubits
- **CliffordSimulator** + **StabilizerState**: Polynomial-time Clifford circuits via stabilizer tableau, unbounded system size
- **MPSSimulator** + **MPSState**: Approximate large-system simulation, $O(n\chi^2)$ space, tunable bond dimension $\chi$

All simulators follow common interface: `Result run(circuit, params)`

Deep dive: [docs/api/simulators.md](api/simulators.md)

Mathematical constants (`PI`, `INV_SQRT2`, ...) live in one header for the whole
library: [docs/api/constants.md](api/constants.md)

### Minimal examples

**Exact statevector**:

```cpp
#include "lindblad/simulators/statevector_sim.hpp"

StatevectorSimulator sim;
auto result = sim.run(circuit, 1024);  // 1024 shots
```

**Noisy via density matrix**:

```cpp
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/noise.hpp"

DensityMatrixSimulator sim;
NoiseModel noise = NoiseModel::from_t1_t2(n_qubits, t1, t2, gate_time);
auto result = sim.run(circuit, noise, 1024);
```

## Transpiler

Headers:

- `include/lindblad/transpiler.hpp`
- `include/lindblad/dag.hpp`

Transpiler classes and types:

- **CouplingMap**: Hardware topology (linear, grid, heavy-hex, all-to-all); connectivity queries and shortest-path
- **DAGCircuit**: Directed acyclic graph representation; dependency analysis and topological operations
- **TranspilationPass**: Abstract pass interface; concrete implementations for layout, routing, optimization
- **PassManager**: Orchestrate pass sequence; preset managers for optimization levels 0–3
- **Layout passes**: TrivialLayout, SabreLayout (heuristic distance minimization)
- **Routing passes**: SabreSwap (greedy with lookahead), StochasticSwap (multi-trial)
- **Basis passes**: BasisTranslator (gate decomposition into hardware-native basis)
- **Optimization passes**: Optimize1qGates (ZYZ consolidation), ConsolidateBlocks (KAK decomposition), CXCancellation, CommutativeCancellation, RemoveDiagonalGatesBeforeMeasure, RemoveResetInZeroState
- **Scheduling passes**: ASAPSchedule, ALAPSchedule (timing assignment)

High-level entry point: `transpile(circuit, coupling_map, basis_gates, optimization_level)`

Deep dive: [docs/api/transpiler.md](api/transpiler.md)

### Minimal example

**Transpile for a linear 5-qubit hardware**:

```cpp
#include "lindblad/transpiler.hpp"

lindblad::QuantumCircuit circuit = ...;
lindblad::CouplingMap coupling = lindblad::CouplingMap::linear(5);
lindblad::QuantumCircuit optimized = lindblad::transpile(circuit, coupling, {"cx", "u3", "rz"}, 2);
```

## Backends

Header: `include/lindblad/backends/local_backend.hpp`

Backend abstraction:

- **LocalBackend**: Unified simulator wrapper with AUTO selection heuristic
- **BackendResult**: Unified result type with counts, timing, and status
- **SimType**: Simulator selection enum (STATEVECTOR, DENSITY_MATRIX, CLIFFORD, MPS, AUTO)
- **Config**: Configuration struct with thread/memory limits and bond dimension

Key features:

- AUTO heuristic selects optimal simulator based on circuit properties and noise
- Transparent integration with noise models (switches to DENSITY_MATRIX when noise configured)
- Thread and memory limits for resource-constrained environments
- Batch execution with parallel circuit distribution
- Timing and metadata tracking

Deep dive: [docs/api/backends.md](api/backends.md)

### Minimal example

**Basic execution with AUTO selection**:

```cpp
#include "lindblad/backends/local_backend.hpp"

lindblad::backends::LocalBackend backend;
lindblad::QuantumCircuit circuit = ...;

auto result = backend.run(circuit, 1024);  // AUTO selects best simulator
for (const auto& [bitstring, count] : result.counts) {
    std::cout << bitstring << ": " << count << "\n";
}
```

## Noise

Header: `include/lindblad/noise.hpp`

Core abstractions:

- `KrausChannel`
- `NoiseChannels` factory-style channel construction
- `NoiseModel` for attaching errors to operations

Noise models are consumed by primitives and simulator paths that support noisy execution.

Deep dive: [docs/api/noise.md](api/noise.md)

## Quantum Information

Header: `include/lindblad/operators.hpp`

Includes:

- Pauli string/operator representations
- Sparse Pauli operators used for Hamiltonians
- Operator algebra helpers and metrics support

Deep dive: [docs/api/operators.md](api/operators.md)

## Transpiler

Header: `include/lindblad/transpiler.hpp`

Contains structures for:

- Coupling map constraints
- Layout and routing utilities
- Optimization passes and pass manager orchestration

Use transpiler interfaces before simulation when target constraints or optimization are required.

## Gates

Namespace: `lindblad::gates`

Header: `include/lindblad/gates.hpp`

Gate operations on `Statevector`:

- **Single-qubit gates**: Pauli (X, Y, Z), Hadamard, phase gates (S, T), sqrt-X, rotations (RX, RY, RZ, P), general unitary (U)
- **Two-qubit gates**: Controlled gates (CX, CY, CZ, CH), swap operations (SWAP, iSWAP), controlled rotations (CRX, CRY, CRZ, CP, CU), Ising interactions (RXX, RYY, RZZ, RZX), echoed cross-resonance (ECR)
- **Three-qubit gates**: Toffoli (CCX), CCZ, Fredkin (CSWAP), Margolus (RCCX)
- **N-qubit unitary**: Arbitrary unitary matrices via `apply_unitary`
- **Multi-controlled and permutation (R.1.13)**: `MCX` (multi-controlled X),
  `MCP` (multi-controlled phase), and `PERMUTATION` (basis-index map) applied
  natively without a dense matrix; used by Grover's diffusion and Shor's
  modular multiplication. See [gates API](api/gates.md).

Performance features:

- SIMD vectorization with 64-byte alignment for AVX-512
- OpenMP parallelization for dimensions ≥ 2^20
- Cache-optimized loop structure for two-qubit gates (hi/lo step decomposition)
- Specialized diagonal paths for phase gates (RZ, P, CZ, CP, RZZ)

Deep dive: [docs/api/gates.md](api/gates.md)

## Primitives

Header: `include/lindblad/primitives.hpp`

Primitive APIs:

- `Estimator`: expectation value computation
- `Sampler`: bitstring sampling

These APIs decouple algorithm logic from backend implementation details.

Deep dives: [docs/api/estimator.md](api/estimator.md), [docs/api/sampler.md](api/sampler.md)

## Problem Representations and Dispatch

Headers:

- `include/lindblad/ising.hpp`
- `include/lindblad/dispatch.hpp`

Main types:

- `lindblad::IsingHamiltonian`
- `lindblad::SoftDispatchResult`

`IsingHamiltonian` is the native QUBO/Ising conversion layer used by optimization algorithms.
`SoftDispatchResult` turns sampled bitstring counts into soft assignments and rounded dispatch outputs.

Deep dives: [docs/api/ising.md](api/ising.md), [docs/api/dispatch.md](api/dispatch.md)

## Algorithms

Header: `include/lindblad/algorithms.hpp`

Main classes:

- `algorithms::VQE`
- `algorithms::QAOA`
- `algorithms::MAQAOA`
- `algorithms::QPE`
- `algorithms::Grover`
- `algorithms::DeutschJozsa`
- `algorithms::BernsteinVazirani`
- `algorithms::RecursiveBernsteinVazirani`
- `algorithms::ProbabilisticBernsteinVazirani`
- `algorithms::Simon`
- `algorithms::QuditBernsteinVazirani`
- `algorithms::QuditDeutschJozsa`
- `algorithms::QuditGrover`
- `algorithms::QuditPhaseEstimation`
- `algorithms::QuditSimon`
- `algorithms::Shor`

Detailed usage notes for these algorithms live under the family pages in [docs/algorithms/](algorithms/).

Deep dives: [docs/api/vqe.md](api/vqe.md), [docs/api/qaoa.md](api/qaoa.md), [docs/api/maqaoa.md](api/maqaoa.md), [docs/api/qpe.md](api/qpe.md), [docs/api/grover.md](api/grover.md), [docs/api/deutsch-jozsa.md](api/deutsch-jozsa.md), [docs/api/bernstein-vazirani.md](api/bernstein-vazirani.md), [docs/api/simon.md](api/simon.md), [docs/api/shor.md](api/shor.md), [docs/api/qudit-simulators.md](api/qudit-simulators.md)

Variational algorithms use `Estimator`/`Sampler` and optimizer settings in their `Options` structures.

## Backend Integration

Header: `include/lindblad/backends/local_backend.hpp`

Provides local backend abstraction to execute circuits while selecting simulator strategy.

## API Stability Notes

- Keep new API additions in headers under `include/lindblad/` and maintain backward compatibility where possible.
- Prefer extending option/result structures over breaking method signatures.