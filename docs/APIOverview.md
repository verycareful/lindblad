# API Overview

This document summarizes the primary public interfaces declared under `include/qpp/`.

## Namespaces and Header Conventions

- Primary namespace: `qpp`
- Algorithm namespace: `qpp::algorithms`
- Public headers are organized by subsystem and intended to be included selectively.

## Circuit Construction

Header: `include/qpp/circuit.hpp`

Primary types:

- `Instruction`: normalized gate operation record
- `QuantumCircuit`: circuit container and fluent builder

Key capabilities:

- Add single-, two-, and three-qubit gates
- Add custom unitaries
- Add measure/reset/barrier operations
- Use symbolic parameters and later bind values with `assign_parameters`
- Export/import QASM where implemented

### Minimal example

```cpp
#include "qpp/circuit.hpp"

qpp::QuantumCircuit bell(2, 2, "bell");
bell.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
```

## Simulators

Headers:

- `include/qpp/simulators/statevector_sim.hpp`
- `include/qpp/simulators/density_matrix_sim.hpp`
- `include/qpp/simulators/clifford_sim.hpp`
- `include/qpp/simulators/mps_sim.hpp`

Statevector simulator entry point:

- `StatevectorSimulator::run(const QuantumCircuit&, int shots = 0, uint64_t seed = 0)`

Output includes final state and optional sampled counts.

## Noise

Header: `include/qpp/noise.hpp`

Core abstractions:

- `KrausChannel`
- `NoiseChannels` factory-style channel construction
- `NoiseModel` for attaching errors to operations

Noise models are consumed by primitives and simulator paths that support noisy execution.

## Quantum Information

Header: `include/qpp/operators.hpp`

Includes:

- Pauli string/operator representations
- Sparse Pauli operators used for Hamiltonians
- Operator algebra helpers and metrics support

## Transpiler

Header: `include/qpp/transpiler.hpp`

Contains structures for:

- Coupling map constraints
- Layout and routing utilities
- Optimization passes and pass manager orchestration

Use transpiler interfaces before simulation when target constraints or optimization are required.

## Primitives

Header: `include/qpp/primitives.hpp`

Primitive APIs:

- `Estimator`: expectation value computation
- `Sampler`: bitstring sampling

These APIs decouple algorithm logic from backend implementation details.

## Algorithms

Header: `include/qpp/algorithms.hpp`

Main classes:

- `algorithms::VQE`
- `algorithms::QAOA`
- `algorithms::MAQAOA`
- `algorithms::QPE`
- `algorithms::Grover`

Variational algorithms use `Estimator`/`Sampler` and optimizer settings in their `Options` structures.

## Backend Integration

Header: `include/qpp/backends/local_backend.hpp`

Provides local backend abstraction to execute circuits while selecting simulator strategy.

## API Stability Notes

- Keep new API additions in headers under `include/qpp/` and maintain backward compatibility where possible.
- Prefer extending option/result structures over breaking method signatures.