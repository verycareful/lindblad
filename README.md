# Lindblad
[![CI](https://img.shields.io/github/actions/workflow/status/verycareful/lindblad/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/verycareful/lindblad/actions/workflows/ci.yml)
[![C++](https://img.shields.io/badge/C%2B%2B-23-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.21+-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org/)
[![License: Lindblad v2.3](https://img.shields.io/badge/License-Lindblad%20v2.3-red.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](.)
[![Version](https://img.shields.io/badge/version-R.1.20.4-blue?style=flat-square)](CHANGELOG.md)

> **License Notice:** This software is **proprietary and source-available**. Free for non-commercial and academic use only. Commercial use of any kind requires a separate written license agreement. Private non-commercial redistribution to specific peers/collaborators is permitted under the same license terms, for both unmodified copies (§3.1) and modifications (§3.2 of [LICENSE](LICENSE)). **Public redistribution in any form — including forks, copies, mirrors, package registries, and derivative works — is strictly prohibited without explicit written authorization from the author.** Public GitHub forks are technically permitted by GitHub's platform but are **not licensed** under this agreement for any purpose other than reviewing or submitting contributions via pull request; any other use of a fork constitutes a violation. By submitting any contribution (pull request, code snippet, bug fix, or similar) you grant the author a perpetual, irrevocable license to use and commercialize it while retaining your own copyright — see §6.3 of [LICENSE](LICENSE). See [LICENSE](LICENSE) for full terms — `lindblad.software@proton.me` for licensing inquiries.

Lindblad is a high-performance C++23 quantum computing framework for circuit construction, simulation, noise modeling, transpilation, and variational algorithms. It is distributed as a static core library with optional Python bindings and benchmarking targets.

---

## Contents

- [Documentation](#documentation)
- [Features](#features)
- [Repository Layout](#repository-layout)
- [Build Requirements](#build-requirements)
- [Quick Start](#quick-start)
- [License](#license)

---

## Documentation

| Document | Description |
|---|---|
| [docs/MasterDocumentation.md](docs/MasterDocumentation.md) | Canonical doc-writing guide and recovery plan for future sessions |
| [docs/Architecture.md](docs/Architecture.md) | Subsystem layout and runtime flow |
| [walkthrough.md](walkthrough.md) | Practical end-to-end flow and where to find each subsystem |
| [docs/APIOverview.md](docs/APIOverview.md) | Public header and class index |
| [docs/BuildAndTest.md](docs/BuildAndTest.md) | Platform-specific build and test instructions |
| [docs/Benchmarks.md](docs/Benchmarks.md) | Head-to-head benchmark results vs Qiskit / Qiskit Aer |
| [docs/DevelopmentGuide.md](docs/DevelopmentGuide.md) | Contributor workflow and quality checks |

### Algorithm Pages

| Algorithm | Page | API Reference |
|---|---|---|
| VQE | [docs/algorithms/vqe.md](docs/algorithms/vqe.md) | [docs/api/vqe.md](docs/api/vqe.md) |
| QAOA | [docs/algorithms/qaoa.md](docs/algorithms/qaoa.md) | [docs/api/qaoa.md](docs/api/qaoa.md) |
| MA-QAOA | [docs/algorithms/maqaoa.md](docs/algorithms/maqaoa.md) | [docs/api/maqaoa.md](docs/api/maqaoa.md) |
| QPE | [docs/algorithms/qpe.md](docs/algorithms/qpe.md) | [docs/api/qpe.md](docs/api/qpe.md) |
| Grover | [docs/algorithms/grover.md](docs/algorithms/grover.md) | [docs/api/grover.md](docs/api/grover.md) |
| Deutsch-Jozsa | [docs/algorithms/deutsch-jozsa.md](docs/algorithms/deutsch-jozsa.md) | [docs/api/deutsch-jozsa.md](docs/api/deutsch-jozsa.md) |
| Bernstein-Vazirani (all variants) | [docs/algorithms/bernstein-vazirani.md](docs/algorithms/bernstein-vazirani.md) | [docs/api/bernstein-vazirani.md](docs/api/bernstein-vazirani.md) |
| Simon | [docs/algorithms/simon.md](docs/algorithms/simon.md) | [docs/api/simon.md](docs/api/simon.md) |
| QFT / IQFT / AQFT | [docs/algorithms/qft.md](docs/algorithms/qft.md) | [docs/api/qft.md](docs/api/qft.md) |
| Ising / QUBO | [docs/algorithms/ising.md](docs/algorithms/ising.md) | [docs/api/ising.md](docs/api/ising.md) |
| Dispatch | [docs/algorithms/dispatch.md](docs/algorithms/dispatch.md) | [docs/api/dispatch.md](docs/api/dispatch.md) |
| Shor | [docs/algorithms/shor.md](docs/algorithms/shor.md) | [docs/api/shor.md](docs/api/shor.md) |

### API Reference Pages

| Component | Page |
|---|---|
| QuantumCircuit | [docs/api/circuit.md](docs/api/circuit.md) |
| QASM 2 / QASM 3 parsers and serializers | [docs/api/qasm.md](docs/api/qasm.md) |
| Qudit layer | [docs/api/qudit.md](docs/api/qudit.md) |
| Qudit backend simulators | [docs/api/qudit-simulators.md](docs/api/qudit-simulators.md) |
| Gates | [docs/api/gates.md](docs/api/gates.md) |
| Operators | [docs/api/operators.md](docs/api/operators.md) |
| Simulators | [docs/api/simulators.md](docs/api/simulators.md) |
| Backends | [docs/api/backends.md](docs/api/backends.md) |
| Noise | [docs/api/noise.md](docs/api/noise.md) |
| Transpiler | [docs/api/transpiler.md](docs/api/transpiler.md) |
| Estimator | [docs/api/estimator.md](docs/api/estimator.md) |
| Sampler | [docs/api/sampler.md](docs/api/sampler.md) |
| Mathematical constants | [docs/api/constants.md](docs/api/constants.md) |

---

## Features

### Simulators

- **Statevector** — OpenMP-parallelized full amplitude tracking; exact gate application via dense matrix contraction
- **Density Matrix** — Cache-blocked tensor gate application; exact Kraus operator noise channels
- **Clifford** — Stabilizer tableau simulation with exact GF(2) expectation value tracking; exponential speedup for Clifford circuits
- **MPS** — Boundary contraction marginals; Eigen BDCSVD truncation for low-entanglement states
- **AUTO** — Heuristic backend selection based on qubit count and noise model presence

### Noise and Quantum Information

- Composable noise models: depolarizing, amplitude damping, phase damping, T1/T2 thermal relaxation
- `NoiseModel::from_t1_t2()` for device-realistic per-qubit noise from T1/T2/gate-time specs
- Exact quantum information metrics via Eigen eigendecomposition: Uhlmann-Jozsa fidelity, Wootters concurrence, von Neumann entropy, purity

### Transpiler

- **ZYZ decomposition** and **KAK Weyl-chamber block consolidation** for two-qubit gate compression
- **SABRE layout** — 3-pass bidirectional heuristic for initial qubit placement
- **SABRE SWAP routing** — H_basic + H_extended lookahead cost minimization
- **True IBM heavy-hex topology** coupling maps
- **BasisTranslator** — full CX+U3 decomposition of all standard gates for hardware targeting
- Single-qubit optimization, CX cancellation, commutative gate cancellation, diagonal gate removal, scheduling

### Primitives

- **Estimator** — `gradient()` via parameter-shift rule with 2P evaluations fully parallelized; transpiler-result caching (structure-keyed) to skip redundant SABRE/ZYZ on repeated calls
- **Sampler** — configurable shot-based sampling with noise model injection
- `SparsePauliOp::expectation_value_batch()` for vectorized multi-state evaluation

### Algorithms

| Algorithm | Class | Notes |
|---|---|---|
| VQE | `VQE` | EfficientSU2, RealAmplitudes, TwoLocal ansatz builders; COBYLA/Nelder-Mead/Powell optimizers |
| QAOA | `QAOA` | Physics-informed bitstring selection; `initial_params` in result for trajectory analysis |
| MA-QAOA | `MAQAOA` | Layerwise and joint optimization; orbit symmetry reduction; PI-MA-QAOA beta initialization; QSP initial state; direct statevector evolution (3–5× speedup); extended result with per-layer diagnostics |
| QPE | `QPE` | Exact phase estimation; uses `QFT::build_inverse_circuit` internally |
| Grover | `Grover` | MCX-based oracle; auto iteration count via π/4 √N formula |
| Deutsch-Jozsa | `DeutschJozsa` | Constant vs balanced in one oracle query |
| Bernstein-Vazirani | `BernsteinVazirani` | Standard, recursive, probabilistic (Shukla-Vedula 2023), and distributed variants |
| Simon | `Simon` | Period finding via GF(2) Gaussian elimination |
| Shor | `Shor` | Integer factorisation via QPE-based quantum order finding; classical pre-screening + continued-fraction period recovery; SV/DM/MPS backends |
| QFT | `QFT` | Exact QFT, IQFT, AQFT (Kitaev/Coppersmith); Clifford-simulable for n≤2 or AQFT m=1 |
| IsingHamiltonian | `IsingHamiltonian` | `from_qubo()` QUBO→Ising conversion; `to_sparse_pauli_op()` export |
| Dispatch | `SoftDispatchResult` | Post-process MA-QAOA bitstring distributions into unit-commitment dispatch solutions |

**MA-QAOA extensions:** `orbits_by_power()` utility for automatic Orbit-QAOA setup; qubit-indexed gammas by default (N gammas/layer, matches Qiskit baseline); `term_indexed_gammas = true` for term-indexed ablation path.

### I/O

- OpenQASM 2.0 parsing and export (fully wired)
- Optional Python bindings via pybind11

---

## Repository Layout

```text
include/lindblad/          Public C++ API headers
src/                       Core implementations
  algorithms/              Algorithm implementations (VQE, QAOA, MAQAOA, QPE, QFT, ...)
  simulators/              Statevector, DM, Clifford, MPS simulator backends
  transpiler/              Layout, routing, optimization, scheduling passes
  noise/                   Kraus channels and noise model builders
  primitives/              Estimator and Sampler
  qasm/                    OpenQASM 2.0/3.0 parser
  backends/                LocalBackend wrapper
tests/                     Unit tests (GoogleTest)
benchmarks/                Micro and algorithm benchmarks (Google Benchmark)
bindings/                  Optional Python extension module (pybind11)
docs/
  algorithms/              One page per algorithm family
  api/                     Method and class deep-dives
```

---

## Build Requirements

- CMake 3.21 or newer
- C++23 compiler:
  - GCC or Clang (with OpenMP)
  - MSVC with `/openmp` support
- Git (for CMake FetchContent)

Dependencies fetched automatically at configure time:

| Dependency | Version | License | Use |
|---|---|---|---|
| Eigen3 | 3.4.0 | MPL 2.0 | SVD, eigendecomposition, linear algebra |
| NLopt | 2.7.1 | LGPL 2.1 | Classical optimizer (COBYLA, Nelder-Mead, Powell) |
| GoogleTest | 1.14.0 | BSD 3-Clause | Test suite only |
| Google Benchmark | 1.8.3 | Apache 2.0 | Benchmarks only (optional) |
| pybind11 | 2.12.0 | BSD 3-Clause | Python bindings (optional) |

---

## Quick Start

### Configure and build

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --config Release -j
```

> **Note (MSVC / CMake ≥ 3.30):** Pass `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to work around the vendored NLopt source requiring `cmake_minimum_required(VERSION 3.2)`.

### Run tests

```powershell
ctest --test-dir build --output-on-failure
```

### Build options

| Option | Default | Description |
|---|---|---|
| `LINDBLAD_BUILD_TESTS` | `ON` (top-level), `OFF` (dependency) | Build GoogleTest test suite |
| `LINDBLAD_BUILD_BENCHMARKS` | `ON` (top-level), `OFF` (dependency) | Build Google Benchmark targets |
| `LINDBLAD_BUILD_PYTHON` | `OFF` | Build pybind11 Python extension |
| `LINDBLAD_MARCH_NATIVE` | `OFF` | Compile with `-march=native` (non-distributable, max performance) |

When Lindblad is consumed via CMake `FetchContent`, `LINDBLAD_BUILD_TESTS` and `LINDBLAD_BUILD_BENCHMARKS` default to `OFF` so the test and benchmark targets do not propagate into the parent build.

> **Note (direct clone or GitHub archive download):** The test suite and benchmark sources are part of this repository and are present in any direct clone or release archive. If you download the source directly rather than consuming it through CMake, be aware that the test files (under `tests/`) and benchmark files (under `benchmarks/`) are included. GoogleTest and Google Benchmark are fetched at build time; their license notices are in [NOTICE](NOTICE).

Example:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLINDBLAD_BUILD_BENCHMARKS=ON -DLINDBLAD_BUILD_PYTHON=OFF
```

### Building with Clang (recommended for performance)

For maximum performance on Linux with Clang 18+ and libomp:

```bash
cmake -S . -B build-clang -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLINDBLAD_MARCH_NATIVE=ON \
  -DLINDBLAD_BUILD_PYTHON=OFF
cmake --build build-clang -j$(nproc)
```

Verified with Clang 18.1.3+ and LLVM libomp. Use `-DLINDBLAD_MARCH_NATIVE=ON`
rather than putting `-march=native` in `CMAKE_CXX_FLAGS`: user flags are
emitted before the project's compile options, so the default
`-march=x86-64-v3` would silently override a flag-level `-march=native`.

---

## Status

Active development. The core simulation, transpilation, and algorithm subsystems are production-quality. Remaining work is tracked in the project's TODO list.

---

## License

Copyright © 2026 Sricharan Suresh (github.com/verycareful)

This project is licensed under the **[Lindblad Software License Agreement v2.3](LICENSE)** — source-available, free for non-commercial and academic use. Commercial use requires a separate license agreement.

See [LICENSE](LICENSE) for full terms, [NOTICE](NOTICE) for copyright notice, and [CITATION.cff](CITATION.cff) for citation information.

Licensing inquiries: `lindblad.software@proton.me`
 
