# Lindblad
[![C++](https://img.shields.io/badge/C%2B%2B-23-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org/)
[![License: Lindblad v1.0](https://img.shields.io/badge/License-Lindblad%20v1.0-red.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](.)
[![Version](https://img.shields.io/badge/version-R.1.2.2-blue?style=flat-square)](CHANGELOG.md)

> **License Notice:** This software is **proprietary and source-available**. Free for non-commercial and academic use only. Commercial use of any kind requires a separate written license agreement. **Redistribution in any form — including forks, copies, and derivative works — is strictly prohibited without explicit written authorization from the author**, regardless of whether the use is commercial or non-commercial. Public GitHub forks are technically permitted by GitHub's platform but are **not licensed** under this agreement for any purpose other than reviewing or submitting contributions via pull request; any other use of a fork constitutes a violation. By submitting any contribution (pull request, code snippet, bug fix, or similar) you irrevocably assign full copyright ownership of that contribution to the author — see §6.3 of [LICENSE](LICENSE). See [LICENSE](LICENSE) for full terms — `qpp.support@proton.me` for licensing inquiries.

lindblad is a high-performance C++23 quantum computing framework focused on circuit construction, simulation, transpilation, noise modeling, and variational algorithms. The project is structured as a static core library with optional Python bindings and benchmarking targets.

## Documentation Map

The documentation is organized so new users can start broad and then drill into the exact API or algorithm they need.

- [docs/MasterDocumentation.md](docs/MasterDocumentation.md) - canonical doc-writing guide and recovery plan for future sessions
- [docs/Architecture.md](docs/Architecture.md) - subsystem layout and runtime flow
- [docs/APIOverview.md](docs/APIOverview.md) - public header and class index
- [docs/BuildAndTest.md](docs/BuildAndTest.md) - configure, build, and test instructions
- [docs/DevelopmentGuide.md](docs/DevelopmentGuide.md) - contributor workflow and quality checks
- [docs/algorithms/bernstein-vazirani.md](docs/algorithms/bernstein-vazirani.md) - Bernstein-Vazirani family, including recursive and probabilistic variants
- [docs/algorithms/vqe.md](docs/algorithms/vqe.md) - VQE optimization workflow and ansatz helpers
- [docs/algorithms/qaoa.md](docs/algorithms/qaoa.md) - QAOA usage, options, and circuit construction
- [docs/algorithms/maqaoa.md](docs/algorithms/maqaoa.md) - MAQAOA usage, variants, and parameter layouts
- [docs/algorithms/qpe.md](docs/algorithms/qpe.md) - QPE phase estimation workflow and outputs
- [docs/algorithms/grover.md](docs/algorithms/grover.md) - Grover oracle usage and iteration control
- [docs/algorithms/deutsch-jozsa.md](docs/algorithms/deutsch-jozsa.md) - Deutsch-Jozsa constant/balanced classification
- [docs/algorithms/simon.md](docs/algorithms/simon.md) - Simon period finding and GF(2) elimination details
- [docs/algorithms/ising.md](docs/algorithms/ising.md) - IsingHamiltonian and QUBO conversion
- [docs/algorithms/dispatch.md](docs/algorithms/dispatch.md) - dispatch post-processing helpers for sampled bitstrings
- [docs/api/vqe.md](docs/api/vqe.md) - VQE options, results, and ansatz generator deep dive
- [docs/api/ising.md](docs/api/ising.md) - IsingHamiltonian fields, conversion, evaluation, and export
- [docs/api/dispatch.md](docs/api/dispatch.md) - SoftDispatchResult fields, rounding, cost estimation, and top-k helpers
- [docs/api/circuit.md](docs/api/circuit.md) - QuantumCircuit and Instruction API details
- [docs/api/operators.md](docs/api/operators.md) - PauliString, SparsePauliOp, Operator, and QuantumInfo helpers
- [docs/api/noise.md](docs/api/noise.md) - Noise channels, NoiseModel, and readout error handling
- [docs/api/estimator.md](docs/api/estimator.md) - Estimator primitive options, caching, and expectation evaluation
- [docs/api/sampler.md](docs/api/sampler.md) - Sampler primitive options and sampling behavior
- [docs/api/gates.md](docs/api/gates.md) - Gates API covering single-qubit, two-qubit, and N-qubit operations
- [docs/api/simulators.md](docs/api/simulators.md) - Simulators (Statevector, Density Matrix, Clifford, MPS) architecture and selection
- [docs/api/transpiler.md](docs/api/transpiler.md) - Transpiler passes (layout, routing, optimization, basis translation, scheduling)
- [docs/api/backends.md](docs/api/backends.md) - LocalBackend simulator wrapper and AUTO selection heuristic
- [docs/api/qaoa.md](docs/api/qaoa.md) - QAOA options, optimizer behavior, and circuit construction
- [docs/api/maqaoa.md](docs/api/maqaoa.md) - MAQAOA options, parameter layouts, and layerwise optimization
- [docs/api/qpe.md](docs/api/qpe.md) - QPE circuit construction, phase extraction, and implementation notes
- [docs/api/grover.md](docs/api/grover.md) - Grover circuit construction, iteration defaults, and search output
- [docs/api/deutsch-jozsa.md](docs/api/deutsch-jozsa.md) - Deutsch-Jozsa circuit construction and classification logic
- [docs/api/bernstein-vazirani.md](docs/api/bernstein-vazirani.md) - Bernstein-Vazirani family API details and result fields
- [docs/api/simon.md](docs/api/simon.md) - Simon API details for circuit construction and period recovery

Planned algorithm pages live under docs/algorithms/ and will be expanded as the public API is documented in detail.

## Release

- Current release: `R.1.2.2` — 3 regression fixes (StatevectorSimulator sentinel, DeutschJozsa ancilla bitstring, BernsteinVazirani ancilla strip); 94/94 tests passing
- Previous release: `R.1.2.1` — test suite run against R.1.2.0; 33 regressions identified (61/94 passing)

## Project Scope

The current codebase provides:

- Circuit construction with symbolic and numeric parameters
- High-performance exact and approximate simulators:
  - **Statevector**: OpenMP-accelerated full amplitude tracking
  - **Density Matrix**: Cache-blocked tensor gate applications with exact Kraus operator noise
  - **Clifford**: Stabilizer tableau with exact GF(2) expectation value tracking
  - **MPS**: Boundary contraction marginals and Eigen BDCSVD truncation
- Noise channels and composable noise models (T1/T2 thermal relaxation, depolarizing, amplitude/phase damping)
  - `NoiseModel::from_t1_t2()` for device-realistic per-qubit noise from T1/T2/gate-time specs
- Exact quantum information metrics via Eigen eigendecomposition (Uhlmann-Jozsa fidelity, Wootters concurrence, etc.)
- Advanced Transpiler passes:
  - ZYZ decomposition and KAK Weyl-chamber block consolidation
  - 3-pass bidirectional SABRE layout heuristic
  - SABRE SWAP routing with H_basic + H_extended lookahead
  - True IBM heavy-hex topology coupling maps
  - `BasisTranslator`: full CX+U3 decomposition of all standard gates for hardware targeting
- Primitives (Estimator and Sampler)
  - `Estimator::gradient()` via parameter-shift rule (2P evaluations fully parallelised)
  - Transpiler-result caching in Estimator (structure-keyed; skips SABRE/ZYZ on repeated calls)
  - `SparsePauliOp::expectation_value_batch()` for vectorised multi-state evaluation
- Algorithms (VQE, QAOA, layerwise MA-QAOA, exact QPE, Grover with MCX, Bernstein-Vazirani family)
  - `IsingHamiltonian` with `from_qubo()` QUBO→Ising conversion and `to_sparse_pauli_op()`
  - `SoftDispatchResult` for post-processing MA-QAOA bitstring distributions into dispatch solutions
  - Orbit-QAOA: symmetry-reduced parameterisation via `MAQAOA::Options::orbit_assignments`
  - **QAOA optimiser enhancements** (v2.2.0): parameter bounds `[-2π, 2π]`, initial step-size 0.3, and `QAOA::Result::initial_params` for trajectory analysis. Bitstring selection ranks by computational-basis cost eigenvalue (physics-informed) rather than sample count alone. Convergence distinguishes true completion from iteration-limit exhaustion.
  - **PI-MA-QAOA initialisation**: `mixer_weights` + `beta_base` in `MAQAOA::Options` for physics-informed beta init (expensive generators → large angle, cheap generators → small); all initial parameters (gammas and betas) are perturbed by a reproducible `U(-0.05, 0.05)` noise seeded by `MAQAOA::Options::seed`, enabling multi-seed landscape exploration
  - `orbits_by_power(powers, tolerance)` utility: assigns orbit indices by power tier for automatic Orbit-QAOA setup
  - **Direct statevector evolution** in `MAQAOA::optimize()`: inner loop bypasses `QuantumCircuit` construction, parameter binding, transpile cache, and instruction dispatch — expected 3–5× wall-time reduction on N=20 layerwise runs
  - **Qubit-indexed gammas by default** (N gammas per layer, matching Python baseline); `term_indexed_gammas = true` opts into the more expressive term-indexed path for ablation
  - Extended `MAQAOA::Result`: `initial_params`, `per_layer_costs`, `layer_nfev`, `wall_time_by_layer`, `wall_time_seconds` for research-grade convergence analysis
- OpenQASM 2.0 parsing and export (fully wired)
- Unit tests and performance benchmarks
- Optional Python bindings through pybind11

## Repository Layout

```text
include/lindblad/                Public C++ API headers
src/                        Core implementations
tests/                      Unit tests (GoogleTest)
benchmarks/                 Micro and algorithm benchmarks (Google Benchmark)
bindings/                   Optional Python extension module
docs/                       Project documentation
  algorithms/               One page per algorithm family
  api/                      Optional method/class deep-dives
```

The detailed documentation pages are intentionally close to the public headers and tests so the examples stay aligned with the implementation.

## Build Requirements

- CMake 3.20 or newer
- C++23 compiler
  - GCC or Clang (OpenMP enabled)
  - MSVC with OpenMP support
- Git (for FetchContent dependencies)

Dependencies are downloaded during configure using CMake FetchContent:

- Eigen3 (v3.4.0)
- GoogleTest
- Google Benchmark (optional)
- pybind11 (optional)
- NLopt

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

- `LINDBLAD_BUILD_BENCHMARKS=ON|OFF` (default: `ON`)
- `LINDBLAD_BUILD_PYTHON=ON|OFF` (default: `OFF`)

Example:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLINDBLAD_BUILD_BENCHMARKS=ON -DLINDBLAD_BUILD_PYTHON=OFF
```

### Building with Clang (Recommended for Performance)

For maximum performance on Linux with Clang 18+ and libomp:

```bash
cmake -S . -B build-clang -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -Wno-nan-infinity-disabled" \
  -DLINDBLAD_BUILD_PYTHON=OFF
cmake --build build-clang -j$(nproc)
```

This builds with:
- Clang 18.1.3+ (proven compatible)
- libomp (LLVM OpenMP runtime)
- `-O3 -march=native` for maximum CPU-specific optimizations
- `-Wno-nan-infinity-disabled` to suppress NaN/infinity warnings safe under `-ffast-math`

- `docs/Architecture.md`: architecture, module boundaries, and execution flow
- `docs/BuildAndTest.md`: platform-specific build and test instructions
- `docs/APIOverview.md`: public API map and usage guidance
- `docs/DevelopmentGuide.md`: coding workflow and quality checks

## Status

The project already includes a substantial implementation across simulation, transpilation, and algorithms. Remaining work is tracked as local planning and `TODO.txt` is intentionally ignored by git.

## License

Copyright © 2026 Sricharan Suresh (github.com/verycareful)

This project is licensed under the **[Lindblad Software License Agreement v1.0](LICENSE)** — source-available, free for non-commercial and academic use. Commercial use requires a separate license agreement.

See the [LICENSE](LICENSE) file for full terms, [NOTICE](NOTICE) for copyright notice, and [CITATION.cff](CITATION.cff) for citation information.
Licensing inquiries: qpp.support@proton.me
