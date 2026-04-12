# q++ (qpp)
[![C++](https://img.shields.io/badge/C%2B%2B-23-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](.)
[![Version](https://img.shields.io/badge/version-v2.2.0--beta-blue?style=flat-square)](CHANGELOG.md)

q++ is a high-performance C++23 quantum computing framework focused on circuit construction, simulation, transpilation, noise modeling, and variational algorithms. The project is structured as a static core library with optional Python bindings and benchmarking targets.

## Release

- Current release: `v2.2.0-beta` (feat: QAOA optimiser bounds [-2π, 2π], initial step-size 0.3, and computational-basis energy ranking for improved bitstring selection)
- Previous release: `v2.1.2-beta` (fix: QAOA seeded parameter init — gamma/beta drawn from U(-0.05, 0.05) via `QAOA::Options::seed`, matching MAQAOA convention; sampler seed propagated)

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
- Algorithms (VQE, QAOA, layerwise MA-QAOA, exact QPE, Grover with MCX)
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
include/qpp/                Public C++ API headers
src/                        Core implementations
tests/                      Unit tests (GoogleTest)
benchmarks/                 Micro and algorithm benchmarks (Google Benchmark)
bindings/                   Optional Python extension module
docs/                       Project documentation
```

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

- `QPP_BUILD_BENCHMARKS=ON|OFF` (default: `ON`)
- `QPP_BUILD_PYTHON=ON|OFF` (default: `OFF`)

Example:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQPP_BUILD_BENCHMARKS=ON -DQPP_BUILD_PYTHON=OFF
```

### Building with Clang (Recommended for Performance)

For maximum performance on Linux with Clang 18+ and libomp:

```bash
cmake -S . -B build-clang -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -Wno-nan-infinity-disabled" \
  -DQPP_BUILD_PYTHON=OFF
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

This project is licensed under the **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)**.
You may use, modify, and distribute this software in accordance with Apache 2.0 terms.

See the [LICENSE](LICENSE) file for full text and [NOTICE](NOTICE) for attribution information.
