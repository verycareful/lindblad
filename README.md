# q++ (qpp)
[![C++](https://img.shields.io/badge/C%2B%2B-23-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](.)
[![Version](https://img.shields.io/badge/version-v1.6.0--alpha-orange?style=flat-square)](CHANGELOG.md)

q++ is a high-performance C++23 quantum computing framework focused on circuit construction, simulation, transpilation, noise modeling, and variational algorithms. The project is structured as a static core library with optional Python bindings and benchmarking targets.

## Release

- Current release: `v1.6.0-alpha` (perf: eliminate expectation_value cloning, parallel run_batch, DensityMatrix cache fix, NUMA first-touch, MPS threshold; fix: from_qasm2 wired, schedule_time field)
- Previous release: `v1.5.4-alpha` (fix: MA-QAOA layerwise logger off-by-one in layer index)

## Project Scope

The current codebase provides:

- Circuit construction with symbolic and numeric parameters
- High-performance exact and approximate simulators:
  - **Statevector**: OpenMP-accelerated full amplitude tracking
  - **Density Matrix**: Cache-blocked tensor gate applications with exact Kraus operator noise
  - **Clifford**: Stabilizer tableau with exact GF(2) expectation value tracking
  - **MPS**: Boundary contraction marginals and Eigen BDCSVD truncation
- Noise channels and composable noise models (including generalized T1/T2 thermal relaxation)
- Exact quantum information metrics via Eigen eigendecomposition (Uhlmann-Jozsa fidelity, Wootters concurrence, etc.)
- Advanced Transpiler passes:
  - ZYZ decomposition and KAK Weyl-chamber block consolidation
  - 3-pass bidirectional SABRE layout heuristic
  - SABRE SWAP routing with H_basic + H_extended lookahead
  - True IBM heavy-hex topology coupling maps
- Primitives (Estimator and Sampler)
- Algorithms (VQE, QAOA, layerwise MA-QAOA, exact QPE, Grover with MCX)
- OpenQASM parsing and export support
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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

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
