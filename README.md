# q++ (qpp)

q++ is a high-performance C++23 quantum computing framework focused on circuit construction, simulation, transpilation, noise modeling, and variational algorithms. The project is structured as a static core library with optional Python bindings and benchmarking targets.

## Release

- Current release: `v1.0.0-alpha`

## Project Scope

The current codebase provides:

- Circuit construction with symbolic and numeric parameters
- Multiple simulator implementations (statevector, density matrix, Clifford, MPS)
- Noise channels and composable noise models
- Quantum information operators and metrics
- Transpiler passes (layout, routing, and optimization)
- Primitives (Estimator and Sampler)
- Algorithms (VQE, QAOA, MA-QAOA, QPE, Grover)
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

## Documentation Index

- `docs/Architecture.md`: architecture, module boundaries, and execution flow
- `docs/BuildAndTest.md`: platform-specific build and test instructions
- `docs/APIOverview.md`: public API map and usage guidance
- `docs/DevelopmentGuide.md`: coding workflow and quality checks

## Licensing

This project is licensed under:

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

See the full license text in `LICENSE`.

## Status

The project already includes a substantial implementation across simulation, transpilation, and algorithms. Remaining work is tracked as local planning and `TODO.txt` is intentionally ignored by git.
