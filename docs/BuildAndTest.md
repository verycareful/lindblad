# Build and Test Guide

## Supported Toolchains

lindblad targets modern C++23 toolchains.

- Windows: MSVC (Visual Studio 2022/26 recommended)
- Linux: GCC or Clang
- CMake: 3.20+

OpenMP support is expected and linked in `lindblad_core`.

## Configure

From repository root:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Important CMake options:

- `LINDBLAD_BUILD_BENCHMARKS` (default `ON`)
- `LINDBLAD_BUILD_PYTHON` (default `OFF`)
- `LINDBLAD_BUILD_COVERAGE` (default `OFF`)

Example with options:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLINDBLAD_BUILD_BENCHMARKS=ON -DLINDBLAD_BUILD_PYTHON=OFF
```

## Build

### Windows (single-config generators)

```powershell
cmake --build build -j
```

### Windows (multi-config generators, example Release)

```powershell
cmake --build build --config Release -j
```

### Linux

```bash
cmake --build build -j$(nproc)
```

## Test Execution

Run all discovered tests:

```powershell
ctest --test-dir build --output-on-failure
```

For multi-config builds, specify configuration:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

## Coverage

`LINDBLAD_BUILD_COVERAGE=ON` compiles `lindblad_core` and the test tree with
`--coverage -O0 -g` on GCC and Clang (the option is ignored on MSVC). Build a
dedicated coverage tree, run the suite, then generate a report with gcovr:

```bash
cmake -B build-cov -DLINDBLAD_BUILD_COVERAGE=ON
cmake --build build-cov -j$(nproc)
ctest --test-dir build-cov --output-on-failure
gcovr -r . --html-details build-cov/coverage.html --txt build-cov/coverage.txt build-cov
```

The coverage build is unoptimised and instrumented, so it runs slower than a
normal build; use it only for reporting, not for benchmarking.

## Benchmark Execution

If `LINDBLAD_BUILD_BENCHMARKS=ON`, benchmark executables are generated from:

- `benchmarks/bench_gates.cpp`
- `benchmarks/bench_statevector.cpp`
- `benchmarks/bench_maqaoa.cpp`

Run from the build output directory, for example:

```powershell
.\bench_gates
.\bench_statevector
.\bench_maqaoa
```

## Python Binding Build

Enable bindings:

```powershell
cmake -S . -B build -DLINDBLAD_BUILD_PYTHON=ON
cmake --build build -j
```

This builds a `lindblad` Python extension module via pybind11.

## Troubleshooting

### Configure fails while fetching dependencies

- Ensure outbound network access for CMake FetchContent Git repositories.
- Remove stale build directory and reconfigure:

```powershell
Remove-Item -Recurse -Force build
cmake -S . -B build
```

### OpenMP link errors

- Confirm compiler OpenMP support is installed/enabled.
- On MSVC, use a Visual Studio toolchain with OpenMP components.

### Tests not found in CTest

- Verify `enable_testing()` and `gtest_discover_tests()` completed during configure.
- Re-run CMake configure before running CTest.