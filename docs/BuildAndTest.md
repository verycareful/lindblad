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
- `benchmarks/bench_qudit_sv.cpp` (R.1.13: qudit statevector kernels, d=2-7)
- `benchmarks/bench_qudit_dm.cpp` (R.1.13: qudit density-matrix kernels)
- `benchmarks/bench_scaling.cpp` (R.1.13: standard circuit across all four backends)

Run from the build output directory, for example:

```powershell
.\bench_gates
.\bench_statevector
.\bench_maqaoa
.\bench_scaling
```

## Comparison Benchmarks (vs Qiskit / Qiskit Aer)

R.1.14 adds a head-to-head suite comparing Lindblad against Qiskit and Qiskit
Aer on a shared, committed QASM2 corpus. Results are published in
[`Benchmarks.md`](Benchmarks.md).

Components:

- `benchmarks/compare/circuits/` — the shared gate-only QASM2 corpus plus
  observable and coupling-graph files (regenerable via `gen_circuits.py`;
  committed for reproducibility)
- `benchmarks/bench_compare_{sv,dm,mps,clifford,transpiler,estimator}.cpp` —
  the Lindblad half (Google Benchmark)
- `benchmarks/bench_validate.cpp` — the Lindblad half of the cross-engine
  result-parity gate
- `benchmarks/compare/aer_bench.py` — the Qiskit/Aer half (mirrored protocol)
- `tools/bench_report.py` — merges both JSON outputs into `docs/Benchmarks.md`
  and enforces the parity gate

Full sequence (Linux/WSL, from the repo root, build directory `build`):

```bash
# 0. one-time: Python side dependencies
pip install -r benchmarks/compare/requirements.txt

# 1. Lindblad timings (one JSON per domain binary)
for b in sv dm mps clifford transpiler estimator; do
  ./build/benchmarks/bench_compare_$b \
    --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
    --benchmark_format=json --benchmark_out=lb_$b.json
done

# 2. Result-parity halves
./build/benchmarks/bench_validate lindblad_validation.json
python3 benchmarks/compare/aer_bench.py --validate --out aer_validation.json

# 3. Qiskit/Aer timings
python3 benchmarks/compare/aer_bench.py --out aer_results.json

# 4. Merge into docs/Benchmarks.md (refuses stale binaries and parity failures)
python3 tools/bench_report.py \
  --lindblad lb_*.json --aer aer_results.json \
  --validate-lindblad lindblad_validation.json --validate-aer aer_validation.json \
  --expect-version R.<current> --note "<compiler, flags, machine>" \
  --out docs/Benchmarks.md
```

Protocol constants (shots, seed, repetitions) and the twin-built DM noise model
must stay identical between `benchmarks/compare_common.hpp`,
`bench_compare_dm.cpp`, and `aer_bench.py`; each file cross-references the
others. Both engines run out of the box: Aer keeps gate fusion and its own
threading, Lindblad keeps its compiled flags.

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

### Only `lindblad_core` builds; no test / benchmark / app targets

- A clean configure that produces only the static library and no
  `lindblad_tests` / benchmark / app executables means the top-level
  `CMakeLists.txt` is missing its `add_subdirectory(tests|benchmarks|apps)`
  blocks (this happened in the R.1.12.2 release; fixed in R.1.13.0). Confirm
  those blocks are present near the end of `CMakeLists.txt`, then reconfigure.

### Test output shows an unexpected version

- The runner prints `Thank you for using Lindblad Quantum Toolkit R.X.X.X` at
  the end. If that label does not match `LINDBLAD_VERSION_LABEL` in
  `CMakeLists.txt`, you are running a STALE binary from an old, separately
  configured build directory, and the results do not reflect your current
  changes. Rebuild the intended directory (or remove it and reconfigure) before
  trusting a run.