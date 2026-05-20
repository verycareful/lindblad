# Lindblad
[![C++](https://img.shields.io/badge/C%2B%2B-23-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.21+-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org/)
[![License: Lindblad v2.1](https://img.shields.io/badge/License-Lindblad%20v2.1-red.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](.)
[![Version](https://img.shields.io/badge/version-R.1.10.0-blue?style=flat-square)](CHANGELOG.md)

> **License Notice:** This software is **proprietary and source-available**. Free for non-commercial and academic use only. Commercial use of any kind requires a separate written license agreement. Private non-commercial redistribution of unmodified copies to peers/collaborators is permitted under the same license terms (see §3.1 of [LICENSE](LICENSE)). **Public redistribution in any form — including forks, copies, mirrors, package registries, and derivative works — is strictly prohibited without explicit written authorization from the author.** Public GitHub forks are technically permitted by GitHub's platform but are **not licensed** under this agreement for any purpose other than reviewing or submitting contributions via pull request; any other use of a fork constitutes a violation. By submitting any contribution (pull request, code snippet, bug fix, or similar) you irrevocably assign full copyright ownership of that contribution to the author — see §6.3 of [LICENSE](LICENSE). See [LICENSE](LICENSE) for full terms — `qpp.support@proton.me` for licensing inquiries.

Lindblad is a high-performance C++23 quantum computing framework for circuit construction, simulation, noise modeling, transpilation, and variational algorithms. It is distributed as a static core library with optional Python bindings and benchmarking targets.

---

## Contents

- [Documentation](#documentation)
- [Release](#release)
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

---

## Release

| Version | Description |
|---|---|
| `R.1.10.0` | Circuit visualiser: `QuantumCircuit::draw(DrawMode, DrawOptions)` with four backends (ASCII, SVG, LaTeX/Quantikz, HTML) sharing one ASAP-packed `CircuitDocument` layout pass. Three-tier gate catalogue: declarative `GateSymbol` table for single-qubit boxes (`gate_symbols.cpp`), declarative `CompositeGate` table for controlled and multi-bullet gates plus the TallBox role for `RXX`/`RYY`/`RZZ`/`RZX`/`ECR` interaction gates (`composite_catalogue.cpp`), and hand-written builders for `BARRIER`/`MEASURE`/`RESET`/`UNITARY` (`gate_builders.cpp`). `DrawOptions` exposes `fold_width`, `show_clbits`, `show_params`, `ascii_safe`, `param_format` (π-snap or raw decimal), and SVG/HTML cell sizing. SVG output is self-contained (inline `<style>`, semantic `.lb-*` classes, `data-gate`/`data-col`/`data-qubits` hooks for downstream interactivity). LaTeX output is a `quantikz` environment (no document shell). HTML output wraps the SVG with hover styling. Python bindings expose `DrawMode`, `ParamFormat`, `DrawOptions`, and `qc.draw()`. Legacy `to_ascii()` now forwards to `draw(DrawMode::ASCII)` for source compatibility. New `docs/api/visualisation.md` covers the full API. Test suite follows in R.1.10.1 |
| `R.1.9.1` | Test suite for the OpenQASM 3.0 parser: 202 new tests across 2 new suites (`QASM3ParserTest`, `ParamExprTest`) covering every parser branch (lexer punctuation/comments/numerics, all stdgate dispatch entries, modifier fast paths and matrix fallback, custom gates with substitution, classical `if`/`else` conditioning, symbolic parameters, peephole optimisation, multi-register offsets, error reporting, round-trips). Four bug fixes uncovered during test bring-up are also included: bracket-less `c = measure q;` classical assignment, `pow(-n)` negative exponents, `input float` parameter names lost across the QuantumCircuit reconstruction in `run()`, and a SEGFAULT on missing-angle rotations like `rx q[0];` (now throws cleanly). 834 tests across 74 suites, all passing |
| `R.1.9.0` | OpenQASM 3.0 parser: `from_qasm3()` now implemented as a `string_view` tokenizer + recursive-descent parser. Covers multi-register declarations, gate modifiers (`ctrl @`, `inv @`, `pow(n) @`, chained), classical `if`/`else` conditioning, user-defined `gate` bodies, `measure`/`reset`/`barrier`, and symbolic `input float` parameters (`ParamExpr` + `QuantumCircuit::bind_parameters()`). Parse-time peephole cancels self-inverse pairs and `pow(0) @` gates. Timing/loop constructs throw with a descriptive error. Test suite follows in R.1.9.1 |
| `R.1.8.2` | Patch: two `int` overflow fixes in `find_order` (`1ULL` shifts); `Shor::cf_convergents` promoted to public static; 8 new tests (cf_convergents direct tests, unitarity check, tighter find_order assertions); 632 tests across 72 suites — all passing |
| `R.1.8.1` | Test suite for `algorithms::Shor` — 30 new tests covering classical pre-screening, exception paths, circuit structure, order-finding, backend parity, and seed reproducibility; 624 tests across 72 suites — all passing |
| `R.1.8.0` | `algorithms::Shor` — integer factorisation via QPE-based quantum order finding with classical pre-screening, continued-fraction period recovery, and GCD post-processing. Supports SV, DM, MPS backends; CLIFFORD unsupported (non-Clifford modular exponentiation). Practical for N ≤ ~100 |
| `R.1.7.8` | Test suite expansion: `test_primitives.cpp` (Estimator, Sampler), `test_dag.cpp` (DAG properties), and `test_qasm_parser.cpp` (QASM2 multi-register support); 594 tests across 71 suites — all passing |
| `R.1.7.7` | `Estimator` hot path: added `StatevectorSimulator::eval_expectation`; eliminates `final_state` copy on variational loop. Removed 3 stale TODO entries. |
| `R.1.7.6` | Structural fix: `SabreLayout::run` and all SABRE internals extracted from `trivial_layout.cpp` into `sabre_layout.cpp`; 556 tests across 67 suites — all passing |
| `R.1.7.5` | Bug fixes: MPS UNITARY gate correctly dispatched via SV fallback (B3); `Estimator` routes through `DensityMatrixSimulator` for noisy/shot-based runs (B5); `DensityMatrixSimulator` `before_gate` noise now applied (B6); `to_qasm2()` emits valid QASM2 for `UNITARY`/`PARAM_*` gates (B7). Regression suite: 14 new tests, 556 total across 67 suites |
| `R.1.7.4` | License v2.1: §14 third-party components; `NOTICE` rewritten with full dep notices (Eigen MPL-2.0 + `EIGEN_MPL2_ONLY`, NLopt LGPL-2.1 relinking note, GoogleTest/Benchmark/pybind11). CMake: 3.21+, `LINDBLAD_BUILD_TESTS` guard (default OFF as dep), googletest fetch gated. Website: `ThirdPartyNotices` section, KEY_TERMS corrected for v2.1 structure |
| `R.1.7.3` | License revised to v2.0: private non-commercial redistribution permitted under same-license; §6 acknowledgment added. Website synced to R.1.7.2 state: QFT added to capability matrix and catalog, qudit suite referenced, "Nine families". `CITATION.cff` description updated |
| `R.1.7.2` | Correctness fixes: `apply_H` conjugation (x←−z, z←x), `apply_CSUM`/`apply_CSUM_dag` phase cross-term (x_c·z_t), `measure_qudit` linear-system solve + tableau collapse, BV Clifford per-qudit snapshots; `apply_to_bra` inlines conj(U); docs updated |
| `R.1.7.1` | Test suite release — `test_qudit_simulators.cpp`: 90 tests across 14 suites covering `QuditDensityMatrix`, `QuditMPS`, `QuditCliffordSimulator`, `QuditNoiseModel`, and backend dispatch for all 5 qudit algorithms; 526/538 passed, 12 failures fixed in R.1.7.2 |
| `R.1.7.0` | Qudit backend simulator suite: `QuditDensityMatrix` (Kraus/Lindblad/noise/partial trace), `QuditMPS` (SVD tensor-network, SWAP chain), `QuditCliffordSimulator` (stabilizer tableau, prime d); `QuditNoiseModel` (depolarizing, amplitude damping, phase damping, Lindblad); backend dispatch (`STATEVECTOR`/`DENSITY_MATRIX`/`MPS`/`CLIFFORD`) wired into all 5 qudit algorithms |
| `R.1.6.1` | Test suite release — qudit algorithm suite: 45 tests (`test_qudit_bv`) + 63 tests (`test_qudit_algorithms`); 108 tests, 448 total passing |
| `R.1.6.0` | Qudit algorithm suite: `QuditBernsteinVazirani`, `QuditDeutschJozsa`, `QuditGrover`, `QuditPhaseEstimation`, `QuditSimon` for any d ≥ 2; `QuditStatevector` / `QuditGates` / `QuditSimulator` layer; fixed `shift_matrix` forward-shift convention and Grover exact auto-iteration formula |
| `R.1.5.1` | Test suite release — feedforward + iterative QFT: 9 suites, 100 tests across all 4 simulators |
| `R.1.5.0` | Feedforward infrastructure + semi-classical (Griffiths-Niu) QFT; standalone `QFT` class (exact/AQFT/IQFT/iterative); `p_if`/`add_if` circuit API; all 4 simulators support classically-conditioned gates |
| `R.1.4.1` | Test suite release — `DistributedBernsteinVazirani` unit tests |
| `R.1.4.0` | New algorithm: `DistributedBernsteinVazirani`; Qudit BV deferred as future work |
| `R.1.3.1` | Test suite release — 5 new test files, 223 tests across 35 suites |
| `R.1.3.0` | Full audit: 15 correctness + 8 performance + 3 doc fixes across all subsystems |

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
  -DCMAKE_CXX_FLAGS="-O3 -march=native -Wno-nan-infinity-disabled" \
  -DLINDBLAD_BUILD_PYTHON=OFF
cmake --build build-clang -j$(nproc)
```

Verified with Clang 18.1.3+ and LLVM libomp. `-Wno-nan-infinity-disabled` suppresses NaN/infinity warnings that are safe under `-ffast-math`.

---

## Status

Active development. The core simulation, transpilation, and algorithm subsystems are production-quality. Remaining work is tracked in local planning files (`TODO.txt` is gitignored).

---

## License

Copyright © 2026 Sricharan Suresh (github.com/verycareful)

This project is licensed under the **[Lindblad Software License Agreement v2.1](LICENSE)** — source-available, free for non-commercial and academic use. Commercial use requires a separate license agreement.

See [LICENSE](LICENSE) for full terms, [NOTICE](NOTICE) for copyright notice, and [CITATION.cff](CITATION.cff) for citation information.

Licensing inquiries: `qpp.support@proton.me`
