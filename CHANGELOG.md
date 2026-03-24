# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project uses semantic versioning labels for release identifiers.

## [1.5.1-alpha] - 2026-03-24

### Fixed
- **Build:** OpenMP `aligned` clause in single/two-qubit gates now uses local pointers instead of struct member access (Clang compatibility)
- **Build:** `std::swap` on `std::vector<bool>` elements replaced with manual swap in Clifford simulator (proxy reference fix)
- **Build:** Added missing `<random>` and `<unordered_map>` includes in MPS simulator header
- **Build:** Added missing `apply_two_qubit_gate_adjacent` and `apply_swap_adjacent` declarations in `mps_sim.hpp`

## [1.5.0-alpha] - 2026-03-23

### Performance
- Two-qubit gates rewritten with cache-optimised nested lo/hi-step loop structure — sequential memory access within cache lines
- SIMD vectorisation (`#pragma omp simd aligned`) on all two-qubit gate inner loops (controlled-matrix, controlled-phase, 4-index group gates)
- CZ specialised as diagonal gate — avoids full controlled-matrix path, halves memory traffic
- CRZ specialised with split tgt=0/tgt=1 diagonal loops — avoids branch-per-element
- RZZ parity-split into 4 separate SIMD loops — eliminates per-element parity branch
- Clifford `SDG` direct tableau rule — O(N) single pass replacing 3× `apply_s` calls

### Fixed
- **High:** MPS simulator now decomposes 3-qubit gates (CCX, CCZ, CSWAP, RCCX) into 1Q+2Q sequences instead of silently ignoring them — Grover/MCX circuits now produce correct results on MPS backend
- **Medium:** RCCX matrix in density matrix simulator corrected to match statevector decomposition (phases were wrong on |101⟩, |110⟩, |111⟩ states)
- Removed unused `bit_pos` variable in `apply_unitary` (-Wunused-variable)

## [1.4.0-alpha] - 2026-03-23

### Added
- `QuantumCircuit::control(int num_ctrl_qubits)` — build controlled circuit variants (X→CX, CX→CCX, generic→UNITARY)
- `CommutativeCancellation` transpiler pass — merge/cancel rotation gates through commuting intermediates
- `RemoveDiagonalGatesBeforeMeasure` transpiler pass — strip Z-diagonal gates (RZ, P, T, S, Z, U1) before MEASURE
- `RemoveResetInZeroState` transpiler pass — remove redundant RESET on qubits known to be |0⟩
- `ASAPSchedule` / `ALAPSchedule` transpiler passes — assign time-slot annotations to instructions
- Python binding: `QuantumCircuit.control()`

### Improved
- Preset pass manager: level 1+ includes reset/diagonal cleanup; level 2+ includes commutative cancellation
- Transpiler pipeline now covers all 6 standard stages: unroll, layout, routing, basis translation, optimization, scheduling

## [1.3.0-alpha] - 2026-03-23

### Added
- JSON serialization: `QuantumCircuit::to_json()` / `from_json()` for circuit persistence and interop
- QASM2 parser: custom `gate` definitions (`gate foo(a) p,q { ... }`) with parameter substitution and recursive inlining
- Python bindings: `PauliString` class exposed directly; `SparsePauliOp` now accepts `vector<PauliString>` constructor
- Python bindings: `to_json()` / `from_json()` on `QuantumCircuit`

### Improved
- Distinct-qubit validation on all 2-qubit and 3-qubit circuit construction methods (throws `invalid_argument` on `q1==q2`)
- Documented Pauli qubit ordering convention in `operators.hpp` (MSB-first / big-endian, matching Qiskit)
- Documented `process_fidelity` as squared Hilbert-Schmidt inner product with Nielsen 2002 reference

## [1.2.0-alpha] - 2026-03-22

### Fixed
- **Critical:** `simulate_circuit` moved to public API (was private, caused compilation errors in QPE)
- **Critical:** Gate duplication bug in `ConsolidateBlocks` at optimization level 3
- **Critical:** MPS simulator uses sequential measurement for N>25 (was incorrectly sampling qubits independently)
- **High:** Clifford simulator preserves actual final stabilizer state instead of resetting to |0⟩
- **High:** RESET gate implemented in density matrix simulator (was a no-op)
- **High:** Removed dead O(4^N) memory allocation in `DensityMatrix::apply_gate`
- **High:** `rowmult` phase tracking rewritten per Aaronson-Gottesman 2004 Table 1
- **High:** KAK decomposition searches all permutations for Weyl chamber coordinates
- **High:** `inverse()` correctly handles U2, CU, ISWAP gates; throws for unknown gates
- **High:** RXX gate matrix in MPS simulator had dead code overwriting correct values

### Performance
- DAG `to_circuit()` O(N²)→O(N) via hash map lookup
- DAG `successors()`/`predecessors()` O(E)→O(1) via cached adjacency lists
- DAG `depth()` O(N×E)→O(N+E)
- DAG `front_layer()` O(N²)→O(N)

### Improved
- `-march=native` replaced with `-march=x86-64-v3` baseline; native via `QPP_MARCH_NATIVE=ON`
- `substitute_node` routes edges per-wire instead of all-to-first/all-from-last
- QASM2 parser handles `pi/2`, `2*pi`, `-pi/4` expressions correctly
- Python bindings: `final_state` exposed with `to_numpy()` method
- Version numbers unified across all files

### Added
- Integration tests: SV↔DM equivalence, opt-3 correctness, Clifford state, DM reset, QASM2 round-trip

## [1.1.0-alpha] - 2026-03-19

### Added
- Local simulators: Statevector, Density Matrix, Clifford (stabilizer tableau), and MPS backends
- Transpiler de-simplification: ZYZ/KAK decomposition, SABRE layout/routing, IBM heavy-hex topology
- Noise channels and composable noise models with T1/T2 thermal relaxation
- Quantum information metrics (Uhlmann-Jozsa fidelity, Wootters concurrence)
- Primitives: Estimator and Sampler
- Algorithms: VQE, QAOA, MA-QAOA, QPE, Grover with MCX

## [1.0.0-alpha] - 2026-03-19

### Added

- Professional project documentation set:
  - `README.md`
  - `docs/Architecture.md`
  - `docs/BuildAndTest.md`
  - `docs/APIOverview.md`
  - `docs/DevelopmentGuide.md`
  - `walkthrough.md`
- Official `CC BY-NC 4.0` legal text in `LICENSE` sourced from Creative Commons.
- Repository hygiene files:
  - `.gitignore` with CMake/build/editor ignores
  - `TODO.txt` explicitly ignored as local planning content

### Changed

- CMake project version updated from `0.1.0` to `1.0.0` for the alpha release line.
