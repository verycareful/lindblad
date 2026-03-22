# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project uses semantic versioning labels for release identifiers.

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
