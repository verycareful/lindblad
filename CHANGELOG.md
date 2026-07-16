# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project uses semantic versioning labels for release identifiers.

## [R.1.15.1] - 2026-07-12

Test-suite release for the R.1.15.0 transpiler correctness wave (`.1` slot:
tests only). Pins the contracts behind #47/#48/#49 so any future routing,
preset-composition, or basis-translation change that breaks them fails in
ctest. Also corrects one stale doc comment: `QuantumCircuit::mcx()` claimed
the transpiler decomposes MCX to a CX/CCX ladder, which is not implemented --
`BasisTranslator` throws on MCX under a non-empty basis and SABRE routes it
only when every wire pair is already adjacent.

### Tests

- `tests/test_r1151_routing_regression.cpp` -- 21 tests across five suites:
  - Expansion (#47): circuits SMALLER than the coupling map route on
    line/grid/heavy-hex(27) at every optimisation level with output width
    equal to the device width and coupling-legal 2q gates. At levels 0-1
    the contract is pinned EXACTLY at the unitary level:
    `U_routed x U_original^dagger` must be a phase-times-permutation matrix
    (routing only inserts SWAPs; it never changes physics). At all levels a
    sorted-probability-spectrum relabeling check covers SabreLayout, which
    conjugates by its wire relabeling rather than left-multiplying. A
    NON-symmetric basis state (K=5) prepared behind routing-forcing
    DIAGONAL cz gates keeps its exact clbit key "0101" end-to-end at every
    level (per the project convention rule, symmetric outcomes would mask
    ordering bugs), a GHZ superposition samples only its two keys, and
    circuits wider than the device throw from `transpile()` and from each
    layout/routing pass individually.
  - Determinism: transpiling the same input twice yields field-identical
    instruction streams across three topologies and all four levels.
  - Presets (#48): the stage table is pinned by pass names -- exactly one
    layout pass and ONE routing pass per level, `SabreLayout` first at
    level >= 2, `BasisTranslator` composed iff `basis_gates` is non-empty
    and always last; `optimization_level` outside 0..3 throws from both
    `preset_pass_manager` and `transpile()`; an unroutable circuit
    (disconnected coupling islands) throws instead of hanging, and at
    level >= 2 the error message's "SABRE layout" prefix proves the layout
    stage now runs first.
  - initial_layout: a valid layout is honoured (zero SWAPs when the
    interaction is adjacent under it); a PARTIAL layout on an expanded DAG
    is completed deterministically with the unused physical slots in
    ascending order (exact CX placement pinned); out-of-range, negative,
    duplicate, and oversized layouts throw.
  - Basis (#49): with a non-empty basis the transpiled output contains
    ONLY basis gates at every level, stays hardware-legal after
    translation, and preserves semantics; the emitted 1q type follows the
    caller's u/u3 naming; native gates pass through untouched;
    untranslatable inputs throw naming the gate (`mcx` -- with the
    empty-basis no-throw counterpart pinning "translation is composed iff
    basis_gates is non-empty" -- `unitary`, an unbound symbolic `rx`, and
    a basis from which cx+u3 are unreachable); classical conditions are
    propagated onto every decomposed instruction, and a feedforward
    circuit (measure + conditional X) samples its deterministic key
    through the full pipeline at every level.
  Deliberately NO hardcoded golden SWAP counts: determinism is asserted by
  double-run stream equality and correctness by unitary/measurement
  semantics; golden counts would pin heuristic scores that are not part of
  the public contract.
- `tests/CMakeLists.txt` -- suite registered in `LINDBLAD_TEST_SOURCES`.

### Results

- 1880 tests across 150 suites -- all passed (23.7 s, WSL/Clang
  `-march=native`).

## [R.1.15.0] - 2026-07-12

Transpiler correctness wave: fixes the three defects surfaced by the R.1.14
comparison suite -- SABRE frozen-slot (#47), cumulative preset routing order
(#48), silently ignored `basis_gates` (#49) -- plus the fail-loud gaps found
in their blast radius.
The comparison benchmark drops its R.1.14 workarounds, so the transpiler
domain now measures the full pipeline end-to-end against Qiskit.

### Fixed

- SABRE frozen-slot defect (#47): both layout passes now expand the circuit
  to device width by identity embedding (`TrivialLayout` and `SabreLayout`
  share the same output invariant, `n_qubits == n_physical_qubits`), so
  circuits SMALLER than the coupling map are routable -- every physical slot
  holds a (possibly idle) logical wire and idle-wire SWAPs are legal, as in
  Qiskit after ancilla allocation. Previously the occupied slot set was
  frozen at the initial layout image while the heuristic scored distances
  through slots that could never be entered; on degree-sparse maps
  (heavy-hex) routing thrashed until the SWAP-budget guard threw a
  misleading "disconnected components" error.
- `transpile()` / the preset pipelines honour `basis_gates` (#49):
  `BasisTranslator` is composed as the FINAL stage of every level when
  `basis_gates` is non-empty (it was composed into no level, making the
  parameter a silent no-op). The translator now verifies its output -- every returned gate is in
  the target basis or `std::invalid_argument` is thrown naming the offending
  gate (`MCX`/`MCP`/`PERMUTATION` until their lowering ships, `UNITARY`,
  unbound parameterised gates, and bases that omit the cx+u3 targets).
- `BasisTranslator` silently DROPPED classical conditions when decomposing a
  conditioned gate (latent until composition). Conditions are now propagated
  onto every emitted instruction -- exact, since the decompositions are
  unitary-only and never modify clbits.
- `SabreLayout` could loop forever on unroutable input: its internal SABRE
  search lacked the SWAP-budget guard `SabreSwap` already had. It matters now
  that `SabreLayout` runs first at level >= 2; unroutable circuits throw
  `std::runtime_error` on both paths.
- Undefined behaviour on circuits wider than the device (out-of-range
  distance-matrix indexing) in both layout passes and `SabreSwap`: now a
  `std::invalid_argument` with the circuit and device sizes.
- `preset_pass_manager` accepted any level: -1 returned an EMPTY manager
  (silent identity transpile) and >= 4 silently behaved as 3; levels outside
  0..3 now throw `std::invalid_argument`.
- `SabreSwap` silently ignored a wrong-sized `ctx.initial_layout` and
  produced undefined behaviour on out-of-range or duplicate entries; layouts
  are now validated (throws) and a partial layout is completed
  deterministically with the unused physical slots in ascending order.
- `tools/bench_report.py` is now tracked. R.1.14.0 shipped public
  documentation (`docs/BuildAndTest.md`, the generated `docs/Benchmarks.md`)
  and R.1.14.1 a committed test suite (`tests/tools/test_bench_report.py`
  plus its ctest registration) that reference the script, but the entire
  `tools/` directory was gitignored -- public clones could not regenerate
  the benchmark report and the referenced script did not exist for them.
  That was a mistake; the report generator is part of the published
  benchmark workflow and now ships with it. The rest of `tools/` remains
  maintainer-local by design.

### Changed

- Preset levels compose per STAGE, non-cumulatively (#48): one layout choice,
  one routing pass per level. Level >= 2 uses `SabreLayout` in the layout slot
  instead of appending after the level-0 `TrivialLayout` + `SabreSwap` block,
  which had routed the unexpanded DAG first and forced SABRE layout to
  re-route an already-SWAP-laden circuit. Basis translation, when composed,
  is always last -- the optimisation passes are basis-oblivious
  (`Optimize1qGates` emits U/RY/RZ, `ConsolidateBlocks` emits RXX/RYY/RZZ),
  so any earlier slot would break the basis guarantee they follow.
- Transpiled output width now equals the device width when a coupling map is
  present (Qiskit parity); measurement/clbit mapping is unaffected.
- `BasisTranslator` emits the `u` gate type when the basis names `u` without
  `u3` (same 3-parameter gate, caller's naming).
- `PassManager`, `preset_pass_manager`, and `transpile()` moved from
  `optimize_1q.cpp` to the new `src/transpiler/preset_pass_manager.cpp`
  (pipeline composition is not a 1-qubit-optimisation concern); no behaviour
  change from the move itself.
- `benchmarks/bench_compare_transpiler.cpp` / `aer_bench.py`: the R.1.14
  workarounds are reverted -- the transpiler domain now routes n=22 circuits
  onto the 25/27-slot maps (exercising the expansion path) and both engines
  translate into the same `{cx, u3}` basis, so the quality counters compare
  CX against CX across the full pipeline.
- `docs/api/transpiler.md` -- affected sections rewritten: stage-table preset
  composition, shared layout invariant, the `BasisTranslator` contract with
  the real supported/throwing decomposition lists (the old text wrongly
  claimed MCX support), `transpile()` output-width and basis semantics,
  `initial_layout` validation, and pitfalls.
- `docs/BuildAndTest.md` -- documented the benchmark cadence policy: the
  comparison suite is NOT rerun for every release; `docs/Benchmarks.md`
  carries the version stamp of the run that produced it, which may lag the
  current release.

## [R.1.14.1] - 2026-07-10

Test-suite release for the R.1.14.0 comparison benchmark suite (`.1` slot:
tests only). Guards the committed corpus, the loaders every benchmark binary
depends on, and the report generator, so a regression in any of them fails in
ctest instead of silently invalidating published numbers. Also corrects a
figure in the R.1.14.0 entry below: the committed corpus holds 40 gate-only
QASM2 circuits (47 files in total, including the observable and
coupling-graph files).

### Tests

- `tests/test_r1141_corpus.cpp` -- structural: every committed corpus circuit
  parses via `from_qasm2`, is gate-only (measurement is appended by each
  engine at run time, never stored), and declares the qubit count its
  filename names (grover files declare 2s-3 for search width s); the
  benchmark-registered file inventory exists; `measure_all` appends exactly
  one measurement per qubit; missing corpus files fail loudly with the
  regeneration hint; coupling edge lists load symmetrised and connected with
  exact node and edge counts; Heisenberg observable files carry the exact
  3(n-1)+n term structure and reproduce the analytic ground-state expectation.
- `tests/test_r1141_corpus_semantics.cpp` -- behavioural anchors against
  independent references: grover_n8 concentrates above 90% on its
  NON-symmetric marked state with all v-chain ancillas restored to |0> (the
  same anchor the cross-engine parity gate relies on); qft_n8 matches its
  closed-form product state on a non-symmetric input, computed from first
  principles in the test (the corpus QFT is the ascending-loop variant,
  deliberately anchored to its own closed form rather than to
  `QFT::build_circuit`, a different operator); clifford_n8 runs on the
  tableau backend and agrees with the statevector backend within the
  sampling-noise total-variation bound.
- `tests/tools/test_bench_report.py` -- nine cases for `tools/bench_report.py`
  against synthetic fixtures: end-to-end exit codes (clean run, parity
  failure exits 2 and stamps the warning banner, stale version exits 1
  without a report), iteration-fallback median, unpaired-workload
  placeholders, and direct unit tests of the total-variation distance and
  the PASS/WARN/FAIL threshold bands.
- `tests/CMakeLists.txt` -- registers the two gtest suites (benchmarks
  include dir + corpus path definition for the test target) plus two new
  ctest entries: `bench_report_tools` (whenever a Python3 interpreter is
  found) and `bench_validate_smoke` (runs the result-parity validator when
  benchmarks are enabled).

### Results

- 1859 tests across 145 suites -- all passed (22.0 s, WSL/Clang
  `-march=native`), plus `bench_report_tools` (9 Python cases) and
  `bench_validate_smoke` at the ctest level.

## [R.1.14.0] - 2026-07-10

Head-to-head comparison benchmark suite: Lindblad vs Qiskit / Qiskit Aer on a
shared QASM2 corpus, with a generated, correctness-gated results page. Both
engines run natively (Google Benchmark vs a mirrored-protocol Python harness)
at out-of-box settings; no timing table is published unless both engines
provably compute the same answers.

### Added

- `benchmarks/compare/` -- the shared corpus and the Qiskit half of the suite:
  `gen_circuits.py` (deterministic, seeded generator) plus **40(edited in R.1.14.1)** committed
  gate-only QASM2 circuits (layered scaling, QFT, QV-style random, ccx-lowered
  Grover with a non-symmetric marked state, Clifford ladders, measure-free
  ansatz), Heisenberg observable files (Lindblad LSB-first Pauli order,
  documented for Qiskit label reversal), coupling-graph edge lists (27-qubit
  line, 5x5 grid, 27-qubit heavy-hex) consumed by BOTH engines, and
  `aer_bench.py` (mirrored warmup/repetition/median protocol, validation mode,
  per-domain filters) with `requirements.txt`.
- `benchmarks/bench_compare_{sv,dm,mps,clifford,transpiler,estimator}.cpp` --
  the Lindblad half (Google Benchmark, JSON output, corpus path baked in via
  `LINDBLAD_BENCH_QASM_DIR`); registered in `benchmarks/CMakeLists.txt`.
- `benchmarks/bench_validate.cpp` -- cross-engine result-parity gate:
  8192-shot count distributions compared by total-variation distance against
  a sampling-noise-aware threshold, estimator expectation agreement to 1e-6,
  and a `LINDBLAD_VERSION_LABEL` stamp so stale binaries are refused. The
  Grover entry peaks on a non-symmetric marked state, so the gate doubles as
  an end-to-end qubit-ordering convention check between the engines.
- `tools/bench_report.py` -- merges both engines' JSON into
  `docs/Benchmarks.md` with per-workload speedups and environment capture;
  exits nonzero and stamps a warning banner on any parity failure.
- `docs/Benchmarks.md` -- generated results page. First recorded run: 49/49
  workloads paired, parity all-PASS (Ryzen 9 7900X, WSL/Clang,
  `-march=native`; qiskit 2.5.0, qiskit-aer 0.17.2).

### Changed

- `docs/BuildAndTest.md` -- new "Comparison Benchmarks (vs Qiskit / Qiskit
  Aer)" section with the full command sequence; benchmark inventory updated.
- `README.md` -- `docs/Benchmarks.md` added to the documentation table.

### Known Issues

- The transpiler comparison is constrained to circuits sized exactly to their
  coupling map, compared routing-only (no basis translation on either side):
  routing circuits SMALLER than the map trips a SABRE swap-candidate defect
  (the occupied physical set is frozen at the initial layout, and the distance
  heuristic thrashes on sparse maps until the SWAP-budget guard throws), and
  `transpile()` currently does not apply `basis_gates` (`BasisTranslator` is
  not composed into any preset level). Filed as GitHub issues; both target a
  dedicated fix release with a routing regression suite, after which the
  benchmark constraints will be lifted.

### Results

- 1849 tests across 143 suites -- all passed (22.2 s, WSL/Clang
  `-march=native`). No test changes in this release.

## [R.1.13.1] - 2026-07-08

Test-suite release for the R.1.13.0 performance wave (`.1` slot: tests only; no
feature or fix code beyond two stale test assertions). Ten new suites assert the
R.1.13.0 changes against independent brute-force references over non-symmetric
inputs, so a convention or index bug cannot hide behind a symmetric test.

### Tests

- `test_r1131_structured_ops.cpp` — MCX/MCP/PERMUTATION: statevector kernels vs
  brute force, simulator dispatch, density-matrix native path, MPS fallback and
  the <= 2-control gate ladder, builder field population, validation throws,
  inverse (MCX self-inverse, MCP phase negation, PERMUTATION map inversion),
  `control()` of MCX equals CCX, and the QASM2/QASM3/JSON export loud-throw contract.
- `test_r1131_cow_matrix.cpp` — CowMatrix: read API, implicit const-vector view,
  copies share one buffer, assignment rebinds without touching other handles,
  value equality across distinct buffers, and that copying a circuit does not
  deep-copy the gate matrix.
- `test_r1131_estimator_grouping.cpp` — `group_pauli_terms`: grouped and ungrouped
  both converge to the exact expectation, each path is seed-deterministic, and a
  diagonal observable is exact under grouping.
- `test_r1131_mps.cpp` — `svd_method` defaults to Jacobi, the two-site GEMM
  contraction (adjacent and SWAP-chained) matches the statevector, GHZ sampling
  distribution, and BDC selection emits the broken-BDCSVD warning.
- `test_r1131_dm.cpp` — density-matrix rework: `apply_kraus` out-of-place vs an
  embedded sum-of-Kraus reference, per-shot buffer-reuse determinism, folded-phase
  `expectation_value_sparse` (including Y terms), `apply_permutation`, and
  `apply_mcp_phase`.
- `test_r1131_kernels.cpp` — RCCX single-pass vs the density-matrix dense matrix,
  RESET and MEASURE collapse, and both `apply_unitary` work-shape dispatch branches.
- `test_r1131_clifford.cpp` — terminal-measurement fast path vs the statevector
  sampler, `is_clifford` classification, and the reset general path.
- `test_r1131_qudit.cpp` — parallel `apply_1qudit` / `apply_2qudit` / `apply_kqudit`
  vs a serial reference above the OpenMP threshold, `QuditMPS::measure` sampler vs
  the dense distribution, and the qudit BDC warning.
- `test_r1131_algorithms.cpp` — Simon batch and per-sample recovery plus seed
  determinism, Grover MCX diffusion, and Shor's PERMUTATION oracle on the MPS
  fallback backend (8-qubit exact) plus `factorize(15)`.
- `test_r1131_transpiler.cpp` — `CouplingMap::distance_matrix` (BFS) vs known and
  shortest-path distances, and SABRE routing legality on line and grid maps.

### Fixed

- `tests/test_shor.cpp`: the two oracle tests still asserted the pre-R.1.13.0
  dense UNITARY oracle. `CircuitContainsUnitaryGates` becomes
  `CircuitContainsPermutationOracles` (exactly `n_eval` PERMUTATION and zero
  UNITARY instructions), and the now-vacuous `UnitaryGatesAreUnitary` becomes
  `PermutationOraclesAreBijections` (each oracle map is a size-`2^(1+n_target)`
  bijection with control-off sub-states fixed). Closes the single failure carried
  over from R.1.13.0.

### Known Issues

- The MPS backend does not recover Shor's order for N=15 on the 13-qubit
  period-finding circuit, even though bond dimension 64 is mathematically exact
  for any 13-qubit state (maximum Schmidt rank `2^6 = 64`, so no truncation can
  occur). The statevector backend recovers it. This is an accuracy gap in the MPS
  path, not a truncation limit; the PERMUTATION-oracle fallback is verified exact
  at 4 and 8 qubits. Targeted for R.1.13.2.

### Results

- 1849 tests across 143 suites, all passed (23.4 s, WSL / Clang, `-march=native`).

## [R.1.13.0] - 2026-07-07

Performance wave over the entire shipping library (audit
`local/audits/2026-07-06-r113-performance-audit.md`, 24 of 29 findings
implemented). The density-matrix and MPS simulators, the qudit layer, the
estimator, and the transpiler were reworked for speed without giving up
correctness; where a genuine accuracy or speed trade existed it is a
user-selectable option with the exact path as the default. Two structured-oracle
instructions (MCX, MCP, PERMUTATION) replace the dense per-iteration matrices in
Grover and Shor. Contains breaking changes (marked BREAKING below). The DAG
`substitute_node`/`remove_node` refactor (audit F-22) is deferred to its own
release: it has no in-tree callers and its correctness can only be verified
against the routing regression suite.

### Added

- `MCX`, `MCP`, and `PERMUTATION` instructions with builders `mcx(controls,
  target)`, `mcp(lambda, qubits)`, and `permute(perm, qubits)`. MCX and MCP are
  native two-amplitude/diagonal kernels; PERMUTATION applies a basis-index map
  as an O(dim) gather. All three are native in the statevector and
  density-matrix backends (no dense 2^k matrix); the MPS backend reduces MCX
  with <= 2 controls to X/CX/CCX and fails loud on the rest. Grover's diffusion
  now uses `mcx` and Shor's controlled modular multiplication uses `permute`.
- `SVDMethod` option on the qubit and qudit MPS (`MPSState::svd_method`,
  `QuditMPS::svd_method`). Jacobi (accurate) is the default; BDC is a faster
  opt-in that is CURRENTLY BROKEN (Eigen BDCSVD defect, R.1.11.2) and emits a
  loud one-time runtime warning when selected.
- `Estimator::Options::group_pauli_terms` (default true): groups qubit-wise
  commuting observable terms so they share measurement runs (all Z/I terms
  collapse into one run). Set false to restore the pre-R.1.13 one-run-per-term
  sampling (byte-identical seeded stream).
- `Simon::solve(..., bool batch_shots = true)`: draws all equation samples from
  one batched simulation; false restores the pre-R.1.13 per-sample loop.
- Benchmark sources `benchmarks/bench_qudit_sv.cpp`, `bench_qudit_dm.cpp`,
  `bench_scaling.cpp` (standard circuit across all four backends).

### Changed

- BREAKING: `Instruction::matrix` is now a copy-on-write `CowMatrix` (shared,
  immutable buffer) instead of `std::vector<Complex128>`, so Instruction copies
  no longer deep-copy 2^k x 2^k gate data. Read call sites are source-compatible
  (implicit conversion to `const std::vector<Complex128>&`); code that mutated
  the matrix in place must assign a fresh vector instead.
- Density-matrix simulator: `apply_gate` rewritten as an OpenMP row-block AXPY
  over contiguous rows (was serial column-strided); `apply_kraus` is out of
  place (no per-Kraus 4^N restore copy); noise and gate matrices are resolved
  once per circuit instead of per shot (`errors_for_gate` no longer deep-copies
  Kraus operators per call); one density-matrix buffer is reused across shots
  (via a new `DensityMatrix::initialize()` that resets the matrix to
  `|0..0><0..0|` in place); `expectation_value_sparse` uses folded per-term
  phase + popcount parity + an OpenMP reduction.
- MPS sampling: environment contractions are the two-stage O(chi^3) form (were
  O(chi^4)); terminal-measurement sampling computes the shot-invariant right
  environments once and samples each shot read-only (no per-shot MPS copy);
  the two-site gate contraction is a single zero-copy Eigen GEMM. `QuditMPS`
  measurement uses the same sequential-environment sampler instead of
  contracting to a dense d^n statevector.
- Statevector: the MEASURE/RESET/trajectory collapse is one shared OpenMP
  helper; `apply_unitary` parallelises by total work so large-k oracles no
  longer run serially; `RCCX` is one three-level-stride kernel pass instead of
  a nine-kernel ladder.
- Estimator sampling: qubit-wise-commuting term grouping (see
  `group_pauli_terms` above). This changes seeded byte output versus R.1.12
  (statistically identical); seeds reproduce within a version, not across.
- Qudit statevector: `apply_1qudit`/`apply_2qudit`/`apply_kqudit` are now
  OpenMP-parallel.
- Clifford simulator: terminal-measurement circuits run the gate pass once and
  sample each shot from a copy of the stabilizer tableau.
- Transpiler: `CouplingMap::distance_matrix` uses BFS per node (was
  Floyd-Warshall O(V^3)); `SabreLayout` advances successors through an adjacency
  list and tests adjacency via the distance matrix (was O(N*E) edge scans and
  per-query `is_connected`); `SabreSwap` drops a dead full-DAG copy, hoists the
  inverse-layout build, and indexes candidate swaps through physical-neighbour
  lists.
- BREAKING: qubit MPS SVD default changed from BDC to Jacobi (accuracy first;
  see `SVDMethod` above). Truncation results shift numerically versus R.1.12.
- `MAQAOA`/`QAOA` mixer and other paths unchanged in behaviour; only the shared
  simulator/estimator internals they call were sped up.

### Fixed

- Restored the truncated tail of the top-level `CMakeLists.txt`. The
  `add_subdirectory(tests)` / `benchmarks` / `apps` blocks and the
  Python-bindings body were missing, so a clean configure built only
  `lindblad_core` and produced no test, benchmark, app, or Python-binding
  targets. This truncation was present in the published R.1.12.2 release: a
  fresh clone of R.1.12.2 from GitHub could build the library but could not
  build or run its tests, benchmarks, or CLI apps. The regression went unnoticed
  because existing local, separately-configured build directories retained the
  cached rules from when the file was whole and kept building/running the tests
  (against a stale binary), so it only surfaced on an R.1.13 clean build. The
  release version string itself was always correct; only the build of the
  auxiliary targets was affected.
- The MPS backend now applies the new `MCX` (wide) / `MCP` / `PERMUTATION`
  instructions via the bounded statevector fallback (`to_statevector` -> apply
  -> `mps_from_sv`), the same path a 3+ qubit `UNITARY` already used. Without
  this, Shor's oracle (which now emits `PERMUTATION`) would have thrown on the
  MPS backend, whereas its former dense `UNITARY` ran there; the fallback keeps
  Shor-on-MPS working exactly as before.

### Results

- Full suite: 1783/1784 across 133 suites (21.1 s, WSL / Clang 18,
  `-march=native`). The one failure, `ShorTest.CircuitContainsUnitaryGates`,
  asserts the pre-R.1.13 dense-UNITARY modular-multiplication oracle, which F-9
  deliberately replaced with a `PERMUTATION` instruction; the stale assertion is
  updated (and new-op coverage added) in the R.1.13.1 test patch.

## [R.1.12.2] - 2026-07-04

Fix release of the R.1.12 cadence. Resolves the four defects pinned red by the
R.1.12.1 total-coverage suite, closes that suite's line-coverage target with
four additional fill suites, and fixes eight more defects the fill and a full
compiler-warning triage surfaced along the way. Closes issues #30 #31 #32 #33
#34 #35 #36 #37 #38 #40 #42 #43 (#27 closed as duplicate of #30; #39 closed as
intended behaviour, superseded by tracking issue #41).

### Fixed

- `SparsePauliOp::to_matrix()` built a non-Hermitian matrix for any Y-containing
  term: the `i^(#Y)` factor is now a per-term constant folded into the
  coefficient (`Y = iXZ`) instead of a per-column `i^popcount(j & y_mask)`.
  `to_matrix` now agrees with `expectation_value` and composes homomorphically.
  (#30)
- `zyz_decompose` extracted `phi`/`lambda` with a flipped sign
  (`arg(SU[0,0]) = -(phi+lam)/2`, not `+`), so `Optimize1qGates` merged generic
  single-qubit runs into a different unitary and `transpile()` /
  `preset_pass_manager()` silently altered circuits at every optimization
  level. The theta-only-approximately-zero branch had the same defect. (#31)
- Qubit `Grover` auto iteration count now uses the exact-angle optimum
  `max(1, round(pi/(4*asin(1/sqrt(N))) - 1/2))`, mirroring the qudit path;
  N = 4 selects 1 iteration instead of over-rotating with 2. (#32)
- `DensityMatrixSimulator` now applies per-qubit `ReadoutError` confusion to
  every sampled MEASURE outcome on both sampling paths (terminal clbit-keyed
  sampling and per-shot trajectories). The state stays collapsed to the true
  outcome; only the classical record (and feedforward reading it) sees the
  noisy bit. Flips draw from the run's seeded RNG; ideal models consume no
  extra draws, so seeded ideal runs are bit-for-bit unchanged. (#33)
- `BasisTranslator` gained the missing `CU` (CU3 ladder with the gamma phase
  folded into the control U1) and `RCCX` (Margolus H/T/CX ladder) recipes;
  both previously passed through the default `{cx, u3}` target undecomposed
  with no error. (#34)
- QASM2 custom-gate parameter arithmetic: numeric tokens must now parse in
  full (`"2*a"` no longer partial-parses to 2.0), unresolvable tokens throw
  instead of silently becoming 0, and `+`/`-` bind at the lowest precedence so
  `a-pi/2` evaluates as `a - (pi/2)`. (#35)
- `VQE::compute_minimum_eigenvalue` no longer surfaces an uninitialised value
  when NLopt fails: the integer status is checked first, a non-finite result
  falls back to the minimum of the recorded energy history, and a run whose
  objective never evaluated throws. (#36)
- `SabreLayout` was a silent no-op on every circuit: the front layer counted
  wire IN-node edges in its in-degree seeding, so it was permanently empty and
  the pass always returned the trivial layout with zero swaps. Seeding now
  counts OP-source edges only; the heuristic and forward/backward/final passes
  execute for the first time. (#37)
- The QASM3 modifier matrix fallback embedded `ctrl @` controls as the MSB
  (block-diagonal), which under the frozen qubits[0]-is-LSB convention swapped
  control and target for every gate routed through it. It now uses the
  interleaved layout (control = index bit 0), matching `control()` and the
  Shor/QPE controlled matrices. (#38)
- `QAOA` evolved multi-qubit mixer terms as independent per-qubit rotations
  (`RX (x) RY` instead of `exp(-i*beta*XY)`), a wrong ansatz for entangling
  (e.g. constraint-preserving XY) mixers. The mixer loop now uses the same
  per-term Pauli-rotation recipe as the cost unitary; single-qubit mixers are
  bit-for-bit unchanged. (#40)
- QASM3 `pow(n) @` / `inv @` folding scaled every gate parameter by
  `inv*pow`, an identity valid only for single-axis rotations; `u`/`u2`/`u3`
  under those modifiers silently imported a slightly different unitary
  (Euler angles do not scale, and `inv U3(t,p,l) = U3(-t,-l,-p)`). Multi-
  parameter gates now route to the exact matrix fallback. (#42)
- The build handed the project's `-ffast-math` to vendored NLopt, whose
  COBYLA convergence logic assumes IEEE semantics; under Clang with
  `-march=native` the optimiser bailed after the initial evaluation (VQE
  returned the starting energy). NLopt now compiles with `-fno-fast-math`
  (keeping `-O3`/`-march`). (#43)

### Changed

- `MAQAOA::optimize` / `build_circuit` now throw `std::invalid_argument` on a
  non-empty `mixer_hamiltonian` instead of silently ignoring it: the MA-QAOA
  mixer is definitionally the fixed per-qubit RX (Herrman et al. 2022), with
  beta customisation via `options.mixer_weights` / `options.orbit_assignments`.
  First-class custom-mixer support is tracked in #41. (Supersedes #39.)
- All NaN guards in `maqaoa`/`vqe` use the new bit-level
  `lindblad::is_finite_strict`, immune to `-ffast-math` folding
  `std::isfinite` away (GCC never warns about this; Clang's
  `-Wnan-infinity-disabled` does).
- Eigen 3.4.0 keyed its test registration on the standard `BUILD_TESTING`
  variable and silently registered ~915 phantom ctest entries; dependencies
  now build with `BUILD_TESTING` forced OFF, so `ctest` lists only Lindblad
  tests. NLopt additionally builds with `-w` to keep warning inventories
  signal-only.
- README Clang recipe corrected: `-DLINDBLAD_MARCH_NATIVE=ON` (a flag-level
  `-march=native` was silently overridden by the project's `-march=x86-64-v3`)
  and an accurate explanation of `-Wno-nan-infinity-disabled`.
- Full GCC and Clang warning inventories driven to zero in project sources:
  dead code removed (`QuditSimon::null_space_ring`, an unused HTML escape
  helper, `glyph_row_span`, phantom `BackendProperties` forward declarations,
  dead locals in the gate kernels), dangling-else sites braced, intentionally
  unused parameters documented.
- Documentation updated across `docs/api/` (operators, grover, maqaoa, qaoa,
  qasm, vqe, noise) to state the corrected contracts, including the QASM3
  modifier folding rules and the readout-error application semantics.

### Added

- Coverage fill suites `tests/test_r1122_fill_{simulators,transpiler,
  algorithms,frontends}.cpp` (~75 tests): every gate branch through MPS and DM
  against the statevector oracle, the MPS statevector-fallback path,
  BasisTranslator full-gate decomposition sweep, SABRE routing contracts,
  scheduling barrier semantics, algorithm option matrices
  (QFT/Simon/MAQAOA/VQE/Shor), QASM2 builtin and custom-gate sweeps, QASM3
  modifier stacks, Estimator transpile cache, LocalBackend AUTO selection,
  every noise-channel factory checked for trace preservation, qudit MPS vs
  QuditStatevector cross-checks, and the visualisation catalogues.
- `lindblad::is_finite_strict(double)` in `types.hpp`: IEEE-754 finiteness
  test via bit pattern, one integer compare, immune to any compiler flag set.

### Results

- Full suite: 1784/1784 across 133 suites. Verified on GCC 13 (WSL) and
  Clang 18 with `-march=native`; zero compiler warnings in project sources.
- Line coverage 95.13% on `src/` + `include/` (gcovr, instrumented build),
  every file at or above 85% except the two documented gcov
  aggregate-initializer artifacts in the visualisation catalogues; the
  R.1.12.1 coverage plan's definition of done is closed.

## [R.1.12.1] - 2026-06-25

R.1.12.1 is the test-only release of the R.1.12 cadence (the `.1` patch slot
carries no production code). It adds a total-coverage suite for the conventions
frozen in R.1.12.0: every public symbol in
[`include/lindblad/`](include/lindblad) is now exercised by at least one test
that asserts its contract, every gate and channel formula is pinned numerically,
every documented error path is checked, and every frozen convention has an
asymmetric (convention-revealing) test.

Driving coverage this thoroughly surfaced four latent correctness defects that
earlier suites had missed. Per the project's release policy each one ships as a
documented failing test that asserts the CORRECT contract, so it becomes a
regression guard the moment the bug is fixed; the fixes themselves are deferred
to R.1.12.2, since the `.1` slot is test-only.

### Tests

- New `LINDBLAD_BUILD_COVERAGE` CMake option (off by default) compiles
  `lindblad_core` and the test tree with `--coverage -O0 -g` on GCC/Clang for
  gcovr line and branch reporting.
- 24 new `tests/test_r1121_*.cpp` suites plus a `LINDBLAD_BUILD_PYTHON`-gated
  Python bindings harness under `tests/python/`. A static symbol
  cross-reference drove the suite to zero unreferenced public symbols.
- Core: `Complex128` algebra and constants, `Statevector` (bounds, sampling key
  convention, move and clone), every gate matrix under the frozen LSB
  convention, the `QuantumCircuit` builder, validation, compose, inverse and
  `control()` surface, JSON and QASM2/QASM3 round-trips, and the `DAGCircuit`
  API.
- Engines: the SV/DM/MPS execution-strategy matrix (nine measurement scenarios
  across shots 0, 1 and 1024 with cross-backend distribution agreement), every
  noise channel (CPTP plus closed-form density-matrix evolution, including the
  full thermal-relaxation T1/T2 grid and the exact depolarizing Bloch
  contraction), `SparsePauliOp` and `Operator` algebra, quantum-information
  metrics (fidelity families, entropies, Werner-state concurrence, partial
  traces), the Clifford conjugation table, and `MPSState` (bond-dimension growth
  and truncation, non-adjacent and reversed two-qubit gates, sequential
  measurement).
- Toolchain: exact `CouplingMap` edge sets for the four named topologies, each
  transpiler pass on crafted DAGs (`ConsolidateBlocks` over the full two-qubit
  gate set, routing validity across topologies at every optimization level), the
  `Estimator` four-mode matrix and the `Sampler` seed scheme, and `LocalBackend`
  selection.
- Qudit layer and algorithms: `QuditStatevector`, density-matrix, MPS and
  Clifford backends at d = 2 to 7 (general-d gate unitarity, the LSB digit
  convention, channel trace preservation, Lindblad decay), and end-to-end QFT,
  Shor, QPE (dyadic and non-dyadic phases), Grover (asymmetric targets), VQE,
  QAOA/MA-QAOA and Ising mapping.
- Visualisation: the full `DrawOptions` matrix across the ASCII, SVG, LaTeX and
  HTML renderers, plus `draw_to_file`.

### Results

- 1693/1702 tests across 129 suites passed (coverage-instrumented build, WSL).
  The 9 failures are the four documented findings below, each asserting the
  correct contract and slated for the R.1.12.2 fix:
  - `SparsePauliOp::to_matrix()` builds a non-Hermitian matrix for any Pauli
    term containing Y (the i^(number of Y) factor is applied per output column
    rather than once per term):
    `R1121Operators.PauliComposeMatchesMatrixProductAllPairs`,
    `.PauliComposeMultiQubitAndLengthThrow`,
    `.ExpectationValueMatchesQuadraticForm`.
  - `Optimize1qGates` (via `zyz_decompose`) merges a generic single-qubit run
    into a gate with the correct rotation angle but wrong phase angles, so
    `transpile()` and `preset_pass_manager()` silently alter a circuit's
    unitary: `R1121Passes.Optimize1qPreservesGenericRun`,
    `.PassManagerComposesAndPreservesSemantics`,
    `.TranspilePresetLevelsPreserveSemantics`,
    `R1121PassesMore.PresetPassManagerPreservesSemanticsUnconstrained`.
  - The density-matrix simulator does not apply `ReadoutError`:
    `R1121NoiseModel.ReadoutErrorPerturbsCounts_EXPECTED_RED`.
  - The qubit `Grover` auto iteration count `round(pi/4 * sqrt(N))` over-rotates
    the N = 4 (two-qubit) search (2 iterations instead of the optimal 1),
    leaving the marked state un-amplified; the qudit Grover in the same file
    already uses the corrected count:
    `R1121Algos.GroverTwoQubitAutoIterationsOverRotate_EXPECTED_RED`.

## [R.1.12.0] - 2026-06-12

Resolves GitHub issues #9 through #26 (every currently open issue) in one
correctness and performance wave, and freezes the project conventions. The
normative convention reference is the new "Conventions" section in
[`docs/Architecture.md`](docs/Architecture.md).

The underlying defects were identified by a full-codebase correctness and
performance audit performed with Claude Fable 5, Anthropic's Mythos-class
model; every finding was verified by independent numerical replication before
being filed as an issue, and several (the non-unitary DM/MPS interaction
matrices, the n >= 4 Grover diffusion, the half-rate T2 dephasing, the
cross-backend convention splits) had survived prior manual review of the codebase.

### Changed

- **Pauli strings are LSB-first project-wide** (#11, BREAKING): `pauli[q]` acts
  on qubit q. This is a deliberate deviation from Qiskit's label order, and
  Pauli strings read in the opposite direction from measurement bitstrings
  (whose rightmost character is qubit 0). `IsingHamiltonian::to_sparse_pauli_op`,
  the Estimator sampling path, and `DensityMatrix::expectation_value_sparse`
  were flipped to match the already-LSB-first `SparsePauliOp` evaluation,
  Clifford `expectation_pauli`, and QAOA internals. This closes the split where
  the same observable evaluated with opposite signs depending on the execution
  path (exact vs sampled vs noisy), and where QAOA returned qubit-reversed
  variable assignments for `IsingHamiltonian` problems.
- **Multi-qubit matrices are `qubits[0]`-is-LSB on every backend** (#13,
  BREAKING): `unitary()` instructions and multi-qubit `KrausChannel` operators
  now mean the same operation on the statevector, density-matrix, and MPS
  simulators; the DM backend gained the bit-reversal bridges the MPS already
  had (`gate_matrix_for_dm` UNITARY case and `apply_kraus`).
- **Qudit subspace matrices flipped to first-operand-is-least-significant-digit**
  (BREAKING for hand-built matrices only): `apply_2qudit`/`apply_kqudit` and
  the `qudit_gates` builders changed in lockstep, so all in-tree algorithms
  behave identically. See [`docs/api/qudit.md`](docs/api/qudit.md).
- **ECR argument order resolved as a documented deliberate deviation** (#24):
  `ecr(a, b)` equals Qiskit `ecr(b, a)`; all three simulators agree. Documented
  in [`docs/api/gates.md`](docs/api/gates.md); swap operands when porting
  Qiskit circuits.
- **CouplingMap edge lists are literal** (BREAKING): an edgeless
  `CouplingMap(n)` declares a graph with no allowed pairs and routing 2-qubit
  gates against it throws; unconstrained routing is expressed by
  `CouplingMap()` (n = 0). SABRE now THROWS on a routing stall instead of
  silently discarding the unroutable remainder of the circuit, enforces
  all-pairs adjacency for 3+ qubit gates, routes barriers freely, and guards
  against unroutable cross-component gates with a SWAP budget.
- **shots == 0 is one seeded trajectory** (BREAKING): classical conditions are
  honoured and MEASURE outcomes are recorded along the way;
  `eval_expectation` and `Estimator` with shots == 0 throw for circuits
  containing measurement or conditional instructions (the exact expectation of
  a single stochastic trajectory is undefined).
- **Terminal-measurement sampling** (#25): circuits whose measurements are all
  terminal (the `measure_all` pattern) now run ONE forward evolution and
  sample outcomes from the final state, keyed by the qubit-to-clbit map
  (partial measurements key only the measured clbits). Replaces per-shot
  re-simulation: roughly shots-times faster on sampling workloads, about 128x
  on `Shor::find_order`. Per-thread RNG stream independence is preserved on
  the fast path (pinned by `BugRegression.B9`).
- **QPE builds U^(2^k) by repeated squaring** (#26) with the base-matrix
  extraction hoisted out of the per-qubit loop: m-1 matrix products instead of
  2^m - 1, and one column extraction instead of m.
- Smaller performance work: O(1) adjacency lookups in SABRE's routing loop,
  direct base-index enumeration in the qudit subspace appliers,
  `partial_trace(Statevector)` no longer materialises the full density matrix,
  and the DM per-shot path moves (not copies) the final trajectory state.
- Known red tests shipped with this release, pending the R.1.12.1 suite
  refresh: `AuditR1112.C4a` (pins the superseded MSB-first string docs),
  `AuditR1112.C16` (pins Qiskit's ECR order), `AuditR1112.C17` (pins the
  pre-freeze unconstrained CouplingMap reading; transpile now correctly
  throws there).

### Fixed

- **DM/MPS interaction-gate matrices** (#9): RZX acted as a local RX
  (non-entangling) on both backends; DM RXX and DM/MPS RYY were non-unitary
  (missing diagonal cos entries) and silently destroyed probability mass. All
  five wrong matrices now match the statevector reference implementations.
- **Grover at 4+ qubits** (#10): the diffusion MCX matrix targeted qubit 0
  while the H sandwich wrapped qubit nq-1, collapsing success probability to
  noise level; the MCX now targets the wrapped qubit.
- **`QuantumCircuit::control()`** (#12): the generic path silently emitted the
  IDENTITY for every gate without a dedicated controlled mapping (controlled
  SX/CZ/RZZ, any gate with 2+ controls) and placed the gate block on the wrong
  qubits for UNITARY instructions. Rebuilt with real gate matrices and the
  interleaved control layout used by Shor/QPE; unresolved parameterised gates
  throw instead of degrading.
- **`thermal_relaxation` dephasing rate** (#14): coherences now decay by
  exactly exp(-t/T2); previously half the requested pure-dephasing rate for
  every T2 < 2*T1 (only the T2 = 2*T1 boundary was correct), which made every
  `NoiseModel::from_t1_t2` model too coherent.
- **Estimator with optimization_level >= 1** (#15): silently deleted every
  gate from the first 2-qubit gate onward (edgeless internal CouplingMap plus
  SABRE's silent stall). The Estimator now passes `CouplingMap()` and stalls
  throw.
- **MPS sampling above 18 qubits** (#16): `measure_sequential` returned
  bit-reversed count keys; qubit 0 is now the rightmost character at every
  register width.
- **MPS mid-circuit measurement and reset** (#17): collapse renormalises by
  the environment-contracted outcome marginal (the local Frobenius norm is
  only valid in canonical form, leaving the state unnormalised); RESET samples
  from normalised probabilities.
- **Feedforward integrity** (#18, #19, #20): the statevector simulator
  honours classical conditions outside the per-shot path (trajectory
  semantics); `Optimize1qGates` no longer merges conditioned gates into
  unconditional rotations, `CXCancellation` requires identical conditioning,
  `CommutativeCancellation` excludes conditioned gates entirely; the DAG gains
  read-after-write and write-after-read classical-dependency edges and a
  deterministic topological order.
- **ConsolidateBlocks** (#21): unsupported 2-qubit gates (ECR, CU, RZX,
  UNITARY) are no longer absorbed as identity inside consolidated blocks;
  unrepresentable gates break the block, and every KAK decomposition is
  verified against the block unitary (up to global phase) with the original
  block kept on mismatch.
- **QASM 2 importer** (#22): whole-register `measure q -> c;` and `reset q;`
  expand to per-bit instructions (register-size mismatches throw);
  unresolvable measure/reset operands THROW instead of being silently dropped
  (previously register-form files imported with all measurements missing);
  `barrier` honours its operand list.
- **`depolarizing(p, n_qubits)`** (#23): general n-qubit Pauli twirl for n in
  [1, 6] with argument validation; previously returned a silent EMPTY channel
  for n >= 3 that annihilated the state when applied.
- Assorted: `to_json()` throws on unbound symbolic parameters instead of
  silently dropping them; `Sampler` batches thread per-circuit seeds without
  mutating `options.seed`; `is_clifford` accepts the 2-pi fmod boundary like
  `run()`; the RCCX phase comment matches the implemented operator;
  `noise.hpp` includes `<array>` directly.

### Results

- 1314/1317 tests across 103 suites passed (14.4 s, WSL). The 3 failures are
  the convention-pinned `AuditR1112` tests (C4a, C16, C17) described under
  Changed: they encode the superseded pre-freeze expectations and are slated
  for the R.1.12.1 test-suite refresh. `BugRegression.B9` (per-thread RNG
  independence) is green on the new terminal-measurement sampling path.

## [R.1.11.2] - 2026-06-02

### Fixed

- **`QuditMPS` dense-statevector reconstruction was wrong for d > 2**, producing
  trivial or incorrect qudit Simon results on the MPS backend. Two independent root
  causes: (1) the dense `QuditMPS(statevector, ...)` constructor decoded interior-site
  SVD `U` rows with a physical-major index (`sigma*chi_L + aL`) while the residual matrix
  was built (and the final site read) with a left-bond-major index (`aL*d + sigma`),
  transposing the physical and bond indices on any interior site with a nontrivial bond
  (chi_L > 1); the extraction now uses `aL*d + sigma` to match. With chi_L = 1 the old and
  new formulas coincide, so only previously-corrupted states change. (2) `Eigen::BDCSVD`
  returned an inaccurate decomposition for a degenerate, rank-deficient complex matrix
  arising at an interior site (the reconstruction `U·S·V†` did not match the input and the
  squared singular values did not sum to 1), giving a non-negligible reconstruction error;
  all four SVD sites (dense constructor, two-qudit gate split, left and right
  canonicalization) now use `Eigen::JacobiSVD`, which is machine-precision accurate for
  these matrices. Reported in [#8](https://github.com/verycareful/lindblad/issues/8); its
  three reproducing cases (`/21`, `/24`, `/29` in `tests/test_qudit_r1111.cpp`) now pass
  (`/21` also needed the composite-recovery fix below).
- **`QuditSimon` composite-d period recovery could return a non-period vector.** The
  hidden subgroup was read off an integer Smith-Normal-Form kernel of the measurement
  matrix, which for composite d could emit a vector that does not annihilate every measured
  outcome (for example returning a candidate outside the true kernel). Recovery now performs
  a direct, oracle-verified search over Z_d^n, returning the first nonzero vector that
  annihilates every measured outcome (mod d) and is a confirmed oracle period. This was a
  latent correctness bug on all backends (statevector, density matrix, MPS), not MPS-only.

### Changed

- **CLI banner, README, and project website license copy clarified** to cite both
  LICENSE v2.3 §3.1 (private sharing of unmodified copies) and §3.2 (private sharing of
  modifications) for non-commercial use. No license change: the text already permitted this
  since v2.3; the public-facing wording previously described only §3.1. Public
  redistribution (forks, mirrors, registries, public derivatives) remains prohibited.

### Results

- 1296 tests across 102 suites, all passed (~72 s, WSL/Clang).

## [R.1.11.1] - 2026-05-31

Test-only release for the R.1.11.0 qudit completeness work.

### Tests

- **`tests/test_qudit_r1111.cpp`** — dedicated suite for the three R.1.11.0 features,
  value-parameterized over an exhaustive (backend, d, n, case) matrix bounded by
  per-backend dimension budgets:
  - **apply_P (odd prime d)**: white-box tableau-transform check at d=3,5,7 (injects
    x >= 2 so the x(x-1) phase term is exercised) plus an exact-statevector cross-check
    on non-symmetric H-P-H circuits at d=2,3,5,7.
  - **Affine DeutschJozsa**: `QuditAffineOracle::eval` unit/throw tests; constant-vs-
    balanced verdict matrix across STATEVECTOR/DENSITY_MATRIX/MPS over d in {2..9},
    n in {1,2,3}, plus CLIFFORD over prime d; opaque-oracle and affine-composite
    CLIFFORD throw paths; affine-vs-opaque agreement.
  - **Composite-d Simon**: period-recovery matrix over composite d {4,6,8,9,12} and a
    prime control, verified against f(x)=f(x+s); injective-is-trivial; affine Simon on
    CLIFFORD recovers ker(A) for prime d; opaque/affine-composite CLIFFORD throw paths.

### Results

- 1269 / 1272 tests across 95 suites passed (~74 s, WSL/Clang).
- **3 known failures**, all `Matrix/SimonComposite.RecoversVerifiedPeriod` on the **MPS
  backend** at d > 2. This is a pre-existing MPS-backend defect (it reproduces on prime
  d=5, so it is not the composite-d ring kernel; STATEVECTOR and DENSITY_MATRIX recover
  all cases correctly). The failing tests are kept in the suite rather than disabled —
  hiding a real correctness failure would violate the project's no-silent-failures rule.
  The MPS fix is tracked for R.1.11.2 (a `.1` release carries tests only, no fixes).

## [R.1.11.0] - 2026-05-31

### Added

- **`QuditCliffordSimulator::apply_P()` now supports odd prime d.** Implements the
  canonical qudit phase gate `P = Σ_k ω^{2⁻¹·k(k−1)} |k⟩⟨k|` (Howard & Vala 2012).
  The Heisenberg-picture tableau update is `z_q → z_q + x_q`, `phase += x_q(x_q−1)
  (mod 2d)`; the inverse-of-2 factor lives only in the gate's matrix definition, so
  the conjugation result is inverse-free. The d=2 S-gate keeps its own branch (2 is
  not invertible mod 2). Previously threw `std::runtime_error` for d > 2.
- **`QuditAffineOracle`** — a structured, Clifford-decomposable oracle `f(x) = A·x + b
  (mod d)`. The reversible function-oracle gadget lowers to `X^{b_j}` plus
  `apply_CSUM` powers on the stabilizer tableau, so affine oracles run on the CLIFFORD
  backend where black-box `std::function` oracles cannot.
- **`QuditDeutschJozsa::solve(const QuditAffineOracle&, ...)`** — affine-oracle overload
  supporting all four backends, including CLIFFORD (prime d). An affine `f` is constant
  iff `a = 0` and balanced iff `a ≠ 0`.
- **`QuditSimon::solve(const QuditAffineOracle&, ...)`** — affine-oracle overload that
  extends the CLIFFORD backend (prime d) to Simon; other backends materialise `f`.
- **Composite-d support for `QuditSimon`.** The hidden subgroup is now recovered over
  the ring `Z_d` via the integer Smith Normal Form of the measurement matrix (uniform
  across all moduli, reproducing the field result for prime d), with candidate
  generators verified against the oracle before being returned. Previously composite d
  threw `std::invalid_argument`.

### Changed

- **`QuditDeutschJozsa` and `QuditSimon` opaque-oracle CLIFFORD path now throws instead
  of silently substituting another backend.** The black-box `std::function` overloads
  reject `QuditBackend::CLIFFORD` with a message directing callers to the
  `QuditAffineOracle` overload (DJ previously fell back silently to STATEVECTOR; Simon's
  message is clarified). Aligns with golden rule #1 (no silent failures).
- **Version label is now sourced from a single compile-time definition.** The CLI
  welcome/goodbye banner (`src/banner.cpp`) and the `lindblad_draw` usage banner
  (`apps/lindblad_draw.cpp`) read the `LINDBLAD_VERSION_LABEL` macro instead of
  hardcoding the string; the definition is exported by `lindblad_core` and now also
  attached to the `lindblad_banner_obj` object library. No behaviour change, but the
  on-screen version can no longer drift from `CMakeLists.txt`.

## [R.1.10.8] - 2026-05-28

### Changed

- **`LICENSE`** -- bumped to Version 2.3. Substantive structural changes over v2.2:
  - **§1.2** -- "sole creator" corrected to "original author, primary copyright holder"
    to reflect that PR #7 introduced contributor code.
  - **Preamble** -- added an explicit declaration that the Agreement governs all
    versions of the Software in the repository, including all prior commits and
    historical revisions, regardless of license text that may appear in earlier
    revisions. This closes the prior-license-text-in-git-history gap at the document
    level.
  - **§3.2** (new) -- Permitted Private Redistribution of Modifications. Modified
    copies may be shared privately with specific, identified individuals for
    Non-Commercial purposes, mirroring the §3.1 conditions for unmodified copies.
    Public redistribution of modifications remains prohibited.
  - **§3.3** (new) -- Community Notification (Advisory). Strongly encourages, but
    does not require, contributors to inform the Author of meaningful modifications
    via GitHub Issues or email. Explicit no-breach-for-non-disclosure language.
  - **§3.4** -- prior §3.2 (Prohibited Acts) renumbered. Clause (b) updated to
    reference §3.2 as the carve-out for permitted private modification sharing.
  - **§10.3** -- survival clause expanded to include §6.6 (acknowledgment) and
    §6.7 (no monetary compensation), so a contributor whose user license is
    terminated cannot retroactively claim compensation and the Author's
    acknowledgment commitment survives termination.
- **Contact email** -- migrated from `qpp.support@proton.me` to
  `lindblad.software@proton.me` across all live files (`LICENSE`, `README.md`,
  `CITATION.cff`, `NOTICE`, `src/banner.cpp`, `lindblad-page/src/pages.tsx`,
  `lindblad-page/src/components/chrome.tsx`). Historical CHANGELOG entries
  intentionally retain the old email as a record of what shipped at those releases.
- **`NOTICE`** -- caught up to Version 2.3 (was still labelled 2.1).

## [R.1.10.7] - 2026-05-24

### Fixed

- **`Estimator::run_single` now performs real Pauli-basis shot sampling.** Pre-fix,
  `options.shots > 0` only toggled the simulator backend from SV to DM — the
  returned expectation was still read off the final state analytically with
  zero variance. VQE/QAOA users relying on finite-shot noise to model real
  hardware silently got exact estimates. `run_single` now decomposes the
  observable into Pauli terms, rotates each non-identity term into the Z
  measurement basis, samples `shots` measurements, and accumulates a
  parity-weighted estimate. Three modes: `shots == 0` (exact, ideal or DM),
  `shots > 0` ideal (SV sampling), `shots > 0` with noise (DM sampling).
  Variance scales as `1/√shots` per Pauli term. (Reported by [@zParik](https://github.com/zParik) in #2.)
- **`QuantumCircuit::to_qasm2()` documents 1-qubit UNITARY global phase loss;
  `to_qasm3()` preserves it via `gphase`.** OpenQASM 2.0 has no representation
  for global phase, so the ZYZ lowering at `circuit.cpp:1095` had been
  silently dropping the `alpha = arg(U[0,0])` factor — invisible for the
  gate in isolation, but a real bug when the unitary is later wrapped under
  a control. `to_qasm2()` now emits an explicit `// global phase: <alpha>`
  comment when alpha is non-zero so the loss is documented and discoverable.
  `to_qasm3()` gains a proper 1-qubit-UNITARY path that emits `gphase(alpha)`
  before `u(theta, phi, lambda)` — gphase commutes through QASM 3 `ctrl @`
  modifiers, so the round-trip is lossless on the emission side. (Parser-side
  `gphase` support in `from_qasm3()` deferred; current emission produces
  spec-correct output.) (Reported by [@zParik](https://github.com/zParik) in #4.)
- **`MPSSimulator` applies 1q and 2q UNITARYs via direct tensor contraction.**
  Pre-fix every UNITARY routed through `to_statevector()` regardless of size,
  which threw `"Too many qubits for full statevector conversion"` from a site
  far from the offending instruction whenever `n_qubits > MPS_SV_MAX_QUBITS`
  (25). Wide MPS circuits with any user-supplied unitary were unusable.
  `mps_apply_instruction` now dispatches 1-qubit UNITARYs to
  `MPSState::apply_single_qubit_gate` (contracts the 2x2 matrix into one
  site tensor, no SVD) and 2-qubit UNITARYs to `apply_two_qubit_gate`
  (contracts into two-site tensor + truncated SVD; non-adjacent qubit pairs
  handled via the existing swap network). 3+ qubit UNITARYs keep the
  full-statevector fallback bounded by `MPS_SV_MAX_QUBITS = 25`; beyond
  that, a clear error names the UNITARY instruction and qubit count.
  The 2q dispatch transposes bit 0 ↔ bit 1 of the matrix row/column
  indices to bridge `apply_unitary`'s LSB-at-first-arg convention with
  `apply_two_qubit_gate`'s MSB-at-first-arg convention (matching the
  existing internal MPS 2q-gate convention). (Reported by [@zParik](https://github.com/zParik) in #5.)
- **`QuantumCircuit::from_qasm2()` throws on unknown gates instead of silently
  skipping.** A legacy "skip on miss" concession for older Qiskit exports was
  eating any gate name the parser didn't recognise (no built-in match, no
  user `gate` definition in scope), producing silently truncated circuits
  whose round-trip mismatches were getting attributed to other components.
  Violates project golden rule #1 (no silent failures). Now throws
  `std::runtime_error("QASM2Parser: unknown gate '<name>' …")` with the
  offending name in the message.

### Added

- `tests/test_bug_regression.cpp` — twelve new regression tests covering all
  four R.1.10.7 fixes: B10 (global-phase comment + gphase), B11 (QASM2
  unknown-gate throws and custom-gate-def still works), B12 (MPS UNITARY at
  n=28: 1q, 2q adjacent, 2q non-adjacent, 3q clearer error), B13 (Estimator
  shot noise: variance across seeds, shots=0 stays exact, convergence to
  exact at high shot count).

## [R.1.10.6] - 2026-05-24

### Added

- **`CONTRIBUTING.md`** -- contribution guide covering accepted contribution types, the
  PR process (issue-first for new features, direct PRs for bug fixes), C++23 style
  conventions, bug report template, and a required AI-verification notice (the codebase
  is ~40,000 lines; AI-generated contributions must be fully verified before submission).
- **`CONTRIBUTORS`** -- contributor acknowledgment file. Parikshieth Harish ([@zParik](https://github.com/zParik))
  acknowledged for the `find_order` bit-extraction fix (PR #7, landed in R.1.10.5).

### Changed

- **`LICENSE`** -- updated to Version 2.2.
  - §6.3 (Contributor License Grant): changed from irrevocable copyright assignment to a
    perpetual, irrevocable, non-exclusive license grant. Contributors now retain copyright
    ownership of their contributions while the author retains full commercial freedom.
  - §6.6 (Acknowledgment): changed from discretionary ("may acknowledge") to a firm
    commitment -- the author will acknowledge contributors in `CONTRIBUTORS` and in the
    release notes for the release in which their contribution first appears.

## [R.1.10.5] - 2026-05-24

### Fixed

- **`QFT::build_circuit` (`do_swaps=true`) now uses the project-wide LSB-at-qubit-0
  convention end-to-end.** The pre-fix gate sequence was textbook QFT (qubit-0=MSB
  internal); `do_swaps=true` placed the bit-reversal SWAPs on the wrong side, exposing
  a uniformly-MSB QFT/IQFT to callers. The SWAPs are now applied on the input side of
  the forward QFT and on the output side of the inverse QFT, delivering a
  uniformly-LSB QFT∘IQFT pair that matches Qiskit. `do_swaps=false` preserved
  unchanged as the raw H+CP sequence escape hatch.
- **`Shor::find_order` recovers the correct order on non-power-of-2 orders.** Two
  underlying bugs: (1) post-PR-#7 the bit-extraction was reading the correct slice
  but the QFT convention mismatch garbled the recovered `m`; the QFT fix above
  resolves that. (2) `find_order` was picking the single most-frequent measured
  bitstring, which is biased toward the exact-`m` peaks at `s=0` and `s=r/2` —
  useless under continued fractions whenever `r` doesn't divide `2^n_eval` (e.g.
  N=21, r=6). `find_order` now iterates observed bitstrings in descending frequency
  and returns the first valid `r`, recovering ~100% success rate. No change to
  shots count or circuit. (Initial slice-fix and N=15/N=21 regression tests
  contributed by [@zParik](https://github.com/zParik) in #7.)
- **`QPE::estimate_phase` bit-extraction fixed.** Pre-fix the routine read the
  leftmost `num_eval_qubits` characters of the measurement bitstring (the target
  register and upper eval qubits) with MSB-first endianness, masked because all
  shipped QPE tests used phase=0 eigenstates. It now reads the rightmost
  `num_eval_qubits` characters in LSB convention, consistent with the fixed IQFT.
  Added a comment directing multi-eigenstate callers (Shor-style) to iterate
  `result.counts` rather than rely on the single-most-frequent strategy.
- **`NoiseModel::add_quantum_error` honours `before_gate` ordering.** The public
  API silently overwrote `ge.after_gate = true`, dropping a `DensityMatrixSimulator`
  capability that has existed for several releases. Both `add_quantum_error` and
  `add_all_qubit_quantum_error` now take an `after_gate` parameter (default
  `true`); the stale "before_gate not yet implemented" comment was removed.
  (Reported by [@zParik](https://github.com/zParik) in #1.)
- **StatevectorSimulator thread-local RNG independence under parallel batches.**
  The RNG was reseeded with the raw caller `seed` on every `run()`, so OpenMP-
  dispatched batches (`Estimator::run_batch` and similar) had every thread reseeded
  to the same Mersenne Twister state, defeating shot independence. The seed is now
  mixed with `omp_get_thread_num()` via `std::seed_seq`; single-threaded
  reproducibility is preserved (tid is 0 outside any parallel region).
  (Reported by [@zParik](https://github.com/zParik) in #3.)

### Added

- `tests/test_qft_convention.cpp` — new regression suite pinning down the LSB
  convention end-to-end: QFT∘IQFT roundtrip, IQFT on uniform superposition,
  forward QFT against standard `QFT|x⟩` amplitudes for n=2/n=3, IQFT against
  expected `|m⟩` peaks for exact phases at n=4/n=5/n=11, IQFT spread test at
  n=11 for non-exact φ=1/6, end-to-end QPE on S gate with non-zero phase, and a
  Shor N=21 m-extraction diagnostic.
- `tests/test_bug_regression.cpp` — B6 tests rewritten to drive through the
  public `NoiseModel` API (the prior tests bypassed it and missed the bug);
  added `B6_AfterGateRemainsDefault`, `B9_StatevectorRngParallelIndependence`
  (OpenMP-gated), `B9_StatevectorRngSingleThreadedReproducible`.
- `CLAUDE.md` "Project Conventions" section and `docs/Architecture.md` "Qubit
  Ordering Convention" section formally declaring LSB-at-qubit-0 as the
  project-wide convention with worked example and the requirement that new
  algorithms include a non-symmetric end-to-end test.

### Changed

- `tests/test_shor.cpp` — `FindOrderA2N15HighSuccessRate` and
  `FindOrderA2N21HighSuccessRate` thresholds set to 18/20 (90%) with full
  theoretical derivation in test comments. The iterate-counts strategy now
  saturates these regimes near 100%.

## [R.1.10.4] - 2026-05-20

### Added

- `lindblad_draw` demo catalogue expanded from 5 entries to 13. The new
  algorithm demos exercise the library's own `build_circuit()` factories
  so the visualiser is showcased on output from real project code rather
  than hand-crafted gate sequences:
  - `--demo bv` : Bernstein-Vazirani n=3 with hidden string `101`,
    via `BernsteinVazirani::build_circuit`.
  - `--demo dj` : Deutsch-Jozsa n=3 with a balanced XOR oracle,
    via `DeutschJozsa::build_circuit`.
  - `--demo grover` : 3-qubit Grover, 1 iteration, marks `|011>`,
    via `Grover::build_circuit` with an X-conjugated CCZ oracle.
  - `--demo qpe` : phase estimation with 3 evaluation qubits and a
    T-gate (P(pi/4)) target, via `QPE::build_circuit`.
  - `--demo simon` : Simon's algorithm n=2 with a minimal period
    oracle, via `Simon::build_circuit`.
  - `--demo vqe` : two-layer hardware-efficient VQE ansatz on 3
    qubits (RY -> CX chain -> RY). Hand-built because VQE is an
    optimiser rather than a circuit factory.
  - `--demo qaoa` : one-layer QAOA on a 3-node triangle MaxCut
    (H init, RZZ on each edge, RX mixer). Hand-built; the
    SparsePauliOp-driven `QAOA::build_circuit` is not exposed on the
    CLI yet.
  - `--demo iqft` : 4-qubit inverse QFT subcircuit via
    `QFT::build_inverse_circuit`.
- The existing `--demo qft` now uses `QFT::build_circuit(4)` instead of
  a hand-crafted approximation, so the demo reflects the actual library
  output.

### Changed

- `README.md` no longer carries a `## Release` table. The section was a
  compressed duplicate of `CHANGELOG.md` and grew unbounded on every
  release. The Contents list entry pointing to it is removed; the
  version badge at the top still links to `CHANGELOG.md`, which is the
  single source of release notes.

### Fixed

Both of the layout bugs in this release were caught by visually
inspecting the new `lindblad_draw --demo qaoa` and `--demo qft`
outputs added in the same release. The QASM3 / catalogue / fixture
tests from R.1.10.0 through R.1.10.3 all passed against the buggy
layout because none of them rendered a non-adjacent multi-qubit gate
with another gate competing for an intermediate wire. Once the demos
exercised real algorithm circuits the visual ambiguity became
obvious, so the demos paid for themselves on day one.

- `src/visualisation/layout.cpp` now reserves every wire row in
  `[min(qubits)..max(qubits)]` whenever a glyph visually owns
  intermediate wires. Two trigger paths:
  - a `BoxPart` with `rowspan > 1` (TallBox composites
    RXX/RYY/RZZ/RZX/ECR, plus UNITARY); previously only UNITARY hit
    this rule. The QAOA demo's `RZZ(0, 2)` exposed the gap when
    `RX(1)` packed into the same column and overlapped on q1's wire.
  - a vertical strut spanning at least two qubit indices (non-adjacent
    `CX` / `CP` / `CZ` / `SWAP` and the controlled rotations). The QFT
    demo's `CP(theta, 3, 0)` etc. exposed this when the H boxes on
    intermediate qubits packed into the same column as the strut
    crossings.
  Adjacent struts (strut span 1, e.g. `CX(0, 1)`) keep the original
  tight packing behaviour. The trade-off is wider diagrams in exchange
  for unambiguous visual structure on non-adjacent multi-qubit gates.
- `tests/test_visualiser_layout.cpp` :
  - `CxBetweenDistantQubitsBlocksMiddleRows` was previously written
    under the OLD (buggy) semantics where `CX(0, 3)` left intermediate
    rows free. The test name was aspirational; the body now matches
    the new (correct) behaviour.
  - new `NonAdjacentTallBoxReservesIntermediateRows`: `RZZ(0, 2)` +
    `X(1)` serialise into two layers.
  - new `NonAdjacentControlledStrutReservesIntermediateRows`:
    `CX(0, 2)` + `H(1)` serialise into two layers.
  - new `AdjacentControlledGateStillPacksTightly`: `CX(0, 1)` +
    `H(2)` stay in a single layer (regression guard).

### Results

- 1045 tests across 84 suites; all passing (Linux/WSL, clang 18,
  Release). +3 tests over R.1.10.3 (three new layout tests pinning
  the intermediate-row reservation rule and its adjacent-strut
  exception).

## [R.1.10.3] - 2026-05-20

### Added

- `QuantumCircuit::draw_to_file(path, mode, opts)` : convenience wrapper
  around `draw()` that opens an `std::ofstream` in binary mode and
  streams the rendered output to `path`. Raises `std::runtime_error`
  with the offending path embedded in the message when the open
  fails, rather than silently dropping the output. Default mode is
  `DrawMode::ASCII`. The Python bindings expose this as
  `qc.draw_to_file("bell.svg", DrawMode.SVG)` for consistency, though
  direct C++ is the recommended path for performance-sensitive
  workflows (the Python wrapper crosses the binding boundary and
  serialises through the GIL).
- `lindblad_draw` : command-line frontend to the visualiser. Reads
  QASM2 (default) or QASM3 (`--qasm3`) from a file or `--stdin`, or
  picks a built-in demo via `--demo <name>` (bell, ghz, parametric,
  tallbox, qft). Output to stdout or `--output <path>`. Backend
  selected by `--mode ascii|svg|latex|html`. Options propagated:
  `--ascii-safe`, `--show-clbits`, `--no-show-params`,
  `--param-format pretty|raw`, `--fold N`, `--cell-px N`, `--legend`.
  `--list-demos` prints the built-in catalogue; `--help` shows usage.
  Built when `LINDBLAD_BUILD_APPS=ON` (default for top-level builds,
  OFF when consumed as a dependency).
- `apps/` directory and matching `LINDBLAD_BUILD_APPS` option for
  shipping user-facing executables alongside `lindblad_core`.

### Tests

- `tests/test_visualiser_file_io.cpp` (new file, 13 tests,
  `DrawToFileTest`): round-trip against ASCII / SVG / LaTeX / HTML
  (each rendered file matches the direct `draw()` output verbatim);
  options round-trip for `ascii_safe` and `show_clbits`; default
  mode is ASCII when none is supplied; file content starts with the
  expected prologue per backend (`<?xml`, `<!DOCTYPE html>`,
  `\begin{quantikz}`); error path on an unwritable destination
  throws `std::runtime_error` with the path embedded in the
  message; overwriting an existing file truncates seeded content
  rather than appending.

### Changed

- `tests/CMakeLists.txt` registers the new `test_visualiser_file_io.cpp`
  source under an `R.1.10.3 additions` comment.
- `CMakeLists.txt` adds the `LINDBLAD_BUILD_APPS` option and
  conditionally builds `apps/` when set.

### Results

- 1042 tests across 84 suites; all passing (Linux/WSL, clang 18,
  Release). +13 tests over R.1.10.2.

## [R.1.10.2] - 2026-05-20

### Fixed

- `src/visualisation/render_latex.cpp` now respects the
  `GateSymbol::latex_macro` override from the Tier 1 catalogue. Daggered
  gates (SDG, TDG, SXDG) emit `\gate{S^{\dagger}}` style output instead
  of `\gate{\text{S†}}`; axis-subscripted rotations (RX, RY, RZ, U1,
  U2, U3 and their `PARAM_*` variants) emit `\gate{R_X\text{(π/2)}}`
  instead of `\gate{\text{RX(π/2)}}`. The catalogue now stores the
  math-mode gate symbol only (e.g. `"R_X"`, `"S^{\\dagger}"`) and the
  renderer composes the `\gate{...}` wrapper at emission time, splitting
  the `BoxPart` label at the first `(` so the parameter suffix is still
  routed through `\text{...}` when it carries non-ASCII bytes.
- `src/visualisation/render_ascii.cpp` annotates conditional-gate
  c-wire crossings with the match value. Under `show_clbits = true`,
  every conditional now writes `=v` immediately after the c-wire cross
  glyph (`═╪=1═` for a `c[k] == 1` gate), so two conditionals with
  different match values are visually distinguishable.
- `src/visualisation/render_ascii.cpp` no longer forces odd column
  width when the layer contains no glyph that needs a centre cell.
  Pure box-only layers (TallBox interaction gates, UNITARY, single-
  qubit Tier 1 boxes) keep even widths so even-length labels fit
  without trailing padding. The visible effect: `RXX(π/4)` renders as
  `┤RXX(π/4)├` (10 cells) instead of `┤RXX(π/4) ├` (11 cells with one
  stray space inside the right border).

### Tests

- `tests/test_visualiser_layout.cpp`
  `DocumentLayoutTest.NonContiguousUnitaryReservesIntermediateRows`
  now looks up the UNITARY glyph by its actual `data_gate` value
  (`"U"` from the fixture's `inst.label`) rather than the fallback
  `"unitary"` used only when no label is provided.
- `tests/test_visualiser_format.cpp`
  `FormatParamsTest.ArbitraryDecimalFormatsWithFourDecimals` uses 0.5
  (exactly representable) instead of 0.31415 (which lands on a `%.4f`
  rounding boundary and rounded differently depending on the compiler
  build).
- `tests/test_visualiser_html_fixtures.cpp`
  `HtmlFixtureTest.LegendAbsentByDefault` looks for
  `<div class="lb-legend">` specifically. The `.lb-legend` CSS rule is
  declared in the page's `<style>` block on every render so user CSS
  overrides have a stable selector; only the legend `<div>` itself is
  gated on `opts.include_legend`.

### Goldens

- LaTeX golden files regenerated to reflect the new `latex_macro`
  emission for daggered and rotation gates.
- ASCII golden files regenerated for `tallbox` (no trailing padding
  on even-length labels) and `feedforward.show_clbits` (new `=v`
  annotation on the c-wire crossing).

### Results

- 1029 tests across 83 suites; all passing (Linux/WSL, clang 18,
  Release). The five R.1.10.1 known failures are resolved; no new
  failures introduced.

## [R.1.10.1] - 2026-05-20

### Note on release contents

This is the test-only release for the circuit visualiser introduced in
R.1.10.0. Per the project's `.0` / `.1` / `.2+` release cadence the `.1`
slot is for tests only. Five tests are shipping in a failing state and
will be resolved in R.1.10.2. They are documented under "Known failures"
below so anyone running the suite can see which ones are expected to
fail and why. The remaining 1024 tests pass under Linux/WSL clang 18.

### Tests

- `tests/test_visualiser_layout.cpp` (new file, 33 tests, `DocumentLayoutTest`):
  - **ASAP packing**: single gate at column 0; disjoint gates share a
    column; overlapping gates serialise; glyph column matches layer
    column; multiple glyphs per layer keep insertion order.
  - **Multi-qubit packing**: CX between adjacent qubits has a strut
    spanning rows; CX between distant qubits blocks the middle rows for
    other gates; CCX strut spans min to max; serialised CX gates that
    share a qubit produce two layers.
  - **Barriers**: BARRIER forces a full-width column break; empty
    barrier still blocks all rows; barrier with explicit qubits still
    forces a full break.
  - **UNITARY**: non-contiguous UNITARY reserves intermediate rows;
    contiguous UNITARY does not inflate the column count.
  - **Conditional gates**: with `show_clbits = false` two conditional
    gates on different qubits share a column; with `show_clbits = true`
    they serialise via the c-wire; condition_clbit / condition_value
    are recorded on the glyph; non-conditional gates carry the sentinel
    values.
  - **c-wire strut**: MEASURE with `show_clbits = true` extends the
    strut to the c-wire row; MEASURE with `show_clbits = false` omits
    it; conditional gate with `show_clbits = true` extends the strut.
  - **Labels**: qubit labels follow Qiskit's `q[i]` convention;
    clbit_labels stay empty unless `show_clbits = true`; 10-qubit case
    renders `q[9]`.
  - **Metadata**: data_gate matches `Instruction::gate_name()`;
    DrawOptions are captured on the document; MeasurePart / ResetPart
    / BarrierPart counts match the instruction list; layer count
    equals depth for serial chains.
- `tests/test_visualiser_catalogue.cpp` (new file, 38 tests, `GateCatalogueTest`):
  - **Tier 1 (`gate_symbols.cpp`)**: catalogue is non-empty; every
    single-qubit box gate has an entry (H, X, Y, Z, S, SDG, T, TDG, SX,
    SXDG, RX, RY, RZ, P, U, U1, U2, U3 plus the PARAM_* variants);
    Hadamard label is "H" with `show_params = false`; rotation gates
    have `show_params = true`; Pauli gates share a colour palette;
    dagger labels contain the Unicode dagger character; PARAM_*
    variants reuse the numeric-sibling label; `latex_macro` is filled
    only where the default macro is wrong.
  - **Tier 2 (`composite_catalogue.cpp`)**: catalogue is non-empty;
    every controlled / multi-bullet gate has an entry; CX is
    `CtrlBullet + XorTarget`; CZ is two `CtrlBullet`s (no
    DotTargetPart in the variant); CCZ is three `CtrlBullet`s; CCX is
    two CtrlBullets plus XorTarget; CY / CH route the box label
    correctly; SWAP / CSWAP use SwapX arms; CRX includes the param in
    the box label and respects opts.show_params = false; iSWAP gives
    both slots the same label.
  - **TallBox (Tier 2 special role)**: RXX produces a single BoxPart
    with `rowspan = 2` and no strut; RYY / RZZ / RZX share this
    structure; ECR uses TallBox without params; RXX label includes
    the parameter.
  - **Tier 3 (`gate_builders.cpp`)**: BARRIER emits one BarrierPart
    per touched qubit (empty list yields empty parts); MEASURE
    records the clbit field on the part (sentinel -1 when absent);
    RESET emits a ResetPart on the target; UNITARY uses the provided
    label or "U" fallback; rowspan covers non-contiguous ranges; an
    empty qubit list returns an empty glyph.
  - **Dispatch**: build_glyph routes MEASURE through Tier 3, CX
    through Tier 2, H through Tier 1; data_gate stamping matches
    `gate_name()`.
- `tests/test_visualiser_format.cpp` (new file, 41 tests, `FormatParamsTest`):
  - **Numeric pi-snap**: every positive entry in the table (pi, 2pi,
    3pi, 4pi, 3pi/2, pi/2, 2pi/3, pi/3, 3pi/4, pi/4, 5pi/6, pi/6,
    pi/8) and the negative counterparts (-pi, -pi/2, -3pi/4, -pi/8).
  - **Edge cases**: zero formats as bare "0" (both positive and
    negative zero); misses fall back to "%.4f"; raw mode never
    pi-snaps; raw mode zero formats as "0.0000".
  - **ParamExpr**: Literal nodes format via the double overload;
    Name nodes pass through unchanged; Add / Sub / Mul / Div render
    with their operator (multiplication uses the UTF-8 middle dot);
    lower-precedence children of `*` get parens; equal-precedence
    chains skip parens; higher-precedence children of `+` skip
    parens; right-child of `*` also gets parens for lower-precedence
    children.
  - **format_gate_label**: Hadamard label is bare "H"; RX label
    appends "(pi/2)" with show_params on; U label includes three
    pi-formatted params; opts.show_params = false strips the parens;
    symbolic param_exprs take precedence over numeric params; raw
    param_format propagates into the label; an RX with no params at
    all produces the bare "RX" label.
- `tests/test_visualiser_palette.cpp` (new file, 24 tests, `PaletteTest`):
  - **ASCII Unicode default**: U+2500 wire, U+2524 / U+251c box
    borders, U+25cf control bullet, U+2295 XOR target, U+2502 strut
    pipe, U+2715 SWAP arm, U+250a barrier all appear in the
    appropriate fixtures.
  - **ASCII safe**: output contains no byte >= 0x80; wire becomes
    "-"; box borders become "[" / "]"; control bullet becomes "*".
  - **SVG**: every `.lb-*` class is present in the inline style
    block; glyphs carry `data-gate` attributes; Hadamard family uses
    #e8eef9 fill; Pauli family uses #fce8e8 fill; barrier uses
    dashed stroke.
  - **LaTeX (Quantikz)**: plain Hadamard emits `\gate{H}`; CX emits
    `\ctrl{1}` and `\targ{}`; SWAP emits `\swap{1}` and `\targX{}`;
    MEASURE emits `\meter{}`; RESET emits `\push{\ket{0}}`; BARRIER
    emits `\barrier[\dashed]`; TallBox emits `\gate[2]`.
- `tests/test_visualiser_folding.cpp` (new file, 5 tests, `FoldingTest`):
  pins the current pre-folding ASCII behaviour. `fold_width = 0`
  sentinel matches the default opts; small fold_width does not crash;
  no "... fold ..." marker yet; long lines tolerated. Tests will
  invert when folding is implemented in a future patch.
- `tests/test_visualiser_ascii_fixtures.cpp` (new file, 16 tests,
  `AsciiFixtureTest`): 13 golden-file fixtures plus three structural
  invariants (no UTF-8 under ascii_safe, output ends with newline,
  empty circuit produces non-empty output).
- `tests/test_visualiser_svg_fixtures.cpp` (new file, 12 tests,
  `SvgFixtureTest`): 6 golden-file fixtures plus six structural
  invariants (XML prolog, UTF-8 encoding declaration, inline style
  block, data-* attribute presence, </svg> closer, one wire line per
  qubit).
- `tests/test_visualiser_latex_fixtures.cpp` (new file, 14 tests,
  `LatexFixtureTest`): 7 golden-file fixtures plus seven structural
  invariants (\begin{quantikz} envelope, math-mode row prefixes, \qw
  defaults, c-wire \setwiretype{c} appears under show_clbits, no
  \documentclass shell, legend comment when include_legend = true).
- `tests/test_visualiser_html_fixtures.cpp` (new file, 11 tests,
  `HtmlFixtureTest`): 3 golden-file fixtures plus eight structural
  invariants (DOCTYPE, charset meta, <html>/<head>/<body> shell,
  embedded <svg>, lb-meta caption, .lb-glyph:hover rule, lb-legend
  toggle).

### Tooling

- `tests/visualiser_fixtures.hpp` : shared fixture circuit factories
  used by both the test binaries and the regen tool.
- `tests/golden_helpers.hpp` : `load_golden(rel_path)` helper that
  resolves paths via the `LINDBLAD_TEST_GOLDEN_DIR` compile definition.
- `tests/visualiser_regen.cpp` : standalone `lindblad_visualiser_regen`
  binary that walks every fixture x backend x variant combination and
  writes ~60 golden files to `tests/golden/visualisation/<backend>/`.
  Not wired into ctest; run manually when intentionally regenerating.
- `tests/golden/visualisation/{ascii,svg,latex,html}/` : new committed
  golden file tree.

### Known failures (deferred to R.1.10.2)

Five tests fail in this release. They are documented here so anyone
running the suite knows which ones are expected to fail:

- `DocumentLayoutTest.NonContiguousUnitaryReservesIntermediateRows` :
  test-side issue. The fixture's `gate_name()` for UNITARY does not
  match the `"x"` lookup; the assertion shape needs reworking.
- `FormatParamsTest.ArbitraryDecimalFormatsWithFourDecimals` :
  test-side issue. The chosen test value `0.31415` is on a
  floating-point rounding boundary so `%.4f` produces either
  `"0.3141"` or `"0.3142"` depending on bit pattern. Pick a less
  ambiguous value.
- `PaletteTest.LatexDaggerEmitsTexBackslashDagger` : implementation
  gap. `src/visualisation/render_latex.cpp` does not consult the
  `GateSymbol::latex_macro` field, so `S^{\dagger}` overrides for
  daggered gates are not emitted. Wire the catalogue lookup into the
  LaTeX renderer.
- `PaletteTest.LatexRotationUsesSubscript` : same root cause as the
  dagger gap above. `R_X` / `R_Y` / `R_Z` macros are unused.
- `HtmlFixtureTest.LegendAbsentByDefault` : test-side issue. The
  `.lb-legend` CSS class is always declared in the page's `<style>`
  block; the test should look for `<div class="lb-legend">`
  specifically.

Two cosmetic gaps were noticed while reviewing the golden files and
also belong in R.1.10.2:

- Conditional gate decoration omits the `c[k]=v` tag near the c-wire
  in ASCII output. The strut drops correctly but the value annotation
  is missing.
- TallBox padding leaves a trailing space inside the label box on the
  anchor row and an over-wide empty box on the bottom row.

### Results

- 1029 tests across 83 suites; 1024 passing, 5 known failures (Linux/
  WSL, clang 18, Release).

## [R.1.10.0] - 2026-05-20

### Added

- `QuantumCircuit::draw(DrawMode, DrawOptions)` : new public API replacing the
  primitive `to_ascii()` rendering with a layered, parameter-aware visualiser.
  Four output backends behind a single layout pass:
  - `DrawMode::ASCII` : monospaced UTF-8 grid with `ascii_safe` portable
    fallback (`-|+*X[]`). Layered ASAP packing, multi-row tall boxes for
    `UNITARY` and the `TallBox` interaction gates, conditional decoration on
    the bundled c-wire.
  - `DrawMode::SVG` : self-contained SVG with inline `<style>`, semantic
    `.lb-*` class names, and `data-gate` / `data-col` / `data-qubits`
    attributes on every `<g class="lb-glyph">` for downstream interactivity.
    No external CSS, no external fonts.
  - `DrawMode::LATEX` : a `quantikz` environment (no `\documentclass` shell).
    Gate boxes via `\gate{}` / `\gate[N]{}`, controls via `\ctrl{offset}` /
    `\octrl{offset}`, targets via `\targ{}`, swaps via `\swap{offset}` /
    `\targX{}`, measurements via `\meter{}`, resets via `\push{\ket{0}}`,
    barriers via `\barrier[\dashed]{N}`. Defensive `\text{...}` wrap for
    labels with parens, commas, or non-ASCII glyphs.
  - `DrawMode::HTML` : a standalone HTML page embedding the SVG with hover
    styling (no JavaScript). Page-level CSS targets the SVG's existing
    `data-*` attributes for `:hover` rules.
- `DrawOptions` : per-call configuration covering `fold_width` (ASCII wrap
  column; 0 disables), `show_clbits` (hide the bundled c-wire by default),
  `show_params`, `ascii_safe`, `param_format` (`ParamFormat::Pretty` snaps
  pi/2, pi/4, etc.; `ParamFormat::Raw` always uses `%.4f`), `cell_width_px`,
  `cell_height_px`, `include_legend`.
- `include/lindblad/visualisation.hpp` : lightweight public re-export header
  for callers that only need the option types.
- Three-tier gate catalogue under `src/visualisation/`:
  - `gate_symbols.cpp` : Tier 1 declarative `GateSymbol` table for the 22
    single-qubit box gates (Hadamard, Pauli, phase, sqrt(X), U-family, plus
    the symbolic `PARAM_*` variants).
  - `composite_catalogue.cpp` : Tier 2 declarative `CompositeGate` table for
    `CX CY CZ CH SWAP ISWAP CRX CRY CRZ CP CU CCX CCZ CSWAP RCCX`, with a
    new `TallBox` role for the symmetric two-qubit interaction gates
    `RXX RYY RZZ RZX ECR`. CZ and CCZ use `CtrlBullet` on every involved
    qubit (symmetric phase, no dedicated dot-target primitive).
  - `gate_builders.cpp` : Tier 3 hand-written builders for `BARRIER`,
    `MEASURE`, `RESET`, `UNITARY` (whose visuals do not fit either table).
- Backend-agnostic intermediate model in `src/visualisation/document.hpp`:
  closed 7-kind `GlyphPart` variant (`BoxPart`, `CtrlBulletPart`,
  `XorTargetPart`, `SwapXPart`, `MeasurePart`, `ResetPart`, `BarrierPart`),
  `Glyph`, `Layer`, `CircuitDocument`.
- `format_params.cpp` : pi-snap table covering 0, +/- pi/8, pi/6, pi/4, pi/3,
  pi/2, 2pi/3, 3pi/4, 5pi/6, pi, 3pi/2, 2pi, 3pi, 4pi within tolerance 1e-6;
  `ParamExpr` recursive renderer with precedence-aware paren wrapping and
  middle-dot multiplication.
- `layout.cpp` : ASAP packing driver `build_document` with full-width barrier
  column breaks, non-contiguous `UNITARY` row reservation, conditional gate
  c-wire serialisation when `show_clbits` is on, and post-build c-wire strut
  extension for measure / conditional glyphs.
- Python bindings expose `lindblad.DrawMode`, `lindblad.ParamFormat`,
  `lindblad.DrawOptions`, and `QuantumCircuit.draw(mode, opts)`.
- `docs/api/visualisation.md` : new full reference covering DrawMode,
  ParamFormat, DrawOptions, examples for all four backends, the three-tier
  catalogue architecture, and known limitations.

### Changed

- `QuantumCircuit::to_ascii()` is now a thin compatibility wrapper around
  `draw(DrawMode::ASCII, {})`. The original ad-hoc per-wire concatenation
  with hard-coded gate dispatch has been deleted from `src/circuit.cpp`; new
  code should call `draw()` directly.
- `docs/api/circuit.md` : replaced the ASCII Visualization section with a
  pointer to `docs/api/visualisation.md`.
- `docs/APIOverview.md` : added `visualisation.md` to the Circuit
  Construction & Manipulation deep-dive link list.

### Deferred

- Matplotlib (`mpl`) backend stays out of `DrawMode`. A `Figure` object can
  only be constructed from Python, so MPL support belongs in a future
  Python-bindings deliverable layered on top of the SVG renderer.
- Multi-line ASCII gate boxes for long parameter labels. Single-line boxes
  cover the current need; widening the column suffices for `U(0.7854, ...)`.
- ASCII folding for very wide circuits. `DrawOptions::fold_width` is wired
  through; the implementation lands when the ergonomic case is real.
- R.1.10.1 test suite for the new visualiser per the project's release
  cadence: this release adds only a `CircuitTest.DrawSmoke` smoke test
  exercising all four `DrawMode` values on a measured Bell circuit.

## [R.1.9.1] - 2026-05-20

### Note on release contents

This is nominally a `.1` test-only release for the QASM 3 parser introduced
in R.1.9.0. Four parser bugs were uncovered while running the new suite:
one of them caused a SEGFAULT mid-execution, which prevented the rest of
the suite from running. Shipping the tests without these fixes would leave
the project in an unusable state, so the fixes are folded into this
release alongside the tests rather than deferred to R.1.9.2. The fixes
are narrow regression patches strictly required for the new tests to
execute end-to-end. No new features.

### Tests
- `tests/test_qasm3_parser.cpp` (new file, 202 tests, 2 suites):
  - **Register declarations** : `qubit[N]`, `bit[N]`, single-qubit/bit
    without brackets, legacy `qreg`/`creg`, multiple registers, missing
    qubit register throws.
  - **Pragmas, comments, whitespace** : `OPENQASM`/`include` skipped,
    `//` and `/* */` comments, tabs/CR tolerated.
  - **Standard gates (1-qubit no-param)** : `h`, `x`, `y`, `z`, `s`,
    `sdg`, `t`, `tdg`, `sx`, `sxdg`, `id` (dropped).
  - **Standard gates (1-qubit parameterised)** : `rx`, `ry`, `rz`, `p`,
    `phase`, `u1`, `u2`, `u3`, `u`, `U`.
  - **Standard gates (2-qubit no-param)** : `cx`, `CX`, `cy`, `cz`,
    `ch`, `swap`, `iswap`, `ecr`.
  - **Standard gates (2-qubit parameterised)** : `crx`, `cry`, `crz`,
    `cp`, `cphase`, `rxx`, `ryy`, `rzz`, `rzx`.
  - **Standard gates (3-qubit)** : `ccx`, `toffoli`, `ccz`, `cswap`,
    `fredkin`, `rccx`.
  - **Measurements, reset, barrier** : both `c[i] = measure q;` and
    `measure q -> c[i];` forms, error on missing classical target, bare
    `reset`, explicit-qubit barrier, full-circuit barrier.
  - **Modifier `ctrl @`** : maps `x`/`y`/`z`/`h`/`swap` to their
    controlled named gates, double-`ctrl` maps `x`/`z` to `ccx`/`ccz`,
    `ctrl @ rx`/`ry`/`rz`/`p`/`phase` to controlled rotations.
  - **Modifier `inv @`** : self-inverse no-ops (H, X, CX, ...),
    pair swaps (`s` ↔ `sdg`, `t` ↔ `tdg`, `sx` ↔ `sxdg`), angle
    negation on `rx`/`ry`/`rz`, double-negation cancels.
  - **Modifier `pow(n) @`** : `pow(0)` drop, `pow(even)` on
    self-inverse drops, `pow(odd)` on self-inverse keeps, `pow(n)`
    on rotations scales the angle, negative exponents supported.
  - **Chained modifiers** : `inv @ pow(3) @ rx(θ)`, `ctrl @ inv @ rx(θ)`,
    `ctrl @ pow(2) @ x` collapses to identity.
  - **Matrix fallback path** : `pow(2) @ s` becomes UNITARY,
    `inv @ iswap` throws (multi-qubit base unsupported), `pow(3) @ t`
    builds a 2x2 UNITARY, `ctrl @ pow(2) @ s` extends to a 4x4 UNITARY.
  - **Parameter expressions** : `pi`/`tau`/`euler` constants,
    `pi/2`, `-pi`, operator precedence (mul/div over add/sub),
    parenthesised expressions, subtraction, division, float exponent
    literals, unary `+`, double-negation, complex nested expressions.
  - **Custom gate definitions** : no-param, parametric, multi-qubit,
    arithmetic in body, nested calls, param-count/qubit-count mismatch
    throws, modifier on user-defined gate throws.
  - **Classical conditioning** : single-statement `if`, block `if`,
    `else` flips condition value, bare `if (c == V)` form for
    single-bit registers, multi-bit `if (c == V)` throws, condition
    state does not leak past the block.
  - **Symbolic parameters** : `input float` registers a parameter
    name; `param_exprs` populated for symbolic angles; `bind_parameters`
    resolves to numeric `params`, throws on missing binding, skips
    numeric instructions, merges into `parameter_bindings`.
  - **ParamExpr factories and eval** : `make_literal`, `make_name`,
    `make_binary` for `+`/`-`/`*`/`/`, nested expressions, missing
    binding throws, division by zero throws, deep-copy semantics on
    copy ctor and copy-assign, move-construct preserves value.
  - **Peephole optimisation** : cancels `h;h`, `x;x`, `cx;cx`,
    `ccx;ccx`, triple-H reduces to single, respects qubit ordering
    (`cx q[0],q[1]` then `cx q[1],q[0]` does not cancel), does not
    cancel different types or different qubits, skips conditioned
    gates, skips parametric gates, cancels across unrelated qubit
    activity.
  - **Multi-register offset resolution** : two `qubit` registers
    produce contiguous global indices, two `bit` registers produce
    contiguous classical offsets, unknown qubit/classical register
    throws.
  - **Round-trips** : Bell state, GHZ, rotation chain, barrier, reset
    all survive `to_qasm3() -> from_qasm3()`.
  - **Error reporting** : unknown gate throws, `for`/`while`/`def`/
    `delay`/`stretch`/`box`/`cal` constructs throw, missing required
    args throw with descriptive message, extra qubit args throw,
    unexpected lexer character throws, missing semicolon throws.
  - **Lexer details** : empty input throws, comments-only throws,
    header-only (no qubit) throws, INT/FLOAT/leading-dot/trailing-dot/
    exponent literals all parse, multiple statements on one line.
  - **Identity gate with modifiers** : `inv @ id`, `pow(n) @ id`, and
    `ctrl @ id` all drop (identity is invariant under any modifier).
  - **Larger integration tests** : 3-qubit GHZ with measurements,
    parametric ansatz with `bind_parameters`, modifier-heavy circuit,
    custom gate with symbolic call site, measurement inside `if` block
    carries the condition, peephole interaction with barriers.
  - **UTF-8 identifier names** : Greek letter `θ` as input parameter
    name round-trips through bind correctly.

### Fixed
- `src/qasm/qasm3_parser.cpp` : classical-assignment dispatch in
  `parse_statement` now recognises bracket-less `c = measure q;` for
  single-bit registers in addition to the indexed `c[i] = measure q[j];`
  form.
- `src/qasm/qasm3_parser.cpp` : `pow(-n)` (negative integer exponent) is
  now accepted by both `parse_gate_call` and `parse_body_call`. An
  optional leading `-` or `+` is consumed before the `INT`, and the sign
  is folded into `pow_exp`.
- `src/qasm/qasm3_parser.cpp` : `input float` parameter names registered
  during `first_pass()` were being discarded by the subsequent
  `qc_ = QuantumCircuit(n_qubits_, n_clbits_)` reset in `run()`. They are
  now saved and restored across the reconstruction so
  `parameter_names` reflects the source.
- `src/qasm/qasm3_parser.cpp` : `emit_matrix_fallback` now validates the
  parameter arity against the gate's `BuiltinSpec` before calling
  `build_1q_base`. Previously a missing-angle call such as `rx q[0];`
  fell through to the fallback and dereferenced `params[0]` on an empty
  vector, causing a SEGFAULT. The fallback now throws
  `QASM3Parser: gate 'rx' expects 1 parameter(s), got 0`.
- `src/qasm/qasm3_parser.cpp` : the `id` gate is now dropped for **any**
  modifier stack (`id`, `inv @ id`, `pow(n) @ id`, `ctrl @ id`), since
  identity is invariant under inversion, exponentiation, and control
  extension. Previously `inv @ id` would slip past the early drop and
  emit an H instruction by mistake (because the `id` table entry used
  `GT::H` as a placeholder type).

### Results
- 834 tests from 74 test suites ran. All passed (Linux/WSL, 3.4s wall).

## [R.1.9.0] - 2026-05-20

### Added
- `src/qasm/qasm3_parser.cpp` — full OpenQASM 3.0 parser replacing the previous stub.
  `QASM3Lexer` is a single-pass, zero-copy tokenizer (tokens are `std::string_view` into
  the source). `QASM3Parser` is a recursive-descent parser covering: multi-register qubit
  and bit declarations (`qubit[N] name;` plus legacy `qreg name[N];` form), gate modifiers
  `ctrl @ / inv @ / pow(n) @` with arbitrary chaining, the `stdgates.inc` library, user-
  defined `gate` bodies with recursive inlining and parameter substitution, classical
  `if (c[i] == V) { ... } else { ... }` conditioning, `measure`/`reset`/`barrier`, and
  symbolic `input float[N] θ;` parameters. Unknown gates and timing/loop constructs throw
  `std::runtime_error` with the offending token and line number (no silent skips).
- `include/lindblad/circuit.hpp` — new `ParamExpr` struct (Literal / Name / BinaryOp tree)
  for symbolic parameter expressions, with deep-copy semantics. `Instruction::param_exprs`
  added alongside the existing numeric `params` so the parser can emit symbolic gates
  without forcing eager evaluation.
- `include/lindblad/circuit.hpp` — `QuantumCircuit::bind_parameters(bindings)` evaluates
  every instruction's `ParamExpr` tree against the supplied bindings, populates `params`,
  and clears `param_exprs`. Designed for VQE-style parameter sweeps over a single circuit.
- `src/qasm/qasm3_parser.cpp` — modifier resolution prefers named fast paths
  (`ctrl @ x → cx`, `ctrl @ ctrl @ x → ccx`, `ctrl @ z → cz`, `ctrl @ rx(θ) → crx(θ)`,
  `inv @ s → sdg`, `inv @ rx(θ) → rx(-θ)`, `pow(n) @ rx(θ) → rx(n·θ)`). Combinations that
  don't map to a named gate fall back to explicit matrix composition (1-qubit base
  matrix → conjugate-transpose on `inv` → binary-exponentiation on `pow(n)` → block-
  diagonal extension on each `ctrl`) and are emitted as `UNITARY`.
- `src/qasm/qasm3_parser.cpp` — parse-time peephole window: tracks a per-qubit history
  stack of live instruction indices and cancels self-inverse pairs (`h; h`, `cx; cx`,
  etc.) on identical qubit lists. `pow(0) @ <gate>` is dropped during modifier
  resolution. Cancellations are recorded in a parallel boolean vector and swept at the
  end so mid-vector erases don't invalidate other indices.

### Changed
- `src/circuit.cpp` — `QuantumCircuit::from_qasm3()` now forwards to the new
  `qasm3_parse_impl()` bridge function instead of throwing "not yet implemented".

### Documentation
- `docs/api/qasm.md` — new API deep-dive page covering both QASM 2 and QASM 3
  parsers and serialisers: header/namespace, lexer + parser internals, supported
  constructs, modifier-resolution table (named fast paths + matrix fallback),
  parse-time peephole semantics, the `ParamExpr` + `bind_parameters()` symbolic
  parameter workflow, the error catalogue for unsupported constructs, three
  round-trip examples (Bell, VQE ansatz with `input float`, gate modifiers),
  and a common-pitfalls section.
- `README.md` — added `docs/api/qasm.md` row to the API Reference Pages table.
- `docs/APIOverview.md` — Circuit Construction section now mentions both QASM
  dialects and the `ParamExpr` / `bind_parameters()` workflow; added
  `docs/api/qasm.md` to the deep-dive links.
- `docs/api/circuit.md` — added `param_exprs` to the `Instruction` field list,
  documented `QuantumCircuit::bind_parameters()` alongside `assign_parameters`,
  and rewrote the `from_qasm3()` description with the full feature list.

## [R.1.8.2] - 2026-05-20

### Fixed
- `src/algorithms/shor.cpp` — `find_order`: two `int` bit-shift overflows corrected to `1ULL` (`m` promoted to `uint64_t`; phase divisor `1 << n_eval` → `1ULL << n_eval`). Overflows triggered silently for `n_eval > 30`, i.e. all `N` requiring more than 15 target qubits.

### Tests
- `tests/test_shor.cpp` — `FindOrderA2N15` / `FindOrderA7N15`: replaced single-shot lenient checks with 10-seed loops asserting at least one valid order is returned; removes the unconditional `r == 0` pass.
- `tests/test_shor.cpp` — `CfConvergents*` (7 tests): direct tests of `Shor::cf_convergents` covering exact rational phases (1/4, 1/2, 1/3), irrational approximation (golden ratio, max-denom enforcement), and edge cases (`max_denom=1`, `x≈0`, `x≈1`).
- `tests/test_shor.cpp` — `UnitaryGatesAreUnitary`: verifies that every `UNITARY` instruction in the period-finding circuit satisfies U†U = I to 1e-10.
- `include/lindblad/algorithms.hpp` — `Shor::cf_convergents` promoted to `public static` method to enable direct testing.

### Results
- 632 tests from 72 test suites ran. All passed.

## [R.1.8.1] - 2026-05-20

### Tests
- `tests/test_shor.cpp` — New test suite for `algorithms::Shor` covering classical pre-screening paths (even N, perfect powers, trial GCDs), exception handling (N < 4, prime N), circuit structural validation (qubit counts, Hadamard initialization, target register |1⟩, UNITARY gate count, no measurements), direct order-finding verification (ord₁₅(2), ord₁₅(7)), backend parity (DM, MPS), seed reproducibility, and Options/Result field correctness.

### Results
- 624 tests from 72 test suites ran. All passed.

## [R.1.8.0] - 2026-05-20

### Added
- `algorithms::Shor` — integer factorisation via quantum order finding (Shor 1994).
  `Shor::factorize(N)` performs classical pre-screening (even N, perfect powers,
  small trial GCDs) then quantum period finding via a QPE-based circuit.
  `Shor::build_period_finding_circuit(a, N, n_eval, n_target)` and
  `Shor::find_order(a, N, n_eval, backend)` are exposed for testing and composition.
  Supports STATEVECTOR (default), DENSITY_MATRIX, and MPS backends.
  CLIFFORD backend is not supported (modular exponentiation is non-Clifford).
  Practical for N ≤ ~100; larger N requires exponentially more simulation memory.

### Documentation
- `docs/algorithms/shor.md` — Algorithm explanation page covering theory, circuit architecture, backend compatibility, and usage examples.
- `docs/api/shor.md` — API deep dive for the `Shor` class, `Options`, `Result`, and all public methods.
- `docs/APIOverview.md` — Added `algorithms::Shor` to the class index and deep dive links.
- `docs/MasterDocumentation.md` — Added `shor.md` to algorithm documentation map.

## [R.1.7.8] - 2026-05-19

### Tests
- `tests/test_primitives.cpp` — New test suite covering `Estimator` and `Sampler` primitives, including exact statevector evaluation, analytical parameter-shift gradient verification, noisy path dispatch, and batch execution consistency.
- `tests/test_dag.cpp` — New test suite for `DAGCircuit` covering topological sorting, parallel layer extraction, dependency tracking, and graph properties.
- `tests/test_qasm_parser.cpp` — New test suite for QASM 2.0 parsing covering `QASM2Parser`, multi-register layouts, and gate resolution.

### Documentation
- `docs/api/estimator.md` — Updated the `run_single` exact path to accurately reflect the R.1.7.7 optimization using `StatevectorSimulator::eval_expectation`.
- `docs/api/simulators.md` — Added a "Fast Expectation Values" section detailing the `eval_expectation` API and its memory benefits.
- `docs/MasterDocumentation.md` — Bumped the documentation status tracker to R.1.7.8.

### Results
- 594 tests from 71 test suites ran. All passed.


## [R.1.7.7] - 2026-05-19

### Changed
- `StatevectorSimulator::eval_expectation` — new method; runs circuit into the
  thread-local working buffer and returns ⟨ψ|H|ψ⟩ directly, with no
  `Result::final_state` allocation. For variational hot paths (VQE, QAOA).
- `Estimator::run_single` — ideal path (no noise, shots=0) now calls
  `eval_expectation` instead of `run().final_state`.

### Results
- 556 tests from 67 test suites ran. All passed.

## [R.1.7.6] - 2026-05-19

### Fixed

- **`src/transpiler/layout/sabre_layout.cpp`**: `SabreLayout::run`, `sabre_run`, `sabre_heuristic`, and `SABRERunResult` extracted from `trivial_layout.cpp` into their own translation unit. File was previously an empty stub; implementation now lives where the build system and structural convention expect it.
- **`src/transpiler/layout/trivial_layout.cpp`**: Stripped to `TrivialLayout::run` only; removed SABRE-specific includes (`<algorithm>`, `<limits>`, `<numeric>`, `<unordered_map>`, `<vector>`) that are no longer needed.
- **`tests/test_bug_regression.cpp`**: B8 header comment updated to reflect the structural fix resolved in R.1.7.6.

## [R.1.7.5] - 2026-05-19

### Fixed

- **`src/simulators/mps_sim.cpp`** (B3): `mps_apply_instruction` now intercepts `UNITARY` gates before the size-based dispatch; the 2-qubit path previously fell through to `gate4x4`'s identity default, silently dropping every custom unitary matrix. `mps_from_sv` now bit-reverses statevector indices on read, reconciling the MPS (qubit 0 = MSB of site index) and statevector (qubit 0 = LSB) conventions.
- **`src/simulators/density_matrix_sim.cpp`** (B6): `apply_inst` lambda now applies `before_gate` Kraus channels before the gate unitary; they were previously silently ignored while `after_gate` channels were applied correctly.
- **`src/primitives/estimator.cpp`** (B5): `run_single` routes through `DensityMatrixSimulator` when `options.noise_model` is non-ideal or `options.shots > 0`. The exact statevector path is retained for the ideal zero-shot case.
- **`src/circuit.cpp`** (B7): `to_qasm2()` no longer emits `UNITARY` or `PARAM_*` gates as comments. 1-qubit `UNITARY` → ZYZ Euler decomposition to `u(θ,φ,λ)`; multi-qubit `UNITARY` → custom gate block definition + call; `PARAM_*` → standard gate name with symbolic parameter string.

### Tests

- **`tests/test_bug_regression.cpp`**: New file — 14 black-box regression tests covering all 8 open bugs from R.1.7.4 (B1 `from_qasm2`, B2 custom register names ×2, B3 MPS UNITARY, B4 DM RCCX ×3, B5 Estimator noisy path ×2, B6 `before_gate` noise ×3, B7 `to_qasm2` UNITARY/PARAM ×3, B8 SabreLayout ×2). 556 tests total across 67 suites — all passing.

## [R.1.7.4] - 2026-05-19

### Changed

- **`NOTICE`**: Rewritten — updates stale Q++/v1.0 content to Lindblad SLA v2.1; adds full third-party component notices for Eigen (MPL-2.0), NLopt (LGPL-2.1, static-link relinking note), GoogleTest (BSD 3-Clause), Google Benchmark (Apache 2.0), and pybind11 (BSD 3-Clause); includes source-download warning for test and benchmark files in GitHub archives.
- **`LICENSE`**: Revised to Version 2.1. §14 Third-Party Components added — points to `NOTICE` for full notices; states that non-commercial licensees' obligations under third-party licenses are independent of this Agreement; omits commercial context (addressed in individual license agreements).
- **`CMakeLists.txt`**: `cmake_minimum_required` bumped to 3.21; `EIGEN_MPL2_ONLY` compile definition added to `lindblad_core` (restricts Eigen inclusion to MPL-2.0-only headers); `LINDBLAD_BUILD_TESTS` option added defaulting to `PROJECT_IS_TOP_LEVEL` with guarded `enable_testing()` / `add_subdirectory(tests)` and conditional `FetchContent_MakeAvailable(googletest)`; `LINDBLAD_BUILD_BENCHMARKS` default changed from `ON` to `PROJECT_IS_TOP_LEVEL`; build options moved before `FetchContent_MakeAvailable` calls so all fetches are conditional.
- **`lindblad-page/src/pages.tsx`**: `ThirdPartyNotices` component and section added to the license page listing all five dependencies with license identifiers and usage notes. `KEY_TERMS` corrected for SLA v2.1 structure: §3↔§4 section numbers swapped (§3 = Redistribution, §4 = Commercial use); §3 body updated to reflect §3.1 permitted private sharing; standalone Modifications row merged into §2 (Permitted use); §7 Citations renamed §5 Attribution & citations; §14 Third-party components row added. PATH A card: redistribution bullet updated to reflect §3.1 permitted private sharing. All SLA version references updated to v2.1.
- **`lindblad-page/src/components/chrome.tsx`**: SLA version updated to v2.1.

## [R.1.7.3] - 2026-05-19

### Changed

- **LICENSE**: Revised to Version 2.0. §3 rewritten: §3.1 permits private non-commercial redistribution of unmodified copies to individual peers and collaborators under same-license passthrough, no-charge, and attribution-preservation conditions; §3.2 enumerates prohibited acts (public distribution, modified distribution without authorization, commercial use, sublicensing). Description updated to "C++23 quantum computing framework" throughout. §6.6 added — Author may acknowledge contributors in documentation or release notes at their sole discretion; former §6.6 renumbered §6.7 (No Monetary Compensation), clarified to distinguish non-monetary acknowledgment from compensation.
- **`CITATION.cff`**: `abstract` and `preferred-citation.title` updated to "C++23 Quantum Computing Framework"; `license` updated to `LicenseRef-Lindblad-2.0`.
- **`lindblad-page/src/pages.tsx`**: Website synced to current project state. Capability matrix: QFT row added (Statevector native, DM/Clifford/MPS supported). Algorithm catalog: QFT entry added (standard · inverse · approximate · semi-classical feedforward variants; all four simulator backends; new `Transform` filter family); header corrected from "Eight families" to "Nine families"; lead paragraph updated to describe the `QuditBackend` surface (Statevector · DM · MPS · Clifford for prime d) used by the qudit algorithm suite. Architecture section: QFT and qudit generalizations (BV · DJ · Grover · QPE · Simon) added to ALGORITHMS row; qudit simulator variants added to SIMULATORS row. License page: lead text corrected to reflect v2.0 private redistribution permission; all SLA version references updated to v2.0.
- **`lindblad-page/src/components/chrome.tsx`**: SLA version updated to v2.0 in footer.

## [R.1.7.2] - 2026-05-17

### Fixed

- **`QuditCliffordSimulator::apply_H`** (`src/qudit/qudit_clifford.cpp`): Corrected
  Hadamard conjugation rule. Was `x ← z, z ← −x`; correct qudit QFT rule is
  `x ← −z mod d, z ← x`. Phase update `phase −= 2·old_x·old_z` was already correct.

- **`QuditCliffordSimulator::apply_CSUM` / `apply_CSUM_dag`** (`src/qudit/qudit_clifford.cpp`):
  Fixed Gottesman-style phase cross-term. Was `phase −= 2·z_c·x_t`; correct
  term (from reordering Z_t^{b_t} past X_c^{a_c} in normal form) is
  `phase −= 2·x_c·z_t` (and `+= 2·x_c·z_t` for the dagger).

- **`QuditCliffordSimulator::measure_qudit`** (`src/qudit/qudit_clifford.cpp`):
  Rewrote deterministic-outcome path. Solves a linear system over Z_d to find
  the unique stabilizer combination equal to Z_q, reads the outcome from that
  combination's phase via `phase + 2·m·k ≡ 0 (mod 2d)`, and collapses the
  tableau to the post-measurement state. The previous scan returned the outcome
  but did not update the tableau.

- **`QuditBernsteinVazirani::solve` (Clifford path)** (`src/algorithms/bernstein_vazirani.cpp`):
  Measures the ancilla once, then recovers each query qudit from an independent
  Clifford state snapshot to avoid cross-talk between successive stateful
  `measure_qudit` calls.

- **`QuditDensityMatrix::apply_to_bra` / `apply_to_bra_2q` / `apply_kraus_1qudit` / `apply_lindblad_step`**
  (`src/qudit/qudit_density_matrix.cpp`): Removed pre-built U†/K†/L† auxiliary
  matrices; inline `U[k·d+l].conj()` directly in the multiply loop. The bra-side
  transform requires the element-wise conjugate conj(U), not the conjugate-transpose
  U† — the previous code applied the wrong operation.

### Changed

- **`docs/api/qudit-simulators.md`**: Corrected Gate API table — `apply_H` rule
  (`x_q ← −z_q, z_q ← x_q`), `apply_CSUM` / `apply_CSUM_dag` phase cross-terms
  (`2·x_c·z_t`; dagger term was previously omitted). Extended measurement
  description to document the linear-system solve used for determined outcomes.
  Fixed broken Simon anchor (`#quditsimone` → `#quditsimonsalgorithm`). Added
  `test_qudit_simulators.cpp` reference.
- **`docs/api/qudit.md`**: Added `tests/test_qudit_simulators.cpp` to Related Files.
- **`docs/algorithms/bernstein-vazirani.md`**: Replaced stale "distributed BV tests are
  planned for R.1.4.1" note with the completed test reference (`tests/test_classic_algorithms.cpp`, R.1.4.1).

## [R.1.7.1] - 2026-05-17

### Tests

- `tests/test_qudit_simulators.cpp` — 90 tests across 14 suites covering the full R.1.7.0 backend suite:
  - `QuditDensityMatrix` (20 tests): zero-state initialisation, purity, statevector construction, `apply_1qudit`/`apply_2qudit` agreement with `QuditStatevector`, trace preservation after gates (d=3), Kraus trace preservation (depolarising d=2), purity decay (d=2/d=3), amplitude damping population decay, `apply_noise` dispatch, Lindblad Euler step, `measure` collapse, `partial_trace` (zero state, entangled state, empty-keep throws)
  - `QuditMPS` (12 tests): zero-state init (d=2/d=3), statevector round-trip (d=2/d=3/d=5), norm unity after gates, `apply_1qudit` agreement with SV (d=2/d=3), `apply_2qudit` adjacent and non-adjacent agreement with SV, `measure` zero-state, `left_canonicalize` norm preservation
  - `QuditClifford` (13 tests): tableau initialisation (d=2/d=3), `apply_X` phase update, `apply_H` X/Z swap and double-H identity, `apply_CSUM` target-X and control-Z updates, `measure` zero-state (d=2/d=3), H-then-measure uniform distribution (d=2), non-prime d=4/d=6 throws, prime d=5 constructs
  - `QuditNoiseModel` (8 tests): Kraus operator counts for depolarising (d=2/d=3), amplitude damping (d=2/d=3), phase damping (d=2/d=3), `add_depolarizing` registration, `amplitude_damping_lindblad` rate
  - `QuditBV_Backend` (8 tests): STATEVECTOR/DENSITY_MATRIX/MPS/CLIFFORD backends recover secret (d=2); Clifford d=3 and d=5; all-backends agreement on d=2
  - `QuditBV_Errors` (2 tests): invalid-d and zero-length-secret throw paths
  - `QuditDJ_Backend` (6 tests): constant and balanced oracles across STATEVECTOR/DENSITY_MATRIX/MPS backends (d=3)
  - `QuditDJ_Errors` (1 test): invalid-d throw
  - `QuditGrover_Backend` (4 tests): STATEVECTOR/DENSITY_MATRIX/MPS success rate; CLIFFORD throws
  - `QuditQPE_Backend` (4 tests): phase recovery across STATEVECTOR/DENSITY_MATRIX/MPS; CLIFFORD throws
  - `QuditQPE_Errors` (2 tests): invalid-d and mismatched-size throws
  - `QuditSimon_Backend` (6 tests): period recovery across STATEVECTOR/DENSITY_MATRIX/MPS for d∈{3,5,7}; CLIFFORD throws
  - `DM_SV_Agreement` (2 tests): density-matrix trace and purity match statevector reference (d=2/d=3) after gate sequence
  - `MPS_SV_Agreement` (2 tests): MPS amplitude agreement with statevector after multi-gate circuit (d=2/d=3)

### Results

- 538 tests across 66 suites — 526 passed, 12 failed (798 ms, WSL/Clang). Failures resolved in R.1.7.2.

## [R.1.7.0] - 2026-05-17

### Added

- **`QuditNoiseModel`** — `include/lindblad/qudit/qudit_noise_model.hpp`, `src/qudit/qudit_noise_model.cpp`:
  - `depolarizing_channel(d, p)` — d²-Kraus-operator depolarizing channel for a single qudit
  - `amplitude_damping_channel(d, gamma)` — d-operator amplitude damping
  - `phase_damping_channel(d, gamma)` — (d+1)-operator phase damping
  - `lindblad_op(d, L)` — single Lindblad operator wrapped as a `QuditLindbladOp`
  - `add_depolarizing` / `add_amplitude_damping` / `add_phase_damping` convenience methods that register channels into the `per_qudit` map for algorithm dispatch

- **`QuditDensityMatrix`** — `include/lindblad/qudit/qudit_density_matrix.hpp`, `src/qudit/qudit_density_matrix.cpp`:
  - Full mixed-state simulation of d^n-dimensional qudit systems; ρ stored as dim²-element flat vector (row-major)
  - `apply_1qudit(q, U)` / `apply_2qudit(q0, q1, U)` — superoperator gate application via `apply_to_ket` + conjugate `apply_to_bra` sweep
  - `apply_kraus_1qudit(q, ops)` / `apply_kraus_2qudit(q0, q1, ops)` — Kraus channel application
  - `apply_noise(model)` — dispatches all per-qudit Kraus channels from a `QuditNoiseModel`
  - `lindblad_step(ops, dt)` — first-order Euler Lindblad master-equation step
  - `apply_function_oracle(n_query, f)` / `apply_phase_oracle(phase_fn)` — unitary oracle application
  - `measure(seed)` — projective measurement with state collapse; returns digit vector
  - `partial_trace(keep_qudits)` — traces out all qudits not in `keep_qudits`; returns reduced density matrix
  - `purity()` — `Tr(ρ²)` (1.0 for pure states)

- **`QuditMPS`** — `include/lindblad/qudit/qudit_mps.hpp`, `src/qudit/qudit_mps.cpp`:
  - Tensor-network representation with `MPSSiteTensor{d, chi_L, chi_R, data}`; Eigen BDCSVD-based left-canonical construction from statevector
  - `apply_1qudit(q, U)` — O(d²·χ²) single-site contraction
  - `apply_2qudit(q0, q1, U)` — adjacent (SVD split, O(d²·χ_L·χ_R + SVD)) or non-adjacent (SWAP chain)
  - `apply_function_oracle(n_query, f)` / `apply_phase_oracle(phase_fn)` — via `to_statevector()` + reconstruct fallback
  - `to_statevector()` — O(d^n) full reconstruction; `norm()` / `left_canonicalize()`
  - `measure(seed)` — single-shot sampling

- **`QuditCliffordSimulator`** — `include/lindblad/qudit/qudit_clifford.hpp`, `src/qudit/qudit_clifford.cpp`:
  - Stabilizer tableau for prime-d qudit systems; 2n rows (rows 0..n-1 = destabilizers, rows n..2n-1 = stabilizers)
  - Heisenberg-Weyl Pauli group with phase tracking mod 2d; tableau columns = (xbits, zbits, phase) per row
  - `apply_X(q)` / `apply_Z(q)` / `apply_H(q)` / `apply_CSUM(ctrl, tgt)` with full conjugation rules
  - `measure(q, seed)` — syndrome extraction; deterministic if qudit is in eigenstate, probabilistic otherwise
  - Throws `std::invalid_argument` for non-prime d

- **`QuditBackend` enum** — `include/lindblad/qudit/qudit_backend.hpp`: `STATEVECTOR`, `DENSITY_MATRIX`, `MPS`, `CLIFFORD`

- **Backend dispatch for all qudit algorithms** — `include/lindblad/algorithms.hpp`, `src/algorithms/{bernstein_vazirani,deutsch_jozsa,grover,qpe,simon}.cpp`:
  - `QuditBernsteinVazirani::solve` — all four backends supported; CLIFFORD is valid (BV uses only Clifford gates)
  - `QuditDeutschJozsa::solve` — STATEVECTOR/DENSITY_MATRIX/MPS supported; CLIFFORD falls back to STATEVECTOR (function oracle not Clifford-simulable in general)
  - `QuditGrover::search` / `search_with_oracle` — STATEVECTOR/DENSITY_MATRIX/MPS supported; CLIFFORD throws `std::invalid_argument`
  - `QuditPhaseEstimation::estimate` — STATEVECTOR/DENSITY_MATRIX/MPS supported; CLIFFORD throws `std::invalid_argument`
  - `QuditSimon::solve` — STATEVECTOR/DENSITY_MATRIX/MPS supported; CLIFFORD throws `std::invalid_argument`
  - Noise (`QuditNoiseModel*`) applied only in the DENSITY_MATRIX path; silently ignored in all others

- **`CMakeLists.txt`** — four new source files added to `lindblad_core`: `src/qudit/qudit_noise_model.cpp`, `src/qudit/qudit_density_matrix.cpp`, `src/qudit/qudit_mps.cpp`, `src/qudit/qudit_clifford.cpp`

### Documentation

- `docs/api/qudit-simulators.md` — new: full API reference for all four qudit backend simulators; backend × algorithm compatibility matrix; usage examples for DM+noise BV, MPS QPE, and Clifford BV; Related Algorithm Pages links
- `docs/api/qudit.md` — updated: backend table expanded to 5 columns (Memory, Restrictions); "Clifford stabilizer (prime d)" corrected
- `docs/algorithms/bernstein-vazirani.md` — updated: QuditBV "Simulator Dependency" replaced with Backend table; Required Inputs expanded with `backend`/`noise`/`shots`/`seed`
- `docs/algorithms/deutsch-jozsa.md` — updated: QuditDJ "Simulator Dependency" replaced with Backend table (CLIFFORD fallback noted); Required Inputs expanded
- `docs/algorithms/grover.md` — updated: QuditGrover "Simulator Dependency" replaced with Backend table (CLIFFORD throws); Exceptions updated
- `docs/algorithms/qpe.md` — updated: QuditQPE "Simulator Dependency" replaced with Backend table (CLIFFORD throws); Exceptions updated
- `docs/algorithms/simon.md` — updated: QuditSimon "Simulator Dependency" replaced with Backend table (CLIFFORD throws); Exceptions updated
- `docs/APIOverview.md` — updated: five qudit algorithm variants added to the Algorithms class list; `qudit-simulators.md` added to deep-dives
- `README.md` — version badge updated to R.1.7.0; R.1.7.0 release row added; `docs/api/qudit-simulators.md` added to API Reference table

## [R.1.6.1] - 2026-05-15

### Tests

- `tests/test_qudit_bv.cpp` — 45 tests: `QuditStatevector` construction, normalisation, `apply_1qudit`, `apply_2qudit`, `measure`; `QuditGates` unitarity and entry-correctness for d ∈ {2,3,4,5,6,7}; `QuditBernsteinVazirani` secret recovery for n=1,2 and d ∈ {2,3,4,5,6,7}; input validation (invalid d, zero-length secret)
- `tests/test_qudit_algorithms.cpp` — 63 tests: `QuditDeutschJozsa` constant/balanced for d ∈ {2,3,5,7}; `QuditGrover` auto-iteration formula correctness, `search` success rate, `search_with_oracle` predicate path; `QuditPhaseEstimation` phase recovery for rational φ; `QuditSimon` period recovery for d ∈ {3,5,7} with GF(d) scalar-equivalence assertions; `shift_eigenstate` construction for d ∈ {2,3,5}. (2 N=2 Grover tests omitted — P=0.5 is the theoretical maximum for a 2-state search regardless of target.)

### Results

- 448 tests across 52 suites — all passed (788 ms, WSL/CLang).

## [R.1.6.0] - 2026-05-15

### Added

- **Qudit statevector layer** — `include/lindblad/qudit/qudit_statevector.hpp`, `src/qudit/qudit_statevector.cpp`:
  - `QuditStatevector(n, d)` — d^n-dimensional statevector with mixed-radix (little-endian) indexing; `amplitudes[0] = 1` at construction
  - `apply_1qudit(q, U)` — applies d×d unitary to qudit q via stride-d tensor contraction
  - `apply_2qudit(q0, q1, U)` — applies d²×d² unitary to qudits (q0, q1); base-index scan avoids full d^n pass
  - `apply_kqudit(qudits, U)` — applies d^k×d^k unitary to k distinct qudits; dispatches to 1/2-qudit fast paths for k≤2
  - `apply_function_oracle(n_query, n_output, f)` — unitary oracle `|x⟩|y⟩ → |x⟩|(y+f(x)) mod d⟩` via amplitude permutation
  - `apply_phase_oracle(phase_fn)` — per-basis-state phase multiplication; used by Grover oracle and diffusion
  - `measure(seed)` — cumulative-probability sampling; returns `vector<int>` of d-ary digits
  - `index_to_digits` / `digits_to_index` — static mixed-radix conversion helpers

- **Qudit gate library** — `include/lindblad/qudit/qudit_gates.hpp`, `src/qudit/qudit_gates.cpp`:
  - `qft_matrix(d)` — d×d QFT: `F[j,k] = ω^{jk}/√d`, `ω = exp(2πi/d)`
  - `iqft_matrix(d)` — d×d inverse QFT (conjugate transpose of `qft_matrix`)
  - `shift_matrix(d, m)` — d×d forward shift: `X|k⟩ = |(k+m) mod d⟩`; `X[j,k] = 1` if `j = (k+m) mod d`
  - `cadd_matrix(d, s)` — d²×d² controlled-ADD: `|x⟩|y⟩ → |x⟩|(y+s·x) mod d⟩`
  - `controlled_power_matrix(d, U, k)` — d²×d² gate where clock value `c` applies `U^{c·k}` to the target qudit; built via `mat_pow` (binary exponentiation, O(log m) matrix multiplications)

- **Qudit simulator** — `include/lindblad/qudit/qudit_simulator.hpp`, `src/qudit/qudit_simulator.cpp`:
  - `QuditSimulator::run(sv, ops, seed)` — applies a sequence of `QuditGateOp` (SINGLE or TWO type) to a statevector, then measures once; returns outcome digits and wall-clock time

- **`QuditBernsteinVazirani`** — `src/algorithms/bernstein_vazirani.cpp`:
  - Recovers secret `s ∈ Z_d^n` from `f(x) = s·x mod d` in a single quantum query; ancilla prepared in `|−⟩_d = F_d·X^{d-1}|0⟩`; CADD oracle applies controlled-ADD per query qudit; IQFT decodes phase kickback; majority-vote over `shots` for robustness
  - `oracle_gate(d, s_i)` — returns `cadd_matrix(d, s_i)` for use in custom circuit construction
  - Works for any `d ≥ 2` including composite d

- **`QuditDeutschJozsa`** — `src/algorithms/deutsch_jozsa.cpp`:
  - Distinguishes constant from balanced `f: Z_d^n → Z_d` in one oracle query; ancilla `|−⟩_d` picks up phase `ω^{f(x)}`; all-zero measurement → CONSTANT, any non-zero digit → BALANCED
  - Requires `d ≥ 2`; works for any n ≥ 1

- **`QuditGrover`** — `src/algorithms/grover.cpp`:
  - QFT-based amplitude amplification for d-ary search: each iteration applies the phase oracle (`−1` to marked state), `F_d†`, `R_0` (phase `−1` to non-`|0...0⟩`), `F_d`
  - Auto-iteration count uses exact formula `round(π/(4·arcsin(1/√N)) − 0.5)` (correct for all N; approximation `round(π/4·√N)` was wrong for N=4 and N=5)
  - `search(n, d, target, ...)` — target given as `vector<int>`; delegates to `search_with_oracle`
  - `search_with_oracle(n, d, is_marked, ...)` — predicate oracle; shots histogram; returns mode outcome and empirical probability

- **`QuditPhaseEstimation`** — `src/algorithms/qpe.cpp`:
  - Estimates eigenphase `φ ∈ [0,1)` of a d×d unitary `U` given its eigenstate: `m` clock qudits in little-endian order; clock qudit `j` applies `controlled_power_matrix(d, U, d^j)`; d-ary IQFT on clock; phase decoded as `φ = Σ_j digit_j / d^{j+1}`
  - Input validation: `d ≥ 2`, `m ≥ 1`, `U` must be `d×d`, eigenstate must have length `d`

- **`QuditSimon`** — `src/algorithms/simon.cpp`:
  - Finds hidden period `s ∈ Z_d^n` of `f(x) = f(y) ↔ x − y ≡ 0 or s (mod d)` using O(n) quantum queries and GF(d) Gaussian elimination
  - Quantum circuit: 2n qudits; `F_d` on query register, `apply_function_oracle`, `F_d†` on query, measure; collect n-1+extra_samples non-zero non-duplicate vectors
  - Classical post-processing: `null_space_gf` — reduced row echelon form over GF(d); modular inverse via Fermat's little theorem (`a^{p-2} mod p`)
  - Requires `d` prime (GF(d) field structure); composite d throws `std::invalid_argument`

### Fixed

- **`shift_matrix` forward-shift convention** (`src/qudit/qudit_gates.cpp`) — implementation set `X[row=j, col=(j+m)%d] = 1` (backward shift: `X|k⟩ = |(k−m) mod d⟩`); corrected to `X[row=(k+m)%d, col=k] = 1` (forward shift: `X|k⟩ = |(k+m) mod d⟩`). This was the root cause of all QuditBV secret-recovery failures for `d ≥ 3` (ancilla prepared in `|1⟩` instead of `|d−1⟩`, yielding `−s mod d` instead of `s`)
- **`qudit_grover_auto_iters` exact formula** (`src/algorithms/grover.cpp`) — replaced `round(π/4·√N)` with `round(π/(4·arcsin(1/√N)) − 0.5)` to minimise `|(2R+1)θ − π/2|`; old formula gave R=2 for N=4 (optimal R=1, P=1.0) and N=5 (optimal R=1, P≈0.97)

### Documentation

- `docs/algorithms/bernstein-vazirani.md` — added full `QuditBernsteinVazirani` section: purpose, theory (phase kickback in Z_d^n), circuit diagram, gate definitions, complexity, invocation, supported d, Result, exceptions, simulator dependency, pitfalls
- `docs/api/bernstein-vazirani.md` — added `QuditBernsteinVazirani` section: `Result`, `solve`, `oracle_gate`, exceptions; updated Family Overview
- `docs/algorithms/deutsch-jozsa.md` — added `QuditDeutschJozsa` section: circuit, phase kickback, promise, invocation, Result, exceptions, simulator dependency
- `docs/api/deutsch-jozsa.md` — added `QuditDeutschJozsa` section: `Verdict` enum, `Result`, `solve` signature
- `docs/algorithms/grover.md` — added `QuditGrover` section: circuit table, iteration formula derivation, `search` + `search_with_oracle` examples, Result, exceptions
- `docs/api/grover.md` — added `QuditGrover` section: `Result`, `search`, `search_with_oracle` signatures and throws
- `docs/algorithms/qpe.md` — added `QuditPhaseEstimation` section: m+1 qudit circuit, little-endian phase decode, `controlled_power_matrix` construction, invocation, pitfalls
- `docs/api/qpe.md` — added `QuditPhaseEstimation` section: `Result`, `estimate` signature
- `docs/algorithms/simon.md` — added `QuditSimon` section: 2n-qudit circuit, GF(d) Gaussian elimination, prime-d restriction, invocation, pitfalls
- `docs/api/simon.md` — added `QuditSimon` section: `Result`, `solve` signature, `extra_samples`, prime-d note
- `docs/api/qudit.md` — new file: qudit layer overview (`QuditStatevector`, `QuditGates`, `QuditSimulator`); implemented algorithm list; design notes on simulator extensibility
- `README.md` — version badge updated to R.1.6.0; R.1.6.0 release row added; `docs/api/qudit.md` added to API Reference table

## [R.1.5.1] - 2026-05-14

### Tests

- `tests/test_feedforward_qft.cpp` — new file: 9 test suites, 100 tests covering feedforward infrastructure and semi-classical QFT:
  - `CircuitFeedforwardAPI` (17) — `p_if`/`add_if` circuit construction: type, qubit, clbit, angle, and `condition_value` fields; fluent chaining; out-of-range qubit/clbit throws
  - `FeedforwardStatevector` (7) — conditional-X on SV: gate applied/skipped for `clval` 0 and 1, two independent classical bits triggering different targets, non-conditional gates after `p_if` still execute, total shots preserved
  - `FeedforwardDensityMatrix` (7) — same conditional-X matrix on DM; Bell-state correctness in single-pass (no-feedforward) mode; mid-circuit DM collapse with feedforward active; total shots preserved
  - `FeedforwardClifford` (15) — conditional-X; `P(0)`=I, `P(π/2)`=S, `P(π)`=Z, `P(3π/2)`=SDG behaviour; non-Clifford `P(π/4)` throws from direct sim and returns `success=false` from `LocalBackend`; `is_clifford()` recognition for all Clifford angles; conditional `p_if` with Clifford angle on Clifford backend
  - `FeedforwardMPS` (3) — conditional-X: gate applied, gate skipped, and `clval=0` case
  - `BuildIterativeCircuit` (17) — qubit/clbit counts for n=1..4; exact instruction count for n=1..4; type, qubit, clbit, angle, and condition per position for n=2 and n=3; all `MEASURE` instructions map qubit j→clbit j; all `p_if` instructions use `clval=1`; invalid n throws
  - `BuildIterativeInverseCircuit` (13) — same structural checks; ascending qubit order (opposite of forward); negative angles; forward vs inverse angle-sign comparison for n=2; same total instruction count as forward for n=1..5
  - `RunIterativeCorrectness` (18) — `shots=0` throws on both overloads; n=1 `|+⟩`→"0" and `|−⟩`→"1" deterministically on all 4 backends; `N2_FeedforwardPGateExercised`: `|+i⟩⊗|−⟩` input gives deterministic "11" on all 4 backends (proves the conditional `P(π/2)` gate actually fired); uniform distribution for n=2 `|00⟩` input; result fields (`n_qubits`, `clifford_compatible`, `success`, total shots)
  - `RunIterativeInverse` (6) — n=1 deterministic on SV; `N2_InverseFeedforwardExercised`: `|−⟩⊗|+i⟩` input gives deterministic "01" on SV, DM, and MPS (proves the negative-angle conditional `P(-π/2)` gate fired); total shots preserved
- `tests/CMakeLists.txt` — registered `test_feedforward_qft.cpp`

### Results

- 340 tests across 45 suites — all passed (921 ms, WSL/Clang).

## [R.1.5.0] - 2026-05-14

### Added

- **`QFT` class** — new standalone algorithm (`include/lindblad/algorithms.hpp`, `src/algorithms/qft.cpp`):
  - `QFT::build_circuit(n)` — exact n-qubit quantum Fourier transform circuit
  - `QFT::build_inverse_circuit(n)` — inverse QFT (IQFT)
  - `QFT::build_aqft_circuit(n, m)` — approximate QFT (Kitaev/Coppersmith) with `m`-level phase truncation
  - `QFT::apply(circuit, opts)` — compose QFT onto an existing circuit; returns `QFTResult`
  - `QFT::run(circuit, shots, seed, opts)` — execute; returns `QFTResult` with backend result, qubit count, and Clifford-compatibility flag
  - `Options` struct: `approximation_degree` (0 = exact), `do_swaps` (default true)
  - Clifford-simulable for n ≤ 2 (exact) or AQFT with `m = 1`
  - `build_iterative_circuit(n)` — semi-classical (Griffiths-Niu) forward QFT: processes qubits `n-1` down to `0`; classically-conditioned `P(π/2^{k-j})` gates replace quantum `CP` gates, driven by prior measurement outcomes `c[k]`; output is `n` classical bits
  - `build_iterative_inverse_circuit(n)` — inverse semi-classical QFT: processes qubits `0` up to `n-1`; `P(-π/2^{j-k})` feedforward rotations
  - `run_iterative(input_state, backend, shots, seed)` — compose `input_state` with the iterative QFT circuit and execute on any of the four backends; `shots` must be `> 0`
  - `run_iterative(input_state, shots, seed)` — convenience overload using the default Statevector backend

- **Feedforward infrastructure** — classically-conditioned gate execution across the full framework:
  - `QuantumCircuit::p_if(angle, qubit, clbit, clval=1)` — apply `P(angle)` only if `clreg[clbit] == clval`; primary primitive for semi-classical algorithms
  - `QuantumCircuit::add_if(clbit, clval, type, qubits, params)` — general conditional gate builder for any `GateType`
  - All four simulators now honour `condition_clbit`/`condition_value` on every instruction:
    - **StatevectorSimulator** — condition check inserted before `apply_instruction` in the per-shot dispatch loop
    - **DensityMatrixSimulator** — new feedforward path: when any instruction has a condition, the simulator switches to per-shot mode; each shot re-initialises from `|0⟩⟨0|`, collapses the density matrix on `MEASURE` (project + renormalise), records outcomes in `clreg[n_clbits]`, and evaluates gate conditions; single-pass mode unchanged for circuits without feedforward
    - **CliffordSimulator** — replaced per-shot `bitstring[q]` indexing with a proper `clreg[n_clbits]` vector; `MEASURE` now writes to `inst.clbits[0]`; condition check added before gate dispatch; `P` gate mapped to equivalent Clifford operation by angle (`P(0)=I`, `P(π/2)=S`, `P(π)=Z`, `P(3π/2)=SDG`); `is_clifford()` updated to accept `P` gates whose angles are Clifford
    - **MPSSimulator** — condition check added in per-shot dispatch (classical register and mid-circuit collapse already existed from R.1.3.2)

### Changed

- `src/algorithms/qpe.cpp` — refactored inline IQFT to use `QFT::build_inverse_circuit`

### Fixed

- `docs/api/qpe.md` — typo: `tatic double estimate_phase` → `static double estimate_phase`
- `docs/algorithms/qpe.md` — corrected measurement description: implementation calls `measure_all()`, not selective evaluation-register measurement
- `docs/api/deutsch-jozsa.md` — removed false `substr(1)` claim; bitstring is an `n`-bit direct comparison with no ancilla stripping
- `docs/api/simon.md` — removed false `2n`/`substr(n)` claims; classical register is `n` bits wide, bitstring reversed directly
- `docs/algorithms/qaoa.md` — removed stale "optimizer not wired" note; `optimizer` field is fully wired (COBYLA, NELDER_MEAD, POWELL)
- `docs/api/maqaoa.md` — corrected `orbits_by_power()` description: sorts powers ascending first (order-independent); removed "order-dependent" claim
- `docs/api/backends.md` — fixed noise model example: `add_gate_error(chan, {"h","cx"})` → two separate `add_quantum_error(chan, gate)` calls per gate
- `docs/api/simulators.md` — fixed `DensityMatrixSimulator` workflow example: `noise_model.gate_error(...)` → `noise_model.add_quantum_error(NoiseChannels::depolarizing(...), gate, qubits)`

### Documentation

- `docs/algorithms/qft.md` — new file: exact QFT, AQFT (Kitaev/Coppersmith), IQFT, and semi-classical (Griffiths-Niu) iterative QFT theory; circuit description, complexity table, usage examples for all variants, Clifford-simulability analysis for n ≤ 2 and AQFT m=1, testing notes
- `docs/api/qft.md` — new file: complete API reference for all `QFT` methods — `build_circuit`, `build_inverse_circuit`, `build_aqft_circuit`, `apply`, `run`, `build_iterative_circuit`, `build_iterative_inverse_circuit`, and both `run_iterative` overloads — with inline examples, exception tables, and `Options` struct documentation
- `docs/MasterDocumentation.md` — added `qft.md` to algorithm documentation map; updated Current Priority section
- `README.md` — complete rewrite: structured documentation map table, algorithm list with class names and notes columns, build options and dependency license tables; added QFT, Dispatch, and Ising entries; removed stale "planned pages" line; added `walkthrough.md` link
- `walkthrough.md` — added `docs/algorithms/` and `docs/api/` to Related Documentation section
- `.gitignore` — added `commercial_prep.md`

## [R.1.4.1] - 2026-05-12

### Tests

- `tests/test_classic_algorithms.cpp` — added 10 unit tests for `DistributedBernsteinVazirani` covering edge cases (single party equivalence, multi-party equal/unequal splits, all-zero and all-one secrets) and algorithmic verification (quantum rounds == 1 vs classical rounds == t, party secret string slicing)

### Results

- 233 tests across 36 suites — all passed (WSL/Clang)

## [R.1.4.0] - 2026-05-12

### Added

- `include/lindblad/algorithms.hpp` — `DistributedBernsteinVazirani` class: t-party distributed BV where the n-bit secret is partitioned as `s = S_{n_0} || ... || S_{n_{t-1}}`; each party holds a local BV oracle on `(n_j + 1)` qubits; combined circuit remaps local qubit indices to the global register with a shared ancilla; recovers the full secret in one quantum round vs t classical rounds; circuit depth `2^max(n_j) + 3` vs `2^n + 3` for monolithic BV
- `src/algorithms/bernstein_vazirani.cpp` — `DistributedBernsteinVazirani::build_circuit` and `::solve` implementation appended to the BV source file

### Documentation

- `docs/algorithms/bernstein-vazirani.md` — added `DistributedBernsteinVazirani` to family list; added theory, circuit steps, complexity table, API reference, usage example, and common pitfalls sections; added **Future Work** section documenting Qudit BV as a planned architectural extension requiring d-dimensional gate support
- `docs/api/bernstein-vazirani.md` — added `DistributedBernsteinVazirani` to family overview; added `Party` struct, `build_circuit`, `solve`, and `Result` API sections
- `README.md` — version badge and release line updated to `R.1.4.0`
- `docs/api/backends.md` — version example updated to `R.1.4.0`

### Notes

- `R.1.4.1` (next patch) is reserved for the `DistributedBernsteinVazirani` test suite per dev-workflow §4 (C bump → mandatory .1 test release before any .2+ patches).

## [R.1.3.2] - 2026-05-11

### Added

- `tests/CMakeLists.txt` — added `diagnose` standalone executable (no gtest dependency) for direct algorithm output inspection

### Fixed

- **F-1** `src/simulators/statevector_sim.cpp` — `run()` called `sample_counts` on an already-collapsed statevector when MEASURE gates were present, producing identical outcomes for every shot; replaced with a per-shot re-execution path that re-initialises the statevector from `|0...0⟩` each shot, collapses on each MEASURE, and records outcomes into the `n_clbits`-wide classical register; fast `sample_counts` path retained for circuits without MEASURE instructions
- **F-2** `src/simulators/mps_sim.cpp` — same root cause as F-1 for the MPS backend; added per-shot re-execution path mirroring the statevector fix; extracted `mps_apply_instruction()` helper to eliminate duplicated gate-dispatch code between the per-shot and fast paths; `to_statevector()` left-to-right contraction placed qubit 0 at the MSB position — framework convention is qubit 0 = bit 0 (LSB); added index bit-reversal after contraction so `fidelity` and `sample_counts` comparisons against statevector results are correct
- **F-3** `src/simulators/clifford_sim.cpp` — `expectation_pauli`: Y-qubit loop incremented `p_phase` during Pauli string parsing, double-counting the i phase already tracked by `pauli_phase_update`; removed the extra increment
- **F-4** `src/quantum_info/metrics.cpp` — `partial_trace(rho, qubits)`: the `qubits` argument specifies qubits to **trace out**; implementation was treating it as the keep set, returning a subsystem of the wrong qubits
- **F-5** `src/quantum_info/operators.cpp` — `expectation_value` and `expectation_value_batch`: Z parity was computed from the output row index `k` instead of the input column index `j = k ^ x_mask`; Y count was computed per basis state (`popcount(k & y_mask)`) instead of per Pauli term (`popcount(y_mask)`, which is constant)
- **Frontend** `src/banner.cpp` — Updated the shutdown banner to display a thank-you message instead of a welcome message

### Tests

- `tests/test_simulators_r130.cpp` — `DM_QubitOrdering::CZ_Symmetric`: old assertion compared `unordered_map::begin()` (undefined iteration order); replaced with set-based comparison of non-zero bitstring keys and a per-key check that qubit 1 never fires

### Documentation

- `docs/algorithms/deutsch-jozsa.md` — updated `solve` description: per-shot classical register returns `n`-char bitstrings; no `substr` stripping needed
- `docs/api/bernstein-vazirani.md` — same correction; removed `substr(1)` ancilla-strip description
- `docs/algorithms/bernstein-vazirani.md` — same correction
- `docs/api/operators.md` — `partial_trace` signature updated from `keep_qubits` to `trace_out_qubits`; semantics clarified
- `docs/api/simulators.md` — execution flow step 5 updated to describe per-shot re-execution path for circuits with MEASURE gates
- `docs/api/backends.md` — version example updated to `R.1.3.2`

### Results

- 223 tests across 35 suites — all passed (WSL/Clang)

## [R.1.3.1] - 2026-05-09

### Tests

- `tests/test_simulators_r130.cpp` (new) — 24 regression tests covering R.1.3.0 fixes: C-1 RZX sign pattern, C-2 SV MEASURE/RESET, C-3 Clifford rowmult phase, C-4 DM qubit ordering, H-1/H-2 Clifford expectation_pauli i^k phase, H-3 DM expectation_value_sparse X/Y terms, C-9 QPE IQFT bit-reversal
- `tests/test_mps_sim.cpp` (new) — 18 tests covering MPS simulator: basic gate application, Bell/GHZ state fidelity vs statevector, to_statevector roundtrip, measure_sequential, probabilities_single, RESET, truncation error
- `tests/test_gates_extended.cpp` (new) — extended gate coverage: SDG, TDG, SX/SXdg, P, U/U1/U2/U3, CY, CH, iSWAP, CRX/CRY/CRZ, CP, ECR, RXX/RYY/RZZ, CCZ, CSWAP, apply_unitary
- `tests/test_operators_qi.cpp` (new) — SparsePauliOp algebra (compose, tensor, to_matrix, batch expectation), Operator (compose, adjoint, power, trace), QuantumInfo (state_fidelity, process_fidelity, entropy, entanglement_entropy, partial_trace), DensityMatrix (purity, trace, is_valid)
- `tests/test_ising.cpp` (new) — IsingHamiltonian (from_hJ, from_qubo, evaluate, evaluate_spins, to_sparse_pauli_op), SoftDispatchResult (threshold_round, top_k, expected_cost)

### Results

- 223 tests from 35 suites — 193 passed, 30 failed (672 ms, Linux/Clang)
- Failures: DeutschJozsa (2), BernsteinVazirani (5), RecursiveBernsteinVazirani (5), ProbabilisticBernsteinVazirani (6), SV_MeasureReset (1), CliffordPhase (2), CliffordExpectation (2), DM_ExpectationSparse (1), DM_QubitOrdering (1), MPSSim (4), QuantumInfo_PartialTrace (1)

## [R.1.3.0] - 2026-05-09

### Fixed

- **C-1** `src/gates/two_qubit.cpp` — `apply_rzx`: sin_half sign pattern was swapped between Z=+1 and Z=-1 subspace blocks; all 8 amplitude terms corrected
- **C-2** `src/simulators/statevector_sim.cpp` — MEASURE and RESET were no-ops; implemented state collapse, projection, renormalisation, and X-flip for RESET; thread_local RNG seeded from `run()`
- **C-2** `src/simulators/mps_sim.cpp` — MEASURE and RESET were no-ops; implemented projection of site tensor and renormalisation; RESET applies X gate if outcome was |1⟩
- **C-3** `src/simulators/clifford_sim.cpp` — `rowmult` phase loop: outer condition was on src (xs,zs) computing phase of src·dest; corrected to outer condition on dest (xd,zd) per Aaronson-Gottesman Table 1
- **C-4** `src/simulators/density_matrix_sim.cpp` — `apply_gate` sub_offsets used sorted qubit order, swapping control/target for unsorted pairs (e.g. CX(2,0)); fixed to use original `qubits[k-1-qi]`; removed unnecessary sort
- **C-5** `src/transpiler/optimisation/optimize_1q.cpp` — KAK decompose emitted only the interaction term (RZZ/RYY/RXX), omitting local correction gates V₀,V₁,W₀,W₁; added Takagi factorization and `tensor_factor`/`emit_1q` helpers; full U=(V1⊗V0)·A·(W1⊗W0) now emitted
- **C-7** `src/simulators/mps_sim.cpp` — `mps_from_sv` reshape stride was `p + 2*c2` instead of `p*half_cols + c2`; swaps physical and bond indices for all non-terminal sites
- **C-8** `src/algorithms/bernstein_vazirani.cpp` — `substr(1)` stripped a character from a length-n bitstring, losing the last secret bit; removed
- **C-8** `src/algorithms/deutsch_jozsa.cpp` — `bits.size() > n` always false for length-n bitstring; changed to `bits == query_zero`; removed wrong `substr(1)` from comparison
- **C-8** `src/algorithms/simon.cpp` — `raw.substr(n)` on a length-n string returns ""; removed substr, raw is already the query register
- **C-9** `src/algorithms/qpe.cpp` — IQFT outputs eval register in bit-reversed qubit order; added swap loop after IQFT to correct ordering
- **H-1/H-2** `src/simulators/clifford_sim.cpp` — `expectation_pauli`: p_phase was bool toggle giving wrong ±i phases for Y-count; changed to int (0–3) tracking i^k; pauli_phase_update corrected for all XY/ZY/YX/YZ cases
- **H-3** `src/simulators/density_matrix_sim.cpp` — `expectation_value_sparse` only accumulated diagonal (I/Z) contributions; X and Y terms silently returned 0; replaced with full Tr(ρP) via flip_mask and complex phase accumulation
- **H-4** `src/transpiler/layout/trivial_layout.cpp` — SABRE `in_degree` and `executed` were `vector<>` indexed by position but edges use node IDs; changed to `unordered_map` keyed by node_id
- **H-5** `src/algorithms/vqe.cpp` — default initial parameters all 0.1 causes symmetric barren plateau; replaced with uniform random init in [-π, π] using `mt19937_64(42)`

### Performance

- **P-1** `src/simulators/clifford_sim.cpp`, `include/lindblad/simulators/clifford_sim.hpp` — tableau storage changed from `vector<vector<bool>>` to flat `vector<uint64_t>` (wpr=ceil(2N/64) words per row) + `vector<uint8_t>` phase array; rowmult XOR is now word-level O(N/64) instead of O(N) bit-by-bit; ~64× speedup at N>100
- **P-2/P-4** `src/gates/multi_qubit.cpp` — CCX/CCZ/CSWAP replaced full-sweep branch loop with four-level stride pattern visiting only dim/8 background addresses; `apply_unitary` replaced per-iteration vector allocation with thread_local pre-sized buffers eliminating ~65K allocations per call on 20 qubits
- **P-3** `src/simulators/density_matrix_sim.cpp` — `apply_kraus` saves original ρ once and restores in-place per Kraus operator; eliminates n_kraus-1 heap allocations per call
- **P-5** `src/simulators/mps_sim.cpp` — `svd_truncate`: replaced O(rows·cols) element-by-element input copy with zero-copy `Eigen::Map<const EigenCMatrix>`; U and Vt written via Eigen assignment
- **P-6** `src/simulators/mps_sim.cpp` — `measure_sequential`: precompute all right_envs[q] O(N·χ³) before loop, then incrementally update left_env after each projection O(χ³) per step — reduces O(N²·χ³) to O(N·χ³)
- **P-7** `src/simulators/mps_sim.cpp` — `to_statevector`: replaced per-basis-state per-site allocation loop with single left-to-right contraction maintaining one growing buffer; O(N) allocations instead of O(N·2^N)
- **P-8** `src/algorithms/maqaoa.cpp` — `active_qubits_per_term` precomputed once in `optimize()` before NLopt; eliminates per-call rebuild (~1M allocations at 10K evals × 100 terms)

### Documentation

- **D-1** `src/transpiler/basis_translator.cpp` — CRX comment corrected from wrong formula to explicit gate sequence: U3(0,0,π/2)·CX·U3(-θ/2,0,0)·CX·U3(θ/2,-π/2,0)
- **D-2** `src/simulators/mps_sim.cpp` — `measure_sequential` comment updated to describe precomputed-right-env algorithm and confirm O(N·χ³) complexity
- **D-5** `include/lindblad/simulators/density_matrix_sim.hpp` — `is_valid` annotated: PSD is not checked (requires eigendecomposition, O(4^N))

## [R.1.2.2] - 2026-05-06

### Fixed

- **R1** `StatevectorSimulator::Result` default constructor — sentinel changed from `Statevector(0)` to `Statevector(1)`; fix 7.1 (R.1.2.0) tightened the valid range to [1, 30], making the old sentinel illegal and crashing all 33 tests before any test body executed
- **R2** `DeutschJozsa::solve` — restored `bits.substr(1) == query_zero` check; fix 1.5 (R.1.2.0) incorrectly replaced it with `bits == query_zero`, but `sample_counts` returns `n+1`-length bitstrings (ancilla at MSB position 0) and `query_zero` has length `n`, so they can never be equal — all DJ calls returned BALANCED
- **R3** `BernsteinVazirani::solve` — ancilla bit stripped before reversing (`best->first.substr(1)` before `std::reverse`); fix 1.6 (R.1.2.0) added `max_element` correctly but omitted the `substr(1)` strip, causing an off-by-one extra bit in every recovered secret

### Documentation

- `docs/api/bernstein-vazirani.md` — `solve` behavior: corrected to `max_element` picks most-frequent bitstring; `substr(1)` ancilla strip before reversal documented explicitly
- `docs/algorithms/bernstein-vazirani.md` — ancilla at MSB position 0 of `n+1` bitstring; `substr(1)` extracts the `n`-bit query register before reversal
- `docs/algorithms/deutsch-jozsa.md` — `solve` compares `bits.substr(1)` (query register only) against `std::string(n, '0')`; ancilla at position 0 explained
- `docs/api/backends.md` — version example updated to `R.1.2.2`

### Results 

- 94/94 tests passed (WSL / Clang).

## [R.1.2.1] - 2026-05-06

### Tests

- Full test suite run against R.1.2.0 codebase. Identified 33 regressions across `SimulatorTest`, `IntegrationTest`, `DeutschJozsa`, `BernsteinVazirani`, `SimonsAlgorithm`, `RecursiveBernsteinVazirani`, and `ProbabilisticBernsteinVazirani` suites.
- Root causes traced to three audit fixes: fix 7.1 (`Statevector` lower-bound tightening broke `StatevectorSimulator::Result()` default constructor — sentinel `Statevector(0)` outside the try-catch); fix 1.5 (`DeutschJozsa` bitstring length mismatch n+1 vs n); fix 1.6 (`BernsteinVazirani` ancilla not stripped before secret extraction).
- Patches tracked in R.1.2.2.

### Results

- 61/94 tests passed. 33 failed — all regressions from R.1.2.0 (601 ms, WSL / Clang).

## [R.1.2.0] - 2026-05-05

### Fixed

- **1.1** `SabreLayout::run` — replaced in-place DAG node mutation with `to_circuit → remap → from_circuit` rebuild; avoids stale wire metadata in adjacency lists
- **1.2** `VQE` — removed dead `nlopt_set_min_objective(nullptr)` call that preceded the correct call with `cb_data`
- **1.3** `StabilizerState::measure` deterministic branch — replaced bare XOR accumulation with `rowmult()` for correct Pauli phase tracking
- **1.4** `QuantumCircuit::control` — replaced `goto generic_control` (UB over variable initialisations) with `emit_generic_ctrl` lambda
- **1.5** `DeutschJozsa::solve` — fixed off-by-one bitstring check (`bits.substr(1) == query_zero` → `bits == query_zero`)
- **1.6** `BernsteinVazirani::solve` — replaced `counts.begin()` (arbitrary unordered_map entry) with `max_element` by count
- **2.2** `orbits_by_power` (MAQAOA) — sort generators by power before orbit assignment; added `#include <numeric>` for `iota`
- **2.4** `MPSState::measure_sequential` — renormalise by `probs[outcome]` from boundary contraction, not local Frobenius norm (invalid for non-canonical MPS)
- **2.5** `SabreSwap` — build local `id_to_idx` map; resize `adj_out`/`in_deg`/`done` by `id_range` not `num_nodes`; all `dag.nodes[nid]` accesses now use correct index regardless of node ID sparsity after deletions
- **2.6** `QPE::build_circuit` — fixed `CU_matrix` convention: control qubit is LSB (interleaved even=ctrl-0 / odd=ctrl-1 rows) to match `apply_unitary`'s targets[0]=LSB contract
- **3.1** `QuantumCircuit::compose` (mapped branch) — expand `result.n_clbits` to accommodate `other.n_clbits` before copying instructions
- **3.2** `QuantumCircuit::inverse` — `UNITARY` gates now emit conjugate-transpose; `PARAM_*` symbolic gates skipped (not thrown)
- **3.3** `Sampler::run` batch — each circuit now receives a distinct seed (`base + i`) instead of sharing the same seed
- **3.4** `Estimator` cache key — `circuit_structure_key` now includes `n_clbits` to avoid cross-circuit cache collisions
- **4.1** `SparsePauliOp::to_matrix` — rewritten from O(n·4^n) tensor-product loop to O(2^n) per term using XOR-mask phase computation
- **4.3** `Operator::power` — binary exponentiation O(log n) replaces naive O(n) repeated multiplication; added `n < 0` guard
- **4.4** `SabreSwap` inner loop — adjacency list replaces O(E) edge scan for successor lookup
- **4.5** `CouplingMap::heavy_hex` dedup — `unordered_set<uint64_t>` hash lookup replaces O(n) `std::find`
- **4.8** `DAGCircuit::substitute_node` / `remove_node` — `adj_out`/`adj_in` changed from `vector<int>` to `vector<DAGEdge>`; edge collection and neighbour cleanup now O(degree) not O(E)
- **5.1** `Estimator::run_single` — double-checked locking: transpilation now executes outside the mutex; only cache insertion/lookup is locked
- **6.1** `QAOA::Options::optimizer` — optimizer selection now honours the `optimizer` field (COBYLA / NELDER_MEAD / POWELL)
- **6.2** `Estimator::Options::shots` — documented that `shots` has no effect; Estimator always uses exact statevector evaluation
- **6.3** `LocalBackend::version()` — updated from stale `"1.9.6-alpha"` to current version string
- **6.4** `Grover::build_circuit` — added doc comment noting the default iteration formula assumes a single marked item
- **6.5** `IsingHamiltonian` — explicit qubit↔bitstring-position mapping documented on `to_sparse_pauli_op` and `evaluate`
- **7.1** `Statevector` constructor — lower bound changed from 0 to 1; valid range is now [1, 30]
- **7.3** `MPSSimulator` — replaced magic numbers 18 and 25 with named constants `MPS_SV_CROSSOVER` and `MPS_SV_MAX_QUBITS`
- **7.4** `NoiseModel` — documented that `before_gate` noise is unimplemented in `DensityMatrixSimulator`
- **7.5** `CommutativeCancellation` — added fixed-point outer loop; pass now repeats until no changes
- **7.6** `QuantumCircuit::to_qasm2` — `UNITARY` and `PARAM_*` gates emitted as comments instead of invalid gate names
- **7.7** `Grover::search` — `num_iterations` resolved once before `build_circuit` to guarantee consistency in result

### Changed

- **5.2** `StabilizerState::measure` — signature changed from `uint64_t seed` to `std::mt19937_64& rng`; eliminates per-measurement RNG construction; `CliffordSimulator::run` passes its top-level `rng` by reference

### Documentation

- `docs/api/transpiler.md` — CommutativeCancellation: fixed-point loop and commutation rules documented; dependency query complexity updated for `adj_out`/`adj_in` (`DAGEdge`) adjacency
- `docs/api/backends.md` — `version()` now documented as returning the `LINDBLAD_VERSION_LABEL` CMake macro; cannot drift from project version
- `docs/api/grover.md` — `search()` resolves `num_iterations` before `build_circuit`; default formula annotated as single-marked-item only
- `docs/api/qpe.md` — CU matrix LSB convention documented (even rows = ctrl-0 identity, odd rows = ctrl-1 U^k)
- `docs/api/operators.md` — `to_matrix()` documented as O(2^n) per term via Pauli XOR-mask action
- `docs/api/noise.md` — `before_gate` application documented as unimplemented in `DensityMatrixSimulator`
- `docs/api/ising.md` — `evaluate()` MSB-first indexing (`bitstring[n-1-i]` = qubit `i`) documented explicitly
- `docs/api/statevector.md` — valid `n_qubits` range corrected from [0, 30] to [1, 30]
- `docs/algorithms/grover.md` — same corrections as `docs/api/grover.md`
- `docs/algorithms/deutsch-jozsa.md` — `solve()` compares against `std::string(n, '0')` (query register width only)
- `docs/algorithms/bernstein-vazirani.md` — `max_element` comparator and MSB→index reversal documented
- `docs/algorithms/maqaoa.md` — `orbits_by_power` sort-first grouping behaviour documented
- `docs/algorithms/qpe.md` — CU matrix LSB convention documented (same as `docs/api/qpe.md`)
- `docs/algorithms/vqe.md` — `optimizer` field documented as wired: COBYLA / NELDER_MEAD / POWELL

## [R.1.1.1] - 2026-05-04

### Tests

- Full existing test suite run against renamed Lindblad codebase to verify no regressions from the Q++/qpp → Lindblad rename (namespace, include paths, CMake targets).

### Results

- 94 tests across 16 suites — all passed.

## [R.1.1.0] - 2026-05-04

### Changed

- **Project renamed: Q++/qpp → Lindblad** — all source files, headers, CMake targets, macros, docs, license, and citation updated project-wide. C++ namespace `qpp::` → `lindblad::`. Include path `include/qpp/` → `include/lindblad/`. CMake project name and all `QPP_` macros renamed to `LINDBLAD_`. CHANGELOG.md intentionally preserved as historical record. Contact email `qpp.support@proton.me` unchanged.
- **LICENSE** — title and all references updated to "Lindblad Software License Agreement" and "Lindblad Quantum Computing Framework".
- **CITATION.cff** — title updated to "Lindblad"; `LicenseRef-Q++-1.0` → `LicenseRef-Lindblad-1.0`.
- **Banner** — shows on both startup and exit; exit banner now includes full license notice text.

## [R.1.0.1] - 2026-05-03

### Tests

- `tests/test_classic_algorithms.cpp` — 3 additional BV edge case tests:
  - `BernsteinVazirani/RecoversSecret_1Qubit` — n=1 minimal single-qubit circuit
  - `RecursiveBernsteinVazirani/Depth2WithShots2` — shots parameter propagates correctly through both levels
  - `ProbabilisticBernsteinVazirani/DuplicateSecretPool` — pool with two oracles encoding the same secret collapses to exactly 1 discovered key

### Results

- 94 tests across 16 suites — all passed (600 ms, WSL / Clang).

## [R.1.0.0] - 2026-05-03

### Added

- **Classic algorithm suite:** Five algorithm families now fully implemented and independently tested:
  - `DeutschJozsa` — constant/balanced oracle classification in one query (`src/algorithms/deutsch_jozsa.cpp`)
  - `BernsteinVazirani` — hidden string recovery from `f(x)=s·x mod 2` in one query (`src/algorithms/bernstein_vazirani.cpp`)
  - `RecursiveBernsteinVazirani` — depth-d BV solving d levels with d oracle calls vs classical d×n queries; proves BQP/BPP superpolynomial separation structure
  - `ProbabilisticBernsteinVazirani` — multi-key probabilistic oracle (Shukla & Vedula 2023, arXiv:2301.10014); recovers any key in 1 shot with certainty, all keys in ~K·ln(K) shots
  - `Simon` — period finding via GF(2) Gaussian elimination (`src/algorithms/simon.cpp`)
- **QPE and Grover extracted to own files:** `src/algorithms/qpe.cpp`, `src/algorithms/grover.cpp`
  — previously inlined in `maqaoa.cpp`; `maqaoa.cpp` now contains MAQAOA only.
- **CLI shutdown banner:** `src/banner.cpp` — `BannerInitializer` destructor now calls
  `print_qpp_exit_banner()` on shutdown, symmetrical with the startup banner.
- **Comprehensive documentation suite** — full API and algorithm family pages:
  - `docs/MasterDocumentation.md` — reusable blueprint for algorithm pages, API pages, update workflow, and context recovery
  - Algorithm family pages: `docs/algorithms/bernstein-vazirani.md` (full BV family), `deutsch-jozsa.md`, `simon.md`, `qpe.md`, `grover.md`, `vqe.md`, `qaoa.md`, `maqaoa.md`, `ising.md`, `dispatch.md`
  - API deep-dive pages: `docs/api/bernstein-vazirani.md`, `deutsch-jozsa.md`, `simon.md`, `qpe.md`, `grover.md`, `vqe.md`, `qaoa.md`, `maqaoa.md`, `ising.md`, `dispatch.md`, `circuit.md`, `operators.md`, `noise.md`, `estimator.md`, `sampler.md`, `gates.md`, `simulators.md`, `transpiler.md`, `backends.md`
  - `docs/APIOverview.md` — updated with all algorithm classes (`DeutschJozsa`, `Simon`, full BV family), all deep-dive links, Backends and Gates sections
  - `README.md` — added full documentation map with links to all algorithm and API pages
- **`CITATION.cff`** — standard CFF software citation file with ORCID, preferred-citation block, and version metadata for academic attribution.

### Changed

- **License:** replaced Apache License 2.0 with the **Q++ Software License Agreement v1.0** — proprietary source-available license; free for non-commercial and academic use, commercial use requires a separate written agreement, redistribution in any form is prohibited. See `LICENSE` for full terms and `qpp.support@proton.me` for licensing inquiries.
- **`README.md`** — prominent license notice added at top (distribution policy, GitHub fork policy, §6.3 copyright assignment notice); release line updated to lead with the BV family as the headline feature; license section updated to reference the new license and `CITATION.cff`.
- **`NOTICE`** — updated to reflect the new license; Apache-format attribution text removed.

### Fixed

- **`src/simulators/density_matrix_sim.cpp` — two correctness bugs in `apply_gate`:**
  1. Right-multiply transpose: `U†[c_in,c_out]` index was `conj(U[c_in*d+c_out])` but must be `conj(U[c_out*d+c_in])`.
  2. Sub-index qubit convention mismatch: `sub_offsets` mapped sub-bit `qi` → `sorted_tgts[qi]` (LSB-first) but all gate matrices use MSB-first ordering; fixed to `sorted_tgts[k-1-qi]`. Fixes `RCCXStatevectorDensityMatrixConsistency`, `StatevectorMatchesDensityMatrix`, `DensityMatrixReset`, `DensityMatrixResetSuperposition`.
- **`src/transpiler/routing/sabre_swap.cpp` — front layer never populated:**
  `in_deg` was counting `IN→OP` edges, making all first OP nodes appear as `in_deg > 0`; the routing loop never executed and produced a 0-instruction output. Fixed to count only `OP→OP` edges.
- **`src/transpiler/optimisation/optimize_1q.cpp` — `ConsolidateBlocks` corrupted single 2Q gates:**
  `kak_decompose` emits only the interaction term without local unitary corrections, so a single CNOT became `RXX(π/2)`. Fixed by tracking `block_count`; KAK is only applied when `block_count ≥ 2`.
- **`src/qasm/qasm2_parser.cpp` — register names no longer hard-coded to `q`/`c`:**
  First pass now builds `qreg_offsets`/`creg_offsets` maps; measure, reset, and gate resolution use `resolve_reg_index` and `parse_qubits_mapped`. Multi-register QASM circuits now parse correctly.
- **`src/simulators/mps_sim.cpp` — `GT::UNITARY` 3-qubit gates no longer throw:**
  Added `mps_from_sv` (sequential SVD via Eigen BDCSVD); `GT::UNITARY` 3-qubit case falls back through statevector → `apply_unitary` → `mps_from_sv`.

### Tests

- `tests/test_classic_algorithms.cpp` — new suite: DeutschJozsa (4), BernsteinVazirani (4), Simon (3), RecursiveBernsteinVazirani (4), ProbabilisticBernsteinVazirani (6) — 21 tests total.
- `tests/test_integration.cpp` — `QASM2NonDefaultRegisterNames`, `QASM2MultipleQregs`, `ConsolidateBlocksNoGateDuplication`.
- `tests/test_simulators.cpp` — `RCCXStatevectorDensityMatrixConsistency`.
- `tests/test_operators.cpp` — `PauliStringCommutation` expectation corrected.
- `tests/test_maqaoa_microgrid.cpp` — `MAQAOA_Layerwise` fixed with `term_indexed_gammas=true`.

### Chore

- Version bumped to `R.1.0.0` (CMake `1.1.0.0`); switches from semver labels to A.B.C.D release scheme.

## [2.3.2-beta] - 2026-04-26

### Added

- **CLI startup ASCII art banner:** A process-level banner is now printed once when
  Q++ is run in an interactive terminal (Windows command line, PowerShell, WSL TTY).
  The banner is emitted by the core library initialization path so CLI executables that
  link `qpp_core` show the branding automatically.

### Changed

- **`src/algorithms/maqaoa.cpp` - MAQAOA initial state now supports seeded per-qubit angles:**
  `MAQAOA::build_circuit()` now initializes qubits with `RY(options.initial_thetas[q])`
  when `initial_thetas` is provided with length equal to the qubit count. If no valid
  seed vector is provided, behavior is unchanged and the circuit falls back to the
  default `|+...+>` initialization via Hadamards.
- **Banner linkage for CLI targets:** `src/banner.cpp` is now linked via a dedicated
  object target into command-line executables (tests and benchmarks), ensuring the
  startup ASCII banner is emitted at runtime in interactive terminals.
- **Banner messaging refresh:** The CLI banner now includes a welcome heading with
  version label, top/bottom separators, and an explicit readiness line around the
  ASCII art branding.
- CMake project version bumped to `2.3.2-beta`.

## [2.3.1-beta] - 2026-04-14

### Fixed

- **`src/primitives/sampler.cpp` — `Sampler::run_single` ignores `noise_model`**: The
  `noise_model` field in `Sampler::Options` was never wired into the execution path.
  `run_single` hardcoded `StatevectorSimulator` unconditionally, causing noisy sampling
  experiments to produce byte-for-byte identical results to ideal simulation regardless
  of the noise model set. Fixed by adding an `is_ideal()` branch: when
  `options.noise_model` is non-ideal, execution routes through `DensityMatrixSimulator::run()`
  with the bound circuit, noise model, shots, and seed; the statevector fast-path is
  unchanged for the ideal case. The `run()` batch method is unaffected (it delegates to
  `run_single`). Adds `#include "qpp/simulators/density_matrix_sim.hpp"` to the
  translation unit.

## [2.3.0-beta] - 2026-04-14

### Added

- **Noisy MAQAOA execution path via `DensityMatrixSimulator`:** When
  `estimator.options.noise_model` is non-ideal, both the non-layerwise objective
  (`maqaoa_objective`) and the layerwise objective (`layer_objective`) now route each
  COBYLA evaluation through `build_circuit()` + `DensityMatrixSimulator::run()` instead
  of the direct `evolve_into()` statevector fast-path. The noiseless fast-path is
  completely unchanged; the branch is a zero-cost `is_ideal()` check on every evaluation.
  This enables `Mode A` (noisy optimisation) in exp_p: setting
  `maqaoa.estimator.options.noise_model` is sufficient to activate density-matrix
  simulation for the full COBYLA loop.

- **Noisy final evaluation and sampling:** After optimisation, if
  `estimator.options.noise_model` is non-ideal the final energy evaluation also routes
  through `DensityMatrixSimulator`. If `sampler.options.noise_model` is non-ideal, the
  final bitstring sampling uses `DensityMatrixSimulator::run()` with the sampler's shot
  count and seed instead of `Statevector::sample_counts()`. Both checks are independent,
  allowing noiseless optimisation + noisy sampling (`Mode B`) or fully noisy
  optimisation + noisy sampling (`Mode A`) without code changes in the caller.

- **`DensityMatrix::expectation_value_sparse(const SparsePauliOp&)`:** Computes
  `Tr(ρH)` for diagonal Ising Hamiltonians (Z and ZZ terms only) in O(2^N × n_terms)
  time without materialising the full 2^N × 2^N matrix representation of H. Iterates
  over basis states, computes the eigenvalue of each diagonal Pauli term via bit
  extraction, weights by `ρ[b,b].real`, and sums. Skips basis states with zero
  diagonal probability. Declared in `include/qpp/simulators/density_matrix_sim.hpp`,
  implemented in `src/simulators/density_matrix_sim.cpp`.

- **`tests/test_maqaoa_noisy.cpp`:** Integration test covering layerwise QSP-MA-QAOA
  with depolarising noise on both estimator and sampler. Verifies run completes without
  error, `result.best_bitstring` has the correct length, `result.optimal_value` is
  finite, and `result.counts` is non-empty.

### Changed

- `MAQAOACallbackData` gains a `const MAQAOA* maqaoa` pointer (re-added after the
  v1.8.0 refactor removed it) so `maqaoa_objective` can call `build_circuit()` on the
  noisy path. The noiseless path continues to use `evolve_into()` via the pre-existing
  `Statevector*` member; no hot-path performance regression.
- CMake project version bumped to `2.3.0-beta`.

## [2.2.0-beta] - 2026-04-12

### Added

- **QAOA optimiser bounds and step-size configuration:** `QAOA::optimize()` now enforces
  parameter bounds `[-2π, 2π]` for gamma and beta parameters, preventing unbounded
  growth during optimisation. Initial step size is set to 0.3 to accelerate convergence.
  These settings improve the robustness and speed of the NLopt COBYLA optimiser.
- **Initial parameter tracking:** `QAOA::Result` now includes `initial_params` field
  storing the seeded initial gamma/beta parameters before optimisation, enabling analysis
  of parameter trajectories and optimiser starting configurations.
- **Computational-basis cost evaluation:** New `computational_basis_cost()` helper
  function evaluates the cost Hamiltonian eigenvalue on computational basis states
  in O(n_terms) time. Used for improved bitstring selection.

### Changed

- **QAOA bitstring selection strategy:** Final bitstring is now chosen by ranking sampled
  computational-basis eigenenergies (via cost Hamiltonian), with ties broken by sample count.
  Previously selected by sample count alone, which could favour suboptimal bitstrings
  sampled more frequently by chance. This change aligns with VQE/QAOA best practices.
- **Convergence criterion refinement:** `QAOA::Result::converged` now explicitly excludes
  `NLOPT_MAXEVAL_REACHED` status, distinguishing iterations exhausted from true convergence.
- CMake project version bumped to `2.2.0-beta`.

## [2.1.2-beta] - 2026-04-10

### Changed

- **QAOA seeded parameter initialisation:** `QAOA::optimize()` now draws initial
  gamma/beta parameters from `U(-0.05, 0.05)` seeded by `QAOA::Options::seed`,
  matching the MAQAOA initialisation convention introduced in v2.0.0-beta.
  Previously all parameters were hardcoded to `0.5` regardless of seed, making
  multi-seed runs produce identical starting points. `sampler.options.seed` is
  also propagated from `options.seed` before the final bitstring sampling call.
- CMake project version bumped to `2.1.2`.

## [2.1.1-beta] - 2026-04-09

### Added

- **QSP-QAOA: per-qubit Ry state preparation (`QAOA::Options::initial_thetas`):**
  When `initial_thetas` is non-empty and sized to `n_qubits`, `build_circuit` replaces
  the standard `H|0⟩` initialisation with `Ry(theta[q])|0⟩` for each qubit. This encodes
  a domain prior `P(qubit q = 1) = sin²(theta[q]/2)` directly into the quantum initial
  state before any QAOA layers are applied. Compute `theta[q] = 2*arcsin(sqrt(p_on[q]))`
  from any prior probability `p_on[q] ∈ [0, 1]`. Empty vector (default) preserves the
  existing `H` initialisation exactly — no behaviour change for existing callers.
  Mirrors the `MAQAOA::Options::initial_thetas` field added in v2.1.0-beta, giving standard
  QAOA (2p parameters) full parity with MA-QAOA for QSP state preparation experiments.
- CMake project version bumped to `2.1.1`.

## [2.1.0-beta] - 2026-04-06

### Added

- **QSP-MA-QAOA: per-qubit Ry state preparation (`MAQAOA::Options::initial_thetas`):**
  When `initial_thetas` is non-empty and sized to `n_qubits`, replaces the standard
  `H|0⟩` initialisation with `Ry(theta[q])|0⟩` for each qubit. This encodes a domain
  prior `P(qubit q = 1) = sin²(theta[q]/2)` directly into the quantum initial state
  before any QAOA layers are applied. Compute `theta[q] = 2*arcsin(sqrt(p_on[q]))` from
  any prior probability `p_on[q] ∈ [0, 1]`. Empty vector (default) preserves existing
  `H` initialisation exactly — no behaviour change for existing callers.
- `evolve_into()` gains an `initial_thetas` parameter (defaulting to `{}`) used at every
  statevector reset: layerwise callback, joint callback, and both final-eval sites.
- `LayerCBData` and `MAQAOACallbackData` carry `initial_thetas` from
  `options.initial_thetas` to each `evolve_into` call.
- `experiments/common.hpp` update deferred — file is not present in the repository.
- CMake project version bumped to `2.1.0`.

## [2.0.2-beta] - 2026-04-06

### Fixed (documentation only)

- Corrected stale comment in `include/qpp/algorithms.hpp` `mixer_weights` block which
  still described the pre-v1.9.5 formula (`w_max / w_i`). Comment now accurately reflects
  the current `beta_base * w_i / w_max` implementation, where expensive generators (large
  `w_i`) receive large initial betas and cheap generators receive small ones.
- Documented the IPI (inverse-PI) pattern: callers who want the opposite behaviour
  (large beta for cheap generators) can pass `ipi_weights[i] = 1.0 / pi_weights[i]` with
  no q++ changes required.
- CMake project version bumped to `2.0.2`.

## [2.0.1-beta] - 2026-04-05

### Added

- **Progressive training mode for MA-QAOA (`MAQAOA::Options::progressive`):** When
  `progressive = true` (requires `layerwise = true`), the layerwise schedule never freezes
  previously trained parameters. At each layer step, `free_start` stays at `0` and
  `n_free` grows by one layer's worth of parameters, so COBYLA optimises all parameters
  from layer 0 through the current layer jointly. Layers already trained provide good warm-start
  values; only the new layer's parameters are freshly initialised. When `progressive = false`
  (default) behaviour is identical to the existing layerwise path — no change.

- CMake project version bumped to `2.0.1`.

## [2.0.0-beta] - 2026-04-04

### Changed

- **Seeded random perturbation replaces fixed alternating init in MA-QAOA and PI-MA-QAOA:** All
  initial parameters — gammas (both paths) and betas (both paths) — are now drawn from
  `U(-0.05, 0.05)` using an `std::mt19937_64` seeded by `MAQAOA::Options::seed`. When `seed == 0`
  a `std::random_device` non-deterministic seed is used. This replaces the deterministic
  `±0.1` alternating pattern that caused all runs to start from the same point regardless of seed,
  making the seed field meaningful for reproducible multi-start landscape exploration.

- **PI-MA-QAOA betas preserve physics-informed bias:** The `beta_base * (mixer_weights[i] / w_max)`
  scaling is retained as the centre of each beta's initial value; the seed perturbation is added on
  top rather than replacing it. This keeps expensive generators biased toward large angles while
  still allowing each seed to explore a different neighbourhood.

- CMake project version bumped to `2.0.0` and user-facing release label updated to `2.0.0-beta`.

## [1.9.6-alpha] - 2026-04-02

### Fixed

- **Layerwise MA-QAOA evaluated full depth at every layer:** Layerwise COBYLA now evaluates only
  `layer + 1` active layers during each optimisation stage instead of always evaluating `p` layers,
  restoring parity with the intended layerwise algorithm.

- **OpenMP threshold off-by-one at N=20:** Parallel kernels now activate at `dim >= (1<<20)`
  (not only strictly above), enabling multithreaded execution for 20-qubit workloads in gate,
  statevector, and expectation-value hot paths.

- **Convergence reporting incorrectly marked maxeval as converged:**
  `NLOPT_MAXEVAL_REACHED` is now treated as non-converged in MA-QAOA result reporting.

- **`best_bitstring` selected by frequency instead of objective cost:** Sampled candidates are now
  ranked by computational-basis cost under the cost Hamiltonian (with count as tie-breaker),
  improving reported solution quality.

### Performance

- **Reduced COBYLA hot-path allocations in MA-QAOA:** Reused callback parameter buffers in the
  standard objective path and reused per-layer gamma/beta work vectors inside `evolve_into()`.

- **PI-MA-QAOA initialisation cleanup:** Precomputed `w_max` once per optimisation run rather than
  recomputing it in each layer loop.

### Changed

- CMake project version bumped to `1.9.6` and user-facing release label updated to `1.9.6-alpha`.

## [1.9.5-alpha] - 2026-04-01

### Fixed

- **PI-MA-QAOA beta scaling inverted (B0-1):** `beta_base * (w_max / mixer_weights[i])` gave
  small angles to expensive generators and large angles to cheap ones — the opposite of the
  intended physics-informed initialisation. Corrected to `beta_base * (mixer_weights[i] / w_max)`
  so that high-cost (expensive) generators receive a large initial beta and cheap generators
  receive a small initial beta. Applies to both the layerwise path and the standard path in
  `src/algorithms/maqaoa.cpp`. README description updated to match.

## [1.9.4-alpha] - 2026-03-31

### Fixed

- **MSVC compatibility: `__restrict__` on class members (S1):** GCC/Clang accept `__restrict__` on
  `Statevector` members, but MSVC only allows `__restrict` (one underscore) on local pointers and function
  parameters, not class members. Removed `__restrict__` from `real_parts` and `imag_parts` declarations
  in `include/qpp/statevector.hpp`. The restrict aliasing information the compiler actually uses comes
  from the local `double* __restrict__` declarations inside each gate function, which remain unchanged.
  **Performance impact: zero** — these declarations were not providing useful information to the compiler.

- **MSVC compatibility: `__builtin_popcountll` intrinsic (S3):** GCC/Clang provide `__builtin_popcountll`
  for population count; MSVC provides `__popcnt64` from `<intrin.h>`. Added `QPP_POPCOUNT64(x)` macro
  in `include/qpp/types.hpp` that maps to the correct intrinsic for each compiler. Applied in
  `src/quantum_info/operators.cpp` in `SparsePauliOp::expectation_value()` and `expectation_value_batch()`.
  **Performance impact: zero** — both compile to the same `POPCNT` hardware instruction on x86.

- **MSVC compatibility: OpenMP 4.0+ pragma clauses (S2):** MSVC uses OpenMP 2.0 (legacy) and silently
  ignores unrecognised `#pragma omp` directives, including the `aligned()` clause on `#pragma omp simd`.
  This caused MSVC to lose SIMD vectorisation hints while GCC/Clang benefited from them. Added
  `QPP_SIMD_LOOP` macro in `include/qpp/types.hpp` that expands to `_Pragma("omp simd aligned(...)")` on
  GCC/Clang and is empty on MSVC. Applied in `src/gates/single_qubit.cpp` (9 instances) and
  `src/gates/two_qubit.cpp` (16 instances). MSVC auto-vectorises these loops with `/O2 /arch:AVX2`
  anyway, so the net effect is no performance change on MSVC and no change on GCC/Clang.

## [1.9.3-alpha] - 2026-03-31

### Fixed

- **Missing `#include <functional>` in `include/qpp/dispatch.hpp`:** `std::function` is used
  in `SoftDispatchResult` but `<functional>` was not included. Relies on transitive inclusion
  on GCC/Clang; fails on MSVC and libc++ in standalone builds. Added before `<string>`.

## [1.9.2-alpha] - 2026-03-31

### Build

- **Wire `ising.cpp` and `dispatch.cpp` into `qpp_core` (B0-1):** Both files existed under
  `src/algorithms/` but were absent from the `add_library(qpp_core STATIC ...)` block.
  `IsingHamiltonian` and `SoftDispatchResult` were compiled into no translation unit, causing
  link errors for any target that used them. Added after `src/algorithms/maqaoa.cpp`.

## [1.9.1-alpha] - 2026-03-31

### Fixed

- **Delete non-default constructors in `VQE`, `QAOA`, `MAQAOA` (I1):** `VQE(const Options&, const Estimator&)`,
  `QAOA(const Options&, const Estimator&, const Sampler&)`, and `MAQAOA(const Options&, const Estimator&, const Sampler&)`
  attempted to copy-construct `Estimator` which holds a `std::mutex` (non-copyable). Constructors deleted;
  configure via member access (`algo.estimator.options.shots = 0`).

- **`M_PI` portability guard in `include/qpp/algorithms.hpp` (I2):** MSVC does not define `M_PI` without
  `_USE_MATH_DEFINES`. Added `#ifndef M_PI` / `#define M_PI 3.14159265358979323846` guard before `<cmath>`.

- **`preset_pass_manager` / `pm.run()` replaced with `qpp::transpile()` in `src/primitives/estimator.cpp` (I3):**
  The two-argument `preset_pass_manager` call was missing the required `basis_gates` argument, and `pm.run()`
  does not accept a `QuantumCircuit` directly. Replaced with `qpp::transpile(circuit, CouplingMap(n), {}, level)`.

- **`M_PI` inline literal in `src/primitives/estimator.cpp` gradient (I4):** `constexpr double shift = M_PI / 2.0`
  fails on MSVC. Replaced with `3.14159265358979323846 / 2.0`.

- **Copy-assignment of `Estimator` in test files replaced with direct member access (I5):**
  `maqaoa.estimator = est` fails because `Estimator::operator=` is deleted (mutex member).
  Fixed in `test_maqaoa_5qubit.cpp`, `test_maqaoa_microgrid.cpp` (×2), and `test_maqaoa_20qubit.cpp`.

- **Out-of-scope `params_per_layer` / `total_params` in `AllMethodsComparison` (I6):**
  Variables defined in `MAQAOA_Layerwise` scope were referenced in a separate test function.
  Replaced with compile-time expressions `2 * N` (= 40) and `2 * N * 6` (= 240).

## [1.9.0-alpha] - 2026-03-28

### Fixed

- **MA-QAOA standard path now qubit-indexed by default (F0-1):** `num_parameters()`,
  `optimize()`, `evolve_into()`, and `build_circuit()` all previously used term-indexed
  gammas (one gamma per Hamiltonian term) in the non-orbit branch, giving 230 params/layer
  at N=20 vs the Python baseline's 40. The standard path now uses N gammas per layer:
  `gamma[i]` drives all cost terms where qubit `i` is the lowest active qubit (Z on qubit
  `i` → `gamma[i]`; ZZ on qubits `i < j` → `gamma[i]`). This matches the Python/Qiskit
  MA-QAOA implementation exactly and makes standard and PI-MA-QAOA runs directly comparable
  to the Python baseline in the paper.

- **`evolve_into()` and `build_circuit()` gamma dispatch restructured (F0-2):** Both
  functions previously computed `gamma_idx` before building the active-qubit list `aq`,
  making qubit-indexed dispatch impossible. Active qubits are now computed first, then
  `gamma_idx` is resolved via a three-way branch: orbit-indexed → `term_orbit_map[t]`,
  term-indexed → `t`, qubit-indexed (default) → `aq[0]`.

### Added

- **`MAQAOA::Options::term_indexed_gammas` flag (A0-1):** `bool term_indexed_gammas = false`.
  When `true`, restores the previous one-gamma-per-term behaviour (230 params/layer at N=20)
  for ablation studies. When `false` (default), uses the qubit-indexed convention (40
  params/layer at N=20). The orbit path is unaffected by this flag.

### Changed

- `tests/test_maqaoa.cpp`: expected param counts updated — `CircuitBuild` (3→4) and
  `MoreParams_MoreLayers` (10→8) reflect the qubit-indexed default. New
  `TermIndexedGammas` test verifies the opt-in flag still produces the old counts.
- `tests/test_maqaoa_20qubit.cpp`: `params_per_layer` now derived from
  `maqaoa.num_parameters(ham) / P_MAX` rather than hardcoded from `ham.terms.size()`.
  Printout updated from "term-indexed" to "qubit-indexed"; comparison table corrected.

## [1.8.0-alpha] - 2026-03-27

### Performance

- **MA-QAOA direct statevector evolution (P0-1):** `MAQAOA::optimize()` now uses `evolve_into()` —
  a file-scope function that applies MAQAOA layers directly onto a pre-allocated `Statevector` via
  `gates::apply_h/rz/cx/rx/s/sdg`. Bypasses `QuantumCircuit` construction, `assign_parameters()`,
  `circuit_structure_key()`, transpile-cache mutex acquisition, and `StatevectorSimulator`
  instruction dispatch on every COBYLA evaluation. Expected 3–5× wall-time reduction on N=20
  layerwise runs (dominant cost shifts from circuit overhead to statevector math).

- **Orbit-map precomputation (P0-2):** `cost_term_orbit_map()` and `count_cost_orbits()` are now
  called **once** at `optimize()` entry and threaded through all callbacks as `const` references.
  Previously recomputed on every `build_circuit` call (every COBYLA evaluation) via `std::map`
  over all Hamiltonian terms.

- **Frozen-vector copy eliminated in layerwise callback (P0-3):** `LayerCBData` now holds a
  pre-allocated `all_params` vector of full run size `[frozen | free]`. The callback does
  `std::copy(x, x + n, all_params.begin() + free_start)` in-place instead of constructing a
  new vector from `d->frozen` and appending `x` on every evaluation. At layer 5 with N=20 this
  removes ~575,000 allocations (500 evals × 1150 frozen doubles) across the full layerwise run.

- **Statevector reused for sampling (P0-4):** The final statevector from `evolve_into()` is used
  directly for `sample_counts()`. Eliminates a redundant `build_circuit` + `Sampler::run_single`
  call (circuit construction + full statevector simulation) that previously ran once after
  optimization to produce the bitstring distribution.

- **`Estimator::optimization_level` defaults to 0 (P0-5):** Was 1, which ran SABRE layout +
  ZYZ decomposition on the first circuit structure seen. For statevector simulation there are no
  coupling constraints — SABRE produces the identity mapping at non-trivial compute cost. Default
  is now 0 (no transpilation); set `optimization_level = 1` explicitly for hardware targeting.

### Added

- **PI-MA-QAOA initialisation (A1-1):** `MAQAOA::Options` gains `mixer_weights` (per-orbit weight
  vector), `beta_base` (default `π/4`), and `lambda_co2` (carbon weighting factor). When
  `mixer_weights` is non-empty and sized to match `n_mixer_orbits`, betas are initialised as
  `beta_base × (w_max / w_i)` — generators with high augmented cost per MW get a small initial
  angle, cheap generators get a large angle. Falls back to the original alternating `±0.1`
  initialisation when `mixer_weights` is empty.

- **`orbits_by_power(powers, tolerance)` (A1-2):** Free function in `qpp::algorithms`. Assigns
  orbit indices by power tier: generators within `tolerance` MW of each other share an orbit.
  Returns a `vector<int>` of size N suitable for direct assignment to
  `MAQAOA::Options::orbit_assignments`. Completes the Orbit-QAOA setup path without requiring
  manual orbit construction.

- **Extended `MAQAOA::Result` (A1-3):** New fields:
  - `initial_params` — per-layer concatenated initial guess (saved before each COBYLA call)
  - `per_layer_costs` — best energy at end of each layer (convergence curve)
  - `layer_nfev` — function evaluations per layer (budget attribution)
  - `wall_time_by_layer` — wall seconds per layer (equal-time comparison)
  - `wall_time_seconds` — total wall time (both paths)

- **`qpp::core` CMake alias (A1-4):** `add_library(qpp::core ALIAS qpp_core)` added after the
  `target_include_directories` block. External repositories using FetchContent can now use the
  conventional namespaced form: `target_link_libraries(my_target PRIVATE qpp::core)`.

### Changed

- `LayerCBData` and `layer_objective` promoted from definitions inside `optimize()`'s for-loop to
  file-scope declarations. No behaviour change; required for the pre-allocated `all_params` member
  and to expose the callback as a plain function pointer to NLopt without a static lambda.
- `MAQAOACallbackData` now holds `term_orbit_map`, `n_cost_params_per_layer`, `n_mixer_orbits`,
  and a `Statevector*` instead of `Estimator*` and `const MAQAOA*`-for-build-circuit. The
  non-layerwise objective also calls `evolve_into` directly.
- `build_circuit` is retained unchanged for API compatibility and offline circuit inspection but
  is no longer invoked anywhere in the `optimize()` hot path.

## [1.7.0-alpha] - 2026-03-26

### Added

- **`IsingHamiltonian` with QUBO conversion (P1-1):** New `include/qpp/ising.hpp` and
  `src/algorithms/ising.cpp`. Exposes `from_qubo(Q, penalty_A)` (QUBO matrix → Ising h/J/offset
  via x_i = (1−s_i)/2 substitution), `to_sparse_pauli_op()`, `evaluate(bitstring)`,
  `evaluate_spins(spins)`, and `from_hJ()`. Eliminates the Python round-trip for problem setup
  in the MA-QAOA research workflow.

- **`SoftDispatchResult` post-processing (P1-2):** New `include/qpp/dispatch.hpp` and
  `src/algorithms/dispatch.cpp`. Extracts fractional dispatch solutions from MA-QAOA bitstring
  distributions: `threshold_round`, `greedy_dispatch` (demand-constrained generator selection),
  `expected_cost` (probability-weighted energy under any cost function), and `top_k`.

- **Orbit-QAOA symmetry reduction (P1-3):** `MAQAOA::Options::orbit_assignments` maps each qubit
  to an orbit index. When set, `num_parameters` and `build_circuit` use orbit-reduced counts —
  one beta per orbit in the mixer, one gamma per orbit-equivalent cost term group. Reduces
  optimisation dimensionality for structured QUBO problems with known symmetry.

- **Parameter-shift gradient in `Estimator` (P1-4):** `Estimator::gradient(circuit, observable,
  params)` computes `dE/dθ_i = (E(θ+π/2 eᵢ) − E(θ−π/2 eᵢ)) / 2`. All 2P shifted evaluations
  are packed into a single `run_batch` call and computed in parallel. Enables gradient-based
  optimisers (L-BFGS-B, Adam) as an alternative to COBYLA.

- **Transpiler caching in `Estimator` (P1-5):** Structure-keyed `unordered_map` cache (mutex
  protected) stores the transpiled-but-unbound circuit after the first evaluation. Subsequent
  calls with different parameter values skip the SABRE layout + ZYZ/KAK passes entirely.
  Effectively free for COBYLA/gradient loops that re-evaluate the same circuit structure
  thousands of times.

- **`BasisTranslator::run()` (P2-2):** Full implementation in `src/transpiler/basis_translator.cpp`.
  Decomposes all standard gates to the CX+U3 target basis (or any user-specified basis in
  `TranspilationContext::basis_gates`) using a fixed equivalence library. Covers all 1-qubit gates,
  all 2-qubit gates (CY, CZ, CH, SWAP, iSWAP, CRX/CRY/CRZ, CP, RZX, RXX, RYY, RZZ, ECR),
  and 3-qubit gates (CCX, CCZ, CSWAP). Required for real-hardware circuit lowering.

- **`NoiseModel::from_t1_t2()` (P2-4):** Static constructor in `src/noise/noise_model.cpp`.
  Accepts per-qubit T1/T2 arrays and a `gate_name → gate_time` map; creates `thermal_relaxation`
  Kraus channels for every (gate, qubit) pair. Matches Qiskit's `NoiseModel.from_backend()`
  workflow for device-realistic noise modelling.

- **`SparsePauliOp::expectation_value_batch()` (P2-6):** Evaluates `⟨H⟩` for a collection of
  statevectors simultaneously. Precomputes Pauli x/z/y masks once (amortising loop overhead),
  then parallelises over states with `#pragma omp parallel for`. Inner loop uses `#pragma omp simd`
  for vectorisation. Benefit grows with batch size and Hamiltonian term count.

## [1.6.0-alpha] - 2026-03-25

### Performance
- **`SparsePauliOp::expectation_value` — eliminate statevector cloning (P0-1):** Rewrote
  `expectation_value` to compute `⟨ψ|P|ψ⟩` directly without cloning the statevector. Uses
  `x_mask`/`z_mask`/`y_mask` bitmasks and `__builtin_popcountll` to compute the Pauli phase
  per basis state in a single streaming pass. For a 20-qubit, 50-term Ising Hamiltonian this
  eliminates ~100 MB of allocation/deallocation and 50M double reads per energy evaluation.
  Expected 5–50× speedup on the MA-QAOA/VQE inner loop.
- **`Estimator::run_batch` — OpenMP parallelisation (P0-2):** Replaced sequential loop with
  `#pragma omp parallel for schedule(dynamic, 1)`. Thread safety guaranteed: `run_single` uses
  only stack-local state (`bound_circuit`, `StatevectorSimulator`, result). Delivers up to
  N_threads× throughput on batch parameter evaluations.
- **`DensityMatrix::apply_gate` — cache-blocked background enumeration (P0-3):** Replaced
  O(4^N)-with-branches nested loop with direct background-index enumeration (bit-insertion,
  no `continue` branches) and an O(2^k) scratch buffer. Eliminated the full O(4^N) density
  matrix copy per gate application. Expected 10–100× for noisy simulation at N≥10.
- **`DensityMatrix` sampling — replace `std::discrete_distribution` (P0-3, fix 4.3):**
  Sampling now uses `std::partial_sum` + `std::lower_bound` on a cumulative probability array,
  consistent with `Statevector::sample_counts`. Avoids the O(2^N) alias table construction
  overhead of `std::discrete_distribution`.
- **`Statevector::initialize()` — NUMA first-touch (P0-4):** Replaced `memset` with a
  parallel `for` loop so OS page-faults are distributed across threads at allocation time.
  Free on UMA (7900X); 2–4× bandwidth improvement on dual-socket/EPYC hardware.
- **MPS `to_statevector()` threshold (P0-5):** Lowered the threshold for full statevector
  contraction from N≤25 to N≤18 (with bond-dimension-aware formula `N + log2(chi) ≤ 24`).
  Prevents potential OOM or multi-minute stalls for MPS circuits at N>18; always uses
  sequential MPS measurement (O(shots × N × χ³)) for larger systems.

### Fixed
- **`QuantumCircuit::from_qasm2()` wired to `QASM2Parser` (P2-1):** Was throwing
  `"QASM2 parser not yet implemented"` despite the parser being fully implemented in
  `qasm2_parser.cpp`. Added `qasm2_parse_impl` bridge function; `from_qasm2` now delegates
  correctly. Fixes `QASM2PiExpressionRoundTrip` integration test.
- **`Instruction::schedule_time` dedicated field (P2-3):** Added `int schedule_time = -1`
  to `Instruction`. `ASAPSchedule` and `ALAPSchedule` now write to `schedule_time` instead
  of overloading `condition_value` with a `-2` sentinel in `condition_clbit`. Eliminates
  the fragile repurposing that could cause classical conditioning checks to misread schedule
  annotations.

## [1.5.4-alpha] - 2026-03-25

### Fixed
- **MA-QAOA layerwise progress logging:** off-by-one in layer index passed to `LayerCBData` — callback
  was reporting `layer+1` (1-indexed) while start/done messages used `layer` (0-indexed), causing
  mismatched labels in output (e.g. `layer=0 starting` followed by `layer=1 eval=50`). Now consistently
  0-indexed throughout.

### Notes
- Benchmark runs completed: 5-qubit microgrid (all methods, exact optimum found) and 20-qubit microgrid
  (MA-QAOA p=6 layerwise, 1380 params). Full results to be published in a future release.

## [1.5.3-alpha] - 2026-03-24

### Added
- **MA-QAOA microgrid benchmark (5-qubit):** `tests/test_maqaoa_microgrid.cpp` — full C++ port of
  `EnergyGridOpt_py/QAOA` pipeline with hard-coded 5-generator critical scenario (demand=10 MW, A=10).
  Includes: QUBO builder, Ising mapper (h_i, J_ij, energy offset), brute-force exact solver (all 32
  states), simulated annealing (T₀=100, Tf=0.01, α=0.995, 100 runs), and MA-QAOA layerwise (p=3,
  500 evals/layer). All three methods find the exact optimal bitstring `10110` (cost=14). Comparison
  table printed for direct benchmarking against Python reference.
- **MA-QAOA microgrid benchmark (20-qubit):** `tests/test_maqaoa_20qubit.cpp` — C++ port of
  `EnergyGridOpt_py/MA QAOA/EnergyGridOptimisation_20qubit`, scenario `critical_tight_A`
  (demand=52 MW, A=20, 20 generators from `generators.csv`). Includes: exact solver enumerating all
  2²⁰=1M states in ~22 ms (vs Python's PuLP ILP solver), simulated annealing with demand-aware warm
  start, and MA-QAOA layerwise (p=6, 500 evals/layer, term-indexed gammas). Exact optimum: bitstring
  `10111100100000000000`, QUBO cost=16 (generators 01,03,04,05,06,09 — exactly 52 MW, zero penalty).
- **MA-QAOA layerwise progress logging:** `src/algorithms/maqaoa.cpp` now prints per-layer start/done
  messages and best-value updates every 50 evaluations — essential for monitoring long 20-qubit runs.

### Changed
- `tests/CMakeLists.txt`: added `test_maqaoa_microgrid.cpp` and `test_maqaoa_20qubit.cpp` to test sources.

### Notes
- Term-indexed parameterisation (one γ per Hamiltonian term) gives 230 free params/layer for the 20-qubit
  problem vs Python's 40 (qubit-indexed γ). More expressive; each layer is ~5 min on 20-qubit statevector.
- SA single-run success rate on 20 qubits is problem-dependent; Python uses a 300 s time budget
  (`shootout.budget_seconds`) across multiple restarts.

## [1.5.2-alpha] - 2026-03-24

### Fixed
- **Build:** Clang 18+ compatibility — added missing `<complex>` and `<array>` standard library headers in algorithm and noise modules
- **Build:** Clang 18+ compatibility — added missing `#include "qpp/gates.hpp"` in operators.cpp (gates namespace functions now properly declared)
- **Build:** Clang 18+ compatibility — made `StatevectorSimulator::apply_instruction()` public to allow internal module access from quantum_info/states.cpp
- **Build:** Resolved `-ffast-math` + `-Werror` conflict by suppressing `-Wnan-infinity-disabled` warning during Clang build — NaN/infinity checks are safe under -ffast-math when performed post-computation

### Build Verification
- ✓ Successfully compiled with Clang 18.1.3 + libomp (Ubuntu 24.04 LTS)
- ✓ Release build: `-O3 -march=native` optimization flags
- ✓ All 130 build targets compiled successfully
- ✓ Main library (libqpp_core.a), test framework, and benchmark suite linked
- ✓ Test executable generated: `build-clang/tests/qpp_tests` (2.4M)

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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             