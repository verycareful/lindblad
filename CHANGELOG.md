# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project uses semantic versioning labels for release identifiers.

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
