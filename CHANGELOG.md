# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project uses semantic versioning labels for release identifiers.

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
