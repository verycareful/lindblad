# Simulators API Deep Dive

The Simulators API provides four distinct quantum state backends, each optimized for different circuit classes and simulation goals. All simulators follow a common interface (`Result run(circuit, ...)`) and dispatch gate operations via the `Instruction` enum defined in the Circuit API.

## Architectural Overview

### Instruction Dispatch Pattern

All simulators implement instruction execution through a common switch/case pattern:

```cpp
void StatevectorSimulator::apply_instruction(Statevector& sv, const Instruction& inst) {
    switch (inst.type) {
        case GT::X:    gates::apply_x(sv, inst.qubits[0]); break;
        case GT::CX:   gates::apply_cx(sv, inst.qubits[0], inst.qubits[1]); break;
        case GT::RX:   gates::apply_rx(sv, inst.qubits[0], inst.params[0]); break;
        // ... 40+ gate types ...
    }
}
```

**Benefits**:
- Centralized gate application logic (reuses `gates::` namespace functions)
- Consistent handling of parameterized gates
- Stateless (no mutable simulator state during simulation)
- Easy to extend with new gate types

### Execution Flow

1. **Circuit validation**: Verify qubit indices, parameter counts
2. **State initialization**: Allocate and initialize state representation (SV, DM, tableau, or MPS)
3. **Instruction iteration**: Loop over `circuit.instructions` in order
4. **Gate dispatch**: Call appropriate `gates::apply_*` or `apply_gate` function
5. **Measurement handling**: see Execution Semantics below
6. **Result collection**: Extract final state and sampled bitstrings (if shots > 0)

### Operand Validation

Every backend runs a pre-flight over `circuit.instructions` at the start of
`run()`, checking that each qubit and classical-bit index lies in range. This
closes the ingress paths that bypass the per-gate circuit builders (`compose`
index remapping, `control`, the QASM parsers, transpiler passes), so no
out-of-range index reaches a kernel. The statevector and density-matrix backends
surface a failure through `Result` (their `run()` wraps execution in a
try/catch); the MPS and Clifford backends surface it by throwing, consistent with
their existing error contract.

Beneath the pre-flight, the low-level apply-primitives (`gates::apply_*`,
`DensityMatrix::apply_gate` / `apply_kraus`, the `StabilizerState` gates, the MPS
gates, and the qudit apply-primitives) each validate independently: index bounds
throw `std::out_of_range`, and operand-structure violations (non-distinct qubits,
wrong matrix / Kraus / permutation size) throw `std::invalid_argument`. Direct
primitive callers therefore get the same guarantees as circuit callers.

### Physical Validity

A second pre-flight, `QuantumCircuit::validate_physical()`, checks that every
instruction carrying a caller-supplied matrix is unitary, under that
instruction's own policy. It runs beside the operand pre-flight and before gate
fusion, so a matrix is judged while it is still the caller's rather than after
it has been multiplied into a block. Because it has already judged every matrix,
execution applies instructions under `Validation::Ignore`: the per-shot
trajectory and the terminal-measurement pass would otherwise re-measure the same
unchanged matrix once per shot.

Kraus and superoperator entry points check trace preservation on the same terms.
Policies, tolerances, the warning channel, and what the library exempts as its
own arithmetic are documented in [validation.md](validation.md).

### Execution Semantics (frozen in R.1.12)

The statevector, density-matrix, and MPS simulators pick one of three
strategies:

- **Terminal-only measurements** (no classical conditioning, and nothing acts on a qubit after it is measured, the `measure_all` pattern): ONE forward pass with MEASURE skipped, then outcomes are sampled from the final state. Counts keys follow the qubit-to-clbit map of the measure instructions (`n_clbits` wide, clbit 0 rightmost); partial measurements key only the measured qubits. This replaces the per-shot re-execution used before R.1.12, which cost `shots` full evolutions for the most common circuit shape.
- **Mid-circuit measurement or feedforward** with `shots > 0`: per-shot trajectories. The circuit is re-executed from `|0...0⟩` once per shot; each MEASURE collapse is drawn independently, conditions are evaluated against the per-shot classical register.
- **shots == 0**: a single seeded trajectory. Classical conditions are honoured and MEASURE outcomes are recorded along the way; `final_state` is one reproducible trajectory. (`eval_expectation` instead THROWS for measure/conditional circuits: the exact expectation of one random trajectory is undefined; estimate from counts with `shots > 0`.)

The MPS simulator collapses a qubit at the chain's orthogonality centre, where
the outcome's marginal is a read of one site and dividing that site by the
marginal's square root restores unit norm (see [Canonical Form](#canonical-form));
sampled MPS bitstrings use the project key convention (qubit 0 rightmost) at
every register width.

### The run harness

Every `run()` takes a trailing `const RunPlan&` that defaults to empty. It
carries the state the run starts from and the observers watching it while it
runs, neither of which belongs in a circuit: a circuit describes physics, and
watching a simulation or substituting its state is not physics. An empty plan
starts at the all-zero state and watches nothing, so a caller who passes no plan
is unaffected.

```cpp
StatevectorSimulator::Result   run(circuit, shots, seed, plan);
DensityMatrixSimulator::Result run(circuit, noise_model, shots, seed, plan);
CliffordSimulator::Result      run(circuit, shots, seed, plan);
MPSSimulator::Result           run(circuit, max_bond_dim, shots, seed, plan);
```

Each `Result` carries an `observations` bundle holding whatever the run's
labelled observers collected, empty unless observers with labels were attached.

Two backends change execution strategy while a plan is watching, and both do so
to keep anchors meaning what they said:

- The statevector simulator suppresses gate fusion, since fusion rewrites
  instructions into blocks and renumbers the positions anchors name.
  `RunPlan::Options::Fusion::Keep` asks for fusion anyway, and anchors then bind
  to the fused circuit.
- The Clifford simulator runs its gate pass on the row-major tableau rather than
  the bit-sliced one, which is not a state an observer can be handed. Slab
  sampling is unaffected, so only the gate pass gives up its speed.

On strategies where one evolution serves every shot, observers fire once,
because that evolution describes all of them. On per-shot trajectory paths they
fire once per shot.

See [Observation and the run harness](observation.md) for anchors, the observer
catalogue, the conversion and cost policy, and the initial state.

## StatevectorSimulator

Exact simulation of pure quantum states using the aligned Statevector representation.

### State Representation

See [Statevector API](statevector.md) for full details. State stored as:
- Dual aligned arrays: `real_parts` and `imag_parts` (64-byte alignment for AVX-512)
- Structure-of-Arrays (SoA) layout enabling SIMD vectorization
- Complexity: $O(2^n)$ space for $n$ qubits

### Options

```cpp
struct Options {
    int max_parallel_threads = 0;  // 0 = auto (all cores)
    uint64_t max_memory_mb = 0;    // 0 = auto
    int precision = 64;            // 32 or 64 bit (not yet used)
    bool zero_threshold = true;
    double threshold = 1e-10;      // Unused; for API compatibility
    bool fusion_enable = true;     // master switch for gate fusion
    int fusion_threshold = 0;      // min qubits to engage fusion; 0 = auto
    int fusion_max_qubit = 5;      // max fused-block width, 2..6
};
```

- **max_parallel_threads**: Cap on OpenMP thread count (0 = system default)
- **max_memory_mb**: Memory budget (0 = no limit); used for preemptive error checking
- **precision**: Reserved for future 32-bit float variants
- **zero_threshold**, **threshold**: Legacy fields; may be removed in future versions
- **fusion_enable**: Master switch for the gate-fusion pre-pass (see Gate Fusion below); `false` runs every circuit unfused
- **fusion_threshold**: Minimum circuit qubit count at which fusion engages. `0` (the default) derives the point from the hardware at runtime: the first $n$ whose statevector ($16 \cdot 2^n$ bytes) exceeds one last-level-cache instance. An explicit value pins the point verbatim; values below the auto point can regress cache-resident circuits, so overriding downward is for experimentation, not production. `run()` rejects negative values.
- **fusion_max_qubit**: Maximum width of a fused block in qubits, valid range 2 to 6. Follows the Qiskit Aer option name; Aer's default width (5) is also the default here.

### Result Structure

```cpp
struct Result {
    Statevector final_state;                          // Full amplitude vector
    std::unordered_map<std::string, int> counts;      // Sampled bitstrings (if shots > 0)
    std::vector<double> expectation_values;           // Empty; use Estimator for expectations
    double simulation_time_seconds = 0.0;
    bool success = true;
    std::string error_message;
};
```

**Fields**:
- **final_state**: Complete quantum state after circuit execution
- **counts**: Measurement outcome histogram; keys are bitstrings (e.g., "01"), values are occurrence counts
- **expectation_values**: Reserved for future use (currently not populated by simulator)
- **success**: `false` when the run failed. `StatevectorSimulator` and `DensityMatrixSimulator` report failures through this field rather than by throwing, so a caller checks it; `MPSSimulator` and `CliffordSimulator` have no such field and throw instead
- **error_message**: what went wrong, when `success` is `false`; empty otherwise
- **observations**: whatever the run's labelled observers collected, empty unless the `RunPlan` attached observers carrying labels. A failed run leaves it empty rather than partly written

### Workflow

**For exact state inspection** (no measurement):

```cpp
StatevectorSimulator sim;
auto result = sim.run(circuit, 0);  // shots=0: no sampling
Statevector state = result.final_state;
```

**For measured sampled outcomes**:

```cpp
auto result = sim.run(circuit, 1024, 42);  // shots=1024, seed=42
std::unordered_map<std::string, int> counts = result.counts;
// E.g., {"00": 256, "11": 768}
```

**Workflow summary**:
1. User calls `run(circuit, shots, seed)`
2. Create Statevector initialized to $|0\cdots0\rangle$
3. Iterate over circuit instructions, calling `apply_instruction`
4. If shots > 0, sample `sv.sample_counts(shots, seed)` via cumulative distribution
5. Return Result with final state and counts

### Fast Expectation Values

For variational inner loops, `StatevectorSimulator::eval_expectation(circuit, observable)` simulates the circuit and computes the expectation value in-place, bypassing the `Result` struct and avoiding an $O(2^n)$ allocation of the final state vector. This is used by the `Estimator` ideal path.

### Gate Fusion (R.1.17)

When the statevector outgrows one last-level-cache instance, `run()` executes a fused equivalent of the circuit: consecutive fusable gates are greedily merged while the union of their supports stays within `fusion_max_qubit` qubits, each block is composed into a dense $2^k \times 2^k$ unitary (by applying the member gates to basis columns through the simulator's own dispatch, so every gate type composes exactly), and the block is applied as a single `apply_unitary` stride pass. Bandwidth-bound simulation is where fusion pays — $k$ gates cost $k$ full sweeps of $2^n$ amplitudes unfused, one sweep fused — while cache-resident states are compute-bound: there the specialised per-gate kernels win and a dense block would only add arithmetic. Measurements on a 32 MiB-L3 machine put the boundary exactly at the cache size (fusing a state equal to the LLC instance ran 3.3x slower; twice the LLC, 2.7x faster), so the engagement point is the first $n$ whose state exceeds one LLC instance — detected at runtime, per L3 instance rather than package total because on multi-CCD parts a thread's working set lives in its own CCD's slice. When the cache size cannot be detected, 32 MiB is assumed (engaging at $n \geq 22$), which errs toward engaging later: a missed fusion win costs far less than a mid-range regression. `fusion_threshold` pins the point explicitly; `fusion_enable = false` disables the pre-pass entirely. Below the engagement point circuits execute unfused and bit-identically to earlier releases.

Semantics are untouched by construction: `MEASURE` / `RESET` / `BARRIER`, classically-conditioned instructions, unresolved parameterised gates, and the structured ops (`MCX` / `MCP` / `PERMUTATION`, which already have fast native paths) are never fused — they flush the current block and pass through verbatim. Single-gate blocks emit the original instruction. The fused plan is built once per `run()` and, on the per-shot trajectory path, reused across every shot.

### Complexity Analysis

- **Time**: $O(2^n k)$ where $k$ = number of instructions; $O(2^{2k})$ for $k$-qubit gates
- **Space**: $O(2^n)$ for statevector
- **Parallelization**: Each gate parallelizes if $2^n \geq 2^{20}$ (via OpenMP)
- **Measurement**: $O(2^n + \text{shots} \cdot \log 2^n)$ to sample (cumulative distribution + binary search)

### Use Cases

- Exact simulation of small-medium circuits (5–20 qubits, depth < 1000)
- Reference implementation for validation
- Analysis of quantum state properties (amplitudes, entanglement)
- Gradient computation via parameter-shift rule (used by Estimator)

### Limitations

- Exponential memory: 2 GB per 28 qubits (double precision), 1 GB per 27 qubits
- No noise support (use DensityMatrixSimulator for noisy circuits)
- Clifford-only circuits require stabilizer tableau (CliffordSimulator is more efficient)

## DensityMatrixSimulator

Exact simulation of mixed quantum states with integrated Kraus operator noise application.

### State Representation: DensityMatrix

```cpp
class DensityMatrix {
    int n_qubits;
    size_t dim;  // 2^n_qubits
    std::vector<Complex128> data;  // row-major, dim × dim matrix
};
```

**Properties**:
- Stores full density matrix $\rho$ in row-major order: `data[i*dim + j]` = $\rho_{ij}$
- Valid states satisfy: $\text{Tr}(\rho) = 1$ and $\rho = \rho^\dagger$ (Hermitian)
- Purity: $\gamma = \text{Tr}(\rho^2) \in [0, 1]$; $\gamma = 1$ iff pure, $\gamma = 1/2^n$ iff maximally mixed
- Complexity: $O(4^n)$ space for $n$ qubits

**Validity and normalization**:

- `trace()` returns $\text{Tr}(\rho)$, `purity()` returns $\text{Tr}(\rho^2)$
- `is_valid(atol)` checks trace and Hermiticity, at the framework tolerance.
  Positive semi-definiteness is NOT verified: a full check needs an
  eigendecomposition and is $O(4^n)$
- `normalize()` divides every entry by the trace. It throws when there is no
  trace to divide out, a zero or non-finite matrix, rather than returning the
  matrix unchanged
- `is_normalized(atol)` is a predicate over $\text{Tr}(\rho) = 1$ alone: it
  answers, and neither repairs nor throws
- `check_normalized(validation)` applies a validation policy, with
  `Repair::Attempt` renormalizing in place

### Density Matrix Initialization

Pure state conversion:
$$\rho = |\psi\rangle\langle\psi| \quad \Rightarrow \quad \rho_{ij} = \psi_i \cdot \overline{\psi_j}$$

Initial state $|0\rangle$ gives:
$$\rho = \begin{bmatrix} 1 & 0 \\ 0 & 0 & \ddots \\ 0 & \cdots & 0 \end{bmatrix}$$

### Gate Application: Localized Tensor Operation

Gate application $\rho \to U \rho U^\dagger$ uses **localized tensor operations** on target qubits only, avoiding the $O(4^n)$ branch-per-element loop.

**Algorithm**:

1. **Sort targets** and compute offset masks: `sub_offsets[s]` maps sub-index $s$ to target qubit positions
2. **Enumerate background indices** (non-target qubits): $2^{n-k}$ distinct background configurations
3. **For each background** configuration and **each column** of the density matrix:
   - Read 2D sub-vector (dimension $2^k \times 2^k$)
   - Apply $U$ matrix multiplication (left multiply): row update
4. **For each row** and **each background** configuration:
   - Read 2D sub-vector
   - Apply $U^\dagger$ matrix multiplication (right multiply): column update

**Code Structure**:

```cpp
void DensityMatrix::apply_gate(const std::vector<Complex128>& U,
                                const std::vector<int>& qubits) {
    int k = qubits.size();
    size_t sub_dim = 1ULL << k;
    std::vector<size_t> sub_offsets(sub_dim);  // Precompute target offsets
    std::vector<size_t> bg_indices(dim >> k);  // Precompute background indices
    std::vector<Complex128> scratch(sub_dim);  // Scratch buffer: O(2^k)

    // Left multiply: rho = U * rho (row update)
    for (size_t bg : bg_indices)
        for (size_t col = 0; col < dim; ++col)
            // Read, apply, write using scratch buffer
            
    // Right multiply: rho = rho * U† (column update)
    for (size_t row = 0; row < dim; ++row)
        for (size_t bg : bg_indices)
            // Read, apply, write using scratch buffer
}
```

**Complexity**: $O(4^n \cdot 2^k)$ for $k$-qubit gate; $O(2^k)$ scratch memory per thread

**Benefits**: 
- Avoids expensive allocations per gate
- Sequential memory access patterns (amenable to SIMD, though not vectorized in current code)
- No full-matrix copy overhead

### Kraus Operator Application

For noise channels with $m$ Kraus operators $\{K_1, \ldots, K_m\}$ satisfying $\sum_i K_i^\dagger K_i = I$:

$$\rho \to \sum_{i=1}^{m} K_i \rho K_i^\dagger$$

**Implementation (R.1.13 gates; R.1.17 channels)**: `apply_gate` is an OpenMP
row-block AXPY over contiguous rows (`rho = U rho U†` via an in-place ket
multiply plus a row-local bra multiply, parallel over background groups /
rows). `apply_kraus` fuses the **whole channel into one superoperator** and
applies it in a **single pass** over $\rho$:

$$S_{(r_o c_o),(r_i c_i)} = \sum_k K_k[r_o, r_i] \cdot \overline{K_k[c_o, c_i]}, \qquad \text{vec}(\rho'_{\text{block}}) = S \cdot \text{vec}(\rho_{\text{block}})$$

per background pair, identity elsewhere. Every $\rho$ element is read and
written exactly once regardless of the operator count $m$ — previously each
operator cost a ket sweep, a bra sweep, and an accumulate pass plus two
$4^n$ scratch allocations per call, so a 16-operator two-qubit depolarizing
channel swept the full matrix ~48 times per noisy gate. The superoperator is
at most $4^k \times 4^k$ ($16 \times 16$ for the 1–2 qubit channels noise
models attach) and costs $O(m \cdot 16^k)$ to build, negligible next to one
sweep. `apply_channel_superop(S, qubits)` exposes the single-pass path
directly for callers that already hold a superoperator (external LSB-first
convention, matching `KrausChannel`; trace preservation is the caller's
responsibility).

**Complexity**: $O(4^n \cdot 4^k)$ per channel — independent of $m$.

The `DensityMatrixSimulator` also pre-resolves each instruction **once per
circuit** into a plan: gate matrices, and every attached channel fused into
its superoperator with its stride tables, so per-shot trajectory execution
pays zero per-call setup. One density-matrix buffer is reused across shots,
and the structured `MCX`/`MCP`/`PERMUTATION` ops apply as a full-register
row/column relabel or diagonal phase (no dense matrix).

### DensityMatrixSimulator Workflow

```cpp
DensityMatrixSimulator sim;
NoiseModel noise_model;
noise_model.add_quantum_error(
    lindblad::NoiseChannels::depolarizing(0.001), "cx", {0, 1});

auto result = sim.run(circuit, noise_model, 1024);  // shots=1024
```

**Execution Steps**:
1. Initialize $\rho = |0\rangle\langle 0|$ (dim $\times$ dim)
2. For each instruction in circuit:
   - If noise is attached and any errors have `after_gate = false`: apply those Kraus channels **before** the unitary via `apply_kraus`
   - Apply gate via `apply_gate` (Hamiltonian evolution)
   - If noise is attached and any errors have `after_gate = true`: apply those Kraus channels **after** the unitary via `apply_kraus`
3. Sample measurement outcomes via spectral decomposition of marginal density matrices
4. Return counts and final state

### Expectation Values

**Hermitian operator expectation** (e.g., Pauli observable):

$$\langle O \rangle = \text{Tr}(\rho O)$$

```cpp
double DensityMatrix::expectation_value(const std::vector<Complex128>& hermitian_op) const {
    double exp = 0.0;
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j)
            exp += (data[i*dim + j] * hermitian_op[j*dim + i]).real;
    return exp;
}
```

It throws `std::invalid_argument` unless the operator has `dim * dim` entries
and is Hermitian to within `DEFAULT_PHYSICAL_ATOL` entrywise: the trace of a
non-Hermitian operator is complex, and the function returns a real number.
Both checks cost O(dim^2), the same as the trace.

**SparsePauliOp expectation** (specialized for sparse Pauli strings):

`expectation_value_sparse` applies the operator rules in
[operators.md](operators.md#width-rule): at least one term, every term exactly
`n_qubits` wide and written with `I`, `X`, `Y` and `Z`, and a Hermitian
operator.

Extracts diagonal terms of Pauli strings using bit masks; computes traces without full matrix multiplication:

$$\langle P \rangle = 2^{-n} \text{Tr}(P_0 \rho) \quad \text{where} \quad P_0 \text{ is diagonal Pauli}$$

**Complexity**: $O(4^n)$ for dense operator, $O(4^n \cdot m)$ where $m$ = number of Pauli terms

### Complexity Analysis

- **Time per gate**: $O(4^n \cdot 2^k)$ for $k$-qubit gate
- **Space**: $O(4^n)$ for density matrix + $O(2^k)$ scratch per gate
- **Noise overhead**: Factor of $m$ for $m$ Kraus operators

### Use Cases

- Noisy simulation (with Kraus operators from `NoiseModel`)
- Analysis of mixed state properties (purity, entanglement)
- Validation of noise models
- Circuits up to 10–12 qubits (exponential storage in pure states impractical)

### Limitations

- 16 GB per 10 qubits (complex128); impractical beyond ~11 qubits
- Slower than statevector for pure-state circuits (overhead of full density matrix)
- No advantage over statevector without noise

## CliffordSimulator

Efficient exact simulation of Clifford circuits using stabilizer tableau representation.

The supported gate set is $\{H, S, S^\dagger, \sqrt{X}, \sqrt{X}^\dagger, X, Y, Z, CX, CY, CZ, SWAP, iSWAP, ECR\}$, together with $P$, $RX$, $RY$, $RZ$, $RXX$, $RYY$, $RZZ$ and $RZX$ at multiples of $\pi/2$, which are the angles at which those rotations are Clifford.

Global phase is not represented in the stabilizer formalism, so the backend reproduces each gate up to phase. That is exact for measurement outcomes and Pauli expectations, which is everything this backend reports.

### State Representation: StabilizerState

**Tableau**: $2N \times (2N+1)$ binary matrix representing stabilizers and destabilizers:

$$\begin{pmatrix}
\text{Destabilizer}_0 & \text{phase}_0 \\
\vdots \\
\text{Destabilizer}_{N-1} & \text{phase}_{N-1} \\
\text{Stabilizer}_0 & \text{phase}_N \\
\vdots \\
\text{Stabilizer}_{N-1} & \text{phase}_{2N-1}
\end{pmatrix}$$

Each row represents a Pauli operator as:
- Columns $0$ to $N-1$: X-part (1 = X or Y component present)
- Columns $N$ to $2N-1$: Z-part (1 = Z or Y component present)
- Column $2N$: phase bit (0 = $+1$, 1 = $-1$)

**Invariant**: All stabilizers commute pairwise; destabilizers anti-commute with corresponding stabilizers.

**Complexity**: $O(N^2)$ space for $N$ qubits (dense binary matrix)

### Clifford Gate Application

Each gate updates the tableau via row operations:

**Hadamard** ($H$): Swap X and Z parts
- For all rows: `(row.x, row.z) = (row.z, row.x)`

**S-gate** ($S = \text{diag}(1, i)$): Updates Z component
- For each row: If `row.x[i] = 1`, XOR with `row.z`

**CNOT** (control=i, target=j): Coupled row operations
- If row has `x[j] = 1`, XOR with `x[i]`
- If row has `x[i] = 1`, XOR with `z[j]`
- Conditionally update phase

**Pauli gates** (X, Y, Z): Update phase based on row anticommutation

### Measurement

**In Clifford simulation**, measurement of qubit $j$:
1. Find a destabilizer row with `x[j] = 1` (must exist for non-trivial state)
2. Invert via Gaussian elimination to convert to stabilizer
3. Measure outcome: randomly 0 or 1 (if deterministic, derived from phase)
4. Project state onto measurement outcome via controlled row operations

**Complexity**: $O(N^2)$ per measurement (row operations)

### Expectation Value: Pauli String

$$\langle \prod_i P_i^{q_i} \rangle = \pm 1 \text{ or } 0 \text{ (indeterminate)}$$

Check if the Pauli string commutes with all stabilizers:
- If yes, expectation is deterministic ($\pm 1$)
- If no, measurement outcome is random (expectation $= 0$)

**Complexity**: $O(N^2)$ per expectation query

### Workflow

```cpp
CliffordSimulator sim;
auto result = sim.run(clifford_circuit, 1024);  // Must be Clifford-only
```

**Terminal-measurement fast path (R.1.13, audit F-19)**: when there is no
feedforward, no `RESET`, and nothing acts on a qubit after it is measured, the
pre-measurement stabilizer state is deterministic. The gate pass then runs
ONCE and each shot samples measurements from a copy of the tableau, instead of
re-applying every gate per shot. Circuits with mid-circuit measurement,
feedforward, or reset use the general per-shot trajectory path.

**Use Cases**:
- Verification of Clifford circuits (error correction, stabilizer codes)
- Analysis of stabilizer codes (exact marginal probabilities)
- Research into Clifford-only gates and measurement outcomes

### Use Cases & Limitations

**Strengths**:
- Polynomial ($O(N^2)$) space and time for Clifford circuits
- Exact for arbitrarily large systems (limited by RAM, not exponential scaling)
- Ideal for stabilizer codes and error correction benchmarks

**Limitations**:
- Only the Clifford gate set above; $T$ and $T^\dagger$ are outside it
- Parameterized rotations are accepted only at multiples of $\pi/2$. That
  covers the one-qubit `p`, `rx`, `ry`, `rz` and the two-qubit Ising rotations
  `rxx`, `ryy`, `rzz`, `rzx`, which the tableau runs as `cx . s . cx` on the
  ZZ axis conjugated by the single-qubit Cliffords that rotate Z into X or Y
- A gate outside the set throws rather than being applied approximately

**Is-Clifford Check**:

`CliffordSimulator::is_clifford` reports whether a circuit can take the tableau
path. The automatic backend selection calls it, so a circuit that fails the
check runs on another simulator rather than failing.

```cpp
QuantumCircuit qc(2);
qc.h(0).sx(1).ecr(0, 1).rz(PI_2, 0);
bool ok = CliffordSimulator::is_clifford(qc);   // true
```

A direct call to `run()` bypasses that check, and an unsupported gate throws
there rather than being skipped.

### Options

```cpp
CliffordSimulator sim;
sim.options.sampling = CliffordSimulator::Options::Sampling::Auto;
sim.options.elimination = StabilizerState::Elimination::Plain;
```

`sampling` chooses how terminal measurements become shots. `Slab` reads the
outcome distribution's affine subspace off the tableau once and draws each shot
as a subset-sum of its free directions. `PerShot` replays a measurement pass
over a copy of the tableau for every shot. `Auto`, the default, picks the slab
wherever it applies and the per-shot route otherwise.

Both sample the same distribution. They consume the random stream differently,
so a given seed produces different individual bitstrings under each; counts
agree in distribution, not shot for shot.

Circuits with mid-circuit measurement, feedforward or reset take the per-shot
route regardless of what is selected, because the subspace describes a terminal
measurement of a fixed state: once a measurement collapses the state
mid-circuit, each trajectory diverges and there is no single subspace left to
read. Selecting `Slab` explicitly for such a circuit emits a note saying the
per-shot route was used instead. `Auto` asked for nothing and stays silent.

`elimination` chooses how that subspace is extracted. `Plain`, the default,
multiplies a row into another only where the pivot bit is set. `FourRussians`
clears a block of pivot columns from each row with one multiplication against a
table of subset products; it is asymptotically better and slower at ordinary
sizes, since the table costs $2^k$ multiplications regardless of how many rows
remain to amortise it over. Selecting it emits a one-time note.

### Outcome Distribution

A stabilizer state's computational-basis outcomes are uniform over a coset of a
linear subspace of $\mathbb{F}_2^n$. `StabilizerState::outcome_slab()` returns
that shape: one point of the coset in `offset`, and the directions along it in
`basis`. Every outcome is `offset` combined with some subset of `basis`, each
equally likely.

```cpp
StabilizerState state = result.final_state;
auto slab = state.outcome_slab();
// slab.dim free directions, so the support has 2^slab.dim outcomes
```

## MPSSimulator

Approximate simulation of arbitrary circuits using Matrix Product State (MPS) representation, enabling simulation of larger systems at the cost of controlled truncation error.

```cpp
#include "lindblad/simulators/mps_sim.hpp"  // MPSSimulator, MPSState, MPSTensor
```

Everything is in namespace `lindblad`. `CanonicalForm`, `SVDMethod` and
`MPS_DEFAULT_CUTOFF` are declared in `lindblad/types.hpp`, which this header
includes.

### State Representation: MPSState

An MPS decomposes an $n$-qubit state as:

$$|\psi\rangle = \sum_{s_0, \ldots, s_{n-1}} M_0^{s_0} \cdot M_1^{s_1} \cdots M_{n-1}^{s_{n-1}} |s_0, \ldots, s_{n-1}\rangle$$

where each $M_i^{s_i}$ is a $(\chi \times \chi)$ matrix (bond dimension $\chi$) and $s_i \in \{0, 1\}$ is the physical index.

**Data Structure**:

```cpp
struct MPSTensor {
    int bond_left, bond_right;
    // shape: (bond_left, physical_dim=2, bond_right)
    std::vector<Complex128> data;
    
    Complex128& operator()(int left, int phys, int right) {
        return data[left * 2 * bond_right + phys * bond_right + right];
    }
};

class MPSState {
public:
    int n_qubits;
    int max_bond_dim;    // chi parameter
    double cutoff;       // max fraction of weight truncation may discard
    SVDMethod svd_method = SVDMethod::BDC;     // bond-split kernel
    bool svd_rescue = true;                    // descend the ladder on a rejected factorisation
    CanonicalForm canonical_form = CanonicalForm::Always;  // when a split moves the centre first

    MPSState(int n_qubits, int max_bond_dim = 64,
             double cutoff = MPS_DEFAULT_CUTOFF);

    const std::vector<MPSTensor>& tensors() const;   // read-only
    void set_tensors(std::vector<MPSTensor> sites);  // validated replacement
    std::pair<int, int> open_span() const;
    void canonicalize(int site);
    // gates, norm, measurement, profile: below
};
```

The site tensors are private. Reading them through `tensors()` is free;
replacing the chain goes through `set_tensors()`, because every operation relies
on the chain's open span (next section) and a direct write would falsify it.

**Norm and normalization**:

- `norm_sq()` contracts the transfer matrix over the open span only, since the
  sites outside it contract to the identity: at a single-site centre it is that
  site's squared Frobenius norm, $O(\chi^2)$, and with the span open over the
  whole chain $O(n \cdot \chi^3)$. There is no flat amplitude array to sweep,
  so this is the only way to read the norm without materialising the state
- `normalize()` rescales the first site of the open span, which is exact because
  the norm is multilinear in the tensors; a site outside the span would lose its
  orthonormality if it were scaled. It throws when there is no norm to divide
  out, a zero or non-finite state, rather than returning the state unchanged
- `is_normalized(atol)` answers without repairing or throwing
- `check_normalized(validation)` applies a policy, with `Repair::Attempt`
  renormalizing. Under `Ignore` with `Repair::None` the contraction does not run
  at all, which matters more here than on the dense classes because this
  measurement is the most expensive of any state type in the library

**SVD kernel**: `svd_method` (declared in `lindblad/types.hpp`, shared with
the qudit MPS) selects the factorisation every bond split asks for first. Four
kernels from two providers; the values name algorithms, prefixed by provider
only where both offer the same one:

| Value | Provider | Method | Accuracy promise |
|---|---|---|---|
| `BDC` (default) | autonne | Householder bidiagonalisation, Gu-Eisenstat divide and conquer | absolute: `abs(s_i - s_i(true)) <= 64 * max(rows, cols) * eps * s_max` |
| `Jacobi` | autonne | one-sided cyclic Jacobi | relative, on every singular value down to a column floor of `2^-500` |
| `EigenBDC` | Eigen | `BDCSVD` | Eigen's |
| `EigenJacobi` | Eigen | `JacobiSVD` | Eigen's |

`BDC` is the default because a bond split truncates on weight, which is what
an absolute bound serves, and because its cost is `O(n^3)` with a constant the
spectrum barely moves: on a 128x128 decaying spectrum it is 2.7x faster than
`Jacobi` and no faster on a flat one. `Jacobi` resolves the tail of a graded
spectrum that an absolute bound treats as noise, and is the ladder's first
rescue (below), so it is the choice when the tail matters more than the
clock. Selecting either Jacobi kernel emits a one-time note per MPS layer to
the warning channel that it is the slower algorithm; selecting `EigenBDC` is
silent. Every kernel is available in every build, so the public API does not
change shape with the build configuration.

The autonne kernels take the project's floating-point flags as they are, being
verified under both models by their own suite; the Eigen kernels run in a
translation unit compiled under strict IEEE arithmetic, since Eigen's entry
guards do not survive `-ffast-math`. All four are held to the same verification
described below. Kernels do not agree bit for bit, so a state truncated under
one differs in its last digits from the same state truncated under another.
The selection reaches every split the chain performs, the rebuild from dense
amplitudes included, so `svd_time_ns()` under a selected kernel is that
kernel's time throughout.

The autonne revision is fetched at a release tag; override it with
`-DLINDBLAD_AUTONNE_GIT_TAG=<tag or commit>`, or point
`-DLINDBLAD_AUTONNE_REPOSITORY` at a local clone to build without network.

**Verified truncation**: the SVD output is not trusted blindly. A third-party
SVD can return a corrupt factorisation on degenerate rank-deficient input (the
class of two-site tensors Shor-style circuits produce), in failure shapes
ranging from NaN singular vectors to a wrong-but-finite kept vector. Every
truncation therefore: selects the kept singular values by bit-level-finite
comparison (immune to ordering corruption), verifies the kept factorisation
against the Frobenius identity `‖M − U·S·V†‖²_F = Σ(discarded σ²)`, and on a
rejection descends a rescue ladder: autonne's `Jacobi` (an independent road to
the same factorisation, skipped when it was the selected kernel), then a
Gram-matrix eigendecomposition (sharing no code with either SVD), then
`std::runtime_error` rather than continuing with a corrupt tensor. Every rung
descended is reported through the warning channel, naming the layer, the block
shape and the kernel that failed, so a run rescued on every bond reads as one.

`svd_rescue = false` forbids the descent: the first rejected factorisation
throws, for a caller who would rather stop than accept a tensor from a kernel
they did not name. `MPSSimulator` carries the same two fields, and
`canonical_form`, and copies all three onto every chain it builds.

That identity is an equality for a true truncated SVD, so the allowance above
the discarded weight is only the backward error a stable SVD is entitled to.
The bound is stated in the amplitude domain,
`‖M − U·S·V†‖_F ≤ sqrt(Σ discarded σ²) + c·N·eps·‖M‖_F` with `N` the larger
matrix dimension, and applied by squaring the right-hand side whole.

Both halves earn their place. Sizing the backward-error term tightly matters
because a factorisation can reconstruct its input to a few parts in `10⁸`,
which is nowhere near backward-stable yet is orders away from producing a NaN;
a looser gate accepts it, and the resulting state carries a norm error while
its Schmidt directions stay exact, which no downstream check would catch.
Keeping the bound in the amplitude domain matters because squaring it termwise
would drop the cross term. Under heavy truncation the residual and the discarded
weight are two large nearly-equal quantities arrived at by different routes, so
their difference carries first-order rounding, and a purely squared bound would
reject sound factorisations there. The cross term vanishes as the discarded
weight does, so a bond that truncated nothing still faces the strict bound.

`max_verify_residual_excess()` reports what the gate actually admitted.
`truncation_error()` counts the weight truncation chose to drop, meaning the
directions the weight budget or the bond cap rejected, and is committed only
after verification. The verification costs roughly one extra rank-slice matrix
multiply per two-qubit gate.

`truncation_error()` accumulates across every split, so it grows with the
number of splits a run performs and two runs are comparable only when they
perform the same ones. Each term is the absolute weight its split threw away. A
split taken in canonical gauge discards that weight from the state, whose norm
its block then carries, so on a chain with no collapse, no normalisation and no
absorbed profile the total equals how far `norm_sq()` has fallen since the chain
was built, to rounding. Under `CanonicalForm::Auto` the splits that run in place
contribute weight at their blocks' rounding level instead.

It is a weight, not a fidelity. How close the chain is to the state an
untruncated run would hold is what `fidelity_estimate()` and
`fidelity_lower_bound()` report (see [Fidelity Figures](#fidelity-figures)), and
those are the figures to compare one bond cap against another with.

The reconstruction residual that decides whether a factorisation is accepted is
computed in a translation unit compiled under strict IEEE floating-point, whichever
kernel produced it. The residual subtracts two nearly identical matrices, and one
computed too small would admit exactly the factorisations the check exists to
reject.

**Ladder observability**. A rescued split is reported through the warning
channel and is otherwise indistinguishable from a clean one, since both yield valid
tensors. The counters on `MPSState` are the record of how often it happened
and what it cost:

- `svd_call_count()`: bond splits performed, one per call into the truncation
  routine. This is the denominator; a rescue count means nothing without it.
- `jacobi_rescue_count()`: splits where the selected kernel's factorisation
  failed verification and autonne's `Jacobi` produced the accepted slice.
- `gram_fallback_count()`: splits where the Gram route produced the accepted
  slice, after the selected kernel and the Jacobi rescue both failed. Only
  successful rescues are counted on either rung, because a split on which every
  rung fails throws.
- `floor_rejected_weight()`: the Gram route's own cost, summed over the splits
  it rescued: singular weight below its validity floor (see the rebuild section
  below). It is not truncation and is kept out of `truncation_error()`. Zero
  unless some split took the Gram rung.
- `max_verify_residual_excess()`: the worst factorisation error verification
  accepted, as a fraction of $\|M\|_F^2$, maximised over splits. The Frobenius
  identity holds with equality for a true truncated SVD, so this reports the
  excess over that ideal rather than the raw residual, and a healthy run sits
  near the square of machine epsilon. It says how close a run came to being
  rescued, and how much error the accepted route let through when it was not.
- `svd_time_ns()`: nanoseconds spent in the truncation routine, over the same
  splits `svd_call_count()` counts. The interval covers the factorisation, the
  verification deciding whether to accept it, and any Gram rescue verification
  forced, and stops before this layer copies the factors into its own storage
  convention. It is therefore an upper bound on what a faster SVD kernel could
  remove rather than an estimate of it: verification costs roughly one extra
  rank-slice matrix multiply per split and survives any change of kernel, as
  does a rescue. A split that threw contributes nothing, having no result to
  profile.

A run with both rescue counts at zero never distrusted its kernel. A nonzero
count is not an error: it is the containment working.

The warning channel reports every rescue, and collapses identical messages the
way it collapses every repeated warning (see the
[validation reference](validation.md)): the
first occurrence of a message is delivered at once, and later identical ones
arrive as a single repeat count at the next `flush_warnings()`.
`MPSSimulator::run()` flushes when it returns. A caller driving an `MPSState` or
a `QuditMPS` directly calls `flush_warnings()` to receive the counts. Rescues of
blocks with the same layer, shape and cause produce identical messages, so the
number of warning lines is not the number of rescues; `jacobi_rescue_count()`
and `gram_fallback_count()` are.

A rescue warning needs no action from a caller. The tensor that continues has
passed verification, and a split no rung can serve throws instead of
continuing. The warning matters only to someone developing the kernel that
declined (autonne or Eigen) or Lindblad itself. Under the default `BDC`, some
blocks are declined and served by the `Jacobi` rung, mostly from qudit chains
at `d = 3`, and which blocks depends on the compiler.

Every figure covers every split the chain has taken, including those from
`rebuild_from_statevector` below. They describe the state rather than the route
that produced it, so a chain rebuilt part way through a run still carries what
the gates before the rebuild cost.

**Folding one chain's figures into another**.

```cpp
void MPSState::absorb_profile(const MPSState& other);
```

Adds `other`'s tallies to this chain's and takes the larger of the two worst
residuals. The tensors, the register width, the cap, the cutoff and the
kernel selection are untouched, so afterwards the chain reports splits it did
not itself perform, exactly as one chain performing both sets would have.

It is what makes the figures on the chain `MPSSimulator::run` returns cover
every split of the run on every path. Mid-circuit measurement or feedforward at
nonzero shots re-simulates per shot, each trajectory on a chain of its own;
each trajectory absorbs the figures the run has gathered so far and becomes the
returned chain, so what comes back is the last trajectory in every respect,
carrying the totals of all of them. `truncation_error()` read there is
everything the run discarded rather than the returned tensors' own history,
which is the accumulate-rather-than-reset contract above applied across
trajectories. So `svd_time_ns()` against the run's wall clock is the share bond
splitting took of the whole run, whichever path it ran on.

**Rebuilding a chain from dense amplitudes**.

```cpp
void MPSState::rebuild_from_statevector(const Statevector& sv);
```

The inverse of `to_statevector()`: replaces the chain with the factorisation of
`sv` by sequential SVD, one truncated split per bond, keeping this state's qubit
count, bond cap and cutoff. It is the route every dense fallback takes, and the
route a supplied initial state is factorised through.

The bond cap still applies, so a state needing more bonds than the cap holds is
truncated rather than refused: that is what running at this cap means, and the
discarded weight is added to `truncation_error()`. Throws
`std::invalid_argument` when `sv` does not cover the same number of qubits as the
chain.

Weight the Gram route's validity floor rejected is reported separately from
`truncation_error()`, as `floor_rejected_weight()`, because it is not
truncation. Forming the Gram matrix
squares the condition number, so a singular value that is exactly zero in the
input returns at the scale of the square root of machine epsilon and carries
weight that was never in the matrix. Counting that as truncation error would
report a bond which discarded nothing as having lost something. The floor is
`sqrt(c·n·eps)·σ_max` with `n` the Gram dimension and `c` the same slack the
verification grants: the top of the band the eigensolver's own error bound
lets a null direction return in, so noise cannot pass as a direction, and a
real singular value below it is one the route could not have resolved.

The counters accumulate over the state's lifetime and are not reset by gate
application. Reconstruction from a statevector runs the ladder like any other
split, through the kernel `svd_method` selects, and advances every counter.

**Complexity**:
- **Space**: $O(n \cdot \chi^2)$ where $\chi$ = max bond dimension (typically 16–256)
- **Time per gate**: $O(\chi^2)$ for single-qubit, $O(\chi^3)$ for two-qubit (the
  contraction and the SVD of a $2\chi \times 2\chi$ block), plus $O(\chi^3)$ per
  step whenever the orthogonality centre moves first
- **Accuracy**: Controlled by $\chi$ and `cutoff`; larger $\chi$ = more accurate

**Truncation rule**. `cutoff` is the maximum fraction of total weight
$\sum \sigma^2$ that a bond truncation may discard. It is not a magnitude
threshold: a bare singular value is never compared against it. Truncation keeps
the largest $k \le \chi$ singular values such that the discarded weight stays
within `cutoff` of the total.

A magnitude threshold asks a question whose answer depends on the scale of the
matrix and on how the target rounded its way there, so the same state could
carry a different bond dimension on a different CPU. A weight fraction is
scale-free and bounds the physical error directly. Note it is a ceiling rather
than a quota: where a spectrum has a clean gap between real content and
numerical noise, nothing extra is discarded and the retained bond dimension is
unchanged.

The default, `MPS_DEFAULT_CUTOFF` (`1e-16`, in `lindblad/types.hpp`), sits at
the relative accuracy to which a double-precision sum of weights is known. At
that value a split removes exact rank deficiency and weight its block's own
total cannot distinguish from rounding, and does not compress; a caller who
wants compression raises it. The qudit layer's `svd_cutoff` means the same thing
and has the same default.

### Canonical Form

A bond split truncates on the singular values of a two-site block, and those
are the state's Schmidt coefficients only when the chain is in mixed canonical
form centred on the block: every site to its left left-orthonormal, every site
to its right right-orthonormal. In any other gauge the block's spectrum is
weighted by the rest of the chain, so a cap or a cutoff applied to it keeps and
drops the wrong directions, and the weight it reports discarding is not a
fraction of the state.

**The open span**. Each chain keeps two site indices, reported by `open_span()`
as `{lo, hi}`:

- every site left of `lo` is left-orthonormal: reshaped as a
  $(2\chi_L) \times \chi_R$ matrix $A$, $A^\dagger A = I$
- every site right of `hi` is right-orthonormal: reshaped as a
  $\chi_L \times (2\chi_R)$ matrix $A$, $A A^\dagger = I$
- the sites in between carry no guarantee

`{c, c}` is mixed canonical form centred on `c`: the whole norm sits in site
`c`, and a two-site block at `c` holds the Schmidt coefficients at its bond. A
new chain is `|0...0>` with its centre on site 0.

**Moving the centre**. The centre moves by a thin QR of a site (rightward, the
$R$ factor multiplied into the next site) or a thin LQ (leftward, the $L$ factor
multiplied into the previous one). Both are exact and never truncate, so moving
the centre changes the gauge and never the state, and neither touches the
truncation total, the fidelity figures or the SVD counters. A bond wider than
the rank its neighbouring site can carry is trimmed to that rank on the way,
which is also lossless.

```cpp
void canonicalize(int site);           // centre to `site`; open_span() becomes {site, site}
std::pair<int, int> open_span() const;
```

**When a split moves the centre first**. `canonical_form`, shared with the
qudit MPS and declared in `lindblad/types.hpp`:

- `CanonicalForm::Always` (default): before every split, so every truncation
  acts on Schmidt coefficients and the bond dimensions do not depend on whether
  the cap is binding
- `CanonicalForm::Auto`: only before a split that can discard real weight,
  meaning one whose block's rank bound $\min(2\chi_L, 2\chi_R)$ exceeds
  `max_bond_dim` so the cap can bind, and every split once `cutoff` is above
  `MPS_DEFAULT_CUTOFF`. Any other split runs where the chain stands and skips
  the QR steps. Its truncation is still sound, since it drops only exact rank
  deficiency (the same in every gauge) and weight below the resolution of its
  block's total, but it keeps too much: a direction that is rounding noise in
  the state can carry real weight in a block weighted by an uncentred
  environment, and survives the cutoff there

That surplus is why `Always` is the default. The bonds `Auto` keeps grow wider
than the state needs, and the extra rank costs more in every later split than
the QR steps saved. On a 16-qubit brickwork at a cap that truncates nothing,
`Auto` ends with a middle bond of 256 where `Always` ends with 143, at the same
fidelity to $10^{-14}$, and takes 1.9x as long. Where the cap binds (the
24-qubit brickwork at $\chi$ = 8 to 64) the two agree to within a few percent.
`Auto` stays selectable for a workload that measures otherwise.

Either way, a split sends its singular values toward the block the chain
touches next, so a SWAP chain finds the centre already in place at each step.

Measurement, reset and terminal sampling move the centre under both policies,
because each reads one site's marginals and that is a local read only at the
centre. Reads that do not move it (`probabilities_single`, `norm_sq`, the bond
spectrum behind the entropy observer) contract only the open span, so they are
cheapest with the centre where they look.

**Replacing the chain**.

```cpp
const std::vector<MPSTensor>& tensors() const;
void set_tensors(std::vector<MPSTensor> sites);
```

`set_tensors` validates before replacing anything: the count must equal
`n_qubits`, every bond must be at least 1 with the two outer ones exactly 1,
neighbouring bonds must agree, each data array must hold
`bond_left * 2 * bond_right` entries, and every entry must be finite. It throws
`std::invalid_argument` naming the first violation and leaves the state as it
was. Nothing is assumed about a chain built by hand, so the open span becomes
the whole chain, and the first operation that needs the centre pays the QR
steps to find it. The fidelity figures reset to exact, since the chain handed in
is what later truncation is measured against; assigning a whole `MPSState`
instead keeps them. The SVD counters and `truncation_error()` are untouched.

### Fidelity Figures

```cpp
std::optional<double> fidelity_estimate() const;
std::optional<double> fidelity_lower_bound() const;
```

Both answer how close the chain is to the state an untruncated run would hold,
as a fidelity $|\langle \text{exact} | \text{chain} \rangle|^2$ between the two
normalised states. Each split $k$ contributes its discarded fraction
$\varepsilon_k = \text{discarded}_k / (\text{kept}_k + \text{discarded}_k)$,
which for a split in canonical gauge is the fraction of the state it removed.

- `fidelity_estimate()` is $\prod_k (1 - \varepsilon_k)$, the standard figure
  (Zhou, Stoudenmire and Waintal, Phys. Rev. X 10, 041038, 2020), accurate in
  practice. It is **not a bound**: the true fidelity can lie on either side of
  it. Two rotations by $a$, each followed by a truncation back onto $|0\rangle$,
  keep $\cos^2 a$ per split, so the product is $\cos^4 a \approx 1 - 2a^2$,
  while the true fidelity is $\cos^2 2a \approx 1 - 4a^2$
- `fidelity_lower_bound()` is $\max(0, 1 - \Delta^2/2)^2$ with
  $\Delta = \sum_k \delta_k$ and $\delta_k = \sqrt{2 - 2\sqrt{1 - \varepsilon_k}}$,
  the exact distance a split moves the normalised state. Gates between splits
  are unitary and move both states alike, so by the triangle inequality the
  exact and truncated states are within $\Delta$. When every split is canonical
  the true fidelity cannot fall below the bound (the last rule below covers
  `CanonicalForm::Auto`). On the example above it is $1 - 4a^2$ to leading order

Rules:

- Both read 1 on a new chain and after `set_tensors()`. Both describe normalised
  states, so `normalize()` changes neither
- Both are **empty** once a measurement or reset has collapsed the chain
  (`measure_qubit`, `measure_sequential`, or a MEASURE or RESET in a run), and
  stay empty: projection renormalises the exact and the truncated state by
  different factors, so neither figure describes the pair afterwards. On the
  per-shot path of `MPSSimulator::run` the returned chain has collapsed, so both
  are empty there
- `rebuild_from_statevector` counts its splits like any other; they are
  canonical by construction
- `absorb_profile` does not fold them: they describe the returned tensors' own
  history, where `truncation_error()` describes the run's splits
- A split `CanonicalForm::Auto` runs in place contributes its fraction of the
  block rather than of the state; by the rule above that fraction is at most
  `MPS_DEFAULT_CUTOFF`. Under `CanonicalForm::Always` every split is canonical
  and the bound is rigorous throughout

### Gate Application via SVD

**Single-qubit gate on site $i$**: the 2x2 matrix is contracted into the
physical index of $M_i$. No SVD and no bond change, and the open span is left as
it is, because a unitary on the physical index preserves both
orthonormalities.

**Two-qubit gate on sites $(i, i+1)$** (adjacent):

1. When `canonical_form` calls for it (see [Canonical Form](#canonical-form)),
   move the orthogonality centre onto the pair first
2. Contract tensors: `M_i @ bond @ M_{i+1}` → rank-4 tensor. The contraction is
   a single zero-copy Eigen GEMM: the MPSTensor data is already contiguous
   row-major in the needed $(\text{bond}_L \cdot 2) \times \chi$ and
   $\chi \times (2 \cdot \text{bond}_R)$ shapes.
3. Apply $U$ gate to physical indices
4. Reshape to matrix and perform SVD (backend per `svd_method`): $U = L \cdot S \cdot R^\dagger$
5. Truncate singular values: keep only $\chi$ largest with sum $\geq (1 - \text{cutoff})$
6. Absorb $S$ toward the block the chain touches next and keep the isometry on
   the other side: $M_i = L$, $M_{i+1} = S R^\dagger$ by default, and the mirror
   image inside a SWAP chain moving left. The site holding $S$ joins the open
   span, and the isometry leaves it when nothing beyond it was in the span
7. Update bond dimension, the truncation total and the fidelity figures

**Non-adjacent gates**: Apply SWAPs to move gates adjacent, then apply gate, then SWAP back. Each split sends its singular values toward the next block, so when splits move the centre it is already in place at each step.

**Arbitrary `UNITARY` gates** (R.1.10.7 — direct tensor dispatch for 1q and 2q):

- **1-qubit UNITARY**: contracts the 2x2 matrix directly into one site tensor
  via `MPSState::apply_single_qubit_gate`. No SVD, no full statevector
  conversion. Memory cost is bounded by the bond dimension and independent
  of `n_qubits`.
- **2-qubit UNITARY**: contracts the 4x4 matrix into the two-site tensor via
  `apply_two_qubit_gate`, followed by a truncated SVD bounded by `max_bond_dim`
  and `cutoff`. Non-adjacent qubit pairs are handled by the existing swap
  network. `apply_two_qubit_gate` takes its matrix in the project convention,
  as `apply_unitary` does: bit 0 of the row and column index is the first
  qubit argument. The circuit's matrix is therefore passed as it stands, and
  the primitive bit-reverses it once into the MSB-first order its two-site
  contraction reads.
- **3+ qubit UNITARY**: falls back to the statevector path —
  `to_statevector()` → `gates::apply_unitary` → `rebuild_from_statevector`. The
  rebuild is a sequential SVD, one truncated split per bond, and the weight each
  split discards is added to `truncation_error()` rather than replacing it, so
  the figure covers everything the chain has lost rather than only the last
  thing that lost it. The fallback
  is bounded by `MPS_SV_MAX_QUBITS` (= 25); beyond that the simulator throws
  with a clear error naming the offending UNITARY and qubit count, rather
  than the generic "Too many qubits for full statevector conversion" surfaced
  from inside `to_statevector()`. Decompose >2q unitaries into 1q/2q factors
  for wider registers.

### Measurement

Every measurement reads a qubit at the orthogonality centre, where its raw
marginals $\langle\psi|P_k|\psi\rangle$ are the squared norms of its site's two
physical slices.

**One qubit**.

```cpp
int MPSState::measure_qubit(int qubit, std::mt19937_64& rng);
```

Moves the centre to `qubit`, draws one uniform from `rng`, and collapses the
chain onto the outcome (returned as 0 or 1): the other slice is zeroed and the
site divided by the square root of the outcome's marginal, which leaves the
state with unit norm and `open_span()` at `{qubit, qubit}`. A MEASURE in a run
and the first half of a RESET are this call. Throws `std::out_of_range` for a
qubit outside the register. The fidelity figures become empty.

**Sequential measurement** (`measure_sequential(rng)`), physically realistic:

1. Move the centre to qubit 0
2. Measure it as above, collapsing the chain onto the outcome
3. Step the centre one site right by a QR and repeat for qubit 1, 2, ...

A full bitstring costs $O(N \cdot \chi^3)$ with no environments, draws one
uniform per qubit, and is returned with qubit 0 as the rightmost character.

**Terminal sampling**. With the centre on qubit 0, every other site is
right-orthonormal, so the environment right of any qubit is the identity and a
shot needs only a row vector $v$ on the bond left of the current qubit: $w_p =
v A_q[p]$, the outcome drawn from $|w_0|^2$ and $|w_1|^2$, then $v \leftarrow
w_{\text{out}} / |w_{\text{out}}|$. When the run samples this way (see
[Choosing the Sampling Path](#choosing-the-sampling-path)), it moves the centre
of the returned chain to qubit 0 once (a gauge change: the state is unchanged)
and samples every shot read-only from it at $O(N \cdot \chi^2)$ per shot, with
nothing precomputed and nothing copied.

### Choosing the Sampling Path

Terminal sampling has two paths that draw from the same distribution, the
chain's own, so `MPSSimulator::run` picks whichever is cheaper for the chain it
holds and the shots requested. Both costs follow from the bond profile, so the
choice is made before either runs:

- **Dense**: contract the chain into $2^n$ amplitudes once
  (`to_statevector()`), then draw every shot from them at almost no cost. The
  contraction costs about $\sum_q 2^{q+1} \chi_L(q) \chi_R(q)$
  multiply-accumulates.
- **MPS sampler**: the vector walk above, about $\sum_q \chi_L(q) \chi_R(q)$
  units of work per shot.

The dense path is chosen when its contraction costs no more than
`shots` walks, with one unit of walk weighted at 1.8 multiply-accumulates (the
measured ratio: 2.1 ns against 1.2 ns, medians over brickwork circuits at
$n$ = 14 to 24 and $\chi$ = 8 to 64). Small registers at high shot counts go
dense; wide registers and low shot counts go through the sampler. On an
8-layer brickwork at $\chi$ = 64 the break-even is about 400 shots at 16
qubits and about 3900 at 20.

The dense path also allocates the $2^n$ amplitudes, where the sampler's memory
follows the bond dimension, so it is taken only while those amplitudes (16
bytes each) fit in one last-level cache instance, as reported by
`hw::llc_bytes()` (4 MiB when detection reports none). On a part with 32 MiB of
L3 per instance that allows $n \le 21$. Wider registers always use the
sampler, whatever the shot count.

`MPS_SV_MAX_QUBITS = 25` is the separate hard limit on `to_statevector()`: it
throws above 25 qubits (about 512 MB at that size), and every dense fallback in
this backend stops there.

### Measurement Normalization

A collapse happens at the centre, where the site carries the state's whole norm.
The squared norm of the kept slice is therefore the outcome's raw marginal, and
dividing that one site by its square root renormalises the whole state. Every
later qubit's marginals are then conditional on the outcomes before it, because
the chain has already collapsed onto them.

### Conversion to Exact Statevector

For small systems, convert back to statevector via boundary contraction:

```cpp
Statevector MPSState::to_statevector() const {
    // Contract all tensors: M_0 @ M_1 @ ... @ M_{n-1}
    // O(n * chi^3) time, O(2^n) space
    // Throws if n_qubits > MPS_SV_MAX_QUBITS (25)
}
```

### Workflow

```cpp
#include "lindblad/simulators/mps_sim.hpp"

#include <cstdio>

using namespace lindblad;

// `circuit` is any QuantumCircuit ending in terminal measurements.
MPSSimulator sim;  // canonical_form = Always, svd_method = BDC
auto result = sim.run(circuit, /*max_bond_dim=*/64, /*shots=*/1024, /*seed=*/42);

// result.counts holds the bitstrings (qubit 0 rightmost). The chain behind
// them is result.final_state; what the bond cap cost it, as fidelities against
// the untruncated state, is empty only when a mid-circuit measurement collapsed
// the chain.
const MPSState& chain = result.final_state;
if (const auto bound = chain.fidelity_lower_bound()) {
    std::printf("fidelity >= %.6f (estimate %.6f), max bond %d\n", *bound,
                *chain.fidelity_estimate(), chain.current_max_bond_dim());
}
```

`MPSSimulator` has only its default constructor: the chain is built inside
`run()`, and the register width comes from the circuit. Note the argument
order, `run(circuit, max_bond_dim, shots, seed)`, which differs from the other
simulators' `run(circuit, shots, seed)`.

### Use Cases

- Approximate simulation of large-scale circuits (20–40+ qubits)
- Research into entanglement dynamics and scaling
- Benchmarking approximate methods against classical baselines
- Analysis of weakly-entangled states (shallow circuits, product states)

### Limitations

- Approximate: truncation error grows with depth and entanglement
- Non-trivial configuration (choosing $\chi$, `cutoff` for target accuracy)
- Slower than statevector for small systems (overhead of SVD)
- Two-qubit gates require adjacency (need SWAPs for non-local gates)
- **Known accuracy gap at scale (targeted R.1.13.2)**: some high-entanglement
  circuits lose accuracy even when the bond dimension is theoretically exact for
  the register size. Concretely, Shor's 13-qubit period-finding circuit for
  $N = 15$ does not recover the order on the MPS backend at `max_bond_dim = 64`
  (which is the *exact* bond dimension for any 13-qubit state, since the maximum
  Schmidt rank across any cut is $2^6 = 64$), whereas the statevector backend
  does. The wide-`PERMUTATION` oracle fallback is verified exact at 4 and 8
  qubits, so this is a scale-specific defect under investigation, not a
  truncation limit. Until it is fixed, use the statevector or density-matrix
  backend for exact results on deep, highly-entangling circuits.

### Parameters

- **max_bond_dim** ($\chi$): Controls truncation; typical range 16–512
  - Larger $\chi$ = more accurate but slower and more memory
  - $\chi = 2^n$ recovers exact simulation
  - Must be at least 1. `MPSState` and `MPSSimulator::run` reject a smaller
    value with `std::invalid_argument`: a retained rank of zero keeps no
    singular values at all, and left unchecked it reaches the truncation step
    indistinguishable from a numerically corrupt spectrum. Note that
    `MPSSimulator::run(circuit, max_bond_dim, shots, seed)` takes the bond
    dimension where `StatevectorSimulator::run(circuit, shots, seed)` takes the
    shot count
- **cutoff**: the fraction of total singular weight ($\sum \sigma^2$) a bond
  split may discard; default `MPS_DEFAULT_CUTOFF` (`1e-16`) on `MPSState`
  - A weight fraction, not a magnitude threshold: no bare singular value is
    compared against it, so the same state carries the same bond dimension on
    every CPU
  - The budget is a ceiling: on a spectrum with nothing between the noise and
    the budget, nothing extra is discarded
- **svd_method** (`SVDMethod`, default `BDC`): the kernel every bond split asks
  for first; `Jacobi`, `EigenBDC` and `EigenJacobi` are selectable (see the
  kernel table under `MPSState`). Set on `MPSSimulator` for `run()` or on an
  `MPSState` driven directly
- **svd_rescue** (`bool`, default `true`): whether a factorisation the verify
  rung rejects may descend the rescue ladder, one warning per rung (identical
  warnings collapse into a repeat count); `false`
  turns the first rejection into a `std::runtime_error`
- **canonical_form** (`CanonicalForm`, default `Always`): which bond splits
  first move the orthogonality centre onto their block; `Auto` moves it only
  where truncation can bind (see [Canonical Form](#canonical-form)). Set on `MPSSimulator` for
  `run()` or on an `MPSState` driven directly

## Simulator Selection Guide

| Circuit Property | Simulator | Notes |
|---|---|---|
| Small, pure, arbitrary gates | **Statevector** | Exact, fast, limited to ~25 qubits |
| Clifford-only, any size | **Clifford** | Polynomial scaling, exact |
| Noisy execution | **DensityMatrix** | Integrates Kraus operators, ~10 qubits max |
| Large-scale, weakly-entangled | **MPS** | Approximate, ~30+ qubits, configure $\chi$ carefully |
| Default/Hybrid | **Auto-dispatch** | Detect circuit properties, pick best simulator |

## Integration with Primitives

- **Estimator** uses `StatevectorSimulator` for ideal zero-shot circuits; automatically routes to `DensityMatrixSimulator` when `options.noise_model` is non-ideal or `options.shots > 0`. See [Estimator API](estimator.md) for the routing rules.
- **Sampler** routes to appropriate simulator based on `NoiseModel` (density matrix if noise present)
- **MAQAOA** supports both statevector (direct evolution) and density matrix (noisy path)

## See Also

- [Gates API](gates.md) — Gate implementation details and optimization
- [Statevector API](statevector.md) — Aligned memory layout and measurement
- [Operators API](operators.md) — Pauli string and sparse operator representations
- [Noise API](noise.md) — Kraus channels and NoiseModel construction
- [Estimator API](estimator.md) — Expectation value computation using simulators
- [Sampler API](sampler.md) — Bitstring sampling via simulator backends
