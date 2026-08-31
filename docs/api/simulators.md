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

The collapse renormalisation in the MPS simulator divides by the
environment-contracted outcome marginal (valid for non-canonical tensors);
sampled MPS bitstrings use the project key convention (qubit 0 rightmost) at
every register width.

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
- `check_normalized(validation)` applies a validation policy, with `Fix`
  renormalizing in place

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

**SparsePauliOp expectation** (specialized for sparse Pauli strings):

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

Efficient exact simulation of Clifford circuits (gates from $\{H, S, CNOT, X, Y, Z\}$) using stabilizer tableau representation.

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
- Only supports Clifford gate set (H, S, CNOT, X, Y, Z, T/T† forbidden)
- No support for parameterized gates (RX, RY, etc.)
- Non-Clifford gates cause error or fallback

**Is-Clifford Check**:
```cpp
bool CliffordSimulator::is_clifford(const QuantumCircuit& circuit) {
    for (const auto& inst : circuit.instructions) {
        if (!is_clifford_gate(inst.type)) return false;
    }
    return true;
}
```

## MPSSimulator

Approximate simulation of arbitrary circuits using Matrix Product State (MPS) representation, enabling simulation of larger systems at the cost of controlled truncation error.

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
    std::vector<MPSTensor> tensors;
    int max_bond_dim;    // chi parameter
    double cutoff;       // max fraction of weight truncation may discard
    SVDMethod svd_method = SVDMethod::BDC;     // SVD backend
};
```

**Norm and normalization**:

- `norm_sq()` contracts the transfer matrix along the chain, $O(n \cdot \chi^3)$.
  There is no flat amplitude array to sweep, so this is the only way to read the
  norm without materialising the state
- `normalize()` rescales the first site tensor, which is exact because the norm
  is multilinear in the tensors. It throws when there is no norm to divide out,
  a zero or non-finite state, rather than returning the state unchanged
- `is_normalized(atol)` answers without repairing or throwing
- `check_normalized(validation)` applies a policy, with `Fix` renormalizing.
  Under `Ignore` the contraction does not run at all, which matters more here
  than on the dense classes because this measurement is the most expensive of
  any state type in the library

**SVD backend**: `svd_method` (declared in `lindblad/types.hpp`, shared with
the qudit MPS) selects the truncation SVD. `SVDMethod::BDC` is the **default**.
It is divide-and-conquer where Jacobi is `O(n^3)` per sweep, so the gap widens
with the block: measured through the truncation ladder, BDC costs roughly a
fifth of Jacobi at 32x32 and a fiftieth at 128x128, which is a two-site theta at
bond dimension 64.

`SVDMethod::Jacobi` remains selectable and emits a one-time note to the warning
channel that it is the slower algorithm. Below Eigen's divide-and-conquer
threshold the choice is nominal: BDCSVD delegates to the Jacobi kernel for
blocks smaller than 16x16, so a theta at bond dimension 4 runs identical code
either way.

Both backends are held to the same verification described below, and both are
accepted on the first attempt on decaying and exactly degenerate spectra alike.
The two do not agree bit for bit, so a state truncated under one backend differs
in its last digits from the same state truncated under the other.

**Verified truncation (R.1.16.0)**: the SVD output is no longer trusted
blindly. Eigen's SVDs were found to return corrupt factorisations on
degenerate rank-deficient inputs (the class of two-site tensors Shor-style
circuits produce), in failure shapes ranging from NaN singular vectors to a
wrong-but-finite kept vector. Every truncation therefore now: selects the
kept singular values by bit-level-finite comparison (immune to ordering
corruption), verifies the kept factorisation against the Frobenius identity
`‖M − U·S·V†‖²_F = Σ(discarded σ²)`, recomputes via a Gram-matrix
eigendecomposition if verification fails, and throws `std::runtime_error`
rather than continue if both routes fail — an MPS run can no longer produce
a silently corrupted state from a bad SVD.

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

The factorisation itself is performed in a translation unit compiled under
strict IEEE floating-point, and so is the reconstruction residual that decides
whether a factorisation is accepted. The residual subtracts two nearly identical
matrices, and one computed too small would admit exactly the factorisations the
check exists to reject.

**Ladder observability**. Which route a bond split took is otherwise invisible
to the caller, since a rescued split and a clean one both yield valid tensors.
Two counters on `MPSState` report it:

- `svd_call_count()`: bond splits performed, one per call into the truncation
  routine. This is the denominator; a fallback count means nothing without it.
- `gram_fallback_count()`: splits where the SVD backend's factorisation failed
  verification and the Gram route was used instead. Only successful rescues are
  counted, because a Gram route that also fails verification throws.
- `max_verify_residual_excess()`: the worst factorisation error verification
  accepted, as a fraction of $\|M\|_F^2$, maximised over splits. The Frobenius
  identity holds with equality for a true truncated SVD, so this reports the
  excess over that ideal rather than the raw residual, and a healthy run sits
  near the square of machine epsilon. It says how close a run came to being
  rescued, and how much error the accepted route let through when it was not.

A run with `gram_fallback_count() == 0` never distrusted its SVD backend. A
nonzero count is not an error: it is the containment working.

Weight the rescue's validity floor rejected is reported separately from
`truncation_error()`, because it is not truncation. Forming the Gram matrix
squares the condition number, so a singular value that is exactly zero in the
input returns at the scale of the square root of machine epsilon and carries
weight that was never in the matrix. Counting that as truncation error would
report a bond which discarded nothing as having lost something.

Both counters accumulate over the state's lifetime and are not reset by gate
application. Reconstruction from a statevector runs the ladder like any other
split and advances both counters.

**Complexity**:
- **Space**: $O(n \cdot \chi^2)$ where $\chi$ = max bond dimension (typically 16–256)
- **Time per gate**: $O(\chi^4)$ for single-qubit, $O(\chi^6)$ for two-qubit
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

The default `1e-16` bounds truncation error at the order of the reconstruction
error an SVD already carries, so nothing the factorisation actually resolved is
thrown away. The qudit layer's `svd_cutoff` means the same thing.

### Gate Application via SVD

**Single-qubit gate on site $i$**:
1. Apply gate matrix to physical index of $M_i$
2. Reshape to matrix form `(bond_left, bond_right × 2)`
3. Absorb into a neighboring bond (left or right)

**Two-qubit gate on sites $(i, i+1)$** (adjacent):
1. Contract tensors: `M_i @ bond @ M_{i+1}` → rank-4 tensor. In R.1.13 (audit
   F-6) this contraction is a single zero-copy Eigen GEMM: the MPSTensor data is
   already contiguous row-major in the needed $(\text{bond}_L \cdot 2) \times \chi$
   and $\chi \times (2 \cdot \text{bond}_R)$ shapes.
2. Apply $U$ gate to physical indices
3. Reshape to matrix and perform SVD (backend per `svd_method`): $U = L \cdot S \cdot R^\dagger$
4. Truncate singular values: keep only $\chi$ largest with sum $\geq (1 - \text{cutoff})$
5. Absorb $L \cdot S$ into $M_i$; absorb $R^\dagger$ into $M_{i+1}$
6. Update bond dimension and track truncation error

**Non-adjacent gates**: Apply SWAPs to move gates adjacent, then apply gate, then SWAP back

**Arbitrary `UNITARY` gates** (R.1.10.7 — direct tensor dispatch for 1q and 2q):

- **1-qubit UNITARY**: contracts the 2x2 matrix directly into one site tensor
  via `MPSState::apply_single_qubit_gate`. No SVD, no full statevector
  conversion. Memory cost is bounded by the bond dimension and independent
  of `n_qubits`.
- **2-qubit UNITARY**: contracts the 4x4 matrix into the two-site tensor via
  `apply_two_qubit_gate`, followed by a truncated SVD bounded by `max_bond_dim`
  and `cutoff`. Non-adjacent qubit pairs are handled by the existing swap
  network. The dispatch swaps bits 0 and 1 of the matrix row/column indices to
  bridge `apply_unitary`'s LSB-at-first-arg convention with
  `apply_two_qubit_gate`'s MSB-at-first-arg convention.
- **3+ qubit UNITARY**: falls back to the statevector path —
  `to_statevector()` → `gates::apply_unitary` → `mps_from_sv`. The fallback
  is bounded by `MPS_SV_MAX_QUBITS` (= 25); beyond that the simulator throws
  with a clear error naming the offending UNITARY and qubit count, rather
  than the generic "Too many qubits for full statevector conversion" surfaced
  from inside `to_statevector()`. Decompose >2q unitaries into 1q/2q factors
  for wider registers.

### Measurement

**Sequential measurement** (physically realistic):
1. Measure qubit 0 by contracting boundary from left
2. Condition state on outcome; collapse corresponding MPS branch
3. Project boundary to next qubit and repeat for qubit 1, 2, ...
4. $O(N \cdot \chi^3)$ time for full bitstring

**Complexity**: $O(n \cdot \chi^3)$ per shot (not $O(2^n)$ like statevector)

**Sampling (R.1.13, audit F-3/F-4)**: the right environments are invariant
across shots, so the terminal-sampling path computes them ONCE and samples each
shot read-only (the left environment is carried incrementally and each site's
outcome slice is read from the unmodified tensors, scaled by $1/p$). There is no
per-shot MPS copy and no environment rebuild, and every contraction is the
two-stage $O(\chi^3)$ form (the previous per-shot path rebuilt environments and
deep-copied the whole MPS, with $O(\chi^4)$ contractions).

### Statevector Crossover

`MPSSimulator::run` uses two named thresholds:

- `MPS_SV_CROSSOVER = 18`: circuits with `n_qubits ≤ 18` sample via full statevector conversion (faster)
- `MPS_SV_MAX_QUBITS = 25`: `to_statevector()` throws for `n_qubits > 25` (memory guard, ~512 MB at that size)

### Measurement Normalization

After `measure_sequential` projects a qubit, the remaining MPS tensors are renormalized by dividing by `sqrt(prob)` where `prob` is the boundary-contraction probability of the observed outcome — **not** the local Frobenius norm (which differs for non-canonical MPS and would not give correct conditional probabilities).

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
MPSSimulator sim(32, 1024);  // n_qubits=32, max_bond_dim=1024

// MPS is allocated internally
auto result = sim.run(large_circuit, 100);  // 100 shots
```

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
- **cutoff**: SVD truncation threshold; typical 1e-12 to 1e-8
  - Discards singular values $< \text{cutoff}$
  - Minimal impact on accuracy for cutoff $\leq 1e-10$

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
