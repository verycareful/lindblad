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
5. **Measurement handling**: If any `GT::MEASURE` instruction is present and `shots > 0`, the circuit is re-executed from `|0...0⟩` once per shot so that each collapse is independent. Outcomes are written into the classical register (`n_clbits`-wide). If no MEASURE instructions are present, the state is simulated once and `sample_counts` samples the final distribution.
6. **Result collection**: Extract final state and sampled bitstrings (if shots > 0)

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
};
```

- **max_parallel_threads**: Cap on OpenMP thread count (0 = system default)
- **max_memory_mb**: Memory budget (0 = no limit); used for preemptive error checking
- **precision**: Reserved for future 32-bit float variants
- **zero_threshold**, **threshold**: Legacy fields; may be removed in future versions

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

**Implementation**:

```cpp
void DensityMatrix::apply_kraus(const std::vector<std::vector<Complex128>>& kraus_ops,
                                 const std::vector<int>& qubits) {
    std::vector<Complex128> result_data(dim * dim, Complex128(0.0, 0.0));
    
    for (const auto& K : kraus_ops) {
        DensityMatrix temp = *this;
        temp.apply_gate(K, qubits);   // temp = K * rho * K†
        for (size_t i = 0; i < dim * dim; ++i)
            result_data[i] += temp.data[i];
    }
    
    data = std::move(result_data);
}
```

**Complexity**: $O(m \cdot 4^n \cdot 2^k)$ where $m$ = number of Kraus operators

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
    double cutoff;       // SVD truncation threshold
};
```

**Complexity**:
- **Space**: $O(n \cdot \chi^2)$ where $\chi$ = max bond dimension (typically 16–256)
- **Time per gate**: $O(\chi^4)$ for single-qubit, $O(\chi^6)$ for two-qubit
- **Accuracy**: Controlled by $\chi$ and `cutoff`; larger $\chi$ = more accurate

### Gate Application via SVD

**Single-qubit gate on site $i$**:
1. Apply gate matrix to physical index of $M_i$
2. Reshape to matrix form `(bond_left, bond_right × 2)`
3. Absorb into a neighboring bond (left or right)

**Two-qubit gate on sites $(i, i+1)$** (adjacent):
1. Contract tensors: `M_i @ bond @ M_{i+1}` → rank-4 tensor
2. Apply $U$ gate to physical indices
3. Reshape to matrix and perform SVD: $U = L \cdot S \cdot R^\dagger$
4. Truncate singular values: keep only $\chi$ largest with sum $\geq (1 - \text{cutoff})$
5. Absorb $L \cdot S$ into $M_i$; absorb $R^\dagger$ into $M_{i+1}$
6. Update bond dimension and track truncation error

**Non-adjacent gates**: Apply SWAPs to move gates adjacent, then apply gate, then SWAP back

**Arbitrary `UNITARY` gates**: Handled via statevector fallback for all qubit counts. The MPS is converted to a full statevector via `to_statevector()`, `gates::apply_unitary` applies the matrix, and `mps_from_sv` reconstructs the MPS via sequential SVD. The bit-reversal between the MPS site-index convention (qubit 0 = MSB) and the statevector convention (qubit 0 = LSB) is handled automatically.

### Measurement

**Sequential measurement** (physically realistic):
1. Measure qubit 0 by contracting boundary from left
2. Condition state on outcome; collapse corresponding MPS branch
3. Project boundary to next qubit and repeat for qubit 1, 2, ...
4. $O(N \cdot \chi^3)$ time for full bitstring

**Complexity**: $O(n \cdot \chi^3)$ per shot (not $O(2^n)$ like statevector)

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

### Parameters

- **max_bond_dim** ($\chi$): Controls truncation; typical range 16–512
  - Larger $\chi$ = more accurate but slower and more memory
  - $\chi = 2^n$ recovers exact simulation
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
