# Gates API Deep Dive

The Gates API provides a comprehensive suite of quantum gate operations for state manipulation in lindblad. All gates operate directly on `Statevector` objects via in-place modifications, leveraging SIMD vectorization and OpenMP parallelization for high performance across single, two, and multi-qubit operations.

## Design Philosophy

Gates in lindblad follow a **functional, in-place modification** paradigm:
- Each gate is a `void` function taking a `Statevector&` reference and qubit indices
- Modifications happen directly on the aligned `real_parts` and `imag_parts` arrays
- Two core patterns underpin all implementations:
  1. **Amplitude pairing**: Group amplitudes by target qubit (step = 1 << qubit) and apply 2×2 unitaries
  2. **Bit manipulation**: For multi-qubit gates, check bit patterns via `(index >> qubit) & 1`

## Single-Qubit Gates

All single-qubit gates iterate over amplitude pairs `(index, index + step)` where `step = 1 << qubit`. The outer loop spans `[0, dim, 2*step)` with inner SIMD loop over `[i, i+step)`, enabling cache-friendly access and SIMD vectorization.

### Pauli Gates

**X Gate** (`apply_x`): Implements $X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$
- **Operation**: Swaps amplitudes between qubit=0 and qubit=1 states
- **Code**: Two `std::swap` operations per amplitude pair (no arithmetic)
- **Complexity**: $O(2^n)$ iterations, one swap per pair

**Y Gate** (`apply_y`): Implements $Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}$
- **Operation**: Maps $|\psi_0\rangle \to -i|\psi_1\rangle$ and $|\psi_1\rangle \to i|\psi_0\rangle$
- **Code**: For each pair (r0, i0, r1, i1):
  - `new_0 = (i1, -r1)` (multiply by $-i$)
  - `new_1 = (-i0, r0)` (multiply by $i$)
- **Complexity**: Two complex multiplications per pair

**Z Gate** (`apply_z`): Implements $Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$
- **Operation**: Negates all amplitudes where qubit=1
- **Code**: Single loop over `[i+step, i+2*step)` range; negate both real and imag parts
- **Complexity**: $O(2^n)$ negations

### Hadamard Gate

(`apply_h`): Implements $H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$
- **Operation**: For each amplitude pair:
  - $\text{new}_0 = \frac{1}{\sqrt{2}}(\text{old}_0 + \text{old}_1)$
  - $\text{new}_1 = \frac{1}{\sqrt{2}}(\text{old}_0 - \text{old}_1)$
- **Code**: Multiply by `INV_SQRT2` constant; compile-time precomputed as `1.0 / sqrt(2)`
- **Complexity**: Four multiplications and two additions per pair

### Phase Gates

**S Gate** (`apply_s`): Implements $S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}$ (phase gate with $\lambda = \pi/2$)
- **Operation**: Multiplies qubit=1 amplitudes by $i$
- **Code**: Only modify `[i+step, i+2*step)` range; complex multiplication $(r + im \cdot i) \cdot i = -im + r \cdot i$
- **Transformation**: `(r, im) → (-im, r)`

**S† Gate** (`apply_sdg`): Adjoint of S; multiplies by $-i$
- **Transformation**: `(r, im) → (im, -r)`

**T Gate** (`apply_t`): Phase gate with $\lambda = \pi/4$; phase = $e^{i\pi/4}$
- **Code**: Uses `apply_diagonal_phase(sv, q, 1.0, 0.0, cos(π/4), sin(π/4))`

**T† Gate** (`apply_tdg`): Adjoint with phase = $e^{-i\pi/4}$

**SX and SX† Gates**: Sqrt-X gates; $SX = \begin{bmatrix} \frac{1+i}{2} & \frac{1-i}{2} \\ \frac{1-i}{2} & \frac{1+i}{2} \end{bmatrix}$
- **Implementation**: Via `apply_single_qubit_matrix` helper

### Rotation Gates

**RX(θ)** (`apply_rx`): Rotation around X-axis
$$RX(\theta) = \begin{bmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}$$
- **Code**: Compute `cos_half = cos(θ/2)` and `sin_half = sin(θ/2)`, then call `apply_single_qubit_matrix` with matrix elements as `(ar, ai)` pairs
- **Complexity**: Two `sin`/`cos` evaluations per gate application

**RY(θ)** (`apply_ry`): Rotation around Y-axis
$$RY(\theta) = \begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}$$
- **Code**: Similar to RX, but imaginary components are zero

**RZ(θ)** (`apply_rz`): Rotation around Z-axis (diagonal gate)
$$RZ(\theta) = \begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}$$
- **Code**: Call `apply_diagonal_phase` with phase0 = $\cos(\theta/2) - i\sin(\theta/2)$ and phase1 = $\cos(\theta/2) + i\sin(\theta/2)$
- **Optimization**: Avoids amplitude mixing; only scales by phase

**Phase Gate P(λ)** (`apply_p`): Implements $P(\lambda) = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{bmatrix}$
- **Code**: Call `apply_diagonal_phase(sv, q, 1.0, 0.0, cos(λ), sin(λ))`

### General Unitary Gates

**U(θ, φ, λ)** (`apply_u`): General single-qubit unitary
$$U(\theta, \phi, \lambda) = \begin{bmatrix} \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\ e^{i\phi}\sin(\theta/2) & e^{i(\lambda+\phi)}\cos(\theta/2) \end{bmatrix}$$
- **Code**: Precompute all phase factors, then apply via `apply_single_qubit_matrix`
- **Parameters**: Three angles (θ, φ, λ) fully specify any single-qubit unitary

**U1(λ)**, **U2(φ, λ)**, **U3(θ, φ, λ)**: Alternative parameterizations used in some quantum frameworks
- All decompose to the U gate or phase gates

## Helper Functions

### `apply_single_qubit_matrix`

Generic single-qubit matrix application for $U = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:
```cpp
static inline void apply_single_qubit_matrix(
    Statevector& sv, int qubit,
    double ar, double ai,  // a = matrix[0][0]
    double br, double bi,  // b = matrix[0][1]
    double cr, double ci,  // c = matrix[1][0]
    double dr, double di   // d = matrix[1][1]
)
```

**Loop Structure**:
```cpp
#pragma omp parallel for schedule(static) if(sv.dim >= (1<<20))
for (size_t i = 0; i < sv.dim; i += 2 * step) {
    #pragma omp simd aligned(real_ptr, imag_ptr: 64)
    for (size_t j = i; j < i + step; ++j) {
        // Load pair
        double r0 = real_ptr[j], i0 = imag_ptr[j];
        double r1 = real_ptr[j + step], i1 = imag_ptr[j + step];
        
        // Apply: [new_0; new_1] = [[a,b],[c,d]] * [old_0; old_1]
        real_ptr[j]        = (ar*r0 - ai*i0) + (br*r1 - bi*i1);
        imag_ptr[j]        = (ar*i0 + ai*r0) + (br*i1 + bi*r1);
        real_ptr[j + step] = (cr*r0 - ci*i0) + (dr*r1 - di*i1);
        imag_ptr[j + step] = (cr*i0 + ci*r0) + (dr*i1 + di*r1);
    }
}
```

**Features**:
- **Outer parallel loop**: Divides amplitude pairs across OpenMP threads with `schedule(static)`; activates only if `dim >= 2^20` (~1M amplitudes)
- **Inner SIMD loop**: `#pragma omp simd aligned(ptr: 64)` ensures 64-byte aligned memory access for AVX-512 vectorization
- **Complex multiplication**: Four multiplications + two additions per matrix element (2×2 matrix × 2 amplitudes = 16 operations total per pair)

### `apply_diagonal_phase`

Optimized path for diagonal gates (phases on diagonal):
```cpp
static inline void apply_diagonal_phase(
    Statevector& sv, int qubit,
    double cos0, double sin0,  // cos/sin of phase0
    double cos1, double sin1   // cos/sin of phase1
)
```

**Optimization**: Only scales amplitudes by phase; no mixing. Two separate SIMD loops (one for qubit=0 amplitudes, one for qubit=1) avoid branch misprediction.

## Two-Qubit Gates

Two-qubit gates employ a **cache-optimized nested loop** structure with lo/hi step decomposition to maintain sequential memory access and enable SIMD vectorization.

### Cache-Optimized Loop Pattern

For a two-qubit operation on controls/targets, sort indices to iterate in memory order:
```cpp
const int lo = std::min(ctrl, tgt);
const int hi = std::max(ctrl, tgt);
const size_t lo_step = 1ULL << lo;
const size_t hi_step = 1ULL << hi;

#pragma omp parallel for schedule(static) if(dim >= (1<<20))
for (int kk = 0; kk < static_cast<int>(dim); kk += static_cast<int>(2 * hi_step)) {
    size_t k = kk;
    for (size_t j = 0; j < hi_step; j += 2 * lo_step) {
        size_t base = k + j;
        // Compute offsets based on which qubit is hi/lo
        size_t off_ctrl1_tgt0, off_partner;
        if (ctrl > tgt) {
            off_ctrl1_tgt0 = hi_step;   // ctrl(hi)=1, tgt(lo)=0
            off_partner    = lo_step;   // tgt(lo)=1
        } else {
            off_ctrl1_tgt0 = lo_step;   // ctrl(lo)=1, tgt(hi)=0
            off_partner    = hi_step;   // tgt(hi)=1
        }
        
        #pragma omp simd aligned(real_ptr, imag_ptr: 64)
        for (size_t i = 0; i < lo_step; ++i) {
            size_t idx0 = base + i + off_ctrl1_tgt0;
            size_t idx1 = idx0 + off_partner;
            // Process indices where ctrl=1, tgt=0/1
        }
    }
}
```

**Benefit**: Memory access is sequential within cache lines, enabling SIMD prefetch and vectorization. No branch per element; all complex logic in outer loops.

### Controlled Gates

**CX (CNOT)** (`apply_cx`): Flips target when control=1
- **Code**: Swaps amplitudes at `[idx0, idx0 + off_partner]` when ctrl=1
- **Implementation**: Two loop swaps for real and imag parts

**CY** (`apply_cy`): Applies Y to target when control=1
- **Code**: Call `apply_controlled_matrix(sv, ctrl, tgt, 0, 0, 0, -1, 0, 1, 0, 0)` for Y = $\begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}$

**CZ** (`apply_cz`): Negates phase when both qubits are 1
- **Optimization**: Uses `apply_controlled_phase` (diagonal path)
- **Code**: Only modifies amplitudes where ctrl=1 AND tgt=1

**CH** (`apply_ch`): Applies Hadamard to target when control=1
- **Code**: Call `apply_controlled_matrix` with Hadamard coefficients

### Swap Gates

**SWAP** (`apply_swap`): Exchanges amplitudes between q1 and q2
- **Code**: For each index pair with `(q1, q2)` bits differing, swap amplitudes

**iSWAP** (`apply_iswap`): Swaps AND applies $i$ phase factor
- **Code**: Similar to SWAP, but prepends $i$ (imaginary unit) to swapped amplitudes

### Parameterized Controlled Rotations

**CRX, CRY, CRZ** (`apply_crx`, `apply_cry`, `apply_crz`): Controlled rotation gates
- **Code**: Call `apply_controlled_matrix` with rotation gate coefficients
- **Complexity**: Compute `cos(θ/2)` and `sin(θ/2)` once; apply to all matching indices

**CP(λ)** (`apply_cp`): Controlled phase gate
- **Code**: Call `apply_controlled_phase` with phase = $e^{i\lambda}$
- **Optimization**: No amplitude mixing; only phase update when ctrl=1 AND tgt=1

**CU(θ, φ, λ, γ)** (`apply_cu`): Controlled general unitary with four parameters
- **Code**: Call `apply_controlled_matrix` with U gate coefficients plus global phase γ

### Interaction Gates

**ECR (Echoed Cross-Resonance)** (`apply_ecr`): Native two-qubit gate for superconducting qubits
- **Implementation**: dedicated 4-index loop over the documented ECR matrix
- **Argument convention (frozen in R.1.12, deliberate Qiskit deviation)**: the FIRST argument binds to the high bit of the documented matrix, so `lindblad ecr(a, b)` equals `Qiskit ecr(b, a)` (equivalently SWAP * ECR_qiskit * SWAP). All three simulators implement the same convention. Swap the operands when porting Qiskit circuits.

**RZX(θ), RXX(θ), RYY(θ), RZZ(θ)**: Ising interaction gates
- **RXX(θ)**: $RXX(\theta) = e^{-i\theta(X \otimes X)/2}$ — applies XX rotation
- **RYY(θ)**: $RYY(\theta) = e^{-i\theta(Y \otimes Y)/2}$ — applies YY rotation
- **RZZ(θ)**: $RZZ(\theta) = e^{-i\theta(Z \otimes Z)/2}$ — applies ZZ rotation (diagonal)
- **RZX(θ)**: $RZX(\theta) = e^{-i\theta(Z \otimes X)/2}$ — mixed ZX rotation
- **Code**: Precompute phase factors; apply via `apply_controlled_matrix` or `apply_controlled_phase` (for RZZ, which is diagonal)

## Three-Qubit Gates

### Toffoli (CCX)

(`apply_ccx`): Flips target when both controls are 1

**Implementation**: Simpler than two-qubit gates; uses direct bit checking:
```cpp
for (int ii = 0; ii < static_cast<int>(dim); ++ii) {
    size_t i = ii;
    // Act when c1=1, c2=1, tgt=0
    if (((i >> c1) & 1) && ((i >> c2) & 1) && !((i >> tgt) & 1)) {
        size_t j = i | (1ULL << tgt);  // Set tgt bit
        std::swap(sv.real_parts[i], sv.real_parts[j]);
        std::swap(sv.imag_parts[i], sv.imag_parts[j]);
    }
}
```

**Complexity**: $O(2^n)$ iterations; conditional swap only for matching indices. OpenMP parallelization via `schedule(static)`.

### CCZ

(`apply_ccz`): Negates phase when all three qubits are 1

**Code**: Check `(i >> c1) & 1 && (i >> c2) & 1 && (i >> tgt) & 1`; negate real and imag parts

### CSWAP (Fredkin)

(`apply_cswap`): Swaps q1 and q2 when control=1

**Code**: Check ctrl=1; then swap amplitudes if q1 and q2 bits differ

### RCCX (Margolus Gate)

(`apply_rccx`): Simplified Toffoli (same as Qiskit RCCX)

**Exact action** (relative-phase Toffoli): `|101> -> -|101>`, `|110> -> i|111>`,
`|111> -> -i|110>`, all other basis states unchanged. Equivalent to
$\text{RCCX} = H \cdot T \cdot CX(c_2,tgt) \cdot T^\dagger \cdot CX(c_1,tgt) \cdot T \cdot CX(c_2,tgt) \cdot T^\dagger \cdot H$ on the target.

- **Implementation (R.1.13, audit F-20)**: a single three-level-stride kernel
  pass over the `dim/8` base groups, applying the exact `±1`/`±i` action above.
  This replaced the previous nine-kernel ladder (which swept the full
  statevector nine times); every coefficient is `±1`/`±i`, so the kernel is
  exact with no floating-point rounding. The nine-gate ladder is still the
  reference decomposition used by the transpiler and the MPS backend.
- **Equivalence**: Same unitary action as CCX up to relative phases.

## Arbitrary N-Qubit Unitary

(`apply_unitary`): Apply a $2^k \times 2^k$ unitary matrix to k target qubits

**Signature**:
```cpp
void apply_unitary(
    Statevector& sv,
    const std::vector<int>& targets,
    const std::vector<Complex128>& matrix  // row-major, 2^k × 2^k
)
```

**Algorithm**:

1. **Validation**: Check that `matrix.size() == (1 << targets.size())^2`
2. **Grouping by Background State**: Partition all $2^n$ amplitudes into $2^{n-k}$ groups, each corresponding to a unique assignment of non-target qubits
3. **Subspace Application**: For each group, apply the $2^k \times 2^k$ matrix to the k target qubits' subspace

**Implementation Details**:
```cpp
size_t n_groups = sv.dim >> k;  // 2^(n-k) groups

#pragma omp parallel for schedule(static) if(n_groups > (1<<15))
for (int gg = 0; gg < static_cast<int>(n_groups); ++gg) {
    size_t g = gg;
    // Map group index g to background index by inserting zeros at target positions
    size_t bg_idx = 0;
    // ... bit manipulation to compute background index ...
    
    // Compute 2^k indices in this subspace
    std::vector<size_t> indices(block_size);
    for (size_t s = 0; s < block_size; ++s) {
        size_t idx = bg_idx;
        for (int ti = 0; ti < k; ++ti) {
            if ((s >> ti) & 1) {
                idx |= target_masks[ti];
            }
        }
        indices[s] = idx;
    }
    
    // Read current amplitudes
    std::vector<double> old_real(block_size), old_imag(block_size);
    for (size_t s = 0; s < block_size; ++s) {
        old_real[s] = sv.real_parts[indices[s]];
        old_imag[s] = sv.imag_parts[indices[s]];
    }
    
    // Apply matrix: new[row] = sum_col matrix[row*block_size + col] * old[col]
    for (size_t row = 0; row < block_size; ++row) {
        double new_r = 0.0, new_i = 0.0;
        for (size_t col = 0; col < block_size; ++col) {
            const Complex128& m = matrix[row * block_size + col];
            new_r += m.real * old_real[col] - m.imag * old_imag[col];
            new_i += m.real * old_imag[col] + m.imag * old_real[col];
        }
        sv.real_parts[indices[row]] = new_r;
        sv.imag_parts[indices[row]] = new_i;
    }
}
```

**Complexity**:
- **Time**: $O(2^n \cdot 2^{2k})$ (iterate over $2^{n-k}$ groups, apply $2^{2k}$ matrix operations)
- **Space**: $O(2^k)$ for temporary vectors per OpenMP thread
- **Parallelization (R.1.13, audit F-8)**: work-shape dispatch. When there are
  many background groups (`n_groups > 2^15`), the outer group loop parallelizes.
  When `k` is large (few groups, big per-group blocks) — e.g. a full-register
  oracle has a single group — the per-group ROW multiply parallelizes instead,
  so large-`k` unitaries no longer run serially at $O(4^n)$.

## Multi-Controlled and Permutation Operations

Structured operations that avoid a dense $2^k \times 2^k$ matrix (R.1.13, audit
F-7/F-9). Declared in `lindblad/gates.hpp`; built via `QuantumCircuit::mcx`,
`mcp`, and `permute` (see [Circuit API](circuit.md)).

### `apply_mcx` — multi-controlled X

```cpp
void apply_mcx(Statevector& sv, const std::vector<int>& controls, int target) noexcept;
```

Flips `target` on every amplitude whose control qubits are all `|1>`. Any number
of controls (0 controls == plain X, 1 == CX, 2 == CCX). $O(\dim)$, visiting
disjoint amplitude pairs (safe under OpenMP). Grover's diffusion uses this
instead of building a dense $2^n \times 2^n$ matrix per iteration.

### `apply_mcp` — multi-controlled phase

```cpp
void apply_mcp(Statevector& sv, const std::vector<int>& qubits, double lambda) noexcept;
```

Multiplies by $e^{i\lambda}$ on every amplitude whose listed qubits are all
`|1>` (symmetric controls; 1 qubit == P, 2 == CP). $O(\dim)$.

### `apply_permutation` — basis permutation

```cpp
void apply_permutation(Statevector& sv, const std::vector<int>& qubits,
                       const std::vector<int>& perm);
```

Applies $|x\rangle \to |\text{perm}[x]\rangle$ within the $2^k$ target subspace
(LSB = `qubits[0]`); `perm` must be a bijection of $[0, 2^k)$. Applied as an
$O(\dim)$ amplitude gather. Shor's controlled modular multiplication is emitted
as a `PERMUTATION` over (control, target), replacing a dense $(2 \cdot 2^{n_t})^2$
matrix with a $2^{n_t}$-entry index map.

Backend support: native in the statevector and density-matrix backends. The MPS
backend reduces `MCX` with `<= 2` controls to X/CX/CCX natively; wider MCX and
MCP/PERMUTATION take the bounded statevector fallback (`to_statevector` ->
apply -> `mps_from_sv`, same as a 3+ qubit UNITARY, capped at
`MPS_SV_MAX_QUBITS`). Peripheral tooling: QASM 3 export emits `ctrl(k) @`
forms and lowers `PERMUTATION` to gates; QASM 2 export throws unless
`QasmExportOptions::decompose_unrepresentable` is set; JSON round-trips all
three natively; the stage-0 `HighLevelDecompose` transpiler pass lowers them
for routing and basis translation (see the [QASM](qasm.md) and
[Transpiler](transpiler.md) pages).

## Performance Optimizations

### SIMD Vectorization

All gates employ `#pragma omp simd aligned(ptr: 64)` to:
- Enable AVX-512 instruction generation (64-byte alignment matches 8×double cache line)
- Allow compiler vectorization without manual intrinsics
- Maintain portability across architectures

### OpenMP Parallelization

- **Single-qubit gates**: Activate parallelization if `dim >= 2^20` (~1M amplitudes)
- **Multi-qubit gates**: Similar threshold based on dimension or number of groups
- **Schedule**: `schedule(static)` for predictable workload distribution without synchronization overhead

### Specialized Diagonal Paths

For gates with diagonal matrices (S, T, RZ, P, CZ, CP, RZZ):
- Use `apply_diagonal_phase` instead of full `apply_single_qubit_matrix`
- Avoids amplitude mixing; only scales by phase
- **Memory traffic**: ~50% of full gate implementation

### Loop Tiling for Two-Qubit Gates

Cache-optimized nested loops (hi_step/lo_step) ensure:
- **Sequential memory access** within SIMD vectors
- **Prefetch efficiency** by accessing contiguous regions
- **Cache utilization** by processing complete blocks before moving to next

### Direct Bit Manipulation for Multi-Qubit Gates

Three-qubit gates (CCX, CCZ, CSWAP) use direct bit checking rather than nested step loops:
- **Simpler code** with fewer nested loops
- **Same complexity** ($O(2^n)$) but potentially better branch prediction on CPUs with specialized bit testing
- **Trade-off**: Slightly worse SIMD efficiency than cache-optimized loops (less predictable memory patterns)

## Integration with QuantumCircuit

Gates are called by `QuantumCircuit` via the public API:
```cpp
circuit.x(q0);           // Single-qubit gate
circuit.cx(ctrl, tgt);   // Two-qubit gate
circuit.ccx(c1, c2, tgt); // Three-qubit gate
circuit.unitary(targets, matrix); // N-qubit unitary
circuit.rx(q, theta);    // Parameterized gate
```

**Gate Construction Pipeline**:
1. User calls circuit method (e.g., `circuit.rx(q, 0.5)`)
2. Circuit computes gate parameters and calls appropriate `apply_*` function
3. Statevector is modified in-place; circuit maintains instruction log (for serialization, inversion, etc.)

## Validation and Error Handling

Every `apply_*` primitive validates its operands before touching memory. The
checks are a handful of integer comparisons at the kernel entry, negligible next
to the O(2^n) amplitude sweep:

- **Index bounds** (`0 <= qubit < n_qubits`): a target, control, or list entry
  outside the register throws `std::out_of_range`. This prevents the shift
  `1 << qubit` and the strided writes it drives from running out of range.
- **Operand structure**: two- and three-qubit gates require distinct qubits;
  `unitary`, `mcx`, `mcp`, and `permutation` require in-range, distinct qubit
  lists and correctly sized matrices/permutations. A violation throws
  `std::invalid_argument`. `permutation` additionally requires its image to be a
  bijection of `[0, 2^k)`.

Messages match the `QuantumCircuit` validators (for example
`"h: qubit index 9 out of range [0, 3)"`). Because they throw, the `apply_*`
functions are not `noexcept`. Direct callers receive the exception; when a gate
is reached through a simulator `run()`, the pre-flight surfaces the same failure
through `Result`.

Physical validity is a third class, and it is governed separately. A primitive
taking a caller-supplied matrix, `apply_unitary` here, also checks that the
matrix is unitary. That check is floating-point work rather than an integer
comparison, so it takes a `ValidationOptions` the caller can set per call:
`Throw` at `1e-12` by default, down to `Ignore` for a caller that has already
proven its input. See [validation.md](validation.md).

## See Also

- [Statevector API](statevector.md) — Quantum state representation and alignment
- [Operators API](operators.md) — Pauli and sparse operator representations
- [Circuit API](circuit.md) — Gate composition and circuit manipulation
- [Estimator API](estimator.md) — Expectation value computation via gate sampling
