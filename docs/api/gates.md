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

All single-qubit gates act on amplitude pairs: member 0 at an index with the target bit clear, member 1 at the index `step = 1 << qubit` above it, with the bit set. Each gate states its formula once, as a function of one pair, and the sweep applies it to every pair (see [Sweeping the State](#sweeping-the-state)).

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

**The pair function**:
```cpp
for_each_pair<kPairBoth>(sv, qubit, [=](auto* r, auto* i) {
    const auto r0 = r[0], i0 = i[0];  // member 0: target bit clear
    const auto r1 = r[1], i1 = i[1];  // member 1: target bit set

    // [new_0; new_1] = [[a,b],[c,d]] * [old_0; old_1]
    r[0] = (ar*r0 - ai*i0) + (br*r1 - bi*i1);
    i[0] = (ar*i0 + ai*r0) + (br*i1 + bi*r1);
    r[1] = (cr*r0 - ci*i0) + (dr*r1 - di*i1);
    i[1] = (cr*i0 + ci*r0) + (dr*i1 + di*r1);
});
```

**Features**:
- **One formula, two widths**: `r` and `i` point at 8-lane values (eight pairs) or at doubles (one pair, inside a vectorized loop); the same expression serves both, and a result does not depend on how many threads ran it
- **Members touched**: `kPairBoth` loads and stores both members; a gate that changes only member 1 (Z, S, S†) passes `kPairSet`
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

**Optimization**: Only scales amplitudes by phase; no mixing. Member 0 is multiplied by the first phase and member 1 by the second.

## Two-Qubit Gates

Two-qubit gates act on groups of four amplitudes, one per setting of the two qubits.

### Groups of Four

A kernel names its two qubits in its own order, `for_each_quad(sv, a, b, g)`, and member `x` of a group has `x = bit_a + 2 * bit_b`. For a controlled gate called with `(ctrl, tgt)`, member 1 is ctrl=1, tgt=0 and member 3 is ctrl=1, tgt=1, whichever of the two qubits is higher. The kernel also names the members it touches as a bit mask, and only those are loaded and stored, so a controlled gate moves only the memory its control selects:

```cpp
void apply_cx(Statevector& sv, int ctrl, int tgt) {
    // Members: 1 = ctrl 1, tgt 0; 3 = ctrl 1, tgt 1.
    for_each_quad<0b1010>(sv, ctrl, tgt, [](auto* r, auto* i) {
        std::swap(r[1], r[3]);
        std::swap(i[1], i[3]);
    });
}
```

As for single-qubit gates, the group function runs on eight groups at a time, by the sweep described in [Sweeping the State](#sweeping-the-state).

### Controlled Gates

**CX (CNOT)** (`apply_cx`): Flips target when control=1
- **Code**: Swaps members 1 and 3 (ctrl=1 with tgt=0 and with tgt=1); members 0 and 2 are not touched

**CY** (`apply_cy`): Applies Y to target when control=1
- **Code**: Call `apply_controlled_matrix(sv, ctrl, tgt, 0, 0, 0, -1, 0, 1, 0, 0)` for Y = $\begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}$

**CZ** (`apply_cz`): Negates phase when both qubits are 1
- **Code**: Negates member 3 (ctrl=1 AND tgt=1); the other members are not touched

**CH** (`apply_ch`): Applies Hadamard to target when control=1
- **Code**: Call `apply_controlled_matrix` with Hadamard coefficients

### Swap Gates

**SWAP** (`apply_swap`): Exchanges amplitudes between q1 and q2
- **Code**: Swaps the two members whose `(q1, q2)` bits differ

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
- **Code**: Precompute `cos(θ/2)` and `sin(θ/2)`; each gate has its own group function over the four members (RZZ is diagonal and multiplies every member by its parity's phase)

## Three-Qubit Gates

### Toffoli (CCX)

(`apply_ccx`): Flips target when both controls are 1

**Implementation**: The gate acts on groups of eight amplitudes, one per setting of the three qubits, and only two members of each move. The groups' base indices (all three bits clear) come in runs of consecutive indices, `2^q_min` long for the lowest of the three qubits, so within a run the c1=1, c2=1, tgt=0 amplitudes are one contiguous stretch and their partners the stretch `2^tgt` above it. The kernel swaps the two stretches:
```cpp
detail::for_each_run(sv.dim, sorted_qubits, 3, [=](size_t run, auto len) {
    const size_t first = run | c1b | c2b;  // c1=1, c2=1, tgt=0
    for_each_in_run(len, [&](size_t k) {
        std::swap(real_ptr[first + k], real_ptr[first + k + tb]);
        std::swap(imag_ptr[first + k], imag_ptr[first + k + tb]);
    });
});
```
`for_each_in_run` runs a stretch of one or two groups (the lowest qubit at 0 or 1) as straight-line code, and a longer one as an `omp simd` loop vectorized across the run.

**Complexity**: $O(2^n / 8)$ swaps: only the pairs that change are visited. The runs are split across threads as in [Sweeping the State](#sweeping-the-state).

### CCZ

(`apply_ccz`): Negates phase when all three qubits are 1

**Code**: Negates the stretch with all three bits set, run by run as for CCX

### CSWAP (Fredkin)

(`apply_cswap`): Swaps q1 and q2 when control=1

**Code**: Swaps the ctrl=1, q1=1, q2=0 stretch with the ctrl=1, q1=0, q2=1 stretch, run by run as for CCX

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
    const std::vector<Complex128>& matrix,  // row-major, 2^k × 2^k
    ValidationOptions validation = {}
)
```

`targets[j]` is bit j of a member index: member `s` of a group is the amplitude
whose `targets[j]` bit equals bit j of `s`, for every j, and the new member `r`
is $\sum_s M[r][s] \cdot \text{old}_s$.

**Algorithm**:

1. **Validation**: the targets must be in range and distinct, `matrix.size()`
   must be $(2^k)^2$, and the matrix must be unitary under `validation` (see
   [Validation and Error Handling](#validation-and-error-handling)).
2. **Path**: one of the three below, chosen by `k`, the state size and whether
   the matrix is diagonal (every off-diagonal entry exactly zero).

**One or two targets, dense matrix**: the $2 \times 2$ or $4 \times 4$ product
is a group function for the sweep the one- and two-qubit gates use (see
[Sweeping the State](#sweeping-the-state)), so it runs on eight groups at a
time with the gate's coefficients held in registers.

**Up to six targets**, on a state of at least $8 \cdot 2^k$ amplitudes: the
product runs on 8-lane values, each lane holding a different group, so one pass
over the matrix updates eight groups.

- A target below qubit 3 pairs amplitudes within one cache line (a low
  target); the others pair whole cache lines (high targets). A tile is
  $2^k$ cache lines of each array that together hold every member of eight
  groups.
- Each tile is loaded, its lanes are shuffled once per low target so that each
  8-lane value holds one member of all eight groups, the matrix multiplies the
  values four rows at a time (each input value is loaded once per four rows),
  and the inverse shuffles put every lane back before the store.
- The matrix is split into real and imaginary parts and renumbered into the
  shuffled order once per call, not once per group.

**Diagonal matrix**, up to six targets: each amplitude is scaled by its own
diagonal entry, one complex multiply per amplitude in place of $2^k$. A block
fused from diagonal gates (Z, S, T, RZ, P, CZ, CP, ...) always takes this path.
Each cache line is multiplied by an 8-lane coefficient holding the entries of
its members. When two lines of one group lie 4 KiB or more apart, they are
visited in separate passes over the state, so no load waits on a store whose
address only appears to overlap; lines closer than that are visited in address
order within one pass.

**More than six targets**, or a state smaller than one tile: one group at a
time. The members' offsets above the group's base are the same for every
group, so they are computed once per call; the base of group `g` is `g` with a
zero inserted at each target position.

**Complexity**:
- **Time**: $O(2^n \cdot 2^k)$ complex multiply-adds for a dense matrix
  ($2^{n-k}$ groups, a $2^k \times 2^k$ product each); $O(2^n)$ for a diagonal
  one.
- **Space**: the matrix in the kernel's layout, $O(2^{2k})$, once per call.
- **Parallelization**: each amplitude of a dense product costs $2^k$ complex
  multiply-adds, so the team is sized for $2^n \cdot 2^k / 2$ amplitudes of
  one-qubit work; a diagonal product is sized for $2^n$ (see
  [OpenMP Parallelization](#openmp-parallelization)). Above six targets, when
  there are fewer groups than that team could use (a full-register oracle has
  a single group), the rows of each group's product are split across the team
  instead. Every amplitude is computed the same way whatever the team, so the
  result does not depend on the thread count.

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
apply -> `rebuild_from_statevector`, same as a 3+ qubit UNITARY, capped at
the chain's dense limit: 25 qubits, or 31 under `QubitLimit::Lift`). Peripheral tooling: QASM 3 export emits `ctrl(k) @`
forms and lowers `PERMUTATION` to gates; QASM 2 export throws unless
`QasmExportOptions::decompose_unrepresentable` is set; JSON round-trips all
three natively; the stage-0 `HighLevelDecompose` transpiler pass lowers them
for routing and basis translation (see the [QASM](qasm.md) and
[Transpiler](transpiler.md) pages).

## Performance Optimizations

### Sweeping the State

One- and two-qubit gates are swept by `detail::for_each_group`, which takes the gate's group function and runs it eight groups at a time on 8-lane values, each lane holding one group. The sweep picks one of two paths by the lowest qubit `q_min` the gate touches:

- **Blocks** (`q_min >= 3`): the groups' base indices come in runs of `2^q_min` consecutive indices, at least eight, so the same member of eight consecutive groups is one whole cache line. Each thread sweeps its share run by run, the next run following from the last by a masked increment. The run is swept a cache line at a time: the sweep loads each member's line directly, calls the group function on the 8-lane values, and stores the lines back. Diagonal gates sweep a long run (eight cache lines or more) as one `omp simd` loop over its amplitudes instead, which the compiler vectorizes for a function that scales one member; a dense group function (a fused $4 \times 4$ block, for one) is too heavy for the compiler to vectorize that way, so whole-group sweeps always use the 8-lane values.
- **Lanes** (`q_min < 3`): a run is shorter than a cache line (one to four amplitudes). The sweep takes tiles of whole cache lines and shuffles their lanes so that one 8-lane value holds the same member of eight different groups. The group function runs once on those values, and the inverse shuffle puts every lane back. A gate qubit at 3 or above chooses which cache lines form a tile, since its partner line lies `2^q` amplitudes away.

Each thread copies the group function and takes `restrict` pointers to the amplitude arrays once for its share, so the gate's coefficients stay in registers across the whole share.

The 8-lane values use GCC and Clang vector extensions, which the compiler lowers to the target's own registers and permutes (AVX-512, AVX2, SSE2 or NEON); other compilers get a plain eight-double array with the same values. A state smaller than one tile (under 16 amplitudes for a one-qubit gate, 32 for a two-qubit gate) runs the group function on doubles, one group at a time.

Three-qubit gates (CCX, CCZ, CSWAP, RCCX) sweep runs only: straight-line code for runs of one or two groups, an `omp simd` loop over longer ones.

### OpenMP Parallelization

- **Team size**: every pass sizes its team by its work: `min(available threads, work / min_amplitudes_per_thread)`, where the work counts amplitudes at one-qubit-gate cost and `min_amplitudes_per_thread` defaults to `DEFAULT_MIN_AMPLITUDES_PER_THREAD` (2048, in `lindblad/types.hpp`). A team of one runs serially. See `min_amplitudes_per_thread` in the [Simulators](simulators.md) page.
- **Even split**: the groups (or tiles) are divided among the team by count, in contiguous shares that start on a cache line, so a gate on the highest qubit keeps every thread busy and no two threads write one line.
- **Determinism**: each amplitude is written by exactly one thread with the same formula, so a gate computes the same bits on one thread or many.

### Specialized Diagonal Paths

For gates with diagonal matrices (Z, S, S†, T, T†, RZ, P, CZ, CP, CRZ, RZZ):
- The kernel states the gate per member, as a function that scales one member by its phase (`detail::for_each_diagonal`); no amplitude mixing
- On the blocks path, each run is swept one member at a time, so the loop streams through one member's memory before the next member's instead of interleaving all of them; the lanes path applies the member function to every member of a tile
- Controlled diagonal gates (CZ, CP) touch one member of four, so they load and store a quarter of the state when both qubits are at 3 or above, half when exactly one is below 3, and all of it when both are (a cache line then holds touched and untouched members alike)

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
is reached through a simulator `run()`, the checks before the first gate refuse
the same fault first, with `lindblad::InvalidArgument` or `lindblad::OutOfRange`
naming the instruction (see [Failures](failures.md)).

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
