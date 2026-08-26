# Transpiler API Deep Dive

The Transpiler provides circuit optimization, layout assignment, and routing passes to adapt circuits for constrained quantum hardware and performance optimization. All passes operate on a Directed Acyclic Graph (DAG) representation of the circuit for efficient dependency analysis and gate reordering.

## Overview and Architecture

**Header**: `include/lindblad/transpiler.hpp`

**Namespace**: `lindblad`

The transpiler pipeline follows a modular pass architecture:

1. **Input**: `QuantumCircuit` or `DAGCircuit`
2. **Passes** (preset stage order):
   - High-level decomposition (lower `MCX` / `MCP` / `PERMUTATION` into routable, translatable gates; composed only when a constrained coupling map or a basis is present)
   - Layout assignment (logical → physical qubit mapping, expanded to device width)
   - Routing (insert SWAP gates to satisfy coupling constraints)
   - Optimization (eliminate redundant gates, consolidate blocks)
   - Basis translation (decompose gates into the hardware-native basis; always the final stage so the basis contract holds on the output)
3. **Output**: Optimized `QuantumCircuit` with physical qubit indices; when a coupling map is present the output width equals `n_physical_qubits`

Scheduling passes (`ASAPSchedule`, `ALAPSchedule`) exist but are not composed into any preset level; append them to a custom `PassManager` when needed.

All pass types inherit from `TranspilationPass` and implement:
```cpp
virtual DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const;
virtual std::string name() const;
```

## DAGCircuit: Graph Representation

**Header**: `include/lindblad/dag.hpp`

A DAG (Directed Acyclic Graph) represents circuit dependencies explicitly, enabling efficient gate reordering and analysis.

### Structure

```cpp
struct DAGNode {
    enum class Type { IN, OUT, OP };
    Type type;           // IN = input boundary, OUT = output boundary, OP = operation
    Instruction op;      // Gate instruction (only for OP nodes)
    int node_id;
    std::vector<int> qubit_wires;    // Qubit indices this node touches.
                                     // Logical before layout; physical after SabreLayout/TrivialLayout.
    std::vector<int> clbit_wires;    // Classical bit indices
};

struct DAGEdge {
    int src_node, dst_node;
    int wire;            // Qubit or classical bit index
    bool is_classical;   // True if dependency is on classical bit, false if quantum
};

class DAGCircuit {
public:
    std::vector<DAGNode> nodes;
    std::vector<DAGEdge> edges;
    int n_qubits, n_clbits;
};
```

**Semantics**:
- Each IN node has outgoing edges to all single-qubit gates on that qubit
- Each OUT node has incoming edges from all final gates on each qubit
- Gates on different qubits with no intermediate dependencies can reorder

### Key Operations

**Conversion**:
```cpp
static DAGCircuit DAGCircuit::from_circuit(const QuantumCircuit& qc);
QuantumCircuit DAGCircuit::to_circuit() const;
```

**Dependency queries**:
```cpp
std::vector<int> topological_sort() const;        // Execution order — O(N+E)
std::vector<int> front_layer() const;             // Gates with no predecessors — O(N+E)
std::vector<int> successors(int node_id) const;   // Immediate dependents — O(degree)
std::vector<int> predecessors(int node_id) const; // Immediate dependencies — O(degree)
```

`adj_out` and `adj_in` store `vector<DAGEdge>` keyed by node ID, giving O(degree) access for `successors`, `predecessors`, `substitute_node`, and `remove_node`. The flat `edges` vector is still present for passes that iterate all edges.

**Analysis**:
```cpp
std::vector<std::pair<int,int>> two_qubit_ops() const;  // All (control, target) pairs
int num_op_nodes() const;                               // Count of instruction nodes
int depth() const;                                      // Critical path length
```

### Complexity

- **Space**: $O(N + E)$ where $N$ = instruction count, $E$ = dependencies
- **Conversion from QuantumCircuit**: $O(N)$ (single pass)
- **Topological sort**: $O(N + E)$ (DFS-based)
- **Front layer extraction**: $O(E)$ (count in-degrees)

## CouplingMap: Hardware Topology

Represents connectivity constraints of the target quantum processor.

```cpp
class CouplingMap {
public:
    int n_physical_qubits;
    std::vector<std::pair<int,int>> edges;  // Directed edges (control, target)
    
    bool is_connected(int q1, int q2) const;
    std::vector<int> shortest_path(int q1, int q2) const;
    std::vector<std::vector<int>> distance_matrix() const;
    bool is_connected_graph() const;
    
    static CouplingMap linear(int n);
    static CouplingMap grid(int rows, int cols);
    static CouplingMap heavy_hex(int n);      // True IBM heavy-hex topology
    static CouplingMap all_to_all(int n);
};
```

### Semantics (frozen in R.1.12)

- The edge list is LITERAL: `CouplingMap(n)` with no edges declares n qubits where no pair may interact. Routing a 2-qubit gate against it throws a routing error.
- "Unconstrained" is expressed by the ABSENCE of a map: `CouplingMap()` (n = 0). Routing passes skip it entirely. Matches Qiskit/tket semantics.
- SABRE never silently drops gates: a routing stall (no viable SWAP candidates, or a gate spanning disconnected components) throws `std::runtime_error` describing the cause. Before R.1.12 the unroutable remainder of the circuit was silently discarded.
- 3+ qubit gates are routed only if every wire pair is already adjacent; otherwise SABRE throws and asks for decomposition to 1q/2q gates first. Barriers carry no routing constraint.
- Classically-conditioned instructions are ordered in the DAG after the measurement writing their condition bit (read-after-write, plus write-after-read for re-measurement), and the optimisation passes (`Optimize1qGates`, `CXCancellation`, `CommutativeCancellation`, `ConsolidateBlocks`) never merge, cancel, or absorb them.
- `ConsolidateBlocks` verifies every KAK decomposition against the consolidated block's unitary (up to global phase) and keeps the original block on mismatch.

### Predefined Topologies

**Linear**: Qubits $0 \to 1 \to 2 \to \cdots \to N-1$
- Maximum distance: $N-1$
- Use for superconducting qubit chains

**Grid**: $\text{rows} \times \text{cols}$ 2D grid
- Maximum distance: $\text{rows} + \text{cols} - 2$
- Use for planar processor layouts

**Heavy-Hex** (`heavy_hex(n)`): Native IBM topology
- Heavy-hexagonal lattice with 8-connectivity and specific qubit ordering
- Constrained routing for actual IBM hardware

**All-to-All**: Complete graph (no routing needed)
- Used for testing or simulator execution

### Example: Construct a 5-qubit linear topology

```cpp
CouplingMap cm = CouplingMap::linear(5);  // Qubits: 0-1-2-3-4
std::vector<int> path = cm.shortest_path(0, 4);  // [0, 1, 2, 3, 4]
```

## TranspilationContext: Pass Configuration

```cpp
struct TranspilationContext {
    CouplingMap coupling_map;
    std::vector<std::string> basis_gates;        // Native gates: e.g., {"cx", "u3", "rz"}
    std::vector<int> initial_layout;             // Logical → physical qubit mapping
    int optimization_level;                      // 0 (minimal) to 3 (aggressive)
};
```

**Fields**:
- **coupling_map**: Hardware connectivity constraints
- **basis_gates**: Target gate set. Non-empty: the preset pipelines compose `BasisTranslator` as the final stage and the transpiled output contains ONLY these gates, or `std::invalid_argument` is thrown. Empty: no basis translation. The equivalence library targets `cx` + `u3`, so a practical basis includes both (plus any gates to keep native).
- **initial_layout**: Optional logical → physical assignment, honoured by `SabreSwap` at optimization levels 0–1 (default: identity). Validated in R.1.15.0: entries must be distinct in-range physical indices, and a partial layout (fewer entries than wires) is completed deterministically with the unused physical slots in ascending order; malformed layouts throw `std::invalid_argument`. Leave it empty at levels ≥ 2 — `SabreLayout` chooses the layout and its output must be routed with an identity layout.
- **optimization_level**: Pass selection, 0–3 (see the preset stage table below). Out-of-range values throw `std::invalid_argument` from `preset_pass_manager` (before R.1.15.0, level −1 silently returned an empty manager and levels ≥ 4 silently behaved as 3).

## TranspilationPass: The Pass Interface

Base class for all optimization and mapping passes.

```cpp
class TranspilationPass {
public:
    virtual DAGCircuit run(
        const DAGCircuit& dag,
        const TranspilationContext& ctx
    ) const = 0;
    virtual std::string name() const = 0;
};
```

**Pattern**: Each pass reads the DAG, applies transformations (gate reordering, insertion, decomposition, or removal), and returns an updated DAG. All passes are **read-only** with respect to the input DAG (const input).

## High-Level Decomposition: Lower MCX / MCP / PERMUTATION

### HighLevelDecompose

```cpp
class HighLevelDecompose : public TranspilationPass { ... };
```

**Behavior**: Replaces every `MCX`, `MCP`, and `PERMUTATION` instruction with an exact standard-gate realization; all other instructions pass through unchanged. A circuit containing none of the three ops (and no `CCX` when the floor below applies) is returned as-is (single scan, no rebuild).

Lowering rules (all exact — no approximation anywhere):

- `MCX` with k ≤ 2 controls → `X` / `CX` / `CCX` directly; k ≥ 3 → `H`-conjugated multi-controlled phase (Barenco-style λ/2 recursion with borrowed-wire halving), output alphabet `{X, H, P, CP, CX, CCX}`
- `MCP` → the same λ/2 recursion directly
- `PERMUTATION` whose basis map is a pure wire relabeling → at most k−1 `SWAP`s; a general basis map → cycle decomposition into transpositions, each realized as a CX-fan conjugated, pattern-controlled MCX (then flattened by the MCX rule)

**Coupling-map floor**: when `ctx.coupling_map` is constrained (`n_physical_qubits > 0`), the output additionally contains no gate wider than two qubits. Every `CCX` — whether produced by the lowering above or written by the user — is flattened into the exact 6-`CX` T-ladder (alphabet `{H, P, CX}`, 15 gates, equal to the Toffoli as a matrix identity with no global-phase slack). Rationale: routing executes a 3-qubit gate only when all three wire pairs are simultaneously adjacent, which triangle-free targets (path, grid, heavy-hex) can never provide, so any `CCX` reaching the router on such a map would throw. The 2-qubit floor routes on any connected map. Unconstrained targets keep `CCX`: with no routing consumer the 3-qubit form is strictly better — a `ccx`-bearing `basis_gates` list keeps it native, and flattening would insert `CX` into streams that never needed it.

The construction is ancilla-free and self-contained on the instruction's own operands; worst-case emitted gate count is cubic in the control count for wide `MCX` / `MCP`. Classical conditions (`condition_clbit` / `condition_value`) are propagated onto every emitted gate, preserving feedforward semantics.

**Composition rule**: the preset pipelines compose this pass as stage 0 (ahead of layout and routing) at EVERY optimization level, but only when the transpile target actually requires the lowering — a constrained coupling map (`n_physical_qubits > 0`; routing handles at most 3-qubit gates) or a non-empty `basis_gates` list (the equivalence library cannot reach the three ops). With neither constraint the ops stay native: every backend executes `MCX` / `MCP` / `PERMUTATION` directly, and unconditional lowering would only pessimize. Append the pass to a custom `PassManager` for unconditional lowering.

**Complexity**: $O(G)$ scan when no high-level op is present; otherwise one circuit rebuild plus the per-op synthesis cost above.

## Layout Passes: Assign Logical Qubits to Physical Qubits

### Shared layout invariant (R.1.15.0)

When a coupling map is present, both layout passes return a DAG with `n_qubits == coupling_map.n_physical_qubits`: the circuit is expanded by identity embedding so every physical slot holds a (possibly idle) logical wire. This is what makes circuits smaller than the device routable — SABRE SWAP candidates require both slots of an edge to be occupied, and before the expansion the occupied set was frozen at the initial layout image while the scoring heuristic used full-graph distances through slots that could never be entered (the frozen-slot defect: on degree-sparse maps such as heavy-hex, routing thrashed until the SWAP-budget guard threw a misleading "disconnected components" error). Idle-wire SWAPs are legal; Qiskit emits the same after ancilla allocation.

Circuits with more qubits than the device throw `std::invalid_argument` from either pass (previously undefined behaviour: out-of-range distance-matrix indexing). With no coupling map (`n_physical == 0`), both passes return the DAG unchanged.

### TrivialLayout

```cpp
class TrivialLayout : public TranspilationPass { ... };
```

**Behavior**: Map logical qubit $i$ to physical qubit $i$ (identity mapping), expanded to device width per the shared invariant above.

**Use**: Baseline for testing or when the circuit already respects the coupling map.

**Complexity**: $O(N)$ when expansion rebuilds the DAG; $O(1)$ when the width already matches or no map is present

### SabreLayout

```cpp
class SabreLayout : public TranspilationPass { ... };
```

**Behavior**: Heuristic layout assignment via the SABRE forward-backward-forward pass sequence (Li et al. 2019, arXiv:1809.02573).

Algorithm:
1. Expand the input to device width (identity embedding, shared invariant above) — the internal SABRE search requires every physical slot to hold a logical wire
2. **Forward pass**: run SABRE routing, record the final logical→physical mapping and SWAP count
3. **Backward pass**: seed from the forward final mapping; run SABRE routing again; update best if fewer SWAPs
4. **Second forward pass**: seed from best result so far; run SABRE routing a final time
5. Apply the winning layout by rebuilding the DAG — qubit indices in every instruction are remapped to physical indices and the DAG is reconstructed from scratch so that all edge `wire` fields and adjacency metadata remain consistent

The internal search carries the same SWAP-budget guard as `SabreSwap` (R.1.15.0): a gate spanning disconnected coupling-map components throws `std::runtime_error` instead of looping forever. This matters because at levels ≥ 2 `SabreLayout` is the first pass and no longer has a routing pass in front of it to throw first.

**Output DAG invariant**: `dag.n_qubits == coupling_map.n_physical_qubits` after this pass. All qubit indices are physical. `SabreSwap` must use an identity `initial_layout` on the output (or read it from `ctx.initial_layout`, which should be identity at this stage).

**Complexity**: $O(3 \cdot G \cdot E)$ where $G$ = gate count, $E$ = coupling edges; dominated by three SABRE passes

**Use**: When layout quality significantly impacts routing cost; recommended for dense circuits.

## Routing Passes: Insert SWAPs to Satisfy Coupling Constraints

### SabreSwap

```cpp
class SabreSwap : public TranspilationPass { ... };
```

**Algorithm** (Li et al. 2019):

1. Build **front layer**: 2Q gates whose OP predecessors are all executed
2. **Execute** any gate whose logical qubits are adjacent in current layout
3. If blocked gates remain:
   - Score candidate SWAP insertions using heuristic:
     $$H = H_{\text{basic}} + W \cdot H_{\text{extended}}$$
     where $H_{\text{basic}}$ = distance reduction for front layer,
     $H_{\text{extended}}$ = distance reduction for lookahead layer (next N gates),
     $W \approx 0.5$ = lookahead weight
   - Execute highest-scoring SWAP; update layout; repeat
4. Continue until all 2Q gates are executable

**Key Design**:
- **Lookahead**: Minimize not just front layer but also anticipated future gates
- **Decay factor**: Penalize reusing same qubits for multiple SWAPs to encourage spreading
- **Greedy**: Select locally optimal SWAP; global optimum is NP-hard

**Complexity**: $O(\text{gates} \cdot \text{candidates} \cdot \text{lookahead_depth})$
- Typically $O(k \log N)$ SWAPs where $k$ = 2Q gate count, $N$ = system size

**Use**: Production circuits; balances quality and speed.

### StochasticSwap

```cpp
class StochasticSwap : public TranspilationPass {
private:
    int trials = 20;
    int seed = 42;
};
```

**Behavior**: Run SABRE routing `trials` times with different random seeds and layout perturbations; return the result with fewest SWAPs.

**Use**: When circuit quality is critical and computation time is available.

**Complexity**: $O(\text{trials} \cdot \text{SabreSwap complexity})$

## Basis Translation: Decompose Gates into Hardware-Native Basis

### BasisTranslator

```cpp
class BasisTranslator : public TranspilationPass { ... };
```

**Behavior**: Decompose any gate not in `ctx.basis_gates` into a sequence of gates from a fixed equivalence library targeting `cx` + `u3`. Default basis when `ctx.basis_gates` is empty: `{"cx", "u3"}`.

**Contract (R.1.15.0)**: the output is verified — every returned gate is in the target basis, or `std::invalid_argument` is thrown naming the offending gate. `MEASURE`/`RESET`/`BARRIER` are exempt (not gates). Before R.1.15.0, anything the library could not reach passed through silently, violating the caller's basis.

**Supported decompositions** (all emit `u3` and `cx` only):
- Every fixed single-qubit gate and numeric rotation (`h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `rx`, `ry`, `rz`, `p`, `u1`, `u2`, `u`/`u3`) → one `u3`
- Two-qubit gates (`cy`, `cz`, `ch`, `swap`, `iswap`, `crx`, `cry`, `crz`, `cp`, `cu`, `ecr`, `rzx`, `rxx`, `ryy`, `rzz`) → `cx` + `u3` ladders on the same qubit pair
- Three-qubit gates (`ccx`, `ccz`, `cswap`, `rccx`) → `cx` + `u3` ladders on the same wire pairs

**No decomposition path (throws when outside the basis)**:
- `MCX`, `MCP`, `PERMUTATION` — lowered by the stage-0 `HighLevelDecompose` pass, which the preset pipelines compose ahead of routing whenever a basis is requested, so these never reach the translator in a preset run. They throw only in hand-built pipelines that skip the stage-0 pass: run `HighLevelDecompose` first or include the gate itself in `basis_gates`.
- `UNITARY` — no synthesis path in the translator.
- Symbolic `PARAM_*` gates — parameters are unbound, so numeric decomposition is impossible; bind parameters first.

**Classical conditions**: a conditioned gate decomposes into the same sequence with the condition propagated onto every emitted instruction — exact, because the decompositions are unitary-only and never modify clbits. (Before R.1.15.0 the condition was silently dropped on decomposition.)

**Routing interaction**: every decomposition stays on the original qubit pair(s), so a post-routing circuit remains hardware-legal. The preset pipelines compose this pass as the final stage; the optimisation passes are basis-oblivious (`Optimize1qGates` emits `U`/`RY`/`RZ`, `ConsolidateBlocks` emits `RXX`/`RYY`/`RZZ`), so any earlier position would let them re-introduce non-basis gates.

**Complexity**: $O(k)$ per gate where $k$ = decomposition size (largest: ECR at 11 instructions), plus an $O(n)$ output verification sweep.

**Use**: Composed automatically by `transpile()` / `preset_pass_manager` when `basis_gates` is non-empty; usable standalone before hardware execution.

## Optimization Passes: Eliminate Redundancy and Consolidate Gates

### Optimize1qGates

**Behavior**: Collect consecutive single-qubit gates on the same qubit, compose their 2×2 unitary matrices, and decompose the result into a single U3 gate via ZYZ decomposition.

**Algorithm**:
1. Scan DAG for runs of 1Q gates on the same qubit (no 2Q gates between)
2. Compose: $U_1 \cdot U_2 \cdot U_3 = U_{\text{combined}}$ (matrix multiply)
3. Decompose via ZYZ:
   $$U_{\text{combined}} = e^{i\phi} \cdot RZ(\text{phi}) \cdot RY(\text{theta}) \cdot RZ(\text{lambda})$$
4. Replace run with single U3 instruction

**ZYZ Decomposition** (phase + 3 angles):
- Given any 2×2 unitary $U$, extract determinant for global phase
- Normalize to SU(2); decompose into rotations around Z, Y, Z axes
- Recover $\phi$, $\theta$, $\lambda$ via eigenvalue analysis

**Complexity**: $O(n)$ scans; $O(1)$ per gate (matrix operations are 2×2)

**Example**:
```
Before:  H(q0) - RX(θ) - S(q0) - Z(q0)  [4 gates]
After:   U3(phi, theta, lam) on q0       [1 gate]
```

### CXCancellation

**Behavior**: Identify and remove adjacent CX pairs that cancel (CX·CX = I).

**Algorithm**: Scan backward through DAG; when two consecutive CX gates on the same pair found, remove both.

**Complexity**: $O(n)$ single pass

### ConsolidateBlocks

**Behavior**: Identify maximal blocks of consecutive 2Q gates on the same qubit pair, compose their 4×4 unitaries, and decompose via KAK (Cartan-Weyl-Kraus) into $\leq 3$ CNOTs + local corrections.

**KAK Decomposition** ($k$-rank):
- Any 2Q unitary decomposes as:
  $$U = (A \otimes B) \cdot CX_{\text{basis}} \cdot (C \otimes D)$$
  where $CX_{\text{basis}}$ is a diagonal unitary in Bell basis.
- Number of CNOTs ≤ 3; often 2 or 1 for structured unitaries.

**Benefit**: Reduces 2Q gate count for circuits with blocks of consecutive gates. This is enforced rather than assumed: a decomposition is kept only when it lowers the two-qubit gate count, with total instruction count as a tie-break. A block whose decomposition would not be cheaper is left exactly as it was.

**Two independent gates decide the outcome**, and both are needed:

- The **verification net** rebuilds the decomposition's 4×4 and requires it to match the block up to global phase. On mismatch the original instructions are kept, so the pass can never emit a wrong circuit.
- The **count guard** then asks whether the valid decomposition is worth keeping.

Two-qubit count is the metric because it is the only count that is stable at this point in the pipeline: the cleanup sweep that follows this pass at level 3 merges and cancels single-qubit gates, so a total-count comparison taken here would judge a circuit that no longer exists by the time it runs. Nothing downstream adds or removes two-qubit gates.

**Weyl coordinates**: the interaction is emitted from coordinates reduced into the canonical Weyl chamber. Many coordinate triples describe the same operator up to local gates and they do not all cost the same, since a triple with a zero entry needs one fewer interaction rotation. The reduction moves are absorbed entirely into the local corrections, so iSWAP costs two rotations rather than three.

**Block formation**: a block is a run of *adjacent* two-qubit gates on one pair. A single-qubit gate on either wire ends the run, so an entangler surrounded by local gates forms a block of one.

**Note**: Only applies KAK if block contains $\geq 2$ gates; single gates left unchanged. Nothing is lost by that: a one-gate block holds a single two-qubit gate, so the count guard could only accept a replacement with fewer instructions overall, and a decomposition emitting an interaction rotation plus local corrections never has fewer.

**Complexity**: $O(n)$ for block identification; $O(1)$ per block (fixed 4×4 decomposition)

### CommutativeCancellation

**Behavior**: Identify commuting rotation gates separated by commuting gates, merge same-type rotations, and cancel inverse pairs. Runs in a **fixed-point loop**: after each full forward pass, if any cancellation or merge occurred the pass repeats until no further changes are found. This ensures cancellations exposed by earlier merges are not missed.

**Example**:
```
H(q0) - RX(θ) - H(q0)
If RX commutes with both H gates:
  → H - RX - H  can reorder to  H - H - RX  →  RX
```

**Commutation rules recognized**:
- Z-diagonal gates (RZ, P, T, S, Z, U1, SDG, TDG) commute with each other
- Z-diagonal gates commute through the control wire of CX, CZ, CP, CRZ
- CX commutes with CX on the same qubit pair

**Complexity**: $O(n^2)$ per fixed-point iteration; typically converges in 2–3 passes

### RemoveDiagonalGatesBeforeMeasure

**Behavior**: Remove diagonal gates (RZ, P, T, S, Z, U1) whose only successor is MEASURE.

**Reasoning**: Diagonal gates commute with measurement and only add a global phase on the measured state (no observable effect).

**Complexity**: $O(n)$ single pass

### RemoveResetInZeroState

**Behavior**: Remove RESET instructions on qubits already in state $|0\rangle$ (at circuit start or after prior RESET).

**Complexity**: $O(n)$ single pass with qubit state tracking

## Scheduling Passes: Assign Time Slots to Instructions

Scheduling assigns each instruction a time slot (depth level) subject to dependencies, enabling precise gate timing for control electronics.

### ASAPSchedule (As-Soon-As-Possible)

**Behavior**: Assign each gate to the earliest slot where all dependencies are satisfied.

```
Front layer gates → slot 0
Their successors → slot 1
...
```

**Complexity**: $O(n + e)$ topological sort

**Use**: Minimize circuit depth for execution.

### ALAPSchedule (As-Late-As-Possible)

**Behavior**: Assign each gate to the latest slot such that the critical path length is preserved.

Reverse topological sort: start from OUTPUT, work backward.

**Complexity**: $O(n + e)$

**Use**: Enable gate reordering analysis; identify which gates have slack.

## PassManager and Preset Managers

**PassManager**: Orchestrate a sequence of passes.

```cpp
class PassManager {
public:
    void append(std::unique_ptr<TranspilationPass> pass);
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const;
};
```

**Workflow**:
```cpp
PassManager pm;
pm.append(std::make_unique<SabreLayout>());
pm.append(std::make_unique<SabreSwap>());
pm.append(std::make_unique<BasisTranslator>());
pm.append(std::make_unique<Optimize1qGates>());
DAGCircuit optimized = pm.run(dag, ctx);
```

**Preset Managers**: Auto-select passes based on `optimization_level`.

```cpp
PassManager preset_pass_manager(
    int optimization_level,                  // 0–3; out of range throws std::invalid_argument
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates
);
```

Levels compose per **stage**, non-cumulatively (R.1.15.0). Before R.1.15.0 the levels composed cumulatively, so level ≥ 2 first ran the level-0 `TrivialLayout` + `SabreSwap` block: the initial routing pass executed on the unexpanded DAG (frozen-slot defect) and `SabreLayout` then had to re-route an already-SWAP-laden circuit whose pass-1 SWAPs it could not undo. Now every level makes one layout choice and runs one routing pass.

Stage 0 — high-level decomposition:
- `HighLevelDecompose`, composed at every level if and only if the coupling map is constrained (`n_physical_qubits > 0`) or `basis_gates` is non-empty — the two situations that require `MCX` / `MCP` / `PERMUTATION` to be lowered. With neither, the ops stay native for the backends (see the pass's section above). `preset_pass_manager` therefore consumes its `coupling_map` argument (previously composition ignored it).

Stage 1 — layout + routing:
- Levels 0–1: `TrivialLayout` → `SabreSwap`
- Levels 2–3: `SabreLayout` → `SabreSwap`

Stage 2 — optimisation:
- Level 0: none
- Level 1: `RemoveResetInZeroState` → `Optimize1qGates` → `CXCancellation` → `RemoveDiagonalGatesBeforeMeasure`
- Level 2: level-1 chain with `CommutativeCancellation` inserted before the diagonal removal
- Level 3: level-2 chain, then `ConsolidateBlocks` (KAK) followed by a second cleanup sweep (`Optimize1qGates` → `CXCancellation` → `CommutativeCancellation` → `RemoveDiagonalGatesBeforeMeasure`)

Stage 3 — basis translation:
- Composed if and only if `basis_gates` is non-empty; always the final stage, so the "output contains only basis gates" guarantee holds on the returned circuit. `transpile()` passes the same `basis_gates` into both the composition and the context, so they always agree; when building a preset manager by hand, pass the same vector you place in `ctx.basis_gates`.

## Convenience Function: Direct Transpilation

```cpp
QuantumCircuit transpile(
    const QuantumCircuit& circuit,
    const CouplingMap& coupling_map = CouplingMap(),
    const std::vector<std::string>& basis_gates = {},
    int optimization_level = 1
);
```

**Workflow**:
1. Convert `circuit` → DAG
2. Build `TranspilationContext` from arguments
3. Get preset `PassManager` for the level (throws on level outside 0–3)
4. Run manager to get optimized DAG
5. Convert DAG → `QuantumCircuit` (with physical qubit indices)
6. Return result

**Output width**: when a coupling map is present, the returned circuit has `n_qubits == n_physical_qubits` (device width); circuits larger than the device throw `std::invalid_argument`.

**basis_gates**: non-empty means the output contains only those gates or `std::invalid_argument` is thrown (see `BasisTranslator`). Before R.1.15.0 this parameter was silently ignored — `BasisTranslator` was composed into no preset level.

**Use**: Quick transpilation without manual pass construction.

### Example: Transpile for IBM Heavy-Hex

```cpp
#include "lindblad/transpiler.hpp"

lindblad::QuantumCircuit circuit = ...;  // Your circuit

// Target: IBM 27-qubit heavy-hex
lindblad::CouplingMap coupling = lindblad::CouplingMap::heavy_hex(27);
std::vector<std::string> basis = {"cx", "u3", "rz"};  // must cover the library's cx+u3 targets

lindblad::QuantumCircuit optimized = lindblad::transpile(
    circuit,
    coupling,
    basis,
    2  // optimization_level=2
);
// optimized.n_qubits == 27 (device width) even if `circuit` is smaller;
// every gate in `optimized` is cx, u3, or rz.
```

## Integration with Simulators and Primitives

- **StatevectorSimulator** and **DensityMatrixSimulator** can optionally transpile circuits via:
  ```cpp
  StatevectorSimulator::Options opts;
  opts.transpile = true;
  opts.coupling_map = CouplingMap::linear(5);
  ```

- **Estimator** and **Sampler** internally transpile circuits before simulation if a coupling map is provided

- **Caching**: Estimator caches transpilation results by circuit structure (keyed by topological sort signature)

## Performance Considerations

### Pass Ordering Matters

The preset stage order is deliberate:
1. **Layout first**: determine the logical→physical mapping (and device-width expansion) early
2. **Routing**: insert SWAPs using the determined layout
3. **Optimization**: clean up inserted gates and canonical forms
4. **Basis translation last**: the optimisation passes are basis-oblivious and would re-introduce non-basis gates after an earlier translation; the final position is what makes the basis contract hold on the output

When composing a custom `PassManager`, keep translation after any pass that synthesises gates (`Optimize1qGates`, `ConsolidateBlocks`), or its output guarantee will not survive.

### Optimization Level Trade-offs

Higher levels spend more transpile time for fewer SWAPs and lower depth: level 0 is routing-only (testing, simulation), level 1 adds cheap linear-time cleanup (the default), level 2 adds SABRE layout and commutative cancellation (quality-critical circuits), and level 3 adds KAK block consolidation with a second cleanup sweep (offline preparation of hot circuits). Measured wall-time and quality comparisons against Qiskit live in `docs/Benchmarks.md` (transpiler domain).

## Common Pitfalls

1. **Basis without `cx` + `u3`**: the equivalence library targets `cx` + `u3`, so a basis that includes neither the source gate nor both targets makes `BasisTranslator` throw `std::invalid_argument`. Include `cx` and `u3` (plus any gates you want kept native).

2. **Malformed `initial_layout`**: out-of-range or duplicate entries, or more entries than circuit wires, throw `std::invalid_argument` from `SabreSwap` (R.1.15.0; previously silently ignored or undefined behaviour). A layout that maps interacting logical qubits far apart is legal but costs SWAPs.

3. **`initial_layout` at levels ≥ 2**: `SabreLayout` chooses the layout and returns a physically-indexed DAG; a non-empty `initial_layout` would be applied on top of it by the subsequent routing pass. Leave it empty at levels ≥ 2.

4. **Mutating DAG nodes directly after layout**: Do not remap `DAGNode::qubit_wires` or `Instruction::qubits` in-place on a DAG. `DAGEdge::wire` fields store qubit indices independently; partial mutation leaves the DAG in an inconsistent state. Always rebuild via `dag.to_circuit()` → remap instructions → `DAGCircuit::from_circuit()`.

5. **Over-optimization**: Level 3 can be slower than execution; use only for pre-computed circuits, not for repeated transpilation in tight loops.

6. **DAG caching**: DAGCircuit is not cached by default; convert once and reuse across multiple passes if possible.

## See Also

- [Gates API](gates.md) — Gate decomposition targets and matrix forms
- [Circuit API](circuit.md) — QuantumCircuit instruction format and manipulation
- [Simulators API](simulators.md) — Simulator integration with transpilation
- [Estimator API](estimator.md) — Primitive-level transpilation and caching
