# Transpiler API Deep Dive

The Transpiler provides circuit optimization, layout assignment, and routing passes to adapt circuits for constrained quantum hardware and performance optimization. All passes operate on a Directed Acyclic Graph (DAG) representation of the circuit for efficient dependency analysis and gate reordering.

## Overview and Architecture

**Header**: `include/lindblad/transpiler.hpp`

**Namespace**: `lindblad`

The transpiler pipeline follows a modular pass architecture:

1. **Input**: `QuantumCircuit` or `DAGCircuit`
2. **Passes** (in sequence):
   - Layout assignment (logical → physical qubit mapping)
   - Routing (insert SWAP gates to satisfy coupling constraints)
   - Basis translation (decompose gates into hardware-native basis)
   - Optimization (eliminate redundant gates, consolidate blocks)
   - Scheduling (assign time slots to instructions)
3. **Output**: Optimized `QuantumCircuit` with physical qubit indices

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
- **basis_gates**: Gate set native to the target hardware (e.g., `{"cx", "u3", "rz"}` for IBM systems)
- **initial_layout**: Optional initial assignment (default: identity mapping)
- **optimization_level**: Control pass selection and aggressiveness
  - 0: Minimal (only layout + routing + basis translation)
  - 1: Standard (add single-qubit optimization)
  - 2: Aggressive (add two-qubit consolidation, commutative cancellation)
  - 3: Maximum (add scheduling, gate removal optimizations)

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

## Layout Passes: Assign Logical Qubits to Physical Qubits

### TrivialLayout

```cpp
class TrivialLayout : public TranspilationPass { ... };
```

**Behavior**: Map logical qubit $i$ to physical qubit $i$ (identity mapping).

**Use**: Baseline for testing or when the circuit already respects the coupling map.

**Complexity**: $O(1)$

### SabreLayout

```cpp
class SabreLayout : public TranspilationPass { ... };
```

**Behavior**: Heuristic layout assignment via the SABRE forward-backward-forward pass sequence (Li et al. 2019, arXiv:1809.02573).

Algorithm:
1. Start with trivial (identity) layout
2. **Forward pass**: run SABRE routing, record the final logical→physical mapping and SWAP count
3. **Backward pass**: seed from the forward final mapping; run SABRE routing again; update best if fewer SWAPs
4. **Second forward pass**: seed from best result so far; run SABRE routing a final time
5. Apply the winning layout by rebuilding the DAG — qubit indices in every instruction are remapped to physical indices and the DAG is reconstructed from scratch so that all edge `wire` fields and adjacency metadata remain consistent

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

**Behavior**: Decompose any gate not in `basis_gates` into a sequence of native gates.

**Supported decompositions**:
- **RX, RY, RZ → U3** (parameterized rotations)
- **H → U3** (Hadamard via U3)
- **T, S, Tdg, Sdg → U1** (phase gates)
- **CX, CY, CZ** → CX + single-qubit basis (if CX in basis)
- **MCX, CNOT variants** → native CX decomposition
- **SWAP, iSWAP** → CX decomposition (typically 3 CNOTs for SWAP)

**Code path**: Look up decomposition in a table or compute via unitary synthesis (Qiskit's `one_qubit_decompose` pattern).

**Complexity**: $O(k)$ per gate where $k$ = decomposition size (typically $k \leq 5$)

**Use**: Right before execution on hardware to ensure all gates are executable.

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

**Benefit**: Reduces 2Q gate count for circuits with blocks of consecutive gates.

**Note**: Only applies KAK if block contains $\geq 2$ gates; single gates left unchanged (avoid losing CNOT).

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
    int optimization_level,                  // 0–3
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates
);
```

**Level 0** (Minimal):
- Layout (Trivial)
- Routing (SABRE)
- Basis Translation

**Level 1** (Standard):
- Layout (Trivial or SABRE)
- Routing (SABRE)
- Basis Translation
- Optimize1qGates
- CXCancellation

**Level 2** (Aggressive):
- Level 1 passes
- ConsolidateBlocks
- CommutativeCancellation

**Level 3** (Maximum):
- Level 2 passes
- Scheduling (ASAP)
- RemoveDiagonalGatesBeforeMeasure
- RemoveResetInZeroState

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
3. Get preset `PassManager` for the level
4. Run manager to get optimized DAG
5. Convert DAG → `QuantumCircuit` (with physical qubit indices)
6. Return result

**Use**: Quick transpilation without manual pass construction.

### Example: Transpile for IBM Heavy-Hex

```cpp
#include "lindblad/transpiler.hpp"

lindblad::QuantumCircuit circuit = ...;  // Your circuit

// Target: IBM 27-qubit heavy-hex
lindblad::CouplingMap coupling = lindblad::CouplingMap::heavy_hex(27);
std::vector<std::string> basis = {"cx", "u3", "rz"};

lindblad::QuantumCircuit optimized = lindblad::transpile(
    circuit,
    coupling,
    basis,
    2  // optimization_level=2
);
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

Order passes to minimize redundant work:
1. **Layout first**: Determine logical→physical mapping early
2. **Routing**: Insert SWAPs using determined layout
3. **Basis translation**: Decompose remaining gates
4. **Optimization**: Clean up inserted gates and canonical forms
5. **Scheduling (optional)**: Final timing assignment

Reordering can increase optimization time without benefit (e.g., routing after optimization may undo savings).

### Optimization Level Trade-offs

| Level | SWAPs (avg) | Depth (avg) | Time (s) | Use Case |
|---|---|---|---|---|
| 0 | ~40% excess | ↓ minimal | ↓ <1s | Testing, simulation |
| 1 | ~20% excess | ↓ moderate | 1–5s | Production default |
| 2 | ~10% excess | ↓ good | 5–30s | Quality-critical |
| 3 | ~5% excess | ↓ excellent | 30–300s | Research, offline prep |

## Common Pitfalls

1. **Wrong basis gates**: If `basis_gates` doesn't include CX, routing may fail. Always include all hardware-native gates.

2. **Disconnected layout**: If `initial_layout` violates coupling map (e.g., maps adjacent logical qubits to non-adjacent physical qubits), routing cannot fix it; use SabreLayout or TrivialLayout instead.

3. **Mutating DAG nodes directly after layout**: Do not remap `DAGNode::qubit_wires` or `Instruction::qubits` in-place on a DAG. `DAGEdge::wire` fields store qubit indices independently; partial mutation leaves the DAG in an inconsistent state. Always rebuild via `dag.to_circuit()` → remap instructions → `DAGCircuit::from_circuit()`.

3. **Over-optimization**: Level 3 can be slower than execution; use only for pre-computed circuits, not for repeated transpilation in tight loops.

4. **DAG caching**: DAGCircuit is not cached by default; convert once and reuse across multiple passes if possible.

## See Also

- [Gates API](gates.md) — Gate decomposition targets and matrix forms
- [Circuit API](circuit.md) — QuantumCircuit instruction format and manipulation
- [Simulators API](simulators.md) — Simulator integration with transpilation
- [Estimator API](estimator.md) — Primitive-level transpilation and caching
