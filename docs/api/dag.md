# DAG Circuit

A circuit as a directed acyclic graph of operations linked by wire
dependencies, rather than as a flat instruction list. This is the form every
transpiler pass works in: routing needs to know which gates are ready to run,
optimisation needs to know which gates actually touch each other, and neither
question is answerable from a list without rediscovering the graph each time.

Two gates that share no qubit are independent no matter how far apart they sit
in the instruction list. A `QuantumCircuit` cannot say that. A DAG says it by
construction, because independence is the absence of a path.

## Header

```cpp
#include "lindblad/dag.hpp"
```

## Namespace

`lindblad`

## Types

### `DAGNode`

One vertex. Three kinds, distinguished by `type`:

- `Type::IN` and `Type::OUT`: the boundary. Every wire enters at an `IN` node
  and leaves at an `OUT` node, so a wire carrying no gates is still a path and
  the graph has no special case for an idle qubit.
- `Type::OP`: an actual operation. `op` holds the `Instruction`, and is only
  meaningful on this node type.

Fields:

- `type`: which of the three kinds this is
- `op`: the instruction, valid only when `type == Type::OP`
- `node_id`: stable identifier, which is **not** the index into `nodes`
- `qubit_wires`: quantum wires this node sits on
- `clbit_wires`: classical wires this node sits on

The distinction between `node_id` and position in `nodes` matters whenever you
hold an id across a mutation. Ids are handed out by an internal counter and an
internal map resolves them to positions.

### `DAGEdge`

One wire dependency, from `src_node` to `dst_node`, carrying:

- `wire`: the qubit or clbit index the dependency runs along
- `is_classical`: whether `wire` names a clbit rather than a qubit

An edge is per wire, not per pair of nodes, so two gates sharing two qubits are
joined by two edges. That is deliberate: a routing pass asks which *wire* forced
an ordering, and collapsing the edges would throw that away.

### `DAGCircuit`

Public data members, readable directly:

- `nodes`: every vertex
- `edges`: every dependency, flat
- `n_qubits`, `n_clbits`: register widths

## Constructors

```cpp
DAGCircuit();
DAGCircuit(int n_qubits, int n_clbits);
```

The default constructor makes an empty graph with no registers. The sized one
creates the `IN` and `OUT` boundary for every wire, so the graph is well formed
before a single operation is added.

## Methods

### Conversion

```cpp
static DAGCircuit from_circuit(const QuantumCircuit& qc);
QuantumCircuit to_circuit() const;
```

`from_circuit` walks the instruction list and links each operation to the last
writer of every wire it touches. `to_circuit` flattens back, using a topological
order.

The round trip is how every transpiler pass in this project runs: in, transform,
out. It is worth knowing that `to_circuit` picks *a* valid order, not the one
you started with. Independent gates may come back in a different sequence, which
is semantically identical and textually different.

### Traversal

```cpp
std::vector<int> topological_sort() const;
std::vector<int> front_layer() const;
std::vector<int> successors(int node_id) const;
std::vector<int> predecessors(int node_id) const;
```

`front_layer` returns the operations with no operation predecessors: the gates
that could execute right now. Boundary nodes do not count as predecessors, which
is what makes the first layer of a circuit come back rather than an empty list.

All four return **node ids, not indices into `nodes`**. There is no public
mapping from an id back to its node, so resolving one means searching `nodes`
for a matching `node_id`. That is linear per lookup, and it is the reason the
routing pass in this project builds its own front layer by iterating `nodes`
directly rather than calling `front_layer`. If you are resolving many ids, build
your own id-to-index map once and reuse it.

### Mutation

```cpp
void substitute_node(int node_id, const DAGCircuit& replacement);
void remove_node(int node_id);
```

Replace one operation with a subgraph, or delete one and reconnect its wires.

Both carry a cost worth knowing before you build a loop around them: each erases
from the flat `edges` vector and rebuilds the id-to-index map, so a long
sequence of substitutions is quadratic in the size of the circuit. Every
transpiler pass in this project instead rebuilds through `from_circuit`, which
is why the cost has never mattered in practice. If you are calling these
repeatedly on a large circuit, that is the reason it is slow, and rebuilding may
genuinely be faster.

### Queries

```cpp
std::vector<std::pair<int,int>> two_qubit_ops() const;
int num_op_nodes() const;
int depth() const;
```

`two_qubit_ops` returns the qubit pairs of every two-qubit operation, which is
what a routing pass needs to check against a coupling map.

`num_op_nodes` counts operations only, excluding the boundary, so it matches the
instruction count of the circuit rather than the vertex count of the graph.

`depth` is the longest path through operation nodes: the number of sequential
layers, not the number of gates.

## Exceptions and preconditions

Node ids passed to `successors`, `predecessors`, `substitute_node` and
`remove_node` must exist in the graph. An id that does not resolve is not a
meaningful query, and holding an id across a mutation that removed it is the
usual way to produce one.

`op` is only valid on a node whose `type` is `Type::OP`. Reading it on a
boundary node gives you a default-constructed instruction rather than an error,
so check `type` first when walking `nodes` directly.

## Example

Reading the parallel structure of a circuit, which is the question the flat
instruction list cannot answer:

```cpp
#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"

#include <iostream>

using namespace lindblad;

QuantumCircuit qc(4);
qc.h(0);
qc.h(2);          // independent of the gate above: different wire
qc.cx(0, 1);
qc.cx(2, 3);      // still independent of everything on wires 0 and 1

const DAGCircuit dag = DAGCircuit::from_circuit(qc);

std::cout << "operations: " << dag.num_op_nodes() << "\n";  // 4
std::cout << "depth:      " << dag.depth() << "\n";         // 2, not 4

// The gates that could run immediately: both Hadamards.
//
// front_layer returns ids, and an id is not a position in `nodes`, so the
// lookup is a search. For a handful of ids that is fine; for many, build an
// id-to-index map once instead.
for (int id : dag.front_layer()) {
    for (const DAGNode& n : dag.nodes) {
        if (n.node_id != id) continue;
        std::cout << "ready: " << n.op.gate_name() << "\n";
        break;
    }
}
```

Depth is 2 rather than 4 because the two chains never touch. That gap between
instruction count and depth is the entire reason this representation exists.

## Related pages

- `docs/api/circuit.md` for `QuantumCircuit` and `Instruction`, the flat form
  this converts to and from.
- `docs/api/transpiler.md` for the passes that consume a `DAGCircuit`, including
  SABRE routing and the optimisation passes.
- `docs/Architecture.md` for where the DAG sits in the transpilation pipeline.
