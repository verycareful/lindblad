#pragma once

#include "lindblad/circuit.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {

// =============================================================================
// Node in the circuit DAG
// =============================================================================

struct DAGNode {
    enum class Type { IN, OUT, OP };
    Type type;
    Instruction op;        // only valid for OP nodes
    int node_id;
    std::vector<int> qubit_wires;
    std::vector<int> clbit_wires;
};

// =============================================================================
// Edge represents a qubit/clbit wire dependency
// =============================================================================

struct DAGEdge {
    int src_node;
    int dst_node;
    int wire;   // qubit or clbit index
    bool is_classical;
};

// =============================================================================
// DAGCircuit — Directed Acyclic Graph representation of a quantum circuit
// =============================================================================

class DAGCircuit {
public:
    std::vector<DAGNode> nodes;
    std::vector<DAGEdge> edges;
    int n_qubits;
    int n_clbits;

public:
    DAGCircuit();
    DAGCircuit(int n_qubits, int n_clbits);

    // Conversion
    static DAGCircuit from_circuit(const QuantumCircuit& qc);
    QuantumCircuit to_circuit() const;

    // DAG operations used by transpiler
    std::vector<int> topological_sort() const;
    std::vector<int> front_layer() const;  // gates with no OP predecessors
    std::vector<int> successors(int node_id) const;
    std::vector<int> predecessors(int node_id) const;
    void substitute_node(int node_id, const DAGCircuit& replacement);
    void remove_node(int node_id);

    // Two-qubit gate extraction (for routing)
    std::vector<std::pair<int,int>> two_qubit_ops() const;

    // Utility
    int num_op_nodes() const;
    int depth() const;

private:
    int next_node_id = 0;

    // O(1) lookup: node_id → index in nodes vector
    std::unordered_map<int, size_t> node_id_to_idx;
    // Adjacency lists for O(1) successors/predecessors
    std::unordered_map<int, std::vector<int>> adj_out;  // node_id → successor node_ids
    std::unordered_map<int, std::vector<int>> adj_in;   // node_id → predecessor node_ids

    int add_node(DAGNode node);
    void add_edge(int src, int dst, int wire, bool is_classical = false);

    // Rebuild adjacency caches (used after bulk operations)
    void rebuild_adjacency();
};

} // namespace lindblad
