#pragma once

#include "qpp/circuit.hpp"

#include <string>
#include <vector>

namespace qpp {

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
    int add_node(DAGNode node);
    void add_edge(int src, int dst, int wire, bool is_classical = false);
};

} // namespace qpp
