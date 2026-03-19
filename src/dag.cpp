#include "qpp/dag.hpp"

#include <algorithm>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace qpp {

// =============================================================================
// Constructors
// =============================================================================

DAGCircuit::DAGCircuit()
    : n_qubits(0), n_clbits(0), next_node_id(0) {}

DAGCircuit::DAGCircuit(int n_qubits, int n_clbits)
    : n_qubits(n_qubits), n_clbits(n_clbits), next_node_id(0) {}

// =============================================================================
// Internal helpers
// =============================================================================

int DAGCircuit::add_node(DAGNode node) {
    node.node_id = next_node_id++;
    nodes.push_back(std::move(node));
    return nodes.back().node_id;
}

void DAGCircuit::add_edge(int src, int dst, int wire, bool is_classical) {
    edges.push_back({src, dst, wire, is_classical});
}

// =============================================================================
// from_circuit — build DAG from QuantumCircuit
// =============================================================================

DAGCircuit DAGCircuit::from_circuit(const QuantumCircuit& qc) {
    DAGCircuit dag(qc.n_qubits, qc.n_clbits);

    // Create IN nodes for each qubit and clbit
    std::vector<int> qubit_in_nodes(qc.n_qubits);
    std::vector<int> clbit_in_nodes(qc.n_clbits);

    for (int q = 0; q < qc.n_qubits; ++q) {
        DAGNode in_node;
        in_node.type = DAGNode::Type::IN;
        in_node.qubit_wires = {q};
        qubit_in_nodes[q] = dag.add_node(std::move(in_node));
    }

    for (int c = 0; c < qc.n_clbits; ++c) {
        DAGNode in_node;
        in_node.type = DAGNode::Type::IN;
        in_node.clbit_wires = {c};
        clbit_in_nodes[c] = dag.add_node(std::move(in_node));
    }

    // Track the last node that touched each wire
    std::vector<int> last_qubit_node = qubit_in_nodes;
    std::vector<int> last_clbit_node = clbit_in_nodes;

    // Add OP nodes for each instruction
    for (const auto& inst : qc.instructions) {
        DAGNode op_node;
        op_node.type = DAGNode::Type::OP;
        op_node.op = inst;
        op_node.qubit_wires = inst.qubits;
        op_node.clbit_wires = inst.clbits;

        int node_id = dag.add_node(std::move(op_node));

        // Add edges from previous nodes on each wire
        for (int q : inst.qubits) {
            dag.add_edge(last_qubit_node[q], node_id, q, false);
            last_qubit_node[q] = node_id;
        }
        for (int c : inst.clbits) {
            dag.add_edge(last_clbit_node[c], node_id, c, true);
            last_clbit_node[c] = node_id;
        }
    }

    // Create OUT nodes
    for (int q = 0; q < qc.n_qubits; ++q) {
        DAGNode out_node;
        out_node.type = DAGNode::Type::OUT;
        out_node.qubit_wires = {q};
        int out_id = dag.add_node(std::move(out_node));
        dag.add_edge(last_qubit_node[q], out_id, q, false);
    }

    for (int c = 0; c < qc.n_clbits; ++c) {
        DAGNode out_node;
        out_node.type = DAGNode::Type::OUT;
        out_node.clbit_wires = {c};
        int out_id = dag.add_node(std::move(out_node));
        dag.add_edge(last_clbit_node[c], out_id, c, true);
    }

    return dag;
}

// =============================================================================
// to_circuit — extract instruction sequence via topological sort
// =============================================================================

QuantumCircuit DAGCircuit::to_circuit() const {
    QuantumCircuit qc(n_qubits, n_clbits);

    auto sorted = topological_sort();
    for (int nid : sorted) {
        // Find the node
        for (const auto& node : nodes) {
            if (node.node_id == nid && node.type == DAGNode::Type::OP) {
                qc.instructions.push_back(node.op);
                break;
            }
        }
    }

    return qc;
}

// =============================================================================
// Topological sort (Kahn's algorithm)
// =============================================================================

std::vector<int> DAGCircuit::topological_sort() const {
    // Build adjacency list and in-degree map
    std::unordered_map<int, std::vector<int>> adj;
    std::unordered_map<int, int> in_degree;

    for (const auto& node : nodes) {
        if (in_degree.find(node.node_id) == in_degree.end()) {
            in_degree[node.node_id] = 0;
        }
    }

    for (const auto& edge : edges) {
        adj[edge.src_node].push_back(edge.dst_node);
        in_degree[edge.dst_node]++;
    }

    // Queue nodes with in-degree 0
    std::queue<int> q;
    for (const auto& [nid, deg] : in_degree) {
        if (deg == 0) q.push(nid);
    }

    std::vector<int> result;
    while (!q.empty()) {
        int nid = q.front();
        q.pop();
        result.push_back(nid);

        if (adj.count(nid)) {
            for (int neighbor : adj.at(nid)) {
                if (--in_degree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }
    }

    return result;
}

// =============================================================================
// Front layer — OP nodes with no OP predecessors
// =============================================================================

std::vector<int> DAGCircuit::front_layer() const {
    std::vector<int> front;

    // Collect OP nodes whose predecessors are all IN nodes
    for (const auto& node : nodes) {
        if (node.type != DAGNode::Type::OP) continue;

        auto preds = predecessors(node.node_id);
        bool all_in = true;
        for (int pid : preds) {
            for (const auto& pnode : nodes) {
                if (pnode.node_id == pid && pnode.type == DAGNode::Type::OP) {
                    all_in = false;
                    break;
                }
            }
            if (!all_in) break;
        }
        if (all_in) {
            front.push_back(node.node_id);
        }
    }

    return front;
}

// =============================================================================
// Successors / Predecessors
// =============================================================================

std::vector<int> DAGCircuit::successors(int node_id) const {
    std::set<int> succs;
    for (const auto& edge : edges) {
        if (edge.src_node == node_id) {
            succs.insert(edge.dst_node);
        }
    }
    return std::vector<int>(succs.begin(), succs.end());
}

std::vector<int> DAGCircuit::predecessors(int node_id) const {
    std::set<int> preds;
    for (const auto& edge : edges) {
        if (edge.dst_node == node_id) {
            preds.insert(edge.src_node);
        }
    }
    return std::vector<int>(preds.begin(), preds.end());
}

// =============================================================================
// substitute_node — replace a node with a sub-DAG
// =============================================================================

void DAGCircuit::substitute_node(int node_id, const DAGCircuit& replacement) {
    // Find the node to replace
    int node_idx = -1;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].node_id == node_id) {
            node_idx = static_cast<int>(i);
            break;
        }
    }
    if (node_idx < 0) throw std::out_of_range("Node not found");

    const DAGNode& target = nodes[node_idx];

    // Get the incoming and outgoing edges
    std::vector<DAGEdge> in_edges, out_edges;
    for (const auto& e : edges) {
        if (e.dst_node == node_id) in_edges.push_back(e);
        if (e.src_node == node_id) out_edges.push_back(e);
    }

    // Remove old edges
    edges.erase(
        std::remove_if(edges.begin(), edges.end(),
            [node_id](const DAGEdge& e) {
                return e.src_node == node_id || e.dst_node == node_id;
            }),
        edges.end()
    );

    // Get OP nodes from replacement in topological order
    auto sorted = replacement.topological_sort();
    std::vector<int> new_op_ids;
    std::unordered_map<int, int> old_to_new;

    for (int rid : sorted) {
        for (const auto& rnode : replacement.nodes) {
            if (rnode.node_id == rid && rnode.type == DAGNode::Type::OP) {
                DAGNode new_node = rnode;
                int new_id = add_node(std::move(new_node));
                old_to_new[rid] = new_id;
                new_op_ids.push_back(new_id);
                break;
            }
        }
    }

    // Re-map replacement internal edges
    for (const auto& re : replacement.edges) {
        if (old_to_new.count(re.src_node) && old_to_new.count(re.dst_node)) {
            add_edge(old_to_new[re.src_node], old_to_new[re.dst_node],
                     re.wire, re.is_classical);
        }
    }

    // Connect incoming edges to replacement's first nodes per wire
    // Connect outgoing edges from replacement's last nodes per wire
    if (!new_op_ids.empty()) {
        // Simple approach: connect all in_edges to first new node, out_edges from last
        for (const auto& ie : in_edges) {
            add_edge(ie.src_node, new_op_ids.front(), ie.wire, ie.is_classical);
        }
        for (const auto& oe : out_edges) {
            add_edge(new_op_ids.back(), oe.dst_node, oe.wire, oe.is_classical);
        }
    } else {
        // No replacement ops — connect in to out directly
        for (const auto& ie : in_edges) {
            for (const auto& oe : out_edges) {
                if (ie.wire == oe.wire) {
                    add_edge(ie.src_node, oe.dst_node, ie.wire, ie.is_classical);
                }
            }
        }
    }

    // Remove the original node
    nodes.erase(nodes.begin() + node_idx);
}

// =============================================================================
// remove_node
// =============================================================================

void DAGCircuit::remove_node(int node_id) {
    // Connect predecessors directly to successors
    std::vector<DAGEdge> in_edges, out_edges;
    for (const auto& e : edges) {
        if (e.dst_node == node_id) in_edges.push_back(e);
        if (e.src_node == node_id) out_edges.push_back(e);
    }

    // Remove all edges involving this node
    edges.erase(
        std::remove_if(edges.begin(), edges.end(),
            [node_id](const DAGEdge& e) {
                return e.src_node == node_id || e.dst_node == node_id;
            }),
        edges.end()
    );

    // Reconnect: for each wire, connect the predecessor to the successor
    for (const auto& ie : in_edges) {
        for (const auto& oe : out_edges) {
            if (ie.wire == oe.wire && ie.is_classical == oe.is_classical) {
                add_edge(ie.src_node, oe.dst_node, ie.wire, ie.is_classical);
            }
        }
    }

    // Remove the node
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
            [node_id](const DAGNode& n) { return n.node_id == node_id; }),
        nodes.end()
    );
}

// =============================================================================
// Two-qubit operations
// =============================================================================

std::vector<std::pair<int,int>> DAGCircuit::two_qubit_ops() const {
    std::vector<std::pair<int,int>> ops;
    for (const auto& node : nodes) {
        if (node.type == DAGNode::Type::OP && node.qubit_wires.size() == 2) {
            ops.emplace_back(node.qubit_wires[0], node.qubit_wires[1]);
        }
    }
    return ops;
}

// =============================================================================
// Utility
// =============================================================================

int DAGCircuit::num_op_nodes() const {
    int count = 0;
    for (const auto& node : nodes) {
        if (node.type == DAGNode::Type::OP) count++;
    }
    return count;
}

int DAGCircuit::depth() const {
    // Longest path through OP nodes
    auto sorted = topological_sort();
    std::unordered_map<int, int> dist;
    for (const auto& node : nodes) {
        dist[node.node_id] = 0;
    }

    int max_depth = 0;
    for (int nid : sorted) {
        bool is_op = false;
        for (const auto& node : nodes) {
            if (node.node_id == nid && node.type == DAGNode::Type::OP) {
                is_op = true;
                break;
            }
        }

        for (const auto& edge : edges) {
            if (edge.src_node == nid) {
                int new_dist = dist[nid] + (is_op ? 1 : 0);
                dist[edge.dst_node] = std::max(dist[edge.dst_node], new_dist);
                max_depth = std::max(max_depth, dist[edge.dst_node]);
            }
        }
    }

    return max_depth;
}

} // namespace qpp
