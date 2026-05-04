#include "lindblad/dag.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace lindblad {

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
    int nid = node.node_id;
    node_id_to_idx[nid] = nodes.size();
    nodes.push_back(std::move(node));
    // Initialize empty adjacency entries
    adj_out[nid];
    adj_in[nid];
    return nid;
}

void DAGCircuit::add_edge(int src, int dst, int wire, bool is_classical) {
    edges.push_back({src, dst, wire, is_classical});
    adj_out[src].push_back(dst);
    adj_in[dst].push_back(src);
}

void DAGCircuit::rebuild_adjacency() {
    node_id_to_idx.clear();
    adj_out.clear();
    adj_in.clear();
    for (size_t i = 0; i < nodes.size(); ++i) {
        int nid = nodes[i].node_id;
        node_id_to_idx[nid] = i;
        adj_out[nid];
        adj_in[nid];
    }
    for (const auto& e : edges) {
        adj_out[e.src_node].push_back(e.dst_node);
        adj_in[e.dst_node].push_back(e.src_node);
    }
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
// to_circuit — O(N) via node_id_to_idx map
// =============================================================================

QuantumCircuit DAGCircuit::to_circuit() const {
    QuantumCircuit qc(n_qubits, n_clbits);

    auto sorted = topological_sort();
    for (int nid : sorted) {
        auto it = node_id_to_idx.find(nid);
        if (it != node_id_to_idx.end()) {
            const auto& node = nodes[it->second];
            if (node.type == DAGNode::Type::OP) {
                qc.instructions.push_back(node.op);
            }
        }
    }

    return qc;
}

// =============================================================================
// Topological sort (Kahn's algorithm) — uses cached adjacency lists
// =============================================================================

std::vector<int> DAGCircuit::topological_sort() const {
    std::unordered_map<int, int> in_degree;
    in_degree.reserve(nodes.size());

    for (const auto& node : nodes) {
        in_degree[node.node_id] = 0;
    }

    for (const auto& [nid, preds] : adj_in) {
        in_degree[nid] = static_cast<int>(preds.size());
    }

    std::queue<int> q;
    for (const auto& [nid, deg] : in_degree) {
        if (deg == 0) q.push(nid);
    }

    std::vector<int> result;
    result.reserve(nodes.size());
    while (!q.empty()) {
        int nid = q.front();
        q.pop();
        result.push_back(nid);

        auto it = adj_out.find(nid);
        if (it != adj_out.end()) {
            for (int neighbor : it->second) {
                if (--in_degree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }
    }

    return result;
}

// =============================================================================
// Front layer — OP nodes with no OP predecessors (O(N) via adjacency + index)
// =============================================================================

std::vector<int> DAGCircuit::front_layer() const {
    std::vector<int> front;

    for (const auto& node : nodes) {
        if (node.type != DAGNode::Type::OP) continue;

        auto preds = predecessors(node.node_id);
        bool all_non_op = true;
        for (int pid : preds) {
            auto it = node_id_to_idx.find(pid);
            if (it != node_id_to_idx.end() && nodes[it->second].type == DAGNode::Type::OP) {
                all_non_op = false;
                break;
            }
        }
        if (all_non_op) {
            front.push_back(node.node_id);
        }
    }

    return front;
}

// =============================================================================
// Successors / Predecessors — O(1) via adjacency lists
// =============================================================================

std::vector<int> DAGCircuit::successors(int node_id) const {
    auto it = adj_out.find(node_id);
    if (it == adj_out.end()) return {};
    // Deduplicate (multi-wire edges may create duplicates)
    std::unordered_set<int> unique(it->second.begin(), it->second.end());
    return std::vector<int>(unique.begin(), unique.end());
}

std::vector<int> DAGCircuit::predecessors(int node_id) const {
    auto it = adj_in.find(node_id);
    if (it == adj_in.end()) return {};
    std::unordered_set<int> unique(it->second.begin(), it->second.end());
    return std::vector<int>(unique.begin(), unique.end());
}

// =============================================================================
// substitute_node — replace a node with a sub-DAG (per-wire edge routing)
// =============================================================================

void DAGCircuit::substitute_node(int node_id, const DAGCircuit& replacement) {
    auto idx_it = node_id_to_idx.find(node_id);
    if (idx_it == node_id_to_idx.end()) throw std::out_of_range("Node not found");

    // Get the incoming and outgoing edges for this node
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
        auto rit = replacement.node_id_to_idx.find(rid);
        if (rit != replacement.node_id_to_idx.end()) {
            const auto& rnode = replacement.nodes[rit->second];
            if (rnode.type == DAGNode::Type::OP) {
                DAGNode new_node = rnode;
                int new_id = add_node(std::move(new_node));
                old_to_new[rid] = new_id;
                new_op_ids.push_back(new_id);
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

    // Per-wire routing: connect in-edges to first new node on their wire,
    // and out-edges from last new node on their wire.
    if (!new_op_ids.empty()) {
        // Build per-wire first/last new node maps
        std::unordered_map<int, int> wire_first_new, wire_last_new;
        for (int nid : new_op_ids) {
            auto nit = node_id_to_idx.find(nid);
            if (nit == node_id_to_idx.end()) continue;
            const auto& n = nodes[nit->second];
            for (int w : n.qubit_wires) {
                if (!wire_first_new.count(w)) wire_first_new[w] = nid;
                wire_last_new[w] = nid;
            }
            for (int w : n.clbit_wires) {
                int key = w + 10000;  // offset to avoid collision with qubit wires
                if (!wire_first_new.count(key)) wire_first_new[key] = nid;
                wire_last_new[key] = nid;
            }
        }

        for (const auto& ie : in_edges) {
            int key = ie.is_classical ? (ie.wire + 10000) : ie.wire;
            int target = wire_first_new.count(key) ? wire_first_new[key] : new_op_ids.front();
            add_edge(ie.src_node, target, ie.wire, ie.is_classical);
        }
        for (const auto& oe : out_edges) {
            int key = oe.is_classical ? (oe.wire + 10000) : oe.wire;
            int source = wire_last_new.count(key) ? wire_last_new[key] : new_op_ids.back();
            add_edge(source, oe.dst_node, oe.wire, oe.is_classical);
        }
    } else {
        // No replacement ops — connect in to out directly
        for (const auto& ie : in_edges) {
            for (const auto& oe : out_edges) {
                if (ie.wire == oe.wire && ie.is_classical == oe.is_classical) {
                    add_edge(ie.src_node, oe.dst_node, ie.wire, ie.is_classical);
                }
            }
        }
    }

    // Remove the original node from nodes vector and rebuild index
    auto nit = node_id_to_idx.find(node_id);
    if (nit != node_id_to_idx.end()) {
        size_t idx = nit->second;
        nodes.erase(nodes.begin() + static_cast<ptrdiff_t>(idx));
        // Rebuild node_id_to_idx since indices shifted
        node_id_to_idx.clear();
        for (size_t i = 0; i < nodes.size(); ++i) {
            node_id_to_idx[nodes[i].node_id] = i;
        }
    }

    // Clean adjacency for removed node
    adj_out.erase(node_id);
    adj_in.erase(node_id);
    // Remove stale references (adj lists are rebuilt on next add_edge calls;
    // for correctness, strip node_id from all adj lists)
    for (auto& [k, v] : adj_out) {
        v.erase(std::remove(v.begin(), v.end(), node_id), v.end());
    }
    for (auto& [k, v] : adj_in) {
        v.erase(std::remove(v.begin(), v.end(), node_id), v.end());
    }
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

    // Remove the node and rebuild index/adjacency
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
            [node_id](const DAGNode& n) { return n.node_id == node_id; }),
        nodes.end()
    );

    // Clean adjacency
    adj_out.erase(node_id);
    adj_in.erase(node_id);
    for (auto& [k, v] : adj_out) {
        v.erase(std::remove(v.begin(), v.end(), node_id), v.end());
    }
    for (auto& [k, v] : adj_in) {
        v.erase(std::remove(v.begin(), v.end(), node_id), v.end());
    }

    // Rebuild node_id_to_idx
    node_id_to_idx.clear();
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_id_to_idx[nodes[i].node_id] = i;
    }
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

// depth — O(N+E) via topological sort + adjacency lists
int DAGCircuit::depth() const {
    auto sorted = topological_sort();
    std::unordered_map<int, int> dist;
    dist.reserve(nodes.size());
    for (const auto& node : nodes) {
        dist[node.node_id] = 0;
    }

    int max_depth = 0;
    for (int nid : sorted) {
        // O(1) node type lookup
        auto it = node_id_to_idx.find(nid);
        bool is_op = (it != node_id_to_idx.end() &&
                      nodes[it->second].type == DAGNode::Type::OP);

        // O(degree) successor traversal via adjacency list
        auto ait = adj_out.find(nid);
        if (ait != adj_out.end()) {
            int new_dist = dist[nid] + (is_op ? 1 : 0);
            for (int succ : ait->second) {
                if (new_dist > dist[succ]) {
                    dist[succ] = new_dist;
                    if (new_dist > max_depth) max_depth = new_dist;
                }
            }
        }
    }

    return max_depth;
}

} // namespace lindblad
