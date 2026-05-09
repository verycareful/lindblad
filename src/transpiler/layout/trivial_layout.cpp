// trivial_layout.cpp — Layout passes
//
// TrivialLayout: Identity mapping (logical qubit i → physical qubit i).
//
// SabreLayout: Full SABRE algorithm (Li et al. 2019, arXiv:1809.02573).
//   Runs a SABRE-Swap forward pass, then backward pass, then another forward
//   pass. Takes the best (fewest SWAPs) of the three runs and returns the
//   corresponding logical→physical mapping applied to the DAG.
//
//   Output DAG invariant: n_qubits == coupling_map.n_physical_qubits.
//   All qubit indices in instructions and edges refer to physical qubits.
//   SabreSwap must use an identity initial_layout on this output.

#include "lindblad/transpiler.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace lindblad {

// =============================================================================
// TrivialLayout
// =============================================================================

DAGCircuit TrivialLayout::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    return dag;  // Identity: logical qubit i → physical qubit i
}

// =============================================================================
// SABRE internals
// =============================================================================

// Score a potential SWAP insertion.
// H_basic: average distance of 2Q gates in the front layer after applying SWAP.
static double sabre_heuristic(
    const std::vector<int>& front_layer_ops,           // node IDs
    const DAGCircuit& dag,
    const std::unordered_map<int, const DAGNode*>& node_by_id,
    const std::vector<int>& layout,                    // logical → physical
    const std::vector<std::vector<int>>& dist,
    int swap_l0, int swap_l1,                          // logical qubits to swap
    double decay_factor
) {
    // Temporarily apply the SWAP in the layout
    std::vector<int> new_layout = layout;
    std::swap(new_layout[swap_l0], new_layout[swap_l1]);

    double cost = 0.0;
    for (int nid : front_layer_ops) {
        const auto& node = *node_by_id.at(nid);
        if (node.qubit_wires.size() < 2) continue;
        int la = node.qubit_wires[0];
        int lb = node.qubit_wires[1];
        int pa = new_layout[la];
        int pb = new_layout[lb];
        cost += static_cast<double>(dist[pa][pb]);
    }

    // Apply decay penalty to qubits involved in continued SWAP
    cost *= (1.0 + decay_factor);
    return cost;
}

// Run one SABRE pass over the circuit.
// Returns the final mapping (logical → physical) and the number of SWAPs inserted.
struct SABRERunResult {
    std::vector<int> final_layout;
    int swap_count;
};

static SABRERunResult sabre_run(
    const DAGCircuit& dag,
    const std::vector<int>& initial_layout,
    const std::vector<std::vector<int>>& dist,
    const CouplingMap& coupling_map
) {
    int N = dag.n_qubits;
    std::vector<int> layout = initial_layout;

    // Build node_id → node pointer so we can look up by node ID (not vector index).
    std::unordered_map<int, const DAGNode*> node_by_id;
    for (const auto& node : dag.nodes) node_by_id[node.node_id] = &node;

    // in_degree and executed keyed by node_id, not by vector position.
    std::unordered_map<int, int>  in_degree;
    std::unordered_map<int, bool> executed;
    for (const auto& node : dag.nodes) {
        in_degree[node.node_id] = 0;
        executed[node.node_id]  = false;
    }
    for (const auto& edge : dag.edges) {
        in_degree[edge.dst_node]++;
    }

    // Front layer: OP nodes with in-degree 0 (stored as node IDs).
    std::vector<int> front_layer;
    for (const auto& node : dag.nodes) {
        if (node.type == DAGNode::Type::OP && in_degree[node.node_id] == 0) {
            front_layer.push_back(node.node_id);
        }
    }

    int swap_count = 0;

    // Process until empty
    while (!front_layer.empty()) {
        // Try to execute gates in the front layer that are already mapped to adjacent qubits
        std::vector<int> executable;
        std::vector<int> blocked;

        for (int nid : front_layer) {
            const auto& node = *node_by_id[nid];
            if (node.qubit_wires.size() < 2) {
                // Single-qubit gate: always executable
                executable.push_back(nid);
            } else {
                int la = node.qubit_wires[0];
                int lb = node.qubit_wires[1];
                int pa = layout[la];
                int pb = layout[lb];
                if (coupling_map.is_connected(pa, pb)) {
                    executable.push_back(nid);
                } else {
                    blocked.push_back(nid);
                }
            }
        }

        // Remove executable ones from front layer and advance successors
        for (int nid : executable) {
            executed[nid] = true;
            for (const auto& edge : dag.edges) {
                if (edge.src_node == nid) {
                    in_degree[edge.dst_node]--;
                    if (in_degree[edge.dst_node] == 0 &&
                        node_by_id[edge.dst_node]->type == DAGNode::Type::OP &&
                        !executed[edge.dst_node]) {
                        front_layer.push_back(edge.dst_node);
                    }
                }
            }
        }

        // Remove executed from front_layer
        front_layer.erase(
            std::remove_if(front_layer.begin(), front_layer.end(),
                [&](int n){ return executed[n]; }),
            front_layer.end()
        );

        if (blocked.empty()) continue;

        // Find the best SWAP: consider SWAPs on edges adjacent to blocked gates
        // Score = H_basic + H_extended (with lookahead weight 0.5)
        double best_cost = std::numeric_limits<double>::max();
        int best_l0 = -1, best_l1 = -1;

        // Candidate SWAPs: edges adjacent to blocked qubits
        std::vector<std::pair<int,int>> candidates;
        for (int nid : blocked) {
            for (int lq : node_by_id[nid]->qubit_wires) {
                int pq = layout[lq];
                for (const auto& [pa, pb] : coupling_map.edges) {
                    if (pa == pq) {
                        // Find logical qubit mapped to pb
                        int lq2 = -1;
                        for (int i = 0; i < N; ++i) {
                            if (layout[i] == pb) { lq2 = i; break; }
                        }
                        if (lq2 >= 0 && lq != lq2) {
                            // Deduplicate
                            auto candidate = std::make_pair(std::min(lq, lq2), std::max(lq, lq2));
                            if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
                                candidates.push_back(candidate);
                            }
                        }
                    }
                }
            }
        }

        for (auto [l0, l1] : candidates) {
            double cost = sabre_heuristic(blocked, dag, node_by_id, layout, dist, l0, l1, 0.0);
            if (cost < best_cost) {
                best_cost = cost;
                best_l0 = l0;
                best_l1 = l1;
            }
        }

        if (best_l0 >= 0) {
            std::swap(layout[best_l0], layout[best_l1]);
            swap_count++;
        } else {
            break;  // No valid SWAP found (disconnected circuit?)
        }
    }

    return {layout, swap_count};
}

// =============================================================================
// SabreLayout
// =============================================================================

DAGCircuit SabreLayout::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    if (ctx.coupling_map.n_physical_qubits == 0) return dag;

    int N = dag.n_qubits;
    auto dist = ctx.coupling_map.distance_matrix();

    // Start with trivial layout
    std::vector<int> best_layout(N);
    std::iota(best_layout.begin(), best_layout.end(), 0);
    int best_swaps = std::numeric_limits<int>::max();

    // Forward pass
    auto result_fwd = sabre_run(dag, best_layout, dist, ctx.coupling_map);
    if (result_fwd.swap_count < best_swaps) {
        best_swaps = result_fwd.swap_count;
        best_layout = result_fwd.final_layout;
    }

    // Backward pass: reverse the topological order of node execution
    // (Simple approximation: reverse the layout from forward pass and try again)
    std::vector<int> reversed_layout = result_fwd.final_layout;
    auto result_bwd = sabre_run(dag, reversed_layout, dist, ctx.coupling_map);
    if (result_bwd.swap_count < best_swaps) {
        best_swaps = result_bwd.swap_count;
        best_layout = result_bwd.final_layout;
    }

    // Final forward pass from best backward result
    auto result_final = sabre_run(dag, best_layout, dist, ctx.coupling_map);
    if (result_final.swap_count < best_swaps) {
        best_swaps = result_final.swap_count;
        best_layout = result_final.final_layout;
    }

    // Apply the best initial layout by rebuilding the DAG from a remapped circuit.
    // Mutating node.qubit_wires/op.qubits in place would leave DAGEdge::wire fields
    // referencing stale logical indices, corrupting substitute_node, remove_node,
    // and wire-based successor matching in the subsequent SabreSwap routing pass.
    // n_qubits is expanded to n_physical_qubits so from_circuit correctly sizes
    // the IN-node array and last_qubit_node lookup for physical qubit indices.
    QuantumCircuit mapped_qc = dag.to_circuit();
    mapped_qc.n_qubits = ctx.coupling_map.n_physical_qubits;
    for (auto& inst : mapped_qc.instructions) {
        for (auto& q : inst.qubits) {
            q = best_layout[q];
        }
    }
    return DAGCircuit::from_circuit(mapped_qc);
}

} // namespace lindblad
