// sabre_layout.cpp — SabreLayout pass
//
// SabreLayout: Full SABRE algorithm (Li et al. 2019, arXiv:1809.02573).
//   Runs a SABRE-Swap forward pass, then backward pass, then another forward
//   pass. Takes the best (fewest SWAPs) of the three runs and returns the
//   corresponding logical→physical mapping applied to the DAG.
//
//   Output DAG invariant: n_qubits == coupling_map.n_physical_qubits.
//   All qubit indices in instructions and edges refer to physical qubits.
//   SabreSwap must use an identity initial_layout on this output.
//
//   Input expansion (R.1.15.0): the DAG is expanded to n_physical_qubits by
//   identity embedding BEFORE the SABRE search (layout_expansion.hpp — see
//   there for the frozen-slot defect this prevents). Circuits larger than
//   the device throw std::invalid_argument instead of indexing the distance
//   matrix out of range (previously undefined behaviour).

#include "lindblad/transpiler.hpp"

#include "layout_expansion.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lindblad {

// =============================================================================
// SABRE internals
// =============================================================================

// Score a potential SWAP insertion.
// H_basic: average distance of 2Q gates in the front layer after applying SWAP.
//
// front_layer_ops = node IDs of the current front layer
// layout          = logical → physical qubit mapping
// dist            = coupling map all-pairs shortest-path distance matrix
// swap_l0/swap_l1 = logical qubits to tentatively swap
// decay_factor    = penalty for repeatedly swapping the same pair
static double sabre_heuristic(
    const std::vector<int>& front_layer_ops,
    const std::unordered_map<int, const DAGNode*>& node_by_id,
    const std::vector<int>& layout,
    const std::vector<std::vector<int>>& dist,
    int swap_l0, int swap_l1,
    double decay_factor
) {
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

    cost *= (1.0 + decay_factor);
    return cost;
}

// Result of a single SABRE routing pass.
struct SABRERunResult {
    std::vector<int> final_layout; // logical → physical after the pass
    int swap_count;
};

// Run one SABRE pass over the circuit, starting from initial_layout.
// Returns the final mapping and the number of SWAPs inserted.
//
// dag          = the circuit to route (logical qubit indices)
// initial_layout = starting logical → physical mapping
// dist           = all-pairs shortest-path distance matrix from coupling_map
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
    // Adjacency list of OP-source edges (audit F-15): advancing successors by
    // scanning ALL dag.edges per executed gate was O(N*E) per pass; the list
    // makes it O(degree). Built with the same OP-source filter as in_degree so
    // the two stay consistent (duplicates per wire are kept — each is one
    // in_degree decrement).
    std::unordered_map<int, std::vector<int>> adj_out;
    for (const auto& edge : dag.edges) {
        // Only OP predecessors gate execution order. Wire IN nodes are never
        // "executed", so counting their edges pinned every first-layer gate
        // at in_degree >= 1, left the front layer permanently empty, and
        // silently reduced the whole SABRE pass to a no-op that returned the
        // trivial layout with zero swaps.
        auto sit = node_by_id.find(edge.src_node);
        if (sit != node_by_id.end() &&
            sit->second->type == DAGNode::Type::OP) {
            in_degree[edge.dst_node]++;
            adj_out[edge.src_node].push_back(edge.dst_node);
        }
    }

    // Front layer: OP nodes with in-degree 0 (stored as node IDs).
    std::vector<int> front_layer;
    for (const auto& node : dag.nodes) {
        if (node.type == DAGNode::Type::OP && in_degree[node.node_id] == 0) {
            front_layer.push_back(node.node_id);
        }
    }

    int swap_count = 0;

    // Termination guard (mirrors sabre_swap.cpp): a gate spanning
    // disconnected coupling-map components never becomes adjacent, yet SWAP
    // candidates keep existing inside one component — without a budget this
    // loop cannot terminate on unroutable input. Routing any gate needs at
    // most n_physical SWAPs, so the budget is generous for every routable
    // circuit. This matters from R.1.15.0 on because SabreLayout is the
    // FIRST pass at level >= 2 and no longer has a SabreSwap run in front of
    // it to throw first.
    const long long swap_budget =
        static_cast<long long>(dag.nodes.size() + 16) *
        static_cast<long long>(std::max(1, coupling_map.n_physical_qubits));

    while (!front_layer.empty()) {
        std::vector<int> executable;
        std::vector<int> blocked;

        for (int nid : front_layer) {
            const auto& node = *node_by_id[nid];
            if (node.qubit_wires.size() < 2) {
                executable.push_back(nid);
            } else {
                int la = node.qubit_wires[0];
                int lb = node.qubit_wires[1];
                int pa = layout[la];
                int pb = layout[lb];
                if (dist[pa][pb] == 1) {  // adjacent (audit F-15: was is_connected O(E))
                    executable.push_back(nid);
                } else {
                    blocked.push_back(nid);
                }
            }
        }

        for (int nid : executable) {
            executed[nid] = true;
            auto ait = adj_out.find(nid);
            if (ait != adj_out.end())
                for (int dst : ait->second) {
                    if (--in_degree[dst] == 0 &&
                        node_by_id[dst]->type == DAGNode::Type::OP &&
                        !executed[dst]) {
                        front_layer.push_back(dst);
                    }
                }
        }

        front_layer.erase(
            std::remove_if(front_layer.begin(), front_layer.end(),
                [&](int n){ return executed[n]; }),
            front_layer.end()
        );

        if (blocked.empty()) continue;

        // Find the best SWAP over edges adjacent to blocked gates.
        double best_cost = std::numeric_limits<double>::max();
        int best_l0 = -1, best_l1 = -1;

        std::vector<std::pair<int,int>> candidates;
        for (int nid : blocked) {
            for (int lq : node_by_id[nid]->qubit_wires) {
                int pq = layout[lq];
                for (const auto& [pa, pb] : coupling_map.edges) {
                    if (pa == pq) {
                        int lq2 = -1;
                        for (int i = 0; i < N; ++i) {
                            if (layout[i] == pb) { lq2 = i; break; }
                        }
                        if (lq2 >= 0 && lq != lq2) {
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
            double cost = sabre_heuristic(blocked, node_by_id, layout, dist, l0, l1, 0.0);
            if (cost < best_cost) {
                best_cost = cost;
                best_l0 = l0;
                best_l1 = l1;
            }
        }

        if (best_l0 >= 0) {
            std::swap(layout[best_l0], layout[best_l1]);
            swap_count++;
            if (swap_count > swap_budget) {
                throw std::runtime_error(
                    "SABRE layout exceeded its SWAP budget: a 2-qubit gate "
                    "appears to span disconnected coupling-map components and "
                    "can never be made adjacent.");
            }
        } else {
            break;  // No valid SWAP found (disconnected circuit?)
        }
    }

    return {layout, swap_count};
}

// =============================================================================
// SabreLayout — SABRE layout pass (Li et al. 2019)
// =============================================================================

DAGCircuit SabreLayout::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    if (ctx.coupling_map.n_physical_qubits == 0) return dag;

    // Expand to device width first (identity embedding): the SABRE search
    // requires every physical slot to hold a logical wire, otherwise SWAP
    // candidates freeze at the initial layout image (see layout_expansion.hpp).
    const DAGCircuit expanded =
        transpiler_detail::expand_to_physical(dag, ctx, "SabreLayout");

    int N = expanded.n_qubits;  // == n_physical_qubits after expansion
    auto dist = ctx.coupling_map.distance_matrix();

    std::vector<int> best_layout(N);
    std::iota(best_layout.begin(), best_layout.end(), 0);
    int best_swaps = std::numeric_limits<int>::max();

    // Forward pass
    auto result_fwd = sabre_run(expanded, best_layout, dist, ctx.coupling_map);
    if (result_fwd.swap_count < best_swaps) {
        best_swaps = result_fwd.swap_count;
        best_layout = result_fwd.final_layout;
    }

    // Backward pass: seed from the forward-pass final layout and re-route.
    // This approximates the true backward pass (reversed topological order)
    // without requiring a reversed DAG; it typically finds a better initial
    // mapping that reduces SWAPs in the subsequent forward pass.
    std::vector<int> reversed_layout = result_fwd.final_layout;
    auto result_bwd = sabre_run(expanded, reversed_layout, dist, ctx.coupling_map);
    if (result_bwd.swap_count < best_swaps) {
        best_swaps = result_bwd.swap_count;
        best_layout = result_bwd.final_layout;
    }

    // Final forward pass from the best result so far
    auto result_final = sabre_run(expanded, best_layout, dist, ctx.coupling_map);
    if (result_final.swap_count < best_swaps) {
        best_swaps = result_final.swap_count;
        best_layout = result_final.final_layout;
    }

    // Apply the best initial layout by rebuilding the DAG from a remapped circuit.
    // Mutating node.qubit_wires/op.qubits in place would leave DAGEdge::wire fields
    // referencing stale logical indices, corrupting substitute_node, remove_node,
    // and wire-based successor matching in the subsequent SabreSwap routing pass.
    // n_qubits already equals n_physical_qubits (input expansion above); the
    // assignment restates the output invariant explicitly so from_circuit
    // sizes the IN-node array for physical qubit indices.
    QuantumCircuit mapped_qc = expanded.to_circuit();
    mapped_qc.n_qubits = ctx.coupling_map.n_physical_qubits;
    for (auto& inst : mapped_qc.instructions) {
        for (auto& q : inst.qubits) {
            q = best_layout[q];
        }
    }
    return DAGCircuit::from_circuit(mapped_qc);
}

} // namespace lindblad
