// sabre_swap.cpp — SABRE SWAP routing with lookahead heuristic
//
// The routing algorithm (Li et al. 2019) works as follows:
//  1. Compute the "front layer" (2Q gates whose predecessors are all mapped).
//  2. Execute any gate whose logical qubits are adjacent in the current layout.
//  3. If blocked gates remain, score all candidate SWAP insertions using:
//        H = H_basic + W * H_extended
//     where H_basic sums the distance improvement for the front layer and
//     H_extended sums the distance improvement for the next "lookahead" layer.
//  4. Execute the best-scoring SWAP, update the layout, and repeat.
//
// StochasticSwap runs SABRE `trials` times with different random seeds for the
// initial randomised layout perturbation and returns the best result.

#include "lindblad/transpiler.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lindblad {

// =============================================================================
// Core SABRE routing pass
// Returns the routed DAGCircuit (with SWAPs inserted) and swap count.
// =============================================================================

struct SabreRoutingResult {
    DAGCircuit dag;
    int swap_count = 0;
};

static SabreRoutingResult sabre_route(
    const DAGCircuit& dag,
    const std::vector<int>& initial_layout,     // logical → physical
    const CouplingMap& coupling_map,
    const std::vector<std::vector<int>>& dist,
    double lookahead_weight = 0.5
) {
    int N = dag.n_qubits;
    std::vector<int> layout = initial_layout;

    // O(1) adjacency lookups for the routing hot loop. is_connected() scans
    // the edge vector linearly per query, which dominated large-map routing.
    std::unordered_set<long long> adj_pairs;
    adj_pairs.reserve(coupling_map.edges.size() * 2 + 1);
    const long long ADJ_KEY = 1000003LL;
    for (const auto& [ea, eb] : coupling_map.edges) {
        adj_pairs.insert(static_cast<long long>(ea) * ADJ_KEY + eb);
        adj_pairs.insert(static_cast<long long>(eb) * ADJ_KEY + ea);
    }
    auto is_adjacent = [&](int a, int b) {
        return adj_pairs.count(static_cast<long long>(a) * ADJ_KEY + b) > 0;
    };

    // Build logical reverse: physical → logical
    auto build_inv = [&]() {
        std::vector<int> inv(coupling_map.n_physical_qubits, -1);
        for (int l = 0; l < N; ++l) inv[layout[l]] = l;
        return inv;
    };

    // Build ID→index map: node IDs can diverge from vector positions after
    // prior passes that call remove_node/substitute_node on the DAG.
    const int num_nodes = static_cast<int>(dag.nodes.size());
    std::unordered_map<int, int> id_to_idx;
    int max_id = 0;
    for (int i = 0; i < num_nodes; ++i) {
        int nid = dag.nodes[i].node_id;
        id_to_idx[nid] = i;
        if (nid > max_id) max_id = nid;
    }
    const int id_range = max_id + 1;

    // Build adjacency list for O(1) successor lookup (O(N+E) total vs O(N*E)).
    // Vectors sized by id_range so node IDs are valid indices regardless of deletions.
    std::vector<std::vector<int>> adj_out(id_range);
    std::vector<int> in_deg(id_range, 0);
    for (const auto& e : dag.edges) {
        if (dag.nodes[id_to_idx.at(e.src_node)].type == DAGNode::Type::OP) {
            adj_out[e.src_node].push_back(e.dst_node);
            in_deg[e.dst_node]++;
        }
    }

    // Initial front layer — iterate by position, look up in_deg by node ID.
    std::vector<int> front;
    for (int i = 0; i < num_nodes; ++i) {
        int nid = dag.nodes[i].node_id;
        if (dag.nodes[i].type == DAGNode::Type::OP && in_deg[nid] == 0)
            front.push_back(nid);
    }

    // Build output DAG
    DAGCircuit out_dag = dag;
    // Clear all OP nodes' qubit wires — we'll rebuild them
    // (We track the routing in the layout and insert SWAP nodes)
    // For a clean approach: accumulate output instructions then rebuild
    std::vector<Instruction> out_instructions;
    std::vector<bool> done(id_range, false);
    int swap_count = 0;

    // Termination guard: a gate spanning disconnected coupling-map components
    // can never become adjacent, yet SWAP candidates may keep existing inside
    // one component. Routing any gate needs at most n_physical SWAPs, so this
    // budget is generous for every routable circuit and only trips on
    // genuinely unroutable input.
    const long long swap_budget =
        static_cast<long long>(num_nodes + 16) *
        static_cast<long long>(std::max(1, coupling_map.n_physical_qubits));

    // Decay parameters (per-qubit, penalise re-using same qubits for SWAPs)
    std::vector<double> decay(N, 1.0);
    constexpr double DECAY_FACTOR = 0.001;

    while (!front.empty()) {
        // Executable gates
        std::vector<int> executable, blocked;
        for (int nid : front) {
            const auto& node = dag.nodes[id_to_idx.at(nid)];
            const size_t nw = node.qubit_wires.size();
            if (nw < 2 || node.op.type == Instruction::GateType::BARRIER) {
                // 0/1-qubit operations and barriers carry no routing
                // constraint (a full-register BARRIER previously fell into
                // the 2-qubit branch and could block routing forever).
                executable.push_back(nid);
            } else if (nw == 2) {
                int la = node.qubit_wires[0];
                int lb = node.qubit_wires[1];
                int pa = layout[la];
                int pb = layout[lb];
                if (is_adjacent(pa, pb)) {
                    executable.push_back(nid);
                } else {
                    blocked.push_back(nid);
                }
            } else {
                // 3+ qubit gates: SABRE only inserts SWAPs for pairs.
                // Executable iff EVERY wire pair is already adjacent;
                // otherwise refuse loudly (decompose to 1q/2q gates before
                // routing) instead of validating only the first two wires.
                bool all_adjacent = true;
                for (size_t a = 0; a < nw && all_adjacent; ++a)
                    for (size_t b = a + 1; b < nw && all_adjacent; ++b)
                        if (!is_adjacent(layout[node.qubit_wires[a]],
                                         layout[node.qubit_wires[b]]))
                            all_adjacent = false;
                if (all_adjacent) {
                    executable.push_back(nid);
                } else {
                    throw std::runtime_error(
                        "SABRE routing: " + std::to_string(nw) +
                        "-qubit gate '" + node.op.gate_name() +
                        "' is not executable on this coupling map; decompose "
                        "multi-qubit gates to 1q/2q gates before routing");
                }
            }
        }

        for (int nid : executable) {
            // Emit the gate with current physical qubit mapping
            const auto& enode = dag.nodes[id_to_idx.at(nid)];
            Instruction inst = enode.op;
            for (size_t qi = 0; qi < inst.qubits.size(); ++qi) {
                inst.qubits[qi] = layout[enode.qubit_wires[qi]];
            }
            out_instructions.push_back(inst);
            done[nid] = true;

            // Reduce decay for used qubits
            for (int lq : enode.qubit_wires) decay[lq] = 1.0;

            // Advance successors using adjacency list — O(degree) not O(E)
            for (int succ : adj_out[nid]) {
                if (--in_deg[succ] == 0 &&
                    dag.nodes[id_to_idx.at(succ)].type == DAGNode::Type::OP &&
                    !done[succ]) {
                    front.push_back(succ);
                }
            }
        }

        front.erase(std::remove_if(front.begin(), front.end(),
            [&](int n){ return done[n]; }), front.end());

        if (blocked.empty()) continue;

        // Increase decay for qubits involved in blocked gates
        for (int nid : blocked) {
            for (int lq : dag.nodes[id_to_idx.at(nid)].qubit_wires) {
                decay[lq] = std::min(decay[lq] + DECAY_FACTOR, 1.0 + DECAY_FACTOR * 10);
            }
        }

        // Collect candidate SWAPs from edges adjacent to blocked logical qubits
        std::vector<std::pair<int,int>> candidates;
        for (int nid : blocked) {
            for (int lq : dag.nodes[id_to_idx.at(nid)].qubit_wires) {
                int pq = layout[lq];
                auto inv = build_inv();
                for (const auto& [pa, pb] : coupling_map.edges) {
                    if (pa == pq) {
                        int other_l = inv[pb];
                        if (other_l >= 0 && other_l != lq) {
                            auto c = std::make_pair(std::min(lq, other_l), std::max(lq, other_l));
                            if (std::find(candidates.begin(), candidates.end(), c) == candidates.end())
                                candidates.push_back(c);
                        }
                    }
                }
            }
        }

        // Build extended (lookahead) layer: successors of blocked nodes
        std::unordered_set<int> extended_set;
        for (int nid : blocked) {
            for (int succ : adj_out[nid]) {
                if (dag.nodes[id_to_idx.at(succ)].qubit_wires.size() >= 2)
                    extended_set.insert(succ);
            }
        }
        std::vector<int> extended(extended_set.begin(), extended_set.end());

        // Score candidates
        double best_score = std::numeric_limits<double>::max();
        int best_l0 = -1, best_l1 = -1;

        for (auto [l0, l1] : candidates) {
            // Tentative SWAP
            std::vector<int> tentative = layout;
            std::swap(tentative[l0], tentative[l1]);

            // H_basic: sum distances in blocked front layer
            double h_basic = 0.0;
            for (int nid : blocked) {
                const auto& wires = dag.nodes[id_to_idx.at(nid)].qubit_wires;
                int pa = tentative[wires[0]], pb = tentative[wires[1]];
                h_basic += static_cast<double>(dist[pa][pb]);
            }

            // H_extended: lookahead into next layer
            double h_ext = 0.0;
            for (int nid : extended) {
                const auto& wires = dag.nodes[id_to_idx.at(nid)].qubit_wires;
                if (wires.size() >= 2) {
                    int pa = tentative[wires[0]], pb = tentative[wires[1]];
                    h_ext += static_cast<double>(dist[pa][pb]);
                }
            }
            if (!extended.empty()) h_ext /= static_cast<double>(extended.size());

            double score = (h_basic + lookahead_weight * h_ext) *
                           std::max(decay[l0], decay[l1]);

            if (score < best_score) {
                best_score = score;
                best_l0 = l0;
                best_l1 = l1;
            }
        }

        if (best_l0 < 0) {
            // No SWAP candidate can make progress. Throwing here is the
            // no-silent-failure contract: the previous `break` discarded
            // every not-yet-emitted gate from the routed circuit.
            throw std::runtime_error(
                "SABRE routing stalled: the coupling map cannot host the "
                "remaining 2-qubit gates (no SWAP candidates). Note that an "
                "edgeless CouplingMap(n) declares a graph with NO allowed "
                "pairs; pass CouplingMap() for unconstrained routing or "
                "CouplingMap::all_to_all(n) for full connectivity.");
        }

        // Apply the SWAP
        int p0 = layout[best_l0], p1 = layout[best_l1];
        std::swap(layout[best_l0], layout[best_l1]);
        swap_count++;
        if (swap_count > swap_budget) {
            throw std::runtime_error(
                "SABRE routing exceeded its SWAP budget: a 2-qubit gate "
                "appears to span disconnected coupling-map components and "
                "can never be made adjacent.");
        }

        // Emit a SWAP instruction
        Instruction sw;
        sw.type = Instruction::GateType::SWAP;
        sw.qubits = {p0, p1};
        out_instructions.push_back(sw);
    }

    // Rebuild output DAG from instructions
    QuantumCircuit out_qc(dag.n_qubits, dag.n_clbits);
    out_qc.instructions = out_instructions;
    return {DAGCircuit::from_circuit(out_qc), swap_count};
}

// =============================================================================
// SabreSwap
// =============================================================================

DAGCircuit SabreSwap::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    if (ctx.coupling_map.n_physical_qubits == 0) return dag;

    int N = dag.n_qubits;
    auto dist = ctx.coupling_map.distance_matrix();

    // Use the initial_layout from context if provided, else trivial
    std::vector<int> layout;
    if (!ctx.initial_layout.empty() && static_cast<int>(ctx.initial_layout.size()) >= N) {
        layout = std::vector<int>(ctx.initial_layout.begin(), ctx.initial_layout.begin() + N);
    } else {
        layout.resize(N);
        std::iota(layout.begin(), layout.end(), 0);
    }

    auto result = sabre_route(dag, layout, ctx.coupling_map, dist, 0.5);
    return result.dag;
}

// =============================================================================
// StochasticSwap — run SABRE with multiple random initial layouts, take best
// =============================================================================

DAGCircuit StochasticSwap::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    if (ctx.coupling_map.n_physical_qubits == 0) return dag;

    int N = dag.n_qubits;
    auto dist = ctx.coupling_map.distance_matrix();

    std::mt19937 rng(seed);
    int best_swaps = std::numeric_limits<int>::max();
    DAGCircuit best_dag = dag;

    for (int t = 0; t < trials; ++t) {
        // Generate a random layout
        std::vector<int> layout(N);
        std::iota(layout.begin(), layout.end(), 0);
        if (N <= ctx.coupling_map.n_physical_qubits) {
            std::shuffle(layout.begin(), layout.end(), rng);
        }

        auto result = sabre_route(dag, layout, ctx.coupling_map, dist, 0.5);
        if (result.swap_count < best_swaps) {
            best_swaps = result.swap_count;
            best_dag = result.dag;
        }
    }

    return best_dag;
}

} // namespace lindblad
