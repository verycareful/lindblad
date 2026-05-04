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

    // Build logical reverse: physical → logical
    auto build_inv = [&]() {
        std::vector<int> inv(coupling_map.n_physical_qubits, -1);
        for (int l = 0; l < N; ++l) inv[layout[l]] = l;
        return inv;
    };

    // Track predecessor count — only OP→OP edges; IN→OP edges must not count
    // or the first gate layer will never reach in_deg==0 and front stays empty.
    std::vector<int> in_deg(dag.nodes.size(), 0);
    for (const auto& e : dag.edges) {
        if (dag.nodes[e.src_node].type == DAGNode::Type::OP)
            in_deg[e.dst_node]++;
    }

    // Initial front layer
    std::vector<int> front;
    for (size_t i = 0; i < dag.nodes.size(); ++i) {
        if (dag.nodes[i].type == DAGNode::Type::OP && in_deg[i] == 0)
            front.push_back(static_cast<int>(i));
    }

    // Build output DAG
    DAGCircuit out_dag = dag;
    // Clear all OP nodes' qubit wires — we'll rebuild them
    // (We track the routing in the layout and insert SWAP nodes)
    // For a clean approach: accumulate output instructions then rebuild
    std::vector<Instruction> out_instructions;
    std::vector<bool> done(dag.nodes.size(), false);
    int swap_count = 0;

    // Decay parameters (per-qubit, penalise re-using same qubits for SWAPs)
    std::vector<double> decay(N, 1.0);
    constexpr double DECAY_FACTOR = 0.001;

    while (!front.empty()) {
        // Executable gates
        std::vector<int> executable, blocked;
        for (int nid : front) {
            const auto& node = dag.nodes[nid];
            if (node.qubit_wires.size() < 2) {
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

        for (int nid : executable) {
            // Emit the gate with current physical qubit mapping
            Instruction inst = dag.nodes[nid].op;
            for (size_t qi = 0; qi < inst.qubits.size(); ++qi) {
                inst.qubits[qi] = layout[dag.nodes[nid].qubit_wires[qi]];
            }
            out_instructions.push_back(inst);
            done[nid] = true;

            // Reduce decay for used qubits
            for (int lq : dag.nodes[nid].qubit_wires) decay[lq] = 1.0;

            // Advance successors
            for (const auto& e : dag.edges) {
                if (e.src_node == nid) {
                    in_deg[e.dst_node]--;
                    if (in_deg[e.dst_node] == 0 &&
                        dag.nodes[e.dst_node].type == DAGNode::Type::OP &&
                        !done[e.dst_node]) {
                        front.push_back(e.dst_node);
                    }
                }
            }
        }

        front.erase(std::remove_if(front.begin(), front.end(),
            [&](int n){ return done[n]; }), front.end());

        if (blocked.empty()) continue;

        // Increase decay for qubits involved in blocked gates
        for (int nid : blocked) {
            for (int lq : dag.nodes[nid].qubit_wires) {
                decay[lq] = std::min(decay[lq] + DECAY_FACTOR, 1.0 + DECAY_FACTOR * 10);
            }
        }

        // Collect candidate SWAPs from edges adjacent to blocked logical qubits
        std::vector<std::pair<int,int>> candidates;
        for (int nid : blocked) {
            for (int lq : dag.nodes[nid].qubit_wires) {
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
            for (const auto& e : dag.edges) {
                if (e.src_node == nid && dag.nodes[e.dst_node].qubit_wires.size() >= 2) {
                    extended_set.insert(e.dst_node);
                }
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
                const auto& wires = dag.nodes[nid].qubit_wires;
                int pa = tentative[wires[0]], pb = tentative[wires[1]];
                h_basic += static_cast<double>(dist[pa][pb]);
            }

            // H_extended: lookahead into next layer
            double h_ext = 0.0;
            for (int nid : extended) {
                const auto& wires = dag.nodes[nid].qubit_wires;
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

        if (best_l0 < 0) break;  // No progress

        // Apply the SWAP
        int p0 = layout[best_l0], p1 = layout[best_l1];
        std::swap(layout[best_l0], layout[best_l1]);
        swap_count++;

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
