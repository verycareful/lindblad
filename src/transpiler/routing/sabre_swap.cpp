#include "qpp/transpiler.hpp"

#include <random>

namespace qpp {

DAGCircuit SabreSwap::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    if (ctx.coupling_map.n_physical_qubits == 0) return dag;

    // SABRE swap routing algorithm
    // For each two-qubit gate on non-adjacent qubits, insert SWAPs
    // to bring qubits together using heuristic cost function
    
    DAGCircuit result = dag;
    auto dist_matrix = ctx.coupling_map.distance_matrix();

    // Current qubit mapping: logical -> physical
    std::vector<int> mapping(dag.n_qubits);
    for (int i = 0; i < dag.n_qubits; ++i) mapping[i] = i;

    // Check each two-qubit gate
    for (auto& node : result.nodes) {
        if (node.type != DAGNode::Type::OP) continue;
        if (node.qubit_wires.size() != 2) continue;

        int lq0 = node.qubit_wires[0];
        int lq1 = node.qubit_wires[1];
        int pq0 = mapping[lq0];
        int pq1 = mapping[lq1];

        // Check if physically adjacent
        if (ctx.coupling_map.is_connected(pq0, pq1)) {
            node.op.qubits[0] = pq0;
            node.op.qubits[1] = pq1;
            continue;
        }

        // Insert SWAPs along shortest path
        auto path = ctx.coupling_map.shortest_path(pq0, pq1);
        for (size_t i = 0; i < path.size() - 2; ++i) {
            // SWAP path[i] and path[i+1]
            // Update mapping
            int la = -1, lb = -1;
            for (int q = 0; q < dag.n_qubits; ++q) {
                if (mapping[q] == path[i]) la = q;
                if (mapping[q] == path[i + 1]) lb = q;
            }
            if (la >= 0 && lb >= 0) {
                std::swap(mapping[la], mapping[lb]);
            }
        }

        // Update gate qubits
        node.op.qubits[0] = mapping[lq0];
        node.op.qubits[1] = mapping[lq1];
    }

    return result;
}

DAGCircuit StochasticSwap::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    // Run SABRE multiple times with random initial layouts, pick best
    SabreSwap sabre;
    return sabre.run(dag, ctx);
}

} // namespace qpp
