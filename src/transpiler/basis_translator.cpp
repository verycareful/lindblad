#include "qpp/transpiler.hpp"

#include <algorithm>
#include <queue>
#include <limits>

namespace qpp {

// =============================================================================
// CouplingMap
// =============================================================================

bool CouplingMap::is_connected(int q1, int q2) const {
    for (const auto& [a, b] : edges) {
        if ((a == q1 && b == q2) || (a == q2 && b == q1)) return true;
    }
    return false;
}

std::vector<int> CouplingMap::shortest_path(int q1, int q2) const {
    if (q1 == q2) return {q1};

    // BFS
    std::vector<int> prev(n_physical_qubits, -1);
    std::vector<bool> visited(n_physical_qubits, false);
    std::queue<int> bfs;
    bfs.push(q1);
    visited[q1] = true;

    while (!bfs.empty()) {
        int curr = bfs.front();
        bfs.pop();

        for (const auto& [a, b] : edges) {
            int neighbor = -1;
            if (a == curr) neighbor = b;
            else if (b == curr) neighbor = a;
            else continue;

            if (!visited[neighbor]) {
                visited[neighbor] = true;
                prev[neighbor] = curr;
                if (neighbor == q2) {
                    // Reconstruct path
                    std::vector<int> path;
                    for (int n = q2; n != -1; n = prev[n]) {
                        path.push_back(n);
                    }
                    std::reverse(path.begin(), path.end());
                    return path;
                }
                bfs.push(neighbor);
            }
        }
    }

    return {};  // not connected
}

std::vector<std::vector<int>> CouplingMap::distance_matrix() const {
    std::vector<std::vector<int>> dist(
        n_physical_qubits,
        std::vector<int>(n_physical_qubits, std::numeric_limits<int>::max() / 2)
    );

    for (int i = 0; i < n_physical_qubits; ++i) dist[i][i] = 0;

    for (const auto& [a, b] : edges) {
        dist[a][b] = 1;
        dist[b][a] = 1;
    }

    // Floyd-Warshall
    for (int k = 0; k < n_physical_qubits; ++k) {
        for (int i = 0; i < n_physical_qubits; ++i) {
            for (int j = 0; j < n_physical_qubits; ++j) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    return dist;
}

bool CouplingMap::is_connected_graph() const {
    if (n_physical_qubits <= 1) return true;
    auto path = shortest_path(0, n_physical_qubits - 1);
    return !path.empty();
}

CouplingMap CouplingMap::linear(int n) {
    CouplingMap cm(n);
    for (int i = 0; i < n - 1; ++i) {
        cm.edges.push_back({i, i + 1});
        cm.edges.push_back({i + 1, i});
    }
    return cm;
}

CouplingMap CouplingMap::grid(int rows, int cols) {
    int n = rows * cols;
    CouplingMap cm(n);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            if (c + 1 < cols) {
                cm.edges.push_back({idx, idx + 1});
                cm.edges.push_back({idx + 1, idx});
            }
            if (r + 1 < rows) {
                cm.edges.push_back({idx, idx + cols});
                cm.edges.push_back({idx + cols, idx});
            }
        }
    }
    return cm;
}

CouplingMap CouplingMap::heavy_hex(int n) {
    // Simplified heavy-hex: just use a grid for now
    return grid(n, n);
}

CouplingMap CouplingMap::all_to_all(int n) {
    CouplingMap cm(n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            cm.edges.push_back({i, j});
            cm.edges.push_back({j, i});
        }
    }
    return cm;
}

} // namespace qpp
