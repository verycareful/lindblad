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

CouplingMap CouplingMap::heavy_hex(int n_qubits) {
    // IBM heavy-hex coupling map (Eagle/Falcon/Heron device topology).
    // The pattern is a hexagonal lattice where:
    // - "data" qubits have degree 2 or 3
    // - "ancilla" qubits (between data qubits on long edges) have degree 2
    //
    // The pattern for an n-qubit heavy-hex proceeds as follows using the
    // canonical IBM numbering (see arXiv:2002.12718 and IBM device specs):
    //
    // For a row-based heavy-hex with `cols` data cols per row:
    //   Row spacing: data qubits at positions 0..cols-1,
    //   ancilla qubits interspersed horizontally and vertically.
    //
    // We build the standard IBM heavy-hex unit cell pattern up to n_qubits.
    // For sizes that don't fit cleanly, we truncate.

    CouplingMap cm(n_qubits);
    if (n_qubits <= 1) return cm;

    // Heavy-hex standard connectivity pattern for IBM devices:
    // Columns of data qubits connected by horizontal ancillas (even-numbered rows)
    // and vertical connections between data rows.
    //
    // For a linear heavy-hex (1D version, e.g. 5-qubit Falcon):
    // 0 - 1 - 2 - 3 - 4  (all connected linearly)
    // For a full 2D heavy-hex, e.g., 27-qubit Falcon r4:
    //   The connectivity is defined by the specific IBM heavy-hex graph.
    //
    // Here we generate the standard IBM heavy-hex graph for up to n_qubits.
    // The connectivity is defined by fixing a row width of 4 data qubits
    // plus 3 ancillas between them per row, with anchoring ancillas between rows.

    // For robustness, build heavy-hex with standard dimensions.
    // The pattern uses rows of width W with ancillas.
    // Each "super-row" has: W data + (W-1) ancilla horizontal = 2W-1 physical slots
    // Between super-rows: W ancillas

    // Find best dimensions
    // Standard: n=27 → 3 rows of 7 data + 6 ancilla each + 7 inter-row ancillas
    // We'll use a simplified but correct pattern: generate up to N edges.

    // Build as: data qubit rows separated by ancilla columns
    // Pattern index: r*cols + c for "data" qubits
    // Ancilla between (r,c) and (r,c+1): has index data_n + r*(cols-1) + c
    // Ancilla between row r and row r+1 at column c (if c is connection column):
    //   index depends on row

    // IBM specific: use cols=7 (Eagle: 127q), cols=4 (Falcon: 27q), etc.
    int cols = 4;  // default: Falcon-like (27q) pattern
    if (n_qubits >= 100) cols = 7;   // Eagle-like
    else if (n_qubits >= 50) cols = 5;

    int data_per_row = cols;
    int anc_per_row_horiz = cols - 1;
    int row_qubit_count = data_per_row + anc_per_row_horiz;  // = 2*cols - 1

    // Inter-row ancillas: placed at alternating columns
    // Even inter-rows: columns 1, 3, 5, ...
    // Odd inter-rows: columns 0, 2, 4, ...

    std::vector<std::pair<int,int>> edge_list;
    auto add_edge = [&](int a, int b) {
        if (a < n_qubits && b < n_qubits) {
            edge_list.push_back({a, b});
            edge_list.push_back({b, a});
        }
    };

    int qubit = 0;
    int rows = (n_qubits + row_qubit_count - 1) / row_qubit_count;

    // For each row of data+horizontal-ancilla qubits
    // and inter-row ancilla qubits

    // Total per "super-row block" = row_qubit_count (data+horiz-anc)
    //                              + cols-1 (inter-row vertical ancillas, every other col)
    // For simplicity: build flat linear indices

    // --- Simple correct approach: build vertex indices for the grid ---

    struct Qubit { int row, col, type; };  // type: 0=data, 1=anc_horiz, 2=anc_vert
    std::vector<Qubit> qubits_list;
    std::vector<std::vector<int>> data_idx(rows, std::vector<int>(cols, -1));
    std::vector<std::vector<int>> anc_h_idx(rows, std::vector<int>(cols-1, -1));
    // Vertical ancillas between row r and r+1, at column c (only for connecting cols)
    std::vector<std::vector<int>> anc_v_idx(rows-1, std::vector<int>(cols, -1));

    int idx = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            data_idx[r][c] = idx++;
            if (idx >= n_qubits) break;
            if (c < cols - 1) {
                anc_h_idx[r][c] = idx++;
                if (idx >= n_qubits) break;
            }
        }
        if (idx >= n_qubits) break;
        if (r < rows - 1) {
            // Inter-row ancillas: every other column, alternating per row parity
            for (int c = (r % 2); c < cols; c += 2) {
                anc_v_idx[r][c] = idx++;
                if (idx >= n_qubits) break;
            }
        }
        if (idx >= n_qubits) break;
    }

    // Add horizontal edges: data[r][c] -- anc_h[r][c] -- data[r][c+1]
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols - 1; ++c) {
            int d0 = data_idx[r][c];
            int ah = anc_h_idx[r][c];
            int d1 = data_idx[r][c+1];
            if (d0 >= 0 && ah >= 0) add_edge(d0, ah);
            if (ah >= 0 && d1 >= 0) add_edge(ah, d1);
        }
    }

    // Add vertical edges: data[r][c] -- anc_v[r][c] -- data[r+1][c]
    for (int r = 0; r < rows - 1; ++r) {
        for (int c = 0; c < cols; ++c) {
            int av = anc_v_idx[r][c];
            if (av < 0) continue;
            int d0 = data_idx[r][c];
            int d1 = data_idx[r+1][c];
            if (d0 >= 0) add_edge(d0, av);
            if (d1 >= 0) add_edge(av, d1);
        }
    }

    // Deduplicate and copy to cm
    std::vector<std::pair<int,int>> seen;
    for (const auto& [a, b] : edge_list) {
        auto key = std::make_pair(std::min(a,b), std::max(a,b));
        if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
            seen.push_back(key);
            cm.edges.push_back({a, b});
            cm.edges.push_back({b, a});
        }
    }

    return cm;
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
