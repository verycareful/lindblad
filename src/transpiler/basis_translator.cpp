#include "lindblad/transpiler.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace lindblad {

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
    const int n = n_physical_qubits;
    const int INF = std::numeric_limits<int>::max() / 2;
    std::vector<std::vector<int>> dist(n, std::vector<int>(n, INF));

    // BFS from each source (audit F-25): O(V·(V+E)) beats Floyd-Warshall's
    // O(V^3) for the sparse device graphs this is used on (e.g. a 127-qubit
    // heavy-hex has ~144 edges, not ~8000). Build the adjacency list once.
    std::vector<std::vector<int>> adj(n);
    for (const auto& [a, b] : edges) {
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    std::queue<int> bfs;
    for (int src = 0; src < n; ++src) {
        auto& d = dist[src];
        d[src] = 0;
        bfs.push(src);
        while (!bfs.empty()) {
            const int u = bfs.front();
            bfs.pop();
            for (int v : adj[u]) {
                if (d[v] > d[u] + 1) {
                    d[v] = d[u] + 1;
                    bfs.push(v);
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

    // Deduplicate and copy to cm — O(1) lookup via hash set
    std::unordered_set<uint64_t> seen;
    for (const auto& [a, b] : edge_list) {
        uint64_t key = (static_cast<uint64_t>(std::min(a,b)) << 32) |
                        static_cast<uint64_t>(std::max(a,b));
        if (seen.insert(key).second) {
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

// =============================================================================
// BasisTranslator::run
//
// Decomposes all gates not in ctx.basis_gates into the target basis using a
// fixed equivalence library. Default target: {"cx", "u3"} (IBM standard).
//
// Each gate is decomposed into CX + U3 or a subset thereof using standard
// textbook / Qiskit-equivalent decompositions. Single-qubit gates are mapped
// to U3. Two-qubit non-CX gates are decomposed into CX + U3 sequences.
//
// Contract (R.1.15.0): the OUTPUT is verified — every gate in the returned
// circuit is in the target basis, or std::invalid_argument is thrown naming
// the offending gate. Anything the equivalence library cannot reach (MCX,
// MCP, PERMUTATION until their lowering ships; UNITARY; symbolic PARAM_*
// gates, whose parameters are unbound) previously passed through SILENTLY,
// violating the caller's basis. MEASURE/RESET/BARRIER are not gates and are
// exempt. Since the library targets cx+u3, a basis that includes neither
// the source gate nor cx+u3 also throws.
//
// Classical conditions: a conditioned gate decomposes into the same sequence
// with the condition propagated onto EVERY emitted instruction. This is
// exact — the decompositions are unitary-only (no measurement), so all
// emitted gates fire under the same runtime clbit state, which the sequence
// never modifies. (Before R.1.15.0 decomposition silently DROPPED the
// condition; the pass was never composed into a preset pipeline, so the bug
// was latent.)
//
// Routing interaction: every decomposition maps a gate to 1q gates + CX on
// the SAME qubit pair(s), so a post-routing circuit stays hardware-legal.
// This pass is composed as the FINAL stage of the preset pipelines (see
// preset_pass_manager.cpp for why it must come after the basis-oblivious
// optimisation passes).
// =============================================================================

namespace {

using GT = Instruction::GateType;

// Is this gate name already in the target basis?
bool in_basis(const std::string& name, const std::unordered_set<std::string>& basis) {
    return basis.empty() || basis.count(name);
}

// Build U3 instruction: U3(theta, phi, lambda) on qubit q
static Instruction u3_inst(double theta, double phi, double lambda, int q) {
    Instruction inst;
    inst.type = GT::U3;
    inst.qubits = {q};
    inst.params = {theta, phi, lambda};
    return inst;
}

// Build CX instruction
static Instruction cx_inst(int ctrl, int tgt) {
    Instruction inst;
    inst.type = GT::CX;
    inst.qubits = {ctrl, tgt};
    return inst;
}

// Decompose a single gate into a sequence of instructions in {CX, U3}.
// Returns empty vector if gate is already in basis.
static std::vector<Instruction> decompose_to_cx_u3(
    const Instruction& inst,
    const std::unordered_set<std::string>& basis
) {
    const std::string name = inst.gate_name();
    if (in_basis(name, basis)) return {};  // already in basis

    const auto& p = inst.params;
    const int q0 = inst.qubits.empty() ? 0 : inst.qubits[0];
    const int q1 = inst.qubits.size() > 1 ? inst.qubits[1] : -1;
    constexpr double pi = 3.14159265358979323846;
    constexpr double pi2 = pi / 2.0;

    std::vector<Instruction> out;

    switch (inst.type) {
        // ---- Single-qubit: direct U3 equivalents ----
        case GT::H:   out.push_back(u3_inst(pi2, 0, pi, q0)); break;
        case GT::X:   out.push_back(u3_inst(pi, 0, pi, q0)); break;
        case GT::Y:   out.push_back(u3_inst(pi, pi2, pi2, q0)); break;
        case GT::Z:   out.push_back(u3_inst(0, 0, pi, q0)); break;
        case GT::S:   out.push_back(u3_inst(0, 0, pi2, q0)); break;
        case GT::SDG: out.push_back(u3_inst(0, 0, -pi2, q0)); break;
        case GT::T:   out.push_back(u3_inst(0, 0, pi / 4.0, q0)); break;
        case GT::TDG: out.push_back(u3_inst(0, 0, -pi / 4.0, q0)); break;
        case GT::SX:  out.push_back(u3_inst(pi2, -pi2, pi2, q0)); break;
        case GT::SXDG:out.push_back(u3_inst(-pi2, -pi2, pi2, q0)); break;
        case GT::RX:  out.push_back(u3_inst(p[0], -pi2, pi2, q0)); break;
        case GT::RY:  out.push_back(u3_inst(p[0], 0, 0, q0)); break;
        case GT::RZ:  out.push_back(u3_inst(0, 0, p[0], q0)); break;
        case GT::P:   out.push_back(u3_inst(0, 0, p[0], q0)); break;
        case GT::U1:  out.push_back(u3_inst(0, 0, p[0], q0)); break;
        case GT::U2:  out.push_back(u3_inst(pi2, p[0], p[1], q0)); break;
        case GT::U:
        case GT::U3:
            // Already U3 — just relabel type if U is used
            out.push_back(u3_inst(p[0], p[1], p[2], q0)); break;

        // ---- Two-qubit: CX + U3 decompositions ----
        case GT::CY:
            // CY = (I⊗S†) CX (I⊗S)
            out.push_back(u3_inst(0, 0, -pi2, q1));  // S† on target
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, pi2, q1));   // S on target
            break;

        case GT::CZ:
            // CZ = (I⊗H) CX (I⊗H)
            out.push_back(u3_inst(pi2, 0, pi, q1));  // H on target
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(pi2, 0, pi, q1));  // H on target
            break;

        case GT::CH:
            // CH = (I⊗(S†·H·T)) CX (I⊗(T†·H·S))
            out.push_back(u3_inst(0, 0, pi2, q1));          // S
            out.push_back(u3_inst(pi2, 0, pi, q1));         // H
            out.push_back(u3_inst(0, 0, pi / 4.0, q1));     // T
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, -pi / 4.0, q1));   // T†
            out.push_back(u3_inst(pi2, 0, pi, q1));         // H
            out.push_back(u3_inst(0, 0, -pi2, q1));         // S†
            break;

        case GT::SWAP:
            // SWAP = CX(0,1) CX(1,0) CX(0,1)
            out.push_back(cx_inst(q0, q1));
            out.push_back(cx_inst(q1, q0));
            out.push_back(cx_inst(q0, q1));
            break;

        case GT::ISWAP:
            // iSWAP = (S⊗S) · (H⊗I) · CX(0,1) · CX(1,0) · (I⊗H)
            out.push_back(u3_inst(0, 0, pi2, q0));   // S on q0
            out.push_back(u3_inst(0, 0, pi2, q1));   // S on q1
            out.push_back(u3_inst(pi2, 0, pi, q0));  // H on q0
            out.push_back(cx_inst(q0, q1));
            out.push_back(cx_inst(q1, q0));
            out.push_back(u3_inst(pi2, 0, pi, q1));  // H on q1
            break;

        case GT::CRX: {
            double theta = p[0];
            // CRX(θ) gate sequence (applied left to right):
            //   U3(0, 0, π/2) · CX · U3(-θ/2, 0, 0) · CX · U3(θ/2, -π/2, 0)
            out.push_back(u3_inst(0, 0, pi2, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(-theta / 2.0, 0, 0, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(theta / 2.0, -pi2, 0, q1));
            break;
        }

        case GT::CRY: {
            double theta = p[0];
            // CRY(θ): RY(θ/2) CX RY(-θ/2) CX
            out.push_back(u3_inst(theta / 2.0, 0, 0, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(-theta / 2.0, 0, 0, q1));
            out.push_back(cx_inst(q0, q1));
            break;
        }

        case GT::CRZ: {
            double theta = p[0];
            // CRZ(θ) = (I⊗RZ(θ/2)) CX (I⊗RZ(-θ/2)) CX
            out.push_back(u3_inst(0, 0, theta / 2.0, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, -theta / 2.0, q1));
            out.push_back(cx_inst(q0, q1));
            break;
        }

        case GT::CP: {
            double lam = p[0];
            // CP(λ) = (RZ(λ/2)⊗I) CX (I⊗RZ(-λ/2)) CX (I⊗RZ(λ/2))
            out.push_back(u3_inst(0, 0, lam / 2.0, q0));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, -lam / 2.0, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, lam / 2.0, q1));
            break;
        }

        case GT::RZX: {
            double theta = p[0];
            // RZX(θ) = (I⊗H) CX (I⊗RZ(θ)) CX (I⊗H)
            out.push_back(u3_inst(pi2, 0, pi, q1));  // H
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, theta, q1)); // RZ
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(pi2, 0, pi, q1));  // H
            break;
        }

        case GT::RXX: {
            double theta = p[0];
            // RXX(θ) = (H⊗H) CX (I⊗RZ(θ)) CX (H⊗H)
            out.push_back(u3_inst(pi2, 0, pi, q0));
            out.push_back(u3_inst(pi2, 0, pi, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, theta, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(pi2, 0, pi, q0));
            out.push_back(u3_inst(pi2, 0, pi, q1));
            break;
        }

        case GT::RYY: {
            double theta = p[0];
            // RYY(θ) = (RX(π/2)⊗RX(π/2)) CX (I⊗RZ(θ)) CX (RX(-π/2)⊗RX(-π/2))
            out.push_back(u3_inst(pi2, -pi2, pi2, q0));  // RX(π/2)
            out.push_back(u3_inst(pi2, -pi2, pi2, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, theta, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(-pi2, -pi2, pi2, q0)); // RX(-π/2)
            out.push_back(u3_inst(-pi2, -pi2, pi2, q1));
            break;
        }

        case GT::RZZ: {
            double theta = p[0];
            // RZZ(θ) = CX (I⊗RZ(θ)) CX
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, theta, q1));
            out.push_back(cx_inst(q0, q1));
            break;
        }

        case GT::ECR: {
            // ECR = (1/√2)[(I⊗X) + ZX] = RZX(π/4) X⊗I RZX(-π/4)
            double pi4 = pi / 4.0;
            // RZX(π/4)
            out.push_back(u3_inst(pi2, 0, pi, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, pi4, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(pi2, 0, pi, q1));
            // X on q0
            out.push_back(u3_inst(pi, 0, pi, q0));
            // RZX(-π/4)
            out.push_back(u3_inst(pi2, 0, pi, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(0, 0, -pi4, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(pi2, 0, pi, q1));
            break;
        }

        // ---- Three-qubit: expand via 2-qubit decompositions ----
        case GT::CCX: {
            // Standard Toffoli decomposition (6 CNOT, 12 single-qubit)
            int c0 = q0, c1 = q1, t = inst.qubits[2];
            double pi4 = pi / 4.0;
            out.push_back(u3_inst(pi2, 0, pi, t));           // H t
            out.push_back(cx_inst(c1, t));
            out.push_back(u3_inst(0, 0, -pi4, t));           // T† t
            out.push_back(cx_inst(c0, t));
            out.push_back(u3_inst(0, 0, pi4, t));            // T t
            out.push_back(cx_inst(c1, t));
            out.push_back(u3_inst(0, 0, -pi4, t));           // T† t
            out.push_back(cx_inst(c0, t));
            out.push_back(u3_inst(0, 0, pi4, c1));           // T c1
            out.push_back(u3_inst(0, 0, pi4, t));            // T t
            out.push_back(u3_inst(pi2, 0, pi, t));           // H t
            out.push_back(cx_inst(c0, c1));
            out.push_back(u3_inst(0, 0, pi4, c0));           // T c0
            out.push_back(u3_inst(0, 0, -pi4, c1));          // T† c1
            out.push_back(cx_inst(c0, c1));
            break;
        }

        case GT::CCZ: {
            // CCZ = (I⊗I⊗H) CCX (I⊗I⊗H)
            int c0 = q0, c1 = q1, t = inst.qubits[2];
            // Recursively expand: just inline CCX decomp with H wrappers
            out.push_back(u3_inst(pi2, 0, pi, t));           // H t
            // CCX body (same as above)
            double pi4 = pi / 4.0;
            out.push_back(u3_inst(pi2, 0, pi, t));
            out.push_back(cx_inst(c1, t));
            out.push_back(u3_inst(0, 0, -pi4, t));
            out.push_back(cx_inst(c0, t));
            out.push_back(u3_inst(0, 0, pi4, t));
            out.push_back(cx_inst(c1, t));
            out.push_back(u3_inst(0, 0, -pi4, t));
            out.push_back(cx_inst(c0, t));
            out.push_back(u3_inst(0, 0, pi4, c1));
            out.push_back(u3_inst(0, 0, pi4, t));
            out.push_back(u3_inst(pi2, 0, pi, t));
            out.push_back(cx_inst(c0, c1));
            out.push_back(u3_inst(0, 0, pi4, c0));
            out.push_back(u3_inst(0, 0, -pi4, c1));
            out.push_back(cx_inst(c0, c1));
            out.push_back(u3_inst(pi2, 0, pi, t));           // H t
            break;
        }

        case GT::CSWAP: {
            // Fredkin = CCX with target and ancilla swapped
            // CSWAP(c, a, b) = CX(b,a) CCX(c,a,b) CX(b,a)
            int c = q0, a = q1, b = inst.qubits[2];
            double pi4 = pi / 4.0;
            out.push_back(cx_inst(b, a));
            out.push_back(u3_inst(pi2, 0, pi, b));
            out.push_back(cx_inst(a, b));
            out.push_back(u3_inst(0, 0, -pi4, b));
            out.push_back(cx_inst(c, b));
            out.push_back(u3_inst(0, 0, pi4, b));
            out.push_back(cx_inst(a, b));
            out.push_back(u3_inst(0, 0, -pi4, b));
            out.push_back(cx_inst(c, b));
            out.push_back(u3_inst(0, 0, pi4, a));
            out.push_back(u3_inst(0, 0, pi4, b));
            out.push_back(u3_inst(pi2, 0, pi, b));
            out.push_back(cx_inst(c, a));
            out.push_back(u3_inst(0, 0, pi4, c));
            out.push_back(u3_inst(0, 0, -pi4, a));
            out.push_back(cx_inst(c, a));
            out.push_back(cx_inst(b, a));
            break;
        }

        case GT::CU: {
            // CU(θ, φ, λ, γ): the controlled block is e^{iγ}·U3(θ, φ, λ)
            // (project convention, see gate builders / the analytic 4x4).
            // Standard CU3 ladder with the γ phase folded into the control's
            // diagonal U1 = U3(0, 0, ·).
            double th = p[0], ph = p[1], la = p[2], ga = p[3];
            out.push_back(u3_inst(0, 0, ga + (la + ph) / 2.0, q0));
            out.push_back(u3_inst(0, 0, (la - ph) / 2.0, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(-th / 2.0, 0, -(ph + la) / 2.0, q1));
            out.push_back(cx_inst(q0, q1));
            out.push_back(u3_inst(th / 2.0, ph, 0, q1));
            break;
        }

        case GT::RCCX: {
            // Margolus (relative-phase Toffoli): H T CX T† CX T CX T† H
            // ladder, matching gates::apply_rccx and the MPS decomposition
            // exactly (not just up to phase).
            int c0 = q0, c1 = q1, t = inst.qubits[2];
            double pi4 = pi / 4.0;
            out.push_back(u3_inst(pi2, 0, pi, t));   // H t
            out.push_back(u3_inst(0, 0, pi4, t));    // T t
            out.push_back(cx_inst(c1, t));
            out.push_back(u3_inst(0, 0, -pi4, t));   // T† t
            out.push_back(cx_inst(c0, t));
            out.push_back(u3_inst(0, 0, pi4, t));    // T t
            out.push_back(cx_inst(c1, t));
            out.push_back(u3_inst(0, 0, -pi4, t));   // T† t
            out.push_back(u3_inst(pi2, 0, pi, t));   // H t
            break;
        }

        default:
            // Pass through: gate is either already in basis or unknown
            out.push_back(inst);
            break;
    }
    return out;
}

} // anonymous namespace

DAGCircuit BasisTranslator::run(
    const DAGCircuit& dag,
    const TranspilationContext& ctx
) const {
    QuantumCircuit qc = dag.to_circuit();
    QuantumCircuit out(qc.n_qubits, qc.n_clbits);
    out.name = qc.name;

    // Default target basis if none specified: CX + U3
    std::unordered_set<std::string> basis_set;
    const auto& bg = ctx.basis_gates;
    if (!bg.empty()) {
        basis_set.insert(bg.begin(), bg.end());
    } else {
        basis_set = {"cx", "u3"};
    }

    // Emitted 1q gate type follows the basis: the library synthesises the
    // 3-parameter U gate, which this codebase carries as both U3 ("u3") and
    // U ("u") with identical semantics (see instruction_to_2x2 /
    // decompose_to_cx_u3's U/U3 case). Emit the variant the caller's basis
    // names; prefer u3 when both (or neither) are present.
    const bool emit_u = basis_set.count("u") > 0 && basis_set.count("u3") == 0;

    for (const auto& inst : qc.instructions) {
        if (inst.type == Instruction::GateType::MEASURE ||
            inst.type == Instruction::GateType::RESET  ||
            inst.type == Instruction::GateType::BARRIER) {
            out.instructions.push_back(inst);
            continue;
        }

        auto decomposed = decompose_to_cx_u3(inst, basis_set);
        if (decomposed.empty()) {
            // Gate already in basis
            out.instructions.push_back(inst);
        } else {
            for (auto& d : decomposed) {
                // Propagate the classical condition onto every emitted gate:
                // exact for these unitary-only decompositions (see header).
                d.condition_clbit = inst.condition_clbit;
                d.condition_value = inst.condition_value;
                if (emit_u && d.type == Instruction::GateType::U3) {
                    d.type = Instruction::GateType::U;  // same gate, basis's name
                }
                out.instructions.push_back(std::move(d));
            }
        }
    }

    // Verify the basis contract on the final output (R.1.15.0): anything the
    // equivalence library could not reach passed through decompose_to_cx_u3
    // unchanged — catch it loudly here instead of silently violating the
    // caller's basis_gates. Deterministic message: basis listed in the
    // caller's order (or the cx+u3 default).
    for (const auto& inst : out.instructions) {
        if (inst.type == Instruction::GateType::MEASURE ||
            inst.type == Instruction::GateType::RESET  ||
            inst.type == Instruction::GateType::BARRIER) {
            continue;
        }
        const std::string name = inst.gate_name();
        if (in_basis(name, basis_set)) continue;

        std::string basis_str;
        if (!bg.empty()) {
            for (const auto& b : bg) basis_str += (basis_str.empty() ? "" : ", ") + b;
        } else {
            basis_str = "cx, u3";
        }
        std::string msg = "lindblad::BasisTranslator: cannot translate gate '" +
            name + "' into the requested basis {" + basis_str + "}";
        if (inst.type == Instruction::GateType::MCX ||
            inst.type == Instruction::GateType::MCP ||
            inst.type == Instruction::GateType::PERMUTATION) {
            msg += "; MCX/MCP/PERMUTATION lowering is not implemented yet: "
                   "include the gate itself in basis_gates or decompose it "
                   "before transpiling";
        } else if (inst.is_parameterised()) {
            msg += "; symbolic parameterised gates cannot be numerically "
                   "decomposed: bind parameters first or include the gate "
                   "in basis_gates";
        } else {
            msg += "; the equivalence library targets cx+u3: include 'cx' "
                   "and 'u3' (or the gate itself) in basis_gates";
        }
        throw std::invalid_argument(msg);
    }

    return DAGCircuit::from_circuit(out);
}

} // namespace lindblad
