// R.1.12.1 total-coverage suite, Batch 3: lindblad/transpiler.hpp CouplingMap.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 3: toolchain".
//
// Pins the four named topologies' connectivity, symmetric is_connected,
// shortest_path / distance_matrix correctness (incl. unreachable), and the
// literal edgeless-vs-unconstrained semantics. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/transpiler.hpp"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {
// Undirected edge set (min,max ordered) extracted from the directed edge list.
std::set<std::pair<int, int>> undirected_edges(const CouplingMap& cm) {
    std::set<std::pair<int, int>> e;
    for (const auto& [a, b] : cm.edges)
        e.insert({std::min(a, b), std::max(a, b)});
    return e;
}
}  // namespace

// =============================================================================
// linear
// =============================================================================

TEST(R1121Coupling, LinearChainConnectivity) {
    auto cm = CouplingMap::linear(4);
    EXPECT_TRUE(cm.is_connected(0, 1));
    EXPECT_TRUE(cm.is_connected(1, 0));   // edges stored both directions
    EXPECT_FALSE(cm.is_connected(0, 2));  // not adjacent
    EXPECT_TRUE(cm.is_connected_graph());

    auto path = cm.shortest_path(0, 3);
    ASSERT_EQ(path.size(), 4u);
    EXPECT_EQ(path.front(), 0);
    EXPECT_EQ(path.back(), 3);
}

TEST(R1121Coupling, DistanceMatrixIsSymmetricWithZeroDiagonal) {
    auto cm = CouplingMap::linear(4);
    auto d = cm.distance_matrix();
    ASSERT_EQ(d.size(), 4u);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(d[i][i], 0);
        for (int j = 0; j < 4; ++j) EXPECT_EQ(d[i][j], d[j][i]);
    }
    EXPECT_EQ(d[0][3], 3);
    EXPECT_EQ(d[1][3], 2);
}

// =============================================================================
// all_to_all / grid / heavy_hex
// =============================================================================

TEST(R1121Coupling, AllToAllEveryPairAdjacent) {
    auto cm = CouplingMap::all_to_all(4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) EXPECT_TRUE(cm.is_connected(i, j));
    auto d = cm.distance_matrix();
    EXPECT_EQ(d[0][3], 1);
    EXPECT_TRUE(cm.is_connected_graph());
}

TEST(R1121Coupling, GridAndHeavyHexAreConnected) {
    auto g = CouplingMap::grid(2, 2);
    EXPECT_EQ(g.n_physical_qubits, 4);
    EXPECT_TRUE(g.is_connected_graph());
    EXPECT_TRUE(g.is_connected(0, 1));   // same row
    EXPECT_TRUE(g.is_connected(0, 2));   // same column (idx + cols)

    auto hh = CouplingMap::heavy_hex(5);
    EXPECT_TRUE(hh.is_connected_graph());
}

// =============================================================================
// Exact edge sets at small n
// =============================================================================

TEST(R1121Coupling, ExactEdgeSets) {
    using E = std::set<std::pair<int, int>>;

    // linear(5): a simple chain.
    EXPECT_EQ(undirected_edges(CouplingMap::linear(5)),
              (E{{0, 1}, {1, 2}, {2, 3}, {3, 4}}));

    // grid(2,3): indices r*cols + c. Horizontal within each row, vertical
    // between rows.
    EXPECT_EQ(undirected_edges(CouplingMap::grid(2, 3)),
              (E{{0, 1}, {1, 2}, {3, 4}, {4, 5}, {0, 3}, {1, 4}, {2, 5}}));

    // all_to_all(4): the complete graph K4 (6 edges).
    EXPECT_EQ(undirected_edges(CouplingMap::all_to_all(4)),
              (E{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}));

    // heavy_hex at small n is a single data row threaded by horizontal ancillas
    // — a linear chain (documented "1D Falcon" behaviour).
    EXPECT_EQ(undirected_edges(CouplingMap::heavy_hex(5)),
              (E{{0, 1}, {1, 2}, {2, 3}, {3, 4}}));
    EXPECT_EQ(undirected_edges(CouplingMap::heavy_hex(7)),
              (E{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}));
}

TEST(R1121Coupling, EdgesAreStoredSymmetrically) {
    // Every topology stores both directions of each undirected edge.
    for (const CouplingMap& cm : {CouplingMap::linear(4), CouplingMap::grid(2, 3),
                                  CouplingMap::all_to_all(4),
                                  CouplingMap::heavy_hex(9)}) {
        std::set<std::pair<int, int>> directed(cm.edges.begin(), cm.edges.end());
        for (const auto& [a, b] : cm.edges)
            EXPECT_TRUE(directed.count({b, a}))
                << "missing reverse edge " << b << "," << a;
    }
}

TEST(R1121Coupling, GridShortestPathAndDistances) {
    auto g = CouplingMap::grid(2, 3);  // 2x3 grid, 6 qubits
    auto d = g.distance_matrix();
    // Manhattan distances on the grid: 0 at (0,0), 5 at (1,2) -> distance 3.
    EXPECT_EQ(d[0][5], 3);
    EXPECT_EQ(d[0][4], 2);  // (0,0) -> (1,1)
    EXPECT_EQ(d[2][3], 3);  // (0,2) -> (1,0): diagonal corners
    auto path = g.shortest_path(0, 5);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(path.front(), 0);
    EXPECT_EQ(path.back(), 5);
    EXPECT_EQ(path.size(), 4u);  // length-3 path has 4 nodes
}

// =============================================================================
// literal edge semantics
// =============================================================================

TEST(R1121Coupling, EdgelessMapIsDisconnectedAndUnreachable) {
    CouplingMap edgeless(3);  // 3 qubits, no edges (literal)
    EXPECT_FALSE(edgeless.is_connected(0, 1));
    EXPECT_FALSE(edgeless.is_connected_graph());
    EXPECT_TRUE(edgeless.shortest_path(0, 2).empty()) << "unreachable -> empty path";
}

TEST(R1121Coupling, DefaultMapHasNoPhysicalQubits) {
    CouplingMap unconstrained;  // n = 0 sentinel
    EXPECT_EQ(unconstrained.n_physical_qubits, 0);
}
