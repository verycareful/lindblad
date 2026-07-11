// R.1.14.1 test patch -- comparison-benchmark corpus and loaders (structural).
// Guards the committed QASM2 corpus (benchmarks/compare/circuits/) and the
// compare_common.hpp loaders that both the bench_compare_* binaries and
// bench_validate depend on. A corpus or loader regression invalidates every
// published number in docs/Benchmarks.md, so these are load-bearing:
//   - every committed .qasm parses via from_qasm2, is gate-only (the harness
//     contract: measurement is appended by each engine, never stored), and its
//     qubit count matches its filename
//   - every workload file registered in a bench_compare_* binary exists
//   - the coupling edge lists load, symmetrise, and describe connected graphs
//     of the expected size
//   - the observable files load with the exact Heisenberg-chain structure and
//     reproduce the analytic <0...0|H|0...0> value
//   - missing corpus files fail loudly with the regeneration hint

#include <gtest/gtest.h>

#include "compare_common.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/primitives.hpp"

#include <filesystem>
#include <regex>
#include <set>
#include <string>

using namespace lindblad;
using namespace lindblad_bench;

namespace {

// Workload files registered in the bench_compare_* BENCHMARK_CAPTURE lists and
// bench_validate. Keep in sync when benchmarks are added.
const std::set<std::string> kRegisteredWorkloads = {
    "scaling_n10.qasm", "scaling_n14.qasm", "scaling_n18.qasm",
    "scaling_n22.qasm", "scaling_n26.qasm",                       // SV scaling
    "qft_n10.qasm", "qft_n14.qasm", "qft_n18.qasm", "qft_n22.qasm",
    "qv_n10.qasm", "qv_n14.qasm", "qv_n18.qasm", "qv_n22.qasm",
    "grover_n8.qasm", "grover_n10.qasm", "grover_n12.qasm",
    "dmscaling_n4.qasm", "dmscaling_n6.qasm", "dmscaling_n8.qasm",
    "dmscaling_n10.qasm",                                          // DM
    "scaling_n16.qasm", "scaling_n24.qasm", "scaling_n32.qasm",
    "scaling_n40.qasm",                                            // MPS
    "clifford_n20.qasm", "clifford_n40.qasm", "clifford_n80.qasm",
    "clifford_n160.qasm",                                          // Clifford
    "qv_n25.qasm", "qv_n27.qasm", "qft_n25.qasm", "qft_n27.qasm",  // transpiler
    "ansatz_n12.qasm", "ansatz_n16.qasm", "ansatz_n20.qasm",       // estimator
    "scaling_n8.qasm", "qft_n8.qasm", "qv_n8.qasm",
    "clifford_n8.qasm", "ansatz_n8.qasm",                          // validation
};

// Expected qubit count from the filename. Grover files are named by the
// SEARCH width s but declare 2s-3 total qubits (s-3 v-chain ancillas).
int expected_qubits(const std::string& stem, int n_in_name) {
    if (stem.rfind("grover", 0) == 0) return 2 * n_in_name - 3;
    return n_in_name;
}

} // namespace

// =============================================================================
// R1141Corpus -- structural checks over every committed circuit file
// =============================================================================

TEST(R1141Corpus, EveryCommittedCircuitParsesGateOnlyWithNamedQubitCount) {
    const std::filesystem::path dir(LINDBLAD_BENCH_QASM_DIR);
    ASSERT_TRUE(std::filesystem::is_directory(dir))
        << "corpus directory missing: " << dir
        << " (run benchmarks/compare/gen_circuits.py)";

    const std::regex name_re(R"(^[a-z]+_n(\d+)$)");
    int qasm_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".qasm") continue;
        ++qasm_count;
        const std::string stem = entry.path().stem().string();

        std::smatch m;
        ASSERT_TRUE(std::regex_match(stem, m, name_re))
            << "unexpected corpus filename: " << entry.path();
        const int n_in_name = std::stoi(m[1]);

        QuantumCircuit qc;
        ASSERT_NO_THROW(qc = load_corpus_circuit(entry.path().filename().string(),
                                                 /*add_measure=*/false))
            << "corpus file fails to parse: " << entry.path();

        EXPECT_EQ(qc.n_qubits, expected_qubits(stem, n_in_name))
            << "qubit count does not match filename: " << entry.path();
        EXPECT_FALSE(qc.instructions.empty()) << entry.path();

        // Harness contract: gate-only files; measurement is appended by each
        // engine at run time, never stored in the corpus.
        for (const auto& ins : qc.instructions) {
            EXPECT_NE(ins.type, Instruction::GateType::MEASURE)
                << "corpus circuit contains MEASURE: " << entry.path();
            EXPECT_NE(ins.type, Instruction::GateType::BARRIER)
                << "corpus circuit contains BARRIER: " << entry.path();
        }
    }
    // 40 circuit files at R.1.14.0 (the corpus directory holds 47 files in
    // total: 40 .qasm + 4 observable .txt + 3 coupling .edges; only the .qasm
    // files are counted here). Only grow this number deliberately.
    EXPECT_GE(qasm_count, 40) << "committed corpus shrank";
}

TEST(R1141Corpus, EveryRegisteredWorkloadFileExists) {
    for (const auto& name : kRegisteredWorkloads) {
        EXPECT_TRUE(std::filesystem::exists(
            std::filesystem::path(LINDBLAD_BENCH_QASM_DIR) / name))
            << "benchmark-registered corpus file missing: " << name;
    }
}

TEST(R1141Corpus, MeasureAllAppendsFullRegisterMeasurement) {
    const auto bare = load_corpus_circuit("scaling_n8.qasm", false);
    const auto measured = load_corpus_circuit("scaling_n8.qasm", true);
    int measures = 0;
    for (const auto& ins : measured.instructions) {
        if (ins.type == Instruction::GateType::MEASURE) ++measures;
    }
    EXPECT_EQ(measures, 8);
    EXPECT_EQ(measured.instructions.size(), bare.instructions.size() + 8);
}

TEST(R1141Corpus, MissingCorpusFileFailsLoudWithRegenerationHint) {
    try {
        load_corpus_circuit("no_such_file.qasm", false);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("gen_circuits.py"), std::string::npos)
            << "error message should point at the corpus generator";
    }
}

// =============================================================================
// R1141Corpus -- coupling-map edge lists
// =============================================================================

TEST(R1141Corpus, CouplingMapsLoadSymmetrisedAndConnected) {
    struct Expect { const char* file; int n; int undirected_edges; };
    const Expect cases[] = {
        {"coupling_linear27.edges", 27, 26},
        {"coupling_grid25.edges", 25, 40},
        {"coupling_heavyhex27.edges", 27, 28},
    };
    for (const auto& c : cases) {
        const auto cmap = load_coupling(c.file);
        EXPECT_EQ(cmap.n_physical_qubits, c.n) << c.file;
        // The loader inserts both directions of every undirected pair.
        EXPECT_EQ(cmap.edges.size(), static_cast<size_t>(2 * c.undirected_edges))
            << c.file;
        EXPECT_TRUE(cmap.is_connected_graph()) << c.file;
        for (const auto& [u, v] : cmap.edges) {
            EXPECT_TRUE(cmap.is_connected(v, u))
                << c.file << ": missing reverse edge " << v << "->" << u;
        }
    }
}

// =============================================================================
// R1141Corpus -- observable files
// =============================================================================

TEST(R1141Corpus, HeisenbergObservablesHaveExactChainStructure) {
    for (int n : {8, 12, 16, 20}) {
        const auto obs = load_observable("obs_heisenberg_n" + std::to_string(n) + ".txt");
        // 3(n-1) coupling terms + n field terms.
        ASSERT_EQ(obs.terms.size(), static_cast<size_t>(3 * (n - 1) + n)) << "n=" << n;
        EXPECT_EQ(obs.n_qubits(), n);
        int coupling = 0, field = 0;
        for (const auto& t : obs.terms) {
            int non_identity = 0;
            for (char p : t.pauli) non_identity += (p != 'I');
            if (non_identity == 2) {
                ++coupling;
                EXPECT_EQ(t.coeff, Complex128(1.0, 0.0));
            } else {
                ASSERT_EQ(non_identity, 1);
                ++field;
                EXPECT_EQ(t.coeff, Complex128(0.5, 0.0));
            }
        }
        EXPECT_EQ(coupling, 3 * (n - 1));
        EXPECT_EQ(field, n);
    }
}

TEST(R1141Corpus, HeisenbergExpectationOnZeroStateIsAnalytic) {
    // On |0...0>: every ZZ term contributes +1, XX/YY contribute 0, and each
    // 0.5*Z field term contributes +0.5. For n = 8: 7*1 + 8*0.5 = 11.
    const auto obs = load_observable("obs_heisenberg_n8.txt");
    Estimator est;
    est.options.shots = 0;
    const double value = est.run_single(QuantumCircuit(8), obs);
    EXPECT_NEAR(value, 11.0, 1e-9);
}
