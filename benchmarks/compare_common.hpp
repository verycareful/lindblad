#pragma once

// =============================================================================
// compare_common.hpp -- shared helpers for the Lindblad vs Qiskit/Aer
// comparison suite (bench_compare_*.cpp, bench_validate.cpp)
// =============================================================================
//
// The shared corpus lives in benchmarks/compare/circuits/ (absolute path baked
// in at configure time via the LINDBLAD_BENCH_QASM_DIR compile definition; see
// benchmarks/CMakeLists.txt). Corpus circuits are gate-only QASM2 with a
// single qreg: sampling workloads append measure_all() HERE, and the Python
// harness (benchmarks/compare/aer_bench.py) does the same on its side, so both
// engines execute byte-identical gate content.
//
// Protocol constants (kShots, kSeed, kValidationShots) are mirrored in
// aer_bench.py. Change them in BOTH places or cross-engine numbers are
// meaningless.

#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/transpiler.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef LINDBLAD_BENCH_QASM_DIR
#error "LINDBLAD_BENCH_QASM_DIR must be defined (see benchmarks/CMakeLists.txt)"
#endif

namespace lindblad_bench {

inline constexpr int kShots = 256;             // timed sampling workloads
inline constexpr int kValidationShots = 8192;  // result-parity validation runs
inline constexpr std::uint64_t kSeed = 42;

inline std::string corpus_path(const std::string& file) {
    return std::string(LINDBLAD_BENCH_QASM_DIR) + "/" + file;
}

inline std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("compare bench: cannot open " + path +
                                 " (corpus missing? run benchmarks/compare/gen_circuits.py)");
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Load a corpus circuit. add_measure = true appends a full-register terminal
// measurement (qubit q -> clbit q), mirroring Qiskit's measure_all(), so both
// engines sample the identical circuit and produce comparable bitstring keys
// (qubit 0 rightmost on both sides).
inline lindblad::QuantumCircuit load_corpus_circuit(const std::string& file,
                                                    bool add_measure) {
    auto qc = lindblad::QuantumCircuit::from_qasm2(read_file(corpus_path(file)));
    if (add_measure) qc.measure_all();
    return qc;
}

// .edges format (written by gen_circuits.py): '#' comment lines, then one line
// with the qubit count, then undirected "u v" pairs. CouplingMap::edges is
// directed, so both directions are inserted here; aer_bench.py symmetrises the
// same list on its side, guaranteeing both engines route on identical graphs.
inline lindblad::CouplingMap load_coupling(const std::string& file) {
    std::istringstream in(read_file(corpus_path(file)));
    lindblad::CouplingMap cmap;
    std::string line;
    bool have_n = false;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        if (!have_n) {
            ls >> cmap.n_physical_qubits;
            have_n = true;
            continue;
        }
        int u = -1, v = -1;
        ls >> u >> v;
        if (u < 0 || v < 0) {
            throw std::runtime_error("compare bench: malformed edge line in " + file);
        }
        cmap.edges.emplace_back(u, v);
        cmap.edges.emplace_back(v, u);
    }
    if (!have_n || cmap.edges.empty()) {
        throw std::runtime_error("compare bench: empty coupling file " + file);
    }
    return cmap;
}

// obs_*.txt format (written by gen_circuits.py): '#' comment lines, then
// "<coeff> <pauli>" per line. Strings are ALREADY in Lindblad order
// (char q acts on qubit q, LSB first); only the Qiskit side reverses.
inline lindblad::SparsePauliOp load_observable(const std::string& file) {
    std::istringstream in(read_file(corpus_path(file)));
    std::vector<std::pair<std::string, lindblad::Complex128>> terms;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        double coeff = 0.0;
        std::string pauli;
        ls >> coeff >> pauli;
        if (pauli.empty()) {
            throw std::runtime_error("compare bench: malformed observable line in " + file);
        }
        terms.emplace_back(pauli, lindblad::Complex128(coeff, 0.0));
    }
    if (terms.empty()) {
        throw std::runtime_error("compare bench: empty observable file " + file);
    }
    return lindblad::SparsePauliOp::from_list(terms);
}

// Two-qubit gate count of a measure-free circuit: the routing/basis-quality
// metric reported alongside transpile time (lower is better at equal legality).
inline int two_qubit_count(const lindblad::QuantumCircuit& qc) {
    int count = 0;
    for (const auto& ins : qc.instructions) {
        if (ins.qubits.size() == 2) ++count;
    }
    return count;
}

} // namespace lindblad_bench
