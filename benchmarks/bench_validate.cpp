// =============================================================================
// bench_validate.cpp -- Lindblad half of the cross-engine result-parity check
// =============================================================================
// Speed numbers are worthless if the answers differ, so the comparison suite
// never publishes a timing table without this gate. This tool runs the small
// validation members of every corpus family and dumps raw results as JSON;
// aer_bench.py --validate produces the twin file, and tools/bench_report.py
// compares the two:
//   - sampled counts (8192 shots): total-variation distance must be within
//     sampling noise of zero
//   - exact estimator expectation: must agree to ~1e-9
// The Grover validation circuit peaks on a NON-symmetric marked state, so it
// doubles as an end-to-end qubit-ordering convention check across engines
// (a symmetric target would mask ordering bugs; see CLAUDE.md).
//
// Usage: bench_validate [output.json]   (default lindblad_validation.json)
// The emitted "version" field carries LINDBLAD_VERSION_LABEL so the report
// script can refuse stale binaries (see CLAUDE.md "Build and Test").

#include "lindblad/noise.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include "compare_common.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

using namespace lindblad;
using namespace lindblad_bench;

namespace {

// Ordered copy so the JSON is deterministic and diff-friendly.
std::map<std::string, int> ordered(const std::unordered_map<std::string, int>& counts) {
    return {counts.begin(), counts.end()};
}

void write_counts(std::ofstream& out, const std::string& key,
                  const std::map<std::string, int>& counts, bool last) {
    out << "    \"" << key << "\": {";
    bool first = true;
    for (const auto& [bits, n] : counts) {
        if (!first) out << ", ";
        first = false;
        out << "\"" << bits << "\": " << n;
    }
    out << "}" << (last ? "" : ",") << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "lindblad_validation.json";

    // Sampled-counts validation set (small members of each timed family).
    StatevectorSimulator sv;
    CliffordSimulator cliff;
    DensityMatrixSimulator dm;
    NoiseModel dm_noise;  // twin of aer_bench.py make_noise_model()
    dm_noise.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.01, 2), "cx");
    dm_noise.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.005), "h");

    std::map<std::string, std::map<std::string, int>> counts;
    counts["val__sv__scaling__n8"] = ordered(
        sv.run(load_corpus_circuit("scaling_n8.qasm", true), kValidationShots, kSeed).counts);
    counts["val__sv__qft__n8"] = ordered(
        sv.run(load_corpus_circuit("qft_n8.qasm", true), kValidationShots, kSeed).counts);
    counts["val__sv__qv__n8"] = ordered(
        sv.run(load_corpus_circuit("qv_n8.qasm", true), kValidationShots, kSeed).counts);
    counts["val__sv__grover__s8"] = ordered(
        sv.run(load_corpus_circuit("grover_n8.qasm", true), kValidationShots, kSeed).counts);
    counts["val__clifford__ladder__n8"] = ordered(
        cliff.run(load_corpus_circuit("clifford_n8.qasm", true), kValidationShots, kSeed).counts);
    counts["val__dm__scaling__n6"] = ordered(
        dm.run(load_corpus_circuit("dmscaling_n6.qasm", true), dm_noise,
               kValidationShots, kSeed).counts);

    // Exact expectation validation (engine-independent ground truth).
    Estimator est;
    est.options.shots = 0;
    const double expectation = est.run_single(
        load_corpus_circuit("ansatz_n8.qasm", false),
        load_observable("obs_heisenberg_n8.txt"));

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "bench_validate: cannot write " << out_path << "\n";
        return 1;
    }
    out << "{\n";
    out << "  \"engine\": \"lindblad\",\n";
    out << "  \"version\": \"" << LINDBLAD_VERSION_LABEL << "\",\n";
    out << "  \"shots\": " << kValidationShots << ",\n";
    out << "  \"seed\": " << kSeed << ",\n";
    out << "  \"counts\": {\n";
    size_t i = 0;
    for (const auto& [key, c] : counts) {
        write_counts(out, key, c, ++i == counts.size());
    }
    out << "  },\n";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.12g", expectation);
    out << "  \"expectation\": { \"val__est__heisenberg__n8\": " << buf << " }\n";
    out << "}\n";

    std::cout << "bench_validate (" << LINDBLAD_VERSION_LABEL << "): wrote "
              << out_path << "\n";
    return 0;
}
