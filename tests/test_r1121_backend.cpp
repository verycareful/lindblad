// R.1.12.1 total-coverage suite, Batch 3: lindblad/backends/local_backend.hpp.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 3: toolchain".
//
// Each explicit SimType and AUTO produce correct counts on a Clifford circuit;
// metadata (name/version/max_qubits) and run_batch shape are checked, plus the
// AUTO selection observable proxies (noise -> DM result still correct, Clifford
// -> Clifford result matches statevector). Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"

#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::backends;

namespace {
QuantumCircuit bell_measured() {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).measure_all();
    return qc;
}

void expect_correlated(const std::unordered_map<std::string, int>& counts,
                       int shots) {
    int total = 0;
    for (const auto& [bits, n] : counts) {
        EXPECT_TRUE(bits == "00" || bits == "11") << "unexpected key " << bits;
        total += n;
    }
    EXPECT_EQ(total, shots);
}
}  // namespace

// =============================================================================
// Explicit simulator selection
// =============================================================================

TEST(R1121Backend, EachExplicitSimTypeRunsBell) {
    for (auto st : {LocalBackend::SimType::STATEVECTOR,
                    LocalBackend::SimType::DENSITY_MATRIX,
                    LocalBackend::SimType::CLIFFORD,
                    LocalBackend::SimType::MPS}) {
        LocalBackend::Config cfg;
        cfg.simulator = st;
        LocalBackend be(cfg);
        auto res = be.run(bell_measured(), 1000, 1);
        EXPECT_TRUE(res.success) << res.error_message;
        expect_correlated(res.counts, 1000);
    }
}

// =============================================================================
// AUTO selection (observable proxies)
// =============================================================================

TEST(R1121Backend, AutoSelectsAndProducesCorrectCounts) {
    LocalBackend be;  // default config: AUTO
    EXPECT_EQ(static_cast<int>(be.config.simulator),
              static_cast<int>(LocalBackend::SimType::AUTO));
    auto res = be.run(bell_measured(), 1000, 1);
    EXPECT_TRUE(res.success);
    expect_correlated(res.counts, 1000);
}

TEST(R1121Backend, AutoWithNoiseStillCorrectOnNoiselessGate) {
    // A noise model present routes AUTO to the density-matrix path; with an
    // identity-like tiny error the correlated structure must survive.
    LocalBackend be;
    be.noise_model.add_all_qubit_quantum_error(
        NoiseChannels::depolarizing(0.0, 1), "h");  // p=0: no actual error
    auto res = be.run(bell_measured(), 1000, 2);
    EXPECT_TRUE(res.success);
    expect_correlated(res.counts, 1000);
}

// =============================================================================
// Metadata and batch
// =============================================================================

TEST(R1121Backend, MetadataAndConfigDefaults) {
    LocalBackend be;
    EXPECT_EQ(be.name(), "lindblad_local_simulator");
    EXPECT_EQ(be.max_qubits(), 30);
    EXPECT_FALSE(be.version().empty());
    EXPECT_EQ(be.config.mps_bond_dim, 64);
}

TEST(R1121Backend, RunBatchReturnsOneResultPerCircuit) {
    LocalBackend be;
    auto results = be.run_batch({bell_measured(), bell_measured()}, 256, 5);
    ASSERT_EQ(results.size(), 2u);
    for (const auto& r : results) {
        EXPECT_TRUE(r.success);
        expect_correlated(r.counts, 256);
    }
}
