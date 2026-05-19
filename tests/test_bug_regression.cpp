// Black-box regression tests for the 8 open bugs listed in docs/plans/TODO.md as of R.1.7.4.
// Each test is designed to FAIL if the corresponding bug is still open.
// Run under ctest (WSL) and compare the pass/fail verdict against the TODO list
// to determine which entries are genuinely open vs. stale.

#include <gtest/gtest.h>
#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/transpiler.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace lindblad;

static constexpr double kTol  = 1e-9;   // exact comparisons
static constexpr double kStat = 0.05;   // statistical fraction tolerance (shots-based)

// =============================================================================
// B1 — from_qasm2() wiring
// TODO: from_qasm2() throws instead of calling qasm2_parse_impl (src/circuit.cpp ~L1023).
// Test: parse a 3-qubit GHZ QASM string; expect no throw, correct qubit/gate count,
//       and that simulation produces only "000"/"111" outcomes.
// =============================================================================

TEST(BugRegression, B1_FromQasm2DoesNotThrow) {
    // Different from existing integration test: 3-qubit GHZ, verify simulation too.
    const std::string qasm = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
)";

    QuantumCircuit qc = QuantumCircuit::from_qasm2(qasm);
    EXPECT_EQ(qc.n_qubits, 3);
    EXPECT_EQ(qc.n_clbits, 3);

    // H + 2×CX + 3×MEASURE = 6 instructions
    EXPECT_EQ(qc.size(), 6);

    StatevectorSimulator sim;
    auto res = sim.run(qc, 1024, 7);
    ASSERT_TRUE(res.success);
    for (const auto& [bits, count] : res.counts)
        EXPECT_TRUE(bits == "000" || bits == "111")
            << "Unexpected bitstring: " << bits;
}

// =============================================================================
// B2 — QASM2 parser hard-codes register names q and c
// TODO: parse_qubits fails for custom register names (src/qasm/qasm2_parser.cpp).
// Test: 4-qubit circuit across two custom qregs; verify qubit offset arithmetic
//       by checking a known final-state distribution.
// =============================================================================

TEST(BugRegression, B2_Qasm2CustomRegisterNames) {
    // "left" register (2 qubits), "right" register (2 qubits).
    // X on right[1] (global qubit 3), measure all.
    // Expected: only "0001" (qubit 3 = 1, rest = 0, MSB-first ordering).
    const std::string qasm = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg left[2];
qreg right[2];
creg out[4];
x right[1];
measure left[0]  -> out[0];
measure left[1]  -> out[1];
measure right[0] -> out[2];
measure right[1] -> out[3];
)";

    QuantumCircuit qc = QuantumCircuit::from_qasm2(qasm);
    ASSERT_EQ(qc.n_qubits, 4);

    StatevectorSimulator sim;
    auto res = sim.run(qc, 128, 11);
    ASSERT_TRUE(res.success);
    EXPECT_EQ(res.counts.size(), 1u);
    // Qubit layout: left[0]=q0, left[1]=q1, right[0]=q2, right[1]=q3.
    // MSB-first bitstring: q3 q2 q1 q0 = 1 0 0 0 = "1000".
    EXPECT_GT(res.counts.count("1000"), 0u)
        << "Custom register offset wrong: right[1] not mapped to qubit 3";
}

// =============================================================================
// B3 — MPS simulator throws std::runtime_error for UNITARY gate type
// TODO: MPS throws at mps_sim.cpp for UNITARY; blocks Grover MCZ oracle on MPS.
// Test: apply a 2-qubit UNITARY (iSWAP-like) via MPS backend; assert no throw,
//       and compare output probability distribution against StatevectorSimulator.
// =============================================================================

TEST(BugRegression, B3_MPSUnitaryGateNoThrow) {
    // Custom 2-qubit unitary: SWAP matrix, supplied as a raw vector.
    // SWAP |ab⟩ = |ba⟩.
    const std::vector<Complex128> SWAP_mat = {
        {1,0},{0,0},{0,0},{0,0},
        {0,0},{0,0},{1,0},{0,0},
        {0,0},{1,0},{0,0},{0,0},
        {0,0},{0,0},{0,0},{1,0}
    };

    // Start |01⟩ — X on qubit 0. After SWAP should be |10⟩.
    auto make_circuit = [&]() {
        QuantumCircuit qc(2);
        qc.x(0);
        qc.unitary(SWAP_mat, {0, 1});
        qc.measure_all();
        return qc;
    };

    // MPS must not throw.
    ASSERT_NO_THROW({
        MPSSimulator mps;
        [[maybe_unused]] auto res = mps.run(make_circuit(), 16, 256, 13);
    });

    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(make_circuit(), 256, 13);
    ASSERT_TRUE(sv_res.success);

    MPSSimulator mps_sim;
    auto mps_res = mps_sim.run(make_circuit(), 16, 256, 13);

    // Both should give "10" (=|10⟩) with 100% probability.
    EXPECT_EQ(sv_res.counts.size(),  1u);
    EXPECT_GT(sv_res.counts.count("10"),  0u) << "SV: expected |10⟩ after SWAP";
    // Dump actual MPS counts to show what the UNITARY produced.
    std::string mps_actual;
    for (const auto& [k, v] : mps_res.counts)
        mps_actual += k + ":" + std::to_string(v) + " ";
    if (mps_actual.empty()) mps_actual = "(empty)";

    EXPECT_EQ(mps_res.counts.size(), 1u) << "MPS counts: " << mps_actual;
    EXPECT_GT(mps_res.counts.count("10"), 0u)
        << "MPS: expected |10⟩ after UNITARY SWAP, got: " << mps_actual;
}

// =============================================================================
// B4 — DensityMatrixSimulator RCCX matrix disagrees with StatevectorSimulator
// TODO: density_matrix_sim.cpp RCCX matrix mismatch (R.1.7.2 candidate fix).
// Test: run RCCX on |110⟩ under both simulators; compare counts + density-matrix
//       diagonal for the pure-state check.
// =============================================================================

TEST(BugRegression, B4_DensityMatrixRCCXMatchesSV) {
    // |110⟩: X on q0, X on q1; rccx(0,1,2).
    // RCCX: |110⟩ → i|111⟩  →  P("111") = 1.
    auto make_circuit = []() {
        QuantumCircuit qc(3);
        qc.x(0).x(1);
        qc.rccx(0, 1, 2);
        qc.measure_all();
        return qc;
    };

    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(make_circuit(), 512, 17);
    ASSERT_TRUE(sv_res.success);

    NoiseModel ideal;
    DensityMatrixSimulator dm_sim;
    auto dm_res = dm_sim.run(make_circuit(), ideal, 512, 17);
    ASSERT_TRUE(dm_res.success);

    // Both must give "111" with probability 1.
    EXPECT_EQ(sv_res.counts.size(), 1u);
    EXPECT_GT(sv_res.counts.count("111"), 0u) << "SV: expected |111⟩ after RCCX|110⟩";
    EXPECT_EQ(dm_res.counts.size(), 1u);
    EXPECT_GT(dm_res.counts.count("111"), 0u) << "DM: expected |111⟩ after RCCX|110⟩";

    // Cross-check: density matrix diagonal element (7,7) = 1 for pure |111⟩.
    // (index 7 = binary 111 in 3-qubit space)
    const auto& dm = dm_res.final_state;
    EXPECT_NEAR(dm(7, 7).real, 1.0, kTol) << "DM rho(7,7) should be 1 for pure |111⟩";
    EXPECT_NEAR(dm(7, 7).imag, 0.0, kTol);
    // All other diagonal elements must be 0.
    for (size_t i = 0; i < dm.dim; ++i) {
        if (i == 7) continue;
        EXPECT_NEAR(dm(i, i).real, 0.0, kTol) << "DM rho(" << i << "," << i << ") != 0";
    }
}

TEST(BugRegression, B4_DensityMatrixRCCXOnState101) {
    // |101⟩ → -|101⟩ (relative phase only); probability stays at |101⟩.
    auto make_circuit = []() {
        QuantumCircuit qc(3);
        qc.x(0).x(2);        // q0=1, q1=0, q2=1 → |101⟩
        qc.rccx(0, 1, 2);
        qc.measure_all();
        return qc;
    };

    StatevectorSimulator sv_sim;
    auto sv_res = sv_sim.run(make_circuit(), 256, 19);
    ASSERT_TRUE(sv_res.success);

    NoiseModel ideal;
    DensityMatrixSimulator dm_sim;
    auto dm_res = dm_sim.run(make_circuit(), ideal, 256, 19);
    ASSERT_TRUE(dm_res.success);

    EXPECT_EQ(sv_res.counts.size(),  1u);
    EXPECT_GT(sv_res.counts.count("101"), 0u) << "SV: expected |101⟩ after RCCX|101⟩";
    EXPECT_EQ(dm_res.counts.size(), 1u);
    EXPECT_GT(dm_res.counts.count("101"), 0u) << "DM: expected |101⟩ after RCCX|101⟩";
}

// =============================================================================
// B5 — Estimator::Options shots and noise_model are never used
// TODO: src/primitives/estimator.cpp always takes the exact statevector path.
// Test: apply 100% depolarizing noise on an X gate; noiseless ⟨Z⟩ = -1,
//       maximally mixed ⟨Z⟩ = 0.  If options are used the result must differ.
// =============================================================================

TEST(BugRegression, B5_EstimatorHonorsNoisyPath) {
    // X|0⟩ = |1⟩.  ⟨Z⟩ = -1 exact.
    // With 100% depolarizing noise on X: final state is maximally mixed → ⟨Z⟩ = 0.
    QuantumCircuit qc(1);
    qc.x(0);

    SparsePauliOp Z_op = SparsePauliOp::from_list({{"Z", Complex128(1.0, 0.0)}});

    // Sanity: noiseless exact value.
    Estimator est_ideal;
    double ideal_val = est_ideal.run_single(qc, Z_op);
    EXPECT_NEAR(ideal_val, -1.0, kTol) << "Noiseless ⟨Z⟩ of X|0⟩ must be -1";

    // Noisy: 100% depolarizing → maximally mixed → ⟨Z⟩ = 0.
    NoiseModel nm;
    nm.add_quantum_error(NoiseChannels::depolarizing(1.0), "x");

    Estimator est_noisy;
    est_noisy.options.shots = 8192;
    est_noisy.options.noise_model = nm;
    double noisy_val = est_noisy.run_single(qc, Z_op);

    // If bug B5 is open, run_single ignores options and returns -1 again.
    EXPECT_TRUE(std::abs(noisy_val - ideal_val) > 0.1)
        << "Estimator ignores noise_model/shots (bug B5 open): "
           "noisy_val=" << noisy_val << " ideal_val=" << ideal_val;
}

TEST(BugRegression, B5_EstimatorHonorsShotsSampling) {
    // If shots > 0, the result should be shot-sampled (not exact).
    // ⟨Z⟩ of |+⟩ is exactly 0 via statevector; shot-based estimate scatters around 0.
    // We cannot distinguish "exactly 0" from "sampled ≈ 0" without noise, so
    // we pair this with a biased noise model to make the true answer non-zero.
    // With 50% amplitude damping on H: state is partly |0⟩ → ⟨Z⟩ > 0.
    QuantumCircuit qc(1);
    qc.h(0);

    SparsePauliOp Z_op = SparsePauliOp::from_list({{"Z", Complex128(1.0, 0.0)}});

    NoiseModel nm;
    nm.add_quantum_error(NoiseChannels::amplitude_damping(0.9), "h");  // 90% damping → mostly |0⟩

    Estimator est;
    est.options.shots = 8192;
    est.options.noise_model = nm;
    double result = est.run_single(qc, Z_op);

    // If bug open: result = exact ⟨Z⟩ of |+⟩ = 0.
    // If bug fixed: result should be significantly > 0 due to amplitude damping.
    EXPECT_GT(result, 0.1)
        << "Estimator ignores noise_model (bug B5 open): result=" << result;
}

// =============================================================================
// B6 — DensityMatrixSimulator only applies after_gate noise; before_gate ignored
// TODO: before_gate entries in NoiseModel are silently ignored in the gate loop.
// Test: inject a 100% bit-flip BEFORE an X gate; the net operation on |0⟩ becomes
//       bit_flip ∘ X = X ∘ X = I → final state |0⟩.  If before_gate is ignored,
//       X is applied without the preceding flip → final state |1⟩.
// =============================================================================

TEST(BugRegression, B6_BeforeGateNoiseApplied) {
    // 100% bit-flip before X:  |0⟩ → |1⟩ (flip) → |0⟩ (X) = |0⟩.
    // If before_gate is ignored: X|0⟩ = |1⟩.
    NoiseModel nm;
    NoiseModel::GateError ge;
    ge.channel    = NoiseChannels::bit_flip(1.0);
    ge.qubits     = {};
    ge.after_gate = false;   // BEFORE the gate
    nm.basis_gate_errors["x"].push_back(ge);
    nm.noisy_gates.push_back("x");

    QuantumCircuit qc(1);
    qc.x(0);
    qc.measure_all();

    DensityMatrixSimulator dm_sim;
    auto res = dm_sim.run(qc, nm, 1024, 23);
    ASSERT_TRUE(res.success);

    // With before_gate noise: net result = |0⟩, so P("0") = 1.
    int count_0 = res.counts.count("0") ? res.counts.at("0") : 0;
    EXPECT_EQ(count_0, 1024)
        << "before_gate noise ignored (bug B6 open): "
           "expected all shots in |0⟩ but got count_0=" << count_0;
}

TEST(BugRegression, B6_BeforeGateNoiseAffectsDensityMatrix) {
    // Same setup, check density matrix directly without measurement.
    // 100% bit-flip before H on |0⟩:
    //   before: |0⟩ → |1⟩,  then H: |1⟩ → |−⟩ = (|0⟩−|1⟩)/√2.
    // If before_gate is ignored: H|0⟩ = |+⟩ = (|0⟩+|1⟩)/√2.
    // |+⟩ density matrix: rho(0,0)=0.5, rho(1,1)=0.5, rho(0,1)=rho(1,0)=0.5.
    // |−⟩ density matrix: rho(0,0)=0.5, rho(1,1)=0.5, rho(0,1)=rho(1,0)=-0.5.
    // The off-diagonal sign distinguishes the two states.
    NoiseModel nm;
    NoiseModel::GateError ge;
    ge.channel    = NoiseChannels::bit_flip(1.0);
    ge.qubits     = {};
    ge.after_gate = false;
    nm.basis_gate_errors["h"].push_back(ge);
    nm.noisy_gates.push_back("h");

    QuantumCircuit qc(1);
    qc.h(0);   // no measurement — inspect final density matrix

    DensityMatrixSimulator dm_sim;
    auto res = dm_sim.run(qc, nm, 0, 0);
    ASSERT_TRUE(res.success);

    const auto& dm = res.final_state;
    // |−⟩⟨−|: rho(0,1).real = -0.5 (if before_gate applied)
    // |+⟩⟨+|: rho(0,1).real = +0.5 (if before_gate ignored)
    EXPECT_NEAR(dm(0, 1).real, -0.5, kTol)
        << "before_gate noise ignored (bug B6 open): expected |−⟩ state but got |+⟩; "
           "rho(0,1).real=" << dm(0, 1).real;
}

// =============================================================================
// B7 — to_qasm2() emits UNITARY gates and PARAM_* gates as comments
// TODO: emit gate block definitions for UNITARY; emit standard calls for PARAM_*.
// Tests:
//   (a) Circuit with UNITARY gate: output must not contain "// gate".
//   (b) Circuit with symbolic PARAM_RX: output must not contain "// gate".
//   (c) Round-trip: UNITARY circuit serialised and re-parsed must have the same
//       gate count (not dropped).
// =============================================================================

TEST(BugRegression, B7_ToQasm2UnitaryNotComment) {
    const std::vector<Complex128> T_mat = {
        {1,0},{0,0},
        {0,0},{M_SQRT1_2, M_SQRT1_2}   // T gate = diag(1, e^{iπ/4})
    };

    QuantumCircuit qc(1);
    qc.unitary(T_mat, {0}, "my_t");

    const std::string qasm = qc.to_qasm2();

    EXPECT_EQ(qasm.find("// gate"), std::string::npos)
        << "to_qasm2() emitted UNITARY as a comment (bug B7 open).\nQASM output:\n" << qasm;
}

TEST(BugRegression, B7_ToQasm2ParamGateNotComment) {
    QuantumCircuit qc(2);
    qc.rx("theta", 0);    // symbolic PARAM_RX
    qc.rz("phi",   1);    // symbolic PARAM_RZ

    const std::string qasm = qc.to_qasm2();

    EXPECT_EQ(qasm.find("// gate"), std::string::npos)
        << "to_qasm2() emitted symbolic PARAM gate as a comment (bug B7 open).\nQASM output:\n"
        << qasm;
}

TEST(BugRegression, B7_ToQasm2UnitaryRoundTrip) {
    // A circuit whose only gate is a UNITARY.
    // Round-trip: to_qasm2() → from_qasm2() must preserve the gate count.
    const std::vector<Complex128> S_mat = {
        {1,0},{0,0},
        {0,0},{0,1}   // S gate = diag(1, i)
    };

    QuantumCircuit qc(1);
    qc.unitary(S_mat, {0}, "my_s");

    const std::string qasm = qc.to_qasm2();
    const QuantumCircuit parsed = QuantumCircuit::from_qasm2(qasm);

    // If UNITARY was emitted as a comment it would be dropped → size() = 0.
    EXPECT_EQ(parsed.size(), 1)
        << "UNITARY gate dropped on to_qasm2() round-trip (bug B7 open).";
}

// =============================================================================
// B8 — SabreLayout structural fix: SabreLayout::run and all SABRE internals
//       extracted from trivial_layout.cpp into sabre_layout.cpp (R.1.7.6).
//       Tests below verify correctness is preserved after the extraction.
// =============================================================================

// =============================================================================
// B4 extended — sweep all 8 computational basis inputs through RCCX on DM vs SV
// For each |abc⟩ input the RCCX truth table must be identical across both backends.
// This catches any off-diagonal error in the DM matrix that the |110⟩ test misses.
// =============================================================================

TEST(BugRegression, B4_DensityMatrixRCCXFullBasisSweep) {
    // RCCX truth table (probability-only; phases are observable only via interference):
    // |000⟩→|000⟩, |001⟩→|001⟩, |010⟩→|010⟩, |011⟩→|011⟩,
    // |100⟩→|100⟩, |101⟩→|101⟩, |110⟩→|111⟩, |111⟩→|110⟩  (probability-level)
    // rccx(0,1,2): flip q2 when q0=1 AND q1=1.
    // Input bits: bit0=q0, bit1=q1, bit2=q2 (LSB = q0).
    static const std::pair<int,int> table[8] = {
        {0b000, 0b000},
        {0b001, 0b001},
        {0b010, 0b010},
        {0b011, 0b111},  // q0=1,q1=1 → q2 flips 0→1
        {0b100, 0b100},
        {0b101, 0b101},
        {0b110, 0b110},  // q0=0 → no flip
        {0b111, 0b011},  // q0=1,q1=1 → q2 flips 1→0
    };

    auto to_bits = [](int v, int n) {
        std::string s(n, '0');
        for (int i = 0; i < n; ++i)
            if ((v >> i) & 1) s[n - 1 - i] = '1';
        return s;
    };

    NoiseModel ideal;
    DensityMatrixSimulator dm_sim;
    StatevectorSimulator  sv_sim;

    for (auto [input, expected] : table) {
        QuantumCircuit qc(3);
        // Prepare input state: set bits 0, 1, 2 by applying X where needed.
        if ((input >> 0) & 1) qc.x(0);
        if ((input >> 1) & 1) qc.x(1);
        if ((input >> 2) & 1) qc.x(2);
        qc.rccx(0, 1, 2);
        qc.measure_all();

        std::string exp_bits = to_bits(expected, 3);

        auto sv_res = sv_sim.run(qc, 128, 31);
        ASSERT_TRUE(sv_res.success) << "SV failed for input=" << to_bits(input,3);
        EXPECT_GT(sv_res.counts.count(exp_bits), 0u)
            << "SV wrong for input=" << to_bits(input,3) << " expected=" << exp_bits;

        auto dm_res = dm_sim.run(qc, ideal, 128, 31);
        ASSERT_TRUE(dm_res.success) << "DM failed for input=" << to_bits(input,3);
        EXPECT_GT(dm_res.counts.count(exp_bits), 0u)
            << "DM wrong for input=" << to_bits(input,3) << " expected=" << exp_bits;
    }
}

// =============================================================================
// B2 extended — three separate qregs; verify the offset arithmetic is correct
// across all three registers by checking qubit identity via X on each reg.
// =============================================================================

TEST(BugRegression, B2_Qasm2ThreeCustomRegisters) {
    // Three registers: alpha[1], beta[2], gamma[1] → 4 qubits total.
    // X only on beta[1] (global qubit 2); all others remain |0⟩.
    // MSB-first bitstring: q3 q2 q1 q0 = 0 1 0 0 = "0100".
    const std::string qasm = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg alpha[1];
qreg beta[2];
qreg gamma[1];
creg res[4];
x beta[1];
measure alpha[0] -> res[0];
measure beta[0]  -> res[1];
measure beta[1]  -> res[2];
measure gamma[0] -> res[3];
)";

    QuantumCircuit qc = QuantumCircuit::from_qasm2(qasm);
    ASSERT_EQ(qc.n_qubits, 4);

    StatevectorSimulator sim;
    auto res = sim.run(qc, 64, 37);
    ASSERT_TRUE(res.success);
    EXPECT_EQ(res.counts.size(), 1u);
    // beta[1] = global qubit 2; MSB-first: q3=0, q2=1, q1=0, q0=0 = "0100".
    EXPECT_GT(res.counts.count("0100"), 0u)
        << "Three-register offset wrong. Got: "
        << (res.counts.empty() ? "empty" : res.counts.begin()->first);
}

// =============================================================================
// B7 extended — multi-qubit UNITARY round-trip through to_qasm2() / from_qasm2()
// A 2-qubit UNITARY (CZ as custom matrix) must survive serialisation.
// =============================================================================

TEST(BugRegression, B7_ToQasm2TwoQubitUnitaryRoundTrip) {
    // CZ = diag(1,1,1,-1) as a raw 4×4 custom unitary.
    const std::vector<Complex128> CZ_mat = {
        {1,0},{0,0},{0,0},{0,0},
        {0,0},{1,0},{0,0},{0,0},
        {0,0},{0,0},{1,0},{0,0},
        {0,0},{0,0},{0,0},{-1,0}
    };

    QuantumCircuit qc(2);
    qc.h(0);
    qc.unitary(CZ_mat, {0, 1}, "my_cz");
    qc.h(0);

    const std::string qasm = qc.to_qasm2();

    // No comment-emission.
    EXPECT_EQ(qasm.find("// gate"), std::string::npos)
        << "to_qasm2() emitted 2-qubit UNITARY as comment (bug B7 open). Output:\n" << qasm;

    // Gate count preserved on round-trip: H + UNITARY + H = 3 instructions.
    const QuantumCircuit parsed = QuantumCircuit::from_qasm2(qasm);
    EXPECT_EQ(parsed.size(), 3)
        << "2-qubit UNITARY dropped on to_qasm2() round-trip (bug B7 open).";
}

// =============================================================================
// B6 extended — before_gate noise across multiple gates in one circuit
// 100% bit-flip before every H on a 2-qubit circuit; both qubits affected.
// =============================================================================

TEST(BugRegression, B6_BeforeGateNoiseMultiQubitCircuit) {
    // 100% bit-flip before H on every qubit:
    //   q0: |0⟩ → |1⟩ (flip) → H → |−⟩
    //   q1: |0⟩ → |1⟩ (flip) → H → |−⟩
    // Product state |−⟩⊗|−⟩ has equal probability for "00","01","10","11"
    // BUT off-diagonals carry the minus signs.
    // If before_gate is ignored: H|0⟩ = |+⟩ for both → same distribution.
    // We distinguish via the density matrix off-diagonal sign on qubit 0.
    NoiseModel nm;
    NoiseModel::GateError ge;
    ge.channel    = NoiseChannels::bit_flip(1.0);
    ge.qubits     = {};
    ge.after_gate = false;
    nm.basis_gate_errors["h"].push_back(ge);
    nm.noisy_gates.push_back("h");

    QuantumCircuit qc(2);
    qc.h(0).h(1);  // no measurement

    DensityMatrixSimulator dm_sim;
    auto res = dm_sim.run(qc, nm, 0, 0);
    ASSERT_TRUE(res.success);

    const auto& dm = res.final_state;
    // For |−⟩⊗|−⟩ the two-qubit density matrix has rho(0,1) = (-0.5)⊗(-0.5) ← sign matters.
    // Specifically rho(0,3): |00⟩⟨11| element = (+0.5)*(+0.5)=+0.25 for |+⟩⊗|+⟩
    //                                          = (-0.5)*(-0.5)=+0.25 for |−⟩⊗|−⟩
    // Use rho(0,1): |00⟩⟨01| = (+0.5)*(−0.5) for |+⟩⊗|−⟩-like terms.
    // Actually easiest: check single-qubit reduced density matrix of qubit 0.
    // rho_0 = Tr_1(rho). For |−⟩: rho_0(0,1) = -0.5; for |+⟩: rho_0(0,1) = +0.5.
    // Two-qubit state: rho(0,1) = rho_0(0,0)*rho_1(0,1) + rho_0(0,1)*rho_1(0,0) — not direct.
    // Simpler: check full 4×4 dm row 0 col 1 = rho(|00⟩, |01⟩).
    // |−⊗−⟩: amp(00)=0.5, amp(01)=−0.5, amp(10)=−0.5, amp(11)=0.5  (MSB-first)
    // rho(0,1) = (0.5)*(−0.5)* = −0.25   for |−⊗−⟩
    // rho(0,1) = (0.5)*(0.5)*  = +0.25   for |+⊗+⟩
    EXPECT_NEAR(dm(0, 1).real, -0.25, kTol)
        << "before_gate noise not applied to H (bug B6 open): "
           "expected |−⟩⊗|−⟩ but got |+⟩⊗|+⟩; rho(0,1).real=" << dm(0, 1).real;
}

// =============================================================================
// B8_SabreLayoutFunctional — split B8 into two sub-tests:
//   (a) level-2 transpile produces a valid GHZ distribution,
//   (b) level-2 transpile on a random circuit does not crash.
// =============================================================================

TEST(BugRegression, B8_SabreLayoutFunctional) {
    // 5-qubit GHZ on a linear 5-qubit topology — not all-to-all, so SabreSwap
    // must insert SWAP gates and SabreLayout must find a good qubit mapping.
    QuantumCircuit qc(5);
    qc.h(0).cx(0,1).cx(1,2).cx(2,3).cx(3,4).measure_all();

    CouplingMap linear5 = CouplingMap::linear(5);
    QuantumCircuit transpiled = transpile(qc, linear5, {}, 2);

    StatevectorSimulator sim;
    auto res = sim.run(transpiled, 2048, 29);
    ASSERT_TRUE(res.success);

    // GHZ: only "00000" and "11111" should appear.
    EXPECT_GE(res.counts.size(), 1u);
    for (const auto& [bits, count] : res.counts)
        EXPECT_TRUE(bits == "00000" || bits == "11111")
            << "Unexpected bitstring after SabreLayout transpile: " << bits;
}

TEST(BugRegression, B8_SabreLayoutNocrashOnDenseCircuit) {
    // All-to-all circuit on a sparse linear map: every qubit interacts with
    // every other qubit, so SabreSwap must insert many SWAPs without crashing.
    QuantumCircuit qc(4);
    qc.cx(0,3).cx(1,3).cx(2,3).cx(0,2).measure_all();

    CouplingMap linear4 = CouplingMap::linear(4);
    QuantumCircuit transpiled;
    ASSERT_NO_THROW({
        transpiled = transpile(qc, linear4, {}, 2);
    }) << "transpile(level=2) crashed on a dense circuit (B8)";

    StatevectorSimulator sim;
    auto res = sim.run(transpiled, 512, 41);
    ASSERT_TRUE(res.success)
        << "Transpiled circuit simulation failed after SabreLayout (B8)";
}
