// R.1.18.1 test suite — QASM / JSON representability of MCX / MCP / PERMUTATION.
// Covers the R.1.18.0 export and import surfaces (#52): the qasm2
// throw-by-default + opt-in decompose contract, the qasm3 `ctrl(k) @`
// emission and always-lowered PERMUTATION, the parser's counted-modifier
// form and wide-stack fast path, and the lossless JSON round-trip with the
// first-class `permutation` field.
//
// Equivalence strategy: exported text is re-imported and simulated against
// the native circuit from |0...0> behind a fixed non-trivial preamble
// (H / T / RZ ladder), so every column of the op's unitary carries distinct
// amplitude and phase weight into the comparison.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

constexpr double kTol = 1e-9;  // text round-trips print %.15g angles

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// Fixed preamble giving every basis state a distinct amplitude and phase, so
// op-equivalence from |0...0> exercises the full unitary.
void preamble(QuantumCircuit& qc) {
    for (int q = 0; q < qc.n_qubits; ++q) qc.h(q);
    for (int q = 0; q + 1 < qc.n_qubits; ++q) qc.t(q);
    qc.rz(0.7, 0);
}

std::vector<Complex128> run_sv(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 0);
    return res.final_state.amplitudes();
}

void expect_amps_close(const std::vector<Complex128>& a,
                       const std::vector<Complex128>& b, double tol = kTol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, tol) << "re mismatch at " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, tol) << "im mismatch at " << i;
    }
}

// The three-op fixture used across the export tests.
QuantumCircuit fixture_mcx(int k) {
    QuantumCircuit qc(k + 1);
    preamble(qc);
    std::vector<int> controls;
    for (int c = 0; c < k; ++c) controls.push_back(c);
    qc.mcx(controls, k);
    return qc;
}

QuantumCircuit fixture_mcp(int m, double lambda) {
    QuantumCircuit qc(m);
    preamble(qc);
    std::vector<int> qubits;
    for (int q = 0; q < m; ++q) qubits.push_back(q);
    qc.mcp(lambda, qubits);
    return qc;
}

QuantumCircuit fixture_perm_general(int nq) {
    QuantumCircuit qc(nq);
    preamble(qc);
    qc.permute({2, 0, 3, 1}, {1, 3});  // general (non-relabel) 2-qubit map
    return qc;
}

int count_type(const QuantumCircuit& qc, GT t) {
    int n = 0;
    for (const auto& inst : qc.instructions) n += (inst.type == t) ? 1 : 0;
    return n;
}

} // namespace

// =============================================================================
// QASM 2: throw by default, lower on request, equivalent when lowered
// =============================================================================

TEST(R1181Export, Qasm2DefaultThrowsForAllThreeOps) {
    EXPECT_THROW(fixture_mcx(3).to_qasm2(), std::runtime_error);
    EXPECT_THROW(fixture_mcp(3, 0.5).to_qasm2(), std::runtime_error);
    EXPECT_THROW(fixture_perm_general(4).to_qasm2(), std::runtime_error);
}

TEST(R1181Export, Qasm2ThrowMessageNamesTheOptOut) {
    QuantumCircuit qc(4);
    qc.mcx({0, 1, 2}, 3);
    try {
        qc.to_qasm2();
        FAIL() << "default to_qasm2 must throw on mcx";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(contains(e.what(), "mcx")) << e.what();
        EXPECT_TRUE(contains(e.what(), "decompose_unrepresentable")) << e.what();
    }
}

TEST(R1181Export, Qasm2OptInDecomposeRoundTripsEquivalently) {
    QasmExportOptions opts;
    opts.decompose_unrepresentable = true;

    for (auto build : { +[]{ return fixture_mcx(3); },
                        +[]{ return fixture_mcp(3, 0.9123); },
                        +[]{ return fixture_perm_general(4); } }) {
        QuantumCircuit qc = build();
        SCOPED_TRACE(qc.n_qubits);
        std::string text;
        ASSERT_NO_THROW(text = qc.to_qasm2(opts));
        // The lowered text must carry no trace of the high-level names.
        EXPECT_FALSE(contains(text, "mcx"));
        EXPECT_FALSE(contains(text, "mcp"));
        EXPECT_FALSE(contains(text, "permutation"));

        auto back = QuantumCircuit::from_qasm2(text);
        expect_amps_close(run_sv(back), run_sv(qc));
    }
}

TEST(R1181Export, Qasm2OptInLeavesCircuitObjectUntouched) {
    QuantumCircuit qc = fixture_mcx(3);
    const size_t before = qc.instructions.size();
    QasmExportOptions opts;
    opts.decompose_unrepresentable = true;
    (void)qc.to_qasm2(opts);
    EXPECT_EQ(qc.instructions.size(), before);
    EXPECT_EQ(count_type(qc, GT::MCX), 1);
}

// =============================================================================
// QASM 3: native ctrl(k) emission, always-lowered PERMUTATION
// =============================================================================

TEST(R1181Export, Qasm3EmitsCountedCtrlForms) {
    {
        auto text = fixture_mcx(3).to_qasm3();
        EXPECT_TRUE(contains(text, "ctrl(3) @ x")) << text;
    }
    {
        auto text = fixture_mcp(3, 0.5).to_qasm3();
        EXPECT_TRUE(contains(text, "ctrl(2) @ p(")) << text;
    }
    {
        // Degenerate forms: control-free mcx -> plain x; 1-qubit mcp -> p.
        QuantumCircuit qc(2);
        qc.mcx({}, 1);
        qc.mcp(0.25, {0});
        auto text = qc.to_qasm3();
        EXPECT_FALSE(contains(text, "ctrl(")) << text;
        EXPECT_TRUE(contains(text, "x q[1]")) << text;
        EXPECT_TRUE(contains(text, "p(")) << text;
    }
}

TEST(R1181Export, Qasm3RoundTripRestoresWideNodes) {
    // k >= 3 comes back as a first-class MCX node (not CCX chains, not a
    // dense UNITARY); m >= 3 as MCP. The amplitudes agree either way.
    {
        QuantumCircuit qc = fixture_mcx(4);
        auto back = QuantumCircuit::from_qasm3(qc.to_qasm3());
        EXPECT_EQ(count_type(back, GT::MCX), 1);
        EXPECT_EQ(count_type(back, GT::UNITARY), 0);
        expect_amps_close(run_sv(back), run_sv(qc));
    }
    {
        QuantumCircuit qc = fixture_mcp(4, -0.777);
        auto back = QuantumCircuit::from_qasm3(qc.to_qasm3());
        EXPECT_EQ(count_type(back, GT::MCP), 1);
        expect_amps_close(run_sv(back), run_sv(qc));
    }
}

TEST(R1181Export, Qasm3SmallStacksCanonicaliseToNamedGates) {
    // Deliberate canonicalisation: mcx k <= 2 re-imports as cx / ccx, and a
    // 2-qubit mcp as cp — identical unitaries under the named types.
    {
        QuantumCircuit qc(2);
        qc.mcx({0}, 1);
        auto back = QuantumCircuit::from_qasm3(qc.to_qasm3());
        EXPECT_EQ(count_type(back, GT::CX), 1);
        EXPECT_EQ(count_type(back, GT::MCX), 0);
    }
    {
        QuantumCircuit qc(3);
        qc.mcx({0, 1}, 2);
        auto back = QuantumCircuit::from_qasm3(qc.to_qasm3());
        EXPECT_EQ(count_type(back, GT::CCX), 1);
        EXPECT_EQ(count_type(back, GT::MCX), 0);
    }
    {
        QuantumCircuit qc(2);
        qc.mcp(0.4, {0, 1});
        auto back = QuantumCircuit::from_qasm3(qc.to_qasm3());
        EXPECT_EQ(count_type(back, GT::CP), 1);
        EXPECT_EQ(count_type(back, GT::MCP), 0);
    }
}

TEST(R1181Export, Qasm3PermutationAlwaysLowersAtExport) {
    {
        // General map: no permutation token survives; simulation agrees.
        QuantumCircuit qc = fixture_perm_general(4);
        auto text = qc.to_qasm3();
        EXPECT_FALSE(contains(text, "permutation")) << text;
        auto back = QuantumCircuit::from_qasm3(text);
        EXPECT_EQ(count_type(back, GT::PERMUTATION), 0);
        expect_amps_close(run_sv(back), run_sv(qc));
    }
    {
        // Relabel subclass: the lowered text is a plain swap network.
        QuantumCircuit qc(3);
        preamble(qc);
        std::vector<int> perm(8);
        for (int x = 0; x < 8; ++x) {
            int img = 0;
            for (int j = 0; j < 3; ++j)
                if (x & (1 << j)) img |= (1 << ((j + 1) % 3));
            perm[x] = img;
        }
        qc.permute(perm, {0, 1, 2});
        auto text = qc.to_qasm3();
        EXPECT_TRUE(contains(text, "swap")) << text;
        auto back = QuantumCircuit::from_qasm3(text);
        expect_amps_close(run_sv(back), run_sv(qc));
    }
}

// =============================================================================
// QASM 3 parser: counted ctrl form, composition, folding, fail-loud count
// =============================================================================

TEST(R1181Export, ParserAcceptsCountedCtrl) {
    const std::string src =
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "qubit[3] q;\n"
        "ctrl(2) @ x q[0], q[1], q[2];\n";
    auto qc = QuantumCircuit::from_qasm3(src);
    EXPECT_EQ(count_type(qc, GT::CCX), 1);
}

TEST(R1181Export, ParserComposesCountedAndBareCtrl) {
    // ctrl(2) @ ctrl @ x = a 3-control stack -> first-class MCX.
    const std::string src =
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "qubit[4] q;\n"
        "ctrl(2) @ ctrl @ x q[0], q[1], q[2], q[3];\n";
    auto qc = QuantumCircuit::from_qasm3(src);
    ASSERT_EQ(count_type(qc, GT::MCX), 1);
    EXPECT_EQ(qc.instructions.back().qubits, (std::vector<int>{0, 1, 2, 3}));
}

TEST(R1181Export, ParserFoldsInvAndPowIntoWideStacks) {
    {
        // inv over a phase stack negates the angle on the restored MCP node.
        const std::string src =
            "OPENQASM 3.0;\n"
            "include \"stdgates.inc\";\n"
            "qubit[3] q;\n"
            "inv @ ctrl(2) @ p(0.5) q[0], q[1], q[2];\n";
        auto qc = QuantumCircuit::from_qasm3(src);
        ASSERT_EQ(count_type(qc, GT::MCP), 1);
        ASSERT_EQ(qc.instructions.back().params.size(), 1u);
        EXPECT_NEAR(qc.instructions.back().params[0], -0.5, 1e-15);
    }
    {
        // pow(2) over a self-inverse wide-X stack collapses to identity.
        const std::string src =
            "OPENQASM 3.0;\n"
            "include \"stdgates.inc\";\n"
            "qubit[4] q;\n"
            "pow(2) @ ctrl(3) @ x q[0], q[1], q[2], q[3];\n";
        auto qc = QuantumCircuit::from_qasm3(src);
        EXPECT_EQ(count_type(qc, GT::MCX), 0);
        EXPECT_EQ(count_type(qc, GT::UNITARY), 0);
    }
}

TEST(R1181Export, ParserRejectsNonPositiveCtrlCount) {
    const std::string src =
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "qubit[2] q;\n"
        "ctrl(0) @ x q[0], q[1];\n";
    EXPECT_THROW(QuantumCircuit::from_qasm3(src), std::runtime_error);
}

// =============================================================================
// JSON: lossless structural round-trip for all three ops
// =============================================================================

TEST(R1181Export, JsonRoundTripsAllThreeOpsLosslessly) {
    QuantumCircuit qc(4);
    preamble(qc);
    qc.mcx({0, 2}, 3);
    qc.mcp(0.9123, {1, 2, 3});
    qc.permute({2, 0, 3, 1}, {1, 3}, "modmul");

    auto back = QuantumCircuit::from_json(qc.to_json());
    ASSERT_EQ(back.instructions.size(), qc.instructions.size());
    for (size_t i = 0; i < qc.instructions.size(); ++i) {
        SCOPED_TRACE("instruction " + std::to_string(i));
        const auto& a = qc.instructions[i];
        const auto& b = back.instructions[i];
        EXPECT_EQ(b.type, a.type);
        EXPECT_EQ(b.qubits, a.qubits);
        EXPECT_EQ(b.permutation, a.permutation);
        ASSERT_EQ(b.params.size(), a.params.size());
        for (size_t j = 0; j < a.params.size(); ++j)
            EXPECT_NEAR(b.params[j], a.params[j], 1e-15);
    }
    expect_amps_close(run_sv(back), run_sv(qc), 1e-12);
}

TEST(R1181Export, JsonPermutationFieldSurvivesNonSymmetricMap) {
    // Non-symmetric Shor-style map: x -> (3*x) mod 16. The exact index map
    // must survive byte-for-byte; a reversed or re-based map cannot match.
    const int dim = 16;
    std::vector<int> perm(dim);
    for (int x = 0; x < dim; ++x) perm[x] = (3 * x) % dim;
    QuantumCircuit qc(4);
    qc.permute(perm, {0, 1, 2, 3});

    auto back = QuantumCircuit::from_json(qc.to_json());
    ASSERT_EQ(count_type(back, GT::PERMUTATION), 1);
    EXPECT_EQ(back.instructions.back().permutation, perm);
}

TEST(R1181Export, JsonImportedGarbagePermutationFailsLoudDownstream) {
    // from_json performs structural parsing only; a hand-crafted non-bijective
    // map must be rejected the moment it reaches a lowering consumer.
    QuantumCircuit qc(2);
    qc.permute({1, 0, 3, 2}, {0, 1});
    std::string good = qc.to_json();
    // Corrupt the index map: {1,0,3,2} -> {1,1,3,2} (duplicate image).
    const std::string needle = "\"permutation\":[1,0,3,2]";
    auto pos = good.find(needle);
    ASSERT_NE(pos, std::string::npos) << good;
    std::string bad = good;
    bad.replace(pos, needle.size(), "\"permutation\":[1,1,3,2]");

    auto back = QuantumCircuit::from_json(bad);
    EXPECT_THROW((void)back.to_qasm3(), std::invalid_argument);
}
