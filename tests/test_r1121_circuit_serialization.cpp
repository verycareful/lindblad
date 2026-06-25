// R.1.12.1 total-coverage suite, Batch 1: QuantumCircuit serialization
// (to_json/from_json, to_qasm2/from_qasm2, to_qasm3/from_qasm3).
// Plan: docs (R.1.12.1 coverage plan), section "Batch 1: foundations".
//
// JSON is exercised by exact field-by-field round-trip (it preserves matrices,
// labels, conditions and full double precision). QASM is exercised by SEMANTIC
// round-trip: emit, re-parse, and compare final statevectors. QASM round-trips
// use only the gate set both the emitter and the parser support (multi-qubit
// UNITARY and global phase are deliberately excluded from QASM2 round-trips
// because that lowering is documented-lossy). Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

// Final amplitudes of a measurement-free circuit (|0...0> input, seed fixed).
std::vector<Complex128> final_amps(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 1);
    EXPECT_TRUE(res.success) << res.error_message;
    return res.final_state.amplitudes();
}

void expect_same_state(const QuantumCircuit& a, const QuantumCircuit& b,
                       double tol = 1e-9) {
    auto aa = final_amps(a);
    auto bb = final_amps(b);
    ASSERT_EQ(aa.size(), bb.size());
    for (size_t i = 0; i < aa.size(); ++i) {
        EXPECT_NEAR(aa[i].real, bb[i].real, tol) << "amp " << i;
        EXPECT_NEAR(aa[i].imag, bb[i].imag, tol) << "amp " << i;
    }
}

void expect_inst_eq(const Instruction& a, const Instruction& b) {
    EXPECT_EQ(a.type, b.type);
    EXPECT_EQ(a.qubits, b.qubits);
    EXPECT_EQ(a.clbits, b.clbits);
    ASSERT_EQ(a.params.size(), b.params.size());
    for (size_t i = 0; i < a.params.size(); ++i)
        EXPECT_NEAR(a.params[i], b.params[i], 1e-12) << "param " << i;
    EXPECT_EQ(a.param_names, b.param_names);
    ASSERT_EQ(a.matrix.size(), b.matrix.size());
    for (size_t i = 0; i < a.matrix.size(); ++i) {
        EXPECT_NEAR(a.matrix[i].real, b.matrix[i].real, 1e-12) << "matrix re " << i;
        EXPECT_NEAR(a.matrix[i].imag, b.matrix[i].imag, 1e-12) << "matrix im " << i;
    }
    EXPECT_EQ(a.label, b.label);
    EXPECT_EQ(a.condition_clbit, b.condition_clbit);
    EXPECT_EQ(a.condition_value, b.condition_value);
}

}  // namespace

// =============================================================================
// JSON: full structural round-trip
// =============================================================================

TEST(R1121Serialization, JsonRoundTripPreservesEveryField) {
    QuantumCircuit qc(3, 2, "round_trip");
    qc.h(0);
    qc.rx(0.1234567890123, 1);                 // full-precision param
    qc.cx(0, 1);
    qc.cu(0.2, -0.3, 0.4, 0.5, 1, 2);          // 4 params, 2 qubits
    qc.measure(0, 0);
    qc.add_if(1, 1, GT::X, {2});               // conditioned gate
    std::vector<Complex128> M = {{0.6, 0.0}, {0.0, 0.8},
                                 {0.0, 0.8}, {0.6, 0.0}};
    qc.unitary(M, {1}, "custom_box");          // matrix + label

    auto back = QuantumCircuit::from_json(qc.to_json());
    EXPECT_EQ(back.n_qubits, 3);
    EXPECT_EQ(back.n_clbits, 2);
    EXPECT_EQ(back.name, "round_trip");
    ASSERT_EQ(back.instructions.size(), qc.instructions.size());
    for (size_t i = 0; i < qc.instructions.size(); ++i) {
        SCOPED_TRACE("instruction " + std::to_string(i));
        expect_inst_eq(qc.instructions[i], back.instructions[i]);
    }
}

TEST(R1121Serialization, JsonPreservesParameterNames) {
    QuantumCircuit qc(2);
    qc.rx("alpha", 0);
    qc.ry("beta", 1);
    auto back = QuantumCircuit::from_json(qc.to_json());
    ASSERT_EQ(back.parameter_names.size(), 2u);
    EXPECT_EQ(back.parameter_names[0], "alpha");
    EXPECT_EQ(back.parameter_names[1], "beta");
    // The symbolic name survives on the instruction even though the concrete
    // type is reconstructed (gate_name() collapses PARAM_RX -> "rx").
    ASSERT_EQ(back.instructions.size(), 2u);
    EXPECT_EQ(back.instructions[0].param_names, (std::vector<std::string>{"alpha"}));
}

TEST(R1121Serialization, JsonEscapesSpecialCharactersInNames) {
    QuantumCircuit qc(1, 0, "weird\"name\\with\nnewline");
    std::vector<Complex128> X = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    qc.unitary(X, {0}, "box\"with\\quote");
    auto back = QuantumCircuit::from_json(qc.to_json());
    EXPECT_EQ(back.name, "weird\"name\\with\nnewline");
    ASSERT_EQ(back.instructions.size(), 1u);
    EXPECT_EQ(back.instructions[0].label, "box\"with\\quote");
}

TEST(R1121Serialization, JsonToleratesUnknownKeys) {
    // Unknown top-level and per-instruction keys must be skipped, not rejected.
    const std::string j =
        "{\"version\":\"1.0\",\"name\":\"u\",\"n_qubits\":2,\"n_clbits\":0,"
        "\"surprise\":42,\"parameter_names\":[],\"instructions\":["
        "{\"gate\":\"h\",\"qubits\":[0],\"clbits\":[],\"params\":[],"
        "\"mystery\":[1,2,3]},"
        "{\"gate\":\"cx\",\"qubits\":[0,1],\"clbits\":[],\"params\":[]}]}";
    auto qc = QuantumCircuit::from_json(j);
    EXPECT_EQ(qc.n_qubits, 2);
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[1].type, GT::CX);
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{0, 1}));
}

TEST(R1121Serialization, ToJsonThrowsOnUnboundSymbolicExpression) {
    QuantumCircuit qc(1);
    Instruction inst;
    inst.type = GT::RZ;
    inst.qubits = {0};
    inst.param_exprs.push_back(ParamExpr::make_name("theta"));
    qc.instructions.push_back(std::move(inst));
    EXPECT_THROW(qc.to_json(), std::runtime_error);
}

// =============================================================================
// QASM 2.0: semantic round-trip + error path
// =============================================================================

TEST(R1121Serialization, Qasm2MinimalRoundTrip) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.7, 1);
    auto back = QuantumCircuit::from_qasm2(qc.to_qasm2());
    expect_same_state(qc, back);
}

TEST(R1121Serialization, Qasm2ComprehensiveRoundTrip) {
    // Every gate here is emitted by to_qasm2() and accepted by from_qasm2().
    QuantumCircuit qc(3);
    qc.h(0).x(1).y(2).z(0).s(1).sdg(2).t(0).tdg(1).sx(2);
    qc.rx(0.7, 0).ry(-0.4, 1).rz(1.1, 2).p(0.3, 0);
    qc.u(0.5, 0.2, -0.6, 1).u1(0.9, 2).u2(0.4, -0.2, 0).u3(0.3, 0.1, 0.8, 1);
    qc.cx(0, 1).cy(1, 2).cz(0, 2).ch(2, 0).swap(0, 1);
    qc.crx(0.6, 0, 1).cry(-0.2, 1, 2).crz(0.8, 2, 0).cp(0.5, 0, 2);
    qc.ccx(0, 1, 2).cswap(1, 0, 2);
    qc.rxx(0.6, 0, 1).ryy(-0.3, 0, 1).rzz(0.5, 1, 2);
    auto back = QuantumCircuit::from_qasm2(qc.to_qasm2());
    expect_same_state(qc, back);
}

TEST(R1121Serialization, Qasm2RoundTripsMeasurement) {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    auto back = QuantumCircuit::from_qasm2(qc.to_qasm2());
    int n_measure = 0;
    for (const auto& inst : back.instructions)
        if (inst.type == GT::MEASURE) ++n_measure;
    EXPECT_EQ(n_measure, 2);
    EXPECT_GE(back.n_clbits, 2);
}

TEST(R1121Serialization, Qasm2ThrowsWhenNoQregDeclared) {
    EXPECT_THROW(QuantumCircuit::from_qasm2("OPENQASM 2.0;\n"),
                 std::runtime_error);
}

// =============================================================================
// QASM 3.0: semantic round-trip + lossless global phase
// =============================================================================

TEST(R1121Serialization, Qasm3MinimalRoundTrip) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.7, 1);
    auto back = QuantumCircuit::from_qasm3(qc.to_qasm3());
    expect_same_state(qc, back);
}

TEST(R1121Serialization, Qasm3RoundTripWithEntanglers) {
    QuantumCircuit qc(3);
    qc.h(0).x(1).z(2).rx(0.5, 0).ry(-0.3, 1).rz(0.9, 2);
    qc.cx(0, 1).cz(1, 2).swap(0, 2);
    auto back = QuantumCircuit::from_qasm3(qc.to_qasm3());
    expect_same_state(qc, back);
}

TEST(R1121Serialization, Qasm3EmitsGlobalPhaseForUnitary) {
    // A 1-qubit UNITARY that is a pure global phase e^{i*0.5} * I must emit a
    // gphase(...) directive (QASM 3 represents global phase losslessly; QASM 2
    // cannot).
    QuantumCircuit qc(1);
    std::vector<Complex128> g = {Complex128::exp_i(0.5), {0.0, 0.0},
                                 {0.0, 0.0}, Complex128::exp_i(0.5)};
    qc.unitary(g, {0});
    const std::string qasm = qc.to_qasm3();
    EXPECT_NE(qasm.find("gphase"), std::string::npos)
        << "QASM3 must emit gphase for a non-trivial global phase";
}
