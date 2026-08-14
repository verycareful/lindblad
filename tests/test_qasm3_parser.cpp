// =============================================================================
// test_qasm3_parser.cpp : full coverage test suite for the QASM 3.0 parser
// =============================================================================
// Verifies every public branch of `QuantumCircuit::from_qasm3()`, the
// `ParamExpr` expression tree, and `QuantumCircuit::bind_parameters()`.
//
// Suite layout:
//   1.  Register declarations (qubit/bit, qreg/creg, multi-register)
//   2.  Pragma directives (OPENQASM, include, comments, whitespace)
//   3.  Standard gates (1q, 2q, 3q, all stdgates and aliases)
//   4.  Measurements, reset, barrier
//   5.  Modifier resolution (ctrl, inv, pow, chained, fast paths)
//   6.  Matrix fallback path
//   7.  Parameter expressions (literals, names, precedence, constants)
//   8.  Custom gate definitions and recursive inlining
//   9.  Classical conditioning (if / else, single-bit)
//   10. Symbolic parameters and bind_parameters()
//   11. ParamExpr factories, eval, copy, errors
//   12. Peephole optimisation
//   13. Round-trips against to_qasm3()
//   14. Lexer error reporting and unsupported-construct throws
//   15. Multi-register offset resolution

#include <gtest/gtest.h>
#include "lindblad/circuit.hpp"

#include <cmath>
#include <stdexcept>
#include <string>


using namespace lindblad;
using GT = Instruction::GateType;

// ============================================================================
// Helpers
// ============================================================================

// Parse a QASM 3 body, prepending the standard header.
static QuantumCircuit parse3(const std::string& body) {
    return QuantumCircuit::from_qasm3(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n" + body);
}

// Parse with custom header (for tests that need an `input` declaration etc).
static QuantumCircuit parse3_raw(const std::string& full) {
    return QuantumCircuit::from_qasm3(full);
}

// Convenience: assert there is exactly one instruction of `t` on `qubits`.
static void expect_single(const QuantumCircuit& qc, GT t,
                          std::vector<int> qubits) {
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, t);
    EXPECT_EQ(qc.instructions[0].qubits, qubits);
}

// ============================================================================
// 1. Register declarations
// ============================================================================

TEST(QASM3ParserTest, SingleQubitRegister) {
    auto qc = parse3("qubit[3] q;\n");
    EXPECT_EQ(qc.n_qubits, 3);
    EXPECT_EQ(qc.n_clbits, 0);
    EXPECT_EQ(qc.instructions.size(), 0u);
}

TEST(QASM3ParserTest, QubitAndBitRegister) {
    auto qc = parse3("qubit[2] q;\nbit[2] c;\n");
    EXPECT_EQ(qc.n_qubits, 2);
    EXPECT_EQ(qc.n_clbits, 2);
}

TEST(QASM3ParserTest, MultipleQubitRegisters) {
    auto qc = parse3("qubit[2] a;\nqubit[3] b;\n");
    EXPECT_EQ(qc.n_qubits, 5);
}

TEST(QASM3ParserTest, MultipleBitRegisters) {
    auto qc = parse3("qubit[1] q;\nbit[2] c;\nbit[3] d;\n");
    EXPECT_EQ(qc.n_clbits, 5);
}

TEST(QASM3ParserTest, SingleQubitNoBrackets) {
    auto qc = parse3("qubit q;\nh q;\n");
    EXPECT_EQ(qc.n_qubits, 1);
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].qubits[0], 0);
}

TEST(QASM3ParserTest, SingleBitNoBrackets) {
    auto qc = parse3("qubit q;\nbit c;\nc = measure q;\n");
    EXPECT_EQ(qc.n_clbits, 1);
}

TEST(QASM3ParserTest, LegacyQregForm) {
    auto qc = parse3("qreg q[2];\n");
    EXPECT_EQ(qc.n_qubits, 2);
}

TEST(QASM3ParserTest, LegacyCregForm) {
    auto qc = parse3("qubit[1] q;\ncreg c[2];\n");
    EXPECT_EQ(qc.n_clbits, 2);
}

TEST(QASM3ParserTest, MissingQubitRegisterThrows) {
    EXPECT_THROW(parse3(""), std::runtime_error);
}

// ============================================================================
// 2. Pragmas, comments, whitespace
// ============================================================================

TEST(QASM3ParserTest, OPENQASMDirectiveSkipped) {
    auto qc = QuantumCircuit::from_qasm3(
        "OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n");
    EXPECT_EQ(qc.n_qubits, 1);
    EXPECT_EQ(qc.instructions.size(), 1u);
}

TEST(QASM3ParserTest, IncludeDirectiveSkipped) {
    auto qc = QuantumCircuit::from_qasm3(
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\n");
    EXPECT_EQ(qc.n_qubits, 1);
}

TEST(QASM3ParserTest, LineCommentSkipped) {
    auto qc = parse3("// a comment\nqubit[1] q;\n// another\nh q[0];\n");
    EXPECT_EQ(qc.n_qubits, 1);
    ASSERT_EQ(qc.instructions.size(), 1u);
}

TEST(QASM3ParserTest, BlockCommentSkipped) {
    auto qc = parse3("qubit[1] q;\n/* multi\nline\ncomment */\nh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}

TEST(QASM3ParserTest, TabsAndCRTolerated) {
    auto qc = parse3("qubit[1]\tq;\r\nh\tq[0];\r\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}

// ============================================================================
// 3. Standard gates : 1-qubit no-param
// ============================================================================

TEST(QASM3ParserTest, GateH) {
    auto qc = parse3("qubit[1] q;\nh q[0];\n");
    expect_single(qc, GT::H, {0});
}
TEST(QASM3ParserTest, GateX) {
    auto qc = parse3("qubit[1] q;\nx q[0];\n");
    expect_single(qc, GT::X, {0});
}
TEST(QASM3ParserTest, GateY) {
    auto qc = parse3("qubit[1] q;\ny q[0];\n");
    expect_single(qc, GT::Y, {0});
}
TEST(QASM3ParserTest, GateZ) {
    auto qc = parse3("qubit[1] q;\nz q[0];\n");
    expect_single(qc, GT::Z, {0});
}
TEST(QASM3ParserTest, GateS) {
    auto qc = parse3("qubit[1] q;\ns q[0];\n");
    expect_single(qc, GT::S, {0});
}
TEST(QASM3ParserTest, GateSdg) {
    auto qc = parse3("qubit[1] q;\nsdg q[0];\n");
    expect_single(qc, GT::SDG, {0});
}
TEST(QASM3ParserTest, GateT) {
    auto qc = parse3("qubit[1] q;\nt q[0];\n");
    expect_single(qc, GT::T, {0});
}
TEST(QASM3ParserTest, GateTdg) {
    auto qc = parse3("qubit[1] q;\ntdg q[0];\n");
    expect_single(qc, GT::TDG, {0});
}
TEST(QASM3ParserTest, GateSX) {
    auto qc = parse3("qubit[1] q;\nsx q[0];\n");
    expect_single(qc, GT::SX, {0});
}
TEST(QASM3ParserTest, GateSXdg) {
    auto qc = parse3("qubit[1] q;\nsxdg q[0];\n");
    expect_single(qc, GT::SXDG, {0});
}
TEST(QASM3ParserTest, GateIdIsDropped) {
    auto qc = parse3("qubit[1] q;\nid q[0];\nh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}

// ============================================================================
// 4. Standard gates : 1-qubit parameterised
// ============================================================================

TEST(QASM3ParserTest, GateRX) {
    auto qc = parse3("qubit[1] q;\nrx(0.5) q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::RX);
    ASSERT_EQ(qc.instructions[0].params.size(), 1u);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.5);
}
TEST(QASM3ParserTest, GateRY) {
    auto qc = parse3("qubit[1] q;\nry(0.25) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::RY);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.25);
}
TEST(QASM3ParserTest, GateRZ) {
    auto qc = parse3("qubit[1] q;\nrz(-0.7) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::RZ);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], -0.7);
}
TEST(QASM3ParserTest, GateP) {
    auto qc = parse3("qubit[1] q;\np(1.0) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::P);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 1.0);
}
TEST(QASM3ParserTest, GatePhaseAlias) {
    auto qc = parse3("qubit[1] q;\nphase(0.4) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::P);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.4);
}
TEST(QASM3ParserTest, GateU1) {
    auto qc = parse3("qubit[1] q;\nu1(0.3) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::U1);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.3);
}
TEST(QASM3ParserTest, GateU2) {
    auto qc = parse3("qubit[1] q;\nu2(0.1, 0.2) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::U2);
    ASSERT_EQ(qc.instructions[0].params.size(), 2u);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.1);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[1], 0.2);
}
TEST(QASM3ParserTest, GateU3) {
    auto qc = parse3("qubit[1] q;\nu3(0.1, 0.2, 0.3) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::U3);
    ASSERT_EQ(qc.instructions[0].params.size(), 3u);
}
TEST(QASM3ParserTest, GateULowercase) {
    auto qc = parse3("qubit[1] q;\nu(0.1, 0.2, 0.3) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::U);
    EXPECT_EQ(qc.instructions[0].params.size(), 3u);
}
TEST(QASM3ParserTest, GateUUppercase) {
    auto qc = parse3("qubit[1] q;\nU(0.1, 0.2, 0.3) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::U);
    EXPECT_EQ(qc.instructions[0].params.size(), 3u);
}

// ============================================================================
// 5. Standard gates : 2-qubit no-param
// ============================================================================

TEST(QASM3ParserTest, GateCX) {
    auto qc = parse3("qubit[2] q;\ncx q[0], q[1];\n");
    expect_single(qc, GT::CX, {0, 1});
}
TEST(QASM3ParserTest, GateCXUppercase) {
    auto qc = parse3("qubit[2] q;\nCX q[0], q[1];\n");
    expect_single(qc, GT::CX, {0, 1});
}
TEST(QASM3ParserTest, GateCY) {
    auto qc = parse3("qubit[2] q;\ncy q[0], q[1];\n");
    expect_single(qc, GT::CY, {0, 1});
}
TEST(QASM3ParserTest, GateCZ) {
    auto qc = parse3("qubit[2] q;\ncz q[0], q[1];\n");
    expect_single(qc, GT::CZ, {0, 1});
}
TEST(QASM3ParserTest, GateCH) {
    auto qc = parse3("qubit[2] q;\nch q[0], q[1];\n");
    expect_single(qc, GT::CH, {0, 1});
}
TEST(QASM3ParserTest, GateSwap) {
    auto qc = parse3("qubit[2] q;\nswap q[0], q[1];\n");
    expect_single(qc, GT::SWAP, {0, 1});
}
TEST(QASM3ParserTest, GateISwap) {
    auto qc = parse3("qubit[2] q;\niswap q[0], q[1];\n");
    expect_single(qc, GT::ISWAP, {0, 1});
}
TEST(QASM3ParserTest, GateECR) {
    auto qc = parse3("qubit[2] q;\necr q[0], q[1];\n");
    expect_single(qc, GT::ECR, {0, 1});
}

// ============================================================================
// 6. Standard gates : 2-qubit parameterised
// ============================================================================

TEST(QASM3ParserTest, GateCRX) {
    auto qc = parse3("qubit[2] q;\ncrx(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CRX);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.5);
}
TEST(QASM3ParserTest, GateCRY) {
    auto qc = parse3("qubit[2] q;\ncry(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CRY);
}
TEST(QASM3ParserTest, GateCRZ) {
    auto qc = parse3("qubit[2] q;\ncrz(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CRZ);
}
TEST(QASM3ParserTest, GateCP) {
    auto qc = parse3("qubit[2] q;\ncp(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CP);
}
TEST(QASM3ParserTest, GateCPhaseAlias) {
    auto qc = parse3("qubit[2] q;\ncphase(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CP);
}
TEST(QASM3ParserTest, GateRXX) {
    auto qc = parse3("qubit[2] q;\nrxx(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::RXX);
}
TEST(QASM3ParserTest, GateRYY) {
    auto qc = parse3("qubit[2] q;\nryy(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::RYY);
}
TEST(QASM3ParserTest, GateRZZ) {
    auto qc = parse3("qubit[2] q;\nrzz(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::RZZ);
}
TEST(QASM3ParserTest, GateRZX) {
    auto qc = parse3("qubit[2] q;\nrzx(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::RZX);
}

// ============================================================================
// 7. Standard gates : 3-qubit
// ============================================================================

TEST(QASM3ParserTest, GateCCX) {
    auto qc = parse3("qubit[3] q;\nccx q[0], q[1], q[2];\n");
    expect_single(qc, GT::CCX, {0, 1, 2});
}
TEST(QASM3ParserTest, GateToffoliAlias) {
    auto qc = parse3("qubit[3] q;\ntoffoli q[0], q[1], q[2];\n");
    expect_single(qc, GT::CCX, {0, 1, 2});
}
TEST(QASM3ParserTest, GateCCZ) {
    auto qc = parse3("qubit[3] q;\nccz q[0], q[1], q[2];\n");
    expect_single(qc, GT::CCZ, {0, 1, 2});
}
TEST(QASM3ParserTest, GateCSwap) {
    auto qc = parse3("qubit[3] q;\ncswap q[0], q[1], q[2];\n");
    expect_single(qc, GT::CSWAP, {0, 1, 2});
}
TEST(QASM3ParserTest, GateFredkinAlias) {
    auto qc = parse3("qubit[3] q;\nfredkin q[0], q[1], q[2];\n");
    expect_single(qc, GT::CSWAP, {0, 1, 2});
}
TEST(QASM3ParserTest, GateRCCX) {
    auto qc = parse3("qubit[3] q;\nrccx q[0], q[1], q[2];\n");
    expect_single(qc, GT::RCCX, {0, 1, 2});
}

// ============================================================================
// 8. Measurements, reset, barrier
// ============================================================================

TEST(QASM3ParserTest, MeasureAssignForm) {
    auto qc = parse3("qubit[1] q;\nbit[1] c;\nc[0] = measure q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::MEASURE);
    EXPECT_EQ(qc.instructions[0].qubits[0], 0);
    EXPECT_EQ(qc.instructions[0].clbits[0], 0);
}
TEST(QASM3ParserTest, MeasureArrowForm) {
    auto qc = parse3("qubit[1] q;\nbit[1] c;\nmeasure q[0] -> c[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::MEASURE);
}
TEST(QASM3ParserTest, MeasureMissingTargetThrows) {
    EXPECT_THROW(parse3("qubit[1] q;\nmeasure q[0];\n"), std::runtime_error);
}
TEST(QASM3ParserTest, Reset) {
    auto qc = parse3("qubit[1] q;\nreset q[0];\n");
    expect_single(qc, GT::RESET, {0});
}
TEST(QASM3ParserTest, BarrierExplicitQubits) {
    auto qc = parse3("qubit[3] q;\nbarrier q[0], q[2];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::BARRIER);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{0, 2}));
}
TEST(QASM3ParserTest, BarrierAllQubits) {
    auto qc = parse3("qubit[3] q;\nbarrier;\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::BARRIER);
    EXPECT_EQ(qc.instructions[0].qubits.size(), 3u);
}

// ============================================================================
// 9. Modifier resolution : ctrl @ fast paths
// ============================================================================

TEST(QASM3ParserTest, CtrlX_to_CX) {
    auto qc = parse3("qubit[2] q;\nctrl @ x q[0], q[1];\n");
    expect_single(qc, GT::CX, {0, 1});
}
TEST(QASM3ParserTest, CtrlY_to_CY) {
    auto qc = parse3("qubit[2] q;\nctrl @ y q[0], q[1];\n");
    expect_single(qc, GT::CY, {0, 1});
}
TEST(QASM3ParserTest, CtrlZ_to_CZ) {
    auto qc = parse3("qubit[2] q;\nctrl @ z q[0], q[1];\n");
    expect_single(qc, GT::CZ, {0, 1});
}
TEST(QASM3ParserTest, CtrlH_to_CH) {
    auto qc = parse3("qubit[2] q;\nctrl @ h q[0], q[1];\n");
    expect_single(qc, GT::CH, {0, 1});
}
TEST(QASM3ParserTest, CtrlSwap_to_CSwap) {
    auto qc = parse3("qubit[3] q;\nctrl @ swap q[0], q[1], q[2];\n");
    expect_single(qc, GT::CSWAP, {0, 1, 2});
}
TEST(QASM3ParserTest, DoubleCtrlX_to_CCX) {
    auto qc = parse3("qubit[3] q;\nctrl @ ctrl @ x q[0], q[1], q[2];\n");
    expect_single(qc, GT::CCX, {0, 1, 2});
}
TEST(QASM3ParserTest, DoubleCtrlZ_to_CCZ) {
    auto qc = parse3("qubit[3] q;\nctrl @ ctrl @ z q[0], q[1], q[2];\n");
    expect_single(qc, GT::CCZ, {0, 1, 2});
}
TEST(QASM3ParserTest, CtrlRX_to_CRX) {
    auto qc = parse3("qubit[2] q;\nctrl @ rx(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CRX);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.5);
}
TEST(QASM3ParserTest, CtrlRY_to_CRY) {
    auto qc = parse3("qubit[2] q;\nctrl @ ry(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CRY);
}
TEST(QASM3ParserTest, CtrlRZ_to_CRZ) {
    auto qc = parse3("qubit[2] q;\nctrl @ rz(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CRZ);
}
TEST(QASM3ParserTest, CtrlP_to_CP) {
    auto qc = parse3("qubit[2] q;\nctrl @ p(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CP);
}
TEST(QASM3ParserTest, CtrlPhase_to_CP) {
    auto qc = parse3("qubit[2] q;\nctrl @ phase(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CP);
}

// ============================================================================
// 10. Modifier resolution : inv @
// ============================================================================

TEST(QASM3ParserTest, InvHIsSelfInverse) {
    auto qc = parse3("qubit[1] q;\ninv @ h q[0];\n");
    expect_single(qc, GT::H, {0});
}
TEST(QASM3ParserTest, InvXIsSelfInverse) {
    auto qc = parse3("qubit[1] q;\ninv @ x q[0];\n");
    expect_single(qc, GT::X, {0});
}
TEST(QASM3ParserTest, InvCX_IsSelfInverse) {
    auto qc = parse3("qubit[2] q;\ninv @ cx q[0], q[1];\n");
    expect_single(qc, GT::CX, {0, 1});
}
TEST(QASM3ParserTest, InvS_To_SDG) {
    auto qc = parse3("qubit[1] q;\ninv @ s q[0];\n");
    expect_single(qc, GT::SDG, {0});
}
TEST(QASM3ParserTest, InvSDG_To_S) {
    auto qc = parse3("qubit[1] q;\ninv @ sdg q[0];\n");
    expect_single(qc, GT::S, {0});
}
TEST(QASM3ParserTest, InvT_To_TDG) {
    auto qc = parse3("qubit[1] q;\ninv @ t q[0];\n");
    expect_single(qc, GT::TDG, {0});
}
TEST(QASM3ParserTest, InvTDG_To_T) {
    auto qc = parse3("qubit[1] q;\ninv @ tdg q[0];\n");
    expect_single(qc, GT::T, {0});
}
TEST(QASM3ParserTest, InvSX_To_SXDG) {
    auto qc = parse3("qubit[1] q;\ninv @ sx q[0];\n");
    expect_single(qc, GT::SXDG, {0});
}
TEST(QASM3ParserTest, InvSXDG_To_SX) {
    auto qc = parse3("qubit[1] q;\ninv @ sxdg q[0];\n");
    expect_single(qc, GT::SX, {0});
}
TEST(QASM3ParserTest, InvRXNegatesAngle) {
    auto qc = parse3("qubit[1] q;\ninv @ rx(0.5) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::RX);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], -0.5);
}
TEST(QASM3ParserTest, InvRYNegatesAngle) {
    auto qc = parse3("qubit[1] q;\ninv @ ry(0.3) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], -0.3);
}
TEST(QASM3ParserTest, InvRZNegatesAngle) {
    auto qc = parse3("qubit[1] q;\ninv @ rz(0.7) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], -0.7);
}
TEST(QASM3ParserTest, InvDoubleNegationIsIdentity) {
    auto qc = parse3("qubit[1] q;\ninv @ inv @ rx(0.5) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.5);
}

// ============================================================================
// 11. Modifier resolution : pow(n) @
// ============================================================================

TEST(QASM3ParserTest, PowZeroIsDropped) {
    auto qc = parse3("qubit[1] q;\npow(0) @ x q[0];\nh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}
TEST(QASM3ParserTest, PowEvenOnSelfInverseIsDropped) {
    auto qc = parse3("qubit[1] q;\npow(2) @ x q[0];\nh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}
TEST(QASM3ParserTest, PowOddOnSelfInverseKeeps) {
    auto qc = parse3("qubit[1] q;\npow(3) @ x q[0];\n");
    expect_single(qc, GT::X, {0});
}
TEST(QASM3ParserTest, PowOnRotationScalesAngle) {
    auto qc = parse3("qubit[1] q;\npow(3) @ rx(0.2) q[0];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::RX);
    EXPECT_NEAR(qc.instructions[0].params[0], 0.6, 1e-12);
}
TEST(QASM3ParserTest, PowOnRotationWithNegativeExponent) {
    auto qc = parse3("qubit[1] q;\npow(-2) @ rx(0.2) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], -0.4, 1e-12);
}
TEST(QASM3ParserTest, PowEvenOnSelfInverseTwoQubit) {
    auto qc = parse3("qubit[2] q;\npow(4) @ cx q[0], q[1];\nh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}

// ============================================================================
// 12. Modifier resolution : chained
// ============================================================================

TEST(QASM3ParserTest, InvPow3RxFolded) {
    auto qc = parse3("qubit[1] q;\ninv @ pow(3) @ rx(0.2) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], -0.6, 1e-12);
}
TEST(QASM3ParserTest, CtrlInvRxNegated) {
    auto qc = parse3("qubit[2] q;\nctrl @ inv @ rx(0.5) q[0], q[1];\n");
    EXPECT_EQ(qc.instructions[0].type, GT::CRX);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], -0.5);
}
TEST(QASM3ParserTest, CtrlPow2X_DroppedAsIdentity) {
    // ctrl @ pow(2) @ x : pow(2) folds to identity, then ctrl @ id is also identity
    auto qc = parse3("qubit[2] q;\nctrl @ pow(2) @ x q[0], q[1];\nh q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}

// ============================================================================
// 13. Matrix fallback path
// ============================================================================

TEST(QASM3ParserTest, MatrixFallbackPow2OnS) {
    // pow(2) @ s has no named target so it becomes UNITARY (= Z).
    auto qc = parse3("qubit[1] q;\npow(2) @ s q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::UNITARY);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{0}));
    EXPECT_EQ(qc.instructions[0].matrix.size(), 4u);
}
TEST(QASM3ParserTest, MatrixFallbackInvIswapThrows) {
    // iswap is 2-qubit and the matrix fallback only handles 1-qubit bases.
    EXPECT_THROW(parse3("qubit[2] q;\ninv @ iswap q[0], q[1];\n"),
                 std::runtime_error);
}
TEST(QASM3ParserTest, MatrixFallbackPow3OnT) {
    auto qc = parse3("qubit[1] q;\npow(3) @ t q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::UNITARY);
}
TEST(QASM3ParserTest, MatrixFallbackPreservesSize2x2) {
    // After fallback, the matrix is row-major 2x2 (1-qubit unitary).
    auto qc = parse3("qubit[1] q;\npow(2) @ s q[0];\n");
    EXPECT_EQ(qc.instructions[0].matrix.size(), 4u);
}
TEST(QASM3ParserTest, MatrixFallbackCtrlExtendsTo4x4) {
    auto qc = parse3("qubit[2] q;\nctrl @ pow(2) @ s q[0], q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::UNITARY);
    EXPECT_EQ(qc.instructions[0].matrix.size(), 16u);  // 4 x 4
}

// ============================================================================
// 14. Parameter expressions
// ============================================================================

TEST(QASM3ParserTest, ParamExprPiLiteral) {
    auto qc = parse3("qubit[1] q;\nrx(pi) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], PI, 1e-12);
}
TEST(QASM3ParserTest, ParamExprTauLiteral) {
    auto qc = parse3("qubit[1] q;\nrx(tau) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 2.0 * PI, 1e-12);
}
TEST(QASM3ParserTest, ParamExprEulerLiteral) {
    auto qc = parse3("qubit[1] q;\nrx(euler) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], E, 1e-12);
}
TEST(QASM3ParserTest, ParamExprPiOverTwo) {
    auto qc = parse3("qubit[1] q;\nrx(pi/2) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], PI / 2.0, 1e-12);
}
TEST(QASM3ParserTest, ParamExprNegativePi) {
    auto qc = parse3("qubit[1] q;\nrx(-pi) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], -PI, 1e-12);
}
TEST(QASM3ParserTest, ParamExprPrecedenceMulOverAdd) {
    auto qc = parse3("qubit[1] q;\nrx(1.0 + 2.0 * 3.0) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 7.0, 1e-12);
}
TEST(QASM3ParserTest, ParamExprParenthesesOverridePrecedence) {
    auto qc = parse3("qubit[1] q;\nrx((1.0 + 2.0) * 3.0) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 9.0, 1e-12);
}
TEST(QASM3ParserTest, ParamExprSubtraction) {
    auto qc = parse3("qubit[1] q;\nrx(5.0 - 2.0) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 3.0, 1e-12);
}
TEST(QASM3ParserTest, ParamExprDivision) {
    auto qc = parse3("qubit[1] q;\nrx(10.0 / 4.0) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 2.5, 1e-12);
}
TEST(QASM3ParserTest, ParamExprFloatExponent) {
    auto qc = parse3("qubit[1] q;\nrx(1.5e-2) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 0.015, 1e-12);
}
TEST(QASM3ParserTest, ParamExprInteger) {
    auto qc = parse3("qubit[1] q;\nrx(3) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 3.0);
}
TEST(QASM3ParserTest, ParamExprUnaryPlus) {
    auto qc = parse3("qubit[1] q;\nrx(+0.5) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.5);
}
TEST(QASM3ParserTest, ParamExprDoubleNegation) {
    auto qc = parse3("qubit[1] q;\nrx(- -0.5) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.5);
}
TEST(QASM3ParserTest, ParamExprComplexNested) {
    auto qc = parse3("qubit[1] q;\nrx(pi/4 + 1.0) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], PI / 4.0 + 1.0, 1e-12);
}

// ============================================================================
// 15. Custom gate definitions
// ============================================================================

TEST(QASM3ParserTest, CustomGateNoParams) {
    auto qc = parse3(
        "gate myh a { h a; }\n"
        "qubit[1] q;\n"
        "myh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}
TEST(QASM3ParserTest, CustomGateWithParams) {
    auto qc = parse3(
        "gate myrx(t) a { rx(t) a; }\n"
        "qubit[1] q;\n"
        "myrx(0.5) q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::RX);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.5);
}
TEST(QASM3ParserTest, CustomGateMultipleQubits) {
    auto qc = parse3(
        "gate bell a, b { h a; cx a, b; }\n"
        "qubit[2] q;\n"
        "bell q[0], q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[1].type, GT::CX);
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{0, 1}));
}
TEST(QASM3ParserTest, CustomGateParamArithmetic) {
    auto qc = parse3(
        "gate scaled(t) a { rx(t * 2.0) a; }\n"
        "qubit[1] q;\n"
        "scaled(0.5) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 1.0, 1e-12);
}
TEST(QASM3ParserTest, CustomGateNested) {
    auto qc = parse3(
        "gate inner a { h a; }\n"
        "gate outer a { inner a; x a; }\n"
        "qubit[1] q;\n"
        "outer q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[1].type, GT::X);
}
TEST(QASM3ParserTest, CustomGateParamCountMismatchThrows) {
    EXPECT_THROW(parse3(
        "gate myrx(t) a { rx(t) a; }\n"
        "qubit[1] q;\n"
        "myrx(0.5, 0.7) q[0];\n"), std::runtime_error);
}
TEST(QASM3ParserTest, CustomGateQubitCountMismatchThrows) {
    EXPECT_THROW(parse3(
        "gate bell a, b { h a; cx a, b; }\n"
        "qubit[2] q;\n"
        "bell q[0];\n"), std::runtime_error);
}
TEST(QASM3ParserTest, CustomGateModifierThrows) {
    EXPECT_THROW(parse3(
        "gate myh a { h a; }\n"
        "qubit[1] q;\n"
        "inv @ myh q[0];\n"), std::runtime_error);
}

// ============================================================================
// 16. Classical conditioning
// ============================================================================

TEST(QASM3ParserTest, IfSingleStatement) {
    auto qc = parse3(
        "qubit[1] q;\nbit[1] c;\nif (c[0] == 1) x q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::X);
    EXPECT_EQ(qc.instructions[0].condition_clbit, 0);
    EXPECT_EQ(qc.instructions[0].condition_value, 1);
}
TEST(QASM3ParserTest, IfBlock) {
    auto qc = parse3(
        "qubit[2] q;\nbit[1] c;\n"
        "if (c[0] == 1) { x q[0]; y q[1]; }\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].condition_clbit, 0);
    EXPECT_EQ(qc.instructions[0].condition_value, 1);
    EXPECT_EQ(qc.instructions[1].condition_clbit, 0);
    EXPECT_EQ(qc.instructions[1].condition_value, 1);
}
TEST(QASM3ParserTest, IfElseFlipsCondition) {
    auto qc = parse3(
        "qubit[2] q;\nbit[1] c;\n"
        "if (c[0] == 1) x q[0]; else y q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::X);
    EXPECT_EQ(qc.instructions[0].condition_value, 1);
    EXPECT_EQ(qc.instructions[1].type, GT::Y);
    EXPECT_EQ(qc.instructions[1].condition_value, 0);
}
TEST(QASM3ParserTest, IfSingleBitRegisterForm) {
    // bit c; (size 1) supports `if (c == V)` form
    auto qc = parse3(
        "qubit[1] q;\nbit c;\nif (c == 1) x q[0];\n");
    EXPECT_EQ(qc.instructions[0].condition_clbit, 0);
    EXPECT_EQ(qc.instructions[0].condition_value, 1);
}
TEST(QASM3ParserTest, IfMultiBitRegisterThrows) {
    EXPECT_THROW(parse3(
        "qubit[1] q;\nbit[2] c;\nif (c == 3) x q[0];\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, IfDoesNotLeakConditionAfterBlock) {
    auto qc = parse3(
        "qubit[1] q;\nbit[1] c;\n"
        "if (c[0] == 1) x q[0];\n"
        "y q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].condition_clbit, 0);
    EXPECT_EQ(qc.instructions[1].condition_clbit, -1);
}

// ============================================================================
// 17. Symbolic parameters and bind_parameters()
// ============================================================================

TEST(QASM3ParserTest, InputParameterRegistered) {
    auto qc = parse3_raw(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] theta;\n"
        "qubit[1] q;\n"
        "rx(theta) q[0];\n");
    ASSERT_EQ(qc.parameter_names.size(), 1u);
    EXPECT_EQ(qc.parameter_names[0], "theta");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_TRUE(qc.instructions[0].params.empty());
    ASSERT_EQ(qc.instructions[0].param_exprs.size(), 1u);
    EXPECT_EQ(qc.instructions[0].param_exprs[0].kind, ParamExpr::Kind::Name);
}
TEST(QASM3ParserTest, SymbolicAngleEvaluatesAfterBind) {
    auto qc = parse3_raw(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] theta;\n"
        "qubit[1] q;\n"
        "rx(theta) q[0];\n");
    qc.bind_parameters({{"theta", 0.42}});
    ASSERT_EQ(qc.instructions[0].params.size(), 1u);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.42);
    EXPECT_TRUE(qc.instructions[0].param_exprs.empty());
}
TEST(QASM3ParserTest, SymbolicExpressionEvaluatesAfterBind) {
    auto qc = parse3_raw(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] theta;\n"
        "qubit[1] q;\n"
        "ry(theta + pi/4) q[0];\n");
    qc.bind_parameters({{"theta", 0.3}});
    EXPECT_NEAR(qc.instructions[0].params[0], 0.3 + PI / 4.0, 1e-12);
}
TEST(QASM3ParserTest, BindMissingParameterThrows) {
    auto qc = parse3_raw(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] theta;\n"
        "qubit[1] q;\n"
        "rx(theta) q[0];\n");
    EXPECT_THROW(qc.bind_parameters({{"phi", 0.1}}), std::runtime_error);
}
TEST(QASM3ParserTest, BindSkipsNumericInstructions) {
    auto qc = parse3("qubit[1] q;\nh q[0];\n");
    qc.bind_parameters({{"theta", 0.42}});
    EXPECT_TRUE(qc.instructions[0].params.empty());
    EXPECT_TRUE(qc.instructions[0].param_exprs.empty());
}
TEST(QASM3ParserTest, BindMergesIntoParameterBindings) {
    auto qc = parse3_raw(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] theta;\n"
        "qubit[1] q;\n"
        "rx(theta) q[0];\n");
    qc.bind_parameters({{"theta", 0.5}});
    auto it = qc.parameter_bindings.find("theta");
    ASSERT_NE(it, qc.parameter_bindings.end());
    EXPECT_DOUBLE_EQ(it->second, 0.5);
}

// ============================================================================
// 18. ParamExpr factories and eval
// ============================================================================

TEST(ParamExprTest, LiteralFactory) {
    auto e = ParamExpr::make_literal(3.14);
    EXPECT_EQ(e.kind, ParamExpr::Kind::Literal);
    EXPECT_DOUBLE_EQ(e.literal, 3.14);
    EXPECT_DOUBLE_EQ(e.eval({}), 3.14);
}
TEST(ParamExprTest, NameFactory) {
    auto e = ParamExpr::make_name("alpha");
    EXPECT_EQ(e.kind, ParamExpr::Kind::Name);
    EXPECT_EQ(e.name, "alpha");
    EXPECT_DOUBLE_EQ(e.eval({{"alpha", 1.5}}), 1.5);
}
TEST(ParamExprTest, BinaryAdd) {
    auto e = ParamExpr::make_binary('+',
        ParamExpr::make_literal(2.0), ParamExpr::make_literal(3.0));
    EXPECT_DOUBLE_EQ(e.eval({}), 5.0);
}
TEST(ParamExprTest, BinarySub) {
    auto e = ParamExpr::make_binary('-',
        ParamExpr::make_literal(5.0), ParamExpr::make_literal(3.0));
    EXPECT_DOUBLE_EQ(e.eval({}), 2.0);
}
TEST(ParamExprTest, BinaryMul) {
    auto e = ParamExpr::make_binary('*',
        ParamExpr::make_literal(4.0), ParamExpr::make_literal(2.5));
    EXPECT_DOUBLE_EQ(e.eval({}), 10.0);
}
TEST(ParamExprTest, BinaryDiv) {
    auto e = ParamExpr::make_binary('/',
        ParamExpr::make_literal(10.0), ParamExpr::make_literal(4.0));
    EXPECT_DOUBLE_EQ(e.eval({}), 2.5);
}
TEST(ParamExprTest, NestedExpression) {
    // (a + b) * 2
    auto e = ParamExpr::make_binary('*',
        ParamExpr::make_binary('+',
            ParamExpr::make_name("a"),
            ParamExpr::make_name("b")),
        ParamExpr::make_literal(2.0));
    EXPECT_DOUBLE_EQ(e.eval({{"a", 1.0}, {"b", 2.0}}), 6.0);
}
TEST(ParamExprTest, MissingBindingThrows) {
    auto e = ParamExpr::make_name("missing");
    EXPECT_THROW(e.eval({}), std::runtime_error);
}
TEST(ParamExprTest, DivisionByZeroThrows) {
    auto e = ParamExpr::make_binary('/',
        ParamExpr::make_literal(1.0), ParamExpr::make_literal(0.0));
    EXPECT_THROW(e.eval({}), std::runtime_error);
}
TEST(ParamExprTest, DeepCopyIndependence) {
    auto orig = ParamExpr::make_binary('+',
        ParamExpr::make_name("x"), ParamExpr::make_literal(1.0));
    auto copy = orig;  // deep copy via custom ctor
    // Mutate the copy.
    copy.lhs = std::make_unique<ParamExpr>(ParamExpr::make_literal(99.0));
    EXPECT_EQ(orig.lhs->kind, ParamExpr::Kind::Name);
    EXPECT_EQ(orig.lhs->name, "x");
}
TEST(ParamExprTest, CopyAssignmentDeepCopies) {
    auto orig = ParamExpr::make_binary('*',
        ParamExpr::make_name("y"), ParamExpr::make_literal(2.0));
    ParamExpr other = ParamExpr::make_literal(0.0);
    other = orig;
    other.rhs = std::make_unique<ParamExpr>(ParamExpr::make_literal(99.0));
    EXPECT_DOUBLE_EQ(orig.rhs->literal, 2.0);
}
TEST(ParamExprTest, MoveConstructsCorrectly) {
    auto orig = ParamExpr::make_literal(7.0);
    ParamExpr moved = std::move(orig);
    EXPECT_DOUBLE_EQ(moved.literal, 7.0);
}

// ============================================================================
// 19. Peephole optimisation
// ============================================================================

TEST(QASM3ParserTest, PeepholeCancelsHH) {
    auto qc = parse3("qubit[1] q;\nh q[0];\nh q[0];\n");
    EXPECT_EQ(qc.instructions.size(), 0u);
}
TEST(QASM3ParserTest, PeepholeCancelsXX) {
    auto qc = parse3("qubit[1] q;\nx q[0];\nx q[0];\n");
    EXPECT_EQ(qc.instructions.size(), 0u);
}
TEST(QASM3ParserTest, PeepholeCancelsCXCX) {
    auto qc = parse3("qubit[2] q;\ncx q[0], q[1];\ncx q[0], q[1];\n");
    EXPECT_EQ(qc.instructions.size(), 0u);
}
TEST(QASM3ParserTest, PeepholeCancelsCCX) {
    auto qc = parse3(
        "qubit[3] q;\n"
        "ccx q[0], q[1], q[2];\n"
        "ccx q[0], q[1], q[2];\n");
    EXPECT_EQ(qc.instructions.size(), 0u);
}
TEST(QASM3ParserTest, PeepholeKeepsTripleHAsSingle) {
    // h; h; h cancels first two; third stays.
    auto qc = parse3("qubit[1] q;\nh q[0];\nh q[0];\nh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}
TEST(QASM3ParserTest, PeepholeRespectsQubitOrder) {
    // cx q[0], q[1] then cx q[1], q[0] is two different gates; no cancel.
    auto qc = parse3(
        "qubit[2] q;\ncx q[0], q[1];\ncx q[1], q[0];\n");
    EXPECT_EQ(qc.instructions.size(), 2u);
}
TEST(QASM3ParserTest, PeepholeDoesNotCancelDifferentTypes) {
    auto qc = parse3("qubit[1] q;\nh q[0];\nx q[0];\n");
    EXPECT_EQ(qc.instructions.size(), 2u);
}
TEST(QASM3ParserTest, PeepholeDoesNotCancelAcrossDifferentQubits) {
    auto qc = parse3("qubit[2] q;\nh q[0];\nh q[1];\n");
    EXPECT_EQ(qc.instructions.size(), 2u);
}
TEST(QASM3ParserTest, PeepholeSkipsConditionedGates) {
    auto qc = parse3(
        "qubit[1] q;\nbit[1] c;\n"
        "if (c[0] == 1) x q[0];\n"
        "if (c[0] == 1) x q[0];\n");
    EXPECT_EQ(qc.instructions.size(), 2u);
}
TEST(QASM3ParserTest, PeepholeDoesNotCancelParametricGates) {
    auto qc = parse3(
        "qubit[1] q;\nrx(0.5) q[0];\nrx(-0.5) q[0];\n");
    EXPECT_EQ(qc.instructions.size(), 2u);
}
TEST(QASM3ParserTest, PeepholeCancelsAcrossBarrierFree) {
    // No barrier means h; h cancels directly.
    auto qc = parse3("qubit[2] q;\nh q[0];\nx q[1];\nh q[0];\n");
    // Latest on q[0] is the second h; it pairs with the first h since
    // x q[1] does not touch q[0].
    EXPECT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::X);
}

// ============================================================================
// 20. Multi-register offset resolution
// ============================================================================

TEST(QASM3ParserTest, MultiRegisterQubitOffsets) {
    auto qc = parse3(
        "qubit[2] a;\nqubit[3] b;\n"
        "h a[0];\nh b[0];\nh b[2];\n");
    ASSERT_EQ(qc.instructions.size(), 3u);
    EXPECT_EQ(qc.instructions[0].qubits[0], 0);
    EXPECT_EQ(qc.instructions[1].qubits[0], 2);
    EXPECT_EQ(qc.instructions[2].qubits[0], 4);
}
TEST(QASM3ParserTest, MultiRegisterClassicalOffsets) {
    auto qc = parse3(
        "qubit[1] q;\nbit[2] x;\nbit[2] y;\n"
        "x[1] = measure q[0];\n"
        "y[0] = measure q[0];\n");
    EXPECT_EQ(qc.instructions[0].clbits[0], 1);
    EXPECT_EQ(qc.instructions[1].clbits[0], 2);
}
TEST(QASM3ParserTest, UnknownQubitRegisterThrows) {
    EXPECT_THROW(parse3("qubit[1] q;\nh other[0];\n"), std::runtime_error);
}
TEST(QASM3ParserTest, UnknownClassicalRegisterThrows) {
    EXPECT_THROW(parse3(
        "qubit[1] q;\nbit[1] c;\nother[0] = measure q[0];\n"),
        std::runtime_error);
}

// ============================================================================
// 21. Round-trip (to_qasm3 -> from_qasm3)
// ============================================================================

TEST(QASM3ParserTest, RoundTripBellState) {
    QuantumCircuit qc(2, 2, "bell");
    qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    auto qasm = qc.to_qasm3();
    auto qc2 = QuantumCircuit::from_qasm3(qasm);
    EXPECT_EQ(qc2.n_qubits, 2);
    EXPECT_EQ(qc2.n_clbits, 2);
    ASSERT_EQ(qc2.instructions.size(), 4u);
    EXPECT_EQ(qc2.instructions[0].type, GT::H);
    EXPECT_EQ(qc2.instructions[1].type, GT::CX);
    EXPECT_EQ(qc2.instructions[2].type, GT::MEASURE);
    EXPECT_EQ(qc2.instructions[3].type, GT::MEASURE);
}
TEST(QASM3ParserTest, RoundTripGHZ) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);
    auto qc2 = QuantumCircuit::from_qasm3(qc.to_qasm3());
    EXPECT_EQ(qc2.n_qubits, 3);
    ASSERT_EQ(qc2.instructions.size(), 3u);
    EXPECT_EQ(qc2.instructions[0].type, GT::H);
    EXPECT_EQ(qc2.instructions[1].type, GT::CX);
    EXPECT_EQ(qc2.instructions[2].type, GT::CX);
}
TEST(QASM3ParserTest, RoundTripRotationCircuit) {
    QuantumCircuit qc(2);
    qc.rx(0.1, 0).ry(0.2, 0).rz(0.3, 1).cx(0, 1);
    auto qc2 = QuantumCircuit::from_qasm3(qc.to_qasm3());
    ASSERT_EQ(qc2.instructions.size(), 4u);
    EXPECT_NEAR(qc2.instructions[0].params[0], 0.1, 1e-12);
    EXPECT_NEAR(qc2.instructions[1].params[0], 0.2, 1e-12);
    EXPECT_NEAR(qc2.instructions[2].params[0], 0.3, 1e-12);
}
TEST(QASM3ParserTest, RoundTripBarrier) {
    QuantumCircuit qc(2);
    qc.h(0).barrier({0, 1}).cx(0, 1);
    auto qc2 = QuantumCircuit::from_qasm3(qc.to_qasm3());
    ASSERT_EQ(qc2.instructions.size(), 3u);
    EXPECT_EQ(qc2.instructions[1].type, GT::BARRIER);
}
TEST(QASM3ParserTest, RoundTripReset) {
    QuantumCircuit qc(1);
    qc.h(0).reset(0);
    auto qc2 = QuantumCircuit::from_qasm3(qc.to_qasm3());
    ASSERT_EQ(qc2.instructions.size(), 2u);
    EXPECT_EQ(qc2.instructions[1].type, GT::RESET);
}

// ============================================================================
// 22. Error reporting and unsupported constructs
// ============================================================================

TEST(QASM3ParserTest, UnknownGateThrows) {
    EXPECT_THROW(parse3("qubit[1] q;\nnotagate q[0];\n"),
                 std::runtime_error);
}
TEST(QASM3ParserTest, ForLoopThrows) {
    EXPECT_THROW(parse3(
        "qubit[2] q;\nfor int i in [0:2] { h q[i]; }\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, WhileLoopThrows) {
    EXPECT_THROW(parse3(
        "qubit[1] q;\nbit[1] c;\nwhile (c[0] == 0) { x q[0]; }\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, DefSubroutineThrows) {
    EXPECT_THROW(parse3(
        "qubit[1] q;\ndef foo(int x) { x q[0]; }\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, DelayThrows) {
    EXPECT_THROW(parse3(
        "qubit[1] q;\ndelay[100ns] q[0];\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, StretchThrows) {
    EXPECT_THROW(parse3(
        "qubit[1] q;\nstretch s = 10ns;\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, BoxThrows) {
    EXPECT_THROW(parse3(
        "qubit[1] q;\nbox { h q[0]; }\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, CalBlockThrows) {
    EXPECT_THROW(parse3(
        "qubit[1] q;\ncal { }\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, RequiredArgsMissingThrows) {
    EXPECT_THROW(parse3("qubit[1] q;\nrx q[0];\n"),
                 std::runtime_error);
}
TEST(QASM3ParserTest, ExtraQubitArgsThrow) {
    EXPECT_THROW(parse3("qubit[2] q;\nh q[0], q[1];\n"),
                 std::runtime_error);
}
TEST(QASM3ParserTest, UnexpectedCharacterThrows) {
    EXPECT_THROW(QuantumCircuit::from_qasm3(
        "OPENQASM 3.0;\nqubit[1] q;\n$ q[0];\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, MissingSemicolonThrows) {
    EXPECT_THROW(parse3("qubit[1] q;\nh q[0]\n"), std::runtime_error);
}

// ============================================================================
// 23. Lexer details
// ============================================================================

TEST(QASM3ParserTest, EmptyInputThrows) {
    EXPECT_THROW(QuantumCircuit::from_qasm3(""), std::runtime_error);
}
TEST(QASM3ParserTest, OnlyCommentsThrows) {
    EXPECT_THROW(QuantumCircuit::from_qasm3(
        "// just a comment\n/* and another */\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, OPENQASMOnlyHeaderThrows) {
    // No qubit register declared
    EXPECT_THROW(QuantumCircuit::from_qasm3(
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n"),
        std::runtime_error);
}
TEST(QASM3ParserTest, NumericIntegerLiteral) {
    auto qc = parse3("qubit[1] q;\nrx(7) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 7.0);
}
TEST(QASM3ParserTest, NumericFloatLiteral) {
    auto qc = parse3("qubit[1] q;\nrx(3.14) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 3.14, 1e-12);
}
TEST(QASM3ParserTest, NumericFloatWithLeadingDot) {
    auto qc = parse3("qubit[1] q;\nrx(.25) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 0.25);
}
TEST(QASM3ParserTest, NumericFloatWithTrailingDot) {
    auto qc = parse3("qubit[1] q;\nrx(5.) q[0];\n");
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 5.0);
}
TEST(QASM3ParserTest, NumericFloatExponentUpperE) {
    auto qc = parse3("qubit[1] q;\nrx(2E-1) q[0];\n");
    EXPECT_NEAR(qc.instructions[0].params[0], 0.2, 1e-12);
}
TEST(QASM3ParserTest, MultipleStatementsOneLine) {
    auto qc = parse3("qubit[2] q; h q[0]; cx q[0], q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
    EXPECT_EQ(qc.instructions[1].type, GT::CX);
}

// ============================================================================
// 24. Identity gate with modifiers
// ============================================================================

TEST(QASM3ParserTest, InvIdIsStillDropped) {
    auto qc = parse3("qubit[1] q;\ninv @ id q[0];\nh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}
TEST(QASM3ParserTest, PowIdIsStillDropped) {
    auto qc = parse3("qubit[1] q;\npow(5) @ id q[0];\nx q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::X);
}
TEST(QASM3ParserTest, CtrlIdIsStillDropped) {
    auto qc = parse3("qubit[2] q;\nctrl @ id q[0], q[1];\nh q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::H);
}

// ============================================================================
// 25. Larger integration tests
// ============================================================================

TEST(QASM3ParserTest, GHZThreeQubit) {
    auto qc = parse3(
        "qubit[3] q;\nbit[3] c;\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "cx q[1], q[2];\n"
        "c[0] = measure q[0];\n"
        "c[1] = measure q[1];\n"
        "c[2] = measure q[2];\n");
    EXPECT_EQ(qc.n_qubits, 3);
    EXPECT_EQ(qc.n_clbits, 3);
    ASSERT_EQ(qc.instructions.size(), 6u);
}
TEST(QASM3ParserTest, ParametricAnsatz) {
    auto qc = parse3_raw(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] t0;\n"
        "input float[64] t1;\n"
        "qubit[2] q;\n"
        "rx(t0) q[0];\n"
        "ry(t1) q[1];\n"
        "cx q[0], q[1];\n"
        "rz(t0 + t1) q[1];\n");
    EXPECT_EQ(qc.parameter_names.size(), 2u);
    ASSERT_EQ(qc.instructions.size(), 4u);
    qc.bind_parameters({{"t0", 0.1}, {"t1", 0.2}});
    EXPECT_NEAR(qc.instructions[0].params[0], 0.1, 1e-12);
    EXPECT_NEAR(qc.instructions[1].params[0], 0.2, 1e-12);
    EXPECT_NEAR(qc.instructions[3].params[0], 0.3, 1e-12);
}
TEST(QASM3ParserTest, ModifierHeavyCircuit) {
    auto qc = parse3(
        "qubit[3] q;\n"
        "ctrl @ x q[0], q[1];\n"
        "ctrl @ ctrl @ x q[0], q[1], q[2];\n"
        "inv @ s q[0];\n"
        "pow(3) @ inv @ rx(0.2) q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 4u);
    EXPECT_EQ(qc.instructions[0].type, GT::CX);
    EXPECT_EQ(qc.instructions[1].type, GT::CCX);
    EXPECT_EQ(qc.instructions[2].type, GT::SDG);
    EXPECT_EQ(qc.instructions[3].type, GT::RX);
    EXPECT_NEAR(qc.instructions[3].params[0], -0.6, 1e-12);
}
TEST(QASM3ParserTest, CustomGateWithSymbolicCallSite) {
    auto qc = parse3_raw(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] theta;\n"
        "gate myrot(t) a { rx(t) a; ry(t/2.0) a; }\n"
        "qubit[1] q;\n"
        "myrot(theta) q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::RX);
    EXPECT_EQ(qc.instructions[1].type, GT::RY);
    qc.bind_parameters({{"theta", 0.4}});
    EXPECT_NEAR(qc.instructions[0].params[0], 0.4, 1e-12);
    EXPECT_NEAR(qc.instructions[1].params[0], 0.2, 1e-12);
}
TEST(QASM3ParserTest, MeasurementInIfBlockCarriesCondition) {
    auto qc = parse3(
        "qubit[1] q;\nbit[2] c;\n"
        "if (c[0] == 1) { x q[0]; c[1] = measure q[0]; }\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].condition_clbit, 0);
    EXPECT_EQ(qc.instructions[0].condition_value, 1);
    EXPECT_EQ(qc.instructions[1].condition_clbit, 0);
    EXPECT_EQ(qc.instructions[1].condition_value, 1);
}
TEST(QASM3ParserTest, BarrierDoesNotCancelGatesAcrossIt) {
    // Barrier acts as a serialization hint but does not enable cancellation.
    auto qc = parse3(
        "qubit[1] q;\nh q[0];\nbarrier q[0];\nh q[0];\n");
    // h ; h would normally cancel, but the barrier sits between them and
    // also lives on q[0], breaking the peephole window. Both gates survive.
    EXPECT_GE(qc.instructions.size(), 2u);
}

// ============================================================================
// 26. Order-sensitivity of peephole
// ============================================================================

TEST(QASM3ParserTest, PeepholeIgnoresSwappedCx) {
    auto qc = parse3(
        "qubit[2] q;\ncx q[0], q[1];\ncx q[1], q[0];\n");
    EXPECT_EQ(qc.instructions.size(), 2u);
}
TEST(QASM3ParserTest, PeepholeIgnoresSwappedCCXOuter) {
    auto qc = parse3(
        "qubit[3] q;\nccx q[0], q[1], q[2];\nccx q[1], q[0], q[2];\n");
    EXPECT_EQ(qc.instructions.size(), 2u);
}

// ============================================================================
// 27. UTF-8 identifier names
// ============================================================================

TEST(QASM3ParserTest, UnicodeParameterName) {
    auto qc = parse3_raw(
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] \xCE\xB8;\n"   // theta as UTF-8 bytes
        "qubit[1] q;\n"
        "rx(\xCE\xB8) q[0];\n");
    ASSERT_EQ(qc.parameter_names.size(), 1u);
    EXPECT_EQ(qc.parameter_names[0], "\xCE\xB8");
    qc.bind_parameters({{"\xCE\xB8", 1.25}});
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], 1.25);
}
