// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.2 - how the QASM parsers turn an operand into a qubit or bit.
//
// Both parsers computed a global index as register offset + index and never
// compared the index with the register's width. `x q[3]` on a two-qubit q
// therefore acted on the second qubit of the next register, and the circuit's
// own bounds check could not object, since that qubit exists. The QASM 2 parser
// also sent an undeclared register to offset 0, took a negative index to the
// previous register, matched register names as substrings (so `aq[0]` could
// resolve through `q` depending on hash-map order), recognised its statement
// keywords anywhere in a line (so `x resets[0]` was a reset and a line on a
// register whose name contained "qreg" was skipped), widened a barrier operand
// it could not resolve to the whole register, and dropped the second statement
// on a line. Every one of those produced a different circuit with no error.
//
// The parsers now follow the grammar they read. A register must be declared,
// an index must lie inside its register, an OpenQASM 2.0 integer is 0 or a
// digit string without a leading zero or sign, and every refusal is the
// parser's std::runtime_error. std::out_of_range, which a literal too large for
// an int used to escape as, is a logic_error and would slip past a caller
// catching runtime_error around the parser, so the overflow tests check the
// type exactly.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

const std::string kHeader2 = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";
const std::string kHeader3 = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";

QuantumCircuit parse2(const std::string& body) {
    return QuantumCircuit::from_qasm2(kHeader2 + body);
}

QuantumCircuit parse3(const std::string& body) {
    return QuantumCircuit::from_qasm3(kHeader3 + body);
}

// Two registers back to back, so an index past the first lands on a real qubit
// of the second: exactly the case the circuit's own check cannot catch.
const std::string kTwoQregs2 = "qreg q[2];\nqreg r[2];\ncreg c[1];\ncreg d[4];\n";
const std::string kTwoQregs3 = "qubit[2] q;\nqubit[2] r;\nbit[1] c;\nbit[4] d;\n";

}  // namespace

// =============================================================================
// QASM 2: register bounds
// =============================================================================

TEST(V11292QasmOperands, Qasm2AnIndexPastItsRegisterIsRefused) {
    for (const char* stmt : {"x q[2];", "cx q[0], q[3];", "measure q[2] -> c[0];",
                             "measure q[0] -> c[1];", "reset q[5];", "barrier q[2];",
                             "if (c == 1) x q[2];"}) {
        SCOPED_TRACE(stmt);
        EXPECT_THROW((void)parse2(kTwoQregs2 + stmt + "\n"), std::runtime_error);
    }
}

TEST(V11292QasmOperands, Qasm2AnIndexInsideItsRegisterLandsOnTheRightQubit) {
    // r follows q, so r[1] is global qubit 3 and d[2] is global bit 3.
    const QuantumCircuit qc = parse2(kTwoQregs2 + "x r[1];\nmeasure r[0] -> d[2];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::X);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{3}));
    EXPECT_EQ(qc.instructions[1].type, GT::MEASURE);
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{2}));
    EXPECT_EQ(qc.instructions[1].clbits, (std::vector<int>{3}));
}

TEST(V11292QasmOperands, Qasm2ANegativeIndexIsRefused) {
    // r[-1] was offset 2 plus -1: qubit 1, which belongs to q.
    for (const char* stmt : {"x r[-1];", "measure r[-1] -> c[0];", "reset r[-1];"}) {
        SCOPED_TRACE(stmt);
        EXPECT_THROW((void)parse2(kTwoQregs2 + stmt + "\n"), std::runtime_error);
    }
}

TEST(V11292QasmOperands, Qasm2IntegersFollowTheGrammar) {
    for (const std::string body : {kTwoQregs2 + "x q[01];\n", kTwoQregs2 + "x q[+1];\n",
                                   std::string("qreg q[02];\n"), kTwoQregs2 + "x q[1.0];\n"}) {
        SCOPED_TRACE(body);
        EXPECT_THROW((void)parse2(body), std::runtime_error);
    }
    // Whitespace between tokens is not part of the integer.
    const QuantumCircuit qc = parse2(kTwoQregs2 + "x q[ 1 ];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{1}));
}

TEST(V11292QasmOperands, Qasm2AnIntegerTooLargeForAnIntIsARuntimeError) {
    for (const std::string body : {std::string("qreg q[99999999999];\n"),
                                   kTwoQregs2 + "x q[99999999999];\n",
                                   kTwoQregs2 + "measure q[0] -> c[99999999999];\n"}) {
        SCOPED_TRACE(body);
        EXPECT_THROW((void)parse2(body), std::runtime_error);
    }
}

// =============================================================================
// QASM 2: register names
// =============================================================================

TEST(V11292QasmOperands, Qasm2AnUndeclaredRegisterIsRefused) {
    for (const char* stmt : {"x s[0];", "cx q[0], s[0];", "measure s[0] -> c[0];",
                             "measure q[0] -> e[0];", "reset s[0];", "barrier s;",
                             "barrier q[0], s[1];"}) {
        SCOPED_TRACE(stmt);
        EXPECT_THROW((void)parse2(kTwoQregs2 + stmt + "\n"), std::runtime_error);
    }
}

TEST(V11292QasmOperands, Qasm2ARegisterNameIsMatchedWholeNeverAsASuffix) {
    // With registers q and aq, `aq[0]` contains "q[0]", and with c and cc,
    // `cc[0]` contains "c[0]". Both declaration orders, since which name a
    // substring search tried first depended on hash-map order. A register's
    // global index is its declaration position, so aq and cc are 1 when they
    // are declared second and 0 when first.
    struct Case {
        std::string decls;
        int aq, cc;
    };
    const Case cases[] = {
        {"qreg q[1];\nqreg aq[1];\ncreg c[1];\ncreg cc[1];\n", 1, 1},
        {"qreg aq[1];\nqreg q[1];\ncreg cc[1];\ncreg c[1];\n", 0, 0},
    };
    for (const Case& k : cases) {
        SCOPED_TRACE(k.decls);
        const QuantumCircuit qc = parse2(k.decls + "x aq[0];\nmeasure aq[0] -> cc[0];\n");
        ASSERT_EQ(qc.instructions.size(), 2u);
        EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{k.aq}));
        EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{k.aq}));
        EXPECT_EQ(qc.instructions[1].clbits, (std::vector<int>{k.cc}));
    }
}

TEST(V11292QasmOperands, Qasm2AKeywordInsideAnIdentifierIsNotAStatement) {
    // Each of these names begins with or contains a statement keyword. Every
    // line is an ordinary gate call and must parse as one.
    const QuantumCircuit qc = parse2(
        "qreg resets[1];\nqreg barriers[1];\nqreg myqreg[1];\nqreg include_q[1];\n"
        "x resets[0];\nh barriers[0];\nx myqreg[0];\nh include_q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 4u);
    const GT types[] = {GT::X, GT::H, GT::X, GT::H};
    for (int i = 0; i < 4; ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(qc.instructions[i].type, types[i]);
        EXPECT_EQ(qc.instructions[i].qubits, (std::vector<int>{i}));
    }
}

TEST(V11292QasmOperands, Qasm2ACustomGateNamedLikeAKeywordIsACall) {
    // "gatefoo" begins with "gate", and reading the call as a definition
    // swallowed every line up to the next closing brace.
    const QuantumCircuit qc =
        parse2("qreg q[2];\ngate gatefoo a { x a; }\ngatefoo q[1];\nh q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::X);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{1}));
    EXPECT_EQ(qc.instructions[1].type, GT::H);
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{0}));
}

// =============================================================================
// QASM 2: barrier, comments, one statement per line
// =============================================================================

TEST(V11292QasmOperands, Qasm2ABareBarrierStillCoversEveryQubit) {
    const QuantumCircuit qc = parse2(kTwoQregs2 + "barrier;\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].type, GT::BARRIER);
}

TEST(V11292QasmOperands, Qasm2ABarrierKeepsItsNamedOperands) {
    const QuantumCircuit qc = parse2(kTwoQregs2 + "barrier q[1], r;\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{1, 2, 3}));
}

TEST(V11292QasmOperands, Qasm2ATrailingCommentIsIgnored) {
    const QuantumCircuit qc =
        parse2(kTwoQregs2 + "x q[1]; // flip\nmeasure q[1] -> c[0]; // read\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{1}));
    EXPECT_EQ(qc.instructions[1].clbits, (std::vector<int>{0}));
}

TEST(V11292QasmOperands, Qasm2TwoStatementsOnOneLineAreRefused) {
    // The second statement was dropped with no error. This parser reads one
    // statement per line, so a second one is refused rather than lost.
    for (const std::string body :
         {kTwoQregs2 + "x q[0]; h q[1];\n", std::string("qreg q[2]; creg c[2];\nx q[0];\n"),
          kTwoQregs2 + "measure q[0] -> c[0]; measure q[1] -> d[0];\n",
          kTwoQregs2 + "if (c == 1) x q[0]; h q[1];\n"}) {
        SCOPED_TRACE(body);
        EXPECT_THROW((void)parse2(body), std::runtime_error);
    }
}

TEST(V11292QasmOperands, Qasm2AMalformedOperandIsRefused) {
    for (const char* stmt : {"x q[0]x;", "x q[0;", "cx q[0], 2*q[1];"}) {
        SCOPED_TRACE(stmt);
        EXPECT_THROW((void)parse2(kTwoQregs2 + stmt + "\n"), std::runtime_error);
    }
}

// =============================================================================
// QASM 3
// =============================================================================

TEST(V11292QasmOperands, Qasm3AnIndexPastItsRegisterIsRefused) {
    for (const char* stmt : {"x q[2];", "cx q[0], q[3];", "c[1] = measure q[0];",
                             "d[4] = measure q[0];", "reset q[5];", "barrier q[2];",
                             "if (c[1] == 1) x q[0];"}) {
        SCOPED_TRACE(stmt);
        EXPECT_THROW((void)parse3(kTwoQregs3 + stmt + "\n"), std::runtime_error);
    }
}

TEST(V11292QasmOperands, Qasm3AnIndexInsideItsRegisterLandsOnTheRightQubit) {
    const QuantumCircuit qc = parse3(kTwoQregs3 + "x r[1];\nd[2] = measure r[0];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{3}));
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{2}));
    EXPECT_EQ(qc.instructions[1].clbits, (std::vector<int>{3}));
}

TEST(V11292QasmOperands, Qasm3AConditionIndexInsideItsRegisterNamesThatBit) {
    const QuantumCircuit qc = parse3(kTwoQregs3 + "if (d[3] == 1) x q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].condition_clbit, 4);
    EXPECT_EQ(qc.instructions[0].condition_value, 1);
}

TEST(V11292QasmOperands, Qasm3AnIntegerTooLargeForAnIntIsARuntimeError) {
    for (const std::string body :
         {std::string("qubit[99999999999] q;\n"), kTwoQregs3 + "x q[99999999999];\n",
          kTwoQregs3 + "if (c[0] == 99999999999) x q[0];\n",
          kTwoQregs3 + "ctrl(99999999999) @ x q[0], q[1];\n",
          kTwoQregs3 + "pow(99999999999) @ x q[0];\n"}) {
        SCOPED_TRACE(body);
        EXPECT_THROW((void)parse3(body), std::runtime_error);
    }
}

TEST(V11292QasmOperands, Qasm3AnAngleOutsideDoubleRangeIsARuntimeError) {
    // std::stod reports both overflow and underflow as std::out_of_range.
    for (const char* stmt : {"rx(1e999) q[0];", "rx(1e-400) q[0];"}) {
        SCOPED_TRACE(stmt);
        EXPECT_THROW((void)parse3(kTwoQregs3 + stmt + "\n"), std::runtime_error);
    }
}

TEST(V11292QasmOperands, Qasm3PowModifiersWhoseProductOverflowsAreRefused) {
    // 65536 * 65536 is 2^32, past INT_MAX: an int product would overflow,
    // which is undefined behaviour, and on a wrapping build reads as pow(0).
    for (const std::string body :
         {kTwoQregs3 + "pow(65536) @ pow(65536) @ x q[0];\n",
          kTwoQregs3 + "pow(-65536) @ pow(65536) @ x q[0];\n",
          kTwoQregs3 + "gate g a { pow(65536) @ pow(65536) @ x a; }\ng q[0];\n"}) {
        SCOPED_TRACE(body);
        EXPECT_THROW((void)parse3(body), std::runtime_error);
    }
}
