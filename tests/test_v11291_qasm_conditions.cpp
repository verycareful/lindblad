// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.1 test wave - classical conditions through the QASM parsers.
//
// 1.1.29.0 taught from_qasm2() the `if (creg == v) qop;` statement that
// to_qasm2() writes for a one-bit register (#90), so a conditional circuit
// survives the QASM 2 round trip. The condition must land on every instruction
// the guarded statement produces, and on nothing else: a custom gate expands
// into several primitives, a whole-register measurement into one instruction
// per bit, and each copy is conditioned or the circuit is not the one written.
//
// The parsers follow the grammar they read, strictly. In OpenQASM 2.0 an `if`
// guards one quantum operation (a gate call, measure or reset), never a
// barrier, and compares against a non-negative integer written without
// leading zeros or a sign. A one-bit register holds only 0 or 1, so any other
// value is a condition that can never hold and is refused rather than parsed
// into dead code. The same value rule binds the OpenQASM 3 parser, whose
// grammar does admit leading zeros in a decimal literal.
//
// FIVE TESTS SHIP RED in this release, each pinning the strict reading above
// against a parser that currently accepts the input:
//   V11291QasmConditions.Qasm2AnEmptyValueIsRefused
//   V11291QasmConditions.Qasm2BarrierAfterIfIsRefused
//   V11291QasmConditions.Qasm2ValueSpellingsTheGrammarForbidsAreRefused
//   V11291QasmConditions.Qasm3BitConditionOutsideZeroOneIsRefused
//   V11291QasmConditions.Qasm3OneBitRegisterConditionOutsideZeroOneIsRefused
// The parser changes that turn them green belong to the next patch release.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

const std::string kHeader = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";

QuantumCircuit parse2(const std::string& body) {
    return QuantumCircuit::from_qasm2(kHeader + body);
}

const std::string kHeader3 = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";

QuantumCircuit parse3(const std::string& body) {
    return QuantumCircuit::from_qasm3(kHeader3 + body);
}

bool conditioned(const Instruction& inst, int clbit, int value) {
    return inst.condition_clbit == clbit && inst.condition_value == value;
}

bool unconditioned(const Instruction& inst) { return inst.condition_clbit < 0; }

// A measurable circuit: a one-bit register fed by a measurement, and a gate
// conditioned on it. Qubit 1 is the target, so the condition's clbit and the
// gate's qubit differ and a parser mixing them up would show.
QuantumCircuit feedforward(int value, GT type) {
    QuantumCircuit qc(2, 1);
    qc.h(0);
    qc.measure(0, 0);
    qc.add_if(0, value, type, {1});
    qc.h(1);
    return qc;
}

QasmExportOptions always() {
    QasmExportOptions o;
    o.condition_export = ConditionExport::Always;
    return o;
}

// Every field the round trip must carry, instruction by instruction.
::testing::AssertionResult same_instructions(const QuantumCircuit& a, const QuantumCircuit& b) {
    if (a.instructions.size() != b.instructions.size())
        return ::testing::AssertionFailure()
               << a.instructions.size() << " vs " << b.instructions.size() << " instructions";
    for (std::size_t i = 0; i < a.instructions.size(); ++i) {
        const Instruction& x = a.instructions[i];
        const Instruction& y = b.instructions[i];
        if (x.type != y.type || x.qubits != y.qubits || x.clbits != y.clbits ||
            x.condition_clbit != y.condition_clbit ||
            (x.condition_clbit >= 0 && x.condition_value != y.condition_value))
            return ::testing::AssertionFailure()
                   << "instruction " << i << " differs: " << x.gate_name() << " vs "
                   << y.gate_name() << ", condition " << x.condition_clbit << "=="
                   << x.condition_value << " vs " << y.condition_clbit << "=="
                   << y.condition_value;
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

// =============================================================================
// The round trip closes
// =============================================================================

TEST(V11291QasmConditions, AOneBitConditionSurvivesTheQasm2RoundTrip) {
    for (int value : {0, 1}) {
        for (GT type : {GT::X, GT::H, GT::Z}) {
            SCOPED_TRACE("value " + std::to_string(value));
            const QuantumCircuit original = feedforward(value, type);
            const QuantumCircuit back = QuantumCircuit::from_qasm2(original.to_qasm2(always()));
            EXPECT_TRUE(same_instructions(original, back));
        }
    }
}

TEST(V11291QasmConditions, AConditionedParametricGateKeepsItsAngleAndCondition) {
    QuantumCircuit qc(2, 1);
    qc.measure(0, 0);
    qc.add_if(0, 1, GT::RZ, {1}, {0.375});
    const QuantumCircuit back = QuantumCircuit::from_qasm2(qc.to_qasm2(always()));
    ASSERT_EQ(back.instructions.size(), 2u);
    EXPECT_TRUE(conditioned(back.instructions[1], 0, 1));
    ASSERT_EQ(back.instructions[1].params.size(), 1u);
    EXPECT_DOUBLE_EQ(back.instructions[1].params[0], 0.375);
}

TEST(V11291QasmConditions, AOneBitRegisterBesideAnotherIsGuardedExactly) {
    // Two one-bit registers: the guard names c, the second measurement writes
    // d, and neither may borrow the other's bit.
    for (int value : {0, 1}) {
        SCOPED_TRACE(value);
        const QuantumCircuit back = parse2(
            "qreg q[2];\ncreg c[1];\ncreg d[1];\nx q[0];\nmeasure q[0] -> c[0];\n"
            "if (c == " + std::to_string(value) + ") x q[1];\nmeasure q[1] -> d[0];\n");
        ASSERT_EQ(back.instructions.size(), 4u);
        EXPECT_TRUE(conditioned(back.instructions[2], 0, value));
        EXPECT_TRUE(unconditioned(back.instructions[3]));
        EXPECT_EQ(back.instructions[3].clbits, std::vector<int>{1});
    }
}

// =============================================================================
// The condition lands on everything the statement produces, and nothing else
// =============================================================================

TEST(V11291QasmConditions, ACustomGateIsConditionedAsAWhole) {
    const QuantumCircuit qc = parse2(
        "gate pair a, b { h a; cx a, b; rz(0.5) b; }\n"
        "qreg q[3];\ncreg c[1];\n"
        "x q[2];\n"
        "if (c == 1) pair q[0], q[1];\n"
        "h q[2];\n");
    ASSERT_EQ(qc.instructions.size(), 5u);
    EXPECT_TRUE(unconditioned(qc.instructions[0]));
    for (int i = 1; i <= 3; ++i) EXPECT_TRUE(conditioned(qc.instructions[static_cast<std::size_t>(i)], 0, 1)) << i;
    EXPECT_TRUE(unconditioned(qc.instructions[4]));
    EXPECT_EQ(qc.instructions[1].type, GT::H);
    EXPECT_EQ(qc.instructions[2].type, GT::CX);
    EXPECT_EQ(qc.instructions[3].type, GT::RZ);
}

TEST(V11291QasmConditions, ANestedCustomGateIsConditionedAllTheWayDown) {
    const QuantumCircuit qc = parse2(
        "gate inner a, b { cx a, b; s b; }\n"
        "gate outer a, b { h a; inner a, b; x a; }\n"
        "qreg q[2];\ncreg c[1];\n"
        "if (c == 0) outer q[1], q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 4u);
    for (const Instruction& inst : qc.instructions) EXPECT_TRUE(conditioned(inst, 0, 0)) << inst.gate_name();
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{1, 0}));
}

TEST(V11291QasmConditions, AWholeRegisterMeasureConditionsEveryBit) {
    const QuantumCircuit qc = parse2(
        "qreg q[3];\ncreg c[1];\ncreg m[3];\nif (c == 1) measure q -> m;\n");
    ASSERT_EQ(qc.instructions.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(qc.instructions[i].type, GT::MEASURE);
        EXPECT_TRUE(conditioned(qc.instructions[i], 0, 1));
        EXPECT_EQ(qc.instructions[i].qubits, std::vector<int>{static_cast<int>(i)});
        EXPECT_EQ(qc.instructions[i].clbits, std::vector<int>{1 + static_cast<int>(i)});
    }
}

TEST(V11291QasmConditions, MeasureAndResetCanBeConditioned) {
    const QuantumCircuit qc = parse2(
        "qreg q[2];\ncreg c[1];\n"
        "if (c == 0) measure q[1] -> c[0];\n"
        "if (c == 1) reset q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 2u);
    EXPECT_EQ(qc.instructions[0].type, GT::MEASURE);
    EXPECT_TRUE(conditioned(qc.instructions[0], 0, 0));
    EXPECT_EQ(qc.instructions[1].type, GT::RESET);
    EXPECT_TRUE(conditioned(qc.instructions[1], 0, 1));
}

TEST(V11291QasmConditions, TheConditionBitIsTheRegistersPlaceInTheClassicalSpace) {
    const QuantumCircuit qc = parse2(
        "qreg q[1];\ncreg a[2];\ncreg b[1];\ncreg z[3];\n"
        "if (b == 1) x q[0];\n");
    ASSERT_EQ(qc.n_clbits, 6);
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_TRUE(conditioned(qc.instructions[0], 2, 1));
}

TEST(V11291QasmConditions, OnlyTheGuardedStatementIsConditioned) {
    const QuantumCircuit qc = parse2(
        "qreg q[2];\ncreg c[1];\n"
        "if (c == 1) x q[0];\n"
        "y q[1];\n"
        "if (c == 0) z q[1];\n"
        "h q[0];\n");
    ASSERT_EQ(qc.instructions.size(), 4u);
    EXPECT_TRUE(conditioned(qc.instructions[0], 0, 1));
    EXPECT_TRUE(unconditioned(qc.instructions[1]));
    EXPECT_TRUE(conditioned(qc.instructions[2], 0, 0));
    EXPECT_TRUE(unconditioned(qc.instructions[3]));
}

TEST(V11291QasmConditions, AConditionOnTheLastLineIsApplied) {
    const QuantumCircuit qc = parse2("qreg q[1];\ncreg c[1];\nif (c == 1) x q[0];");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_TRUE(conditioned(qc.instructions[0], 0, 1));
}

TEST(V11291QasmConditions, SpacingInsideTheGuardDoesNotMatter) {
    for (const char* line : {"if(c==1) x q[0];", "if (c==1) x q[0];", "if ( c == 1 ) x q[0];",
                             "if (c ==1)x q[0];"}) {
        SCOPED_TRACE(line);
        const QuantumCircuit qc = parse2(std::string("qreg q[1];\ncreg c[1];\n") + line + "\n");
        ASSERT_EQ(qc.instructions.size(), 1u);
        EXPECT_TRUE(conditioned(qc.instructions[0], 0, 1));
    }
}

// =============================================================================
// Refusals
// =============================================================================

TEST(V11291QasmConditions, AWiderRegisterIsRefusedNamingItAndItsWidth) {
    try {
        (void)parse2("qreg q[1];\ncreg wide[3];\nif (wide == 1) x q[0];\n");
        FAIL() << "a three-bit register condition was parsed";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("wide"), std::string::npos) << msg;
        EXPECT_NE(msg.find('3'), std::string::npos) << msg;
    }
}

TEST(V11291QasmConditions, AnUndeclaredRegisterIsRefused) {
    EXPECT_THROW((void)parse2("qreg q[1];\ncreg c[1];\nif (d == 1) x q[0];\n"),
                 std::runtime_error);
}

TEST(V11291QasmConditions, AValueAOneBitRegisterCannotHoldIsRefused) {
    for (const char* v : {"2", "3", "10", "-1"}) {
        SCOPED_TRACE(v);
        EXPECT_THROW((void)parse2(std::string("qreg q[1];\ncreg c[1];\nif (c == ") + v +
                                  ") x q[0];\n"),
                     std::runtime_error);
    }
}

TEST(V11291QasmConditions, AGuardThatGuardsNothingIsRefused) {
    EXPECT_THROW((void)parse2("qreg q[1];\ncreg c[1];\nif (c == 1)\nx q[0];\n"),
                 std::runtime_error);
}

TEST(V11291QasmConditions, AMalformedGuardIsRefused) {
    for (const char* line : {"if (c = 1) x q[0];", "if c == 1 x q[0];", "if (== 1) x q[0];",
                             "if (c == 1 x q[0];", "if (c == one) x q[0];"}) {
        SCOPED_TRACE(line);
        EXPECT_THROW((void)parse2(std::string("qreg q[1];\ncreg c[1];\n") + line + "\n"),
                     std::runtime_error);
    }
}

// ---- Inputs the grammar forbids, refused -----------------------------------

TEST(V11291QasmConditions, Qasm2AnEmptyValueIsRefused) {
    // Nothing between `==` and `)` is not a value, and reading it as 0 turns a
    // typo into a condition the author never wrote.
    for (const char* line : {"if (c == ) x q[0];", "if (c ==) x q[0];"}) {
        SCOPED_TRACE(line);
        EXPECT_THROW((void)parse2(std::string("qreg q[1];\ncreg c[1];\n") + line + "\n"),
                     std::runtime_error);
    }
}

TEST(V11291QasmConditions, Qasm2BarrierAfterIfIsRefused) {
    // A barrier is a directive, not a quantum operation, and the grammar
    // admits only a quantum operation after `if`. to_qasm2() never writes one.
    EXPECT_THROW((void)parse2("qreg q[2];\ncreg c[1];\nif (c == 1) barrier q;\n"),
                 std::runtime_error);
    EXPECT_THROW((void)parse2("qreg q[2];\ncreg c[1];\nif (c == 0) barrier q[0], q[1];\n"),
                 std::runtime_error);
}

TEST(V11291QasmConditions, Qasm2ValueSpellingsTheGrammarForbidsAreRefused) {
    // OpenQASM 2.0's integer is 0 or a digit string without a leading zero,
    // and carries no sign.
    for (const char* v : {"01", "00", "+1", "-0", "+0", "1.0", "0x1"}) {
        SCOPED_TRACE(v);
        EXPECT_THROW((void)parse2(std::string("qreg q[1];\ncreg c[1];\nif (c == ") + v +
                                  ") x q[0];\n"),
                     std::runtime_error);
    }
}

TEST(V11291QasmConditions, Qasm3BitConditionOutsideZeroOneIsRefused) {
    for (const char* v : {"2", "5", "10"}) {
        SCOPED_TRACE(v);
        EXPECT_THROW((void)parse3(std::string("qubit[1] q;\nbit[2] c;\nif (c[1] == ") + v +
                                  ") x q[0];\n"),
                     std::runtime_error);
    }
}

TEST(V11291QasmConditions, Qasm3OneBitRegisterConditionOutsideZeroOneIsRefused) {
    for (const char* v : {"2", "7"}) {
        SCOPED_TRACE(v);
        EXPECT_THROW((void)parse3(std::string("qubit[1] q;\nbit[1] c;\nif (c == ") + v +
                                  ") x q[0];\n"),
                     std::runtime_error);
    }
}

// ---- Controls for the strict readings: what must stay accepted -------------

TEST(V11291QasmConditions, Qasm3ZeroAndOneStayAccepted) {
    // Leading zeros are a valid OpenQASM 3 decimal literal, so 01 is the value
    // 1 there; the value rule is what binds, not the spelling.
    for (const char* v : {"0", "1", "01"}) {
        SCOPED_TRACE(v);
        const QuantumCircuit qc =
            parse3(std::string("qubit[1] q;\nbit[2] c;\nif (c[1] == ") + v + ") x q[0];\n");
        ASSERT_EQ(qc.instructions.size(), 1u);
        EXPECT_EQ(qc.instructions[0].condition_clbit, 1);
        EXPECT_EQ(qc.instructions[0].condition_value, std::string(v) == "0" ? 0 : 1);
    }
}

TEST(V11291QasmConditions, Qasm2ACanonicalZeroAndOneStayAccepted) {
    for (int value : {0, 1}) {
        const QuantumCircuit qc = parse2("qreg q[1];\ncreg c[1];\nif (c == " +
                                         std::to_string(value) + ") x q[0];\n");
        ASSERT_EQ(qc.instructions.size(), 1u);
        EXPECT_TRUE(conditioned(qc.instructions[0], 0, value));
    }
}
