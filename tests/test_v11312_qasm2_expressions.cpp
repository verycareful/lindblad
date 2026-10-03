// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.2 - OpenQASM 2.0 parameter expressions, and gate bodies that refuse
// what they cannot expand.
//
// The QASM 2 reader evaluated a gate's arguments with a hand-split
// evaluator. A top-level argument it could not read became 0.0. A number
// before "pi" was read as far as it went, so rx(0.1+pi/2) became 0.05 pi and
// rx(pi/2+0.1) lost its 0.1. Inside a gate body x/0 was 0, a/b/c was a/(b/c),
// and parentheses and the grammar's functions were not read at all. The
// expander of a user-defined gate skipped a body gate it did not know, dropped
// a body operand that named no qubit argument, and ignored a call with the
// wrong number of arguments, while a built-in gate ignored arguments past the
// ones it uses. All of it silently.
//
// Every argument is now read by one parser for the grammar's `exp`, at the top
// level and in a gate body alike:
//
//   exp := real | nninteger | pi | id | exp + exp | exp - exp | exp * exp
//        | exp / exp | -exp | exp ^ exp | ( exp ) | unaryop ( exp )
//   unaryop := sin | cos | tan | exp | ln | sqrt
//
// with + and - lowest and left-associative, then * and / (left), then unary
// minus, then ^ (right-associative), so -2^2 is -4 and 2^3^2 is 512. Anything
// else, a division by zero and a value that is not finite are refused, and so
// is every mismatch the expander used to let through.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "v11311_helpers.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

const std::string kHeader = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\n";

// The one angle `rx(<expr>) q[0];` reads.
double angle_of(const std::string& expr) {
    const QuantumCircuit qc = QuantumCircuit::from_qasm2(kHeader + "rx(" + expr + ") q[0];\n");
    EXPECT_EQ(qc.instructions.size(), 1u) << expr;
    if (qc.instructions.size() != 1 || qc.instructions[0].params.size() != 1) return std::nan("");
    return qc.instructions[0].params[0];
}

// The message a program is refused with, exactly as std::runtime_error.
std::string refusal(const std::string& program) {
    const auto e = v11311::thrown<std::runtime_error>(
        [&] { (void)QuantumCircuit::from_qasm2(kHeader + program); });
    EXPECT_TRUE(e.has_value()) << program;
    return e ? std::string(e->what()) : std::string();
}

}  // namespace

// =============================================================================
// The grammar, at the top level
// =============================================================================

TEST(V11312Qasm2Expressions, EveryFormOfTheGrammarEvaluates) {
    const std::vector<std::pair<std::string, double>> cases = {
        {"0.5", 0.5},
        {".5", 0.5},
        {"5.", 5.0},
        {"1e-3", 1e-3},
        {"2.5E2", 2.5e2},
        {"pi", PI},
        {"-pi/4", -PI / 4},
        {"2*pi/3", 2 * PI / 3},
        {"0.1+pi/2", 0.1 + PI / 2},
        {"pi/2+0.1", PI / 2 + 0.1},
        {"3-2-1", (3.0 - 2.0) - 1.0},
        {"1/2/4", (1.0 / 2.0) / 4.0},
        {"2*-3", 2.0 * -3.0},
        {"(1+2)*3", (1.0 + 2.0) * 3.0},
        {"2^3^2", std::pow(2.0, std::pow(3.0, 2.0))},
        {"-2^2", -std::pow(2.0, 2.0)},
        {"2^-1", std::pow(2.0, -1.0)},
        {"sin(pi/2)", std::sin(PI / 2)},
        {"cos(0)", std::cos(0.0)},
        {"tan(0.3)", std::tan(0.3)},
        {"exp(1)", std::exp(1.0)},
        {"ln(2)", std::log(2.0)},
        {"sqrt(2)", std::sqrt(2.0)},
        {"-(pi - 1) * sqrt( 4 )", -(PI - 1.0) * std::sqrt(4.0)},
        {"sin(cos(0.5)^2)", std::sin(std::pow(std::cos(0.5), 2.0))},
    };
    for (const auto& [expr, want] : cases) {
        SCOPED_TRACE(expr);
        EXPECT_DOUBLE_EQ(angle_of(expr), want);
    }
}

TEST(V11312Qasm2Expressions, EveryArgumentOfAGateIsReadWhole) {
    const QuantumCircuit qc =
        QuantumCircuit::from_qasm2(kHeader + "u(sin(pi/2), (1+1)*pi, -0.25) q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].qubits, std::vector<int>{1});
    ASSERT_EQ(qc.instructions[0].params.size(), 3u);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], std::sin(PI / 2));
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[1], (1.0 + 1.0) * PI);
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[2], -0.25);
}

TEST(V11312Qasm2Expressions, WhatTheGrammarDoesNotHoldIsRefused) {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"0.5x", "unexpected 'x' after the expression"},
        {"theta", "'theta' names no parameter in scope"},
        {"1/0", "division by zero"},
        {"ln(0)", "evaluates to a value that is not finite"},
        {"sqrt(-1)", "evaluates to a value that is not finite"},
        {"(0.5 1)", "expected ')'"},
        {"2*", "expected a number, pi, a parameter, '(' or a function"},
        {"+1", "expected a number, pi, a parameter, '(' or a function"},
        {"sin 1", "expected '(' after 'sin'"},
        {"1e400", "'1e400' is outside the range of a double"},
    };
    for (const auto& [expr, why] : cases) {
        SCOPED_TRACE(expr);
        EXPECT_EQ(refusal("rx(" + expr + ") q[0];\n"),
                  "QASM2Parser: parameter expression '" + expr + "': " + why);
    }
}

TEST(V11312Qasm2Expressions, AnUnbalancedCallIsRefused) {
    EXPECT_EQ(refusal("rx((0.5) q[0];\n"),
              "QASM2Parser: unbalanced parentheses in 'rx((0.5) q[0]'");
}

TEST(V11312Qasm2Expressions, AnEmptyArgumentIsRefusedNotSkipped) {
    EXPECT_EQ(refusal("u(0.1,,0.3) q[0];\n"),
              "QASM2Parser: parameter expression '': the expression is empty");
}

TEST(V11312Qasm2Expressions, ABuiltInGateTakesExactlyItsArguments) {
    EXPECT_EQ(refusal("rx(0.1, 0.2) q[0];\n"),
              "QASM2Parser: gate 'rx' takes 1 parameter and 1 qubit; it was given 2 parameters "
              "and 1 qubit");
    EXPECT_EQ(refusal("h(0.5) q[0];\n"),
              "QASM2Parser: gate 'h' takes 0 parameters and 1 qubit; it was given 1 parameter "
              "and 1 qubit");
    EXPECT_EQ(refusal("u(0.1, 0.2) q[0];\n"),
              "QASM2Parser: gate 'u' takes 3 parameters and 1 qubit; it was given 2 parameters "
              "and 1 qubit");
    EXPECT_EQ(refusal("cx q[0];\n"),
              "QASM2Parser: gate 'cx' takes 0 parameters and 2 qubits; it was given 0 parameters "
              "and 1 qubit");
}

// =============================================================================
// Gate bodies
// =============================================================================

TEST(V11312Qasm2Expressions, AGateBodyReadsTheSameGrammarWithItsParameters) {
    const QuantumCircuit qc = QuantumCircuit::from_qasm2(
        kHeader +
        "gate g(a, b) t { rx(a/b/2) t; ry(-a^2) t; rz(sin(a) + b) t; u(a, -b, (a+b)*pi) t; }\n"
        "g(1.5, 2) q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 4u);
    const double a = 1.5, b = 2.0;
    EXPECT_DOUBLE_EQ(qc.instructions[0].params[0], (a / b) / 2.0);
    EXPECT_DOUBLE_EQ(qc.instructions[1].params[0], -std::pow(a, 2.0));
    EXPECT_DOUBLE_EQ(qc.instructions[2].params[0], std::sin(a) + b);
    ASSERT_EQ(qc.instructions[3].params.size(), 3u);
    EXPECT_DOUBLE_EQ(qc.instructions[3].params[0], a);
    EXPECT_DOUBLE_EQ(qc.instructions[3].params[1], -b);
    EXPECT_DOUBLE_EQ(qc.instructions[3].params[2], (a + b) * PI);
    for (const Instruction& inst : qc.instructions) EXPECT_EQ(inst.qubits, std::vector<int>{1});
}

TEST(V11312Qasm2Expressions, AGateBodyRefusesADivisionByZero) {
    EXPECT_EQ(refusal("gate g(a) t { rx(1/a) t; }\ng(0) q[0];\n"),
              "QASM2Parser: parameter expression '1/a': division by zero");
}

TEST(V11312Qasm2Expressions, AGateBodyRefusesAGateItCannotExpand) {
    EXPECT_EQ(refusal("gate g a { foo a; h a; }\ng q[0];\n"),
              "QASM2Parser: gate 'g' calls 'foo', which is neither built in nor defined");
}

TEST(V11312Qasm2Expressions, AGateBodyRefusesAnOperandThatIsNotAQubitArgument) {
    EXPECT_EQ(refusal("gate g a, b { cx a, b2; }\ng q[0], q[1];\n"),
              "QASM2Parser: gate 'g' names 'b2' in its body, which is not one of its qubit "
              "arguments (a, b)");
}

TEST(V11312Qasm2Expressions, ACustomGateTakesExactlyItsArguments) {
    const std::string def = "gate g(a, b) s, t { rx(a) s; ry(b) t; }\n";
    EXPECT_EQ(refusal(def + "g(0.1) q[0], q[1];\n"),
              "QASM2Parser: gate 'g' takes 2 parameters and 2 qubits; it was given 1 parameter "
              "and 2 qubits");
    EXPECT_EQ(refusal(def + "g(0.1, 0.2, 0.3) q[0], q[1];\n"),
              "QASM2Parser: gate 'g' takes 2 parameters and 2 qubits; it was given 3 parameters "
              "and 2 qubits");
    EXPECT_EQ(refusal(def + "g(0.1, 0.2) q[0];\n"),
              "QASM2Parser: gate 'g' takes 2 parameters and 2 qubits; it was given 2 parameters "
              "and 1 qubit");
}

TEST(V11312Qasm2Expressions, ANestedGateIsCheckedAtEveryLevel) {
    // The inner call passes one parameter where the inner gate takes two.
    EXPECT_EQ(refusal("gate inner(a, b) t { rx(a+b) t; }\n"
                      "gate outer(c) t { inner(c) t; }\n"
                      "outer(0.5) q[0];\n"),
              "QASM2Parser: gate 'inner' takes 2 parameters and 1 qubit; it was given 1 "
              "parameter and 1 qubit");
}

TEST(V11312Qasm2Expressions, AGateThatCallsItselfIsRefused) {
    // Definitions are collected before any call is expanded, so a body can
    // name its own gate; expanding it would never end.
    EXPECT_EQ(refusal("gate a t { b t; }\ngate b t { a t; }\na q[0];\n"),
              "QASM2Parser: gate 'a' calls itself (a -> b -> a)");
}

TEST(V11312Qasm2Expressions, AStatementWithNoOperandsIsRefused) {
    EXPECT_EQ(refusal("foo;\n"), "QASM2Parser: statement 'foo' names no operands");
}

TEST(V11312Qasm2Expressions, TheGrammarsOwnUAndCxAndABodyBarrierAreRead) {
    const QuantumCircuit qc = QuantumCircuit::from_qasm2(
        kHeader + "gate g a, b { U(pi, 0, pi) a; barrier a, b; CX a, b; }\ng q[0], q[1];\n");
    ASSERT_EQ(qc.instructions.size(), 3u);
    EXPECT_EQ(qc.instructions[0].type, Instruction::GateType::U);
    EXPECT_EQ(qc.instructions[0].params, (std::vector<double>{PI, 0.0, PI}));
    EXPECT_EQ(qc.instructions[1].type, Instruction::GateType::BARRIER);
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{0, 1}));
    EXPECT_EQ(qc.instructions[2].type, Instruction::GateType::CX);
    EXPECT_EQ(qc.instructions[2].qubits, (std::vector<int>{0, 1}));
}

TEST(V11312Qasm2Expressions, WhatTheExporterWritesReadsBackExactly) {
    // to_qasm2 writes 15 significant digits; reading them back gives the
    // double those digits name, exponents included.
    QuantumCircuit qc(1);
    qc.rx(1e-5, 0).ry(-PI / 3, 0).rz(123456.789, 0);
    const QuantumCircuit back = QuantumCircuit::from_qasm2(qc.to_qasm2());
    ASSERT_EQ(back.instructions.size(), 3u);
    EXPECT_EQ(back.to_qasm2(), qc.to_qasm2());
}
