// R.1.22.2 - a classical condition survives QASM export, or is refused loudly.
//
// This suite exists because of a hole in the test matrix rather than a hard
// bug report. Conditions were covered on the parse direction (QASM3ParserTest
// asserts condition_clbit after reading `if (...)`) and on the JSON round trip.
// Export was covered without conditions. Sixteen test files call add_if or
// p_if; two of those also call to_qasm; not one TEST block did both. So neither
// exporter ever wrote an `if`, for any gate type, and nothing noticed.
//
// Every test below therefore crosses the two: build a conditional circuit, then
// export it. That crossing is the point, and it is what the older suites cannot
// do by construction.
//
// The two formats differ because their `if` forms differ, not because the
// exporters disagree. QASM 3's names one bit and takes a block, so an
// instruction's text goes inside it and the round trip closes. QASM 2's
// compares a WHOLE classical register to an integer and takes a single quantum
// operation, so a single-bit condition has an exact spelling there only when
// the register is that one bit.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"

#include <stdexcept>
#include <string>

using namespace lindblad;

namespace {

using GT = Instruction::GateType;

std::size_t occurrences(const std::string& haystack, const std::string& needle) {
    std::size_t n = 0;
    for (std::size_t p = haystack.find(needle); p != std::string::npos;
         p = haystack.find(needle, p + needle.size()))
        ++n;
    return n;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// One conditional X on qubit 0, in a register of the requested width. The
// register width is the whole subject on the QASM 2 side, so it is a parameter
// rather than a constant.
QuantumCircuit conditional_x(int n_clbits) {
    QuantumCircuit qc(1, n_clbits);
    qc.add_if(0, 1, GT::X, {0});
    return qc;
}

QasmExportOptions with_conditions(ConditionExport mode) {
    QasmExportOptions o;
    o.condition_export = mode;
    return o;
}

}  // namespace

// =============================================================================
// QASM 3, which can say it exactly
// =============================================================================

TEST(R1222ConditionalExport, Qasm3WritesTheConditionByDefault) {
    const std::string text = conditional_x(1).to_qasm3();
    EXPECT_TRUE(contains(text, "if (c[0] == 1)"))
        << "QASM 3 names one bit and takes a block, so nothing is lost and "
           "there is nothing to consent to.\n"
        << text;
    EXPECT_TRUE(contains(text, "x q[0];")) << text;
}

// The whole reason the block form is emitted rather than a refusal: the parser
// on the other side already reads it, so the circuit comes back intact.
TEST(R1222ConditionalExport, Qasm3RoundTripsAConditionalCircuit) {
    const QuantumCircuit in = conditional_x(1);
    const QuantumCircuit back = QuantumCircuit::from_qasm3(in.to_qasm3());

    ASSERT_EQ(back.instructions.size(), 1u) << in.to_qasm3();
    EXPECT_EQ(back.instructions[0].type, GT::X);
    EXPECT_EQ(back.instructions[0].condition_clbit, 0)
        << "the condition was written and read, so it must survive the trip";
    EXPECT_EQ(back.instructions[0].condition_value, 1);
}

TEST(R1222ConditionalExport, Qasm3RefusesUnderNever) {
    EXPECT_THROW(conditional_x(1).to_qasm3(with_conditions(ConditionExport::Never)),
                 std::runtime_error)
        << "Never means refuse in both formats, which is what makes a value "
           "read at a call site mean one thing wherever it is used";
}

// A 2-qubit UNITARY lowers into several gates. Conditioning each of them is
// equivalent to conditioning the group, since none writes a classical bit, and
// QASM 3 can simply put the group in one block.
TEST(R1222ConditionalExport, AConditionalTwoQubitUnitaryLowersInsideOneBlock) {
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);

    QuantumCircuit qc(2, 1);
    qc.unitary(Operator::from_circuit(src).data, {0, 1}, "u2");
    qc.instructions.back().condition_clbit = 0;
    qc.instructions.back().condition_value = 1;

    std::string text;
    ASSERT_NO_THROW(text = qc.to_qasm3())
        << "the operand lowers, and the condition rides on every gate it "
           "lowered into, so there is nothing here to refuse";

    EXPECT_EQ(occurrences(text, "if (c[0] == 1)"), 1u)
        << "the lowered sequence belongs in ONE block, not one block per "
           "emitted gate.\n"
        << text;
    EXPECT_GT(occurrences(text, "q["), 1u) << text;
}

// =============================================================================
// QASM 2, whose `if` reaches a whole register
// =============================================================================

TEST(R1222ConditionalExport, Qasm2RefusesByDefault) {
    try {
        conditional_x(2).to_qasm2();
        ADD_FAILURE() << "a condition OpenQASM 2.0 cannot spell must not be "
                         "dropped silently";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_TRUE(contains(msg, "condition_export"))
            << "the message must name the setting that changes the outcome: "
            << msg;
    }
}

// The one case OpenQASM 2.0 says exactly: with a single-bit register, "the
// register equals v" and "bit 0 equals v" are the same statement.
TEST(R1222ConditionalExport, Qasm2EmitsExactlyForAOneBitRegister) {
    r1211::WarningProbe probe;
    const std::string text =
        conditional_x(1).to_qasm2(with_conditions(ConditionExport::Always));

    EXPECT_TRUE(contains(text, "if (c == 1) x q[0];")) << text;
    EXPECT_FALSE(contains(text, "dropped condition"))
        << "nothing was lost here, so nothing should be recorded as lost.\n"
        << text;
    EXPECT_EQ(probe.count(), 0u)
        << "an exact export must not warn; a warning here would train a caller "
           "to ignore the one that means something";
}

// Wider registers cannot say it, so the export records what it dropped in both
// channels: the diagnostic sink for the caller, and the text for the file.
TEST(R1222ConditionalExport, Qasm2RecordsTheLossForAWiderRegister) {
    r1211::WarningProbe probe;
    const std::string text =
        conditional_x(3).to_qasm2(with_conditions(ConditionExport::Always));

    EXPECT_TRUE(contains(text, "// dropped condition: c[0] == 1"))
        << "the loss has to survive in the file, not only in a warning the "
           "reader of the file never saw.\n"
        << text;
    EXPECT_FALSE(contains(text, "if (c"))
        << "a register-wide `if` says something different from a single-bit "
           "condition, so it must not be written.\n"
        << text;
    EXPECT_TRUE(contains(text, "x q[0];")) << text;
    EXPECT_TRUE(probe.any_contains("dropped the classical condition"))
        << "the caller asked to proceed, not to proceed quietly";
}

// A barrier is a directive, and the OpenQASM 2.0 grammar admits only a quantum
// operation after `if`, so prefixing one would emit text no parser accepts.
TEST(R1222ConditionalExport, AConditionalBarrierNeverGetsAQasm2If) {
    QuantumCircuit qc(1, 1);
    qc.add_if(0, 1, GT::BARRIER, {0});

    const std::string text = qc.to_qasm2(with_conditions(ConditionExport::Always));
    EXPECT_TRUE(contains(text, "barrier")) << text;
    EXPECT_FALSE(contains(text, "if (c == 1) barrier")) << text;
}

// =============================================================================
// The unconditional case, which must be untouched by any of this
// =============================================================================

TEST(R1222ConditionalExport, AnUnconditionalCircuitGainsNoIfInEitherFormat) {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);

    for (ConditionExport mode : {ConditionExport::FormatDefault,
                                 ConditionExport::Always,
                                 ConditionExport::Never}) {
        const QasmExportOptions opts = with_conditions(mode);
        const std::string q2 = qc.to_qasm2(opts);
        const std::string q3 = qc.to_qasm3(opts);
        EXPECT_FALSE(contains(q2, "if (")) << q2;
        EXPECT_FALSE(contains(q3, "if (")) << q3;
        EXPECT_TRUE(contains(q2, "cx q[0], q[1];") ||
                    contains(q2, "cx q[0],q[1];")) << q2;
    }
}
