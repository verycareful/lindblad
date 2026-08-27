// R.1.21.1 test wave - how a physical-validity policy travels.
//
// The policy lives on the Instruction whose matrix it describes, so it moves
// with that matrix wherever the instruction is copied whole. That covers
// compose, repeat, inverse and the DAG round trip. Each is pinned below
// individually, because travelling by copy is an implementation detail rather
// than a guarantee, and a route that rebuilds an instruction field by field
// drops the policy without any other symptom.
//
// Establishing that lossless set is what turns the findings below into claims
// rather than anecdotes. The policy survives every route that copies, so a
// route that loses it is one that rebuilds, and rebuilding without carrying
// the field is an oversight at that site rather than a property of the design.
//
// Two routes currently lose it, and the tests asserting otherwise are RED on
// purpose: `control()` and the JSON round trip each keep the matrix and reset
// the policy that governs it, which turns a circuit the caller legitimately
// opted out of into one that will not run. That is issue #74, owed to
// R.1.21.2. The contract is what is asserted here, not the behaviour.
//
// QASM is a third route, and investigating it turned up something worse than a
// lost policy: to_qasm2 writes a multi-qubit UNITARY as a custom gate whose
// body is a literal `cx`, so the operator itself is silently replaced. That is
// issue #75, and the test asserting otherwise is RED too.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

std::vector<Complex128> good_1q() {
    constexpr double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

std::vector<Complex128> good_2q() {   // CNOT, qubits[0] the control
    std::vector<Complex128> m(16, Complex128(0.0, 0.0));
    m[0 * 4 + 0] = Complex128(1.0, 0.0);
    m[1 * 4 + 3] = Complex128(1.0, 0.0);
    m[2 * 4 + 2] = Complex128(1.0, 0.0);
    m[3 * 4 + 1] = Complex128(1.0, 0.0);
    return m;
}

// CZ: unitary, two-qubit, and distinguishable from the `cx` the QASM exporter
// writes as a placeholder body. Using a CNOT here would let the substitution
// pass unnoticed.
std::vector<Complex128> good_2q_distinguishable() {
    std::vector<Complex128> m(16, Complex128(0.0, 0.0));
    m[0 * 4 + 0] = Complex128(1.0, 0.0);
    m[1 * 4 + 1] = Complex128(1.0, 0.0);
    m[2 * 4 + 2] = Complex128(1.0, 0.0);
    m[3 * 4 + 3] = Complex128(-1.0, 0.0);
    return m;
}

// A matrix that is deliberately not unitary. Under Ignore it is a legitimate
// thing to put in a circuit, which is exactly why losing the Ignore matters:
// the circuit stops running rather than starting to produce wrong numbers.
std::vector<Complex128> non_unitary_1q() {
    return {Complex128(1.5, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(1.5, 0.0)};
}

// Finds the single UNITARY instruction in a circuit and returns its policy.
const ValidationOptions& sole_unitary_policy(const QuantumCircuit& qc) {
    for (const auto& inst : qc.instructions)
        if (inst.type == Instruction::GateType::UNITARY) return inst.validation;
    ADD_FAILURE() << "no UNITARY instruction survived the transformation, so "
                     "there is nothing to read a policy from";
    static const ValidationOptions fallback{};
    return fallback;
}

std::size_t count_unitaries(const QuantumCircuit& qc) {
    std::size_t n = 0;
    for (const auto& inst : qc.instructions)
        if (inst.type == Instruction::GateType::UNITARY) ++n;
    return n;
}

// A circuit holding one UNITARY under a distinctive, non-default policy. Both
// members differ from the default, so a transformation that reset either one
// is visible.
QuantumCircuit marked_circuit(int n_qubits = 2) {
    QuantumCircuit qc(n_qubits);
    qc.unitary(good_1q(), {0}, "marked", {Validation::Warn, 7.5e-9});
    return qc;
}

void expect_marked(const QuantumCircuit& qc, const char* route) {
    const auto& v = sole_unitary_policy(qc);
    EXPECT_EQ(v.policy, Validation::Warn)
        << route << " reset the policy to the default";
    EXPECT_EQ(v.atol, 7.5e-9) << route << " reset the tolerance to the default";
}

} // namespace

// =============================================================================
// In-memory transformations
// =============================================================================

TEST(R1211Propagation, PlainCopyPreservesThePolicy) {
    const auto original = marked_circuit();
    const QuantumCircuit copied = original;
    expect_marked(copied, "copy construction");

    QuantumCircuit assigned(2);
    assigned = original;
    expect_marked(assigned, "copy assignment");
}

TEST(R1211Propagation, ComposePreservesThePolicy) {
    QuantumCircuit base(2);
    base.h(0);
    expect_marked(base.compose(marked_circuit()), "compose");
}

TEST(R1211Propagation, ComposePreservesThePolicyOfBothOperands) {
    // Two marked circuits with different policies must not converge on one.
    QuantumCircuit first(2);
    first.unitary(good_1q(), {0}, "ignored", {Validation::Ignore, 0.0});
    QuantumCircuit second(2);
    second.unitary(good_1q(), {1}, "warned", {Validation::Warn, 1e-7});

    const auto joined = first.compose(second);
    ASSERT_EQ(count_unitaries(joined), 2u);

    std::vector<Validation> seen;
    for (const auto& inst : joined.instructions)
        if (inst.type == Instruction::GateType::UNITARY)
            seen.push_back(inst.validation.policy);
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0], Validation::Ignore);
    EXPECT_EQ(seen[1], Validation::Warn)
        << "compose collapsed two distinct policies into one";
}

TEST(R1211Propagation, RepeatPreservesThePolicyOnEveryCopy) {
    const auto repeated = marked_circuit().repeat(3);
    ASSERT_EQ(count_unitaries(repeated), 3u);
    for (const auto& inst : repeated.instructions) {
        if (inst.type != Instruction::GateType::UNITARY) continue;
        EXPECT_EQ(inst.validation.policy, Validation::Warn);
        EXPECT_EQ(inst.validation.atol, 7.5e-9);
    }
}

TEST(R1211Propagation, InversePreservesThePolicy) {
    // The inverse of a matrix is unitary exactly when the matrix is, so the
    // policy the caller set for one is the right policy for the other.
    expect_marked(marked_circuit().inverse(), "inverse");
}

TEST(R1211Propagation, ControlPreservesThePolicy) {
    // RED, issue #74. control() synthesises a fresh Instruction for the
    // controlled matrix and sets type, matrix, qubits, label and condition,
    // leaving the policy at its default. inverse() copies the whole instruction
    // and is correct, and the reasoning that justifies inverse() applies here
    // unchanged: the controlled matrix is unitary exactly when the block it was
    // built from is, because the control structure contributes identity on
    // every unselected slice.
    expect_marked(marked_circuit().control(1), "control");
}

TEST(R1211Propagation, ControlOfAnOptedOutMatrixStaysRunnable) {
    // RED, issue #74. The consequence stated end to end: a circuit that is
    // legal because the caller opted out must not stop being legal purely by
    // being controlled.
    QuantumCircuit qc(1);
    qc.unitary(non_unitary_1q(), {0}, "opted_out", {Validation::Ignore, 0.0});
    ASSERT_NO_THROW(qc.validate_physical()) << "the source circuit is legal";

    EXPECT_NO_THROW(qc.control(1).validate_physical())
        << "issue #74: controlling an opted-out matrix produced a circuit that "
           "cannot run, with no diagnostic explaining where the opt-out went";
}

TEST(R1211Propagation, DagRoundTripPreservesThePolicy) {
    const auto dag = DAGCircuit::from_circuit(marked_circuit());
    expect_marked(dag.to_circuit(), "DAG round trip");
}

TEST(R1211Propagation, IgnoreSurvivesEveryInMemoryRoute) {
    // Ignore is the policy whose loss actually changes behaviour, since it is
    // the one attached to matrices that would otherwise be rejected. Running
    // it through each route and then executing is the end-to-end statement.
    QuantumCircuit qc(2);
    qc.unitary(non_unitary_1q(), {0}, "opted_out", {Validation::Ignore, 0.0});

    EXPECT_NO_THROW(qc.validate_physical()) << "the source circuit";
    EXPECT_NO_THROW(QuantumCircuit(qc).validate_physical()) << "after copy";
    EXPECT_NO_THROW(qc.repeat(2).validate_physical()) << "after repeat";
    EXPECT_NO_THROW(qc.inverse().validate_physical()) << "after inverse";
    EXPECT_NO_THROW(qc.control(1).validate_physical())
        << "after control (RED, issue #74)";
    EXPECT_NO_THROW(
        DAGCircuit::from_circuit(qc).to_circuit().validate_physical())
        << "after a DAG round trip";

    QuantumCircuit base(2);
    base.h(1);
    EXPECT_NO_THROW(base.compose(qc).validate_physical()) << "after compose";
}

// =============================================================================
// Serialization
// =============================================================================

TEST(R1211SerialisationLoss, QasmRoundTripMustNotSilentlyChangeTheOperator) {
    // RED, issue #75, and a heavier defect than the policy loss this file was
    // written to investigate.
    //
    // to_qasm2 emits a multi-qubit UNITARY as a custom gate definition whose
    // body is the literal text `cx q0,q1`, whatever the matrix actually holds.
    // The matrix is never written. Re-importing therefore yields a CNOT, and
    // for a k-qubit unitary with k > 2 it yields a CNOT on the first two wires.
    // Nothing is reported at either end.
    //
    // The same function refuses MCX, MCP and PERMUTATION loudly on exactly this
    // ground, so the loud path already exists and this operand does not take
    // it. Either outcome satisfies the assertion below: refuse the export, or
    // round-trip the operator. Silently substituting a different operator is
    // the one thing that must not happen.
    QuantumCircuit qc(2);
    qc.unitary(good_2q_distinguishable(), {0, 1}, "custom_two_qubit");

    const Operator before = Operator::from_circuit(qc);

    std::string qasm;
    try {
        qasm = qc.to_qasm2();
    } catch (const std::runtime_error&) {
        // std::runtime_error is the exporter's type for an operand with no
        // faithful OpenQASM 2.0 spelling, pinned for MCX / MCP / PERMUTATION by
        // R1181Export.Qasm2DefaultThrowsForAllThreeOps. A multi-qubit UNITARY
        // is refused on the same ground and so reports the same way.
        SUCCEED() << "the export refused this operand loudly, which is the "
                     "other acceptable outcome";
        return;
    }

    const auto reimported = QuantumCircuit::from_qasm2(qasm);
    const Operator after = Operator::from_circuit(reimported);

    ASSERT_EQ(after.data.size(), before.data.size())
        << "issue #75: the round trip changed the operator's dimension";
    for (std::size_t k = 0; k < before.data.size(); ++k) {
        EXPECT_NEAR(after.data[k].real, before.data[k].real, 1e-12)
            << "issue #75: entry " << k
            << " changed across a QASM round trip that reported no error";
        EXPECT_NEAR(after.data[k].imag, before.data[k].imag, 1e-12)
            << "issue #75: entry " << k << " changed";
    }
}

TEST(R1211SerialisationLoss, QasmSingleQubitUnitaryRoundTripsFaithfully) {
    // The one-qubit case is Euler-decomposed to u(theta, phi, lambda), which
    // preserves the map exactly. What comes back is a standard gate rather
    // than a UNITARY, so it carries no policy, and that is correct rather than
    // lossy: a named gate's unitarity is the library's own business.
    //
    // The lowering is requested explicitly. Restructuring a circuit for export
    // is the caller's decision at every width, so the one-qubit path is not a
    // quiet exception to it, and this test asks for the same consent a
    // two-qubit operand would need.
    QuantumCircuit qc(1);
    qc.unitary(good_1q(), {0}, "euler_me");

    QasmExportOptions opts;
    opts.unitary_lowering = UnitaryLowering::Always;

    const Operator before = Operator::from_circuit(qc);
    const auto reimported = QuantumCircuit::from_qasm2(qc.to_qasm2(opts));
    const Operator after = Operator::from_circuit(reimported);

    EXPECT_EQ(count_unitaries(reimported), 0u)
        << "a one-qubit unitary returns as a standard gate, by design";
    ASSERT_EQ(after.data.size(), before.data.size());
    for (std::size_t k = 0; k < before.data.size(); ++k) {
        EXPECT_NEAR(after.data[k].real, before.data[k].real, 1e-12)
            << "entry " << k;
        EXPECT_NEAR(after.data[k].imag, before.data[k].imag, 1e-12)
            << "entry " << k;
    }
}

TEST(R1211SerialisationLoss, JsonRoundTripPreservesThePolicy) {
    // RED, issue #74. The JSON format is the library's own and writes the
    // matrix out entry by entry, so the round trip is faithful in every respect
    // except this one: the emitted object carries no policy, and from_json
    // therefore produces instructions at the default.
    //
    // A circuit that ran before being written to disk consequently throws when
    // read back, with nothing in the file to explain why and no record of the
    // opt-out the caller made. Persist and reload is what the format exists
    // for, so this is where the loss has a practical cost.
    QuantumCircuit qc(2);
    qc.unitary(non_unitary_1q(), {0}, "opted_out", {Validation::Ignore, 0.0});
    ASSERT_NO_THROW(qc.validate_physical())
        << "the source circuit is legal: the caller opted out";

    const auto reloaded = QuantumCircuit::from_json(qc.to_json());
    ASSERT_EQ(count_unitaries(reloaded), 1u)
        << "the matrix must survive the round trip; if it no longer does, the "
           "defect has moved rather than been fixed";

    EXPECT_EQ(sole_unitary_policy(reloaded).policy, Validation::Ignore)
        << "issue #74: the policy was reset to the default by serialisation";
    EXPECT_NO_THROW(reloaded.validate_physical())
        << "issue #74: a circuit that was legal before being written is "
           "rejected after being read back";
}

TEST(R1211SerialisationLoss, JsonRoundTripPreservesTheMatrixItself) {
    // The half that does work, and the reason the policy loss is worth fixing
    // rather than accepting: everything else about the instruction survives.
    QuantumCircuit qc(2);
    qc.unitary(good_2q(), {0, 1}, "faithful");

    const auto reloaded = QuantumCircuit::from_json(qc.to_json());
    ASSERT_EQ(count_unitaries(reloaded), 1u);

    const std::vector<Complex128>& original = qc.instructions[0].matrix;
    const std::vector<Complex128>& returned = reloaded.instructions[0].matrix;
    ASSERT_EQ(returned.size(), original.size());
    for (std::size_t k = 0; k < original.size(); ++k) {
        EXPECT_DOUBLE_EQ(returned[k].real, original[k].real) << "entry " << k;
        EXPECT_DOUBLE_EQ(returned[k].imag, original[k].imag) << "entry " << k;
    }
    EXPECT_EQ(reloaded.instructions[0].qubits, qc.instructions[0].qubits);
}

TEST(R1211SerialisationLoss, JsonRoundTripPreservesTheTolerance) {
    // RED, issue #74. Both members are lost, not only the policy, so a caller
    // who widened the tolerance for a specific matrix loses that too.
    QuantumCircuit qc(2);
    qc.unitary(good_1q(), {0}, "warned", {Validation::Warn, 1e-3});

    const auto reloaded = QuantumCircuit::from_json(qc.to_json());
    ASSERT_EQ(count_unitaries(reloaded), 1u);
    EXPECT_EQ(sole_unitary_policy(reloaded).policy, Validation::Warn)
        << "issue #74";
    EXPECT_EQ(sole_unitary_policy(reloaded).atol, 1e-3)
        << "issue #74: the tolerance is lost along with the policy";
}

TEST(R1211SerialisationLoss, LosingAPolicyFailsTowardsTheStricterSetting) {
    // The one mitigating property of issue #74, and the reason it is medium
    // rather than high: the default landed on is Throw, so a lost policy fails
    // loudly rather than quietly. This passes today and must keep passing
    // whatever the fix does, since a fix that defaulted to Ignore would turn a
    // loud failure into a silent wrong answer.
    QuantumCircuit qc(2);
    qc.unitary(good_1q(), {0}, "defaulted");
    const auto reloaded = QuantumCircuit::from_json(qc.to_json());
    ASSERT_EQ(count_unitaries(reloaded), 1u);
    EXPECT_NE(sole_unitary_policy(reloaded).policy, Validation::Ignore)
        << "an absent policy must never be read back as an opt-out";
}
