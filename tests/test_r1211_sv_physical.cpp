// R.1.21.1 test wave - Class C at the statevector layer.
//
// Four entry points reach the unitarity check on this backend: the kernel
// (gates::apply_unitary), the circuit ingress (QuantumCircuit::unitary), the
// instruction dispatcher (StatevectorSimulator::apply_instruction, in both its
// two- and three-argument forms), and the pre-flight that run() and
// simulate_circuit perform over a whole circuit.
//
// The two dispatcher overloads are the interesting pair. The two-argument form
// applies the instruction's own options, which is what a caller driving
// instructions by hand wants; the three-argument form applies the caller's,
// which is how run() passes Ignore after its pre-flight has already measured
// every matrix once. Confusing the two would either check every matrix once per
// shot or stop checking altogether, and neither shows up as a wrong amplitude.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using r1211::expect_accepts_valid;
using r1211::expect_rejects_invalid;
using r1211::expect_repairs_invalid;
using r1211::expect_tolerance_is_honoured;
using r1211::WarningProbe;

namespace {

// A correct single-qubit gate. Its residual is a couple of ulp, far inside the
// 1e-12 default.
std::vector<Complex128> good_1q() {
    constexpr double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

// Columns of norm 1.5 rather than 1. The deviation is exactly 1.25, chosen so
// nothing about the failure depends on rounding.
std::vector<Complex128> bad_1q() {
    return {Complex128(1.5, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(1.5, 0.0)};
}

std::vector<Complex128> good_2q() {   // CNOT, qubits[0] the control
    std::vector<Complex128> m(16, Complex128(0.0, 0.0));
    m[0 * 4 + 0] = Complex128(1.0, 0.0);
    m[1 * 4 + 3] = Complex128(1.0, 0.0);
    m[2 * 4 + 2] = Complex128(1.0, 0.0);
    m[3 * 4 + 1] = Complex128(1.0, 0.0);
    return m;
}

// CNOT with one basis image deleted, so a column loses its norm entirely.
std::vector<Complex128> bad_2q() {
    auto m = good_2q();
    m[1 * 4 + 3] = Complex128(0.0, 0.0);
    return m;
}

std::vector<Complex128> good_3q() {   // a diagonal sign pattern, exactly unitary
    std::vector<Complex128> m(64, Complex128(0.0, 0.0));
    for (std::size_t i = 0; i < 8; ++i)
        m[i * 8 + i] = Complex128((i % 3 == 0) ? -1.0 : 1.0, 0.0);
    return m;
}

std::vector<Complex128> bad_3q() {
    auto m = good_3q();
    m[5 * 8 + 5] = Complex128(0.5, 0.0);
    return m;
}

// A matrix whose deviation is a chosen size, for testing that atol is read
// rather than a constant being used. U = diag(sqrt(1+d), 1, ...) would need a
// square root; scaling one diagonal entry by (1 + d/2) gives a deviation of
// d to first order, so the exact construction below is used instead:
// |1 + e|^2 - 1 = 2e + e^2, so e = (sqrt(1 + dev) - 1) gives exactly `dev`.
std::vector<Complex128> deviating_1q(double deviation) {
    const double scale = std::sqrt(1.0 + deviation);
    return {Complex128(scale, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
}

Instruction make_unitary_instruction(const std::vector<Complex128>& m,
                                     const std::vector<int>& qubits) {
    Instruction inst;
    inst.type = Instruction::GateType::UNITARY;
    inst.qubits = qubits;
    inst.matrix = m;
    return inst;
}

} // namespace

// =============================================================================
// gates::apply_unitary
// =============================================================================

TEST(R1211SvKernel, SingleQubitPolicyMatrix) {
    expect_repairs_invalid("apply_unitary k=1", [](ValidationOptions v) {
        Statevector sv(3);
        gates::apply_unitary(sv, {1}, bad_1q(), v);
    });
    expect_accepts_valid("apply_unitary k=1", [](ValidationOptions v) {
        Statevector sv(3);
        gates::apply_unitary(sv, {1}, good_1q(), v);
    });
}

TEST(R1211SvKernel, TwoQubitPolicyMatrix) {
    expect_repairs_invalid("apply_unitary k=2", [](ValidationOptions v) {
        Statevector sv(4);
        gates::apply_unitary(sv, {0, 2}, bad_2q(), v);
    });
    expect_accepts_valid("apply_unitary k=2", [](ValidationOptions v) {
        Statevector sv(4);
        gates::apply_unitary(sv, {0, 2}, good_2q(), v);
    });
}

TEST(R1211SvKernel, ThreeQubitPolicyMatrix) {
    expect_repairs_invalid("apply_unitary k=3", [](ValidationOptions v) {
        Statevector sv(4);
        gates::apply_unitary(sv, {0, 1, 3}, bad_3q(), v);
    });
    expect_accepts_valid("apply_unitary k=3", [](ValidationOptions v) {
        Statevector sv(4);
        gates::apply_unitary(sv, {0, 1, 3}, good_3q(), v);
    });
}

TEST(R1211SvKernel, ToleranceIsConsultedRatherThanFixed) {
    expect_tolerance_is_honoured(
        "apply_unitary atol",
        [](ValidationOptions v) {
            Statevector sv(2);
            gates::apply_unitary(sv, {0}, deviating_1q(1e-9), v);
        },
        1e-9);
}

TEST(R1211SvKernel, MessageNamesTheEntryPointAndTheResidual) {
    Statevector sv(2);
    try {
        gates::apply_unitary(sv, {0}, bad_1q(), ValidationOptions{});
        FAIL() << "a non-unitary matrix reached the kernel without objection";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not unitary"), std::string::npos) << msg;
        EXPECT_NE(msg.find("U†U - I"), std::string::npos) << msg;
        EXPECT_NE(msg.find("1.25"), std::string::npos)
            << "the residual is exactly 1.25 and must be reported; got: " << msg;
    }
}

TEST(R1211SvKernel, StructureCheckRunsBeforeThePhysicalCheck) {
    // A wrong-sized matrix is a Class B failure. Measuring unitarity first
    // would read past the operand, so the ordering is a memory-safety property
    // rather than a matter of which message is nicer.
    Statevector sv(3);
    std::vector<Complex128> too_small(2, Complex128(1.0, 0.0));   // k=2 needs 16
    try {
        gates::apply_unitary(sv, {0, 1}, too_small, ValidationOptions{});
        FAIL() << "a 2-entry matrix was accepted for a 2-qubit target";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("U†U - I"), std::string::npos)
            << "the physical check ran on an operand of the wrong size; got: "
            << msg;
    }
}

TEST(R1211SvKernel, IgnoreDoesNotSuppressTheStructureCheck) {
    // Ignore opts out of Class C only. Class A and Class B are always on,
    // because they are the difference between a loud error and undefined
    // behaviour.
    Statevector sv(3);
    std::vector<Complex128> too_small(2, Complex128(1.0, 0.0));
    EXPECT_THROW(
        gates::apply_unitary(sv, {0, 1}, too_small, {Validation::Ignore}),
        std::invalid_argument);
}

TEST(R1211SvKernel, IgnoreStillAppliesTheMatrix) {
    // Opting out of the check must not opt out of the work. A non-unitary
    // matrix under Ignore produces a well-defined unphysical state, and the
    // amplitudes must show the map was applied.
    Statevector sv(1);
    gates::apply_unitary(sv, {0}, bad_1q(), {Validation::Ignore});
    const auto amps = sv.amplitudes();
    ASSERT_EQ(amps.size(), 2u);
    EXPECT_DOUBLE_EQ(amps[0].real, 1.5)
        << "the 1.5-scaled matrix was not applied under Ignore";
    EXPECT_DOUBLE_EQ(amps[1].real, 0.0);
}

// =============================================================================
// QuantumCircuit::unitary - ingress
// =============================================================================

TEST(R1211SvCircuitIngress, PolicyMatrixAtIngress) {
    expect_repairs_invalid("QuantumCircuit::unitary", [](ValidationOptions v) {
        QuantumCircuit qc(2);
        qc.unitary(bad_1q(), {0}, "probe", v);
    });
    expect_accepts_valid("QuantumCircuit::unitary", [](ValidationOptions v) {
        QuantumCircuit qc(2);
        qc.unitary(good_1q(), {0}, "probe", v);
    });
}

TEST(R1211SvCircuitIngress, RejectionLeavesNoInstructionBehind) {
    QuantumCircuit qc(2);
    EXPECT_THROW(qc.unitary(bad_1q(), {0}, "rejected", ValidationOptions{}),
                 std::invalid_argument);
    EXPECT_TRUE(qc.instructions.empty())
        << "a rejected gate must not be half-appended; a caller catching the "
           "exception would otherwise hold a circuit containing the matrix "
           "that was just refused";
}

TEST(R1211SvCircuitIngress, PolicyIsStoredOnTheInstruction) {
    QuantumCircuit qc(2);
    qc.unitary(good_1q(), {0}, "kept", {Validation::Warn, 5e-9});
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].validation.policy, Validation::Warn)
        << "run()'s pre-flight reads this; losing it would silently restore "
           "the default policy for a matrix the caller had already ruled on";
    EXPECT_EQ(qc.instructions[0].validation.atol, 5e-9);
}

TEST(R1211SvCircuitIngress, DefaultedArgumentGivesThrowAtTheDefaultTolerance) {
    QuantumCircuit qc(2);
    qc.unitary(good_1q(), {0});
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].validation.policy, Validation::Throw);
    EXPECT_EQ(qc.instructions[0].validation.atol, 1e-12);
}

TEST(R1211SvCircuitIngress, IgnoreAtIngressAdmitsTheMatrix) {
    QuantumCircuit qc(2);
    qc.unitary(bad_1q(), {0}, "admitted", {Validation::Ignore});
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].validation.policy, Validation::Ignore);
}

// =============================================================================
// QuantumCircuit::validate_physical - the pre-flight
// =============================================================================

TEST(R1211SvPreflight, CatchesAMatrixThatBypassedIngress) {
    // The QASM parser builds instructions directly, so a matrix can reach a
    // circuit without passing through unitary(). The pre-flight is the one
    // place every matrix is guaranteed to be seen.
    QuantumCircuit qc(2);
    qc.instructions.push_back(make_unitary_instruction(bad_1q(), {0}));
    EXPECT_THROW(qc.validate_physical(), std::invalid_argument);
}

TEST(R1211SvPreflight, AcceptsAValidCircuit) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).t(2);
    qc.unitary(good_2q(), {0, 2}, "cnot_like");
    EXPECT_NO_THROW(qc.validate_physical());
}

TEST(R1211SvPreflight, SkipsInstructionsMarkedIgnore) {
    QuantumCircuit qc(2);
    auto inst = make_unitary_instruction(bad_1q(), {0});
    inst.validation = {Validation::Ignore, 0.0};
    qc.instructions.push_back(inst);
    EXPECT_NO_THROW(qc.validate_physical());
}

TEST(R1211SvPreflight, HonoursPerInstructionPolicyIndependently) {
    // Two matrices, two policies, one circuit. The pre-flight must read each
    // instruction's own options rather than any single circuit-wide setting.
    QuantumCircuit qc(2);
    auto ignored = make_unitary_instruction(bad_1q(), {0});
    ignored.validation = {Validation::Ignore, 0.0};
    qc.instructions.push_back(ignored);
    EXPECT_NO_THROW(qc.validate_physical());

    qc.instructions.push_back(make_unitary_instruction(bad_1q(), {1}));
    EXPECT_THROW(qc.validate_physical(), std::invalid_argument)
        << "the second instruction carries the default Throw and must be "
           "checked even though the first opted out";
}

TEST(R1211SvPreflight, WarnPolicyReportsWithoutThrowing) {
    QuantumCircuit qc(2);
    auto inst = make_unitary_instruction(bad_1q(), {0});
    inst.validation = {Validation::Warn, 1e-12};
    qc.instructions.push_back(inst);

    WarningProbe probe;
    EXPECT_NO_THROW(qc.validate_physical());
    EXPECT_GE(probe.count(), 1u);
}

TEST(R1211SvPreflight, IgnoresNonUnitaryInstructionTypes) {
    // Only UNITARY instructions carry a caller-supplied matrix. A named gate's
    // matrix is the library's own, and measure, reset and barrier have none.
    QuantumCircuit qc(2, 2);   // measure needs clbits; the default is zero
    qc.h(0).cx(0, 1).measure(0, 0).reset(1).barrier();
    EXPECT_NO_THROW(qc.validate_physical());
}

TEST(R1211SvPreflight, WrongSizedMatricesAreCaughtByTheKernelNotThePreflight) {
    // Neither circuit-level check owns matrix size. validate_physical skips an
    // instruction whose matrix is not rows*rows, because measuring unitarity
    // against an assumed side length would read past the operand, and
    // validate_operands inspects only qubit and clbit indices, which here are
    // valid. The guard is the kernel's own Class B size check.
    //
    // What must hold is that the defect is caught somewhere before it can
    // corrupt a state, so that is what is asserted.
    QuantumCircuit qc(2);
    auto inst = make_unitary_instruction({Complex128(1.0, 0.0)}, {0, 1});
    qc.instructions.push_back(inst);

    EXPECT_NO_THROW(qc.validate_physical())
        << "the physical check cannot measure an operand of the wrong size";
    EXPECT_NO_THROW(qc.validate_operands())
        << "the indices are valid; operand structure at the circuit layer is "
           "about indices";

    StatevectorSimulator sim;
    const auto result = sim.run(qc, 0, 0);
    EXPECT_FALSE(result.success)
        << "a 1-entry matrix on two qubits reached execution and was applied";
    EXPECT_NE(result.error_message.find("size mismatch"), std::string::npos)
        << "the failure must name the size defect. Got: "
        << result.error_message;
}

TEST(R1211SvPreflight, ReportsTheGateLabelAsContext) {
    QuantumCircuit qc(2);
    auto inst = make_unitary_instruction(bad_1q(), {0});
    inst.label = "my_custom_gate";
    qc.instructions.push_back(inst);
    try {
        qc.validate_physical();
        FAIL() << "expected a rejection";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("my_custom_gate"), std::string::npos)
            << "a circuit can hold many unitaries; the message must say which "
               "one failed. Got: " << e.what();
    }
}

// =============================================================================
// StatevectorSimulator::apply_instruction - the two overloads
// =============================================================================

TEST(R1211SvDispatch, TwoArgumentFormUsesTheInstructionsOwnPolicy) {
    StatevectorSimulator sim;

    {
        Statevector sv(2);
        auto inst = make_unitary_instruction(bad_1q(), {0});
        // Default Throw on the instruction.
        EXPECT_THROW(sim.apply_instruction(sv, inst), std::invalid_argument);
    }
    {
        Statevector sv(2);
        auto inst = make_unitary_instruction(bad_1q(), {0});
        inst.validation = {Validation::Ignore, 0.0};
        EXPECT_NO_THROW(sim.apply_instruction(sv, inst))
            << "the instruction opted out and the two-argument form must "
               "honour that rather than imposing a default";
    }
}

TEST(R1211SvDispatch, ThreeArgumentFormOverridesTheInstructionsPolicy) {
    StatevectorSimulator sim;

    {
        // run() passes Ignore after its pre-flight, so a default-Throw
        // instruction must not be re-checked per shot.
        Statevector sv(2);
        auto inst = make_unitary_instruction(bad_1q(), {0});
        EXPECT_NO_THROW(sim.apply_instruction(sv, inst, {Validation::Ignore}));
    }
    {
        // And the reverse: an instruction marked Ignore is checked when the
        // caller asks for it.
        Statevector sv(2);
        auto inst = make_unitary_instruction(bad_1q(), {0});
        inst.validation = {Validation::Ignore, 0.0};
        EXPECT_THROW(sim.apply_instruction(sv, inst, ValidationOptions{}),
                     std::invalid_argument);
    }
}

TEST(R1211SvDispatch, PolicyMatrixThroughTheDispatcher) {
    StatevectorSimulator sim;
    expect_repairs_invalid("apply_instruction", [&sim](ValidationOptions v) {
        Statevector sv(2);
        auto inst = make_unitary_instruction(bad_1q(), {0});
        sim.apply_instruction(sv, inst, v);
    });
    expect_accepts_valid("apply_instruction", [&sim](ValidationOptions v) {
        Statevector sv(2);
        auto inst = make_unitary_instruction(good_1q(), {0});
        sim.apply_instruction(sv, inst, v);
    });
}

// =============================================================================
// run() and simulate_circuit - the pre-flight in place
// =============================================================================

TEST(R1211SvRun, RunRejectsANonUnitaryCircuit) {
    QuantumCircuit qc(2);
    qc.instructions.push_back(make_unitary_instruction(bad_1q(), {0}));

    // run() converts an exception into Result::success plus error_message
    // rather than letting it escape, so the rejection is read from the result.
    StatevectorSimulator sim;
    const auto result = sim.run(qc, 0, 0);
    EXPECT_FALSE(result.success)
        << "run() must pre-flight the circuit; without it the matrix is "
           "checked per shot or not at all";
    EXPECT_NE(result.error_message.find("not unitary"), std::string::npos)
        << "the failure must name the physical defect rather than surfacing as "
           "a generic one. Got: " << result.error_message;
}

TEST(R1211SvRun, SimulateCircuitRejectsANonUnitaryCircuit) {
    // simulate_circuit is a public entry that does not pass through run(), so
    // it performs the pre-flight itself.
    QuantumCircuit qc(2);
    qc.instructions.push_back(make_unitary_instruction(bad_1q(), {0}));
    StatevectorSimulator sim;
    Statevector sv(2);
    EXPECT_THROW(sim.simulate_circuit(sv, qc), std::invalid_argument);
}

TEST(R1211SvRun, RunAcceptsACircuitMarkedIgnore) {
    QuantumCircuit qc(1);
    auto inst = make_unitary_instruction(bad_1q(), {0});
    inst.validation = {Validation::Ignore, 0.0};
    qc.instructions.push_back(inst);
    StatevectorSimulator sim;
    EXPECT_NO_THROW(sim.run(qc, 0, 0));
}

TEST(R1211SvRun, ShotsDoNotMultiplyTheWarningOutput) {
    // Warn inside a shots loop reports the same violation on the same
    // unchanged matrix once per shot. Deduplication is what keeps that
    // readable, and the pre-flight is what keeps it to one measurement.
    // A deviation of 1e-6 is six orders outside the default tolerance and so
    // is certainly reported, while leaving the state norm within 1e-6 of 1 so
    // that measurement sampling behaves normally. A grossly scaled matrix
    // would test the sampler's response to an unnormalised state instead of
    // testing the warning count.
    QuantumCircuit qc(1, 1);   // measure needs clbits; the default is zero
    qc.h(0);
    auto inst = make_unitary_instruction(deviating_1q(1e-6), {0});
    inst.validation = {Validation::Warn, 1e-12};
    qc.instructions.push_back(inst);
    qc.measure(0, 0);

    WarningProbe probe;
    StatevectorSimulator sim;
    EXPECT_NO_THROW(sim.run(qc, 64, 1234));
    EXPECT_LE(probe.count(), 2u)
        << "64 shots produced " << probe.count()
        << " warning lines; one delivery plus at most one repeat summary is "
           "the contract";
    EXPECT_GE(probe.count(), 1u) << "the violation was not reported at all";
}

TEST(R1211SvRun, ValidCircuitsRunUnchangedUnderEveryPolicy) {
    // The framework must be invisible to correct input. A Bell state is a Bell
    // state whatever the policy says.
    for (auto policy : {Validation::Throw, Validation::Warn, Validation::Fix,
                        Validation::Ignore}) {
        QuantumCircuit qc(2);
        qc.h(0);
        qc.unitary(good_2q(), {0, 1}, "cnot_like", {policy, 1e-12});

        WarningProbe probe;
        StatevectorSimulator sim;
        const auto result = sim.run(qc, 0, 0);
        const auto amps = result.final_state.amplitudes();
        ASSERT_EQ(amps.size(), 4u);
        EXPECT_NEAR(amps[0].real, INV_SQRT2, 1e-12)
            << "policy " << static_cast<int>(policy);
        EXPECT_NEAR(amps[3].real, INV_SQRT2, 1e-12)
            << "policy " << static_cast<int>(policy);
        EXPECT_EQ(probe.count(), 0u)
            << "a correct circuit warned under policy "
            << static_cast<int>(policy);
    }
}
