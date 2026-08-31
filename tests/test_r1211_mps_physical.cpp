// R.1.21.1 test wave - Class C at the MPS layer.
//
// Two public entry points take a caller-supplied matrix here, and both are
// fixed size (4 or 16 entries), so the check is 8 or 64 complex multiplies
// against an SVD. That makes MPS the backend where the check is cheapest
// relative to the work it guards, and the one where leaving it on by default
// costs least.
//
// The internal re-entries are the part worth pinning. apply_two_qubit_gate
// re-enters itself with the operands swapped when q1 > q2, and passes Ignore on
// the grounds that a symmetric row/column permutation of a matrix does not
// change whether it is unitary. That reasoning is sound, but it means the outer
// call is the only place the check happens, so a test has to establish the
// swapped path still rejects.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <array>
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

std::array<Complex128, 4> good_1q() {
    constexpr double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

// Deviation exactly 1.25.
std::array<Complex128, 4> bad_1q() {
    return {Complex128(1.5, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(1.5, 0.0)};
}

std::array<Complex128, 4> deviating_1q(double deviation) {
    const double scale = std::sqrt(1.0 + deviation);   // scale² - 1 == deviation
    return {Complex128(scale, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
}

std::array<Complex128, 16> good_2q() {   // CNOT
    std::array<Complex128, 16> m{};
    m[0 * 4 + 0] = Complex128(1.0, 0.0);
    m[1 * 4 + 3] = Complex128(1.0, 0.0);
    m[2 * 4 + 2] = Complex128(1.0, 0.0);
    m[3 * 4 + 1] = Complex128(1.0, 0.0);
    return m;
}

// A symmetric two-qubit unitary: identical under the row/column permutation
// the operand swap applies, so it is valid from either operand order.
std::array<Complex128, 16> good_2q_symmetric() {   // CZ
    std::array<Complex128, 16> m{};
    m[0 * 4 + 0] = Complex128(1.0, 0.0);
    m[1 * 4 + 1] = Complex128(1.0, 0.0);
    m[2 * 4 + 2] = Complex128(1.0, 0.0);
    m[3 * 4 + 3] = Complex128(-1.0, 0.0);
    return m;
}

std::array<Complex128, 16> bad_2q() {
    auto m = good_2q();
    m[1 * 4 + 3] = Complex128(0.0, 0.0);   // a basis image deleted
    return m;
}

// good_2q with one basis image deleted, chosen so the loss is observable.
//
// Which column to delete depends on the operand layout, and the MPS layer does
// not use the circuit layer's. Here the FIRST operand is the HIGH bit of the
// matrix index (row = po1*2 + po2, as the operand-swap block in mps_sim.cpp
// spells out), so after H on qubit 0 the amplitude sits on matrix indices 0 and
// 2, not 0 and 1. Deleting the image of column 2 destroys half the norm;
// deleting column 1 or 3 would be just as non-unitary and entirely invisible on
// this state.
std::array<Complex128, 16> truncating_2q() {
    auto m = good_2q();
    m[2 * 4 + 2] = Complex128(0.0, 0.0);
    return m;
}

// The circuit layer stores matrices as vectors while the MPS primitives take
// fixed-size arrays, so the circuit-level tests below need the same operand in
// the other container.
std::vector<Complex128> as_vector(const std::array<Complex128, 4>& m) {
    return {m[0], m[1], m[2], m[3]};
}

} // namespace

// =============================================================================
// apply_single_qubit_gate
// =============================================================================

TEST(R1211MpsGate1q, PolicyMatrix) {
    expect_repairs_invalid("MPSState::apply_single_qubit_gate",
                           [](ValidationOptions v) {
                               MPSState mps(4);
                               mps.apply_single_qubit_gate(bad_1q(), 2, v);
                           });
    expect_accepts_valid("MPSState::apply_single_qubit_gate",
                         [](ValidationOptions v) {
                             MPSState mps(4);
                             mps.apply_single_qubit_gate(good_1q(), 2, v);
                         });
}

TEST(R1211MpsGate1q, ToleranceIsConsultedRatherThanFixed) {
    expect_tolerance_is_honoured(
        "MPSState::apply_single_qubit_gate atol",
        [](ValidationOptions v) {
            MPSState mps(3);
            mps.apply_single_qubit_gate(deviating_1q(1e-9), 0, v);
        },
        1e-9);
}

TEST(R1211MpsGate1q, MessageNamesUnitarity) {
    MPSState mps(3);
    try {
        mps.apply_single_qubit_gate(bad_1q(), 0, ValidationOptions{});
        FAIL() << "a non-unitary single-qubit gate was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not unitary"), std::string::npos) << msg;
        EXPECT_NE(msg.find("1.25"), std::string::npos) << msg;
    }
}

TEST(R1211MpsGate1q, CheckAppliesAtEveryQubitPosition) {
    // The check must not depend on where in the chain the gate lands. A
    // boundary tensor has a different shape from an interior one, and the
    // orthogonality centre moves between calls.
    for (int q = 0; q < 4; ++q) {
        MPSState mps(4);
        EXPECT_THROW(mps.apply_single_qubit_gate(bad_1q(), q, ValidationOptions{}),
                     std::invalid_argument)
            << "qubit " << q << " accepted a non-unitary gate";
    }
}

TEST(R1211MpsGate1q, IgnoreStillAppliesTheGate) {
    MPSState mps(1);
    mps.apply_single_qubit_gate(bad_1q(), 0, {Validation::Ignore});
    const auto amps = mps.to_statevector().amplitudes();
    ASSERT_EQ(amps.size(), 2u);
    EXPECT_NEAR(amps[0].real, 1.5, 1e-12)
        << "the 1.5-scaled gate was not applied under Ignore";
}

// =============================================================================
// apply_two_qubit_gate
// =============================================================================

TEST(R1211MpsGate2q, PolicyMatrix) {
    expect_repairs_invalid("MPSState::apply_two_qubit_gate",
                           [](ValidationOptions v) {
                               MPSState mps(4);
                               mps.apply_two_qubit_gate(bad_2q(), 0, 1, v);
                           });
    expect_accepts_valid("MPSState::apply_two_qubit_gate",
                         [](ValidationOptions v) {
                             MPSState mps(4);
                             mps.apply_two_qubit_gate(good_2q(), 0, 1, v);
                         });
}

TEST(R1211MpsGate2q, SwappedOperandOrderStillRejects) {
    // q1 > q2 re-enters the function with the operands swapped, passing Ignore
    // because a symmetric row/column permutation preserves unitarity. That
    // makes the outer call the only place the check runs, so it has to run.
    MPSState mps(4);
    EXPECT_THROW(mps.apply_two_qubit_gate(bad_2q(), 2, 1, ValidationOptions{}),
                 std::invalid_argument)
        << "the descending operand order took the swapped re-entry and lost "
           "its check on the way";
}

TEST(R1211MpsGate2q, SwappedOperandOrderStillAcceptsAValidGate) {
    // The other half: the Ignore on the re-entry must not be reachable in a
    // way that makes a valid gate fail, and a symmetric unitary is valid from
    // either direction.
    MPSState mps(4);
    EXPECT_NO_THROW(
        mps.apply_two_qubit_gate(good_2q_symmetric(), 2, 1, ValidationOptions{}));
    EXPECT_NO_THROW(
        mps.apply_two_qubit_gate(good_2q_symmetric(), 1, 2, ValidationOptions{}));
}

TEST(R1211MpsGate2q, NonAdjacentOperandsAreStillChecked) {
    // Non-adjacent pairs are routed through a SWAP chain, which is more
    // internal re-entry and more chances for the policy to be dropped.
    MPSState mps(5);
    EXPECT_THROW(mps.apply_two_qubit_gate(bad_2q(), 0, 4, ValidationOptions{}),
                 std::invalid_argument);
    EXPECT_THROW(mps.apply_two_qubit_gate(bad_2q(), 4, 0, ValidationOptions{}),
                 std::invalid_argument);
}

TEST(R1211MpsGate2q, ValidNonAdjacentGateIsAccepted) {
    MPSState mps(5);
    EXPECT_NO_THROW(mps.apply_two_qubit_gate(good_2q(), 0, 4, ValidationOptions{}));
    EXPECT_NO_THROW(mps.apply_two_qubit_gate(good_2q(), 4, 0, ValidationOptions{}));
}

TEST(R1211MpsGate2q, IgnoreStillAppliesTheGate) {
    MPSState mps(2);
    mps.apply_single_qubit_gate(good_1q(), 0, ValidationOptions{});
    // After H on qubit 0 the amplitude sits on basis states 0 and 1, so the
    // matrix used here is the one whose deleted image is state 1: it destroys
    // half the norm. Deleting the image of state 3 instead would be equally
    // non-unitary and equally invisible on this state.
    mps.apply_two_qubit_gate(truncating_2q(), 0, 1, {Validation::Ignore});

    const auto amps = mps.to_statevector().amplitudes();
    ASSERT_EQ(amps.size(), 4u);
    double norm_sq = 0.0;
    for (const auto& a : amps) norm_sq += a.norm_sq();
    EXPECT_NEAR(norm_sq, 0.5, 1e-12)
        << "the truncating gate was not applied; norm_sq = " << norm_sq;
}

// =============================================================================
// Standard-gate paths
// =============================================================================

TEST(R1211MpsStandardGates, LibraryBuiltGatesRunUnderTheDefaultPolicy) {
    // The gate2x2 and gate4x4 builders pass Ignore: those matrices are the
    // library's own arithmetic, not a caller's declaration. A circuit of named
    // gates must therefore run clean under the strictest policy, and warn
    // about nothing.
    //
    // This is built up one gate at a time and each prefix is run, so a failure
    // names the exact gate that introduces it rather than only the whole
    // circuit. The full sequence currently fails: the SVD ladder throws
    // "both the SVD backend and the Gram-eigendecomposition fallback failed
    // verification on a 2x2 matrix", which is a library defect rather than a
    // property of these gates, and the prefix that first trips it is the
    // information needed to characterise it.
    struct Step { const char* label; void (*apply)(QuantumCircuit&); };
    const Step steps[]{
        {"h(0)",        [](QuantumCircuit& c) { c.h(0); }},
        {"cx(0,1)",     [](QuantumCircuit& c) { c.cx(0, 1); }},
        {"cx(1,2)",     [](QuantumCircuit& c) { c.cx(1, 2); }},
        {"t(3)",        [](QuantumCircuit& c) { c.t(3); }},
        {"s(2)",        [](QuantumCircuit& c) { c.s(2); }},
        {"rz(0.7,1)",   [](QuantumCircuit& c) { c.rz(0.7, 1); }},
        {"ry(0.3,3)",   [](QuantumCircuit& c) { c.ry(0.3, 3); }},
        {"cz(0,3)",     [](QuantumCircuit& c) { c.cz(0, 3); }},
        {"swap(1,3)",   [](QuantumCircuit& c) { c.swap(1, 3); }},
    };

    WarningProbe probe;
    QuantumCircuit qc(4);
    for (const auto& step : steps) {
        step.apply(qc);
        MPSSimulator sim;
        try {
            sim.run(qc, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/0);
        } catch (const std::exception& e) {
            FAIL() << "the MPS backend rejected a circuit of standard library "
                      "gates once " << step.label << " was appended: " << e.what();
        }
    }
    EXPECT_EQ(probe.count(), 0u)
        << "a circuit of standard gates produced a physical-validity warning";
}

TEST(R1211MpsStandardGates, PreflightRejectsANonUnitaryCircuit) {
    QuantumCircuit qc(2);
    Instruction inst;
    inst.type = Instruction::GateType::UNITARY;
    inst.qubits = {0};
    inst.matrix = as_vector(bad_1q());
    qc.instructions.push_back(inst);

    MPSSimulator sim;
    EXPECT_THROW(sim.run(qc, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/0),
                 std::invalid_argument)
        << "run() must pre-flight on this backend too";
}

TEST(R1211MpsStandardGates, PreflightHonoursIgnoreOnTheInstruction) {
    QuantumCircuit qc(1);
    Instruction inst;
    inst.type = Instruction::GateType::UNITARY;
    inst.qubits = {0};
    inst.matrix = as_vector(bad_1q());
    inst.validation = {Validation::Ignore, 0.0};
    qc.instructions.push_back(inst);

    MPSSimulator sim;
    EXPECT_NO_THROW(sim.run(qc, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/0));
}

TEST(R1211MpsStandardGates, StructureCheckSurvivesIgnore) {
    // Class B is unconditional on this backend as well.
    MPSState mps(2);
    EXPECT_THROW(mps.apply_single_qubit_gate(good_1q(), 7, {Validation::Ignore}),
                 std::out_of_range)
        << "an out-of-range qubit index is Class A and must stay loud whatever "
           "the physical-validity policy says";
}
