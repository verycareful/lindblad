// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// R.1.19.1 test wave — run() pre-flight, ingress paths, and the Statevector
// constructor guard.
//
// The per-gate circuit builders validate indices at construction, but an
// instruction can also enter a circuit through compose() index remapping,
// control(), the QASM parsers, or a transpiler pass, none of which re-check the
// remapped target. R.1.19.0 added QuantumCircuit::validate_operands(), a
// per-instruction sweep every backend run() invokes up front, so a bad index
// arriving by any of those routes is reported before it reaches a kernel.
//
// Every backend's run() refuses a bad index before its first gate with
// lindblad::OutOfRange naming the run and the instruction; validate_operands(),
// called directly, still throws a plain std::out_of_range. Both are pinned here.
//
// This file uses compose() with an out-of-range qubit mapping as a concrete,
// builder-bypassing ingress: the composed circuit carries an X on a qubit index
// beyond its register, which validate_operands() must catch. It also pins the
// bundled Statevector constructor fix (the count is validated before it reaches
// `1ULL << n` in the initializer list, so a negative count is a clean throw
// rather than shift-width UB).

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/types.hpp"
#include "v11311_helpers.hpp"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// Build a circuit whose only instruction (an X) sits on a qubit index beyond
// its register, produced through the compose() ingress path rather than a
// builder call. inner has one qubit; mapping its qubit 0 to position 5 while
// the composed result keeps outer's two qubits leaves an X on qubit 5 with
// n_qubits == 2 — exactly the compose-remap gap validate_operands() closes.
QuantumCircuit make_bad_index_circuit() {
    QuantumCircuit inner(1);
    inner.x(0);
    QuantumCircuit outer(2);
    return outer.compose(inner, {5});
}

// The refusal every backend's run() gives for make_bad_index_circuit(): exactly
// OutOfRange from the run, before the first gate (so no failed-run record),
// naming the index, the register and the instruction.
void expect_bad_index_refused(const std::string& entry_point, const std::function<void()>& run) {
    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<OutOfRange>(run);
    EXPECT_EQ(detail::failed_run_stores(), stores) << "a refusal left a failed-run record";
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), entry_point);
    EXPECT_EQ(std::string(e->what()),
              entry_point + ": qubit index 5 out of range [0, 2) (instruction 0: x on qubit 5)");
    v11311::expect_point(e->where(), -1, 0, "x", {5});
}

// A well-formed reference circuit (Bell pair) for the positive controls.
QuantumCircuit make_valid_circuit() {
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    return qc;
}

} // namespace

// =============================================================================
// validate_operands() directly
// =============================================================================

TEST(R1191Preflight, ValidateOperandsThrowsOnBadIndex) {
    QuantumCircuit bad = make_bad_index_circuit();
    const auto e = v11311::thrown<std::out_of_range>([&] { bad.validate_operands(); });
    ASSERT_TRUE(e.has_value());
    EXPECT_STREQ(e->what(), "Qubit index 5 out of range [0, 2)");
}

TEST(R1191Preflight, ValidateOperandsPassesOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    EXPECT_NO_THROW(ok.validate_operands());
}

// =============================================================================
// Every backend: refused before the first gate, naming the run and the gate
// =============================================================================

TEST(R1191Preflight, StatevectorRunRefusesBadIndexBeforeTheFirstGate) {
    QuantumCircuit bad = make_bad_index_circuit();
    StatevectorSimulator sim;
    expect_bad_index_refused("StatevectorSimulator::run", [&] { (void)sim.run(bad); });
}

TEST(R1191Preflight, StatevectorRunSucceedsOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    StatevectorSimulator sim;
    EXPECT_NO_THROW((void)sim.run(ok));
}

TEST(R1191Preflight, DensityMatrixRunRefusesBadIndexBeforeTheFirstGate) {
    QuantumCircuit bad = make_bad_index_circuit();
    NoiseModel noise;  // empty: exercises the noiseless path, pre-flight still runs
    DensityMatrixSimulator sim;
    expect_bad_index_refused("DensityMatrixSimulator::run", [&] { (void)sim.run(bad, noise); });
}

TEST(R1191Preflight, DensityMatrixRunSucceedsOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    NoiseModel noise;
    DensityMatrixSimulator sim;
    EXPECT_NO_THROW((void)sim.run(ok, noise));
}

TEST(R1191Preflight, MpsRunThrowsOnBadIndex) {
    QuantumCircuit bad = make_bad_index_circuit();
    MPSSimulator sim;
    expect_bad_index_refused("MPSSimulator::run", [&] { (void)sim.run(bad); });
}

TEST(R1191Preflight, MpsRunSucceedsOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    MPSSimulator sim;
    EXPECT_NO_THROW(sim.run(ok));
}

TEST(R1191Preflight, CliffordRunThrowsOnBadIndex) {
    QuantumCircuit bad = make_bad_index_circuit();  // X is Clifford; the index is not
    CliffordSimulator sim;
    expect_bad_index_refused("CliffordSimulator::run", [&] { (void)sim.run(bad, /*shots=*/16); });
}

TEST(R1191Preflight, CliffordRunSucceedsOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    CliffordSimulator sim;
    EXPECT_NO_THROW(sim.run(ok, /*shots=*/16));
}

// =============================================================================
// Statevector constructor guard (bundled R.1.19.0 fix)
// =============================================================================

TEST(R1191Preflight, StatevectorCtorRejectsBadCount) {
    // A negative count used to compute `1ULL << n` in the initializer list
    // BEFORE the range guard ran: shift-width undefined behaviour. The count is
    // validated first now, so every out-of-range count is a clean throw.
    for (const int n : {0, -1, -1000, ENFORCED_MAX_QUBITS + 1, 64}) {
        SCOPED_TRACE(n);
        const auto e = v11311::thrown<InvalidArgument>([&] { Statevector sv(n); });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(e->entry_point(), "Statevector");
        v11311::expect_message(*e, {"Statevector: n_qubits must be in [1, " +
                                    std::to_string(ENFORCED_MAX_QUBITS) + "], got " +
                                    std::to_string(n)});
        // Only a width Lift would accept is told about Lift.
        const bool liftable = n > ENFORCED_MAX_QUBITS && n <= LIFTED_MAX_QUBITS;
        EXPECT_EQ(std::string(e->what()).find("QubitLimit::Lift") != std::string::npos, liftable)
            << e->what();
    }
}

TEST(R1191Preflight, StatevectorCtorAcceptsValidCount) {
    EXPECT_NO_THROW(Statevector(1));
    EXPECT_NO_THROW(Statevector{ENFORCED_MAX_QUBITS});  // the documented maximum
}
