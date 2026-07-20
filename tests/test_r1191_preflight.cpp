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
// The two run() reporting styles differ and are pinned here:
//   - StatevectorSimulator / DensityMatrixSimulator catch inside run() and
//     report through Result{success=false, error_message}.
//   - MPSSimulator / CliffordSimulator have no Result error channel and
//     surface the failure by throwing.
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
    EXPECT_THROW(bad.validate_operands(), std::out_of_range);
}

TEST(R1191Preflight, ValidateOperandsPassesOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    EXPECT_NO_THROW(ok.validate_operands());
}

// =============================================================================
// Statevector / DensityMatrix: report through Result, do not throw out of run()
// =============================================================================

TEST(R1191Preflight, StatevectorRunReportsBadIndexViaResult) {
    QuantumCircuit bad = make_bad_index_circuit();
    StatevectorSimulator sim;
    auto result = sim.run(bad);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(R1191Preflight, StatevectorRunSucceedsOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    StatevectorSimulator sim;
    auto result = sim.run(ok);
    EXPECT_TRUE(result.success);
}

TEST(R1191Preflight, DensityMatrixRunReportsBadIndexViaResult) {
    QuantumCircuit bad = make_bad_index_circuit();
    NoiseModel noise;  // empty: exercises the noiseless path, pre-flight still runs
    DensityMatrixSimulator sim;
    auto result = sim.run(bad, noise);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(R1191Preflight, DensityMatrixRunSucceedsOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    NoiseModel noise;
    DensityMatrixSimulator sim;
    auto result = sim.run(ok, noise);
    EXPECT_TRUE(result.success);
}

// =============================================================================
// MPS / Clifford: no Result error channel, so run() throws
// =============================================================================

TEST(R1191Preflight, MpsRunThrowsOnBadIndex) {
    QuantumCircuit bad = make_bad_index_circuit();
    MPSSimulator sim;
    EXPECT_THROW(sim.run(bad), std::out_of_range);
}

TEST(R1191Preflight, MpsRunSucceedsOnValidCircuit) {
    QuantumCircuit ok = make_valid_circuit();
    MPSSimulator sim;
    EXPECT_NO_THROW(sim.run(ok));
}

TEST(R1191Preflight, CliffordRunThrowsOnBadIndex) {
    QuantumCircuit bad = make_bad_index_circuit();  // X is Clifford; the index is not
    CliffordSimulator sim;
    EXPECT_THROW(sim.run(bad, /*shots=*/16), std::out_of_range);
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
    EXPECT_THROW(Statevector(0), std::invalid_argument);
    EXPECT_THROW(Statevector(-1), std::invalid_argument);
    EXPECT_THROW(Statevector(-1000), std::invalid_argument);
    EXPECT_THROW(Statevector(31), std::invalid_argument);
    EXPECT_THROW(Statevector(64), std::invalid_argument);
}

TEST(R1191Preflight, StatevectorCtorAcceptsValidCount) {
    EXPECT_NO_THROW(Statevector(1));
    EXPECT_NO_THROW(Statevector(30));  // the documented maximum
}
