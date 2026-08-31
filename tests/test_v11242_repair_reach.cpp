// test_v11242_repair_reach.cpp - Validation::Fix, once it reaches everywhere.
//
// The previous release shipped twenty-four failing tests asserting that Fix
// repairs unitarity wherever it is asked to, and that a circuit rejects a
// wrong-sized matrix as its neighbouring builders do. Those close here. What
// they do not cover is everything the fix had to add in order to close them,
// which is what this file is for.
//
// Three things, and each is a place where a plausible implementation is wrong
// in a way the twenty-four cannot see.
//
// THE COPY. A borrowing entry point cannot repair in place: its operand is a
// const reference the caller still owns. So it repairs a copy, and the caller's
// matrix must come back untouched. The pre-flight has the same problem one level
// up, against a whole circuit, and there the copy shares every gate matrix with
// the original through copy-on-write, so a repair written through the copy would
// reach the caller's circuit anyway.
//
// THE NOTE. A repair that does not persist is worth reporting once, because a
// caller looping over a bent matrix pays a projection every iteration and has no
// other way to find out. It has to fire from the paths where the repair
// evaporates and stay silent on the one where it sticks, which is the exact
// inverse of what a naive placement inside the projection would do.
//
// THE PRE-FLIGHT ACTUALLY APPLYING. A backend reads each instruction's policy
// once and then runs every gate under Ignore, so a repair that is measured but
// not carried into what executes would leave the run using the bent matrix while
// reporting success. That is the failure mode with no symptom, and the only way
// to see it is to check the state the run produced.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

using lindblad::Complex128;
using lindblad::MPSSimulator;
using lindblad::QuantumCircuit;
using lindblad::Statevector;
using lindblad::StatevectorSimulator;
using lindblad::Validation;
using lindblad::ValidationOptions;
using lindblad::detail::unitarity_deviation;
using r1211::WarningProbe;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kAtol = lindblad::DEFAULT_PHYSICAL_ATOL;
const ValidationOptions kFix{Validation::Fix, kAtol};

// a times the identity. Its thin SVD is I (a I) I-dagger, so the nearest
// unitary is EXACTLY the identity whatever a is. Every expectation below is
// therefore an exact value rather than a tolerance on a rotated state: a
// correctly repaired gate is the identity and leaves the register alone, and a
// gate applied unrepaired scales an amplitude by a.
std::vector<Complex128> scaled_identity(int n, double a) {
    std::vector<Complex128> m(static_cast<size_t>(n) * n, Complex128(0.0, 0.0));
    for (int i = 0; i < n; ++i)
        m[static_cast<size_t>(i) * n + i] = Complex128(a, 0.0);
    return m;
}

std::vector<Complex128> bent_1q() { return scaled_identity(2, 1.5); }

// An instruction whose matrix never passed through the repairing constructor,
// which is what QASM import, compose, and appending to the public vector all
// produce. Stored under Ignore, then given the policy it should have carried.
QuantumCircuit circuit_with_unrepaired_gate(int n_qubits, int target) {
    QuantumCircuit qc(n_qubits);
    qc.unitary(bent_1q(), {target}, "unrepaired",
               {Validation::Ignore, kAtol});
    qc.instructions[0].validation = kFix;
    return qc;
}

std::vector<Complex128> amplitudes(const Statevector& sv) {
    std::vector<Complex128> a(sv.dim);
    for (size_t i = 0; i < sv.dim; ++i)
        a[i] = Complex128(sv.real_parts[i], sv.imag_parts[i]);
    return a;
}

// |0...0> is unchanged by the identity, so an amplitude of 1 at index 0 and 0
// everywhere else is what a correctly repaired run must produce. An unrepaired
// one scales index 0 to 1.5.
void expect_ground_state(const std::vector<Complex128>& a, const char* what) {
    ASSERT_FALSE(a.empty()) << what;
    EXPECT_NEAR(a[0].real, 1.0, 64.0 * kEps)
        << what << ": amplitude 0 is " << a[0].real
        << ", which is what applying the operand UNREPAIRED would give";
    EXPECT_NEAR(a[0].imag, 0.0, 64.0 * kEps) << what;
    for (size_t i = 1; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, 0.0, 64.0 * kEps) << what << " at " << i;
        EXPECT_NEAR(a[i].imag, 0.0, 64.0 * kEps) << what << " at " << i;
    }
}

}  // namespace

// =============================================================================
// The copy: a borrowed operand comes back untouched
// =============================================================================

TEST(V11242RepairReach, ABorrowingKernelLeavesTheCallersMatrixAlone) {
    // The repair goes into storage the entry point owns. A caller who handed
    // over a const reference must find their own vector exactly as they left
    // it, whatever the policy did with it.
    auto mine = bent_1q();
    const auto before = mine;

    Statevector sv(2);
    ASSERT_NO_THROW(lindblad::gates::apply_unitary(sv, {0}, mine, kFix));

    for (size_t i = 0; i < mine.size(); ++i) {
        EXPECT_EQ(mine[i].real, before[i].real) << "entry " << i << " real";
        EXPECT_EQ(mine[i].imag, before[i].imag) << "entry " << i << " imag";
    }
    EXPECT_GT(unitarity_deviation(mine.data(), 2), kAtol)
        << "the caller's matrix was repaired behind its back";
}

TEST(V11242RepairReach, TheRepairedOperandIsWhatActuallyGetsApplied) {
    // Repairing and then applying the original would leave every policy test
    // green while the arithmetic used the bent matrix. The fixture makes the
    // difference visible: repaired is the identity, unrepaired scales by 1.5.
    Statevector sv(2);
    ASSERT_NO_THROW(lindblad::gates::apply_unitary(sv, {0}, bent_1q(), kFix));
    expect_ground_state(amplitudes(sv), "statevector kernel under Fix");
}

TEST(V11242RepairReach, NothingIsCopiedWhenTheOperandIsAlreadyUnitary) {
    // The storage the entry point passes down stays empty on the common path.
    // Asserted through the observable proxy: an operand inside tolerance is
    // applied unchanged and reports nothing, so no projection ran.
    WarningProbe probe;
    std::vector<Complex128> good{Complex128(1.0, 0.0), Complex128(0.0, 0.0),
                                 Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
    Statevector sv(2);
    ASSERT_NO_THROW(lindblad::gates::apply_unitary(sv, {0}, good, kFix));
    expect_ground_state(amplitudes(sv), "identity under Fix");
    EXPECT_EQ(probe.count(), 0u)
        << "a matrix already inside tolerance was repaired, which costs a "
           "factorisation and changes nothing";
}

// =============================================================================
// The note
// =============================================================================

TEST(V11242RepairReach, ABorrowingRepairReportsItselfExactlyOnce) {
    WarningProbe probe;

    Statevector a(2), b(2);
    lindblad::gates::apply_unitary(a, {0}, bent_1q(), kFix);
    lindblad::gates::apply_unitary(b, {0}, bent_1q(), kFix);

    EXPECT_EQ(probe.count(), 1u)
        << "two repairs reported " << probe.count()
        << " times; the note is once per accounting, not once per call";
    EXPECT_TRUE(probe.any_contains("Validation::Fix repaired"))
        << "the note does not say what happened";
    EXPECT_TRUE(probe.any_contains("unchanged"))
        << "the note must say the caller's matrix was not modified, since that "
           "is the reason the cost repeats";
    EXPECT_TRUE(probe.any_contains("QuantumCircuit::unitary"))
        << "the note must name the way to pay the cost once instead";
}

TEST(V11242RepairReach, TheStoringPathRepairsSilently) {
    // The inverse half, and the one a note placed inside the projection would
    // get wrong. QuantumCircuit::unitary keeps the repaired matrix, so the cost
    // is paid once when the instruction is built and there is nothing to warn
    // about.
    WarningProbe probe;
    QuantumCircuit qc(2);
    ASSERT_NO_THROW(qc.unitary(bent_1q(), {0}, "stored", kFix));
    ASSERT_EQ(qc.instructions.size(), 1u);
    EXPECT_LE(unitarity_deviation(qc.instructions[0].matrix.data(), 2), kAtol);
    EXPECT_EQ(probe.count(), 0u)
        << "the storing path reported a repeated cost it does not pay";
}

TEST(V11242RepairReach, NothingIsReportedWhenNoRepairRuns) {
    for (auto policy : {Validation::Throw, Validation::Warn, Validation::Ignore,
                        Validation::Fix}) {
        WarningProbe probe;
        std::vector<Complex128> good{Complex128(1.0, 0.0), Complex128(0.0, 0.0),
                                     Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
        Statevector sv(2);
        EXPECT_NO_THROW(
            lindblad::gates::apply_unitary(sv, {0}, good, {policy, kAtol}));
        EXPECT_EQ(probe.count(), 0u)
            << "policy " << static_cast<int>(policy)
            << " reported something about an operand that was already unitary";
    }
}

// =============================================================================
// The pre-flight
// =============================================================================

TEST(V11242PreFlight, ReturnsNothingWhenNoRepairIsOwed) {
    // The copy is the expensive part, so it must not be taken speculatively. A
    // circuit needing no repair returns an empty optional, which is how a
    // backend knows to execute the caller's own circuit.
    QuantumCircuit clean(2);
    clean.h(0).cx(0, 1);
    EXPECT_FALSE(clean.validated_physical().has_value())
        << "a circuit with no caller-supplied matrix produced a copy";

    QuantumCircuit stored(2);
    stored.unitary(bent_1q(), {0}, "already-repaired", kFix);
    EXPECT_FALSE(stored.validated_physical().has_value())
        << "an instruction repaired when it was built was repaired again";
}

TEST(V11242PreFlight, RepairsAnInstructionThatArrivedUnrepaired) {
    QuantumCircuit qc = circuit_with_unrepaired_gate(2, 0);
    const auto before = qc.instructions[0].matrix;

    const auto repaired = qc.validated_physical();
    ASSERT_TRUE(repaired.has_value())
        << "an operand outside tolerance under Fix produced no repair";
    ASSERT_EQ(repaired->instructions.size(), 1u);
    EXPECT_LE(unitarity_deviation(repaired->instructions[0].matrix.data(), 2),
              kAtol);

    // The caller's circuit is untouched. This is the copy-on-write trap: the
    // returned circuit was copied from this one and shares its buffers, so a
    // repair written in place would have reached here too.
    ASSERT_EQ(qc.instructions[0].matrix.size(), before.size());
    for (size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(qc.instructions[0].matrix[i].real, before[i].real) << i;
        EXPECT_EQ(qc.instructions[0].matrix[i].imag, before[i].imag) << i;
    }
    EXPECT_GT(unitarity_deviation(qc.instructions[0].matrix.data(), 2), kAtol)
        << "the caller's circuit was repaired in place";
}

TEST(V11242PreFlight, RepairsEveryInstructionThatNeedsItAndLeavesTheRestAlone) {
    // Stopping at the first repair would leave later gates bent while reporting
    // success, so the loop has to continue rather than return on the first hit.
    QuantumCircuit qc(3);
    qc.unitary(bent_1q(), {0}, "bent-a", {Validation::Ignore, kAtol});
    qc.h(1);
    qc.unitary(bent_1q(), {2}, "bent-b", {Validation::Ignore, kAtol});
    qc.instructions[0].validation = kFix;
    qc.instructions[2].validation = kFix;

    const auto repaired = qc.validated_physical();
    ASSERT_TRUE(repaired.has_value());
    ASSERT_EQ(repaired->instructions.size(), 3u);
    EXPECT_LE(unitarity_deviation(repaired->instructions[0].matrix.data(), 2),
              kAtol) << "the first repairable instruction";
    EXPECT_LE(unitarity_deviation(repaired->instructions[2].matrix.data(), 2),
              kAtol) << "the second repairable instruction";
    EXPECT_EQ(repaired->instructions[1].type, qc.instructions[1].type)
        << "an instruction needing no repair was altered";
}

TEST(V11242PreFlight, HonoursTheOtherThreePoliciesExactlyAsTheConstFormDoes) {
    {
        QuantumCircuit qc = circuit_with_unrepaired_gate(2, 0);
        qc.instructions[0].validation = {Validation::Throw, kAtol};
        EXPECT_THROW(qc.validated_physical(), std::invalid_argument);
        EXPECT_THROW(qc.validate_physical(), std::invalid_argument);
    }
    {
        QuantumCircuit qc = circuit_with_unrepaired_gate(2, 0);
        qc.instructions[0].validation = {Validation::Warn, kAtol};
        WarningProbe probe;
        std::optional<QuantumCircuit> out;
        EXPECT_NO_THROW(out = qc.validated_physical());
        EXPECT_FALSE(out.has_value())
            << "Warn repaired; Warn describes and does not repair";
        EXPECT_GE(probe.count(), 1u) << "Warn proceeded without reporting";
    }
    {
        QuantumCircuit qc = circuit_with_unrepaired_gate(2, 0);
        qc.instructions[0].validation = {Validation::Ignore, kAtol};
        WarningProbe probe;
        std::optional<QuantumCircuit> out;
        EXPECT_NO_THROW(out = qc.validated_physical());
        EXPECT_FALSE(out.has_value()) << "Ignore repaired the operand it was "
                                         "told not to look at";
        EXPECT_EQ(probe.count(), 0u) << "Ignore reported something";
    }
}

// =============================================================================
// The pre-flight, end to end through a backend
// =============================================================================
// A backend reads the policy once here and then applies every gate under
// Ignore, so a repair measured but not carried into what executes leaves the run
// using the bent matrix and reporting success. Only the produced state shows it.

TEST(V11242PreFlight, TheStatevectorRunAppliesTheRepairedMatrix) {
    QuantumCircuit qc = circuit_with_unrepaired_gate(2, 0);
    StatevectorSimulator sim;
    const auto res = sim.run(qc, /*shots=*/0);
    ASSERT_TRUE(res.success) << res.error_message;
    expect_ground_state(amplitudes(res.final_state), "statevector run");
}

TEST(V11242PreFlight, TheMpsRunAppliesTheRepairedMatrix) {
    QuantumCircuit qc = circuit_with_unrepaired_gate(2, 0);
    MPSSimulator sim;
    auto res = sim.run(qc, /*max_bond_dim=*/8, /*shots=*/0, /*seed=*/1);
    const Statevector sv = res.final_state.to_statevector();
    expect_ground_state(amplitudes(sv), "MPS run");
}

TEST(V11242PreFlight, ARunLeavesTheCallersCircuitUnrepaired) {
    // The whole reason the pre-flight returns a copy rather than mutating. A
    // caller inspecting their circuit after a run must find what they built.
    QuantumCircuit qc = circuit_with_unrepaired_gate(2, 0);
    const auto before = qc.instructions[0].matrix;

    StatevectorSimulator sim;
    ASSERT_TRUE(sim.run(qc, /*shots=*/0).success);

    ASSERT_EQ(qc.instructions[0].matrix.size(), before.size());
    for (size_t i = 0; i < before.size(); ++i)
        EXPECT_EQ(qc.instructions[0].matrix[i].real, before[i].real) << i;
    EXPECT_GT(unitarity_deviation(qc.instructions[0].matrix.data(), 2), kAtol)
        << "run() rewrote the caller's circuit";
}

TEST(V11242PreFlight, RunningTwiceGivesTheSameAnswer) {
    // The repair is per-run, so a second run repeats it. What must not happen is
    // the two runs disagreeing, which is what consuming a partially repaired
    // circuit would produce.
    QuantumCircuit qc = circuit_with_unrepaired_gate(3, 1);
    StatevectorSimulator sim;
    const auto first = sim.run(qc, /*shots=*/0);
    const auto second = sim.run(qc, /*shots=*/0);
    ASSERT_TRUE(first.success);
    ASSERT_TRUE(second.success);

    const auto a = amplitudes(first.final_state);
    const auto b = amplitudes(second.final_state);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].real, b[i].real) << "amplitude " << i;
        EXPECT_EQ(a[i].imag, b[i].imag) << "amplitude " << i;
    }
    expect_ground_state(a, "first run");
}

// =============================================================================
// The builder's size check
// =============================================================================

TEST(V11242Builder, EveryBuilderRejectsAMalformedOperandWhereItIsHandedOver) {
    // The convention #97 brought `unitary` into line with. Stated together so
    // the next builder added has something to copy.
    QuantumCircuit qc(3);

    EXPECT_THROW(qc.unitary(std::vector<Complex128>(5, Complex128(1.0, 0.0)),
                            {0}, "bad-shape"),
                 std::invalid_argument)
        << "a 1-qubit gate needs four entries";
    EXPECT_THROW(qc.permute({0, 1, 2}, {0}, "bad-perm"), std::invalid_argument)
        << "a 1-qubit permutation needs two images";
    EXPECT_THROW(qc.permute({1, 1}, {0}, "not-a-bijection"),
                 std::invalid_argument);
    EXPECT_THROW(qc.mcx({0}, 0), std::invalid_argument)
        << "a control equal to its target";
    EXPECT_THROW(qc.mcp(0.5, {}), std::invalid_argument)
        << "an empty qubit list";

    EXPECT_TRUE(qc.instructions.empty())
        << "a rejected operand was appended anyway, so a caller catching the "
           "exception still holds a circuit containing it";
}

TEST(V11242Builder, TheSizeCheckRunsBeforeTheUnitarityCheck) {
    // Order matters rather than being incidental: unitarity is read as
    // U-dagger U over a rows x rows operand, so measuring a wrong-sized buffer
    // reads past its end. The diagnostic must name the shape, not the physics.
    QuantumCircuit qc(2);
    try {
        qc.unitary(std::vector<Complex128>(5, Complex128(1.0, 0.0)), {0},
                   "bad-shape", kFix);
        FAIL() << "a wrong-sized operand was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("entries"), std::string::npos)
            << "the diagnostic does not describe the shape. Got: " << msg;
        // Searching for the property's own phrasing rather than for the word
        // `unitary`, which every message from this entry point carries as its
        // context: the gate is named `unitary`, so a shape error legitimately
        // begins with it.
        EXPECT_EQ(msg.find("is not unitary"), std::string::npos)
            << "a shape error was reported as a physics error. Got: " << msg;
    }
}
