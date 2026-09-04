// 1.1.27.1 test wave - the six reachable validation policies (#110).
//
// ValidationOptions used to carry one enum answering two questions, so four of
// the six policies it could express were sayable and two were not. The two
// missing ones were the entire top-right of the table: repair, and warn when
// the repair is impossible; repair, and stay quiet when it is.
//
//                     Throw                 Warn                    Ignore
//   Repair::None      reject                report, proceed         proceed
//   Repair::Attempt   repair, else reject   repair, else report     repair, else proceed
//
// The shared probe in r1211_policy_probe.hpp now walks that grid at all
// nineteen entry points, which establishes that each one reaches the machinery
// with both knobs intact. This file asserts the things the probe cannot see,
// because they are about the EFFECT rather than about the contract: whether a
// repair actually ran, whether a measurement was taken at all, and which of the
// two knobs decided each outcome.
//
// The one combination that measures nothing is Ignore with Repair::None, and
// that is not a cost detail: it is the definition of plain ignore. Every other
// combination consumes the residual, Ignore with Repair::Attempt included,
// because a repair cannot know whether it is owed without measuring first. The
// difference is observable, and asserting it is what stops the cheap path from
// quietly becoming the only path.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "r1211_policy_probe.hpp"

#include "lindblad/detail/validate_physical.hpp"

#include <string>
#include <vector>

using namespace lindblad;
using r1211::WarningProbe;

namespace {

constexpr double kAtol = DEFAULT_PHYSICAL_ATOL;

// A Hadamard scaled off unitarity by an amount far outside atol and far inside
// what the polar projection recovers. The deviation is derived rather than
// observed: scaling U by (1+e) makes U†U = (1+e)²I, so the residual is
// 2e + e², which at e = 1e-6 is about 2e-6.
constexpr double kDrift = 1e-6;

std::vector<Complex128> drifted_hadamard() {
    const double h = INV_SQRT2 * (1.0 + kDrift);
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

std::vector<Complex128> exact_hadamard() {
    const double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

double deviation_of(const std::vector<Complex128>& m) {
    return detail::unitarity_deviation(m.data(), 2);
}

}  // namespace

// =============================================================================
// The defaults
// =============================================================================

TEST(V11271PolicyGrid, TheDefaultIsThrowWithNoRepair) {
    // Both halves matter. A default that repaired would rewrite operands nobody
    // asked it to touch; a default that did not throw would make every
    // unchecked call site a silent one.
    const ValidationOptions v;
    EXPECT_EQ(v.policy, Validation::Throw);
    EXPECT_EQ(v.repair, Repair::None);
    EXPECT_EQ(v.atol, DEFAULT_PHYSICAL_ATOL);
}

TEST(V11271PolicyGrid, TheRepairKnobIsTheThirdMember) {
    // The aggregate is written positionally as {policy, atol} at more than
    // eighty sites across the library, its tests and its docs. A knob inserted
    // ahead of atol would bind a tolerance to an enum at every one of them, so
    // the position is part of the contract rather than a formatting choice.
    const ValidationOptions v{Validation::Warn, 1e-9};
    EXPECT_EQ(v.policy, Validation::Warn);
    EXPECT_EQ(v.atol, 1e-9);
    EXPECT_EQ(v.repair, Repair::None)
        << "the two-member form must still name policy and atol";

    const ValidationOptions w{Validation::Warn, 1e-9, Repair::Attempt};
    EXPECT_EQ(w.atol, 1e-9);
    EXPECT_EQ(w.repair, Repair::Attempt);
}

// =============================================================================
// Which knob decides what
// =============================================================================

TEST(V11271PolicyGrid, TheRepairRunsUnderEveryResponse) {
    // The response governs an operand that is still invalid, and a repaired one
    // is not. So the repair happens under all three, and an implementation that
    // only repaired under Throw would look correct to any test that never asked
    // for a repair alongside Warn or Ignore.
    //
    // QuantumCircuit::unitary is the observable: it STORES the operand, so the
    // repaired matrix is still there afterwards to be measured.
    for (auto response : r1211::kResponses) {
        WarningProbe probe;
        QuantumCircuit qc(1);
        ASSERT_NO_THROW(qc.unitary(drifted_hadamard(), {0}, "u",
                                   {response, kAtol, Repair::Attempt}))
            << r1211::response_name(response);
        ASSERT_EQ(qc.instructions.size(), 1u);
        EXPECT_LE(detail::unitarity_deviation(qc.instructions[0].matrix.data(), 2),
                  kAtol)
            << "the stored matrix was not repaired under "
            << r1211::response_name(response);
    }
}

TEST(V11271PolicyGrid, NoRepairRunsUnderRepairNone) {
    // The mirror image, and the reason Repair::None has to be a command rather
    // than a description. Warn and Ignore both proceed, and what they proceed
    // with is the operand exactly as the caller wrote it.
    for (auto response : {Validation::Warn, Validation::Ignore}) {
        WarningProbe probe;
        QuantumCircuit qc(1);
        ASSERT_NO_THROW(qc.unitary(drifted_hadamard(), {0}, "u",
                                   {response, kAtol}))
            << r1211::response_name(response);
        ASSERT_EQ(qc.instructions.size(), 1u);
        EXPECT_NEAR(detail::unitarity_deviation(qc.instructions[0].matrix.data(), 2),
                    deviation_of(drifted_hadamard()), 1e-15)
            << "the operand was rewritten under Repair::None with "
            << r1211::response_name(response);
    }
}

TEST(V11271PolicyGrid, TheResponseDecidesOnlyWhatIsSaid) {
    // Same operand, same repair, three responses. The state of the world after
    // the call is identical in all three; only the reporting differs. That is
    // what makes them orthogonal knobs rather than one knob with six names.
    struct Expectation { Validation response; std::size_t warnings; };
    for (const Expectation e : {Expectation{Validation::Throw, 0u},
                                Expectation{Validation::Warn, 0u},
                                Expectation{Validation::Ignore, 0u}}) {
        WarningProbe probe;
        QuantumCircuit qc(1);
        ASSERT_NO_THROW(qc.unitary(drifted_hadamard(), {0}, "u",
                                   {e.response, kAtol, Repair::Attempt}));
        EXPECT_LE(detail::unitarity_deviation(qc.instructions[0].matrix.data(), 2),
                  kAtol);
        EXPECT_EQ(probe.count(), e.warnings)
            << "a repair that succeeded reported under "
            << r1211::response_name(e.response)
            << "; the response describes a FAILED repair, and this one worked";
    }
}

// =============================================================================
// Plain ignore does not look
// =============================================================================

TEST(V11271PolicyGrid, IgnoreWithNoRepairTakesNoMeasurement) {
    // The claim cannot be observed by timing, so it is observed by consequence.
    // A NaN operand is the sharpest probe available: the residual comparison is
    // written so that a NaN fails it, so ANY measurement of this matrix rejects
    // it. Passing silently therefore proves no measurement was taken.
    std::vector<Complex128> nan_matrix = exact_hadamard();
    nan_matrix[0] = Complex128(std::nan(""), 0.0);

    WarningProbe probe;
    Statevector sv(1);
    EXPECT_NO_THROW(
        gates::apply_unitary(sv, {0}, nan_matrix, {Validation::Ignore, kAtol}))
        << "a NaN operand was rejected under a policy that does not look";
    EXPECT_EQ(probe.count(), 0u);
}

TEST(V11271PolicyGrid, IgnoreWithARepairDoesLook) {
    // The other half, and the reason the two are not the same policy. Asking
    // for a repair means the residual has to be taken, because a repair cannot
    // know whether it is owed otherwise. So the drift is found and corrected,
    // while Ignore with no repair leaves it alone.
    QuantumCircuit looked(1);
    looked.unitary(drifted_hadamard(), {0}, "u",
                   {Validation::Ignore, kAtol, Repair::Attempt});
    EXPECT_LE(detail::unitarity_deviation(looked.instructions[0].matrix.data(), 2),
              kAtol)
        << "Ignore with Repair::Attempt did not measure, so it could not repair";

    QuantumCircuit did_not(1);
    did_not.unitary(drifted_hadamard(), {0}, "u", {Validation::Ignore, kAtol});
    EXPECT_GT(detail::unitarity_deviation(did_not.instructions[0].matrix.data(), 2),
              kAtol)
        << "Ignore with Repair::None repaired something it never measured";
}

TEST(V11271PolicyGrid, IgnoreWithARepairStaysSilent) {
    // Looking is not reporting. The measurement is taken for the repair's
    // benefit, and nothing about that reaches the caller.
    WarningProbe probe;
    QuantumCircuit qc(1);
    qc.unitary(drifted_hadamard(), {0}, "u",
               {Validation::Ignore, kAtol, Repair::Attempt});
    EXPECT_EQ(probe.count(), 0u)
        << "Ignore reported; it is the response that says nothing";
}

// =============================================================================
// A valid operand owes nothing to either knob
// =============================================================================

TEST(V11271PolicyGrid, AnOperandInsideToleranceIsNeverRewritten) {
    // An implementation that projected unconditionally rather than on a
    // measured violation would pass every test that hands it a broken operand.
    // This is the one that catches it: the exact Hadamard must come back
    // bit-for-bit, because the projection is not a no-op even on a unitary
    // input, it is an SVD round trip.
    const auto exact = exact_hadamard();
    for (const auto& v : r1211::kAllPolicies) {
        WarningProbe probe;
        QuantumCircuit qc(1);
        ASSERT_NO_THROW(qc.unitary(exact, {0}, "u", v)) << r1211::policy_name(v);
        ASSERT_EQ(qc.instructions.size(), 1u);
        const auto& stored = qc.instructions[0].matrix;
        ASSERT_EQ(stored.size(), exact.size());
        for (std::size_t i = 0; i < exact.size(); ++i) {
            EXPECT_EQ(stored[i].real, exact[i].real)
                << r1211::policy_name(v) << ", entry " << i;
            EXPECT_EQ(stored[i].imag, exact[i].imag)
                << r1211::policy_name(v) << ", entry " << i;
        }
        EXPECT_EQ(probe.count(), 0u) << r1211::policy_name(v);
    }
}

// =============================================================================
// The tolerance is consulted under both knobs
// =============================================================================

TEST(V11271PolicyGrid, AtolDecidesWhetherARepairIsOwed) {
    // atol is not only the reject threshold. It is also what tells a repair
    // that there is nothing to do, so a tolerance wide enough to accept the
    // drift must leave the operand alone even under Repair::Attempt.
    const double wide = deviation_of(drifted_hadamard()) * 10.0;
    const double tight = deviation_of(drifted_hadamard()) / 10.0;

    QuantumCircuit accepted(1);
    accepted.unitary(drifted_hadamard(), {0}, "u",
                     {Validation::Throw, wide, Repair::Attempt});
    EXPECT_NEAR(detail::unitarity_deviation(accepted.instructions[0].matrix.data(), 2),
                deviation_of(drifted_hadamard()), 1e-15)
        << "a repair ran on an operand the caller's tolerance accepts";

    QuantumCircuit repaired(1);
    repaired.unitary(drifted_hadamard(), {0}, "u",
                     {Validation::Throw, tight, Repair::Attempt});
    EXPECT_LE(detail::unitarity_deviation(repaired.instructions[0].matrix.data(), 2),
              tight)
        << "a repair did not run on an operand the caller's tolerance rejects";
}

TEST(V11271PolicyGrid, AtolIsHonouredUnderEveryResponse) {
    const double wide = deviation_of(drifted_hadamard()) * 10.0;
    for (const auto response : r1211::kResponses) {
        WarningProbe probe;
        Statevector sv(1);
        EXPECT_NO_THROW(gates::apply_unitary(sv, {0}, drifted_hadamard(),
                                             {response, wide}))
            << r1211::response_name(response)
            << " rejected a residual inside the tolerance it was given";
        EXPECT_EQ(probe.count(), 0u)
            << r1211::response_name(response)
            << " reported a residual inside the tolerance it was given";
    }
}

// =============================================================================
// The borrowing note
// =============================================================================

TEST(V11271PolicyGrid, ABorrowingRepairSaysTheCostRepeats) {
    // A primitive takes its operand by const reference, projects a copy,
    // applies it and lets it go, so the caller still holds the matrix it passed
    // and a loop pays a projection every iteration. The note names the storing
    // route as the way to pay once, and it is emitted for the repair rather
    // than for the violation, so it must appear under every response.
    for (const auto response : r1211::kResponses) {
        WarningProbe probe;
        Statevector sv(1);
        ASSERT_NO_THROW(gates::apply_unitary(sv, {0}, drifted_hadamard(),
                                             {response, kAtol, Repair::Attempt}))
            << r1211::response_name(response);
        EXPECT_GE(probe.count(), 1u)
            << "no note under " << r1211::response_name(response);
        EXPECT_TRUE(probe.any_contains("Repair::Attempt"))
            << "the note does not name the knob that caused it";
        EXPECT_TRUE(probe.any_contains("unchanged"))
            << "the note must say the caller's matrix was not modified, since "
               "that is why the cost repeats";
    }
}

TEST(V11271PolicyGrid, TheStoringRouteRepairsWithoutTheNote) {
    // The inverse, and the reason the note lives at the borrowing entry points
    // rather than inside the projection they share. QuantumCircuit::unitary
    // keeps the repaired matrix, so the cost is paid once and there is nothing
    // to report.
    WarningProbe probe;
    QuantumCircuit qc(1);
    ASSERT_NO_THROW(qc.unitary(drifted_hadamard(), {0}, "u",
                               {Validation::Throw, kAtol, Repair::Attempt}));
    EXPECT_EQ(probe.count(), 0u)
        << "the storing route reported a repeated cost it does not pay";
}
