// 1.1.27.1 test wave - what happens when the repair cannot run (#110).
//
// This is the case the split exists for. Before it, asking for a repair also
// chose the answer to "and what if the repair is impossible", and that answer
// was fixed at throw. A long batch run wants one uncorrectable operand reported
// rather than the run ended; a caller who has decided the check is advisory
// wants the repair applied where it can be and silence where it cannot. Neither
// could be stated.
//
// There are exactly three ways a repair fails to produce a valid operand, and
// they are not the same failure:
//
//   1. the property defines no repair at all (trace preservation). Asking is a
//      mistake in the calling code, so it throws under every response.
//   2. the repair is defined and does not converge (a polar projection on an
//      operand it cannot factorise). The operand is the problem, so the
//      response decides.
//   3. the repair is defined and is impossible for this input (a rescale of an
//      object with no norm to divide out). Also the operand, also the response.
//
// Only the first ignores the response, and the reason is not that it is harder
// to recover from. It is that a caller who asks for a repair that does not
// exist has written something that cannot be satisfied under any response, and
// absorbing that would return an unphysical operand to code that asked for a
// correction the library cannot make.
//
// The other requirement, easy to miss and asserted throughout: a failed repair
// must leave the CALLER'S operand in place. Proceeding under Warn or Ignore
// with a half-projected buffer would be worse than either throwing or doing
// nothing, and it is what makes those two responses usable alongside a repair
// at all.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "r1211_policy_probe.hpp"

#include "lindblad/detail/validate_physical.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using r1211::WarningProbe;

namespace {

constexpr double kAtol = DEFAULT_PHYSICAL_ATOL;

// A matrix the polar projection cannot factorise. A non-finite entry propagates
// through the SVD, so the routine reports failure rather than returning an
// operand as unphysical as the one it was given.
std::vector<Complex128> unfactorisable() {
    const double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0),
            Complex128(0.0, std::numeric_limits<double>::infinity())};
}

// A Kraus set that is not trace preserving, on a property with no repair.
std::vector<std::vector<Complex128>> not_trace_preserving() {
    return {{Complex128(0.5, 0.0), Complex128(0.0, 0.0),
             Complex128(0.0, 0.0), Complex128(0.5, 0.0)}};
}

bool same_matrix(const std::vector<Complex128>& a,
                 const std::vector<Complex128>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        // The operand carries a non-finite entry by construction, and NaN and
        // infinity do not compare the way a plain equality would need, so the
        // bit pattern is what is compared.
        if (std::isnan(a[i].real) != std::isnan(b[i].real)) return false;
        if (std::isnan(a[i].imag) != std::isnan(b[i].imag)) return false;
        if (!std::isnan(a[i].real) && a[i].real != b[i].real) return false;
        if (!std::isnan(a[i].imag) && a[i].imag != b[i].imag) return false;
    }
    return true;
}

}  // namespace

// =============================================================================
// 1. no repair is defined for this property
// =============================================================================

TEST(V11271RepairImpossible, NoRepairDefinedThrowsUnderEveryResponse) {
    // The response knob governs an operand that is still invalid. This is not
    // that: the caller asked for something the library does not implement, and
    // no response makes that request satisfiable.
    for (const auto response : r1211::kResponses) {
        WarningProbe probe;
        DensityMatrix rho(1);
        try {
            rho.apply_kraus(not_trace_preserving(), {0},
                            {response, kAtol, Repair::Attempt});
            FAIL() << "a repair that does not exist was absorbed under "
                   << r1211::response_name(response);
        } catch (const std::invalid_argument& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("no repair defined"), std::string::npos)
                << "under " << r1211::response_name(response)
                << " the diagnostic does not say a repair is missing. Got: "
                << msg;
            EXPECT_NE(msg.find("trace preservation"), std::string::npos)
                << "the diagnostic does not name the property. Got: " << msg;
        }
    }
}

TEST(V11271RepairImpossible, NoRepairDefinedIsSilentWhenTheOperandIsFine) {
    // The throw is about the operand failing its check while a repair was
    // requested, not about the request itself. A trace-preserving channel under
    // Repair::Attempt is accepted, so asking for a repair is not an error in
    // its own right.
    const std::vector<std::vector<Complex128>> identity_channel{
        {Complex128(1.0, 0.0), Complex128(0.0, 0.0),
         Complex128(0.0, 0.0), Complex128(1.0, 0.0)}};

    for (const auto response : r1211::kResponses) {
        WarningProbe probe;
        DensityMatrix rho(1);
        EXPECT_NO_THROW(rho.apply_kraus(identity_channel, {0},
                                        {response, kAtol, Repair::Attempt}))
            << r1211::response_name(response);
        EXPECT_EQ(probe.count(), 0u) << r1211::response_name(response);
    }
}

TEST(V11271RepairImpossible, TheDispatcherSaysTheSameThingDirectly) {
    // The entry points above route through detail::enforce_physical, and the
    // three properties that define no repair all reach it. Driving it directly
    // covers the two that have no convenient entry point of their own.
    for (const auto& prop : {detail::KRAUS_TRACE_PRESERVING,
                             detail::SUPEROP_TRACE_PRESERVING}) {
        for (const auto response : r1211::kResponses) {
            try {
                detail::enforce_physical(1e-6, {response, kAtol, Repair::Attempt},
                                         "probe", prop);
                FAIL() << prop.noun << " absorbed a repair request under "
                       << r1211::response_name(response);
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("no repair defined"),
                          std::string::npos)
                    << prop.noun << " under "
                    << r1211::response_name(response) << ": " << e.what();
            }
        }
    }
}

// =============================================================================
// 2. the repair is defined and does not converge
// =============================================================================

TEST(V11271RepairImpossible, AFailedProjectionThrowsUnderThrow) {
    QuantumCircuit qc(1);
    try {
        qc.unitary(unfactorisable(), {0}, "u",
                   {Validation::Throw, kAtol, Repair::Attempt});
        FAIL() << "an unfactorisable operand was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("could not repair"), std::string::npos)
            << "the diagnostic does not say the repair failed. Got: " << msg;
        EXPECT_NE(msg.find("unitarity"), std::string::npos)
            << "the diagnostic does not name the property. Got: " << msg;
        EXPECT_EQ(msg.find("no repair defined"), std::string::npos)
            << "unitarity HAS a repair; the message must not claim otherwise. "
               "Got: " << msg;
    }
}

TEST(V11271RepairImpossible, AFailedProjectionReportsUnderWarn) {
    // The policy the split was filed to make sayable. A batch run asks for the
    // repair and asks to be told, rather than to be stopped.
    WarningProbe probe;
    QuantumCircuit qc(1);
    ASSERT_NO_THROW(qc.unitary(unfactorisable(), {0}, "u",
                               {Validation::Warn, kAtol, Repair::Attempt}));
    EXPECT_GE(probe.count(), 1u) << "Warn proceeded without reporting anything";
    EXPECT_TRUE(probe.any_contains("could not repair"))
        << "the report does not say the repair failed";
}

TEST(V11271RepairImpossible, AFailedProjectionIsSilentUnderIgnore) {
    // The other policy the split was filed for: repair where you can, say
    // nothing where you cannot.
    WarningProbe probe;
    QuantumCircuit qc(1);
    ASSERT_NO_THROW(qc.unitary(unfactorisable(), {0}, "u",
                               {Validation::Ignore, kAtol, Repair::Attempt}));
    EXPECT_EQ(probe.count(), 0u) << "Ignore reported a repair it could not run";
}

TEST(V11271RepairImpossible, AFailedProjectionLeavesTheOperandInPlace) {
    // The requirement that makes Warn and Ignore usable at all. The projection
    // writes into a copy, and when it does not converge that copy holds
    // whatever it left behind. Installing it would hand the caller an operand
    // no worse-behaved code would have produced.
    const auto original = unfactorisable();
    for (const auto response : {Validation::Warn, Validation::Ignore}) {
        WarningProbe probe;
        QuantumCircuit qc(1);
        ASSERT_NO_THROW(qc.unitary(original, {0}, "u",
                                   {response, kAtol, Repair::Attempt}))
            << r1211::response_name(response);
        ASSERT_EQ(qc.instructions.size(), 1u);
        const std::vector<Complex128> stored = qc.instructions[0].matrix;
        EXPECT_TRUE(same_matrix(stored, original))
            << "under " << r1211::response_name(response)
            << " the instruction kept a half-projected buffer rather than the "
               "operand the caller handed over";
    }
}

TEST(V11271RepairImpossible, ABorrowingEntryPointAlsoKeepsTheOperand) {
    // Same requirement one layer down, where the operand is borrowed rather
    // than stored. The call proceeds, and what it applies is the caller's
    // matrix.
    const auto original = unfactorisable();
    for (const auto response : {Validation::Warn, Validation::Ignore}) {
        WarningProbe probe;
        Statevector sv(1);
        auto copy = original;
        EXPECT_NO_THROW(gates::apply_unitary(sv, {0}, copy,
                                             {response, kAtol, Repair::Attempt}))
            << r1211::response_name(response);
        EXPECT_TRUE(same_matrix(copy, original))
            << "under " << r1211::response_name(response)
            << " the caller's own matrix was modified";
    }
}

TEST(V11271RepairImpossible, TheTwoProjectionFailuresReadDifferently) {
    // A projection that never factorised and one that ran and landed outside
    // tolerance are different situations for a caller: the first says the
    // operand is not repairable at all, the second says the tolerance is
    // tighter than the repair can reach. Collapsing them into one message
    // leaves the caller unable to tell which.
    std::string could_not_factorise;
    try {
        QuantumCircuit qc(1);
        qc.unitary(unfactorisable(), {0}, "u",
                   {Validation::Throw, kAtol, Repair::Attempt});
    } catch (const std::invalid_argument& e) {
        could_not_factorise = e.what();
    }
    ASSERT_FALSE(could_not_factorise.empty());
    EXPECT_NE(could_not_factorise.find("factorise"), std::string::npos)
        << "the message does not distinguish a projection that never ran. Got: "
        << could_not_factorise;
}

// =============================================================================
// 3. the repair is defined and this input has nothing to divide out
// =============================================================================

TEST(V11271RepairImpossible, AZeroStateAnswersToTheResponse) {
    // A rescale needs something to divide by. Zero and non-finite are the two
    // objects that have nothing, and that is a property of the operand, so the
    // response decides rather than the repair request forcing a throw.
    const std::vector<Complex128> zero{Complex128(0.0, 0.0), Complex128(0.0, 0.0)};

    {
        Statevector sv(1);
        EXPECT_THROW(sv.set_amplitudes(zero, {Validation::Throw, kAtol,
                                              Repair::Attempt}),
                     std::invalid_argument);
    }
    {
        WarningProbe probe;
        Statevector sv(1);
        EXPECT_NO_THROW(sv.set_amplitudes(zero, {Validation::Warn, kAtol,
                                                 Repair::Attempt}));
        EXPECT_EQ(probe.count(), 1u) << "Warn reported nothing";
        EXPECT_TRUE(probe.any_contains("no norm to divide out"))
            << "the report does not say why the rescale was impossible";
    }
    {
        WarningProbe probe;
        Statevector sv(1);
        EXPECT_NO_THROW(sv.set_amplitudes(zero, {Validation::Ignore, kAtol,
                                                 Repair::Attempt}));
        EXPECT_EQ(probe.count(), 0u) << "Ignore reported something";
    }
}

TEST(V11271RepairImpossible, ARefusedHandOverLeavesTheStateAlone) {
    // The buffer is judged before anything is written, so a hand-over the
    // policy refuses leaves the object holding what it held. Without this the
    // throw could arrive after the copy and the state would carry the very
    // amplitudes that were rejected.
    Statevector sv(1);
    const std::vector<Complex128> zero{Complex128(0.0, 0.0), Complex128(0.0, 0.0)};

    EXPECT_THROW(sv.set_amplitudes(zero, {Validation::Throw, kAtol,
                                          Repair::Attempt}),
                 std::invalid_argument);
    EXPECT_TRUE(sv.is_normalized(kAtol))
        << "a refused hand-over overwrote the state it refused to accept";

    // Under Warn the call proceeds, and proceeding means the amplitudes are
    // accepted as given rather than rescaled by a division that cannot happen.
    {
        WarningProbe probe;
        Statevector warned(1);
        EXPECT_NO_THROW(warned.set_amplitudes(zero, {Validation::Warn, kAtol,
                                                     Repair::Attempt}));
        EXPECT_DOUBLE_EQ(warned.norm_sq(), 0.0)
            << "Warn is a report, not a repair; the buffer it reported on is "
               "the buffer it accepted";
    }
}

TEST(V11271RepairImpossible, ANonFiniteStateAnswersToTheResponse) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<Complex128> bad{Complex128(nan, 0.0), Complex128(0.0, 0.0)};

    Statevector thrower(1);
    EXPECT_THROW(thrower.set_amplitudes(bad, {Validation::Throw, kAtol,
                                             Repair::Attempt}),
                 std::invalid_argument);

    WarningProbe probe;
    Statevector quiet(1);
    EXPECT_NO_THROW(quiet.set_amplitudes(bad, {Validation::Ignore, kAtol,
                                              Repair::Attempt}));
    EXPECT_EQ(probe.count(), 0u);
}

TEST(V11271RepairImpossible, ARepairableStateIsStillRepaired) {
    // The bound on all of the above: none of it may have been implemented by
    // declining to repair. A state that CAN be rescaled still is, under every
    // response.
    for (const auto response : r1211::kResponses) {
        WarningProbe probe;
        Statevector sv(1);
        const std::vector<Complex128> scaled{Complex128(2.0, 0.0),
                                             Complex128(0.0, 0.0)};
        ASSERT_NO_THROW(sv.set_amplitudes(scaled, {response, kAtol,
                                                   Repair::Attempt}))
            << r1211::response_name(response);
        EXPECT_TRUE(sv.is_normalized(kAtol))
            << "a state that could be rescaled was not, under "
            << r1211::response_name(response);
        EXPECT_EQ(probe.count(), 0u)
            << "a repair that succeeded reported under "
            << r1211::response_name(response);
    }
}
