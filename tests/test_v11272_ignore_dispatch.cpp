// 1.1.27.2 test wave - an arrived Ignore is a broken contract, not a policy
// (#123).
//
// Ignore means the check does not run. The entry points implement that by
// returning before measuring, which is what `measurement_unused` decides, so
// the dispatchers below are never reached with it in normal use.
//
// So what should a dispatcher do if one arrives anyway? It means something
// measured the operand despite being told not to look, and found it bad. The
// answer is to raise it. Treating it as an accept discards a real finding and,
// worse, removes the only signal an entry point that lost its early return
// would ever produce: every such entry point would then pass its tests while
// silently accepting violations.
//
// The reason this needs a suite of its own rather than one assertion is that
// the case is invisible from outside. No public call can reach it, so nothing
// in the library exercises it, and a change to either dispatcher's switch is
// unobservable except here. That is exactly how it regressed: a comment
// claiming Ignore only arrives alongside Repair::Attempt was written, the
// behaviour was changed to match, and only the one pre-existing assertion
// noticed. The claim was false because Repair::Attempt is settled two lines
// above the switch, so the switch sees Repair::None and nothing else.
//
// The asymmetry with respond_unrepaired is deliberate and is asserted here so
// that it cannot be mistaken for the same oversight. That function runs only
// after a repair was requested and attempted, so its Ignore is Ignore WITH a
// repair, which is a policy a caller holds on purpose: fix what you can, say
// nothing about what you cannot.

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "r1211_policy_probe.hpp"

#include "lindblad/detail/validate_physical.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using r1211::WarningProbe;

namespace {

constexpr double kAtol = DEFAULT_PHYSICAL_ATOL;

// Comfortably outside kAtol, so every property judges it a violation.
constexpr double kBadResidual = 1e-6;

// The properties reaching each dispatcher. enforce_physical serves the ones
// with no repair; enforce_physical_repairable serves the ones that have one.
const detail::PhysicalProperty kUnrepairable[]{
    detail::KRAUS_TRACE_PRESERVING, detail::SUPEROP_TRACE_PRESERVING};
const detail::PhysicalProperty kRepairable[]{
    detail::UNITARITY, detail::STATE_NORMALIZATION,
    detail::DENSITY_NORMALIZATION};

}  // namespace

// =============================================================================
// Both dispatchers refuse an arrived Ignore
// =============================================================================

TEST(V11272IgnoreDispatch, EnforcePhysicalRaisesAnArrivedIgnore) {
    for (const auto& prop : kUnrepairable) {
        WarningProbe probe;
        EXPECT_THROW(detail::enforce_physical(kBadResidual,
                                              {Validation::Ignore, kAtol},
                                              "probe", prop),
                     std::invalid_argument)
            << prop.noun
            << ": a residual that was measured and found bad was discarded on "
               "the strength of the policy that should have skipped measuring "
               "it";
        EXPECT_EQ(probe.count(), 0u)
            << prop.noun << ": the refusal is a throw, not a report";
    }
}

TEST(V11272IgnoreDispatch, EnforcePhysicalRepairableRaisesAnArrivedIgnore) {
    // The half that had no coverage at all when this regressed. Both
    // dispatchers carry the same switch and the same reasoning, so a test on
    // one of them leaves the other free to drift.
    for (const auto& prop : kRepairable) {
        WarningProbe probe;
        EXPECT_THROW((void)detail::enforce_physical_repairable(
                         kBadResidual, {Validation::Ignore, kAtol}, "probe",
                         prop),
                     std::invalid_argument)
            << prop.noun
            << ": the repairing dispatcher accepted an Ignore that was never "
               "supposed to be measured";
        EXPECT_EQ(probe.count(), 0u) << prop.noun;
    }
}

TEST(V11272IgnoreDispatch, TheTwoDispatchersAgreeOnIt) {
    // Unitarity reaches both, depending on whether the entry point stores or
    // borrows its operand. The same options and the same residual must produce
    // the same verdict, or which route a caller happened to take would decide
    // whether their violation was reported.
    std::string from_plain;
    std::string from_repairing;

    try {
        detail::enforce_physical(kBadResidual, {Validation::Ignore, kAtol},
                                 "probe", detail::UNITARITY);
    } catch (const std::invalid_argument& e) {
        from_plain = e.what();
    }
    try {
        (void)detail::enforce_physical_repairable(
            kBadResidual, {Validation::Ignore, kAtol}, "probe",
            detail::UNITARITY);
    } catch (const std::invalid_argument& e) {
        from_repairing = e.what();
    }

    ASSERT_FALSE(from_plain.empty()) << "enforce_physical accepted it";
    ASSERT_FALSE(from_repairing.empty())
        << "enforce_physical_repairable accepted it";
    EXPECT_EQ(from_plain, from_repairing)
        << "the two dispatchers describe the same violation differently";
}

TEST(V11272IgnoreDispatch, TheRefusalReadsAsAViolationNotAsAMissingRepair) {
    // The message has to name what was actually wrong. An arrived Ignore is a
    // violation that was measured, not a repair request that could not be
    // served, and the two diagnostics send a reader to different places.
    try {
        detail::enforce_physical(kBadResidual, {Validation::Ignore, kAtol},
                                 "probe", detail::KRAUS_TRACE_PRESERVING);
        FAIL() << "an arrived Ignore was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not trace preserving"), std::string::npos)
            << "the message does not describe the violation. Got: " << msg;
        EXPECT_EQ(msg.find("no repair defined"), std::string::npos)
            << "no repair was asked for, so the message must not claim one is "
               "missing. Got: " << msg;
    }
}

// =============================================================================
// Nothing above changed what Ignore means at the entry points
// =============================================================================

TEST(V11272IgnoreDispatch, AnEntryPointStillDoesNotMeasureUnderPlainIgnore) {
    // The fix restores a refusal in a place no public call reaches, so the
    // thing most worth checking is that it did not leak outward. Plain Ignore
    // must still take no measurement, which a NaN operand proves: any
    // measurement at all rejects it.
    std::vector<Complex128> nan_matrix{
        Complex128(std::nan(""), 0.0), Complex128(0.0, 0.0),
        Complex128(0.0, 0.0), Complex128(1.0, 0.0)};

    WarningProbe probe;
    Statevector sv(1);
    EXPECT_NO_THROW(
        gates::apply_unitary(sv, {0}, nan_matrix, {Validation::Ignore, kAtol}))
        << "the dispatcher's refusal reached a caller who asked not to be "
           "checked, which is the opposite of what Ignore means";
    EXPECT_EQ(probe.count(), 0u);
}

TEST(V11272IgnoreDispatch, MeasurementUnusedIsExactlyPlainIgnore) {
    // The guard that makes the refusal above unreachable from outside. If it
    // ever widened to cover Ignore with a repair, the repair could no longer
    // tell whether it was owed; if it narrowed, plain Ignore would start paying
    // for a residual nothing reads.
    EXPECT_TRUE(detail::measurement_unused({Validation::Ignore, kAtol}));
    EXPECT_FALSE(detail::measurement_unused(
        {Validation::Ignore, kAtol, Repair::Attempt}));
    EXPECT_FALSE(detail::measurement_unused({Validation::Throw, kAtol}));
    EXPECT_FALSE(detail::measurement_unused({Validation::Warn, kAtol}));
    EXPECT_FALSE(detail::measurement_unused(
        {Validation::Throw, kAtol, Repair::Attempt}));
    EXPECT_FALSE(detail::measurement_unused(
        {Validation::Warn, kAtol, Repair::Attempt}));
}

// =============================================================================
// The deliberate asymmetry
// =============================================================================

TEST(V11272IgnoreDispatch, RespondUnrepairedStillHonoursIgnore) {
    // Reached only after a repair was requested AND attempted, so its Ignore is
    // Ignore with Repair::Attempt. That is a policy a caller holds on purpose,
    // and honouring it is the whole point of splitting the enum: repair what
    // you can, stay quiet about what you cannot.
    WarningProbe probe;
    EXPECT_NO_THROW(detail::respond_unrepaired(
        {Validation::Ignore, kAtol, Repair::Attempt}, "probe",
        detail::UNITARITY, "the probe said so"))
        << "the response a caller chose deliberately was overridden";
    EXPECT_EQ(probe.count(), 0u) << "Ignore reported something";
}

TEST(V11272IgnoreDispatch, RespondUnrepairedStillReportsAndRejects) {
    // The other two responses at the same site, so the test above cannot pass
    // by the function having stopped doing anything.
    {
        WarningProbe probe;
        EXPECT_NO_THROW(detail::respond_unrepaired(
            {Validation::Warn, kAtol, Repair::Attempt}, "probe",
            detail::UNITARITY, "the probe said so"));
        EXPECT_EQ(probe.count(), 1u) << "Warn reported nothing";
    }
    EXPECT_THROW(detail::respond_unrepaired(
                     {Validation::Throw, kAtol, Repair::Attempt}, "probe",
                     detail::UNITARITY, "the probe said so"),
                 std::invalid_argument);
}

TEST(V11272IgnoreDispatch, TheAsymmetryIsBetweenTwoDifferentIgnores) {
    // Stated in one place so it cannot be read as an inconsistency. The
    // dispatchers refuse Ignore with NO repair, which no entry point should
    // have measured. respond_unrepaired honours Ignore WITH a repair, which is
    // a caller's deliberate choice. They are different policies that happen to
    // share a response.
    EXPECT_THROW(detail::enforce_physical(kBadResidual,
                                          {Validation::Ignore, kAtol}, "probe",
                                          detail::KRAUS_TRACE_PRESERVING),
                 std::invalid_argument);

    WarningProbe probe;
    EXPECT_NO_THROW(detail::respond_unrepaired(
        {Validation::Ignore, kAtol, Repair::Attempt}, "probe",
        detail::UNITARITY, "the probe said so"));
    EXPECT_EQ(probe.count(), 0u);
}
