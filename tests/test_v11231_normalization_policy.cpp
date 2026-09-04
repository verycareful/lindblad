// 1.1.23.1 test wave - the normalization policy surface, on every state class.
//
// 1.1.23.0 gave the repair knob its first real repair and put normalization
// behind the Class C framework. That added the same four members to six
// classes: is_normalized, check_normalized, a normalize() that now refuses
// rather than returning silently, and (on Statevector) a set_amplitudes that
// judges before it writes. None of it had a test.
//
// The shape of this file follows from a property the design insists on: every
// state class gets all four policies and none gets a reduced enum. A claim made
// six times is worth checking six times, so the policy body is written once and
// instantiated per class through an adapter. Each class still gets its own
// named test, so a failure names the class rather than a template.
//
// The residual arithmetic here is exact on purpose. Scaling a normalized state
// by 2 puts norm_sq at exactly 4 and a trace at exactly 2, so the deviation
// asserted against is an integer and no tolerance question enters a test whose
// subject is tolerances.

#include <gtest/gtest.h>

#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// Scaling a normalized state by this puts a squared norm at 4 and a trace at 2.
constexpr double kScale = 2.0;

// The two states with no repair: nothing to divide out, and a norm that
// poisons every amplitude it divides.
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// -----------------------------------------------------------------------------
// Warning capture
// -----------------------------------------------------------------------------
// The channel counts repeats of an identical message rather than re-emitting
// it, and that state is process-wide. A test that did not reset it could see a
// warning consumed by whichever test ran first, so the channel is cleared on
// both sides of every use.

class WarningCapture {
public:
    WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
        set_warning_handler([this](const std::string& m) { lines_.push_back(m); });
    }

    ~WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
    }

    std::size_t count() const { return lines_.size(); }
    const std::string& only() const { return lines_.front(); }

private:
    std::vector<std::string> lines_;
};

// -----------------------------------------------------------------------------
// Adapters
// -----------------------------------------------------------------------------
// Each adapter teaches the shared bodies below how to reach one state class:
// build an instance off normalization by a known factor, build the two
// unrepairable ones, measure what the object currently sums to, and submit it
// to a policy. `measure` returns 1 for a normalized object on every class,
// whether the underlying quantity is a squared norm or a trace.

struct SvAdapter {
    using State = Statevector;
    static const char* name() { return "Statevector"; }

    static State scaled(double f) {
        State s(1);
        s.set_amplitudes({Complex128(f, 0.0), Complex128(0.0, 0.0)},
                         {Validation::Ignore});
        return s;
    }
    static State off_normalization() { return scaled(kScale); }
    static State zero() { return scaled(0.0); }
    static State non_finite() { return scaled(kNaN); }

    static double measure(const State& s) { return s.norm_sq(); }
    static void check(State& s, ValidationOptions v) { s.check_normalized(v); }
    static bool predicate(const State& s, double atol) { return s.is_normalized(atol); }
    static void normalize(State& s) { s.normalize(); }
};

struct DmAdapter {
    using State = DensityMatrix;
    static const char* name() { return "DensityMatrix"; }

    static State scaled(double f) {
        State r(1);              // |0><0|, so the trace sits in one entry
        r.data[0] = Complex128(f, 0.0);
        return r;
    }
    static State off_normalization() { return scaled(kScale); }
    static State zero() { return scaled(0.0); }
    static State non_finite() { return scaled(kNaN); }

    static double measure(const State& r) { return r.trace(); }
    static void check(State& r, ValidationOptions v) { r.check_normalized(v); }
    static bool predicate(const State& r, double atol) { return r.is_normalized(atol); }
    static void normalize(State& r) { r.normalize(); }
};

struct MpsAdapter {
    using State = MPSState;
    static const char* name() { return "MPSState"; }

    static State scaled(double f) {
        State m(2);
        for (Complex128& e : m.tensors[0].data) {
            e.real *= f;
            e.imag *= f;
        }
        return m;
    }
    static State off_normalization() { return scaled(kScale); }
    static State zero() { return scaled(0.0); }
    static State non_finite() {
        State m(2);
        m.tensors[0].data[0] = Complex128(kNaN, 0.0);
        return m;
    }

    static double measure(const State& m) { return m.norm_sq(); }
    static void check(State& m, ValidationOptions v) { m.check_normalized(v); }
    static bool predicate(const State& m, double atol) { return m.is_normalized(atol); }
    static void normalize(State& m) { m.normalize(); }
};

struct QsvAdapter {
    using State = QuditStatevector;
    static const char* name() { return "QuditStatevector"; }

    static State scaled(double f) {
        State s(2, 3);
        s.amplitudes[0] = Complex128(f, 0.0);
        return s;
    }
    static State off_normalization() { return scaled(kScale); }
    static State zero() { return scaled(0.0); }
    static State non_finite() { return scaled(kNaN); }

    static double measure(const State& s) { return s.norm_sq(); }
    static void check(State& s, ValidationOptions v) { s.check_normalized(v); }
    static bool predicate(const State& s, double atol) { return s.is_normalized(atol); }
    static void normalize(State& s) { s.normalize(); }
};

struct QdmAdapter {
    using State = QuditDensityMatrix;
    static const char* name() { return "QuditDensityMatrix"; }

    static State scaled(double f) {
        State r(2, 3);
        r.rho[0] = Complex128(f, 0.0);
        return r;
    }
    static State off_normalization() { return scaled(kScale); }
    static State zero() { return scaled(0.0); }
    static State non_finite() { return scaled(kNaN); }

    static double measure(const State& r) { return r.trace(); }
    static void check(State& r, ValidationOptions v) { r.check_normalized(v); }
    static bool predicate(const State& r, double atol) { return r.is_normalized(atol); }
    static void normalize(State& r) { r.normalize(); }
};

struct QmpsAdapter {
    using State = QuditMPS;
    static const char* name() { return "QuditMPS"; }

    static State scaled(double f) {
        State m(2, 3);
        for (Complex128& e : m.tensors[0].data) {
            e.real *= f;
            e.imag *= f;
        }
        return m;
    }
    static State off_normalization() { return scaled(kScale); }
    static State zero() { return scaled(0.0); }
    static State non_finite() {
        State m(2, 3);
        m.tensors[0].data[0] = Complex128(kNaN, 0.0);
        return m;
    }

    static double measure(const State& m) { return m.norm_sq(); }
    static void check(State& m, ValidationOptions v) { m.check_normalized(v); }
    static bool predicate(const State& m, double atol) { return m.is_normalized(atol); }
    static void normalize(State& m) { m.normalize(); }
};

// -----------------------------------------------------------------------------
// The shared bodies
// -----------------------------------------------------------------------------

// Throw is the default, so a bare check_normalized() rejects an off-normalization
// object without the caller naming a policy.
template <typename A>
void expect_throw_is_default() {
    SCOPED_TRACE(A::name());
    typename A::State s = A::off_normalization();
    EXPECT_THROW(A::check(s, ValidationOptions{}), std::invalid_argument);

    typename A::State t = A::off_normalization();
    ValidationOptions defaulted;
    EXPECT_EQ(defaulted.policy, Validation::Throw)
        << "the default policy is what a caller gets by writing nothing";
    EXPECT_THROW(A::check(t, defaulted), std::invalid_argument);
}

// Warn reports and returns, leaving the object still violating the property.
template <typename A>
void expect_warn_reports_without_repairing() {
    SCOPED_TRACE(A::name());
    typename A::State s = A::off_normalization();
    const double before = A::measure(s);

    std::size_t warnings = 0;
    std::string message;
    {
        WarningCapture cap;
        A::check(s, {Validation::Warn});
        warnings = cap.count();
        if (warnings > 0) message = cap.only();
    }

    ASSERT_EQ(warnings, 1u) << "Warn reports exactly once per violation";
    EXPECT_NE(message.find("not normalized"), std::string::npos)
        << "the warning names the property, got: " << message;
    EXPECT_DOUBLE_EQ(A::measure(s), before)
        << "Warn describes; it does not repair";
}

// Fix repairs in place, so the object measures 1 afterwards.
template <typename A>
void expect_fix_repairs() {
    SCOPED_TRACE(A::name());
    typename A::State s = A::off_normalization();
    ASSERT_NE(A::measure(s), 1.0) << "the fixture must start off normalization";

    WarningCapture cap;
    A::check(s, {Validation::Throw, DEFAULT_PHYSICAL_ATOL, Repair::Attempt});
    EXPECT_EQ(cap.count(), 0u) << "Fix repairs silently; it is not a warning";
    EXPECT_NEAR(A::measure(s), 1.0, DEFAULT_PHYSICAL_ATOL)
        << "Fix leaves the object satisfying the property it was judged against";
    EXPECT_TRUE(A::predicate(s, DEFAULT_PHYSICAL_ATOL))
        << "the predicate and the policy must agree once Fix has run";
}

// Ignore measures nothing, so it neither throws nor reports nor changes anything.
template <typename A>
void expect_ignore_is_silent() {
    SCOPED_TRACE(A::name());
    typename A::State s = A::off_normalization();
    const double before = A::measure(s);

    WarningCapture cap;
    EXPECT_NO_THROW(A::check(s, {Validation::Ignore}));
    EXPECT_EQ(cap.count(), 0u) << "Ignore is the policy that costs nothing";
    EXPECT_DOUBLE_EQ(A::measure(s), before) << "Ignore leaves the object alone";
}

// Inside tolerance the six policies cannot be told apart: nothing throws,
// nothing warns, nothing moves. The repairing three are included because an
// implementation that rescaled unconditionally rather than on a measured
// violation would be invisible to the other three.
template <typename A>
void expect_policies_agree_within_tolerance() {
    SCOPED_TRACE(A::name());
    const ValidationOptions all[] = {
        {Validation::Throw}, {Validation::Warn}, {Validation::Ignore},
        {Validation::Throw, DEFAULT_PHYSICAL_ATOL, Repair::Attempt},
        {Validation::Warn, DEFAULT_PHYSICAL_ATOL, Repair::Attempt},
        {Validation::Ignore, DEFAULT_PHYSICAL_ATOL, Repair::Attempt}};
    for (const ValidationOptions& p : all) {
        typename A::State s = A::scaled(1.0);
        ASSERT_NEAR(A::measure(s), 1.0, DEFAULT_PHYSICAL_ATOL)
            << "the fixture must start normalized";
        const double before = A::measure(s);

        WarningCapture cap;
        EXPECT_NO_THROW(A::check(s, {p}));
        EXPECT_EQ(cap.count(), 0u)
            << "a satisfied property is not reportable under any policy";
        EXPECT_DOUBLE_EQ(A::measure(s), before);
    }
}

// A rescale is impossible for exactly two objects, so asking for one leaves
// the operand still violating the property and the response decides. Under
// Throw that is std::invalid_argument, the same type every other rejection in
// this family raises: the caller handed over an object that cannot be a state,
// which is a statement about the argument.
template <typename A>
void expect_repair_throws_on_unrepairable() {
    SCOPED_TRACE(A::name());
    const ValidationOptions repairing{Validation::Throw, DEFAULT_PHYSICAL_ATOL,
                                      Repair::Attempt};

    typename A::State z = A::zero();
    EXPECT_THROW(A::check(z, repairing), std::invalid_argument)
        << "a zero object has no direction to normalize toward";

    typename A::State n = A::non_finite();
    EXPECT_THROW(A::check(n, repairing), std::invalid_argument)
        << "dividing by a non-finite norm spreads it rather than removing it";
}

// The same two objects under the other two responses. This is the half the
// split exists for: a caller who asked for a repair and said what should happen
// when one is impossible gets that, rather than the throw being the only
// answer available.
template <typename A>
void expect_repair_respects_the_response_on_unrepairable() {
    SCOPED_TRACE(A::name());

    for (auto response : {Validation::Warn, Validation::Ignore}) {
        const ValidationOptions v{response, DEFAULT_PHYSICAL_ATOL,
                                  Repair::Attempt};
        for (bool use_zero : {true, false}) {
            typename A::State s = use_zero ? A::zero() : A::non_finite();
            WarningCapture cap;
            EXPECT_NO_THROW(A::check(s, v))
                << "a repair that cannot run must answer to the response, and "
                   "neither Warn nor Ignore says throw";
            if (response == Validation::Warn) {
                EXPECT_EQ(cap.count(), 1u)
                    << "Warn proceeded without reporting the repair it could "
                       "not perform, which is indistinguishable from Ignore";
            } else {
                EXPECT_EQ(cap.count(), 0u)
                    << "Ignore reported something; it is the response that says "
                       "nothing";
            }
        }
    }
}

// normalize() refuses the same two objects, through the same test.
template <typename A>
void expect_normalize_refuses_unrepairable() {
    SCOPED_TRACE(A::name());
    typename A::State z = A::zero();
    EXPECT_THROW(A::normalize(z), std::runtime_error);

    typename A::State n = A::non_finite();
    EXPECT_THROW(A::normalize(n), std::runtime_error);

    typename A::State ok = A::off_normalization();
    EXPECT_NO_THROW(A::normalize(ok));
    EXPECT_NEAR(A::measure(ok), 1.0, DEFAULT_PHYSICAL_ATOL);
}

// The predicate answers and does nothing else: it does not repair, and a
// non-finite object answers false rather than throwing.
template <typename A>
void expect_predicate_only_answers() {
    SCOPED_TRACE(A::name());
    typename A::State s = A::off_normalization();
    const double before = A::measure(s);
    EXPECT_FALSE(A::predicate(s, DEFAULT_PHYSICAL_ATOL));
    EXPECT_DOUBLE_EQ(A::measure(s), before) << "a predicate does not repair";

    typename A::State n = A::non_finite();
    bool answered = true;
    EXPECT_NO_THROW(answered = A::predicate(n, DEFAULT_PHYSICAL_ATOL))
        << "a predicate answers rather than throwing";
    EXPECT_FALSE(answered) << "a non-finite object is not normalized";

    typename A::State good = A::scaled(1.0);
    EXPECT_TRUE(A::predicate(good, DEFAULT_PHYSICAL_ATOL));
}

// A tolerance wide enough to span the deviation accepts it, which is what makes
// the parameter a tolerance rather than decoration.
template <typename A>
void expect_predicate_respects_atol() {
    SCOPED_TRACE(A::name());
    typename A::State s = A::off_normalization();
    const double deviation = std::abs(A::measure(s) - 1.0);
    ASSERT_GT(deviation, 0.0);

    EXPECT_FALSE(A::predicate(s, deviation / 2.0))
        << "a tolerance below the deviation rejects";
    EXPECT_TRUE(A::predicate(s, deviation * 2.0))
        << "a tolerance above the deviation accepts";
}

}  // namespace

// =============================================================================
// Throw is the default on every class
// =============================================================================

TEST(V11231NormalizationPolicy, StatevectorThrowsByDefault) { expect_throw_is_default<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixThrowsByDefault) { expect_throw_is_default<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStateThrowsByDefault) { expect_throw_is_default<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorThrowsByDefault) { expect_throw_is_default<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixThrowsByDefault) { expect_throw_is_default<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsThrowsByDefault) { expect_throw_is_default<QmpsAdapter>(); }

// =============================================================================
// Warn reports without repairing
// =============================================================================

TEST(V11231NormalizationPolicy, StatevectorWarnReportsOnly) { expect_warn_reports_without_repairing<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixWarnReportsOnly) { expect_warn_reports_without_repairing<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStateWarnReportsOnly) { expect_warn_reports_without_repairing<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorWarnReportsOnly) { expect_warn_reports_without_repairing<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixWarnReportsOnly) { expect_warn_reports_without_repairing<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsWarnReportsOnly) { expect_warn_reports_without_repairing<QmpsAdapter>(); }

// =============================================================================
// Fix repairs
// =============================================================================

TEST(V11231NormalizationPolicy, StatevectorFixRepairs) { expect_fix_repairs<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixFixRepairs) { expect_fix_repairs<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStateFixRepairs) { expect_fix_repairs<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorFixRepairs) { expect_fix_repairs<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixFixRepairs) { expect_fix_repairs<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsFixRepairs) { expect_fix_repairs<QmpsAdapter>(); }

// =============================================================================
// Ignore is silent
// =============================================================================

TEST(V11231NormalizationPolicy, StatevectorIgnoreIsSilent) { expect_ignore_is_silent<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixIgnoreIsSilent) { expect_ignore_is_silent<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStateIgnoreIsSilent) { expect_ignore_is_silent<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorIgnoreIsSilent) { expect_ignore_is_silent<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixIgnoreIsSilent) { expect_ignore_is_silent<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsIgnoreIsSilent) { expect_ignore_is_silent<QmpsAdapter>(); }

// =============================================================================
// Inside tolerance the policies are indistinguishable
// =============================================================================

TEST(V11231NormalizationPolicy, StatevectorPoliciesAgreeWhenValid) { expect_policies_agree_within_tolerance<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixPoliciesAgreeWhenValid) { expect_policies_agree_within_tolerance<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStatePoliciesAgreeWhenValid) { expect_policies_agree_within_tolerance<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorPoliciesAgreeWhenValid) { expect_policies_agree_within_tolerance<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixPoliciesAgreeWhenValid) { expect_policies_agree_within_tolerance<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsPoliciesAgreeWhenValid) { expect_policies_agree_within_tolerance<QmpsAdapter>(); }

// =============================================================================
// Fix throws where no repair exists
// =============================================================================

TEST(V11231NormalizationPolicy, StatevectorRepairThrowsOnUnrepairable) { expect_repair_throws_on_unrepairable<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixRepairThrowsOnUnrepairable) { expect_repair_throws_on_unrepairable<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStateRepairThrowsOnUnrepairable) { expect_repair_throws_on_unrepairable<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorRepairThrowsOnUnrepairable) { expect_repair_throws_on_unrepairable<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixRepairThrowsOnUnrepairable) { expect_repair_throws_on_unrepairable<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsRepairThrowsOnUnrepairable) { expect_repair_throws_on_unrepairable<QmpsAdapter>(); }
TEST(V11231NormalizationPolicy, StatevectorUnrepairableAnswersToTheResponse) { expect_repair_respects_the_response_on_unrepairable<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixUnrepairableAnswersToTheResponse) { expect_repair_respects_the_response_on_unrepairable<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStateUnrepairableAnswersToTheResponse) { expect_repair_respects_the_response_on_unrepairable<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorUnrepairableAnswersToTheResponse) { expect_repair_respects_the_response_on_unrepairable<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixUnrepairableAnswersToTheResponse) { expect_repair_respects_the_response_on_unrepairable<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsUnrepairableAnswersToTheResponse) { expect_repair_respects_the_response_on_unrepairable<QmpsAdapter>(); }

// =============================================================================
// normalize() refuses rather than returning unchanged
// =============================================================================

TEST(V11231NormalizationPolicy, StatevectorNormalizeRefuses) { expect_normalize_refuses_unrepairable<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixNormalizeRefuses) { expect_normalize_refuses_unrepairable<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStateNormalizeRefuses) { expect_normalize_refuses_unrepairable<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorNormalizeRefuses) { expect_normalize_refuses_unrepairable<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixNormalizeRefuses) { expect_normalize_refuses_unrepairable<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsNormalizeRefuses) { expect_normalize_refuses_unrepairable<QmpsAdapter>(); }

// =============================================================================
// The predicate answers and does nothing else
// =============================================================================

TEST(V11231NormalizationPolicy, StatevectorPredicateOnlyAnswers) { expect_predicate_only_answers<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixPredicateOnlyAnswers) { expect_predicate_only_answers<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStatePredicateOnlyAnswers) { expect_predicate_only_answers<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorPredicateOnlyAnswers) { expect_predicate_only_answers<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixPredicateOnlyAnswers) { expect_predicate_only_answers<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsPredicateOnlyAnswers) { expect_predicate_only_answers<QmpsAdapter>(); }

TEST(V11231NormalizationPolicy, StatevectorPredicateRespectsAtol) { expect_predicate_respects_atol<SvAdapter>(); }
TEST(V11231NormalizationPolicy, DensityMatrixPredicateRespectsAtol) { expect_predicate_respects_atol<DmAdapter>(); }
TEST(V11231NormalizationPolicy, MpsStatePredicateRespectsAtol) { expect_predicate_respects_atol<MpsAdapter>(); }
TEST(V11231NormalizationPolicy, QuditStatevectorPredicateRespectsAtol) { expect_predicate_respects_atol<QsvAdapter>(); }
TEST(V11231NormalizationPolicy, QuditDensityMatrixPredicateRespectsAtol) { expect_predicate_respects_atol<QdmAdapter>(); }
TEST(V11231NormalizationPolicy, QuditMpsPredicateRespectsAtol) { expect_predicate_respects_atol<QmpsAdapter>(); }

// =============================================================================
// set_amplitudes: the hand-over point
// =============================================================================

TEST(V11231SetAmplitudes, ThrowIsTheDefaultOnBothOverloads) {
    const std::vector<Complex128> bad = {Complex128(kScale, 0.0), Complex128(0.0, 0.0)};

    Statevector a(1);
    EXPECT_THROW(a.set_amplitudes(bad), std::invalid_argument);

    const double re[] = {kScale, 0.0};
    const double im[] = {0.0, 0.0};
    Statevector b(1);
    EXPECT_THROW(b.set_amplitudes(re, im, 2), std::invalid_argument);
}

// The policy is judged against the caller's buffer BEFORE anything is written,
// which is what lets a refused hand-over be a no-op rather than a partial one.
TEST(V11231SetAmplitudes, RefusedHandOverLeavesTheObjectUnchanged) {
    Statevector sv(1);
    const std::vector<Complex128> good = {Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
    sv.set_amplitudes(good);
    ASSERT_TRUE(sv.is_normalized());

    const std::vector<Complex128> bad = {Complex128(kScale, 0.0), Complex128(0.0, 0.0)};
    EXPECT_THROW(sv.set_amplitudes(bad), std::invalid_argument);

    const std::vector<Complex128> after = sv.amplitudes();
    ASSERT_EQ(after.size(), good.size());
    for (std::size_t i = 0; i < after.size(); ++i) {
        EXPECT_DOUBLE_EQ(after[i].real, good[i].real) << "amplitude " << i;
        EXPECT_DOUBLE_EQ(after[i].imag, good[i].imag) << "amplitude " << i;
    }
}

TEST(V11231SetAmplitudes, RefusedHandOverLeavesTheObjectUnchangedPointerOverload) {
    Statevector sv(1);
    const double good_re[] = {0.0, 1.0};
    const double good_im[] = {0.0, 0.0};
    sv.set_amplitudes(good_re, good_im, 2);
    ASSERT_TRUE(sv.is_normalized());

    const double bad_re[] = {kScale, 0.0};
    const double bad_im[] = {0.0, 0.0};
    EXPECT_THROW(sv.set_amplitudes(bad_re, bad_im, 2), std::invalid_argument);

    const std::vector<Complex128> after = sv.amplitudes();
    EXPECT_DOUBLE_EQ(after[0].real, good_re[0]);
    EXPECT_DOUBLE_EQ(after[1].real, good_re[1]);
}

TEST(V11231SetAmplitudes, FixNormalizesTheAmplitudesGiven) {
    const std::vector<Complex128> bad = {Complex128(kScale, 0.0), Complex128(0.0, 0.0)};

    Statevector sv(1);
    sv.set_amplitudes(bad, {Validation::Throw, DEFAULT_PHYSICAL_ATOL, Repair::Attempt});
    EXPECT_NEAR(sv.norm_sq(), 1.0, DEFAULT_PHYSICAL_ATOL);
    EXPECT_NEAR(sv.amplitudes()[0].real, 1.0, DEFAULT_PHYSICAL_ATOL)
        << "the direction the caller gave is preserved; only the length changes";

    Statevector ptr(1);
    const double re[] = {kScale, 0.0};
    const double im[] = {0.0, 0.0};
    ptr.set_amplitudes(re, im, 2, {Validation::Throw, DEFAULT_PHYSICAL_ATOL, Repair::Attempt});
    EXPECT_NEAR(ptr.norm_sq(), 1.0, DEFAULT_PHYSICAL_ATOL);
}

TEST(V11231SetAmplitudes, IgnoreWritesTheBufferUnchanged) {
    const std::vector<Complex128> bad = {Complex128(kScale, 0.0), Complex128(0.0, 0.0)};

    WarningCapture cap;
    Statevector sv(1);
    EXPECT_NO_THROW(sv.set_amplitudes(bad, {Validation::Ignore}));
    EXPECT_EQ(cap.count(), 0u);
    EXPECT_DOUBLE_EQ(sv.amplitudes()[0].real, kScale)
        << "Ignore hands the buffer through exactly as given";
    EXPECT_DOUBLE_EQ(sv.norm_sq(), kScale * kScale);
}

TEST(V11231SetAmplitudes, WarnWritesTheBufferAndReports) {
    const std::vector<Complex128> bad = {Complex128(kScale, 0.0), Complex128(0.0, 0.0)};

    std::size_t warnings = 0;
    Statevector sv(1);
    {
        WarningCapture cap;
        EXPECT_NO_THROW(sv.set_amplitudes(bad, {Validation::Warn}));
        warnings = cap.count();
    }
    EXPECT_EQ(warnings, 1u);
    EXPECT_DOUBLE_EQ(sv.amplitudes()[0].real, kScale)
        << "Warn proceeds with the hand-over it reported";
}

TEST(V11231SetAmplitudes, RepairThrowsOnAnUnrepairableBuffer) {
    Statevector sv(1);
    const std::vector<Complex128> zero = {Complex128(0.0, 0.0), Complex128(0.0, 0.0)};
    EXPECT_THROW(sv.set_amplitudes(zero, {Validation::Throw, DEFAULT_PHYSICAL_ATOL,
                                          Repair::Attempt}),
                 std::invalid_argument);

    const std::vector<Complex128> nan = {Complex128(kNaN, 0.0), Complex128(0.0, 0.0)};
    EXPECT_THROW(sv.set_amplitudes(nan, {Validation::Throw, DEFAULT_PHYSICAL_ATOL,
                                         Repair::Attempt}),
                 std::invalid_argument);

    // The buffer is judged before anything is written, so a hand-over refused
    // here leaves the object holding what it held. Without this the throw could
    // arrive after the copy and the state would carry the rejected amplitudes.
    EXPECT_TRUE(sv.is_normalized(DEFAULT_PHYSICAL_ATOL))
        << "a refused hand-over overwrote the state it refused to accept";
}

// =============================================================================
// The message describes a scalar residual as a scalar
// =============================================================================

// A normalization residual is one number. Every property before it reduced over
// a matrix and reported its worst entry, so the formatter used to claim a
// maximum for all of them. These pin that the claim comes from the property
// rather than from the formatter.

TEST(V11231NormalizationMessage, ScalarResidualsClaimNoMaximum) {
    EXPECT_EQ(std::string(detail::STATE_NORMALIZATION.residual).find("max"),
              std::string::npos)
        << "a state norm is one number and takes no maximum, got: "
        << detail::STATE_NORMALIZATION.residual;
    EXPECT_EQ(std::string(detail::DENSITY_NORMALIZATION.residual).find("max"),
              std::string::npos)
        << "a trace is one number and takes no maximum, got: "
        << detail::DENSITY_NORMALIZATION.residual;
}

TEST(V11231NormalizationMessage, MatrixResidualsStillClaimTheirMaximum) {
    EXPECT_NE(std::string(detail::UNITARITY.residual).find("max"), std::string::npos)
        << "unitarity does reduce over a matrix, so it keeps its maximum";
    EXPECT_NE(std::string(detail::KRAUS_TRACE_PRESERVING.residual).find("max"),
              std::string::npos);
}

TEST(V11231NormalizationMessage, TheRenderedMessageCarriesThePropertyResidual) {
    Statevector sv(1);
    const std::vector<Complex128> bad = {Complex128(kScale, 0.0), Complex128(0.0, 0.0)};

    std::string message;
    try {
        sv.set_amplitudes(bad);
        FAIL() << "the default policy must reject this hand-over";
    } catch (const std::invalid_argument& e) {
        message = e.what();
    }

    EXPECT_NE(message.find(detail::STATE_NORMALIZATION.subject), std::string::npos)
        << "got: " << message;
    EXPECT_NE(message.find(detail::STATE_NORMALIZATION.residual), std::string::npos)
        << "the message spells the residual the property declares, got: " << message;
    EXPECT_EQ(message.find("max |"), std::string::npos)
        << "no maximum is taken over a scalar, got: " << message;
}

TEST(V11231NormalizationMessage, DensityMatrixReportsItsOwnResidual) {
    DensityMatrix rho(1);
    rho.data[0] = Complex128(kScale, 0.0);

    std::string message;
    try {
        rho.check_normalized();
        FAIL() << "the default policy must reject an off-normalization trace";
    } catch (const std::invalid_argument& e) {
        message = e.what();
    }

    EXPECT_NE(message.find(detail::DENSITY_NORMALIZATION.residual), std::string::npos)
        << "a density matrix reports its trace, not a state norm, got: " << message;
    EXPECT_EQ(message.find(detail::STATE_NORMALIZATION.residual), std::string::npos)
        << "the two properties are separate residuals over different objects";
}
