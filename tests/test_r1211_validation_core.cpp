// R.1.21.1 test wave - the Class C physical-validity framework's own core.
//
// Everything here drives detail/validate_physical.hpp directly rather than
// through a backend, so a residual or a policy decision is pinned once at the
// place it is computed instead of once per entry point. The per-backend files
// then only have to establish that each primitive reaches this machinery with
// the caller's options intact.
//
// Three properties are measured (unitarity, Kraus trace preservation,
// superoperator trace preservation) and six policies act on the result, three
// responses each with and without a repair asked for, so the dispatcher is
// covered as a full cross product: every policy against a passing and a failing
// residual, for every property.

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "r1211_reference_residuals.hpp"

#include "lindblad/detail/validate.hpp"
#include "lindblad/detail/validate_physical.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace lindblad;
using detail::DENSITY_NORMALIZATION;
using detail::KRAUS_TRACE_PRESERVING;
using detail::STATE_NORMALIZATION;
using detail::SUPEROP_TRACE_PRESERVING;
using detail::UNITARITY;

namespace {

// -----------------------------------------------------------------------------
// Matrix builders
// -----------------------------------------------------------------------------

std::vector<Complex128> identity_matrix(std::size_t n) {
    std::vector<Complex128> m(n * n);
    for (std::size_t i = 0; i < n; ++i) m[i * n + i] = Complex128(1.0, 0.0);
    return m;
}

// diag(entries), the shape used wherever an exactly-representable deviation is
// wanted: U†U is then diag(|entries|²) and the residual is read off directly.
std::vector<Complex128> diagonal_matrix(const std::vector<Complex128>& entries) {
    const std::size_t n = entries.size();
    std::vector<Complex128> m(n * n);
    for (std::size_t i = 0; i < n; ++i) m[i * n + i] = entries[i];
    return m;
}

std::vector<Complex128> pauli_x_matrix() {
    return {Complex128(0.0, 0.0), Complex128(1.0, 0.0),
            Complex128(1.0, 0.0), Complex128(0.0, 0.0)};
}

std::vector<Complex128> pauli_y_matrix() {
    return {Complex128(0.0, 0.0), Complex128(0.0, -1.0),
            Complex128(0.0, 1.0), Complex128(0.0, 0.0)};
}

std::vector<Complex128> hadamard_matrix() {
    constexpr double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

// -----------------------------------------------------------------------------
// Independent references
// -----------------------------------------------------------------------------
// reference_unitarity_deviation is declared in r1211_reference_residuals.hpp
// and defined in a translation unit compiled with -fno-fast-math. It walks
// every entry of U†U where the library walks only the upper triangle, so
// comparing the two is a direct test of the Hermitian-symmetry argument rather
// than of the arithmetic they share. That comparison is only meaningful if the
// reference is evaluated under IEEE semantics; the header explains what the
// permissive model does to it otherwise.
using r1211ref::reference_unitarity_deviation;

// Builds a superoperator from a Kraus set: S[(ro,co),(ri,ci)] = Σ_k K_k[ro,ri]·
// conj(K_k[co,ci]). A trace-preserving Kraus set therefore yields a
// trace-preserving superoperator, which is what lets one construction feed both
// residual functions and keeps them from being tested against separate fixtures.
std::vector<Complex128> superop_from_kraus(
    const std::vector<std::vector<Complex128>>& ops, std::size_t dim) {
    const std::size_t side = dim * dim;
    std::vector<Complex128> S(side * side);
    for (const auto& K : ops) {
        for (std::size_t ro = 0; ro < dim; ++ro)
            for (std::size_t co = 0; co < dim; ++co)
                for (std::size_t ri = 0; ri < dim; ++ri)
                    for (std::size_t ci = 0; ci < dim; ++ci)
                        S[(ro * dim + co) * side + (ri * dim + ci)] =
                            S[(ro * dim + co) * side + (ri * dim + ci)] +
                            K[ro * dim + ri] * K[co * dim + ci].conj();
    }
    return S;
}

// The amplitude-damping Kraus set at damping probability p. Trace preserving
// for every p in [0,1], and not unitary in any single operator, so it separates
// "trace preserving" from "each operator unitary".
std::vector<std::vector<Complex128>> amplitude_damping(double p) {
    const double s = std::sqrt(1.0 - p);
    return {{Complex128(1.0, 0.0), Complex128(0.0, 0.0),
             Complex128(0.0, 0.0), Complex128(s, 0.0)},
            {Complex128(0.0, 0.0), Complex128(std::sqrt(p), 0.0),
             Complex128(0.0, 0.0), Complex128(0.0, 0.0)}};
}

const char* const CTX = "test_ctx";

// Runs enforce_physical and reports whether it threw, plus the message.
struct DispatchOutcome {
    bool threw = false;
    std::string message;
};

DispatchOutcome dispatch(double deviation, const ValidationOptions& v,
                         const detail::PhysicalProperty& p) {
    DispatchOutcome out;
    try {
        detail::enforce_physical(deviation, v, CTX, p);
    } catch (const std::invalid_argument& e) {
        out.threw = true;
        out.message = e.what();
    }
    return out;
}

// Collects warnings for the duration of a scope and restores the previous sink.
// The channel is global, so a test that installs a handler and does not remove
// it makes every later test order-dependent.
class WarningCapture {
public:
    WarningCapture() {
        set_warning_handler([this](const std::string& m) { lines_.push_back(m); });
    }
    ~WarningCapture() {
        set_warning_handler(nullptr);
        flush_warnings();
    }
    WarningCapture(const WarningCapture&) = delete;
    WarningCapture& operator=(const WarningCapture&) = delete;

    const std::vector<std::string>& lines() const { return lines_; }
    std::size_t count() const { return lines_.size(); }

private:
    std::vector<std::string> lines_;
};

} // namespace

// =============================================================================
// ValidationOptions - the aggregate itself
// =============================================================================

TEST(R1211Options, DefaultsToThrowAtOneEMinusTwelve) {
    const ValidationOptions v;
    EXPECT_EQ(v.policy, Validation::Throw)
        << "the default policy is the loud one; a silent default would make "
           "every unchecked call site a silent one";
    EXPECT_EQ(v.atol, 1e-12);
}

TEST(R1211Options, BracedInitialisationOmittingAtolKeepsTheDefault) {
    const ValidationOptions v{Validation::Warn};
    EXPECT_EQ(v.policy, Validation::Warn);
    EXPECT_EQ(v.atol, 1e-12)
        << "call sites throughout the library write {Validation::Ignore} with "
           "no second member and must not silently get atol 0";
}

TEST(R1211Options, EveryMemberSettable) {
    const ValidationOptions v{Validation::Warn, 1e-6, Repair::Attempt};
    EXPECT_EQ(v.policy, Validation::Warn);
    EXPECT_EQ(v.atol, 1e-6);
    EXPECT_EQ(v.repair, Repair::Attempt)
        << "the repair knob is the third member and must be settable "
           "positionally, since that is how the aggregate is written across "
           "the tree";
}

TEST(R1211Options, IsTriviallyCopyable) {
    // The fusion pre-pass copies an Instruction, and therefore this, per fused
    // block. The header static_asserts it; asserting it here as well means the
    // requirement is stated where a reader looking for the reason will find it.
    EXPECT_TRUE(std::is_trivially_copyable<ValidationOptions>::value);
    EXPECT_TRUE(std::is_standard_layout<ValidationOptions>::value);
}

TEST(R1211Options, CopiesPreserveBothMembers) {
    const ValidationOptions a{Validation::Ignore, 3.5e-9};
    const ValidationOptions b = a;
    EXPECT_EQ(b.policy, Validation::Ignore);
    EXPECT_EQ(b.atol, 3.5e-9);
}

// =============================================================================
// unitarity_deviation
// =============================================================================

TEST(R1211Unitarity, IdentityIsExactlyZero) {
    for (std::size_t n : {std::size_t(1), std::size_t(2), std::size_t(4),
                          std::size_t(8)}) {
        const auto I = identity_matrix(n);
        EXPECT_EQ(detail::unitarity_deviation(I.data(), n), 0.0)
            << "identity at n = " << n
            << " must give a bit-exact zero, not merely a small residual";
    }
}

TEST(R1211Unitarity, PermutationAndSignMatricesAreExactlyZero) {
    const auto X = pauli_x_matrix();
    EXPECT_EQ(detail::unitarity_deviation(X.data(), 2), 0.0);

    const auto Y = pauli_y_matrix();
    EXPECT_EQ(detail::unitarity_deviation(Y.data(), 2), 0.0)
        << "the imaginary entries cancel exactly; conj() must be applied to the "
           "left factor for that to happen";
}

TEST(R1211Unitarity, HadamardSitsFarBelowTheDefaultTolerance) {
    const auto H = hadamard_matrix();
    const double dev = detail::unitarity_deviation(H.data(), 2);
    EXPECT_GT(dev, 0.0)
        << "1/sqrt(2) is not exact in binary, so the residual is nonzero; a "
           "zero here would mean the residual is not being computed";
    EXPECT_LT(dev, 1e-15)
        << "a single gate over irrational amplitudes must land near machine "
           "epsilon, four orders inside the 1e-12 default; measured " << dev;
}

TEST(R1211Unitarity, DiagonalScalingGivesTheExactAnalyticResidual) {
    // U = diag(1.5, 1) -> U†U = diag(2.25, 1) -> max |U†U - I| = 1.25 exactly.
    // Every value here is a dyadic rational, so the expected residual is
    // representable and the comparison can be exact rather than approximate.
    const auto U = diagonal_matrix({Complex128(1.5, 0.0), Complex128(1.0, 0.0)});
    EXPECT_EQ(detail::unitarity_deviation(U.data(), 2), 1.25);

    const auto V = diagonal_matrix({Complex128(0.5, 0.0), Complex128(1.0, 0.0)});
    EXPECT_EQ(detail::unitarity_deviation(V.data(), 2), 0.75);
}

TEST(R1211Unitarity, ResidualIsTheWorstEntryNotTheFirstOrTheSum) {
    // Three deviating entries of different sizes: the answer must be the
    // largest, whichever position it occupies.
    const auto U = diagonal_matrix({Complex128(1.5, 0.0),    // |.|²-1 = 1.25
                                    Complex128(1.25, 0.0),   // 0.5625
                                    Complex128(1.0, 0.0)});  // 0
    EXPECT_EQ(detail::unitarity_deviation(U.data(), 3), 1.25);

    const auto V = diagonal_matrix({Complex128(1.0, 0.0),
                                    Complex128(1.25, 0.0),
                                    Complex128(1.5, 0.0)});
    EXPECT_EQ(detail::unitarity_deviation(V.data(), 3), 1.25)
        << "the worst entry moved to the last position and must still be found";
}

TEST(R1211Unitarity, OffDiagonalNonOrthogonalityIsDetected) {
    // Two identical unit columns: every diagonal entry of U†U is 1, so a check
    // reading only the diagonal would pass this. The off-diagonal is 1.
    const std::vector<Complex128> U{
        Complex128(1.0, 0.0), Complex128(1.0, 0.0),
        Complex128(0.0, 0.0), Complex128(0.0, 0.0)};
    EXPECT_EQ(detail::unitarity_deviation(U.data(), 2), 1.0)
        << "columns that are unit-norm but not orthogonal must be rejected";
}

TEST(R1211Unitarity, GlobalPhaseDoesNotAffectTheResidual) {
    const auto H = hadamard_matrix();
    std::vector<Complex128> phased(H.size());
    // Multiply by i, which is exact in binary floating point.
    for (std::size_t k = 0; k < H.size(); ++k)
        phased[k] = H[k] * Complex128(0.0, 1.0);
    EXPECT_EQ(detail::unitarity_deviation(phased.data(), 2),
              detail::unitarity_deviation(H.data(), 2))
        << "U†U is invariant under U -> e^{iθ}U, and at θ = π/2 that invariance "
           "is exact, so the two residuals must agree bit for bit";
}

TEST(R1211Unitarity, UpperTriangleWalkAgreesWithTheFullMatrix) {
    // The library halves its work by relying on U†U being Hermitian. These
    // matrices span exact, near-exact and badly deviating cases so the shortcut
    // is checked where it could plausibly differ.
    const std::vector<std::vector<Complex128>> two_by_two{
        identity_matrix(2), pauli_x_matrix(), pauli_y_matrix(),
        hadamard_matrix(),
        diagonal_matrix({Complex128(1.5, 0.0), Complex128(0.25, 0.0)}),
        {Complex128(1.0, 0.0), Complex128(1.0, 0.0),
         Complex128(0.0, 0.0), Complex128(0.0, 0.0)},
        {Complex128(0.3, 0.4), Complex128(-0.4, 0.3),
         Complex128(0.4, 0.3), Complex128(0.3, -0.4)}};

    for (std::size_t k = 0; k < two_by_two.size(); ++k) {
        EXPECT_EQ(detail::unitarity_deviation(two_by_two[k].data(), 2),
                  reference_unitarity_deviation(two_by_two[k], 2))
            << "matrix " << k << ": the Hermitian shortcut disagreed with a "
               "full walk of U†U";
    }
}

TEST(R1211Unitarity, UpperTriangleWalkAgreesOnALargerDenseMatrix) {
    // A dense 8x8 with all-distinct entries: no symmetry the shortcut could
    // accidentally exploit, and every off-diagonal of U†U is populated.
    const std::size_t n = 8;
    std::vector<Complex128> U(n * n);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c)
            U[r * n + c] = Complex128(0.1 + 0.03 * static_cast<double>(r * n + c),
                                      -0.2 + 0.017 * static_cast<double>(c));
    EXPECT_EQ(detail::unitarity_deviation(U.data(), n),
              reference_unitarity_deviation(U, n));
}

TEST(R1211Unitarity, SingleEntryMatrixIsHandled) {
    const std::vector<Complex128> unit{Complex128(1.0, 0.0)};
    EXPECT_EQ(detail::unitarity_deviation(unit.data(), 1), 0.0);

    const std::vector<Complex128> half{Complex128(0.5, 0.0)};
    EXPECT_EQ(detail::unitarity_deviation(half.data(), 1), 0.75);
}

// =============================================================================
// kraus_tp_deviation
// =============================================================================

TEST(R1211KrausTP, SingleUnitaryOperatorIsTracePreserving) {
    const std::vector<std::vector<Complex128>> ops{pauli_x_matrix()};
    EXPECT_EQ(detail::kraus_tp_deviation(ops, 2), 0.0)
        << "a one-element Kraus set is trace preserving exactly when its "
           "operator is unitary";
}

TEST(R1211KrausTP, AmplitudeDampingIsTracePreservingAcrossItsRange) {
    for (double p : {0.0, 0.125, 0.25, 0.5, 0.75, 1.0}) {
        const double dev = detail::kraus_tp_deviation(amplitude_damping(p), 2);
        EXPECT_LT(dev, 1e-15)
            << "amplitude damping at p = " << p
            << " is trace preserving analytically; measured " << dev;
    }
}

TEST(R1211KrausTP, ExactlyRepresentableDampingIsBitExactlyTracePreserving) {
    // p = 0 and p = 1 are the only parameters where sqrt(p) and sqrt(1-p) are
    // both exactly representable AND square back exactly, so the operator sum
    // is bit-exactly the identity. Anywhere else the residual is a few ulp,
    // and a tolerance-based assertion there could not tell a correct channel
    // from one wrong by 1e-17. Pinning the two exact parameters separately is
    // what makes the swept test above a statement about the channel rather
    // than about the tolerance it was compared against.
    EXPECT_EQ(detail::kraus_tp_deviation(amplitude_damping(0.0), 2), 0.0);
    EXPECT_EQ(detail::kraus_tp_deviation(amplitude_damping(1.0), 2), 0.0);
}

TEST(R1211KrausTP, NonUnitOperatorSumIsDetected) {
    // A single operator scaled by 1.5: Σ K†K = 2.25·I, residual 1.25 exactly.
    std::vector<Complex128> K = identity_matrix(2);
    for (auto& e : K) e = e * 1.5;
    EXPECT_EQ(detail::kraus_tp_deviation({K}, 2), 1.25);
}

TEST(R1211KrausTP, DroppingOneOperatorBreaksTracePreservation) {
    const auto full = amplitude_damping(0.5);
    EXPECT_LT(detail::kraus_tp_deviation(full, 2), 1e-15);

    const std::vector<std::vector<Complex128>> partial{full[0]};
    EXPECT_NEAR(detail::kraus_tp_deviation(partial, 2), 0.5, 1e-15)
        << "without K1, Σ K†K = diag(1, 0.5) and the residual is the missing "
           "0.5; a check that only summed the first operator would pass";
}

TEST(R1211KrausTP, OffDiagonalDefectIsDetected) {
    // The operator sum is [[1, 0.25], [0.25, 1.0625]]: its (0,0) entry is
    // exactly 1, so a check reading only the leading diagonal entry would
    // accept this. The off-diagonal 0.25 is the largest residual present and
    // must be what is reported.
    const std::vector<Complex128> M{
        Complex128(1.0, 0.0), Complex128(0.25, 0.0),
        Complex128(0.0, 0.0), Complex128(0.0, 0.0)};
    const std::vector<Complex128> N{
        Complex128(0.0, 0.0), Complex128(1.0, 0.0),
        Complex128(0.0, 0.0), Complex128(0.0, 0.0)};
    EXPECT_EQ(detail::kraus_tp_deviation({M, N}, 2), 0.25);
}

TEST(R1211KrausTP, ManyOperatorsAccumulateCorrectly) {
    // Four copies of I/2: Σ K†K = 4 · (I/4) = I exactly.
    std::vector<Complex128> K = identity_matrix(2);
    for (auto& e : K) e = e * 0.5;
    EXPECT_EQ(detail::kraus_tp_deviation({K, K, K, K}, 2), 0.0);
    // Three copies fall short by exactly 0.25.
    EXPECT_EQ(detail::kraus_tp_deviation({K, K, K}, 2), 0.25);
}

TEST(R1211KrausTP, FourDimensionalChannelIsHandled) {
    // The two-qubit identity channel, to establish the dim parameter is used
    // for strides rather than assumed to be 2.
    const std::vector<std::vector<Complex128>> ops{identity_matrix(4)};
    EXPECT_EQ(detail::kraus_tp_deviation(ops, 4), 0.0);
}

// =============================================================================
// superop_tp_deviation
// =============================================================================

TEST(R1211SuperopTP, IdentityChannelSuperoperatorIsTracePreserving) {
    const auto S = superop_from_kraus({identity_matrix(2)}, 2);
    EXPECT_EQ(detail::superop_tp_deviation(S.data(), 2), 0.0);
}

TEST(R1211SuperopTP, UnitaryChannelSuperoperatorIsTracePreserving) {
    for (const auto& U : {pauli_x_matrix(), pauli_y_matrix(), hadamard_matrix()}) {
        const auto S = superop_from_kraus({U}, 2);
        EXPECT_LT(detail::superop_tp_deviation(S.data(), 2), 1e-15);
    }
}

TEST(R1211SuperopTP, AgreesWithTheKrausResidualOnTheSameChannel) {
    // The two residuals measure the same physical property through different
    // representations, so on a channel built from one and converted to the
    // other they must both report trace preservation.
    for (double p : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const auto ops = amplitude_damping(p);
        const auto S = superop_from_kraus(ops, 2);
        EXPECT_LT(detail::kraus_tp_deviation(ops, 2), 1e-15) << "p = " << p;
        EXPECT_LT(detail::superop_tp_deviation(S.data(), 2), 1e-15) << "p = " << p;
    }
}

TEST(R1211SuperopTP, ScaledSuperoperatorIsDetected) {
    auto S = superop_from_kraus({identity_matrix(2)}, 2);
    for (auto& e : S) e = e * 1.5;
    EXPECT_EQ(detail::superop_tp_deviation(S.data(), 2), 0.5)
        << "scaling the whole map by 1.5 makes Σ_r S[(r,r),(i,i)] = 1.5, a "
           "residual of exactly 0.5 on the diagonal input indices";
}

TEST(R1211SuperopTP, DefectOnAnOffDiagonalInputIndexIsDetected) {
    // Σ_ro S[(ro,ro),(ri,ci)] must be zero when ri != ci. Perturbing exactly
    // one such column is invisible to any check that only looks at ri == ci.
    auto S = superop_from_kraus({identity_matrix(2)}, 2);
    const std::size_t dim = 2, side = 4;
    const std::size_t in = 0 * dim + 1;              // (ri, ci) = (0, 1)
    S[(0 * dim + 0) * side + in] = S[(0 * dim + 0) * side + in] +
                                   Complex128(0.25, 0.0);
    EXPECT_EQ(detail::superop_tp_deviation(S.data(), 2), 0.25);
}

TEST(R1211SuperopTP, ZeroMapIsNotTracePreserving) {
    const std::vector<Complex128> S(16, Complex128(0.0, 0.0));
    EXPECT_EQ(detail::superop_tp_deviation(S.data(), 2), 1.0)
        << "the map sending every state to zero destroys all trace and must "
           "read a residual of exactly 1 on the diagonal input indices";
}

TEST(R1211SuperopTP, FourDimensionalSuperoperatorIsHandled) {
    const auto S = superop_from_kraus({identity_matrix(4)}, 4);
    EXPECT_EQ(detail::superop_tp_deviation(S.data(), 4), 0.0)
        << "a two-qubit channel indexes a 16x16 array; the stride must come "
           "from dim rather than being assumed";
}

// =============================================================================
// enforce_physical - the policy cross product
// =============================================================================

TEST(R1211Dispatch, EveryPolicyAcceptsAResidualWithinTolerance) {
    for (const auto& prop : {UNITARITY, KRAUS_TRACE_PRESERVING,
                             SUPEROP_TRACE_PRESERVING}) {
        for (auto policy : {Validation::Throw, Validation::Warn,
                            Validation::Ignore}) {
            for (auto repair : {Repair::None, Repair::Attempt}) {
                WarningCapture capture;
                const auto out = dispatch(1e-15, {policy, 1e-12, repair}, prop);
                EXPECT_FALSE(out.threw)
                    << prop.noun
                    << ": a passing residual must be accepted under every "
                       "policy, including one that asked for a repair";
                EXPECT_EQ(capture.count(), 0u)
                    << prop.noun
                    << ": nothing to warn about when the check passes";
            }
        }
    }
}

TEST(R1211Dispatch, ThrowRejectsAFailingResidual) {
    for (const auto& prop : {UNITARITY, KRAUS_TRACE_PRESERVING,
                             SUPEROP_TRACE_PRESERVING}) {
        const auto out = dispatch(1e-6, {Validation::Throw, 1e-12}, prop);
        EXPECT_TRUE(out.threw) << prop.noun;
        EXPECT_NE(out.message.find(prop.subject), std::string::npos)
            << "the message must name the property that failed; got: "
            << out.message;
    }
}

TEST(R1211Dispatch, WarnReportsOnceAndProceeds) {
    for (const auto& prop : {UNITARITY, KRAUS_TRACE_PRESERVING,
                             SUPEROP_TRACE_PRESERVING}) {
        WarningCapture capture;
        const auto out = dispatch(1e-6, {Validation::Warn, 1e-12}, prop);
        EXPECT_FALSE(out.threw)
            << prop.noun << ": Warn proceeds; throwing would make it Throw";
        ASSERT_EQ(capture.count(), 1u) << prop.noun;
        EXPECT_NE(capture.lines()[0].find(prop.subject), std::string::npos);
    }
}

// Fix has two dispatchers, and which one an entry point calls is the whole of
// whether a repair happens. The pair below states both halves, because either
// alone reads as a claim about the properties rather than about the routing.
//
// enforce_physical is the non-repairing one. It holds no repair for ANY
// property, so under Fix it says so and throws; a Fix that quietly did nothing
// would return an unphysical result under a policy that promised a correction.
// This is a fact about the dispatcher and not about which properties are
// repairable, and the test below covers that.
TEST(R1211Dispatch, TheNonRepairingDispatcherSaysItHoldsNoRepair) {
    for (const auto& prop : {UNITARITY, KRAUS_TRACE_PRESERVING,
                             SUPEROP_TRACE_PRESERVING}) {
        const auto out = dispatch(1e-6, {Validation::Throw, 1e-12, Repair::Attempt}, prop);
        ASSERT_TRUE(out.threw) << prop.noun << ": Fix must not fall through";
        EXPECT_NE(out.message.find("no repair defined"), std::string::npos)
            << "got: " << out.message;
        EXPECT_NE(out.message.find(prop.noun), std::string::npos)
            << "the message must name which property it could not repair; got: "
            << out.message;
    }
}

// enforce_physical_repairable is what an entry point calls when the property
// HAS a repair, and its Fix branch returns true so the caller performs one.
// Three properties reach it: unitarity through the polar projection, and both
// normalizations through division by what the object actually sums to.
//
// Trace preservation is deliberately absent. There is no cheap canonical
// nearest trace-preserving Kraus set: unlike unitarity, which has the polar
// factor as a unique closed-form answer, it is a constrained optimisation over
// the whole operator list. Reporting beats guessing.
TEST(R1211Dispatch, TheRepairingDispatcherHandsFixBackToTheCaller) {
    for (const auto& prop : {UNITARITY, STATE_NORMALIZATION,
                             DENSITY_NORMALIZATION}) {
        EXPECT_TRUE(detail::enforce_physical_repairable(
            1e-6, {Validation::Throw, 1e-12, Repair::Attempt}, CTX, prop))
            << prop.noun
            << ": Fix did not ask the caller to repair a property that has a "
               "repair";
        EXPECT_FALSE(detail::enforce_physical_repairable(
            1e-15, {Validation::Throw, 1e-12, Repair::Attempt}, CTX, prop))
            << prop.noun
            << ": an operand already inside tolerance was sent for repair, "
               "which costs a factorisation and changes nothing";
    }
}

TEST(R1211Dispatch, TheRepairingDispatcherLeavesTheOtherPoliciesAlone) {
    // Only Fix differs between the two dispatchers. Throw still rejects, Warn
    // still reports without repairing, and neither may quietly become the
    // repairing path just because one is available.
    EXPECT_THROW(detail::enforce_physical_repairable(
                     1e-6, {Validation::Throw, 1e-12}, CTX, UNITARITY),
                 std::invalid_argument);
    {
        WarningCapture cap;
        EXPECT_FALSE(detail::enforce_physical_repairable(
            1e-6, {Validation::Warn, 1e-12}, CTX, UNITARITY))
            << "Warn asked for a repair; Warn describes and does not repair";
        EXPECT_GE(cap.lines().size(), 1u) << "Warn reported nothing";
    }
}

TEST(R1211Dispatch, FixMessageDiffersFromTheThrowMessage) {
    const auto fixed = dispatch(1e-6, {Validation::Throw, 1e-12, Repair::Attempt}, UNITARITY);
    const auto thrown = dispatch(1e-6, {Validation::Throw, 1e-12}, UNITARITY);
    ASSERT_TRUE(fixed.threw);
    ASSERT_TRUE(thrown.threw);
    EXPECT_NE(fixed.message, thrown.message)
        << "'Fix has no repair' and 'this is not unitary' are different "
           "diagnoses and must not collapse into one message";
    EXPECT_EQ(thrown.message.find("no repair"), std::string::npos);
}

TEST(R1211Dispatch, IgnoreReachingTheDispatcherStillRejects) {
    // The check entry points return before measuring under Ignore with no
    // repair asked for, so the dispatcher never sees that combination in normal
    // use. If it ever does, treating it as an accept would turn a measured
    // violation into a silent one, so the dispatcher falls through to the
    // throw.
    //
    // RED against issue #123: the dispatcher currently returns here. Ignore
    // reaches this switch only with Repair::None, because Repair::Attempt is
    // settled before it, so this is exactly the combination the fall-through
    // guards.
    const auto out = dispatch(1e-6, {Validation::Ignore, 1e-12}, UNITARITY);
    EXPECT_TRUE(out.threw)
        << "a residual that was measured and found bad must not be discarded "
           "on the strength of the policy that should have skipped measuring it";
}

TEST(R1211Dispatch, ToleranceBoundaryIsInclusive) {
    const double atol = 1e-12;
    EXPECT_FALSE(dispatch(atol, {Validation::Throw, atol}, UNITARITY).threw)
        << "a residual exactly equal to atol is within tolerance";

    const double just_above = std::nextafter(atol, 1.0);
    EXPECT_TRUE(dispatch(just_above, {Validation::Throw, atol}, UNITARITY).threw)
        << "one ulp above atol is outside it; the comparison must not carry "
           "slack beyond the tolerance the caller named";

    const double just_below = std::nextafter(atol, 0.0);
    EXPECT_FALSE(dispatch(just_below, {Validation::Throw, atol}, UNITARITY).threw);
}

TEST(R1211Dispatch, ZeroToleranceAcceptsOnlyAnExactZero) {
    EXPECT_FALSE(dispatch(0.0, {Validation::Throw, 0.0}, UNITARITY).threw);
    EXPECT_TRUE(dispatch(std::numeric_limits<double>::denorm_min(),
                         {Validation::Throw, 0.0}, UNITARITY).threw)
        << "atol 0 means exact; the smallest subnormal is not exact";
}

TEST(R1211Dispatch, InfiniteToleranceAcceptsEveryFiniteResidual) {
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(dispatch(1e300, {Validation::Throw, inf}, UNITARITY).threw);
    EXPECT_FALSE(dispatch(inf, {Validation::Throw, inf}, UNITARITY).threw)
        << "inf <= inf holds, so an infinite residual under an infinite "
           "tolerance is accepted rather than throwing";
}

TEST(R1211Dispatch, NanResidualFailsUnderEveryTolerance) {
    // This is what the !(dev <= tol) spelling buys: a NaN compares false
    // against everything, so writing the test as (dev > tol) would accept it.
    const double nan = quiet_nan_strict();
    for (double atol : {0.0, 1e-12, 1.0, std::numeric_limits<double>::infinity()}) {
        EXPECT_TRUE(dispatch(nan, {Validation::Throw, atol}, UNITARITY).threw)
            << "NaN residual accepted at atol = " << atol;
    }
}

TEST(R1211Dispatch, InfiniteResidualFailsUnderAFiniteTolerance) {
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(dispatch(inf, {Validation::Throw, 1e-12}, UNITARITY).threw);
}

TEST(R1211Dispatch, NanResidualUnderWarnReportsRatherThanThrowing) {
    WarningCapture capture;
    const auto out = dispatch(quiet_nan_strict(), {Validation::Warn, 1e-12},
                              UNITARITY);
    EXPECT_FALSE(out.threw);
    ASSERT_EQ(capture.count(), 1u)
        << "a NaN residual is a failure, and under Warn a failure is reported";
}

// =============================================================================
// Message formatting
// =============================================================================

TEST(R1211Messages, CarriesContextResidualNameValueAndTolerance) {
    const auto out = dispatch(2.5e-7, {Validation::Throw, 1e-12}, UNITARITY);
    ASSERT_TRUE(out.threw);
    EXPECT_NE(out.message.find(CTX), std::string::npos)
        << "without the context a caller cannot tell which gate failed; got: "
        << out.message;
    EXPECT_NE(out.message.find("U†U - I"), std::string::npos)
        << "the residual must be named so the number can be interpreted; got: "
        << out.message;
    EXPECT_NE(out.message.find("2.5e-07"), std::string::npos)
        << "got: " << out.message;
    EXPECT_NE(out.message.find("1e-12"), std::string::npos)
        << "got: " << out.message;
}

TEST(R1211Messages, SmallResidualsKeepSignificantDigits) {
    // std::to_string would render every one of these as "0.000000", which is
    // exactly the information a reader needs in order to choose a tolerance.
    // format_residual exists for this and nothing else pins it.
    struct Case { double value; const char* expect; };
    const Case cases[]{{1e-9, "1e-09"},
                       {3.75e-14, "3.75e-14"},
                       {1.5e-300, "1.5e-300"},
                       {2.0, "2"}};
    for (const auto& c : cases) {
        const std::string s = detail::format_residual(c.value);
        EXPECT_NE(s.find(c.expect), std::string::npos)
            << "format_residual(" << c.value << ") gave '" << s
            << "', which does not contain '" << c.expect << "'";
        EXPECT_NE(s, "0.000000");
    }
}

TEST(R1211Messages, NonFiniteResidualsAreRendered) {
    const std::string nan_text = detail::format_residual(quiet_nan_strict());
    EXPECT_FALSE(nan_text.empty());
    EXPECT_EQ(nan_text.find("0.000000"), std::string::npos)
        << "a NaN residual must not print as a small number; got: " << nan_text;

    const auto out = dispatch(quiet_nan_strict(), {Validation::Throw, 1e-12},
                              UNITARITY);
    ASSERT_TRUE(out.threw);
    EXPECT_NE(out.message.find(UNITARITY.subject), std::string::npos);
}

TEST(R1211Messages, EachPropertyNamesItsOwnResidual) {
    EXPECT_NE(dispatch(1.0, {Validation::Throw, 1e-12},
                       KRAUS_TRACE_PRESERVING).message.find("Σ K†K - I"),
              std::string::npos);
    EXPECT_NE(dispatch(1.0, {Validation::Throw, 1e-12},
                       SUPEROP_TRACE_PRESERVING).message.find("S[(r,r),(i,j)]"),
              std::string::npos);
}

TEST(R1211Messages, ThrownTypeIsInvalidArgument) {
    // The project's convention is out_of_range for bounds and invalid_argument
    // for structure and physical validity. A caller catching the documented
    // type must not have a Class C failure escape as something else.
    const auto U = diagonal_matrix({Complex128(1.5, 0.0), Complex128(1.0, 0.0)});
    EXPECT_THROW(detail::check_unitary(U, 2, ValidationOptions{}, CTX),
                 std::invalid_argument);
}

// =============================================================================
// Check entry points - guards and the Ignore short circuit
// =============================================================================

TEST(R1211Guards, IgnoreSkipsAnObviouslyBadOperand) {
    const auto U = diagonal_matrix({Complex128(9.0, 0.0), Complex128(1.0, 0.0)});
    EXPECT_NO_THROW(detail::check_unitary(U, 2, {Validation::Ignore}, CTX));

    std::vector<Complex128> K = identity_matrix(2);
    for (auto& e : K) e = e * 4.0;
    EXPECT_NO_THROW(detail::check_kraus_tp({K}, 2, {Validation::Ignore}, CTX));

    const std::vector<Complex128> S(16, Complex128(0.0, 0.0));
    EXPECT_NO_THROW(detail::check_superop_tp(S, 2, {Validation::Ignore}, CTX));
}

TEST(R1211Guards, IgnoreDoesNotReadTheOperand) {
    // Ignore returns before dereferencing, which is what makes it safe to pass
    // at a call site whose matrix has not been sized yet.
    EXPECT_NO_THROW(
        detail::check_unitary(nullptr, 4, {Validation::Ignore}, CTX));
}

TEST(R1211Guards, ZeroSizedOperandsReturnWithoutMeasuring) {
    EXPECT_NO_THROW(detail::check_unitary(nullptr, 0, ValidationOptions{}, CTX));

    const std::vector<Complex128> empty;
    EXPECT_NO_THROW(detail::check_superop_tp(empty, 0, ValidationOptions{}, CTX));
    // dim == 0 means there is no operand to measure against, which is distinct
    // from an empty operator list at a real dimension; that case is
    // R1211Guards.EmptyKrausSetIsRejected.
    EXPECT_NO_THROW(detail::check_kraus_tp({}, 0, ValidationOptions{}, CTX));
}

TEST(R1211Guards, EmptyKrausSetIsRejected) {
    // Issue #76. Sum over no operators is the zero matrix, so an empty set
    // fuses to an all-zero superoperator and drives the trace to 0.
    //
    // The rejection is Class B, not Class C: a list with no operators is a
    // malformed argument rather than a channel whose physics is off, so it is
    // refused where it is supplied and without consulting a policy.
    EXPECT_THROW(detail::check_kraus_nonempty(0, CTX), std::invalid_argument)
        << "issue #76: an empty Kraus set is accepted";
    EXPECT_NO_THROW(detail::check_kraus_nonempty(1, CTX));
}

TEST(R1211Guards, EmptyKrausSetIsNotMeasuredAsAResidual) {
    // The Class C residual deliberately does not own this case. Measuring it
    // would report a deviation of exactly 1 against the trace-preservation
    // tolerance, describing a structural defect as suppressible physics.
    EXPECT_NO_THROW(detail::check_kraus_tp({}, 2, ValidationOptions{}, CTX));
    EXPECT_NO_THROW(detail::check_kraus_tp({}, 2, {Validation::Ignore}, CTX));
}

TEST(R1211Guards, VectorOverloadMatchesThePointerOverload) {
    const auto U = diagonal_matrix({Complex128(1.5, 0.0), Complex128(1.0, 0.0)});
    EXPECT_THROW(detail::check_unitary(U, 2, ValidationOptions{}, CTX),
                 std::invalid_argument);
    EXPECT_THROW(detail::check_unitary(U.data(), 2, ValidationOptions{}, CTX),
                 std::invalid_argument);
}

TEST(R1211Guards, CheckEntryPointsAcceptValidOperandsUnderThrow) {
    EXPECT_NO_THROW(
        detail::check_unitary(hadamard_matrix(), 2, ValidationOptions{}, CTX));
    EXPECT_NO_THROW(
        detail::check_kraus_tp(amplitude_damping(0.3), 2, ValidationOptions{}, CTX));
    const auto S = superop_from_kraus(amplitude_damping(0.3), 2);
    EXPECT_NO_THROW(detail::check_superop_tp(S, 2, ValidationOptions{}, CTX));
}

// =============================================================================
// Non-finite operands
// =============================================================================
// A matrix carrying a non-finite entry is what these checks exist to catch, and
// the tests below assert that at every position a non-finite entry can occupy.
//
// They currently FAIL for every position outside the final column, which is
// issue #73. The running maximum inside each residual is written
// `if (!(d <= worst_sq)) worst_sq = d;`. That spelling is NaN-safe for a single
// comparison against a fixed tolerance, which is what enforce_physical does,
// but not for a running maximum: once worst_sq holds NaN, every later
// `d <= worst_sq` is false, so the negation is true and the next finite entry
// overwrites the NaN. Detection therefore survives only when the non-finite
// entry is the last one examined, which is the final column.
//
// The contract is what the header states, and these hold the implementation to
// it at every index rather than at the one an ordering accident happens to
// reach.

TEST(R1211NonFinite, NanIsDetectedAtEveryPosition) {
    // The residual for the index pair (i, j) is NaN whenever column i or column
    // j holds a non-finite entry, so every position below genuinely produces a
    // NaN somewhere in the measurement. Whether it survives to the returned
    // value is the question.
    for (std::size_t rows : {std::size_t(2), std::size_t(4), std::size_t(8)}) {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < rows; ++c) {
                std::vector<Complex128> U = identity_matrix(rows);
                U[r * rows + c] = Complex128(quiet_nan_strict(), 0.0);

                ASSERT_TRUE(std::isnan(reference_unitarity_deviation(U, rows)))
                    << "rows=" << rows << " r=" << r << " c=" << c
                    << ": the independent reference does not see a NaN here, so "
                       "this fixture is wrong rather than the library";

                EXPECT_TRUE(std::isnan(detail::unitarity_deviation(U.data(), rows)))
                    << "rows=" << rows << " r=" << r << " c=" << c
                    << ": a NaN at this position is not reported. Issue #73: "
                       "the running maximum is displaced by the next finite "
                       "entry, so only the final column is guarded.";
            }
        }
    }
}

TEST(R1211NonFinite, NanIsRejectedAtEveryPosition) {
    // The same statement one level up, at the entry point a caller actually
    // reaches, under the default Throw policy.
    for (std::size_t rows : {std::size_t(2), std::size_t(4)}) {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < rows; ++c) {
                std::vector<Complex128> U = identity_matrix(rows);
                U[r * rows + c] = Complex128(quiet_nan_strict(), 0.0);
                EXPECT_THROW(detail::check_unitary(U, rows, ValidationOptions{}, CTX),
                             std::invalid_argument)
                    << "rows=" << rows << " r=" << r << " c=" << c
                    << ": accepted a matrix carrying a NaN (issue #73)";
            }
        }
    }
}

TEST(R1211NonFinite, SingleQubitMatrixWithANanIsRejected) {
    // Issue #73 at the size a caller is most likely to hit: a one-qubit gate
    // matrix holding a NaN in its first entry.
    std::vector<Complex128> U = identity_matrix(2);
    U[0] = Complex128(quiet_nan_strict(), 0.0);

    EXPECT_TRUE(std::isnan(detail::unitarity_deviation(U.data(), 2)))
        << "issue #73: the NaN is displaced by the finite (1,1) entry";
    EXPECT_THROW(detail::check_unitary(U, 2, ValidationOptions{}, CTX),
                 std::invalid_argument)
        << "issue #73: a single-qubit matrix carrying a NaN is accepted under "
           "the policy whose entire purpose is to reject it";
}

TEST(R1211NonFinite, InfinityIsDetectedAtEveryPosition) {
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            std::vector<Complex128> U = identity_matrix(2);
            U[r * 2 + c] = Complex128(std::numeric_limits<double>::infinity(), 0.0);
            EXPECT_THROW(detail::check_unitary(U, 2, ValidationOptions{}, CTX),
                         std::invalid_argument)
                << "r=" << r << " c=" << c
                << ": accepted a matrix carrying an infinity";
        }
    }
}

TEST(R1211NonFinite, KrausNanIsDetectedAtEveryPosition) {
    // kraus_tp_deviation carries the same running maximum, so it carries the
    // same defect. Issue #73 covers all three residual functions.
    for (std::size_t k = 0; k < 4; ++k) {
        std::vector<Complex128> K = identity_matrix(2);
        K[k] = Complex128(quiet_nan_strict(), 0.0);
        EXPECT_THROW(detail::check_kraus_tp({K}, 2, ValidationOptions{}, CTX),
                     std::invalid_argument)
            << "entry " << k << ": accepted a Kraus operator carrying a NaN "
               "(issue #73)";
    }
}

TEST(R1211NonFinite, SuperoperatorNanIsDetectedAtEveryInputIndex) {
    // And the third. The input index pair (ri, ci) selects which column of the
    // output-diagonal sum is measured, so sweeping it sweeps the positions
    // whose NaN must survive to the returned value.
    for (std::size_t ri = 0; ri < 2; ++ri) {
        for (std::size_t ci = 0; ci < 2; ++ci) {
            auto S = superop_from_kraus({identity_matrix(2)}, 2);
            const std::size_t dim = 2, side = 4;
            S[(0 * dim + 0) * side + (ri * dim + ci)] =
                Complex128(quiet_nan_strict(), 0.0);
            EXPECT_THROW(detail::check_superop_tp(S, 2, ValidationOptions{}, CTX),
                         std::invalid_argument)
                << "input index (" << ri << ", " << ci
                << "): accepted a superoperator carrying a NaN (issue #73)";
        }
    }
}

TEST(R1211NonFinite, AnAllNanOperandIsAlwaysCaught) {
    // Every entry is NaN, so no finite value exists to displace the running
    // maximum and the position dependence in issue #73 cannot apply. This is
    // the case the guard was shaped for and it must hold regardless.
    const std::vector<Complex128> U(4, Complex128(quiet_nan_strict(), 0.0));
    EXPECT_THROW(detail::check_unitary(U, 2, ValidationOptions{}, CTX),
                 std::invalid_argument);

    const std::vector<Complex128> K(4, Complex128(quiet_nan_strict(), 0.0));
    EXPECT_THROW(detail::check_kraus_tp({K}, 2, ValidationOptions{}, CTX),
                 std::invalid_argument);
}

TEST(R1211NonFinite, IgnoreSkipsNonFiniteOperandsToo) {
    // Ignore returns before measuring, so it is unaffected by anything above.
    const std::vector<Complex128> U(4, Complex128(quiet_nan_strict(), 0.0));
    EXPECT_NO_THROW(detail::check_unitary(U, 2, {Validation::Ignore}, CTX));
}
