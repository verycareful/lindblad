// R.1.21.1 test wave - Class C at the density-matrix layer.
//
// This backend is the only one that checks all three physical properties:
// unitarity for apply_gate, Kraus trace preservation for apply_kraus, and the
// superoperator trace-preservation condition for apply_channel_superop. The
// three are separate properties and separate residuals, so a channel can be
// trace preserving without any of its operators being unitary, and each entry
// point must enforce the one that applies to it rather than the nearest
// available.
//
// The superoperator entry point carries a second obligation. Its argument is in
// the external convention (bit b of every sub-index addresses qubits[b]), and
// the backend bridges to an MSB-first internal layout by bit reversal. The
// check runs on the external form, before that bridge, so the condition is
// stated in the convention the caller actually supplied. Checking after the
// bridge would still detect a defect, but it would report residual indices the
// caller cannot map back to their own matrix.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/constants.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using r1211::expect_accepts_valid;
using r1211::expect_rejects_invalid;
using r1211::expect_tolerance_is_honoured;
using r1211::WarningProbe;

namespace {

std::vector<Complex128> identity_matrix(std::size_t n) {
    std::vector<Complex128> m(n * n);
    for (std::size_t i = 0; i < n; ++i) m[i * n + i] = Complex128(1.0, 0.0);
    return m;
}

std::vector<Complex128> good_1q() {
    constexpr double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

// Deviation exactly 1.25, independent of rounding.
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

std::vector<Complex128> bad_2q() {
    auto m = good_2q();
    m[1 * 4 + 3] = Complex128(0.0, 0.0);
    return m;
}

std::vector<Complex128> deviating_1q(double deviation) {
    const double scale = std::sqrt(1.0 + deviation);   // scale² - 1 == deviation
    return {Complex128(scale, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(1.0, 0.0)};
}

// Amplitude damping: trace preserving at every p, and no single operator here
// is unitary, so this separates trace preservation from operator unitarity.
std::vector<std::vector<Complex128>> good_kraus(double p = 0.3) {
    const double keep = std::sqrt(1.0 - p);
    const double lost = std::sqrt(p);
    return {{Complex128(1.0, 0.0), Complex128(0.0, 0.0),
             Complex128(0.0, 0.0), Complex128(keep, 0.0)},
            {Complex128(0.0, 0.0), Complex128(lost, 0.0),
             Complex128(0.0, 0.0), Complex128(0.0, 0.0)}};
}

// The damping set with its jump operator removed: the surviving operator sums
// to diag(1, 1-p) rather than the identity, so trace leaks away.
std::vector<std::vector<Complex128>> bad_kraus(double p = 0.3) {
    return {good_kraus(p)[0]};
}

// Two-qubit depolarising-style set that is not trace preserving: each operator
// is unitary, but they are unweighted, so the sum is 3·I rather than I. This
// is the case that separates the two properties in the other direction.
std::vector<std::vector<Complex128>> bad_kraus_2q() {
    std::vector<std::vector<Complex128>> ops;
    for (int k = 0; k < 3; ++k) ops.push_back(identity_matrix(4));
    return ops;
}

// S[(ro,co),(ri,ci)] = Σ_k K[ro,ri]·conj(K[co,ci]), the external convention the
// backend documents for apply_channel_superop.
std::vector<Complex128> superop_from_kraus(
    const std::vector<std::vector<Complex128>>& ops, std::size_t dim) {
    const std::size_t side = dim * dim;
    std::vector<Complex128> S(side * side);
    for (const auto& K : ops)
        for (std::size_t ro = 0; ro < dim; ++ro)
            for (std::size_t co = 0; co < dim; ++co)
                for (std::size_t ri = 0; ri < dim; ++ri)
                    for (std::size_t ci = 0; ci < dim; ++ci)
                        S[(ro * dim + co) * side + (ri * dim + ci)] =
                            S[(ro * dim + co) * side + (ri * dim + ci)] +
                            K[ro * dim + ri] * K[co * dim + ci].conj();
    return S;
}

std::vector<Complex128> good_superop_1q() {
    return superop_from_kraus(good_kraus(), 2);
}

// Scaled by 1.5, so Σ_ro S[(ro,ro),(ri,ci)] = 1.5·δ and the residual is 0.5.
std::vector<Complex128> bad_superop_1q() {
    auto S = good_superop_1q();
    for (auto& e : S) e = e * 1.5;
    return S;
}

DensityMatrix fresh(int n) {
    DensityMatrix rho(n);
    return rho;
}

} // namespace

// =============================================================================
// apply_gate - unitarity
// =============================================================================

TEST(R1211DmGate, SingleQubitPolicyMatrix) {
    expect_rejects_invalid("DensityMatrix::apply_gate k=1", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_gate(bad_1q(), {1}, v);
    });
    expect_accepts_valid("DensityMatrix::apply_gate k=1", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_gate(good_1q(), {1}, v);
    });
}

TEST(R1211DmGate, TwoQubitPolicyMatrix) {
    expect_rejects_invalid("DensityMatrix::apply_gate k=2", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_gate(bad_2q(), {0, 2}, v);
    });
    expect_accepts_valid("DensityMatrix::apply_gate k=2", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_gate(good_2q(), {0, 2}, v);
    });
}

TEST(R1211DmGate, ToleranceIsConsultedRatherThanFixed) {
    expect_tolerance_is_honoured(
        "DensityMatrix::apply_gate atol",
        [](ValidationOptions v) {
            auto rho = fresh(2);
            rho.apply_gate(deviating_1q(1e-9), {0}, v);
        },
        1e-9);
}

TEST(R1211DmGate, MessageNamesUnitarity) {
    auto rho = fresh(2);
    try {
        rho.apply_gate(bad_1q(), {0}, ValidationOptions{});
        FAIL() << "a non-unitary gate reached the density matrix";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not unitary"), std::string::npos) << msg;
        EXPECT_NE(msg.find("1.25"), std::string::npos) << msg;
    }
}

TEST(R1211DmGate, IgnoreStillEvolvesTheState) {
    // Opting out of the check must not opt out of the work.
    auto rho = fresh(1);
    rho.apply_gate(bad_1q(), {0}, {Validation::Ignore});
    EXPECT_NEAR(rho.trace(), 2.25, 1e-12)
        << "rho -> U rho U† with U = 1.5·I scales the trace by 2.25; if the "
           "trace is still 1 the gate was skipped rather than applied";
}

TEST(R1211DmGate, ValidGateLeavesTheTraceAtOne) {
    auto rho = fresh(2);
    rho.apply_gate(good_1q(), {0}, ValidationOptions{});
    rho.apply_gate(good_2q(), {0, 1}, ValidationOptions{});
    EXPECT_NEAR(rho.trace(), 1.0, 1e-14);
}

// =============================================================================
// apply_kraus - trace preservation
// =============================================================================

TEST(R1211DmKraus, SingleQubitPolicyMatrix) {
    expect_rejects_invalid("DensityMatrix::apply_kraus k=1", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_kraus(bad_kraus(), {1}, v);
    });
    expect_accepts_valid("DensityMatrix::apply_kraus k=1", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_kraus(good_kraus(), {1}, v);
    });
}

TEST(R1211DmKraus, TwoQubitPolicyMatrix) {
    expect_rejects_invalid("DensityMatrix::apply_kraus k=2", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_kraus(bad_kraus_2q(), {0, 2}, v);
    });
    expect_accepts_valid("DensityMatrix::apply_kraus k=2", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_kraus({identity_matrix(4)}, {0, 2}, v);
    });
}

TEST(R1211DmKraus, MessageNamesTracePreservationNotUnitarity) {
    // Every operator in good_kraus is non-unitary and the set is still valid,
    // so reporting a unitarity failure here would name the wrong property and
    // send the caller looking at the wrong thing.
    auto rho = fresh(2);
    try {
        rho.apply_kraus(bad_kraus(), {0}, ValidationOptions{});
        FAIL() << "a non-trace-preserving Kraus set was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("trace preserving"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("not unitary"), std::string::npos)
            << "the operators are legitimately non-unitary; the defect is in "
               "their sum. Got: " << msg;
        EXPECT_NE(msg.find("K†K"), std::string::npos) << msg;
    }
}

TEST(R1211DmKraus, UnitaryOperatorsDoNotExcuseANonTracePreservingSet) {
    // Three copies of the identity: each operator is perfectly unitary and the
    // set still triples the trace.
    auto rho = fresh(3);
    EXPECT_THROW(rho.apply_kraus(bad_kraus_2q(), {0, 1}, ValidationOptions{}),
                 std::invalid_argument);
}

TEST(R1211DmKraus, NonUnitaryOperatorsDoNotFailAValidSet) {
    // The mirror of the test above: no operator in amplitude damping is
    // unitary, and the channel is correct.
    auto rho = fresh(2);
    EXPECT_NO_THROW(rho.apply_kraus(good_kraus(0.4), {0}, ValidationOptions{}));
    EXPECT_NEAR(rho.trace(), 1.0, 1e-14);
}

TEST(R1211DmKraus, TracePreservationHoldsAcrossTheDampingRange) {
    for (double p : {0.0, 0.1, 0.25, 0.5, 0.9, 1.0}) {
        auto rho = fresh(2);
        EXPECT_NO_THROW(rho.apply_kraus(good_kraus(p), {0}, ValidationOptions{}))
            << "amplitude damping at p = " << p << " was rejected";
    }
}

TEST(R1211DmKraus, EmptyChannelIsRejectedRatherThanAnnihilatingTheState) {
    // RED, issue #76. An empty operator list reaches the kernel unchallenged:
    // check_qubits and check_all_distinct pass, the per-operator size loop has
    // no iterations, and check_kraus_tp returns early on ops.empty(). The
    // fused superoperator is then all zeros, so the addressed sub-blocks are
    // zeroed and the trace goes to 0.
    //
    // A state whose trace is 0 is not a state. Producing one silently, from a
    // call that reported success, is the failure mode this whole framework
    // exists to prevent.
    auto rho = fresh(2);
    ASSERT_NEAR(rho.trace(), 1.0, 1e-14);

    EXPECT_THROW(rho.apply_kraus({}, {0}, ValidationOptions{}),
                 std::invalid_argument)
        << "issue #76: an empty Kraus set was accepted";
    EXPECT_NEAR(rho.trace(), 1.0, 1e-14)
        << "issue #76: the state was annihilated by a channel with no "
           "operators; trace is now " << rho.trace();
}

TEST(R1211DmKraus, EmptyChannelUnderIgnoreIsTheCallersChoice) {
    // Ignore means the caller has taken responsibility, so this must keep
    // working however issue #76 is fixed, unless the fix routes the rejection
    // through Class B instead. Either outcome is defensible; what is pinned
    // here is only that the two policies do not behave identically.
    auto rho = fresh(2);
    rho.apply_kraus({}, {0}, {Validation::Ignore});
    SUCCEED() << "trace after an ignored empty channel: " << rho.trace();
}

TEST(R1211DmKraus, IgnoreStillAppliesTheChannel) {
    auto rho = fresh(1);
    rho.apply_kraus(bad_kraus(0.5), {0}, {Validation::Ignore});
    EXPECT_NEAR(rho.trace(), 1.0, 1e-12)
        << "on |0><0| the surviving damping operator is the identity, so the "
           "trace is unchanged; the point is that the call completed";
}

// =============================================================================
// apply_channel_superop - superoperator trace preservation
// =============================================================================

TEST(R1211DmSuperop, SingleQubitPolicyMatrix) {
    expect_rejects_invalid("apply_channel_superop k=1", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_channel_superop(bad_superop_1q(), {1}, v);
    });
    expect_accepts_valid("apply_channel_superop k=1", [](ValidationOptions v) {
        auto rho = fresh(3);
        rho.apply_channel_superop(good_superop_1q(), {1}, v);
    });
}

TEST(R1211DmSuperop, MessageNamesTheSuperoperatorCondition) {
    auto rho = fresh(2);
    try {
        rho.apply_channel_superop(bad_superop_1q(), {0}, ValidationOptions{});
        FAIL() << "a non-trace-preserving superoperator was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("trace preserving"), std::string::npos) << msg;
        EXPECT_NE(msg.find("S[(r,r),(i,j)]"), std::string::npos)
            << "the residual must be stated in superoperator index terms so "
               "the caller can find it in their own array. Got: " << msg;
    }
}

TEST(R1211DmSuperop, DefectOnAnOffDiagonalInputIndexIsCaught) {
    // Σ_ro S[(ro,ro),(ri,ci)] must vanish when ri != ci. A check that only
    // examined the ri == ci columns would pass this.
    auto S = good_superop_1q();
    const std::size_t dim = 2, side = 4;
    const std::size_t in = 0 * dim + 1;
    S[(0 * dim + 0) * side + in] = S[(0 * dim + 0) * side + in] +
                                   Complex128(0.25, 0.0);
    auto rho = fresh(2);
    EXPECT_THROW(rho.apply_channel_superop(S, {0}, ValidationOptions{}),
                 std::invalid_argument);
}

TEST(R1211DmSuperop, CheckedInTheCallersConventionNotTheInternalOne) {
    // The backend bridges the external LSB-first layout to an MSB-first
    // internal one by bit reversal. On two qubits that bridge is a genuine
    // permutation, so a superoperator that is trace preserving in the external
    // convention must be accepted, and the operand order must not matter.
    const auto S = superop_from_kraus({identity_matrix(4)}, 4);
    {
        auto rho = fresh(3);
        EXPECT_NO_THROW(rho.apply_channel_superop(S, {0, 1}, ValidationOptions{}));
    }
    {
        auto rho = fresh(3);
        EXPECT_NO_THROW(rho.apply_channel_superop(S, {1, 0}, ValidationOptions{}))
            << "reversing the operand order changes the bridge but not the "
               "external condition, which is what is being checked";
    }
    {
        auto rho = fresh(3);
        EXPECT_NO_THROW(rho.apply_channel_superop(S, {2, 0}, ValidationOptions{}))
            << "non-adjacent operands exercise a different bridge again";
    }
}

TEST(R1211DmSuperop, AgreesWithApplyKrausOnTheSameChannel) {
    // The two entry points describe one channel two ways. A set that
    // apply_kraus accepts must produce a superoperator apply_channel_superop
    // accepts, and both must leave the trace at 1.
    for (double p : {0.0, 0.3, 0.7, 1.0}) {
        auto via_kraus = fresh(2);
        via_kraus.apply_kraus(good_kraus(p), {0}, ValidationOptions{});

        auto via_superop = fresh(2);
        via_superop.apply_channel_superop(superop_from_kraus(good_kraus(p), 2),
                                          {0}, ValidationOptions{});

        EXPECT_NEAR(via_kraus.trace(), 1.0, 1e-14) << "p = " << p;
        EXPECT_NEAR(via_superop.trace(), 1.0, 1e-14) << "p = " << p;
    }
}

TEST(R1211DmSuperop, RejectedChannelsMatchAcrossBothEntryPoints) {
    // The mirror: a set apply_kraus rejects must yield a superoperator
    // apply_channel_superop also rejects. If the two disagreed, a caller could
    // route around the check by converting representation.
    {
        auto rho = fresh(2);
        EXPECT_THROW(rho.apply_kraus(bad_kraus(0.5), {0}, ValidationOptions{}),
                     std::invalid_argument);
    }
    {
        auto rho = fresh(2);
        EXPECT_THROW(rho.apply_channel_superop(
                         superop_from_kraus(bad_kraus(0.5), 2), {0},
                         ValidationOptions{}),
                     std::invalid_argument);
    }
}

TEST(R1211DmSuperop, IgnoreStillAppliesTheChannel) {
    auto rho = fresh(1);
    rho.apply_channel_superop(bad_superop_1q(), {0}, {Validation::Ignore});
    EXPECT_NEAR(rho.trace(), 1.5, 1e-12)
        << "the superoperator was scaled by 1.5 and must have been applied";
}

// =============================================================================
// Cross-property independence
// =============================================================================

TEST(R1211DmProperties, EachEntryPointEnforcesOnlyItsOwnProperty) {
    // A Kraus set whose operators are non-unitary passes apply_kraus, and a
    // matrix that is unitary passes apply_gate. Enforcing the wrong property
    // at either point would reject correct input, which is the failure mode a
    // default-on check makes expensive.
    {
        auto rho = fresh(2);
        EXPECT_NO_THROW(rho.apply_kraus(good_kraus(0.6), {0}, ValidationOptions{}))
            << "non-unitary Kraus operators are normal and must not be "
               "measured against unitarity";
    }
    {
        auto rho = fresh(2);
        EXPECT_NO_THROW(rho.apply_gate(good_1q(), {0}, ValidationOptions{}));
    }
}

TEST(R1211DmProperties, StructureCheckStillRunsUnderIgnore) {
    auto rho = fresh(2);
    std::vector<Complex128> too_small(2, Complex128(1.0, 0.0));
    EXPECT_THROW(rho.apply_gate(too_small, {0, 1}, {Validation::Ignore}),
                 std::invalid_argument)
        << "Ignore opts out of physical validity only; operand structure is "
           "always checked because it guards memory safety";
}
