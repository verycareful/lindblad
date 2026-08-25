// R.1.21.1 test wave - Class C across the three qudit backends.
//
// Nine entry points here take a caller-supplied operator: three on
// QuditStatevector, three on QuditMPS, and four on QuditDensityMatrix (two
// unitary, two Kraus). The qudit layer has no circuit type and therefore no
// pre-flight, so unlike the qubit backends the primitive call is the only place
// a check can happen. That makes per-primitive coverage here load-bearing
// rather than redundant.
//
// Everything is exercised at d = 3 by default, with d = 2 and d = 5 spot checks
// where the dimension enters an index calculation. A d = 2 qudit is a qubit, so
// running the same operand at d = 2 is what catches a residual that silently
// assumed the qubit case.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;
using r1211::expect_accepts_valid;
using r1211::expect_rejects_invalid;
using r1211::expect_tolerance_is_honoured;

namespace {

std::vector<Complex128> identity_matrix(std::size_t n) {
    std::vector<Complex128> m(n * n);
    for (std::size_t i = 0; i < n; ++i) m[i * n + i] = Complex128(1.0, 0.0);
    return m;
}

// The generalised X (shift) gate at dimension d: |x> -> |x+1 mod d>. A
// permutation matrix, so it is exactly unitary at every d with no rounding.
std::vector<Complex128> shift_gate(int d) {
    std::vector<Complex128> m(static_cast<std::size_t>(d * d));
    for (int x = 0; x < d; ++x)
        m[static_cast<std::size_t>(((x + 1) % d) * d + x)] = Complex128(1.0, 0.0);
    return m;
}

// The identity scaled by 1.5: deviation exactly 1.25 at every dimension.
std::vector<Complex128> scaled_identity(std::size_t n, double s) {
    auto m = identity_matrix(n);
    for (auto& e : m) e = e * s;
    return m;
}

std::vector<Complex128> bad_matrix(std::size_t n) { return scaled_identity(n, 1.5); }

std::vector<Complex128> deviating_matrix(std::size_t n, double deviation) {
    auto m = identity_matrix(n);
    m[0] = Complex128(std::sqrt(1.0 + deviation), 0.0);   // scale² - 1 == deviation
    return m;
}

// A trace-preserving qudit channel: a probabilistic mixture of the identity
// and the shift, weights summing to 1. No operator here is unitary on its own.
std::vector<std::vector<Complex128>> good_kraus(int d, double p = 0.25) {
    auto keep = identity_matrix(static_cast<std::size_t>(d));
    for (auto& e : keep) e = e * std::sqrt(1.0 - p);
    auto jump = shift_gate(d);
    for (auto& e : jump) e = e * std::sqrt(p);
    return {keep, jump};
}

// The same mixture with the weights left off, so the operators sum to 2·I.
std::vector<std::vector<Complex128>> bad_kraus(int d) {
    return {identity_matrix(static_cast<std::size_t>(d)), shift_gate(d)};
}

std::size_t pow_size(int d, int k) {
    std::size_t r = 1;
    for (int i = 0; i < k; ++i) r *= static_cast<std::size_t>(d);
    return r;
}

} // namespace

// =============================================================================
// QuditStatevector
// =============================================================================

TEST(R1211QuditSv, Apply1quditPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditStatevector::apply_1qudit", [](ValidationOptions v) {
        QuditStatevector sv(4, d);
        sv.apply_1qudit(1, bad_matrix(pow_size(d, 1)), v);
    });
    expect_accepts_valid("QuditStatevector::apply_1qudit", [](ValidationOptions v) {
        QuditStatevector sv(4, d);
        sv.apply_1qudit(1, shift_gate(d), v);
    });
}

TEST(R1211QuditSv, Apply2quditPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditStatevector::apply_2qudit", [](ValidationOptions v) {
        QuditStatevector sv(4, d);
        sv.apply_2qudit(0, 2, bad_matrix(pow_size(d, 2)), v);
    });
    expect_accepts_valid("QuditStatevector::apply_2qudit", [](ValidationOptions v) {
        QuditStatevector sv(4, d);
        sv.apply_2qudit(0, 2, identity_matrix(pow_size(d, 2)), v);
    });
}

TEST(R1211QuditSv, ApplyKquditPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditStatevector::apply_kqudit", [](ValidationOptions v) {
        QuditStatevector sv(4, d);
        sv.apply_kqudit({0, 1, 3}, bad_matrix(pow_size(d, 3)), v);
    });
    expect_accepts_valid("QuditStatevector::apply_kqudit", [](ValidationOptions v) {
        QuditStatevector sv(4, d);
        sv.apply_kqudit({0, 1, 3}, identity_matrix(pow_size(d, 3)), v);
    });
}

TEST(R1211QuditSv, CheckHoldsAcrossDimensions) {
    // d = 2 is the case that would pass if a residual had quietly assumed the
    // qubit dimension; d = 5 is the case that would pass if it had assumed a
    // fixed larger one.
    for (int d : {2, 3, 5}) {
        {
            QuditStatevector sv(3, d);
            EXPECT_THROW(
                sv.apply_1qudit(0, bad_matrix(pow_size(d, 1)), ValidationOptions{}),
                std::invalid_argument) << "d = " << d;
        }
        {
            QuditStatevector sv(3, d);
            EXPECT_NO_THROW(sv.apply_1qudit(0, shift_gate(d), ValidationOptions{}))
                << "the shift gate is a permutation and exactly unitary at d = "
                << d;
        }
    }
}

TEST(R1211QuditSv, ToleranceIsConsultedRatherThanFixed) {
    expect_tolerance_is_honoured(
        "QuditStatevector::apply_1qudit atol",
        [](ValidationOptions v) {
            QuditStatevector sv(3, 3);
            sv.apply_1qudit(0, deviating_matrix(3, 1e-9), v);
        },
        1e-9);
}

TEST(R1211QuditSv, IgnoreStillAppliesTheOperator) {
    QuditStatevector sv(2, 3);
    sv.apply_1qudit(0, bad_matrix(3), {Validation::Ignore});
    EXPECT_NEAR(sv.norm_sq(), 2.25, 1e-12)
        << "the 1.5-scaled operator was not applied under Ignore";
}

TEST(R1211QuditSv, StructureCheckSurvivesIgnore) {
    QuditStatevector sv(2, 3);
    std::vector<Complex128> wrong_size(4, Complex128(1.0, 0.0));   // d=3 needs 9
    EXPECT_THROW(sv.apply_1qudit(0, wrong_size, {Validation::Ignore}),
                 std::invalid_argument)
        << "Ignore opts out of physical validity only";
}

// =============================================================================
// QuditMPS
// =============================================================================

TEST(R1211QuditMps, Apply1quditPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditMPS::apply_1qudit", [](ValidationOptions v) {
        QuditMPS mps(4, d);
        mps.apply_1qudit(1, bad_matrix(pow_size(d, 1)), v);
    });
    expect_accepts_valid("QuditMPS::apply_1qudit", [](ValidationOptions v) {
        QuditMPS mps(4, d);
        mps.apply_1qudit(1, shift_gate(d), v);
    });
}

TEST(R1211QuditMps, Apply2quditAdjacentPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditMPS::apply_2qudit_adjacent", [](ValidationOptions v) {
        QuditMPS mps(4, d);
        mps.apply_2qudit_adjacent(1, bad_matrix(pow_size(d, 2)), v);
    });
    expect_accepts_valid("QuditMPS::apply_2qudit_adjacent", [](ValidationOptions v) {
        QuditMPS mps(4, d);
        mps.apply_2qudit_adjacent(1, identity_matrix(pow_size(d, 2)), v);
    });
}

TEST(R1211QuditMps, Apply2quditPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditMPS::apply_2qudit", [](ValidationOptions v) {
        QuditMPS mps(4, d);
        mps.apply_2qudit(0, 3, bad_matrix(pow_size(d, 2)), v);
    });
    expect_accepts_valid("QuditMPS::apply_2qudit", [](ValidationOptions v) {
        QuditMPS mps(4, d);
        mps.apply_2qudit(0, 3, identity_matrix(pow_size(d, 2)), v);
    });
}

TEST(R1211QuditMps, NonAdjacentAndReversedOperandsAreStillChecked) {
    // Both orders route through a SWAP chain, and every internal hop is a
    // further opportunity for the policy to be dropped on the way.
    const int d = 3;
    for (auto pair : {std::pair<int, int>{0, 3}, std::pair<int, int>{3, 0}}) {
        QuditMPS mps(4, d);
        EXPECT_THROW(mps.apply_2qudit(pair.first, pair.second,
                                      bad_matrix(pow_size(d, 2)),
                                      ValidationOptions{}),
                     std::invalid_argument)
            << "operands (" << pair.first << ", " << pair.second << ")";
    }
}

TEST(R1211QuditMps, CheckHoldsAcrossDimensions) {
    for (int d : {2, 3, 5}) {
        QuditMPS mps(3, d);
        EXPECT_THROW(mps.apply_1qudit(0, bad_matrix(pow_size(d, 1)),
                                      ValidationOptions{}),
                     std::invalid_argument) << "d = " << d;
    }
}

// =============================================================================
// QuditDensityMatrix - unitary entry points
// =============================================================================

TEST(R1211QuditDm, Apply1quditPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditDensityMatrix::apply_1qudit", [](ValidationOptions v) {
        QuditDensityMatrix rho(3, d);
        rho.apply_1qudit(1, bad_matrix(pow_size(d, 1)), v);
    });
    expect_accepts_valid("QuditDensityMatrix::apply_1qudit", [](ValidationOptions v) {
        QuditDensityMatrix rho(3, d);
        rho.apply_1qudit(1, shift_gate(d), v);
    });
}

TEST(R1211QuditDm, Apply2quditPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditDensityMatrix::apply_2qudit", [](ValidationOptions v) {
        QuditDensityMatrix rho(3, d);
        rho.apply_2qudit(0, 2, bad_matrix(pow_size(d, 2)), v);
    });
    expect_accepts_valid("QuditDensityMatrix::apply_2qudit", [](ValidationOptions v) {
        QuditDensityMatrix rho(3, d);
        rho.apply_2qudit(0, 2, identity_matrix(pow_size(d, 2)), v);
    });
}

TEST(R1211QuditDm, IgnoreStillEvolvesTheState) {
    QuditDensityMatrix rho(2, 3);
    rho.apply_1qudit(0, bad_matrix(3), {Validation::Ignore});
    EXPECT_NEAR(rho.trace(), 2.25, 1e-12)
        << "rho -> U rho U† with U = 1.5·I scales the trace by 2.25";
}

// =============================================================================
// QuditDensityMatrix - Kraus entry points
// =============================================================================

TEST(R1211QuditDmKraus, Apply1quditPolicyMatrix) {
    const int d = 3;
    expect_rejects_invalid("QuditDensityMatrix::apply_kraus_1qudit",
                           [](ValidationOptions v) {
                               QuditDensityMatrix rho(3, d);
                               rho.apply_kraus_1qudit(1, bad_kraus(d), v);
                           });
    expect_accepts_valid("QuditDensityMatrix::apply_kraus_1qudit",
                         [](ValidationOptions v) {
                             QuditDensityMatrix rho(3, d);
                             rho.apply_kraus_1qudit(1, good_kraus(d), v);
                         });
}

TEST(R1211QuditDmKraus, Apply2quditPolicyMatrix) {
    const int d = 3;
    const std::size_t dim2 = pow_size(d, 2);
    expect_rejects_invalid(
        "QuditDensityMatrix::apply_kraus_2qudit", [dim2](ValidationOptions v) {
            QuditDensityMatrix rho(3, d);
            rho.apply_kraus_2qudit(
                0, 2, {identity_matrix(dim2), identity_matrix(dim2)}, v);
        });
    expect_accepts_valid(
        "QuditDensityMatrix::apply_kraus_2qudit", [dim2](ValidationOptions v) {
            QuditDensityMatrix rho(3, d);
            rho.apply_kraus_2qudit(0, 2, {identity_matrix(dim2)}, v);
        });
}

TEST(R1211QuditDmKraus, MessageNamesTracePreservationNotUnitarity) {
    QuditDensityMatrix rho(2, 3);
    try {
        rho.apply_kraus_1qudit(0, bad_kraus(3), ValidationOptions{});
        FAIL() << "an unweighted Kraus mixture was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("trace preserving"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("not unitary"), std::string::npos)
            << "both operators here are individually unitary; the defect is "
               "that they were not weighted. Got: " << msg;
    }
}

TEST(R1211QuditDmKraus, WeightedMixtureIsAcceptedAcrossItsRange) {
    for (double p : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        QuditDensityMatrix rho(2, 3);
        EXPECT_NO_THROW(rho.apply_kraus_1qudit(0, good_kraus(3, p),
                                               ValidationOptions{}))
            << "a correctly weighted mixture at p = " << p << " was rejected";
        EXPECT_NEAR(rho.trace(), 1.0, 1e-14) << "p = " << p;
    }
}

TEST(R1211QuditDmKraus, NonUnitaryOperatorsDoNotFailAValidChannel) {
    // Every operator in good_kraus is a scaled permutation and so not unitary,
    // and the channel is correct. Enforcing unitarity here would reject the
    // whole qudit noise catalogue.
    QuditDensityMatrix rho(2, 3);
    EXPECT_NO_THROW(rho.apply_kraus_1qudit(0, good_kraus(3, 0.4),
                                           ValidationOptions{}));
}

TEST(R1211QuditDmKraus, CheckHoldsAcrossDimensions) {
    for (int d : {2, 3, 5}) {
        {
            QuditDensityMatrix rho(2, d);
            EXPECT_THROW(rho.apply_kraus_1qudit(0, bad_kraus(d), ValidationOptions{}),
                         std::invalid_argument) << "d = " << d;
        }
        {
            QuditDensityMatrix rho(2, d);
            EXPECT_NO_THROW(rho.apply_kraus_1qudit(0, good_kraus(d, 0.3),
                                                   ValidationOptions{}))
                << "d = " << d;
        }
    }
}
