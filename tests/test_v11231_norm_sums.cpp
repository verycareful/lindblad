// 1.1.23.1 test wave - the normalizability guard and the residual sums.
//
// Three pieces of 1.1.23.0 that nothing exercised: is_normalizable, which
// decides whether a state can be repaired at all; the two state_norm_sq
// overloads and density_trace_real, which are what every normalization verdict
// is computed from; and the change from a running total to a balanced tree.
//
// The summation test is the one worth reading. Rather than asserting that a
// tree is more accurate, it builds a vector where a single running total
// provably loses every small term, computes that naive sum in the test body to
// show it does, and then shows the library recovers essentially all of it. The
// fixture is exact in binary: every term is a power of two, so the exact answer
// is known and no tolerance enters a test about accuracy.

#include <gtest/gtest.h>

#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

// std::to_string gives a double six decimal places, so every residual here
// would record as "0.000000". format_residual keeps significant digits, and the
// value is printed as well as recorded because RecordProperty reaches only the
// XML report, which an ordinary run does not produce.
void report_residual(const std::string& key, double value) {
    const std::string text = detail::format_residual(value);
    ::testing::Test::RecordProperty(key, text);
    std::cout << "[ MARGIN   ] " << key << " = " << text << std::endl;
}

}  // namespace

// =============================================================================
// is_normalizable - the guard that decides whether a repair exists
// =============================================================================

TEST(V11231Normalizable, RejectsTheTwoStatesWithNoRepair) {
    EXPECT_FALSE(is_normalizable(0.0)) << "a zero state has no direction to keep";
    EXPECT_FALSE(is_normalizable(kNaN))
        << "dividing by NaN spreads it rather than removing it";
    EXPECT_FALSE(is_normalizable(kInf));
    EXPECT_FALSE(is_normalizable(-kInf));
}

TEST(V11231Normalizable, RejectsAnythingAtOrBelowMachineEpsilon) {
    // The floor is derived from the type, not chosen: epsilon is the spacing of
    // doubles at 1.0, so a norm at or below it cannot be told from zero at the
    // scale a normalized state occupies.
    EXPECT_FALSE(is_normalizable(kEps)) << "the boundary itself is rejected";
    EXPECT_FALSE(is_normalizable(kEps / 2.0));
    EXPECT_FALSE(is_normalizable(std::numeric_limits<double>::denorm_min()));
    EXPECT_FALSE(is_normalizable(std::numeric_limits<double>::min()))
        << "the smallest normal double is still far below the floor";

    EXPECT_TRUE(is_normalizable(std::nextafter(kEps, 1.0)))
        << "one representable step above the floor is normalizable";
}

TEST(V11231Normalizable, AcceptsOrdinaryNorms) {
    EXPECT_TRUE(is_normalizable(1.0));
    EXPECT_TRUE(is_normalizable(2.0));
    EXPECT_TRUE(is_normalizable(std::numeric_limits<double>::max()));
}

TEST(V11231Normalizable, RejectsNegativeNorms) {
    // A norm is a magnitude. A negative value reaching here means the caller
    // computed something other than one, and it must not be divided out.
    EXPECT_FALSE(is_normalizable(-1.0));
    EXPECT_FALSE(is_normalizable(-kEps));
}

// is_finite_strict reads the exponent bits rather than asking the FPU, which is
// what keeps it working in a translation unit built with permissive floating
// point where isnan() may be folded to false.
TEST(V11231Normalizable, FiniteStrictSeparatesFiniteFromNot) {
    EXPECT_TRUE(is_finite_strict(0.0));
    EXPECT_TRUE(is_finite_strict(-0.0));
    EXPECT_TRUE(is_finite_strict(1.0));
    EXPECT_TRUE(is_finite_strict(-1.0));
    EXPECT_TRUE(is_finite_strict(std::numeric_limits<double>::max()));
    EXPECT_TRUE(is_finite_strict(std::numeric_limits<double>::denorm_min()));

    EXPECT_FALSE(is_finite_strict(kNaN));
    EXPECT_FALSE(is_finite_strict(kInf));
    EXPECT_FALSE(is_finite_strict(-kInf));
}

// =============================================================================
// state_norm_sq - both overloads
// =============================================================================

TEST(V11231NormSums, InterleavedOverloadSumsSquaredMagnitudes) {
    const std::vector<Complex128> unit = {Complex128(1.0, 0.0), Complex128(0.0, 0.0)};
    EXPECT_DOUBLE_EQ(detail::state_norm_sq(unit.data(), unit.size()), 1.0);

    // 3-4-5, so the squared magnitude is exactly 25 and no rounding enters.
    const std::vector<Complex128> pythag = {Complex128(3.0, 4.0)};
    EXPECT_DOUBLE_EQ(detail::state_norm_sq(pythag.data(), pythag.size()), 25.0);

    const std::vector<Complex128> both = {Complex128(1.0, 0.0), Complex128(0.0, 1.0)};
    EXPECT_DOUBLE_EQ(detail::state_norm_sq(both.data(), both.size()), 2.0)
        << "the imaginary part contributes exactly as the real one does";
}

TEST(V11231NormSums, StructureOfArraysOverloadAgreesWithInterleaved) {
    const std::vector<double> re = {0.5, -0.25, 0.125, 0.0};
    const std::vector<double> im = {0.0, 0.5, -0.25, 0.125};

    std::vector<Complex128> packed;
    packed.reserve(re.size());
    for (std::size_t i = 0; i < re.size(); ++i)
        packed.push_back(Complex128(re[i], im[i]));

    const double soa = detail::state_norm_sq(re.data(), im.data(), re.size());
    const double interleaved = detail::state_norm_sq(packed.data(), packed.size());

    EXPECT_DOUBLE_EQ(soa, interleaved)
        << "the two layouts describe the same state and must measure identically";
}

TEST(V11231NormSums, EmptyStateSumsToZero) {
    EXPECT_DOUBLE_EQ(detail::state_norm_sq(static_cast<const Complex128*>(nullptr), 0),
                     0.0);
    EXPECT_DOUBLE_EQ(detail::state_norm_sq(static_cast<const double*>(nullptr),
                                           static_cast<const double*>(nullptr), 0),
                     0.0);
}

TEST(V11231NormSums, NonFiniteAmplitudePropagatesToTheSum) {
    // The sum does not sanitise: a non-finite state must reach the caller as a
    // non-finite residual, which then fails every comparison it is put through.
    const std::vector<Complex128> bad = {Complex128(kNaN, 0.0), Complex128(1.0, 0.0)};
    EXPECT_FALSE(is_finite_strict(detail::state_norm_sq(bad.data(), bad.size())));
}

// =============================================================================
// density_trace_real
// =============================================================================

TEST(V11231NormSums, DensityTraceSumsTheDiagonalOnly) {
    const std::size_t dim = 2;
    std::vector<Complex128> rho(dim * dim, Complex128(0.0, 0.0));
    rho[0] = Complex128(0.25, 0.0);   // (0,0)
    rho[3] = Complex128(0.75, 0.0);   // (1,1)
    rho[1] = Complex128(9.0, 9.0);    // off diagonal, must be ignored
    rho[2] = Complex128(-9.0, 9.0);

    EXPECT_DOUBLE_EQ(detail::density_trace_real(rho.data(), dim), 1.0);
}

TEST(V11231NormSums, DensityTraceTakesTheRealPart) {
    // A Hermitian matrix has a real diagonal. An imaginary part on it means the
    // caller handed over something that is not one, and the trace it is judged
    // by is still the real part.
    const std::size_t dim = 2;
    std::vector<Complex128> rho(dim * dim, Complex128(0.0, 0.0));
    rho[0] = Complex128(0.5, 7.0);
    rho[3] = Complex128(0.5, -3.0);

    EXPECT_DOUBLE_EQ(detail::density_trace_real(rho.data(), dim), 1.0);
}

TEST(V11231NormSums, DensityTraceOfASingleEntry) {
    const std::vector<Complex128> rho = {Complex128(1.0, 0.0)};
    EXPECT_DOUBLE_EQ(detail::density_trace_real(rho.data(), 1), 1.0);
}

// =============================================================================
// Balanced-tree order, demonstrated rather than asserted
// =============================================================================

// The fixture: one amplitude of magnitude 1, then N-1 amplitudes whose squared
// magnitude is 2^-54. The spacing of doubles at 1.0 is 2^-52, so 1.0 + 2^-54
// rounds back to 1.0: a single running total that starts at the large term
// absorbs none of the small ones and returns exactly 1. Every quantity here is
// a power of two, so the exact answer is known with no rounding anywhere.
TEST(V11231NormSums, TreeOrderRecoversWhatARunningTotalDrops) {
    const std::size_t n = std::size_t(1) << 16;
    const double amp = std::ldexp(1.0, -27);          // amp * amp == 2^-54 exactly
    const double term = amp * amp;
    ASSERT_DOUBLE_EQ(term, std::ldexp(1.0, -54))
        << "the small term must be exact for the rest of this test to mean anything";

    std::vector<Complex128> amps(n, Complex128(amp, 0.0));
    amps[0] = Complex128(1.0, 0.0);

    // What a single accumulator does with this input, computed here so the
    // claim is demonstrated in the test rather than asserted about the library.
    //
    // volatile is load-bearing. This translation unit is built with the
    // project's permissive floating point, under which the compiler may
    // vectorise the loop into several partial accumulators, and a set of
    // partial accumulators is already a small tree: it recovers most of the
    // small mass and the contrast being drawn here disappears. Forcing a
    // load-add-store per term is what makes this one genuine running total.
    volatile double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = amps[i].real * amps[i].real + amps[i].imag * amps[i].imag;
        acc = acc + t;
    }
    const double running = acc;
    ASSERT_DOUBLE_EQ(running, 1.0)
        << "the fixture is built so a running total loses every small term";

    const double measured = detail::state_norm_sq(amps.data(), n);
    const double exact_small = static_cast<double>(n - 1) * term;
    const double recovered = measured - 1.0;

    EXPECT_GT(recovered, 0.0)
        << "a running total returns exactly 1.0 here, so anything above it is "
           "mass the summation order preserved";
    EXPECT_GE(recovered, 0.99 * exact_small)
        << "the tree recovers essentially all of the small mass";
    EXPECT_LE(recovered, 1.01 * exact_small)
        << "and does not invent any";
}

// The same shape reached through the structure-of-arrays overload, since the
// two share the summation but not the term expression.
TEST(V11231NormSums, TreeOrderHoldsOnTheStructureOfArraysOverload) {
    const std::size_t n = std::size_t(1) << 16;
    const double amp = std::ldexp(1.0, -27);
    const double term = amp * amp;

    std::vector<double> re(n, amp);
    std::vector<double> im(n, 0.0);
    re[0] = 1.0;

    // volatile for the same reason as the interleaved twin above.
    volatile double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = re[i] * re[i] + im[i] * im[i];
        acc = acc + t;
    }
    const double running = acc;
    ASSERT_DOUBLE_EQ(running, 1.0);

    const double measured = detail::state_norm_sq(re.data(), im.data(), n);
    const double exact_small = static_cast<double>(n - 1) * term;

    EXPECT_GT(measured - 1.0, 0.0);
    EXPECT_GE(measured - 1.0, 0.99 * exact_small);
    EXPECT_LE(measured - 1.0, 1.01 * exact_small);
}

// The order is fixed by the length alone, so the same input measures the same
// on every call. A parallel reduction would let this move with the free core
// count, which is what a check must never do.
TEST(V11231NormSums, TheSumIsDeterministicAcrossRepeatedCalls) {
    const std::size_t n = std::size_t(1) << 14;
    std::vector<Complex128> amps(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i + 1);
        amps[i] = Complex128(1.0 / x, 1.0 / (x + 0.5));
    }

    const double first = detail::state_norm_sq(amps.data(), n);
    for (int repeat = 0; repeat < 8; ++repeat)
        EXPECT_EQ(detail::state_norm_sq(amps.data(), n), first)
            << "bit-identical on every call, repeat " << repeat;
}

// A normalized state of real size measures 1 to well inside the framework
// tolerance. This is the property the tree order exists to protect: the
// tolerance is absolute, so the measurement must not drift with register size.
TEST(V11231NormSums, ANormalizedStateMeasuresOneAtRegisterSizesThatMatter) {
    for (int nq : {12, 16, 20}) {
        SCOPED_TRACE("qubits = " + std::to_string(nq));
        const std::size_t dim = std::size_t(1) << nq;

        // Uniform superposition: every amplitude is 2^(-nq/2), whose square is
        // exactly 2^-nq when nq is even, so the exact sum is 1.
        const double a = std::ldexp(1.0, -nq / 2);
        std::vector<Complex128> amps(dim, Complex128(a, 0.0));

        const double residual =
            std::abs(detail::state_norm_sq(amps.data(), dim) - 1.0);
        report_residual("exact_residual_n" + std::to_string(nq), residual);
        EXPECT_LE(residual, DEFAULT_PHYSICAL_ATOL)
            << "a correct state must not be reported unnormalized by the "
               "measurement at nq = " << nq;
    }
}

// The sweep above is exact in binary, so it proves correctness and can show no
// drift at all. This one rounds at every step: the amplitudes are unequal and
// not representable, so the sum accumulates real error, and the state is
// normalized through the library before being measured again.
TEST(V11231NormSums, ARoundedNormalizedStateStillMeasuresOne) {
    for (int nq : {12, 16, 20}) {
        SCOPED_TRACE("qubits = " + std::to_string(nq));
        const std::size_t dim = std::size_t(1) << nq;

        std::vector<Complex128> amps(dim);
        for (std::size_t k = 0; k < dim; ++k) {
            // Unequal, non-representable, and spanning several orders of
            // magnitude, so small terms have to survive alongside large ones.
            const double x = static_cast<double>(k + 1);
            amps[k] = Complex128(1.0 / std::sqrt(x), 1.0 / (3.0 * x));
        }

        const double norm = std::sqrt(detail::state_norm_sq(amps.data(), dim));
        ASSERT_TRUE(is_normalizable(norm));
        for (Complex128& a : amps) {
            a.real /= norm;
            a.imag /= norm;
        }

        const double residual =
            std::abs(detail::state_norm_sq(amps.data(), dim) - 1.0);
        report_residual("rounded_residual_n" + std::to_string(nq), residual);
        EXPECT_LE(residual, DEFAULT_PHYSICAL_ATOL)
            << "normalizing then re-measuring must land inside the tolerance "
               "the framework judges by, at nq = " << nq;
    }
}
