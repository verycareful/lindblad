// test_v11241_unitary_repair.cpp - #57, the unitary polar projection behind
// Validation::Fix.
//
// The repair shipped with a single test touching it, and that test asserted the
// behaviour it replaced. Nothing exercised the projection itself, nothing
// exercised its failure contract, and nothing exercised the aliasing guarantee
// the call site depends on.
//
// The suite is in four parts.
//
// PROJECTION. The polar factor is the nearest unitary in the Frobenius sense,
// so "nearest" is an assertable property rather than a description: the squared
// distance from M to its projection is exactly the sum of (sigma - 1)^2 over the
// spectrum, which this suite derives from the same seam the repair uses. Two
// fixtures make that checkable without any tolerance at all, since both have
// the identity as their exact polar factor.
//
// FAILURE. project_to_unitary promises that a refused repair leaves the
// caller's buffer untouched rather than partly written. A caller that asked for
// a repair and did not get one must still hold what it handed over, and the
// only way to see a partial write is to look for it.
//
// ALIASING. QuantumCircuit stores gate matrices in a copy-on-write buffer that
// every Instruction copied from it shares. Repairing in place would rewrite a
// matrix other instructions are reading, so the call site takes a copy on the
// Fix path and only there. That is a correctness guarantee about instructions
// the caller never mentioned, and it was untested.
//
// REACH. Unitarity has a repair, and fourteen of the fifteen entry points that
// check unitarity still report that no repair is defined. Those cases are owed
// by 1.1.24.2 and are RED until it lands. They are written here, against the
// ruled behaviour, because a test that documents the defect is worth more than
// a gap, and because the release that fixes the library should turn them green
// without anyone having to remember they were owed.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/detail/unitary_repair.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

using lindblad::Complex128;
using lindblad::DensityMatrix;
using lindblad::MPSState;
using lindblad::QuantumCircuit;
using lindblad::QuditStatevector;
using lindblad::Statevector;
using lindblad::Validation;
using lindblad::ValidationOptions;
using lindblad::detail::project_to_unitary;
using lindblad::detail::repair_unitary;
using lindblad::detail::unitarity_deviation;
using lindblad::detail::unitary_needs_repair;
using r1211::WarningProbe;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kAtol = lindblad::DEFAULT_PHYSICAL_ATOL;

// A scalar multiple of the identity. Its thin SVD is I * (a I) * I-dagger, so
// the polar factor is exactly the identity whatever `a` is, and the expected
// output of a repair needs no tolerance and no reference implementation.
std::vector<Complex128> scaled_identity(int n, double a) {
    std::vector<Complex128> m(static_cast<size_t>(n) * n, Complex128(0.0, 0.0));
    for (int i = 0; i < n; ++i)
        m[static_cast<size_t>(i) * n + i] = Complex128(a, 0.0);
    return m;
}

// Diagonal, positive, and far from unitary in both directions at once. Polar
// factor is again exactly the identity, and the distance to it is derivable by
// hand: (2-1)^2 + (1/2-1)^2.
std::vector<Complex128> stretched_2x2() {
    return {Complex128(2.0, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0), Complex128(0.5, 0.0)};
}

// A genuine unitary that is not diagonal, so the projection has something to do
// in every entry rather than only on the diagonal.
std::vector<Complex128> hadamard() {
    const double s = lindblad::INV_SQRT2;
    return {Complex128(s, 0.0), Complex128(s, 0.0),
            Complex128(s, 0.0), Complex128(-s, 0.0)};
}

// The Hadamard with one entry stretched. Near a unitary but outside the default
// tolerance by six orders of magnitude, which is the shape a matrix acquires by
// drifting rather than by being wrong.
std::vector<Complex128> drifted_hadamard() {
    auto m = hadamard();
    m[0] = Complex128(m[0].real * (1.0 + 1e-6), 0.0);
    return m;
}

double frob_distance_sq(const std::vector<Complex128>& a,
                        const std::vector<Complex128>& b) {
    double t = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double dr = a[i].real - b[i].real;
        const double di = a[i].imag - b[i].imag;
        t += dr * dr + di * di;
    }
    return t;
}

// Singular values through the same seam the repair uses, so the expected
// distance below is derived from the factorisation rather than transcribed.
std::vector<double> spectrum(const std::vector<Complex128>& m, int n) {
    std::vector<std::complex<double>> in(m.size());
    for (size_t i = 0; i < m.size(); ++i)
        in[i] = std::complex<double>(m[i].real, m[i].imag);
    std::vector<std::complex<double>> U(m.size()), V(m.size());
    std::vector<double> S(static_cast<size_t>(n));
    const bool ok = lindblad::detail::svd_thin(
        in.data(), n, n, lindblad::detail::MatrixOrder::RowMajor,
        lindblad::SVDMethod::Jacobi, U.data(), S.data(), V.data());
    return ok ? S : std::vector<double>{};
}

void expect_identity(const std::vector<Complex128>& m, int n, double tol,
                     const char* what) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const Complex128& z = m[static_cast<size_t>(i) * n + j];
            EXPECT_NEAR(z.real, (i == j) ? 1.0 : 0.0, tol)
                << what << ": real part at (" << i << "," << j << ")";
            EXPECT_NEAR(z.imag, 0.0, tol)
                << what << ": imaginary part at (" << i << "," << j << ")";
        }
}

}  // namespace

// =============================================================================
// project_to_unitary
// =============================================================================

TEST(V11241UnitaryRepair, AScaledIdentityProjectsToTheIdentity) {
    // No tolerance is needed on the expected value: the polar factor of a I is
    // the identity for every a, so this pins the answer and not an
    // approximation of it.
    for (int n : {2, 4, 8}) {
        for (double a : {0.25, 1.5, 7.0}) {
            auto m = scaled_identity(n, a);
            ASSERT_TRUE(project_to_unitary(m.data(), static_cast<size_t>(n)))
                << "n=" << n << " a=" << a;
            expect_identity(m, n, 64.0 * n * kEps, "scaled identity");
        }
    }
}

TEST(V11241UnitaryRepair, AnAlreadyUnitaryMatrixSurvivesTheProjection) {
    auto m = hadamard();
    const auto before = m;
    ASSERT_TRUE(project_to_unitary(m.data(), 2));
    EXPECT_LT(frob_distance_sq(m, before), 64.0 * kEps * kEps)
        << "the projection moved a matrix that was already unitary";
    EXPECT_LE(unitarity_deviation(m.data(), 2), kAtol);
}

TEST(V11241UnitaryRepair, ADriftedMatrixLandsInsideTolerance) {
    auto m = drifted_hadamard();
    ASSERT_GT(unitarity_deviation(m.data(), 2), kAtol)
        << "the fixture is already unitary, so it repairs nothing";
    ASSERT_TRUE(project_to_unitary(m.data(), 2));
    EXPECT_LE(unitarity_deviation(m.data(), 2), kAtol)
        << "the whole postcondition of the repair is that this holds";
}

TEST(V11241UnitaryRepair, AMatrixFarFromUnitaryIsRepairedJustAsWillingly) {
    // Documented deliberately: Fix asks for the nearest unitary, not for a
    // diagnosis, so distance from unitarity is not a reason to refuse.
    auto m = stretched_2x2();
    ASSERT_GT(unitarity_deviation(m.data(), 2), 1.0)
        << "fixture is not far from unitary, so it tests nothing";
    ASSERT_TRUE(project_to_unitary(m.data(), 2));
    expect_identity(m, 2, 64.0 * kEps, "stretched diagonal");
}

TEST(V11241UnitaryRepair, TheProjectionIsTheNearestUnitary) {
    // For M = W S V-dagger the polar factor W V-dagger satisfies
    // ||M - W V-dagger||_F^2 = sum_i (sigma_i - 1)^2, and no unitary does
    // better. Both halves are checked: the exact distance, and that a sample of
    // other unitaries is further away.
    const int n = 2;
    const auto original = stretched_2x2();
    const auto sigmas = spectrum(original, n);
    ASSERT_EQ(sigmas.size(), static_cast<size_t>(n));

    double expected = 0.0;
    for (double s : sigmas) expected += (s - 1.0) * (s - 1.0);

    auto projected = original;
    ASSERT_TRUE(project_to_unitary(projected.data(), n));
    EXPECT_NEAR(frob_distance_sq(original, projected), expected,
                64.0 * n * kEps)
        << "the projection is not at the polar factor's distance";

    // Nothing unitary may sit closer. The identity IS the projection here, so
    // the alternatives are other unitaries: a Hadamard and a phase flip.
    const std::vector<Complex128> flip{Complex128(1.0, 0.0), Complex128(0.0, 0.0),
                                       Complex128(0.0, 0.0), Complex128(-1.0, 0.0)};
    for (const auto& other : {hadamard(), flip}) {
        ASSERT_LE(unitarity_deviation(other.data(), n), kAtol)
            << "the alternative is not unitary, so it proves nothing";
        EXPECT_LT(frob_distance_sq(original, projected),
                  frob_distance_sq(original, other))
            << "a different unitary sat closer than the polar factor";
    }
}

TEST(V11241UnitaryRepair, ARefusedProjectionLeavesTheBufferUntouched) {
    // The documented failure contract. A caller that asked for a repair and did
    // not get one must still hold exactly what it passed in, so a partial write
    // is as much a defect as a wrong answer.
    auto m = drifted_hadamard();
    m[1] = Complex128(std::numeric_limits<double>::quiet_NaN(), 0.0);
    const auto before = m;

    EXPECT_FALSE(project_to_unitary(m.data(), 2))
        << "a non-finite operand was accepted";
    for (size_t i = 0; i < m.size(); ++i) {
        if (std::isnan(before[i].real)) {
            EXPECT_TRUE(std::isnan(m[i].real)) << "entry " << i;
        } else {
            EXPECT_EQ(m[i].real, before[i].real) << "entry " << i << " real";
        }
        EXPECT_EQ(m[i].imag, before[i].imag) << "entry " << i << " imag";
    }
}

TEST(V11241UnitaryRepair, AZeroSizedOperandIsRefusedRatherThanProjected) {
    EXPECT_FALSE(project_to_unitary(nullptr, 0));
}

// =============================================================================
// unitary_needs_repair and repair_unitary
// =============================================================================

TEST(V11241UnitaryRepair, NeedsRepairIsTrueUnderFixAndOnlyOutsideTolerance) {
    const auto bent = drifted_hadamard();
    const auto clean = hadamard();

    EXPECT_TRUE(unitary_needs_repair(bent.data(), 2, {Validation::Fix, kAtol},
                                     "probe"));
    EXPECT_FALSE(unitary_needs_repair(clean.data(), 2, {Validation::Fix, kAtol},
                                      "probe"))
        << "a matrix already inside tolerance must not be projected; the "
           "repair would be a no-op that still costs a factorisation";
}

TEST(V11241UnitaryRepair, NeedsRepairFollowsTheOtherThreePolicies) {
    const auto bent = drifted_hadamard();

    EXPECT_FALSE(unitary_needs_repair(bent.data(), 2,
                                      {Validation::Ignore, kAtol}, "probe"))
        << "Ignore measured the operand it was told not to look at";

    EXPECT_THROW(unitary_needs_repair(bent.data(), 2,
                                      {Validation::Throw, kAtol}, "probe"),
                 std::invalid_argument);

    {
        WarningProbe probe;
        EXPECT_FALSE(unitary_needs_repair(bent.data(), 2,
                                          {Validation::Warn, kAtol}, "probe"))
            << "Warn repaired; Warn describes and does not repair";
        EXPECT_GE(probe.count(), 1u) << "Warn proceeded without reporting";
    }
}

TEST(V11241UnitaryRepair, NeedsRepairDeclinesAZeroSizedOperand) {
    // rows == 0 returns before measuring, so an empty operand cannot reach the
    // residual and be reported as a violation of a property it cannot have.
    const std::vector<Complex128> empty;
    EXPECT_FALSE(unitary_needs_repair(empty.data(), 0, {Validation::Fix, kAtol},
                                      "probe"));
    EXPECT_FALSE(unitary_needs_repair(empty.data(), 0,
                                      {Validation::Throw, kAtol}, "probe"));
}

TEST(V11241UnitaryRepair, RepairVerifiesItsOwnResultAndRaisesWhenItCannot) {
    // The repair is checked rather than trusted, so a projection that cannot be
    // performed raises instead of returning an operand as unphysical as the one
    // it replaced.
    auto broken = drifted_hadamard();
    broken[3] = Complex128(0.0, std::numeric_limits<double>::infinity());
    EXPECT_THROW(repair_unitary(broken.data(), 2, {Validation::Fix, kAtol},
                                "probe"),
                 std::invalid_argument);
}

TEST(V11241UnitaryRepair, RepairSucceedsAndReportsNothingOnADriftedOperand) {
    WarningProbe probe;
    auto m = drifted_hadamard();
    EXPECT_NO_THROW(repair_unitary(m.data(), 2, {Validation::Fix, kAtol},
                                   "probe"));
    EXPECT_LE(unitarity_deviation(m.data(), 2), kAtol);
    EXPECT_EQ(probe.count(), 0u)
        << "a repair that succeeded warned about the input it fixed";
}

// =============================================================================
// QuantumCircuit::unitary - the one wired site
// =============================================================================

TEST(V11241UnitaryRepair, CircuitIngressStoresTheRepairedMatrix) {
    QuantumCircuit qc(2);
    ASSERT_NO_THROW(qc.unitary(drifted_hadamard(), {0}, "fixed",
                               {Validation::Fix, kAtol}));
    ASSERT_EQ(qc.instructions.size(), 1u);

    const auto& stored = qc.instructions[0].matrix;
    ASSERT_EQ(stored.size(), 4u);
    EXPECT_LE(unitarity_deviation(stored.data(), 2), kAtol)
        << "Fix stored the operand it was asked to repair";
}

TEST(V11241UnitaryRepair, CircuitIngressLeavesTheCallersMatrixAlone) {
    // Documented: the caller's own matrix is not modified. The repair belongs
    // to the instruction, not to the vector the caller still holds.
    auto mine = drifted_hadamard();
    const auto before = mine;

    QuantumCircuit qc(2);
    qc.unitary(mine, {0}, "fixed", {Validation::Fix, kAtol});

    for (size_t i = 0; i < mine.size(); ++i) {
        EXPECT_EQ(mine[i].real, before[i].real) << "entry " << i << " real";
        EXPECT_EQ(mine[i].imag, before[i].imag) << "entry " << i << " imag";
    }
    EXPECT_GT(unitarity_deviation(mine.data(), 2), kAtol)
        << "the caller's matrix was repaired behind its back";
}

TEST(V11241UnitaryRepair, RepairingOneInstructionDoesNotRewriteItsSiblings) {
    // The aliasing guarantee. A CowMatrix shares one buffer across every
    // Instruction copied from it, so an in-place repair would silently rewrite
    // a matrix that another instruction is reading. The Fix path takes a copy
    // for exactly this reason, and this is the only test that can see it.
    const auto bent = drifted_hadamard();

    QuantumCircuit qc(2);
    qc.unitary(bent, {0}, "kept-as-is", {Validation::Ignore, kAtol});
    qc.unitary(bent, {1}, "repaired", {Validation::Fix, kAtol});
    ASSERT_EQ(qc.instructions.size(), 2u);

    const auto& untouched = qc.instructions[0].matrix;
    const auto& repaired = qc.instructions[1].matrix;
    ASSERT_EQ(untouched.size(), 4u);
    ASSERT_EQ(repaired.size(), 4u);

    EXPECT_GT(unitarity_deviation(untouched.data(), 2), kAtol)
        << "repairing the second instruction rewrote the first, which asked "
           "for no validation at all";
    EXPECT_LE(unitarity_deviation(repaired.data(), 2), kAtol);
}

TEST(V11241UnitaryRepair, CircuitIngressRejectsAWrongSizedMatrix) {
    // RED until the library catches up. Every other QuantumCircuit builder
    // validates a caller-supplied operand's structure at ingress and throws:
    // permute rejects a permutation whose size is not 2^k, whose images fall
    // out of range, or which is not a bijection; mcx rejects a control equal to
    // its target; mcp rejects an empty qubit list. unitary is the one builder
    // that stores a structurally invalid operand and defers, which leaves an
    // unusable instruction sitting in a public vector for anything reading the
    // circuit to find.
    //
    // permute makes exactly this check, on exactly this argument shape, one
    // function away. There is no property of a matrix that makes it harder to
    // measure than a permutation.
    QuantumCircuit qc(2);
    EXPECT_THROW(qc.unitary(std::vector<Complex128>(5, Complex128(1.0, 0.0)),
                            {0}, "bad-shape", {Validation::Fix, kAtol}),
                 std::invalid_argument)
        << "a 1-qubit gate needs a 2x2 matrix and got five entries, which "
           "permute would have refused at the same point";
    EXPECT_TRUE(qc.instructions.empty())
        << "the wrong-sized operand was appended anyway, so a caller catching "
           "the exception would still hold a circuit containing it";
}

TEST(V11241UnitaryRepair, AWrongSizedMatrixNeverReachesTheProjection) {
    // Independent of the ruling above, and true either way: a shape that cannot
    // be measured must not be measured. Unitarity is read as U-dagger U over a
    // rows x rows operand, so running it on a wrong-sized buffer reads past the
    // end. The guard is a size comparison beside the Fix branch, and this pins
    // that the branch is what it skips rather than something the projection
    // discovers for itself.
    //
    // Asserted at the kernel, which rejects the shape today, so this case holds
    // whether or not ingress starts rejecting it too.
    Statevector sv(2);
    const std::vector<Complex128> wrong(5, Complex128(1.0, 0.0));
    EXPECT_THROW(lindblad::gates::apply_unitary(sv, {0}, wrong,
                                                {Validation::Fix, kAtol}),
                 std::invalid_argument);

    // The diagnostic must name the shape, not the physics. A caller told their
    // matrix is not unitary would go looking for a numerical problem in an
    // operand whose real defect is that it is the wrong size.
    try {
        lindblad::gates::apply_unitary(sv, {0}, wrong, {Validation::Fix, kAtol});
        FAIL() << "a wrong-sized operand was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("size"), std::string::npos)
            << "the diagnostic does not mention size. Got: " << msg;
        EXPECT_EQ(msg.find("not unitary"), std::string::npos)
            << "a shape error was reported as a physics error. Got: " << msg;
    }
}

TEST(V11241UnitaryRepair, TheRunTimePreFlightCannotRepairWhatIngressDidNot) {
    // Pins the boundary #87 describes, WITHOUT ruling on it, because that issue
    // is open with three candidate answers and one of them is to document the
    // boundary rather than move it.
    //
    // The shape: run() reads each instruction's policy once, through
    // QuantumCircuit::validate_physical(), and hands every kernel below it
    // Validation::Ignore so an unchanged matrix is not re-measured per gate per
    // shot. That is the right trade, and it means the pre-flight is the only
    // consumer of the policy on this path.
    //
    // validate_physical() is const, so it cannot repair. An instruction that
    // reached it unrepaired therefore gets a diagnostic rather than a
    // correction, even under Fix. QuantumCircuit::unitary avoids this by
    // repairing at ingress, so the gap belongs to instructions that arrive some
    // other way: appended to the public vector, imported from QASM, or produced
    // by compose.
    //
    // Simulated here by storing under Ignore and then setting the policy, which
    // is what any of those routes produces: an operand the ingress repair never
    // saw.
    QuantumCircuit qc(1);
    qc.unitary(drifted_hadamard(), {0}, "unrepaired", {Validation::Ignore, kAtol});
    ASSERT_EQ(qc.instructions.size(), 1u);
    ASSERT_GT(unitarity_deviation(qc.instructions[0].matrix.data(), 2), kAtol)
        << "Ignore repaired, so this fixture cannot reach the pre-flight bent";

    qc.instructions[0].validation = {Validation::Fix, kAtol};

    lindblad::StatevectorSimulator sim;
    const auto res = sim.run(qc, /*shots=*/0);

    EXPECT_FALSE(res.success)
        << "the pre-flight accepted an operand outside tolerance under a policy "
           "that promised to correct it, and no repair ran";
    EXPECT_NE(res.error_message.find("unitar"), std::string::npos)
        << "the failure does not name the property. Got: " << res.error_message;
}

// =============================================================================
// The fourteen borrowing entry points
// =============================================================================
// RED until 1.1.24.2 wires the repair through them. Each asserts the ruled
// behaviour: Fix is opt-in, the caller asked for a repair, and one exists, so
// reporting that no repair is defined is a false diagnostic.
//
// Every fixture below is a scalar multiple of the identity, whose polar factor
// IS the identity. A correctly repairing entry point therefore applies the
// identity and leaves the state exactly as it found it, which makes the
// expected result exact rather than a tolerance on a rotated state.

namespace {

std::vector<Complex128> bent_1q() { return scaled_identity(2, 1.5); }
std::vector<Complex128> bent_2q() { return scaled_identity(4, 1.5); }

// The MPS gate entry points take fixed-size arrays rather than vectors, so the
// same operand is spelled twice. Same value either way: 1.5 times the identity,
// whose polar factor is exactly the identity.
template <std::size_t N>
std::array<Complex128, N> bent_array(int n) {
    std::array<Complex128, N> m{};
    for (int i = 0; i < n; ++i)
        m[static_cast<std::size_t>(i) * n + i] = Complex128(1.5, 0.0);
    return m;
}

}  // namespace

TEST(V11241UnitaryRepairReach, StatevectorKernelRepairsUnderFix) {
    // bent_1q() is 1.5 times the identity, whose polar factor is exactly the
    // identity, so a correctly repairing entry point applies the identity and
    // leaves the state where it found it. That makes the expected result exact
    // rather than a tolerance on a rotated state.
    Statevector sv(3);
    std::vector<Complex128> before(sv.dim);
    for (size_t i = 0; i < sv.dim; ++i)
        before[i] = Complex128(sv.real_parts[i], sv.imag_parts[i]);

    ASSERT_NO_THROW(
        lindblad::gates::apply_unitary(sv, {1}, bent_1q(),
                                       {Validation::Fix, kAtol}))
        << "Fix reported no repair for unitarity, which has one";

    for (size_t i = 0; i < before.size(); ++i) {
        EXPECT_NEAR(sv.real_parts[i], before[i].real, 64.0 * kEps) << i;
        EXPECT_NEAR(sv.imag_parts[i], before[i].imag, 64.0 * kEps) << i;
    }
}

TEST(V11241UnitaryRepairReach, StatevectorTwoQubitKernelRepairsUnderFix) {
    Statevector sv(4);
    ASSERT_NO_THROW(
        lindblad::gates::apply_unitary(sv, {0, 2}, bent_2q(),
                                       {Validation::Fix, kAtol}));
}

TEST(V11241UnitaryRepairReach, MpsSingleQubitGateRepairsUnderFix) {
    MPSState mps(4);
    ASSERT_NO_THROW(mps.apply_single_qubit_gate(bent_array<4>(2), 2,
                                                {Validation::Fix, kAtol}));
}

TEST(V11241UnitaryRepairReach, MpsTwoQubitGateRepairsUnderFix) {
    MPSState mps(4);
    ASSERT_NO_THROW(mps.apply_two_qubit_gate(bent_array<16>(4), 0, 1,
                                             {Validation::Fix, kAtol}));
}

TEST(V11241UnitaryRepairReach, DensityMatrixGateRepairsUnderFix) {
    DensityMatrix rho(3);
    ASSERT_NO_THROW(rho.apply_gate(bent_1q(), {1}, {Validation::Fix, kAtol}));
}

TEST(V11241UnitaryRepairReach, QuditStatevectorRepairsUnderFix) {
    const int d = 3;
    QuditStatevector sv(4, d);
    ASSERT_NO_THROW(sv.apply_1qudit(1, scaled_identity(d, 1.5),
                                    {Validation::Fix, kAtol}));
}

TEST(V11241UnitaryRepairReach, TheDiagnosticNamesTheRealReasonWhenItRefuses) {
    // Whichever way the ruling lands, "no repair defined" is false for
    // unitarity. If an entry point declines to repair, it owes the caller the
    // actual reason rather than a claim the library contradicts elsewhere.
    Statevector sv(2);
    try {
        lindblad::gates::apply_unitary(sv, {0}, bent_1q(),
                                       {Validation::Fix, kAtol});
        SUCCEED() << "the entry point repaired, which is the ruled behaviour";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("no repair defined"), std::string::npos)
            << "unitarity has a repair and ships one, so this diagnostic is "
               "false. Got: "
            << msg;
    }
}
