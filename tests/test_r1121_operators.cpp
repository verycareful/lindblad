// R.1.12.1 total-coverage suite, Batch 2: lindblad/operators.hpp
// (PauliString, SparsePauliOp, Operator). Plan section "Batch 2: engines".
//
// Correctness is pinned by convention-robust cross-checks: Pauli compose vs
// matrix product, to_matrix vs Operator::from_pauli, expectation_value vs
// <psi|M|psi>, tensor vs Kronecker placement, plus the documented throw paths.
// Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/operators.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 1e-10;
using Mat = std::vector<Complex128>;

Mat pauli_mat(const PauliString& p) {
    return SparsePauliOp(std::vector<PauliString>{p}).to_matrix();
}

Mat matmul(const Mat& A, const Mat& B, size_t dim) {
    Mat C(dim * dim, Complex128(0, 0));
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j) {
            Complex128 acc(0, 0);
            for (size_t k = 0; k < dim; ++k) acc += A[i * dim + k] * B[k * dim + j];
            C[i * dim + j] = acc;
        }
    return C;
}

// Kronecker product hi (x) lo, with hi the high (MSB) factor.
Mat kron(const Mat& hi, size_t dh, const Mat& lo, size_t dl) {
    const size_t d = dh * dl;
    Mat C(d * d, Complex128(0, 0));
    for (size_t rh = 0; rh < dh; ++rh)
        for (size_t ch = 0; ch < dh; ++ch)
            for (size_t rl = 0; rl < dl; ++rl)
                for (size_t cl = 0; cl < dl; ++cl)
                    C[(rh * dl + rl) * d + (ch * dl + cl)] =
                        hi[rh * dh + ch] * lo[rl * dl + cl];
    return C;
}

void expect_mat_eq(const Mat& a, const Mat& b, double tol = kTol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, tol) << "re @ " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, tol) << "im @ " << i;
    }
}

}  // namespace

// =============================================================================
// PauliString
// =============================================================================
//
// REGRESSION (shipped red in R.1.12.1, fixed in R.1.12.2):
// SparsePauliOp::to_matrix() applies the i^(#Y) factor as a per-term constant
// folded into the coefficient (Y = i*X*Z, so one i per Y in the string). The
// pre-fix code applied i^popcount(j & y_mask) per column, which made every
// Y-containing matrix non-Hermitian. The three tests below pin the correct
// contract: to_matrix is Hermitian for Hermitian operators, agrees with
// expectation_value, and is a homomorphism under compose().

TEST(R1121Operators, PauliComposeMatchesMatrixProductAllPairs) {
    const std::vector<std::string> ps = {"I", "X", "Y", "Z"};
    for (const auto& a : ps)
        for (const auto& b : ps) {
            PauliString pa(a), pb(b);
            auto composed = pauli_mat(pa.compose(pb));
            auto product = matmul(pauli_mat(pa), pauli_mat(pb), 2);
            SCOPED_TRACE(a + " * " + b);
            expect_mat_eq(composed, product);
        }
}

// Full single-qubit Pauli multiplication table at the PauliString level
// (result letter + phase). This pins the algebra DIRECTLY, independent of the
// to_matrix-Y bug, so it is green. compose(a, b) is the matrix product a*b:
//   X*Y=+iZ, Y*Z=+iX, Z*X=+iY ; Y*X=-iZ, Z*Y=-iX, X*Z=-iY ; a*a=I ; I*b=b.
TEST(R1121Operators, PauliSingleQubitProductTablePhasesAndLetters) {
    struct Row { char a, b, result; double re, im; };
    const Row table[] = {
        {'I','I','I',1, 0}, {'I','X','X',1, 0}, {'I','Y','Y',1, 0}, {'I','Z','Z',1, 0},
        {'X','I','X',1, 0}, {'X','X','I',1, 0}, {'X','Y','Z',0, 1}, {'X','Z','Y',0,-1},
        {'Y','I','Y',1, 0}, {'Y','X','Z',0,-1}, {'Y','Y','I',1, 0}, {'Y','Z','X',0, 1},
        {'Z','I','Z',1, 0}, {'Z','X','Y',0, 1}, {'Z','Y','X',0,-1}, {'Z','Z','I',1, 0},
    };
    for (const Row& r : table) {
        PauliString pa(std::string(1, r.a)), pb(std::string(1, r.b));
        auto c = pa.compose(pb);
        SCOPED_TRACE(std::string(1, r.a) + " * " + std::string(1, r.b));
        EXPECT_EQ(c.pauli, std::string(1, r.result));
        EXPECT_NEAR(c.coeff.real, r.re, kTol);
        EXPECT_NEAR(c.coeff.imag, r.im, kTol);
    }
}

// Multi-qubit phase accumulation: phases from each position multiply.
TEST(R1121Operators, PauliMultiQubitPhaseAccumulation) {
    // "XZ".compose("ZX"): pos0 X*Z = -iY, pos1 Z*X = +iY -> (-i)(i) = 1, "YY".
    auto a = PauliString("XZ").compose(PauliString("ZX"));
    EXPECT_EQ(a.pauli, "YY");
    EXPECT_NEAR(a.coeff.real, 1.0, kTol);
    EXPECT_NEAR(a.coeff.imag, 0.0, kTol);

    // "XY".compose("YX"): pos0 X*Y = +iZ, pos1 Y*X = -iZ -> (i)(-i) = 1, "ZZ".
    auto b = PauliString("XY").compose(PauliString("YX"));
    EXPECT_EQ(b.pauli, "ZZ");
    EXPECT_NEAR(b.coeff.imag, 0.0, kTol);

    // "XX".compose("YY"): pos0 X*Y=iZ, pos1 X*Y=iZ -> (i)(i) = -1, "ZZ".
    auto cc = PauliString("XX").compose(PauliString("YY"));
    EXPECT_EQ(cc.pauli, "ZZ");
    EXPECT_NEAR(cc.coeff.real, -1.0, kTol);

    // Carried coefficients multiply alongside the phase.
    auto d = PauliString("X", Complex128(2.0, 0.0))
                 .compose(PauliString("Y", Complex128(0.0, 3.0)));  // 2 * 3i * (iZ)
    EXPECT_EQ(d.pauli, "Z");
    EXPECT_NEAR(d.coeff.real, -6.0, kTol);  // 2 * 3i * i = -6
    EXPECT_NEAR(d.coeff.imag, 0.0, kTol);
}

TEST(R1121Operators, CommutesWithThreeQubitParity) {
    // Three anticommuting positions -> odd -> anticommute.
    EXPECT_FALSE(PauliString("XXX").commutes_with(PauliString("ZZZ")));
    // Two anticommuting positions (one shared identity) -> even -> commute.
    EXPECT_TRUE(PauliString("XXI").commutes_with(PauliString("ZZI")));
    // Mixed length mismatch -> false (documented).
    EXPECT_FALSE(PauliString("XX").commutes_with(PauliString("X")));
}

TEST(R1121Operators, PauliComposeMultiQubitAndLengthThrow) {
    PauliString a("XZ"), b("ZX");
    auto composed = pauli_mat(a.compose(b));
    auto product = matmul(pauli_mat(a), pauli_mat(b), 4);
    expect_mat_eq(composed, product);

    EXPECT_THROW(PauliString("XY").compose(PauliString("X")),
                 std::invalid_argument);
}

TEST(R1121Operators, PauliAdjointConjugatesCoefficient) {
    PauliString p("Y", Complex128(0.0, 1.0));
    auto adj = p.adjoint();
    EXPECT_EQ(adj.pauli, "Y");
    EXPECT_NEAR(adj.coeff.real, 0.0, kTol);
    EXPECT_NEAR(adj.coeff.imag, -1.0, kTol);
}

TEST(R1121Operators, CommutesWithCountsAnticommutingPositions) {
    EXPECT_TRUE(PauliString("X").commutes_with(PauliString("X")));
    EXPECT_FALSE(PauliString("X").commutes_with(PauliString("Z")));
    // Two anticommuting positions -> overall commute.
    EXPECT_TRUE(PauliString("XZ").commutes_with(PauliString("ZX")));
    // One anticommuting position -> anticommute.
    EXPECT_FALSE(PauliString("XI").commutes_with(PauliString("ZI")));
}

// =============================================================================
// SparsePauliOp
// =============================================================================

TEST(R1121Operators, SimplifyMergesDuplicatesAndPrunesSmall) {
    auto op = SparsePauliOp::from_list(
        {{"X", Complex128(1.0, 0.0)},
         {"X", Complex128(2.0, 0.0)},
         {"Z", Complex128(1e-10, 0.0)}});
    auto s = op.simplify();
    ASSERT_EQ(s.size(), 1u) << "duplicate X merged; tiny Z pruned";
    EXPECT_EQ(s.terms[0].pauli, "X");
    EXPECT_NEAR(s.terms[0].coeff.real, 3.0, kTol);
}

TEST(R1121Operators, SimplifyAtolPruningBoundary) {
    // A term with |coeff| just below atol is pruned; just above is kept.
    const double atol = 1e-6;
    auto below = SparsePauliOp::from_list({{"X", Complex128(0.5e-6, 0.0)}}).simplify(atol);
    EXPECT_EQ(below.size(), 0u) << "coeff below atol pruned";
    auto above = SparsePauliOp::from_list({{"X", Complex128(2e-6, 0.0)}}).simplify(atol);
    EXPECT_EQ(above.size(), 1u) << "coeff above atol kept";

    // Complex coefficients that cancel to zero are pruned after merge.
    auto cancel = SparsePauliOp::from_list(
        {{"Z", Complex128(1.0, 1.0)}, {"Z", Complex128(-1.0, -1.0)}}).simplify();
    EXPECT_EQ(cancel.size(), 0u) << "Z and -Z cancel and prune";
}

TEST(R1121Operators, ComposeThenSimplifyCancelsCrossTerms) {
    // (X + Z) composed with (X + Z): XX + XZ + ZX + ZZ.
    //   XX = I, ZZ = I, XZ = -iY, ZX = +iY -> the Y terms cancel -> 2I.
    SparsePauliOp xz(std::vector<PauliString>{PauliString("X"), PauliString("Z")});
    auto prod = xz.compose(xz);
    ASSERT_EQ(prod.size(), 1u) << "cross Y terms cancel, X*X and Z*Z merge to 2I";
    EXPECT_EQ(prod.terms[0].pauli, "I");
    EXPECT_NEAR(prod.terms[0].coeff.real, 2.0, kTol);
    EXPECT_NEAR(prod.terms[0].coeff.imag, 0.0, kTol);
}

TEST(R1121Operators, AdjointConjugatesAllCoefficients) {
    auto op = SparsePauliOp::from_list(
        {{"XZ", Complex128(0.0, 1.0)}, {"ZX", Complex128(2.0, -3.0)}});
    auto adj = op.adjoint();
    ASSERT_EQ(adj.size(), 2u);
    // adjoint keeps the labels (Paulis are Hermitian) and conjugates coeffs.
    for (const auto& t : adj.terms) {
        if (t.pauli == "XZ") { EXPECT_NEAR(t.coeff.imag, -1.0, kTol); }
        if (t.pauli == "ZX") { EXPECT_NEAR(t.coeff.imag, 3.0, kTol); }
    }
}

TEST(R1121Operators, ExpectationBatchManyStatesMatchesLoopUnderOpenMP) {
    // Z/X-only observable (no Y -> independent of the to_matrix-Y bug; and the
    // batch path is what we are exercising vs the per-state loop).
    auto op = SparsePauliOp::from_list(
        {{"ZI", Complex128(1.0, 0.0)}, {"IZ", Complex128(-0.5, 0.0)},
         {"XX", Complex128(0.25, 0.0)}});
    std::vector<Statevector> store;
    store.reserve(64);
    for (int k = 0; k < 64; ++k) {
        Statevector sv(2);
        std::vector<Complex128> amps = {
            Complex128(0.3 + 0.01 * k, 0.1), Complex128(-0.2, 0.05 * k - 1.0),
            Complex128(0.4, -0.2), Complex128(0.1 * k - 3.0, 0.2)};
        // The amplitudes vary with k rather than being normalized, so the
        // hand-over repairs them: Fix normalizes within the same call.
        sv.set_amplitudes(amps, {Validation::Throw, DEFAULT_PHYSICAL_ATOL, Repair::Attempt});
        store.push_back(std::move(sv));
    }
    std::vector<const Statevector*> ptrs;
    for (const auto& s : store) ptrs.push_back(&s);

    auto batch = op.expectation_value_batch(ptrs);
    ASSERT_EQ(batch.size(), store.size());
    for (size_t i = 0; i < store.size(); ++i) {
        SCOPED_TRACE("state " + std::to_string(i));
        EXPECT_NEAR(batch[i], op.expectation_value(store[i]), 1e-10)
            << "parallel batch must equal the serial per-state result";
    }
}

TEST(R1121Operators, ToMatrixMatchesFromPauli) {
    auto op = SparsePauliOp::from_list(
        {{"XI", Complex128(0.5, 0.0)},
         {"ZZ", Complex128(-0.3, 0.0)},
         {"IY", Complex128(0.2, 0.0)}});
    expect_mat_eq(op.to_matrix(), Operator::from_pauli(op).data);
}

TEST(R1121Operators, ExpectationValueMatchesQuadraticForm) {
    auto op = SparsePauliOp::from_list(
        {{"XI", Complex128(0.5, 0.0)},
         {"ZZ", Complex128(-0.3, 0.0)},
         {"IY", Complex128(0.2, 0.0)}});
    auto M = op.to_matrix();

    Statevector sv(2);
    std::vector<Complex128> amps = {
        Complex128(0.5, 0.1), Complex128(-0.2, 0.4),
        Complex128(0.3, -0.3), Complex128(0.5, 0.2)};
    // The amplitudes are chosen to be distinct rather than normalized, so the
    // hand-over repairs them: Fix normalizes within the same call.
    sv.set_amplitudes(amps, {Validation::Throw, DEFAULT_PHYSICAL_ATOL, Repair::Attempt});
    auto a = sv.amplitudes();

    // <psi|M|psi>
    Complex128 acc(0, 0);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            acc += a[i].conj() * M[i * 4 + j] * a[j];
    EXPECT_NEAR(op.expectation_value(sv), acc.real, 1e-9);
    EXPECT_NEAR(acc.imag, 0.0, 1e-9) << "Hermitian observable -> real expectation";
}

TEST(R1121Operators, ExpectationBatchEqualsPerStateLoop) {
    auto op = SparsePauliOp::from_list({{"ZI", Complex128(1.0, 0.0)},
                                        {"IZ", Complex128(0.5, 0.0)}});
    Statevector a(2), b(2);
    a.initialize_basis(1);
    b.initialize_basis(2);
    std::vector<const Statevector*> states = {&a, &b};
    auto batch = op.expectation_value_batch(states);
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_NEAR(batch[0], op.expectation_value(a), 1e-12);
    EXPECT_NEAR(batch[1], op.expectation_value(b), 1e-12);
}

TEST(R1121Operators, TensorPlacesThisOnLowQubits) {
    SparsePauliOp a(std::vector<PauliString>{PauliString("X")});
    SparsePauliOp b(std::vector<PauliString>{PauliString("Z")});
    auto t = a.tensor(b);  // X on low qubit, Z on high qubit
    // full matrix == Z (high) (x) X (low)
    expect_mat_eq(t.to_matrix(), kron(pauli_mat(PauliString("Z")), 2,
                                      pauli_mat(PauliString("X")), 2));
}

TEST(R1121Operators, IdentityZeroAndFromList) {
    auto id = SparsePauliOp::identity(2);
    Mat I4(16, Complex128(0, 0));
    for (size_t i = 0; i < 4; ++i) I4[i * 4 + i] = {1, 0};
    expect_mat_eq(id.to_matrix(), I4);

    auto z = SparsePauliOp::zero(2);
    Mat Z4(16, Complex128(0, 0));
    expect_mat_eq(z.to_matrix(), Z4);

    EXPECT_EQ(SparsePauliOp::identity(3).n_qubits(), 3);
}

TEST(R1121Operators, ScalarMultiplyAndAddAreLinear) {
    auto x = SparsePauliOp::from_list({{"Z", Complex128(1.0, 0.0)}});
    auto scaled = x * 2.0;
    auto summed = x + x;
    // 2Z and Z+Z both equal the matrix 2*diag(1,-1).
    expect_mat_eq(scaled.to_matrix(), summed.simplify().to_matrix());
}

// =============================================================================
// Operator
// =============================================================================

TEST(R1121Operators, FromCircuitGivesGateMatrix) {
    QuantumCircuit qc(1);
    qc.h(0);
    auto op = Operator::from_circuit(qc);
    Mat H = {{INV_SQRT2, 0}, {INV_SQRT2, 0}, {INV_SQRT2, 0}, {-INV_SQRT2, 0}};
    expect_mat_eq(op.data, H);
    EXPECT_EQ(op.n_qubits, 1);
    EXPECT_EQ(op.dim(), 2u);
}

TEST(R1121Operators, PowerComposeAdjointAndPredicates) {
    QuantumCircuit qc(1);
    qc.h(0);
    auto H = Operator::from_circuit(qc);

    // H is unitary and Hermitian; H^2 = I.
    EXPECT_TRUE(H.is_unitary());
    EXPECT_TRUE(H.is_hermitian());
    Mat I2 = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    expect_mat_eq(H.power(0).data, I2);
    expect_mat_eq(H.power(2).data, I2);
    expect_mat_eq(H.compose(H).data, I2);
    expect_mat_eq(H.power(1).data, H.data);
    // odd power returns H again.
    expect_mat_eq(H.power(5).data, H.data);
    EXPECT_THROW(H.power(-1), std::invalid_argument);

    // adjoint of H is H.
    expect_mat_eq(H.adjoint().data, H.data);
}

TEST(R1121Operators, NonHermitianNonUnitaryDetected) {
    QuantumCircuit qc(1);
    qc.t(0);  // T is unitary but not Hermitian
    auto T = Operator::from_circuit(qc);
    EXPECT_TRUE(T.is_unitary());
    EXPECT_FALSE(T.is_hermitian());

    // A scaled identity 2I is Hermitian but not unitary.
    Operator twoI({Complex128(2, 0), Complex128(0, 0),
                   Complex128(0, 0), Complex128(2, 0)}, 1);
    EXPECT_TRUE(twoI.is_hermitian());
    EXPECT_FALSE(twoI.is_unitary());
    EXPECT_NEAR(twoI.trace().real, 4.0, kTol);
}

TEST(R1121Operators, TensorOfOperatorsKronStructure) {
    QuantumCircuit qx(1); qx.x(0);
    QuantumCircuit qz(1); qz.z(0);
    auto X = Operator::from_circuit(qx);
    auto Z = Operator::from_circuit(qz);
    auto t = X.tensor(Z);
    EXPECT_EQ(t.n_qubits, 2);
    // Operator::tensor(other): this on one factor, other on the other.
    // Verify it is one of the two Kronecker orders (associativity-agnostic).
    auto k1 = kron(Z.data, 2, X.data, 2);
    auto k2 = kron(X.data, 2, Z.data, 2);
    bool matches_k1 = true, matches_k2 = true;
    for (size_t i = 0; i < 16; ++i) {
        if (std::abs(t.data[i].real - k1[i].real) > kTol ||
            std::abs(t.data[i].imag - k1[i].imag) > kTol) matches_k1 = false;
        if (std::abs(t.data[i].real - k2[i].real) > kTol ||
            std::abs(t.data[i].imag - k2[i].imag) > kTol) matches_k2 = false;
    }
    EXPECT_TRUE(matches_k1 || matches_k2) << "tensor must be a Kronecker product";
}
