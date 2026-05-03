#include <gtest/gtest.h>
#include "qpp/operators.hpp"
#include "qpp/statevector.hpp"
#include "qpp/gates.hpp"

using namespace qpp;

TEST(OperatorsTest, PauliStringCompose) {
    PauliString a("XY", Complex128(1, 0));
    PauliString b("YX", Complex128(1, 0));
    auto result = a.compose(b);
    EXPECT_EQ(result.pauli, "ZZ");
}

TEST(OperatorsTest, PauliStringCommutation) {
    PauliString a("XY");
    PauliString b("YX");
    EXPECT_TRUE(a.commutes_with(b));  // XY and YX: 2 anticommuting positions → even → commute

    PauliString c("XX");
    PauliString d("ZZ");
    EXPECT_TRUE(c.commutes_with(d));
}

TEST(OperatorsTest, SparsePauliOpExpectation) {
    // ⟨0|Z|0⟩ = 1
    Statevector sv(1);
    SparsePauliOp op({PauliString("Z")});
    double ev = op.expectation_value(sv);
    EXPECT_NEAR(ev, 1.0, 1e-10);

    // ⟨1|Z|1⟩ = -1
    Statevector sv1(1);
    gates::apply_x(sv1, 0);
    double ev1 = op.expectation_value(sv1);
    EXPECT_NEAR(ev1, -1.0, 1e-10);
}

TEST(OperatorsTest, SparsePauliOpSimplify) {
    SparsePauliOp op({
        PauliString("ZZ", Complex128(0.5, 0)),
        PauliString("ZZ", Complex128(0.5, 0)),
        PauliString("XI", Complex128(1e-12, 0))
    });
    auto simplified = op.simplify();
    EXPECT_EQ(simplified.terms.size(), 1u);
    EXPECT_NEAR(simplified.terms[0].coeff.real, 1.0, 1e-10);
}

TEST(OperatorsTest, OperatorIsUnitary) {
    // Hadamard matrix
    double inv = INV_SQRT2;
    Operator H({
        Complex128(inv, 0), Complex128(inv, 0),
        Complex128(inv, 0), Complex128(-inv, 0)
    }, 1);
    EXPECT_TRUE(H.is_unitary());
}

TEST(OperatorsTest, IdentityIsHermitian) {
    Operator I({
        Complex128(1, 0), Complex128(0, 0),
        Complex128(0, 0), Complex128(1, 0)
    }, 1);
    EXPECT_TRUE(I.is_hermitian());
}
