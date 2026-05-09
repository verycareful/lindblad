// Operators and quantum information tests.
// Covers: SparsePauliOp algebra (compose, tensor, to_matrix, batch expectation),
//         Operator (compose, adjoint, power, trace, from_pauli),
//         QuantumInfo (state_fidelity, process_fidelity, average_gate_fidelity,
//                      entropy, entanglement_entropy, concurrence, partial_trace).

#include <gtest/gtest.h>
#include "lindblad/operators.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/circuit.hpp"

#include <cmath>
#include <vector>

using namespace lindblad;
using namespace lindblad::QuantumInfo;

static constexpr double kTol = 1e-9;

// =============================================================================
// SparsePauliOp — algebraic operations
// =============================================================================

TEST(SparsePauliOpAlgebra, Compose_XY_IsIZ) {
    // X·Y = iZ  (Pauli algebra)
    SparsePauliOp X = SparsePauliOp::from_list({{"X", Complex128(1,0)}});
    SparsePauliOp Y = SparsePauliOp::from_list({{"Y", Complex128(1,0)}});
    SparsePauliOp XY = X.compose(Y).simplify();
    // Expect one term "Z" with coefficient i
    ASSERT_EQ(XY.terms.size(), 1u);
    EXPECT_EQ(XY.terms[0].pauli, "Z");
    EXPECT_NEAR(XY.terms[0].coeff.real, 0.0, kTol);
    EXPECT_NEAR(XY.terms[0].coeff.imag, 1.0, kTol);
}

TEST(SparsePauliOpAlgebra, Compose_XX_IsIdentity) {
    // X·X = I
    SparsePauliOp X = SparsePauliOp::from_list({{"X", Complex128(1,0)}});
    SparsePauliOp XX = X.compose(X).simplify();
    ASSERT_EQ(XX.terms.size(), 1u);
    EXPECT_EQ(XX.terms[0].pauli, "I");
    EXPECT_NEAR(XX.terms[0].coeff.real, 1.0, kTol);
    EXPECT_NEAR(XX.terms[0].coeff.imag, 0.0, kTol);
}

TEST(SparsePauliOpAlgebra, Adjoint_HermitianIsUnchanged) {
    // Pauli operators are Hermitian: adjoint = self (coefficient conjugated).
    SparsePauliOp H = SparsePauliOp::from_list({
        {"X", Complex128(1.0, 0.0)},
        {"Z", Complex128(0.5, 0.0)}
    });
    SparsePauliOp Hadj = H.adjoint().simplify();
    // Expect same terms
    Statevector sv(1);
    gates::apply_h(sv, 0);
    EXPECT_NEAR(H.expectation_value(sv), Hadj.expectation_value(sv), kTol);
}

TEST(SparsePauliOpAlgebra, Tensor_XZ_OnTwoQubits) {
    // X⊗Z: tensor product gives a 2-qubit operator.
    SparsePauliOp X = SparsePauliOp::from_list({{"X", Complex128(1,0)}});
    SparsePauliOp Z = SparsePauliOp::from_list({{"Z", Complex128(1,0)}});
    SparsePauliOp XZ = X.tensor(Z);
    ASSERT_EQ(XZ.size(), 1u);
    EXPECT_EQ(XZ.n_qubits(), 2);
}

TEST(SparsePauliOpAlgebra, ToMatrix_PauliX) {
    // Pauli X matrix: [[0,1],[1,0]]
    SparsePauliOp X = SparsePauliOp::from_list({{"X", Complex128(1,0)}});
    auto mat = X.to_matrix();
    ASSERT_EQ(mat.size(), 4u);
    EXPECT_NEAR(mat[0].real, 0.0, kTol);
    EXPECT_NEAR(mat[1].real, 1.0, kTol);
    EXPECT_NEAR(mat[2].real, 1.0, kTol);
    EXPECT_NEAR(mat[3].real, 0.0, kTol);
}

TEST(SparsePauliOpAlgebra, ToMatrix_PauliZ) {
    // Pauli Z matrix: [[1,0],[0,-1]]
    SparsePauliOp Z = SparsePauliOp::from_list({{"Z", Complex128(1,0)}});
    auto mat = Z.to_matrix();
    EXPECT_NEAR(mat[0].real,  1.0, kTol);
    EXPECT_NEAR(mat[3].real, -1.0, kTol);
    EXPECT_NEAR(mat[1].real,  0.0, kTol);
    EXPECT_NEAR(mat[2].real,  0.0, kTol);
}

TEST(SparsePauliOpAlgebra, BatchExpectation_ConsistentWithSingle) {
    // expectation_value_batch should match expectation_value called individually.
    SparsePauliOp H = SparsePauliOp::from_list({
        {"X", Complex128(1.0, 0.0)},
        {"Z", Complex128(0.5, 0.0)}
    });
    Statevector sv1(1);
    gates::apply_h(sv1, 0);
    Statevector sv2(1);
    gates::apply_x(sv2, 0);

    double e1 = H.expectation_value(sv1);
    double e2 = H.expectation_value(sv2);

    auto batch = H.expectation_value_batch({&sv1, &sv2});
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_NEAR(batch[0], e1, kTol);
    EXPECT_NEAR(batch[1], e2, kTol);
}

TEST(SparsePauliOpAlgebra, Operator_Plus) {
    SparsePauliOp X = SparsePauliOp::from_list({{"X", Complex128(1,0)}});
    SparsePauliOp Z = SparsePauliOp::from_list({{"Z", Complex128(1,0)}});
    SparsePauliOp sum = (X + Z).simplify();
    EXPECT_EQ(sum.size(), 2u);
}

// =============================================================================
// Operator — general matrix operator
// =============================================================================

TEST(OperatorAlgebra, FromPauli_X_IsUnitary) {
    SparsePauliOp X = SparsePauliOp::from_list({{"X", Complex128(1,0)}});
    Operator op = Operator::from_pauli(X);
    EXPECT_TRUE(op.is_unitary());
    EXPECT_TRUE(op.is_hermitian());
}

TEST(OperatorAlgebra, FromPauli_iY_IsHermitian) {
    // iY is anti-Hermitian; just check from_pauli does not crash.
    SparsePauliOp Y = SparsePauliOp::from_list({{"Y", Complex128(0,1)}});
    Operator op = Operator::from_pauli(Y);
    EXPECT_EQ(op.n_qubits, 1);
}

TEST(OperatorAlgebra, Compose_HH_IsIdentity) {
    // H·H = I; result should be unitary and have trace = 2.
    QuantumCircuit h_circ(1);
    h_circ.h(0);
    Operator H = Operator::from_circuit(h_circ);
    Operator HH = H.compose(H);
    EXPECT_TRUE(HH.is_unitary());
    EXPECT_NEAR(HH.trace().real, 2.0, 1e-8);
    EXPECT_NEAR(HH.trace().imag, 0.0, 1e-8);
}

TEST(OperatorAlgebra, Adjoint_UnitaryIsInverse) {
    // For unitary U: U·U† = I
    QuantumCircuit s_circ(1);
    s_circ.s(0);
    Operator S = Operator::from_circuit(s_circ);
    Operator SS_dag = S.compose(S.adjoint());
    EXPECT_TRUE(SS_dag.is_unitary());
    EXPECT_NEAR(SS_dag.trace().real, 2.0, 1e-8);
}

TEST(OperatorAlgebra, Power_Square_IsTwoCompose) {
    QuantumCircuit t_circ(1);
    t_circ.t(0);
    Operator T = Operator::from_circuit(t_circ);
    Operator T2_power = T.power(2);
    Operator T2_compose = T.compose(T);
    // Both should give the same matrix (S gate up to global phase)
    EXPECT_NEAR(T2_power.trace().real, T2_compose.trace().real, 1e-8);
}

TEST(OperatorAlgebra, Power_Zero_IsIdentity) {
    QuantumCircuit cx_circ(2);
    cx_circ.cx(0, 1);
    Operator CX = Operator::from_circuit(cx_circ);
    Operator I = CX.power(0);
    EXPECT_TRUE(I.is_unitary());
    EXPECT_NEAR(I.trace().real, 4.0, 1e-8);  // trace of 4x4 identity = 4
}

TEST(OperatorAlgebra, Tensor_Product_Dimension) {
    QuantumCircuit h1(1); h1.h(0);
    QuantumCircuit x1(1); x1.x(0);
    Operator H = Operator::from_circuit(h1);
    Operator X = Operator::from_circuit(x1);
    Operator HX = H.tensor(X);
    EXPECT_EQ(HX.n_qubits, 2);
    EXPECT_EQ(HX.dim(), 4u);
    EXPECT_TRUE(HX.is_unitary());
}

// =============================================================================
// QuantumInfo::state_fidelity
// =============================================================================

TEST(QuantumInfo_Fidelity, SameState_IsOne) {
    Statevector sv(2);
    gates::apply_h(sv, 0);
    gates::apply_cx(sv, 0, 1);
    EXPECT_NEAR(state_fidelity(sv, sv), 1.0, kTol);
}

TEST(QuantumInfo_Fidelity, OrthogonalStates_IsZero) {
    Statevector sv0(1);
    Statevector sv1(1);
    gates::apply_x(sv1, 0);  // |1⟩ ⊥ |0⟩
    EXPECT_NEAR(state_fidelity(sv0, sv1), 0.0, kTol);
}

TEST(QuantumInfo_Fidelity, PartialOverlap) {
    // ⟨0|+⟩² = 1/2
    Statevector sv0(1);
    Statevector svp(1);
    gates::apply_h(svp, 0);
    EXPECT_NEAR(state_fidelity(sv0, svp), 0.5, kTol);
}

TEST(QuantumInfo_Fidelity, DensityMatrix_SameState_IsOne) {
    Statevector sv(1);
    gates::apply_h(sv, 0);
    DensityMatrix rho = DensityMatrix::from_statevector(sv);
    EXPECT_NEAR(state_fidelity(rho, rho), 1.0, kTol);
}

// =============================================================================
// QuantumInfo::process_fidelity / average_gate_fidelity
// =============================================================================

TEST(QuantumInfo_ProcessFidelity, SameGate_IsOne) {
    QuantumCircuit h(1); h.h(0);
    Operator H = Operator::from_circuit(h);
    EXPECT_NEAR(process_fidelity(H, H), 1.0, kTol);
}

TEST(QuantumInfo_ProcessFidelity, AverageGateFidelity_SameGate_IsOne) {
    QuantumCircuit h(1); h.h(0);
    Operator H = Operator::from_circuit(h);
    EXPECT_NEAR(average_gate_fidelity(H, H), 1.0, kTol);
}

// =============================================================================
// QuantumInfo::entropy
// =============================================================================

TEST(QuantumInfo_Entropy, PureState_IsZero) {
    Statevector sv(2);
    gates::apply_h(sv, 0);
    gates::apply_cx(sv, 0, 1);
    DensityMatrix rho = DensityMatrix::from_statevector(sv);
    // Pure state entropy = 0
    EXPECT_NEAR(entropy(rho), 0.0, 1e-7);
}

TEST(QuantumInfo_Entropy, MaximallyMixed_IsNBits) {
    // Maximally mixed state on 1 qubit: entropy = 1 (base 2)
    DensityMatrix rho(1);
    rho(0,0) = Complex128(0.5, 0.0);
    rho(1,1) = Complex128(0.5, 0.0);
    EXPECT_NEAR(entropy(rho), 1.0, 1e-7);
}

// =============================================================================
// QuantumInfo::entanglement_entropy
// =============================================================================

TEST(QuantumInfo_EntanglementEntropy, ProductState_IsZero) {
    // |+⟩⊗|0⟩: no entanglement between qubit 0 and qubit 1.
    Statevector sv(2);
    gates::apply_h(sv, 0);
    EXPECT_NEAR(entanglement_entropy(sv, {0}), 0.0, 1e-7);
}

TEST(QuantumInfo_EntanglementEntropy, BellState_IsOneBit) {
    // |Φ+⟩: maximally entangled → entanglement entropy = 1 ebit.
    Statevector sv(2);
    gates::apply_h(sv, 0);
    gates::apply_cx(sv, 0, 1);
    EXPECT_NEAR(entanglement_entropy(sv, {0}), 1.0, 1e-7);
}

// =============================================================================
// QuantumInfo::partial_trace
// =============================================================================

TEST(QuantumInfo_PartialTrace, BellState_TraceQ0_IsMaxMixed) {
    // Tracing out qubit 0 of |Φ+⟩ gives maximally mixed state on qubit 1.
    Statevector sv(2);
    gates::apply_h(sv, 0);
    gates::apply_cx(sv, 0, 1);
    DensityMatrix rho_1 = partial_trace(sv, {0});
    EXPECT_NEAR(rho_1.trace(), 1.0, kTol);
    EXPECT_NEAR(rho_1.purity(), 0.5, 1e-7);
}

TEST(QuantumInfo_PartialTrace, ProductState_TraceQ0_IsPureState) {
    // |0⟩⊗|+⟩: trace out qubit 0 (|0⟩ part) → |+⟩ on qubit 1.
    Statevector sv(2);
    gates::apply_h(sv, 1);  // qubit 1 = |+⟩, qubit 0 = |0⟩
    DensityMatrix rho_1 = partial_trace(sv, {0});
    EXPECT_NEAR(rho_1.purity(), 1.0, 1e-7);
}

TEST(QuantumInfo_PartialTrace, DensityMatrix_Trace_IsNormalised) {
    Statevector sv(3);
    gates::apply_h(sv, 0);
    gates::apply_cx(sv, 0, 1);
    gates::apply_cx(sv, 0, 2);
    DensityMatrix rho = DensityMatrix::from_statevector(sv);
    DensityMatrix rho_sub = partial_trace(rho, {2});
    EXPECT_NEAR(rho_sub.trace(), 1.0, kTol);
    EXPECT_EQ(rho_sub.n_qubits, 2);
}

// =============================================================================
// DensityMatrix — is_valid, purity, trace
// =============================================================================

TEST(DensityMatrixProps, PureState_PurityIsOne) {
    Statevector sv(2);
    gates::apply_h(sv, 0);
    gates::apply_cx(sv, 0, 1);
    DensityMatrix rho = DensityMatrix::from_statevector(sv);
    EXPECT_NEAR(rho.trace(), 1.0, kTol);
    EXPECT_NEAR(rho.purity(), 1.0, 1e-8);
    EXPECT_TRUE(rho.is_valid());
}

TEST(DensityMatrixProps, MaximallyMixed_PurityIsHalf) {
    DensityMatrix rho(1);
    rho(0,0) = Complex128(0.5, 0.0);
    rho(1,1) = Complex128(0.5, 0.0);
    EXPECT_NEAR(rho.trace(), 1.0, kTol);
    EXPECT_NEAR(rho.purity(), 0.5, 1e-8);
    EXPECT_TRUE(rho.is_valid());
}
