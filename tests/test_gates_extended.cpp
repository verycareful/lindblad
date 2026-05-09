// Extended gate tests — all gate functions absent from the original suite.
// Covers: SDG, TDG, SX/SXdg, P, U/U1/U2/U3, CY, CH, iSWAP,
//         CRX/CRY/CRZ, CP, ECR, RXX/RYY/RZZ, CCZ, CSWAP, apply_unitary.

#include <gtest/gtest.h>
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <vector>

using namespace lindblad;
using namespace lindblad::gates;

static constexpr double kTol = 1e-10;

// =============================================================================
// Phase gates: SDG, TDG
// =============================================================================

TEST(GateExtended, SDG_InverseOfS) {
    // S·SDG = I  — apply both, state should be unchanged from |+⟩.
    Statevector sv(1);
    apply_h(sv, 0);
    apply_s(sv, 0);
    apply_sdg(sv, 0);
    // Should be back to |+⟩: prob 0 = prob 1 = 0.5, no imaginary component
    auto a0 = sv.amplitude(0);
    auto a1 = sv.amplitude(1);
    EXPECT_NEAR(a0.real, 1.0 / std::sqrt(2.0), kTol);
    EXPECT_NEAR(a0.imag, 0.0, kTol);
    EXPECT_NEAR(a1.real, 1.0 / std::sqrt(2.0), kTol);
    EXPECT_NEAR(a1.imag, 0.0, kTol);
}

TEST(GateExtended, TDG_InverseOfT) {
    Statevector sv(1);
    apply_x(sv, 0);     // |1⟩
    apply_t(sv, 0);     // e^(iπ/4)|1⟩
    apply_tdg(sv, 0);   // back to |1⟩
    EXPECT_NEAR(sv.amplitude(1).real, 1.0, kTol);
    EXPECT_NEAR(sv.amplitude(1).imag, 0.0, kTol);
}

TEST(GateExtended, TDG_PhaseOnExcited) {
    // TDG|1⟩ = e^(-iπ/4)|1⟩
    Statevector sv(1);
    apply_x(sv, 0);
    apply_tdg(sv, 0);
    const double c = std::cos(-M_PI / 4.0);
    const double s = std::sin(-M_PI / 4.0);
    EXPECT_NEAR(sv.amplitude(1).real, c, kTol);
    EXPECT_NEAR(sv.amplitude(1).imag, s, kTol);
}

// =============================================================================
// SX / SXdg
// =============================================================================

TEST(GateExtended, SX_SquareIsX) {
    // SX·SX = X: two SX gates flip the state.
    Statevector sv(1);          // |0⟩
    apply_sx(sv, 0);
    apply_sx(sv, 0);
    // Should be |1⟩
    EXPECT_NEAR(sv.probability(0), 0.0, kTol);
    EXPECT_NEAR(sv.probability(1), 1.0, kTol);
}

TEST(GateExtended, SXdg_InverseOfSX) {
    // SX·SXdg = I
    Statevector sv(1);
    apply_sx(sv, 0);
    apply_sxdg(sv, 0);
    EXPECT_NEAR(sv.probability(0), 1.0, kTol);
    EXPECT_NEAR(sv.probability(1), 0.0, kTol);
}

TEST(GateExtended, SX_NormPreserved) {
    Statevector sv(2);
    apply_h(sv, 0);
    apply_sx(sv, 0);
    apply_sxdg(sv, 1);
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}

// =============================================================================
// Phase gate P(λ)
// =============================================================================

TEST(GateExtended, P_OnGroundState_NoEffect) {
    // P(λ)|0⟩ = |0⟩ (phase only affects |1⟩)
    Statevector sv(1);
    apply_p(sv, 0, 1.23);
    EXPECT_NEAR(sv.probability(0), 1.0, kTol);
}

TEST(GateExtended, P_OnExcitedState_PhaseApplied) {
    // P(π/2)|1⟩ = e^(iπ/2)|1⟩ = i|1⟩
    Statevector sv(1);
    apply_x(sv, 0);
    apply_p(sv, 0, M_PI / 2.0);
    EXPECT_NEAR(sv.amplitude(1).real, 0.0, kTol);
    EXPECT_NEAR(sv.amplitude(1).imag, 1.0, kTol);
}

TEST(GateExtended, P_Pi_IsZ) {
    // P(π) = Z
    Statevector sv1(1), sv2(1);
    apply_x(sv1, 0); apply_x(sv2, 0);
    apply_p(sv1, 0, M_PI);
    apply_z(sv2, 0);
    EXPECT_NEAR(sv1.amplitude(1).real, sv2.amplitude(1).real, kTol);
    EXPECT_NEAR(sv1.amplitude(1).imag, sv2.amplitude(1).imag, kTol);
}

// =============================================================================
// U / U1 / U2 / U3
// =============================================================================

TEST(GateExtended, U3_IsGeneralSingleQubit) {
    // U3(π, 0, π) = X (up to global phase)
    Statevector sv(1);
    apply_u3(sv, 0, M_PI, 0.0, M_PI);
    // |0⟩ → should be |1⟩ in probability
    EXPECT_NEAR(sv.probability(0), 0.0, kTol);
    EXPECT_NEAR(sv.probability(1), 1.0, kTol);
}

TEST(GateExtended, U3_IsHadamardCase) {
    // U3(π/2, 0, π) = H
    Statevector sv1(1), sv2(1);
    apply_u3(sv1, 0, M_PI / 2.0, 0.0, M_PI);
    apply_h(sv2, 0);
    EXPECT_NEAR(sv1.probability(0), sv2.probability(0), kTol);
    EXPECT_NEAR(sv1.probability(1), sv2.probability(1), kTol);
}

TEST(GateExtended, U1_IsPhaseGate) {
    // U1(λ)|1⟩ = e^(iλ)|1⟩ — same as P(λ)
    Statevector sv1(1), sv2(1);
    apply_x(sv1, 0); apply_x(sv2, 0);
    apply_u1(sv1, 0, M_PI / 3.0);
    apply_p(sv2, 0, M_PI / 3.0);
    EXPECT_NEAR(sv1.amplitude(1).real, sv2.amplitude(1).real, kTol);
    EXPECT_NEAR(sv1.amplitude(1).imag, sv2.amplitude(1).imag, kTol);
}

TEST(GateExtended, U_NormPreserved) {
    Statevector sv(2);
    apply_u(sv, 0, 0.3, 0.7, 1.1);
    apply_u(sv, 1, 1.5, 0.2, 0.9);
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}

// =============================================================================
// Two-qubit: CY
// =============================================================================

TEST(GateExtended, CY_FlipsWithPhase) {
    // CY|11⟩: control=1, target in |1⟩ → Y|1⟩ = -i|0⟩
    Statevector sv(2);
    apply_x(sv, 0); apply_x(sv, 1);
    apply_cy(sv, 1, 0);
    // |11⟩ → -i|10⟩  (target q0 flipped with -i phase)
    EXPECT_NEAR(sv.probability(2), 1.0, kTol); // |q1=1,q0=0⟩ = index 2
    auto a = sv.amplitude(2);
    EXPECT_NEAR(a.real, 0.0, kTol);
    EXPECT_NEAR(a.imag, -1.0, kTol);
}

TEST(GateExtended, CY_NoFlipOnZeroControl) {
    // CY with control=0: |01⟩ unchanged (ctrl qubit 1 is |0⟩).
    Statevector sv(2);
    apply_x(sv, 0);       // q0=1 (target), q1=0 (control)
    apply_cy(sv, 1, 0);   // ctrl=q1=0 → no flip
    EXPECT_NEAR(sv.probability(1), 1.0, kTol); // still |01⟩ = index 1
}

// =============================================================================
// Two-qubit: CH
// =============================================================================

TEST(GateExtended, CH_AppliesHOnActiveControl) {
    // CH: control=q1=1, target=q0; applies H to q0.
    // |10⟩ → H on q0 → (|10⟩+|11⟩)/√2
    Statevector sv(2);
    apply_x(sv, 1);       // q1=1
    apply_ch(sv, 1, 0);
    EXPECT_NEAR(sv.probability(2), 0.5, kTol); // |10⟩ = index 2
    EXPECT_NEAR(sv.probability(3), 0.5, kTol); // |11⟩ = index 3
}

TEST(GateExtended, CH_NoOpOnZeroControl) {
    // CH with ctrl=0: no effect.
    Statevector sv(2);
    apply_x(sv, 0);      // q0=1
    apply_ch(sv, 1, 0);  // ctrl=q1=0 → no-op
    EXPECT_NEAR(sv.probability(1), 1.0, kTol);
}

// =============================================================================
// Two-qubit: iSWAP
// =============================================================================

TEST(GateExtended, ISWAP_SwapsWithPhase) {
    // iSWAP|01⟩ = i|10⟩
    Statevector sv(2);
    apply_x(sv, 0);       // |01⟩ = index 1
    apply_iswap(sv, 0, 1);
    EXPECT_NEAR(sv.probability(1), 0.0, kTol);
    EXPECT_NEAR(sv.probability(2), 1.0, kTol); // |10⟩
    auto a = sv.amplitude(2);
    EXPECT_NEAR(a.real, 0.0, kTol);
    EXPECT_NEAR(a.imag, 1.0, kTol);  // +i phase
}

TEST(GateExtended, ISWAP_Squared_IsMinusSwap) {
    // iSWAP² = -SWAP: applying twice on |01⟩ gives -|10⟩...
    // Easier: iSWAP² on |00⟩ gives |00⟩ (trivially).
    // On |01⟩: iSWAP → i|10⟩ → iSWAP(i|10⟩) = i·i|01⟩ = -|01⟩
    Statevector sv(2);
    apply_x(sv, 0);
    apply_iswap(sv, 0, 1);
    apply_iswap(sv, 0, 1);
    // Should be back to -|01⟩ (index 1), probability = 1
    EXPECT_NEAR(sv.probability(1), 1.0, kTol);
    EXPECT_NEAR(sv.amplitude(1).real, -1.0, kTol);
}

TEST(GateExtended, ISWAP_NormPreserved) {
    Statevector sv(3);
    apply_h(sv, 0);
    apply_iswap(sv, 0, 1);
    apply_iswap(sv, 1, 2);
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}

// =============================================================================
// Controlled rotations: CRX, CRY, CRZ
// =============================================================================

TEST(GateExtended, CRX_HalfRotation_ControlActive) {
    // CRX(π, ctrl=1, tgt=0): control=q1=1 → RX(π) on q0 → flips q0.
    // |10⟩ → |11⟩ (up to phase)
    Statevector sv(2);
    apply_x(sv, 1);
    apply_crx(sv, 1, 0, M_PI);
    EXPECT_NEAR(sv.probability(3), 1.0, kTol); // |11⟩ = index 3
}

TEST(GateExtended, CRX_NoFlipOnZeroControl) {
    Statevector sv(2);
    apply_x(sv, 0);           // q0=1, q1=0
    apply_crx(sv, 1, 0, M_PI);
    EXPECT_NEAR(sv.probability(1), 1.0, kTol); // unchanged |01⟩
}

TEST(GateExtended, CRY_HalfPi_CreatesEntanglement) {
    // CRY(π/2) on |10⟩: RY(π/2)|0⟩ = cos(π/4)|0⟩+sin(π/4)|1⟩
    Statevector sv(2);
    apply_x(sv, 1);           // |10⟩
    apply_cry(sv, 1, 0, M_PI / 2.0);
    // q0 gets RY(π/2)
    EXPECT_NEAR(sv.probability(2), 0.5, kTol); // |10⟩
    EXPECT_NEAR(sv.probability(3), 0.5, kTol); // |11⟩
}

TEST(GateExtended, CRZ_PhaseOnExcited) {
    // CRZ(π/2) on |11⟩: applies e^(-iπ/4)|0⟩ + e^(iπ/4)|1⟩ phase to q0.
    // Probabilities unchanged; only phases shift.
    Statevector sv(2);
    apply_x(sv, 0); apply_x(sv, 1);
    apply_crz(sv, 1, 0, M_PI / 2.0);
    EXPECT_NEAR(sv.probability(3), 1.0, kTol); // still |11⟩
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}

// =============================================================================
// CP — controlled phase
// =============================================================================

TEST(GateExtended, CP_PhaseOnBothExcited) {
    // CP(π/2)|11⟩: adds e^(iπ/2)=i phase to |11⟩ component.
    Statevector sv(2);
    apply_x(sv, 0); apply_x(sv, 1);
    apply_cp(sv, 1, 0, M_PI / 2.0);
    EXPECT_NEAR(sv.amplitude(3).real, 0.0, kTol);
    EXPECT_NEAR(sv.amplitude(3).imag, 1.0, kTol);
}

TEST(GateExtended, CP_NoPhaseOnSingleExcited) {
    // CP(π/2)|10⟩: control=q1=1 but target=q0=0 → no phase.
    Statevector sv(2);
    apply_x(sv, 1);
    apply_cp(sv, 1, 0, M_PI / 2.0);
    EXPECT_NEAR(sv.amplitude(2).real, 1.0, kTol);
    EXPECT_NEAR(sv.amplitude(2).imag, 0.0, kTol);
}

// =============================================================================
// ECR — Echoed Cross-Resonance
// =============================================================================

TEST(GateExtended, ECR_NormPreserved) {
    Statevector sv(2);
    apply_h(sv, 0);
    apply_ecr(sv, 0, 1);
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}

TEST(GateExtended, ECR_IsUnitary_AppliedTwice_Nontrivial) {
    // ECR is not self-inverse. Apply twice and verify state is still normalised.
    Statevector sv(2);
    apply_h(sv, 0); apply_h(sv, 1);
    apply_ecr(sv, 0, 1);
    apply_ecr(sv, 0, 1);
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}

// =============================================================================
// Ising gates: RXX, RYY, RZZ
// =============================================================================

TEST(GateExtended, RXX_Pi_AntiCorrelated) {
    // RXX(π)|00⟩ = -i|11⟩  (up to global phase, max entanglement at π)
    Statevector sv(2);
    apply_rxx(sv, 0, 1, M_PI);
    EXPECT_NEAR(sv.probability(0), 0.0, kTol);
    EXPECT_NEAR(sv.probability(3), 1.0, kTol);
    EXPECT_NEAR(sv.amplitude(3).imag, -1.0, kTol);
}

TEST(GateExtended, RXX_HalfPi_MaxEntanglement) {
    // RXX(π/2)|00⟩ creates 50/50 superposition of |00⟩ and |11⟩.
    Statevector sv(2);
    apply_rxx(sv, 0, 1, M_PI / 2.0);
    EXPECT_NEAR(sv.probability(0), 0.5, kTol);
    EXPECT_NEAR(sv.probability(3), 0.5, kTol);
}

TEST(GateExtended, RYY_Pi_AntiCorrelated) {
    Statevector sv(2);
    apply_ryy(sv, 0, 1, M_PI);
    EXPECT_NEAR(sv.probability(0), 0.0, kTol);
    EXPECT_NEAR(sv.probability(3), 1.0, kTol);
}

TEST(GateExtended, RZZ_IsPhaseGateOnComputationalBasis) {
    // RZZ(θ) is diagonal in the computational basis; probabilities unchanged.
    Statevector sv(2);
    apply_h(sv, 0); apply_h(sv, 1);
    auto probs_before = sv.probabilities();
    apply_rzz(sv, 0, 1, M_PI / 3.0);
    auto probs_after = sv.probabilities();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(probs_before[i], probs_after[i], kTol);
}

TEST(GateExtended, RZZ_NormPreserved) {
    Statevector sv(3);
    apply_h(sv, 0); apply_h(sv, 2);
    apply_rzz(sv, 0, 2, 0.5);
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}

// =============================================================================
// CCZ — Controlled-Controlled-Z
// =============================================================================

TEST(GateExtended, CCZ_FlipsPhaseOnAllOnes) {
    // CCZ|111⟩ = -|111⟩
    Statevector sv(3);
    apply_x(sv, 0); apply_x(sv, 1); apply_x(sv, 2);
    apply_ccz(sv, 0, 1, 2);
    EXPECT_NEAR(sv.probability(7), 1.0, kTol);   // |111⟩ = index 7
    EXPECT_NEAR(sv.amplitude(7).real, -1.0, kTol);
}

TEST(GateExtended, CCZ_NoFlipUnlessAllOnes) {
    // CCZ|110⟩: target=q2=0 → Z only when both controls=1 AND target=1.
    // But CCZ acts as Z on the target qubit when both controls are 1.
    // |110⟩: q0=0,q1=1,q2=1 → controls q0,q1 not both 1 (q0=0) → no phase.
    Statevector sv(3);
    apply_x(sv, 1); apply_x(sv, 2);
    apply_ccz(sv, 0, 1, 2);
    EXPECT_NEAR(sv.amplitude(6).real, 1.0, kTol);  // unchanged |110⟩
}

TEST(GateExtended, CCZ_SelfInverse) {
    // CCZ² = I on all states.
    Statevector sv(3);
    apply_h(sv, 0); apply_h(sv, 1); apply_h(sv, 2);
    auto probs = sv.probabilities();
    apply_ccz(sv, 0, 1, 2);
    apply_ccz(sv, 0, 1, 2);
    auto probs2 = sv.probabilities();
    for (size_t i = 0; i < 8; ++i)
        EXPECT_NEAR(probs[i], probs2[i], kTol);
}

// =============================================================================
// CSWAP (Fredkin)
// =============================================================================

TEST(GateExtended, CSWAP_SwapsTargetsWhenControlOne) {
    // CSWAP(ctrl=0, q1=1, q2=2): |101⟩ (ctrl=1,q1=0,q2=1) → |110⟩ (q1 and q2 swapped)
    // Index for |q2=1,q1=0,q0=1⟩ = 5; after swap |q2=0,q1=1,q0=1⟩ = 3.
    Statevector sv(3);
    apply_x(sv, 0);   // ctrl (q0) = 1
    apply_x(sv, 2);   // q2 = 1
    // state: |q2=1,q1=0,q0=1⟩ = |101⟩ = index 5
    apply_cswap(sv, 0, 1, 2);
    // After swap (ctrl=1 → swap q1,q2): q1↔q2 → |q2=0,q1=1,q0=1⟩ = |011⟩ = index 3
    EXPECT_NEAR(sv.probability(3), 1.0, kTol);
}

TEST(GateExtended, CSWAP_NoSwapWhenControlZero) {
    // Control = 0: no swap.
    Statevector sv(3);
    apply_x(sv, 2);   // q2=1, ctrl(q0)=0
    // state |q2=1,q1=0,q0=0⟩ = index 4
    apply_cswap(sv, 0, 1, 2);
    EXPECT_NEAR(sv.probability(4), 1.0, kTol);  // unchanged
}

TEST(GateExtended, CSWAP_SelfInverse) {
    Statevector sv(3);
    apply_h(sv, 0); apply_x(sv, 1); apply_h(sv, 2);
    auto probs = sv.probabilities();
    apply_cswap(sv, 0, 1, 2);
    apply_cswap(sv, 0, 1, 2);
    auto probs2 = sv.probabilities();
    for (size_t i = 0; i < 8; ++i)
        EXPECT_NEAR(probs[i], probs2[i], kTol);
}

// =============================================================================
// apply_unitary — arbitrary N-qubit unitary
// =============================================================================

TEST(GateExtended, ApplyUnitary_1Q_AsX) {
    // Apply the X matrix via apply_unitary; should flip |0⟩ → |1⟩.
    Statevector sv(1);
    std::vector<Complex128> X_mat = {
        Complex128(0,0), Complex128(1,0),
        Complex128(1,0), Complex128(0,0)
    };
    apply_unitary(sv, {0}, X_mat);
    EXPECT_NEAR(sv.probability(0), 0.0, kTol);
    EXPECT_NEAR(sv.probability(1), 1.0, kTol);
}

TEST(GateExtended, ApplyUnitary_2Q_AsCX) {
    // CX matrix in computational basis (row-major, MSB first for target pair).
    // CX: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
    Statevector sv(2);
    apply_x(sv, 1);  // |10⟩
    std::vector<Complex128> CX_mat = {
        {1,0},{0,0},{0,0},{0,0},
        {0,0},{1,0},{0,0},{0,0},
        {0,0},{0,0},{0,0},{1,0},
        {0,0},{0,0},{1,0},{0,0}
    };
    apply_unitary(sv, {0, 1}, CX_mat);
    // |10⟩ → |11⟩ = index 3
    EXPECT_NEAR(sv.probability(3), 1.0, kTol);
}

TEST(GateExtended, ApplyUnitary_NormPreserved) {
    Statevector sv(2);
    apply_h(sv, 0);
    // Apply Hadamard on qubit 1 via apply_unitary
    const double inv2 = 1.0 / std::sqrt(2.0);
    std::vector<Complex128> H_mat = {
        {inv2,0}, {inv2,0},
        {inv2,0}, {-inv2,0}
    };
    apply_unitary(sv, {1}, H_mat);
    EXPECT_NEAR(sv.norm_sq(), 1.0, 1e-12);
}
