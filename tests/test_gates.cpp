#include <gtest/gtest.h>
#include "lindblad/statevector.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/types.hpp"

#include <cmath>

using namespace lindblad;

class GateTest : public ::testing::Test {
protected:
    static constexpr double EPS = 1e-10;

    void expect_amplitude(const Statevector& sv, size_t idx, double re, double im) {
        auto amp = sv.amplitude(idx);
        EXPECT_NEAR(amp.real, re, EPS) << "State " << idx << " real part";
        EXPECT_NEAR(amp.imag, im, EPS) << "State " << idx << " imag part";
    }
};

// =============================================================================
// Single-qubit gate tests
// =============================================================================

TEST_F(GateTest, PauliX_FlipsBasis) {
    Statevector sv(1);
    gates::apply_x(sv, 0);
    expect_amplitude(sv, 0, 0.0, 0.0);
    expect_amplitude(sv, 1, 1.0, 0.0);

    gates::apply_x(sv, 0);
    expect_amplitude(sv, 0, 1.0, 0.0);
    expect_amplitude(sv, 1, 0.0, 0.0);
}

TEST_F(GateTest, PauliY_FlipsBasisWithPhase) {
    Statevector sv(1);
    gates::apply_y(sv, 0);
    expect_amplitude(sv, 0, 0.0, 0.0);
    expect_amplitude(sv, 1, 0.0, 1.0);
}

TEST_F(GateTest, PauliZ_PhaseFlip) {
    Statevector sv(1);
    gates::apply_h(sv, 0);
    gates::apply_z(sv, 0);
    double inv_sqrt2 = INV_SQRT2;
    expect_amplitude(sv, 0, inv_sqrt2, 0.0);
    expect_amplitude(sv, 1, -inv_sqrt2, 0.0);
}

TEST_F(GateTest, Hadamard_CreatesSuperposition) {
    Statevector sv(1);
    gates::apply_h(sv, 0);
    double inv_sqrt2 = INV_SQRT2;
    expect_amplitude(sv, 0, inv_sqrt2, 0.0);
    expect_amplitude(sv, 1, inv_sqrt2, 0.0);
}

TEST_F(GateTest, Hadamard_SelfInverse) {
    Statevector sv(1);
    gates::apply_h(sv, 0);
    gates::apply_h(sv, 0);
    expect_amplitude(sv, 0, 1.0, 0.0);
    expect_amplitude(sv, 1, 0.0, 0.0);
}

TEST_F(GateTest, SGate_QuarterPhase) {
    Statevector sv(1);
    gates::apply_x(sv, 0);
    gates::apply_s(sv, 0);
    expect_amplitude(sv, 0, 0.0, 0.0);
    expect_amplitude(sv, 1, 0.0, 1.0);
}

TEST_F(GateTest, TGate_EighthPhase) {
    Statevector sv(1);
    gates::apply_x(sv, 0);
    gates::apply_t(sv, 0);
    double expected_re = std::cos(PI / 4.0);
    double expected_im = std::sin(PI / 4.0);
    expect_amplitude(sv, 1, expected_re, expected_im);
}

TEST_F(GateTest, RX_FullRotation) {
    Statevector sv(1);
    gates::apply_rx(sv, 0, 2.0 * PI);
    // Full rotation should give back |0⟩ (up to global phase -1)
    expect_amplitude(sv, 0, -1.0, 0.0);
    EXPECT_NEAR(sv.probability(1), 0.0, EPS);
}

TEST_F(GateTest, RY_HalfRotation) {
    Statevector sv(1);
    gates::apply_ry(sv, 0, PI);
    // RY(pi)|0⟩ = |1⟩
    EXPECT_NEAR(sv.probability(0), 0.0, EPS);
    EXPECT_NEAR(sv.probability(1), 1.0, EPS);
}

TEST_F(GateTest, NormPreserved_SingleQubit) {
    Statevector sv(3);
    gates::apply_h(sv, 0);
    gates::apply_h(sv, 1);
    gates::apply_h(sv, 2);
    gates::apply_rx(sv, 0, 1.234);
    gates::apply_ry(sv, 1, 2.345);
    gates::apply_rz(sv, 2, 3.456);
    EXPECT_NEAR(sv.norm(), 1.0, EPS);
}

// =============================================================================
// Two-qubit gate tests
// =============================================================================

TEST_F(GateTest, CX_Basic) {
    // CX|10⟩ = |11⟩
    Statevector sv(2);
    gates::apply_x(sv, 0);  // |10⟩
    gates::apply_cx(sv, 0, 1);
    EXPECT_NEAR(sv.probability(3), 1.0, EPS);
}

TEST_F(GateTest, CX_NoFlipOnZeroControl) {
    // CX|00⟩ = |00⟩
    Statevector sv(2);
    gates::apply_cx(sv, 0, 1);
    EXPECT_NEAR(sv.probability(0), 1.0, EPS);
}

TEST_F(GateTest, SWAP_Exchanges) {
    // SWAP|10⟩ = |01⟩
    Statevector sv(2);
    gates::apply_x(sv, 0);  // |10⟩
    gates::apply_swap(sv, 0, 1);
    EXPECT_NEAR(sv.probability(2), 1.0, EPS);
}

TEST_F(GateTest, CZ_PhaseFlip) {
    // CZ|11⟩ = -|11⟩
    Statevector sv(2);
    gates::apply_x(sv, 0);
    gates::apply_x(sv, 1);  // |11⟩
    gates::apply_cz(sv, 0, 1);
    expect_amplitude(sv, 3, -1.0, 0.0);
}

TEST_F(GateTest, BellState) {
    // H|0⟩ then CX → Bell state (|00⟩ + |11⟩)/√2
    Statevector sv(2);
    gates::apply_h(sv, 0);
    gates::apply_cx(sv, 0, 1);
    EXPECT_NEAR(sv.probability(0), 0.5, EPS);
    EXPECT_NEAR(sv.probability(3), 0.5, EPS);
}

// =============================================================================
// Three-qubit gate tests
// =============================================================================

TEST_F(GateTest, CCX_Toffoli) {
    // CCX|110⟩ = |111⟩
    Statevector sv(3);
    gates::apply_x(sv, 0);
    gates::apply_x(sv, 1);  // |110⟩
    gates::apply_ccx(sv, 0, 1, 2);
    EXPECT_NEAR(sv.probability(7), 1.0, EPS);
}

TEST_F(GateTest, CCX_NoFlipOneControl) {
    // CCX|100⟩ = |100⟩
    Statevector sv(3);
    gates::apply_x(sv, 0);  // |100⟩
    gates::apply_ccx(sv, 0, 1, 2);
    EXPECT_NEAR(sv.probability(1), 1.0, EPS);
}

TEST_F(GateTest, NormPreserved_MultiQubit) {
    Statevector sv(4);
    gates::apply_h(sv, 0);
    gates::apply_h(sv, 1);
    gates::apply_cx(sv, 0, 1);
    gates::apply_ccx(sv, 0, 1, 2);
    gates::apply_swap(sv, 2, 3);
    EXPECT_NEAR(sv.norm(), 1.0, EPS);
}
