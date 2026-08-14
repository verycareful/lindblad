#pragma once

// =============================================================================
// tests/visualiser_fixtures.hpp : shared circuit fixtures for R.1.10.1
// =============================================================================
// One source of truth for the canonical circuit shapes used across every
// visualiser test suite. Both the test binaries (test_visualiser_*.cpp) and
// the regen tool (visualiser_regen.cpp) construct fixtures via these helpers
// so that golden-file output and EXPECT_EQ assertions stay synchronised.
//
// Every helper is constexpr-free and side-effect-free: it returns a fresh
// QuantumCircuit by value. Callers must not cache the result across tests
// because builders such as h() / cx() append to the instruction list.

#include "lindblad/circuit.hpp"

#include <string>

namespace lindblad::vfx {

// Bell pair preparation : H(0) + CX(0,1) + measure(0,0) + measure(1,1).
// Two qubits, two classical bits. The canonical "does anything work" demo.
inline QuantumCircuit bell_with_measures() {
    QuantumCircuit qc(2, 2, "bell");
    qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    return qc;
}

// Bell pair without measurements : same prep, no classical wires required.
// Useful for clbit-off variants and to keep the LaTeX golden small.
inline QuantumCircuit bell_unmeasured() {
    QuantumCircuit qc(2, 0, "bell_unmeasured");
    qc.h(0).cx(0, 1);
    return qc;
}

// Three-qubit GHZ : H(0) + CX(0,1) + CX(1,2). Tests strut depth and ASAP
// packing of two CX gates that must serialise because they share qubit 1.
inline QuantumCircuit ghz_3q() {
    QuantumCircuit qc(3, 0, "ghz");
    qc.h(0).cx(0, 1).cx(1, 2);
    return qc;
}

// Parametric rotations on three qubits: RX(pi/2) + RY(pi/4) + CRX(pi/3).
// Exercises Tier 1 parameter labels, Tier 2 composite (CRX is ctrl + box)
// and the pi-snap formatter.
inline QuantumCircuit parametric_rotations() {
    QuantumCircuit qc(3, 0, "param_rot");
    constexpr double kPi = PI;  // short local alias, library-sourced value
    qc.rx(kPi / 2.0, 0).ry(kPi / 4.0, 1).crx(kPi / 3.0, 0, 2);
    return qc;
}

// Barrier-then-measure : H(0) + H(1) + BARRIER + measure(0,0) + measure(1,1).
// Tests barrier full-width column break and MEASURE strut placement.
inline QuantumCircuit barrier_and_measure() {
    QuantumCircuit qc(2, 2, "barrier_measure");
    qc.h(0).h(1).barrier({}).measure(0, 0).measure(1, 1);
    return qc;
}

// Conditional gate via add_if : H(0) + measure(0,0) + p_if(pi, 1, 0, 1).
// Tests conditional gate decoration with show_clbits on and off.
inline QuantumCircuit conditional_feedforward() {
    constexpr double kPi = PI;  // short local alias, library-sourced value
    QuantumCircuit qc(2, 1, "feedforward");
    qc.h(0).measure(0, 0).p_if(kPi, 1, 0, 1);
    return qc;
}

// Non-contiguous UNITARY on qubits {0, 3} of a 4-qubit circuit.
// Tests row reservation across intermediate qubits 1 and 2.
inline QuantumCircuit non_contiguous_unitary() {
    QuantumCircuit qc(4, 0, "noncontig_u");
    // 4x4 identity placeholder (the visual is what matters, not the math).
    std::vector<Complex128> mat(16, {0.0, 0.0});
    for (int i = 0; i < 4; ++i) { mat[i * 4 + i] = {1.0, 0.0}; }
    qc.unitary(mat, {0, 3}, "U");
    return qc;
}

// TallBox demo : RXX(pi/4) + RYY(pi/6) + ECR. Three two-qubit interaction
// gates all rendered as single labelled tall boxes spanning two wires.
inline QuantumCircuit tallbox_demo() {
    constexpr double kPi = PI;  // short local alias, library-sourced value
    QuantumCircuit qc(2, 0, "tallbox");
    qc.rxx(kPi / 4.0, 0, 1).ryy(kPi / 6.0, 0, 1).ecr(0, 1);
    return qc;
}

// All single-qubit standard gates in sequence on q0 : H X Y Z S SDG T TDG SX SXDG.
// Tier 1 box coverage for ASCII palette assertions.
inline QuantumCircuit all_single_qubit_unparam() {
    QuantumCircuit qc(1, 0, "all_1q_unparam");
    qc.h(0).x(0).y(0).z(0).s(0).sdg(0).t(0).tdg(0).sx(0).sxdg(0);
    return qc;
}

// Single-qubit parameterised gates on q0 : RX(pi/2) RY(pi/4) RZ(pi/8) P(pi/3) U(pi/2, pi/4, pi/6).
// Tests Tier 1 show_params toggle and pi-snap formatting.
inline QuantumCircuit all_single_qubit_param() {
    QuantumCircuit qc(1, 0, "all_1q_param");
    constexpr double kPi = PI;  // short local alias, library-sourced value
    qc.rx(kPi / 2.0, 0).ry(kPi / 4.0, 0).rz(kPi / 8.0, 0)
      .p(kPi / 3.0, 0).u(kPi / 2.0, kPi / 4.0, kPi / 6.0, 0);
    return qc;
}

// All two-qubit standard gates on q0,q1 : CX CY CZ CH SWAP ISWAP ECR.
// Tests Tier 2 composite catalogue coverage.
inline QuantumCircuit all_two_qubit_unparam() {
    QuantumCircuit qc(2, 0, "all_2q_unparam");
    qc.cx(0, 1).cy(0, 1).cz(0, 1).ch(0, 1).swap(0, 1).iswap(0, 1).ecr(0, 1);
    return qc;
}

// All three-qubit standard gates : CCX CCZ CSWAP RCCX.
// Tests three-qubit composite struts.
inline QuantumCircuit all_three_qubit() {
    QuantumCircuit qc(3, 0, "all_3q");
    qc.ccx(0, 1, 2).ccz(0, 1, 2).cswap(0, 1, 2).rccx(0, 1, 2);
    return qc;
}

// Empty circuit : zero instructions on two qubits. Tests degenerate rendering.
inline QuantumCircuit empty_2q() {
    QuantumCircuit qc(2, 0, "empty");
    return qc;
}

// Reset and re-prepare : reset(0) + H(0). Tests RESET glyph.
inline QuantumCircuit reset_and_h() {
    QuantumCircuit qc(1, 0, "reset_h");
    qc.reset(0).h(0);
    return qc;
}

} // namespace lindblad::vfx
