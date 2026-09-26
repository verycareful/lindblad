// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.2 - MPSState::apply_two_qubit_gate reads its matrix in the project
// convention.
//
// The frozen conventions say every externally supplied multi-qubit matrix is
// qubits[0]-is-LSB, as gates::apply_unitary reads it. The public MPS primitive
// read its 4x4 with the first operand as the HIGH bit instead, so a caller
// porting a gate from the statevector or the density matrix applied it with
// the two qubits' roles exchanged. Every test that called it passed either a
// gate symmetric in its two qubits or a matrix written for the MSB-first
// reading, so nothing pinned the order. The primitive now takes the project
// order and converts to the contraction's own order internally (a breaking
// change for any caller that wrote MSB-first matrices).
//
// The oracle is gates::apply_unitary on a statevector given the SAME matrix, on
// a non-symmetric state, for every ordered pair of three qubits: adjacent,
// non-adjacent and reversed operands each take a different path through the
// SWAP chain and the operand reorder.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 256.0 * std::numeric_limits<double>::epsilon();

// A two-qubit unitary with no symmetry under exchanging its operands: a CX
// between two different single-qubit rotations.
std::vector<Complex128> asymmetric_gate() {
    QuantumCircuit qc(2);
    qc.ry(0.3, 0);
    qc.rz(0.7, 1);
    qc.cx(0, 1);
    qc.rx(1.1, 1);
    return Operator::from_circuit(qc).data;
}

std::array<Complex128, 16> as_array16(const std::vector<Complex128>& m) {
    std::array<Complex128, 16> a{};
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = m[i];
    return a;
}

// Distinct single-qubit rotations on each qubit, so every basis amplitude of
// the three-qubit state is different and a role exchange cannot hide.
std::vector<std::vector<Complex128>> preparation() {
    std::vector<std::vector<Complex128>> out;
    for (double angle : {0.4, 1.3, 2.2}) {
        QuantumCircuit qc(1);
        qc.ry(angle, 0);
        qc.rz(angle / 2.0, 0);
        out.push_back(Operator::from_circuit(qc).data);
    }
    return out;
}

std::array<Complex128, 4> as_array4(const std::vector<Complex128>& m) {
    return {m[0], m[1], m[2], m[3]};
}

}  // namespace

TEST(V11292MpsTwoQubitOrder, EveryOrderedPairMatchesTheStatevector) {
    const std::vector<Complex128> u = asymmetric_gate();
    const auto prep = preparation();
    const std::pair<int, int> pairs[] = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {0, 2}, {2, 0}};

    for (const auto& [a, b] : pairs) {
        SCOPED_TRACE(std::to_string(a) + ", " + std::to_string(b));
        MPSState mps(3);
        Statevector sv(3);
        for (int q = 0; q < 3; ++q) {
            mps.apply_single_qubit_gate(as_array4(prep[static_cast<std::size_t>(q)]), q);
            gates::apply_unitary(sv, {q}, prep[static_cast<std::size_t>(q)]);
        }
        mps.apply_two_qubit_gate(as_array16(u), a, b);
        gates::apply_unitary(sv, {a, b}, u);

        const std::vector<Complex128> got = mps.to_statevector().amplitudes();
        const std::vector<Complex128> want = sv.amplitudes();
        ASSERT_EQ(got.size(), want.size());
        for (std::size_t k = 0; k < got.size(); ++k) {
            EXPECT_NEAR(got[k].real, want[k].real, kTol) << "amplitude " << k;
            EXPECT_NEAR(got[k].imag, want[k].imag, kTol) << "amplitude " << k;
        }
    }
}

TEST(V11292MpsTwoQubitOrder, TheControlOfACxIsItsFirstOperand) {
    // CX with its control on bit 0 of the index, the project layout: basis
    // states 1 and 3 exchange. With qubit 0 set, CX(0 -> 1) sets qubit 1 too,
    // and CX(1 -> 0) leaves the state alone because qubit 1 is clear.
    std::array<Complex128, 16> cx{};
    cx[0 * 4 + 0] = cx[1 * 4 + 3] = cx[2 * 4 + 2] = cx[3 * 4 + 1] = Complex128(1.0, 0.0);
    const std::array<Complex128, 4> x = {Complex128(0.0, 0.0), Complex128(1.0, 0.0),
                                         Complex128(1.0, 0.0), Complex128(0.0, 0.0)};

    MPSState forward(2);
    forward.apply_single_qubit_gate(x, 0);
    forward.apply_two_qubit_gate(cx, 0, 1);
    const auto f = forward.to_statevector().amplitudes();
    EXPECT_NEAR(f[3].real, 1.0, kTol) << "control 0, target 1 did not reach |11>";

    MPSState backward(2);
    backward.apply_single_qubit_gate(x, 0);
    backward.apply_two_qubit_gate(cx, 1, 0);
    const auto g = backward.to_statevector().amplitudes();
    EXPECT_NEAR(g[1].real, 1.0, kTol) << "control 1 was clear, yet the state moved";
}
