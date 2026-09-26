// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.2 - which gamma drives which MA-QAOA cost term.
//
// Two defects in one place.
//
// An all-identity cost term (the constant offset every Ising-derived cost
// carries) drives no gate, and the cost layer skips it, yet under orbit sharing
// and term-indexed gammas it still took a gamma: a dimension the optimiser
// searched that moved nothing, and a meaningless entry in optimal_params. The
// mixer layout already gave identity terms no beta. Now the cost layout gives
// them no gamma either, so a cost with and without its offset has the same
// parameters and builds the same circuit.
//
// The gamma for a term was chosen by testing whether the gamma count equalled
// the term count, not by reading the mode. Under the default qubit-indexed
// gammas the count is n_qubits, so a cost with exactly n_qubits terms (a ring
// MaxCut, the commonest QAOA benchmark) was silently given one gamma per term
// instead of the documented one gamma per lowest active qubit. The mode is now
// read from the options.
//
// Every expectation here is derived from the documented layout, and the ring
// uses distinct gammas so that the two layouts give different angles.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/ising.hpp"
#include "lindblad/operators.hpp"

#include <limits>
#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;
using GT = Instruction::GateType;

namespace {

constexpr double kTol = 64.0 * std::numeric_limits<double>::epsilon();

// Three-qubit ring with distinct weights: as many terms as qubits.
SparsePauliOp ring_3q() {
    return SparsePauliOp::from_list({{"ZZI", Complex128(1.0, 0.0)},
                                     {"IZZ", Complex128(0.5, 0.0)},
                                     {"ZIZ", Complex128(0.25, 0.0)}});
}

// The same ring with a constant offset, placed first so that a layout giving
// it a slot would shift every other term's gamma.
SparsePauliOp ring_3q_with_offset() {
    return SparsePauliOp::from_list({{"III", Complex128(0.75, 0.0)},
                                     {"ZZI", Complex128(1.0, 0.0)},
                                     {"IZZ", Complex128(0.5, 0.0)},
                                     {"ZIZ", Complex128(0.25, 0.0)}});
}

// The RZ angles of a circuit, in order: one per ZZ cost term here.
std::vector<double> rz_angles(const QuantumCircuit& qc) {
    std::vector<double> out;
    for (const auto& inst : qc.instructions)
        if (inst.type == GT::RZ) out.push_back(inst.params[0]);
    return out;
}

bool same_circuit(const QuantumCircuit& a, const QuantumCircuit& b) {
    if (a.n_qubits != b.n_qubits || a.instructions.size() != b.instructions.size()) return false;
    for (std::size_t i = 0; i < a.instructions.size(); ++i) {
        const auto& x = a.instructions[i];
        const auto& y = b.instructions[i];
        if (x.type != y.type || x.qubits != y.qubits || x.params != y.params) return false;
    }
    return true;
}

}  // namespace

// =============================================================================
// Qubit-indexed gammas (the default)
// =============================================================================

TEST(V11292MaqaoaGammas, QubitIndexedGammasFollowTheLowestActiveQubit) {
    // Terms ZZI, IZZ and ZIZ have lowest active qubits 0, 1 and 0, so gamma 0
    // drives the first and the last and gamma 2 drives nothing. The layout this
    // replaces gave the last term gamma 2.
    MAQAOA m;
    m.options.p = 1;
    const SparsePauliOp cost = ring_3q();
    ASSERT_EQ(m.num_parameters(cost), 3 + 3);

    const double g0 = 0.3, g1 = 0.7, g2 = 1.3;
    const QuantumCircuit qc = m.build_circuit(cost, SparsePauliOp(), {g0, g1, g2, 0.1, 0.2, 0.4});
    const std::vector<double> angles = rz_angles(qc);
    ASSERT_EQ(angles.size(), 3u);
    EXPECT_NEAR(angles[0], 2.0 * g0 * cost.terms[0].coeff.real, kTol);
    EXPECT_NEAR(angles[1], 2.0 * g1 * cost.terms[1].coeff.real, kTol);
    EXPECT_NEAR(angles[2], 2.0 * g0 * cost.terms[2].coeff.real, kTol)
        << "the ZIZ term took a gamma of its own rather than qubit 0's";
}

// =============================================================================
// All-identity terms
// =============================================================================

TEST(V11292MaqaoaGammas, AnOffsetTakesNoGammaInAnyMode) {
    struct Mode {
        const char* name;
        bool term_indexed;
        std::vector<int> orbits;
    };
    const Mode modes[] = {{"qubit-indexed", false, {}},
                          {"term-indexed", true, {}},
                          {"orbits", false, {0, 0, 1}}};
    for (const Mode& mode : modes) {
        SCOPED_TRACE(mode.name);
        MAQAOA m;
        m.options.p = 2;
        m.options.term_indexed_gammas = mode.term_indexed;
        m.options.orbit_assignments = mode.orbits;
        EXPECT_EQ(m.num_parameters(ring_3q_with_offset()), m.num_parameters(ring_3q()));
    }
}

TEST(V11292MaqaoaGammas, AnOffsetLeavesTheCircuitUnchanged) {
    // With no slot for the offset, every other term keeps its gamma, so the
    // same parameters build the same gates.
    MAQAOA m;
    m.options.p = 1;
    m.options.term_indexed_gammas = true;
    const std::vector<double> params = {0.3, 0.7, 1.3, 0.1, 0.2, 0.4};
    ASSERT_EQ(m.num_parameters(ring_3q_with_offset()), static_cast<int>(params.size()));
    EXPECT_TRUE(same_circuit(m.build_circuit(ring_3q_with_offset(), SparsePauliOp(), params),
                             m.build_circuit(ring_3q(), SparsePauliOp(), params)));
}

TEST(V11292MaqaoaGammas, TheOptimiserSearchesOnlyLiveGammas) {
    // An Ising model built from h, J and an offset, as a caller would.
    const std::vector<std::vector<double>> J = {{0.0, 1.0, 0.25}, {0.0, 0.0, 0.5}, {0.0, 0.0, 0.0}};
    const SparsePauliOp cost = IsingHamiltonian::from_hJ({0.0, 0.0, 0.0}, J, 0.75).to_sparse_pauli_op();

    int live_terms = 0;
    for (const auto& term : cost.terms)
        if (term.pauli.find_first_not_of('I') != std::string::npos) ++live_terms;
    ASSERT_LT(live_terms, static_cast<int>(cost.terms.size())) << "premise: the cost has an offset";

    MAQAOA m;
    m.options.p = 1;
    m.options.term_indexed_gammas = true;
    m.options.max_iterations = 5;
    m.options.seed = 11292;
    const int n = m.num_parameters(cost);
    EXPECT_EQ(n, live_terms + cost.n_qubits());
    const auto res = m.optimize(cost);
    EXPECT_EQ(res.optimal_params.size(), static_cast<std::size_t>(n));
}
