// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.29.1 test wave - RXX, RYY, RZZ and RZX on the stabilizer tableau.
//
// 1.1.29.0 taught the tableau the four two-qubit Ising rotations at multiples
// of pi/2, and taught is_clifford() to accept them on the same grid, so a
// circuit using them no longer leaves the fast path under automatic selection.
// Each rotation is checked here against the statevector on every Pauli
// expectation, which is equality of the states up to the global phase the
// stabilizer formalism does not carry. The input state is a non-symmetric
// stabilizer state, and every rotation is applied in both operand orders, so
// RZX (which is not symmetric in its operands) and any operand swap in the
// decomposition would show.
//
// The tableau has two layouts: the bit-sliced one the gate pass normally runs
// on, and the row-major one it switches to when a run plan attaches an
// observer. Both are exercised, as are both routes from the final tableau to
// sampled counts.
//
// Acceptance and execution are one question asked twice. is_clifford() is
// what automatic selection consults, and run() is what executes; an angle one
// accepts and the other rejects either fails a run that selection sent to the
// tableau, or leaves a Clifford circuit on a slower backend. The grid tests
// ask both on the same angles.

#include <gtest/gtest.h>

#include "v11251_clifford_oracle.hpp"

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace lindblad;

using v11251::all_pauli_strings;
using v11251::clifford_matches_statevector_exhaustive;
using v11251::exact_statevector_distribution;
using v11251::sv_expectation_pauli_sign;

namespace {

enum class Ising { XX, YY, ZZ, ZX };
constexpr Ising kGates[] = {Ising::XX, Ising::YY, Ising::ZZ, Ising::ZX};

const char* name(Ising g) {
    switch (g) {
        case Ising::XX: return "rxx";
        case Ising::YY: return "ryy";
        case Ising::ZZ: return "rzz";
        case Ising::ZX: return "rzx";
    }
    return "?";
}

void apply(QuantumCircuit& qc, Ising g, double theta, int a, int b) {
    switch (g) {
        case Ising::XX: qc.rxx(theta, a, b); break;
        case Ising::YY: qc.ryy(theta, a, b); break;
        case Ising::ZZ: qc.rzz(theta, a, b); break;
        case Ising::ZX: qc.rzx(theta, a, b); break;
    }
}

// A three-qubit stabilizer state with no symmetry between any two qubits:
// qubit 0 in |+i>, qubits 1 and 2 entangled with qubit 2 rotated off the Z
// axis. Followed by the rotation under test, then more Cliffords, so the
// rotation's effect is carried into correlations it did not create.
QuantumCircuit sandwich(Ising g, double theta, int a, int b) {
    QuantumCircuit qc(3);
    qc.h(0).s(0);
    qc.h(1).cx(1, 2).sx(2);
    apply(qc, g, theta, a, b);
    qc.h(0).cx(2, 0).s(1);
    return qc;
}

const std::vector<std::pair<int, int>> kPairs = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};

// Quarter turns, including negative multiples and whole extra turns, which
// reduce to the same four cases.
std::vector<double> quarter_turns() {
    std::vector<double> out;
    for (int k = -5; k <= 9; ++k) out.push_back(k * PI_2);
    out.push_back(-0.0);
    out.push_back(TWO_PI + PI_2);
    out.push_back(-TWO_PI - PI);
    return out;
}

// Angles at which none of the four rotations is Clifford.
const double kNonClifford[] = {PI_4, 3.0 * PI_4, -PI_4, PI_2 + 1e-6, PI - 1e-6, 1.0, 0.1};

// An observer that asks nothing, attached only so the run takes the row-major
// gate pass.
class Silent final : public Observer {
public:
    void observe(const ObservationContext&) override {}
};

RunPlan row_major_plan() {
    RunPlan plan;
    plan.observations.observe(Anchor::at_end(), std::make_shared<Silent>());
    return plan;
}

::testing::AssertionResult row_major_matches_statevector(const QuantumCircuit& qc) {
    CliffordSimulator cs;
    auto cr = cs.run(qc, /*shots=*/0, /*seed=*/1, row_major_plan());
    StatevectorSimulator ss;
    auto sr = ss.run(qc, /*shots=*/0, /*seed=*/1);
    for (const std::string& p : all_pauli_strings(qc.n_qubits)) {
        const int got = cr.final_state.expectation_pauli(p);
        const int want = sv_expectation_pauli_sign(sr.final_state, p);
        if (got != want)
            return ::testing::AssertionFailure()
                   << "Pauli '" << p << "': tableau " << got << ", statevector " << want;
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

// =============================================================================
// Every rotation, every quarter turn, both layouts
// =============================================================================

TEST(V11291CliffordIsing, EveryRotationMatchesTheStatevectorOnTheBitSlicedPass) {
    for (Ising g : kGates) {
        for (double theta : quarter_turns()) {
            for (const auto& [a, b] : kPairs) {
                SCOPED_TRACE(std::string(name(g)) + "(" + std::to_string(theta) + ", " +
                             std::to_string(a) + ", " + std::to_string(b) + ")");
                EXPECT_TRUE(clifford_matches_statevector_exhaustive(sandwich(g, theta, a, b)));
            }
        }
    }
}

TEST(V11291CliffordIsing, EveryRotationMatchesTheStatevectorOnTheRowMajorPass) {
    for (Ising g : kGates) {
        for (double theta : quarter_turns()) {
            for (const auto& [a, b] : kPairs) {
                SCOPED_TRACE(std::string(name(g)) + "(" + std::to_string(theta) + ", " +
                             std::to_string(a) + ", " + std::to_string(b) + ")");
                EXPECT_TRUE(row_major_matches_statevector(sandwich(g, theta, a, b)));
            }
        }
    }
}

TEST(V11291CliffordIsing, RzxIsNotSymmetricInItsOperands) {
    // Z on the first operand and X on the second. On qubit 0 in |0> and qubit
    // 1 in |+>, rzx(0, 1) finds each operand in an eigenstate of its Pauli and
    // changes the state only by a phase, while rzx(1, 0) entangles the two, so
    // the final states differ; the tableau must agree with the statevector on
    // both.
    for (bool swapped : {false, true}) {
        QuantumCircuit qc(2);
        qc.h(1);
        if (swapped) qc.rzx(PI_2, 1, 0);
        else         qc.rzx(PI_2, 0, 1);
        EXPECT_TRUE(clifford_matches_statevector_exhaustive(qc));
        EXPECT_TRUE(row_major_matches_statevector(qc));
    }
    QuantumCircuit ab(2), ba(2);
    ab.h(1).rzx(PI_2, 0, 1);
    ba.h(1).rzx(PI_2, 1, 0);
    CliffordSimulator cs;
    const auto rab = cs.run(ab, 0, 1), rba = cs.run(ba, 0, 1);
    bool differ = false;
    for (const std::string& p : all_pauli_strings(2))
        if (rab.final_state.expectation_pauli(p) != rba.final_state.expectation_pauli(p))
            differ = true;
    EXPECT_TRUE(differ) << "rzx(a, b) and rzx(b, a) gave the same state";
}

TEST(V11291CliffordIsing, AHalfTurnIsThePauliProduct) {
    // At theta = pi each rotation is P (x) Q up to a global phase, so the
    // tableau must hold exactly the state the two Paulis produce.
    struct Case { Ising g; const char* first; const char* second; };
    const Case cases[] = {{Ising::XX, "x", "x"}, {Ising::YY, "y", "y"},
                          {Ising::ZZ, "z", "z"}, {Ising::ZX, "z", "x"}};
    auto pauli = [](QuantumCircuit& qc, const char* p, int q) {
        if (p[0] == 'x') qc.x(q);
        else if (p[0] == 'y') qc.y(q);
        else qc.z(q);
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(name(c.g));
        QuantumCircuit rot(3), prod(3);
        for (QuantumCircuit* qc : {&rot, &prod}) qc->h(0).s(0).h(1).cx(1, 2).sx(2);
        apply(rot, c.g, PI, 2, 0);
        pauli(prod, c.first, 2);
        pauli(prod, c.second, 0);
        CliffordSimulator cs;
        const auto a = cs.run(rot, 0, 1), b = cs.run(prod, 0, 1);
        EXPECT_TRUE(v11251::states_equal_exhaustive(a.final_state, b.final_state));
    }
}

// =============================================================================
// Acceptance and execution agree
// =============================================================================

TEST(V11291CliffordIsing, IsCliffordAcceptsEveryQuarterTurn) {
    for (Ising g : kGates) {
        for (double theta : quarter_turns()) {
            SCOPED_TRACE(std::string(name(g)) + " " + std::to_string(theta));
            QuantumCircuit qc(2);
            apply(qc, g, theta, 0, 1);
            EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
            CliffordSimulator cs;
            EXPECT_NO_THROW((void)cs.run(qc, 16, 1));
        }
    }
}

TEST(V11291CliffordIsing, IsCliffordRejectsEveryOtherAngleAndRunRefusesIt) {
    for (Ising g : kGates) {
        for (double theta : kNonClifford) {
            SCOPED_TRACE(std::string(name(g)) + " " + std::to_string(theta));
            QuantumCircuit qc(2);
            qc.h(0);
            apply(qc, g, theta, 0, 1);
            EXPECT_FALSE(CliffordSimulator::is_clifford(qc));
            // A direct run bypasses is_clifford, and must not round the angle
            // to the nearest quarter turn.
            CliffordSimulator cs;
            EXPECT_ANY_THROW((void)cs.run(qc, 16, 1));
        }
    }
}

TEST(V11291CliffordIsing, OneNonCliffordRotationRejectsTheWholeCircuit) {
    QuantumCircuit qc(3);
    qc.rzz(PI_2, 0, 1).rxx(PI, 1, 2).ryy(PI_4, 0, 2).rzx(-PI_2, 2, 1);
    EXPECT_FALSE(CliffordSimulator::is_clifford(qc));
}

// =============================================================================
// Sampling and automatic selection
// =============================================================================

TEST(V11291CliffordIsing, BothSamplingRoutesStayOnTheStatesSupport) {
    QuantumCircuit prep(4);
    prep.h(0).h(2).rzz(PI_2, 0, 1).rxx(-PI_2, 1, 2).ryy(PI, 2, 3).rzx(3 * PI_2, 3, 0);
    const auto support = exact_statevector_distribution(prep);

    QuantumCircuit measured = prep;
    measured.measure_all();
    for (auto mode : {CliffordSimulator::Options::Sampling::Slab,
                      CliffordSimulator::Options::Sampling::PerShot}) {
        CliffordSimulator cs;
        cs.options.sampling = mode;
        const auto r = cs.run(measured, 2000, 7);
        int total = 0;
        for (const auto& [bits, count] : r.counts) {
            EXPECT_TRUE(support.count(std::stoull(bits, nullptr, 2)))
                << "outcome " << bits << " has no amplitude in the exact state";
            total += count;
        }
        EXPECT_EQ(total, 2000);
    }
}

TEST(V11291CliffordIsing, AutomaticSelectionRunsThemExactlyBeyondTheStatevectorRange) {
    // 24 qubits, past the size at which automatic selection stops choosing the
    // statevector. The circuit builds entanglement across every pair with the
    // four rotations and then undoes it in reverse, so the exact answer is
    // |0...0> on every shot. The tableau reaches it exactly.
    constexpr int n = 24;
    QuantumCircuit qc(n);
    for (int q = 0; q < n; ++q) qc.h(q);
    std::vector<std::function<void(QuantumCircuit&, bool)>> steps;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const Ising g = kGates[(i + j) % 4];
            steps.push_back([g, i, j](QuantumCircuit& c, bool inverse) {
                apply(c, g, inverse ? -PI_2 : PI_2, i, j);
            });
        }
    }
    for (const auto& s : steps) s(qc, false);
    for (auto it = steps.rbegin(); it != steps.rend(); ++it) (*it)(qc, true);
    for (int q = 0; q < n; ++q) qc.h(q);
    qc.measure_all();

    ASSERT_TRUE(CliffordSimulator::is_clifford(qc));
    backends::LocalBackend backend;
    const auto r = backend.run(qc, 200, 3);
    ASSERT_TRUE(r.success) << r.error_message;
    ASSERT_EQ(r.counts.size(), 1u);
    EXPECT_EQ(r.counts.begin()->first, std::string(n, '0'));
    EXPECT_EQ(r.counts.begin()->second, 200);
}
