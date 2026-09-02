// 1.1.25.1 test wave - the gates the Clifford backend learned in 1.1.25.0.
//
// Two additions are covered here. SX, SXDG, CY, ISWAP and ECR gained tableau
// implementations, each one sweep over the 2N rows rather than a chain of calls
// to the primitives, so each carries its own composed bit and sign rule that
// nothing else in the tree evaluates. And is_clifford() began accepting RX, RY,
// RZ and P at multiples of pi/2, which widens the set of circuits the AUTO
// dispatch hands to this backend.
//
// Every gate claim is checked against the statevector simulator through
// v11251::clifford_matches_statevector_exhaustive, which compares all 4^n Pauli
// expectations. Since rho = 2^-n * sum_P <P> P, agreement on every P is
// equality of the two states up to the global phase the stabilizer formalism
// does not represent, so these are complete comparisons rather than spot
// checks, and they carry no sampling tolerance.
//
// The invariant with the most reach is AcceptanceAndDispatchAgreeOnAngleGrid:
// is_clifford() and run() classify through one function precisely so the set of
// angles the backend accepts cannot drift from the set it can execute, and a
// grid walked through both is what would catch it if they ever did.

#include <gtest/gtest.h>

#include "v11251_clifford_oracle.hpp"

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/clifford_sim.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// The four quarter turns, plus the two ways an angle can land on one without
// being written as one: a full turn folds to zero, and so does a small negative
// that fmod leaves just below 2*pi.
const double kQuarterTurns[] = {0.0, PI_2, PI, 3.0 * PI_2};
const double kFoldsToZero[] = {TWO_PI, -1e-12, 2.0 * TWO_PI};

// Angles that are not multiples of pi/2. Each is derived rather than
// transcribed, apart from 0.7, which is an arbitrary input and not an expected
// value.
const double kNonQuarterTurns[] = {PI_4, PI / 3.0, 0.7, PI_4 / 2.0,
                                   PI_2 + PI_4, -PI_4};

enum class Rot { RX, RY, RZ, P };

const Rot kRotations[] = {Rot::RX, Rot::RY, Rot::RZ, Rot::P};

const char* rot_name(Rot r) {
    switch (r) {
        case Rot::RX: return "rx";
        case Rot::RY: return "ry";
        case Rot::RZ: return "rz";
        case Rot::P:  return "p";
    }
    return "?";
}

Instruction::GateType rot_type(Rot r) {
    switch (r) {
        case Rot::RX: return Instruction::GateType::RX;
        case Rot::RY: return Instruction::GateType::RY;
        case Rot::RZ: return Instruction::GateType::RZ;
        case Rot::P:  return Instruction::GateType::P;
    }
    return Instruction::GateType::RX;
}

void add_rotation(QuantumCircuit& qc, Rot r, double angle, int q) {
    switch (r) {
        case Rot::RX: qc.rx(angle, q); break;
        case Rot::RY: qc.ry(angle, q); break;
        case Rot::RZ: qc.rz(angle, q); break;
        case Rot::P:  qc.p(angle, q); break;
    }
}

// A rotation instruction carrying no angle at all. The circuit builders always
// populate params, so reaching this state means building the instruction
// directly, which the QASM front ends and any caller assembling a circuit by
// hand can also do.
QuantumCircuit rotation_without_params(Rot r, int n_qubits, int q) {
    QuantumCircuit qc(n_qubits);
    Instruction inst;
    inst.type = rot_type(r);
    inst.qubits = {q};
    qc.instructions.push_back(inst);
    return qc;
}

// A state with X, Y and Z structure on every qubit, so a gate applied after it
// has something to act on. Acting on |0...0> exercises almost none of a
// tableau: most rows are still in their initial one-bit form.
void prepare_nontrivial(QuantumCircuit& qc) {
    const int n = qc.n_qubits;
    for (int q = 0; q < n; ++q) {
        if (q % 3 == 0) qc.h(q);
        if (q % 3 == 1) { qc.h(q); qc.s(q); }
        if (q % 3 == 2) qc.x(q);
    }
    for (int q = 0; q + 1 < n; ++q) qc.cx(q, q + 1);
    if (n >= 2) qc.s(n - 1);
}

enum class New { SX, SXDG, CY, ISWAP, ECR };

const New kNewGates[] = {New::SX, New::SXDG, New::CY, New::ISWAP, New::ECR};

const char* new_name(New g) {
    switch (g) {
        case New::SX: return "sx";
        case New::SXDG: return "sxdg";
        case New::CY: return "cy";
        case New::ISWAP: return "iswap";
        case New::ECR: return "ecr";
    }
    return "?";
}

bool new_is_two_qubit(New g) {
    return g == New::CY || g == New::ISWAP || g == New::ECR;
}

void add_new_gate(QuantumCircuit& qc, New g, int a, int b) {
    switch (g) {
        case New::SX:    qc.sx(a); break;
        case New::SXDG:  qc.sxdg(a); break;
        case New::CY:    qc.cy(a, b); break;
        case New::ISWAP: qc.iswap(a, b); break;
        case New::ECR:   qc.ecr(a, b); break;
    }
}

}  // namespace

// =============================================================================
// The five new gates against the statevector
// =============================================================================

TEST(V11251CliffordGates, EachNewGateAloneMatchesStatevector) {
    for (New g : kNewGates) {
        const int n = new_is_two_qubit(g) ? 2 : 1;
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                if (new_is_two_qubit(g) && a == b) continue;
                if (!new_is_two_qubit(g) && b != 0) continue;
                SCOPED_TRACE(std::string(new_name(g)) + "(" + std::to_string(a) +
                             "," + std::to_string(b) + ")");
                QuantumCircuit qc(n);
                add_new_gate(qc, g, a, b);
                EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(qc));
            }
        }
    }
}

TEST(V11251CliffordGates, EachNewGateOnPreparedStateMatchesStatevector) {
    constexpr int n = 3;
    for (New g : kNewGates) {
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                if (new_is_two_qubit(g) && a == b) continue;
                if (!new_is_two_qubit(g) && b != 0) continue;
                SCOPED_TRACE(std::string(new_name(g)) + "(" + std::to_string(a) +
                             "," + std::to_string(b) + ")");
                QuantumCircuit qc(n);
                prepare_nontrivial(qc);
                add_new_gate(qc, g, a, b);
                EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(qc));
            }
        }
    }
}

// Every new gate twice over, interleaved with the primitives and with each
// other. A gate applied once to a prepared state still leaves the tableau's
// phase bits mostly untouched; a chain of them does not.
TEST(V11251CliffordGates, AllNewGatesMixedIntoOneCircuitMatchStatevector) {
    constexpr int n = 4;
    QuantumCircuit qc(n);
    prepare_nontrivial(qc);
    qc.sx(0).cy(1, 2).iswap(0, 3).ecr(2, 1).sxdg(3);
    qc.h(2).cx(3, 0).cz(1, 3).swap(0, 2);
    qc.sx(1).ecr(0, 3).cy(3, 2).iswap(1, 2).sxdg(0);
    qc.s(3).sdg(1).y(2).z(0);
    EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(qc));
}

// Applying a gate and its inverse must return the state exactly, which pins the
// sign rules rather than only the bit rules: a phase error that the forward
// gate introduces does not cancel against a wrong inverse.
TEST(V11251CliffordGates, SxAndSxdgAreInverses) {
    constexpr int n = 3;
    for (int q = 0; q < n; ++q) {
        SCOPED_TRACE("q=" + std::to_string(q));
        QuantumCircuit prepared(n);
        prepare_nontrivial(prepared);

        QuantumCircuit round_trip(n);
        prepare_nontrivial(round_trip);
        round_trip.sx(q).sxdg(q);

        CliffordSimulator sim;
        EXPECT_TRUE(v11251::states_equal_exhaustive(
            sim.run(prepared, 0, 1).final_state,
            sim.run(round_trip, 0, 1).final_state));

        QuantumCircuit other_order(n);
        prepare_nontrivial(other_order);
        other_order.sxdg(q).sx(q);
        EXPECT_TRUE(v11251::states_equal_exhaustive(
            sim.run(prepared, 0, 1).final_state,
            sim.run(other_order, 0, 1).final_state));
    }
}

// =============================================================================
// Operand order
// =============================================================================

// ECR is not symmetric. The library's operand order is a deliberate deviation
// from Qiskit's, documented in docs/api/gates.md, and both directions are
// pinned against the statevector so the tableau cannot silently adopt the other
// convention.
TEST(V11251CliffordGates, EcrIsNotSymmetricInItsOperands) {
    constexpr int n = 2;
    QuantumCircuit forward(n);
    prepare_nontrivial(forward);
    forward.ecr(0, 1);

    QuantumCircuit reversed(n);
    prepare_nontrivial(reversed);
    reversed.ecr(1, 0);

    EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(forward));
    EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(reversed));

    CliffordSimulator sim;
    EXPECT_FALSE(v11251::states_equal_exhaustive(
        sim.run(forward, 0, 1).final_state, sim.run(reversed, 0, 1).final_state))
        << "ecr(0,1) and ecr(1,0) must not produce the same state";
}

// CY distinguishes control from target: on |+0> the control carries the
// superposition and the gate entangles, on |0+> it does not.
TEST(V11251CliffordGates, CyDistinguishesControlFromTarget) {
    constexpr int n = 2;
    QuantumCircuit forward(n);
    prepare_nontrivial(forward);
    forward.cy(0, 1);

    QuantumCircuit reversed(n);
    prepare_nontrivial(reversed);
    reversed.cy(1, 0);

    EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(forward));
    EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(reversed));

    CliffordSimulator sim;
    EXPECT_FALSE(v11251::states_equal_exhaustive(
        sim.run(forward, 0, 1).final_state, sim.run(reversed, 0, 1).final_state))
        << "cy(0,1) and cy(1,0) must not produce the same state";
}

// ISWAP is symmetric, which is asserted here rather than assumed: the composed
// sweep applies cx(a,b), s(b), cx(b,a), cx(a,b), and nothing about that
// sequence is symmetric on its face.
TEST(V11251CliffordGates, IswapIsSymmetricInItsOperands) {
    constexpr int n = 3;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            SCOPED_TRACE("a=" + std::to_string(a) + " b=" + std::to_string(b));
            QuantumCircuit forward(n);
            prepare_nontrivial(forward);
            forward.iswap(a, b);

            QuantumCircuit reversed(n);
            prepare_nontrivial(reversed);
            reversed.iswap(b, a);

            CliffordSimulator sim;
            EXPECT_TRUE(v11251::states_equal_exhaustive(
                sim.run(forward, 0, 1).final_state,
                sim.run(reversed, 0, 1).final_state));
        }
    }
}

// CZ and SWAP changed implementation in the same rework, from calls to the
// primitives into a single composed sweep, so they are re-pinned alongside the
// five that are new.
TEST(V11251CliffordGates, CzIsSymmetricAndMatchesStatevector) {
    constexpr int n = 3;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            SCOPED_TRACE("a=" + std::to_string(a) + " b=" + std::to_string(b));
            QuantumCircuit forward(n);
            prepare_nontrivial(forward);
            forward.cz(a, b);
            EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(forward));

            QuantumCircuit reversed(n);
            prepare_nontrivial(reversed);
            reversed.cz(b, a);

            CliffordSimulator sim;
            EXPECT_TRUE(v11251::states_equal_exhaustive(
                sim.run(forward, 0, 1).final_state,
                sim.run(reversed, 0, 1).final_state));
        }
    }
}

TEST(V11251CliffordGates, SwapIsSymmetricAndMatchesStatevector) {
    constexpr int n = 3;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            SCOPED_TRACE("a=" + std::to_string(a) + " b=" + std::to_string(b));
            QuantumCircuit forward(n);
            prepare_nontrivial(forward);
            forward.swap(a, b);
            EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(forward));

            QuantumCircuit reversed(n);
            prepare_nontrivial(reversed);
            reversed.swap(b, a);

            CliffordSimulator sim;
            EXPECT_TRUE(v11251::states_equal_exhaustive(
                sim.run(forward, 0, 1).final_state,
                sim.run(reversed, 0, 1).final_state));
        }
    }
}

// =============================================================================
// Classification and dispatch of the five new gates
// =============================================================================

TEST(V11251CliffordGates, IsCliffordAcceptsEachNewGate) {
    for (New g : kNewGates) {
        SCOPED_TRACE(new_name(g));
        QuantumCircuit qc(2, 2);
        add_new_gate(qc, g, 0, 1);
        qc.measure_all();
        EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
    }
}

TEST(V11251CliffordGates, DirectRunAcceptsEachNewGate) {
    for (New g : kNewGates) {
        SCOPED_TRACE(new_name(g));
        QuantumCircuit qc(2, 2);
        add_new_gate(qc, g, 0, 1);
        qc.measure_all();
        CliffordSimulator sim;
        EXPECT_NO_THROW(sim.run(qc, /*shots=*/16, /*seed=*/1));
    }
}

// is_clifford() is the exact predicate the AUTO dispatch consults for an ideal
// noise model, so a circuit it accepts is a circuit that reaches the tableau
// instead of falling to a slower simulator with no diagnostic. Both halves are
// asserted: the predicate, and that the route taken returns the right answer.
TEST(V11251CliffordGates, AutoBackendRoutesCircuitsWithTheNewGatesToTheTableau) {
    for (New g : kNewGates) {
        SCOPED_TRACE(new_name(g));
        QuantumCircuit qc(2, 2);
        qc.h(0);
        add_new_gate(qc, g, 0, 1);
        qc.measure_all();

        ASSERT_TRUE(CliffordSimulator::is_clifford(qc));

        backends::LocalBackend backend;
        ASSERT_EQ(backend.config.simulator, backends::LocalBackend::SimType::AUTO);
        const auto res = backend.run(qc, /*shots=*/4096, /*seed=*/11);
        ASSERT_TRUE(res.success) << res.error_message;

        // The support must be exactly the support of the true distribution, and
        // that is read off the tableau's slab rather than sampled.
        QuantumCircuit unmeasured(2);
        unmeasured.h(0);
        add_new_gate(unmeasured, g, 0, 1);
        CliffordSimulator sim;
        const v11251::Dist exact =
            v11251::exact_clifford_distribution(sim.run(unmeasured, 0, 1).final_state);

        int total = 0;
        for (const auto& [key, count] : res.counts) {
            total += count;
            const uint64_t outcome = std::stoull(key, nullptr, 2);
            EXPECT_GT(exact.count(outcome), 0u)
                << "counts key '" << key << "' is outside the exact support";
        }
        EXPECT_EQ(total, 4096);
    }
}

// =============================================================================
// Rotation gates at Clifford angles
// =============================================================================

TEST(V11251CliffordGates, IsCliffordAcceptsQuarterTurns) {
    for (Rot r : kRotations) {
        for (double a : kQuarterTurns) {
            SCOPED_TRACE(std::string(rot_name(r)) + "(" + std::to_string(a) + ")");
            QuantumCircuit qc(1, 1);
            add_rotation(qc, r, a, 0);
            qc.measure(0, 0);
            EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
        }
    }
}

TEST(V11251CliffordGates, IsCliffordAcceptsAnglesThatFoldToZero) {
    for (Rot r : kRotations) {
        for (double a : kFoldsToZero) {
            SCOPED_TRACE(std::string(rot_name(r)) + "(" + std::to_string(a) + ")");
            QuantumCircuit qc(1, 1);
            add_rotation(qc, r, a, 0);
            EXPECT_TRUE(CliffordSimulator::is_clifford(qc));
        }
    }
}

TEST(V11251CliffordGates, IsCliffordRejectsNonQuarterTurns) {
    for (Rot r : kRotations) {
        for (double a : kNonQuarterTurns) {
            SCOPED_TRACE(std::string(rot_name(r)) + "(" + std::to_string(a) + ")");
            QuantumCircuit qc(1, 1);
            add_rotation(qc, r, a, 0);
            EXPECT_FALSE(CliffordSimulator::is_clifford(qc));
        }
    }
}

TEST(V11251CliffordGates, IsCliffordRejectsRotationWithNoAngle) {
    for (Rot r : kRotations) {
        SCOPED_TRACE(rot_name(r));
        EXPECT_FALSE(CliffordSimulator::is_clifford(rotation_without_params(r, 1, 0)));
    }
}

// An angle beyond one turn, and a negative angle, must classify and execute as
// the quarter turn they reduce to rather than being rejected or rounded.
TEST(V11251CliffordGates, AnglesBeyondOneTurnBehaveAsTheirReduction) {
    struct Pair { double raw; double reduced; const char* label; };
    const Pair pairs[] = {
        {5.0 * PI_2, PI_2, "5pi/2 == pi/2"},
        {-PI_2, 3.0 * PI_2, "-pi/2 == 3pi/2"},
        {-PI, PI, "-pi == pi"},
        {-3.0 * PI_2, PI_2, "-3pi/2 == pi/2"},
        {TWO_PI + PI, PI, "2pi+pi == pi"},
        {4.0 * TWO_PI + PI_2, PI_2, "4 turns + pi/2 == pi/2"},
    };
    constexpr int n = 2;
    for (Rot r : kRotations) {
        for (const Pair& p : pairs) {
            SCOPED_TRACE(std::string(rot_name(r)) + " " + p.label);
            QuantumCircuit raw(n);
            prepare_nontrivial(raw);
            add_rotation(raw, r, p.raw, 0);
            EXPECT_TRUE(CliffordSimulator::is_clifford(raw));

            QuantumCircuit reduced(n);
            prepare_nontrivial(reduced);
            add_rotation(reduced, r, p.reduced, 0);

            CliffordSimulator sim;
            EXPECT_TRUE(v11251::states_equal_exhaustive(
                sim.run(raw, 0, 1).final_state, sim.run(reduced, 0, 1).final_state));
        }
    }
}

// Every rotation at every quarter turn, against the statevector. This is what
// pins the dispatch's choice of primitive for each arm: rz(pi/2) reaching
// apply_sdg instead of apply_s would still be Clifford and still execute.
TEST(V11251CliffordGates, EveryRotationAtEveryQuarterTurnMatchesStatevector) {
    constexpr int n = 2;
    for (Rot r : kRotations) {
        for (double a : kQuarterTurns) {
            for (int q = 0; q < n; ++q) {
                SCOPED_TRACE(std::string(rot_name(r)) + "(" + std::to_string(a) +
                             ") on q" + std::to_string(q));
                QuantumCircuit qc(n);
                prepare_nontrivial(qc);
                add_rotation(qc, r, a, q);
                EXPECT_TRUE(v11251::clifford_matches_statevector_exhaustive(qc));
            }
        }
    }
}

// The named equivalences, asserted as tableau equalities rather than through
// the statevector, so they hold as statements about this backend's own
// dispatch. RY has no named quarter-turn gate, so it is covered by the
// statevector comparison above and by its own reduction here.
TEST(V11251CliffordGates, RotationsEqualTheirNamedGates) {
    constexpr int n = 3;
    struct Case {
        const char* label;
        void (*rotation)(QuantumCircuit&, int);
        void (*named)(QuantumCircuit&, int);
    };
    const Case cases[] = {
        {"rx(pi/2) == sx",
         [](QuantumCircuit& c, int q) { c.rx(PI_2, q); },
         [](QuantumCircuit& c, int q) { c.sx(q); }},
        {"rx(3pi/2) == sxdg",
         [](QuantumCircuit& c, int q) { c.rx(3.0 * PI_2, q); },
         [](QuantumCircuit& c, int q) { c.sxdg(q); }},
        {"rx(pi) == x",
         [](QuantumCircuit& c, int q) { c.rx(PI, q); },
         [](QuantumCircuit& c, int q) { c.x(q); }},
        {"rz(pi/2) == s",
         [](QuantumCircuit& c, int q) { c.rz(PI_2, q); },
         [](QuantumCircuit& c, int q) { c.s(q); }},
        {"rz(3pi/2) == sdg",
         [](QuantumCircuit& c, int q) { c.rz(3.0 * PI_2, q); },
         [](QuantumCircuit& c, int q) { c.sdg(q); }},
        {"rz(pi) == z",
         [](QuantumCircuit& c, int q) { c.rz(PI, q); },
         [](QuantumCircuit& c, int q) { c.z(q); }},
        {"p(pi/2) == s",
         [](QuantumCircuit& c, int q) { c.p(PI_2, q); },
         [](QuantumCircuit& c, int q) { c.s(q); }},
        {"p(3pi/2) == sdg",
         [](QuantumCircuit& c, int q) { c.p(3.0 * PI_2, q); },
         [](QuantumCircuit& c, int q) { c.sdg(q); }},
        {"p(pi) == z",
         [](QuantumCircuit& c, int q) { c.p(PI, q); },
         [](QuantumCircuit& c, int q) { c.z(q); }},
        {"ry(pi) == y",
         [](QuantumCircuit& c, int q) { c.ry(PI, q); },
         [](QuantumCircuit& c, int q) { c.y(q); }},
    };

    CliffordSimulator sim;
    for (const Case& c : cases) {
        for (int q = 0; q < n; ++q) {
            SCOPED_TRACE(std::string(c.label) + " on q" + std::to_string(q));
            QuantumCircuit a(n);
            prepare_nontrivial(a);
            c.rotation(a, q);

            QuantumCircuit b(n);
            prepare_nontrivial(b);
            c.named(b, q);

            EXPECT_TRUE(v11251::states_equal_exhaustive(
                sim.run(a, 0, 1).final_state, sim.run(b, 0, 1).final_state));
        }
    }
}

// A zero angle must be the identity on every rotation, not merely Clifford.
TEST(V11251CliffordGates, ZeroAngleIsTheIdentity) {
    constexpr int n = 3;
    CliffordSimulator sim;
    for (Rot r : kRotations) {
        for (double a : {0.0, TWO_PI, -1e-12}) {
            SCOPED_TRACE(std::string(rot_name(r)) + "(" + std::to_string(a) + ")");
            QuantumCircuit plain(n);
            prepare_nontrivial(plain);

            QuantumCircuit rotated(n);
            prepare_nontrivial(rotated);
            add_rotation(rotated, r, a, 1);

            EXPECT_TRUE(v11251::states_equal_exhaustive(
                sim.run(plain, 0, 1).final_state, sim.run(rotated, 0, 1).final_state));
        }
    }
}

// =============================================================================
// Acceptance and dispatch cannot drift apart
// =============================================================================

// The classifier is shared, so every angle is in exactly one of two states:
// accepted by is_clifford() and executable by run(), or rejected by both. An
// angle accepted but not executable would be a circuit the AUTO dispatch hands
// to a backend that then throws; an angle rejected but executable would be a
// circuit needlessly pushed to a slower simulator.
TEST(V11251CliffordGates, AcceptanceAndDispatchAgreeOnAngleGrid) {
    constexpr int kSteps = 64;
    CliffordSimulator sim;
    for (Rot r : kRotations) {
        for (int i = -kSteps; i <= kSteps; ++i) {
            // A grid in sixteenths of a turn: every fourth point is a quarter
            // turn and the rest are not, over four turns in each direction.
            const double angle = static_cast<double>(i) * (TWO_PI / 16.0);
            SCOPED_TRACE(std::string(rot_name(r)) + "(" + std::to_string(angle) +
                         ") i=" + std::to_string(i));

            QuantumCircuit qc(1);
            add_rotation(qc, r, angle, 0);
            const bool accepted = CliffordSimulator::is_clifford(qc);

            // Sixteenths land on a quarter turn exactly when i is a multiple
            // of 4, which is the classification the grid is checking.
            EXPECT_EQ(accepted, (i % 4) == 0)
                << "angle " << angle << " is a quarter turn iff i is a multiple of 4";

            if (accepted) {
                EXPECT_NO_THROW(sim.run(qc, /*shots=*/1, /*seed=*/1))
                    << "is_clifford accepted an angle run() cannot execute";
            } else {
                EXPECT_THROW(sim.run(qc, /*shots=*/1, /*seed=*/1), std::runtime_error)
                    << "run() executed an angle is_clifford rejected";
            }
        }
    }
}

// =============================================================================
// A direct run refuses what it cannot execute
// =============================================================================

// The AUTO dispatch is gated by is_clifford(), so a non-Clifford angle reaches
// run() only through a direct call. It must be rejected rather than rounded to
// the nearest quarter turn, and the message must carry enough to locate the
// offending instruction.
TEST(V11251CliffordGates, NonCliffordAngleInDirectRunThrowsNamingGateAndAngle) {
    for (Rot r : kRotations) {
        for (double a : kNonQuarterTurns) {
            SCOPED_TRACE(std::string(rot_name(r)) + "(" + std::to_string(a) + ")");
            QuantumCircuit qc(1, 1);
            add_rotation(qc, r, a, 0);
            qc.measure(0, 0);
            ASSERT_FALSE(qc.instructions.empty());
            const std::string name = qc.instructions.front().gate_name();

            CliffordSimulator sim;
            try {
                sim.run(qc, /*shots=*/4, /*seed=*/1);
                ADD_FAILURE() << "run() accepted a non-Clifford angle";
            } catch (const std::runtime_error& e) {
                const std::string msg = e.what();
                EXPECT_NE(msg.find("CliffordSimulator"), std::string::npos) << msg;
                EXPECT_NE(msg.find(name), std::string::npos)
                    << "message does not name the gate: " << msg;
                EXPECT_NE(msg.find(std::to_string(a)), std::string::npos)
                    << "message does not name the angle: " << msg;
                EXPECT_NE(msg.find("not Clifford"), std::string::npos) << msg;
            }
        }
    }
}

TEST(V11251CliffordGates, RotationWithNoAngleInDirectRunThrows) {
    for (Rot r : kRotations) {
        SCOPED_TRACE(rot_name(r));
        QuantumCircuit qc = rotation_without_params(r, 1, 0);
        const std::string name = qc.instructions.front().gate_name();

        CliffordSimulator sim;
        try {
            sim.run(qc, /*shots=*/4, /*seed=*/1);
            ADD_FAILURE() << "run() accepted a rotation carrying no angle";
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("CliffordSimulator"), std::string::npos) << msg;
            EXPECT_NE(msg.find(name), std::string::npos)
                << "message does not name the gate: " << msg;
            EXPECT_NE(msg.find("no angle parameter"), std::string::npos) << msg;
        }
    }
}

// A gate outside the Clifford group entirely still reaches the dispatch default
// and must throw there, which is a different message and a different exception
// type from the angle rejection.
TEST(V11251CliffordGates, NonCliffordGateInDirectRunThrowsInvalidArgument) {
    QuantumCircuit qc(1, 1);
    qc.t(0);
    qc.measure(0, 0);
    CliffordSimulator sim;
    EXPECT_THROW(sim.run(qc, /*shots=*/4, /*seed=*/1), std::invalid_argument);
    EXPECT_FALSE(CliffordSimulator::is_clifford(qc));
}
