// R.1.18.2 regression suite — the HighLevelDecompose coupling-map floor
// (issue #67). Under a constrained coupling map the stage-0 pass must emit
// no gate wider than two qubits: SABRE executes a 3-qubit gate only when all
// three wire pairs are adjacent, which triangle-free maps (path, grid,
// heavy-hex) can never provide, so before this fix any CCX-bearing lowered
// stream made transpile() throw on exactly the constrained targets the pass
// exists to serve. Unconstrained targets must KEEP CCX (a ccx-bearing basis
// stays native). The ladder itself is checked as an exact matrix identity on
// non-symmetric states before the routed end-to-end cases.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/transpiler.hpp"

#include "../src/transpiler/high_level_decompose.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

constexpr double kTol = 1e-12;

// Apply an instruction stream to |0...0> and return the statevector.
Statevector run_insts(int nq, const std::vector<Instruction>& insts) {
    Statevector sv(nq);
    sv.initialize_basis(0);
    StatevectorSimulator sim;
    for (const Instruction& inst : insts) { sim.apply_instruction(sv, inst); }
    return sv;
}

// Non-symmetric product preparation: distinct magnitude AND phase per qubit,
// with phase weights chosen so every basis state's accumulated phase is
// distinct (no 0.3+0.6 == 0.9 style collision that could mask a control-mask
// or operand-order bug behind equal amplitudes).
void prep_distinct(QuantumCircuit& qc) {
    for (int q = 0; q < qc.n_qubits; ++q) {
        qc.ry(0.4 * (q + 1), q);
        qc.p(0.3 + 0.4 * q * q, q);
    }
}

void expect_states_close(int nq, const Statevector& a, const Statevector& b) {
    const size_t dim = size_t(1) << nq;
    for (size_t i = 0; i < dim; ++i) {
        EXPECT_NEAR(a.real_parts[i], b.real_parts[i], kTol) << "amp " << i;
        EXPECT_NEAR(a.imag_parts[i], b.imag_parts[i], kTol) << "amp " << i;
    }
}

} // namespace

// =============================================================================
// The ladder is an exact CCX (matrix identity, not just up to phase)
// =============================================================================

TEST(R1182CcxFloor, LadderMatchesCcxOnDistinctStates) {
    // Cycle the control/control/target roles so an operand-order mistake in
    // the ladder cannot hide behind the symmetric (0,1,2) assignment.
    const int orders[3][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}};
    for (const auto& o : orders) {
        SCOPED_TRACE("ccx(" + std::to_string(o[0]) + "," + std::to_string(o[1]) +
                     "," + std::to_string(o[2]) + ")");
        QuantumCircuit ref(3), lad(3);
        prep_distinct(ref);
        prep_distinct(lad);
        ref.ccx(o[0], o[1], o[2]);

        std::vector<Instruction> lad_insts = lad.instructions;
        std::vector<Instruction> ladder = hld::lower_ccx(o[0], o[1], o[2]);
        lad_insts.insert(lad_insts.end(), ladder.begin(), ladder.end());

        // Alphabet: the floor's whole point is 2-qubit routability.
        for (const Instruction& inst : ladder) {
            EXPECT_LE(inst.qubits.size(), 2u);
            EXPECT_TRUE(inst.type == GT::H || inst.type == GT::P ||
                        inst.type == GT::CX)
                << "unexpected gate '" << inst.gate_name() << "' in the ladder";
        }

        expect_states_close(3, run_insts(3, ref.instructions),
                            run_insts(3, lad_insts));
    }
}

TEST(R1182CcxFloor, LadderRejectsDuplicateOperands) {
    EXPECT_THROW(hld::lower_ccx(1, 1, 2), std::invalid_argument);
    EXPECT_THROW(hld::lower_ccx(0, 2, 2), std::invalid_argument);
}

// =============================================================================
// Pass-level floor: constrained context => no 3-qubit gate in the output
// =============================================================================

TEST(R1182CcxFloor, PassOutputHasNoThreeQubitGateWhenConstrained) {
    QuantumCircuit qc(4);
    qc.ccx(0, 1, 2);          // user-written CCX is floored too
    qc.mcx({0, 1, 2}, 3);
    qc.mcp(0.7, {0, 1, 2});

    TranspilationContext ctx;
    ctx.coupling_map = CouplingMap::linear(4);
    HighLevelDecompose pass;
    const auto out = pass.run(DAGCircuit::from_circuit(qc), ctx).to_circuit();

    ASSERT_FALSE(out.instructions.empty());
    for (const Instruction& inst : out.instructions) {
        EXPECT_LE(inst.qubits.size(), 2u)
            << "3-qubit gate '" << inst.gate_name() << "' survived the floor";
    }
}

TEST(R1182CcxFloor, PassKeepsCcxWhenUnconstrained) {
    // Basis-only composition (no coupling map): CCX is the correct bottom.
    // A ccx from the user and one from the k = 2 MCX lowering both survive.
    QuantumCircuit qc(3);
    qc.ccx(0, 1, 2);
    qc.mcx({0, 2}, 1);

    TranspilationContext ctx;
    ctx.basis_gates = {"ccx"};
    HighLevelDecompose pass;
    const auto out = pass.run(DAGCircuit::from_circuit(qc), ctx).to_circuit();

    int ccx = 0;
    for (const Instruction& inst : out.instructions) {
        EXPECT_EQ(inst.type, GT::CCX);
        ++ccx;
    }
    EXPECT_EQ(ccx, 2);
}

// =============================================================================
// Routed end-to-end (the contract the three R1181 pins aim at)
// =============================================================================

TEST(R1182CcxFloor, RoutedEndToEndMcxOnGridMap) {
    // Same forced-flip construction as the linear-map pin, on the 2x2 grid —
    // triangle-free like every realistic topology, but 2D: |1110> -> |1111>.
    QuantumCircuit qc(4, 4);
    qc.x(0).x(1).x(2);
    qc.mcx({0, 1, 2}, 3);
    qc.measure_all();

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap::grid(2, 2), {"cx", "u3"}, level);
        for (const auto& inst : out.instructions) {
            EXPECT_NE(inst.type, GT::MCX);
            EXPECT_NE(inst.type, GT::CCX);
        }
        StatevectorSimulator sim;
        auto res = sim.run(out, 128, 17);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            (void)count;
            EXPECT_EQ(bits, "1111") << "level " << level;
        }
    }
}

TEST(R1182CcxFloor, UserWrittenCcxRoutesOnLinearMap) {
    // Pre-existing, separable half of issue #67: a plain user CCX was already
    // unroutable on triangle-free maps before the pass existed. The floor
    // fixes it as a consequence of the output invariant.
    QuantumCircuit qc(3, 3);
    qc.x(0).x(1);
    qc.ccx(0, 1, 2);
    qc.measure_all();

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap::linear(3), {"cx", "u3"}, level);
        for (const auto& inst : out.instructions) {
            EXPECT_NE(inst.type, GT::CCX);
        }
        StatevectorSimulator sim;
        auto res = sim.run(out, 64, 19);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            (void)count;
            EXPECT_EQ(bits, "111") << "level " << level;
        }
    }
}

TEST(R1182CcxFloor, ConditionedCcxRoutedPreservesFeedforward) {
    // The ladder replacement must carry the classical condition onto every
    // emitted gate: x(0) forces clbit 0 to 1, the conditioned CCX then fires
    // deterministically and the final key is 111.
    QuantumCircuit qc(3, 3);
    qc.x(0);
    qc.measure(0, 0);
    qc.x(1);
    qc.add_if(0, 1, GT::CCX, {0, 1, 2});
    qc.measure_all();

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap::linear(3), {"cx", "u3"}, level);
        StatevectorSimulator sim;
        auto res = sim.run(out, 64, 23);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            (void)count;
            EXPECT_EQ(bits, "111") << "level " << level;
        }
    }
}

TEST(R1182CcxFloor, CcxKeptNativeOnBasisOnlyTarget) {
    // Per-consumer floor, preservation half: with no coupling map and a basis
    // that contains ccx, the whole preset pipeline keeps the gate native.
    QuantumCircuit qc(3);
    qc.mcx({0, 1}, 2);  // lowers to exactly one CCX in stage 0

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap(), {"ccx"}, level);
        ASSERT_EQ(out.instructions.size(), 1u);
        EXPECT_EQ(out.instructions[0].type, GT::CCX);
    }
}
