// R.1.12.1 total-coverage suite, Batch 3: individual transpiler passes.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 3: toolchain".
//
// Each optimisation pass is checked for semantic preservation (full-matrix
// equivalence) and, where it has a structural effect, for that effect. Routing
// is checked by the coupling-adjacency invariant and the edgeless-throw
// contract. transpile() preset levels 0-3 must preserve semantics when no
// routing is required. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/transpiler.hpp"

#include <memory>
#include <vector>

using namespace lindblad;

namespace {

QuantumCircuit run_pass(const TranspilationPass& pass, const QuantumCircuit& qc,
                        TranspilationContext ctx = {}) {
    auto dag = DAGCircuit::from_circuit(qc);
    return pass.run(dag, ctx).to_circuit();
}

// Compare two circuit unitaries up to an unobservable global phase: transpiler
// passes (ZYZ merges etc.) preserve the operator only up to e^{i alpha}.
void expect_equiv(const QuantumCircuit& a, const QuantumCircuit& b,
                  double tol = 1e-7) {
    ASSERT_EQ(a.n_qubits, b.n_qubits);
    auto ma = Operator::from_circuit(a).data;
    auto mb = Operator::from_circuit(b).data;
    ASSERT_EQ(ma.size(), mb.size());
    Complex128 phase(1, 0);
    for (size_t i = 0; i < ma.size(); ++i) {
        if (mb[i].norm_sq() > 1e-12 && ma[i].norm_sq() > 1e-12) {
            phase = ma[i] * Complex128(mb[i].real, -mb[i].imag) * (1.0 / mb[i].norm_sq());
            break;
        }
    }
    for (size_t i = 0; i < ma.size(); ++i) {
        Complex128 bp = mb[i] * phase;
        EXPECT_NEAR(ma[i].real, bp.real, tol) << "re @ " << i;
        EXPECT_NEAR(ma[i].imag, bp.imag, tol) << "im @ " << i;
    }
}

QuantumCircuit identity_circuit(int n) { return QuantumCircuit(n); }

}  // namespace

// =============================================================================
// Optimize1qGates
// =============================================================================

TEST(R1121Passes, Optimize1qCancelsHadamardPair) {
    QuantumCircuit qc(1);
    qc.h(0).h(0);
    auto out = run_pass(Optimize1qGates(), qc);
    expect_equiv(out, identity_circuit(1));
    EXPECT_LE(out.size(), 1) << "H H collapses to (near) identity";
}

// REGRESSION (shipped red in R.1.12.1, fixed in R.1.12.2): zyz_decompose's
// phase extraction was sign-flipped (arg(SU[0,0]) = -(phi+lam)/2, not
// +(phi+lam)/2), so Optimize1qGates merged a generic single-qubit run into a
// gate with the correct theta but wrong phi/lambda, and transpile() /
// preset_pass_manager() silently changed circuit unitaries. This test and
// PassManagerComposesAndPreservesSemantics /
// TranspilePresetLevelsPreserveSemantics (here) plus
// PresetPassManagerPreservesSemanticsUnconstrained (passes_more) pin the
// correct contract: passes preserve the unitary up to a global phase.
TEST(R1121Passes, Optimize1qPreservesGenericRun) {
    QuantumCircuit qc(1);
    qc.rx(0.7, 0).ry(-0.4, 0).rz(1.1, 0).h(0);
    expect_equiv(run_pass(Optimize1qGates(), qc), qc);
}

// =============================================================================
// CXCancellation / CommutativeCancellation / ConsolidateBlocks
// =============================================================================

TEST(R1121Passes, CxCancellationRemovesAdjacentPair) {
    QuantumCircuit qc(2);
    qc.cx(0, 1).cx(0, 1);
    auto out = run_pass(CXCancellation(), qc);
    expect_equiv(out, identity_circuit(2));
}

TEST(R1121Passes, CommutativeCancellationPreservesSemantics) {
    QuantumCircuit qc(2);
    qc.rz(0.3, 0).cx(0, 1).rz(0.4, 0).cx(0, 1);
    expect_equiv(run_pass(CommutativeCancellation(), qc), qc);
}

TEST(R1121Passes, ConsolidateBlocksPreservesSemantics) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.5, 1).h(0);
    expect_equiv(run_pass(ConsolidateBlocks(), qc), qc);
}

// ConsolidateBlocks over the FULL two-qubit gate set, asserting that the pass
// preserves the operator up to global phase for every entangler.
//
// WHAT THIS DOES NOT COVER, since the shape of the fixtures decides it: a block
// is a run of ADJACENT two-qubit gates on one pair, and a single-qubit gate on
// either wire ends that run. Every fixture below separates its two entanglers
// with local gates, so each forms a block of ONE and is returned by the
// block_count == 1 early-out without the decomposition being reached at all.
// The same holds for the raw-UNITARY case at the end, where an `rz` sits
// between the two `unitary` calls.
//
// So this pins the passthrough path. It cannot distinguish a working
// decomposition from a discarded one either way, because the pass keeps the
// ORIGINAL instructions whenever its verification net rejects a
// factorisation, and the original instructions are equivalent by definition.
// Structural coverage of the decomposition needs blocks of adjacent
// entanglers and an assertion about the OUTPUT SHAPE rather than the operator.
TEST(R1121Passes, ConsolidateBlocksFullTwoQubitSetPreservesSemantics) {
    struct Case { const char* name; void (*apply)(QuantumCircuit&); };
    const Case cases[] = {
        {"cx",    [](QuantumCircuit& c){ c.cx(0, 1); }},
        {"cy",    [](QuantumCircuit& c){ c.cy(0, 1); }},
        {"cz",    [](QuantumCircuit& c){ c.cz(0, 1); }},
        {"ch",    [](QuantumCircuit& c){ c.ch(0, 1); }},
        {"swap",  [](QuantumCircuit& c){ c.swap(0, 1); }},
        {"iswap", [](QuantumCircuit& c){ c.iswap(0, 1); }},
        {"ecr",   [](QuantumCircuit& c){ c.ecr(0, 1); }},
        {"cp",    [](QuantumCircuit& c){ c.cp(0.6, 0, 1); }},
        {"crx",   [](QuantumCircuit& c){ c.crx(0.7, 0, 1); }},
        {"cry",   [](QuantumCircuit& c){ c.cry(-0.4, 0, 1); }},
        {"crz",   [](QuantumCircuit& c){ c.crz(1.1, 0, 1); }},
        {"rxx",   [](QuantumCircuit& c){ c.rxx(0.5, 0, 1); }},
        {"ryy",   [](QuantumCircuit& c){ c.ryy(-0.3, 0, 1); }},
        {"rzz",   [](QuantumCircuit& c){ c.rzz(0.9, 0, 1); }},
        {"rzx",   [](QuantumCircuit& c){ c.rzx(0.8, 0, 1); }},
        {"cu",    [](QuantumCircuit& c){ c.cu(0.4, 0.5, -0.6, 0.2, 0, 1); }},
    };
    for (const Case& cs : cases) {
        // Local gates on both wires around two copies of the entangler. The
        // local gates end the run, so this is two blocks of one rather than
        // one block of two.
        QuantumCircuit qc(2);
        qc.ry(0.3, 0).rz(0.5, 1);
        cs.apply(qc);
        qc.rx(0.4, 0).h(1);
        cs.apply(qc);
        qc.rz(-0.2, 0);
        SCOPED_TRACE(cs.name);
        expect_equiv(run_pass(ConsolidateBlocks(), qc), qc);
    }

    // A raw 2-qubit UNITARY block (LSB convention) must also be preserved.
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);
    auto u4 = Operator::from_circuit(src).data;
    QuantumCircuit uq(2);
    uq.ry(0.2, 0).unitary(u4, {0, 1}).rz(0.3, 1).unitary(u4, {0, 1});
    expect_equiv(run_pass(ConsolidateBlocks(), uq), uq);
}

TEST(R1121Passes, ConsolidateBlocksDegenerateKakCases) {
    // Degenerate KAK spectra (equal interaction coefficients) stress the
    // Takagi step; the verification net guarantees correctness regardless.
    //
    // Unlike the full-gate-set case above, both fixtures here place their two
    // entanglers ADJACENTLY, with no local gate between them, so each really is
    // one block of two and the decomposition is reached. What is still not
    // asserted is whether the decomposition was kept: equivalence holds either
    // way, since the fallback returns the input.
    {
        QuantumCircuit qc(2);
        qc.swap(0, 1).cz(0, 1);  // [SWAP then CZ]
        expect_equiv(run_pass(ConsolidateBlocks(), qc), qc);
    }
    {
        QuantumCircuit qc(2);
        qc.iswap(0, 1).iswap(0, 1);  // [ISWAP, ISWAP] = identity up to phase
        expect_equiv(run_pass(ConsolidateBlocks(), qc), qc);
    }
}

TEST(R1121Passes, ConsolidateBlocksDoesNotAbsorbConditionedGate) {
    // A conditioned 2q gate breaks the block (it cannot be merged into a
    // unitary block); it must survive the pass.
    QuantumCircuit qc(2, 1);
    qc.h(0).cx(0, 1).cx(0, 1);
    qc.add_if(0, 1, Instruction::GateType::CX, {0, 1});
    auto out = run_pass(ConsolidateBlocks(), qc);
    int conditioned = 0;
    for (const auto& inst : out.instructions)
        if (inst.condition_clbit >= 0) ++conditioned;
    EXPECT_EQ(conditioned, 1) << "conditioned CX must be preserved";
}

// =============================================================================
// CommutativeCancellation — control-side merge, target-side block, conditions
// =============================================================================

TEST(R1121Passes, CommutativeCancellationAdjacentRzCancels) {
    QuantumCircuit qc(1);
    qc.rz(0.4, 0).rz(-0.4, 0);  // sums to zero
    auto out = run_pass(CommutativeCancellation(), qc);
    expect_equiv(out, identity_circuit(1));
    EXPECT_LE(out.count_ops().count("rz") ? out.count_ops().at("rz") : 0, 1);
}

TEST(R1121Passes, CommutativeCancellationControlVsTargetSideRz) {
    // rz on the CONTROL commutes through CX; rz on the TARGET does not. Both
    // patterns must preserve the unitary (the pass must never wrongly cancel).
    QuantumCircuit ctrl(2);
    ctrl.rz(0.3, 0).cx(0, 1).rz(0.5, 0).cx(0, 1);  // control-side
    expect_equiv(run_pass(CommutativeCancellation(), ctrl), ctrl);

    QuantumCircuit tgt(2);
    tgt.rz(0.3, 1).cx(0, 1).rz(0.5, 1).cx(0, 1);  // target-side (blocked)
    expect_equiv(run_pass(CommutativeCancellation(), tgt), tgt);
}

TEST(R1121Passes, CommutativeCancellationKeepsConditionedGate) {
    QuantumCircuit qc(2, 1);
    qc.cx(0, 1);
    qc.add_if(0, 1, Instruction::GateType::CX, {0, 1});  // can't cancel the pair
    auto out = run_pass(CommutativeCancellation(), qc);
    int cx = out.count_ops().count("cx") ? out.count_ops().at("cx") : 0;
    EXPECT_GE(cx, 2) << "an unconditional and a conditional CX must not cancel";
}

// =============================================================================
// transpile() routing validity across topologies (structural; robust to the
// Optimize1qGates/zyz defect, which corrupts the unitary but not the routing)
// =============================================================================

TEST(R1121Passes, RoutingValidAcrossTopologiesAndLevels) {
    struct Topo { const char* name; CouplingMap cm; };
    const Topo topos[] = {
        {"linear5",    CouplingMap::linear(5)},
        {"grid2x3",    CouplingMap::grid(2, 3)},
        {"all_to_all4",CouplingMap::all_to_all(4)},
        {"heavy_hex7", CouplingMap::heavy_hex(7)},
    };
    QuantumCircuit qc(4, 4);
    qc.h(0).cx(0, 3).cx(1, 3).cx(2, 0).cx(1, 2).measure_all();
    for (const Topo& t : topos) {
        for (int level = 0; level <= 3; ++level) {
            SCOPED_TRACE(std::string(t.name) + " level " + std::to_string(level));
            auto out = transpile(qc, t.cm, {}, level);
            for (const auto& inst : out.instructions) {
                if (inst.qubits.size() == 2)
                    EXPECT_TRUE(t.cm.is_connected(inst.qubits[0], inst.qubits[1]))
                        << "routed 2q gate on non-adjacent " << inst.qubits[0]
                        << "," << inst.qubits[1];
            }
        }
    }
}

// =============================================================================
// RemoveDiagonalGatesBeforeMeasure / RemoveResetInZeroState (structural)
// =============================================================================

TEST(R1121Passes, RemoveDiagonalBeforeMeasureDropsRz) {
    QuantumCircuit qc(1, 1);
    qc.rz(0.5, 0).measure(0, 0);
    auto out = run_pass(RemoveDiagonalGatesBeforeMeasure(), qc);
    EXPECT_EQ(out.count_ops().count("rz"), 0u) << "diagonal gate before measure removed";
    EXPECT_EQ(out.count_ops().at("measure"), 1);
}

TEST(R1121Passes, RemoveDiagonalKeepsMidCircuitRz) {
    QuantumCircuit qc(1, 1);
    qc.rz(0.5, 0).h(0).measure(0, 0);  // rz is not immediately before measure
    auto out = run_pass(RemoveDiagonalGatesBeforeMeasure(), qc);
    EXPECT_EQ(out.count_ops().at("rz"), 1) << "rz with a non-measure successor is kept";
}

TEST(R1121Passes, RemoveResetInZeroStateDropsInitialReset) {
    QuantumCircuit qc(1);
    qc.reset(0).h(0);
    auto out = run_pass(RemoveResetInZeroState(), qc);
    EXPECT_EQ(out.count_ops().count("reset"), 0u) << "reset at |0> start removed";
    EXPECT_EQ(out.count_ops().at("h"), 1);
}

// =============================================================================
// Scheduling
// =============================================================================

TEST(R1121Passes, AsapScheduleAssignsTimes) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).x(1);
    auto out = run_pass(ASAPSchedule(), qc);
    for (const auto& inst : out.instructions)
        EXPECT_GE(inst.schedule_time, 0) << "ASAP assigns a schedule time";
}

// =============================================================================
// PassManager composition
// =============================================================================

TEST(R1121Passes, PassManagerComposesAndPreservesSemantics) {
    QuantumCircuit qc(2);
    qc.h(0).h(0).cx(0, 1).cx(0, 1).rz(0.3, 0);  // reducible
    PassManager pm;
    pm.append(std::make_unique<Optimize1qGates>());
    pm.append(std::make_unique<CXCancellation>());
    auto dag = DAGCircuit::from_circuit(qc);
    TranspilationContext ctx;
    auto out = pm.run(dag, ctx).to_circuit();
    expect_equiv(out, qc);
}

// =============================================================================
// transpile() preset levels — semantics preserved without routing
// =============================================================================

TEST(R1121Passes, TranspilePresetLevelsPreserveSemantics) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.7, 1).h(0).x(1);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("optimization_level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap(), {}, level);  // unconstrained: no routing
        expect_equiv(out, qc);
    }
}

// =============================================================================
// Routing: coupling adjacency invariant + edgeless throw
// =============================================================================

TEST(R1121Passes, RoutingMakesAllTwoQubitGatesAdjacent) {
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 3).cx(1, 3);  // 0-3 not adjacent on a line
    auto cm = CouplingMap::linear(4);
    auto out = transpile(qc, cm, {}, 1);
    for (const auto& inst : out.instructions) {
        if (inst.qubits.size() == 2) {
            EXPECT_TRUE(cm.is_connected(inst.qubits[0], inst.qubits[1]))
                << "routed 2q gate on non-adjacent qubits "
                << inst.qubits[0] << "," << inst.qubits[1];
        }
    }
}

TEST(R1121Passes, EdgelessCouplingMapThrowsOnTwoQubitGate) {
    QuantumCircuit qc(3);
    qc.cx(0, 1).cx(1, 2);
    // CouplingMap(3) is literal: 3 qubits, zero edges -> unroutable.
    EXPECT_ANY_THROW(transpile(qc, CouplingMap(3), {}, 1));
}
