// R.1.12.2 coverage-fill suite, batch F2: transpiler. Closes the
// line-coverage gaps measured on the instrumented R.1.12.1 run:
//
//   - transpiler.hpp: the inline name() of every pass.
//   - basis_translator.cpp: decompose_to_cx_u3 recipes for every gate type
//     (the default {cx, u3} target), explicit-basis passthrough, the
//     MEASURE/RESET/BARRIER passthrough, degenerate CouplingMap constructors.
//   - sabre_layout.cpp: the heuristic and forward/backward/final passes via a
//     routing-pressure circuit on a linear coupling map.
//   - sabre_swap.cpp: the 3q all-pairs adjacency contract (execute vs throw),
//     the disconnected-components SWAP-budget throw, ctx.initial_layout.
//   - scheduling.cpp: the BARRIER synchronisation branches of ASAP and ALAP.
//   - optimize_1q.cpp: the remaining instruction_to_2x2 branches via
//     inverse-pair runs that merge to identity (deliberately identity-only:
//     generic merges are the open zyz_decompose finding and stay red in their
//     own pinned tests), and the RZZ/RYY interaction emission of the KAK
//     consolidation path.
//   - commutative_cancellation.cpp: zero-sum rotation removal and the
//     classical-condition guard in commutes_on_wire.
//   - remove_diagonal.cpp: RemoveResetInZeroState keeping the first RESET on
//     a dirtied qubit.
//
// Semantics oracle: full-matrix equivalence up to global phase, matching the
// R.1.12.1 pass suites.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/transpiler.hpp"

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

QuantumCircuit run_pass(const TranspilationPass& pass, const QuantumCircuit& qc,
                        TranspilationContext ctx = {}) {
    auto dag = DAGCircuit::from_circuit(qc);
    return pass.run(dag, ctx).to_circuit();
}

// Unitary equivalence up to an unobservable global phase.
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

int count_type(const QuantumCircuit& qc, GT t) {
    int n = 0;
    for (const auto& inst : qc.instructions)
        if (inst.type == t) ++n;
    return n;
}

std::map<GT, int> type_multiset(const QuantumCircuit& qc) {
    std::map<GT, int> m;
    for (const auto& inst : qc.instructions) ++m[inst.type];
    return m;
}

void prep(QuantumCircuit& qc) {
    for (int q = 0; q < qc.n_qubits; ++q) {
        qc.h(q);
        qc.rx(0.3 + 0.2 * q, q);
        qc.t(q);
    }
    if (qc.n_qubits >= 2) qc.cx(0, 1);
}

using Apply = std::function<void(QuantumCircuit&)>;

// Every constructible gate, for the basis-translation sweep.
std::vector<std::pair<std::string, Apply>> all_gates() {
    return {
        {"h", [](QuantumCircuit& q) { q.h(0); }},
        {"x", [](QuantumCircuit& q) { q.x(0); }},
        {"y", [](QuantumCircuit& q) { q.y(0); }},
        {"z", [](QuantumCircuit& q) { q.z(0); }},
        {"s", [](QuantumCircuit& q) { q.s(0); }},
        {"sdg", [](QuantumCircuit& q) { q.sdg(0); }},
        {"t", [](QuantumCircuit& q) { q.t(0); }},
        {"tdg", [](QuantumCircuit& q) { q.tdg(0); }},
        {"sx", [](QuantumCircuit& q) { q.sx(0); }},
        {"sxdg", [](QuantumCircuit& q) { q.sxdg(0); }},
        {"rx", [](QuantumCircuit& q) { q.rx(0.37, 0); }},
        {"ry", [](QuantumCircuit& q) { q.ry(0.53, 0); }},
        {"rz", [](QuantumCircuit& q) { q.rz(0.71, 0); }},
        {"p", [](QuantumCircuit& q) { q.p(0.29, 0); }},
        {"u", [](QuantumCircuit& q) { q.u(0.41, 0.23, 0.17, 0); }},
        {"u1", [](QuantumCircuit& q) { q.u1(0.61, 0); }},
        {"u2", [](QuantumCircuit& q) { q.u2(0.43, 0.19, 0); }},
        {"u3", [](QuantumCircuit& q) { q.u3(0.57, 0.31, 0.13, 0); }},
        {"cx", [](QuantumCircuit& q) { q.cx(0, 1); }},
        {"cy", [](QuantumCircuit& q) { q.cy(0, 1); }},
        {"cz", [](QuantumCircuit& q) { q.cz(0, 1); }},
        {"ch", [](QuantumCircuit& q) { q.ch(0, 1); }},
        {"swap", [](QuantumCircuit& q) { q.swap(0, 1); }},
        {"iswap", [](QuantumCircuit& q) { q.iswap(0, 1); }},
        {"crx", [](QuantumCircuit& q) { q.crx(0.37, 0, 1); }},
        {"cry", [](QuantumCircuit& q) { q.cry(0.53, 0, 1); }},
        {"crz", [](QuantumCircuit& q) { q.crz(0.71, 0, 1); }},
        {"cp", [](QuantumCircuit& q) { q.cp(0.29, 0, 1); }},
        {"cu", [](QuantumCircuit& q) { q.cu(0.41, 0.23, 0.17, 0.11, 0, 1); }},
        {"ecr", [](QuantumCircuit& q) { q.ecr(0, 1); }},
        {"rzx", [](QuantumCircuit& q) { q.rzx(0.33, 0, 1); }},
        {"rxx", [](QuantumCircuit& q) { q.rxx(0.47, 0, 1); }},
        {"ryy", [](QuantumCircuit& q) { q.ryy(0.59, 0, 1); }},
        {"rzz", [](QuantumCircuit& q) { q.rzz(0.27, 0, 1); }},
        {"ccx", [](QuantumCircuit& q) { q.ccx(0, 1, 2); }},
        {"ccz", [](QuantumCircuit& q) { q.ccz(0, 1, 2); }},
        {"cswap", [](QuantumCircuit& q) { q.cswap(0, 1, 2); }},
        {"rccx", [](QuantumCircuit& q) { q.rccx(0, 1, 2); }},
    };
}

}  // namespace

// =============================================================================
// Pass identity and preset composition
// =============================================================================

TEST(R1122FillTrans, PassNamesAreStable) {
    EXPECT_EQ(TrivialLayout().name(), "TrivialLayout");
    EXPECT_EQ(SabreLayout().name(), "SabreLayout");
    EXPECT_EQ(SabreSwap().name(), "SabreSwap");
    EXPECT_EQ(StochasticSwap().name(), "StochasticSwap");
    EXPECT_EQ(BasisTranslator().name(), "BasisTranslator");
    EXPECT_EQ(Optimize1qGates().name(), "Optimize1qGates");
    EXPECT_EQ(CXCancellation().name(), "CXCancellation");
    EXPECT_EQ(ConsolidateBlocks().name(), "ConsolidateBlocks");
    EXPECT_EQ(CommutativeCancellation().name(), "CommutativeCancellation");
    EXPECT_EQ(RemoveDiagonalGatesBeforeMeasure().name(),
              "RemoveDiagonalGatesBeforeMeasure");
    EXPECT_EQ(RemoveResetInZeroState().name(), "RemoveResetInZeroState");
    EXPECT_EQ(ASAPSchedule().name(), "ASAPSchedule");
    EXPECT_EQ(ALAPSchedule().name(), "ALAPSchedule");
}

TEST(R1122FillTrans, PresetPassManagerLevelsComposeOnCxOnlyCircuit) {
    // cx-only input: no single-qubit runs, so the open zyz_decompose finding
    // cannot perturb this test; it pins composition and the return path.
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    auto dag = DAGCircuit::from_circuit(qc);

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto pm = preset_pass_manager(level, CouplingMap(), {});
        std::vector<std::string> names;
        for (const auto& p : pm.passes) names.push_back(p->name());
        if (level >= 1) {
            EXPECT_FALSE(names.empty());
        }

        TranspilationContext ctx;
        ctx.optimization_level = level;
        auto out = pm.run(dag, ctx).to_circuit();
        expect_equiv(out, qc);
    }
}

// =============================================================================
// BasisTranslator
// =============================================================================

TEST(R1122FillTrans, BasisTranslatorDecomposesEveryGateToDefaultBasis) {
    for (const auto& [name, apply] : all_gates()) {
        SCOPED_TRACE(name);
        const int nq = (name == "ccx" || name == "ccz" || name == "cswap" ||
                        name == "rccx")
                           ? 3
                           : 2;
        QuantumCircuit qc(nq);
        apply(qc);

        auto out = run_pass(BasisTranslator(), qc);  // default basis {cx, u3}
        for (const auto& inst : out.instructions) {
            const auto n = inst.gate_name();
            EXPECT_TRUE(n == "cx" || n == "u3")
                << name << " decomposed to non-basis gate " << n;
        }
        expect_equiv(out, qc);
    }
}

TEST(R1122FillTrans, BasisTranslatorPassthroughAndExplicitBasis) {
    QuantumCircuit qc(2, 1);
    qc.h(0);
    qc.t(1);
    qc.barrier();
    qc.reset(1);
    qc.cx(0, 1);
    qc.measure(0, 0);

    TranspilationContext ctx;
    ctx.basis_gates = {"h", "t", "cx"};
    auto out = run_pass(BasisTranslator(), qc, ctx);

    EXPECT_EQ(count_type(out, GT::H), 1);        // in basis: untouched
    EXPECT_EQ(count_type(out, GT::T), 1);        // in basis: untouched
    EXPECT_EQ(count_type(out, GT::CX), 1);
    EXPECT_EQ(count_type(out, GT::BARRIER), 1);  // special ops pass through
    EXPECT_EQ(count_type(out, GT::RESET), 1);
    EXPECT_EQ(count_type(out, GT::MEASURE), 1);
}

// =============================================================================
// CouplingMap constructors: degenerate shapes
// =============================================================================

TEST(R1122FillTrans, CouplingMapDegenerateConstructors) {
    auto l1 = CouplingMap::linear(1);
    EXPECT_EQ(l1.n_physical_qubits, 1);
    EXPECT_TRUE(l1.edges.empty());

    auto row = CouplingMap::grid(1, 3);
    EXPECT_EQ(row.n_physical_qubits, 3);
    EXPECT_EQ(row.edges.size(), 4u);  // 2 undirected horizontal links

    auto col = CouplingMap::grid(3, 1);
    EXPECT_EQ(col.n_physical_qubits, 3);
    EXPECT_EQ(col.edges.size(), 4u);  // 2 undirected vertical links

    auto a1 = CouplingMap::all_to_all(1);
    EXPECT_EQ(a1.n_physical_qubits, 1);
    EXPECT_TRUE(a1.edges.empty());

    auto a3 = CouplingMap::all_to_all(3);
    EXPECT_EQ(a3.edges.size(), 6u);  // 3 undirected links, both directions
}

// =============================================================================
// SabreLayout
// =============================================================================

TEST(R1122FillTrans, SabreLayoutRemapsOntoCouplingMap) {
    // Routing pressure: long-range pairs on a linear chain force the
    // heuristic and the forward/backward/final passes to do real work.
    QuantumCircuit qc(4);
    prep(qc);
    qc.cx(0, 3).cx(1, 2).cx(0, 2).cx(3, 1);

    TranspilationContext ctx;
    ctx.coupling_map = CouplingMap::linear(4);

    auto out = run_pass(SabreLayout(), qc, ctx);
    EXPECT_EQ(out.n_qubits, 4);
    EXPECT_EQ(out.instructions.size(), qc.instructions.size())
        << "layout must relabel, never insert or drop";
    EXPECT_EQ(type_multiset(out), type_multiset(qc));
    for (const auto& inst : out.instructions)
        for (int q : inst.qubits) {
            EXPECT_GE(q, 0);
            EXPECT_LT(q, 4);
        }

    // Deterministic: same input, same layout decision.
    auto again = run_pass(SabreLayout(), qc, ctx);
    ASSERT_EQ(again.instructions.size(), out.instructions.size());
    for (size_t i = 0; i < out.instructions.size(); ++i)
        EXPECT_EQ(again.instructions[i].qubits, out.instructions[i].qubits);
}

// =============================================================================
// SabreSwap: contracts
// =============================================================================

TEST(R1122FillTrans, SabreSwapThreeQubitAllPairsAdjacencyContract) {
    QuantumCircuit qc(3);
    qc.ccx(0, 1, 2);

    // All-to-all: every wire pair adjacent, the gate is executable as-is.
    TranspilationContext ok;
    ok.coupling_map = CouplingMap::all_to_all(3);
    auto out = run_pass(SabreSwap(), qc, ok);
    EXPECT_EQ(count_type(out, GT::CCX), 1);

    // Linear chain: pair (0,2) is not adjacent; SABRE must refuse loudly.
    TranspilationContext bad;
    bad.coupling_map = CouplingMap::linear(3);
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_THROW(SabreSwap().run(dag, bad), std::runtime_error);
}

TEST(R1122FillTrans, SabreSwapDisconnectedComponentsExhaustBudget) {
    CouplingMap cm(4);
    cm.edges = {{0, 1}, {1, 0}, {2, 3}, {3, 2}};  // two disconnected islands
    QuantumCircuit qc(4);
    qc.cx(0, 2);  // spans the islands: can never be made adjacent

    TranspilationContext ctx;
    ctx.coupling_map = cm;
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_THROW(SabreSwap().run(dag, ctx), std::runtime_error);
}

TEST(R1122FillTrans, SabreSwapHonoursInitialLayout) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);

    TranspilationContext ctx;
    ctx.coupling_map = CouplingMap::linear(2);
    ctx.initial_layout = {1, 0};  // logical q -> physical 1-q

    auto out = run_pass(SabreSwap(), qc, ctx);
    EXPECT_EQ(count_type(out, GT::CX), 1);
    EXPECT_EQ(count_type(out, GT::SWAP), 0) << "adjacent under the layout";
    for (const auto& inst : out.instructions) {
        if (inst.type == GT::CX) {
            EXPECT_TRUE(ctx.coupling_map.is_connected(inst.qubits[0], inst.qubits[1]) ||
                        ctx.coupling_map.is_connected(inst.qubits[1], inst.qubits[0]));
        }
    }
}

// =============================================================================
// Scheduling: BARRIER synchronisation
// =============================================================================

TEST(R1122FillTrans, AsapAndAlapScheduleBarrierSynchronisation) {
    QuantumCircuit qc(3);
    qc.h(0);
    qc.barrier();       // syncs all wires
    qc.x(1);
    qc.barrier({0, 1});
    qc.cx(0, 1);
    qc.y(2);            // gated only by the full barrier: has ALAP slack

    // The DAG linearisation may reorder independent instructions (Kahn FIFO),
    // so times are asserted by instruction identity, never by position. The
    // two barriers are distinguished by operand count.
    auto time_of = [](const QuantumCircuit& s, GT t, size_t n_ops) -> int {
        for (const auto& inst : s.instructions)
            if (inst.type == t && inst.qubits.size() == n_ops)
                return inst.schedule_time;
        ADD_FAILURE() << "instruction not found in scheduled circuit";
        return -1;
    };

    auto asap = run_pass(ASAPSchedule(), qc);
    ASSERT_EQ(asap.instructions.size(), 6u);
    EXPECT_EQ(time_of(asap, GT::H, 1), 0);
    EXPECT_EQ(time_of(asap, GT::BARRIER, 3), 1);
    EXPECT_EQ(time_of(asap, GT::X, 1), 1);
    EXPECT_EQ(time_of(asap, GT::Y, 1), 1);
    EXPECT_EQ(time_of(asap, GT::BARRIER, 2), 2);
    EXPECT_EQ(time_of(asap, GT::CX, 2), 2);

    // ALAP: y(2) must move late; the tight chain must not move.
    auto alap = run_pass(ALAPSchedule(), qc);
    ASSERT_EQ(alap.instructions.size(), 6u);
    EXPECT_EQ(time_of(alap, GT::H, 1), 0);
    EXPECT_EQ(time_of(alap, GT::BARRIER, 3), 1);
    EXPECT_EQ(time_of(alap, GT::X, 1), 1);
    EXPECT_EQ(time_of(alap, GT::Y, 1), 2);
    EXPECT_EQ(time_of(alap, GT::BARRIER, 2), 2);
    EXPECT_EQ(time_of(alap, GT::CX, 2), 2);
}

// =============================================================================
// Optimize1qGates: remaining accumulate branches (identity-only merges)
// =============================================================================

TEST(R1122FillTrans, Optimize1qInversePairsCollapseToIdentity) {
    const double pi_2 = 1.5707963267948966;
    std::vector<std::pair<std::string, Apply>> pairs = {
        {"xx", [](QuantumCircuit& q) { q.x(0).x(0); }},
        {"yy", [](QuantumCircuit& q) { q.y(0).y(0); }},
        {"zz", [](QuantumCircuit& q) { q.z(0).z(0); }},
        {"s.sdg", [](QuantumCircuit& q) { q.s(0).sdg(0); }},
        {"t.tdg", [](QuantumCircuit& q) { q.t(0).tdg(0); }},
        {"sx.sxdg", [](QuantumCircuit& q) { q.sx(0).sxdg(0); }},
        {"rx", [](QuantumCircuit& q) { q.rx(0.6, 0).rx(-0.6, 0); }},
        {"ry", [](QuantumCircuit& q) { q.ry(0.8, 0).ry(-0.8, 0); }},
        {"rz", [](QuantumCircuit& q) { q.rz(1.1, 0).rz(-1.1, 0); }},
        {"p", [](QuantumCircuit& q) { q.p(0.4, 0).p(-0.4, 0); }},
        {"u1", [](QuantumCircuit& q) { q.u1(0.3, 0).u1(-0.3, 0); }},
        {"u2.inv", [=](QuantumCircuit& q) { q.u2(0.43, 0.19, 0).u(-pi_2, -0.19, -0.43, 0); }},
        {"u3.inv", [](QuantumCircuit& q) { q.u3(0.5, 0.3, 0.2, 0).u(-0.5, -0.2, -0.3, 0); }},
    };
    for (const auto& [name, apply] : pairs) {
        SCOPED_TRACE(name);
        QuantumCircuit qc(1);
        apply(qc);
        auto out = run_pass(Optimize1qGates(), qc);
        expect_equiv(out, QuantumCircuit(1));
        EXPECT_LE(out.instructions.size(), 1u) << "inverse pair must collapse";
    }
}

// =============================================================================
// ConsolidateBlocks: three-axis KAK interaction emission
// =============================================================================

TEST(R1122FillTrans, ConsolidateBlocksThreeAxisBlockPreservesSemantics) {
    // Canonical coordinates with all three axes non-zero force the RZZ and
    // RYY emission branches of the KAK rebuild (RXX alone is the common case).
    QuantumCircuit qc(2);
    qc.rzz(0.8, 0, 1).ryy(0.6, 0, 1).rxx(0.4, 0, 1);
    auto out = run_pass(ConsolidateBlocks(), qc);
    expect_equiv(out, qc);
}

// =============================================================================
// CommutativeCancellation
// =============================================================================

TEST(R1122FillTrans, CommutativeCancellationZeroSumRotationsRemoveBoth) {
    QuantumCircuit qc(1);
    qc.rz(0.7, 0).rz(-0.7, 0);
    auto out = run_pass(CommutativeCancellation(), qc);
    expect_equiv(out, QuantumCircuit(1));
    EXPECT_EQ(out.instructions.size(), 0u)
        << "opposite rotations sum to zero and both gates must go";
}

TEST(R1122FillTrans, CommutativeCancellationConditionBlocksStaticMotion) {
    // A classically-conditioned gate between two mergeable rotations must
    // stop the merge: its action depends on runtime state.
    QuantumCircuit qc(1, 1);
    qc.rz(0.3, 0);
    qc.add_if(0, 1, GT::Z, {0});
    qc.rz(0.4, 0);
    auto out = run_pass(CommutativeCancellation(), qc);
    EXPECT_EQ(out.instructions.size(), 3u)
        << "nothing may merge across a conditioned gate";
}

TEST(R1122FillTrans, CommutativeCancellationOddCxChainKeepsSemantics) {
    QuantumCircuit qc(2);
    qc.cx(0, 1).cx(0, 1).cx(0, 1);
    auto out = run_pass(CommutativeCancellation(), qc);
    QuantumCircuit ref(2);
    ref.cx(0, 1);
    expect_equiv(out, ref);
}

// =============================================================================
// RemoveResetInZeroState
// =============================================================================

TEST(R1122FillTrans, RemoveResetKeepsFirstResetOnDirtyQubit) {
    QuantumCircuit qc(1);
    qc.reset(0);      // qubit starts |0>: removable
    qc.x(0);          // dirties the qubit
    qc.reset(0);      // meaningful: must stay
    qc.reset(0);      // qubit known |0> again: removable
    auto out = run_pass(RemoveResetInZeroState(), qc);
    EXPECT_EQ(count_type(out, GT::RESET), 1);
    EXPECT_EQ(count_type(out, GT::X), 1);
}
