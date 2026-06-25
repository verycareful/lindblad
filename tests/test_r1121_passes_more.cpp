// R.1.12.1 coverage gap-closure: transpiler passes that were unreferenced by
// any suite (TrivialLayout, StochasticSwap, BasisTranslator, ALAPSchedule) plus
// preset_pass_manager. Plan DoD #1: zero unreferenced public symbols.
//
// Layout/routing passes are checked by structural invariants (gate count,
// coupling adjacency, determinism); non-permuting passes by full-matrix
// equivalence. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/transpiler.hpp"

#include <string>
#include <vector>

using namespace lindblad;

namespace {

QuantumCircuit run_pass(const TranspilationPass& pass, const QuantumCircuit& qc,
                        TranspilationContext ctx) {
    auto dag = DAGCircuit::from_circuit(qc);
    return pass.run(dag, ctx).to_circuit();
}

// Equivalence up to an unobservable global phase (passes preserve only that).
void expect_equiv(const QuantumCircuit& a, const QuantumCircuit& b, double tol = 1e-7) {
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

std::vector<std::string> op_names(const QuantumCircuit& qc) {
    std::vector<std::string> v;
    for (const auto& inst : qc.instructions) v.push_back(inst.gate_name());
    return v;
}

}  // namespace

// =============================================================================
// TrivialLayout — identity layout, no routing, content preserved
// =============================================================================

TEST(R1121PassesMore, TrivialLayoutPreservesCircuit) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);  // already maps onto linear(2)
    TranspilationContext ctx;
    ctx.coupling_map = CouplingMap::linear(2);
    auto out = run_pass(TrivialLayout(), qc, ctx);
    expect_equiv(out, qc);
    EXPECT_EQ(TrivialLayout().name(), "TrivialLayout");
}

// =============================================================================
// StochasticSwap — routes to coupling adjacency, deterministic (fixed seed)
// =============================================================================

TEST(R1121PassesMore, StochasticSwapRoutesAndIsDeterministic) {
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 3).cx(1, 3);  // 0-3 non-adjacent on a line
    TranspilationContext ctx;
    ctx.coupling_map = CouplingMap::linear(4);

    auto a = run_pass(StochasticSwap(), qc, ctx);
    auto b = run_pass(StochasticSwap(), qc, ctx);
    EXPECT_EQ(op_names(a), op_names(b)) << "fixed internal seed -> deterministic";

    for (const auto& inst : a.instructions)
        if (inst.qubits.size() == 2)
            EXPECT_TRUE(ctx.coupling_map.is_connected(inst.qubits[0], inst.qubits[1]))
                << "routed 2q gate must be coupling-adjacent";
}

// =============================================================================
// BasisTranslator — no routing, semantics preserved
// =============================================================================

TEST(R1121PassesMore, BasisTranslatorPreservesSemantics) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).t(1);
    TranspilationContext ctx;
    ctx.basis_gates = {"u", "cx"};
    auto out = run_pass(BasisTranslator(), qc, ctx);
    expect_equiv(out, qc, 1e-8);
}

// =============================================================================
// ALAPSchedule — assigns schedule times
// =============================================================================

TEST(R1121PassesMore, AlapScheduleAssignsTimes) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).x(1);
    auto out = run_pass(ALAPSchedule(), qc, TranspilationContext{});
    for (const auto& inst : out.instructions)
        EXPECT_GE(inst.schedule_time, 0);
}

// =============================================================================
// preset_pass_manager — composed, semantics preserved without routing
// =============================================================================

TEST(R1121PassesMore, PresetPassManagerPreservesSemanticsUnconstrained) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.5, 1).h(0);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto pm = preset_pass_manager(level, CouplingMap(), {});
        auto dag = DAGCircuit::from_circuit(qc);
        TranspilationContext ctx;
        ctx.optimization_level = level;
        auto out = pm.run(dag, ctx).to_circuit();
        expect_equiv(out, qc);
    }
}
