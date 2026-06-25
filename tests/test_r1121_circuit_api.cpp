// R.1.12.1 total-coverage suite, Batch 1: lindblad/circuit.hpp (construction,
// validation, analysis, parameterisation, circuit algebra, control()).
// Plan: docs (R.1.12.1 coverage plan), section "Batch 1: foundations".
// Serialization (QASM/JSON) lives in test_r1121_circuit_serialization.cpp.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

constexpr double kTol = 1e-10;

// Final statevector of a (measurement-free) circuit.
Statevector run_sv(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 1);
    EXPECT_TRUE(res.success) << res.error_message;
    return std::move(res.final_state);
}

// Dense matrices of two circuits agree entry-by-entry.
void expect_equivalent(const QuantumCircuit& a, const QuantumCircuit& b,
                       double tol = 1e-9) {
    ASSERT_EQ(a.n_qubits, b.n_qubits);
    auto ma = Operator::from_circuit(a).data;
    auto mb = Operator::from_circuit(b).data;
    ASSERT_EQ(ma.size(), mb.size());
    for (size_t i = 0; i < ma.size(); ++i) {
        EXPECT_NEAR(ma[i].real, mb[i].real, tol) << "entry " << i;
        EXPECT_NEAR(ma[i].imag, mb[i].imag, tol) << "entry " << i;
    }
}

}  // namespace

// =============================================================================
// Construction and validation
// =============================================================================

TEST(R1121CircuitApi, ConstructorsAndNegativeCountsThrow) {
    QuantumCircuit d;
    EXPECT_EQ(d.n_qubits, 0);
    EXPECT_EQ(d.n_clbits, 0);

    QuantumCircuit named(3, 2, "bell_prep");
    EXPECT_EQ(named.n_qubits, 3);
    EXPECT_EQ(named.n_clbits, 2);
    EXPECT_EQ(named.name, "bell_prep");

    EXPECT_THROW(QuantumCircuit(-1, 0), std::invalid_argument);
    EXPECT_THROW(QuantumCircuit(2, -1), std::invalid_argument);
}

TEST(R1121CircuitApi, EveryBuilderValidatesQubitRange) {
    QuantumCircuit qc(2, 1);
    EXPECT_THROW(qc.h(2), std::out_of_range);
    EXPECT_THROW(qc.x(-1), std::out_of_range);
    EXPECT_THROW(qc.rx(0.5, 2), std::out_of_range);
    EXPECT_THROW(qc.u(0.1, 0.2, 0.3, 5), std::out_of_range);
    EXPECT_THROW(qc.u1(0.1, 2), std::out_of_range);
    EXPECT_THROW(qc.u2(0.1, 0.2, 2), std::out_of_range);
    EXPECT_THROW(qc.u3(0.1, 0.2, 0.3, 2), std::out_of_range);
    EXPECT_THROW(qc.cx(0, 2), std::out_of_range);
    EXPECT_THROW(qc.cry(0.1, 0, 2), std::out_of_range);
    EXPECT_THROW(qc.ccx(0, 1, 2), std::out_of_range);
    EXPECT_THROW(qc.measure(2, 0), std::out_of_range);
    EXPECT_THROW(qc.measure(0, 1), std::out_of_range);  // clbit out of range
    EXPECT_THROW(qc.reset(2), std::out_of_range);
    EXPECT_THROW(qc.barrier({0, 2}), std::out_of_range);
    EXPECT_THROW(qc.p_if(0.5, 0, 1), std::out_of_range);   // clbit
    EXPECT_THROW(qc.add_if(1, 1, GT::X, {0}), std::out_of_range);
}

TEST(R1121CircuitApi, DistinctOperandValidation) {
    QuantumCircuit qc(3);
    EXPECT_THROW(qc.cx(1, 1), std::invalid_argument);
    EXPECT_THROW(qc.swap(0, 0), std::invalid_argument);
    EXPECT_THROW(qc.ecr(2, 2), std::invalid_argument);
    EXPECT_THROW(qc.crz(0.3, 1, 1), std::invalid_argument);
    EXPECT_THROW(qc.cu(0.1, 0.2, 0.3, 0.4, 2, 2), std::invalid_argument);
    EXPECT_THROW(qc.rzz(0.3, 0, 0), std::invalid_argument);
    EXPECT_THROW(qc.ccx(0, 0, 1), std::invalid_argument);
    EXPECT_THROW(qc.ccz(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(qc.cswap(2, 0, 2), std::invalid_argument);
    EXPECT_THROW(qc.rccx(1, 2, 1), std::invalid_argument);
}

TEST(R1121CircuitApi, BuildersRecordOperandsAndParams) {
    QuantumCircuit qc(3, 3);
    qc.cry(0.25, 2, 0);
    qc.crz(-0.5, 1, 2);
    qc.cu(0.1, 0.2, 0.3, 0.4, 0, 1);
    qc.u1(0.7, 1);
    qc.u2(0.7, -0.3, 2);
    qc.u3(0.1, 0.2, 0.3, 0);

    ASSERT_EQ(qc.instructions.size(), 6u);
    EXPECT_EQ(qc.instructions[0].type, GT::CRY);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{2, 0}));
    EXPECT_EQ(qc.instructions[0].params, (std::vector<double>{0.25}));
    EXPECT_EQ(qc.instructions[2].type, GT::CU);
    EXPECT_EQ(qc.instructions[2].params.size(), 4u);
    EXPECT_EQ(qc.instructions[3].type, GT::U1);
    EXPECT_EQ(qc.instructions[4].type, GT::U2);
    EXPECT_EQ(qc.instructions[4].params, (std::vector<double>{0.7, -0.3}));
    EXPECT_EQ(qc.instructions[5].type, GT::U3);
}

TEST(R1121CircuitApi, FluentInterfaceChains) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.3, 1).barrier().x(0);
    EXPECT_EQ(qc.instructions.size(), 5u);
}

// =============================================================================
// Special operations
// =============================================================================

TEST(R1121CircuitApi, MeasureAllWidensClassicalRegister) {
    QuantumCircuit qc(3, 1);
    qc.measure_all();
    EXPECT_EQ(qc.n_clbits, 3);
    EXPECT_EQ(qc.instructions.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(qc.instructions[i].type, GT::MEASURE);
        EXPECT_EQ(qc.instructions[i].qubits[0], i);
        EXPECT_EQ(qc.instructions[i].clbits[0], i);
    }

    QuantumCircuit wide(2, 5);
    wide.measure_all();
    EXPECT_EQ(wide.n_clbits, 5) << "wider registers must not shrink";
}

TEST(R1121CircuitApi, BarrierDefaultCoversAllQubitsSubsetKept) {
    QuantumCircuit qc(3);
    qc.barrier();
    EXPECT_EQ(qc.instructions[0].qubits.size(), 3u);
    qc.barrier({1});
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{1}));
}

TEST(R1121CircuitApi, ConditionalBuildersRecordConditions) {
    QuantumCircuit qc(2, 2);
    qc.p_if(0.7, 1, 0);  // default clval = 1
    qc.add_if(1, 0, GT::CX, {0, 1});
    EXPECT_EQ(qc.instructions[0].type, GT::P);
    EXPECT_EQ(qc.instructions[0].condition_clbit, 0);
    EXPECT_EQ(qc.instructions[0].condition_value, 1);
    EXPECT_EQ(qc.instructions[1].type, GT::CX);
    EXPECT_EQ(qc.instructions[1].condition_clbit, 1);
    EXPECT_EQ(qc.instructions[1].condition_value, 0);
}

// =============================================================================
// Instruction utilities
// =============================================================================

TEST(R1121CircuitApi, GateNameCoversEveryConstructibleType) {
    QuantumCircuit qc(3, 3);
    qc.h(0).x(0).y(0).z(0).s(0).sdg(0).t(0).tdg(0).sx(0).sxdg(0);
    qc.rx(0.1, 0).ry(0.1, 0).rz(0.1, 0).p(0.1, 0);
    qc.u(0.1, 0.2, 0.3, 0).u1(0.1, 0).u2(0.1, 0.2, 0).u3(0.1, 0.2, 0.3, 0);
    qc.cx(0, 1).cy(0, 1).cz(0, 1).ch(0, 1).swap(0, 1).iswap(0, 1);
    qc.crx(0.1, 0, 1).cry(0.1, 0, 1).crz(0.1, 0, 1).cp(0.1, 0, 1);
    qc.cu(0.1, 0.2, 0.3, 0.4, 0, 1).ecr(0, 1);
    qc.rzx(0.1, 0, 1).rxx(0.1, 0, 1).ryy(0.1, 0, 1).rzz(0.1, 0, 1);
    qc.ccx(0, 1, 2).ccz(0, 1, 2).cswap(0, 1, 2).rccx(0, 1, 2);
    qc.measure(0, 0).reset(0).barrier();

    const char* expected[] = {
        "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg",
        "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
        "cx", "cy", "cz", "ch", "swap", "iswap",
        "crx", "cry", "crz", "cp", "cu", "ecr",
        "rzx", "rxx", "ryy", "rzz",
        "ccx", "ccz", "cswap", "rccx",
        "measure", "reset", "barrier"};
    ASSERT_EQ(qc.instructions.size(), sizeof(expected) / sizeof(expected[0]));
    for (size_t i = 0; i < qc.instructions.size(); ++i)
        EXPECT_EQ(qc.instructions[i].gate_name(), expected[i]) << "index " << i;
}

TEST(R1121CircuitApi, UnitaryGateNameUsesLabel) {
    QuantumCircuit qc(1);
    std::vector<Complex128> X = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    qc.unitary(X, {0});
    EXPECT_EQ(qc.instructions[0].gate_name(), "unitary");
    qc.unitary(X, {0}, "my_box");
    EXPECT_EQ(qc.instructions[1].gate_name(), "my_box");
}

TEST(R1121CircuitApi, InstructionPredicates) {
    QuantumCircuit qc(2, 1);
    qc.rx("alpha", 0);
    qc.measure(0, 0);
    qc.reset(1);
    qc.cx(0, 1);
    EXPECT_TRUE(qc.instructions[0].is_parameterised());
    EXPECT_FALSE(qc.instructions[3].is_parameterised());
    EXPECT_TRUE(qc.instructions[1].is_classical());
    EXPECT_TRUE(qc.instructions[2].is_classical());
    EXPECT_FALSE(qc.instructions[3].is_classical());
    EXPECT_EQ(qc.instructions[3].num_qubits(), 2);
}

// =============================================================================
// Analysis: depth, size, count_ops, num_parameters
// =============================================================================

TEST(R1121CircuitApi, DepthCountsCriticalPathAndIgnoresBarriers) {
    QuantumCircuit qc(3);
    EXPECT_EQ(qc.depth(), 0);
    qc.h(0);          // wire 0 depth 1
    qc.h(1);          // parallel: depth stays 1
    qc.barrier();     // ignored by depth
    qc.cx(0, 1);      // depth 2 on wires 0, 1
    qc.x(2);          // wire 2 depth 1
    qc.cx(1, 2);      // max(2, 1) + 1 = 3
    EXPECT_EQ(qc.depth(), 3);
}

TEST(R1121CircuitApi, SizeExcludesBarriersCountOpsIncludesEverything) {
    QuantumCircuit qc(2, 2);
    qc.h(0).barrier().cx(0, 1).barrier().measure(0, 0);
    EXPECT_EQ(qc.size(), 3);  // h, cx, measure
    auto ops = qc.count_ops();
    EXPECT_EQ(ops.at("h"), 1);
    EXPECT_EQ(ops.at("cx"), 1);
    EXPECT_EQ(ops.at("barrier"), 2);
    EXPECT_EQ(ops.at("measure"), 1);
}

// =============================================================================
// Symbolic parameters
// =============================================================================

TEST(R1121CircuitApi, ParameterRegistryDeduplicates) {
    QuantumCircuit qc(2);
    qc.rx("a", 0);
    qc.ry("b", 1);
    qc.rz("a", 1);  // duplicate name
    EXPECT_EQ(qc.num_parameters(), 2);
}

TEST(R1121CircuitApi, AssignParametersBindsAndConverts) {
    QuantumCircuit qc(1);
    qc.rx("a", 0);
    qc.rz("b", 0);
    auto bound = qc.assign_parameters({{"a", 0.7}});

    EXPECT_EQ(bound.instructions[0].type, GT::RX);
    ASSERT_EQ(bound.instructions[0].params.size(), 1u);
    EXPECT_NEAR(bound.instructions[0].params[0], 0.7, kTol);
    EXPECT_TRUE(bound.instructions[1].is_parameterised())
        << "unbound parameters must remain symbolic";
    EXPECT_EQ(bound.num_parameters(), 1);  // only "b" remains
    EXPECT_EQ(qc.num_parameters(), 2) << "assign_parameters must not mutate";

    auto full = bound.assign_parameters({{"b", -0.3}});
    EXPECT_EQ(full.num_parameters(), 0);
    EXPECT_EQ(full.instructions[1].type, GT::RZ);

    // Fully bound circuit equals the directly-built one.
    QuantumCircuit direct(1);
    direct.rx(0.7, 0);
    direct.rz(-0.3, 0);
    expect_equivalent(full, direct);
}

TEST(R1121CircuitApi, ParamExprEvalTreeAndErrors) {
    // (a + 2) * b / 4
    auto expr = ParamExpr::make_binary(
        '/',
        ParamExpr::make_binary(
            '*',
            ParamExpr::make_binary('+', ParamExpr::make_name("a"),
                                   ParamExpr::make_literal(2.0)),
            ParamExpr::make_name("b")),
        ParamExpr::make_literal(4.0));

    std::unordered_map<std::string, double> bind = {{"a", 1.0}, {"b", 8.0}};
    EXPECT_NEAR(expr.eval(bind), 6.0, kTol);

    // Deep copy: mutating the copy's source must not alias.
    ParamExpr copy = expr;
    EXPECT_NEAR(copy.eval(bind), 6.0, kTol);

    std::unordered_map<std::string, double> missing = {{"a", 1.0}};
    EXPECT_THROW(expr.eval(missing), std::runtime_error);

    auto div0 = ParamExpr::make_binary('/', ParamExpr::make_literal(1.0),
                                       ParamExpr::make_literal(0.0));
    EXPECT_THROW(div0.eval({}), std::runtime_error);
}

TEST(R1121CircuitApi, BindParametersResolvesExpressions) {
    QuantumCircuit qc(1);
    Instruction inst;
    inst.type = GT::RZ;
    inst.qubits = {0};
    inst.param_exprs.push_back(ParamExpr::make_binary(
        '*', ParamExpr::make_name("theta"), ParamExpr::make_literal(2.0)));
    qc.instructions.push_back(std::move(inst));

    EXPECT_THROW(qc.bind_parameters({}), std::runtime_error);

    qc.bind_parameters({{"theta", 0.35}});
    ASSERT_EQ(qc.instructions[0].params.size(), 1u);
    EXPECT_NEAR(qc.instructions[0].params[0], 0.7, kTol);
    EXPECT_TRUE(qc.instructions[0].param_exprs.empty());
}

// =============================================================================
// Circuit algebra: compose, repeat, inverse
// =============================================================================

TEST(R1121CircuitApi, ComposeDirectAndMapped) {
    QuantumCircuit a(2);
    a.h(0);
    QuantumCircuit b(2);
    b.cx(0, 1);

    auto direct = a.compose(b);
    EXPECT_EQ(direct.instructions.size(), 2u);
    EXPECT_EQ(direct.instructions[1].qubits, (std::vector<int>{0, 1}));

    // Map b's qubits {0,1} onto {2,1} of a 3-qubit host.
    QuantumCircuit host(3);
    auto mapped = host.compose(b, {2, 1});
    EXPECT_EQ(mapped.instructions[0].qubits, (std::vector<int>{2, 1}));

    EXPECT_THROW(host.compose(b, {0}), std::invalid_argument);
}

TEST(R1121CircuitApi, RepeatConcatenates) {
    QuantumCircuit qc(1);
    qc.x(0);
    EXPECT_EQ(qc.repeat(0).instructions.size(), 0u);
    EXPECT_EQ(qc.repeat(3).instructions.size(), 3u);

    // X repeated twice is the identity.
    auto twice = qc.repeat(2);
    auto sv = run_sv(twice);
    EXPECT_NEAR(sv.probability(0), 1.0, kTol);
}

TEST(R1121CircuitApi, InverseUndoesEveryGateFamily) {
    // One representative of every invertible family, asymmetric parameters.
    QuantumCircuit qc(3);
    qc.h(0).x(1).y(2).z(0).s(1).sdg(2).t(0).tdg(1).sx(2).sxdg(0);
    qc.rx(0.7, 0).ry(-0.4, 1).rz(1.1, 2).p(0.3, 0);
    qc.u(0.5, 0.2, -0.6, 1).u1(0.9, 2).u2(0.4, -0.2, 0).u3(0.3, 0.1, 0.8, 1);
    qc.cx(0, 1).cy(1, 2).cz(0, 2).ch(2, 0).swap(0, 1).iswap(1, 2);
    qc.crx(0.6, 0, 1).cry(-0.2, 1, 2).crz(0.8, 2, 0).cp(0.5, 0, 2);
    qc.cu(0.3, 0.4, -0.5, 0.2, 1, 0).ecr(0, 1);
    qc.rzx(0.4, 1, 2).rxx(0.6, 0, 2).ryy(-0.3, 0, 1).rzz(0.5, 1, 2);
    qc.ccx(0, 1, 2).ccz(2, 1, 0).cswap(1, 0, 2);
    std::vector<Complex128> arbitrary = {
        {0.6, 0.0}, {0.0, 0.8}, {0.0, 0.8}, {0.6, 0.0}};
    // Make it unitary: this is 0.6 I + 0.8i X, a valid unitary.
    qc.unitary(arbitrary, {1}, "u_custom");

    auto inv = qc.inverse();
    auto round_trip = qc.compose(inv);
    auto sv = run_sv(round_trip);
    EXPECT_NEAR(sv.probability(0), 1.0, 1e-8)
        << "circuit followed by its inverse must restore |000>";
}

TEST(R1121CircuitApi, InverseDropsMeasureKeepsBarrier) {
    QuantumCircuit qc(1, 1);
    qc.h(0).barrier().measure(0, 0);
    auto inv = qc.inverse();
    auto ops = inv.count_ops();
    EXPECT_EQ(ops.count("measure"), 0u);
    EXPECT_EQ(ops.at("barrier"), 1);
    EXPECT_EQ(ops.at("h"), 1);
}

TEST(R1121CircuitApi, InverseU2ParameterRule) {
    // U2(phi, lambda)^dagger == U2(-lambda - pi, -phi + pi): verify by round trip.
    QuantumCircuit qc(1);
    qc.u2(0.4, -0.7, 0);
    auto rt = qc.compose(qc.inverse());
    auto sv = run_sv(rt);
    EXPECT_NEAR(sv.probability(0), 1.0, 1e-9);

    auto inv = qc.inverse();  // bind temporary; reference into it must not dangle
    const auto& inv_inst = inv.instructions[0];
    ASSERT_EQ(inv_inst.type, GT::U2);
    EXPECT_NEAR(inv_inst.params[0], 0.7 - PI, kTol);
    EXPECT_NEAR(inv_inst.params[1], -0.4 + PI, kTol);
}

TEST(R1121CircuitApi, InverseISwapBecomesUnitaryAdjoint) {
    QuantumCircuit qc(2);
    qc.iswap(0, 1);
    auto inv = qc.inverse();
    ASSERT_EQ(inv.instructions.size(), 1u);
    EXPECT_EQ(inv.instructions[0].type, GT::UNITARY);
    auto rt = qc.compose(inv);
    auto sv = run_sv(rt);
    EXPECT_NEAR(sv.probability(0), 1.0, 1e-9);
}

// =============================================================================
// control(): handled mappings, generic path, multi-control, conditions
// =============================================================================

TEST(R1121CircuitApi, ControlHandledSingleQubitMappings) {
    struct Case {
        void (*build)(QuantumCircuit&);
        GT expected;
    };
    // X->CX, H->CH, RZ->CRZ, P->CP, S->CP(pi/2), T->CP(pi/4), U->CU
    QuantumCircuit bx(1); bx.x(0);
    EXPECT_EQ(bx.control(1).instructions[0].type, GT::CX);
    QuantumCircuit bh(1); bh.h(0);
    EXPECT_EQ(bh.control(1).instructions[0].type, GT::CH);
    QuantumCircuit brz(1); brz.rz(0.3, 0);
    EXPECT_EQ(brz.control(1).instructions[0].type, GT::CRZ);
    QuantumCircuit bp(1); bp.p(0.3, 0);
    EXPECT_EQ(bp.control(1).instructions[0].type, GT::CP);

    QuantumCircuit bs(1); bs.s(0);
    auto bsc = bs.control(1);  // bind temporary; reference into it must not dangle
    const auto& cs = bsc.instructions[0];
    EXPECT_EQ(cs.type, GT::CP);
    EXPECT_NEAR(cs.params[0], PI_2, kTol);

    QuantumCircuit bt(1); bt.tdg(0);
    auto btc = bt.control(1);
    const auto& ct = btc.instructions[0];
    EXPECT_EQ(ct.type, GT::CP);
    EXPECT_NEAR(ct.params[0], -PI_4, kTol);

    QuantumCircuit bu(1); bu.u(0.5, 0.2, -0.1, 0);
    auto buc = bu.control(1);
    const auto& cu = buc.instructions[0];
    EXPECT_EQ(cu.type, GT::CU);
    ASSERT_EQ(cu.params.size(), 4u);
    EXPECT_NEAR(cu.params[3], 0.0, kTol);  // gamma defaults to 0

    QuantumCircuit bcx(2); bcx.cx(0, 1);
    EXPECT_EQ(bcx.control(1).instructions[0].type, GT::CCX);
    QuantumCircuit bsw(2); bsw.swap(0, 1);
    EXPECT_EQ(bsw.control(1).instructions[0].type, GT::CSWAP);
}

TEST(R1121CircuitApi, ControlGenericPathMatchesReference) {
    // control(1) of CZ must equal CCZ (compare dense matrices).
    QuantumCircuit base(2);
    base.cz(0, 1);
    auto ctrl = base.control(1);
    QuantumCircuit ref(3);
    ref.ccz(0, 1, 2);
    expect_equivalent(ctrl, ref);

    // control(2) of X must equal CCX with controls {0, 1} and target 2.
    QuantumCircuit bx(1);
    bx.x(0);
    auto ccx = bx.control(2);
    QuantumCircuit ref2(3);
    ref2.ccx(0, 1, 2);
    expect_equivalent(ccx, ref2);
}

TEST(R1121CircuitApi, ControlledSXActsOnTarget) {
    QuantumCircuit base(1);
    base.sx(0);
    auto csx = base.control(1);

    QuantumCircuit qc(2);
    qc.x(0);  // control = |1>
    for (const auto& inst : csx.instructions) qc.instructions.push_back(inst);
    auto sv = run_sv(qc);
    EXPECT_NEAR(sv.probability(1), 0.5, 1e-9);
    EXPECT_NEAR(sv.probability(3), 0.5, 1e-9);

    QuantumCircuit qc0(2);  // control = |0>: identity
    for (const auto& inst : csx.instructions) qc0.instructions.push_back(inst);
    auto sv0 = run_sv(qc0);
    EXPECT_NEAR(sv0.probability(0), 1.0, 1e-9);
}

TEST(R1121CircuitApi, ControlPropagatesConditionsAndRejectsSymbolic) {
    QuantumCircuit base(1, 1);
    base.p_if(0.4, 0, 0);
    auto ctrl = base.control(1);
    EXPECT_EQ(ctrl.instructions[0].condition_clbit, 0);
    EXPECT_EQ(ctrl.instructions[0].condition_value, 1);

    QuantumCircuit sym(1);
    sym.rx("a", 0);
    // Handled mapping exists for RX only with bound params; symbolic gates
    // route to the generic path, which must throw rather than emit identity.
    EXPECT_THROW(sym.control(1), std::runtime_error);

    QuantumCircuit any(1);
    any.x(0);
    EXPECT_THROW(any.control(0), std::invalid_argument);
}

TEST(R1121CircuitApi, ControlPassesThroughMeasureResetBarrier) {
    QuantumCircuit base(1, 1);
    base.barrier();
    base.measure(0, 0);
    base.reset(0);
    auto ctrl = base.control(1);
    ASSERT_EQ(ctrl.instructions.size(), 3u);
    EXPECT_EQ(ctrl.instructions[0].type, GT::BARRIER);
    EXPECT_EQ(ctrl.instructions[1].type, GT::MEASURE);
    EXPECT_EQ(ctrl.instructions[1].qubits[0], 1) << "shifted past the control";
    EXPECT_EQ(ctrl.instructions[2].type, GT::RESET);
}

// =============================================================================
// unitary() builder
// =============================================================================

TEST(R1121CircuitApi, UnitaryBuilderValidatesQubits) {
    QuantumCircuit qc(2);
    std::vector<Complex128> X = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    EXPECT_THROW(qc.unitary(X, {2}), std::out_of_range);
    EXPECT_NO_THROW(qc.unitary(X, {1}));
    EXPECT_EQ(qc.instructions[0].matrix.size(), 4u);
}

TEST(R1121CircuitApi, UnitaryInstructionExecutesWithLSBConvention) {
    // CX-shaped matrix, control = qubits[0]: on qubits {1, 0} of n=2 the
    // control is q1 and the target is q0.
    std::vector<Complex128> CXm(16, Complex128(0, 0));
    CXm[0 * 4 + 0] = {1, 0};
    CXm[1 * 4 + 3] = {1, 0};
    CXm[2 * 4 + 2] = {1, 0};
    CXm[3 * 4 + 1] = {1, 0};

    QuantumCircuit qc(2);
    qc.x(1);                    // set q1 (the control under {1, 0})
    qc.unitary(CXm, {1, 0});
    auto sv = run_sv(qc);
    EXPECT_NEAR(sv.probability(3), 1.0, 1e-9)
        << "control q1 = 1 must flip target q0";
}
