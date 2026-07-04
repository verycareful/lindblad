// R.1.12.2 coverage-fill suite, batch F1: simulator engines. Closes the
// line-coverage gaps measured on the instrumented R.1.12.1 run:
//
//   - mps_sim.cpp: per-gate branches of the analytic gate2x2/gate4x4 builders,
//     the three-qubit decompositions (CCX/CCZ/CSWAP/RCCX), the statevector
//     fallback for multi-qubit UNITARYs (to_statevector -> apply_unitary ->
//     mps_from_sv), UNITARY matrix-size validation, the wide-register guard,
//     unbound-parameter rejection, malformed-arity rejection, and the
//     conditional forward pass of a circuit without measurements.
//   - density_matrix_sim.cpp: per-gate branches of gate_matrix_for_dm, the
//     dense expectation_value, sparse-operator qubit-count validation,
//     apply_gate matrix-size validation, unbound-parameter failure path.
//   - statevector_sim.cpp: direct MEASURE/BARRIER handling in
//     apply_instruction, unbound-parameter throw, empty-register guards on
//     run() and eval_expectation().
//   - clifford_sim.cpp: Y/Z/RESET/BARRIER dispatch and the Pauli string
//     length validation in expectation_pauli.
//
// The statevector engine is the oracle throughout: MPS and DM results must
// reproduce its amplitudes exactly (same analytic matrices, same LSB
// conventions), so every comparison is entrywise, not up-to-phase. Inputs are
// prepared non-symmetric per the convention-testing rule in the architecture
// docs. The defensive identity fallbacks (gate2x2/gate4x4 default, DM default)
// are unreachable through validated instructions and are deliberately not
// exercised here; they are handled by the coverage residue triage instead.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

constexpr double kTol = 1e-9;

// Final statevector via the statevector engine (shots == 0, seeded).
Statevector sv_final(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 1);
    EXPECT_TRUE(res.success) << res.error_message;
    return std::move(res.final_state);
}

// Final statevector via the MPS engine (shots == 0, seeded, ample bond dim).
Statevector mps_final(const QuantumCircuit& qc, int chi = 64) {
    MPSSimulator sim;
    auto res = sim.run(qc, chi, 0, 1);
    return res.final_state.to_statevector();
}

// Final density matrix via the DM engine under an ideal noise model.
DensityMatrix dm_final(const QuantumCircuit& qc) {
    DensityMatrixSimulator sim;
    NoiseModel ideal;
    auto res = sim.run(qc, ideal, 0, 1);
    EXPECT_TRUE(res.success) << res.error_message;
    return std::move(res.final_state);
}

void expect_sv_equal(const Statevector& a, const Statevector& b,
                     double tol = kTol) {
    ASSERT_EQ(a.dim, b.dim);
    for (size_t i = 0; i < a.dim; ++i) {
        EXPECT_NEAR(a.real_parts[i], b.real_parts[i], tol) << "amp " << i;
        EXPECT_NEAR(a.imag_parts[i], b.imag_parts[i], tol) << "amp " << i;
    }
}

// rho must equal |psi><psi| entrywise.
void expect_dm_is_pure(const DensityMatrix& dm, const Statevector& sv,
                       double tol = kTol) {
    ASSERT_EQ(dm.dim, sv.dim);
    for (size_t i = 0; i < sv.dim; ++i) {
        for (size_t j = 0; j < sv.dim; ++j) {
            const double re =
                sv.real_parts[i] * sv.real_parts[j] + sv.imag_parts[i] * sv.imag_parts[j];
            const double im =
                sv.imag_parts[i] * sv.real_parts[j] - sv.real_parts[i] * sv.imag_parts[j];
            EXPECT_NEAR(dm(i, j).real, re, tol) << "rho(" << i << "," << j << ")";
            EXPECT_NEAR(dm(i, j).imag, im, tol) << "rho(" << i << "," << j << ")";
        }
    }
}

// Non-symmetric preparation: generic complex amplitudes on every qubit plus
// one entangler, so convention bugs and operand-order bugs cannot cancel.
void prep(QuantumCircuit& qc) {
    for (int q = 0; q < qc.n_qubits; ++q) {
        qc.h(q);
        qc.rx(0.3 + 0.2 * q, q);
        qc.t(q);
    }
    if (qc.n_qubits >= 2) qc.cx(0, 1);
}

using Apply = std::function<void(QuantumCircuit&)>;

const std::vector<std::pair<std::string, Apply>>& one_qubit_gates() {
    static const std::vector<std::pair<std::string, Apply>> g = {
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
    };
    return g;
}

std::vector<std::pair<std::string, Apply>> two_qubit_gates(int a, int b) {
    return {
        {"cx", [=](QuantumCircuit& q) { q.cx(a, b); }},
        {"cy", [=](QuantumCircuit& q) { q.cy(a, b); }},
        {"cz", [=](QuantumCircuit& q) { q.cz(a, b); }},
        {"ch", [=](QuantumCircuit& q) { q.ch(a, b); }},
        {"swap", [=](QuantumCircuit& q) { q.swap(a, b); }},
        {"iswap", [=](QuantumCircuit& q) { q.iswap(a, b); }},
        {"crx", [=](QuantumCircuit& q) { q.crx(0.37, a, b); }},
        {"cry", [=](QuantumCircuit& q) { q.cry(0.53, a, b); }},
        {"crz", [=](QuantumCircuit& q) { q.crz(0.71, a, b); }},
        {"cp", [=](QuantumCircuit& q) { q.cp(0.29, a, b); }},
        {"cu", [=](QuantumCircuit& q) { q.cu(0.41, 0.23, 0.17, 0.11, a, b); }},
        {"ecr", [=](QuantumCircuit& q) { q.ecr(a, b); }},
        {"rzx", [=](QuantumCircuit& q) { q.rzx(0.33, a, b); }},
        {"rxx", [=](QuantumCircuit& q) { q.rxx(0.47, a, b); }},
        {"ryy", [=](QuantumCircuit& q) { q.ryy(0.59, a, b); }},
        {"rzz", [=](QuantumCircuit& q) { q.rzz(0.27, a, b); }},
    };
}

const std::vector<std::pair<std::string, Apply>>& three_qubit_gates() {
    static const std::vector<std::pair<std::string, Apply>> g = {
        {"ccx", [](QuantumCircuit& q) { q.ccx(0, 1, 2); }},
        {"ccz", [](QuantumCircuit& q) { q.ccz(0, 1, 2); }},
        {"cswap", [](QuantumCircuit& q) { q.cswap(0, 1, 2); }},
        {"rccx", [](QuantumCircuit& q) { q.rccx(0, 1, 2); }},
    };
    return g;
}

// CCX (controls qubits[0], qubits[1]; target qubits[2]) as a raw 8x8 in the
// project-wide matrix convention: bit 0 (LSB) of the index is qubits[0].
std::vector<Complex128> ccx_matrix_lsb() {
    std::vector<Complex128> m(64, Complex128(0.0, 0.0));
    for (size_t c = 0; c < 8; ++c) {
        const size_t r = ((c & 3) == 3) ? (c ^ 4) : c;
        m[r * 8 + c] = Complex128(1.0, 0.0);
    }
    return m;
}

Instruction raw_instruction(GT type, std::vector<int> qubits) {
    Instruction inst;
    inst.type = type;
    inst.qubits = std::move(qubits);
    return inst;
}

}  // namespace

// =============================================================================
// MPS: analytic gate builders vs the statevector oracle
// =============================================================================

TEST(R1122FillSim, MPSOneQubitGateBranchesMatchStatevector) {
    for (const auto& [name, apply] : one_qubit_gates()) {
        SCOPED_TRACE(name);
        QuantumCircuit qc(2);
        prep(qc);
        apply(qc);
        expect_sv_equal(mps_final(qc), sv_final(qc));
    }
}

TEST(R1122FillSim, MPSTwoQubitGateBranchesMatchStatevectorBothOrders) {
    for (auto [a, b] : {std::pair<int, int>{0, 1}, std::pair<int, int>{1, 0}}) {
        for (const auto& [name, apply] : two_qubit_gates(a, b)) {
            SCOPED_TRACE(name + "(" + std::to_string(a) + "," + std::to_string(b) + ")");
            QuantumCircuit qc(2);
            prep(qc);
            apply(qc);
            expect_sv_equal(mps_final(qc), sv_final(qc));
        }
    }
}

TEST(R1122FillSim, MPSThreeQubitDecompositionsMatchStatevector) {
    for (const auto& [name, apply] : three_qubit_gates()) {
        SCOPED_TRACE(name);
        QuantumCircuit qc(3);
        prep(qc);
        apply(qc);
        expect_sv_equal(mps_final(qc), sv_final(qc));
    }
}

TEST(R1122FillSim, MPSMultiQubitUnitaryFallbackRoundTrip) {
    // A 3-qubit UNITARY routes through to_statevector -> apply_unitary ->
    // mps_from_sv. Using the CCX matrix keeps an independent oracle: the same
    // circuit with the native ccx builder.
    QuantumCircuit ref(3);
    prep(ref);
    ref.ccx(0, 1, 2);

    QuantumCircuit uc(3);
    prep(uc);
    uc.unitary(ccx_matrix_lsb(), {0, 1, 2});

    auto oracle = sv_final(ref);
    expect_sv_equal(sv_final(uc), oracle);   // UNITARY == native gate on SV
    expect_sv_equal(mps_final(uc), oracle);  // exercises mps_from_sv
}

// =============================================================================
// MPS: validation and failure paths
// =============================================================================

TEST(R1122FillSim, MPSUnitaryMatrixSizeValidationThrows) {
    MPSSimulator sim;

    QuantumCircuit one(2);
    auto bad1 = raw_instruction(GT::UNITARY, {0});
    bad1.matrix = std::vector<Complex128>(2, Complex128(1.0, 0.0));  // not 4
    one.instructions.push_back(bad1);
    EXPECT_THROW(sim.run(one, 8, 0, 1), std::runtime_error);

    QuantumCircuit two(2);
    auto bad2 = raw_instruction(GT::UNITARY, {0, 1});
    bad2.matrix = std::vector<Complex128>(4, Complex128(1.0, 0.0));  // not 16
    two.instructions.push_back(bad2);
    EXPECT_THROW(sim.run(two, 8, 0, 1), std::runtime_error);
}

TEST(R1122FillSim, MPSWideRegisterMultiQubitUnitaryThrows) {
    // 26 qubits exceeds MPS_SV_MAX_QUBITS (25), so a 3-qubit UNITARY cannot
    // take the statevector fallback and must raise the descriptive error.
    QuantumCircuit qc(26);
    std::vector<Complex128> identity8(64, Complex128(0.0, 0.0));
    for (size_t i = 0; i < 8; ++i) identity8[i * 8 + i] = Complex128(1.0, 0.0);
    qc.unitary(identity8, {0, 1, 2});
    MPSSimulator sim;
    EXPECT_THROW(sim.run(qc, 4, 0, 1), std::runtime_error);
}

TEST(R1122FillSim, MPSMalformedGateAritiesThrow) {
    MPSSimulator sim;

    // A 3-operand instruction of a non-3-qubit type hits the 3q default.
    QuantumCircuit three(3);
    three.instructions.push_back(raw_instruction(GT::H, {0, 1, 2}));
    EXPECT_THROW(sim.run(three, 8, 0, 1), std::runtime_error);

    // A 4-operand instruction has no MPS path at all.
    QuantumCircuit four(4);
    four.instructions.push_back(raw_instruction(GT::CX, {0, 1, 2, 3}));
    EXPECT_THROW(sim.run(four, 8, 0, 1), std::runtime_error);
}

TEST(R1122FillSim, MPSUnboundParameterisedGateThrows) {
    QuantumCircuit qc(1);
    auto sym = raw_instruction(GT::PARAM_RX, {0});
    sym.param_names = {"theta"};
    qc.instructions.push_back(sym);
    MPSSimulator sim;
    EXPECT_THROW(sim.run(qc, 8, 0, 1), std::runtime_error);
}

TEST(R1122FillSim, MPSConditionalForwardPassWithoutMeasure) {
    // No MEASURE anywhere: shots > 0 takes the single forward pass, which must
    // still evaluate classical conditions against the all-zero register.
    MPSSimulator sim;

    QuantumCircuit met(1, 1);
    met.add_if(0, 0, GT::X, {0});  // clbit 0 == 0 holds initially -> applied
    auto counts_met = sim.run(met, 8, 256, 7).counts;
    ASSERT_EQ(counts_met.size(), 1u);
    EXPECT_EQ(counts_met.begin()->first, "1");
    EXPECT_EQ(counts_met.begin()->second, 256);

    QuantumCircuit skipped(1, 1);
    skipped.add_if(0, 1, GT::X, {0});  // requires clbit 0 == 1 -> never true
    auto counts_skip = sim.run(skipped, 8, 256, 7).counts;
    ASSERT_EQ(counts_skip.size(), 1u);
    EXPECT_EQ(counts_skip.begin()->first, "0");
    EXPECT_EQ(counts_skip.begin()->second, 256);
}

// =============================================================================
// Density matrix: analytic gate builders vs the statevector oracle
// =============================================================================

TEST(R1122FillSim, DMOneQubitGateBranchesMatchStatevector) {
    for (const auto& [name, apply] : one_qubit_gates()) {
        SCOPED_TRACE(name);
        QuantumCircuit qc(2);
        prep(qc);
        apply(qc);
        expect_dm_is_pure(dm_final(qc), sv_final(qc));
    }
}

TEST(R1122FillSim, DMTwoQubitGateBranchesMatchStatevectorBothOrders) {
    for (auto [a, b] : {std::pair<int, int>{0, 1}, std::pair<int, int>{1, 0}}) {
        for (const auto& [name, apply] : two_qubit_gates(a, b)) {
            SCOPED_TRACE(name + "(" + std::to_string(a) + "," + std::to_string(b) + ")");
            QuantumCircuit qc(2);
            prep(qc);
            apply(qc);
            expect_dm_is_pure(dm_final(qc), sv_final(qc));
        }
    }
}

TEST(R1122FillSim, DMThreeQubitGateBranchesMatchStatevector) {
    for (const auto& [name, apply] : three_qubit_gates()) {
        SCOPED_TRACE(name);
        QuantumCircuit qc(3);
        prep(qc);
        apply(qc);
        expect_dm_is_pure(dm_final(qc), sv_final(qc));
    }
}

// =============================================================================
// Density matrix: expectation values and validation
// =============================================================================

TEST(R1122FillSim, DMDenseExpectationMatchesSparseAndStatevector) {
    QuantumCircuit qc(2);
    prep(qc);
    auto dm = dm_final(qc);
    auto sv = sv_final(qc);

    // <Z_0> and <X_0> straight from the amplitudes (qubit 0 = index bit 0).
    double z0 = 0.0, x0 = 0.0;
    for (size_t i = 0; i < sv.dim; ++i) {
        const double p = sv.real_parts[i] * sv.real_parts[i] +
                         sv.imag_parts[i] * sv.imag_parts[i];
        z0 += ((i & 1) ? -p : p);
        const size_t f = i ^ 1;  // X_0 couples i <-> i^1
        x0 += sv.real_parts[i] * sv.real_parts[f] +
              sv.imag_parts[i] * sv.imag_parts[f];
    }

    // Dense operators in the amp-index basis: Z_0 = diag(+1,-1,+1,-1),
    // X_0 has ones on the bit-0 flip positions.
    std::vector<Complex128> z_dense(16, Complex128(0.0, 0.0));
    std::vector<Complex128> x_dense(16, Complex128(0.0, 0.0));
    for (size_t i = 0; i < 4; ++i) {
        z_dense[i * 4 + i] = Complex128((i & 1) ? -1.0 : 1.0, 0.0);
        x_dense[i * 4 + (i ^ 1)] = Complex128(1.0, 0.0);
    }

    EXPECT_NEAR(dm.expectation_value(z_dense), z0, kTol);
    EXPECT_NEAR(dm.expectation_value(x_dense), x0, kTol);

    // Sparse path agrees (Pauli strings LSB-first: pauli[0] acts on qubit 0).
    SparsePauliOp z_op(std::vector<PauliString>{PauliString("ZI")});
    SparsePauliOp x_op(std::vector<PauliString>{PauliString("XI")});
    EXPECT_NEAR(dm.expectation_value_sparse(z_op), z0, kTol);
    EXPECT_NEAR(dm.expectation_value_sparse(x_op), x0, kTol);
}

TEST(R1122FillSim, DMSparseExpectationQubitMismatchThrows) {
    QuantumCircuit qc(2);
    prep(qc);
    auto dm = dm_final(qc);
    SparsePauliOp one_qubit(std::vector<PauliString>{PauliString("Z")});
    EXPECT_THROW(dm.expectation_value_sparse(one_qubit), std::invalid_argument);
}

TEST(R1122FillSim, DMApplyGateMatrixSizeMismatchThrows) {
    DensityMatrix dm(1);
    std::vector<Complex128> wrong(2, Complex128(1.0, 0.0));  // needs 4 entries
    EXPECT_THROW(dm.apply_gate(wrong, {0}), std::invalid_argument);
}

TEST(R1122FillSim, DMUnboundParameterisedGateFailsGracefully) {
    QuantumCircuit qc(1);
    auto sym = raw_instruction(GT::PARAM_P, {0});
    sym.param_names = {"phi"};
    qc.instructions.push_back(sym);
    DensityMatrixSimulator sim;
    NoiseModel ideal;
    auto res = sim.run(qc, ideal, 0, 1);
    EXPECT_FALSE(res.success);
    EXPECT_NE(res.error_message.find("assign_parameters"), std::string::npos)
        << res.error_message;
}

// =============================================================================
// Statevector: direct apply_instruction paths and register guards
// =============================================================================

TEST(R1122FillSim, SVApplyInstructionMeasureCollapsesDeterministically) {
    StatevectorSimulator sim;

    // |1>: MEASURE must give outcome 1 and leave the state normalised.
    Statevector one(1);
    sim.apply_instruction(one, raw_instruction(GT::X, {0}));
    auto m = raw_instruction(GT::MEASURE, {0});
    m.clbits = {0};
    sim.apply_instruction(one, m);
    EXPECT_NEAR(one.real_parts[0] * one.real_parts[0] +
                    one.imag_parts[0] * one.imag_parts[0],
                0.0, kTol);
    EXPECT_NEAR(one.real_parts[1] * one.real_parts[1] +
                    one.imag_parts[1] * one.imag_parts[1],
                1.0, kTol);

    // |0>: MEASURE must give outcome 0 and be a no-op on the amplitudes.
    Statevector zero(1);
    sim.apply_instruction(zero, m);
    EXPECT_NEAR(zero.real_parts[0], 1.0, kTol);
    EXPECT_NEAR(zero.real_parts[1], 0.0, kTol);

    // BARRIER through the same entry point is a no-op.
    sim.apply_instruction(zero, raw_instruction(GT::BARRIER, {}));
    EXPECT_NEAR(zero.real_parts[0], 1.0, kTol);
}

TEST(R1122FillSim, SVApplyInstructionUnboundParamThrows) {
    StatevectorSimulator sim;
    Statevector sv(1);
    auto sym = raw_instruction(GT::PARAM_U, {0});
    sym.param_names = {"a", "b", "c"};
    EXPECT_THROW(sim.apply_instruction(sv, sym), std::runtime_error);
}

TEST(R1122FillSim, SVEmptyRegisterGuards) {
    QuantumCircuit qc(0, 0);
    StatevectorSimulator sim;

    auto res = sim.run(qc, 0, 1);
    EXPECT_FALSE(res.success);
    EXPECT_NE(res.error_message.find("at least 1"), std::string::npos)
        << res.error_message;

    SparsePauliOp z(std::vector<PauliString>{PauliString("Z")});
    EXPECT_THROW(sim.eval_expectation(qc, z), std::invalid_argument);
}

// =============================================================================
// Clifford: remaining dispatch branches and Pauli validation
// =============================================================================

TEST(R1122FillSim, CliffordYZResetBarrierDispatch) {
    QuantumCircuit qc(2, 2);
    qc.y(0);        // |0> -> i|1>: q0 measures 1
    qc.z(1);        // phase only
    qc.barrier();
    qc.x(1);
    qc.reset(1);    // deterministic |1> -> |0>
    qc.measure_all();

    CliffordSimulator sim;
    ASSERT_TRUE(CliffordSimulator::is_clifford(qc));
    auto res = sim.run(qc, 64, 5);
    ASSERT_EQ(res.counts.size(), 1u);
    EXPECT_EQ(res.counts.begin()->first, "01");  // q1=0, q0=1 (q0 rightmost)
    EXPECT_EQ(res.counts.begin()->second, 64);
}

TEST(R1122FillSim, CliffordExpectationPauliLengthValidation) {
    StabilizerState st(2);
    EXPECT_THROW(st.expectation_pauli("XYZ"), std::invalid_argument);
    EXPECT_THROW(st.expectation_pauli(""), std::invalid_argument);
    EXPECT_EQ(st.expectation_pauli("ZZ"), 1);  // |00> stabilised by Z tensor Z
}
