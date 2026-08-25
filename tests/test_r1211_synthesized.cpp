// R.1.21.1 test wave - the library's own matrices under the default policy.
//
// Turning a default-on check across every backend creates one failure mode that
// no amount of per-primitive testing finds: the library rejecting its own
// arithmetic. Wherever the library builds a matrix rather than receiving one,
// that matrix has to clear the same 1e-12 bar a caller's would, or the primitive
// stops working for reasons the caller cannot see or fix.
//
// R.1.21.0 found exactly one site where it does not. QPE raises its controlled
// unitary by repeated squaring, so the operator handed to the last evaluation
// qubit is U^(2^(k-1)) with k-1 rounds of accumulated rounding in it. Its
// distance from exact unitarity is arithmetic drift rather than a caller's
// declaration, so that instruction carries Ignore. The first test below is the
// regression pin for that: at the default policy and without the opt-out, QPE
// throws before it reaches its last evaluation qubit.
//
// Everything else here is the survey that establishes the opt-out is needed in
// that one place and nowhere else.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;
using r1211::WarningProbe;

namespace {

// A one-qubit unitary whose |0> eigenvalue is exp(2*pi*i*phi).
QuantumCircuit phase_unitary(double phi) {
    QuantumCircuit u(1);
    u.x(0).p(2.0 * PI * phi, 0).x(0);
    return u;
}

} // namespace

// =============================================================================
// QPE - the one library-synthesized matrix that needs the opt-out
// =============================================================================

TEST(R1211Synthesized, QpeRunsAtEveryEvaluationWidth) {
    // REGRESSION PIN. Each added evaluation qubit doubles the exponent of the
    // controlled operator and adds one squaring's worth of rounding. Growing
    // the width is therefore the axis along which the drift grows, and any
    // width that throws means the opt-out was lost.
    for (int eval_qubits : {1, 2, 3, 4, 6, 8, 10}) {
        WarningProbe probe;
        EXPECT_NO_THROW({
            QuantumCircuit qc = QPE::build_circuit(phase_unitary(0.25), eval_qubits);
            qc.validate_physical();
        }) << "QPE at " << eval_qubits
           << " evaluation qubits was rejected by its own pre-flight";
        EXPECT_EQ(probe.count(), 0u)
            << "QPE at " << eval_qubits << " evaluation qubits warned about "
               "its own arithmetic";
    }
}

TEST(R1211Synthesized, QpeControlledOperatorCarriesTheOptOut) {
    // Stated as a property of the circuit rather than only as an absence of
    // throwing, so the reason it runs is pinned along with the fact that it
    // does. If a later release computes the powers without accumulating drift,
    // this is the test that should be revisited.
    const QuantumCircuit qc = QPE::build_circuit(phase_unitary(1.0 / 3.0), 8);

    std::size_t opted_out = 0;
    std::size_t unitaries = 0;
    for (const auto& inst : qc.instructions) {
        if (inst.type != Instruction::GateType::UNITARY) continue;
        ++unitaries;
        if (inst.validation.policy == Validation::Ignore) ++opted_out;
    }
    ASSERT_GT(unitaries, 0u) << "QPE built no dense unitary at all";
    EXPECT_EQ(opted_out, unitaries)
        << opted_out << " of " << unitaries
        << " QPE unitaries carry Ignore; the controlled powers are built by "
           "repeated squaring and their drift is the library's own, so all of "
           "them must";
}

TEST(R1211Synthesized, QpeStillProducesTheRightAnswer) {
    // The opt-out must not have been bought by breaking the algorithm.
    EXPECT_NEAR(QPE::estimate_phase(phase_unitary(5.0 / 16.0), 4, 4000, 1),
                5.0 / 16.0, 1e-9);
}

// =============================================================================
// The rest of the algorithm catalogue
// =============================================================================

TEST(R1211Synthesized, GroverRunsUnderTheDefaultPolicy) {
    // The oracle here is a caller-supplied dense unitary, so this exercises the
    // ingress check as well as everything Grover builds around it. A diagonal
    // of plus and minus ones is exactly unitary, with no rounding at all.
    WarningProbe probe;
    for (int n = 2; n <= 4; ++n) {
        const std::size_t dim = std::size_t(1) << n;
        const std::size_t target = dim - 2;   // asymmetric, so a convention bug shows
        std::vector<Complex128> diag(dim * dim, Complex128(0.0, 0.0));
        for (std::size_t i = 0; i < dim; ++i) diag[i * dim + i] = Complex128(1.0, 0.0);
        diag[target * dim + target] = Complex128(-1.0, 0.0);

        std::vector<int> qubits(static_cast<std::size_t>(n));
        for (int q = 0; q < n; ++q) qubits[static_cast<std::size_t>(q)] = q;

        QuantumCircuit oracle(n);
        oracle.unitary(diag, qubits);

        EXPECT_NO_THROW(Grover::build_circuit(oracle).validate_physical())
            << "Grover at n = " << n << " was rejected by its own pre-flight";
        EXPECT_NO_THROW(Grover::search(oracle, -1, 200, 1))
            << "Grover at n = " << n << " failed to run";
    }
    EXPECT_EQ(probe.count(), 0u);
}

TEST(R1211Synthesized, QftRunsUnderTheDefaultPolicy) {
    WarningProbe probe;
    for (int n = 2; n <= 6; ++n) {
        const QuantumCircuit forward =
            QFT::build_circuit(n, QFT::Options(true, 0, false));
        EXPECT_NO_THROW(forward.validate_physical()) << "QFT at n = " << n;

        const QuantumCircuit inverted =
            QFT::build_circuit(n, QFT::Options(true, 0, true));
        EXPECT_NO_THROW(inverted.validate_physical())
            << "inverse QFT at n = " << n;

        EXPECT_NO_THROW(forward.inverse().validate_physical())
            << "the conjugate transpose of QFT at n = " << n
            << " must be as unitary as the original";
    }
    EXPECT_EQ(probe.count(), 0u);
}

TEST(R1211Synthesized, StandardGateCircuitsNeverWarn) {
    // Every named gate the library builds itself must clear 1e-12. A parameter
    // sweep is what catches a rotation whose matrix is assembled from
    // trigonometry rather than from exact constants.
    WarningProbe probe;
    QuantumCircuit qc(3);
    for (int k = 0; k < 32; ++k) {
        const double theta = 2.0 * PI * static_cast<double>(k) / 32.0;
        qc.rx(theta, 0).ry(theta, 1).rz(theta, 2);
        qc.p(theta, 0).u(theta, theta / 2.0, theta / 3.0, 1);
        qc.crx(theta, 0, 1).cry(theta, 1, 2).crz(theta, 0, 2);
    }
    qc.h(0).s(1).sdg(1).t(2).tdg(2).x(0).y(1).z(2).sx(0).cx(0, 1).cz(1, 2).swap(0, 2);

    EXPECT_NO_THROW(qc.validate_physical());
    StatevectorSimulator sim;
    EXPECT_NO_THROW(sim.run(qc, 0, 0));
    EXPECT_EQ(probe.count(), 0u)
        << "a circuit of library-built gates produced a physical-validity "
           "warning; the first line was: "
        << (probe.count() ? probe.lines()[0] : std::string("(none)"));
}

// =============================================================================
// Basis-column extraction
// =============================================================================

TEST(R1211Synthesized, OperatorFromCircuitValidatesOnceThenExtracts) {
    // from_circuit recovers a matrix by simulating 2^n basis columns. Checking
    // per column would measure the same circuit 2^n times, so it validates up
    // front and applies the columns under Ignore.
    WarningProbe probe;
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).t(2).cz(0, 2);

    Operator op = Operator::from_circuit(qc);
    EXPECT_TRUE(op.is_unitary())
        << "the extracted operator must be unitary; it was built from a "
           "circuit of unitary gates";
    EXPECT_EQ(probe.count(), 0u)
        << "basis-column extraction warned about the library's own arithmetic";
}

TEST(R1211Synthesized, OperatorFromCircuitStillRejectsABadCircuit) {
    // The up-front validation is what makes the per-column Ignore safe. If it
    // were dropped, a non-unitary circuit would be extracted silently.
    QuantumCircuit qc(2);
    Instruction inst;
    inst.type = Instruction::GateType::UNITARY;
    inst.qubits = {0};
    inst.matrix = std::vector<Complex128>{
        Complex128(1.5, 0.0), Complex128(0.0, 0.0),
        Complex128(0.0, 0.0), Complex128(1.5, 0.0)};
    qc.instructions.push_back(inst);

    EXPECT_THROW(Operator::from_circuit(qc), std::invalid_argument)
        << "the one validation from_circuit performs is the only one there is";
}

// =============================================================================
// Fusion
// =============================================================================

TEST(R1211Synthesized, FusedBlocksDoNotSelfReject) {
    // The fusion pre-pass multiplies member gates into one dense block, which
    // accumulates rounding across every gate it absorbed. That block is the
    // library's own arithmetic and carries Ignore; without it a long fused
    // window would eventually drift past 1e-12 and reject itself.
    WarningProbe probe;

    QuantumCircuit qc(12);
    for (int layer = 0; layer < 12; ++layer)
        for (int q = 0; q + 1 < 12; ++q) {
            qc.rx(0.3 + 0.01 * layer, q);
            qc.cx(q, q + 1);
        }

    StatevectorSimulator::Options opts;
    opts.fusion_enable = true;
    opts.fusion_threshold = 2;    // force engagement rather than waiting for the
                                  // hardware-derived default
    opts.fusion_max_qubit = 5;
    StatevectorSimulator sim(opts);

    EXPECT_NO_THROW(sim.run(qc, 0, 0));
    EXPECT_EQ(probe.count(), 0u)
        << "fusion produced a block that failed the unitarity check";
}

TEST(R1211Synthesized, FusionAndNoFusionAgreeUnderTheDefaultPolicy) {
    // The check must not change results, only reject bad input. Running the
    // same circuit both ways under the strictest policy is the statement that
    // it does not.
    QuantumCircuit qc(8);
    for (int layer = 0; layer < 6; ++layer)
        for (int q = 0; q + 1 < 8; ++q) {
            qc.ry(0.2 + 0.05 * layer, q);
            qc.cx(q, q + 1);
        }

    StatevectorSimulator::Options fused_opts;
    fused_opts.fusion_enable = true;
    fused_opts.fusion_threshold = 2;
    StatevectorSimulator fused(fused_opts);

    StatevectorSimulator::Options plain_opts;
    plain_opts.fusion_enable = false;
    StatevectorSimulator plain(plain_opts);

    const auto a = fused.run(qc, 0, 0).final_state.amplitudes();
    const auto b = plain.run(qc, 0, 0).final_state.amplitudes();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        EXPECT_NEAR(a[k].real, b[k].real, 1e-12) << "amplitude " << k;
        EXPECT_NEAR(a[k].imag, b[k].imag, 1e-12) << "amplitude " << k;
    }
}
