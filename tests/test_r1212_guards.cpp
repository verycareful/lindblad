// R.1.21.2 - guards owed by the R.1.21.1 defect wave that the wave did not pin.
//
// Issue #77 was found after that wave closed, so no contract was written for
// it. Issue #75 has one (R1211SerialisationLoss.QasmRoundTripMustNotSilently-
// ChangeTheOperator), but it accepts either of two outcomes; the tests here
// state which one the exporter actually implements and why the other is not
// reachable yet.
//
// The rest of the wave's RED tests state their contracts where they already
// live (test_r1211_*.cpp), and are not duplicated here.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// Equality up to a global phase, which OpenQASM 2.0 cannot express.
void expect_same_operator(const Operator& before, const Operator& after,
                          const char* what) {
    ASSERT_EQ(after.data.size(), before.data.size()) << what;

    std::size_t anchor = 0;
    double best = 0.0;
    for (std::size_t k = 0; k < before.data.size(); ++k) {
        const double mag = std::hypot(before.data[k].real, before.data[k].imag);
        if (mag > best) { best = mag; anchor = k; }
    }
    ASSERT_GT(best, 1e-12) << what;

    const std::complex<double> b(before.data[anchor].real, before.data[anchor].imag);
    const std::complex<double> a(after.data[anchor].real,  after.data[anchor].imag);
    const std::complex<double> phase = a / b;
    EXPECT_NEAR(std::abs(phase), 1.0, 1e-9) << what;

    for (std::size_t k = 0; k < before.data.size(); ++k) {
        const std::complex<double> want =
            phase * std::complex<double>(before.data[k].real, before.data[k].imag);
        EXPECT_NEAR(after.data[k].real, want.real(), 1e-9) << what << ", entry " << k;
        EXPECT_NEAR(after.data[k].imag, want.imag(), 1e-9) << what << ", entry " << k;
    }
}

std::vector<Complex128> cz_matrix() {
    std::vector<Complex128> m(16, Complex128(0.0, 0.0));
    m[0]  = Complex128(1.0, 0.0);
    m[5]  = Complex128(1.0, 0.0);
    m[10] = Complex128(1.0, 0.0);
    m[15] = Complex128(-1.0, 0.0);
    return m;
}

std::vector<Complex128> identity_matrix(std::size_t rows) {
    std::vector<Complex128> m(rows * rows, Complex128(0.0, 0.0));
    for (std::size_t i = 0; i < rows; ++i) m[i * rows + i] = Complex128(1.0, 0.0);
    return m;
}

} // namespace

// =============================================================================
// Issue #77 - a non-positive max_bond_dim is a caller argument, not an SVD fault
// =============================================================================

TEST(R1212BondDim, MpsStateRejectsANonPositiveBondDimension) {
    // Zero retains no singular values. Unchecked it reaches svd_truncate as
    // k == 0, enters the rescue branch written for a corrupt spectrum, and the
    // verify step then rejects both routes and blames the SVD backend.
    for (const int chi : {0, -1, -64}) {
        EXPECT_THROW(MPSState(4, chi), std::invalid_argument)
            << "issue #77: max_bond_dim " << chi << " was accepted";
    }
    EXPECT_NO_THROW(MPSState(4, 1));
}

TEST(R1212BondDim, MpsRunRejectsANonPositiveBondDimension) {
    // The argument order differs from StatevectorSimulator::run(circuit, shots,
    // seed), so run(qc, 0, 0) meaning shots is a live way to arrive here.
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 1);

    MPSSimulator sim;
    EXPECT_THROW(sim.run(qc, /*max_bond_dim=*/0, /*shots=*/0, /*seed=*/0),
                 std::invalid_argument)
        << "issue #77: max_bond_dim 0 was accepted";
}

TEST(R1212BondDim, TheMessageNamesTheArgumentRatherThanTheSvd) {
    // The whole severity of #77 is that the failure was loud about the wrong
    // thing: a maintainer reading it would start by suspecting Eigen.
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 1);
    MPSSimulator sim;
    try {
        sim.run(qc, 0, 0, 0);
        FAIL() << "issue #77: max_bond_dim 0 was accepted";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("max_bond_dim"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("SVD"), std::string::npos)
            << "issue #77: the message still blames the SVD backend: " << msg;
    }
}

TEST(R1212BondDim, AValidBondDimensionStillRuns) {
    // The guard must reject the out-of-range argument and nothing else.
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 1);
    MPSSimulator sim;
    EXPECT_NO_THROW((void)sim.run(qc, 64, 0, 0));
    EXPECT_NO_THROW((void)sim.run(qc, 1, 0, 0));
}

// =============================================================================
// Issue #75 - a multi-qubit UNITARY is refused rather than silently replaced
// =============================================================================
// OpenQASM 2.0 has no literal-matrix syntax, so the operand cannot be written
// as itself, and no exact lowering for one exists in the tree. Refusing is the
// only remaining honest answer: the defect was that the exporter took the
// third option and wrote a `cx` placeholder whatever the matrix held.

TEST(R1212QasmUnitary, MultiQubitUnitariesAreRefusedUnderBothSettings) {
    // decompose_unrepresentable lowers MCX / MCP / PERMUTATION. It does not
    // reach a raw matrix, so it must not change the outcome here.
    QasmExportOptions opts;
    opts.decompose_unrepresentable = true;

    QuantumCircuit two(2);
    two.unitary(cz_matrix(), {0, 1}, "my_cz");
    EXPECT_THROW((void)two.to_qasm2(), std::runtime_error)
        << "issue #75: a 2-qubit UNITARY was exported";
    EXPECT_THROW((void)two.to_qasm2(opts), std::runtime_error)
        << "issue #75: the opt-in flag exported an operand it cannot lower";

    QuantumCircuit three(3);
    three.unitary(identity_matrix(8), {0, 1, 2}, "wide");
    EXPECT_THROW((void)three.to_qasm2(), std::runtime_error);
    EXPECT_THROW((void)three.to_qasm2(opts), std::runtime_error);
}

TEST(R1212QasmUnitary, TheRefusalPointsAtSomethingThatActuallyWorks) {
    // Naming decompose_unrepresentable here would send the caller in a circle,
    // since setting it changes nothing for this operand.
    QuantumCircuit qc(2);
    qc.unitary(cz_matrix(), {0, 1}, "my_cz");
    try {
        (void)qc.to_qasm2();
        FAIL() << "issue #75: a 2-qubit UNITARY was exported by default";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("my_cz"), std::string::npos)
            << "the message must name the operand: " << msg;
        EXPECT_NE(msg.find("to_json"), std::string::npos)
            << "the message must name a route that works: " << msg;
        EXPECT_EQ(msg.find("decompose_unrepresentable"), std::string::npos)
            << "that option cannot lower this operand: " << msg;
    }
}

TEST(R1212QasmUnitary, NoPlaceholderGateDefinitionIsEmittedAnywhere) {
    // The placeholder was emitted in a pre-pass over the whole instruction
    // list, so a circuit that exports successfully must carry no trace of it.
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.4, 1);
    const std::string qasm = qc.to_qasm2();

    EXPECT_EQ(qasm.find("gate "), std::string::npos)
        << "no custom gate definition should be emitted: " << qasm;
    EXPECT_EQ(qasm.find("cx q0,q1"), std::string::npos)
        << "the placeholder body must be gone: " << qasm;
}

TEST(R1212QasmUnitary, OneQubitUnitariesAreUnaffected) {
    // The Euler path is correct and must stay reachable. Reaching it now takes
    // the same consent every other width takes: the operand is representable
    // only by being restructured, and that is the caller's call.
    constexpr double h = INV_SQRT2;
    QuantumCircuit qc(1);
    qc.unitary({Complex128(h, 0.0), Complex128(h, 0.0),
                Complex128(h, 0.0), Complex128(-h, 0.0)}, {0}, "euler_me");

    EXPECT_THROW((void)qc.to_qasm2(), std::runtime_error)
        << "width does not exempt an operand from the export door";

    QasmExportOptions opts;
    opts.unitary_lowering = UnitaryLowering::Always;

    std::string qasm;
    ASSERT_NO_THROW(qasm = qc.to_qasm2(opts));
    const Operator before = Operator::from_circuit(qc);
    const Operator after =
        Operator::from_circuit(QuantumCircuit::from_qasm2(qasm));
    expect_same_operator(before, after, "1q euler");
}
