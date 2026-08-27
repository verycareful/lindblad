// R.1.22.1 test wave - the QASM UNITARY export contract.
//
// R.1.22.0 gave a 2-qubit UNITARY an exact lowering and then gave the two
// formats different defaults, because the formats differ in what they can
// represent. QASM 3 has a first-class `gphase` and can restore the operand
// exactly, so it lowers by default. QASM 2 has no representation for a global
// phase at all, so it refuses by default and requires two separate consents:
// one to restructure the circuit, one to accept the loss.
//
// Two consents rather than one, because they are different decisions. A caller
// may be perfectly happy to have a matrix rewritten as gates and still need to
// know that a phase went missing.
//
// The suite also covers the seam under both exporters. tqd::lower_2q_unitary
// lives in src/, not include/, so it is reached here by relative path: it is an
// internal contract between the transpiler and the exporters, and the fields it
// promises to carry (classical condition, validation policy) are not observable
// in the emitted text, since neither exporter writes conditions at all.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/validation.hpp"

#include "../src/transpiler/two_qubit_decompose.hpp"

#include <cmath>
#include <functional>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

::testing::AssertionResult same_operator_up_to_phase(const QuantumCircuit& a,
                                                     const QuantumCircuit& b,
                                                     double tol = 1e-7) {
    const auto ma = Operator::from_circuit(a).data;
    const auto mb = Operator::from_circuit(b).data;
    if (ma.size() != mb.size())
        return ::testing::AssertionFailure()
               << "operator size " << ma.size() << " vs " << mb.size();

    Complex128 phase(1, 0);
    for (size_t i = 0; i < ma.size(); ++i) {
        if (mb[i].norm_sq() > 1e-12 && ma[i].norm_sq() > 1e-12) {
            phase = ma[i] * Complex128(mb[i].real, -mb[i].imag) *
                    (1.0 / mb[i].norm_sq());
            break;
        }
    }
    for (size_t i = 0; i < ma.size(); ++i) {
        const Complex128 bp = mb[i] * phase;
        if (std::abs(ma[i].real - bp.real) > tol ||
            std::abs(ma[i].imag - bp.imag) > tol)
            return ::testing::AssertionFailure()
                   << "entry " << i << ": expected (" << ma[i].real << ", "
                   << ma[i].imag << "), got (" << bp.real << ", " << bp.imag
                   << ")";
    }
    return ::testing::AssertionSuccess();
}

std::vector<Complex128> scaled(std::vector<Complex128> m, double alpha) {
    const Complex128 s(std::cos(alpha), std::sin(alpha));
    for (auto& e : m) e = e * s;
    return m;
}

// U(theta, phi, lambda) reproduces H exactly, with no leftover phase, so this
// is the width-1 operand for which lowering is LOSSLESS. Chosen rather than
// asserted: a fixture whose losslessness had to be measured would be measuring
// the thing under test.
std::vector<Complex128> h_matrix() {
    QuantumCircuit qc(1);
    qc.h(0);
    return Operator::from_circuit(qc).data;
}

std::vector<Complex128> generic_2q() {
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);
    return Operator::from_circuit(src).data;
}

// A 2-qubit operand whose lowering carries NO global phase.
//
// Built rather than corrected, because the obvious correction does not work.
// Measuring a generic operand's phase and dividing it out does NOT produce a
// phase-free operand: u(theta, phi, lambda) has cos(theta/2) in its top-left
// entry, which is real and non-negative, so the set of matrices the emitted
// form can represent EXACTLY is not closed under a global phase rotation.
// Rotating an operand moves it off that set and it simply needs a different
// phase to get back.
//
// So the fixture is assembled from factors that lie on the set by construction.
// H is u(pi/2, 0, pi) and S is u(0, 0, pi/2), both exactly, so their tensor
// product is exactly reproducible by two emitted u gates and nothing is left
// over. The operand is local, which is fine here: these tests are about the
// export doors at width 2, and the entangling path is covered by the round-trip
// and seam tests that do not care whether a phase is present.
std::vector<Complex128> lossless_2q() {
    QuantumCircuit qc(2);
    qc.h(0).s(1);
    return Operator::from_circuit(qc).data;
}

QuantumCircuit with_unitary(int n, const std::vector<Complex128>& m,
                            const std::vector<int>& qubits) {
    QuantumCircuit qc(n);
    qc.unitary(m, qubits);
    return qc;
}

QasmExportOptions door_open_accept_loss() {
    QasmExportOptions o;
    o.unitary_lowering = UnitaryLowering::Always;
    o.accept_global_phase_loss = true;
    return o;
}

QasmExportOptions door_open_refuse_loss() {
    QasmExportOptions o;
    o.unitary_lowering = UnitaryLowering::Always;
    return o;
}

QasmExportOptions door_shut_accept_loss() {
    QasmExportOptions o;
    o.accept_global_phase_loss = true;
    return o;
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

std::string what_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

// =============================================================================
// QASM 3: lossless by default
// =============================================================================

TEST(R1221Qasm3Unitary, OneAndTwoQubitOperandsRoundTrip) {
    struct Case { const char* name; int n; std::vector<int> q; std::vector<Complex128> m; };
    const Case cases[] = {
        {"1q lossless", 1, {0},    h_matrix()},
        {"1q with phase", 1, {0},  scaled(h_matrix(), PI / 5.0)},
        {"2q lossless", 2, {0, 1}, lossless_2q()},
        {"2q with phase", 2, {0, 1}, scaled(lossless_2q(), PI / 5.0)},
    };
    for (const Case& cs : cases) {
        SCOPED_TRACE(cs.name);
        const QuantumCircuit in = with_unitary(cs.n, cs.m, cs.q);
        std::string text;
        ASSERT_NO_THROW(text = in.to_qasm3());
        QuantumCircuit back(cs.n);
        ASSERT_NO_THROW(back = QuantumCircuit::from_qasm3(text))
            << "emitted QASM 3:\n" << text;
        EXPECT_TRUE(same_operator_up_to_phase(in, back))
            << "emitted QASM 3:\n" << text;
    }
}

// The phase is what QASM 3 can carry and QASM 2 cannot, so its presence in the
// text is the whole reason the two formats have different defaults.
TEST(R1221Qasm3Unitary, GphaseAppearsExactlyWhenTheOperandCarriesOne) {
    const QuantumCircuit lossy =
        with_unitary(2, scaled(lossless_2q(), PI / 5.0), {0, 1});
    EXPECT_TRUE(contains(lossy.to_qasm3(), "gphase"))
        << lossy.to_qasm3();

    const QuantumCircuit clean = with_unitary(2, lossless_2q(), {0, 1});
    EXPECT_FALSE(contains(clean.to_qasm3(), "gphase"))
        << "a lowering that drops nothing must not emit a phase of zero\n"
        << clean.to_qasm3();
}

TEST(R1221Qasm3Unitary, ThreeQubitOperandsRefuseWithoutNamingAFlag) {
    QuantumCircuit src(3);
    src.h(0).cx(0, 1).cx(1, 2).t(2);
    const QuantumCircuit in = with_unitary(3, Operator::from_circuit(src).data,
                                           {0, 1, 2});

    const std::string msg = what_of([&] { (void)in.to_qasm3(); });
    ASSERT_FALSE(msg.empty()) << "a 3-qubit UNITARY has no exact lowering";
    EXPECT_TRUE(contains(msg, "no exact lowering above two qubits")) << msg;

    // Naming a setting here would send the caller to try something that cannot
    // work at any setting, which is worse than refusing plainly.
    EXPECT_FALSE(contains(msg, "unitary_lowering")) << msg;
    EXPECT_FALSE(contains(msg, "accept_global_phase_loss")) << msg;
}

TEST(R1221Qasm3Unitary, NeverRefusesAndTheOtherTwoLower) {
    const QuantumCircuit in = with_unitary(2, lossless_2q(), {0, 1});

    QasmExportOptions never;
    never.unitary_lowering = UnitaryLowering::Never;
    const std::string msg = what_of([&] { (void)in.to_qasm3(never); });
    ASSERT_FALSE(msg.empty());
    EXPECT_TRUE(contains(msg, "Never")) << msg;

    QasmExportOptions always;
    always.unitary_lowering = UnitaryLowering::Always;
    EXPECT_NO_THROW((void)in.to_qasm3(always));
    EXPECT_NO_THROW((void)in.to_qasm3());  // FormatDefault
}

// A circuit the options cannot apply to must not notice they were set.
TEST(R1221Qasm3Unitary, ACircuitWithoutAUnitaryIsUnaffectedByEverySetting) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rz(0.3, 1).measure_all();

    QasmExportOptions never;
    never.unitary_lowering = UnitaryLowering::Never;

    const std::string base = qc.to_qasm3();
    EXPECT_EQ(qc.to_qasm3(never), base);
    EXPECT_EQ(qc.to_qasm3(door_open_accept_loss()), base);
    EXPECT_EQ(qc.to_qasm2(), qc.to_qasm2());
}

// =============================================================================
// QASM 2: two consents, and the same rule at both widths
// =============================================================================

// The point of the R.1.22.0 change is that width stopped mattering: a 1-qubit
// UNITARY meets exactly the same doors as a 2-qubit one. Every case below is
// therefore asserted at both widths from one table.
namespace {
struct Width { const char* name; int n; std::vector<int> q;
               std::vector<Complex128> clean, lossy; };
Width widths(int which) {
    if (which == 0)
        return {"width 1", 1, {0}, h_matrix(), scaled(h_matrix(), PI / 5.0)};
    return {"width 2", 2, {0, 1}, lossless_2q(),
            scaled(lossless_2q(), PI / 5.0)};
}
}  // namespace

TEST(R1221Qasm2Doors, DoorShutRefusesAndNamesBothSettings) {
    for (int w = 0; w < 2; ++w) {
        const Width x = widths(w);
        SCOPED_TRACE(x.name);
        const QuantumCircuit in = with_unitary(x.n, x.clean, x.q);

        const std::string msg = what_of([&] { (void)in.to_qasm2(); });
        ASSERT_FALSE(msg.empty()) << "the door is shut by default";
        EXPECT_TRUE(contains(msg, "unitary_lowering")) << msg;
        EXPECT_TRUE(contains(msg, "accept_global_phase_loss")) << msg;
    }
}

// Accepting the loss does not open the door. They are independent consents and
// the door is the outer one, so setting only the inner one changes nothing.
TEST(R1221Qasm2Doors, AcceptingTheLossDoesNotOpenTheDoor) {
    for (int w = 0; w < 2; ++w) {
        const Width x = widths(w);
        SCOPED_TRACE(x.name);
        const QuantumCircuit in = with_unitary(x.n, x.lossy, x.q);
        const std::string msg =
            what_of([&] { (void)in.to_qasm2(door_shut_accept_loss()); });
        ASSERT_FALSE(msg.empty()) << "the door must still refuse";
        EXPECT_TRUE(contains(msg, "unitary_lowering")) << msg;
    }
}

TEST(R1221Qasm2Doors, DoorOpenAndNothingLostLowersSilently) {
    for (int w = 0; w < 2; ++w) {
        const Width x = widths(w);
        SCOPED_TRACE(x.name);
        r1211::WarningProbe probe;
        const QuantumCircuit in = with_unitary(x.n, x.clean, x.q);

        std::string text;
        ASSERT_NO_THROW(text = in.to_qasm2(door_open_refuse_loss()));
        EXPECT_FALSE(contains(text, "global phase"))
            << "nothing was dropped, so nothing should be recorded\n" << text;
        EXPECT_EQ(probe.count(), 0u) << "and nothing should be warned about";
    }
}

TEST(R1221Qasm2Doors, DoorOpenAndLossRefusedNamesTheDroppedAngle) {
    for (int w = 0; w < 2; ++w) {
        const Width x = widths(w);
        SCOPED_TRACE(x.name);
        const QuantumCircuit in = with_unitary(x.n, x.lossy, x.q);
        const std::string msg =
            what_of([&] { (void)in.to_qasm2(door_open_refuse_loss()); });
        ASSERT_FALSE(msg.empty()) << "the loss needs its own consent";
        EXPECT_TRUE(contains(msg, "global phase")) << msg;
        EXPECT_TRUE(contains(msg, "accept_global_phase_loss")) << msg;
    }
}

// ONE warning per lossy UNITARY OPERAND, not one per gate the lowering
// produced. The distinction matters because the lowering emits up to seven
// instructions, so a per-gate rule would make the warning count track an
// implementation detail: improve the decomposition and the count moves while
// nothing about the loss changes.
//
// Emission is per operand. DELIVERY is per distinct message, because the
// warning channel deduplicates on text. The two come apart as soon as a circuit
// holds more than one lossy operand, which the test below pins.
TEST(R1221Qasm2Doors, DoorOpenAndLossAcceptedWarnsOnceAndRecordsIt) {
    for (int w = 0; w < 2; ++w) {
        const Width x = widths(w);
        SCOPED_TRACE(x.name);
        r1211::WarningProbe probe;
        const QuantumCircuit in = with_unitary(x.n, x.lossy, x.q);

        std::string text;
        ASSERT_NO_THROW(text = in.to_qasm2(door_open_accept_loss()));
        EXPECT_TRUE(contains(text, "// global phase:"))
            << "the loss must survive in the file itself\n" << text;
        EXPECT_EQ(probe.count(), 1u)
            << "one operand dropped one phase, so one warning";
        EXPECT_TRUE(probe.any_contains("dropped a global phase"));
    }
}

TEST(R1221Qasm2Doors, ThreeQubitOperandsRefuseUnderEveryFlagCombination) {
    QuantumCircuit src(3);
    src.h(0).cx(0, 1).cx(1, 2).t(2);
    const QuantumCircuit in = with_unitary(3, Operator::from_circuit(src).data,
                                           {0, 1, 2});

    QasmExportOptions never;
    never.unitary_lowering = UnitaryLowering::Never;
    const QasmExportOptions combos[] = {QasmExportOptions{},
                                        door_shut_accept_loss(),
                                        door_open_refuse_loss(),
                                        door_open_accept_loss(), never};
    for (size_t i = 0; i < sizeof(combos) / sizeof(combos[0]); ++i) {
        SCOPED_TRACE("combination " + std::to_string(i));
        const std::string msg = what_of([&] { (void)in.to_qasm2(combos[i]); });
        ASSERT_FALSE(msg.empty()) << "no setting can make this representable";
        EXPECT_TRUE(contains(msg, "no exact lowering above two qubits")) << msg;
    }
}

// Exporting text that nothing can read back is the #79 failure mode exactly,
// and #84 showed the project had shipped it in QASM 3 for years. So the
// emitted QASM 2 is fed back through its own parser.
TEST(R1221Qasm2Doors, EmittedTextReparsesToTheSameOperator) {
    for (int w = 0; w < 2; ++w) {
        const Width x = widths(w);
        SCOPED_TRACE(x.name);
        const QuantumCircuit in = with_unitary(x.n, x.clean, x.q);

        std::string text;
        ASSERT_NO_THROW(text = in.to_qasm2(door_open_refuse_loss()));
        QuantumCircuit back(x.n);
        ASSERT_NO_THROW(back = QuantumCircuit::from_qasm2(text))
            << "emitted QASM 2:\n" << text;
        EXPECT_TRUE(same_operator_up_to_phase(in, back))
            << "emitted QASM 2:\n" << text;
    }
}

// =============================================================================
// The seam, which neither exporter's text can show
// =============================================================================

// The header promises the source instruction's classical condition and
// validation policy are carried onto EVERY emitted instruction. Neither
// exporter writes conditions into its text, so this contract is invisible from
// outside and is asserted against the seam directly.
TEST(R1221LoweringSeam, EveryEmittedInstructionCarriesConditionAndPolicy) {
    Instruction inst;
    inst.type = Instruction::GateType::UNITARY;
    inst.qubits = {0, 1};
    inst.matrix = generic_2q();
    inst.condition_clbit = 3;
    inst.condition_value = 1;
    inst.validation = ValidationOptions{Validation::Warn, 1e-9};

    const auto low = tqd::lower_2q_unitary(inst);
    ASSERT_TRUE(low.has_value()) << "a generic 2-qubit operand must lower";
    ASSERT_FALSE(low->instructions.empty());

    for (size_t i = 0; i < low->instructions.size(); ++i) {
        const Instruction& g = low->instructions[i];
        SCOPED_TRACE("emitted instruction " + std::to_string(i));
        EXPECT_EQ(g.condition_clbit, 3);
        EXPECT_EQ(g.condition_value, 1);
        EXPECT_EQ(g.validation.policy, Validation::Warn);
        EXPECT_DOUBLE_EQ(g.validation.atol, 1e-9);
    }
}

TEST(R1221LoweringSeam, OperandsThatAreNotTwoQubitsReturnNullopt) {
    for (int k : {1, 3}) {
        SCOPED_TRACE("k = " + std::to_string(k));
        QuantumCircuit src(k);
        src.h(0);
        Instruction inst;
        inst.type = Instruction::GateType::UNITARY;
        inst.qubits = (k == 1) ? std::vector<int>{0} : std::vector<int>{0, 1, 2};
        inst.matrix = Operator::from_circuit(src).data;
        EXPECT_FALSE(tqd::lower_2q_unitary(inst).has_value())
            << "this module represents 2-qubit operands only";
    }
}

// The emitted sequence times exp(i * global_phase) IS the operand. That is the
// whole reason the phase comes back separately rather than folded in, so it is
// asserted rather than assumed by the exporters that rely on it.
TEST(R1221LoweringSeam, TheOperandIsTheEmittedSequenceTimesTheReportedPhase) {
    Instruction inst;
    inst.type = Instruction::GateType::UNITARY;
    inst.qubits = {0, 1};
    inst.matrix = scaled(generic_2q(), PI / 5.0);

    const auto low = tqd::lower_2q_unitary(inst);
    ASSERT_TRUE(low.has_value());

    QuantumCircuit rebuilt(2);
    rebuilt.instructions = low->instructions;
    for (auto& g : rebuilt.instructions) {
        g.condition_clbit = -1;
        g.condition_value = 0;
    }

    const auto want = inst.matrix;
    const auto got = scaled(Operator::from_circuit(rebuilt).data,
                            low->global_phase);
    ASSERT_EQ(want.size(), got.size());
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(want[i].real, got[i].real, 1e-9) << "real at " << i;
        EXPECT_NEAR(want[i].imag, got[i].imag, 1e-9) << "imag at " << i;
    }
}

// =============================================================================
// gphase in the QASM 3 parser (#84)
// =============================================================================

namespace {

QuantumCircuit parse3(int n, const std::string& body) {
    return QuantumCircuit::from_qasm3(
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[" +
        std::to_string(n) + "] q;\n" + body);
}

// A controlled gphase puts the phase on the all-controls-1 slice, which for k
// symmetric controls is the last basis state. The OPERATOR is asserted rather
// than the instruction type: the relative phase is the observable thing, and a
// different spelling that produced the same operator would be equally correct.
void expect_phase_on_all_ones(const QuantumCircuit& qc, int k, double angle) {
    const auto m = Operator::from_circuit(qc).data;
    const size_t dim = size_t(1) << k;
    ASSERT_EQ(m.size(), dim * dim);
    for (size_t r = 0; r < dim; ++r) {
        for (size_t c = 0; c < dim; ++c) {
            const bool on_diagonal = (r == c);
            const bool all_ones = on_diagonal && (r == dim - 1);
            const double want_re =
                all_ones ? std::cos(angle) : (on_diagonal ? 1.0 : 0.0);
            const double want_im = all_ones ? std::sin(angle) : 0.0;
            EXPECT_NEAR(m[r * dim + c].real, want_re, 1e-12)
                << "real at (" << r << ", " << c << ")";
            EXPECT_NEAR(m[r * dim + c].imag, want_im, 1e-12)
                << "imag at (" << r << ", " << c << ")";
        }
    }
}

}  // namespace

// QuantumCircuit models circuits up to a global phase, so there is nowhere to
// put an uncontrolled one. Discarding it is correct; refusing the file would
// reject valid QASM 3 that any other tool accepts.
TEST(R1221Gphase, UncontrolledGphaseIsParsedAndDiscarded) {
    const QuantumCircuit with = parse3(1, "gphase(0.5);\nh q[0];\n");
    const QuantumCircuit without = parse3(1, "h q[0];\n");

    const auto a = Operator::from_circuit(with).data;
    const auto b = Operator::from_circuit(without).data;
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, 1e-12) << "real at " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, 1e-12) << "imag at " << i;
    }
}

// Under a control the phase stops being global: it applies on one branch only,
// which is a relative phase between branches and is observable. Dropping it
// there would silently change the operator, which is the #84 defect.
TEST(R1221Gphase, ControlledGphaseIsAnObservableRelativePhase) {
    expect_phase_on_all_ones(parse3(1, "ctrl @ gphase(0.5) q[0];\n"), 1, 0.5);
    expect_phase_on_all_ones(parse3(2, "ctrl(2) @ gphase(0.5) q[0], q[1];\n"),
                             2, 0.5);
    expect_phase_on_all_ones(
        parse3(3, "ctrl(3) @ gphase(0.5) q[0], q[1], q[2];\n"), 3, 0.5);
}

TEST(R1221Gphase, InvNegatesTheAngleAndPowScalesIt) {
    expect_phase_on_all_ones(parse3(1, "inv @ ctrl @ gphase(0.5) q[0];\n"),
                             1, -0.5);
    expect_phase_on_all_ones(parse3(1, "pow(3) @ ctrl @ gphase(0.5) q[0];\n"),
                             1, 1.5);
}

TEST(R1221Gphase, MalformedGphaseIsRejected) {
    // Wrong arity.
    EXPECT_THROW((void)parse3(1, "gphase(0.5, 0.25);\n"), std::runtime_error);
    EXPECT_THROW((void)parse3(1, "gphase();\n"), std::runtime_error);

    // Uncontrolled, but handed qubit operands it cannot act on.
    EXPECT_THROW((void)parse3(1, "gphase(0.5) q[0];\n"), std::runtime_error);

    // Control count and operand count must agree.
    EXPECT_THROW((void)parse3(2, "ctrl(2) @ gphase(0.5) q[0];\n"),
                 std::runtime_error);

    // A controlled gphase becomes an observable phase, so it needs a number
    // now rather than a promise of one later.
    EXPECT_THROW((void)QuantumCircuit::from_qasm3(
                     "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n"
                     "input float[64] a;\nqubit[1] q;\n"
                     "ctrl @ gphase(a) q[0];\n"),
                 std::runtime_error);
}

// Emission is per operand; delivery is per distinct message. A caller counting
// with a handler sees the second, so both halves are stated.
//
// This is the warning channel behaving as designed rather than an accident of
// this call site: identical text is delivered once and counted, so a lossy
// operand inside a shots loop cannot bury the log. Nothing is lost, since the
// repeat count is reported at the next flush.
TEST(R1221Qasm2Doors, WarningsAreOnePerOperandAndDeduplicatedByText) {
    const auto lossy_a = scaled(h_matrix(), PI / 5.0);
    const auto lossy_b = scaled(h_matrix(), PI / 7.0);

    // Two operands, two DIFFERENT dropped angles, so two different messages.
    {
        r1211::WarningProbe probe;
        QuantumCircuit qc(2);
        qc.unitary(lossy_a, {0});
        qc.unitary(lossy_b, {1});
        ASSERT_NO_THROW((void)qc.to_qasm2(door_open_accept_loss()));
        EXPECT_EQ(probe.count(), 2u)
            << "two operands dropped two different phases, so neither report "
               "may be folded into the other";
    }

    // Two operands, the SAME dropped angle, so one message twice. The channel
    // delivers it once and counts the repeat.
    {
        r1211::WarningProbe probe;
        QuantumCircuit qc(2);
        qc.unitary(lossy_a, {0});
        qc.unitary(lossy_a, {1});
        ASSERT_NO_THROW((void)qc.to_qasm2(door_open_accept_loss()));
        EXPECT_EQ(probe.count(), 1u)
            << "identical text is deduplicated by the warning channel, which is "
               "what keeps a lossy operand in a shots loop from burying the log";
    }
}
