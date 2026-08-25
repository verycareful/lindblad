// R.1.21.1 test wave - how physical the library's own operators actually are.
//
// Two tolerances coexist in the tree and they disagree by four orders of
// magnitude. The Class C framework judges caller-supplied matrices at
// atol = 1e-12, while Operator::is_unitary and KrausChannel::is_valid default
// to 1e-8. The looser pair predate the framework and nobody has measured
// whether they need the room.
//
// This file measures it. Every named gate, swept over parameters where it takes
// them, and every noise channel across its parameter range, is measured against
// the residual the framework itself computes, and each measurement is asserted
// against the framework's own constant rather than against the looser default.
// Derived operators are included too, since is_unitary is applied to products
// and tensor factors and not only to primitives, and composition is where
// rounding actually accumulates.
//
// Every suite reports its worst measured residual through RecordProperty, so a
// run produces the numbers even when it passes. That is the evidence a later
// release needs in order to tighten the two defaults, and it is worth having on
// the record whether or not the tightening happens.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// The tolerance the Class C framework applies to a caller's matrix. Every
// measurement below is held to this, not to the 1e-8 the older predicates use.
constexpr double FRAMEWORK_ATOL = 1e-12;

// The looser default currently carried by Operator::is_unitary and
// KrausChannel::is_valid.
constexpr double LEGACY_ATOL = 1e-8;

// Accumulates the worst residual seen, so a suite reports one number rather
// than only a verdict.
class WorstResidual {
public:
    void observe(double residual, const std::string& what) {
        if (!(residual <= worst_)) {   // NaN-safe: a NaN residual becomes worst
            worst_ = residual;
            worst_name_ = what;
        }
    }
    double worst() const { return worst_; }
    const std::string& worst_name() const { return worst_name_; }

private:
    double worst_ = 0.0;
    std::string worst_name_ = "(none)";
};

// Builds the dense operator a single instruction produces, by basis-column
// extraction. Going through Operator::from_circuit measures the matrix the
// library actually hands to a backend rather than an internal builder's output.
double gate_residual(const QuantumCircuit& qc) {
    const Operator op = Operator::from_circuit(qc);
    const std::size_t rows = static_cast<std::size_t>(1) << qc.n_qubits;
    return detail::unitarity_deviation(op.data.data(), rows);
}

// A spread of angles that includes the exact ones (0, pi/2, pi) where a matrix
// entry is representable and the awkward ones where it is not.
std::vector<double> angle_sweep() {
    std::vector<double> angles{0.0, PI_4, PI_2, PI, 2.0 * PI, -PI_2, 1.0, 0.1};
    for (int k = 1; k < 12; ++k)
        angles.push_back(2.0 * PI * static_cast<double>(k) / 12.0);
    return angles;
}

} // namespace

// =============================================================================
// Single-qubit gates
// =============================================================================

TEST(R1211GateMargins, FixedSingleQubitGates) {
    WorstResidual worst;
    struct Case { const char* name; void (*build)(QuantumCircuit&); };
    const Case cases[]{
        {"h",    [](QuantumCircuit& c) { c.h(0); }},
        {"x",    [](QuantumCircuit& c) { c.x(0); }},
        {"y",    [](QuantumCircuit& c) { c.y(0); }},
        {"z",    [](QuantumCircuit& c) { c.z(0); }},
        {"s",    [](QuantumCircuit& c) { c.s(0); }},
        {"sdg",  [](QuantumCircuit& c) { c.sdg(0); }},
        {"t",    [](QuantumCircuit& c) { c.t(0); }},
        {"tdg",  [](QuantumCircuit& c) { c.tdg(0); }},
        {"sx",   [](QuantumCircuit& c) { c.sx(0); }},
        {"sxdg", [](QuantumCircuit& c) { c.sxdg(0); }},
    };

    for (const auto& c : cases) {
        QuantumCircuit qc(1);
        c.build(qc);
        const double residual = gate_residual(qc);
        worst.observe(residual, c.name);
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << c.name << " deviates from unitarity by " << residual
            << ", outside the framework's own tolerance";
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_gate", worst.worst_name());
}

TEST(R1211GateMargins, ParametrisedSingleQubitGates) {
    WorstResidual worst;
    for (double theta : angle_sweep()) {
        struct Case { const char* name; QuantumCircuit qc; };
        std::vector<Case> cases;

        {
            QuantumCircuit qc(1); qc.rx(theta, 0);
            cases.push_back({"rx", qc});
        }
        {
            QuantumCircuit qc(1); qc.ry(theta, 0);
            cases.push_back({"ry", qc});
        }
        {
            QuantumCircuit qc(1); qc.rz(theta, 0);
            cases.push_back({"rz", qc});
        }
        {
            QuantumCircuit qc(1); qc.p(theta, 0);
            cases.push_back({"p", qc});
        }
        {
            QuantumCircuit qc(1); qc.u1(theta, 0);
            cases.push_back({"u1", qc});
        }
        {
            QuantumCircuit qc(1); qc.u2(theta, theta / 2.0, 0);
            cases.push_back({"u2", qc});
        }
        {
            QuantumCircuit qc(1); qc.u3(theta, theta / 2.0, theta / 3.0, 0);
            cases.push_back({"u3", qc});
        }
        {
            QuantumCircuit qc(1); qc.u(theta, theta / 2.0, theta / 3.0, 0);
            cases.push_back({"u", qc});
        }

        for (const auto& c : cases) {
            const double residual = gate_residual(c.qc);
            const std::string label =
                std::string(c.name) + "(theta=" + std::to_string(theta) + ")";
            worst.observe(residual, label);
            EXPECT_LE(residual, FRAMEWORK_ATOL)
                << label << " deviates by " << residual;
        }
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_gate", worst.worst_name());
}

// =============================================================================
// Two- and three-qubit gates
// =============================================================================

TEST(R1211GateMargins, FixedMultiQubitGates) {
    WorstResidual worst;

    struct Case { const char* name; QuantumCircuit qc; };
    std::vector<Case> cases;
    {
        QuantumCircuit qc(2); qc.cx(0, 1);    cases.push_back({"cx", qc});
    }
    {
        QuantumCircuit qc(2); qc.cy(0, 1);    cases.push_back({"cy", qc});
    }
    {
        QuantumCircuit qc(2); qc.cz(0, 1);    cases.push_back({"cz", qc});
    }
    {
        QuantumCircuit qc(2); qc.ch(0, 1);    cases.push_back({"ch", qc});
    }
    {
        QuantumCircuit qc(2); qc.swap(0, 1);  cases.push_back({"swap", qc});
    }
    {
        QuantumCircuit qc(2); qc.iswap(0, 1); cases.push_back({"iswap", qc});
    }
    {
        QuantumCircuit qc(2); qc.ecr(0, 1);   cases.push_back({"ecr", qc});
    }
    {
        QuantumCircuit qc(3); qc.ccx(0, 1, 2); cases.push_back({"ccx", qc});
    }
    {
        QuantumCircuit qc(3); qc.ccz(0, 1, 2); cases.push_back({"ccz", qc});
    }
    {
        QuantumCircuit qc(3); qc.cswap(0, 1, 2); cases.push_back({"cswap", qc});
    }

    for (const auto& c : cases) {
        const double residual = gate_residual(c.qc);
        worst.observe(residual, c.name);
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << c.name << " deviates by " << residual;
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_gate", worst.worst_name());
}

TEST(R1211GateMargins, ParametrisedMultiQubitGates) {
    WorstResidual worst;
    for (double theta : angle_sweep()) {
        struct Case { const char* name; QuantumCircuit qc; };
        std::vector<Case> cases;
        {
            QuantumCircuit qc(2); qc.crx(theta, 0, 1); cases.push_back({"crx", qc});
        }
        {
            QuantumCircuit qc(2); qc.cry(theta, 0, 1); cases.push_back({"cry", qc});
        }
        {
            QuantumCircuit qc(2); qc.crz(theta, 0, 1); cases.push_back({"crz", qc});
        }
        {
            QuantumCircuit qc(2); qc.cp(theta, 0, 1);  cases.push_back({"cp", qc});
        }
        {
            QuantumCircuit qc(2); qc.rxx(theta, 0, 1); cases.push_back({"rxx", qc});
        }
        {
            QuantumCircuit qc(2); qc.ryy(theta, 0, 1); cases.push_back({"ryy", qc});
        }
        {
            QuantumCircuit qc(2); qc.rzz(theta, 0, 1); cases.push_back({"rzz", qc});
        }
        {
            QuantumCircuit qc(2); qc.rzx(theta, 0, 1); cases.push_back({"rzx", qc});
        }
        {
            QuantumCircuit qc(2);
            qc.cu(theta, theta / 2.0, theta / 3.0, theta / 4.0, 0, 1);
            cases.push_back({"cu", qc});
        }

        for (const auto& c : cases) {
            const double residual = gate_residual(c.qc);
            const std::string label =
                std::string(c.name) + "(theta=" + std::to_string(theta) + ")";
            worst.observe(residual, label);
            EXPECT_LE(residual, FRAMEWORK_ATOL)
                << label << " deviates by " << residual;
        }
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_gate", worst.worst_name());
}

// =============================================================================
// Derived operators
// =============================================================================

TEST(R1211OperatorMargins, CompositionAccumulatesSlowly) {
    // is_unitary is applied to products, and a product is where rounding
    // actually builds up. If a deep composition needed the 1e-8 room, this is
    // the test that would show it.
    WorstResidual worst;

    for (int depth : {1, 2, 4, 8, 16, 32, 64, 128}) {
        QuantumCircuit qc(2);
        for (int k = 0; k < depth; ++k) {
            qc.h(0).t(1).cx(0, 1).ry(0.37, 1).cz(1, 0).sx(0);
        }
        const double residual = gate_residual(qc);
        worst.observe(residual, "depth " + std::to_string(depth));
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << "a " << depth
            << "-layer composition deviates by " << residual
            << ", which would mean the framework's default is too tight for "
               "the library's own products";
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_case", worst.worst_name());
}

TEST(R1211OperatorMargins, TensorProductsStayUnitary) {
    WorstResidual worst;

    QuantumCircuit single(1);
    single.u(0.7, 1.1, -0.3, 0);
    Operator op = Operator::from_circuit(single);

    Operator accumulated = op;
    for (int factors = 2; factors <= 6; ++factors) {
        accumulated = accumulated.tensor(op);
        const std::size_t rows = static_cast<std::size_t>(1) << factors;
        const double residual =
            detail::unitarity_deviation(accumulated.data.data(), rows);
        worst.observe(residual, std::to_string(factors) + " factors");
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << factors << " tensor factors deviate by " << residual;
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_case", worst.worst_name());
}

TEST(R1211OperatorMargins, ComposeChainStaysUnitary) {
    WorstResidual worst;

    QuantumCircuit a(2); a.h(0).cx(0, 1);
    QuantumCircuit b(2); b.ry(0.41, 1).cz(0, 1);
    Operator left = Operator::from_circuit(a);
    const Operator right = Operator::from_circuit(b);

    for (int steps = 1; steps <= 40; ++steps) {
        left = left.compose(right);
        const double residual = detail::unitarity_deviation(left.data.data(), 4);
        worst.observe(residual, std::to_string(steps) + " compositions");
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << steps << " compositions deviate by " << residual;
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_case", worst.worst_name());
}

// =============================================================================
// Noise channels
// =============================================================================

TEST(R1211ChannelMargins, EveryBuiltInChannelIsTracePreserving) {
    WorstResidual worst;

    auto measure = [&worst](const std::string& label, const KrausChannel& ch) {
        const std::size_t dim = static_cast<std::size_t>(1)
                                << static_cast<std::size_t>(ch.n_qubits);
        const double residual = detail::kraus_tp_deviation(ch.operators, dim);
        worst.observe(residual, label);
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << label << " deviates from trace preservation by " << residual;
    };

    for (double p : {0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0}) {
        const std::string tag = "(p=" + std::to_string(p) + ")";
        measure("depolarizing" + tag, NoiseChannels::depolarizing(p, 1));
        measure("amplitude_damping" + tag, NoiseChannels::amplitude_damping(p));
        measure("phase_damping" + tag, NoiseChannels::phase_damping(p));
        measure("bit_flip" + tag, NoiseChannels::bit_flip(p));
        measure("phase_flip" + tag, NoiseChannels::phase_flip(p));
        measure("bit_phase_flip" + tag, NoiseChannels::bit_phase_flip(p));
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_channel", worst.worst_name());
}

TEST(R1211ChannelMargins, TwoQubitDepolarizingIsTracePreserving) {
    WorstResidual worst;
    for (double p : {0.0, 0.05, 0.25, 0.5, 1.0}) {
        const KrausChannel ch = NoiseChannels::depolarizing(p, 2);
        const double residual = detail::kraus_tp_deviation(ch.operators, 4);
        worst.observe(residual, "depolarizing2(p=" + std::to_string(p) + ")");
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << "two-qubit depolarizing at p = " << p << " deviates by "
            << residual;
    }
    RecordProperty("worst_residual", std::to_string(worst.worst()));
}

TEST(R1211ChannelMargins, PauliAndResetChannels) {
    WorstResidual worst;

    for (double px : {0.0, 0.1, 0.3}) {
        for (double py : {0.0, 0.1, 0.2}) {
            for (double pz : {0.0, 0.1, 0.4}) {
                if (px + py + pz > 1.0) continue;
                const KrausChannel ch = NoiseChannels::pauli(px, py, pz);
                const double residual = detail::kraus_tp_deviation(ch.operators, 2);
                worst.observe(residual, "pauli");
                EXPECT_LE(residual, FRAMEWORK_ATOL)
                    << "pauli(" << px << ", " << py << ", " << pz
                    << ") deviates by " << residual;
            }
        }
    }

    for (double p0 : {0.0, 0.25, 0.5, 1.0}) {
        const double p1 = 1.0 - p0;
        const KrausChannel ch = NoiseChannels::reset(p0, p1);
        const double residual = detail::kraus_tp_deviation(ch.operators, 2);
        worst.observe(residual, "reset");
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << "reset(" << p0 << ", " << p1 << ") deviates by " << residual;
    }

    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_channel", worst.worst_name());
}

TEST(R1211ChannelMargins, ThermalRelaxationAcrossItsValidRegion) {
    // T2 <= 2*T1 is the physical constraint. The dephasing factor of 2 in the
    // channel construction is load-bearing at the T2 = 2*T1 boundary, so the
    // sweep includes it.
    WorstResidual worst;
    for (double t1 : {10.0, 100.0, 1000.0}) {
        for (double ratio : {0.1, 0.5, 1.0, 1.5, 2.0}) {
            const double t2 = t1 * ratio;
            for (double gate_time : {0.1, 1.0, 10.0}) {
                const KrausChannel ch =
                    NoiseChannels::thermal_relaxation(t1, t2, gate_time);
                const double residual =
                    detail::kraus_tp_deviation(ch.operators, 2);
                const std::string label =
                    "thermal(T1=" + std::to_string(t1) +
                    ", T2=" + std::to_string(t2) +
                    ", t=" + std::to_string(gate_time) + ")";
                worst.observe(residual, label);
                EXPECT_LE(residual, FRAMEWORK_ATOL)
                    << label << " deviates by " << residual;
            }
        }
    }
    RecordProperty("worst_residual", std::to_string(worst.worst()));
    RecordProperty("worst_channel", worst.worst_name());
}

TEST(R1211ChannelMargins, CoherentUnitaryIsBothUnitaryAndTracePreserving) {
    // A one-operator channel, so the two properties coincide and both
    // residuals must agree.
    WorstResidual worst;
    for (double theta : angle_sweep()) {
        const KrausChannel ch =
            NoiseChannels::coherent_unitary(theta, theta / 2.0, theta / 3.0);
        ASSERT_EQ(ch.operators.size(), 1u)
            << "a coherent error is a single unitary; if it now has more "
               "operators the two residuals no longer have to agree";

        const double tp = detail::kraus_tp_deviation(ch.operators, 2);
        const double unitarity =
            detail::unitarity_deviation(ch.operators[0].data(), 2);
        worst.observe(tp, "coherent_unitary(theta=" + std::to_string(theta) + ")");

        EXPECT_LE(tp, FRAMEWORK_ATOL);
        EXPECT_LE(unitarity, FRAMEWORK_ATOL);
        EXPECT_NEAR(tp, unitarity, 1e-15)
            << "for a single Kraus operator the trace-preservation residual is "
               "the unitarity residual; they must not diverge";
    }
    RecordProperty("worst_residual", std::to_string(worst.worst()));
}

// =============================================================================
// What the margins mean for the older predicates
// =============================================================================

TEST(R1211LegacyTolerances, BuiltInOperatorsClearTheFrameworkConstant) {
    // The decision this file exists to inform. Operator::is_unitary defaults to
    // 1e-8; if the library's own operators sit near machine epsilon then that
    // default is four orders wider than anything needs, and a matrix wrong by
    // 1e-9 passes a predicate whose whole job is to catch it.
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).t(2).cswap(0, 1, 2).ry(0.83, 1).ccx(0, 1, 2);
    const Operator op = Operator::from_circuit(qc);

    EXPECT_TRUE(op.is_unitary(LEGACY_ATOL))
        << "the current default must keep accepting this";
    EXPECT_TRUE(op.is_unitary(FRAMEWORK_ATOL))
        << "and so must the framework's constant, which is what makes "
           "tightening the default a safe change";
}

TEST(R1211LegacyTolerances, BuiltInChannelsClearTheFrameworkConstant) {
    const KrausChannel ch = NoiseChannels::thermal_relaxation(100.0, 80.0, 1.0);
    EXPECT_TRUE(ch.is_valid(LEGACY_ATOL));
    EXPECT_TRUE(ch.is_valid(FRAMEWORK_ATOL))
        << "the same argument for KrausChannel::is_valid";
}

TEST(R1211LegacyTolerances, TheLooseDefaultAcceptsWhatTheFrameworkRejects) {
    // The gap made concrete, so the cost of leaving the defaults alone is on
    // the record rather than implied. A matrix wrong by 1e-9 is accepted by
    // is_unitary and rejected by every Class C primitive in the library.
    const double deviation = 1e-9;
    const Operator op(
        {Complex128(std::sqrt(1.0 + deviation), 0.0), Complex128(0.0, 0.0),
         Complex128(0.0, 0.0), Complex128(1.0, 0.0)},
        1);

    EXPECT_TRUE(op.is_unitary(LEGACY_ATOL))
        << "the 1e-8 default accepts a matrix off by 1e-9";
    EXPECT_FALSE(op.is_unitary(FRAMEWORK_ATOL))
        << "the framework's constant rejects the same matrix, and every "
           "backend primitive already does";
}
