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
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// The tolerance the Class C framework applies to a caller's matrix. Every
// measurement below is held to this, not to the 1e-8 the older predicates use.
// Taken from the framework rather than repeated, so this suite cannot end up
// measuring against a number the library has stopped using.
constexpr double FRAMEWORK_ATOL = DEFAULT_PHYSICAL_ATOL;

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

// -----------------------------------------------------------------------------
// Reporting a measurement so it can actually be read
// -----------------------------------------------------------------------------
// RecordProperty on its own does not deliver a number to anyone. It writes to
// the XML report, which a normal run does not produce, and std::to_string
// renders a double with six decimal places, so every residual here, all of them
// far below 1e-6, arrives as "0.000000".
//
// RecordProperty is qualified below because these macros are used from free
// helper functions as well as from test bodies, and the unqualified name
// resolves only inside a Test-derived class. It is a public static member, so
// the qualified form works in both places.
//
// So a value goes through format_residual, which keeps significant digits, and
// is printed as well as recorded. The tag is greppable on purpose: the archived
// console capture then carries the margins on its own, which is what makes a
// later tightening decision reviewable without re-running anything.
#define REPORT_RESIDUAL(key, value)                                       \
    do {                                                                  \
        const std::string rk_ = (key);                                    \
        const std::string rv_ = detail::format_residual(value);           \
        ::testing::Test::RecordProperty(rk_, rv_);                        \
        std::cout << "[ MARGIN   ] " << rk_ << " = " << rv_ << std::endl; \
    } while (0)

// One point of a sweep, so the CURVE is readable and not only its maximum.
//
// A worst value is a single point, and every sweep here whose worst lands on
// its deepest case is still climbing where the sweep stops: the maximum alone
// cannot say whether the next depth would clear the tolerance. The loops
// already compute every value and discarded all but one, so recording each is
// the whole change. The archived capture then carries the slope, which is what
// a later question about a deeper circuit actually needs.
#define REPORT_CURVE_POINT(key, x, value)                                     do {                                                                          const std::string ck_ = std::string(key) + "@" + (x);                     const std::string cv_ = detail::format_residual(value);                   ::testing::Test::RecordProperty(ck_, cv_);                                std::cout << "[ MARGIN   ] " << ck_ << " = " << cv_ << std::endl;     } while (0)

#define REPORT_WHERE(key, name)                                           \
    do {                                                                  \
        const std::string wk_ = (key);                                    \
        const std::string wv_ = (name);                                   \
        ::testing::Test::RecordProperty(wk_, wv_);                        \
        std::cout << "[ MARGIN   ] " << wk_ << " = " << wv_ << std::endl; \
    } while (0)


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

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_gate", worst.worst_name());
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

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_gate", worst.worst_name());
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

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_gate", worst.worst_name());
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

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_gate", worst.worst_name());
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
        REPORT_CURVE_POINT("composition_residual", std::to_string(depth),
                           residual);
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << "a " << depth
            << "-layer composition deviates by " << residual
            << ", which would mean the framework's default is too tight for "
               "the library's own products";
    }

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_case", worst.worst_name());
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
        REPORT_CURVE_POINT("tensor_residual", std::to_string(factors),
                           residual);
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << factors << " tensor factors deviate by " << residual;
    }

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_case", worst.worst_name());
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
        REPORT_CURVE_POINT("compose_chain_residual", std::to_string(steps),
                           residual);
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << steps << " compositions deviate by " << residual;
    }

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_case", worst.worst_name());
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

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_channel", worst.worst_name());
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
    REPORT_RESIDUAL("worst_residual", worst.worst());
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

    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_channel", worst.worst_name());
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
    REPORT_RESIDUAL("worst_residual", worst.worst());
    REPORT_WHERE("worst_channel", worst.worst_name());
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
    REPORT_RESIDUAL("worst_residual", worst.worst());
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

// =============================================================================
// DensityMatrix::is_valid on matrices that have EVOLVED
// =============================================================================
// The sweeps above measure operators the caller SUPPLIES. This one measures an
// object the library produced: a density matrix carried through a noisy circuit,
// where rounding has had a chance to accumulate across every gate and every
// channel application. That is the case the earlier tightening never covered,
// and the two in-tree callers of is_valid hand it essentially exact matrices,
// so a green suite elsewhere is not evidence about this.
//
// Both residuals is_valid actually tests are recorded: the trace, and the worst
// Hermiticity violation, which is |rho_ij - conj(rho_ji)| since the predicate
// compares its square against atol squared.

namespace {

double trace_residual(const DensityMatrix& rho) {
    return std::abs(detail::density_trace_real(rho.data.data(), rho.dim) - 1.0);
}

double hermiticity_residual(const DensityMatrix& rho) {
    double worst = 0.0;
    for (std::size_t i = 0; i < rho.dim; ++i) {
        for (std::size_t j = i + 1; j < rho.dim; ++j) {
            const Complex128 d =
                rho.data[i * rho.dim + j] - rho.data[j * rho.dim + i].conj();
            const double m = std::sqrt(d.norm_sq());
            if (!(m <= worst)) worst = m;   // NaN-safe: a NaN becomes the worst
        }
    }
    return worst;
}

// A circuit that entangles across the register and repeats, so depth means
// accumulated arithmetic rather than a longer list of commuting gates.
QuantumCircuit noisy_layers(int n_qubits, int depth) {
    QuantumCircuit qc(n_qubits);
    for (int d = 0; d < depth; ++d) {
        for (int q = 0; q < n_qubits; ++q) qc.h(q);
        for (int q = 0; q + 1 < n_qubits; ++q) qc.cx(q, q + 1);
        for (int q = 0; q < n_qubits; ++q)
            qc.rz(0.37 * static_cast<double>(d + 1) + 0.11 * static_cast<double>(q), q);
    }
    return qc;
}

NoiseModel damping_and_depolarizing() {
    NoiseModel nm;
    nm.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.01), "h");
    nm.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.02), "rz");
    nm.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.015, 2), "cx");
    return nm;
}

} // namespace

TEST(V11231EvolvedDensityMargin, TraceAndHermiticityHoldAcrossDepthAndWidth) {
    WorstResidual worst_trace;
    WorstResidual worst_herm;

    const NoiseModel nm = damping_and_depolarizing();
    DensityMatrixSimulator sim;

    for (int nq : {2, 3, 4}) {
        for (int depth : {1, 4, 16}) {
            const std::string what =
                "n=" + std::to_string(nq) + " depth=" + std::to_string(depth);
            SCOPED_TRACE(what);

            const auto res = sim.run(noisy_layers(nq, depth), nm, 0, 20260829);
            ASSERT_TRUE(res.success) << res.error_message;
            const DensityMatrix& rho = res.final_state;

            worst_trace.observe(trace_residual(rho), what);
            worst_herm.observe(hermiticity_residual(rho), what);

            EXPECT_TRUE(rho.is_valid(FRAMEWORK_ATOL))
                << "an evolved density matrix must satisfy the tolerance the "
                   "predicate now judges by";
        }
    }

    REPORT_RESIDUAL("worst_evolved_trace_residual", worst_trace.worst());
    REPORT_WHERE("worst_evolved_trace_at", worst_trace.worst_name());
    REPORT_RESIDUAL("worst_evolved_hermiticity_residual", worst_herm.worst());
    REPORT_WHERE("worst_evolved_hermiticity_at", worst_herm.worst_name());

    EXPECT_LE(worst_trace.worst(), FRAMEWORK_ATOL)
        << "worst trace residual at " << worst_trace.worst_name();
    EXPECT_LE(worst_herm.worst(), FRAMEWORK_ATOL)
        << "worst Hermiticity residual at " << worst_herm.worst_name();
}

// The trace is what a noisy channel is most likely to move, so it is worth
// measuring against a channel chosen to be lossy rather than a mild one.
TEST(V11231EvolvedDensityMargin, StrongDampingStillPreservesTheTrace) {
    NoiseModel nm;
    nm.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.25), "h");
    nm.add_all_qubit_quantum_error(NoiseChannels::phase_damping(0.25), "rz");

    DensityMatrixSimulator sim;
    WorstResidual worst;

    for (int depth : {1, 8, 32}) {
        const std::string what = "depth=" + std::to_string(depth);
        const auto res = sim.run(noisy_layers(3, depth), nm, 0, 20260829);
        ASSERT_TRUE(res.success) << res.error_message;
        const double residual = trace_residual(res.final_state);
        worst.observe(residual, what);
        REPORT_CURVE_POINT("damped_trace_residual", std::to_string(depth),
                           residual);
    }

    REPORT_RESIDUAL("worst_damped_trace_residual", worst.worst());
    EXPECT_LE(worst.worst(), FRAMEWORK_ATOL)
        << "a trace-preserving channel must preserve the trace to the tolerance "
           "the predicate uses, worst at " << worst.worst_name();
}

// =============================================================================
// The normalization residual against register size
// =============================================================================
// state_norm_sq reduces over 2^n positive terms, so its error grows with the
// register while the tolerance it is compared against does not. A balanced tree
// is what keeps a correct state from being reported unnormalized by its own
// measurement; this records where the residual actually sits rather than where
// the bound says it should.
//
// ON THE UPPER LIMIT. The default sweep stops at 24, which is 268 MB, so the
// suite stays runnable on a CI runner and on an ordinary machine. Thirty qubits
// is 17.2 GB: large, but only a question of where the suite runs, so it lives
// in the opt-in test below rather than being left out. Where neither is run,
// the trend still carries: tree error grows like log2(N), so 24 to 30 adds a
// quarter to the number of levels, not sixty-four times the error a running
// total would have accumulated.

namespace {

// A normalized state of nq qubits whose amplitudes are neither representable
// nor equal, so every partial sum rounds and small terms have to survive beside
// large ones. A uniform superposition would be exact in binary and would
// measure nothing at all. Returns how far the measured norm sits from 1, or a
// non-finite value if the state could not be normalized.
double normalization_residual(int nq) {
    const std::size_t dim = std::size_t(1) << nq;
    std::vector<Complex128> amps(dim);
    for (std::size_t k = 0; k < dim; ++k) {
        const double x = static_cast<double>(k + 1);
        amps[k] = Complex128(1.0 / std::sqrt(x), 1.0 / (3.0 * x));
    }
    const double norm = std::sqrt(detail::state_norm_sq(amps.data(), dim));
    if (!is_normalizable(norm)) return std::numeric_limits<double>::quiet_NaN();
    for (Complex128& a : amps) {
        a.real /= norm;
        a.imag /= norm;
    }
    return std::abs(detail::state_norm_sq(amps.data(), dim) - 1.0);
}

void sweep_register_sizes(const std::vector<int>& sizes, const char* worst_key,
                          const char* worst_where_key, WorstResidual& worst) {
    for (int nq : sizes) {
        const std::string what = "n=" + std::to_string(nq);
        SCOPED_TRACE(what);
        const double residual = normalization_residual(nq);
        ASSERT_TRUE(is_finite_strict(residual)) << "could not normalize at " << what;
        worst.observe(residual, what);
        REPORT_RESIDUAL("norm_residual_" + what, residual);
        EXPECT_LE(residual, FRAMEWORK_ATOL)
            << "the measurement must not invent a violation at " << what;
    }
    REPORT_RESIDUAL(worst_key, worst.worst());
    REPORT_WHERE(worst_where_key, worst.worst_name());
}

} // namespace

TEST(V11231NormalizationMargin, TheResidualStaysFlatAcrossRegisterSize) {
    WorstResidual worst;
    sweep_register_sizes({12, 16, 20, 22, 24}, "worst_norm_residual",
                         "worst_norm_residual_at", worst);
}

// The sweep above stops at 24 so the suite stays runnable everywhere: a CI
// runner and an ordinary laptop cannot hold what the next sizes need. That is a
// portability limit and not a physical one, so the large end is opt-in rather
// than absent, and it reports through exactly the same channel.
//
//   n = 26   1.1 GB      n = 28   4.3 GB      n = 30   17.2 GB
//
// One size is held at a time, so peak resident memory is the largest of them.
// Run with LINDBLAD_BIG_MARGIN_SWEEP=1 on a machine with the headroom.
TEST(V11231NormalizationMargin, TheResidualAtLargeRegisterSizes) {
    if (std::getenv("LINDBLAD_BIG_MARGIN_SWEEP") == nullptr) {
        GTEST_SKIP() << "set LINDBLAD_BIG_MARGIN_SWEEP=1 to run this; it needs "
                        "about 17.2 GB of memory at n=30";
    }
    WorstResidual worst;
    sweep_register_sizes({26, 28, 30}, "worst_large_norm_residual",
                         "worst_large_norm_residual_at", worst);
}
