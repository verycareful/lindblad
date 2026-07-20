// R.1.12.2 coverage-fill suite, batch F4: front ends, primitives, backend,
// noise factories, qudit MPS, circuit/DAG utilities, and the visualisation
// catalogues. Closes the line-coverage gaps measured on the instrumented
// R.1.12.1 run:
//
//   - qasm2_parser.cpp: every built-in gate keyword, pi-expression variants,
//     multi-line custom gate definitions with parameter arithmetic, and the
//     fail-loud error paths (malformed measure, register size mismatch,
//     unknown gate, unresolvable reset).
//   - estimator.cpp: the sampled path (Y basis change, identity terms, Pauli
//     validation), the transpile cache (miss then hit), the noisy shots==0
//     density-matrix evaluation.
//   - sampler.cpp: batch seeding, parameter binding, the noisy backend.
//   - local_backend.cpp: AUTO simulator selection (SV and the >20-qubit MPS
//     route) and run_batch.
//   - channels.cpp / noise_model.cpp: every channel factory (checked CPTP),
//     thermal_relaxation validation, from_t1_t2, qubit-specific error lookup.
//   - qudit_mps.cpp: reversed-operand and non-adjacent 2-qudit application
//     against the QuditStatevector oracle, validation throws, canonical
//     forms, deterministic measurement.
//   - circuit.cpp / dag.cpp: QASM2 export of barrier operand lists and
//     symbolic gates, JSON round-trip over the full gate-name map, the DAG
//     default constructor and two_qubit_ops.
//   - gate_symbols.cpp / composite_catalogue.cpp / render_html.cpp: the
//     visualisation catalogues (never queried by any earlier test) and HTML
//     escaping of user-supplied labels.

#include <gtest/gtest.h>

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/visualisation.hpp"

#include "../src/visualisation/composite_catalogue.hpp"
#include "../src/visualisation/gate_symbols.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

constexpr double kTol = 1e-9;

int count_type(const QuantumCircuit& qc, GT t) {
    int n = 0;
    for (const auto& inst : qc.instructions)
        if (inst.type == t) ++n;
    return n;
}

int counts_total(const std::unordered_map<std::string, int>& c) {
    int n = 0;
    for (const auto& [k, v] : c) n += v;
    return n;
}

// Kraus completeness: sum_k K_k^dagger K_k == I (dim x dim).
void expect_cptp(const KrausChannel& ch, size_t dim, const std::string& name) {
    std::vector<std::complex<double>> acc(dim * dim, {0.0, 0.0});
    for (const auto& K : ch.operators) {
        ASSERT_EQ(K.size(), dim * dim) << name;
        for (size_t i = 0; i < dim; ++i)
            for (size_t j = 0; j < dim; ++j) {
                std::complex<double> s(0.0, 0.0);
                for (size_t k = 0; k < dim; ++k) {
                    std::complex<double> a(K[k * dim + i].real, -K[k * dim + i].imag);
                    std::complex<double> b(K[k * dim + j].real, K[k * dim + j].imag);
                    s += a * b;
                }
                acc[i * dim + j] += s;
            }
    }
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j) {
            const double want = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(acc[i * dim + j].real(), want, 1e-9)
                << name << " completeness re(" << i << "," << j << ")";
            EXPECT_NEAR(acc[i * dim + j].imag(), 0.0, 1e-9)
                << name << " completeness im(" << i << "," << j << ")";
        }
}

// d = 3 discrete Fourier transform, F[j][k] = w^{jk} / sqrt(3).
std::vector<Complex128> dft3() {
    std::vector<Complex128> F(9);
    const double inv = 1.0 / std::sqrt(3.0);
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            const double a = 2.0 * M_PI * j * k / 3.0;
            F[j * 3 + k] = Complex128(inv * std::cos(a), inv * std::sin(a));
        }
    return F;
}

}  // namespace

// =============================================================================
// QASM2 parser
// =============================================================================

TEST(R1122FillFront, Qasm2EveryBuiltinGateAndPiExpressionParses) {
    // The parser is line-oriented: one statement per line.
    const std::string src = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
x q[0];
y q[0];
z q[0];
s q[0];
sdg q[0];
t q[0];
tdg q[0];
sx q[0];
rx(0.3) q[0];
ry(pi) q[1];
rz(pi/2) q[2];
p(-pi/4) q[0];
u(0.1,0.2,0.3) q[0];
u1(2*pi) q[1];
u2(pi,-pi) q[2];
u3(0.1,0.2,0.3) q[0];
cx q[0],q[1];
cy q[1],q[2];
cz q[0],q[2];
ch q[0],q[1];
swap q[1],q[2];
crx(0.5) q[0],q[1];
cry(0.5) q[1],q[2];
crz(0.5) q[0],q[1];
cp(0.5) q[1],q[2];
rxx(0.3) q[0],q[1];
ryy(0.3) q[1],q[2];
rzz(0.3) q[0],q[1];
ccx q[0],q[1],q[2];
cswap q[0],q[1],q[2];
barrier q;
reset q[0];
measure q -> c;
)";
    auto qc = QuantumCircuit::from_qasm2(src);
    EXPECT_EQ(qc.n_qubits, 3);
    EXPECT_EQ(qc.n_clbits, 3);
    EXPECT_EQ(count_type(qc, GT::MEASURE), 3) << "whole-register measure expands";
    EXPECT_EQ(count_type(qc, GT::RESET), 1);
    EXPECT_EQ(count_type(qc, GT::BARRIER), 1);
    EXPECT_EQ(count_type(qc, GT::CCX), 1);
    EXPECT_EQ(count_type(qc, GT::CSWAP), 1);
    EXPECT_EQ(qc.instructions.size(), 36u) << "31 gates + barrier + reset + 3 measures";

    // pi-expression spot checks (evaluate_pi_expr variants).
    auto param_of = [&](GT t) -> double {
        for (const auto& inst : qc.instructions)
            if (inst.type == t) return inst.params[0];
        ADD_FAILURE() << "gate not found";
        return 0.0;
    };
    EXPECT_NEAR(param_of(GT::RY), M_PI, kTol);
    EXPECT_NEAR(param_of(GT::RZ), M_PI / 2.0, kTol);
    EXPECT_NEAR(param_of(GT::P), -M_PI / 4.0, kTol);
    EXPECT_NEAR(param_of(GT::U1), 2.0 * M_PI, kTol);
    EXPECT_NEAR(param_of(GT::U2), M_PI, kTol);
}

TEST(R1122FillFront, Qasm2CustomGateDefinitionParameterArithmetic) {
    const std::string src = R"(OPENQASM 2.0;
qreg q[2];
creg c[2];
gate foo(a) p, r {
  rz(a) p;
  rz(a/2) r;
  rx(2*a) p;
  ry(a+pi) r;
  rz(-a) p;
  rz(a-pi/2) r;
  cx p, r;
}
foo(0.8) q[0], q[1];
)";
    auto qc = QuantumCircuit::from_qasm2(src);
    ASSERT_EQ(qc.instructions.size(), 7u) << "custom gate expands to its body";
    EXPECT_EQ(qc.instructions[0].type, GT::RZ);
    EXPECT_NEAR(qc.instructions[0].params[0], 0.8, kTol);
    EXPECT_EQ(qc.instructions[0].qubits, (std::vector<int>{0}));
    EXPECT_NEAR(qc.instructions[1].params[0], 0.4, kTol) << "a/2";
    EXPECT_EQ(qc.instructions[1].qubits, (std::vector<int>{1}));
    EXPECT_NEAR(qc.instructions[2].params[0], 1.6, kTol) << "2*a";
    EXPECT_NEAR(qc.instructions[3].params[0], 0.8 + M_PI, kTol) << "a+pi";
    EXPECT_NEAR(qc.instructions[4].params[0], -0.8, kTol) << "-a";
    EXPECT_NEAR(qc.instructions[5].params[0], 0.8 - M_PI / 2.0, kTol)
        << "a-pi/2 must bind subtraction before division";
    EXPECT_EQ(qc.instructions[6].type, GT::CX);
    EXPECT_EQ(qc.instructions[6].qubits, (std::vector<int>{0, 1}));
}

TEST(R1122FillFront, Qasm2ErrorPathsThrow) {
    const std::string head = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\n";
    EXPECT_THROW(QuantumCircuit::from_qasm2(head + "measure q[0] c[0];\n"),
                 std::runtime_error) << "measure without ->";
    EXPECT_THROW(QuantumCircuit::from_qasm2(
                     "OPENQASM 2.0;\nqreg q[2];\ncreg c[1];\nmeasure q -> c;\n"),
                 std::runtime_error) << "register size mismatch";
    EXPECT_THROW(QuantumCircuit::from_qasm2(head + "frobnicate q[0];\n"),
                 std::runtime_error) << "unknown gates must surface";
    EXPECT_THROW(QuantumCircuit::from_qasm2(head + "reset nosuch;\n"),
                 std::runtime_error) << "unresolvable reset operand";
}

TEST(R1122FillFront, Qasm3ModifierMatrixFallbackCoversRotationBases) {
    // pow(n) @ and non-named ctrl @ stacks resolve through the QASM3 modifier
    // matrix fallback; this sweeps the base-gate branches the named fast
    // paths never touch (sdg/tdg/sx/sxdg/rx/ry/rz/p/u/u2) plus mat_dagger.
    const std::string src =
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "qubit[2] q;\n"
        "pow(2) @ sdg q[0];\n"
        "pow(2) @ tdg q[0];\n"
        "pow(2) @ sx q[0];\n"
        "inv @ sxdg q[0];\n"
        "pow(2) @ rx(0.3) q[0];\n"
        "pow(2) @ ry(0.4) q[0];\n"
        "pow(2) @ rz(0.5) q[0];\n"
        "pow(2) @ p(0.6) q[0];\n"
        "pow(2) @ u(0.1, 0.2, 0.3) q[0];\n"
        "pow(-2) @ t q[0];\n"
        "ctrl @ u2(0.4, 0.2) q[0], q[1];\n";
    auto parsed = QuantumCircuit::from_qasm3(src);
    ASSERT_EQ(parsed.n_qubits, 2);
    ASSERT_EQ(parsed.instructions.size(), 11u);

    // Exact classical identities for every stack.
    QuantumCircuit ref(2);
    const double pi_2 = 1.5707963267948966;
    ref.z(0);              // Sdg^2 = P(-pi) = Z
    ref.sdg(0);            // Tdg^2 = Sdg
    ref.x(0);              // SX^2 = X
    ref.sx(0);             // inv(SXdg) = SX
    ref.rx(0.6, 0);
    ref.ry(0.8, 0);
    ref.rz(1.0, 0);
    ref.p(1.2, 0);
    // U3(0.1,0.2,0.3)^2 as an explicit matrix (no named identity exists).
    {
        const double th = 0.1, ph = 0.2, la = 0.3;
        const std::complex<double> u00(std::cos(th / 2), 0.0);
        const std::complex<double> u01 = -std::sin(th / 2) *
            std::complex<double>(std::cos(la), std::sin(la));
        const std::complex<double> u10 = std::sin(th / 2) *
            std::complex<double>(std::cos(ph), std::sin(ph));
        const std::complex<double> u11 = std::cos(th / 2) *
            std::complex<double>(std::cos(ph + la), std::sin(ph + la));
        const std::complex<double> m00 = u00 * u00 + u01 * u10;
        const std::complex<double> m01 = u00 * u01 + u01 * u11;
        const std::complex<double> m10 = u10 * u00 + u11 * u10;
        const std::complex<double> m11 = u10 * u01 + u11 * u11;
        ref.unitary({Complex128(m00.real(), m00.imag()),
                     Complex128(m01.real(), m01.imag()),
                     Complex128(m10.real(), m10.imag()),
                     Complex128(m11.real(), m11.imag())},
                    {0});
    }
    ref.sdg(0);                         // pow(-2) @ t = Tdg^2 = Sdg (negative pow -> dagger)
    ref.cu(pi_2, 0.4, 0.2, 0.0, 0, 1);  // ctrl @ u2(phi,lambda) = CU(pi/2,phi,lambda,0)

    // Full-matrix equivalence up to a global phase.
    auto ma = Operator::from_circuit(parsed).data;
    auto mb = Operator::from_circuit(ref).data;
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
        EXPECT_NEAR(ma[i].real, bp.real, 1e-7) << "re @ " << i;
        EXPECT_NEAR(ma[i].imag, bp.imag, 1e-7) << "im @ " << i;
    }
}

// =============================================================================
// Estimator
// =============================================================================

TEST(R1122FillFront, EstimatorSampledPathBranchesAndValidation) {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);

    Estimator est;
    est.options.shots = 2048;
    est.options.seed = 3;

    // Y basis change plus an identity term riding along.
    SparsePauliOp obs(std::vector<PauliString>{
        PauliString("YI", Complex128(0.5, 0.0)),
        PauliString("II", Complex128(0.25, 0.0))});
    const double v = est.run_single(qc, obs, {});
    // <Y_0> of a Bell state is 0: the estimate must sit near the identity
    // offset 0.25 (sampling noise bounded by 0.5 / sqrt(shots) * few sigma).
    EXPECT_NEAR(v, 0.25, 0.05);

    SparsePauliOp too_short(std::vector<PauliString>{PauliString("Z")});
    EXPECT_THROW(est.run_single(qc, too_short, {}), std::invalid_argument);

    SparsePauliOp bad_char(std::vector<PauliString>{PauliString("QI")});
    EXPECT_THROW(est.run_single(qc, bad_char, {}), std::invalid_argument);
}

TEST(R1122FillFront, EstimatorTranspileCacheAndNoisyEvaluation) {
    // Parameterised circuit: the cache key ignores parameter values, so the
    // second call must reuse the transpiled layout (hit path). CX commutes
    // with Z on its control, so <Z_0> = cos(a) exactly.
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    qc.rx("a", 0);
    qc.cx(0, 1);
    SparsePauliOp z0(std::vector<PauliString>{PauliString("ZI")});

    Estimator cached;
    cached.options.optimization_level = 1;
    EXPECT_NEAR(cached.run_single(qc, z0, {0.4}), std::cos(0.4), 1e-6);
    EXPECT_NEAR(cached.run_single(qc, z0, {1.1}), std::cos(1.1), 1e-6);

    // Noisy exact path: shots == 0 routes through the density matrix.
    QuantumCircuit flip(1);
    flip.x(0);
    SparsePauliOp z(std::vector<PauliString>{PauliString("Z")});

    Estimator noisy;
    noisy.options.noise_model.add_quantum_error(
        NoiseChannels::depolarizing(0.2), "x", {});
    const double noisy_z = noisy.run_single(flip, z, {});
    EXPECT_LT(noisy_z, -0.5) << "still predominantly |1>";
    EXPECT_GT(noisy_z, -1.0 + 1e-6) << "depolarizing must bite";

    Estimator ideal;
    EXPECT_NEAR(ideal.run_single(flip, z, {}), -1.0, kTol);
}

// =============================================================================
// Sampler
// =============================================================================

TEST(R1122FillFront, SamplerBatchSeedingParametersAndNoise) {
    QuantumCircuit rot(1, 1);
    rot.rx("t", 0);
    rot.measure(0, 0);
    QuantumCircuit fixed(1, 1);
    fixed.x(0).measure(0, 0);

    Sampler s;
    s.options.shots = 128;
    s.options.seed = 11;
    auto batch = s.run({rot, fixed}, {{M_PI}});  // params only for circuit 0
    ASSERT_EQ(batch.size(), 2u);
    ASSERT_EQ(batch[0].size(), 1u);
    EXPECT_EQ(batch[0].begin()->first, "1") << "rx(pi) flips deterministically";
    EXPECT_EQ(batch[1].begin()->first, "1");
    EXPECT_EQ(counts_total(batch[0]), 128);

    Sampler noisy;
    noisy.options.shots = 256;
    noisy.options.seed = 13;
    noisy.options.noise_model.add_quantum_error(
        NoiseChannels::bit_flip(0.25), "x", {});
    auto counts = noisy.run_single(fixed, {});
    EXPECT_EQ(counts_total(counts), 256);
    EXPECT_GT(counts.count("0"), 0u) << "bit flip must surface in the counts";
}

// =============================================================================
// LocalBackend
// =============================================================================

TEST(R1122FillFront, LocalBackendAutoSelectionAndBatch) {
    backends::LocalBackend auto_backend;  // Config.simulator = AUTO

    // Non-Clifford, small register: statevector route.
    QuantumCircuit small(2, 2);
    small.rx(0.3, 0).cx(0, 1).measure_all();
    auto r_small = auto_backend.run(small, 64, 5);
    EXPECT_TRUE(r_small.success) << r_small.error_message;
    EXPECT_EQ(counts_total(r_small.counts), 64);

    // Non-Clifford, more than 20 qubits: MPS route (bond dim stays tiny).
    QuantumCircuit wide(21, 21);
    wide.rx(0.3, 0);
    wide.measure_all();
    auto r_wide = auto_backend.run(wide, 8, 5);
    EXPECT_TRUE(r_wide.success) << r_wide.error_message;
    EXPECT_EQ(counts_total(r_wide.counts), 8);

    auto batch = auto_backend.run_batch({small, small}, 16, 7);
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_TRUE(batch[0].success);
    EXPECT_EQ(counts_total(batch[1].counts), 16);
}

// =============================================================================
// Noise channel factories and NoiseModel utilities
// =============================================================================

TEST(R1122FillFront, NoiseChannelFactoriesAreTracePreserving) {
    expect_cptp(NoiseChannels::depolarizing(0.1), 2, "depolarizing");
    expect_cptp(NoiseChannels::amplitude_damping(0.15), 2, "amplitude_damping");
    expect_cptp(NoiseChannels::phase_damping(0.2), 2, "phase_damping");
    expect_cptp(NoiseChannels::pauli(0.1, 0.05, 0.02), 2, "pauli");
    expect_cptp(NoiseChannels::bit_flip(0.1), 2, "bit_flip");
    expect_cptp(NoiseChannels::phase_flip(0.1), 2, "phase_flip");
    expect_cptp(NoiseChannels::bit_phase_flip(0.1), 2, "bit_phase_flip");
    expect_cptp(NoiseChannels::reset(0.1, 0.05), 2, "reset");
    expect_cptp(NoiseChannels::coherent_unitary(0.3, 0.2, 0.1), 2,
                "coherent_unitary");
    expect_cptp(NoiseChannels::thermal_relaxation(1.0, 0.8, 0.01, 0.1), 2,
                "thermal_relaxation with excited population");

    EXPECT_THROW(NoiseChannels::thermal_relaxation(1.0, 0.8, -0.01),
                 std::invalid_argument) << "negative gate time";
    EXPECT_THROW(NoiseChannels::thermal_relaxation(1.0, 2.5, 0.01),
                 std::invalid_argument) << "T2 > 2*T1";
}

TEST(R1122FillFront, NoiseModelFromT1T2AndQubitSpecificLookup) {
    EXPECT_THROW(NoiseModel::from_t1_t2({1.0}, {0.5, 0.5}, {{"x", 0.01}}, {}),
                 std::invalid_argument) << "t1/t2 length mismatch";
    EXPECT_THROW(NoiseModel::from_t1_t2({1.0}, {2.5}, {{"x", 0.01}}, {}),
                 std::invalid_argument) << "T2 > 2*T1";

    // No per-gate qubit list: the error applies to every qubit.
    auto model = NoiseModel::from_t1_t2({1.0, 1.2}, {0.7, 0.9},
                                        {{"x", 0.01}}, {});
    EXPECT_FALSE(model.is_ideal());
    EXPECT_EQ(model.errors_for_gate("x", {0}).size(), 1u);
    EXPECT_EQ(model.errors_for_gate("x", {1}).size(), 1u);

    // Qubit-specific registration matches exactly that operand list.
    NoiseModel targeted;
    targeted.add_quantum_error(NoiseChannels::bit_flip(0.1), "x", {1});
    EXPECT_EQ(targeted.errors_for_gate("x", {1}).size(), 1u);
    EXPECT_TRUE(targeted.errors_for_gate("x", {0}).empty());
    EXPECT_TRUE(targeted.errors_for_gate("h", {1}).empty());
}

// =============================================================================
// Qudit MPS
// =============================================================================

TEST(R1122FillFront, QuditMpsReversedNonAdjacentPairMatchesStatevector) {
    const int n = 3, d = 3;
    const auto F = dft3();
    const auto shift = qudit_gates::shift_matrix(d, 1);
    const auto cadd = qudit_gates::cadd_matrix(d, 1);

    QuditStatevector sv(n, d);
    QuditMPS mps(n, d, 27);
    for (int q = 0; q < n; ++q) {
        sv.apply_1qudit(q, F);
        mps.apply_1qudit(q, F);
    }
    sv.apply_1qudit(1, shift);
    mps.apply_1qudit(1, shift);
    // Reversed operand order AND non-adjacent pair: exercises the digit-role
    // swap and the SWAP-chain routing.
    sv.apply_2qudit(2, 0, cadd);
    mps.apply_2qudit(2, 0, cadd);

    auto back = mps.to_statevector();
    ASSERT_EQ(back.dim, sv.dim);
    for (size_t i = 0; i < sv.dim; ++i) {
        EXPECT_NEAR(back.amplitudes[i].real, sv.amplitudes[i].real, 1e-9) << i;
        EXPECT_NEAR(back.amplitudes[i].imag, sv.amplitudes[i].imag, 1e-9) << i;
    }
}

TEST(R1122FillFront, QuditMpsValidationThrows) {
    QuditMPS mps(3, 3, 9);
    std::vector<Complex128> right_size(81, Complex128(0.0, 0.0));
    std::vector<Complex128> wrong_size(9, Complex128(0.0, 0.0));
    // Bounds violations throw std::out_of_range; structure violations
    // (non-distinct qudits, wrong matrix size) throw std::invalid_argument.
    // This mirrors the circuit layer and the harmonised checker in
    // include/lindblad/detail/validate.hpp; the qudit-MPS bounds path used to
    // throw std::invalid_argument and was realigned here (R.1.19.1).
    EXPECT_THROW(mps.apply_2qudit(1, 1, right_size), std::invalid_argument);
    EXPECT_THROW(mps.apply_2qudit(0, 3, right_size), std::out_of_range);
    EXPECT_THROW(mps.apply_2qudit(-1, 1, right_size), std::out_of_range);
    EXPECT_THROW(mps.apply_2qudit(0, 1, wrong_size), std::invalid_argument);
}

TEST(R1122FillFront, QuditMpsCanonicalFormsPreserveState) {
    const int n = 3, d = 3;
    const auto F = dft3();
    const auto cadd = qudit_gates::cadd_matrix(d, 2);

    QuditMPS mps(n, d, 27);
    mps.apply_1qudit(0, F);
    mps.apply_2qudit(0, 1, cadd);
    mps.apply_2qudit(1, 2, cadd);
    auto before = mps.to_statevector();

    mps.left_canonicalize();
    EXPECT_NEAR(mps.norm_sq(), 1.0, 1e-9);
    mps.right_canonicalize();
    EXPECT_NEAR(mps.norm_sq(), 1.0, 1e-9);

    auto after = mps.to_statevector();
    for (size_t i = 0; i < before.dim; ++i) {
        EXPECT_NEAR(after.amplitudes[i].real, before.amplitudes[i].real, 1e-9) << i;
        EXPECT_NEAR(after.amplitudes[i].imag, before.amplitudes[i].imag, 1e-9) << i;
    }
}

TEST(R1122FillFront, QuditMpsMeasureDeterministicOnBasisState) {
    QuditMPS mps(3, 3, 9);
    mps.apply_1qudit(1, qudit_gates::shift_matrix(3, 1));  // |0,1,0>
    auto digits = mps.measure(5);
    EXPECT_EQ(digits, (std::vector<int>{0, 1, 0}));
}

// =============================================================================
// Circuit serialisation utilities and DAG helpers
// =============================================================================

TEST(R1122FillFront, CircuitQasm2ExportBarrierAndSymbolicGates) {
    QuantumCircuit qc(3, 1);
    qc.barrier({0, 2});
    qc.rx("a", 0);
    Instruction pu;
    pu.type = GT::PARAM_U;
    pu.qubits = {1};
    pu.param_names = {"al", "be", "ga"};
    qc.instructions.push_back(pu);
    qc.measure(0, 0);
    qc.reset(2);

    const auto qasm = qc.to_qasm2();
    EXPECT_NE(qasm.find("barrier q[0], q[2];"), std::string::npos) << qasm;
    EXPECT_NE(qasm.find("u(al,be,ga) q[1];"), std::string::npos) << qasm;
    EXPECT_NE(qasm.find("measure q[0] -> c[0];"), std::string::npos);
    EXPECT_NE(qasm.find("reset q[2];"), std::string::npos);
}

TEST(R1122FillFront, CircuitJsonRoundTripCoversGateNameMap) {
    QuantumCircuit qc(3, 3);
    qc.h(0).x(1).y(2).z(0).s(1).sdg(2).t(0).tdg(1).sx(2).sxdg(0);
    qc.rx(0.1, 0).ry(0.2, 1).rz(0.3, 2).p(0.4, 0);
    qc.u(0.1, 0.2, 0.3, 1).u1(0.5, 2).u2(0.6, 0.7, 0).u3(0.1, 0.2, 0.3, 1);
    qc.cx(0, 1).cy(1, 2).cz(0, 2).ch(0, 1).swap(1, 2).iswap(0, 1);
    qc.crx(0.1, 0, 1).cry(0.2, 1, 2).crz(0.3, 0, 2).cp(0.4, 0, 1);
    qc.cu(0.1, 0.2, 0.3, 0.4, 1, 2).ecr(0, 1);
    qc.rzx(0.1, 0, 1).rxx(0.2, 1, 2).ryy(0.3, 0, 2).rzz(0.4, 0, 1);
    qc.ccx(0, 1, 2).ccz(0, 1, 2).cswap(0, 1, 2).rccx(0, 1, 2);
    qc.barrier();
    qc.measure(0, 0).reset(1);

    auto round = QuantumCircuit::from_json(qc.to_json());
    ASSERT_EQ(round.instructions.size(), qc.instructions.size());
    for (size_t i = 0; i < qc.instructions.size(); ++i) {
        EXPECT_EQ(round.instructions[i].type, qc.instructions[i].type) << "inst " << i;
        EXPECT_EQ(round.instructions[i].qubits, qc.instructions[i].qubits) << "inst " << i;
        ASSERT_EQ(round.instructions[i].params.size(), qc.instructions[i].params.size());
        for (size_t p = 0; p < qc.instructions[i].params.size(); ++p)
            EXPECT_NEAR(round.instructions[i].params[p], qc.instructions[i].params[p],
                        1e-12);
    }
}

TEST(R1122FillFront, DagDefaultCtorAndTwoQubitOps) {
    DAGCircuit empty;
    EXPECT_EQ(empty.n_qubits, 0);
    EXPECT_EQ(empty.n_clbits, 0);

    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cz(1, 2).ccx(0, 1, 2).x(2);
    auto dag = DAGCircuit::from_circuit(qc);
    EXPECT_EQ(dag.to_circuit().instructions.size(), qc.instructions.size());

    auto pairs = dag.two_qubit_ops();
    ASSERT_EQ(pairs.size(), 2u) << "cx and cz; ccx is 3q, h/x are 1q";
    EXPECT_EQ(pairs[0], (std::pair<int, int>{0, 1}));
    EXPECT_EQ(pairs[1], (std::pair<int, int>{1, 2}));
}

// =============================================================================
// Visualisation catalogues and HTML escaping
// =============================================================================

TEST(R1122FillFront, VizCataloguesExposeEveryEntry) {
    // 23 entries: the ten fixed 1q gates, the eight rotation/U-family gates,
    // and the five PARAM_* symbolic variants. Multi-qubit gates live in the
    // composite catalogue instead.
    const auto& sym = viz::symbol_catalogue();
    EXPECT_GE(sym.size(), 23u);
    ASSERT_TRUE(sym.count(GT::H));
    EXPECT_EQ(sym.at(GT::H).label, "H");
    ASSERT_TRUE(sym.count(GT::RX));
    EXPECT_TRUE(sym.at(GT::RX).show_params);
    ASSERT_TRUE(sym.count(GT::PARAM_U));

    const auto& comp = viz::composite_catalogue();
    EXPECT_GE(comp.size(), 20u);
    ASSERT_TRUE(comp.count(GT::CX));
    ASSERT_EQ(comp.at(GT::CX).parts.size(), 2u);
    EXPECT_EQ(comp.at(GT::CX).parts[1].role,
              viz::CompositePart::Role::XorTarget);
    ASSERT_TRUE(comp.count(GT::CCZ));
    EXPECT_EQ(comp.at(GT::CCZ).parts.size(), 3u);
    ASSERT_TRUE(comp.count(GT::RXX));
    EXPECT_EQ(comp.at(GT::RXX).parts[0].role,
              viz::CompositePart::Role::TallBox);
}

TEST(R1122FillFront, VizHtmlEscapesLabelsAndRendersConditions) {
    QuantumCircuit qc(2, 1);
    std::vector<Complex128> eye4(16, Complex128(0.0, 0.0));
    for (int i = 0; i < 4; ++i) eye4[i * 4 + i] = Complex128(1.0, 0.0);
    qc.unitary(eye4, {0, 1}, "u<1>&\"x\"");
    qc.add_if(0, 1, GT::X, {0});
    qc.measure(1, 0);

    const auto html = qc.draw(DrawMode::HTML);
    EXPECT_FALSE(html.empty());
    EXPECT_NE(html.find("&lt;"), std::string::npos)
        << "user-supplied label must be HTML-escaped";
    EXPECT_EQ(html.find("u<1>"), std::string::npos)
        << "raw '<' from a label must never reach the HTML output";

    // The conditioned gate and the same label through the other renderers.
    EXPECT_FALSE(qc.draw(DrawMode::ASCII).empty());
    EXPECT_FALSE(qc.draw(DrawMode::SVG).empty());
    EXPECT_FALSE(qc.draw(DrawMode::LATEX).empty());
}
