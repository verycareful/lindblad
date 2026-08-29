// R.1.13.1 test patch — structured multi-controlled / permutation operations.
// Covers the R.1.13.0 feature (audit F-7 / F-9): the MCX / MCP / PERMUTATION
// instruction family and its native (statevector, density-matrix) and
// fallback (MPS) execution, plus the circuit-level builders, inverse(),
// control(), gate_name, and the export contract (updated when the ops became
// representable: QASM 2 throws by default with an explicit opt-out, QASM 3
// and JSON succeed; see the R1181 suites for the deep coverage).
//
// Reference strategy: every structured op is compared against an independent,
// hand-written brute-force reference over an arbitrary NON-symmetric input
// state, so a convention bug (LSB ordering, control mask, permutation
// direction) cannot hide behind a symmetric test.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 1e-12;

// A deterministic, all-distinct, non-symmetric amplitude vector for nq qubits.
// Not normalised: the structured ops are exact amplitude permutations / phase
// multiplies, so normalisation is irrelevant and distinct values expose any
// index mistake.
std::vector<Complex128> distinct_state(int nq) {
    const size_t dim = size_t(1) << nq;
    std::vector<Complex128> v(dim);
    for (size_t k = 0; k < dim; ++k)
        v[k] = Complex128(0.1 * static_cast<double>(k + 1),
                          -0.05 * static_cast<double>(k + 2));
    return v;
}

// Reference MCX: flip `target` bit on every basis state whose control bits are
// all 1. |k> -> |k ^ (1<<target)> when (k & cmask) == cmask.
std::vector<Complex128> ref_mcx(const std::vector<Complex128>& in,
                                const std::vector<int>& controls, int target) {
    size_t cmask = 0;
    for (int c : controls) cmask |= (size_t(1) << c);
    const size_t tmask = size_t(1) << target;
    std::vector<Complex128> out(in.size());
    for (size_t k = 0; k < in.size(); ++k) {
        // Control bits are unchanged by flipping the target, so the source of
        // out[k] is k^tmask exactly when the control bits of k are all set.
        out[k] = ((k & cmask) == cmask) ? in[k ^ tmask] : in[k];
    }
    return out;
}

// Reference MCP: multiply amplitude by exp(i*lambda) when all listed qubits =1.
std::vector<Complex128> ref_mcp(const std::vector<Complex128>& in,
                                const std::vector<int>& qubits, double lambda) {
    size_t mask = 0;
    for (int q : qubits) mask |= (size_t(1) << q);
    const Complex128 ph = Complex128::exp_i(lambda);
    std::vector<Complex128> out = in;
    for (size_t k = 0; k < in.size(); ++k)
        if ((k & mask) == mask) out[k] = in[k] * ph;
    return out;
}

// Reference permutation on the k target qubits: |x> -> |perm[x]> within the
// target subspace (LSB = qubits[0]); untouched qubits keep their value.
std::vector<Complex128> ref_perm(const std::vector<Complex128>& in,
                                 const std::vector<int>& qubits,
                                 const std::vector<int>& perm) {
    const int k = static_cast<int>(qubits.size());
    std::vector<Complex128> out(in.size(), Complex128(0.0, 0.0));
    for (size_t idx = 0; idx < in.size(); ++idx) {
        size_t sub = 0;
        for (int i = 0; i < k; ++i)
            if (idx & (size_t(1) << qubits[i])) sub |= (size_t(1) << i);
        const size_t nsub = static_cast<size_t>(perm[sub]);
        size_t nidx = idx;
        for (int i = 0; i < k; ++i) {
            const size_t bit = size_t(1) << qubits[i];
            if (nsub & (size_t(1) << i)) nidx |= bit; else nidx &= ~bit;
        }
        out[nidx] = in[idx];
    }
    return out;
}

void expect_amps_close(const std::vector<Complex128>& a,
                       const std::vector<Complex128>& b, double tol = kTol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, tol) << "re mismatch at " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, tol) << "im mismatch at " << i;
    }
}

// Build a non-trivial pure state on nq qubits via a fixed gate recipe, returning
// the resulting statevector amplitudes. Used as a common input for backend
// cross-checks (SV vs DM vs MPS).
QuantumCircuit prep_circuit(int nq) {
    QuantumCircuit qc(nq);
    for (int q = 0; q < nq; ++q) qc.h(q);
    for (int q = 0; q + 1 < nq; ++q) qc.t(q);
    qc.rz(0.7, 0).ry(1.1, nq - 1);
    return qc;
}

std::vector<Complex128> run_sv(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 0);
    return res.final_state.amplitudes();
}

} // namespace

// =============================================================================
// Native statevector kernels vs brute-force reference
// =============================================================================

TEST(R1131StructuredOps, ApplyMcxMatchesReference) {
    for (const std::vector<int>& controls :
         {std::vector<int>{}, std::vector<int>{0}, std::vector<int>{0, 2},
          std::vector<int>{0, 1, 2}}) {
        const int nq = 4;
        auto in = distinct_state(nq);
        const int target = 3;
        Statevector sv(nq);
        sv.set_amplitudes(in, {Validation::Ignore});
        gates::apply_mcx(sv, controls, target);
        expect_amps_close(sv.amplitudes(), ref_mcx(in, controls, target));
    }
}

TEST(R1131StructuredOps, ApplyMcpMatchesReference) {
    const int nq = 4;
    auto in = distinct_state(nq);
    const std::vector<int> qubits{0, 1, 3};
    const double lambda = 0.9123;
    Statevector sv(nq);
    sv.set_amplitudes(in, {Validation::Ignore});
    gates::apply_mcp(sv, qubits, lambda);
    expect_amps_close(sv.amplitudes(), ref_mcp(in, qubits, lambda));
}

TEST(R1131StructuredOps, ApplyPermutationMatchesReference) {
    const int nq = 4;
    auto in = distinct_state(nq);
    const std::vector<int> qubits{1, 3};            // 2-qubit target subspace
    const std::vector<int> perm{2, 0, 3, 1};        // a non-identity bijection
    Statevector sv(nq);
    sv.set_amplitudes(in, {Validation::Ignore});
    gates::apply_permutation(sv, qubits, perm);
    expect_amps_close(sv.amplitudes(), ref_perm(in, qubits, perm));
}

TEST(R1131StructuredOps, PermutationSingleQubitEqualsX) {
    // perm {1,0} on one qubit is exactly the X gate — a minimal convention check.
    const int nq = 3;
    auto in = distinct_state(nq);
    Statevector a(nq), b(nq);
    a.set_amplitudes(in, {Validation::Ignore});
    b.set_amplitudes(in, {Validation::Ignore});
    gates::apply_permutation(a, {1}, {1, 0});
    gates::apply_x(b, 1);
    expect_amps_close(a.amplitudes(), b.amplitudes());
}

TEST(R1131StructuredOps, McxTwoControlsEqualsCcx) {
    const int nq = 3;
    auto in = distinct_state(nq);
    Statevector a(nq), b(nq);
    a.set_amplitudes(in, {Validation::Ignore});
    b.set_amplitudes(in, {Validation::Ignore});
    gates::apply_mcx(a, {0, 1}, 2);
    gates::apply_ccx(b, 0, 1, 2);
    expect_amps_close(a.amplitudes(), b.amplitudes());
}

// =============================================================================
// Statevector-simulator dispatch (circuit path exercises the GateType switch)
// =============================================================================

TEST(R1131StructuredOps, SimulatorDispatchMcxMcpPermutation) {
    const int nq = 4;
    QuantumCircuit qc = prep_circuit(nq);
    qc.mcx({0, 1}, 2);
    qc.mcp(0.6, {1, 3});
    qc.permute({2, 0, 3, 1}, {0, 2});
    auto got = run_sv(qc);

    // Independent reference: prep amplitudes, then apply each op via the
    // brute-force kernels in sequence.
    auto ref = run_sv(prep_circuit(nq));
    ref = ref_mcx(ref, {0, 1}, 2);
    ref = ref_mcp(ref, {1, 3}, 0.6);
    ref = ref_perm(ref, {0, 2}, {2, 0, 3, 1});
    expect_amps_close(got, ref);
}

// =============================================================================
// Density-matrix native path (apply_permutation / apply_mcp_phase)
// =============================================================================

TEST(R1131StructuredOps, DensityMatrixMatchesStatevector) {
    const int nq = 3;
    QuantumCircuit qc = prep_circuit(nq);
    qc.mcx({0, 1}, 2);
    qc.mcp(0.4, {0, 2});

    auto sv_amps = run_sv(qc);

    DensityMatrixSimulator dm_sim;
    NoiseModel ideal;
    auto dm_res = dm_sim.run(qc, ideal, /*shots=*/0, /*seed=*/0);
    const DensityMatrix& rho = dm_res.final_state;

    for (size_t i = 0; i < sv_amps.size(); ++i)
        for (size_t j = 0; j < sv_amps.size(); ++j) {
            Complex128 expected = sv_amps[i] * sv_amps[j].conj();
            EXPECT_NEAR(rho(i, j).real, expected.real, 1e-9);
            EXPECT_NEAR(rho(i, j).imag, expected.imag, 1e-9);
        }
}

// =============================================================================
// MPS fallback: wide MCX (>2 controls), MCP and PERMUTATION take the bounded
// statevector fallback and must reproduce the exact statevector result. This is
// the Shor-on-MPS regression guard (R.1.13.0 correctness fix).
// =============================================================================

TEST(R1131StructuredOps, MpsFallbackMatchesStatevector) {
    const int nq = 4;
    QuantumCircuit qc = prep_circuit(nq);
    qc.mcx({0, 1, 2}, 3);          // 3 controls -> wide, statevector fallback
    qc.mcp(0.5, {0, 1, 3});
    qc.permute({2, 0, 3, 1}, {1, 3});

    auto sv_amps = run_sv(qc);

    MPSSimulator mps_sim;
    auto mps_res = mps_sim.run(qc, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/0);
    auto mps_amps = mps_res.final_state.to_statevector().amplitudes();

    expect_amps_close(mps_amps, sv_amps, 1e-9);
}

TEST(R1131StructuredOps, MpsNarrowMcxUsesGateLadder) {
    // <=2 control MCX lowers to X/CX/CCX on MPS (no fallback). Correctness must
    // still match the statevector backend.
    const int nq = 3;
    QuantumCircuit qc = prep_circuit(nq);
    qc.mcx({0, 1}, 2);

    auto sv_amps = run_sv(qc);
    MPSSimulator mps_sim;
    auto mps_res = mps_sim.run(qc, 64, 0, 0);
    expect_amps_close(mps_res.final_state.to_statevector().amplitudes(), sv_amps,
                      1e-9);
}

// =============================================================================
// Builders, field population, and validation
// =============================================================================

TEST(R1131StructuredOps, BuildersPopulateFields) {
    QuantumCircuit qc(4);
    qc.mcx({0, 2}, 3);
    qc.mcp(0.75, {1, 2});
    qc.permute({1, 0, 3, 2}, {0, 1}, "swap01");

    ASSERT_EQ(qc.instructions.size(), 3u);

    const auto& mcx = qc.instructions[0];
    EXPECT_EQ(mcx.type, Instruction::GateType::MCX);
    EXPECT_EQ(mcx.qubits, (std::vector<int>{0, 2, 3}));  // controls..., target
    EXPECT_EQ(mcx.gate_name(), "mcx");

    const auto& mcp = qc.instructions[1];
    EXPECT_EQ(mcp.type, Instruction::GateType::MCP);
    EXPECT_EQ(mcp.qubits, (std::vector<int>{1, 2}));
    ASSERT_EQ(mcp.params.size(), 1u);
    EXPECT_NEAR(mcp.params[0], 0.75, kTol);
    EXPECT_EQ(mcp.gate_name(), "mcp");

    const auto& perm = qc.instructions[2];
    EXPECT_EQ(perm.type, Instruction::GateType::PERMUTATION);
    EXPECT_EQ(perm.permutation, (std::vector<int>{1, 0, 3, 2}));
    EXPECT_EQ(perm.qubits, (std::vector<int>{0, 1}));
    EXPECT_EQ(perm.gate_name(), "swap01");  // label used when present
}

TEST(R1131StructuredOps, PermuteRejectsNonBijection) {
    QuantumCircuit qc(2);
    // size 4 but {0,1,1,3} is not a bijection (2 missing, 1 doubled).
    EXPECT_THROW(qc.permute({0, 1, 1, 3}, {0, 1}), std::invalid_argument);
    // wrong size (must be 2^k)
    EXPECT_THROW(qc.permute({0, 1, 2}, {0, 1}), std::invalid_argument);
    // out-of-range image
    EXPECT_THROW(qc.permute({0, 1, 2, 9}, {0, 1}), std::invalid_argument);
}

TEST(R1131StructuredOps, McxRejectsControlEqualsTarget) {
    QuantumCircuit qc(3);
    EXPECT_THROW(qc.mcx({0, 1}, 1), std::invalid_argument);
}

TEST(R1131StructuredOps, McpRejectsEmptyQubits) {
    QuantumCircuit qc(2);
    EXPECT_THROW(qc.mcp(0.5, {}), std::invalid_argument);
}

// =============================================================================
// inverse() semantics for the structured family
// =============================================================================

TEST(R1131StructuredOps, InverseMcxIsSelfInverse) {
    QuantumCircuit qc(3);
    qc.mcx({0, 1}, 2);
    auto inv = qc.inverse();
    ASSERT_EQ(inv.instructions.size(), 1u);
    EXPECT_EQ(inv.instructions[0].type, Instruction::GateType::MCX);
    EXPECT_EQ(inv.instructions[0].qubits, (std::vector<int>{0, 1, 2}));
}

TEST(R1131StructuredOps, InverseMcpNegatesPhase) {
    QuantumCircuit qc(2);
    qc.mcp(0.83, {0, 1});
    auto inv = qc.inverse();
    ASSERT_EQ(inv.instructions.size(), 1u);
    EXPECT_EQ(inv.instructions[0].type, Instruction::GateType::MCP);
    ASSERT_EQ(inv.instructions[0].params.size(), 1u);
    EXPECT_NEAR(inv.instructions[0].params[0], -0.83, kTol);
}

TEST(R1131StructuredOps, InversePermutationInvertsMap) {
    const std::vector<int> perm{2, 0, 3, 1};
    QuantumCircuit qc(2);
    qc.permute(perm, {0, 1});
    auto inv = qc.inverse();
    ASSERT_EQ(inv.instructions.size(), 1u);
    const auto& iperm = inv.instructions[0].permutation;
    ASSERT_EQ(iperm.size(), perm.size());
    // perm[x] = y  <=>  iperm[y] = x
    for (size_t x = 0; x < perm.size(); ++x)
        EXPECT_EQ(iperm[static_cast<size_t>(perm[x])], static_cast<int>(x));

    // And perm followed by its inverse is the identity on the state.
    const int nq = 2;
    auto in = distinct_state(nq);
    Statevector sv(nq);
    sv.set_amplitudes(in, {Validation::Ignore});
    gates::apply_permutation(sv, {0, 1}, perm);
    gates::apply_permutation(sv, {0, 1}, iperm);
    expect_amps_close(sv.amplitudes(), in);
}

// =============================================================================
// control() on a structured op (generic materialise-and-control path)
// =============================================================================

TEST(R1131StructuredOps, ControlOfMcxEqualsCcx) {
    // MCX({0},1) is CX(0,1). Adding one control (new control at qubit 0, the
    // gate's qubits shift up by 1) must equal CCX(0,1,2).
    QuantumCircuit base(2);
    base.mcx({0}, 1);
    auto controlled = base.control(1);
    ASSERT_EQ(controlled.n_qubits, 3);

    QuantumCircuit ref(3);
    ref.ccx(0, 1, 2);

    auto got = Operator::from_circuit(controlled).data;
    auto want = Operator::from_circuit(ref).data;
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i].real, want[i].real, 1e-9);
        EXPECT_NEAR(got[i].imag, want[i].imag, 1e-9);
    }
}

// =============================================================================
// Export contract (updated for the MCX/MCP/PERMUTATION representability
// feature): QASM 2 still throws by DEFAULT (no faithful encoding exists) but
// lowers on the explicit opt-out; QASM 3 and JSON now succeed. Deep coverage
// of the new behaviour lives in the R1181Export suite; this pin holds the
// contract shape.
// =============================================================================

TEST(R1131StructuredOps, QasmAndJsonExportContract) {
    QasmExportOptions decompose;
    decompose.decompose_unrepresentable = true;
    {
        QuantumCircuit qc(3);
        qc.mcx({0, 1}, 2);
        EXPECT_THROW(qc.to_qasm2(), std::runtime_error);
        EXPECT_NO_THROW(qc.to_qasm2(decompose));
        EXPECT_NO_THROW(qc.to_qasm3());
        EXPECT_NO_THROW(qc.to_json());
    }
    {
        QuantumCircuit qc(2);
        qc.mcp(0.5, {0, 1});
        EXPECT_THROW(qc.to_qasm2(), std::runtime_error);
        EXPECT_NO_THROW(qc.to_qasm2(decompose));
        EXPECT_NO_THROW(qc.to_qasm3());
        EXPECT_NO_THROW(qc.to_json());
    }
    {
        QuantumCircuit qc(2);
        qc.permute({1, 0, 3, 2}, {0, 1});
        EXPECT_THROW(qc.to_qasm2(), std::runtime_error);
        EXPECT_NO_THROW(qc.to_qasm2(decompose));
        EXPECT_NO_THROW(qc.to_qasm3());
        EXPECT_NO_THROW(qc.to_json());
    }
}
