// =============================================================================
// R.1.17.1 test suite — density-matrix channel superoperators (R1171DmChannelSuperop)
// =============================================================================
// R.1.17.0 reworked DensityMatrix::apply_kraus into a single fused-superoperator
// sweep (operator-count independent) and added the public
// DensityMatrix::apply_channel_superop(S, qubits), which takes S in the EXTERNAL
// LSB-first convention (bit b of every sub-index addresses qubits[b], matching
// KrausChannel and apply_unitary) and size-validates it. run() precomputes each
// attached channel's internal-convention superoperator once.
//
// Everything is verified against an INDEPENDENT, test-local reference: the
// per-operator definition ρ' = Σ_k K_k ρ K_k† built here with plain
// embed/matmul/dagger, and analytic single-qubit channel endpoints. The
// superoperator formula under test is
//   S[(r_out·2^k + c_out)·4^k + (r_in·2^k + c_in)] = Σ_k K[r_out,r_in]·conj(K[c_out,c_in]),
// applied as vec(ρ_block)' = S·vec(ρ_block) per background pair — the reference
// builds S the same way and cross-checks it against the per-operator sum.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace lindblad;

namespace {

using Ops = std::vector<std::vector<Complex128>>;

// --- Dense reference machinery (n = 2..3, so O(dim³) is trivial) -------------

std::vector<Complex128> matmul(const std::vector<Complex128>& A,
                               const std::vector<Complex128>& B, size_t d) {
    std::vector<Complex128> C(d * d, Complex128(0, 0));
    for (size_t i = 0; i < d; ++i)
        for (size_t k = 0; k < d; ++k) {
            const Complex128 aik = A[i * d + k];
            for (size_t j = 0; j < d; ++j)
                C[i * d + j] += aik * B[k * d + j];
        }
    return C;
}

std::vector<Complex128> dagger(const std::vector<Complex128>& M, size_t d) {
    std::vector<Complex128> R(d * d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j)
            R[j * d + i] = M[i * d + j].conj();
    return R;
}

// Embed a k-qubit operator M (row-major 2^k × 2^k, EXTERNAL LSB-first: bit b of
// the sub-index addresses qubits[b]) into the full 2^n × 2^n register.
std::vector<Complex128> embed(const std::vector<Complex128>& M,
                              const std::vector<int>& qubits, int n) {
    const int k = static_cast<int>(qubits.size());
    const size_t sd = size_t(1) << k;
    const size_t dim = size_t(1) << n;
    size_t tmask = 0;
    for (int q : qubits) tmask |= (size_t(1) << q);
    auto subidx = [&](size_t a) {
        size_t x = 0;
        for (int b = 0; b < k; ++b)
            if ((a >> qubits[static_cast<size_t>(b)]) & 1) x |= (size_t(1) << b);
        return x;
    };
    std::vector<Complex128> F(dim * dim, Complex128(0, 0));
    for (size_t a = 0; a < dim; ++a)
        for (size_t b = 0; b < dim; ++b) {
            if ((a & ~tmask) != (b & ~tmask)) continue;  // background bits equal
            F[a * dim + b] = M[subidx(a) * sd + subidx(b)];
        }
    return F;
}

// Independent per-operator channel: ρ' = Σ_k K_k ρ K_k† (K_k embedded on qubits).
std::vector<Complex128> ref_channel(const std::vector<Complex128>& rho,
                                    const Ops& ops,
                                    const std::vector<int>& qubits, int n) {
    const size_t dim = size_t(1) << n;
    std::vector<Complex128> out(dim * dim, Complex128(0, 0));
    for (const auto& K : ops) {
        const auto Kf = embed(K, qubits, n);
        const auto Kd = dagger(Kf, dim);
        const auto t = matmul(matmul(Kf, rho, dim), Kd, dim);
        for (size_t i = 0; i < out.size(); ++i) out[i] += t[i];
    }
    return out;
}

// Channel superoperator in the EXTERNAL convention (the apply_channel_superop
// contract): S[(ro·sd+co)·sd²+(ri·sd+ci)] = Σ_k K[ro,ri]·conj(K[co,ci]).
std::vector<Complex128> build_superop_ext(const Ops& ops, int k) {
    const size_t sd = size_t(1) << k;
    const size_t sd2 = sd * sd;
    std::vector<Complex128> S(sd2 * sd2, Complex128(0, 0));
    for (const auto& K : ops)
        for (size_t ro = 0; ro < sd; ++ro)
            for (size_t co = 0; co < sd; ++co)
                for (size_t ri = 0; ri < sd; ++ri)
                    for (size_t ci = 0; ci < sd; ++ci)
                        S[(ro * sd + co) * sd2 + (ri * sd + ci)] +=
                            K[ro * sd + ri] * K[co * sd + ci].conj();
    return S;
}

// LSB-first tensor A ⊗ B where A acts on operand 0 (bit 0) and B on operand 1
// (bit 1): full index = a0 + 2·a1, so M[(r0+2r1),(c0+2c1)] = A[r0,c0]·B[r1,c1].
std::vector<Complex128> kron_lsb(const std::vector<Complex128>& A,
                                 const std::vector<Complex128>& B) {
    std::vector<Complex128> R(16);
    for (int r0 = 0; r0 < 2; ++r0)
        for (int r1 = 0; r1 < 2; ++r1)
            for (int c0 = 0; c0 < 2; ++c0)
                for (int c1 = 0; c1 < 2; ++c1)
                    R[(r0 + 2 * r1) * 4 + (c0 + 2 * c1)] =
                        A[r0 * 2 + c0] * B[r1 * 2 + c1];
    return R;
}

const std::vector<Complex128> kIdentity2 = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};

// Prepare a density matrix from the pure state a small circuit builds from |0…0⟩.
DensityMatrix prep_dm(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    auto res = sim.run(qc, 0, 0);
    return DensityMatrix::from_statevector(res.final_state);
}

void expect_mat_close(const std::vector<Complex128>& got,
                      const std::vector<Complex128>& want, double tol = 1e-12) {
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i].real, want[i].real, tol) << "re @ " << i;
        EXPECT_NEAR(got[i].imag, want[i].imag, tol) << "im @ " << i;
    }
}

void expect_hermitian(const DensityMatrix& dm, double tol = 1e-12) {
    for (size_t i = 0; i < dm.dim; ++i)
        for (size_t j = i + 1; j < dm.dim; ++j) {
            EXPECT_NEAR(dm(i, j).real, dm(j, i).real, tol) << "@ " << i << "," << j;
            EXPECT_NEAR(dm(i, j).imag, -dm(j, i).imag, tol) << "@ " << i << "," << j;
        }
}

}  // namespace

// -----------------------------------------------------------------------------
// apply_kraus vs the per-operator reference, and the non-symmetric convention
// -----------------------------------------------------------------------------

// 1. 1q amplitude damping (γ = 0.3) on qubit 0 of n = 2 matches the per-operator
//    reference, and the NON-SYMMETRIC endpoint pins LSB-at-qubit-0: starting
//    from the qubit-0-excited state (index 1), the excited population decays;
//    the mirror on qubit 1 (ground there) leaves it untouched.
TEST(R1171DmChannelSuperop, AmplitudeDampingConventionNonSymmetric) {
    const double gamma = 0.3;
    const auto ad = NoiseChannels::amplitude_damping(gamma);

    // |index 1⟩ = q1 q0 = |01⟩ bitstring: qubit 0 (the 1's place) excited.
    QuantumCircuit prep(2);
    prep.x(0);
    DensityMatrix dm0 = prep_dm(prep);

    DensityMatrix dm = dm0;  // copy
    dm.apply_kraus(ad.operators, {0});
    expect_mat_close(dm.data, ref_channel(dm0.data, ad.operators, {0}, 2));
    // Excited qubit-0 population decays: ρ11 = 1-γ, ρ00 = γ.
    EXPECT_NEAR(dm(1, 1).real, 1.0 - gamma, 1e-12);
    EXPECT_NEAR(dm(0, 0).real, gamma, 1e-12);

    // Mirror on qubit 1 (ground in |01⟩): the state is unchanged.
    DensityMatrix dmm = dm0;
    dmm.apply_kraus(ad.operators, {1});
    expect_mat_close(dmm.data, ref_channel(dm0.data, ad.operators, {1}, 2));
    EXPECT_NEAR(dmm(1, 1).real, 1.0, 1e-12);   // index 1 population intact
    EXPECT_NEAR(dmm(0, 0).real, 0.0, 1e-12);
}

// 2. 1q phase damping and 1q depolarizing on both qubits of n = 2: reference match.
TEST(R1171DmChannelSuperop, PhaseDampingAndDepolarizingReferenceMatch) {
    const auto pd = NoiseChannels::phase_damping(0.4);
    const auto dep = NoiseChannels::depolarizing(0.2, 1);

    QuantumCircuit prep(2);
    prep.h(0).h(1).rz(0.6, 0).ry(0.9, 1);  // a generic 2-qubit pure state
    const DensityMatrix dm0 = prep_dm(prep);

    for (int q = 0; q < 2; ++q) {
        DensityMatrix a = dm0;
        a.apply_kraus(pd.operators, {q});
        expect_mat_close(a.data, ref_channel(dm0.data, pd.operators, {q}, 2));

        DensityMatrix b = dm0;
        b.apply_kraus(dep.operators, {q});
        expect_mat_close(b.data, ref_channel(dm0.data, dep.operators, {q}, 2));
    }
}

// 3. 2q depolarizing (16 operators) on an ADJACENT pair (q0,q1) and a
//    NON-adjacent pair (q0,q2) of n = 3: the k = 2 stride / bit-reversal bridge
//    must reproduce the per-operator reference.
TEST(R1171DmChannelSuperop, TwoQubitDepolarizingAdjacentAndNonAdjacent) {
    const auto dep2 = NoiseChannels::depolarizing(0.1, 2);
    ASSERT_EQ(dep2.operators.size(), 16u);

    QuantumCircuit prep(3);
    prep.h(0).h(1).h(2).cx(0, 1).rz(0.5, 2).ry(0.7, 0);
    const DensityMatrix dm0 = prep_dm(prep);

    for (const std::vector<int>& pair :
         {std::vector<int>{0, 1}, std::vector<int>{0, 2}}) {
        DensityMatrix dm = dm0;
        dm.apply_kraus(dep2.operators, pair);
        expect_mat_close(dm.data, ref_channel(dm0.data, dep2.operators, pair, 3));
    }
}

// 4. Reversed qubit order with an ASYMMETRIC 2q channel (amplitude damping on
//    operand 0, identity on operand 1): applying on (q0,q2) vs (q2,q0) damps
//    different qubits, so the results DIFFER exactly as the convention predicts
//    (first listed operand is least significant), each matching its reference.
TEST(R1171DmChannelSuperop, ReversedQubitOrderAsymmetricChannel) {
    const auto ad = NoiseChannels::amplitude_damping(0.5);
    Ops chan;  // {K⊗I} : damping on operand 0 only
    for (const auto& K : ad.operators) chan.push_back(kron_lsb(K, kIdentity2));

    QuantumCircuit prep(3);
    prep.x(0).h(2).rz(0.3, 2);  // q0 excited, q2 in a superposition, q1 = |0⟩
    const DensityMatrix dm0 = prep_dm(prep);

    DensityMatrix a = dm0;  // operand 0 = q0
    a.apply_kraus(chan, {0, 2});
    expect_mat_close(a.data, ref_channel(dm0.data, chan, {0, 2}, 3));

    DensityMatrix b = dm0;  // operand 0 = q2
    b.apply_kraus(chan, {2, 0});
    expect_mat_close(b.data, ref_channel(dm0.data, chan, {2, 0}, 3));

    // The two must genuinely differ (they damp different physical qubits).
    double max_diff = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        max_diff = std::max(max_diff, (a.data[i] - b.data[i]).norm());
    EXPECT_GT(max_diff, 1e-3);
}

// 5. Trace and Hermiticity preservation across the CPTP channels above.
TEST(R1171DmChannelSuperop, TracePreservationAndHermiticity) {
    QuantumCircuit prep(2);
    prep.h(0).ry(0.8, 1).cx(0, 1).rz(0.4, 0);
    const DensityMatrix dm0 = prep_dm(prep);

    const Ops channels_1q_q0[] = {
        NoiseChannels::amplitude_damping(0.35).operators,
        NoiseChannels::phase_damping(0.5).operators,
        NoiseChannels::depolarizing(0.25, 1).operators,
    };
    for (const auto& ops : channels_1q_q0) {
        DensityMatrix dm = dm0;
        dm.apply_kraus(ops, {0});
        EXPECT_NEAR(dm.trace(), 1.0, 1e-12);
        expect_hermitian(dm);
    }
    // A 2q CPTP channel too.
    DensityMatrix dm2 = dm0;
    dm2.apply_kraus(NoiseChannels::depolarizing(0.3, 2).operators, {0, 1});
    EXPECT_NEAR(dm2.trace(), 1.0, 1e-12);
    expect_hermitian(dm2);
}

// 6. Empty-operator channel. RED, issue #76: a Kraus set with no operators is
//    applied as an all-zero superoperator, which zeroes the addressed
//    sub-blocks and takes the trace to 0. A state whose trace is 0 is not a
//    state, and it is produced by a call that reported success.
//
//    Sum over no operators is the zero matrix, so the trace-preservation
//    residual is exactly 1: this is the most non-trace-preserving channel that
//    exists, not a borderline one. Nothing rejects it. check_qubits and
//    check_all_distinct pass, the per-operator size loop has no iterations, and
//    check_kraus_tp returns early on ops.empty().
TEST(R1171DmChannelSuperop, EmptyOperatorChannelIsRejected) {
    QuantumCircuit prep(2);
    prep.h(0).h(1);
    DensityMatrix dm = prep_dm(prep);
    ASSERT_NEAR(dm.trace(), 1.0, 1e-12);

    EXPECT_THROW(dm.apply_kraus(Ops{}, {0}), std::invalid_argument)
        << "issue #76: a channel with no operators was accepted";
    EXPECT_NEAR(dm.trace(), 1.0, 1e-12)
        << "issue #76: the state was annihilated; trace is now " << dm.trace();
}

// 7. RESET instruction path (a 2-operator 1q channel through apply_kraus at
//    run time): the final ρ of a reset-containing circuit matches the analytic
//    result. reset is a deterministic channel here (single-pass, no collapse).
TEST(R1171DmChannelSuperop, ResetInstructionPathAnalytic) {
    QuantumCircuit qc(2);
    qc.h(0).x(1).reset(1);  // q1 driven to |1⟩ then reset → |0⟩; q0 = |+⟩
    NoiseModel ideal;
    const auto res = DensityMatrixSimulator().run(qc, ideal, /*shots=*/0, /*seed=*/0);
    ASSERT_TRUE(res.success);

    // Analytic: |+⟩_q0 ⊗ |0⟩_q1 = H(0)|00⟩.
    QuantumCircuit expect_qc(2);
    expect_qc.h(0);
    const DensityMatrix want = prep_dm(expect_qc);
    expect_mat_close(res.final_state.data, want.data);
}

// -----------------------------------------------------------------------------
// apply_channel_superop — public superoperator entry point
// -----------------------------------------------------------------------------

// 8. apply_channel_superop(S) == apply_kraus(ops) for the same channel, at
//    k = 1 and k = 2, when S is built in the EXTERNAL convention on the test side.
TEST(R1171DmChannelSuperop, SuperopMatchesKraus) {
    QuantumCircuit prep(3);
    prep.h(0).h(1).h(2).rz(0.4, 0).cx(1, 2).ry(0.6, 1);
    const DensityMatrix dm0 = prep_dm(prep);

    // k = 1: amplitude damping on q1.
    {
        const auto ad = NoiseChannels::amplitude_damping(0.4);
        DensityMatrix viakraus = dm0;
        viakraus.apply_kraus(ad.operators, {1});
        DensityMatrix viasuperop = dm0;
        viasuperop.apply_channel_superop(build_superop_ext(ad.operators, 1), {1});
        expect_mat_close(viasuperop.data, viakraus.data);
    }
    // k = 2: 2q depolarizing on the non-adjacent pair (q0,q2).
    {
        const auto dep2 = NoiseChannels::depolarizing(0.1, 2);
        DensityMatrix viakraus = dm0;
        viakraus.apply_kraus(dep2.operators, {0, 2});
        DensityMatrix viasuperop = dm0;
        viasuperop.apply_channel_superop(build_superop_ext(dep2.operators, 2), {0, 2});
        expect_mat_close(viasuperop.data, viakraus.data);
    }
}

// 9. apply_channel_superop size validation: a wrong-dimension S throws loudly
//    (its documented contract). Qubit-range guarding is intentionally NOT this
//    primitive's job — it lives at the circuit-building layer (QuantumCircuit::*
//    validate qubit indices); the low-level apply-primitives (apply_gate,
//    apply_kraus, gates::apply_*) all validate operand SIZE only, and
//    apply_channel_superop is consistent with them. So only the size contract
//    is asserted here; a correctly-sized S on a valid target must not throw.
TEST(R1171DmChannelSuperop, SuperopSizeValidation) {
    DensityMatrix dm(2);
    // 2-qubit target needs a 16×16 = 256-element S; a 4×4 = 16-element S is wrong.
    const std::vector<Complex128> too_small(16, Complex128(0, 0));
    EXPECT_THROW(dm.apply_channel_superop(too_small, {0, 1}), std::invalid_argument);

    // 1-qubit target needs a 4×4 = 16-element S; a 256-element S is wrong.
    const std::vector<Complex128> too_big(256, Complex128(0, 0));
    EXPECT_THROW(dm.apply_channel_superop(too_big, {0}), std::invalid_argument);

    // Correctly-sized identity superop on a valid target: no throw.
    EXPECT_NO_THROW(dm.apply_channel_superop(build_superop_ext({kIdentity2}, 1), {0}));
}

// 10. Identity-channel superop leaves an arbitrary ρ exactly unchanged (S = I⊗I
//     path) — catches an accidental transpose or conjugation in the bridge.
TEST(R1171DmChannelSuperop, IdentitySuperopLeavesRhoUnchanged) {
    QuantumCircuit prep(3);
    prep.h(0).ry(0.7, 1).cx(0, 1).rz(0.9, 2).cx(1, 2).rx(0.3, 0);
    const DensityMatrix dm0 = prep_dm(prep);

    // k = 1 identity on q1.
    DensityMatrix a = dm0;
    a.apply_channel_superop(build_superop_ext({kIdentity2}, 1), {1});
    expect_mat_close(a.data, dm0.data);

    // k = 2 identity (I⊗I) on (q0,q2).
    DensityMatrix b = dm0;
    b.apply_channel_superop(build_superop_ext({kron_lsb(kIdentity2, kIdentity2)}, 2),
                            {0, 2});
    expect_mat_close(b.data, dm0.data);
}

// 11. Non-symmetric convention through the PUBLIC superop API: a single-qubit
//     damping channel embedded at qubit 0 vs qubit 1 of n = 2 (mirror of test 1).
TEST(R1171DmChannelSuperop, SuperopConventionNonSymmetric) {
    const double gamma = 0.4;
    const auto ad = NoiseChannels::amplitude_damping(gamma);
    const auto S = build_superop_ext(ad.operators, 1);

    QuantumCircuit prep(2);
    prep.x(0);  // qubit 0 excited (index 1)
    const DensityMatrix dm0 = prep_dm(prep);

    DensityMatrix on0 = dm0;
    on0.apply_channel_superop(S, {0});
    EXPECT_NEAR(on0(1, 1).real, 1.0 - gamma, 1e-12);  // excited-side decays
    EXPECT_NEAR(on0(0, 0).real, gamma, 1e-12);

    DensityMatrix on1 = dm0;
    on1.apply_channel_superop(S, {1});
    EXPECT_NEAR(on1(1, 1).real, 1.0, 1e-12);           // qubit 1 ground: untouched
    EXPECT_NEAR(on1(0, 0).real, 0.0, 1e-12);
}

// -----------------------------------------------------------------------------
// Noisy run() integration
// -----------------------------------------------------------------------------

// 12. Noisy run() evolve-once path: a terminal-measure circuit with a
//     2q-depolarizing noise model. Seeded counts reproduce EXACTLY across two
//     identical run() calls, and the pre-sampling ρ (shots = 0) matches the
//     per-operator reference evolution at 1e-12.
TEST(R1171DmChannelSuperop, NoisyRunEvolveOnceDeterminismAndReference) {
    const auto dep2 = NoiseChannels::depolarizing(0.05, 2);
    NoiseModel noise;
    noise.add_all_qubit_quantum_error(dep2, "cx");  // after every cx

    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1);

    // Seeded-counts determinism (two identical calls agree exactly).
    QuantumCircuit qm = qc;
    qm.measure_all();
    const auto c1 = DensityMatrixSimulator().run(qm, noise, 4096, 13579).counts;
    const auto c2 = DensityMatrixSimulator().run(qm, noise, 4096, 13579).counts;
    EXPECT_EQ(c1, c2);

    // Pre-sampling ρ matches the independent reference: ideal (h,cx) state, then
    // the depolarizing channel applied per-operator on the cx qubits.
    const auto pre = DensityMatrixSimulator().run(qc, NoiseModel{}, 0, 0).final_state;
    const auto want = ref_channel(pre.data, dep2.operators, {0, 1}, 2);
    const auto noisy = DensityMatrixSimulator().run(qc, noise, 0, 0).final_state;
    expect_mat_close(noisy.data, want);
}

// 13. Noisy run() per-shot path: a feedforward circuit + noise. Seeded counts
//     are identical across two runs (the plan-time superops on the trajectory
//     path are deterministic). The measured qubit sits near a basis state, so
//     the branch taken never rides an RNG boundary.
TEST(R1171DmChannelSuperop, NoisyRunPerShotDeterminism) {
    NoiseModel noise;
    noise.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.03, 1), "h");

    QuantumCircuit qc(2, 2);
    qc.ry(0.2, 0);                                    // q0 ≈ |0⟩
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});   // feedforward
    qc.h(1);
    qc.measure(1, 1);

    const auto a = DensityMatrixSimulator().run(qc, noise, 1024, 246810).counts;
    const auto b = DensityMatrixSimulator().run(qc, noise, 1024, 246810).counts;
    EXPECT_EQ(a, b);
}

// 14. Operator-count independence: a 1q channel given as 16 redundant Kraus
//     operators (each of the 4 depolarizing ops split into 4 half-weight copies)
//     equals the 4-operator form at 1e-12 — the superoperator fusion is
//     decomposition-invariant.
TEST(R1171DmChannelSuperop, OperatorCountIndependence) {
    const auto dep = NoiseChannels::depolarizing(0.2, 1);
    ASSERT_EQ(dep.operators.size(), 4u);

    // Split each K into 4 copies of K/2: 4·(K/2)ρ(K/2)† = KρK†, same channel.
    Ops padded;
    for (const auto& K : dep.operators) {
        std::vector<Complex128> half(4);
        for (int i = 0; i < 4; ++i) half[i] = K[i] * 0.5;
        for (int c = 0; c < 4; ++c) padded.push_back(half);
    }
    ASSERT_EQ(padded.size(), 16u);

    QuantumCircuit prep(2);
    prep.h(0).ry(0.5, 1).cx(0, 1);
    const DensityMatrix dm0 = prep_dm(prep);

    DensityMatrix four = dm0;
    four.apply_kraus(dep.operators, {0});
    DensityMatrix sixteen = dm0;
    sixteen.apply_kraus(padded, {0});
    expect_mat_close(sixteen.data, four.data);
}

// 15. k = 2 channel on the TOP qubit pair (q1,q2) of n = 3: sub-block offset
//     arithmetic at the high end must still reproduce the per-operator reference.
TEST(R1171DmChannelSuperop, TwoQubitChannelTopPair) {
    const auto dep2 = NoiseChannels::depolarizing(0.15, 2);
    QuantumCircuit prep(3);
    prep.h(0).h(1).h(2).cx(1, 2).rz(0.6, 0).ry(0.4, 1);
    const DensityMatrix dm0 = prep_dm(prep);

    DensityMatrix dm = dm0;
    dm.apply_kraus(dep2.operators, {1, 2});
    expect_mat_close(dm.data, ref_channel(dm0.data, dep2.operators, {1, 2}, 3));
}
