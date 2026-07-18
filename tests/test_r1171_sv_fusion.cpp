// =============================================================================
// R.1.17.1 test suite — statevector gate fusion (R1171SvFusion)
// =============================================================================
// R.1.17.0 added a gate-fusion pre-pass to StatevectorSimulator::run(): runs of
// consecutive fusable gates whose combined support stays within
// Options::fusion_max_qubit are composed into one dense UNITARY block and
// applied through the existing apply_unitary kernel. MEASURE / RESET / BARRIER,
// classically-conditioned gates, unresolved PARAM_* gates, and the structured
// ops (MCX / MCP / PERMUTATION) flush the current block and pass through
// verbatim. Single-member blocks re-emit the original instruction. Engagement
// is Options::fusion_enable && n_qubits ≥ (fusion_threshold > 0 ? threshold :
// hardware auto). Everything here is observed through the PUBLIC Options / run()
// / apply_instruction surface — no internal is exposed.
//
// Forced engagement (design device): the auto threshold is ≥ 17 on every
// machine (statevector must exceed one LLC instance, clamped to ≥ 1 MiB), so
// small-n circuits never fuse by default. Every fused-path test therefore sets
// fusion_threshold = 1 to FORCE fusion at n = 3..10, where a dense unfused
// reference is cheap and exact. The unfused reference is a run() with
// fusion_enable = false.
//
// Tolerances (per the plan): a fused final state differs from the unfused one
// only by float re-association inside block composition, so equivalence is
// asserted at per-amplitude |Δ| ≤ 1e-12 and fidelity ≥ 1 − 1e-12. Where
// bit-identity is the actual contract (single-member blocks, the master switch,
// sub-threshold / auto-floor non-engagement), EXACT equality of both component
// arrays is asserted instead — a fused multi-gate block differs at
// machine-epsilon almost surely, so bit-identity is a sharp detector of
// wrongful engagement.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;  // QFT::build_circuit

namespace {

using Options = StatevectorSimulator::Options;

// Unfused reference: the master switch off, so run() executes the original
// instruction stream with no fusion under any threshold.
Options opt_unfused() {
    Options o;
    o.fusion_enable = false;
    return o;
}

// Forced fusion at any n ≥ 1: threshold = 1 overrides the hardware auto point.
Options opt_forced(int max_qubit = 5) {
    Options o;
    o.fusion_enable = true;
    o.fusion_threshold = 1;
    o.fusion_max_qubit = max_qubit;
    return o;
}

StatevectorSimulator::Result run_with(const QuantumCircuit& qc, const Options& o,
                                      int shots = 0, uint64_t seed = 0) {
    StatevectorSimulator sim(o);
    return sim.run(qc, shots, seed);
}

// Per-amplitude closeness + fidelity, for fused-vs-unfused equivalence.
void expect_states_close(const Statevector& a, const Statevector& b,
                         double tol = 1e-12) {
    ASSERT_EQ(a.dim, b.dim);
    const auto va = a.amplitudes();
    const auto vb = b.amplitudes();
    double max_abs = 0.0;
    for (size_t i = 0; i < va.size(); ++i) {
        EXPECT_NEAR(va[i].real, vb[i].real, tol) << "re @ " << i;
        EXPECT_NEAR(va[i].imag, vb[i].imag, tol) << "im @ " << i;
        max_abs = std::max(max_abs, (va[i] - vb[i]).norm());
    }
    // Fidelity |⟨a|b⟩|² for the two (near-)pure states.
    const double fid = a.inner_product(b).norm_sq();
    EXPECT_GE(fid, 1.0 - 1e-12) << "fidelity too low; max |Δ| = " << max_abs;
}

// Bit-for-bit identity of both component arrays — the sharp non-engagement /
// single-member-block detector.
void expect_states_exact(const Statevector& a, const Statevector& b) {
    ASSERT_EQ(a.dim, b.dim);
    for (size_t i = 0; i < a.dim; ++i) {
        EXPECT_EQ(a.real_parts[i], b.real_parts[i]) << "re @ " << i;
        EXPECT_EQ(a.imag_parts[i], b.imag_parts[i]) << "im @ " << i;
    }
}

// A dense fixed 6-qubit circuit (~40 fusable gates): rz/h/s/t/cx plus one ccx so
// support growth exercises >2 slots. Deterministic — no RNG, hardcoded angles.
QuantumCircuit dense6() {
    QuantumCircuit qc(6);
    qc.h(0).h(1).h(2).h(3).h(4).h(5);
    qc.rz(0.31, 0).rz(0.72, 1).rz(1.13, 2).s(3).t(4).rz(0.44, 5);
    qc.cx(0, 1).cx(2, 3).cx(4, 5);
    qc.ry(0.57, 0).rx(0.91, 2).rz(0.25, 4);
    qc.ccx(0, 1, 2);
    qc.cx(1, 2).cx(3, 4).cx(0, 5);
    qc.t(0).s(1).h(2).rz(0.66, 3).ry(0.19, 4).rx(0.83, 5);
    qc.ccx(3, 4, 5);
    qc.cx(2, 1).cx(4, 3).cx(5, 0);
    qc.rz(0.37, 0).rz(0.52, 1).h(2).s(3).t(4).rx(0.28, 5);
    qc.cx(0, 2).cx(1, 3).cx(4, 5);
    return qc;
}

// Index of the largest-probability basis state in a counts map.
std::string top_key(const std::unordered_map<std::string, int>& counts) {
    std::string best;
    int best_n = -1;
    for (const auto& [k, v] : counts)
        if (v > best_n) { best_n = v; best = k; }
    return best;
}

}  // namespace

// -----------------------------------------------------------------------------
// Equivalence under forced fusion
// -----------------------------------------------------------------------------

// 1. Dense random 1q/2q/3q circuit: fused ≡ unfused within 1e-12 + fidelity.
TEST(R1171SvFusion, DenseCircuitEquivalence) {
    const auto qc = dense6();
    const auto fused = run_with(qc, opt_forced());
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(fused.success);
    ASSERT_TRUE(ref.success);
    expect_states_close(fused.final_state, ref.final_state);
}

// 2. Same circuit, fusion_max_qubit swept over {2,3,4,5,6}: block-boundary
//    placement must not change the result.
TEST(R1171SvFusion, MaxQubitSweepEquivalence) {
    const auto qc = dense6();
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(ref.success);
    for (int mq = 2; mq <= 6; ++mq) {
        const auto fused = run_with(qc, opt_forced(mq));
        ASSERT_TRUE(fused.success) << "max_qubit = " << mq;
        SCOPED_TRACE("fusion_max_qubit = " + std::to_string(mq));
        expect_states_close(fused.final_state, ref.final_state);
    }
}

// 3. Disjoint-support runs: support growth must embed identity slots correctly.
TEST(R1171SvFusion, DisjointSupportEquivalence) {
    QuantumCircuit qc(6);
    qc.h(0).h(1).h(2).h(3).h(4).h(5);
    qc.rz(0.3, 0).cx(0, 1);              // block on {0,1}
    qc.ry(0.7, 4).cx(4, 5);              // block on {4,5}
    qc.rx(1.1, 2).cx(2, 3);              // block on {2,3}
    qc.rz(0.9, 0).ry(0.4, 5).rx(0.6, 3); // re-touch, forces regrowth
    const auto fused = run_with(qc, opt_forced());
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(fused.success && ref.success);
    expect_states_close(fused.final_state, ref.final_state);
}

// 4. Overlapping cx ladder with a wraparound: the support-union-overflow flush
//    path (union > max_qubit) must still be exact.
TEST(R1171SvFusion, OverlappingChainOverflowFlush) {
    QuantumCircuit qc(6);
    qc.h(0).h(1).h(2).h(3).h(4).h(5);
    qc.cx(0, 1).cx(1, 2).cx(2, 3).cx(3, 4).cx(4, 5).cx(0, 5);
    qc.rz(0.5, 0).rz(0.5, 3).ry(0.8, 5);
    const auto fused = run_with(qc, opt_forced(3));  // narrow window forces overflow
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(fused.success && ref.success);
    expect_states_close(fused.final_state, ref.final_state);
}

// 5. Explicit UNITARY member gates (1q and 2q) compose exactly through fusion.
TEST(R1171SvFusion, UnitaryMemberGatesEquivalence) {
    // A fixed 1-qubit unitary (RY(0.7)) and a fixed 2-qubit unitary
    // (RY(0.9) ⊗ RX(1.3), qubits[0] = LSB) — both interpreted identically by the
    // fused and unfused paths, so exact convention is irrelevant here.
    const double c1 = std::cos(0.35), s1 = std::sin(0.35);
    const std::vector<Complex128> u1 = {
        {c1, 0}, {-s1, 0}, {s1, 0}, {c1, 0}};
    auto kron = [](const std::vector<Complex128>& A,
                   const std::vector<Complex128>& B) {
        std::vector<Complex128> R(16);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    for (int l = 0; l < 2; ++l)
                        R[(i * 2 + k) * 4 + (j * 2 + l)] = A[i * 2 + j] * B[k * 2 + l];
        return R;
    };
    const double ca = std::cos(0.45), sa = std::sin(0.45);
    const double cb = std::cos(0.65), sb = std::sin(0.65);
    const std::vector<Complex128> ry = {{ca, 0}, {-sa, 0}, {sa, 0}, {ca, 0}};
    const std::vector<Complex128> rx = {{cb, 0}, {0, -sb}, {0, -sb}, {cb, 0}};
    const std::vector<Complex128> u2 = kron(ry, rx);

    QuantumCircuit qc(4);
    qc.h(0).h(1).h(2).h(3);
    qc.unitary(u1, {1});
    qc.unitary(u2, {0, 2});
    qc.rz(0.4, 3).cx(3, 1);
    qc.unitary(u1, {2});

    const auto fused = run_with(qc, opt_forced());
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(fused.success && ref.success);
    expect_states_close(fused.final_state, ref.final_state);
}

// 6. Single-member blocks (each fusable gate isolated by a barrier) re-emit the
//    ORIGINAL instruction — same kernels, same float path — so fused is
//    BIT-IDENTICAL to unfused.
TEST(R1171SvFusion, SingleMemberBlocksBitIdentical) {
    QuantumCircuit qc(4);
    qc.h(0).barrier();
    qc.rz(0.7, 1).barrier();
    qc.cx(0, 1).barrier();
    qc.ry(0.3, 2).barrier();
    qc.t(3).barrier();
    qc.cx(2, 3);
    const auto fused = run_with(qc, opt_forced());
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(fused.success && ref.success);
    expect_states_exact(fused.final_state, ref.final_state);
}

// -----------------------------------------------------------------------------
// Convention (MANDATORY, non-symmetric) — through the fused path
// -----------------------------------------------------------------------------

// 7. n = 4, prepare basis state K = 5 (X on q0 and X on q2 fused into one
//    window). LSB-at-qubit-0: amp[5] must be exactly 1, and measure_all must
//    yield "0101" (the CLAUDE.md worked example) through the fused path.
TEST(R1171SvFusion, ConventionBasisStateK5) {
    QuantumCircuit qc(4);
    qc.x(0).x(2);  // disjoint fusable gates → one fused block on {0,2}
    const auto fused = run_with(qc, opt_forced());
    ASSERT_TRUE(fused.success);
    const auto amps = fused.final_state.amplitudes();
    for (size_t i = 0; i < amps.size(); ++i) {
        const double want = (i == 5) ? 1.0 : 0.0;
        EXPECT_EQ(amps[i].real, want) << "re @ " << i;
        EXPECT_EQ(amps[i].imag, 0.0) << "im @ " << i;
    }

    QuantumCircuit qm(4, 4);
    qm.x(0).x(2).measure_all();
    const auto counts = run_with(qm, opt_forced(), /*shots=*/1000, /*seed=*/7).counts;
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.begin()->first, "0101");
    EXPECT_EQ(counts.at("0101"), 1000);
}

// 8. Fused QFT/IQFT round-trip on a non-symmetric value m = 11 (n = 6): the
//    IQFT with do_swaps = true recovers a clean |11⟩ peak, and fused vs unfused
//    counts are identical under the same seed.
TEST(R1171SvFusion, ConventionFusedQftNonSymmetric) {
    QuantumCircuit qc(6, 6);
    qc.x(0).x(1).x(3);  // prepare |11⟩ = 001011b
    // Forward QFT builds the Fourier state; IQFT (do_swaps=true) recovers |11⟩.
    qc = qc.compose(QFT::build_circuit(6, QFT::Options(true, 0, false)));
    qc = qc.compose(QFT::build_circuit(6, QFT::Options(true, 0, true)));

    const auto fused = run_with(qc, opt_forced());
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(fused.success && ref.success);
    expect_states_close(fused.final_state, ref.final_state);

    // Peak at amp index 11 in both.
    const auto pf = fused.final_state.probabilities();
    const auto pr = ref.final_state.probabilities();
    size_t peak_f = 0, peak_r = 0;
    for (size_t i = 0; i < pf.size(); ++i) {
        if (pf[i] > pf[peak_f]) peak_f = i;
        if (pr[i] > pr[peak_r]) peak_r = i;
    }
    EXPECT_EQ(peak_f, 11u);
    EXPECT_EQ(peak_r, 11u);
    EXPECT_GT(pf[11], 0.99);

    // Seeded counts identical fused vs unfused, and "001011" dominates.
    QuantumCircuit qm = qc;
    qm.measure_all();
    const auto cf = run_with(qm, opt_forced(), 2048, 20250718).counts;
    const auto cr = run_with(qm, opt_unfused(), 2048, 20250718).counts;
    EXPECT_EQ(cf, cr);
    EXPECT_EQ(top_key(cf), "001011");
}

// -----------------------------------------------------------------------------
// Flush semantics / strategy preservation (forced fusion)
// -----------------------------------------------------------------------------

// 9. Terminal-only measurement fast path: seeded counts identical fused vs
//    unfused. Well-separated (non-dyadic) probabilities keep every draw clear
//    of a cumulative-boundary flip under the 1e-12 amplitude difference.
TEST(R1171SvFusion, TerminalMeasureCountsIdentical) {
    QuantumCircuit qc(4, 4);
    qc.ry(0.9, 0).ry(1.4, 1).ry(2.1, 2).ry(0.6, 3);  // cos²(θ/2): ~0.79/0.42/0.24/0.91
    qc.cx(0, 1).cx(2, 3).rz(0.5, 1).rx(0.3, 2);
    qc.measure_all();
    const auto cf = run_with(qc, opt_forced(), 4096, 424242).counts;
    const auto cr = run_with(qc, opt_unfused(), 4096, 424242).counts;
    EXPECT_EQ(cf, cr);
    EXPECT_FALSE(cf.empty());
}

// 10. Mid-circuit measure + feedforward (conditioned X): per-shot trajectory
//     path. The measured qubit is driven near a basis state so the collapse
//     outcome is well clear of the RNG boundary; identical counts fused vs
//     unfused proves the reused fused plan does not perturb collapse order.
TEST(R1171SvFusion, FeedforwardCountsIdentical) {
    QuantumCircuit qc(3, 3);
    qc.ry(0.2, 0);            // qubit 0 ≈ |0⟩ (P(1) ≈ 0.01): outcome ~always 0
    qc.h(1).rz(0.5, 1);
    qc.measure(0, 0);
    // Conditioned X on qubit 2 iff clbit 0 == 1 (rarely taken, but deterministic
    // per seed and never near a boundary).
    qc.add_if(0, 1, Instruction::GateType::X, {2});
    qc.cx(1, 2).ry(0.7, 1);
    qc.measure(1, 1);
    qc.measure(2, 2);
    const auto cf = run_with(qc, opt_forced(), 512, 99991).counts;
    const auto cr = run_with(qc, opt_unfused(), 512, 99991).counts;
    EXPECT_EQ(cf, cr);
}

// 11. RESET mid-circuit under forced fusion: the reset flushes the block and
//     passes through; the final state matches unfused. The reset qubit is put
//     in a definite basis state so the collapse outcome is deterministic.
TEST(R1171SvFusion, ResetMidCircuitMatchesUnfused) {
    QuantumCircuit qc(4);
    qc.h(0).h(1).h(3);  // NOT qubit 2: keep the reset target in a definite state
    qc.x(2);            // qubit 2 → |1⟩ exactly (prob₀ = 0): collapse is deterministic
    qc.reset(2);        // non-fusable: flush; outcome fixed regardless of the draw
    qc.ry(0.6, 2).cx(2, 3).rz(0.4, 0);
    const auto fused = run_with(qc, opt_forced(), 0, 555);
    const auto ref = run_with(qc, opt_unfused(), 0, 555);
    ASSERT_TRUE(fused.success && ref.success);
    expect_states_close(fused.final_state, ref.final_state);
}

// 12. MCX / MCP / PERMUTATION interleaved with fusable gates: the structured
//     ops pass through verbatim while the fusable gates around them fuse.
TEST(R1171SvFusion, StructuredOpsPassthrough) {
    QuantumCircuit qc(6);
    qc.h(0).h(1).h(2).h(3).h(4).h(5);
    qc.rz(0.3, 0).cx(0, 1);
    qc.mcx({0, 1}, 2);                    // non-fusable → flush + verbatim
    qc.ry(0.5, 3).cx(3, 4);
    qc.mcp(0.4, {1, 3});                  // non-fusable → flush + verbatim
    qc.permute({0, 2, 1, 3}, {4, 5});     // non-fusable → flush + verbatim
    qc.rx(0.7, 0).cx(2, 5).rz(0.9, 4);
    const auto fused = run_with(qc, opt_forced());
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(fused.success && ref.success);
    expect_states_close(fused.final_state, ref.final_state);
}

// 13. shots == 0 single-trajectory semantics (conditions honoured, measures
//     recorded) unchanged under forced fusion: seeded final state matches.
//     The measured qubit is near a basis state so the branch taken is stable.
TEST(R1171SvFusion, ShotsZeroTrajectoryMatchesUnfused) {
    QuantumCircuit qc(3, 3);
    qc.ry(0.15, 0);                       // qubit 0 ≈ |0⟩
    qc.h(1).rz(0.3, 1);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {2});
    qc.ry(0.9, 2).cx(1, 2);
    const auto fused = run_with(qc, opt_forced(), 0, 31337);
    const auto ref = run_with(qc, opt_unfused(), 0, 31337);
    ASSERT_TRUE(fused.success && ref.success);
    expect_states_close(fused.final_state, ref.final_state);
}

// -----------------------------------------------------------------------------
// Options surface
// -----------------------------------------------------------------------------

// 14. fusion_max_qubit outside [2,6] → an error Result (success = false, non-
//     empty message), not a throw to the caller.
TEST(R1171SvFusion, MaxQubitOutOfRangeErrorResult) {
    const auto qc = dense6();
    Options lo = opt_forced();
    lo.fusion_max_qubit = 1;
    const auto rlo = run_with(qc, lo);
    EXPECT_FALSE(rlo.success);
    EXPECT_FALSE(rlo.error_message.empty());

    Options hi = opt_forced();
    hi.fusion_max_qubit = 7;
    const auto rhi = run_with(qc, hi);
    EXPECT_FALSE(rhi.success);
    EXPECT_FALSE(rhi.error_message.empty());
}

// 15. fusion_threshold < 0 → error Result.
TEST(R1171SvFusion, NegativeThresholdErrorResult) {
    const auto qc = dense6();
    Options o = opt_forced();
    o.fusion_threshold = -1;
    const auto r = run_with(qc, o);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

// 16. fusion_enable = false with threshold = 1 at n = 6 is BIT-IDENTICAL to a
//     default-options run (which at n = 6 never auto-engages): the master
//     switch wins over the threshold.
TEST(R1171SvFusion, EnableFalseIsBitIdenticalToDefault) {
    const auto qc = dense6();
    Options off = opt_forced();
    off.fusion_enable = false;  // master switch off despite threshold = 1
    const auto disabled = run_with(qc, off);
    const auto def = run_with(qc, Options{});  // defaults: auto (≥17) never fires at n=6
    ASSERT_TRUE(disabled.success && def.success);
    expect_states_exact(disabled.final_state, def.final_state);
}

// 17. Auto floor: at n = 10, threshold = 0 (auto) must NOT engage (auto ≥ 17),
//     so it is BIT-IDENTICAL to fusion_enable = false.
TEST(R1171SvFusion, AutoFloorNoEngagementAtN10) {
    QuantumCircuit qc(10);
    for (int q = 0; q < 10; ++q) qc.h(q);
    for (int q = 0; q + 1 < 10; ++q) qc.cx(q, q + 1);
    for (int q = 0; q < 10; ++q) qc.rz(0.1 * (q + 1), q);
    Options autoo;  // fusion_enable = true, fusion_threshold = 0 (auto)
    const auto a = run_with(qc, autoo);
    const auto b = run_with(qc, opt_unfused());
    ASSERT_TRUE(a.success && b.success);
    expect_states_exact(a.final_state, b.final_state);
}

// 18. Threshold boundary: threshold = n engages (≥ comparison), matching
//     unfused within 1e-12; threshold = n + 1 does NOT engage and is bit-
//     identical to disabled.
TEST(R1171SvFusion, ThresholdBoundaryEngagement) {
    const auto qc = dense6();  // n = 6
    const auto ref = run_with(qc, opt_unfused());
    ASSERT_TRUE(ref.success);

    Options at;
    at.fusion_threshold = 6;  // n ≥ 6 → engages
    const auto engaged = run_with(qc, at);
    ASSERT_TRUE(engaged.success);
    expect_states_close(engaged.final_state, ref.final_state);

    Options above;
    above.fusion_threshold = 7;  // n = 6 < 7 → no engagement
    const auto not_engaged = run_with(qc, above);
    ASSERT_TRUE(not_engaged.success);
    expect_states_exact(not_engaged.final_state, ref.final_state);
}

// 19. apply_instruction / step-wise path is unaffected by any Options fusion
//     setting: options act only inside run(). Two sims with opposite fusion
//     configs must produce bit-identical states when driven step-wise.
TEST(R1171SvFusion, ApplyInstructionIgnoresOptions) {
    const auto qc = dense6();
    StatevectorSimulator sim_fused(opt_forced());
    StatevectorSimulator sim_unfused(opt_unfused());
    Statevector a(qc.n_qubits), b(qc.n_qubits);
    for (const auto& inst : qc.instructions) {
        sim_fused.apply_instruction(a, inst);
        sim_unfused.apply_instruction(b, inst);
    }
    expect_states_exact(a, b);
}
