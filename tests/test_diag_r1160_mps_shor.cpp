// TEMPORARY DIAGNOSTIC (R.1.16.0, issue #44) — not release test content.
// Remove (or promote the keepers into the R.1.16.1 suite) before the
// R.1.16.0 release commit.
//
// Purpose: discriminate the three suspects behind "MPS backend does not
// recover Shor's order at 13 qubits despite bond dim 64 being exact":
//   S1  wide-PERMUTATION dense fallback round-trip (to_statevector ->
//       apply -> rebuild), invoked once per eval qubit — truncation or
//       drift accumulating across rebuilds;
//   S2  MPS sampling (measure_sequential) at 13 qubits;
//   S3  MPS IQFT / gate application at scale.
//
// Instruments:
//   - Stage sweep: run PREFIXES of Shor::build_period_finding_circuit(2, 15,
//     9, 4) on MPS (bond 64, shots = 0: pure evolution, pristine final
//     state per the R.1.12 shots convention) and on the statevector
//     simulator; report fidelity, accumulated truncation error, and the
//     reached bond dimension at every cut. Bond 64 is MATHEMATICALLY exact
//     for any 13-qubit state (max Schmidt rank 2^6 = 64), so ANY nonzero
//     truncation error is itself the defect, and the cut where fidelity
//     first drops names the guilty layer (S1 cuts sit right after each
//     PERMUTATION; S3 shows up only in the IQFT tail).
//   - Sampler check: MPS counts at 4096 shots vs the EXACT eval-register
//     marginal of the MPS's OWN final state. A healthy sampler matches its
//     own state regardless of whether that state is correct — separating
//     S2 from S1/S3.
//   - End to end: find_order(2, 15, 9, backend, seed) on MPS and SV for
//     five seeds — captures the user-visible symptom (r = 4 expected).
//
// Read the numbers from a direct gtest run (ctest hides passing stdout):
//   ./build-clang/tests/lindblad_tests --gtest_filter='DiagR1160MpsShor.*'

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace lindblad;
using algorithms::Shor;
using backends::LocalBackend;

namespace {

constexpr uint64_t kA = 2;
constexpr uint64_t kN = 15;
constexpr int kEval = 9;
constexpr int kTarget = 4;
constexpr int kQubits = kEval + kTarget;  // 13
constexpr int kBond = 64;                 // exact at 13 qubits
constexpr uint64_t kExpectedOrder = 4;    // ord_15(2)

QuantumCircuit prefix_of(const QuantumCircuit& qc, size_t k) {
    QuantumCircuit p(qc.n_qubits, qc.n_clbits);
    p.instructions.assign(qc.instructions.begin(),
                          qc.instructions.begin() + static_cast<long>(k));
    return p;
}

std::vector<std::complex<double>> sv_state_of(const QuantumCircuit& qc) {
    Statevector sv(qc.n_qubits);
    sv.initialize_basis(0);
    StatevectorSimulator sim;
    for (const auto& inst : qc.instructions) sim.apply_instruction(sv, inst);
    const size_t dim = size_t{1} << qc.n_qubits;
    std::vector<std::complex<double>> psi(dim);
    for (size_t i = 0; i < dim; ++i) psi[i] = {sv.real_parts[i], sv.imag_parts[i]};
    return psi;
}

std::vector<std::complex<double>> mps_state_to_vec(const MPSState& st) {
    Statevector sv = st.to_statevector();
    const size_t dim = size_t{1} << st.n_qubits;
    std::vector<std::complex<double>> psi(dim);
    for (size_t i = 0; i < dim; ++i) psi[i] = {sv.real_parts[i], sv.imag_parts[i]};
    return psi;
}

double fidelity(const std::vector<std::complex<double>>& a,
                const std::vector<std::complex<double>>& b) {
    std::complex<double> ov(0, 0);
    for (size_t i = 0; i < a.size(); ++i) ov += std::conj(a[i]) * b[i];
    return std::norm(ov);
}

// Eval-register marginal (qubits 0..kEval-1 = low bits of the amp index).
std::vector<double> eval_marginal(const std::vector<std::complex<double>>& psi) {
    std::vector<double> m(size_t{1} << kEval, 0.0);
    for (size_t i = 0; i < psi.size(); ++i) m[i & ((1u << kEval) - 1)] += std::norm(psi[i]);
    return m;
}

}  // namespace

// -----------------------------------------------------------------------------
// Probe 1 — stage sweep: fidelity + truncation + bond at every stage boundary
// -----------------------------------------------------------------------------
// R.1.16.1: probe retired (kept verbatim for the R.1.16.0 audit trail).
// Assertion form: R1161MpsShor.StateExactAtEveryStage below.
#if 0
TEST(DiagR1160MpsShor, StageSweepFidelityAndTruncation) {
    const auto qc = Shor::build_period_finding_circuit(kA, kN, kEval, kTarget);
    ASSERT_EQ(qc.n_qubits, kQubits);

    // Inventory: what the circuit is actually made of, and where the
    // PERMUTATION blocks (dense-fallback triggers on MPS) sit.
    std::map<std::string, int> ops;
    std::vector<size_t> perm_ends;  // cut AFTER each permutation
    for (size_t i = 0; i < qc.instructions.size(); ++i) {
        const auto& inst = qc.instructions[i];
        ops[inst.gate_name()]++;
        if (inst.type == Instruction::GateType::PERMUTATION) {
            perm_ends.push_back(i + 1);
        }
    }
    std::cout << "\n[inventory] " << qc.instructions.size() << " instructions:";
    for (const auto& [name, n] : ops) std::cout << "  " << name << " x" << n;
    std::cout << "\n[inventory] PERMUTATION count: " << perm_ends.size() << "\n";

    // Cuts: before the first permutation (prep), after every permutation,
    // then through the IQFT tail in quarters. Fallback to deciles if the
    // circuit encodes the multiplies differently than expected.
    std::vector<size_t> cuts;
    if (!perm_ends.empty()) {
        cuts.push_back(perm_ends.front() - 1);  // end of prep layer
        for (size_t e : perm_ends) cuts.push_back(e);
        const size_t tail_start = perm_ends.back();
        const size_t tail = qc.instructions.size() - tail_start;
        for (int q = 1; q <= 4; ++q) {
            cuts.push_back(tail_start + (tail * static_cast<size_t>(q)) / 4);
        }
    } else {
        for (int d = 1; d <= 10; ++d) {
            cuts.push_back(qc.instructions.size() * static_cast<size_t>(d) / 10);
        }
    }

    std::cout << "[sweep] k = prefix length; fid = |<sv|mps>|^2; "
                 "terr = accumulated truncation; chi = reached bond\n"
              << std::fixed << std::setprecision(12);

    double prev_fid = 1.0;
    for (size_t k : cuts) {
        const auto pre = prefix_of(qc, k);
        MPSSimulator mps;
        auto res = mps.run(pre, kBond, /*shots=*/0, /*seed=*/42);
        const double fid = fidelity(sv_state_of(pre),
                                    mps_state_to_vec(res.final_state));
        const double terr = res.final_state.truncation_error();
        const int chi = res.final_state.current_max_bond_dim();
        const std::string last =
            (k == 0) ? std::string("-") : pre.instructions.back().gate_name();

        std::cout << "[sweep] k=" << std::setw(4) << k
                  << "  last=" << std::setw(12) << last
                  << "  fid=" << fid
                  << "  terr=" << std::scientific << terr << std::fixed
                  << "  chi=" << chi
                  << ((fid < prev_fid - 1e-9) ? "   <-- fidelity drop" : "")
                  << "\n";
        prev_fid = fid;

        // Bond 64 is exact at 13 qubits: any truncation IS the defect.
        EXPECT_NEAR(terr, 0.0, 1e-12)
            << "truncation occurred at k=" << k
            << " even though chi=64 is mathematically sufficient";
    }
}
#endif  // retired probe 1

// -----------------------------------------------------------------------------
// Probe 2 — sampler vs the MPS's OWN state (separates S2 from S1/S3)
// -----------------------------------------------------------------------------
// R.1.16.1: probe retired. Assertion form:
// R1161MpsShor.SamplerWithinNoiseAt13Qubits below.
#if 0
TEST(DiagR1160MpsShor, SamplerMatchesItsOwnState) {
    auto qc = Shor::build_period_finding_circuit(kA, kN, kEval, kTarget);
    qc.n_clbits = kQubits;
    qc.measure_all();

    MPSSimulator mps;
    const int shots = 4096;
    auto res = mps.run(qc, kBond, shots, /*seed=*/42);

    // Exact eval-register marginal of the state the sampler itself saw.
    const auto own = eval_marginal(mps_state_to_vec(res.final_state));

    // Sampled eval-register marginal: eval register = qubits 0..8 = the
    // RIGHTMOST 9 characters of the bitstring key (qubit 0 rightmost).
    std::vector<double> sampled(size_t{1} << kEval, 0.0);
    int total = 0;
    for (const auto& [bits, count] : res.counts) {
        ASSERT_EQ(bits.size(), static_cast<size_t>(kQubits));
        const std::string eval_bits = bits.substr(bits.size() - kEval);
        sampled[std::stoul(eval_bits, nullptr, 2)] += count;
        total += count;
    }
    ASSERT_EQ(total, shots);
    for (auto& p : sampled) p /= shots;

    double tvd = 0.0;
    int support = 0;
    for (size_t i = 0; i < own.size(); ++i) {
        tvd += std::abs(own[i] - sampled[i]);
        support += (own[i] > 1.0 / (10.0 * shots)) ? 1 : 0;
    }
    tvd /= 2.0;

    std::cout << "\n[sampler] eval-marginal TVD(sampled, own-state) = "
              << std::fixed << std::setprecision(6) << tvd
              << "  (support " << support << ", shots " << shots << ")\n";
    std::cout << "[sampler] top outcomes (eval register, ideal peaks at "
                 "0/128/256/384 for r=4):\n";
    std::vector<std::pair<double, size_t>> top;
    for (size_t i = 0; i < own.size(); ++i) top.push_back({own[i], i});
    std::sort(top.rbegin(), top.rend());
    for (int i = 0; i < 8; ++i) {
        std::cout << "[sampler]   eval=" << std::setw(3) << top[i].second
                  << "  p_own=" << top[i].first
                  << "  p_sampled=" << sampled[top[i].second] << "\n";
    }

    const double noise_scale =
        std::sqrt(static_cast<double>(std::max(support, 1)) /
                  (3.14159265358979 * shots));
    EXPECT_LT(tvd, 2.5 * noise_scale + 0.02)
        << "sampler disagrees with its OWN state beyond sampling noise: "
           "measure_sequential (S2) is implicated";
}
#endif  // retired probe 2

// -----------------------------------------------------------------------------
// Probe 4 (added after run 1) — NaN bisection. KEPT ACTIVE in R.1.16.1: it
// is already assertion-shaped (fix-aware early return; baseline and
// per-step corruption checks) and doubles as the mirror-path regression.
//
// Run-1 verdict: all 9 PERMUTATION round-trips are EXACT (fid 1.0, terr 0,
// chi 4) — S1 exonerated. The state goes NaN somewhere in instructions
// 20..31 (IQFT h/cp region). terr = NaN means the SVD RECEIVED NaN, so the
// corruption precedes the truncation that reported it. This probe applies
// the remaining instructions ONE AT A TIME through the same public MPSState
// API the simulator dispatch uses (verified against mps_apply_instruction:
// h -> gate2x2/apply_single_qubit_gate, cp/swap -> gate4x4/
// apply_two_qubit_gate), scanning every tensor for NaN/Inf after each. On
// the first corrupt instruction it reruns that instruction's internal SWAP
// chain step by step (mirroring apply_two_qubit_gate: swap q2-1..q1+1 down,
// gate at q1, swap back) to name the exact adjacent SVD call and tensor
// shapes that produce the NaN.
// -----------------------------------------------------------------------------
namespace {

constexpr double kS2 = 0.7071067811865475;

std::array<Complex128, 4> h2x2() {
    return {Complex128(kS2, 0), Complex128(kS2, 0),
            Complex128(kS2, 0), Complex128(-kS2, 0)};
}
std::array<Complex128, 4> x2x2() {
    return {Complex128(0, 0), Complex128(1, 0),
            Complex128(1, 0), Complex128(0, 0)};
}
std::array<Complex128, 16> cp4x4(double lam) {
    std::array<Complex128, 16> U{};
    U[0 * 4 + 0] = U[1 * 4 + 1] = U[2 * 4 + 2] = Complex128(1, 0);
    U[3 * 4 + 3] = Complex128(std::cos(lam), std::sin(lam));
    return U;
}
std::array<Complex128, 16> swap4x4() {
    std::array<Complex128, 16> U{};
    U[0 * 4 + 0] = U[1 * 4 + 2] = U[2 * 4 + 1] = U[3 * 4 + 3] = Complex128(1, 0);
    return U;
}

// Mirrors mps_apply_instruction for the gate types present in the IQFT tail.
void apply_mirror(MPSState& st, const Instruction& inst) {
    switch (inst.type) {
        case Instruction::GateType::H:
            st.apply_single_qubit_gate(h2x2(), inst.qubits[0]);
            break;
        case Instruction::GateType::X:
            st.apply_single_qubit_gate(x2x2(), inst.qubits[0]);
            break;
        case Instruction::GateType::CP:
            st.apply_two_qubit_gate(cp4x4(inst.params[0]), inst.qubits[0],
                                    inst.qubits[1]);
            break;
        case Instruction::GateType::SWAP:
            st.apply_two_qubit_gate(swap4x4(), inst.qubits[0], inst.qubits[1]);
            break;
        default:
            FAIL() << "unexpected gate in IQFT tail: " << inst.gate_name();
    }
}

// NaN/Inf detection that SURVIVES -ffast-math: the tests compile with
// -ffinite-math-only, under which std::isnan/std::isinf constant-fold to
// false (this is also why probe 1's EXPECT_NEAR(terr, 0) "passed" on a NaN
// value). Check the raw exponent bits instead.
bool fp_bad(double x) {
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return ((b >> 52) & 0x7FFu) == 0x7FFu;  // Inf or NaN exponent
}

struct TensorScan {
    bool corrupt = false;   // any NaN or Inf entry
    double max_abs = 0.0;
    int chi = 0;
};

TensorScan scan_tensors(const MPSState& st) {
    TensorScan s;
    for (const auto& t : st.tensors) {
        s.chi = std::max({s.chi, t.bond_left, t.bond_right});
        for (const auto& c : t.data) {
            const double m = std::sqrt(c.real * c.real + c.imag * c.imag);
            if (fp_bad(c.real) || fp_bad(c.imag)) s.corrupt = true;
            s.max_abs = std::max(s.max_abs, m);
        }
    }
    return s;
}

void print_tensor_profile(const MPSState& st, const char* tag) {
    std::cout << "[bisect] tensor profile " << tag << ":\n";
    for (int q = 0; q < st.n_qubits; ++q) {
        const auto& t = st.tensors[static_cast<size_t>(q)];
        double mx = 0.0;
        bool bad = false;
        for (const auto& c : t.data) {
            const double m = std::sqrt(c.real * c.real + c.imag * c.imag);
            mx = std::max(mx, m);
            bad = bad || fp_bad(c.real) || fp_bad(c.imag);
        }
        std::cout << "[bisect]   site " << std::setw(2) << q << "  ("
                  << t.bond_left << " x 2 x " << t.bond_right << ")  max|e|="
                  << std::scientific << std::setprecision(3) << mx
                  << (bad ? "  <-- NaN/Inf" : "") << "\n";
    }
    std::cout << std::fixed;
}

}  // namespace

TEST(DiagR1160MpsShor, NanBisectPerInstructionAndPerSwapStep) {
    const auto qc = Shor::build_period_finding_circuit(kA, kN, kEval, kTarget);
    const size_t kStart = 19;  // run 1: state exact here (after 9th PERMUTATION)

    MPSSimulator sim;
    auto base = sim.run(prefix_of(qc, kStart), kBond, /*shots=*/0, /*seed=*/42);
    MPSState st = base.final_state;  // copy: probe owns its state

    {
        const auto s0 = scan_tensors(st);
        ASSERT_FALSE(s0.corrupt) << "baseline at k=19 already corrupt";
        std::cout << "\n[bisect] baseline k=19: chi=" << s0.chi
                  << " max|e|=" << std::scientific << std::setprecision(3)
                  << s0.max_abs << std::fixed << "\n";
        print_tensor_profile(st, "at k=19");
    }

    size_t first_bad = 0;
    for (size_t i = kStart; i < qc.instructions.size(); ++i) {
        const auto& inst = qc.instructions[i];
        apply_mirror(st, inst);
        const auto s = scan_tensors(st);
        std::cout << "[bisect] i=" << std::setw(3) << i << "  "
                  << std::setw(5) << inst.gate_name() << " q=(";
        for (size_t j = 0; j < inst.qubits.size(); ++j) {
            std::cout << inst.qubits[j] << (j + 1 < inst.qubits.size() ? "," : "");
        }
        std::cout << ")  chi=" << std::setw(3) << s.chi << "  max|e|="
                  << std::scientific << std::setprecision(3) << s.max_abs
                  << std::fixed << "  terr=" << std::scientific
                  << st.truncation_error() << std::fixed
                  << (s.corrupt ? "   <-- FIRST NaN/Inf" : "") << "\n";
        if (s.corrupt) {
            first_bad = i;
            break;
        }
    }
    if (first_bad == 0) {
        std::cout << "[bisect] NO corruption through the full tail — the "
                     "svd_truncate hardening is effective on this path.\n";
        return;  // fix-aware: pre-fix this probe existed to FIND the NaN
    }

    // Zoom: replay the offending instruction's internal SWAP chain manually,
    // scanning after every adjacent operation.
    const auto& bad = qc.instructions[first_bad];
    ASSERT_EQ(bad.qubits.size(), 2u);
    int q1 = bad.qubits[0], q2 = bad.qubits[1];
    const bool is_swap = (bad.type == Instruction::GateType::SWAP);
    std::array<Complex128, 16> G =
        is_swap ? swap4x4() : cp4x4(bad.params[0]);
    if (q1 > q2) {
        std::swap(q1, q2);
        // CP and SWAP are symmetric in their qubits: no matrix reindex needed.
    }

    MPSState zoom = base.final_state;  // fresh copy
    for (size_t i = kStart; i < first_bad; ++i) apply_mirror(zoom, qc.instructions[i]);
    ASSERT_FALSE(scan_tensors(zoom).corrupt);
    print_tensor_profile(zoom, "immediately BEFORE the failing instruction");

    std::cout << "[bisect] failing instruction i=" << first_bad << " "
              << bad.gate_name() << "(" << q1 << "," << q2
              << "), span=" << (q2 - q1) << "; replaying its swap chain:\n";

    auto step_scan = [&](const char* what, int at) {
        const auto s = scan_tensors(zoom);
        std::cout << "[bisect]   " << what << " @" << at << "  chi=" << s.chi
                  << "  max|e|=" << std::scientific << std::setprecision(3)
                  << s.max_abs << std::fixed
                  << (s.corrupt ? "   <-- NaN/Inf HERE" : "") << "\n";
        return s.corrupt;
    };

    bool located = false;
    for (int i2 = q2 - 1; i2 > q1 && !located; --i2) {
        zoom.apply_two_qubit_gate(swap4x4(), i2, i2 + 1);
        located = step_scan("swap-down", i2);
    }
    if (!located) {
        zoom.apply_two_qubit_gate(G, q1, q1 + 1);
        located = step_scan(is_swap ? "swap-core" : "cp-core", q1);
    }
    for (int i2 = q1 + 1; i2 < q2 && !located; ++i2) {
        zoom.apply_two_qubit_gate(swap4x4(), i2, i2 + 1);
        located = step_scan("swap-up", i2);
    }
    if (located) print_tensor_profile(zoom, "at first corruption");
    EXPECT_TRUE(located)
        << "instruction-level corruption did not reproduce step-by-step "
           "(ordering-sensitive: report this)";
}

// -----------------------------------------------------------------------------
// Probe 5 (added after run 2) — FAST-MATH twin of the strict-FP SVD tests.
//
// This TU compiles under the project-wide -ffast-math; its twin
// (test_diag_r1160_strictfp.cpp) compiles the SAME builders and SAME Eigen
// SVDs under -fno-fast-math. The pair answers, with everything else equal:
// is the JacobiSVD NaN (#44) and/or the R.1.11.2 "BDCSVD accuracy bug" a
// fast-math casualty? NOTE: a caveat on this leg — inlining/codegen here
// can differ from mps_sim.cpp's instantiation, so a clean result HERE under
// fast-math does not exonerate fast-math (the library-run leg in the strict
// twin still corrupts); a corrupt result here is direct confirmation.
// Print-only where the outcome is the experiment's answer.
// -----------------------------------------------------------------------------

#include "diag_r1160_matrices.hpp"

namespace {

void print_svd_report(const char* tag, const diag_r1160::SvdReport& r) {
    std::cout << std::scientific << std::setprecision(6);
    diag_r1160::print_svd_report_line(std::cout, tag, r);
    std::cout << std::fixed;
}

}  // namespace

// R.1.16.1: probes retired. Assertion forms:
// R1161MpsShor.PoisonThetaKeptSliceContract / .Simon36JacobiReference below.
#if 0
TEST(DiagR1160MpsShor, FastMathSvdOnPoisonTheta) {
    const auto theta = diag_r1160::build_poison_theta();
    ASSERT_FALSE(diag_r1160::matrix_bad(theta))
        << "poison theta reconstruction corrupt before any SVD";
    const auto rj =
        diag_r1160::run_svd_report<Eigen::JacobiSVD<Eigen::MatrixXcd>>(theta);
    const auto rb =
        diag_r1160::run_svd_report<Eigen::BDCSVD<Eigen::MatrixXcd>>(theta);
    print_svd_report("fast/jacobi/poison", rj);
    print_svd_report("fast/bdc/poison", rb);
    // No hard assertions: the values ARE the answer (see twin TU).
}

TEST(DiagR1160MpsShor, FastMathSvdOnR1112BugMatrix) {
    const auto M = diag_r1160::build_bdcsvd_bug_matrix();
    ASSERT_FALSE(diag_r1160::matrix_bad(M));
    const auto rb =
        diag_r1160::run_svd_report<Eigen::BDCSVD<Eigen::MatrixXcd>>(M);
    const auto rj =
        diag_r1160::run_svd_report<Eigen::JacobiSVD<Eigen::MatrixXcd>>(M);
    print_svd_report("fast/bdc/simon36", rb);
    print_svd_report("fast/jacobi/simon36", rj);
    // Expected from the R.1.11.2 note (recorded under fast-math): Jacobi
    // correct (twelve 0.288675, recon ~1e-16), BDC wrong (norm violation +
    // subnormal tell). Reproduction here re-validates the bug note's data
    // point on the current Eigen pin before the strict twin reinterprets it.
}
#endif  // retired probe 5

// -----------------------------------------------------------------------------
// Probe 6 (added after run 4) — does QUDIT MPS share the manifestation?
//
// The qudit MPS tests all sit at n <= 6 sites (dense-ctor round-trips and
// short gate checks), so the suite being green does NOT clear qudit_mps.cpp
// — its four SVD rank loops have the same unguarded 'S(i) > threshold'
// shape. This probe drives QuditMPS at d=2, 13 sites through the EXACT
// failing scenario: the k=19 Shor state (injected via the dense ctor from
// the statevector simulator's exact state) followed by the same IQFT tail.
// d=2 makes the 2x2/4x4 matrices numerically identical, and CP/SWAP are
// symmetric under qubit-order exchange, so operand conventions cannot
// confound the comparison. After every instruction the state is scanned
// bit-level; at the end, fidelity against the statevector reference.
// -----------------------------------------------------------------------------

#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"

// R.1.16.1: probe retired. Assertion form (a latent-pattern CANARY for the
// unguarded qudit truncation loops): R1161MpsShor.QuditTwinStaysExact below.
#if 0
TEST(DiagR1160MpsShor, QuditMpsTwinOnSameIqftTail) {
    const auto qc = Shor::build_period_finding_circuit(kA, kN, kEval, kTarget);

    // Exact k=19 state from the statevector simulator, as a d=2 qudit state
    // (both layouts are LSB-first: amplitude index bit q = qubit/qudit q).
    const auto prefix19 = prefix_of(qc, 19);
    const auto psi19 = sv_state_of(prefix19);
    QuditStatevector qsv(kQubits, /*d=*/2);
    for (size_t i = 0; i < psi19.size(); ++i) {
        qsv.amplitudes[i] = Complex128(psi19[i].real(), psi19[i].imag());
    }
    QuditMPS qm(qsv, /*max_bond_dim=*/kBond);

    constexpr double s2 = 0.7071067811865475;
    const std::vector<Complex128> H2 = {
        Complex128(s2, 0), Complex128(s2, 0),
        Complex128(s2, 0), Complex128(-s2, 0)};
    auto cp4 = [](double lam) {
        std::vector<Complex128> U(16, Complex128(0, 0));
        U[0] = U[5] = U[10] = Complex128(1, 0);
        U[15] = Complex128(std::cos(lam), std::sin(lam));
        return U;
    };
    std::vector<Complex128> SW(16, Complex128(0, 0));
    SW[0] = SW[6] = SW[9] = SW[15] = Complex128(1, 0);

    auto qudit_state_bad = [&](const QuditMPS& m) {
        const auto sv = m.to_statevector();
        for (const auto& a : sv.amplitudes) {
            std::uint64_t br, bi;
            std::memcpy(&br, &a.real, sizeof(br));
            std::memcpy(&bi, &a.imag, sizeof(bi));
            if (((br >> 52) & 0x7FFu) == 0x7FFu ||
                ((bi >> 52) & 0x7FFu) == 0x7FFu) {
                return true;
            }
        }
        return false;
    };

    ASSERT_FALSE(qudit_state_bad(qm)) << "dense-ctor injection already corrupt";

    bool corrupted = false;
    for (size_t i = 19; i < qc.instructions.size(); ++i) {
        const auto& inst = qc.instructions[i];
        if (inst.type == Instruction::GateType::H) {
            qm.apply_1qudit(inst.qubits[0], H2);
        } else if (inst.type == Instruction::GateType::CP) {
            qm.apply_2qudit(inst.qubits[0], inst.qubits[1], cp4(inst.params[0]));
        } else if (inst.type == Instruction::GateType::SWAP) {
            qm.apply_2qudit(inst.qubits[0], inst.qubits[1], SW);
        } else {
            FAIL() << "unexpected gate in IQFT tail: " << inst.gate_name();
        }
        if (qudit_state_bad(qm)) {
            corrupted = true;
            std::cout << "[qudit] FIRST NaN/Inf at i=" << i << "  "
                      << inst.gate_name() << " q=(" << inst.qubits[0]
                      << (inst.qubits.size() > 1
                              ? "," + std::to_string(inst.qubits[1])
                              : std::string())
                      << ")\n";
            break;
        }
    }

    if (!corrupted) {
        // Full-circuit fidelity vs the statevector reference.
        const auto psi_ref = sv_state_of(prefix_of(qc, qc.instructions.size()));
        const auto qfinal = qm.to_statevector();
        std::complex<double> ov(0, 0);
        for (size_t i = 0; i < psi_ref.size(); ++i) {
            ov += std::conj(psi_ref[i]) *
                  std::complex<double>(qfinal.amplitudes[i].real,
                                       qfinal.amplitudes[i].imag);
        }
        std::cout << "[qudit] IQFT tail completed clean; |<sv|qudit-mps>|^2 = "
                  << std::fixed << std::setprecision(12) << std::norm(ov)
                  << "\n";
    }
    std::cout << "[qudit] verdict: qudit MPS "
              << (corrupted ? "SHARES the manifestation (same fix needed)"
                            : "does NOT manifest on the exact failing "
                              "scenario (latent pattern only)")
              << "\n";
    // No hard assertion: this probe IS the owner's decision input.
}
#endif  // retired probe 6

// -----------------------------------------------------------------------------
// Probe 3 — the user-visible symptom, both backends, five seeds
// -----------------------------------------------------------------------------
// R.1.16.1: probe retired. Assertion form:
// R1161MpsShor.OrderRecoveredOnBothBackends below.
#if 0
TEST(DiagR1160MpsShor, EndToEndFindOrderBothBackends) {
    LocalBackend::Config sv_cfg;
    sv_cfg.simulator = LocalBackend::SimType::STATEVECTOR;
    LocalBackend::Config mps_cfg;
    mps_cfg.simulator = LocalBackend::SimType::MPS;
    mps_cfg.mps_bond_dim = kBond;

    std::cout << "\n[e2e] find_order(2, 15, 9, backend, seed); expected r="
              << kExpectedOrder << "\n";
    int sv_ok = 0, mps_ok = 0;
    for (uint64_t seed = 1; seed <= 5; ++seed) {
        LocalBackend sv_backend(sv_cfg);
        LocalBackend mps_backend(mps_cfg);
        const uint64_t r_sv = Shor::find_order(kA, kN, kEval, sv_backend, seed);
        const uint64_t r_mps = Shor::find_order(kA, kN, kEval, mps_backend, seed);
        sv_ok += (r_sv == kExpectedOrder) ? 1 : 0;
        mps_ok += (r_mps == kExpectedOrder) ? 1 : 0;
        std::cout << "[e2e] seed=" << seed << "  r_sv=" << r_sv
                  << "  r_mps=" << r_mps << "\n";
    }
    std::cout << "[e2e] recovered order: SV " << sv_ok << "/5, MPS " << mps_ok
              << "/5\n";

    EXPECT_GE(sv_ok, 4) << "statevector reference itself failing: the defect "
                           "is upstream of the MPS layer";
    // No assertion on mps_ok: capturing the symptom is this probe's job.
}
#endif  // retired probe 3

// =============================================================================
// =============================================================================
// R.1.16.1 REGRESSION SUITE — assertion forms of the retired probes above.
//
// The R.1.16.0 investigation's central lesson (diagnostic run 5): a
// rank-3-DAMAGED state still recovered Shor's order on 5/5 seeds, so order
// recovery alone is a worthless acceptance criterion. This suite pins STATE
// EXACTNESS — fidelity against the statevector reference, accumulated
// truncation error, and reached bond dimension — at every pipeline stage,
// plus the sampler bound, the qudit latent-pattern canary, the two Eigen-
// defect reproducers as standing assertions, and svd_truncate's fail-loud
// contract. Reuses the probe helpers and tests/diag_r1160_matrices.hpp.
// =============================================================================
// =============================================================================

// State exactness at every stage boundary of the 13-qubit period-finding
// circuit: prep, after each of the 9 PERMUTATIONs, and the IQFT in quarters.
// Bond 64 is mathematically exact at 13 qubits, so fidelity must be 1 and
// truncation error must be numerical dust at EVERY cut (observed post-fix:
// fid 1.000000000000, terr <= 4.3e-30, chi == 4 throughout).
TEST(R1161MpsShor, StateExactAtEveryStage) {
    const auto qc = Shor::build_period_finding_circuit(kA, kN, kEval, kTarget);
    ASSERT_EQ(qc.n_qubits, kQubits);

    std::vector<size_t> cuts;
    std::vector<size_t> perm_ends;
    for (size_t i = 0; i < qc.instructions.size(); ++i) {
        if (qc.instructions[i].type == Instruction::GateType::PERMUTATION) {
            perm_ends.push_back(i + 1);
        }
    }
    ASSERT_EQ(perm_ends.size(), 9u) << "circuit shape changed; update suite";
    cuts.push_back(perm_ends.front() - 1);
    for (size_t e : perm_ends) cuts.push_back(e);
    const size_t tail_start = perm_ends.back();
    const size_t tail = qc.instructions.size() - tail_start;
    for (int q = 1; q <= 4; ++q) {
        cuts.push_back(tail_start + (tail * static_cast<size_t>(q)) / 4);
    }

    for (size_t k : cuts) {
        SCOPED_TRACE("prefix k=" + std::to_string(k));
        const auto pre = prefix_of(qc, k);
        MPSSimulator mps;
        auto res = mps.run(pre, kBond, /*shots=*/0, /*seed=*/42);

        const double fid =
            fidelity(sv_state_of(pre), mps_state_to_vec(res.final_state));
        EXPECT_NEAR(fid, 1.0, 1e-9)
            << "MPS state diverged from the statevector reference";
        EXPECT_LT(res.final_state.truncation_error(), 1e-24)
            << "real weight was truncated even though chi=64 is exact here";
        EXPECT_LE(res.final_state.current_max_bond_dim(), 8);

        const auto scan = scan_tensors(res.final_state);
        EXPECT_FALSE(scan.corrupt) << "non-finite tensor entry";
    }

    // Final state: the exact Schmidt structure of this circuit.
    {
        MPSSimulator mps;
        auto res = mps.run(qc, kBond, /*shots=*/0, /*seed=*/42);
        EXPECT_EQ(res.final_state.current_max_bond_dim(), 4)
            << "final bond must equal the exact Schmidt rank";
    }
}

// The upgraded R.1.13.1 anchor: order recovery on BOTH backends, per seed —
// meaningful only in combination with StateExactAtEveryStage (run-5 lesson).
TEST(R1161MpsShor, OrderRecoveredOnBothBackends) {
    LocalBackend::Config sv_cfg;
    sv_cfg.simulator = LocalBackend::SimType::STATEVECTOR;
    LocalBackend::Config mps_cfg;
    mps_cfg.simulator = LocalBackend::SimType::MPS;
    mps_cfg.mps_bond_dim = kBond;

    for (uint64_t seed = 1; seed <= 5; ++seed) {
        SCOPED_TRACE("seed " + std::to_string(seed));
        LocalBackend sv_backend(sv_cfg);
        LocalBackend mps_backend(mps_cfg);
        EXPECT_EQ(Shor::find_order(kA, kN, kEval, sv_backend, seed),
                  kExpectedOrder);
        EXPECT_EQ(Shor::find_order(kA, kN, kEval, mps_backend, seed),
                  kExpectedOrder)
            << "MPS order recovery regressed (issue #44)";
    }
}

// Sampler agreement with its own (now exact) state at 13 qubits, on the
// eval-register marginal. Post-fix observation: TVD 0.0027 at 4096 shots
// against ideal 0.25 peaks at {0, 128, 256, 384}.
TEST(R1161MpsShor, SamplerWithinNoiseAt13Qubits) {
    auto qc = Shor::build_period_finding_circuit(kA, kN, kEval, kTarget);
    qc.n_clbits = kQubits;
    qc.measure_all();

    MPSSimulator mps;
    const int shots = 4096;
    auto res = mps.run(qc, kBond, shots, /*seed=*/42);

    const auto own = eval_marginal(mps_state_to_vec(res.final_state));
    std::vector<double> sampled(size_t{1} << kEval, 0.0);
    int total = 0;
    for (const auto& [bits, count] : res.counts) {
        ASSERT_EQ(bits.size(), static_cast<size_t>(kQubits));
        sampled[std::stoul(bits.substr(bits.size() - kEval), nullptr, 2)] +=
            count;
        total += count;
    }
    ASSERT_EQ(total, shots);
    for (auto& p : sampled) p /= shots;

    double tvd = 0.0;
    int support = 0;
    for (size_t i = 0; i < own.size(); ++i) {
        tvd += std::abs(own[i] - sampled[i]);
        support += (own[i] > 1.0 / (10.0 * shots)) ? 1 : 0;
    }
    tvd /= 2.0;

    EXPECT_EQ(support, 4) << "the exact state has exactly four eval peaks";
    for (size_t peak : {size_t{0}, size_t{128}, size_t{256}, size_t{384}}) {
        EXPECT_NEAR(own[peak], 0.25, 1e-9) << "peak " << peak;
    }
    const double noise_scale =
        std::sqrt(static_cast<double>(std::max(support, 1)) /
                  (3.14159265358979 * shots));
    EXPECT_LT(tvd, 2.5 * noise_scale + 0.02);
}

// Latent-pattern CANARY: qudit MPS carries the same unguarded truncation
// loops the qubit MPS had (owner ruling: untouched until it misbehaves).
// This drives QuditMPS at d=2, 13 sites through the exact R.1.16.0 failing
// scenario and must stay exact — if this ever goes red, port the
// select-verify-fallback hardening to qudit_mps.cpp.
TEST(R1161MpsShor, QuditTwinStaysExact) {
    const auto qc = Shor::build_period_finding_circuit(kA, kN, kEval, kTarget);
    const auto prefix19 = prefix_of(qc, 19);
    const auto psi19 = sv_state_of(prefix19);
    QuditStatevector qsv(kQubits, /*d=*/2);
    for (size_t i = 0; i < psi19.size(); ++i) {
        qsv.amplitudes[i] = Complex128(psi19[i].real(), psi19[i].imag());
    }
    QuditMPS qm(qsv, /*max_bond_dim=*/kBond);

    constexpr double s2 = 0.7071067811865475;
    const std::vector<Complex128> H2 = {
        Complex128(s2, 0), Complex128(s2, 0),
        Complex128(s2, 0), Complex128(-s2, 0)};
    std::vector<Complex128> SW(16, Complex128(0, 0));
    SW[0] = SW[6] = SW[9] = SW[15] = Complex128(1, 0);

    for (size_t i = 19; i < qc.instructions.size(); ++i) {
        const auto& inst = qc.instructions[i];
        if (inst.type == Instruction::GateType::H) {
            qm.apply_1qudit(inst.qubits[0], H2);
        } else if (inst.type == Instruction::GateType::CP) {
            std::vector<Complex128> CP(16, Complex128(0, 0));
            CP[0] = CP[5] = CP[10] = Complex128(1, 0);
            CP[15] = Complex128(std::cos(inst.params[0]),
                                std::sin(inst.params[0]));
            qm.apply_2qudit(inst.qubits[0], inst.qubits[1], CP);
        } else if (inst.type == Instruction::GateType::SWAP) {
            qm.apply_2qudit(inst.qubits[0], inst.qubits[1], SW);
        } else {
            FAIL() << "unexpected gate in IQFT tail: " << inst.gate_name();
        }
    }

    const auto psi_ref = sv_state_of(prefix_of(qc, qc.instructions.size()));
    const auto qfinal = qm.to_statevector();
    std::complex<double> ov(0, 0);
    for (size_t i = 0; i < psi_ref.size(); ++i) {
        const auto& a = qfinal.amplitudes[i];
        ASSERT_FALSE(diag_r1160::fp_bad(a.real) || diag_r1160::fp_bad(a.imag))
            << "qudit MPS produced a non-finite amplitude at index " << i
            << ": the latent pattern manifested — port the R.1.16.0 fix";
        ov += std::conj(psi_ref[i]) * std::complex<double>(a.real, a.imag);
    }
    EXPECT_NEAR(std::norm(ov), 1.0, 1e-9)
        << "qudit MPS diverged on the R.1.16.0 scenario";
}

// Eigen-defect reproducer 1 as a standing assertion (fast-math TU leg; the
// strict-FP twin lives in R1161StrictFP): whatever garbage Eigen emits in
// the null space, the KEPT rank-4 slice of the poison theta must be finite
// and reconstruct it exactly. This is precisely the contract the R.1.16.0
// svd_truncate hardening relies on.
TEST(R1161MpsShor, PoisonThetaKeptSliceContract) {
    const auto theta = diag_r1160::build_poison_theta();
    ASSERT_FALSE(diag_r1160::matrix_bad(theta));
    const auto r =
        diag_r1160::run_svd_report<Eigen::JacobiSVD<Eigen::MatrixXcd>>(theta);
    print_svd_report("r1161/jacobi/poison", r);
    EXPECT_EQ(r.rank_1e12, 4) << "poison theta has exact Schmidt rank 4";
    EXPECT_NEAR(r.sum_sq, r.frob_sq, 1e-12);
    EXPECT_FALSE(r.kept_slice_bad);
    EXPECT_GE(r.trunc_recon_err, 0.0);
    EXPECT_LT(r.trunc_recon_err, 1e-12);
}

// Eigen-defect reproducer 2 as a standing assertion: JacobiSVD is the
// correct reference on the R.1.11.2 Simon matrix (twelve-fold degenerate
// spectrum). The BDCSVD leg stays PRINT-ONLY documentation — it is the
// known-broken upstream path; if Eigen ever fixes it, the printout is how
// we notice.
TEST(R1161MpsShor, Simon36JacobiReference) {
    const auto M = diag_r1160::build_bdcsvd_bug_matrix();
    ASSERT_FALSE(diag_r1160::matrix_bad(M));
    const auto rj =
        diag_r1160::run_svd_report<Eigen::JacobiSVD<Eigen::MatrixXcd>>(M);
    const auto rb =
        diag_r1160::run_svd_report<Eigen::BDCSVD<Eigen::MatrixXcd>>(M);
    print_svd_report("r1161/jacobi/simon36", rj);
    print_svd_report("r1161/bdc/simon36", rb);  // documentation only
    EXPECT_FALSE(rj.corrupt);
    EXPECT_EQ(rj.rank_1e12, 12);
    EXPECT_NEAR(rj.sum_sq, rj.frob_sq, 1e-10);
    EXPECT_LT(rj.recon_err, 1e-10);
}

// svd_truncate's fail-loud contract: an MPS whose tensors already carry
// non-finite data (injected via the public tensors member) must make the
// next two-qubit gate THROW — never continue with a corrupt tensor. Both
// the SVD path and the Gram fallback receive garbage, so the double-failure
// branch is exercised.
TEST(R1161MpsShor, SvdTruncateThrowsOnUnrecoverableInput) {
    MPSState st(2, kBond);
    st.tensors[0].data[0] = Complex128(
        std::numeric_limits<double>::quiet_NaN(), 0.0);

    std::array<Complex128, 16> CZ{};
    CZ[0 * 4 + 0] = CZ[1 * 4 + 1] = CZ[2 * 4 + 2] = Complex128(1, 0);
    CZ[3 * 4 + 3] = Complex128(-1, 0);
    EXPECT_THROW(st.apply_two_qubit_gate(CZ, 0, 1), std::runtime_error)
        << "a non-finite theta must fail loud, not propagate";
}

// -----------------------------------------------------------------------------
// Breadth sweep (R.1.16.1 gap 1): the Eigen trigger family is flat
// degenerate Schmidt spectra with exact zeros — which GHZ and stabilizer
// states are made of — plus dense generic spectra. At n = 12 with chi = 64
// = 2^(n/2), NO 12-qubit state can exceed the bond cap, so every circuit
// below must be simulated EXACTLY regardless of what its spectrum looks
// like. Turns "the fix works on Shor" into "the fix works on the family".
// -----------------------------------------------------------------------------

namespace {

void expect_exact_at_bond64(const QuantumCircuit& qc, const char* what,
                            int expected_final_chi = -1) {
    ASSERT_LE(qc.n_qubits, 12) << "exact-regime guarantee needs n <= 12";
    MPSSimulator mps;
    auto res = mps.run(qc, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/42);
    const double fid =
        fidelity(sv_state_of(qc), mps_state_to_vec(res.final_state));
    EXPECT_NEAR(fid, 1.0, 1e-9) << what << ": MPS diverged from statevector";
    EXPECT_LT(res.final_state.truncation_error(), 1e-18)
        << what << ": truncated in the exact regime";
    EXPECT_FALSE(scan_tensors(res.final_state).corrupt) << what;
    EXPECT_LE(res.final_state.current_max_bond_dim(), 64) << what;
    if (expected_final_chi > 0) {
        EXPECT_EQ(res.final_state.current_max_bond_dim(), expected_final_chi)
            << what << ": final Schmidt rank is analytic";
    }
}

}  // namespace

// GHZ-12 over a deliberately long-range CX chain (every gate swap-routed):
// flat rank-2 spectrum with exact zeros — the smallest member of the
// trigger family.
TEST(R1161MpsShor, BreadthGhz12LongRangeExact) {
    QuantumCircuit qc(12);
    qc.h(0);
    qc.cx(0, 11).cx(11, 5).cx(5, 8).cx(8, 2).cx(2, 9).cx(9, 4);
    qc.cx(4, 10).cx(10, 1).cx(1, 6).cx(6, 3).cx(3, 7);
    expect_exact_at_bond64(qc, "GHZ-12", /*expected_final_chi=*/2);
}

// Stabilizer ladder at n = 12: stabilizer states have exactly FLAT Schmidt
// spectra (all kept sigmas equal, rest exact zeros) — the degenerate
// pattern that broke Eigen's SVDs in R.1.16.0, at every cut, layer after
// layer.
TEST(R1161MpsShor, BreadthCliffordLadder12Exact) {
    QuantumCircuit qc(12);
    for (int layer = 0; layer < 10; ++layer) {
        for (int i = 0; i < 12; ++i) qc.h(i);
        for (int i = layer % 2; i + 1 < 12; i += 2) qc.cx(i, i + 1);
        for (int i = 0; i < 12; ++i) {
            if ((i + layer) % 3 == 0) qc.s(i);
        }
    }
    expect_exact_at_bond64(qc, "Clifford ladder-12");
}

// Dense brickwork at n = 12, depth 16, deterministic angles: generic
// non-degenerate spectra that saturate chi toward the 64 cap — the
// complementary stress to the flat-spectrum cases.
TEST(R1161MpsShor, BreadthBrickwork12Exact) {
    QuantumCircuit qc(12);
    for (int layer = 0; layer < 16; ++layer) {
        for (int i = 0; i < 12; ++i) {
            switch ((i + layer) % 3) {
                case 0: qc.h(i); break;
                case 1: qc.t(i); break;
                default: qc.rz(0.31 + 0.07 * layer + 0.011 * i, i); break;
            }
        }
        for (int i = layer % 2; i + 1 < 12; i += 2) qc.cx(i, i + 1);
    }
    expect_exact_at_bond64(qc, "brickwork-12");
}

// -----------------------------------------------------------------------------
// Gram fallback route (R.1.16.1 gap 2): the rescue MATH validated
// standalone on both reproducer matrices (this TU = fast-math leg; the
// strict-FP twin lives in R1161StrictFP). Direct wiring coverage of the
// library's fallback (engagement counter + seam) is queued follow-up work —
// it needs a src change and cannot ship in a tests-only release.
// -----------------------------------------------------------------------------

TEST(R1161MpsShor, GramRouteReconstructsPoisonTheta) {
    const auto theta = diag_r1160::build_poison_theta();
    const auto r = diag_r1160::run_gram_route_report(theta);
    print_svd_report("r1161/gram/poison", r);
    EXPECT_EQ(r.rank_1e12, 4);
    EXPECT_FALSE(r.kept_slice_bad);
    EXPECT_GE(r.trunc_recon_err, 0.0);
    EXPECT_LT(r.trunc_recon_err, 1e-10);
    EXPECT_NEAR(r.sum_sq, r.frob_sq, 1e-10);
}

TEST(R1161MpsShor, GramRouteReconstructsSimon36) {
    const auto M = diag_r1160::build_bdcsvd_bug_matrix();
    const auto r = diag_r1160::run_gram_route_report(M);
    print_svd_report("r1161/gram/simon36", r);
    EXPECT_EQ(r.rank_1e12, 12);
    EXPECT_FALSE(r.kept_slice_bad);
    EXPECT_GE(r.trunc_recon_err, 0.0);
    EXPECT_LT(r.trunc_recon_err, 1e-8);
    EXPECT_NEAR(r.sum_sq, r.frob_sq, 1e-8)
        << "the Gram route must preserve the Frobenius norm BDCSVD violates";
}
