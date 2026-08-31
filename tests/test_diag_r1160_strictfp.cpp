// The R.1.16.0 (#44) diagnostic's strict leg.
//
// This TU compiles -fno-fast-math (per-source override in tests/CMakeLists.txt)
// while test_diag_r1160_mps_shor.cpp compiles the same shared helpers under the
// project-wide -ffast-math. The flag governs THIS FILE's arithmetic: how it
// reads a factorisation into a report and how it evaluates an assertion.
//
// It does not govern Eigen's, and no per-file flag can. Eigen is header-only,
// so its instantiations carry vague linkage: every naming TU emits its own weak
// symbol, the linker keeps one per binary chosen by mangled name, and compile
// flags are no part of that name. Every factorisation these two suites perform
// therefore goes through lindblad::detail::eigen_backend, which is strict
// inside its own translation unit and is the tree's only emitter.
// test_v11241_seam_rules.cpp fails if a second one appears.
//
// What the pair still answers is narrower than it once claimed, and still worth
// having: whether the SUITE's own reading of a factorisation depends on the
// floating-point model it was compiled under. Two legs agreeing is what makes
// these reports evidence rather than an artifact of the flags the reader was
// built with.
//
// The three probes below the retired marker are the audit trail of the
// R.1.16.0 investigation and are not compiled.

#include <gtest/gtest.h>

#include "diag_r1160_matrices.hpp"

#include "lindblad/simulators/statevector_sim.hpp"

#include <iomanip>
#include <iostream>

using namespace diag_r1160;

namespace {

void print_report(const char* tag, const SvdReport& r) {
    std::cout << std::scientific << std::setprecision(6);
    print_svd_report_line(std::cout, tag, r);
    std::cout << std::fixed;
}

}  // namespace

// Probe run 2 falsified the fast-math hypothesis: Jacobi-on-poison is
// corrupt under BOTH FP models (spectrum exact, NaN in the singular
// VECTORS), so the defect is Eigen 3.4.0 JacobiSVD behaviour on this
// degenerate + rank-deficient input, not the flags. The fix-viability
// claim therefore moves down a level: svd_truncate only ever KEEPS the
// leading rank columns — if that kept slice is finite and reconstructs M,
// a hardened truncation (fast-math-immune rank loop + fail-loud corrupt-
// slice check) fixes #44 without touching Eigen or the flags.
//
// R.1.16.1: probe retired (kept for the audit trail). Assertion form:
// R1161StrictFP.PoisonThetaKeptSliceContract below.
#if 0
TEST(DiagR1160StrictFP, JacobiPoisonKeptSliceIsCleanAndExact) {
    const auto theta = build_poison_theta();
    ASSERT_FALSE(matrix_bad(theta)) << "poison theta reconstruction corrupt "
                                       "before any SVD — library evolution "
                                       "differs from the probe run";
    const auto r = run_svd_report(theta, lindblad::SVDMethod::Jacobi);
    print_report("strict/jacobi/poison", r);

    EXPECT_NEAR(r.sum_sq, r.frob_sq, 1e-12);
    EXPECT_EQ(r.rank, 4) << "poison theta has exact Schmidt rank 4";
    EXPECT_FALSE(r.kept_slice_bad)
        << "NaN inside the KEPT columns: slice-based hardening insufficient";
    EXPECT_GE(r.trunc_recon_err, 0.0);
    EXPECT_LT(r.trunc_recon_err, 1e-12)
        << "rank-4 slice does not reconstruct theta";
}

TEST(DiagR1160StrictFP, BdcOnR1112BugMatrix) {
    const auto M = build_bdcsvd_bug_matrix();
    ASSERT_FALSE(matrix_bad(M));
    const auto rb = run_svd_report(M, lindblad::SVDMethod::BDC);
    const auto rj = run_svd_report(M, lindblad::SVDMethod::Jacobi);
    print_report("strict/bdc/simon36", rb);
    print_report("strict/jacobi/simon36", rj);

    // Jacobi is the reference (correct in R.1.11.2 under fast-math even).
    EXPECT_FALSE(rj.corrupt);
    EXPECT_NEAR(rj.sum_sq, rj.frob_sq, 1e-10);
    EXPECT_LT(rj.recon_err, 1e-10);
    EXPECT_EQ(rj.rank, 12) << "expected the twelve-fold degenerate "
                                   "spectrum from the bug note";

    // BDC is bit-identically WRONG under strict FP too (0.9861 norm violation,
    // spurious 0.2635, 3.8e-170 subnormal), so the R.1.11.2 finding is a
    // GENUINE Eigen 3.4.0 BDCSVD defect and not a fast-math artifact. No hard
    // assertion: this test records the strict-FP data point.
}

TEST(DiagR1160StrictFP, LibraryGateApplicationStillCorruptsUnderItsOwnFlags) {
    // Same pre-poison state, but let lindblad_core (fast-math TU) do the
    // gate + SVD: corruption must reproduce REGARDLESS of this TU's flags,
    // pinning the defect to the library TU's codegen.
    const auto qc = lindblad::algorithms::Shor::build_period_finding_circuit(
        2, 15, 9, 4);
    lindblad::QuantumCircuit prefix(qc.n_qubits, qc.n_clbits);
    prefix.instructions.assign(qc.instructions.begin(),
                               qc.instructions.begin() + 27);  // incl. i=26
    lindblad::MPSSimulator sim;
    auto res = sim.run(prefix, 64, /*shots=*/0, /*seed=*/42);

    bool corrupt = false;
    for (const auto& t : res.final_state.tensors)
        for (const auto& c : t.data)
            corrupt = corrupt || fp_bad(c.real) || fp_bad(c.imag);
    std::cout << "[svd:strict-TU/library-run] state corrupt after i<=26: "
              << (corrupt ? "YES (expected while mps_sim.cpp is fast-math)"
                          : "no (library flags already fixed?)")
              << "\n";
}
#endif  // retired probes (all three; audit trail for the R.1.16.0 saga)

// =============================================================================
// R.1.16.1 REGRESSION SUITE — strict-FP twins of the R1161MpsShor Eigen
// reproducers, plus the library-evolution exactness check. This TU compiles
// -fno-fast-math (per-source property in tests/CMakeLists.txt), so these
// assertions pin the defects-and-contracts as FLAG-INDEPENDENT facts: if a
// future Eigen upgrade or flag change alters the picture, the fast/strict
// pair diverging is the first signal.
// =============================================================================

// Strict-FP twin of R1161MpsShor.PoisonThetaKeptSliceContract, made portable
// across compilers in R.1.18.2. The original hard-pinned WHERE Eigen's
// garbage lands (null space only) — true under WSL/clang, but GCC 13 at -O3
// was observed relocating the NaN INTO S, displacing a real sigma (the
// 'interleaved' morph from svd_truncate's catalogue; -O0 matches clang, so
// this is pure codegen). Eigen guarantees nothing about the landing site, so
// the honest portable contract is acceptance-soundness, mirroring the
// verify ladder svd_truncate actually runs: if the detectors ACCEPT the kept
// slice (bit-finite and reconstructing), it must genuinely be the exact
// rank-4 truncation; if they REJECT, the Gram fallback takes over — its math
// is pinned by GramRouteReconstructsPoisonTheta and the library wiring by
// LibraryEvolutionExactThroughPoisonRegion, so the reject branch documents
// the per-config shape via the printout instead of asserting it.
TEST(R1161StrictFP, PoisonThetaKeptSliceContract) {
    const auto theta = build_poison_theta();
    ASSERT_FALSE(matrix_bad(theta));
    const auto r = run_svd_report(theta, lindblad::SVDMethod::Jacobi);
    print_report("r1161-strict/jacobi/poison", r);

    const bool accepted = !r.kept_slice_bad && r.trunc_recon_err >= 0.0 &&
                          r.trunc_recon_err < 1e-9;
    if (accepted) {
        // Acceptance must be sound: exact spectrum, exact reconstruction.
        // A factorisation that clears the detectors while being wrong is the
        // one outcome that would silently corrupt the library.
        EXPECT_EQ(r.rank, 4);
        EXPECT_NEAR(r.sum_sq, r.frob_sq, 1e-12);
        EXPECT_GE(r.trunc_recon_err, 0.0);
        EXPECT_LT(r.trunc_recon_err, 1e-12);
    } else {
        // Detectably corrupt => svd_truncate falls back (rescue proven by the
        // Gram-route and library-level tests above/below). No hard assertion
        // on the corruption's shape: it morphs with compiler and opt level.
        std::cout << "[svd:r1161-strict/jacobi/poison] verify ladder rejects "
                     "this factorisation; Gram fallback path engages\n";
    }
}

// Twin of R1161MpsShor.Simon36JacobiReference.
// Both backends are asserted. The 36x36 Simon residual is the matrix whose
// BDCSVD factorisation violated the Frobenius identity on Eigen 3.4.0, returning
// a spurious singular value in place of a real one while every entry stayed
// finite. The pinned Eigen returns the correct spectrum, so the leg that used to
// be printed is now held to the same bar as the reference, and a pin that
// reintroduces the defect fails here rather than being noticed in a printout.
//
// The Frobenius identity is what carries these assertions. A wrong spectrum that
// is entirely finite has nothing to pattern-match on: only the norm catches it,
// which is why the sum is compared against the matrix rather than the factors
// being scanned.
TEST(R1161StrictFP, Simon36JacobiReference) {
    const auto M = build_bdcsvd_bug_matrix();
    ASSERT_FALSE(matrix_bad(M));
    const auto rj = run_svd_report(M, lindblad::SVDMethod::Jacobi);
    const auto rb = run_svd_report(M, lindblad::SVDMethod::BDC);
    print_report("r1161-strict/jacobi/simon36", rj);
    print_report("r1161-strict/bdc/simon36", rb);
    for (const auto* leg : {&rj, &rb}) {
        const bool jacobi = (leg == &rj);
        const char* which = jacobi ? "jacobi" : "bdc";
        EXPECT_FALSE(leg->corrupt) << which << ": factorisation corrupt";
        EXPECT_EQ(leg->rank, 12) << which << ": retained rank";
        EXPECT_NEAR(leg->sum_sq, leg->frob_sq, 1e-10)
            << which << ": sum of sigma^2 is " << leg->sum_sq
            << " against ||M||_F^2 " << leg->frob_sq
            << ", so the spectrum is wrong rather than imprecise";
        EXPECT_LT(leg->recon_err, 1e-10)
            << which << ": U S V^H does not reconstruct M";
    }
}

// The library's own evolution through the original poison region must now
// be EXACT (the retired probe merely checked for non-finite entries; a
// finite-but-wrong state slipped past that in diagnostic run 6 — fidelity
// is the honest criterion).
TEST(R1161StrictFP, LibraryEvolutionExactThroughPoisonRegion) {
    const auto qc = lindblad::algorithms::Shor::build_period_finding_circuit(
        2, 15, 9, 4);
    lindblad::QuantumCircuit prefix(qc.n_qubits, qc.n_clbits);
    prefix.instructions.assign(qc.instructions.begin(),
                               qc.instructions.begin() + 27);  // incl. i=26
    lindblad::MPSSimulator sim;
    auto res = sim.run(prefix, 64, /*shots=*/0, /*seed=*/42);

    // Statevector reference for the same prefix.
    lindblad::Statevector sv(prefix.n_qubits);
    sv.initialize_basis(0);
    lindblad::StatevectorSimulator ssim;
    for (const auto& inst : prefix.instructions) ssim.apply_instruction(sv, inst);

    const lindblad::Statevector mps_sv = res.final_state.to_statevector();
    std::complex<double> ov(0, 0);
    const size_t dim = size_t{1} << prefix.n_qubits;
    for (size_t i = 0; i < dim; ++i) {
        ASSERT_FALSE(fp_bad(mps_sv.real_parts[i]) || fp_bad(mps_sv.imag_parts[i]))
            << "non-finite amplitude at " << i;
        ov += std::conj(std::complex<double>(sv.real_parts[i], sv.imag_parts[i])) *
              std::complex<double>(mps_sv.real_parts[i], mps_sv.imag_parts[i]);
    }
    EXPECT_NEAR(std::norm(ov), 1.0, 1e-9)
        << "library MPS evolution diverged in the poison region";
    EXPECT_LT(res.final_state.truncation_error(), 1e-24);
}

// Strict-FP twins of the Gram-route validation (R.1.16.1 gap 2): the rescue
// math must be exact under IEEE semantics on both reproducer matrices.
TEST(R1161StrictFP, GramRouteReconstructsPoisonTheta) {
    const auto theta = build_poison_theta();
    const auto r = run_gram_route_report(theta);
    print_report("r1161-strict/gram/poison", r);
    EXPECT_EQ(r.rank, 4);
    EXPECT_GT(r.sigma_floor, 1e-12)
        << "the Gram route must count rank against its own sqrt(eps)-scaled "
           "floor, not the absolute default";
    EXPECT_FALSE(r.kept_slice_bad);
    EXPECT_GE(r.trunc_recon_err, 0.0);
    EXPECT_LT(r.trunc_recon_err, 1e-10);
    EXPECT_NEAR(r.sum_sq, r.frob_sq, 1e-10);
}

TEST(R1161StrictFP, GramRouteReconstructsSimon36) {
    const auto M = build_bdcsvd_bug_matrix();
    const auto r = run_gram_route_report(M);
    print_report("r1161-strict/gram/simon36", r);
    EXPECT_EQ(r.rank, 12);
    EXPECT_GT(r.sigma_floor, 1e-12)
        << "the Gram route must count rank against its own sqrt(eps)-scaled "
           "floor, not the absolute default";
    EXPECT_FALSE(r.kept_slice_bad);
    EXPECT_GE(r.trunc_recon_err, 0.0);
    EXPECT_LT(r.trunc_recon_err, 1e-8);
    EXPECT_NEAR(r.sum_sq, r.frob_sq, 1e-8);
}
