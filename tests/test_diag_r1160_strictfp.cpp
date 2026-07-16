// TEMPORARY DIAGNOSTIC (R.1.16.0, issue #44) — STRICT-FP twin.
//
// This TU is compiled with -fno-fast-math (per-source override in
// tests/CMakeLists.txt) while test_diag_r1160_mps_shor.cpp compiles under
// the project-wide -ffast-math. Eigen is header-only, so each TU
// instantiates JacobiSVD/BDCSVD under its own flags: identical inputs,
// identical Eigen version, only the FP model differs. Together the two
// suites answer, with everything else held constant:
//   1. Does JacobiSVD on the EXACT first-NaN Shor matrix (poison theta)
//      misbehave under fast-math and behave under strict FP?  -> #44 root
//      cause is the FP model, and strict-FP on mps_sim.cpp is the fix.
//   2. Does BDCSVD on the R.1.11.2 bug matrix (36x36 degenerate complex)
//      recover under strict FP?  -> the "Eigen BDCSVD accuracy bug"
//      (local note, worked around via JacobiSVD) was a fast-math casualty,
//      not an Eigen defect — re-evaluate BDC after the fix.
//      If it stays wrong here, it is a genuine Eigen bug and the note's
//      upstream-report plan stands.
//
// A third check calls the LIBRARY's own gate application (lindblad_core is
// compiled with fast-math regardless of this TU's flags): it must still
// corrupt, proving the defect lives in the library TU's codegen, not in
// anything this test does differently.

#include <gtest/gtest.h>

#include "diag_r1160_matrices.hpp"

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
TEST(DiagR1160StrictFP, JacobiPoisonKeptSliceIsCleanAndExact) {
    const auto theta = build_poison_theta();
    ASSERT_FALSE(matrix_bad(theta)) << "poison theta reconstruction corrupt "
                                       "before any SVD — library evolution "
                                       "differs from the probe run";
    const auto r = run_svd_report<Eigen::JacobiSVD<Eigen::MatrixXcd>>(theta);
    print_report("strict/jacobi/poison", r);

    EXPECT_NEAR(r.sum_sq, r.frob_sq, 1e-12);
    EXPECT_EQ(r.rank_1e12, 4) << "poison theta has exact Schmidt rank 4";
    EXPECT_FALSE(r.kept_slice_bad)
        << "NaN inside the KEPT columns: slice-based hardening insufficient";
    EXPECT_GE(r.trunc_recon_err, 0.0);
    EXPECT_LT(r.trunc_recon_err, 1e-12)
        << "rank-4 slice does not reconstruct theta";
}

TEST(DiagR1160StrictFP, BdcOnR1112BugMatrix) {
    const auto M = build_bdcsvd_bug_matrix();
    ASSERT_FALSE(matrix_bad(M));
    const auto rb = run_svd_report<Eigen::BDCSVD<Eigen::MatrixXcd>>(M);
    const auto rj = run_svd_report<Eigen::JacobiSVD<Eigen::MatrixXcd>>(M);
    print_report("strict/bdc/simon36", rb);
    print_report("strict/jacobi/simon36", rj);

    // Jacobi is the reference (correct in R.1.11.2 under fast-math even).
    EXPECT_FALSE(rj.corrupt);
    EXPECT_NEAR(rj.sum_sq, rj.frob_sq, 1e-10);
    EXPECT_LT(rj.recon_err, 1e-10);
    EXPECT_EQ(rj.rank_1e12, 12) << "expected the twelve-fold degenerate "
                                   "spectrum from the bug note";

    // Probe run 2 answered the maintainer's question: BDC is bit-identically
    // WRONG under strict FP too (0.9861 norm violation, spurious 0.2635,
    // 3.8e-170 subnormal) => the R.1.11.2 finding is a GENUINE Eigen 3.4.0
    // BDCSVD defect, not fast-math. The upstream-report plan in
    // local/plans/eigen-bdcsvd-bug.md stands, now with a strict-FP data
    // point. No hard assertion: this test documents, the note decides.
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
