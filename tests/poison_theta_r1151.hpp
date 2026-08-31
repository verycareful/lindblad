// poison_theta_r1151.hpp - the #44 matrix, frozen.
//
// The 8x8 two-site theta that Eigen 3.4.0 mis-factorises: JacobiSVD returns a
// NaN inside column 6 of U, a null-space column, while S, V and the leading
// rank-4 slice stay exact. BDCSVD delegates to Jacobi at this size and returns
// the same output.
//
// A LITERAL, deliberately. diag_r1160_matrices.hpp builds its theta by evolving
// a circuit prefix through the library, which means the matrix tracks the
// library rather than holding still: the R.1.16.0 fix quarantined
// src/simulators/mps_sim.cpp to -fno-fast-math and hardened the truncation, and
// issue #70 later moved the Hadamard amplitude by one ULP. Either alone changes
// what comes out, and the defect does not survive the change. A reproducer that
// is a recipe cannot outlive the code it calls; this one is a fact.
//
// Captured at commit 17b224e (R.1.15.1), the last commit before the fix, built
// against Eigen 3.4.0 under the project's -ffast-math. Emitted with %a, so
// every value round-trips exactly rather than through a decimal approximation.

#pragma once

#include <Eigen/Dense>

#include <complex>

namespace poison_r1151 {
namespace {  // internal linkage, same reason as diag_r1160_matrices.hpp

inline Eigen::MatrixXcd build_poison_theta_r1151() {
    Eigen::MatrixXcd M(8, 8);
    static const double kRe[] = {
        0x1.6a09e667f3bcdp-1, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x1.6a09e667f3bc9p-1, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x1.6a09e667f3bccp-1, -0x1.a827999fcef36p-271, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x1.6a09e667f3bcbp-1, 0x1.a827999fcef36p-271, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x0p+0, 0x1.6a09e667f3bcbp-1, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x0p+0, 0x1.6a09e667f3bc4p-1, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x0p+0, 0x0p+0, -0x1.6a09e667f3bcbp-1, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x0p+0, 0x0p+0, -0x1.6a09e667f3bc7p-1, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
    };
    static const double kIm[] = {
        0x1.092b8a0189396p-386, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x1.092b8a0189394p-386, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, -0x1.6a09e667f3bcdp-56, -0x1.a827999fcef35p-216, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, -0x1.6a09e667f3bccp-56, 0x1.a827999fcef36p-216, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x0p+0, 0x1.239c9da146be2p-57, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x0p+0, 0x1.239c9da146bddp-57, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x0p+0, 0x0p+0, 0x1.3fffffffffffep-53, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
        0x0p+0, 0x0p+0, 0x0p+0, 0x1.3fffffffffffdp-53, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 
    };
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            M(r, c) = std::complex<double>(kRe[r * 8 + c],
                                           kIm[r * 8 + c]);
    return M;
}

}  // namespace (internal linkage)

}  // namespace poison_r1151
