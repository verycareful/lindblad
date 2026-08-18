// diag_r1160_matrices.hpp — shared builders for the R.1.16.0 (#44) SVD
// diagnostics. TEMPORARY diagnostic infrastructure, header-only ON PURPOSE:
// Eigen is header-only, so each including TU instantiates JacobiSVD/BDCSVD
// under ITS OWN compile flags. test_diag_r1160_mps_shor.cpp includes this
// under the project-wide -ffast-math; test_diag_r1160_strictfp.cpp includes
// it under -fno-fast-math (per-source CMake override). Comparing the two
// answers "is the SVD failure a fast-math casualty or a genuine Eigen bug"
// with everything else held constant.
//
// Builders:
//   build_poison_theta()  — the EXACT 8x8 complex matrix fed to
//     MPSState::svd_truncate at the first NaN of the 13-qubit Shor run
//     (instruction i=26, cp(5,7): after its swap-down, the adjacent cp core
//     at sites (5,6); shapes bl=4, bm=4, br=4). Reconstructed through the
//     library's own MPS evolution (clean up to that point, verified by the
//     NanBisect probe), then contracted and gate-applied HERE, mirroring
//     apply_two_qubit_gate_adjacent's theta construction.
//   build_bdcsvd_bug_matrix() — the 36x36 complex, rank-12, 12-fold
//     degenerate matrix BDCSVD mishandles (R.1.11.2): qudit Simon post-oracle
//     state at d=6, n=2, s={2,4}; dense-ctor site-0 peel (exact) leaves this
//     residual. True
//     spectrum: twelve sigma = 1/(2*sqrt(3)), rest zero; sum sigma^2 = 1.

#pragma once

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <vector>

namespace diag_r1160 {

// NaN/Inf detection by raw exponent bits, so the diagnostic does not depend on
// how the standard library spells the classification.
inline bool fp_bad(double x) {
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return ((b >> 52) & 0x7FFu) == 0x7FFu;
}

inline bool matrix_bad(const Eigen::MatrixXcd& M) {
    for (Eigen::Index c = 0; c < M.cols(); ++c)
        for (Eigen::Index r = 0; r < M.rows(); ++r)
            if (fp_bad(M(r, c).real()) || fp_bad(M(r, c).imag())) return true;
    return false;
}

inline bool vector_bad(const Eigen::VectorXd& v) {
    for (Eigen::Index i = 0; i < v.size(); ++i)
        if (fp_bad(v(i))) return true;
    return false;
}

// --- Poison theta (13-qubit Shor, first-NaN SVD input) ----------------------

inline Eigen::MatrixXcd build_poison_theta() {
    using GT = lindblad::Instruction::GateType;
    const auto qc = lindblad::algorithms::Shor::build_period_finding_circuit(
        2, 15, /*n_eval=*/9, /*n_target=*/4);

    // Evolve to just before instruction 26 through the library (verified
    // clean by the NanBisect probe), mirroring the simulator dispatch.
    lindblad::QuantumCircuit prefix(qc.n_qubits, qc.n_clbits);
    prefix.instructions.assign(qc.instructions.begin(),
                               qc.instructions.begin() + 19);
    lindblad::MPSSimulator sim;
    auto base = sim.run(prefix, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/42);
    lindblad::MPSState st = base.final_state;

    // The hand-applied gates below must use the SAME Hadamard amplitude the
    // library used for the evolved prefix above, or the reconstructed theta is
    // not the matrix svd_truncate actually saw. The literal previously here was
    // one ULP below correctly-rounded 1/√2, which is exactly the divergence #70
    // removed from the library — leaving it would have re-created it inside the
    // reproducer. Short local alias, single-sourced value.
    constexpr double s2 = lindblad::INV_SQRT2;
    const std::array<lindblad::Complex128, 4> H = {
        lindblad::Complex128(s2, 0), lindblad::Complex128(s2, 0),
        lindblad::Complex128(s2, 0), lindblad::Complex128(-s2, 0)};
    auto cp = [](double lam) {
        std::array<lindblad::Complex128, 16> U{};
        U[0] = U[5] = U[10] = lindblad::Complex128(1, 0);
        U[15] = lindblad::Complex128(std::cos(lam), std::sin(lam));
        return U;
    };
    std::array<lindblad::Complex128, 16> SW{};
    SW[0] = SW[6] = SW[9] = SW[15] = lindblad::Complex128(1, 0);

    for (size_t i = 19; i < 26; ++i) {
        const auto& inst = qc.instructions[i];
        if (inst.type == GT::H) {
            st.apply_single_qubit_gate(H, inst.qubits[0]);
        } else if (inst.type == GT::CP) {
            st.apply_two_qubit_gate(cp(inst.params[0]), inst.qubits[0],
                                    inst.qubits[1]);
        } else {
            st.apply_two_qubit_gate(SW, inst.qubits[0], inst.qubits[1]);
        }
    }
    // Instruction 26 is cp(5,7): its internal swap chain first swaps sites
    // (6,7) — clean per the probe — leaving the adjacent cp core at (5,6).
    st.apply_two_qubit_gate(SW, 6, 7);

    const auto& T1 = st.tensors[5];
    const auto& T2 = st.tensors[6];
    const int bl = T1.bond_left, bm = T1.bond_right, br = T2.bond_right;
    const int rows = bl * 2, cols = 2 * br;

    // theta[(l*2+p1), (p2*br+r)] = sum_m T1[l,p1,m] * T2[m,p2,r]
    Eigen::MatrixXcd theta = Eigen::MatrixXcd::Zero(rows, cols);
    for (int l = 0; l < bl; ++l)
        for (int p1 = 0; p1 < 2; ++p1)
            for (int m = 0; m < bm; ++m)
                for (int p2 = 0; p2 < 2; ++p2)
                    for (int r = 0; r < br; ++r) {
                        const auto& a = T1.data[(l * 2 + p1) * bm + m];
                        const auto& b = T2.data[(m * 2 + p2) * br + r];
                        theta(l * 2 + p1, p2 * br + r) +=
                            std::complex<double>(a.real, a.imag) *
                            std::complex<double>(b.real, b.imag);
                    }

    // Apply the cp(5,7) gate (lambda from instruction 26) exactly as
    // apply_two_qubit_gate_adjacent does before its SVD.
    const double lam = qc.instructions[26].params[0];
    const auto G = cp(lam);
    Eigen::MatrixXcd theta_new = Eigen::MatrixXcd::Zero(rows, cols);
    for (int l = 0; l < bl; ++l)
        for (int po1 = 0; po1 < 2; ++po1)
            for (int po2 = 0; po2 < 2; ++po2)
                for (int r = 0; r < br; ++r) {
                    std::complex<double> sum(0, 0);
                    for (int pi1 = 0; pi1 < 2; ++pi1)
                        for (int pi2 = 0; pi2 < 2; ++pi2) {
                            const auto& g =
                                G[(po1 * 2 + po2) * 4 + (pi1 * 2 + pi2)];
                            sum += std::complex<double>(g.real, g.imag) *
                                   theta(l * 2 + pi1, pi2 * br + r);
                        }
                    theta_new(l * 2 + po1, po2 * br + r) = sum;
                }
    return theta_new;
}

// --- R.1.11.2 BDCSVD bug matrix (36x36, d=6 Simon site-1 residual) ----------

inline Eigen::MatrixXcd build_bdcsvd_bug_matrix() {
    const int d = 6;
    lindblad::QuditStatevector psi(/*n_qudits=*/4, d);  // |x0 x1 y0 y1>

    const auto F = lindblad::qudit_gates::qft_matrix(d);
    psi.apply_1qudit(0, F);
    psi.apply_1qudit(1, F);

    // Coset-canonical Simon oracle, hidden period s = {2, 4}:
    // f(x) = lexicographic min over { (x + k*s) mod 6 : k = 0..5 }.
    const std::vector<int> s = {2, 4};
    psi.apply_function_oracle(2, 2, [&](const std::vector<int>& x) {
        std::vector<int> best = x;
        for (int k = 1; k < d; ++k) {
            std::vector<int> cand = {(x[0] + k * s[0]) % d,
                                     (x[1] + k * s[1]) % d};
            if (cand < best) best = cand;
        }
        return best;
    });

    // Dense-ctor chain, site 0 (exact per the bug note): amplitude index is
    // LSB-first (digit q has weight d^q), so qudit 0 is idx % 6 and the
    // remaining three qudits form the column index.
    const int rest0 = 6 * 6 * 6;  // 216
    Eigen::MatrixXcd M0(d, rest0);
    for (int x0 = 0; x0 < d; ++x0)
        for (int c = 0; c < rest0; ++c) {
            const auto& a = psi.amplitudes[static_cast<size_t>(c) * d + x0];
            M0(x0, c) = std::complex<double>(a.real, a.imag);
        }

    Eigen::JacobiSVD<Eigen::MatrixXcd> svd0(
        M0, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& s0 = svd0.singularValues();
    int k0 = 0;
    for (Eigen::Index i = 0; i < s0.size(); ++i) k0 += (s0(i) > 1e-12) ? 1 : 0;
    // Residual block = S * V^dagger (k0 x 216), reshaped so site 1's digit
    // joins the row index: M[(alpha*6 + x1), c2] with c2 over the last two
    // qudits (36 columns).
    Eigen::MatrixXcd block =
        s0.head(k0).asDiagonal() *
        svd0.matrixV().leftCols(k0).adjoint();  // k0 x 216

    const int rest1 = 36;
    Eigen::MatrixXcd M1(k0 * d, rest1);
    for (int alpha = 0; alpha < k0; ++alpha)
        for (int x1 = 0; x1 < d; ++x1)
            for (int c2 = 0; c2 < rest1; ++c2)
                M1(alpha * d + x1, c2) = block(alpha, c2 * d + x1);
    return M1;  // 36 x 36
}

// --- SVD report --------------------------------------------------------------

struct SvdReport {
    bool corrupt = false;      // NaN/Inf anywhere in U, S, or V
    bool s_bad = false;        // NaN/Inf in the singular VALUES themselves
    double sum_sq = 0.0;       // sum of sigma^2 (must equal ||M||_F^2)
    double frob_sq = 0.0;      // ||M||_F^2
    double recon_err = 0.0;    // max |U S V^H - M| (full, only if !corrupt)
    double smallest_pos = 0.0; // smallest sigma > 0 (subnormal tell)
    int rank = 0;              // sigmas above sigma_floor (finite sigmas only)
    double sigma_floor = 0.0;  // threshold rank was counted against
    std::vector<double> top;   // leading singular values (up to 14)
    // WHERE the corruption sits, and whether the TRUNCATED slice (the only part
    // svd_truncate keeps) is usable.
    std::vector<int> u_bad_cols;   // U columns containing NaN/Inf
    std::vector<int> v_bad_cols;   // V columns containing NaN/Inf
    bool kept_slice_bad = true;    // NaN/Inf inside U/V cols < rank
    double trunc_recon_err = -1.0; // max |U_r S_r V_r^H - M| at rank
                                   // (-1 if the kept slice is corrupt)
};

// Shared report builder over explicit factors, so the Eigen-SVD wrapper and
// the in-test Gram-route replication produce comparable diagnostics.
//
// sigma_floor = threshold the significant sigmas are counted against. A backend
// SVD resolves down to roughly eps * sigma_max, so the absolute default holds
// there. The Gram route squares the condition number and resolves only to about
// sqrt(eps) * sigma_max, so it passes the same validity floor it built its
// partner factors with; counting below that floor reports its own noise as
// retained rank.
inline SvdReport report_from_factors(const Eigen::MatrixXcd& M,
                                     const Eigen::VectorXd& S,
                                     const Eigen::MatrixXcd& U,
                                     const Eigen::MatrixXcd& V,
                                     double sigma_floor = 1e-12) {
    SvdReport r;
    r.sigma_floor = sigma_floor;
    r.s_bad = vector_bad(S);
    for (Eigen::Index c = 0; c < U.cols(); ++c) {
        for (Eigen::Index i = 0; i < U.rows(); ++i) {
            if (fp_bad(U(i, c).real()) || fp_bad(U(i, c).imag())) {
                r.u_bad_cols.push_back(static_cast<int>(c));
                break;
            }
        }
    }
    for (Eigen::Index c = 0; c < V.cols(); ++c) {
        for (Eigen::Index i = 0; i < V.rows(); ++i) {
            if (fp_bad(V(i, c).real()) || fp_bad(V(i, c).imag())) {
                r.v_bad_cols.push_back(static_cast<int>(c));
                break;
            }
        }
    }
    r.corrupt = r.s_bad || !r.u_bad_cols.empty() || !r.v_bad_cols.empty();
    r.frob_sq = M.squaredNorm();
    for (Eigen::Index i = 0; i < S.size(); ++i) {
        if (!fp_bad(S(i))) {
            r.sum_sq += S(i) * S(i);
            if (S(i) > sigma_floor) r.rank++;
            if (S(i) > 0.0 &&
                (r.smallest_pos == 0.0 || S(i) < r.smallest_pos)) {
                r.smallest_pos = S(i);
            }
        }
        if (i < 14) r.top.push_back(S(i));
    }
    if (!r.corrupt) {
        r.recon_err =
            (U * S.asDiagonal() * V.adjoint() - M).cwiseAbs().maxCoeff();
    }
    // The candidate-fix question: is everything at columns < rank finite,
    // and does the truncated slice alone reconstruct M?
    const int k = r.rank;
    r.kept_slice_bad = false;
    for (int c : r.u_bad_cols) r.kept_slice_bad |= (c < k);
    for (int c : r.v_bad_cols) r.kept_slice_bad |= (c < k);
    for (Eigen::Index i = 0; i < std::min<Eigen::Index>(k, S.size()); ++i) {
        r.kept_slice_bad |= fp_bad(S(i));
    }
    if (!r.kept_slice_bad && k > 0) {
        r.trunc_recon_err = (U.leftCols(k) * S.head(k).asDiagonal() *
                                 V.leftCols(k).adjoint() -
                             M)
                                .cwiseAbs()
                                .maxCoeff();
    }
    return r;
}

template <typename SVD>
SvdReport run_svd_report(const Eigen::MatrixXcd& M) {
    SVD svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return report_from_factors(M, svd.singularValues(), svd.matrixU(),
                               svd.matrixV());
}

// In-test replication of the R.1.16.0 svd_truncate Gram fallback route
// (G = M^H M or M M^H on the smaller side, SelfAdjointEigenSolver, sigmas
// descending from sqrt(max(lambda, 0)), partner factor built only above the
// sqrt(eps)-scaled floor). Validates the rescue MATH standalone under each
// including TU's FP flags; the library WIRING (engagement counter + direct
// seam) is tracked as follow-up work in the plans.
inline SvdReport run_gram_route_report(const Eigen::MatrixXcd& M) {
    const auto rows = M.rows();
    const auto cols = M.cols();
    const bool tall = rows >= cols;
    const Eigen::MatrixXcd G =
        tall ? Eigen::MatrixXcd(M.adjoint() * M)
             : Eigen::MatrixXcd(M * M.adjoint());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(G);
    SvdReport fail;  // default report reads as failure: rank 0, kept_slice_bad
    if (es.info() != Eigen::Success) return fail;

    const int gd = static_cast<int>(es.eigenvalues().size());
    Eigen::VectorXd S(gd);
    for (int i = 0; i < gd; ++i) {
        S(i) = std::sqrt(std::max(0.0, es.eigenvalues()(gd - 1 - i)));
    }
    const double smax = (gd > 0) ? S(0) : 0.0;
    const double floor_g = std::max(1e-12, 1.5e-8 * smax);

    Eigen::MatrixXcd U(rows, gd), V(cols, gd);
    if (tall) {
        for (int i = 0; i < gd; ++i) V.col(i) = es.eigenvectors().col(gd - 1 - i);
        for (int i = 0; i < gd; ++i) {
            if (S(i) > floor_g) U.col(i) = (M * V.col(i)) / S(i);
            else U.col(i).setZero();
        }
    } else {
        for (int i = 0; i < gd; ++i) U.col(i) = es.eigenvectors().col(gd - 1 - i);
        for (int i = 0; i < gd; ++i) {
            if (S(i) > floor_g) V.col(i) = (M.adjoint() * U.col(i)) / S(i);
            else V.col(i).setZero();
        }
    }
    return report_from_factors(M, S, U, V, floor_g);
}

inline void print_svd_report_line(std::ostream& os, const char* tag,
                                  const SvdReport& r) {
    os << "[svd:" << tag << "] corrupt=" << (r.corrupt ? "YES" : "no")
       << " (S_bad=" << (r.s_bad ? "YES" : "no") << ", badU_cols=";
    if (r.u_bad_cols.empty()) os << "-";
    for (size_t i = 0; i < r.u_bad_cols.size(); ++i)
        os << (i ? "," : "") << r.u_bad_cols[i];
    os << ", badV_cols=";
    if (r.v_bad_cols.empty()) os << "-";
    for (size_t i = 0; i < r.v_bad_cols.size(); ++i)
        os << (i ? "," : "") << r.v_bad_cols[i];
    os << ")\n[svd:" << tag << "] sum(s^2)=" << r.sum_sq
       << "  |M|_F^2=" << r.frob_sq << "  rank(>" << r.sigma_floor
       << ")=" << r.rank
       << "  kept_slice=" << (r.kept_slice_bad ? "CORRUPT" : "clean")
       << "  trunc_recon_err=" << r.trunc_recon_err
       << "  full_recon_err=" << r.recon_err
       << "  smallest_pos=" << r.smallest_pos << "\n[svd:" << tag
       << "] top sigmas:";
    for (double s : r.top) os << " " << s;
    os << "\n";
}

}  // namespace diag_r1160
