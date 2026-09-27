// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

// Helpers shared by the 1.1.30.1 canonical-form suites.
//
// Every claim those suites make about a chain is checked against something the
// chain did not compute: dense amplitudes, a Schmidt decomposition of them, or
// the orthonormality of a site read straight off its entries. The references
// live here so the four suites cannot drift apart in what they mean by "the
// state" or by "canonical".
//
// Two rules shape the code below.
//
//   - No Eigen decomposition is instantiated here. Every factorisation goes
//     through lindblad::detail::eigen_backend, the tree's one strict emitter,
//     which test_v11241_seam_rules.cpp enforces for tests as well as for the
//     library. The helpers hold plain std::complex buffers and do their own
//     products, which are small.
//
//   - Nothing random depends on argument evaluation order. Every draw is taken
//     into a named local before it is used.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace v11301 {

using lindblad::Complex128;
using lindblad::MPSSiteTensor;
using lindblad::MPSState;
using lindblad::MPSTensor;
using lindblad::QuditMPS;
using lindblad::QuditStatevector;
using Cplx = std::complex<double>;
using Amplitudes = std::vector<Cplx>;

constexpr double kEps = std::numeric_limits<double>::epsilon();

// Backward-error allowance per operation, in units of size * eps. The same
// figure the SVD ladder's verify rung grants, so these checks and the ladder
// cannot disagree about what rounding is.
constexpr double kSlack = 64.0;

// =============================================================================
// Seeded complex entries
// =============================================================================

class ComplexStream {
public:
    explicit ComplexStream(std::uint64_t seed) : rng_(seed) {}

    Cplx next() {
        const double re = unit_(rng_);
        const double im = unit_(rng_);
        return {re, im};
    }

    double real() { return unit_(rng_); }

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> unit_{-1.0, 1.0};
};

inline Complex128 to_c128(const Cplx& z) { return Complex128(z.real(), z.imag()); }
inline Cplx to_std(const Complex128& z) { return Cplx(z.real, z.imag); }

// =============================================================================
// Hand-built chains
// =============================================================================

// A qubit chain with the given interior bonds (bonds[q] joins sites q and q+1)
// and every entry drawn from the stream. Nothing about its gauge is known,
// which is the point: set_tensors must assume nothing about it either.
inline std::vector<MPSTensor> random_chain(int n, const std::vector<int>& bonds,
                                           std::uint64_t seed) {
    if (static_cast<int>(bonds.size()) != std::max(n - 1, 0)) {
        throw std::logic_error("random_chain: one bond per interior cut");
    }
    ComplexStream s(seed);
    std::vector<MPSTensor> sites;
    sites.reserve(static_cast<std::size_t>(n));
    for (int q = 0; q < n; ++q) {
        const int bl = (q == 0) ? 1 : bonds[static_cast<std::size_t>(q - 1)];
        const int br = (q == n - 1) ? 1 : bonds[static_cast<std::size_t>(q)];
        MPSTensor t(bl, br);
        for (auto& z : t.data) z = to_c128(s.next());
        sites.push_back(std::move(t));
    }
    return sites;
}

inline std::vector<MPSSiteTensor> random_qudit_chain(int n, int d,
                                                     const std::vector<int>& bonds,
                                                     std::uint64_t seed) {
    if (static_cast<int>(bonds.size()) != std::max(n - 1, 0)) {
        throw std::logic_error("random_qudit_chain: one bond per interior cut");
    }
    ComplexStream s(seed);
    std::vector<MPSSiteTensor> sites;
    sites.reserve(static_cast<std::size_t>(n));
    for (int q = 0; q < n; ++q) {
        const int bl = (q == 0) ? 1 : bonds[static_cast<std::size_t>(q - 1)];
        const int br = (q == n - 1) ? 1 : bonds[static_cast<std::size_t>(q)];
        MPSSiteTensor t(d, bl, br);
        for (auto& z : t.data) z = to_c128(s.next());
        sites.push_back(std::move(t));
    }
    return sites;
}

// =============================================================================
// Unitaries, through the seam's QR
// =============================================================================

// The Q factor of a thin QR of a square random matrix: unitary, and generic
// enough to have no symmetry a layout mistake could hide behind. Row-major.
inline std::vector<Cplx> random_unitary(int dim, std::uint64_t seed) {
    ComplexStream s(seed);
    std::vector<Cplx> m(static_cast<std::size_t>(dim) * dim);
    for (auto& z : m) z = s.next();
    std::vector<Cplx> q(m.size()), r(m.size());
    if (!lindblad::detail::qr_thin(m.data(), dim, dim,
                                   lindblad::detail::MatrixOrder::ColMajor,
                                   q.data(), r.data())) {
        throw std::logic_error("random_unitary: the seam refused a square matrix");
    }
    // q is column-major; hand it back row-major, which every gate takes.
    std::vector<Cplx> out(m.size());
    for (int row = 0; row < dim; ++row)
        for (int col = 0; col < dim; ++col)
            out[static_cast<std::size_t>(row) * dim + col] =
                q[static_cast<std::size_t>(col) * dim + row];
    return out;
}

inline std::vector<Complex128> as_c128(const std::vector<Cplx>& m) {
    std::vector<Complex128> out;
    out.reserve(m.size());
    for (const auto& z : m) out.push_back(to_c128(z));
    return out;
}

inline std::array<Complex128, 16> as_gate4(const std::vector<Cplx>& m) {
    if (m.size() != 16) throw std::logic_error("as_gate4: need a 4x4 matrix");
    std::array<Complex128, 16> out{};
    for (std::size_t i = 0; i < 16; ++i) out[i] = to_c128(m[i]);
    return out;
}

inline std::array<Complex128, 4> as_gate2(const std::vector<Cplx>& m) {
    if (m.size() != 4) throw std::logic_error("as_gate2: need a 2x2 matrix");
    std::array<Complex128, 4> out{};
    for (std::size_t i = 0; i < 4; ++i) out[i] = to_c128(m[i]);
    return out;
}

// =============================================================================
// Gauge distortion
// =============================================================================

// Insert G G^-1 on the bond between sites q and q+1: site q's right index is
// multiplied by G and site q+1's left index by G^-1, which leaves the state
// exactly as it was in exact arithmetic and the gauge far from canonical.
// G = W diag(lambda), so G^-1 = diag(1/lambda) W^H with no inversion needed,
// W unitary from the seam's QR. lambda spans `spread` geometrically, which is
// the condition number of G.
namespace detail_oracle {

inline std::vector<double> geometric(int dim, double spread) {
    std::vector<double> lam(static_cast<std::size_t>(dim), 1.0);
    for (int i = 0; i < dim; ++i) {
        lam[static_cast<std::size_t>(i)] =
            (dim == 1) ? 1.0 : std::pow(spread, static_cast<double>(i) / (dim - 1));
    }
    return lam;
}

}  // namespace detail_oracle

inline void distort_bond(std::vector<MPSTensor>& sites, int q, double spread,
                         std::uint64_t seed) {
    MPSTensor& A = sites[static_cast<std::size_t>(q)];
    MPSTensor& B = sites[static_cast<std::size_t>(q + 1)];
    const int chi = A.bond_right;
    const std::vector<Cplx> W = random_unitary(chi, seed);
    const std::vector<double> lam = detail_oracle::geometric(chi, spread);

    MPSTensor A2(A.bond_left, chi);
    for (int l = 0; l < A.bond_left; ++l)
        for (int p = 0; p < 2; ++p)
            for (int r = 0; r < chi; ++r) {
                Cplx acc(0.0, 0.0);
                for (int m = 0; m < chi; ++m)
                    acc += to_std(A(l, p, m)) * W[static_cast<std::size_t>(m) * chi + r];
                A2(l, p, r) = to_c128(acc * lam[static_cast<std::size_t>(r)]);
            }
    MPSTensor B2(chi, B.bond_right);
    for (int r = 0; r < chi; ++r)
        for (int p = 0; p < 2; ++p)
            for (int x = 0; x < B.bond_right; ++x) {
                Cplx acc(0.0, 0.0);
                for (int m = 0; m < chi; ++m)
                    acc += std::conj(W[static_cast<std::size_t>(m) * chi + r]) *
                           to_std(B(m, p, x));
                B2(r, p, x) = to_c128(acc / lam[static_cast<std::size_t>(r)]);
            }
    A = std::move(A2);
    B = std::move(B2);
}

inline void distort_bond(std::vector<MPSSiteTensor>& sites, int q, double spread,
                         std::uint64_t seed) {
    MPSSiteTensor& A = sites[static_cast<std::size_t>(q)];
    MPSSiteTensor& B = sites[static_cast<std::size_t>(q + 1)];
    const int d = A.d;
    const int chi = A.chi_R;
    const std::vector<Cplx> W = random_unitary(chi, seed);
    const std::vector<double> lam = detail_oracle::geometric(chi, spread);

    MPSSiteTensor A2(d, A.chi_L, chi);
    for (int s = 0; s < d; ++s)
        for (int l = 0; l < A.chi_L; ++l)
            for (int r = 0; r < chi; ++r) {
                Cplx acc(0.0, 0.0);
                for (int m = 0; m < chi; ++m)
                    acc += to_std(A.at(s, l, m)) * W[static_cast<std::size_t>(m) * chi + r];
                A2.at(s, l, r) = to_c128(acc * lam[static_cast<std::size_t>(r)]);
            }
    MPSSiteTensor B2(d, chi, B.chi_R);
    for (int s = 0; s < d; ++s)
        for (int r = 0; r < chi; ++r)
            for (int x = 0; x < B.chi_R; ++x) {
                Cplx acc(0.0, 0.0);
                for (int m = 0; m < chi; ++m)
                    acc += std::conj(W[static_cast<std::size_t>(m) * chi + r]) *
                           to_std(B.at(s, m, x));
                B2.at(s, r, x) = to_c128(acc / lam[static_cast<std::size_t>(r)]);
            }
    A = std::move(A2);
    B = std::move(B2);
}

// =============================================================================
// Dense amplitudes
// =============================================================================
// Index convention everywhere: digit q has weight d^q, so qubit q is bit q.

inline Amplitudes dense(const lindblad::Statevector& sv) {
    Amplitudes a(sv.dim);
    for (std::size_t i = 0; i < sv.dim; ++i) a[i] = Cplx(sv.real_parts[i], sv.imag_parts[i]);
    return a;
}

inline Amplitudes dense(const MPSState& s) { return dense(s.to_statevector()); }

inline Amplitudes dense(const QuditStatevector& sv) {
    Amplitudes a(sv.amplitudes.size());
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = to_std(sv.amplitudes[i]);
    return a;
}

inline Amplitudes dense(const QuditMPS& s) { return dense(s.to_statevector()); }

inline lindblad::Statevector to_statevector(const Amplitudes& a, int n) {
    lindblad::Statevector sv(n);
    for (std::size_t i = 0; i < a.size(); ++i) {
        sv.real_parts[i] = a[i].real();
        sv.imag_parts[i] = a[i].imag();
    }
    return sv;
}

inline QuditStatevector to_qudit_statevector(const Amplitudes& a, int n, int d) {
    QuditStatevector sv(n, d);
    for (std::size_t i = 0; i < a.size(); ++i) sv.amplitudes[i] = to_c128(a[i]);
    return sv;
}

inline double norm_sq(const Amplitudes& a) {
    double s = 0.0;
    for (const auto& z : a) s += std::norm(z);
    return s;
}

// <a|b>
inline Cplx inner(const Amplitudes& a, const Amplitudes& b) {
    Cplx s(0.0, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
    return s;
}

// |<a|b>|^2 between the normalised states.
inline double fidelity(const Amplitudes& a, const Amplitudes& b) {
    const double na = norm_sq(a);
    const double nb = norm_sq(b);
    if (!(na > 0.0) || !(nb > 0.0)) return 0.0;
    return std::norm(inner(a, b)) / (na * nb);
}

inline double max_abs_diff(const Amplitudes& a, const Amplitudes& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

// Rounding allowance at amplitude scale for a state reached through `ops`
// operations over `dim` amplitudes: each can move an amplitude by a few eps of
// the state's norm.
inline double amplitude_tol(std::size_t ops, std::size_t dim, double norm) {
    return kSlack * static_cast<double>(ops + dim) * kEps * norm;
}

// Raw marginal <psi|P_k|psi> of qubit q, k = 0, 1.
inline std::array<double, 2> qubit_marginals(const Amplitudes& a, int q) {
    std::array<double, 2> p{0.0, 0.0};
    for (std::size_t i = 0; i < a.size(); ++i) p[(i >> q) & 1u] += std::norm(a[i]);
    return p;
}

// Raw marginals of qudit q.
inline std::vector<double> qudit_marginals(const Amplitudes& a, int q, int d) {
    std::vector<double> p(static_cast<std::size_t>(d), 0.0);
    std::size_t stride = 1;
    for (int i = 0; i < q; ++i) stride *= static_cast<std::size_t>(d);
    for (std::size_t i = 0; i < a.size(); ++i)
        p[(i / stride) % static_cast<std::size_t>(d)] += std::norm(a[i]);
    return p;
}

// =============================================================================
// Schmidt decomposition, through the seam
// =============================================================================

struct Schmidt {
    int rows = 0;               // d^cut: the low digits
    int cols = 0;               // the rest
    std::vector<double> sigma;  // descending
    std::vector<Cplx> U;        // rows x k, column-major
    std::vector<Cplx> V;        // cols x k, column-major
};

// The reshaped state across the cut between digits cut-1 and cut: row index
// the low `cut` digits, column index the rest.
inline Schmidt schmidt(const Amplitudes& a, int d, int n, int cut) {
    Schmidt s;
    s.rows = 1;
    for (int i = 0; i < cut; ++i) s.rows *= d;
    s.cols = static_cast<int>(a.size()) / s.rows;
    if (cut <= 0 || cut >= n || static_cast<std::size_t>(s.rows) * s.cols != a.size()) {
        throw std::logic_error("schmidt: the cut must split the register");
    }
    const int k = std::min(s.rows, s.cols);
    std::vector<Cplx> m(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::size_t row = i % static_cast<std::size_t>(s.rows);
        const std::size_t col = i / static_cast<std::size_t>(s.rows);
        m[col * static_cast<std::size_t>(s.rows) + row] = a[i];
    }
    s.sigma.assign(static_cast<std::size_t>(k), 0.0);
    s.U.assign(static_cast<std::size_t>(s.rows) * k, Cplx(0.0, 0.0));
    s.V.assign(static_cast<std::size_t>(s.cols) * k, Cplx(0.0, 0.0));
    if (!lindblad::detail::svd_thin(m.data(), s.rows, s.cols,
                                    lindblad::detail::MatrixOrder::ColMajor,
                                    lindblad::SVDMethod::EigenJacobi, s.U.data(),
                                    s.sigma.data(), s.V.data())) {
        throw std::logic_error("schmidt: the seam refused the reshaped state");
    }
    return s;
}

// The state with only its `keep` leading Schmidt directions across the cut:
// (U_k U_k^H x I) applied to it. This is the optimal truncation, and it is
// basis-independent whenever sigma[keep-1] > sigma[keep].
inline Amplitudes project_top(const Amplitudes& a, int d, int n, int cut, int keep) {
    const Schmidt s = schmidt(a, d, n, cut);
    const int rows = s.rows;
    const int cols = s.cols;
    Amplitudes out(a.size(), Cplx(0.0, 0.0));
    for (int col = 0; col < cols; ++col) {
        // c_j = sum_row conj(U(row, j)) a(row, col) for the kept j.
        std::vector<Cplx> c(static_cast<std::size_t>(keep), Cplx(0.0, 0.0));
        for (int j = 0; j < keep; ++j)
            for (int row = 0; row < rows; ++row)
                c[static_cast<std::size_t>(j)] +=
                    std::conj(s.U[static_cast<std::size_t>(j) * rows + row]) *
                    a[static_cast<std::size_t>(col) * rows + row];
        for (int row = 0; row < rows; ++row) {
            Cplx acc(0.0, 0.0);
            for (int j = 0; j < keep; ++j)
                acc += s.U[static_cast<std::size_t>(j) * rows + row] *
                       c[static_cast<std::size_t>(j)];
            out[static_cast<std::size_t>(col) * rows + row] = acc;
        }
    }
    return out;
}

// =============================================================================
// Orthonormality of a site, read off its entries
// =============================================================================

// max |(A^H A - I)_ij| with A the (d chi_L) x chi_R reshape: zero for a
// left-orthonormal site.
inline double left_deviation(const MPSTensor& t) {
    double worst = 0.0;
    for (int i = 0; i < t.bond_right; ++i)
        for (int j = 0; j < t.bond_right; ++j) {
            Cplx acc(0.0, 0.0);
            for (int l = 0; l < t.bond_left; ++l)
                for (int p = 0; p < 2; ++p)
                    acc += std::conj(to_std(t(l, p, i))) * to_std(t(l, p, j));
            worst = std::max(worst, std::abs(acc - Cplx(i == j ? 1.0 : 0.0, 0.0)));
        }
    return worst;
}

// max |(A A^H - I)_ij| with A the chi_L x (d chi_R) reshape: zero for a
// right-orthonormal site.
inline double right_deviation(const MPSTensor& t) {
    double worst = 0.0;
    for (int i = 0; i < t.bond_left; ++i)
        for (int j = 0; j < t.bond_left; ++j) {
            Cplx acc(0.0, 0.0);
            for (int p = 0; p < 2; ++p)
                for (int r = 0; r < t.bond_right; ++r)
                    acc += to_std(t(i, p, r)) * std::conj(to_std(t(j, p, r)));
            worst = std::max(worst, std::abs(acc - Cplx(i == j ? 1.0 : 0.0, 0.0)));
        }
    return worst;
}

inline double left_deviation(const MPSSiteTensor& t) {
    double worst = 0.0;
    for (int i = 0; i < t.chi_R; ++i)
        for (int j = 0; j < t.chi_R; ++j) {
            Cplx acc(0.0, 0.0);
            for (int s = 0; s < t.d; ++s)
                for (int l = 0; l < t.chi_L; ++l)
                    acc += std::conj(to_std(t.at(s, l, i))) * to_std(t.at(s, l, j));
            worst = std::max(worst, std::abs(acc - Cplx(i == j ? 1.0 : 0.0, 0.0)));
        }
    return worst;
}

inline double right_deviation(const MPSSiteTensor& t) {
    double worst = 0.0;
    for (int i = 0; i < t.chi_L; ++i)
        for (int j = 0; j < t.chi_L; ++j) {
            Cplx acc(0.0, 0.0);
            for (int s = 0; s < t.d; ++s)
                for (int r = 0; r < t.chi_R; ++r)
                    acc += to_std(t.at(s, i, r)) * std::conj(to_std(t.at(s, j, r)));
            worst = std::max(worst, std::abs(acc - Cplx(i == j ? 1.0 : 0.0, 0.0)));
        }
    return worst;
}

inline int site_rows(const MPSTensor& t) { return 2 * t.bond_left; }
inline int site_cols(const MPSTensor& t) { return 2 * t.bond_right; }
inline int site_rows(const MPSSiteTensor& t) { return t.d * t.chi_L; }
inline int site_cols(const MPSSiteTensor& t) { return t.d * t.chi_R; }

// The invariant: the span lies inside the chain, every site left of it is
// left-orthonormal and every site right of it right-orthonormal. The sites
// inside carry no guarantee and are not read. Each orthonormal site was made
// by one QR or one SVD, so it is orthonormal to a few eps of its size.
template <typename Chain>
::testing::AssertionResult is_canonical(const Chain& chain) {
    const auto& t = chain.tensors();
    const int n = static_cast<int>(t.size());
    const auto [lo, hi] = chain.open_span();
    if (n == 0) {
        if (lo == 0 && hi == -1) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure()
               << "a chain with no sites reports {" << lo << ", " << hi << "}";
    }
    if (lo < 0 || hi >= n || lo > hi) {
        return ::testing::AssertionFailure()
               << "open span {" << lo << ", " << hi << "} is not inside a "
               << n << "-site chain";
    }
    for (int q = 0; q < lo; ++q) {
        const auto& s = t[static_cast<std::size_t>(q)];
        const double dev = left_deviation(s);
        const double tol = kSlack * static_cast<double>(site_rows(s)) * kEps;
        if (!(dev <= tol)) {
            return ::testing::AssertionFailure()
                   << "site " << q << " is left of the span {" << lo << ", " << hi
                   << "} but deviates from left-orthonormal by " << dev
                   << " (tolerance " << tol << ")";
        }
    }
    for (int q = hi + 1; q < n; ++q) {
        const auto& s = t[static_cast<std::size_t>(q)];
        const double dev = right_deviation(s);
        const double tol = kSlack * static_cast<double>(site_cols(s)) * kEps;
        if (!(dev <= tol)) {
            return ::testing::AssertionFailure()
                   << "site " << q << " is right of the span {" << lo << ", " << hi
                   << "} but deviates from right-orthonormal by " << dev
                   << " (tolerance " << tol << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

// =============================================================================
// Bitwise snapshots, for "this call changed nothing"
// =============================================================================

inline bool same_bits(double a, double b) {
    std::uint64_t x, y;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    return x == y;
}

inline bool same_site(const MPSTensor& a, const MPSTensor& b) {
    if (a.bond_left != b.bond_left || a.bond_right != b.bond_right ||
        a.data.size() != b.data.size())
        return false;
    for (std::size_t i = 0; i < a.data.size(); ++i)
        if (!same_bits(a.data[i].real, b.data[i].real) ||
            !same_bits(a.data[i].imag, b.data[i].imag))
            return false;
    return true;
}

inline bool same_site(const MPSSiteTensor& a, const MPSSiteTensor& b) {
    if (a.d != b.d || a.chi_L != b.chi_L || a.chi_R != b.chi_R ||
        a.data.size() != b.data.size())
        return false;
    for (std::size_t i = 0; i < a.data.size(); ++i)
        if (!same_bits(a.data[i].real, b.data[i].real) ||
            !same_bits(a.data[i].imag, b.data[i].imag))
            return false;
    return true;
}

template <typename Site>
bool same_sites(const std::vector<Site>& a, const std::vector<Site>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_site(a[i], b[i])) return false;
    return true;
}

// Everything a chain reports about the splits it performed and the
// truncation they cost. Two snapshots comparing equal means none of it moved.
struct Profile {
    std::size_t svd_calls = 0;
    std::size_t jacobi = 0;
    std::size_t gram = 0;
    double floor = 0.0;
    std::uint64_t nanos = 0;
    double truncation = 0.0;
    std::optional<double> estimate;
    std::optional<double> lower_bound;

    bool operator==(const Profile& o) const {
        const auto same_opt = [](const std::optional<double>& x,
                                 const std::optional<double>& y) {
            if (x.has_value() != y.has_value()) return false;
            return !x.has_value() || same_bits(*x, *y);
        };
        return svd_calls == o.svd_calls && jacobi == o.jacobi && gram == o.gram &&
               same_bits(floor, o.floor) && nanos == o.nanos &&
               same_bits(truncation, o.truncation) &&
               same_opt(estimate, o.estimate) && same_opt(lower_bound, o.lower_bound);
    }
};

template <typename Chain>
Profile profile_of(const Chain& c) {
    Profile p;
    p.svd_calls = c.svd_call_count();
    p.jacobi = c.jacobi_rescue_count();
    p.gram = c.gram_fallback_count();
    p.floor = c.floor_rejected_weight();
    p.nanos = c.svd_time_ns();
    p.truncation = c.truncation_error();
    p.estimate = c.fidelity_estimate();
    p.lower_bound = c.fidelity_lower_bound();
    return p;
}

inline std::string describe(const Profile& p) {
    std::ostringstream os;
    os << "{svd_calls " << p.svd_calls << ", jacobi " << p.jacobi << ", gram "
       << p.gram << ", floor " << p.floor << ", ns " << p.nanos << ", truncation "
       << p.truncation << ", estimate "
       << (p.estimate ? std::to_string(*p.estimate) : std::string("empty"))
       << ", floor_bound "
       << (p.lower_bound ? std::to_string(*p.lower_bound) : std::string("empty"))
       << "}";
    return os.str();
}

// =============================================================================
// Circuits
// =============================================================================

// Random two-qubit blocks on alternating bond parities, each three CX between
// layers of random u3: the shape benchmarks/compare/gen_circuits.py gives the
// corpus brickwork, and the generator the 1.1.30.0 measurements used, seeded
// from (n, layers) alone so every build draws one circuit.
inline lindblad::QuantumCircuit brickwork(int n, int layers) {
    lindblad::QuantumCircuit qc(n);
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL ^
                        (static_cast<std::uint64_t>(n) << 16) ^
                        static_cast<std::uint64_t>(layers));
    std::uniform_real_distribution<double> angle(0.0, lindblad::TWO_PI);
    const auto u3 = [&](int q) {
        const double theta = angle(rng);
        const double phi = angle(rng);
        const double lam = angle(rng);
        qc.u3(theta, phi, lam, q);
    };
    for (int layer = 0; layer < layers; ++layer) {
        for (int a = layer % 2; a + 1 < n; a += 2) {
            u3(a);
            u3(a + 1);
            for (int k = 0; k < 3; ++k) {
                qc.cx(a, a + 1);
                u3(a);
                u3(a + 1);
            }
        }
    }
    return qc;
}

}  // namespace v11301
