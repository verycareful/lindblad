// R.1.12.1 total-coverage suite, Batch 1: lindblad/gates.hpp.
// Plan: docs (R.1.12.1 coverage plan), section "Batch 1: foundations".
//
// Pins EVERY statevector gate kernel numerically: exact matrix elements at a
// generic angle, unitarity, inverse pairs, special angles, operand-order
// behaviour for non-symmetric gates, embeddings on non-adjacent qubits, and
// the apply_unitary contract (targets[0] = LSB of the matrix index).
// Expected values follow the conventions frozen in R.1.12.0
// (docs/Architecture.md "Conventions") and were derived independently of the
// implementation before being written down.

#include <gtest/gtest.h>

#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 1e-12;
constexpr double kTheta = 0.7;  // generic non-symmetric angle

using Matrix = std::vector<Complex128>;
using Entry = std::tuple<size_t, size_t, double, double>;  // row, col, re, im

// Extract the full 2^n x 2^n matrix of an applier via basis columns.
Matrix extract(int n, const std::function<void(Statevector&)>& apply) {
    const size_t dim = 1ULL << n;
    Matrix M(dim * dim);
    for (size_t col = 0; col < dim; ++col) {
        Statevector sv(n);
        sv.initialize_basis(col);
        apply(sv);
        for (size_t row = 0; row < dim; ++row)
            M[row * dim + col] = sv.amplitude(row);
    }
    return M;
}

// All listed entries match; every other entry is zero.
void expect_sparse(const Matrix& M, size_t dim, const std::vector<Entry>& want,
                   double tol = kTol) {
    Matrix expect(dim * dim, Complex128(0.0, 0.0));
    for (const auto& [r, c, re, im] : want) expect[r * dim + c] = {re, im};
    for (size_t r = 0; r < dim; ++r)
        for (size_t c = 0; c < dim; ++c) {
            EXPECT_NEAR(M[r * dim + c].real, expect[r * dim + c].real, tol)
                << "real mismatch at (" << r << "," << c << ")";
            EXPECT_NEAR(M[r * dim + c].imag, expect[r * dim + c].imag, tol)
                << "imag mismatch at (" << r << "," << c << ")";
        }
}

void expect_unitary(const Matrix& M, size_t dim, double tol = 1e-10) {
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < dim; ++j) {
            Complex128 acc(0.0, 0.0);
            for (size_t k = 0; k < dim; ++k)
                acc += M[k * dim + i].conj() * M[k * dim + j];
            const double want = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(acc.real, want, tol) << "(M+M)[" << i << "," << j << "]";
            EXPECT_NEAR(acc.imag, 0.0, tol) << "(M+M)[" << i << "," << j << "]";
        }
}

void expect_identity(const Matrix& M, size_t dim, double tol = 1e-10) {
    std::vector<Entry> id;
    for (size_t i = 0; i < dim; ++i) id.push_back({i, i, 1.0, 0.0});
    expect_sparse(M, dim, id, tol);
}

}  // namespace

// =============================================================================
// Single-qubit gates: exact matrices on n = 1
// =============================================================================

TEST(R1121Gates, HadamardExactMatrix) {
    auto M = extract(1, [](Statevector& s) { gates::apply_h(s, 0); });
    expect_sparse(M, 2, {{0, 0, INV_SQRT2, 0}, {0, 1, INV_SQRT2, 0},
                         {1, 0, INV_SQRT2, 0}, {1, 1, -INV_SQRT2, 0}});
}

TEST(R1121Gates, PauliExactMatrices) {
    auto X = extract(1, [](Statevector& s) { gates::apply_x(s, 0); });
    expect_sparse(X, 2, {{0, 1, 1, 0}, {1, 0, 1, 0}});

    auto Y = extract(1, [](Statevector& s) { gates::apply_y(s, 0); });
    expect_sparse(Y, 2, {{0, 1, 0, -1}, {1, 0, 0, 1}});

    auto Z = extract(1, [](Statevector& s) { gates::apply_z(s, 0); });
    expect_sparse(Z, 2, {{0, 0, 1, 0}, {1, 1, -1, 0}});
}

TEST(R1121Gates, PhaseFamilyExactMatrices) {
    auto S = extract(1, [](Statevector& s) { gates::apply_s(s, 0); });
    expect_sparse(S, 2, {{0, 0, 1, 0}, {1, 1, 0, 1}});

    auto Sdg = extract(1, [](Statevector& s) { gates::apply_sdg(s, 0); });
    expect_sparse(Sdg, 2, {{0, 0, 1, 0}, {1, 1, 0, -1}});

    auto T = extract(1, [](Statevector& s) { gates::apply_t(s, 0); });
    expect_sparse(T, 2, {{0, 0, 1, 0}, {1, 1, INV_SQRT2, INV_SQRT2}});

    auto Tdg = extract(1, [](Statevector& s) { gates::apply_tdg(s, 0); });
    expect_sparse(Tdg, 2, {{0, 0, 1, 0}, {1, 1, INV_SQRT2, -INV_SQRT2}});

    const double c = std::cos(kTheta), s = std::sin(kTheta);
    auto P = extract(1, [](Statevector& sv) { gates::apply_p(sv, 0, kTheta); });
    expect_sparse(P, 2, {{0, 0, 1, 0}, {1, 1, c, s}});

    auto U1 = extract(1, [](Statevector& sv) { gates::apply_u1(sv, 0, kTheta); });
    expect_sparse(U1, 2, {{0, 0, 1, 0}, {1, 1, c, s}});
}

TEST(R1121Gates, SqrtXExactMatrices) {
    auto SX = extract(1, [](Statevector& s) { gates::apply_sx(s, 0); });
    expect_sparse(SX, 2, {{0, 0, 0.5, 0.5}, {0, 1, 0.5, -0.5},
                          {1, 0, 0.5, -0.5}, {1, 1, 0.5, 0.5}});

    auto SXdg = extract(1, [](Statevector& s) { gates::apply_sxdg(s, 0); });
    expect_sparse(SXdg, 2, {{0, 0, 0.5, -0.5}, {0, 1, 0.5, 0.5},
                            {1, 0, 0.5, 0.5}, {1, 1, 0.5, -0.5}});
}

TEST(R1121Gates, RotationExactMatrices) {
    const double c = std::cos(kTheta / 2), s = std::sin(kTheta / 2);

    auto RX = extract(1, [](Statevector& sv) { gates::apply_rx(sv, 0, kTheta); });
    expect_sparse(RX, 2, {{0, 0, c, 0}, {0, 1, 0, -s}, {1, 0, 0, -s}, {1, 1, c, 0}});

    auto RY = extract(1, [](Statevector& sv) { gates::apply_ry(sv, 0, kTheta); });
    expect_sparse(RY, 2, {{0, 0, c, 0}, {0, 1, -s, 0}, {1, 0, s, 0}, {1, 1, c, 0}});

    auto RZ = extract(1, [](Statevector& sv) { gates::apply_rz(sv, 0, kTheta); });
    expect_sparse(RZ, 2, {{0, 0, c, -s}, {1, 1, c, s}});
}

TEST(R1121Gates, UGateFamilyExactAndAliases) {
    const double th = 0.7, ph = 0.3, la = -0.4;
    const double c = std::cos(th / 2), s = std::sin(th / 2);
    std::vector<Entry> expU = {
        {0, 0, c, 0},
        {0, 1, -s * std::cos(la), -s * std::sin(la)},
        {1, 0, s * std::cos(ph), s * std::sin(ph)},
        {1, 1, c * std::cos(ph + la), c * std::sin(ph + la)}};

    auto U = extract(1, [&](Statevector& sv) { gates::apply_u(sv, 0, th, ph, la); });
    expect_sparse(U, 2, expU);

    auto U3 = extract(1, [&](Statevector& sv) { gates::apply_u3(sv, 0, th, ph, la); });
    expect_sparse(U3, 2, expU);

    // U2(phi, lambda) == U(pi/2, phi, lambda)
    auto U2 = extract(1, [&](Statevector& sv) { gates::apply_u2(sv, 0, ph, la); });
    auto Uhalf = extract(1, [&](Statevector& sv) { gates::apply_u(sv, 0, PI_2, ph, la); });
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(U2[i].real, Uhalf[i].real, kTol);
        EXPECT_NEAR(U2[i].imag, Uhalf[i].imag, kTol);
    }
}

TEST(R1121Gates, SingleQubitInversePairs) {
    auto id1 = extract(1, [](Statevector& s) {
        gates::apply_s(s, 0);
        gates::apply_sdg(s, 0);
    });
    expect_identity(id1, 2);

    auto id2 = extract(1, [](Statevector& s) {
        gates::apply_t(s, 0);
        gates::apply_tdg(s, 0);
    });
    expect_identity(id2, 2);

    auto id3 = extract(1, [](Statevector& s) {
        gates::apply_sx(s, 0);
        gates::apply_sxdg(s, 0);
    });
    expect_identity(id3, 2);

    auto id4 = extract(1, [](Statevector& s) {
        gates::apply_rx(s, 0, kTheta);
        gates::apply_rx(s, 0, -kTheta);
    });
    expect_identity(id4, 2);

    // U(th, ph, la)^dagger = U(-th, -la, -ph)
    auto id5 = extract(1, [](Statevector& s) {
        gates::apply_u(s, 0, 0.7, 0.3, -0.4);
        gates::apply_u(s, 0, -0.7, 0.4, -0.3);
    });
    expect_identity(id5, 2);
}

TEST(R1121Gates, SpecialAngles) {
    // RX(0) = identity exactly
    auto rx0 = extract(1, [](Statevector& s) { gates::apply_rx(s, 0, 0.0); });
    expect_identity(rx0, 2, kTol);

    // RZ(2*pi) = -I (the spinor sign)
    auto rz2pi = extract(1, [](Statevector& s) { gates::apply_rz(s, 0, 2.0 * PI); });
    expect_sparse(rz2pi, 2, {{0, 0, -1, 0}, {1, 1, -1, 0}}, 1e-10);

    // P(0) = identity
    auto p0 = extract(1, [](Statevector& s) { gates::apply_p(s, 0, 0.0); });
    expect_identity(p0, 2, kTol);

    // SX * SX = X
    auto sx2 = extract(1, [](Statevector& s) {
        gates::apply_sx(s, 0);
        gates::apply_sx(s, 0);
    });
    expect_sparse(sx2, 2, {{0, 1, 1, 0}, {1, 0, 1, 0}}, 1e-10);
}

TEST(R1121Gates, SingleQubitGateOnHighQubitEmbedsAsTensor) {
    // H on qubit 1 of a 2-qubit register: mixes index pairs (0,2) and (1,3).
    auto M = extract(2, [](Statevector& s) { gates::apply_h(s, 1); });
    expect_sparse(M, 4, {{0, 0, INV_SQRT2, 0}, {0, 2, INV_SQRT2, 0},
                         {2, 0, INV_SQRT2, 0}, {2, 2, -INV_SQRT2, 0},
                         {1, 1, INV_SQRT2, 0}, {1, 3, INV_SQRT2, 0},
                         {3, 1, INV_SQRT2, 0}, {3, 3, -INV_SQRT2, 0}});
}

// =============================================================================
// Two-qubit gates: exact matrices on n = 2 (physical idx = q0 + 2*q1)
// =============================================================================

TEST(R1121Gates, CXExactBothOperandOrders) {
    auto M01 = extract(2, [](Statevector& s) { gates::apply_cx(s, 0, 1); });
    expect_sparse(M01, 4, {{0, 0, 1, 0}, {2, 2, 1, 0}, {1, 3, 1, 0}, {3, 1, 1, 0}});

    auto M10 = extract(2, [](Statevector& s) { gates::apply_cx(s, 1, 0); });
    expect_sparse(M10, 4, {{0, 0, 1, 0}, {1, 1, 1, 0}, {2, 3, 1, 0}, {3, 2, 1, 0}});
}

TEST(R1121Gates, CYExact) {
    auto M = extract(2, [](Statevector& s) { gates::apply_cy(s, 0, 1); });
    // ctrl q0 = 1: Y on q1: |q0=1,q1=0>(1) -> i|q0=1,q1=1>(3); (3) -> -i (1)
    expect_sparse(M, 4, {{0, 0, 1, 0}, {2, 2, 1, 0}, {3, 1, 0, 1}, {1, 3, 0, -1}});
}

TEST(R1121Gates, CZAndCHExact) {
    auto CZ = extract(2, [](Statevector& s) { gates::apply_cz(s, 0, 1); });
    expect_sparse(CZ, 4, {{0, 0, 1, 0}, {1, 1, 1, 0}, {2, 2, 1, 0}, {3, 3, -1, 0}});

    auto CH = extract(2, [](Statevector& s) { gates::apply_ch(s, 0, 1); });
    expect_sparse(CH, 4, {{0, 0, 1, 0}, {2, 2, 1, 0},
                          {1, 1, INV_SQRT2, 0}, {1, 3, INV_SQRT2, 0},
                          {3, 1, INV_SQRT2, 0}, {3, 3, -INV_SQRT2, 0}});
}

TEST(R1121Gates, SwapAndISwapExact) {
    auto SW = extract(2, [](Statevector& s) { gates::apply_swap(s, 0, 1); });
    expect_sparse(SW, 4, {{0, 0, 1, 0}, {1, 2, 1, 0}, {2, 1, 1, 0}, {3, 3, 1, 0}});

    auto ISW = extract(2, [](Statevector& s) { gates::apply_iswap(s, 0, 1); });
    expect_sparse(ISW, 4, {{0, 0, 1, 0}, {1, 2, 0, 1}, {2, 1, 0, 1}, {3, 3, 1, 0}});
}

TEST(R1121Gates, ControlledRotationsExact) {
    const double c = std::cos(kTheta / 2), s = std::sin(kTheta / 2);

    auto CRX = extract(2, [](Statevector& sv) { gates::apply_crx(sv, 0, 1, kTheta); });
    expect_sparse(CRX, 4, {{0, 0, 1, 0}, {2, 2, 1, 0},
                           {1, 1, c, 0}, {1, 3, 0, -s}, {3, 1, 0, -s}, {3, 3, c, 0}});

    auto CRY = extract(2, [](Statevector& sv) { gates::apply_cry(sv, 0, 1, kTheta); });
    expect_sparse(CRY, 4, {{0, 0, 1, 0}, {2, 2, 1, 0},
                           {1, 1, c, 0}, {1, 3, -s, 0}, {3, 1, s, 0}, {3, 3, c, 0}});

    auto CRZ = extract(2, [](Statevector& sv) { gates::apply_crz(sv, 0, 1, kTheta); });
    expect_sparse(CRZ, 4, {{0, 0, 1, 0}, {2, 2, 1, 0},
                           {1, 1, c, -s}, {3, 3, c, s}});

    const double cl = std::cos(kTheta), sl = std::sin(kTheta);
    auto CP = extract(2, [](Statevector& sv) { gates::apply_cp(sv, 0, 1, kTheta); });
    expect_sparse(CP, 4, {{0, 0, 1, 0}, {1, 1, 1, 0}, {2, 2, 1, 0}, {3, 3, cl, sl}});
}

TEST(R1121Gates, CUWithGlobalPhaseExact) {
    const double th = 0.7, ph = 0.3, la = -0.4, ga = 0.9;
    const double c = std::cos(th / 2), s = std::sin(th / 2);
    auto CU = extract(2, [&](Statevector& sv) {
        gates::apply_cu(sv, 0, 1, th, ph, la, ga);
    });
    // Active block on ctrl=1 indices {1, 3}; every element carries e^{i*ga}.
    expect_sparse(CU, 4,
        {{0, 0, 1, 0}, {2, 2, 1, 0},
         {1, 1, std::cos(ga) * c, std::sin(ga) * c},
         {1, 3, -std::cos(ga + la) * s, -std::sin(ga + la) * s},
         {3, 1, std::cos(ga + ph) * s, std::sin(ga + ph) * s},
         {3, 3, std::cos(ga + ph + la) * c, std::sin(ga + ph + la) * c}});
}

TEST(R1121Gates, ECRExactBothOperandOrders) {
    const double r = INV_SQRT2;
    // Frozen convention (docs/api/gates.md): ecr(a, b) binds the FIRST
    // argument to the HIGH bit of the documented matrix, so ecr(1, 0) is the
    // documented matrix verbatim in physical indexing (idx = q0 + 2*q1)...
    auto M10 = extract(2, [](Statevector& s) { gates::apply_ecr(s, 1, 0); });
    expect_sparse(M10, 4, {{0, 2, r, 0}, {0, 3, 0, r},
                           {1, 2, 0, r}, {1, 3, r, 0},
                           {2, 0, r, 0}, {2, 1, 0, -r},
                           {3, 0, 0, -r}, {3, 1, r, 0}});

    // ...and ecr(0, 1) is its SWAP conjugate (== Qiskit's ecr(1, 0)).
    auto M01 = extract(2, [](Statevector& s) { gates::apply_ecr(s, 0, 1); });
    expect_sparse(M01, 4, {{0, 1, r, 0}, {0, 3, 0, r},
                           {1, 0, r, 0}, {1, 2, 0, -r},
                           {2, 1, 0, r}, {2, 3, r, 0},
                           {3, 0, 0, -r}, {3, 2, r, 0}});
}

TEST(R1121Gates, RZXExactBothOperandOrders) {
    const double c = std::cos(kTheta / 2), s = std::sin(kTheta / 2);
    // rzx(0, 1): Z on q0, X on q1: pairs (0,2) with -i*s, (1,3) with +i*s.
    auto M01 = extract(2, [](Statevector& sv) { gates::apply_rzx(sv, 0, 1, kTheta); });
    expect_sparse(M01, 4, {{0, 0, c, 0}, {2, 2, c, 0}, {0, 2, 0, -s}, {2, 0, 0, -s},
                           {1, 1, c, 0}, {3, 3, c, 0}, {1, 3, 0, s}, {3, 1, 0, s}});

    // rzx(1, 0): Z on q1, X on q0: pairs (0,1) with -i*s, (2,3) with +i*s.
    auto M10 = extract(2, [](Statevector& sv) { gates::apply_rzx(sv, 1, 0, kTheta); });
    expect_sparse(M10, 4, {{0, 0, c, 0}, {1, 1, c, 0}, {0, 1, 0, -s}, {1, 0, 0, -s},
                           {2, 2, c, 0}, {3, 3, c, 0}, {2, 3, 0, s}, {3, 2, 0, s}});
}

TEST(R1121Gates, RXXRYYRZZExact) {
    const double c = std::cos(kTheta / 2), s = std::sin(kTheta / 2);

    auto RXX = extract(2, [](Statevector& sv) { gates::apply_rxx(sv, 0, 1, kTheta); });
    expect_sparse(RXX, 4, {{0, 0, c, 0}, {1, 1, c, 0}, {2, 2, c, 0}, {3, 3, c, 0},
                           {0, 3, 0, -s}, {3, 0, 0, -s}, {1, 2, 0, -s}, {2, 1, 0, -s}});

    auto RYY = extract(2, [](Statevector& sv) { gates::apply_ryy(sv, 0, 1, kTheta); });
    expect_sparse(RYY, 4, {{0, 0, c, 0}, {1, 1, c, 0}, {2, 2, c, 0}, {3, 3, c, 0},
                           {0, 3, 0, s}, {3, 0, 0, s}, {1, 2, 0, -s}, {2, 1, 0, -s}});

    auto RZZ = extract(2, [](Statevector& sv) { gates::apply_rzz(sv, 0, 1, kTheta); });
    expect_sparse(RZZ, 4, {{0, 0, c, -s}, {1, 1, c, s}, {2, 2, c, s}, {3, 3, c, -s}});
}

TEST(R1121Gates, TwoQubitGatesOnNonAdjacentQubits) {
    // CX(0, 2) on n = 3: flips q2 when q0 = 1: pairs (1,5) and (3,7).
    auto M = extract(3, [](Statevector& s) { gates::apply_cx(s, 0, 2); });
    std::vector<Entry> want = {{0, 0, 1, 0}, {2, 2, 1, 0}, {4, 4, 1, 0}, {6, 6, 1, 0},
                               {1, 5, 1, 0}, {5, 1, 1, 0}, {3, 7, 1, 0}, {7, 3, 1, 0}};
    expect_sparse(M, 8, want);
}

TEST(R1121Gates, AllTwoQubitGatesAreUnitary) {
    using A = std::function<void(Statevector&)>;
    std::vector<A> appliers = {
        [](Statevector& s) { gates::apply_cx(s, 0, 1); },
        [](Statevector& s) { gates::apply_cy(s, 0, 1); },
        [](Statevector& s) { gates::apply_cz(s, 0, 1); },
        [](Statevector& s) { gates::apply_ch(s, 0, 1); },
        [](Statevector& s) { gates::apply_swap(s, 0, 1); },
        [](Statevector& s) { gates::apply_iswap(s, 0, 1); },
        [](Statevector& s) { gates::apply_crx(s, 0, 1, kTheta); },
        [](Statevector& s) { gates::apply_cry(s, 0, 1, kTheta); },
        [](Statevector& s) { gates::apply_crz(s, 0, 1, kTheta); },
        [](Statevector& s) { gates::apply_cp(s, 0, 1, kTheta); },
        [](Statevector& s) { gates::apply_cu(s, 0, 1, 0.7, 0.3, -0.4, 0.9); },
        [](Statevector& s) { gates::apply_ecr(s, 0, 1); },
        [](Statevector& s) { gates::apply_rzx(s, 0, 1, kTheta); },
        [](Statevector& s) { gates::apply_rxx(s, 0, 1, kTheta); },
        [](Statevector& s) { gates::apply_ryy(s, 0, 1, kTheta); },
        [](Statevector& s) { gates::apply_rzz(s, 0, 1, kTheta); },
    };
    for (size_t i = 0; i < appliers.size(); ++i) {
        auto M = extract(2, appliers[i]);
        SCOPED_TRACE("two-qubit applier #" + std::to_string(i));
        expect_unitary(M, 4);
    }
}

// =============================================================================
// Three-qubit gates on n = 3
// =============================================================================

TEST(R1121Gates, CCXExactAndPermutedOperands) {
    auto M = extract(3, [](Statevector& s) { gates::apply_ccx(s, 0, 1, 2); });
    std::vector<Entry> want;
    for (size_t i = 0; i < 8; ++i)
        if (i != 3 && i != 7) want.push_back({i, i, 1.0, 0.0});
    want.push_back({3, 7, 1.0, 0.0});
    want.push_back({7, 3, 1.0, 0.0});
    expect_sparse(M, 8, want);

    // Controls q2 and q0, target q1: swap 0b101=5 and 0b111=7.
    auto P = extract(3, [](Statevector& s) { gates::apply_ccx(s, 2, 0, 1); });
    std::vector<Entry> wantP;
    for (size_t i = 0; i < 8; ++i)
        if (i != 5 && i != 7) wantP.push_back({i, i, 1.0, 0.0});
    wantP.push_back({5, 7, 1.0, 0.0});
    wantP.push_back({7, 5, 1.0, 0.0});
    expect_sparse(P, 8, wantP);
}

TEST(R1121Gates, CCZExact) {
    auto M = extract(3, [](Statevector& s) { gates::apply_ccz(s, 0, 1, 2); });
    std::vector<Entry> want;
    for (size_t i = 0; i < 8; ++i)
        want.push_back({i, i, (i == 7) ? -1.0 : 1.0, 0.0});
    expect_sparse(M, 8, want);
}

TEST(R1121Gates, CSwapExact) {
    // ctrl q0 = 1: swap q1 and q2: indices 0b011=3 and 0b101=5.
    auto M = extract(3, [](Statevector& s) { gates::apply_cswap(s, 0, 1, 2); });
    std::vector<Entry> want;
    for (size_t i = 0; i < 8; ++i)
        if (i != 3 && i != 5) want.push_back({i, i, 1.0, 0.0});
    want.push_back({3, 5, 1.0, 0.0});
    want.push_back({5, 3, 1.0, 0.0});
    expect_sparse(M, 8, want);
}

TEST(R1121Gates, RCCXExactActionAndUnitarity) {
    // Verified action (matches the B4 regression family): with c1=q0, c2=q1,
    // t=q2: -1 on idx5 (c1=1,c2=0,t=1); idx3 (c1,c2=1,t=0) -> i*idx7;
    // idx7 -> -i*idx3; identity elsewhere.
    auto M = extract(3, [](Statevector& s) { gates::apply_rccx(s, 0, 1, 2); });
    std::vector<Entry> want;
    for (size_t i = 0; i < 8; ++i) {
        if (i == 3 || i == 7) continue;
        want.push_back({i, i, (i == 5) ? -1.0 : 1.0, 0.0});
    }
    want.push_back({7, 3, 0.0, 1.0});   // |110> -> i|111>
    want.push_back({3, 7, 0.0, -1.0});  // |111> -> -i|110>
    expect_sparse(M, 8, want, 1e-10);
    expect_unitary(M, 8);
}

TEST(R1121Gates, RCCXEqualsDefiningSequence) {
    auto direct = extract(3, [](Statevector& s) { gates::apply_rccx(s, 0, 1, 2); });
    auto seq = extract(3, [](Statevector& s) {
        gates::apply_h(s, 2);
        gates::apply_t(s, 2);
        gates::apply_cx(s, 1, 2);
        gates::apply_tdg(s, 2);
        gates::apply_cx(s, 0, 2);
        gates::apply_t(s, 2);
        gates::apply_cx(s, 1, 2);
        gates::apply_tdg(s, 2);
        gates::apply_h(s, 2);
    });
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(direct[i].real, seq[i].real, 1e-10);
        EXPECT_NEAR(direct[i].imag, seq[i].imag, 1e-10);
    }
}

TEST(R1121Gates, ThreeQubitSelfInverses) {
    auto ccx2 = extract(3, [](Statevector& s) {
        gates::apply_ccx(s, 0, 1, 2);
        gates::apply_ccx(s, 0, 1, 2);
    });
    expect_identity(ccx2, 8);

    auto cswap2 = extract(3, [](Statevector& s) {
        gates::apply_cswap(s, 0, 1, 2);
        gates::apply_cswap(s, 0, 1, 2);
    });
    expect_identity(cswap2, 8);
}

// =============================================================================
// apply_unitary: the targets[0]-is-LSB contract
// =============================================================================

TEST(R1121Gates, ApplyUnitarySingleQubitOnHighWire) {
    std::vector<Complex128> X = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    auto M = extract(2, [&](Statevector& s) { gates::apply_unitary(s, {1}, X); });
    expect_sparse(M, 4, {{0, 2, 1, 0}, {2, 0, 1, 0}, {1, 3, 1, 0}, {3, 1, 1, 0}});
}

TEST(R1121Gates, ApplyUnitaryTwoQubitDescendingNonContiguousTargets) {
    // CX-shaped matrix with control = targets[0]; targets = {2, 0} on n = 3.
    // Control is q2, target is q0: pairs (4,5) and (6,7) swap.
    std::vector<Complex128> CXm(16, Complex128(0, 0));
    CXm[0 * 4 + 0] = {1, 0};
    CXm[1 * 4 + 3] = {1, 0};
    CXm[2 * 4 + 2] = {1, 0};
    CXm[3 * 4 + 1] = {1, 0};
    auto M = extract(3, [&](Statevector& s) { gates::apply_unitary(s, {2, 0}, CXm); });
    std::vector<Entry> want = {{0, 0, 1, 0}, {1, 1, 1, 0}, {2, 2, 1, 0}, {3, 3, 1, 0},
                               {4, 5, 1, 0}, {5, 4, 1, 0}, {6, 7, 1, 0}, {7, 6, 1, 0}};
    expect_sparse(M, 8, want);
}

TEST(R1121Gates, ApplyUnitaryThreeQubitMatchesCCX) {
    // Permutation matrix swapping sub-states 3 and 7 == CCX(c=targets[0],[1]).
    std::vector<Complex128> U(64, Complex128(0, 0));
    for (size_t i = 0; i < 8; ++i) {
        size_t j = (i == 3) ? 7 : (i == 7) ? 3 : i;
        U[j * 8 + i] = {1, 0};
    }
    auto viaUnitary = extract(3, [&](Statevector& s) {
        gates::apply_unitary(s, {0, 1, 2}, U);
    });
    auto viaGate = extract(3, [](Statevector& s) { gates::apply_ccx(s, 0, 1, 2); });
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(viaUnitary[i].real, viaGate[i].real, kTol);
        EXPECT_NEAR(viaUnitary[i].imag, viaGate[i].imag, kTol);
    }
}

TEST(R1121Gates, ApplyUnitarySizeMismatchThrows) {
    Statevector sv(2);
    std::vector<Complex128> tooSmall(4, Complex128(1, 0));
    EXPECT_THROW(gates::apply_unitary(sv, {0, 1}, tooSmall), std::invalid_argument);
}

TEST(R1121Gates, ApplyUnitaryScratchBufferReuseAcrossSizes) {
    // Exercises the thread-local scratch resize path: 2q, then 1q, then 3q on
    // the SAME thread; each application must remain exact.
    std::vector<Complex128> X = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    std::vector<Complex128> SWAPm(16, Complex128(0, 0));
    SWAPm[0 * 4 + 0] = {1, 0};
    SWAPm[1 * 4 + 2] = {1, 0};
    SWAPm[2 * 4 + 1] = {1, 0};
    SWAPm[3 * 4 + 3] = {1, 0};
    std::vector<Complex128> I8(64, Complex128(0, 0));
    for (size_t i = 0; i < 8; ++i) I8[i * 8 + i] = {1, 0};

    Statevector sv(3);
    sv.initialize_basis(1);                      // |q0=1>
    gates::apply_unitary(sv, {0, 1}, SWAPm);     // -> |q1=1> = idx 2
    EXPECT_NEAR(sv.probability(2), 1.0, kTol);
    gates::apply_unitary(sv, {2}, X);            // -> idx 6
    EXPECT_NEAR(sv.probability(6), 1.0, kTol);
    gates::apply_unitary(sv, {0, 1, 2}, I8);     // identity, larger scratch
    EXPECT_NEAR(sv.probability(6), 1.0, kTol);
    EXPECT_NEAR(sv.norm_sq(), 1.0, kTol);
}

// =============================================================================
// Global sanity: every single-qubit gate is unitary at a generic angle
// =============================================================================

TEST(R1121Gates, AllSingleQubitGatesAreUnitary) {
    using A = std::function<void(Statevector&)>;
    std::vector<A> appliers = {
        [](Statevector& s) { gates::apply_h(s, 0); },
        [](Statevector& s) { gates::apply_x(s, 0); },
        [](Statevector& s) { gates::apply_y(s, 0); },
        [](Statevector& s) { gates::apply_z(s, 0); },
        [](Statevector& s) { gates::apply_s(s, 0); },
        [](Statevector& s) { gates::apply_sdg(s, 0); },
        [](Statevector& s) { gates::apply_t(s, 0); },
        [](Statevector& s) { gates::apply_tdg(s, 0); },
        [](Statevector& s) { gates::apply_sx(s, 0); },
        [](Statevector& s) { gates::apply_sxdg(s, 0); },
        [](Statevector& s) { gates::apply_rx(s, 0, kTheta); },
        [](Statevector& s) { gates::apply_ry(s, 0, kTheta); },
        [](Statevector& s) { gates::apply_rz(s, 0, kTheta); },
        [](Statevector& s) { gates::apply_p(s, 0, kTheta); },
        [](Statevector& s) { gates::apply_u(s, 0, 0.7, 0.3, -0.4); },
        [](Statevector& s) { gates::apply_u1(s, 0, kTheta); },
        [](Statevector& s) { gates::apply_u2(s, 0, 0.3, -0.4); },
        [](Statevector& s) { gates::apply_u3(s, 0, 0.7, 0.3, -0.4); },
    };
    for (size_t i = 0; i < appliers.size(); ++i) {
        auto M = extract(1, appliers[i]);
        SCOPED_TRACE("single-qubit applier #" + std::to_string(i));
        expect_unitary(M, 2);
    }
}
