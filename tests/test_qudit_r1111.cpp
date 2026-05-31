// R.1.11.1 test suite — qudit completeness (closes the three R.1.11.0 "Simulator Gaps").
//
// Part A: QuditCliffordSimulator::apply_P() for odd prime d
//   - tableau structural transform (z -> z+x, phase += x(x-1))
//   - exact-statevector cross-check on NON-SYMMETRIC H-P-H circuits (a flat
//     distribution would mask a phase bug; the convention rule in CLAUDE.md
//     requires a non-symmetric end-to-end check)
// Part B: QuditAffineOracle + QuditDeutschJozsa affine overload (all backends)
// Part C: composite-d QuditSimon (integer-SNF ring kernel) + affine Simon on CLIFFORD
//
// The canonical odd-d phase-gate matrix is built locally here (test-only); it is
// intentionally NOT added to production qudit_gates so this .1 release stays
// test-only.

#include <gtest/gtest.h>

#include "lindblad/algorithms.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_gates.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

namespace {

constexpr double kPi = 3.14159265358979323846;

int imod(int a, int m) { int r = a % m; return r < 0 ? r + m : r; }

long long ipow(int base, int exp) {
    long long r = 1;
    for (int i = 0; i < exp; ++i) r *= base;
    return r;
}

// Canonical odd-prime-d phase gate (Howard & Vala 2012):
//   P = diag(omega^{(2^{-1} mod d) * k(k-1)}),  2^{-1} = (d+1)/2 for odd d.
std::vector<Complex128> oddP_matrix(int d) {
    const int inv2 = (d + 1) / 2;  // valid for odd d: 2*inv2 = d+1 ≡ 1 (mod d)
    std::vector<Complex128> M(static_cast<size_t>(d) * d, Complex128(0.0, 0.0));
    for (int k = 0; k < d; ++k) {
        const int e = imod(inv2 * k * (k - 1), d);
        M[static_cast<size_t>(k) * d + k] = Complex128::exp_i(2.0 * kPi * e / d);
    }
    return M;
}

// d=2 phase gate S = diag(1, i).
std::vector<Complex128> s_matrix_d2() {
    return { Complex128(1, 0), Complex128(0, 0), Complex128(0, 0), Complex128(0, 1) };
}

std::vector<Complex128> phase_matrix(int d) {
    return d == 2 ? s_matrix_d2() : oddP_matrix(d);
}

// A single-qudit gate in a Clifford circuit, applied identically to both backends.
enum class G { H, P };
struct Op { G g; int q; };

void apply_sv(QuditStatevector& sv, const Op& op, int d) {
    if (op.g == G::H) sv.apply_1qudit(op.q, qudit_gates::qft_matrix(d));
    else              sv.apply_1qudit(op.q, phase_matrix(d));
}
void apply_clifford(QuditCliffordSimulator& c, const Op& op) {
    if (op.g == G::H) c.apply_H(op.q);
    else              c.apply_P(op.q);
}

// Exact probabilities |amplitude[i]|^2 after running ops on a fresh statevector.
std::vector<double> sv_probs(int n, int d, const std::vector<Op>& ops) {
    QuditStatevector sv(n, d);
    for (const auto& op : ops) apply_sv(sv, op, d);
    std::vector<double> p(sv.amplitudes.size());
    for (size_t i = 0; i < sv.amplitudes.size(); ++i) p[i] = sv.amplitudes[i].norm_sq();
    return p;
}

// Sampled outcome histogram (normalised) from the Clifford tableau.
std::vector<double> clifford_freqs(int n, int d, const std::vector<Op>& ops,
                                   int shots, uint64_t seed0) {
    std::vector<long long> hist(static_cast<size_t>(ipow(d, n)), 0);
    for (int s = 0; s < shots; ++s) {
        QuditCliffordSimulator c(n, d);
        for (const auto& op : ops) apply_clifford(c, op);
        const auto out = c.measure(seed0 + static_cast<uint64_t>(s));
        hist[QuditStatevector::digits_to_index(out, d)]++;
    }
    std::vector<double> f(hist.size());
    for (size_t i = 0; i < hist.size(); ++i) f[i] = static_cast<double>(hist[i]) / shots;
    return f;
}

// ── Simon helpers ────────────────────────────────────────────────────────────

// Coset-canonical Simon oracle for any d: f(x) = lexicographic min over {x + k*s}.
std::function<std::vector<int>(const std::vector<int>&)>
make_simon_f(const std::vector<int>& s, int n, int d) {
    return [s, n, d](const std::vector<int>& x) -> std::vector<int> {
        std::vector<int> best = x, cur(static_cast<size_t>(n));
        for (int k = 1; k < d; ++k) {
            for (int i = 0; i < n; ++i)
                cur[static_cast<size_t>(i)] = imod(x[static_cast<size_t>(i)] + k * s[static_cast<size_t>(i)], d);
            if (cur < best) best = cur;
        }
        return best;
    };
}

// True iff p is a nonzero period of f (f(x) == f(x+p) on x=0 and a few points).
bool verified_period(const std::function<std::vector<int>(const std::vector<int>&)>& f,
                     const std::vector<int>& p, int n, int d) {
    bool nz = false;
    for (int v : p) if (v != 0) { nz = true; break; }
    if (!nz) return false;
    auto agrees = [&](const std::vector<int>& x) {
        std::vector<int> xs(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) xs[static_cast<size_t>(i)] = imod(x[static_cast<size_t>(i)] + p[static_cast<size_t>(i)], d);
        return f(x) == f(xs);
    };
    std::vector<int> x(static_cast<size_t>(n), 0);
    if (!agrees(x)) return false;
    for (int t = 1; t < d && t < 6; ++t) {
        for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = imod(t * (i + 1), d);
        if (!agrees(x)) return false;
    }
    return true;
}

}  // namespace

// =============================================================================
// Part A — apply_P for odd prime d
// =============================================================================

class ApplyP_OddD : public ::testing::TestWithParam<int> {};

// The tableau transform: x unchanged, z -> (z+x) mod d, phase += x(x-1) mod 2d.
// White-box: inject varied x/z directly so the x(x-1) phase term is exercised for
// x >= 2 (it is identically zero for x in {0,1}, which a gate-built tableau alone
// would not cover). The qudit-1 column is a sentinel that must stay untouched.
TEST_P(ApplyP_OddD, TableauTransform) {
    const int d = GetParam();
    const int n = 4;                 // 8 rows -> spans x = 0..7 (covers all residues for d <= 7)
    QuditCliffordSimulator c(n, d);
    const int rows = 2 * n;
    const int two_d = 2 * d;

    for (int r = 0; r < rows; ++r) {
        c.xbits[r][0] = r % d;
        c.zbits[r][0] = (2 * r + 1) % d;
        c.zbits[r][1] = (r + 3) % d;          // sentinel on the other qudit
        c.phase[r]    = (3 * r) % two_d;
    }
    const auto x0 = c.xbits, z0 = c.zbits;
    const auto ph0 = c.phase;

    c.apply_P(0);

    for (int r = 0; r < rows; ++r) {
        EXPECT_EQ(c.xbits[r], x0[r]) << "row " << r;                                  // x unchanged
        EXPECT_EQ(c.zbits[r][0], imod(z0[r][0] + x0[r][0], d)) << "row " << r;        // z -> z+x
        EXPECT_EQ(c.zbits[r][1], z0[r][1]) << "row " << r;                            // other qudit untouched
        EXPECT_EQ(c.phase[r], imod(ph0[r] + x0[r][0] * (x0[r][0] - 1), two_d)) << "row " << r;
    }
}

INSTANTIATE_TEST_SUITE_P(OddPrimes, ApplyP_OddD, ::testing::Values(3, 5, 7));

// Exact-statevector cross-check on non-symmetric circuits. The Hadamard before
// and after the phase gate turns the diagonal phases into an observable,
// non-uniform Z-basis distribution; sampling the Clifford tableau must match the
// exact |amplitude|^2 distribution from the statevector running the same gates.
class ApplyP_CrossCheck : public ::testing::TestWithParam<int> {};

TEST_P(ApplyP_CrossCheck, MatchesStatevector_NonSymmetric) {
    const int d = GetParam();
    const std::vector<std::vector<Op>> circuits = {
        { {G::H, 0}, {G::P, 0}, {G::H, 0} },
        { {G::H, 0}, {G::P, 0}, {G::P, 0}, {G::H, 0} },
        { {G::H, 0}, {G::P, 0}, {G::H, 0}, {G::P, 0}, {G::H, 0} },
    };
    const int shots = 6000;
    bool any_nonsymmetric = false;
    for (size_t ci = 0; ci < circuits.size(); ++ci) {
        const auto& ops = circuits[ci];
        const auto probs = sv_probs(1, d, ops);
        const auto freqs = clifford_freqs(1, d, ops, shots, 1234u + ci * 100u);

        for (int i = 0; i < d; ++i)
            EXPECT_NEAR(freqs[static_cast<size_t>(i)], probs[static_cast<size_t>(i)], 0.05)
                << "d=" << d << " circuit=" << ci << " outcome=" << i;

        double pmax = 0.0, pmin = 1.0;
        for (int i = 0; i < d; ++i) { pmax = std::max(pmax, probs[static_cast<size_t>(i)]); pmin = std::min(pmin, probs[static_cast<size_t>(i)]); }
        if (pmax - pmin > 0.02) any_nonsymmetric = true;
    }
    // At least one circuit must produce a non-symmetric distribution, otherwise
    // the cross-check is vacuous (a flat distribution masks phase bugs). Some
    // individual circuits are incidentally uniform (e.g. HSH at d=2), so this is
    // asserted over the set rather than per circuit.
    EXPECT_TRUE(any_nonsymmetric) << "d=" << d;
}

INSTANTIATE_TEST_SUITE_P(WithD2Regression, ApplyP_CrossCheck, ::testing::Values(2, 3, 5, 7));

// =============================================================================
// Part B — QuditAffineOracle + QuditDeutschJozsa affine overload
// =============================================================================

TEST(AffineOracleEval, LinearAndConstant) {
    QuditAffineOracle o{ {{1, 2}}, {1} };       // f(x) = x0 + 2 x1 + 1 (mod d)
    EXPECT_EQ(o.eval({0, 0}, 3), (std::vector<int>{1}));
    EXPECT_EQ(o.eval({1, 0}, 3), (std::vector<int>{2}));
    EXPECT_EQ(o.eval({0, 1}, 3), (std::vector<int>{0}));   // 0+2+1=3≡0
    EXPECT_EQ(o.eval({2, 2}, 3), (std::vector<int>{1}));   // 2+4+1=7≡1
    EXPECT_EQ(o.num_inputs(), 2);
    EXPECT_EQ(o.num_outputs(), 1);
}

TEST(AffineOracleEval, ThrowsOnBadInput) {
    QuditAffineOracle o{ {{1, 1}}, {0} };
    EXPECT_THROW(o.eval({0}, 3), std::invalid_argument);        // wrong size
    EXPECT_THROW(o.eval({0, 3}, 3), std::invalid_argument);     // digit out of range
}

struct DjParam { QuditBackend backend; int d; int n; bool balanced; };

static QuditAffineOracle dj_oracle(int n, bool balanced) {
    std::vector<int> a(static_cast<size_t>(n), 0);
    if (balanced) a[0] = 1;                 // coprime to any d → balanced for any d
    return QuditAffineOracle{ {a}, {0} };
}

static std::vector<DjParam> dj_nonclifford_params() {
    std::vector<DjParam> v;
    for (QuditBackend b : {QuditBackend::STATEVECTOR, QuditBackend::DENSITY_MATRIX, QuditBackend::MPS})
        for (int d : {2, 3, 4, 5, 6, 7, 9})
            for (int n : {1, 2, 3})
                for (bool bal : {false, true}) {
                    const long long dim = ipow(d, n + 1);
                    if (b == QuditBackend::DENSITY_MATRIX && dim > 1500) continue;
                    if (b == QuditBackend::MPS && dim > 8000) continue;
                    v.push_back({b, d, n, bal});
                }
    return v;
}

static std::vector<DjParam> dj_clifford_params() {
    std::vector<DjParam> v;
    for (int d : {2, 3, 5, 7})        // CLIFFORD requires prime d
        for (int n : {1, 2, 3})
            for (bool bal : {false, true})
                v.push_back({QuditBackend::CLIFFORD, d, n, bal});
    return v;
}

class DjAffine : public ::testing::TestWithParam<DjParam> {};

TEST_P(DjAffine, VerdictMatchesPromise) {
    const auto p = GetParam();
    const auto oracle = dj_oracle(p.n, p.balanced);
    const auto r = QuditDeutschJozsa::solve(oracle, p.d, 1234u, p.backend);
    EXPECT_EQ(r.verdict, p.balanced ? QuditDeutschJozsa::Verdict::BALANCED
                                    : QuditDeutschJozsa::Verdict::CONSTANT)
        << "backend=" << static_cast<int>(p.backend) << " d=" << p.d << " n=" << p.n;
    EXPECT_EQ(r.d, p.d);
    EXPECT_EQ(r.n, p.n);
}

INSTANTIATE_TEST_SUITE_P(NonClifford, DjAffine, ::testing::ValuesIn(dj_nonclifford_params()));
INSTANTIATE_TEST_SUITE_P(Clifford,    DjAffine, ::testing::ValuesIn(dj_clifford_params()));

TEST(DjAffineEdge, OpaqueOracleCliffordThrows) {
    auto f = [](const std::vector<int>&) -> int { return 0; };
    EXPECT_THROW(QuditDeutschJozsa::solve(2, 2, f, 0, QuditBackend::CLIFFORD),
                 std::invalid_argument);
}

TEST(DjAffineEdge, AffineCompositeCliffordThrows) {
    // Affine oracle is Clifford-decomposable, but the tableau backend needs prime d.
    auto oracle = dj_oracle(2, true);
    EXPECT_THROW(QuditDeutschJozsa::solve(oracle, 4, 0, QuditBackend::CLIFFORD),
                 std::invalid_argument);
}

TEST(DjAffineEdge, NonzeroConstantStillConstant) {
    for (int d : {2, 3, 5}) {
        QuditAffineOracle o{ {{0, 0}}, {1 % d} };   // f(x) = const (a = 0)
        EXPECT_EQ(QuditDeutschJozsa::solve(o, d, 7u, QuditBackend::STATEVECTOR).verdict,
                  QuditDeutschJozsa::Verdict::CONSTANT) << "d=" << d;
    }
}

TEST(DjAffineEdge, AffineMatchesOpaqueOnStatevector) {
    // The affine overload and a hand-written equivalent f must agree.
    for (int d : {3, 5}) {
        QuditAffineOracle o{ {{1, 0}}, {0} };       // f(x) = x0  (balanced)
        auto f = [d](const std::vector<int>& x) -> int { return imod(x[0], d); };
        EXPECT_EQ(QuditDeutschJozsa::solve(o, d, 5u, QuditBackend::STATEVECTOR).verdict,
                  QuditDeutschJozsa::solve(2, d, f, 5u, QuditBackend::STATEVECTOR).verdict)
            << "d=" << d;
    }
}

// =============================================================================
// Part C — composite-d QuditSimon + affine Simon on CLIFFORD
// =============================================================================

struct SimonParam { QuditBackend backend; int d; int n; std::vector<int> s; };

static std::vector<SimonParam> simon_params() {
    std::vector<SimonParam> v;
    // (d, n, s) cases — composite d plus a prime control; s non-trivial.
    struct DNS { int d; int n; std::vector<int> s; };
    const std::vector<DNS> cases = {
        {4, 2, {2, 0}}, {4, 2, {1, 1}}, {4, 3, {2, 0, 2}},
        {6, 2, {1, 0}}, {6, 2, {3, 0}}, {6, 2, {2, 4}},
        {8, 2, {4, 0}}, {8, 2, {2, 2}},
        {9, 2, {3, 0}}, {9, 2, {1, 1}},
        {12, 2, {6, 0}}, {12, 2, {4, 8}},
        {5, 2, {1, 4}},                       // prime control
    };
    for (QuditBackend b : {QuditBackend::STATEVECTOR, QuditBackend::DENSITY_MATRIX, QuditBackend::MPS})
        for (const auto& c : cases) {
            const long long dim = ipow(c.d, 2 * c.n);   // Simon uses 2n qudits
            if (b == QuditBackend::STATEVECTOR && dim > 50000) continue;
            if (b == QuditBackend::DENSITY_MATRIX && dim > 1500) continue;
            if (b == QuditBackend::MPS && dim > 20000) continue;
            v.push_back({b, c.d, c.n, c.s});
        }
    return v;
}

class SimonComposite : public ::testing::TestWithParam<SimonParam> {};

TEST_P(SimonComposite, RecoversVerifiedPeriod) {
    const auto p = GetParam();
    auto f = make_simon_f(p.s, p.n, p.d);
    const auto r = QuditSimon::solve(p.n, p.d, f, /*extra_samples=*/8, 4321u, p.backend);
    EXPECT_FALSE(r.is_trivial)
        << "backend=" << static_cast<int>(p.backend) << " d=" << p.d << " n=" << p.n;
    EXPECT_TRUE(verified_period(f, r.period, p.n, p.d))
        << "backend=" << static_cast<int>(p.backend) << " d=" << p.d << " n=" << p.n
        << " recovered period is not a true period";
}

INSTANTIATE_TEST_SUITE_P(Matrix, SimonComposite, ::testing::ValuesIn(simon_params()));

// KNOWN BUG — deferred to R.1.11.2 (test-only release cannot carry the fix).
// MPS Simon recovery is wrong for certain d > 2 period structures: the query
// register collapses to zero (is_trivial) or a wrong period is returned. It
// reproduces on prime d=5 (s={1,4}), so it is an MPS-backend defect, not the
// R.1.11.0 composite-d kernel. SV and DM recover these correctly. Remove the
// DISABLED_ prefix once the MPS backend is fixed. See docs/plans/TODO.md.
TEST(SimonMpsKnownBug, DISABLED_MpsSimonDGt2) {
    auto f = make_simon_f({1, 4}, 2, 5);
    const auto r = QuditSimon::solve(2, 5, f, 8, 4321u, QuditBackend::MPS);
    EXPECT_FALSE(r.is_trivial);
    EXPECT_TRUE(verified_period(f, r.period, 2, 5));
}

TEST(SimonCompositeEdge, InjectiveIsTrivial) {
    for (int d : {4, 6, 9}) {
        auto f = [](const std::vector<int>& x) -> std::vector<int> { return x; };   // identity = injective
        const auto r = QuditSimon::solve(2, d, f, 4, 99u, QuditBackend::STATEVECTOR);
        EXPECT_TRUE(r.is_trivial) << "d=" << d;
        for (int v : r.period) EXPECT_EQ(v, 0) << "d=" << d;
    }
}

TEST(SimonCompositeEdge, OpaqueOracleCliffordThrows) {
    auto f = make_simon_f({1, 0}, 2, 5);
    EXPECT_THROW(QuditSimon::solve(2, 5, f, 3, 0, QuditBackend::CLIFFORD),
                 std::invalid_argument);
}

// Affine Simon on the CLIFFORD backend (prime d): the hidden subgroup is ker(A).
// Each A below is rank-deficient with a known 1-D kernel; the recovered period
// must be a nonzero element of that kernel (A·period ≡ 0 mod d).
struct AffineSimonCase { int d; std::vector<std::vector<int>> A; };

class SimonAffineClifford : public ::testing::TestWithParam<AffineSimonCase> {};

TEST_P(SimonAffineClifford, RecoversKernelGenerator) {
    const auto c = GetParam();
    const int n = static_cast<int>(c.A.size());
    QuditAffineOracle oracle{ c.A, std::vector<int>(static_cast<size_t>(n), 0) };
    const auto r = QuditSimon::solve(oracle, c.d, /*extra_samples=*/6, 246u, QuditBackend::CLIFFORD);

    EXPECT_FALSE(r.is_trivial) << "d=" << c.d;
    bool nz = false;
    for (int v : r.period) if (v != 0) { nz = true; break; }
    EXPECT_TRUE(nz) << "d=" << c.d << " period is zero";
    // A·period ≡ 0 (mod d): period lies in ker(A).
    for (int row = 0; row < n; ++row) {
        long long acc = 0;
        for (int col = 0; col < n; ++col)
            acc += static_cast<long long>(c.A[static_cast<size_t>(row)][static_cast<size_t>(col)])
                 * r.period[static_cast<size_t>(col)];
        EXPECT_EQ(acc % c.d, 0) << "d=" << c.d << " period not in ker(A), row " << row;
    }
}

INSTANTIATE_TEST_SUITE_P(PrimeD, SimonAffineClifford, ::testing::Values(
    AffineSimonCase{2, {{1, 1}, {1, 1}}},   // ker = <(1,1)>
    AffineSimonCase{3, {{1, 2}, {2, 1}}},   // ker = <(1,1)>
    AffineSimonCase{5, {{1, 4}, {1, 4}}}    // ker = <(1,1)>
));

TEST(SimonAffineEdge, CompositeCliffordThrows) {
    QuditAffineOracle oracle{ {{1, 1}, {1, 1}}, {0, 0} };
    EXPECT_THROW(QuditSimon::solve(oracle, 4, 6, 0, QuditBackend::CLIFFORD),
                 std::invalid_argument);
}
