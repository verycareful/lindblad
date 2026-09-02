#pragma once

// Oracles shared by the 1.1.25.1 Clifford suites.
//
// StabilizerState has no equality operator and its tableau is private, so every
// claim that two states match is made through public surface. Three oracles are
// provided because what they prove and what they cost differ sharply.
//
// states_equal_exhaustive compares all 4^n Pauli expectations. Since
// rho = 2^-n * sum_P <P> P, agreement on every P IS equality of the states, so
// this is a proof and not a probe. expectation_pauli runs a Gaussian
// elimination per call, which makes the 4^n calls affordable only up to about
// n = 6.
//
// fingerprint serves the boundary sizes, where 4^n is unreachable and one
// expectation_pauli call already costs O(n^3). It reads the outcome slab under
// three conjugations (Z, X and Y) and adds two measurement sweeps. The slab is
// in reduced row echelon form, so each of the three is a canonical description
// of the Z-only subgroup of the conjugated stabilizer group, and any relabelling
// of rows or columns by a wrong block walk moves at least one of them.
//
// exact_clifford_distribution and sv_expectation_pauli are the statevector
// cross-checks. Both are EXACT: the Clifford distribution is read off the slab
// rather than sampled, and the Pauli expectation is summed over amplitudes, so
// a comparison between the two backends carries no sampling tolerance at all.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace v11251 {

using lindblad::Complex128;
using lindblad::QuantumCircuit;
using lindblad::StabilizerState;
using lindblad::Statevector;
using lindblad::StatevectorSimulator;

// =============================================================================
// The one-time FourRussians note
// =============================================================================

// The library announces Elimination::FourRussians once per PROCESS, behind a
// function-local static that flush_warnings() cannot reset. Which test sees it
// would therefore depend on the order gtest happens to run them in, and any
// suite that selected the block route first would consume it.
//
// So every FourRussians selection in the 1.1.25.1 wave calls this first. Being
// an inline function, its own static is shared across all the translation units
// that include this header, so the first caller anywhere performs the first
// selection under a capturing handler and every later caller reads what it
// recorded.
inline const std::vector<std::string>& four_russians_first_notes() {
    static const std::vector<std::string> notes = [] {
        std::vector<std::string> captured;
        lindblad::set_warning_handler(
            [&captured](const std::string& m) { captured.push_back(m); });
        StabilizerState probe(4);
        probe.apply_h(0);
        probe.apply_cx(0, 1);
        probe.apply_cx(1, 2);
        (void)probe.outcome_slab(StabilizerState::Elimination::FourRussians);
        lindblad::set_warning_handler(nullptr);
        lindblad::flush_warnings();
        return captured;
    }();
    return notes;
}

// =============================================================================
// Gate scripts, in the vocabulary both tableau layouts share
// =============================================================================

// The gates StabilizerState and ColumnTableau both implement, which is what
// lets one script drive either layout. Rotations are absent because only the
// circuit-level dispatch decomposes them; neither layout takes an angle.
enum class Op { H, S, SDG, X, Y, Z, SX, SXDG, CX, CY, CZ, SWAP, ISWAP, ECR };

inline constexpr Op kLayoutOps[] = {
    Op::H, Op::S, Op::SDG, Op::X, Op::Y, Op::Z, Op::SX, Op::SXDG,
    Op::CX, Op::CY, Op::CZ, Op::SWAP, Op::ISWAP, Op::ECR};

struct Step {
    Op op;
    int a;
    int b;  // ignored by the single-qubit ops
};

inline const char* op_name(Op op) {
    switch (op) {
        case Op::H: return "h";
        case Op::S: return "s";
        case Op::SDG: return "sdg";
        case Op::X: return "x";
        case Op::Y: return "y";
        case Op::Z: return "z";
        case Op::SX: return "sx";
        case Op::SXDG: return "sxdg";
        case Op::CX: return "cx";
        case Op::CY: return "cy";
        case Op::CZ: return "cz";
        case Op::SWAP: return "swap";
        case Op::ISWAP: return "iswap";
        case Op::ECR: return "ecr";
    }
    return "?";
}

inline bool is_two_qubit(Op op) {
    switch (op) {
        case Op::CX: case Op::CY: case Op::CZ:
        case Op::SWAP: case Op::ISWAP: case Op::ECR:
            return true;
        default:
            return false;
    }
}

template <class Tableau>
void apply_step(Tableau& t, const Step& s) {
    switch (s.op) {
        case Op::H:     t.apply_h(s.a); break;
        case Op::S:     t.apply_s(s.a); break;
        case Op::SDG:   t.apply_sdg(s.a); break;
        case Op::X:     t.apply_x(s.a); break;
        case Op::Y:     t.apply_y(s.a); break;
        case Op::Z:     t.apply_z(s.a); break;
        case Op::SX:    t.apply_sx(s.a); break;
        case Op::SXDG:  t.apply_sxdg(s.a); break;
        case Op::CX:    t.apply_cx(s.a, s.b); break;
        case Op::CY:    t.apply_cy(s.a, s.b); break;
        case Op::CZ:    t.apply_cz(s.a, s.b); break;
        case Op::SWAP:  t.apply_swap(s.a, s.b); break;
        case Op::ISWAP: t.apply_iswap(s.a, s.b); break;
        case Op::ECR:   t.apply_ecr(s.a, s.b); break;
    }
}

inline StabilizerState run_row_major(int n, const std::vector<Step>& script) {
    StabilizerState st(n);
    for (const Step& s : script) apply_step(st, s);
    return st;
}

inline StabilizerState run_bit_sliced(int n, const std::vector<Step>& script) {
    StabilizerState::ColumnTableau cols(n);
    for (const Step& s : script) apply_step(cols, s);
    return cols.to_state();
}

// =============================================================================
// Exhaustive Pauli equality (a proof, for small n)
// =============================================================================

// Every Pauli string of length n, in the project's LSB-first order: index q of
// the string acts on qubit q. The odometer runs the 4^n strings in two-bit
// digits, so digit q of the counter selects the letter at position q.
inline std::vector<std::string> all_pauli_strings(int n) {
    const char letters[4] = {'I', 'X', 'Y', 'Z'};
    long long total = 1;
    for (int i = 0; i < n; ++i) total *= 4;
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(total));
    std::string p(static_cast<size_t>(n), 'I');
    for (long long code = 0; code < total; ++code) {
        long long v = code;
        for (int q = 0; q < n; ++q) {
            p[static_cast<size_t>(q)] = letters[v & 3];
            v >>= 2;
        }
        out.push_back(p);
    }
    return out;
}

inline ::testing::AssertionResult states_equal_exhaustive(
    const StabilizerState& a, const StabilizerState& b) {
    if (a.n_qubits != b.n_qubits) {
        return ::testing::AssertionFailure()
               << "n_qubits differ: " << a.n_qubits << " vs " << b.n_qubits;
    }
    for (const std::string& p : all_pauli_strings(a.n_qubits)) {
        const int ea = a.expectation_pauli(p);
        const int eb = b.expectation_pauli(p);
        if (ea != eb) {
            return ::testing::AssertionFailure()
                   << "Pauli '" << p << "': " << ea << " vs " << eb;
        }
    }
    return ::testing::AssertionSuccess();
}

// =============================================================================
// Structural fingerprint (for the boundary sizes)
// =============================================================================

struct SlabKey {
    int dim = 0;
    std::vector<uint64_t> offset;
    std::vector<std::vector<uint64_t>> basis;

    bool operator==(const SlabKey& o) const {
        return dim == o.dim && offset == o.offset && basis == o.basis;
    }
    bool operator!=(const SlabKey& o) const { return !(*this == o); }
};

inline SlabKey slab_key(
    const StabilizerState& s,
    StabilizerState::Elimination m = StabilizerState::Elimination::Plain) {
    const StabilizerState::OutcomeSlab slab = s.outcome_slab(m);
    SlabKey k;
    k.dim = slab.dim;
    k.offset = slab.offset;
    k.basis = slab.basis;
    return k;
}

struct Fingerprint {
    int n = 0;
    SlabKey z;                  // the state as it stands
    SlabKey x;                  // after H on every qubit: measures X
    SlabKey y;                  // after SDG then H on every qubit: measures Y
    std::vector<int> seeded;    // measure(q, random = true) sweep, fixed seed
    std::vector<int> forced;    // measure(q, random = false) sweep

    bool operator==(const Fingerprint& o) const {
        return n == o.n && z == o.z && x == o.x && y == o.y &&
               seeded == o.seeded && forced == o.forced;
    }
    bool operator!=(const Fingerprint& o) const { return !(*this == o); }
};

// The Y entry conjugates by H·SDG. Reading Z on H SDG |psi> is reading
// (H SDG)^dagger Z (H SDG) = S X S^dagger = Y on |psi>.
inline Fingerprint fingerprint(const StabilizerState& s) {
    Fingerprint f;
    f.n = s.n_qubits;
    f.z = slab_key(s);
    {
        StabilizerState t = s;
        for (int q = 0; q < t.n_qubits; ++q) t.apply_h(q);
        f.x = slab_key(t);
    }
    {
        StabilizerState t = s;
        for (int q = 0; q < t.n_qubits; ++q) { t.apply_sdg(q); t.apply_h(q); }
        f.y = slab_key(t);
    }
    {
        // A random outcome consumes the generator and a deterministic one does
        // not, so two states that disagree about WHICH qubits are determined
        // desynchronise the stream and diverge from that point on.
        StabilizerState t = s;
        std::mt19937_64 rng(0x5EEDu);
        for (int q = 0; q < t.n_qubits; ++q) f.seeded.push_back(t.measure(q, true, rng));
    }
    {
        // random = false pins every undetermined outcome to 0, so this sweep
        // reads the determined values alone and needs no generator.
        StabilizerState t = s;
        std::mt19937_64 unused(0);
        for (int q = 0; q < t.n_qubits; ++q) f.forced.push_back(t.measure(q, false, unused));
    }
    return f;
}

inline ::testing::AssertionResult fingerprints_equal(const Fingerprint& a,
                                                     const Fingerprint& b) {
    if (a.n != b.n) {
        return ::testing::AssertionFailure()
               << "n differs: " << a.n << " vs " << b.n;
    }
    if (a.z != b.z) return ::testing::AssertionFailure() << "Z-basis slab differs";
    if (a.x != b.x) return ::testing::AssertionFailure() << "X-basis slab differs";
    if (a.y != b.y) return ::testing::AssertionFailure() << "Y-basis slab differs";
    if (a.seeded != b.seeded) {
        for (size_t i = 0; i < a.seeded.size(); ++i) {
            if (a.seeded[i] != b.seeded[i]) {
                return ::testing::AssertionFailure()
                       << "seeded measurement sweep differs first at qubit " << i;
            }
        }
        return ::testing::AssertionFailure() << "seeded measurement sweep length differs";
    }
    if (a.forced != b.forced) {
        for (size_t i = 0; i < a.forced.size(); ++i) {
            if (a.forced[i] != b.forced[i]) {
                return ::testing::AssertionFailure()
                       << "forced measurement sweep differs first at qubit " << i;
            }
        }
        return ::testing::AssertionFailure() << "forced measurement sweep length differs";
    }
    return ::testing::AssertionSuccess();
}

inline ::testing::AssertionResult states_equal_structural(const StabilizerState& a,
                                                          const StabilizerState& b) {
    return fingerprints_equal(fingerprint(a), fingerprint(b));
}

// =============================================================================
// Exact distributions (no sampling anywhere)
// =============================================================================

using Dist = std::map<uint64_t, double>;

// Standard deviation of a binomial frequency estimate over `trials` draws.
//
// p is clamped into [0, 1] before use. A reference probability summed from
// floating-point terms can land a few ulps ABOVE one, and p * (1 - p) is then
// negative, so the square root is NaN. NaN compares false against every bound,
// which turns a correct measurement into a failure that reads as a real
// disagreement. Clamping costs nothing and removes the whole class.
inline double binomial_sigma(double p, int trials) {
    const double clamped = p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p);
    return std::sqrt(clamped * (1.0 - clamped) / static_cast<double>(trials));
}

// The slab is uniform over offset xor every subset-sum of basis, so enumerating
// the 2^dim subsets gives the distribution exactly. Confined to one word, which
// covers every size these comparisons use.
inline Dist exact_clifford_distribution(const StabilizerState& s) {
    const StabilizerState::OutcomeSlab slab = s.outcome_slab();
    Dist d;
    const int dim = slab.dim;
    EXPECT_LE(s.n_qubits, 64) << "exact_clifford_distribution packs one word";
    EXPECT_LE(dim, 24) << "2^dim enumeration would not terminate usefully";
    const double p = std::ldexp(1.0, -dim);
    const uint64_t base = slab.offset.empty() ? 0ULL : slab.offset[0];
    for (uint64_t m = 0; m < (1ULL << dim); ++m) {
        uint64_t y = base;
        for (int b = 0; b < dim; ++b) {
            if ((m >> b) & 1ULL) y ^= slab.basis[static_cast<size_t>(b)][0];
        }
        d[y] += p;
    }
    return d;
}

// Probabilities of the statevector produced by a circuit that must carry no
// measurement: shots == 0 is then a plain forward pass.
inline Dist exact_statevector_distribution(const QuantumCircuit& qc,
                                           double floor = 1e-12) {
    StatevectorSimulator sim;
    auto r = sim.run(qc, /*shots=*/0, /*seed=*/1);
    Dist d;
    for (size_t i = 0; i < r.final_state.dim; ++i) {
        const double p = r.final_state.probability(i);
        if (p > floor) d[static_cast<uint64_t>(i)] = p;
    }
    return d;
}

inline ::testing::AssertionResult distributions_equal(const Dist& a, const Dist& b,
                                                      double tol = 1e-9) {
    for (const auto& [k, pa] : a) {
        const auto it = b.find(k);
        if (it == b.end()) {
            return ::testing::AssertionFailure()
                   << "outcome " << k << " has probability " << pa
                   << " on the left and no support on the right";
        }
        if (std::abs(pa - it->second) > tol) {
            return ::testing::AssertionFailure()
                   << "outcome " << k << ": " << pa << " vs " << it->second;
        }
    }
    for (const auto& [k, pb] : b) {
        if (a.find(k) == a.end()) {
            return ::testing::AssertionFailure()
                   << "outcome " << k << " has probability " << pb
                   << " on the right and no support on the left";
        }
    }
    return ::testing::AssertionSuccess();
}

// =============================================================================
// Exact Pauli expectation of a statevector
// =============================================================================

// <psi|P|psi> for an LSB-first Pauli string. Writing P as i^(number of Y) times
// an X mask followed by a Z mask, P|k> = i^nY * (-1)^popcount(k & zmask) *
// |k xor xmask>, so the expectation is one pass over the amplitudes.
inline Complex128 sv_expectation_pauli(const Statevector& sv, const std::string& pauli) {
    const int n = sv.n_qubits;
    uint64_t xmask = 0, zmask = 0;
    int n_y = 0;
    for (int q = 0; q < n; ++q) {
        const char c = pauli[static_cast<size_t>(q)];
        if (c == 'X') { xmask |= 1ULL << q; }
        else if (c == 'Y') { xmask |= 1ULL << q; zmask |= 1ULL << q; ++n_y; }
        else if (c == 'Z') { zmask |= 1ULL << q; }
    }
    Complex128 acc(0.0, 0.0);
    for (size_t k = 0; k < sv.dim; ++k) {
        const double sign =
            (std::popcount(static_cast<uint64_t>(k) & zmask) & 1) ? -1.0 : 1.0;
        acc += sv.amplitude(static_cast<size_t>(k ^ xmask)).conj() *
               sv.amplitude(k) * sign;
    }
    // i^n_y
    switch (n_y & 3) {
        case 0: break;
        case 1: acc *= Complex128(0.0, 1.0); break;
        case 2: acc *= Complex128(-1.0, 0.0); break;
        default: acc *= Complex128(0.0, -1.0); break;
    }
    return acc;
}

// The stabilizer formalism reports +1, -1 or 0 for a Pauli. A statevector's
// expectation is a continuous number, so it is rounded to that alphabet here
// and the rounding is checked: anything strictly between the three values means
// the reference state is not a stabilizer state and the comparison would be
// meaningless.
inline int sv_expectation_pauli_sign(const Statevector& sv, const std::string& pauli,
                                     double tol = 1e-9) {
    const Complex128 e = sv_expectation_pauli(sv, pauli);
    EXPECT_LT(std::abs(e.imag), tol)
        << "Pauli '" << pauli << "' expectation has imaginary part " << e.imag;
    const double re = e.real;
    if (std::abs(re - 1.0) < tol) return 1;
    if (std::abs(re + 1.0) < tol) return -1;
    EXPECT_LT(std::abs(re), tol)
        << "Pauli '" << pauli << "' expectation " << re
        << " is not one of -1, 0, +1; the reference is not a stabilizer state";
    return 0;
}

// Runs a measurement-free circuit on both backends and compares EVERY Pauli
// expectation. Complete up to global phase, which the stabilizer formalism does
// not represent, so this is the strongest statement available about a gate.
inline ::testing::AssertionResult clifford_matches_statevector_exhaustive(
    const QuantumCircuit& qc) {
    lindblad::CliffordSimulator cs;
    auto cr = cs.run(qc, /*shots=*/0, /*seed=*/1);

    StatevectorSimulator ss;
    auto sr = ss.run(qc, /*shots=*/0, /*seed=*/1);

    for (const std::string& p : all_pauli_strings(qc.n_qubits)) {
        const int got = cr.final_state.expectation_pauli(p);
        const int want = sv_expectation_pauli_sign(sr.final_state, p);
        if (got != want) {
            return ::testing::AssertionFailure()
                   << "Pauli '" << p << "': tableau " << got
                   << ", statevector " << want;
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace v11251
