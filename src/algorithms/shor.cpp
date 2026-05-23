#include "lindblad/algorithms.hpp"
#include "lindblad/backends/local_backend.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace lindblad {
namespace algorithms {

// =============================================================================
// Classical helpers
// =============================================================================

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b) { a %= b; std::swap(a, b); }
    return a;
}

// Modular exponentiation: base^exp mod m. Uses __uint128_t to avoid overflow.
static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = static_cast<uint64_t>(
            static_cast<__uint128_t>(result) * base % mod);
        base = static_cast<uint64_t>(
            static_cast<__uint128_t>(base) * base % mod);
        exp >>= 1;
    }
    return result;
}

// Deterministic Miller-Rabin for all N < 3.2×10⁹ using witnesses {2,3,5,7,11,13}.
static bool is_prime(uint64_t N) {
    if (N < 2) return false;
    for (uint64_t p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL})
        if (N == p) return true;
    for (uint64_t p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL})
        if (N % p == 0) return false;

    uint64_t d = N - 1; int r = 0;
    while (d % 2 == 0) { d /= 2; ++r; }

    for (uint64_t a : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL}) {
        if (a >= N) continue;
        uint64_t x = mod_pow(a, d, N);
        if (x == 1 || x == N - 1) continue;
        bool composite = true;
        for (int i = 0; i < r - 1; ++i) {
            x = static_cast<uint64_t>(
                static_cast<__uint128_t>(x) * x % N);
            if (x == N - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

// Check N = b^e for integer b ≥ 2, e ≥ 2. Returns true and sets base_out.
static bool is_perfect_power(uint64_t N, uint64_t& base_out) {
    for (int e = 2; e <= 63; ++e) {
        uint64_t b = static_cast<uint64_t>(
            std::round(std::pow(static_cast<double>(N), 1.0 / e)));
        for (uint64_t c : {b > 0 ? b - 1 : 0, b, b + 1}) {
            if (c < 2) continue;
            uint64_t pw = 1;
            bool overflow = false;
            for (int i = 0; i < e; ++i) {
                if (pw > N / c + 1) { overflow = true; break; }
                pw *= c;
            }
            if (!overflow && pw == N) { base_out = c; return true; }
        }
    }
    return false;
}

// Continued-fraction convergents of x in (0,1). Returns (numerator, denominator)
// pairs with denominator ≤ max_denom, ordered smallest-to-largest denominator.
std::vector<std::pair<uint64_t, uint64_t>>
Shor::cf_convergents(double x, uint64_t max_denom) {
    std::vector<std::pair<uint64_t, uint64_t>> conv;
    uint64_t h0 = 1, h1 = 0, k0 = 0, k1 = 1;
    for (int iter = 0; iter < 64; ++iter) {
        uint64_t a = static_cast<uint64_t>(std::floor(x));
        uint64_t h2 = a * h0 + h1;
        uint64_t k2 = a * k0 + k1;
        if (k2 > max_denom) break;
        conv.emplace_back(h2, k2);
        double rem = x - static_cast<double>(a);
        if (rem < 1e-12) break;
        x = 1.0 / rem;
        h1 = h0; h0 = h2;
        k1 = k0; k0 = k2;
    }
    return conv;
}

// =============================================================================
// build_period_finding_circuit
//
// Layout: eval register qubits [0, n_eval), target register [n_eval, n_eval+n_target).
//
// 1. H on each eval qubit.
// 2. X on qubit n_eval (target register = |1⟩).
// 3. For k = 0..n_eval-1:
//      Build U_{a^(2^k)}: permutation |x⟩→|a^(2^k)·x mod N⟩ on target register.
//      Wrap as a 1-control + n_target-target UNITARY gate.
// 4. IQFT on eval register (do_swaps=true).
// =============================================================================

QuantumCircuit Shor::build_period_finding_circuit(
    uint64_t a, uint64_t N, int n_eval, int n_target
) {
    const int total = n_eval + n_target;
    QuantumCircuit qc(total);
    const size_t td = 1ULL << n_target;  // target Hilbert-space dimension

    // Step 1: Hadamards on eval register
    for (int i = 0; i < n_eval; ++i) qc.h(i);

    // Step 2: |1⟩ in target register
    qc.x(n_eval);

    // Step 3: controlled-U^(2^k) gates
    // ak = a^(2^k) mod N, updated by squaring each iteration.
    uint64_t ak = a % N;
    for (int k = 0; k < n_eval; ++k) {
        // Build td×td permutation matrix U_ak: U[j,i] = 1 iff j = ak*i mod N (i<N)
        std::vector<Complex128> U(td * td, Complex128(0.0, 0.0));
        for (size_t x = 0; x < td; ++x) {
            size_t y = (x < N)
                ? static_cast<size_t>(
                    static_cast<__uint128_t>(ak) * x % N)
                : x;   // identity on out-of-range states
            U[y * td + x] = Complex128(1.0, 0.0);
        }

        // Build (2*td)×(2*td) controlled-U:
        //   ctrl=0 block (even indices) → identity
        //   ctrl=1 block (odd indices)  → U
        // apply_unitary convention: targets[0] (ctrl qubit k) maps to bit 0 (LSB)
        // of the matrix index, so ctrl=0 ↔ even and ctrl=1 ↔ odd.
        const size_t fd = 2 * td;
        std::vector<Complex128> CU(fd * fd, Complex128(0.0, 0.0));
        for (size_t i = 0; i < td; ++i)
            CU[(2 * i) * fd + (2 * i)] = Complex128(1.0, 0.0);
        for (size_t r = 0; r < td; ++r)
            for (size_t c = 0; c < td; ++c)
                CU[(2 * r + 1) * fd + (2 * c + 1)] = U[r * td + c];

        Instruction inst;
        inst.type   = Instruction::GateType::UNITARY;
        inst.qubits = {k};  // control = eval qubit k
        for (int tq = 0; tq < n_target; ++tq)
            inst.qubits.push_back(n_eval + tq);
        inst.matrix = std::move(CU);
        qc.instructions.push_back(inst);

        // a^(2^{k+1}) mod N = (a^(2^k))^2 mod N
        ak = static_cast<uint64_t>(
            static_cast<__uint128_t>(ak) * ak % N);
    }

    // Step 4: IQFT on eval register
    auto iqft = QFT::build_inverse_circuit(n_eval, /*do_swaps=*/true);
    for (const auto& inst : iqft.instructions)
        qc.instructions.push_back(inst);

    return qc;
}

// =============================================================================
// find_order
// =============================================================================

uint64_t Shor::find_order(
    uint64_t a, uint64_t N, int n_eval,
    backends::LocalBackend& backend,
    uint64_t seed
) {
    const int n_target = static_cast<int>(
        std::ceil(std::log2(static_cast<double>(N) + 1.0)));

    auto circuit = build_period_finding_circuit(a, N, n_eval, n_target);
    circuit.measure_all();

    auto br = backend.run(circuit, 128, seed);
    if (br.counts.empty()) return 0;

    // Iterate observed bitstrings in descending frequency. Return on the first
    // shot that yields a valid r (mod_pow(a, r, N) == 1 for some convergent
    // denominator).
    //
    // Why not pick a single most-frequent bitstring: when r divides 2^n_eval
    // (e.g. r=4 for N=15) every QPE peak lands on an exact m and the most-
    // frequent bitstring is a uniform pick across the r peaks, half of which
    // yield valid orders. When r does NOT divide 2^n_eval (e.g. r=6 for N=21),
    // the s=0 and s=r/2 peaks land on exact m (m=0, m=2^n_eval/2) and absorb
    // their whole ~shots/r probability mass into a single (m, target)
    // bitstring, while the spread-distributed useful peaks (s with gcd(s,r)=1)
    // dilute across many bitstrings. The "single most-frequent" strategy then
    // almost always picks an exact-but-useless peak. Iterating across counts
    // recovers ~100% success — the useful peaks are well represented in the
    // distribution, just not as the top bin.
    //
    // measure_all() maps qubit q → clbit q for all n_eval + n_target qubits.
    // The statevector simulator builds the bitstring with clbit 0 at the
    // rightmost character, so bits[total - 1 - q] is qubit q's outcome. The
    // eval register occupies qubits [0, n_eval); under the project LSB-at-
    // qubit-0 convention (see CLAUDE.md), eval qubit q contributes bit q of
    // the QPE integer m.
    std::vector<std::pair<std::string, int>> sorted_counts(
        br.counts.begin(), br.counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& [bits, count] : sorted_counts) {
        const int total = static_cast<int>(bits.size());
        // 1ULL prevents int overflow when n_eval > 30.
        uint64_t m = 0;
        for (int q = 0; q < n_eval && q < total; ++q) {
            if (bits[total - 1 - q] == '1') m |= (1ULL << q);
        }
        if (m == 0) continue;

        const double phase = static_cast<double>(m) /
                             static_cast<double>(1ULL << n_eval);
        auto convs = cf_convergents(phase, N);
        for (const auto& [s, r] : convs) {
            if (r == 0 || r >= N) continue;
            if (mod_pow(a, r, N) == 1) return r;
        }
    }
    return 0;
}

// =============================================================================
// factorize
// =============================================================================

Shor::Result Shor::factorize(uint64_t N) const {
    if (N < 4)
        throw std::invalid_argument("Shor::factorize: N must be >= 4");
    if (is_prime(N))
        throw std::invalid_argument(
            "Shor::factorize: N is prime — cannot factor a prime number");

    // Trivial: even
    if (N % 2 == 0)
        return {2, N / 2, true, 0, "trivial_gcd"};

    // Perfect power
    uint64_t pw_base = 0;
    if (is_perfect_power(N, pw_base))
        return {pw_base, N / pw_base, true, 0, "perfect_power"};

    // Small trial GCDs (catch small factors quickly without QPE)
    for (uint64_t a : {3ULL, 5ULL, 7ULL, 11ULL, 13ULL}) {
        uint64_t g = gcd_u64(a, N);
        if (g > 1 && g < N) return {g, N / g, true, 0, "trivial_gcd"};
    }

    // Quantum order finding
    const int n_target = static_cast<int>(
        std::ceil(std::log2(static_cast<double>(N) + 1.0)));
    const int n_eval = options.n_eval_qubits > 0
                       ? options.n_eval_qubits
                       : 2 * n_target + 1;

    backends::LocalBackend::Config cfg;
    cfg.simulator = options.simulator;
    backends::LocalBackend backend(cfg);

    std::mt19937_64 rng(options.seed == 0
                        ? static_cast<uint64_t>(std::random_device{}())
                        : options.seed);
    std::uniform_int_distribution<uint64_t> dist(2, N - 2);

    for (int attempt = 1; attempt <= options.max_attempts; ++attempt) {
        uint64_t a  = dist(rng);
        uint64_t g  = gcd_u64(a, N);
        if (g > 1) return {g, N / g, true, attempt, "trivial_gcd"};

        const uint64_t run_seed = (options.seed == 0)
                                  ? rng()
                                  : options.seed + static_cast<uint64_t>(attempt);
        uint64_t r = find_order(a, N, n_eval, backend, run_seed);
        if (r == 0 || r % 2 != 0) continue;

        uint64_t half = mod_pow(a, r / 2, N);
        if (half == N - 1) continue;  // a^(r/2) ≡ -1 (mod N)

        if (half > 0) {
            uint64_t p = gcd_u64(half - 1, N);
            if (p > 1 && p < N) return {p, N / p, true, attempt, "quantum"};
        }
        uint64_t q = gcd_u64(half + 1, N);
        if (q > 1 && q < N) return {q, N / q, true, attempt, "quantum"};
    }

    return {0, 0, false, options.max_attempts, "quantum"};
}

} // namespace algorithms
} // namespace lindblad
