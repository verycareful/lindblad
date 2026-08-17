#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace lindblad {
namespace algorithms {

QuantumCircuit Simon::build_circuit(const QuantumCircuit& oracle, int n) {
    QuantumCircuit qc(2 * n, n);
    for (int i = 0; i < n; ++i) qc.h(i);
    for (const auto& inst : oracle.instructions) qc.instructions.push_back(inst);
    for (int i = 0; i < n; ++i) qc.h(i);
    for (int i = 0; i < n; ++i) qc.measure(i, i);
    return qc;
}

std::string Simon::gaussian_eliminate(const std::vector<std::string>& eqs, int n) {
    std::vector<std::vector<int>> M(eqs.size(), std::vector<int>(n + 1, 0));
    for (int i = 0; i < (int)eqs.size(); ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = eqs[i][j] - '0';

    int row = 0;
    std::vector<int> pivot_col(n, -1);
    for (int col = 0; col < n && row < (int)M.size(); ++col) {
        int piv = -1;
        for (int r = row; r < (int)M.size(); ++r)
            if (M[r][col]) { piv = r; break; }
        if (piv < 0) continue;
        std::swap(M[row], M[piv]);
        pivot_col[col] = row;
        for (int r = 0; r < (int)M.size(); ++r)
            if (r != row && M[r][col])
                for (int c = 0; c <= n; ++c)
                    M[r][c] ^= M[row][c];
        ++row;
    }

    std::vector<int> s(n, 0);
    for (int col = n - 1; col >= 0; --col) {
        if (pivot_col[col] < 0) {
            s[col] = 1;
        } else {
            int r = pivot_col[col];
            int val = 0;
            for (int c = col + 1; c < n; ++c)
                val ^= M[r][c] & s[c];
            s[col] = val & 1;
        }
    }

    std::string result(n, '0');
    for (int i = 0; i < n; ++i) result[i] = '0' + s[i];
    return result;
}

Simon::Result Simon::solve(const QuantumCircuit& oracle, int n,
                            uint64_t seed, int extra_samples, bool batch_shots) {
    auto qc = build_circuit(oracle, n);
    StatevectorSimulator sim;

    std::vector<std::string> equations;

    if (batch_shots) {
        // One batched simulation: the Simon circuit's measurements
        // are terminal, so a single run samples all shots from one forward pass.
        // Harvest the DISTINCT non-zero equations; a std::set keeps the order
        // deterministic (independent of the counts map's iteration order), so a
        // given seed is reproducible run to run.
        const int shots = std::max(1, (n + extra_samples) * 8);
        auto res = sim.run(qc, shots, seed);
        std::set<std::string> seen;
        for (const auto& [raw, cnt] : res.counts) {
            (void)cnt;
            std::string y = raw;                  // query register, MSB-first
            std::reverse(y.begin(), y.end());     // y[j] = qubit j
            if (y == std::string(n, '0')) continue;
            seen.insert(y);
        }
        equations.assign(seen.begin(), seen.end());
    } else {
        // Per-sample loop: one single-shot simulation per equation. Consumes
        // the RNG stream in a different order from the batched path, so a given
        // seed yields a different (statistically equivalent) equation stream.
        std::mt19937_64 rng(seed);
        for (int attempt = 0;
             attempt < n * 4 && (int)equations.size() < n - 1 + extra_samples;
             ++attempt) {
            auto res = sim.run(qc, 1, rng());
            std::string raw = res.counts.begin()->first;
            std::string y = raw;
            std::reverse(y.begin(), y.end());
            if (y == std::string(n, '0')) continue;
            bool dup = false;
            for (auto& eq : equations) if (eq == y) { dup = true; break; }
            if (!dup) equations.push_back(y);
        }
    }

    std::string period = gaussian_eliminate(equations, n);
    return { period, equations };
}

// =============================================================================
// QuditSimon
// =============================================================================

bool QuditSimon::is_prime(int d) {
    if (d < 2) return false;
    if (d == 2) return true;
    if (d % 2 == 0) return false;
    for (int i = 3; i * i <= d; i += 2)
        if (d % i == 0) return false;
    return true;
}

int QuditSimon::mod_inv(int a, int p) {
    // Fermat's little theorem: a^{p-2} mod p (p prime, a != 0)
    int result = 1;
    a = ((a % p) + p) % p;
    int exp = p - 2;
    while (exp > 0) {
        if (exp % 2 == 1) result = (result * a) % p;
        a = (a * a) % p;
        exp /= 2;
    }
    return result;
}

std::vector<std::vector<int>> QuditSimon::null_space_gf(
    std::vector<std::vector<int>> M, int n, int d)
{
    const int rows = static_cast<int>(M.size());
    std::vector<int> pivot_col_for_row(static_cast<size_t>(rows), -1);
    std::vector<bool> col_has_pivot(static_cast<size_t>(n), false);

    int r = 0;
    for (int c = 0; c < n && r < rows; ++c) {
        // Find pivot in column c at or below row r
        int piv = -1;
        for (int i = r; i < rows; ++i)
            if (M[static_cast<size_t>(i)][static_cast<size_t>(c)] != 0) {
                piv = i; break;
            }
        if (piv < 0) continue;

        std::swap(M[static_cast<size_t>(r)], M[static_cast<size_t>(piv)]);
        pivot_col_for_row[static_cast<size_t>(r)] = c;
        col_has_pivot[static_cast<size_t>(c)] = true;

        // Scale row r so leading entry = 1 (mod d)
        const int inv = mod_inv(M[static_cast<size_t>(r)][static_cast<size_t>(c)], d);
        for (int j = 0; j < n; ++j)
            M[static_cast<size_t>(r)][static_cast<size_t>(j)] =
                (M[static_cast<size_t>(r)][static_cast<size_t>(j)] * inv) % d;

        // Eliminate column c in all other rows
        for (int i = 0; i < rows; ++i) {
            if (i == r || M[static_cast<size_t>(i)][static_cast<size_t>(c)] == 0)
                continue;
            const int factor = M[static_cast<size_t>(i)][static_cast<size_t>(c)];
            for (int j = 0; j < n; ++j)
                M[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    ((M[static_cast<size_t>(i)][static_cast<size_t>(j)]
                      - factor * M[static_cast<size_t>(r)][static_cast<size_t>(j)]) % d + d) % d;
        }
        ++r;
    }
    const int rank = r;

    // Build null space: one vector per free variable (column without pivot)
    std::vector<std::vector<int>> null_vecs;
    for (int fc = 0; fc < n; ++fc) {
        if (col_has_pivot[static_cast<size_t>(fc)]) continue;

        // Set free variable fc = 1, other free variables = 0; back-substitute pivots
        std::vector<int> v(static_cast<size_t>(n), 0);
        v[static_cast<size_t>(fc)] = 1;

        for (int ri = 0; ri < rank; ++ri) {
            const int pc = pivot_col_for_row[static_cast<size_t>(ri)];
            if (pc < 0) continue;
            // pivot row ri: v[pc] = −M[ri][fc] * v[fc] mod d  (other free vars = 0)
            v[static_cast<size_t>(pc)] =
                ((-M[static_cast<size_t>(ri)][static_cast<size_t>(fc)]) % d + d) % d;
        }
        null_vecs.push_back(v);
    }
    return null_vecs;
}

QuditSimon::Result QuditSimon::solve(
    int n, int d,
    const std::function<std::vector<int>(const std::vector<int>&)>& f,
    int extra_samples, uint64_t seed,
    QuditBackend backend,
    const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument("QuditSimon::solve: d must be >= 2");
    if (n < 1)
        throw std::invalid_argument("QuditSimon::solve: n must be >= 1");

    if (backend == QuditBackend::CLIFFORD)
        throw std::invalid_argument(
            "QuditSimon::solve: CLIFFORD backend does not support a black-box "
            "std::function oracle (no Clifford decomposition exists). Use the "
            "QuditAffineOracle overload for a Clifford-decomposable oracle.");

    const auto F  = qudit_gates::qft_matrix(d);
    const auto Fd = qudit_gates::iqft_matrix(d);

    std::mt19937_64 rng(seed == 0
        ? static_cast<uint64_t>(std::random_device{}())
        : seed);

    const int target_samples = n - 1 + extra_samples;
    const int max_attempts   = (n + extra_samples) * 6;

    auto validated_f = [&](const std::vector<int>& x) -> std::vector<int> {
        auto fx = f(x);
        if (static_cast<int>(fx.size()) != n)
            throw std::invalid_argument("QuditSimon::solve: f returned wrong size vector");
        for (int v : fx)
            if (v < 0 || v >= d)
                throw std::invalid_argument("QuditSimon::solve: f returned digit out of [0, d)");
        return fx;
    };

    std::vector<std::vector<int>> equations;
    int queries = 0;

    // ── DENSITY_MATRIX path ──────────────────────────────────────────────────
    if (backend == QuditBackend::DENSITY_MATRIX) {
        for (int attempt = 0;
             attempt < max_attempts && static_cast<int>(equations.size()) < target_samples;
             ++attempt)
        {
            QuditDensityMatrix dm(2 * n, d);
            for (int i = 0; i < n; ++i) dm.apply_1qudit(i, F);
            dm.apply_function_oracle(n, n, validated_f);
            for (int i = 0; i < n; ++i) dm.apply_1qudit(i, Fd);
            if (noise) dm.apply_noise(*noise);
            const auto outcome = dm.measure(rng());
            ++queries;

            std::vector<int> y(outcome.begin(), outcome.begin() + n);
            bool is_zero = true;
            for (int v : y) if (v != 0) { is_zero = false; break; }
            if (is_zero) continue;
            bool dup = false;
            for (const auto& eq : equations) if (eq == y) { dup = true; break; }
            if (!dup) equations.push_back(y);
        }
    }
    // ── MPS path ─────────────────────────────────────────────────────────────
    else if (backend == QuditBackend::MPS) {
        for (int attempt = 0;
             attempt < max_attempts && static_cast<int>(equations.size()) < target_samples;
             ++attempt)
        {
            QuditMPS mps(2 * n, d);
            for (int i = 0; i < n; ++i) mps.apply_1qudit(i, F);
            // MPS function oracle: takes flat query index, returns flat addend
            mps.apply_function_oracle(n, n,
                [&](int x_flat) -> int {
                    auto digits = QuditStatevector::index_to_digits(
                        static_cast<size_t>(x_flat), d, n);
                    auto out = validated_f(digits);
                    return static_cast<int>(QuditStatevector::digits_to_index(out, d));
                });
            for (int i = 0; i < n; ++i) mps.apply_1qudit(i, Fd);
            const auto outcome = mps.measure(rng());
            ++queries;

            std::vector<int> y(outcome.begin(), outcome.begin() + n);
            bool is_zero = true;
            for (int v : y) if (v != 0) { is_zero = false; break; }
            if (is_zero) continue;
            bool dup = false;
            for (const auto& eq : equations) if (eq == y) { dup = true; break; }
            if (!dup) equations.push_back(y);
        }
    }
    // ── STATEVECTOR path (default) ───────────────────────────────────────────
    else {
        for (int attempt = 0;
             attempt < max_attempts && static_cast<int>(equations.size()) < target_samples;
             ++attempt)
        {
            QuditStatevector sv(2 * n, d);
            for (int i = 0; i < n; ++i) sv.apply_1qudit(i, F);
            sv.apply_function_oracle(n, n, validated_f);
            for (int i = 0; i < n; ++i) sv.apply_1qudit(i, Fd);
            const auto outcome = sv.measure(rng());
            ++queries;

            std::vector<int> y(outcome.begin(), outcome.begin() + n);
            bool is_zero = true;
            for (int v : y) if (v != 0) { is_zero = false; break; }
            if (is_zero) continue;
            bool dup = false;
            for (const auto& eq : equations) if (eq == y) { dup = true; break; }
            if (!dup) equations.push_back(y);
        }
    }

    // Period verifier (used only for composite d): a nonzero s is a true period
    // iff f(x) = f((x + s) mod d) for all x. Sample x = 0 plus a few random points.
    auto is_period = [&](const std::vector<int>& s) -> bool {
        bool nonzero = false;
        for (int v : s) if (v != 0) { nonzero = true; break; }
        if (!nonzero) return false;

        std::mt19937_64 vrng(seed ^ 0x9e3779b97f4a7c15ULL);
        std::uniform_int_distribution<int> dist(0, d - 1);
        auto agrees = [&](const std::vector<int>& x) -> bool {
            std::vector<int> xs(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
                xs[static_cast<size_t>(i)] = (x[static_cast<size_t>(i)] + s[static_cast<size_t>(i)]) % d;
            return validated_f(x) == validated_f(xs);
        };
        std::vector<int> x(static_cast<size_t>(n), 0);
        if (!agrees(x)) return false;               // x = 0
        for (int trial = 0; trial < 8; ++trial) {
            for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = dist(vrng);
            if (!agrees(x)) return false;
        }
        return true;
    };

    return post_process(equations, n, d, queries, is_period);
}

// =============================================================================
// QuditSimon — shared post-processing (equations → Result)
//
// prime d:     field Gaussian elimination (null_space_gf); the first kernel
//              vector is the period generator.
// composite d: direct search over Z_d^n for a vector that annihilates every
//              measured y (mod d) and is a verified period of the oracle. Correct
//              for the ring Z_d and always feasible (d^n << the d^{2n} the sim
//              already paid for). Returns the zero vector (trivial) when f is
//              injective.
// =============================================================================

QuditSimon::Result QuditSimon::post_process(
    const std::vector<std::vector<int>>& equations,
    int n, int d, int queries,
    const std::function<bool(const std::vector<int>&)>& is_period)
{
    const std::vector<int> zero(static_cast<size_t>(n), 0);

    if (is_prime(d)) {
        const std::vector<std::vector<int>> null_vecs = null_space_gf(equations, n, d);
        if (null_vecs.empty())
            return Result{zero, true, d, n, queries};
        const std::vector<int>& period = null_vecs[0];
        bool trivial = true;
        for (int v : period) if (v != 0) { trivial = false; break; }
        return Result{period, trivial, d, n, queries};
    }

    // ── composite d ──────────────────────────────────────────────────────────
    // Z_d is a ring, so field Gaussian elimination does not apply. Find a nonzero
    // period by a direct search over Z_d^n: s must satisfy y·s ≡ 0 (mod d) for every
    // measured y AND be a true period of the oracle (is_period). This is provably
    // correct and always feasible — the quantum simulation already materialised
    // d^{2n} amplitudes, so enumerating the d^n candidates is cheap by comparison.
    // (Reading the kernel off an integer Smith Normal Form, as an earlier revision
    // did, could emit spurious non-kernel vectors for composite d.)
    long long total = 1;
    for (int i = 0; i < n; ++i) {
        total *= d;
        if (total > (1LL << 30))   // d^n this large implies an unrunnable d^{2n} sim
            throw std::runtime_error(
                "QuditSimon::post_process: composite-d period search space too large");
    }

    auto annihilates_all = [&](const std::vector<int>& s) {
        for (const auto& y : equations) {
            long long dot = 0;
            for (int i = 0; i < n; ++i)
                dot += static_cast<long long>(y[static_cast<size_t>(i)]) * s[static_cast<size_t>(i)];
            if (dot % d != 0) return false;
        }
        return true;
    };

    // Odometer over Z_d^n (little-endian), skipping s = 0. Return the first vector
    // that annihilates every measured y and is a verified period of the oracle.
    std::vector<int> s(static_cast<size_t>(n), 0);
    for (long long code = 1; code < total; ++code) {
        long long c = code;
        for (int i = 0; i < n; ++i) { s[static_cast<size_t>(i)] = static_cast<int>(c % d); c /= d; }
        if (annihilates_all(s) && is_period(s))
            return Result{s, false, d, n, queries};
    }
    return Result{zero, true, d, n, queries};   // injective f: no nonzero period
}

// =============================================================================
// QuditSimon — structured (affine) oracle overload
//
// f(x) = A·x + b (mod d), A square n×n. Hidden subgroup H = ker_{Z_d}(A); b is
// irrelevant to the period. The CLIFFORD backend (prime d) simulates the linear
// oracle directly on the stabilizer tableau (CSUM powers + X^b per output); all
// other backends materialise f and delegate to the function-oracle overload.
// =============================================================================

QuditSimon::Result QuditSimon::solve(
    const QuditAffineOracle& oracle, int d,
    int extra_samples, uint64_t seed,
    QuditBackend backend, const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument("QuditSimon::solve: d must be >= 2");
    const int n = oracle.num_inputs();
    if (oracle.num_outputs() != n)
        throw std::invalid_argument(
            "QuditSimon::solve: affine oracle must be square (num_outputs == num_inputs)");
    if (n < 1)
        throw std::invalid_argument("QuditSimon::solve: n must be >= 1");
    if (static_cast<int>(oracle.b.size()) != n)
        throw std::invalid_argument("QuditSimon::solve: affine oracle b must have n entries");
    for (const auto& row : oracle.A) {
        if (static_cast<int>(row.size()) != n)
            throw std::invalid_argument("QuditSimon::solve: affine oracle A must be n x n");
        for (int v : row)
            if (v < 0 || v >= d)
                throw std::invalid_argument("QuditSimon::solve: affine coefficient out of [0, d)");
    }
    for (int v : oracle.b)
        if (v < 0 || v >= d)
            throw std::invalid_argument("QuditSimon::solve: affine constant out of [0, d)");

    // Period verifier (exact for affine f): s is a period iff A·s ≡ 0 (mod d).
    auto is_period = [&](const std::vector<int>& s) -> bool {
        bool nonzero = false;
        for (int v : s) if (v != 0) { nonzero = true; break; }
        if (!nonzero) return false;
        for (int j = 0; j < n; ++j) {
            long long acc = 0;
            for (int i = 0; i < n; ++i)
                acc += static_cast<long long>(oracle.A[static_cast<size_t>(j)][static_cast<size_t>(i)])
                     * s[static_cast<size_t>(i)];
            if (acc % d != 0) return false;
        }
        return true;
    };

    if (backend == QuditBackend::CLIFFORD) {
        if (!is_prime(d))
            throw std::invalid_argument(
                "QuditSimon::solve: CLIFFORD backend requires prime d");

        std::mt19937_64 rng(seed == 0
            ? static_cast<uint64_t>(std::random_device{}())
            : seed);
        const int target_samples = n - 1 + extra_samples;
        const int max_attempts   = (n + extra_samples) * 6;

        std::vector<std::vector<int>> equations;
        int queries = 0;
        for (int attempt = 0;
             attempt < max_attempts && static_cast<int>(equations.size()) < target_samples;
             ++attempt)
        {
            QuditCliffordSimulator c(2 * n, d);
            for (int i = 0; i < n; ++i) c.apply_H(i);              // F_d on query register
            // U_f: output_j += A[j][i]·x_i (+ b_j); CADD(k) = CSUM applied k times.
            for (int j = 0; j < n; ++j) {
                if (oracle.b[static_cast<size_t>(j)] != 0)
                    c.apply_X(n + j, oracle.b[static_cast<size_t>(j)]);
                for (int i = 0; i < n; ++i)
                    for (int rep = 0; rep < oracle.A[static_cast<size_t>(j)][static_cast<size_t>(i)]; ++rep)
                        c.apply_CSUM(i, n + j);
            }
            for (int i = 0; i < n; ++i) { c.apply_H(i); c.apply_H(i); c.apply_H(i); }  // F_d†

            // Joint measurement of the query register (sequential collapse is correct).
            const uint64_t s0 = rng();
            std::vector<int> y(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
                y[static_cast<size_t>(i)] = c.measure_qudit(i, s0 + static_cast<uint64_t>(i));
            ++queries;

            bool is_zero = true;
            for (int v : y) if (v != 0) { is_zero = false; break; }
            if (is_zero) continue;
            bool dup = false;
            for (const auto& eq : equations) if (eq == y) { dup = true; break; }
            if (!dup) equations.push_back(std::move(y));
        }
        return post_process(equations, n, d, queries, is_period);
    }

    // Non-Clifford backends (any d): materialise f and delegate.
    auto affine_f = [&oracle, d](const std::vector<int>& x) -> std::vector<int> {
        return oracle.eval(x, d);
    };
    return solve(n, d, affine_f, extra_samples, seed, backend, noise);
}

} // namespace algorithms
} // namespace lindblad
