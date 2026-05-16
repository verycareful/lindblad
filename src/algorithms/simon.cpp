#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"

#include <random>
#include <string>
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
                            uint64_t seed, int extra_samples) {
    auto qc = build_circuit(oracle, n);
    StatevectorSimulator sim;

    std::vector<std::string> equations;
    std::mt19937_64 rng(seed);

    for (int attempt = 0;
         attempt < n * 4 && (int)equations.size() < n - 1 + extra_samples;
         ++attempt) {
        auto res = sim.run(qc, 1, rng());
        std::string raw = res.counts.begin()->first; // n chars, MSB-first (only query qubits 0..n-1 measured)
        std::string y = raw;                          // already the query register
        std::reverse(y.begin(), y.end());            // y[j] = qubit j
        if (y == std::string(n, '0')) continue;
        bool dup = false;
        for (auto& eq : equations) if (eq == y) { dup = true; break; }
        if (!dup) equations.push_back(y);
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
    if (!is_prime(d))
        throw std::invalid_argument(
            "QuditSimon::solve: d must be prime for GF(d) post-processing");
    if (n < 1)
        throw std::invalid_argument("QuditSimon::solve: n must be >= 1");

    if (backend == QuditBackend::CLIFFORD)
        throw std::invalid_argument(
            "QuditSimon::solve: CLIFFORD backend is not supported "
            "(Simon's oracle is a general function oracle, not a Clifford circuit)");

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

    const std::vector<std::vector<int>> null_vecs = null_space_gf(equations, n, d);
    if (null_vecs.empty())
        return Result{std::vector<int>(static_cast<size_t>(n), 0), true, d, n, queries};

    const std::vector<int>& period = null_vecs[0];
    bool trivial = true;
    for (int v : period) if (v != 0) { trivial = false; break; }
    return Result{period, trivial, d, n, queries};
}

} // namespace algorithms
} // namespace lindblad
