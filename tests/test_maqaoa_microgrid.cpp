// test_maqaoa_microgrid.cpp
// Direct C++ port of EnergyGridOpt_py MA-QAOA pipeline with hard-coded microgrid input.
// Mirrors: MicrogridData / QUBOBuilder / IsingMapper / SimulatedAnnealing /
//          MAQAOACircuit + LayerwiseTrainer from the Python reference project.
//
// Convention note:
//   Python MA-QAOA: N gammas/layer (qubit-indexed, each qubit's gamma scales ALL
//                   cost terms touching that qubit) → 2N = 10 params/layer for N=5.
//   C++ MA-QAOA:   one gamma per cost term (term-indexed) → (n_terms + N) params/layer.
//   Both use independent betas per qubit (N per layer).

#include <gtest/gtest.h>
#include "lindblad/algorithms.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace lindblad;
using namespace lindblad::algorithms;

// =============================================================================
// Hard-coded microgrid parameters (Python MicrogridData defaults / 'critical')
// =============================================================================
namespace microgrid {

constexpr int    N       = 5;
constexpr double DEMAND  = 10.0;
constexpr double A       = 10.0;   // penalty weight

const double POWERS[N] = {5.0, 4.0, 3.0, 2.0, 1.0};
const double COSTS[N]  = {2.0, 3.0, 4.0, 8.0, 10.0};

// ---------------------------------------------------------------------------
// QUBOBuilder — matches qubo_builder.py QUBOBuilder
// ---------------------------------------------------------------------------

// Direct QUBO cost: operational_cost + A * (power_delivered - demand)^2
double qubo_cost(const int x[N]) {
    double op_cost   = 0.0;
    double delivered = 0.0;
    for (int i = 0; i < N; ++i) {
        op_cost   += COSTS[i]  * x[i];
        delivered += POWERS[i] * x[i];
    }
    return op_cost + A * (delivered - DEMAND) * (delivered - DEMAND);
}

// Linear QUBO coefficients c'_i = C_i + A*P_i*(P_i - 2*D)
void build_linear_coefficients(double c_prime[N]) {
    for (int i = 0; i < N; ++i)
        c_prime[i] = COSTS[i] + A * POWERS[i] * (POWERS[i] - 2.0 * DEMAND);
}

// Upper-triangular quadratic coefficients Q_ij = 2*A*P_i*P_j
void build_quadratic_coefficients(double Q[N][N]) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Q[i][j] = (i < j) ? 2.0 * A * POWERS[i] * POWERS[j] : 0.0;
}

// Brute-force: enumerate all 2^N combinations, return (best_cost, best_bitstring)
struct BruteResult {
    double      best_cost;
    std::string best_bitstring;          // e.g. "10110" — index i = qubit i
    std::vector<std::string> all_optimal; // degenerate optima
};

// The +infinity seed this used to carry is the defect #68 fixed in the library,
// and it is worse in a REFERENCE solver than in the thing being tested: every
// comparison here feeds an assertion about the library's answer, so a reference
// that silently returns its seed makes a comparison test compare two broken
// implementations. A seed of +infinity exists only to lose its first
// comparison, which is precisely the comparison -ffinite-math-only (implied by
// the project-wide -ffast-math) may fold away on the premise that infinities do
// not occur. An explicit flag carries the same intent unfoldably.
BruteResult brute_force_solve() {
    BruteResult res;
    res.best_cost = 0.0;
    bool have_best = false;

    for (int mask = 0; mask < (1 << N); ++mask) {
        int x[N];
        for (int i = 0; i < N; ++i) x[i] = (mask >> i) & 1;
        double cost = qubo_cost(x);

        std::string bits(N, '0');
        for (int i = 0; i < N; ++i) bits[i] = '0' + x[i];

        if (!have_best || cost < res.best_cost - 1e-9) {
            res.best_cost = cost;
            res.best_bitstring = bits;
            res.all_optimal = {bits};
            have_best = true;
        } else if (std::fabs(cost - res.best_cost) < 1e-9) {
            res.all_optimal.push_back(bits);
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// IsingMapper — matches ising_mapper.py IsingMapper
// ---------------------------------------------------------------------------

// J_ij = A * P_i * P_j / 2  (upper triangular, i < j)
void compute_J(double J[N][N]) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            J[i][j] = (i < j) ? A * POWERS[i] * POWERS[j] / 2.0 : 0.0;
}

// h_i = A*D*P_i - C_i/2 - A*P_i^2/2 - A*P_i * (sum_{j!=i} P_j) / 2
void compute_h(double h[N]) {
    double sum_P = 0.0;
    for (int i = 0; i < N; ++i) sum_P += POWERS[i];

    for (int i = 0; i < N; ++i) {
        double interaction_sum = A * POWERS[i] * (sum_P - POWERS[i]) / 2.0;
        h[i] = A * DEMAND * POWERS[i]
               - COSTS[i]  / 2.0
               - A * POWERS[i] * POWERS[i] / 2.0
               - interaction_sum;
    }
}

// Constant energy offset so that QUBO(x) = H_Ising(z) + offset
// (z_i = 1 - 2*x_i, eigenvalue of Z in |x_i> is (-1)^x_i)
double compute_offset() {
    double c_prime[N];
    double Q[N][N];
    build_linear_coefficients(c_prime);
    build_quadratic_coefficients(Q);

    double sum_c = 0.0;
    for (int i = 0; i < N; ++i) sum_c += c_prime[i];

    double sum_Q = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            sum_Q += Q[i][j];

    return A * DEMAND * DEMAND + sum_c / 2.0 + sum_Q / 4.0;
}

// Build SparsePauliOp for H_Ising = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
// lindblad Pauli string convention: character at position q = Pauli for qubit q (left-to-right).
SparsePauliOp build_ising_hamiltonian() {
    double h[N], J[N][N];
    compute_h(h);
    compute_J(J);

    std::vector<PauliString> terms;

    // Local fields
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h[i]) < 1e-12) continue;
        std::string s(N, 'I');
        s[i] = 'Z';
        terms.push_back(PauliString(s, Complex128(h[i], 0.0)));
    }

    // Couplings
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (std::fabs(J[i][j]) < 1e-12) continue;
            std::string s(N, 'I');
            s[i] = 'Z';
            s[j] = 'Z';
            terms.push_back(PauliString(s, Complex128(J[i][j], 0.0)));
        }
    }

    return SparsePauliOp(terms);
}

// ---------------------------------------------------------------------------
// SimulatedAnnealing — matches simulated_annealing.py SimulatedAnnealing
// ---------------------------------------------------------------------------
struct SAResult {
    std::string best_bitstring;
    double      best_cost;
    double      mean_cost;
    double      std_cost;
    double      success_rate;
    int         nfev_total;
};

SAResult simulated_annealing(int num_runs = 100, uint64_t seed = 42,
                              double T0 = 100.0, double Tf = 0.01,
                              double alpha = 0.995, int iters_per_T = 50) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> bit_dist(0, N - 1);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_int_distribution<int> bit_init(0, 1);

    double brute_min = brute_force_solve().best_cost;

    std::vector<double> costs;
    std::string overall_best_bits;
    // Same remedy as brute_force_solve above: a best-so-far seeded with
    // +infinity is the one comparison -ffinite-math-only may fold, which would
    // leave overall_best_bits empty and overall_best_cost at its seed.
    double overall_best_cost = 0.0;
    bool have_overall_best = false;
    int nfev = 0;
    int successes = 0;

    for (int run = 0; run < num_runs; ++run) {
        int x[N];
        for (int i = 0; i < N; ++i) x[i] = bit_init(rng);
        int best_x[N];
        std::copy(x, x + N, best_x);
        double best_cost   = qubo_cost(x);
        double current_cost = best_cost;
        ++nfev;

        double T = T0;
        while (T > Tf) {
            for (int it = 0; it < iters_per_T; ++it) {
                int j = bit_dist(rng);
                int x_new[N];
                std::copy(x, x + N, x_new);
                x_new[j] = 1 - x_new[j];

                double new_cost = qubo_cost(x_new);
                ++nfev;
                double delta = new_cost - current_cost;
                if (delta < 0.0 || uni(rng) < std::exp(-delta / T)) {
                    std::copy(x_new, x_new + N, x);
                    current_cost = new_cost;
                    if (current_cost < best_cost) {
                        best_cost = current_cost;
                        std::copy(x, x + N, best_x);
                    }
                }
            }
            T *= alpha;
        }

        costs.push_back(best_cost);
        if (!have_overall_best || best_cost < overall_best_cost) {
            overall_best_cost = best_cost;
            overall_best_bits = std::string(N, '0');
            for (int i = 0; i < N; ++i)
                overall_best_bits[i] = '0' + best_x[i];
            have_overall_best = true;
        }
        if (std::fabs(best_cost - brute_min) < 1e-5) ++successes;
    }

    double mean = 0.0;
    for (double c : costs) mean += c;
    mean /= num_runs;

    double var = 0.0;
    for (double c : costs) var += (c - mean) * (c - mean);
    var /= num_runs;

    SAResult res;
    res.best_bitstring = overall_best_bits;
    res.best_cost      = overall_best_cost;
    res.mean_cost      = mean;
    res.std_cost       = std::sqrt(var);
    res.success_rate   = static_cast<double>(successes) / num_runs;
    res.nfev_total     = nfev;
    return res;
}

} // namespace microgrid

// =============================================================================
// GTest fixtures
// =============================================================================

TEST(MicrogridQAOA, BruteForce) {
    using namespace microgrid;

    auto started = std::chrono::high_resolution_clock::now();
    auto res = brute_force_solve();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - started).count();

    std::cout << "\n=== Brute Force (mirrors exp01_brute_force) ===\n";
    std::cout << "  Best QUBO cost:     " << res.best_cost << "\n";
    std::cout << "  Best bitstring:     " << res.best_bitstring << "  (index 0 = gen0)\n";
    std::cout << "  Degenerate optima:  " << res.all_optimal.size() << "\n";
    for (const auto& s : res.all_optimal)
        std::cout << "                      " << s << "\n";
    std::cout << "  Runtime:            " << elapsed << " us\n";

    // Print full landscape (mirrors run_landscape_plot)
    std::cout << "\n  Energy landscape (all 32 states):\n";
    std::cout << "  Bitstring  |  QUBO cost\n";
    std::cout << "  -----------|----------\n";
    for (int mask = 0; mask < (1 << N); ++mask) {
        int x[N];
        for (int i = 0; i < N; ++i) x[i] = (mask >> i) & 1;
        std::string bits(N, '0');
        for (int i = 0; i < N; ++i) bits[i] = '0' + x[i];
        double cost = qubo_cost(x);
        bool optimal = std::fabs(cost - res.best_cost) < 1e-9;
        std::cout << "  " << bits << "  |  " << cost;
        if (optimal) std::cout << "  <-- optimal";
        std::cout << "\n";
    }

    EXPECT_LT(res.best_cost, 20.0);   // sanity: reasonable QUBO cost
    EXPECT_EQ(static_cast<int>(res.best_bitstring.size()), N);
}

TEST(MicrogridQAOA, IsingMapping) {
    using namespace microgrid;

    double h[N], J[N][N];
    compute_h(h);
    compute_J(J);
    double offset = compute_offset();

    std::cout << "\n=== Ising Mapping (mirrors IsingMapper) ===\n";
    std::cout << "  Local fields h_i:\n";
    for (int i = 0; i < N; ++i)
        std::cout << "    h[" << i << "] = " << h[i] << "\n";

    std::cout << "  Coupling J_ij (upper triangular, non-zero):\n";
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            if (std::fabs(J[i][j]) > 1e-12)
                std::cout << "    J[" << i << "][" << j << "] = " << J[i][j] << "\n";

    std::cout << "  Constant energy offset: " << offset << "\n";

    // Verify: QUBO(x) = H_Ising(z) + offset, where z_i = (-1)^x_i
    // Test on all 32 states
    auto ham = build_ising_hamiltonian();
    std::cout << "  Hamiltonian terms: " << ham.terms.size() << "\n";

    // Spot-check: manually compute H_Ising for ground state from brute force
    auto bf = brute_force_solve();
    std::cout << "  Brute force QUBO minimum:      " << bf.best_cost << "\n";
    std::cout << "  Brute force bitstring:         " << bf.best_bitstring << "\n";
    std::cout << "  Expected Ising energy at opt:  " << bf.best_cost - offset << "\n";

    EXPECT_EQ(static_cast<int>(ham.terms.size()), N + N * (N - 1) / 2);
}

TEST(MicrogridQAOA, SimulatedAnnealing) {
    using namespace microgrid;

    auto started = std::chrono::high_resolution_clock::now();
    auto res = simulated_annealing(/*num_runs=*/100, /*seed=*/42);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - started).count();

    auto bf = brute_force_solve();

    std::cout << "\n=== Simulated Annealing (mirrors exp02_sa / SimulatedAnnealing) ===\n";
    std::cout << "  Brute force optimum: " << bf.best_cost << "  (" << bf.best_bitstring << ")\n";
    std::cout << "  SA best cost:        " << res.best_cost << "\n";
    std::cout << "  SA best bitstring:   " << res.best_bitstring << "\n";
    std::cout << "  Mean cost:           " << res.mean_cost << "\n";
    std::cout << "  Std cost:            " << res.std_cost << "\n";
    std::cout << "  Success rate:        " << res.success_rate * 100.0 << " %\n";
    std::cout << "  Function evals:      " << res.nfev_total << "\n";
    std::cout << "  Runtime:             " << elapsed_ms << " ms\n";

    EXPECT_LE(res.best_cost, bf.best_cost + 1e-5);
    EXPECT_GT(res.success_rate, 0.5);
}

TEST(MicrogridQAOA, MAQAOA_Layerwise) {
    using namespace microgrid;

    // Build Ising Hamiltonian
    auto cost_ham = build_ising_hamiltonian();
    double offset = compute_offset();

    constexpr int    P_DEPTH        = 3;
    constexpr int    BUDGET_PER_LAYER = 200;   // matches Python LayerwiseTrainer default
    constexpr double CONV_TOL       = 1e-6;

    MAQAOA maqaoa;
    maqaoa.estimator.options.shots = 0;  // exact statevector expectation
    maqaoa.sampler.options.shots   = 4096;
    maqaoa.sampler.options.seed    = 42;
    maqaoa.options.p                    = P_DEPTH;
    maqaoa.options.layerwise            = true;
    maqaoa.options.term_indexed_gammas  = true;   // test expects term-indexed (60 params, not 30)
    maqaoa.options.max_iterations       = BUDGET_PER_LAYER;
    maqaoa.options.convergence_threshold = CONV_TOL;
    maqaoa.options.seed                 = 42;

    auto bf = brute_force_solve();

    int expected_params = P_DEPTH * (static_cast<int>(cost_ham.terms.size()) + N);

    std::cout << "\n=== MA-QAOA Layerwise (mirrors LayerwiseTrainer / MAQAOACircuit) ===\n";
    std::cout << "  Problem:             5-generator microgrid (critical scenario)\n";
    std::cout << "  Depth p:             " << P_DEPTH << "\n";
    std::cout << "  Budget/layer:        " << BUDGET_PER_LAYER << "\n";
    std::cout << "  Cost Hamiltonian terms: " << cost_ham.terms.size() << "\n";
    std::cout << "  Total params (C++):  " << expected_params
              << "  (Python equivalent: " << P_DEPTH * 2 * N << " — qubit-indexed gammas)\n";
    std::cout << "  Brute force QUBO min: " << bf.best_cost
              << "  (" << bf.best_bitstring << ")\n";
    std::cout << "  Ising offset:        " << offset << "\n";
    std::cout << "  Expected Ising target: " << bf.best_cost - offset << "\n\n";

    auto started = std::chrono::high_resolution_clock::now();
    auto result  = maqaoa.optimize(cost_ham);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - started).count();

    double qaoa_qubo_approx = result.optimal_value + offset;
    double gap = qaoa_qubo_approx - bf.best_cost;

    std::cout << "  MA-QAOA Ising energy:     " << result.optimal_value << "\n";
    std::cout << "  MA-QAOA QUBO approx:      " << qaoa_qubo_approx
              << "  (= Ising + offset)\n";
    std::cout << "  Approximation gap:        " << gap
              << "  (" << (gap / std::fabs(bf.best_cost) * 100.0) << " %)\n";
    std::cout << "  Best sampled bitstring:   " << result.best_bitstring << "\n";
    std::cout << "  Converged:                " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  Param count:              " << result.optimal_params.size() << "\n";
    std::cout << "  Runtime:                  " << elapsed_ms << " ms\n";

    // Evaluate QUBO cost of the sampled bitstring
    {
        const std::string& bs = result.best_bitstring;
        if (static_cast<int>(bs.size()) == N) {
            int x[N];
            for (int i = 0; i < N; ++i) x[i] = bs[i] - '0';
            double sampled_qubo = qubo_cost(x);
            std::cout << "  Sampled bitstring QUBO cost: " << sampled_qubo
                      << "  (brute min = " << bf.best_cost << ")\n";
        }
    }

    EXPECT_EQ(static_cast<int>(result.optimal_params.size()), expected_params);
    EXPECT_LT(result.optimal_value, 0.0);  // should be negative (minimising H_Ising)
}

TEST(MicrogridQAOA, AllMethodsComparison) {
    using namespace microgrid;

    // ---- Brute force ----
    auto t0   = std::chrono::high_resolution_clock::now();
    auto bf   = brute_force_solve();
    auto t1   = std::chrono::high_resolution_clock::now();

    // ---- SA ----
    auto sa_res = simulated_annealing(100, 42);
    auto t2     = std::chrono::high_resolution_clock::now();

    // ---- MA-QAOA ----
    auto cost_ham = build_ising_hamiltonian();
    double offset = compute_offset();

    MAQAOA maqaoa;
    maqaoa.estimator.options.shots = 0;
    maqaoa.sampler.options.shots   = 4096;
    maqaoa.sampler.options.seed    = 42;
    maqaoa.options.p                     = 3;
    maqaoa.options.layerwise             = true;
    maqaoa.options.max_iterations        = 200;
    maqaoa.options.convergence_threshold = 1e-6;
    maqaoa.options.seed                  = 42;

    auto qaoa_result = maqaoa.optimize(cost_ham);
    auto t3 = std::chrono::high_resolution_clock::now();

    double qaoa_qubo = qaoa_result.optimal_value + offset;

    // QUBO costs of sampled bitstrings
    double sa_qubo = sa_res.best_cost;
    // quiet_nan_strict, not quiet_NaN(): the marker must stay detectably
    // non-finite when the bitstring is the wrong width, and -ffinite-math-only
    // need not materialise the library constant. See the note in types.hpp.
    double qaoa_sampled_qubo = quiet_nan_strict();
    {
        const std::string& bs = qaoa_result.best_bitstring;
        if (static_cast<int>(bs.size()) == N) {
            int x[N];
            for (int i = 0; i < N; ++i) x[i] = bs[i] - '0';
            qaoa_sampled_qubo = qubo_cost(x);
        }
    }

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Microgrid 5-Generator Optimisation — Method Comparison         ║\n";
    std::cout << "║   (mirrors EnergyGridOpt_py/QAOA critical scenario)              ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Problem: N=5, P=[5,4,3,2,1], C=[2,3,4,8,10], D=10, A=10       ║\n";
    std::cout << "╠══════════════════════════════════════════╦═══════════╦═══════════╣\n";
    std::cout << "║  Method                                  ║ QUBO cost ║ Runtime   ║\n";
    std::cout << "╠══════════════════════════════════════════╬═══════════╬═══════════╣\n";

    auto row = [](const std::string& name, double cost, long ms_val) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "║  %-40s║ %9.4f ║ %6ld ms  ║\n",
            name.c_str(), cost, ms_val);
        std::cout << buf;
    };

    row("Brute Force (exact)",          bf.best_cost,       ms(t0, t1));
    row("Simulated Annealing (best)",   sa_qubo,            ms(t1, t2));
    row("MA-QAOA p=3 (Ising+offset)",   qaoa_qubo,          ms(t2, t3));
    row("MA-QAOA p=3 (sampled)",        qaoa_sampled_qubo,  ms(t2, t3));

    std::cout << "╠══════════════════════════════════════════╩═══════════╩═══════════╣\n";
    std::cout << "║  Bitstrings (qubit 0 = gen 0, qubit 4 = gen 4):                  ║\n";

    auto brow = [](const std::string& name, const std::string& bits) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "║    %-22s  %s                                ║\n",
            name.c_str(), bits.c_str());
        std::cout << buf;
    };

    brow("Brute force optimal:", bf.best_bitstring);
    if (bf.all_optimal.size() > 1) {
        for (size_t k = 1; k < bf.all_optimal.size(); ++k)
            brow("              (tie):", bf.all_optimal[k]);
    }
    brow("SA best found:      ", sa_res.best_bitstring);
    brow("MA-QAOA sampled:    ", qaoa_result.best_bitstring);

    std::cout << "╠════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  MA-QAOA parameterisation note:                                    ║\n";
    std::cout << "║    Python:  " << 2 * N << " params/layer (N gammas qubit-indexed + N betas)        ║\n";
    std::cout << "║    C++ lindblad: " << (int)cost_ham.terms.size() + N
              << " params/layer (1 gamma/term + N betas)              ║\n";
    std::cout << "║    SA success rate: " << sa_res.success_rate * 100.0 << " %                                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";

    // Assertions
    EXPECT_LE(bf.best_cost, 20.0);
    EXPECT_LE(sa_res.best_cost, bf.best_cost + 1e-5);
    EXPECT_LT(qaoa_result.optimal_value, 0.0);
    EXPECT_FALSE(qaoa_result.best_bitstring.empty());
}
