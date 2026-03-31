// test_maqaoa_20qubit.cpp
// C++ benchmark for the 20-generator microgrid problem.
// Mirrors EnergyGridOpt_py/MA QAOA/EnergyGridOptimisation_20qubit
// Scenario: critical_tight_A (demand=52 MW, A=20, generators from generators.csv)
//
// Uses the existing MAQAOA class (qubit-indexed gammas — N gammas per layer,
// matching the Python baseline). Term-indexed path available via term_indexed_gammas=true.
// convention (one gamma per qubit).
//
// To save output:
//   ./tests/qpp_tests --gtest_filter=MicrogridQAOA20.* 2>&1 | tee /mnt/c/Sricharan/Projects/Github/q++_cp/outputs/maqaoa_20q_results.txt

#include <gtest/gtest.h>
#include "qpp/algorithms.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace qpp;
using namespace qpp::algorithms;

// =============================================================================
// Hard-coded microgrid data — generators.csv + critical_tight_A scenario
// =============================================================================
namespace mg20 {

constexpr int    N      = 20;
constexpr double DEMAND = 52.0;
constexpr double A      = 20.0;

// From generators.csv (gen01..gen20), power_mw and cost_per_mwh
const double POWER[N] = {12, 4, 8, 6, 10, 7, 5, 3, 9, 6, 4, 3, 8, 6, 5, 2, 4, 3, 2, 1};
const double COST[N]  = {1.8, 2.1, 2.5, 2.3, 2.8, 3.1, 3.4, 3.8, 3.5, 4.2,
                          3.9, 5.1, 7.5, 8.2, 6.8, 9.1, 11.5, 12.0, 13.5, 15.0};

const char* NAMES[N] = {
    "gen01(Solar 12MW)", "gen02(Solar  4MW)", "gen03(Solar  8MW)", "gen04(Solar  6MW)",
    "gen05(Wind  10MW)", "gen06(Wind   7MW)", "gen07(Wind   5MW)", "gen08(Wind   3MW)",
    "gen09(Hydro  9MW)", "gen10(Bio    6MW)", "gen11(Hydro  4MW)", "gen12(Biogas 3MW)",
    "gen13(Gas    8MW)", "gen14(Gas    6MW)", "gen15(CHP    5MW)", "gen16(Gas    2MW)",
    "gen17(Diesel 4MW)", "gen18(Diesel 3MW)", "gen19(Diesel 2MW)", "gen20(Diesel 1MW)"
};

// ---------------------------------------------------------------------------
// QUBO cost (direct) — matches qubo.py QUBOBuilder.evaluate()
// ---------------------------------------------------------------------------
double qubo_cost(const int x[N]) {
    double op = 0.0, delivered = 0.0;
    for (int i = 0; i < N; ++i) {
        op        += COST[i]  * x[i];
        delivered += POWER[i] * x[i];
    }
    return op + A * (delivered - DEMAND) * (delivered - DEMAND);
}

// ---------------------------------------------------------------------------
// Ising mapping — matches ising.py IsingMapper._build()
// ---------------------------------------------------------------------------
struct Ising {
    double h[N];
    double J[N][N];  // upper triangular
    double offset;
};

Ising build_ising() {
    Ising g{};
    for (int i = 0; i < N; ++i) {
        double coupling_sum = 0.0;
        for (int j = 0; j < N; ++j)
            if (j != i) coupling_sum += A * POWER[i] * POWER[j] / 2.0;
        g.h[i] = A * DEMAND * POWER[i] - COST[i] / 2.0
                 - A * POWER[i] * POWER[i] / 2.0 - coupling_sum;
    }
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            g.J[i][j] = A * POWER[i] * POWER[j] / 2.0;

    double sum_c = 0.0, sum_Q = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_c += COST[i] + A * POWER[i] * (POWER[i] - 2.0 * DEMAND);
        for (int j = i + 1; j < N; ++j)
            sum_Q += 2.0 * A * POWER[i] * POWER[j];
    }
    g.offset = A * DEMAND * DEMAND + sum_c / 2.0 + sum_Q / 4.0;
    return g;
}

// Build SparsePauliOp — q++ convention: string position q = qubit q
SparsePauliOp build_hamiltonian(const Ising& g) {
    std::vector<PauliString> terms;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(g.h[i]) < 1e-12) continue;
        std::string s(N, 'I'); s[i] = 'Z';
        terms.push_back(PauliString(s, Complex128(g.h[i], 0.0)));
    }
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            if (std::fabs(g.J[i][j]) < 1e-12) continue;
            std::string s(N, 'I'); s[i] = 'Z'; s[j] = 'Z';
            terms.push_back(PauliString(s, Complex128(g.J[i][j], 0.0)));
        }
    return SparsePauliOp(terms);
}

// ---------------------------------------------------------------------------
// Exact solver — enumerate all 2^20 (1M iterations, fast in C++)
// Python uses PuLP ILP for N>15; we can brute-force in C++.
// ---------------------------------------------------------------------------
struct ExactResult {
    double      best_cost;
    std::string best_bitstring;
    std::vector<std::string> all_optimal;
};

ExactResult exact_solve() {
    ExactResult res{ std::numeric_limits<double>::infinity(), "", {} };
    for (int mask = 0; mask < (1 << N); ++mask) {
        int x[N];
        for (int i = 0; i < N; ++i) x[i] = (mask >> i) & 1;
        double cost = qubo_cost(x);
        std::string bits(N, '0');
        for (int i = 0; i < N; ++i) bits[i] = '0' + x[i];
        if (cost < res.best_cost - 1e-9) {
            res.best_cost = cost; res.best_bitstring = bits; res.all_optimal = {bits};
        } else if (std::fabs(cost - res.best_cost) < 1e-9) {
            res.all_optimal.push_back(bits);
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Simulated Annealing — matches classical.py SimulatedAnnealing
// Demand-aware warm start: cheapest cost/power generators first.
// ---------------------------------------------------------------------------
struct SAResult {
    std::string best_bitstring;
    double      best_cost;
    bool        success;
    int         evaluations;
    double      wall_time_seconds;
};

void warm_start(int x[N], std::mt19937_64& rng) {
    int order[N];
    std::iota(order, order + N, 0);
    std::sort(order, order + N, [](int a, int b) {
        return COST[a] / std::max(POWER[a], 1e-12) < COST[b] / std::max(POWER[b], 1e-12);
    });
    std::fill(x, x + N, 0);
    double running = 0.0;
    for (int k = 0; k < N; ++k) {
        if (running >= DEMAND) break;
        x[order[k]] = 1;
        running += POWER[order[k]];
    }
    // Small random perturbation (matches _initial_state)
    std::uniform_int_distribution<int> bit_dist(0, N - 1);
    int flips = static_cast<int>(
        std::uniform_int_distribution<int>(0, std::max(2, N / 5))(rng));
    for (int f = 0; f < flips; ++f) x[bit_dist(rng)] ^= 1;
}

SAResult simulated_annealing(double exact_optimal_cost, int seed = 42,
                              double T0 = 100.0, double Tf = 0.01,
                              double alpha = 0.995, int iters_per_T = 50) {
    auto t0 = std::chrono::steady_clock::now();
    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_int_distribution<int> bit_dist(0, N - 1);

    int cur[N], best_x[N], cand[N];
    warm_start(cur, rng);
    std::copy(cur, cur + N, best_x);
    double cur_cost = qubo_cost(cur), best_cost = cur_cost;
    int evals = 1;

    for (double T = T0; T > Tf; T *= alpha)
        for (int it = 0; it < iters_per_T; ++it) {
            std::copy(cur, cur + N, cand);
            cand[bit_dist(rng)] ^= 1;
            double cc = qubo_cost(cand); ++evals;
            double delta = cc - cur_cost;
            if (delta <= 0.0 || uni(rng) < std::exp(-delta / std::max(T, 1e-12))) {
                std::copy(cand, cand + N, cur); cur_cost = cc;
            }
            if (cur_cost < best_cost) { best_cost = cur_cost; std::copy(cur, cur + N, best_x); }
        }

    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string bits(N, '0');
    for (int i = 0; i < N; ++i) bits[i] = '0' + best_x[i];
    return {bits, best_cost, std::fabs(best_cost - exact_optimal_cost) <= 1e-5, evals, wall};
}

} // namespace mg20

// =============================================================================
// Tests
// =============================================================================

TEST(MicrogridQAOA20, ExactSolver) {
    using namespace mg20;

    std::cout << "\n=== Exact Solver (20-qubit, enumerate all 2^20) ===\n";
    auto t0  = std::chrono::high_resolution_clock::now();
    auto res = exact_solve();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - t0).count();

    std::cout << "  Best QUBO cost:     " << res.best_cost << "\n";
    std::cout << "  Best bitstring:     " << res.best_bitstring
              << "  (index 0 = gen01)\n";
    std::cout << "  Degenerate optima:  " << res.all_optimal.size() << "\n";
    std::cout << "  Runtime:            " << ms << " ms\n\n";

    // Decode active generators
    double total_power = 0.0, total_op = 0.0;
    std::cout << "  Active generators:\n";
    for (int i = 0; i < N; ++i)
        if (res.best_bitstring[i] == '1') {
            std::cout << "    " << NAMES[i] << "  cost=" << COST[i] << "/MWh\n";
            total_power += POWER[i]; total_op += COST[i];
        }
    std::cout << "  Total dispatch: " << total_power << " MW  (demand=" << DEMAND << ")\n";
    std::cout << "  Op cost:  " << total_op << "\n";
    std::cout << "  Penalty:  " << A * (total_power - DEMAND) * (total_power - DEMAND) << "\n";

    EXPECT_LT(res.best_cost, 1000.0);
}

TEST(MicrogridQAOA20, SimulatedAnnealing) {
    using namespace mg20;
    auto exact = exact_solve();

    std::cout << "\n=== Simulated Annealing (mirrors classical.py SimulatedAnnealing) ===\n";
    std::cout << "  Exact optimum:  " << exact.best_cost
              << "  (" << exact.best_bitstring << ")\n";

    auto t0  = std::chrono::high_resolution_clock::now();
    auto res = simulated_annealing(exact.best_cost, 42);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - t0).count();

    std::cout << "  SA best cost:   " << res.best_cost << "\n";
    std::cout << "  SA best bits:   " << res.best_bitstring << "\n";
    std::cout << "  Success:        " << (res.success ? "yes" : "no") << "\n";
    std::cout << "  Evaluations:    " << res.evaluations << "\n";
    std::cout << "  Runtime:        " << ms << " ms\n";

    // Single SA run — Python uses a 300s time budget (experiment_config.yaml shootout.budget_seconds).
    // A single schedule on 20 qubits may not reach the exact optimum; allow 2× gap.
    EXPECT_LE(res.best_cost, exact.best_cost * 2.0);
}

TEST(MicrogridQAOA20, MAQAOA_Layerwise) {
    using namespace mg20;

    // Matching experiment_config.yaml
    constexpr int    P_MAX  = 6;
    constexpr int    BUDGET = 500;   // budget_per_layer_evals
    constexpr int    SEED   = 42;

    auto exact = exact_solve();
    auto ising  = build_ising();
    auto ham    = build_hamiltonian(ising);

    MAQAOA maqaoa;
    maqaoa.estimator.options.shots = 0;
    maqaoa.sampler.options.shots   = 8192;
    maqaoa.sampler.options.seed    = SEED;
    maqaoa.options.p                     = P_MAX;
    maqaoa.options.layerwise             = true;
    maqaoa.options.max_iterations        = BUDGET;
    maqaoa.options.convergence_threshold = 1e-6;
    maqaoa.options.seed                  = SEED;

    int params_per_layer = maqaoa.num_parameters(ham) / P_MAX;
    int total_params     = P_MAX * params_per_layer;

    std::cout << "\n=== MA-QAOA Layerwise (20 qubits) ===\n";
    std::cout << "  N=20, demand=" << DEMAND << " MW, A=" << A << "\n";
    std::cout << "  p_max=" << P_MAX << ", budget/layer=" << BUDGET << ", seed=" << SEED << "\n";
    std::cout << "  Hamiltonian terms: " << ham.terms.size()
              << " (" << N << " Z + " << N*(N-1)/2 << " ZZ)\n";
    std::cout << "  Params/layer (qubit-indexed): " << params_per_layer
              << "  (term-indexed would be " << ham.terms.size() + N << ")\n";
    std::cout << "  Total params: " << total_params << "\n";
    std::cout << "  Exact QUBO optimum:  " << exact.best_cost
              << "  (" << exact.best_bitstring << ")\n";
    std::cout << "  Ising offset:        " << ising.offset << "\n";
    std::cout << "  Target Ising energy: " << exact.best_cost - ising.offset << "\n\n";

    auto t0     = std::chrono::steady_clock::now();
    auto result = maqaoa.optimize(ham);
    auto total_s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();

    double final_qubo = result.optimal_value + ising.offset;
    double sampled_qubo = std::numeric_limits<double>::quiet_NaN();
    if (static_cast<int>(result.best_bitstring.size()) == N) {
        int x[N];
        for (int i = 0; i < N; ++i) x[i] = result.best_bitstring[i] - '0';
        sampled_qubo = qubo_cost(x);
    }

    std::cout << "  MA-QAOA Ising energy:       " << result.optimal_value << "\n";
    std::cout << "  MA-QAOA QUBO (Ising+offset):" << final_qubo << "\n";
    std::cout << "  Exact QUBO optimum:         " << exact.best_cost << "\n";
    std::cout << "  Approximation ratio:        "
              << (final_qubo > 0 ? exact.best_cost / final_qubo : 0.0) << "\n";
    std::cout << "  Best sampled bitstring:     " << result.best_bitstring << "\n";
    std::cout << "  Sampled QUBO cost:          " << sampled_qubo << "\n";
    std::cout << "  Success (exact):            "
              << (std::fabs(sampled_qubo - exact.best_cost) < 1e-5 ? "yes" : "no") << "\n";
    std::cout << "  Converged:                  " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  Param count:                " << result.optimal_params.size() << "\n";
    std::cout << "  Total wall time:            " << total_s << " s\n";

    EXPECT_EQ(static_cast<int>(result.optimal_params.size()), total_params);
    EXPECT_FALSE(result.best_bitstring.empty());
}

TEST(MicrogridQAOA20, AllMethodsComparison) {
    using namespace mg20;

    auto t0    = std::chrono::steady_clock::now();
    auto exact = exact_solve();
    auto t1    = std::chrono::steady_clock::now();
    auto sa    = simulated_annealing(exact.best_cost, 42);
    auto t2    = std::chrono::steady_clock::now();

    double exact_s = std::chrono::duration<double>(t1 - t0).count();
    double sa_s    = std::chrono::duration<double>(t2 - t1).count();

    auto ising = build_ising();
    auto ham   = build_hamiltonian(ising);

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Microgrid 20-Generator Optimisation — Method Comparison             ║\n";
    std::cout << "║  (mirrors EnergyGridOptimisation_20qubit / critical_tight_A)         ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  N=20, demand=52 MW, A=20, generators from generators.csv            ║\n";
    std::cout << "╠═══════════════════════════╦═══════════════╦═════════════════════════╣\n";
    std::cout << "║  Method                   ║  QUBO cost    ║  Notes                  ║\n";
    std::cout << "╠═══════════════════════════╬═══════════════╬═════════════════════════╣\n";

    char buf[256];
    auto row = [&](const char* name, double cost, const char* note) {
        std::snprintf(buf, sizeof(buf),
            "║  %-25s║  %13.4f║  %-23s  ║\n", name, cost, note);
        std::cout << buf;
    };

    char exact_note[64], sa_note[64];
    std::snprintf(exact_note, sizeof(exact_note), "%.3f s, %zu optima",
                  exact_s, exact.all_optimal.size());
    std::snprintf(sa_note, sizeof(sa_note), "%.3f s, %s",
                  sa_s, sa.success ? "exact match" : "approx");

    row("Exact (enumerate 2^20)",  exact.best_cost, exact_note);
    row("Simulated Annealing",     sa.best_cost,    sa_note);
    row("MA-QAOA p=6 (layerwise)", 0.0,             "run MAQAOA_Layerwise test");

    std::cout << "╠═══════════════════════════╩═══════════════╩═════════════════════════╣\n";
    std::cout << "║  Bitstrings:                                                          ║\n";
    std::cout << "║    Exact optimal: " << exact.best_bitstring << " ║\n";
    std::cout << "║    SA best found: " << sa.best_bitstring    << " ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  C++ q++ MA-QAOA: qubit-indexed gammas (matches Python)               ║\n";
    std::snprintf(buf, sizeof(buf),
        "║    %d params/layer (2*N), %d total for p=6                             ║\n",
        2 * N, 2 * N * 6);
    std::cout << buf;
    std::cout << "║  Python MA-QAOA:  qubit-indexed gammas                                ║\n";
    std::snprintf(buf, sizeof(buf),
        "║    40 params/layer (2*N), %d total for p=6                             ║\n",
        2 * N * 6);
    std::cout << buf;
    std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";

    EXPECT_LT(exact.best_cost, 1000.0);
    EXPECT_LE(sa.best_cost, exact.best_cost * 2.0);
}
