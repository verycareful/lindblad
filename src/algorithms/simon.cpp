#include "qpp/algorithms.hpp"
#include "qpp/simulators/statevector_sim.hpp"

#include <random>
#include <string>
#include <vector>

namespace qpp {
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
        std::string raw = res.counts.begin()->first; // 2n chars, MSB-first
        std::string y = raw.substr(n);               // query register, still MSB-first
        std::reverse(y.begin(), y.end());            // y[j] = qubit j
        if (y == std::string(n, '0')) continue;
        bool dup = false;
        for (auto& eq : equations) if (eq == y) { dup = true; break; }
        if (!dup) equations.push_back(y);
    }

    std::string period = gaussian_eliminate(equations, n);
    return { period, equations };
}

} // namespace algorithms
} // namespace qpp
