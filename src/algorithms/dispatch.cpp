#include "lindblad/dispatch.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace lindblad {

SoftDispatchResult::SoftDispatchResult(
    const std::unordered_map<std::string, int>& counts_in
) : counts(counts_in), total_shots(0) {
    for (const auto& [bs, cnt] : counts) total_shots += cnt;
    compute();
}

void SoftDispatchResult::compute() {
    if (counts.empty() || total_shots == 0) return;

    // Infer n_qubits from the first bitstring
    int n = static_cast<int>(counts.begin()->first.size());
    soft_assignment.assign(n, 0.0);

    best_bitstring.clear();
    best_probability = 0.0;

    for (const auto& [bs, cnt] : counts) {
        double prob = static_cast<double>(cnt) / total_shots;
        if (prob > best_probability) {
            best_probability = prob;
            best_bitstring = bs;
        }
        for (int i = 0; i < n; ++i) {
            if (bs[i] == '1') soft_assignment[i] += prob;
        }
    }
}

std::string SoftDispatchResult::threshold_round(double threshold) const {
    int n = static_cast<int>(soft_assignment.size());
    std::string result(n, '0');
    for (int i = 0; i < n; ++i) {
        if (soft_assignment[i] >= threshold) result[i] = '1';
    }
    return result;
}

std::vector<int> SoftDispatchResult::greedy_dispatch(
    const std::vector<double>& generator_capacities,
    double demand
) const {
    int n = static_cast<int>(soft_assignment.size());
    if (static_cast<int>(generator_capacities.size()) != n) {
        throw std::invalid_argument(
            "generator_capacities size must match soft_assignment size"
        );
    }

    // Sort generators by soft_assignment descending
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return soft_assignment[a] > soft_assignment[b];
    });

    std::vector<int> selected;
    double supplied = 0.0;
    for (int idx : order) {
        if (supplied >= demand) break;
        selected.push_back(idx);
        supplied += generator_capacities[idx];
    }
    return selected;
}

double SoftDispatchResult::expected_cost(
    const std::function<double(const std::string&)>& cost_fn
) const {
    if (total_shots == 0) return 0.0;
    double total_cost = 0.0;
    for (const auto& [bs, cnt] : counts) {
        double prob = static_cast<double>(cnt) / total_shots;
        total_cost += prob * cost_fn(bs);
    }
    return total_cost;
}

std::vector<std::pair<std::string, double>> SoftDispatchResult::top_k(int k) const {
    std::vector<std::pair<std::string, double>> items;
    items.reserve(counts.size());
    for (const auto& [bs, cnt] : counts) {
        items.push_back({bs, static_cast<double>(cnt) / total_shots});
    }
    int actual_k = std::min(k, static_cast<int>(items.size()));
    std::partial_sort(
        items.begin(), items.begin() + actual_k, items.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );
    items.resize(actual_k);
    return items;
}

} // namespace lindblad
