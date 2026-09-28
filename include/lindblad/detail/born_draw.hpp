// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

// =============================================================================
// born_draw - one Born-rule draw from a list of outcome weights
// =============================================================================
// Every dense sampler and collapse draws the same way: an outcome with weight
// w_s is chosen with probability w_s / total, where total is the sum of the
// weights, so a state short of unit norm is sampled as its own normalised
// distribution rather than having its missing weight pushed onto one outcome.
//
// A uniform u in [0, 1) is scaled to target = u * total, and the outcome is the
// FIRST whose running sum exceeds the target, strictly. The running sum only
// grows at an outcome of positive weight, so the comparison can never stop on
// an outcome of weight zero. Rounding can leave a target equal to the total
// (u a hair below 1 rounds the product up); no running sum exceeds it then, and
// the draw falls back to the last outcome of positive weight, which is where
// the target would have landed exactly.
//
// Both entries require what every caller establishes first with
// require_norm_to_sample (validate_physical.hpp): a finite total above zero.
// The weights must be non-negative; a caller holding raw weights that could be
// negative (a density matrix's diagonal) clamps them before the draw.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace lindblad {
namespace detail {

// Binary search over precomputed running sums, for many draws from one state:
// cum[s] = w_0 + ... + w_s, so cum.back() is the total.
inline std::size_t born_draw_cumulative(const std::vector<double>& cum, double u) {
    const double target = u * cum.back();
    const auto it = std::upper_bound(cum.begin(), cum.end(), target);
    if (it != cum.end()) return static_cast<std::size_t>(it - cum.begin());
    std::size_t s = cum.size() - 1;
    while (s > 0 && cum[s] <= cum[s - 1]) --s;
    return s;
}

// One draw by a linear walk, for a single draw where building the running sums
// would cost memory the walk does not need. `total` must be the sum of the
// same weights in the same order, so the walk ends on it exactly.
template <class Weight>
std::size_t born_draw_linear(std::size_t k, double total, double u, Weight weight) {
    const double target = u * total;
    double running = 0.0;
    std::size_t last_positive = 0;
    for (std::size_t s = 0; s < k; ++s) {
        const double w = weight(s);
        if (w <= 0.0) continue;
        running += w;
        last_positive = s;
        if (target < running) return s;
    }
    return last_positive;
}

}  // namespace detail
}  // namespace lindblad
