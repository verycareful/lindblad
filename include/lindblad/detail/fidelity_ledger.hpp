// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include <algorithm>
#include <cmath>
#include <optional>

// =============================================================================
// detail::FidelityLedger - how far truncation has moved an MPS
// =============================================================================
// Both MPS layers report how close their chain is to the state an untruncated
// run would hold, built from each split's discarded FRACTION
//
//     eps_k = discarded_k / (kept_k + discarded_k).
//
// For a split in canonical gauge that is the fraction of the state the split
// removed: the block then carries the state's whole norm, and keeping its
// leading singular directions is an orthogonal projection of the state.
//
// estimate() = prod_k (1 - eps_k), the standard figure (Zhou, Stoudenmire and
// Waintal, Phys. Rev. X 10, 041038, 2020) and accurate in practice. It is NOT a
// bound. Two rotations by a, each followed by a truncation back onto |0>, keep
// cos²a per split, so the product is cos⁴a ≈ 1 - 2a², while the true fidelity
// is cos²(2a) ≈ 1 - 4a²: the overlap between successive projections has no
// fixed sign.
//
// lower_bound() = max(0, 1 - Delta²/2)², Delta = sum_k delta_k, with
//
//     delta_k = sqrt(2 - 2 sqrt(1 - eps_k)),
//
// the exact distance between a unit vector and its renormalised projection
// keeping 1 - eps_k of its weight. Gates between splits are unitary and move
// both states alike, so by the triangle inequality the normalised exact and
// truncated states are within Delta of each other, and for unit vectors
// |<phi|psi>| >= Re<phi|psi> = 1 - |phi - psi|²/2. On the example above it is
// 1 - 4a² to leading order in a, the true fidelity.
// Both figures compare normalised states, so rescaling the chain changes
// neither.
//
// A split that ran outside canonical gauge contributes its fraction of the
// BLOCK, which is not a fraction of the state. CanonicalForm::Auto runs a split
// in place only when that fraction cannot exceed MPS_DEFAULT_CUTOFF, so under
// Auto the figures carry that rounding-level approximation, and under
// CanonicalForm::Always none.

namespace lindblad {
namespace detail {

struct StateFileAccess;

class FidelityLedger {
public:
    // One split: kept = Σ sigma² over the retained singular values, discarded
    // = the weight the split threw away, unresolved = weight the split could
    // not resolve (the Gram route's floor), which may be real or may be noise
    // the route manufactured. The estimate leaves unresolved weight out; the
    // bound counts it as removed, so that it stays a bound either way. A block
    // with no weight removes none.
    void record(double kept, double discarded, double unresolved = 0.0) noexcept {
        const double total = kept + discarded;
        const double eps = (total > 0.0) ? std::clamp(discarded / total, 0.0, 1.0)
                                         : 0.0;
        retained_ *= 1.0 - eps;
        const double total_b = total + unresolved;
        const double eps_b =
            (total_b > 0.0) ? std::clamp((discarded + unresolved) / total_b, 0.0, 1.0)
                            : 0.0;
        // 2 - 2 sqrt(1 - eps) written as 2 eps / (1 + sqrt(1 - eps)): the
        // direct form subtracts two numbers within eps of each other, which at
        // a rounding-level eps returns zero or noise instead of the distance.
        distance_ += std::sqrt(2.0 * eps_b / (1.0 + std::sqrt(1.0 - eps_b)));
    }

    // A measurement collapsed the chain. Projection renormalises the exact and
    // the truncated state by different factors, so neither figure describes
    // the collapsed pair, and nothing recorded afterwards restores them.
    void invalidate() noexcept { valid_ = false; }

    // The chain as it stands becomes the reference: both figures read exact.
    void reset() noexcept { *this = FidelityLedger{}; }

    std::optional<double> estimate() const noexcept {
        if (!valid_) return std::nullopt;
        return retained_;
    }

    std::optional<double> lower_bound() const noexcept {
        if (!valid_) return std::nullopt;
        const double overlap = std::max(0.0, 1.0 - 0.5 * distance_ * distance_);
        return overlap * overlap;
    }

private:
    // The failed-run state file writes and restores these figures with the
    // chain they describe.
    friend struct StateFileAccess;

    double retained_ = 1.0;  // prod_k (1 - eps_k)
    double distance_ = 0.0;  // Delta = sum_k delta_k
    bool valid_ = true;
};

} // namespace detail
} // namespace lindblad
