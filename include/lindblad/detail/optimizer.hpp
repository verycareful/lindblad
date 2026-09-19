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
// optimizer - the one seam between the variational algorithms and the
//             derivative-free minimisers they run on
// =============================================================================
//
// VQE, QAOA and MA-QAOA all pose the same problem: minimise a scalar objective
// of a real parameter vector, derivative-free, under an evaluation cap and an
// x tolerance, with or without a box. This header is the only place that knows
// which library answers it. Two libraries do: FLOP (verycareful/FLOP) supplies
// the default COBYLA; NLopt supplies its own COBYLA, Nelder-Mead and BOBYQA.
// An algorithm names a backend, hands over its objective, and reads a
// library-neutral outcome. Nothing above this seam includes <nlopt.h> or
// <flop/flop.hpp>.
//
// The public knob stays a string (VQE::Options::optimizer and its two
// siblings), so the mapping from names to backends lives here as well and the
// three call sites cannot drift apart on spelling.

#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lindblad/types.hpp"

namespace lindblad::detail {

// =============================================================================
// Initial parameters
// =============================================================================

// The RNG every variational algorithm seeds its starting point from. seed == 0
// means "not seeded" and draws the seed from std::random_device, the
// convention every simulator's run(shots, seed) already follows.
inline std::mt19937_64 seeded_rng(std::uint64_t seed) {
    return std::mt19937_64(seed == 0 ? static_cast<std::uint64_t>(std::random_device{}())
                                     : seed);
}

// A draw in [lo, hi) that is the same bits on every compiler and every
// floating-point model, given the same engine state.
//
// std::uniform_real_distribution is header code, so a library built with
// -ffast-math and a caller built without it can turn the same engine output
// into values an ulp apart, and a derivative-free optimiser amplifies that
// ulp into a different trajectory. This draw uses only operations IEEE 754
// pins to one result: the top 53 bits of the engine word converted to a
// double (exact), scaled by 2^-53 (exact), then one correctly rounded
// std::fma. A compiler is not permitted to contract, split or reassociate
// any of it.
inline double uniform_in(std::mt19937_64& rng, double lo, double hi) {
    const double u = static_cast<double>(rng() >> 11) * 0x1p-53;
    return std::fma(hi - lo, u, lo);
}

// =============================================================================
// Backends
// =============================================================================

// Which library and method answers a minimisation request. FlopCobyla is the
// project default; the three NLopt values exist because FLOP ships only
// COBYLA today and the public options already promise Nelder-Mead and BOBYQA.
enum class OptimizerBackend { FlopCobyla, NloptCobyla, NloptNelderMead, NloptBobyqa };

// The public name for each backend, as a caller writes it into
// Options::optimizer: "COBYLA", "NLOPT_COBYLA", "NELDER_MEAD", "BOBYQA".
std::string_view optimizer_name(OptimizerBackend backend) noexcept;

// Every accepted public name, for diagnostics and documentation.
std::span<const std::string_view> optimizer_names() noexcept;

// Map a public name to its backend. Returns false, leaving `out` untouched,
// for a name that is not in optimizer_names().
bool parse_optimizer_name(std::string_view name, OptimizerBackend& out) noexcept;

// Map a public name to its backend for an algorithm entry point. An unknown
// name emits a warning through the warning channel naming `where` and the
// offending string, and resolves to FlopCobyla, so a misspelt option degrades
// to the documented default loudly rather than failing the run.
OptimizerBackend resolve_optimizer(const std::string& name, const char* where);

// What an algorithm asks of a minimiser.
//
// max_evaluations caps objective calls (the public field is named
// max_iterations; for every method here one iteration is one evaluation).
// xtol_rel is the relative x tolerance; each library measures it its own way
// (FLOP against initial_step, NLopt against |x|), and both stop when the
// trust region or simplex has shrunk to that scale. initial_step is the
// displacement of the first n trial points from x0 along each axis, and every
// backend is given it explicitly so no library-side default (NLopt's is
// |x0_i| per coordinate) decides how much of a small budget the initial
// simplex spends. lower/upper are either both empty (unbounded) or both of
// size n.
struct OptimizerSpec {
    OptimizerBackend backend = OptimizerBackend::FlopCobyla;
    int max_evaluations = 0;
    double xtol_rel = 0.0;
    double initial_step = 0.0;
    std::vector<double> lower;
    std::vector<double> upper;
};

// What comes back, in terms no library owns.
//
// x and f are the library's returned point and value when it reported a
// finite one, and otherwise the best finite point inside the box that the
// seam itself saw the objective evaluated at. The seam keeps that record on
// every run, so a backend that returns without writing its result (NLopt on
// a failure code) cannot surface an uninitialised value.
//
// converged is true only when the backend stopped on a tolerance (x or f)
// with a finite best value: reaching the evaluation cap, a roundoff-limited
// exit, a library failure code and a non-finite objective all report false.
// hit_evaluation_cap says whether the cap was the reason. status carries the
// backend's own exit description for diagnostics.
struct OptimizerOutcome {
    std::vector<double> x;
    double f = quiet_nan_strict();
    int evaluations = 0;
    bool converged = false;
    bool hit_evaluation_cap = false;
    bool saw_non_finite = false;
    std::string status;
};

// The objective. Receives the trial point, returns its value. It may throw;
// an exception propagates out of minimize() unchanged on the FLOP backend
// and is re-thrown after the NLopt call returns on the NLopt backends (the
// C frames in between are not unwound through).
using Objective = std::function<double(std::span<const double>)>;

// Run the minimisation described by `spec` from `x0`.
//
// Throws std::invalid_argument for a malformed request before any evaluation:
// empty x0, a non-positive evaluation cap, a non-positive initial step, a
// negative tolerance, bounds of the wrong size or a bound the start point
// violates. The message names `where`.
//
// A non-finite objective value (checked by bit pattern, see is_finite_strict)
// ends the run: the outcome carries the best finite point seen so far,
// converged = false and saw_non_finite = true. A run whose objective never
// produced a finite value throws std::runtime_error, since there is nothing
// to return that a caller could mistake for a result.
OptimizerOutcome minimize(const OptimizerSpec& spec, const Objective& objective,
                          std::span<const double> x0, const char* where);

}  // namespace lindblad::detail
