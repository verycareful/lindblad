// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// =============================================================================
// optimizer - FLOP and NLopt behind the seam declared in
//             lindblad/detail/optimizer.hpp
// =============================================================================
//
// Both libraries are driven the same way: the seam wraps the caller's
// objective in a tracker that counts calls, checks every returned value by
// bit pattern, and remembers the best finite feasible point, then hands the
// tracker to the library. The library's own answer is used when it is finite;
// the tracker's is the fallback for every exit on which the library wrote
// nothing usable. NLopt is C, so nothing is thrown through it: the wrapper
// records what happened, asks NLopt to stop, and the seam acts once the call
// has returned.

#include "lindblad/detail/optimizer.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <array>
#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <flop/cobyla.hpp>
#include <nlopt.h>

namespace lindblad::detail {

namespace {

constexpr std::array<std::string_view, 4> kNames = {
    "COBYLA",        // OptimizerBackend::FlopCobyla
    "NLOPT_COBYLA",  // OptimizerBackend::NloptCobyla
    "NELDER_MEAD",   // OptimizerBackend::NloptNelderMead
    "BOBYQA",        // OptimizerBackend::NloptBobyqa
};

// =============================================================================
// Tracker - the objective as the libraries see it
// =============================================================================

// Wraps the caller's objective. Counts evaluations, records the best finite
// point inside the box, and on a non-finite value or an exception remembers
// the event so the seam can end the run and report it. The FLOP path throws
// out of operator() directly (FLOP is C++ and propagates); the NLopt path
// stores the exception and force-stops instead, because an exception must
// not cross NLopt's C frames.
struct Tracker {
    const Objective& objective;
    const OptimizerSpec& spec;
    std::vector<double> best_x;
    double best_f = quiet_nan_strict();
    bool have_best = false;
    int evaluations = 0;
    bool saw_non_finite = false;
    std::exception_ptr pending;

    Tracker(const Objective& f, const OptimizerSpec& s) : objective(f), spec(s) {}

    bool feasible(std::span<const double> x) const noexcept {
        if (spec.lower.empty()) return true;
        for (std::size_t i = 0; i < x.size(); ++i)
            if (x[i] < spec.lower[i] || x[i] > spec.upper[i]) return false;
        return true;
    }

    // One evaluation. Returns the value for the library. A non-finite value
    // is reported through saw_non_finite and returned as is; what the
    // library then does with it is the backend's business (FLOP stops on
    // it, NLopt is told to stop).
    double evaluate(std::span<const double> x) {
        ++evaluations;
        const double f = objective(x);
        if (!is_finite_strict(f)) {
            saw_non_finite = true;
            return f;
        }
        if ((!have_best || f < best_f) && feasible(x)) {
            best_x.assign(x.begin(), x.end());
            best_f = f;
            have_best = true;
        }
        return f;
    }
};

// The exception that unwinds FLOP when the objective produced a non-finite
// value. FLOP throws std::runtime_error itself on seeing a NaN, but only where
// the consuming build's floating-point flags let the NaN reach its check (see
// FLOP's documentation on -ffinite-math-only). The seam decides from its own
// bit-pattern check so the contract does not depend on those flags, and
// unwinds with a type of its own so that FLOP's runtime_error, if it ever
// fires first, is not mistaken for the seam's.
struct NonFiniteStop final : std::exception {
    const char* what() const noexcept override { return "non-finite objective"; }
};

// =============================================================================
// Request validation
// =============================================================================

void validate(const OptimizerSpec& spec, std::span<const double> x0, const char* where) {
    const std::string prefix = std::string(where) + ": ";
    if (x0.empty())
        throw std::invalid_argument(prefix + "the parameter vector is empty");
    for (double v : x0)
        if (!is_finite_strict(v))
            throw std::invalid_argument(prefix + "the initial parameters contain a "
                                                 "non-finite value");
    if (spec.max_evaluations <= 0)
        throw std::invalid_argument(prefix + "max_iterations must be positive (got " +
                                    std::to_string(spec.max_evaluations) + ")");
    if (!is_finite_strict(spec.xtol_rel) || spec.xtol_rel < 0.0)
        throw std::invalid_argument(prefix + "convergence_threshold must be finite and "
                                             "non-negative");
    if (!is_finite_strict(spec.initial_step) || !(spec.initial_step > 0.0))
        throw std::invalid_argument(prefix + "initial_step must be positive and finite");
    if (spec.lower.size() != spec.upper.size())
        throw std::invalid_argument(prefix + "lower and upper bounds differ in size");
    if (!spec.lower.empty()) {
        if (spec.lower.size() != x0.size())
            throw std::invalid_argument(prefix + "bounds have " +
                                        std::to_string(spec.lower.size()) + " entries for " +
                                        std::to_string(x0.size()) + " parameters");
        for (std::size_t i = 0; i < x0.size(); ++i) {
            if (!is_finite_strict(spec.lower[i]) || !is_finite_strict(spec.upper[i]))
                throw std::invalid_argument(prefix + "a bound is non-finite");
            if (!(spec.lower[i] < spec.upper[i]))
                throw std::invalid_argument(prefix + "lower bound is not below upper bound "
                                                     "at parameter " + std::to_string(i));
            if (x0[i] < spec.lower[i] || x0[i] > spec.upper[i])
                throw std::invalid_argument(prefix + "initial parameter " + std::to_string(i) +
                                            " lies outside its bounds");
        }
    }
}

// =============================================================================
// Outcome assembly
// =============================================================================

// Combine what the library returned with what the tracker saw. The library's
// point wins when it reported a finite value; otherwise the tracker's best
// finite feasible point is the answer, and if there is none the run produced
// nothing a caller could use, which is an error rather than a result.
OptimizerOutcome finish(const Tracker& tracker, std::optional<std::span<const double>> lib_x,
                        double lib_f, bool stopped_on_tolerance, bool hit_cap,
                        std::string status, const char* where) {
    OptimizerOutcome out;
    out.evaluations = tracker.evaluations;
    out.saw_non_finite = tracker.saw_non_finite;
    out.hit_evaluation_cap = hit_cap;
    out.status = std::move(status);

    if (lib_x && is_finite_strict(lib_f)) {
        out.x.assign(lib_x->begin(), lib_x->end());
        out.f = lib_f;
    } else if (tracker.have_best) {
        out.x = tracker.best_x;
        out.f = tracker.best_f;
    } else {
        throw std::runtime_error(std::string(where) + ": the optimiser produced no finite "
                                 "objective value (" + out.status + ", " +
                                 std::to_string(tracker.evaluations) + " evaluations)");
    }
    out.converged = stopped_on_tolerance && !tracker.saw_non_finite && is_finite_strict(out.f);
    return out;
}

// =============================================================================
// FLOP
// =============================================================================

OptimizerOutcome minimize_flop(const OptimizerSpec& spec, const Objective& objective,
                               std::span<const double> x0, const char* where) {
    Tracker tracker(objective, spec);

    flop::cobyla::Options opts;
    opts.stopping.max_evaluations = static_cast<std::size_t>(spec.max_evaluations);
    opts.stopping.xtol_rel = spec.xtol_rel;
    opts.initial_step = spec.initial_step;
    if (!spec.lower.empty()) opts.bounds = flop::Bounds::box(spec.lower, spec.upper);

    auto f = [&tracker](std::span<const double> x) -> double {
        const double v = tracker.evaluate(x);
        if (tracker.saw_non_finite) throw NonFiniteStop{};
        return v;
    };

    try {
        const flop::Result r = flop::cobyla::minimize(f, x0, opts);
        const bool on_tolerance = flop::converged(r.status);
        const bool cap = r.status == flop::Status::MaxEvaluationsReached;
        return finish(tracker, std::span<const double>(r.x), r.f, on_tolerance, cap,
                      std::string("FLOP COBYLA ") + std::string(flop::to_string(r.status)),
                      where);
    } catch (const NonFiniteStop&) {
        return finish(tracker, std::nullopt, quiet_nan_strict(), false, false,
                      "FLOP COBYLA stopped on a non-finite objective value", where);
    }
}

// =============================================================================
// NLopt
// =============================================================================

nlopt_algorithm nlopt_algorithm_for(OptimizerBackend backend) {
    switch (backend) {
        case OptimizerBackend::NloptCobyla:     return NLOPT_LN_COBYLA;
        case OptimizerBackend::NloptNelderMead: return NLOPT_LN_NELDERMEAD;
        case OptimizerBackend::NloptBobyqa:     return NLOPT_LN_BOBYQA;
        case OptimizerBackend::FlopCobyla:      break;
    }
    throw std::logic_error("nlopt_algorithm_for: not an NLopt backend");
}

// The C callback. Anything the objective throws is parked in the tracker and
// NLopt is asked to stop; the seam re-throws once nlopt_optimize has
// returned. A non-finite value is likewise turned into a forced stop, since
// NLopt's own handling of NaN differs between its methods.
struct NloptCallbackData {
    Tracker* tracker;
    nlopt_opt opt;
};

double nlopt_objective(unsigned n, const double* x, double* /*grad*/, void* data) {
    auto* cb = static_cast<NloptCallbackData*>(data);
    try {
        const double v = cb->tracker->evaluate(std::span<const double>(x, n));
        if (cb->tracker->saw_non_finite) nlopt_force_stop(cb->opt);
        return v;
    } catch (...) {
        cb->tracker->pending = std::current_exception();
        nlopt_force_stop(cb->opt);
        return quiet_nan_strict();
    }
}

// Owns the nlopt_opt for the duration of one call.
struct NloptHandle {
    nlopt_opt opt;
    NloptHandle(nlopt_algorithm algo, unsigned n) : opt(nlopt_create(algo, n)) {
        if (!opt) throw std::runtime_error("nlopt_create failed");
    }
    ~NloptHandle() { nlopt_destroy(opt); }
    NloptHandle(const NloptHandle&) = delete;
    NloptHandle& operator=(const NloptHandle&) = delete;
};

OptimizerOutcome minimize_nlopt(const OptimizerSpec& spec, const Objective& objective,
                                std::span<const double> x0, const char* where) {
    Tracker tracker(objective, spec);
    const unsigned n = static_cast<unsigned>(x0.size());

    NloptHandle h(nlopt_algorithm_for(spec.backend), n);
    nlopt_set_maxeval(h.opt, spec.max_evaluations);
    nlopt_set_xtol_rel(h.opt, spec.xtol_rel);
    if (!spec.lower.empty()) {
        nlopt_set_lower_bounds(h.opt, spec.lower.data());
        nlopt_set_upper_bounds(h.opt, spec.upper.data());
    }
    std::vector<double> step(n, spec.initial_step);
    nlopt_set_initial_step(h.opt, step.data());

    NloptCallbackData cb{&tracker, h.opt};
    nlopt_set_min_objective(h.opt, nlopt_objective, &cb);

    // NLopt can return a failure code without writing min_val; the marker is
    // a bit-built NaN so an unwritten value is detectable under every
    // floating-point model.
    std::vector<double> x(x0.begin(), x0.end());
    double min_val = quiet_nan_strict();
    const nlopt_result res = nlopt_optimize(h.opt, x.data(), &min_val);

    if (tracker.pending) std::rethrow_exception(tracker.pending);

    const bool on_tolerance = res == NLOPT_SUCCESS || res == NLOPT_STOPVAL_REACHED ||
                              res == NLOPT_FTOL_REACHED || res == NLOPT_XTOL_REACHED;
    const bool cap = res == NLOPT_MAXEVAL_REACHED;
    const std::optional<std::span<const double>> lib_x =
        res > 0 ? std::optional<std::span<const double>>(x) : std::nullopt;
    return finish(tracker, lib_x, min_val, on_tolerance, cap,
                  std::string("NLopt ") + std::string(optimizer_name(spec.backend)) + " " +
                      nlopt_result_to_string(res),
                  where);
}

}  // namespace

// =============================================================================
// Public seam
// =============================================================================

std::string_view optimizer_name(OptimizerBackend backend) noexcept {
    return kNames[static_cast<std::size_t>(backend)];
}

std::span<const std::string_view> optimizer_names() noexcept { return kNames; }

bool parse_optimizer_name(std::string_view name, OptimizerBackend& out) noexcept {
    for (std::size_t i = 0; i < kNames.size(); ++i) {
        if (name == kNames[i]) {
            out = static_cast<OptimizerBackend>(i);
            return true;
        }
    }
    return false;
}

OptimizerBackend resolve_optimizer(const std::string& name, const char* where) {
    OptimizerBackend backend = OptimizerBackend::FlopCobyla;
    if (!parse_optimizer_name(name, backend)) {
        std::string known;
        for (std::string_view k : kNames) {
            if (!known.empty()) known += ", ";
            known += k;
        }
        emit_warning(std::string(where) + ": unknown optimizer '" + name +
                     "', defaulting to COBYLA (known: " + known + ")");
        backend = OptimizerBackend::FlopCobyla;
    }
    return backend;
}

OptimizerOutcome minimize(const OptimizerSpec& spec, const Objective& objective,
                          std::span<const double> x0, const char* where) {
    validate(spec, x0, where);
    if (spec.backend == OptimizerBackend::FlopCobyla)
        return minimize_flop(spec, objective, x0, where);
    return minimize_nlopt(spec, objective, x0, where);
}

}  // namespace lindblad::detail
