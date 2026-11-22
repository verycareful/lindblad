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
// ScopedThreadCap - an OpenMP thread cap for the length of one call
// =============================================================================
// max_parallel_threads caps the threads every parallel region inside one
// public call may use. OpenMP takes the thread count of a new region from the
// calling thread's own setting, so the cap sets that setting on entry and puts
// the caller's back on exit, however the call ends. The caller's parallel code
// before and after the call, and every other thread (a batch running
// simulators side by side), keep their own.
//
// 0 leaves OpenMP's own choice in force (OMP_NUM_THREADS, else every core); a
// negative cap is refused. In a build without OpenMP there are no parallel
// regions and a cap changes nothing.

#include "lindblad/detail/validate.hpp"
#include "lindblad/types.hpp"

#include <cstddef>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lindblad {
namespace detail {

class ScopedThreadCap {
public:
    ScopedThreadCap(int cap, const char* ctx) {
        check_require(cap >= 0, ctx,
                      "max_parallel_threads must be >= 0 (0 = the OpenMP default), got " +
                          std::to_string(cap));
#ifdef _OPENMP
        if (cap > 0) {
            previous_ = omp_get_max_threads();
            omp_set_num_threads(cap);
        }
#endif
    }

    ~ScopedThreadCap() {
#ifdef _OPENMP
        if (previous_ > 0) omp_set_num_threads(previous_);
#endif
    }

    ScopedThreadCap(const ScopedThreadCap&) = delete;
    ScopedThreadCap& operator=(const ScopedThreadCap&) = delete;

private:
    int previous_ = 0;  // the caller's setting, when a cap replaced it
};

// =============================================================================
// ScopedParallelGrain - the statevector parallel grain for the length of a call
// =============================================================================
// The amplitudes each thread must get before a statevector gate kernel runs
// in parallel (DEFAULT_MIN_AMPLITUDES_PER_THREAD in types.hpp). The kernels
// read it from parallel_grain(), a per-thread setting, so a simulator's
// min_amplitudes_per_thread reaches them without passing through every gate's
// signature. Like ScopedThreadCap it holds for one public call and puts the
// calling thread's previous value back however the call ends, and other
// threads keep theirs. 0 keeps the value in force; a gate called outside any
// simulator uses the default.
inline thread_local std::size_t t_parallel_grain = DEFAULT_MIN_AMPLITUDES_PER_THREAD;

// The automatic team model of the calling thread (team_model.hpp), or null: a
// pass then takes the grain rule. A simulator whose min_amplitudes_per_thread
// is 0 installs one for the length of a call; no other thread sees it.
class TeamModel;
inline thread_local TeamModel* t_team_model = nullptr;

inline std::size_t parallel_grain() noexcept { return t_parallel_grain; }

// =============================================================================
// ScopedGrainRule - the grain rule for the length of a scope
// =============================================================================
// Sets the calling thread's team model aside and leaves the grain as it is. A
// measurement whose figures are defined on the grain rule (the fusion costs)
// holds this while it times its kernels, so what it measures does not follow
// the automatic team.
class ScopedGrainRule {
public:
    ScopedGrainRule() : previous_model_(t_team_model) { t_team_model = nullptr; }
    ~ScopedGrainRule() { t_team_model = previous_model_; }

    ScopedGrainRule(const ScopedGrainRule&) = delete;
    ScopedGrainRule& operator=(const ScopedGrainRule&) = delete;

private:
    TeamModel* previous_model_;
};

// A non-zero grain also puts the grain rule in force: the model is set aside
// for the scope's length. A measurement that forces its own teams (a grain of
// one amplitude) therefore never reads the model it is feeding.
class ScopedParallelGrain {
public:
    explicit ScopedParallelGrain(std::size_t grain)
        : previous_(t_parallel_grain), previous_model_(t_team_model) {
        if (grain > 0) {
            t_parallel_grain = grain;
            t_team_model = nullptr;
        }
    }

    ~ScopedParallelGrain() {
        t_parallel_grain = previous_;
        t_team_model = previous_model_;
    }

    ScopedParallelGrain(const ScopedParallelGrain&) = delete;
    ScopedParallelGrain& operator=(const ScopedParallelGrain&) = delete;

private:
    std::size_t previous_;          // the calling thread's value on entry
    TeamModel* previous_model_;     // the calling thread's model on entry
};

}  // namespace detail
}  // namespace lindblad
