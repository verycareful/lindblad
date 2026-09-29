// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/errors.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lindblad {

// Defined in observation.hpp; declared here so this header stays light.
enum class Response;

namespace detail {

// =============================================================================
// report - the one way a failure is raised or a knobbed refusal delivered
// =============================================================================
// Which to call:
//   raise_internal   a check that can only fail if Lindblad itself is wrong
//   raise<E>         any other failure without a knob: InvalidArgument for a
//                    caller mistake or an exceeded limit, OutOfRange for an
//                    index outside its range, RuntimeFailure for a failure met
//                    while running that makes the answer wrong
//   respond          a failure that leaves the answer intact, governed by the
//                    caller's Response
//
// Outside namespace lindblad::detail, write detail::raise<E>(...): <csignal>
// declares a global raise(int), and the qualified name can never reach it.

// "<entry point>: <what>", then " (instruction <i>: <gate> on qubits a, b)"
// when an instruction is known and " at shot <k>" when a shot is. An empty
// entry point yields the text alone, for messages that already name their
// requester.
std::string failure_message(std::string_view entry_point, std::string_view what,
                            const std::optional<FailurePoint>& where);

// A failure without a knob. Never returns.
template <class E>
[[noreturn]] void raise(std::string_view entry_point, std::string_view what,
                        std::optional<FailurePoint> where = {}) {
    std::string message = failure_message(entry_point, what, where);
    throw E(std::move(message), std::string(entry_point), std::move(where));
}

// A defect in Lindblad. Appends the request to report it. Never returns.
[[noreturn]] void raise_internal(std::string_view entry_point, std::string_view what,
                                 std::optional<FailurePoint> where = {});

// A knobbed refusal (a failure that leaves the answer intact). Under Throw,
// and under Auto before the first gate, raises InvalidArgument. Under Warn,
// and under Auto mid-run, emits one warning ending "The observation is
// omitted." and returns false. Under Ignore returns false silently.
bool respond(Response policy, RunPhase phase, std::string_view entry_point,
             std::string_view what, std::optional<FailurePoint> where = {});

// The failure path's access to Error::rethrow_saved.
struct FailurePathAccess {
    [[noreturn]] static void rethrow_saved(const Error& e, const std::filesystem::path& path) {
        e.rethrow_saved(path);
    }
};

}  // namespace detail
}  // namespace lindblad
