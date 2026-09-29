// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lindblad {

namespace detail { struct FailurePathAccess; }

// =============================================================================
// FailurePoint - where a failure happened
// =============================================================================
// Filled when the failure belongs to one instruction or one shot. -1 in a
// field means the failure has no such coordinate: a refusal before the run has
// no shot, a register-wide refusal has no instruction.
struct FailurePoint {
    int shot = -1;
    int instruction = -1;
    std::string gate;         // empty when instruction == -1
    std::vector<int> qubits;  // the instruction's operands
};

// =============================================================================
// RunPhase - whether any work would be lost by stopping now
// =============================================================================
// BeforeFirstGate holds until the first instruction of the first shot has
// executed: stopping costs nothing. MidRun is everything after, including an
// anchor at the start of any shot after the first.
enum class RunPhase { BeforeFirstGate, MidRun };

// =============================================================================
// Error - what every exception Lindblad raises itself carries
// =============================================================================
// A mixin, deliberately NOT derived from std::exception. Each concrete type
// below already has one std::exception through its std base; a second would
// make catch (const std::exception&) ambiguous, and an ambiguous handler does
// not match at all.
//
// saved_to() is set when the run's partial results were saved to disk. The
// message then ends with the same path, since what() is fixed once a std
// exception exists; that is why the failure path throws a fresh copy through
// rethrow_saved instead of editing this one.
class Error {
public:
    virtual ~Error() = default;

    const std::string& entry_point() const noexcept { return entry_point_; }
    const std::optional<FailurePoint>& where() const noexcept { return where_; }
    const std::optional<std::filesystem::path>& saved_to() const noexcept {
        return saved_to_;
    }

protected:
    Error(std::string entry_point, std::optional<FailurePoint> where)
        : entry_point_(std::move(entry_point)), where_(std::move(where)) {}
    Error(const Error&) = default;
    Error& operator=(const Error&) = default;

    // Throws a copy of this object's dynamic type whose message ends with
    // "Partial results saved to <path>." and whose saved_to() is `path`.
    [[noreturn]] virtual void rethrow_saved(const std::filesystem::path& path) const = 0;

    // The message a saved copy carries: the original, then the path.
    static std::string saved_message(const char* what, const std::filesystem::path& path) {
        std::string out(what);
        if (!out.empty() && out.back() != '.') out += '.';
        out += " Partial results saved to " + path.string() + ".";
        return out;
    }

    std::optional<std::filesystem::path> saved_to_;

private:
    std::string entry_point_;
    std::optional<FailurePoint> where_;

    friend struct detail::FailurePathAccess;
};

// =============================================================================
// The four concrete types
// =============================================================================
// Each is also its std counterpart, so an existing catch of the std type keeps
// working. What each means:
//   InvalidArgument - a caller mistake or an exceeded limit, known before the
//                     first gate, or a refusal the caller's knob set to Throw
//   OutOfRange      - an index outside a register or a range
//   RuntimeFailure  - a failure met while running that makes the answer wrong
//   InternalError   - a defect in Lindblad; the message asks for a report

class InvalidArgument : public std::invalid_argument, public Error {
public:
    InvalidArgument(const std::string& message, std::string entry_point,
                    std::optional<FailurePoint> where = {})
        : std::invalid_argument(message), Error(std::move(entry_point), std::move(where)) {}

protected:
    [[noreturn]] void rethrow_saved(const std::filesystem::path& path) const override {
        InvalidArgument copy(saved_message(what(), path), entry_point(), where());
        copy.saved_to_ = path;
        throw copy;
    }
};

class OutOfRange : public std::out_of_range, public Error {
public:
    OutOfRange(const std::string& message, std::string entry_point,
               std::optional<FailurePoint> where = {})
        : std::out_of_range(message), Error(std::move(entry_point), std::move(where)) {}

protected:
    [[noreturn]] void rethrow_saved(const std::filesystem::path& path) const override {
        OutOfRange copy(saved_message(what(), path), entry_point(), where());
        copy.saved_to_ = path;
        throw copy;
    }
};

class RuntimeFailure : public std::runtime_error, public Error {
public:
    RuntimeFailure(const std::string& message, std::string entry_point,
                   std::optional<FailurePoint> where = {})
        : std::runtime_error(message), Error(std::move(entry_point), std::move(where)) {}

protected:
    [[noreturn]] void rethrow_saved(const std::filesystem::path& path) const override {
        RuntimeFailure copy(saved_message(what(), path), entry_point(), where());
        copy.saved_to_ = path;
        throw copy;
    }
};

class InternalError : public std::logic_error, public Error {
public:
    InternalError(const std::string& message, std::string entry_point,
                  std::optional<FailurePoint> where = {})
        : std::logic_error(message), Error(std::move(entry_point), std::move(where)) {}

protected:
    [[noreturn]] void rethrow_saved(const std::filesystem::path& path) const override {
        InternalError copy(saved_message(what(), path), entry_point(), where());
        copy.saved_to_ = path;
        throw copy;
    }
};

}  // namespace lindblad
