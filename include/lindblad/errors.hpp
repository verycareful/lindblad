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

namespace detail {

// What a failed run's failure path adds to the copy of a Lindblad exception it
// rethrows: the folder a save wrote and a sentence naming what that folder
// does not hold, and, for an exception that named no position of its own,
// where the run was together with that position as a message words it
// (" (instruction 4: cx on qubits 0, 1) at shot 7").
struct Amendment {
    std::optional<std::filesystem::path> saved_to;
    std::string unsaved;
    std::optional<FailurePoint> where;
    std::string position;
};

}  // namespace detail

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
// rethrow_amended instead of editing this one.
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

    // Throws a copy of this object's dynamic type carrying `a`: its where()
    // is a.where when set and this one's otherwise, its saved_to() is
    // a.saved_to, and its message is amended_message(what(), a).
    [[noreturn]] virtual void rethrow_amended(const detail::Amendment& a) const = 0;

    // The original message, then the position when the original named none,
    // then "Partial results saved to <path>." and what the folder does not
    // hold, when a save was made.
    static std::string amended_message(const char* what, const detail::Amendment& a) {
        std::string out(what);
        if (!a.position.empty()) {
            if (!out.empty() && out.back() == '.') out.pop_back();
            out += a.position;
        }
        if (a.saved_to) {
            if (!out.empty() && out.back() != '.') out += '.';
            out += " Partial results saved to " + a.saved_to->string() + ".";
            if (!a.unsaved.empty()) out += " " + a.unsaved;
        }
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
    [[noreturn]] void rethrow_amended(const detail::Amendment& a) const override {
        InvalidArgument copy(amended_message(what(), a), entry_point(), a.where ? a.where : where());
        copy.saved_to_ = a.saved_to;
        throw copy;
    }
};

class OutOfRange : public std::out_of_range, public Error {
public:
    OutOfRange(const std::string& message, std::string entry_point,
               std::optional<FailurePoint> where = {})
        : std::out_of_range(message), Error(std::move(entry_point), std::move(where)) {}

protected:
    [[noreturn]] void rethrow_amended(const detail::Amendment& a) const override {
        OutOfRange copy(amended_message(what(), a), entry_point(), a.where ? a.where : where());
        copy.saved_to_ = a.saved_to;
        throw copy;
    }
};

class RuntimeFailure : public std::runtime_error, public Error {
public:
    RuntimeFailure(const std::string& message, std::string entry_point,
                   std::optional<FailurePoint> where = {})
        : std::runtime_error(message), Error(std::move(entry_point), std::move(where)) {}

protected:
    [[noreturn]] void rethrow_amended(const detail::Amendment& a) const override {
        RuntimeFailure copy(amended_message(what(), a), entry_point(), a.where ? a.where : where());
        copy.saved_to_ = a.saved_to;
        throw copy;
    }
};

class InternalError : public std::logic_error, public Error {
public:
    InternalError(const std::string& message, std::string entry_point,
                  std::optional<FailurePoint> where = {})
        : std::logic_error(message), Error(std::move(entry_point), std::move(where)) {}

protected:
    [[noreturn]] void rethrow_amended(const detail::Amendment& a) const override {
        InternalError copy(amended_message(what(), a), entry_point(), a.where ? a.where : where());
        copy.saved_to_ = a.saved_to;
        throw copy;
    }
};

}  // namespace lindblad
