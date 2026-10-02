// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/detail/report.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/validation.hpp"

#include <cstddef>
#include <string>

namespace lindblad {
namespace detail {

std::string failure_message(std::string_view entry_point, std::string_view what,
                            const std::optional<FailurePoint>& where) {
    std::string out;
    if (!entry_point.empty()) {
        out.append(entry_point);
        out += ": ";
    }
    out.append(what);
    if (where) {
        if (where->instruction >= 0) {
            out += " (instruction " + std::to_string(where->instruction);
            if (!where->gate.empty()) out += ": " + where->gate;
            if (!where->qubits.empty()) {
                out += " on qubit";
                if (where->qubits.size() > 1) out += 's';
                for (std::size_t i = 0; i < where->qubits.size(); ++i) {
                    out += (i == 0 ? " " : ", ");
                    out += std::to_string(where->qubits[i]);
                }
            }
            out += ')';
        }
        if (where->shot >= 0) out += " at shot " + std::to_string(where->shot);
    }
    return out;
}

void raise_internal(std::string_view entry_point, std::string_view what,
                    std::optional<FailurePoint> where) {
    std::string text(what);
    if (!text.empty() && text.back() != '.') text += '.';
    text += " This is a defect in Lindblad, not in your code. Please report it at "
            "https://github.com/verycareful/lindblad/issues with this message.";
    raise<InternalError>(entry_point, text, std::move(where));
}

bool respond(Response policy, RunPhase phase, std::string_view entry_point,
             std::string_view what, std::optional<FailurePoint> where) {
    Response effective = policy;
    if (policy == Response::Auto) {
        effective = phase == RunPhase::BeforeFirstGate ? Response::Throw : Response::Warn;
    }
    switch (effective) {
        case Response::Throw:
            raise<InvalidArgument>(entry_point, what, std::move(where));
        case Response::Warn: {
            std::string text = failure_message(entry_point, what, where);
            if (!text.empty() && text.back() != '.') text += '.';
            emit_warning("note: " + text + " The observation is omitted.");
            return false;
        }
        case Response::Ignore:
        case Response::Auto:
            return false;
    }
    return false;
}

}  // namespace detail
}  // namespace lindblad
