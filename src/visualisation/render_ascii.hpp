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
// src/visualisation/render_ascii.hpp : ASCII backend declaration
// =============================================================================
// One free function consumed by QuantumCircuit::draw() when the caller
// requests DrawMode::ASCII. The renderer walks a CircuitDocument produced by
// build_document() and emits a UTF-8 text grid. ASCII-safe fallback is
// driven entirely by DrawOptions::ascii_safe; the renderer picks the right
// palette per call.

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <string>

namespace lindblad::viz {

std::string render_ascii(const CircuitDocument& doc, const DrawOptions& opts);

} // namespace lindblad::viz
