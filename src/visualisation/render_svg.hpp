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
// src/visualisation/render_svg.hpp : SVG backend declaration
// =============================================================================
// Self-contained SVG (no external CSS, no external fonts). Inline <style>
// block plus per-element data attributes drive interactivity hooks for
// downstream consumers (lindblad-page, HTML wrapper).

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <string>

namespace lindblad::viz {

std::string render_svg(const CircuitDocument& doc, const DrawOptions& opts);

} // namespace lindblad::viz
