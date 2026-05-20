#pragma once

// =============================================================================
// src/visualisation/render_html.hpp : HTML backend declaration
// =============================================================================
// Standalone HTML page wrapping the SVG output. Adds page-level CSS that
// targets the SVG's data-* attributes for hover styling. No JavaScript.

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <string>

namespace lindblad::viz {

std::string render_html(const CircuitDocument& doc, const DrawOptions& opts);

} // namespace lindblad::viz
