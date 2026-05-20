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
