#pragma once

// =============================================================================
// src/visualisation/render_latex.hpp : Quantikz LaTeX backend declaration
// =============================================================================
// Emits a `quantikz` environment (no document shell, no \documentclass) that
// the caller pastes inside their own LaTeX source. Quantikz is the modern
// TikZ-based dialect; the older `qcircuit` package is not supported.

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <string>

namespace lindblad::viz {

std::string render_latex(const CircuitDocument& doc, const DrawOptions& opts);

} // namespace lindblad::viz
