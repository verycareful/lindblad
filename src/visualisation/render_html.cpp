// =============================================================================
// src/visualisation/render_html.cpp : HTML backend
// =============================================================================
// Wraps the SVG output (from render_svg) in a styled HTML page. The page-
// level CSS targets the SVG's existing .lb-glyph elements for hover effects
// without any JavaScript: `:hover .lb-gate { stroke-width: 2.5 }` is a pure
// CSS state change that browsers handle natively.
//
// The HTML page is meant to be saved to a `.html` file and opened in a
// browser. It carries the circuit name in <title> and a small <div class=
// "lb-meta"> caption summarising qubit count and layer count.

#include "render_html.hpp"
#include "render_svg.hpp"

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <sstream>
#include <string>

namespace lindblad::viz {

// =============================================================================
// render_html : public entry point
// =============================================================================

std::string render_html(const CircuitDocument& doc, const DrawOptions& opts) {
    const std::string svg = render_svg(doc, opts);

    std::ostringstream out;
    out << "<!DOCTYPE html>\n"
        << "<html lang=\"en\">\n"
        << "<head>\n"
        << "<meta charset=\"utf-8\">\n"
        << "<title>Lindblad circuit</title>\n"
        << "<style>\n"
        << "body { font-family: -apple-system, system-ui, sans-serif;\n"
        << "       margin: 2rem; background: #fafafa; }\n"
        << ".lb-circuit { display: inline-block; padding: 1rem;\n"
        << "              background: white; border: 1px solid #ddd;\n"
        << "              border-radius: 6px; }\n"
        << ".lb-meta { font-size: 13px; color: #666; margin-bottom: 0.5rem; }\n"
        << ".lb-glyph { cursor: default; }\n"
        << ".lb-glyph:hover .lb-gate { stroke-width: 2.5; }\n"
        << ".lb-legend { margin-top: 1rem; font-size: 13px; color: #444; }\n"
        << "</style>\n"
        << "</head>\n"
        << "<body>\n"
        << "<div class=\"lb-circuit\">\n"
        << "<div class=\"lb-meta\">"
        << doc.n_qubits << " qubits, "
        << doc.layers.size() << " layers";
    if (opts.show_clbits) {
        out << ", " << doc.n_clbits << " clbits";
    }
    out << "</div>\n"
        << svg;
    if (opts.include_legend) {
        out << "<div class=\"lb-legend\">"
            << "Filled circle = control, open circle = anti-control, "
            << "X = CNOT target, x = SWAP arm, dashed line = barrier."
            << "</div>\n";
    }
    out << "</div>\n"
        << "</body>\n"
        << "</html>\n";

    return out.str();
}

} // namespace lindblad::viz
