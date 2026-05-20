// =============================================================================
// src/visualisation/render_svg.cpp : SVG backend
// =============================================================================
// Emit a self-contained SVG document encoding the CircuitDocument. The output
// declares its UTF-8 encoding and inlines a small <style> block carrying the
// .lb-* class palette spec section 7.2. Every glyph is wrapped in a
// <g class="lb-glyph"> with data-gate / data-col / data-qubits attributes so
// downstream consumers (lindblad-page, HTML wrapper) can wire interactivity
// without re-parsing the SVG.
//
// Coordinate system:
//   x_centre(col) = prefix_px + cell_width  * (col + 0.5)
//   y_centre(q)   =            cell_height * (q   + 0.5)
//   y_centre(c)   =            cell_height * n_qubits + cell_height * 0.5
//
// prefix_px is sized so the row label ("q[0]:") fits on the left. The choice
// of cell_width / cell_height comes from DrawOptions; defaults are 48 px.
//
// SVG strings escape <, >, &, " in text content drawn from gate labels.

#include "render_svg.hpp"

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace lindblad::viz {

namespace {

// XML attribute / text content escaper. Only the five XML-significant
// characters need handling; the rest pass through unchanged (UTF-8 is fine in
// SVG text and attributes).
std::string xml_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '<': out += "&lt;";   break;
            case '>': out += "&gt;";   break;
            case '&': out += "&amp;";  break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += c;          break;
        }
    }
    return out;
}

// Approximate visual width in CSS pixels for a label. The font is monospace,
// so 8 px per ASCII character is a reasonable working estimate. UTF-8 multi-
// byte glyphs (pi, dagger) are counted as single visual columns: we split on
// UTF-8 codepoint boundaries.
int codepoint_count(const std::string& s) {
    int n = 0;
    for (std::size_t i = 0; i < s.size(); ) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t cp_len = 1;
        if ((c & 0x80) == 0x00)      cp_len = 1;
        else if ((c & 0xe0) == 0xc0) cp_len = 2;
        else if ((c & 0xf0) == 0xe0) cp_len = 3;
        else if ((c & 0xf8) == 0xf0) cp_len = 4;
        if (cp_len > s.size() - i) { cp_len = s.size() - i; }
        ++n;
        i += cp_len;
    }
    return n;
}

// Joined qubit-row list for the data-qubits attribute ("0,1" for a CX).
std::string join_qubits(const Glyph& g) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& [qrow, part] : g.parts) {
        (void)part;
        if (!first) { oss << ','; }
        oss << qrow;
        first = false;
    }
    return oss.str();
}

// =============================================================================
// Emit per-part SVG
// =============================================================================

void emit_box(std::ostringstream& out,
              double cx, double cy,
              double cell_w, double cell_h,
              const BoxPart& p) {
    // rect spans the cell horizontally with a small inset; vertically it
    // spans either one cell or `rowspan` cells.
    const double inset = 4.0;
    const double w = cell_w - 2.0 * inset;
    const double h = (p.rowspan > 0 ? p.rowspan : 1) * cell_h - 2.0 * inset;
    const double x = cx - w * 0.5;
    const double y = cy - cell_h * 0.5 + inset;

    const std::string fill   = p.svg_fill.empty()   ? "#f5f5f5" : p.svg_fill;
    const std::string stroke = p.svg_stroke.empty() ? "#444"    : p.svg_stroke;

    out << "<rect class=\"lb-gate\" x=\"" << x << "\" y=\"" << y
        << "\" width=\"" << w << "\" height=\"" << h
        << "\" rx=\"3\" ry=\"3\" fill=\"" << xml_escape(fill)
        << "\" stroke=\"" << xml_escape(stroke) << "\"/>";

    const double text_y = cy + (p.rowspan > 1 ? (p.rowspan - 1) * 0.5 * cell_h : 0.0);
    out << "<text class=\"lb-label\" x=\"" << cx << "\" y=\"" << text_y << "\">"
        << xml_escape(p.label) << "</text>";
}

void emit_ctrl(std::ostringstream& out, double cx, double cy, bool anti) {
    if (anti) {
        out << "<circle class=\"lb-ctrl\" cx=\"" << cx << "\" cy=\"" << cy
            << "\" r=\"5\" fill=\"white\" stroke=\"#444\" stroke-width=\"1.5\"/>";
    } else {
        out << "<circle class=\"lb-ctrl\" cx=\"" << cx << "\" cy=\"" << cy
            << "\" r=\"5\" fill=\"#444\"/>";
    }
}

void emit_xor_target(std::ostringstream& out, double cx, double cy) {
    const double r = 11.0;
    out << "<circle class=\"lb-gate\" cx=\"" << cx << "\" cy=\"" << cy
        << "\" r=\"" << r << "\" fill=\"white\" stroke=\"#444\" stroke-width=\"1.5\"/>";
    out << "<line class=\"lb-gate\" x1=\"" << (cx - r) << "\" y1=\"" << cy
        << "\" x2=\"" << (cx + r) << "\" y2=\"" << cy
        << "\" stroke=\"#444\" stroke-width=\"1.5\"/>";
    out << "<line class=\"lb-gate\" x1=\"" << cx << "\" y1=\"" << (cy - r)
        << "\" x2=\"" << cx << "\" y2=\"" << (cy + r)
        << "\" stroke=\"#444\" stroke-width=\"1.5\"/>";
}

void emit_swap_x(std::ostringstream& out, double cx, double cy) {
    const double s = 7.0;
    out << "<line class=\"lb-gate\" x1=\"" << (cx - s) << "\" y1=\"" << (cy - s)
        << "\" x2=\"" << (cx + s) << "\" y2=\"" << (cy + s)
        << "\" stroke=\"#444\" stroke-width=\"2\"/>";
    out << "<line class=\"lb-gate\" x1=\"" << (cx - s) << "\" y1=\"" << (cy + s)
        << "\" x2=\"" << (cx + s) << "\" y2=\"" << (cy - s)
        << "\" stroke=\"#444\" stroke-width=\"2\"/>";
}

void emit_measure(std::ostringstream& out, double cx, double cy,
                  double cell_w, double cell_h) {
    const double inset = 4.0;
    const double w = cell_w - 2.0 * inset;
    const double h = cell_h - 2.0 * inset;
    const double x = cx - w * 0.5;
    const double y = cy - h * 0.5;
    // Box.
    out << "<rect class=\"lb-gate\" x=\"" << x << "\" y=\"" << y
        << "\" width=\"" << w << "\" height=\"" << h
        << "\" rx=\"3\" ry=\"3\" fill=\"#fff\" stroke=\"#444\" stroke-width=\"1.5\"/>";
    // Arc (semicircle) suggesting the meter dial.
    const double r = h * 0.35;
    const double arc_y = cy + h * 0.1;
    out << "<path class=\"lb-gate\" d=\"M" << (cx - r) << "," << arc_y
        << " A" << r << "," << r << " 0 0 1 " << (cx + r) << "," << arc_y
        << "\" fill=\"none\" stroke=\"#444\" stroke-width=\"1.5\"/>";
    // Needle.
    out << "<line class=\"lb-gate\" x1=\"" << cx << "\" y1=\"" << arc_y
        << "\" x2=\"" << (cx + r * 0.7) << "\" y2=\"" << (arc_y - r * 0.9)
        << "\" stroke=\"#444\" stroke-width=\"1.5\"/>";
}

void emit_reset(std::ostringstream& out, double cx, double cy) {
    // The reset marker prints as a small "|0>" text fragment.
    out << "<text class=\"lb-label\" x=\"" << cx << "\" y=\"" << cy << "\">|0&#x27E9;</text>";
}

void emit_barrier(std::ostringstream& out, double cx, double y_top, double y_bot) {
    out << "<line class=\"lb-barrier\" x1=\"" << cx << "\" y1=\"" << y_top
        << "\" x2=\"" << cx << "\" y2=\"" << y_bot
        << "\" stroke=\"#888\" stroke-dasharray=\"2 3\" stroke-width=\"1.2\"/>";
}

} // anonymous namespace

// =============================================================================
// render_svg : public entry point
// =============================================================================

std::string render_svg(const CircuitDocument& doc, const DrawOptions& opts) {
    const double cell_w  = static_cast<double>(opts.cell_width_px);
    const double cell_h  = static_cast<double>(opts.cell_height_px);
    const int    n_q     = doc.n_qubits;
    const bool   show_c  = opts.show_clbits;
    const int    n_cols  = static_cast<int>(doc.layers.size());

    // Prefix width: enough for the longest row label. Compute from the longest
    // qubit label or "c:" assuming ~8 px per codepoint.
    int max_label_codepoints = 3; // "q[0]" minimum
    for (const auto& s : doc.qubit_labels) {
        int cp = codepoint_count(s);
        if (cp > max_label_codepoints) { max_label_codepoints = cp; }
    }
    if (show_c) {
        int cp = codepoint_count("c");
        if (cp > max_label_codepoints) { max_label_codepoints = cp; }
    }
    const double prefix_px = max_label_codepoints * 9.0 + 14.0; // characters + ": " padding

    const double svg_width  = prefix_px + n_cols * cell_w + 16.0;
    const double svg_height = (n_q + (show_c ? 1.0 : 0.0)) * cell_h + 16.0;

    auto y_qubit  = [&](int q) { return cell_h * (q + 0.5) + 8.0; };
    auto y_cwire  = [&]()       { return cell_h * (n_q + 0.5) + 8.0; };
    auto x_centre = [&](int c)  { return prefix_px + cell_w * (c + 0.5); };

    std::ostringstream out;
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
        << svg_width << " " << svg_height << "\">\n";

    // Inline style block per spec section 7.2.
    out << "<style>\n"
        << ".lb-wire    { stroke: #444; stroke-width: 1.5; fill: none; }\n"
        << ".lb-cwire   { stroke: #444; stroke-width: 1.5; fill: none; }\n"
        << ".lb-strut   { stroke: #444; stroke-width: 1.5; fill: none; }\n"
        << ".lb-gate    { stroke-width: 1.5; }\n"
        << ".lb-ctrl    { fill: #444; }\n"
        << ".lb-label   { font-family: monospace; font-size: 14px;\n"
        << "              text-anchor: middle; dominant-baseline: central;\n"
        << "              fill: #111; }\n"
        << ".lb-rowlabel{ font-family: monospace; font-size: 13px;\n"
        << "              dominant-baseline: central; fill: #444; }\n"
        << ".lb-barrier { stroke: #888; stroke-dasharray: 2 3; }\n"
        << "</style>\n";

    // ------------------------------------------------------------------------
    // Wires: one <line> per qubit + the bundled c-wire (if visible).
    // ------------------------------------------------------------------------
    const double wire_x0 = prefix_px - 4.0;
    const double wire_x1 = prefix_px + n_cols * cell_w + 4.0;

    out << "<g class=\"lb-wires\">\n";
    for (int q = 0; q < n_q; ++q) {
        out << "<line class=\"lb-wire\" x1=\"" << wire_x0
            << "\" y1=\"" << y_qubit(q)
            << "\" x2=\"" << wire_x1
            << "\" y2=\"" << y_qubit(q) << "\"/>\n";
    }
    if (show_c) {
        // Double line for the c-wire (two thin parallel strokes 2 px apart).
        out << "<line class=\"lb-cwire\" x1=\"" << wire_x0
            << "\" y1=\"" << (y_cwire() - 1.5)
            << "\" x2=\"" << wire_x1
            << "\" y2=\"" << (y_cwire() - 1.5) << "\"/>\n";
        out << "<line class=\"lb-cwire\" x1=\"" << wire_x0
            << "\" y1=\"" << (y_cwire() + 1.5)
            << "\" x2=\"" << wire_x1
            << "\" y2=\"" << (y_cwire() + 1.5) << "\"/>\n";
    }
    out << "</g>\n";

    // ------------------------------------------------------------------------
    // Row labels: prefix text aligned right of the prefix column.
    // ------------------------------------------------------------------------
    out << "<g class=\"lb-rowlabels\">\n";
    for (int q = 0; q < n_q; ++q) {
        std::string lbl = (q < static_cast<int>(doc.qubit_labels.size())
                           ? doc.qubit_labels[q]
                           : ("q[" + std::to_string(q) + "]"));
        out << "<text class=\"lb-rowlabel\" x=\"" << (prefix_px - 8.0)
            << "\" y=\"" << y_qubit(q) << "\" text-anchor=\"end\">"
            << xml_escape(lbl) << "</text>\n";
    }
    if (show_c) {
        out << "<text class=\"lb-rowlabel\" x=\"" << (prefix_px - 8.0)
            << "\" y=\"" << y_cwire() << "\" text-anchor=\"end\">c</text>\n";
    }
    out << "</g>\n";

    // ------------------------------------------------------------------------
    // Glyphs: one <g class="lb-glyph"> per Glyph, carrying data attributes.
    // ------------------------------------------------------------------------
    out << "<g class=\"lb-glyphs\">\n";
    for (const Layer& L : doc.layers) {
        for (const Glyph& g : L.glyphs) {
            out << "<g class=\"lb-glyph\" data-gate=\""
                << xml_escape(g.data_gate)
                << "\" data-col=\"" << g.column
                << "\" data-qubits=\"" << join_qubits(g) << "\">";

            const double cx = x_centre(g.column);

            // Strut first, so part marks sit on top of it.
            if (g.has_strut && g.strut_top != g.strut_bot) {
                const double y_top = y_qubit(g.strut_top);
                double y_bot = y_qubit(g.strut_bot);
                if (show_c && g.strut_bot >= n_q) {
                    y_bot = y_cwire();
                }
                out << "<line class=\"lb-strut\" x1=\"" << cx
                    << "\" y1=\"" << y_top
                    << "\" x2=\"" << cx
                    << "\" y2=\"" << y_bot << "\"/>";
            }

            // Conditional decoration: drop a strut to the c-wire row when
            // show_clbits is on. Tag "c[k]=v" near the c-wire.
            if (show_c && g.condition_clbit >= 0) {
                int top_qubit = n_q;
                for (const auto& [qrow, part] : g.parts) {
                    (void)part;
                    if (qrow < top_qubit) { top_qubit = qrow; }
                }
                if (top_qubit < n_q) {
                    out << "<line class=\"lb-strut\" x1=\"" << cx
                        << "\" y1=\"" << y_qubit(top_qubit)
                        << "\" x2=\"" << cx
                        << "\" y2=\"" << y_cwire() << "\"/>";
                }
                out << "<text class=\"lb-rowlabel\" x=\"" << (cx + 6.0)
                    << "\" y=\"" << y_cwire() << "\">c[" << g.condition_clbit
                    << "]=" << g.condition_value << "</text>";
            }

            // Per-part marks.
            for (const auto& [qrow, part] : g.parts) {
                const double cy = y_qubit(qrow);
                std::visit([&](auto&& p) {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, BoxPart>) {
                        emit_box(out, cx, cy, cell_w, cell_h, p);
                    } else if constexpr (std::is_same_v<T, CtrlBulletPart>) {
                        emit_ctrl(out, cx, cy, p.anti);
                    } else if constexpr (std::is_same_v<T, XorTargetPart>) {
                        emit_xor_target(out, cx, cy);
                    } else if constexpr (std::is_same_v<T, SwapXPart>) {
                        emit_swap_x(out, cx, cy);
                    } else if constexpr (std::is_same_v<T, MeasurePart>) {
                        emit_measure(out, cx, cy, cell_w, cell_h);
                    } else if constexpr (std::is_same_v<T, ResetPart>) {
                        emit_reset(out, cx, cy);
                    } else if constexpr (std::is_same_v<T, BarrierPart>) {
                        emit_barrier(out, cx, cy - cell_h * 0.5,
                                     cy + cell_h * 0.5);
                    }
                }, part);
            }

            out << "</g>\n";
        }
    }
    out << "</g>\n";

    out << "</svg>\n";
    return out.str();
}

} // namespace lindblad::viz
