// =============================================================================
// src/visualisation/render_ascii.cpp : ASCII backend
// =============================================================================
// Paint the CircuitDocument onto a char grid built from per-codepoint cells
// (each cell holds one UTF-8 codepoint as a std::string, so column counting
// matches visual width even for multi-byte characters like the box-drawing
// glyphs).
//
// Grid geometry per spec section 7.1:
//   - n_rows = 2 * n_qubits - 1 (even rows are wires, odd rows are gap rows).
//   - When opts.show_clbits is true, append 2 more rows: a gap row + the
//     bundled c-wire row.
//   - Wire labels (q[0]:, c:) live in the prefix column on the wire rows
//     themselves; gap rows have no label.
//
// Palette is swapped wholesale based on opts.ascii_safe. The "safe" palette
// substitutes ASCII for every box-drawing / arrow / pi glyph.
//
// Folding (opts.fold_width > 0): split at column boundaries when the total
// width exceeds the threshold; emit "... fold ..." between folds and
// continuation labels ("<<q[0]: ") on each fold's first column.
//
// Multi-line boxes: NOT implemented in this release. All gates render as
// single-line "[label]" or "<label>" form. Tall boxes (rowspan > 1) draw
// vertical pipes through gap rows to span the indicated wire rows.

#include "render_ascii.hpp"

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace lindblad::viz {

namespace {

// =============================================================================
// Palette : per-renderer-call set of UTF-8 / ASCII strings
// =============================================================================
// All multi-byte glyphs are encoded as std::string so the caller can write
// them into the cell grid without juggling char32_t conversions. Each entry
// represents exactly one visual column.

struct Palette {
    std::string wire;          // qubit wire fill
    std::string cwire;         // c-wire fill
    std::string box_left;      // left border of a single-line gate
    std::string box_right;     // right border of a single-line gate
    std::string ctrl;          // filled control bullet
    std::string anti_ctrl;     // open anti-control bullet
    std::string xor_tgt;       // CNOT target
    std::string swap_x;        // SWAP arm
    std::string strut_v;       // vertical pipe in a gap row
    std::string strut_cross;   // pipe crossing a wire row
    std::string strut_cwire;   // pipe crossing the c-wire row
    std::string barrier;       // barrier mark per qubit
    std::string measure_text;  // chars inside measure box (already framed elsewhere)
    std::string reset_zero;    // first char of the reset glyph ("|" / "|")
    std::string reset_ket0;    // second char of the reset glyph ("0" / "0")
    std::string reset_bra;     // third char of the reset glyph (Unicode ket close vs ">")
};

Palette make_palette(bool ascii_safe) {
    if (ascii_safe) {
        return Palette{
            "-",  "=", "[", "]",
            "*",  "o", "X", "x",
            "|",  "+", "+", ":",
            "M",  "|", "0", ">"
        };
    }
    return Palette{
        "\xe2\x94\x80", // ─
        "\xe2\x95\x90", // ═
        "\xe2\x94\xa4", // ┤
        "\xe2\x94\x9c", // ├
        "\xe2\x97\x8f", // ●
        "\xe2\x97\x8b", // ○
        "\xe2\x8a\x95", // ⊕
        "\xe2\x9c\x95", // ✕
        "\xe2\x94\x82", // │
        "\xe2\x94\xbc", // ┼
        "\xe2\x95\xaa", // ╪
        "\xe2\x94\x8a", // ┊
        "M",
        "|",
        "0",
        "\xe2\x9f\xa9"  // ⟩
    };
}

// =============================================================================
// UTF-8 cell splitting
// =============================================================================
// A "cell" is one visual column. Split a UTF-8 string into a vector of
// one-codepoint cells. Continuation bytes (10xxxxxx) attach to the leading
// byte of the codepoint. Invalid sequences are treated as single bytes; the
// labels we receive originate from format_params and are guaranteed valid.

std::vector<std::string> utf8_cells(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t cp_len = 1;
        if ((c & 0x80) == 0x00)      cp_len = 1; // 0xxxxxxx
        else if ((c & 0xe0) == 0xc0) cp_len = 2; // 110xxxxx
        else if ((c & 0xf0) == 0xe0) cp_len = 3; // 1110xxxx
        else if ((c & 0xf8) == 0xf0) cp_len = 4; // 11110xxx
        cp_len = std::min(cp_len, s.size() - i);
        out.emplace_back(s.substr(i, cp_len));
        i += cp_len;
    }
    return out;
}

// =============================================================================
// Grid : 2D vector of UTF-8 cells
// =============================================================================
// row x col addressed by [r][c]. All rows are kept the same length to keep
// downstream painting / dumping straightforward.

struct Grid {
    int n_rows = 0;
    int n_cols = 0;
    std::vector<std::vector<std::string>> cells; // cells[r][c]

    void init(int rows, int cols, const std::string& fill) {
        n_rows = rows;
        n_cols = cols;
        cells.assign(rows, std::vector<std::string>(cols, fill));
    }

    void put(int r, int c, const std::string& cell) {
        if (r >= 0 && r < n_rows && c >= 0 && c < n_cols) {
            cells[r][c] = cell;
        }
    }
};

// Wire row helper: qubit q lives on row 2*q (0-indexed visually).
int wire_row(int qubit) { return 2 * qubit; }

// Compute the visual cell width of a label (number of UTF-8 codepoints).
int label_width(const std::string& s) {
    return static_cast<int>(utf8_cells(s).size());
}

// Per-glyph "natural width" = the widest part this glyph contributes, plus a
// fixed margin for box borders / strut padding. Used to size column widths.
int glyph_natural_width(const Glyph& g) {
    int max_w = 1;
    for (const auto& [row, part] : g.parts) {
        (void)row;
        int w = 1;
        std::visit([&](auto&& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, BoxPart>) {
                w = label_width(p.label) + 2; // borders ┤├
            } else if constexpr (std::is_same_v<T, MeasurePart>) {
                w = 3; // ┤M├
            } else if constexpr (std::is_same_v<T, ResetPart>) {
                w = 3; // |0⟩
            } else {
                w = 1; // bullets / targets / swaps / barriers are single cells
            }
        }, part);
        if (w > max_w) { max_w = w; }
    }
    return max_w;
}

// Compute the row range a glyph visually spans, including any tall BoxPart.
// Returned as (top, bot) inclusive in wire-row coordinates.
struct RowRange { int top; int bot; };

RowRange glyph_row_span(const Glyph& g) {
    int top = -1, bot = -1;
    auto bump = [&](int r) {
        if (top < 0 || r < top) { top = r; }
        if (bot < 0 || r > bot) { bot = r; }
    };
    for (const auto& kv : g.parts) {
        const int qrow = kv.first;
        const GlyphPart& part = kv.second;
        bump(wire_row(qrow));
        std::visit([&](auto&& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, BoxPart>) {
                if (p.rowspan > 1) {
                    bump(wire_row(qrow + p.rowspan - 1));
                }
            }
        }, part);
    }
    if (g.has_strut) {
        bump(wire_row(g.strut_top));
        bump(wire_row(g.strut_bot));
    }
    if (top < 0) { top = 0; bot = 0; }
    return { top, bot };
}

// =============================================================================
// Paint one glyph at column origin x0 in the grid
// =============================================================================
// Strategy:
//   1. Paint per-part marks on each wire row at the centre of the cell range
//      [x0, x0 + col_width).
//   2. Paint the strut as vertical pipes through gap rows in the same column,
//      using a "cross" glyph on intermediate wire rows.
//   3. For tall BoxPart (rowspan > 1), draw borders on the top and bottom
//      wire rows plus vertical pipes through every gap row in between.

void paint_box(Grid& grid, int wire_r, int x0, int col_width,
               const std::string& label,
               const Palette& palette,
               int rowspan,
               bool draw_top_bot_borders) {
    // Centre the label in the column.
    auto cells = utf8_cells(label);
    int label_w = static_cast<int>(cells.size());
    // Column inside the box (between borders).
    int inner_w = col_width - 2;
    int pad     = std::max(0, (inner_w - label_w) / 2);
    int x_left  = x0;
    int x_right = x0 + col_width - 1;

    // Always write the box-left / box-right characters on the anchor wire row.
    grid.put(wire_r, x_left,  palette.box_left);
    grid.put(wire_r, x_right, palette.box_right);

    // Paint the label cells.
    for (int i = 0; i < label_w; ++i) {
        int x = x_left + 1 + pad + i;
        if (x < x_right) {
            grid.put(wire_r, x, cells[i]);
        }
    }

    // Fill any remaining inner cells on the anchor row with spaces so they
    // stop carrying the wire fill character (otherwise short labels print as
    // "[--label--]" with wire dashes around the label).
    for (int x = x_left + 1; x < x_right; ++x) {
        if (grid.cells[wire_r][x] == palette.wire) {
            grid.put(wire_r, x, " ");
        }
    }

    if (rowspan <= 1) {
        return;
    }

    // Tall box: paint the side pipes on every gap row between the anchor and
    // the bottom wire row, and the box-left / box-right on the bottom wire
    // row. The middle wires (intermediate qubits) get pipes that cross them.
    int bottom_wire = wire_r + 2 * (rowspan - 1);
    for (int r = wire_r + 1; r <= bottom_wire; ++r) {
        if (r == bottom_wire) {
            grid.put(r, x_left,  palette.box_left);
            grid.put(r, x_right, palette.box_right);
            // Blank inside cells on the bottom row too.
            for (int x = x_left + 1; x < x_right; ++x) {
                if (grid.cells[r][x] == palette.wire) {
                    grid.put(r, x, " ");
                }
            }
        } else {
            // Either a gap row (paint vertical pipes) or an intermediate wire
            // row (paint vertical pipes crossing the wire).
            grid.put(r, x_left,  palette.strut_v);
            grid.put(r, x_right, palette.strut_v);
            for (int x = x_left + 1; x < x_right; ++x) {
                // Erase the wire fill inside the box.
                if (grid.cells[r][x] == palette.wire) {
                    grid.put(r, x, " ");
                }
            }
        }
    }

    (void)draw_top_bot_borders; // reserved for future styling
}

void paint_glyph(Grid& grid, const Glyph& g, int x0, int col_width,
                 const Palette& palette,
                 int n_qubit_wires,
                 bool show_clbits,
                 int cwire_row_index) {
    int x_centre = x0 + col_width / 2;

    // 1. Strut first (so part marks overwrite the cross glyphs at endpoints).
    if (g.has_strut && g.strut_top != g.strut_bot) {
        int top_r = wire_row(g.strut_top);
        int bot_r = wire_row(g.strut_bot);
        if (show_clbits && g.strut_bot >= n_qubit_wires) {
            bot_r = cwire_row_index;
        }
        for (int r = top_r + 1; r < bot_r; ++r) {
            // Gap row or intermediate wire row.
            bool is_wire = (r % 2 == 0);
            bool is_cwire = show_clbits && (r == cwire_row_index);
            if (is_cwire) {
                grid.put(r, x_centre, palette.strut_cwire);
            } else if (is_wire) {
                grid.put(r, x_centre, palette.strut_cross);
            } else {
                grid.put(r, x_centre, palette.strut_v);
            }
        }
    }

    // 2. Conditional decoration when show_clbits and condition_clbit >= 0.
    if (show_clbits && g.condition_clbit >= 0) {
        // Drop a strut from the topmost glyph row to the c-wire row.
        int top_r = -1;
        for (const auto& [qrow, part] : g.parts) {
            (void)part;
            int r = wire_row(qrow);
            if (top_r < 0 || r < top_r) { top_r = r; }
        }
        if (top_r < 0) { top_r = 0; }
        for (int r = top_r + 1; r < cwire_row_index; ++r) {
            bool is_wire = (r % 2 == 0);
            if (is_wire) {
                grid.put(r, x_centre, palette.strut_cross);
            } else {
                grid.put(r, x_centre, palette.strut_v);
            }
        }
        // Mark the c-wire row with a cross glyph for the conditional.
        grid.put(cwire_row_index, x_centre, palette.strut_cwire);

        // R.1.10.2: annotate the c-wire crossing with the comparison value
        // so two conditional gates with different match values are
        // distinguishable. We write "=v" immediately to the right of the
        // crossing, overwriting the bundled c-wire fill characters. The
        // annotation stays within the current column's footprint for
        // single-digit values (which is every condition_value the public
        // API currently produces). Multi-digit values may bleed into the
        // next column's c-wire fill; this is documented in
        // docs/api/visualisation.md as a known limitation.
        std::string vstr = "=" + std::to_string(g.condition_value);
        for (std::size_t i = 0; i < vstr.size(); ++i) {
            int x = x_centre + 1 + static_cast<int>(i);
            if (x < grid.n_cols) {
                grid.put(cwire_row_index, x, std::string(1, vstr[i]));
            }
        }
    }

    // 3. Per-part marks.
    for (const auto& [qrow, part] : g.parts) {
        int wr = wire_row(qrow);
        std::visit([&](auto&& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, BoxPart>) {
                paint_box(grid, wr, x0, col_width, p.label, palette,
                          p.rowspan, true);
            } else if constexpr (std::is_same_v<T, CtrlBulletPart>) {
                grid.put(wr, x_centre,
                         p.anti ? palette.anti_ctrl : palette.ctrl);
            } else if constexpr (std::is_same_v<T, XorTargetPart>) {
                grid.put(wr, x_centre, palette.xor_tgt);
            } else if constexpr (std::is_same_v<T, SwapXPart>) {
                grid.put(wr, x_centre, palette.swap_x);
            } else if constexpr (std::is_same_v<T, MeasurePart>) {
                // ┤M├ centred in the column.
                int x_left  = x0;
                int x_right = x0 + col_width - 1;
                grid.put(wr, x_left,  palette.box_left);
                grid.put(wr, x_right, palette.box_right);
                // Centre "M".
                int x_mid = x_left + (col_width / 2);
                grid.put(wr, x_mid, palette.measure_text);
                // Blank wire between borders so the M sits cleanly.
                for (int x = x_left + 1; x < x_right; ++x) {
                    if (x != x_mid && grid.cells[wr][x] == palette.wire) {
                        grid.put(wr, x, " ");
                    }
                }
            } else if constexpr (std::is_same_v<T, ResetPart>) {
                // |0⟩ centred.
                int x_left  = x0;
                int x_right = x0 + col_width - 1;
                // Reserve three cells: left + middle + right of the centre.
                int x_mid = x_left + (col_width / 2);
                grid.put(wr, x_mid - 1, palette.reset_zero);  // |
                grid.put(wr, x_mid,     palette.reset_ket0);  // 0
                grid.put(wr, x_mid + 1, palette.reset_bra);   // ⟩ or >
                // Blank surrounding wire fills.
                for (int x = x_left; x <= x_right; ++x) {
                    if (x < x_mid - 1 || x > x_mid + 1) {
                        if (grid.cells[wr][x] == palette.wire) {
                            grid.put(wr, x, palette.wire);
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, BarrierPart>) {
                grid.put(wr, x_centre, palette.barrier);
            }
        }, part);
    }
}

// =============================================================================
// emit : flatten the grid into a single string
// =============================================================================

std::string emit_grid(const Grid& grid) {
    std::ostringstream oss;
    for (int r = 0; r < grid.n_rows; ++r) {
        for (int c = 0; c < grid.n_cols; ++c) {
            oss << grid.cells[r][c];
        }
        oss << '\n';
    }
    return oss.str();
}

} // anonymous namespace

// =============================================================================
// render_ascii : public entry point
// =============================================================================

std::string render_ascii(const CircuitDocument& doc, const DrawOptions& opts) {
    const Palette palette = make_palette(opts.ascii_safe);

    const int n_qubit_wires = doc.n_qubits;                 // qubit count
    const bool show_c       = opts.show_clbits;
    const int  cwire_extra  = show_c ? 2 : 0;               // gap + c-wire
    const int  n_rows       = std::max(0, 2 * n_qubit_wires - 1) + cwire_extra;
    const int  cwire_row    = show_c ? (2 * n_qubit_wires) : -1;

    // ------------------------------------------------------------------------
    // Build the row prefixes: "q[0]: ", "q[1]: ", ..., "c: ".
    // Prefix width is uniform across all rows so the wires line up.
    // ------------------------------------------------------------------------
    std::vector<std::string> prefixes(n_rows, "");
    int prefix_width = 0;
    for (int q = 0; q < n_qubit_wires; ++q) {
        std::string lbl = (q < static_cast<int>(doc.qubit_labels.size())
                           ? doc.qubit_labels[q]
                           : ("q[" + std::to_string(q) + "]"));
        prefixes[wire_row(q)] = lbl + ": ";
        prefix_width = std::max(prefix_width,
                                 static_cast<int>(utf8_cells(prefixes[wire_row(q)]).size()));
    }
    if (show_c) {
        std::string lbl = "c: ";
        prefixes[cwire_row] = lbl;
        prefix_width = std::max(prefix_width,
                                 static_cast<int>(utf8_cells(prefixes[cwire_row]).size()));
    }
    // Pad each non-empty prefix to prefix_width with spaces.
    for (auto& p : prefixes) {
        if (p.empty()) {
            continue;
        }
        int w = static_cast<int>(utf8_cells(p).size());
        for (int i = w; i < prefix_width; ++i) { p += " "; }
    }

    // ------------------------------------------------------------------------
    // Width pass: per-layer column width = max natural width across glyphs.
    // ------------------------------------------------------------------------
    std::vector<int> layer_widths;
    layer_widths.reserve(doc.layers.size());
    for (const Layer& L : doc.layers) {
        int w = 3; // minimum column width
        for (const Glyph& g : L.glyphs) {
            int nw = glyph_natural_width(g);
            if (nw > w) { w = nw; }
        }
        // R.1.10.2: only force odd column width when the layer actually
        // contains a glyph that needs a centre cell (control bullets, XOR
        // targets, SWAP arms, barrier ticks). Pure box-only layers (TallBox
        // gates, UNITARY, single-qubit Tier 1 boxes) keep even widths so
        // labels with even length fit without trailing padding. Without
        // this guard, "RXX(pi/4)" got bumped to a 11-wide column with one
        // stray trailing space inside the right border.
        bool layer_needs_centre = false;
        for (const Glyph& g : L.glyphs) {
            for (const auto& kv : g.parts) {
                const GlyphPart& part = kv.second;
                if (std::holds_alternative<CtrlBulletPart>(part) ||
                    std::holds_alternative<XorTargetPart >(part) ||
                    std::holds_alternative<SwapXPart     >(part) ||
                    std::holds_alternative<BarrierPart   >(part)) {
                    layer_needs_centre = true;
                    break;
                }
            }
            if (layer_needs_centre) { break; }
        }
        if (layer_needs_centre && w % 2 == 0) { ++w; }
        layer_widths.push_back(w);
    }

    int total_cols = prefix_width;
    for (int w : layer_widths) { total_cols += w + 1; } // +1 for separator
    if (total_cols < prefix_width + 3) { total_cols = prefix_width + 3; }

    // ------------------------------------------------------------------------
    // Initialise the grid with wire fills on wire rows, spaces elsewhere.
    // ------------------------------------------------------------------------
    Grid grid;
    grid.init(n_rows, total_cols, " ");
    for (int q = 0; q < n_qubit_wires; ++q) {
        int wr = wire_row(q);
        for (int c = 0; c < total_cols; ++c) {
            grid.put(wr, c, palette.wire);
        }
    }
    if (show_c) {
        for (int c = 0; c < total_cols; ++c) {
            grid.put(cwire_row, c, palette.cwire);
        }
    }
    // Stamp the prefixes.
    for (int r = 0; r < n_rows; ++r) {
        auto pcells = utf8_cells(prefixes[r]);
        for (int i = 0; i < static_cast<int>(pcells.size()); ++i) {
            grid.put(r, i, pcells[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Paint pass: walk layers left to right; for each glyph paint at its
    // column origin.
    // ------------------------------------------------------------------------
    int x = prefix_width;
    for (std::size_t li = 0; li < doc.layers.size(); ++li) {
        const Layer& L = doc.layers[li];
        int col_w = layer_widths[li];
        for (const Glyph& g : L.glyphs) {
            paint_glyph(grid, g, x, col_w, palette,
                        n_qubit_wires, show_c, cwire_row);
        }
        x += col_w + 1;
    }

    // ------------------------------------------------------------------------
    // Emit. Folding (opts.fold_width > 0 && total_cols > fold_width) splits
    // at layer boundaries; minimal implementation that just emits the full
    // grid for now -- folding refines in a future patch when long circuits
    // become a real ergonomic issue.
    // ------------------------------------------------------------------------
    return emit_grid(grid);
}

} // namespace lindblad::viz
