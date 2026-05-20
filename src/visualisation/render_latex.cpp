// =============================================================================
// src/visualisation/render_latex.cpp : Quantikz backend
// =============================================================================
// Emit a Quantikz environment per spec section 7.3. Quantikz takes a row-
// major matrix of cells separated by `&`; rows are separated by `\\`. Layout
// already groups glyphs by column, so emission is a double loop:
//
//   for each row r in [0..n_q-1] (plus optional c-wire row):
//     for each column c in [0..n_layers-1]:
//       emit the per-cell token for whatever Glyph touches (r, c)
//
// Per-row prefix is "\lstick{$q_{r}$}". For the bundled c-wire row we use
// "\lstick{$c$}" plus "\setwiretype{c}" so Quantikz draws the double line.
// Empty cells on qubit rows get "\qw"; on the c-wire row they get "\cw".
//
// Token mapping per spec section 7.3:
//   BoxPart            -> \gate{...} or \gate[N]{...} for tall boxes
//   CtrlBulletPart     -> \ctrl{offset} or \octrl{offset}
//   XorTargetPart      -> \targ{}
//   SwapXPart          -> \swap{offset} on the first slot, \targX{} on second
//   MeasurePart        -> \meter{}
//   ResetPart          -> \push{\ket{0}}
//   BarrierPart        -> \barrier[\dashed]{N} on the top barrier row
//
// Defensive escaping: gate labels containing characters that are not safe in
// math mode (parens, commas, non-ASCII pi, dagger, etc.) get wrapped in
// \text{...} so Quantikz does not silently mangle the output.

#include "render_latex.hpp"

#include "document.hpp"

#include "lindblad/circuit.hpp"

#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace lindblad::viz {

namespace {

// Decide whether a label needs \text{} wrapping for safety in math mode.
// Math mode is safe for ASCII letters, digits, and certain punctuation; the
// catalogue may produce labels with parens, commas, UTF-8 pi (0xCF 0x80), or
// the dagger character (UTF-8 0xE2 0x80 0xA0). Wrap when in doubt.
bool label_needs_text_wrap(const std::string& s) {
    for (unsigned char c : s) {
        // Anything non-ASCII => wrap.
        if (c >= 128) { return true; }
        // Parens / comma / spaces / arithmetic glyphs => wrap.
        switch (c) {
            case '(': case ')': case ',': case ' ':
            case '+': case '-': case '*': case '/':
                return true;
            default: break;
        }
    }
    return false;
}

// Render one BoxPart cell as a Quantikz token. The `rowspan` knob selects
// between \gate{} and \gate[N]{} (Quantikz multi-wire syntax).
std::string box_token(const BoxPart& p) {
    std::ostringstream out;
    if (p.rowspan > 1) {
        out << "\\gate[" << p.rowspan << "]{";
    } else {
        out << "\\gate{";
    }
    if (label_needs_text_wrap(p.label)) {
        out << "\\text{" << p.label << "}";
    } else {
        out << p.label;
    }
    out << "}";
    return out.str();
}

// Find the offset from `this_row` to the matching target row on the same
// glyph's strut. For a typical CX, the control sits on this_row=qubits[0]
// and the target on this_row=qubits[1]; we return target - this_row. For
// multi-target gates (CCX), we point to the nearest target row.
int find_offset_to_other(const Glyph& g, int this_row) {
    int best_offset = 0;
    bool found = false;
    for (const auto& [qrow, part] : g.parts) {
        (void)part;
        if (qrow == this_row) { continue; }
        int off = qrow - this_row;
        if (!found || std::abs(off) < std::abs(best_offset)) {
            best_offset = off;
            found       = true;
        }
    }
    // When the strut overrides the parts list (rare), prefer the strut's far
    // endpoint. Quantikz needs a non-zero offset to draw the connector.
    if (!found && g.has_strut) {
        int top = g.strut_top;
        int bot = g.strut_bot;
        if (this_row == top) { best_offset = bot - top; }
        else                 { best_offset = top - this_row; }
    }
    return best_offset;
}

// Build a 2D grid of Quantikz cell tokens. Outer index = row (qubit 0..n-1,
// optional c-wire at index n), inner index = column (one per layer).
// Default fill is "\qw" on qubit rows and "\cw" on the c-wire row; we
// overwrite cells touched by glyphs.

struct LatexGrid {
    int n_rows = 0;
    int n_cols = 0;
    std::vector<std::vector<std::string>> cells;

    void init(int rows, int cols, int n_qubits, bool show_clbits) {
        n_rows = rows;
        n_cols = cols;
        cells.assign(rows, std::vector<std::string>(cols, ""));
        for (int r = 0; r < rows; ++r) {
            const bool is_cwire = show_clbits && (r == n_qubits);
            const std::string fill = is_cwire ? "\\cw" : "\\qw";
            for (int c = 0; c < cols; ++c) {
                cells[r][c] = fill;
            }
        }
    }
};

} // anonymous namespace

// =============================================================================
// render_latex : public entry point
// =============================================================================

std::string render_latex(const CircuitDocument& doc, const DrawOptions& opts) {
    const int n_q     = doc.n_qubits;
    const bool show_c = opts.show_clbits;
    const int n_rows  = n_q + (show_c ? 1 : 0);
    const int n_cols  = static_cast<int>(doc.layers.size());

    LatexGrid grid;
    grid.init(n_rows, std::max(n_cols, 1), n_q, show_c);

    // ------------------------------------------------------------------------
    // Stamp each glyph's tokens into the grid. Layers are already in column-
    // major order from the layout pass; iterate by layer index for the
    // column coordinate.
    // ------------------------------------------------------------------------
    for (std::size_t li = 0; li < doc.layers.size(); ++li) {
        const Layer& L = doc.layers[li];
        const int col = static_cast<int>(li);
        for (const Glyph& g : L.glyphs) {
            // Determine if this is a full-width barrier (no parts but a
            // barrier-like glyph). We treat it as a no-op cell -- the
            // implementation here keeps the default "\qw" so the column
            // still exists; users wanting Quantikz \barrier{} on every row
            // can extend in a future patch.
            // Per-part token emission. We unpack g.parts via pair accessors
            // rather than structured bindings so the std::visit lambda can
            // capture the row index without tripping clang's "capturing a
            // structured binding is not yet supported in OpenMP" diagnostic.
            for (const auto& kv : g.parts) {
                const int qrow = kv.first;
                const GlyphPart& part = kv.second;
                std::string token;
                std::visit([&](auto&& p) {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, BoxPart>) {
                        token = box_token(p);
                    } else if constexpr (std::is_same_v<T, CtrlBulletPart>) {
                        int off = find_offset_to_other(g, qrow);
                        std::ostringstream s;
                        s << (p.anti ? "\\octrl{" : "\\ctrl{") << off << "}";
                        token = s.str();
                    } else if constexpr (std::is_same_v<T, XorTargetPart>) {
                        token = "\\targ{}";
                    } else if constexpr (std::is_same_v<T, SwapXPart>) {
                        // First swap arm in the parts list emits \swap{offset};
                        // the rest emit \targX{}. Detect "first arm" by
                        // checking whether qrow equals the smallest qrow with
                        // a SwapXPart in this glyph. Inner loop also uses
                        // pair accessors to avoid the structured-binding /
                        // OpenMP capture issue.
                        int first_swap_row = -1;
                        for (const auto& kv2 : g.parts) {
                            const int r2 = kv2.first;
                            const GlyphPart& p2 = kv2.second;
                            if (std::holds_alternative<SwapXPart>(p2)) {
                                if (first_swap_row < 0 || r2 < first_swap_row) {
                                    first_swap_row = r2;
                                }
                            }
                        }
                        if (qrow == first_swap_row) {
                            int off = find_offset_to_other(g, qrow);
                            std::ostringstream s;
                            s << "\\swap{" << off << "}";
                            token = s.str();
                        } else {
                            token = "\\targX{}";
                        }
                    } else if constexpr (std::is_same_v<T, MeasurePart>) {
                        token = "\\meter{}";
                    } else if constexpr (std::is_same_v<T, ResetPart>) {
                        token = "\\push{\\ket{0}}";
                    } else if constexpr (std::is_same_v<T, BarrierPart>) {
                        // Quantikz \barrier wants to be on the topmost row
                        // with rowspan = N. We emit it as a per-row dashed
                        // separator: a leading \qw kept, then \barrier on
                        // the topmost barrier row, others stay \qw.
                        // Smallest barrier-bearing row in this glyph:
                        int top_b = -1;
                        int bot_b = -1;
                        for (const auto& kv2 : g.parts) {
                            const int r2 = kv2.first;
                            const GlyphPart& p2 = kv2.second;
                            if (std::holds_alternative<BarrierPart>(p2)) {
                                if (top_b < 0 || r2 < top_b) { top_b = r2; }
                                if (bot_b < 0 || r2 > bot_b) { bot_b = r2; }
                            }
                        }
                        if (qrow == top_b) {
                            int span = (bot_b - top_b + 1);
                            std::ostringstream s;
                            s << "\\barrier[\\dashed]{" << span << "}";
                            token = s.str();
                        } else {
                            token = ""; // keep default \qw fill below
                        }
                    }
                }, part);

                if (!token.empty() && col >= 0 && col < grid.n_cols
                    && qrow >= 0 && qrow < grid.n_rows) {
                    grid.cells[qrow][col] = token;
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Emit the Quantikz body. lstick prefixes per row, & between cells, \\
    // between rows.
    // ------------------------------------------------------------------------
    std::ostringstream out;
    out << "\\begin{quantikz}\n";

    for (int r = 0; r < n_rows; ++r) {
        const bool is_cwire = show_c && (r == n_q);
        if (is_cwire) {
            out << "\\lstick{$c$} \\setwiretype{c} ";
        } else {
            out << "\\lstick{$q_{" << r << "}$} ";
        }
        for (int c = 0; c < grid.n_cols; ++c) {
            out << "& " << grid.cells[r][c] << " ";
        }
        // No `\\` after the last row.
        if (r + 1 < n_rows) {
            out << "\\\\\n";
        } else {
            out << "\n";
        }
    }

    out << "\\end{quantikz}\n";

    if (opts.include_legend) {
        out << "% legend: filled circle = control, open circle = anti-control, "
            << "X = CNOT target, x = SWAP arm, dashed line = barrier\n";
    }

    return out.str();
}

} // namespace lindblad::viz
