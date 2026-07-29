// =============================================================================
// src/visualisation/format_params.cpp : parameter and label formatting helpers
// =============================================================================
// Three helpers used by the Tier 1 / Tier 2 builders to assemble gate labels.
//
// Pretty mode snaps recognised rational multiples of pi to symbolic strings
// (e.g. "pi/2", "-3pi/4") within tolerance 1e-6 before falling back to "%.4f".
// Raw mode always emits "%.4f". The pi character is the UTF-8 U+03C0
// codepoint ("\xCF\x80") embedded directly into the output string.
//
// Symbolic expressions (ParamExpr) recursively format literals via the double
// overload, names pass through unchanged, and binary ops render as
// "lhs <op> rhs" with parens around child binary ops only when precedence
// demands. Multiplication renders as the UTF-8 middle dot U+00B7 ("\xC2\xB7")
// rather than ASCII "*" to keep labels visually clean inside boxes.
//
// format_gate_label assembles the final box label: GateSymbol::label followed
// by an optional parenthesised list of parameters when both the catalogue and
// the caller's DrawOptions agree that parameters are visible.

#include "format_params.hpp"

#include "gate_symbols.hpp"

#include "lindblad/circuit.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace lindblad::viz {

namespace {

// UTF-8 byte sequences for the two non-ASCII glyphs we embed in labels.
// Centralised here so a future palette tweak (e.g. switching to ASCII "pi")
// only edits one place.
constexpr const char* kPiUtf8     = "\xCF\x80"; // U+03C0 GREEK SMALL LETTER PI
constexpr const char* kMidDotUtf8 = "\xC2\xB7"; // U+00B7 MIDDLE DOT

// One row of the pi-snap table: factor * pi exactly representable as a tidy
// label. Sign is handled separately so the table only enumerates the positive
// magnitudes; negatives reuse the same string with a leading "-".
//
// Stored as (numerator, denominator) so the label assembler can choose between
// "pi", "pi/2", "3pi/4", "2pi" etc. without parsing floats. The double value
// is precomputed once at table-build time to keep the matcher branch-free.
struct PiSnapEntry {
    int    num;       // numerator
    int    den;       // denominator (1 = integer multiple)
    double value;     // num * pi / den
    const char* label_no_sign; // formatted with no leading "-"
};

// Build the positive half of the table. Negatives are handled by prepending
// "-" at match time. Tolerance for matching is 1e-6 (per the spec).
const std::array<PiSnapEntry, 14>& pi_snap_table() {
    static const std::array<PiSnapEntry, 14> table = { {
        // Integer multiples of pi.
        { 1, 1, 1.0          * PI,                     "pi"           },
        { 2, 1, 2.0          * PI,                     "2pi"          },
        { 3, 1, 3.0          * PI,                     "3pi"          },
        { 4, 1, 4.0          * PI,                     "4pi"          },
        // Half / third / quarter etc.
        { 3, 2, 3.0 / 2.0    * PI,                     "3pi/2"        },
        { 1, 2, 0.5          * PI,                     "pi/2"         },
        { 2, 3, 2.0 / 3.0    * PI,                     "2pi/3"        },
        { 1, 3, PI         / 3.0,                      "pi/3"         },
        { 3, 4, 3.0 * PI   / 4.0,                      "3pi/4"        },
        { 1, 4, PI         / 4.0,                      "pi/4"         },
        { 5, 6, 5.0 * PI   / 6.0,                      "5pi/6"        },
        { 1, 6, PI         / 6.0,                      "pi/6"         },
        { 1, 8, PI         / 8.0,                      "pi/8"         },
        // Zero anchor (formatted as "0", not "0pi").
        { 0, 1, 0.0,                                     "0"            },
    } };
    return table;
}

// Format a string with the pi character substituted for the literal "pi" so
// the label_no_sign entries above can be authored in ASCII (easier to read in
// source) while still printing the UTF-8 glyph downstream.
std::string substitute_pi(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ) {
        if (i + 1 < input.size() && input[i] == 'p' && input[i + 1] == 'i') {
            out += kPiUtf8;
            i += 2;
        } else {
            out += input[i];
            ++i;
        }
    }
    return out;
}

// Format a double via "%.4f" with no trailing-zero trimming. The fixed width
// keeps box widths consistent across columns; tests / golden files can rely
// on exact byte sequences.
std::string format_fixed_4(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4f", v);
    return std::string(buf);
}

// Recursive helper for ParamExpr that propagates a "needs parens around this
// subtree" flag based on parent precedence. Returns the rendered expression
// with no surrounding parens.
//
// Precedence:
//   '+' '-' : 1 (lowest)
//   '*' '/' : 2
//
// A child binary op renders with parens iff its precedence is strictly less
// than the parent's. Atomic nodes (literal, name) never need parens.
int op_precedence(char op) {
    switch (op) {
        case '*':
        case '/':
            return 2;
        case '+':
        case '-':
            return 1;
        default:
            return 0;
    }
}

std::string render_param_expr(const ParamExpr& e, ParamFormat fmt, int parent_prec) {
    switch (e.kind) {
        case ParamExpr::Kind::Literal:
            return format_param(e.literal, fmt);
        case ParamExpr::Kind::Name:
            return e.name;
        case ParamExpr::Kind::BinaryOp: {
            const int my_prec = op_precedence(e.op);

            std::string lhs = e.lhs ? render_param_expr(*e.lhs, fmt, my_prec)
                                    : std::string("?");
            std::string rhs = e.rhs ? render_param_expr(*e.rhs, fmt, my_prec)
                                    : std::string("?");

            // Use middle dot for multiplication; the rest pass through as ASCII.
            const char* op_str = (e.op == '*') ? kMidDotUtf8 : nullptr;
            char op_buf[2] = { e.op, '\0' };

            std::string combined = lhs;
            combined += (op_str ? op_str : op_buf);
            combined += rhs;

            // Wrap in parens iff our precedence is strictly less than the
            // parent's. Same precedence: no parens (left-associative default
            // is good enough for label readability).
            if (my_prec < parent_prec) {
                return "(" + combined + ")";
            }
            return combined;
        }
    }
    return std::string();
}

} // anonymous namespace

// =============================================================================
// format_param(double) : numeric -> string
// =============================================================================
// Pretty: scan the pi-snap table for a match within 1e-6, prepend "-" when
// the input is negative, return the table label with pi glyph substituted.
// Miss: fall back to "%.4f". Raw: always "%.4f". Zero is special-cased to
// "0" in Pretty mode for both signs.

std::string format_param(double v, ParamFormat fmt) {
    if (fmt == ParamFormat::Raw) {
        return format_fixed_4(v);
    }

    constexpr double kTol = 1e-6;

    // Zero anchor: cover both +0 and -0 to "0".
    if (std::fabs(v) < kTol) {
        return std::string("0");
    }

    const double mag  = std::fabs(v);
    const bool   neg  = (v < 0.0);

    for (const PiSnapEntry& entry : pi_snap_table()) {
        if (entry.value == 0.0) {
            continue; // already handled above
        }
        if (std::fabs(mag - entry.value) < kTol) {
            std::string out = substitute_pi(entry.label_no_sign);
            if (neg) {
                out.insert(out.begin(), '-');
            }
            return out;
        }
    }

    return format_fixed_4(v);
}

// =============================================================================
// format_param(const ParamExpr&) : symbolic expression -> string
// =============================================================================
// Recursive tree walk; parens added only where precedence demands them. The
// top-level call uses parent precedence 0 (lower than anything) so the
// outermost expression never gets wrapped.

std::string format_param(const ParamExpr& e, ParamFormat fmt) {
    return render_param_expr(e, fmt, 0);
}

// =============================================================================
// format_gate_label : assemble the final box label
// =============================================================================
// Returns sym.label unchanged when params are hidden (either the catalogue
// disables them or the caller did). When visible, append a parenthesised
// comma-separated list. Symbolic param_exprs take precedence over numeric
// params because QASM 3 input gates store the expression tree even after a
// binding pass; falling back to params keeps the legacy numeric path working.

std::string format_gate_label(const Instruction& inst,
                              const GateSymbol& sym,
                              const DrawOptions& opts) {
    if (!sym.show_params || !opts.show_params) {
        return sym.label;
    }

    const bool have_exprs   = !inst.param_exprs.empty();
    const bool have_numeric = !inst.params.empty();

    if (!have_exprs && !have_numeric) {
        return sym.label;
    }

    std::string out = sym.label;
    out += "(";

    if (have_exprs) {
        for (std::size_t i = 0; i < inst.param_exprs.size(); ++i) {
            if (i > 0) { out += ", "; }
            out += format_param(inst.param_exprs[i], opts.param_format);
        }
    } else {
        for (std::size_t i = 0; i < inst.params.size(); ++i) {
            if (i > 0) { out += ", "; }
            out += format_param(inst.params[i], opts.param_format);
        }
    }

    out += ")";
    return out;
}

} // namespace lindblad::viz
