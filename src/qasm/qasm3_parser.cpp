// =============================================================================
// qasm3_parser.cpp — OpenQASM 3.0 parser
// =============================================================================
// Production-grade tokenizer + recursive-descent parser covering the full
// Qiskit-exportable QASM 3 dialect:
//   • multiple named qubit / bit registers
//   • gate modifiers: ctrl @, inv @, pow(n) @, with chained composition
//   • standard gate library (`stdgates.inc`)
//   • user-defined `gate name(params) qargs { body }` with recursive inlining
//   • classical `if (c[i] == V) ...` (with optional `else`)
//   • `measure`, `reset`, `barrier`
//   • symbolic `input float[N] name;` parameters → ParamExpr
//   • parse-time peephole cancellation of self-inverse pairs and pow(0) gates
//
// Design notes:
//   • The lexer keeps tokens as std::string_view into the original source —
//     zero copies on the hot path.
//   • The token vector is pre-reserved at source.size()/5 (empirically ~5
//     chars/token for QASM 3).
//   • Built-in gate dispatch goes through a single static unordered_map
//     keyed on the gate name string_view — O(1) average lookup.
//   • Modifier resolution prefers named fast paths (`ctrl @ x` → cx,
//     `inv @ rx(θ)` → rx(-θ), `pow(n) @ rx(θ)` → rx(n·θ)). Combinations that
//     don't map to a named gate fall back to explicit matrix composition
//     (build base 2^k matrix → apply inv as conjugate-transpose → apply
//     pow(n) by repeated multiply → apply ctrl as block-diagonal extension)
//     and emit a UNITARY instruction.
//   • Peephole window: a per-qubit history stack of live instruction indices.
//     After every emission we check the most recent neighbour on the same
//     exact qubit list and cancel both if the gate is self-inverse with no
//     parameters. Cancelled instructions are marked with a `cancelled_` flag
//     and stripped in a final sweep, so mid-vector erases don't invalidate
//     other indices.

#include "lindblad/circuit.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lindblad {

namespace {

// =============================================================================
// Token model
// =============================================================================

enum class TT {
    KEYWORD, IDENT, INT, FLOAT, STRING,
    LBRACKET, RBRACKET, LPAREN, RPAREN, LBRACE, RBRACE,
    SEMICOLON, COMMA, EQUALS, EQEQ, ARROW,
    AT, STAR, SLASH, PLUS, MINUS,
    END
};

struct Token {
    TT type;
    std::string_view text;
    int line;
};

// =============================================================================
// QASM3Lexer — single-pass, zero-copy tokenizer
// =============================================================================
// Emits a flat std::vector<Token>. Tokens hold string_views into the source,
// so the caller must keep the source string alive for the parser's lifetime.

class QASM3Lexer {
public:
    static std::vector<Token> tokenize(std::string_view src);

private:
    static bool is_digit(unsigned char c) noexcept { return c >= '0' && c <= '9'; }
    static bool is_ident_start(unsigned char c) noexcept {
        // Accept ASCII letters, underscore, and any non-ASCII byte (UTF-8
        // continuation/leading bytes are allowed inside identifiers so Greek
        // letters in parameter names — `θ`, `φ` — round-trip cleanly).
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c >= 0x80;
    }
    static bool is_ident_cont(unsigned char c) noexcept {
        return is_ident_start(c) || is_digit(c);
    }
    static bool is_keyword(std::string_view s) noexcept;
};

bool QASM3Lexer::is_keyword(std::string_view s) noexcept {
    // We treat the QASM 3 reserved words as KEYWORD so the parser can
    // distinguish them from user identifiers without a second pass. Some
    // QASM 2 leftovers (`qreg`, `creg`) are tolerated since Qiskit's exporter
    // occasionally emits them when the user constructs a circuit with the
    // legacy ClassicalRegister type.
    static const std::unordered_set<std::string_view> kw = {
        "OPENQASM", "include", "qubit", "bit", "gate", "if", "else",
        "ctrl", "negctrl", "inv", "pow", "measure", "reset", "barrier",
        "input", "output", "const", "let", "for", "while", "def",
        "delay", "stretch", "box", "duration", "cal", "defcal",
        "qreg", "creg",
        "true", "false",
        "float", "int", "uint", "complex", "bool", "bit_t", "angle"
    };
    return kw.count(s) > 0;
}

std::vector<Token> QASM3Lexer::tokenize(std::string_view src) {
    std::vector<Token> out;
    // Empirical heuristic: QASM 3 averages ~5 source characters per token
    // (gate names + brackets + qubit indices). Pre-reserving avoids ~10
    // reallocations on a typical 4-kqubit circuit.
    out.reserve(src.size() / 5 + 16);

    int line = 1;
    size_t i = 0;

    auto push = [&](TT t, size_t start, size_t end) {
        out.push_back({t, src.substr(start, end - start), line});
    };

    while (i < src.size()) {
        unsigned char c = static_cast<unsigned char>(src[i]);

        // Whitespace
        if (c == ' ' || c == '\t' || c == '\r') { ++i; continue; }
        if (c == '\n') { ++line; ++i; continue; }

        // Comments
        if (c == '/' && i + 1 < src.size()) {
            if (src[i + 1] == '/') {
                while (i < src.size() && src[i] != '\n') ++i;
                continue;
            }
            if (src[i + 1] == '*') {
                i += 2;
                while (i + 1 < src.size()) {
                    if (src[i] == '*' && src[i + 1] == '/') { i += 2; break; }
                    if (src[i] == '\n') ++line;
                    ++i;
                }
                continue;
            }
        }

        // Multi-char operators (must come before single-char checks)
        if (c == '-' && i + 1 < src.size() && src[i + 1] == '>') {
            push(TT::ARROW, i, i + 2); i += 2; continue;
        }
        if (c == '=' && i + 1 < src.size() && src[i + 1] == '=') {
            push(TT::EQEQ, i, i + 2); i += 2; continue;
        }

        // Single-char punctuation
        switch (c) {
            case '[': push(TT::LBRACKET,  i, i + 1); ++i; continue;
            case ']': push(TT::RBRACKET,  i, i + 1); ++i; continue;
            case '(': push(TT::LPAREN,    i, i + 1); ++i; continue;
            case ')': push(TT::RPAREN,    i, i + 1); ++i; continue;
            case '{': push(TT::LBRACE,    i, i + 1); ++i; continue;
            case '}': push(TT::RBRACE,    i, i + 1); ++i; continue;
            case ';': push(TT::SEMICOLON, i, i + 1); ++i; continue;
            case ',': push(TT::COMMA,     i, i + 1); ++i; continue;
            case '@': push(TT::AT,        i, i + 1); ++i; continue;
            case '*': push(TT::STAR,      i, i + 1); ++i; continue;
            case '/': push(TT::SLASH,     i, i + 1); ++i; continue;
            case '+': push(TT::PLUS,      i, i + 1); ++i; continue;
            case '-': push(TT::MINUS,     i, i + 1); ++i; continue;
            case '=': push(TT::EQUALS,    i, i + 1); ++i; continue;
            default: break;
        }

        // String literal — only used by `include "stdgates.inc";`
        if (c == '"') {
            size_t start = i + 1;
            ++i;
            while (i < src.size() && src[i] != '"') {
                if (src[i] == '\n') ++line;
                ++i;
            }
            push(TT::STRING, start, i);
            if (i < src.size()) ++i;
            continue;
        }

        // Numeric literal — INT or FLOAT, with optional exponent
        if (is_digit(c) ||
            (c == '.' && i + 1 < src.size() && is_digit(static_cast<unsigned char>(src[i + 1])))) {
            size_t start = i;
            bool is_float = (c == '.');
            if (is_float) ++i;
            while (i < src.size()) {
                unsigned char cc = static_cast<unsigned char>(src[i]);
                if (is_digit(cc)) { ++i; }
                else if (cc == '.' && !is_float) { is_float = true; ++i; }
                else if (cc == 'e' || cc == 'E') {
                    is_float = true; ++i;
                    if (i < src.size() && (src[i] == '+' || src[i] == '-')) ++i;
                    while (i < src.size() && is_digit(static_cast<unsigned char>(src[i]))) ++i;
                    break;
                } else {
                    break;
                }
            }
            push(is_float ? TT::FLOAT : TT::INT, start, i);
            continue;
        }

        // Identifier or keyword
        if (is_ident_start(c)) {
            size_t start = i;
            ++i;
            while (i < src.size() && is_ident_cont(static_cast<unsigned char>(src[i]))) ++i;
            auto text = src.substr(start, i - start);
            push(is_keyword(text) ? TT::KEYWORD : TT::IDENT, start, i);
            continue;
        }

        throw std::runtime_error(
            "QASM3Lexer: unexpected character '" +
            std::string(1, static_cast<char>(c)) +
            "' at line " + std::to_string(line));
    }

    out.push_back({TT::END, std::string_view{}, line});
    return out;
}

// =============================================================================
// Built-in gate dispatch table
// =============================================================================
// Maps a stdgates name to (n_params_expected, n_qubits_expected, GateType).
// Kept as a static unordered_map for O(1) average lookup, replacing the long
// if-else chain of the QASM 2 parser.

struct BuiltinSpec {
    int n_params;
    int n_qubits;
    Instruction::GateType type;
};

const std::unordered_map<std::string_view, BuiltinSpec>& builtin_table() {
    using GT = Instruction::GateType;
    static const std::unordered_map<std::string_view, BuiltinSpec> t = {
        // 1-qubit, 0-param
        {"h",     {0, 1, GT::H}},   {"x",    {0, 1, GT::X}},
        {"y",     {0, 1, GT::Y}},   {"z",    {0, 1, GT::Z}},
        {"s",     {0, 1, GT::S}},   {"sdg",  {0, 1, GT::SDG}},
        {"t",     {0, 1, GT::T}},   {"tdg",  {0, 1, GT::TDG}},
        {"sx",    {0, 1, GT::SX}},  {"sxdg", {0, 1, GT::SXDG}},
        // 1-qubit identity (handled by an early drop in emit_gate before any
        // dispatch happens; the table entry exists only so a stray `id` inside
        // `try_emit_named` doesn't fall to the matrix fallback path)
        {"id",    {0, 1, GT::H}},
        // 1-qubit, parameterised
        {"rx",    {1, 1, GT::RX}},  {"ry",   {1, 1, GT::RY}},
        {"rz",    {1, 1, GT::RZ}},  {"p",    {1, 1, GT::P}},
        {"phase", {1, 1, GT::P}},   // QASM3 alias
        {"u1",    {1, 1, GT::U1}},  {"u2",   {2, 1, GT::U2}},
        {"u3",    {3, 1, GT::U3}},  {"u",    {3, 1, GT::U}},
        {"U",     {3, 1, GT::U}},   // QASM3 native U
        // 2-qubit, 0-param
        {"cx",    {0, 2, GT::CX}},  {"CX",   {0, 2, GT::CX}},
        {"cy",    {0, 2, GT::CY}},  {"cz",   {0, 2, GT::CZ}},
        {"ch",    {0, 2, GT::CH}},
        {"swap",  {0, 2, GT::SWAP}}, {"iswap",{0, 2, GT::ISWAP}},
        {"ecr",   {0, 2, GT::ECR}},
        // 2-qubit, parameterised
        {"crx",   {1, 2, GT::CRX}}, {"cry",  {1, 2, GT::CRY}},
        {"crz",   {1, 2, GT::CRZ}}, {"cp",   {1, 2, GT::CP}},
        {"cphase",{1, 2, GT::CP}},
        {"rxx",   {1, 2, GT::RXX}}, {"ryy",  {1, 2, GT::RYY}},
        {"rzz",   {1, 2, GT::RZZ}}, {"rzx",  {1, 2, GT::RZX}},
        // 3-qubit
        {"ccx",   {0, 3, GT::CCX}}, {"toffoli", {0, 3, GT::CCX}},
        {"ccz",   {0, 3, GT::CCZ}}, {"cswap",   {0, 3, GT::CSWAP}},
        {"fredkin",{0, 3, GT::CSWAP}}, {"rccx", {0, 3, GT::RCCX}}
    };
    return t;
}

// Gates that are their own inverse with no parameters — used by the peephole
// pass and by `inv @ <gate>` modifier folding. Includes 1- and 2- and
// 3-qubit Hermitian-self-inverse gates from the stdgates set.
bool is_self_inverse(Instruction::GateType t) noexcept {
    using GT = Instruction::GateType;
    switch (t) {
        case GT::H: case GT::X: case GT::Y: case GT::Z:
        case GT::CX: case GT::CY: case GT::CZ: case GT::CH:
        case GT::SWAP: case GT::ECR:
        case GT::CCX: case GT::CCZ: case GT::CSWAP:
            return true;
        default:
            return false;
    }
}

// =============================================================================
// Matrix fallback — used when a modifier stack doesn't map to a named gate.
// Only purely-numeric parameters are supported in this slow path.
// =============================================================================

using Mat = std::vector<Complex128>;  // row-major 2^k × 2^k

// 1-qubit base matrices — column-stable ordering [row0col0, row0col1, row1col0, row1col1].
Mat build_1q_base(std::string_view name, const std::vector<double>& p) {
    Mat M(4);
    const Complex128 ZERO{0.0, 0.0};
    const Complex128 ONE{1.0, 0.0};
    const Complex128 I_UNIT{0.0, 1.0};
    if (name == "h") {
        const double s = 1.0 / std::sqrt(2.0);
        M = { Complex128{s, 0}, Complex128{s, 0},
              Complex128{s, 0}, Complex128{-s, 0} };
    } else if (name == "x") {
        M = { ZERO, ONE, ONE, ZERO };
    } else if (name == "y") {
        M = { ZERO, Complex128{0, -1}, Complex128{0, 1}, ZERO };
    } else if (name == "z") {
        M = { ONE, ZERO, ZERO, Complex128{-1, 0} };
    } else if (name == "s") {
        M = { ONE, ZERO, ZERO, I_UNIT };
    } else if (name == "sdg") {
        M = { ONE, ZERO, ZERO, Complex128{0, -1} };
    } else if (name == "t") {
        const double r = 1.0 / std::sqrt(2.0);
        M = { ONE, ZERO, ZERO, Complex128{r, r} };
    } else if (name == "tdg") {
        const double r = 1.0 / std::sqrt(2.0);
        M = { ONE, ZERO, ZERO, Complex128{r, -r} };
    } else if (name == "sx") {
        M = { Complex128{0.5, 0.5}, Complex128{0.5, -0.5},
              Complex128{0.5, -0.5}, Complex128{0.5, 0.5} };
    } else if (name == "sxdg") {
        M = { Complex128{0.5, -0.5}, Complex128{0.5, 0.5},
              Complex128{0.5, 0.5}, Complex128{0.5, -0.5} };
    } else if (name == "rx") {
        const double c = std::cos(p[0] / 2.0), si = std::sin(p[0] / 2.0);
        M = { Complex128{c, 0}, Complex128{0, -si},
              Complex128{0, -si}, Complex128{c, 0} };
    } else if (name == "ry") {
        const double c = std::cos(p[0] / 2.0), si = std::sin(p[0] / 2.0);
        M = { Complex128{c, 0}, Complex128{-si, 0},
              Complex128{si, 0}, Complex128{c, 0} };
    } else if (name == "rz") {
        const double c = std::cos(p[0] / 2.0), si = std::sin(p[0] / 2.0);
        M = { Complex128{c, -si}, ZERO, ZERO, Complex128{c, si} };
    } else if (name == "p" || name == "phase" || name == "u1") {
        M = { ONE, ZERO, ZERO, Complex128{std::cos(p[0]), std::sin(p[0])} };
    } else if (name == "u" || name == "U" || name == "u3") {
        const double th = p[0], ph = p[1], la = p[2];
        const double c = std::cos(th / 2.0), si = std::sin(th / 2.0);
        M = {
            Complex128{c, 0},
            Complex128{-std::cos(la) * si, -std::sin(la) * si},
            Complex128{ std::cos(ph) * si,  std::sin(ph) * si},
            Complex128{ std::cos(ph + la) * c, std::sin(ph + la) * c }
        };
    } else if (name == "u2") {
        const double ph = p[0], la = p[1];
        const double r = 1.0 / std::sqrt(2.0);
        M = {
            Complex128{r, 0},
            Complex128{-r * std::cos(la), -r * std::sin(la)},
            Complex128{ r * std::cos(ph),  r * std::sin(ph)},
            Complex128{ r * std::cos(ph + la), r * std::sin(ph + la) }
        };
    } else {
        throw std::runtime_error(
            "QASM3Parser: modifier matrix fallback does not know base gate '" +
            std::string(name) + "'");
    }
    return M;
}

// Conjugate-transpose of a (row-major, k-qubit) matrix.
Mat mat_dagger(const Mat& M, int k) {
    const size_t d = size_t(1) << k;
    Mat R(d * d);
    for (size_t r = 0; r < d; ++r) {
        for (size_t c = 0; c < d; ++c) {
            R[r * d + c] = M[c * d + r].conj();
        }
    }
    return R;
}

// d×d matrix multiply: C = A · B (row-major).
Mat mat_mul(const Mat& A, const Mat& B, int k) {
    const size_t d = size_t(1) << k;
    Mat C(d * d, Complex128{0.0, 0.0});
    for (size_t r = 0; r < d; ++r) {
        for (size_t c = 0; c < d; ++c) {
            Complex128 acc{0.0, 0.0};
            for (size_t i = 0; i < d; ++i) {
                acc += A[r * d + i] * B[i * d + c];
            }
            C[r * d + c] = acc;
        }
    }
    return C;
}

// Integer matrix power. Negative exponents flip via dagger first.
Mat mat_pow(const Mat& M, int n, int k) {
    const size_t d = size_t(1) << k;
    Mat I(d * d, Complex128{0.0, 0.0});
    for (size_t i = 0; i < d; ++i) I[i * d + i] = Complex128{1.0, 0.0};
    if (n == 0) return I;
    Mat base = (n > 0) ? M : mat_dagger(M, k);
    int exp = std::abs(n);
    Mat result = I;
    // Binary exponentiation — O(log n) matmuls.
    while (exp > 0) {
        if (exp & 1) result = mat_mul(result, base, k);
        exp >>= 1;
        if (exp > 0) base = mat_mul(base, base, k);
    }
    return result;
}

// Extend a k-qubit matrix M with a single control to a (k+1)-qubit matrix.
//
// Project convention (docs/Architecture.md "Conventions", frozen in R.1.12.0):
// Instruction::matrix is qubits[0]-is-LSB, and `ctrl @` prepends the control
// as qubits[0]. The control is therefore index BIT 0 and the layout is the
// interleaved controlled form (controls = low index bits): even indices
// (control = 0) are identity, and M maps over the odd indices with the
// original k bits shifted up by one. Iterating this helper for ctrl @ ctrl @
// stacks yields the all-controls-1 slice on the low bits, matching control()
// and the Shor/QPE controlled matrices.
//
// Until R.1.12.2 this helper used a block-diagonal layout (control = MSB),
// which under the frozen operand order silently SWAPPED control and target
// for every gate routed through the matrix fallback. Named fast paths were
// unaffected, which is why no earlier test caught it.
Mat mat_add_control(const Mat& M, int k) {
    const size_t d = size_t(1) << k;
    const size_t D = 2 * d;
    Mat R(D * D, Complex128{0.0, 0.0});
    // ctrl = 0 (even indices): identity
    for (size_t i = 0; i < d; ++i) R[(2 * i) * D + (2 * i)] = Complex128{1.0, 0.0};
    // ctrl = 1 (odd indices): M over the remaining bits, shifted up by one
    for (size_t r = 0; r < d; ++r) {
        for (size_t c = 0; c < d; ++c) {
            R[(2 * r + 1) * D + (2 * c + 1)] = M[r * d + c];
        }
    }
    return R;
}

// =============================================================================
// Reusable: walk a ParamExpr tree and substitute Name nodes
// =============================================================================
// Used during custom-gate inlining — formal parameter names get replaced by
// the corresponding call-site ParamExpr (which may itself be symbolic).

ParamExpr substitute_in_expr(
    const ParamExpr& src,
    const std::unordered_map<std::string, ParamExpr>& param_map
) {
    switch (src.kind) {
        case ParamExpr::Kind::Literal:
            return ParamExpr::make_literal(src.literal);
        case ParamExpr::Kind::Name: {
            auto it = param_map.find(src.name);
            if (it != param_map.end()) {
                return it->second;  // ParamExpr copy ctor deep-clones
            }
            return ParamExpr::make_name(src.name);
        }
        case ParamExpr::Kind::BinaryOp: {
            ParamExpr l = substitute_in_expr(*src.lhs, param_map);
            ParamExpr r = substitute_in_expr(*src.rhs, param_map);
            return ParamExpr::make_binary(src.op, std::move(l), std::move(r));
        }
    }
    return ParamExpr::make_literal(0.0);
}

// Does this ParamExpr reduce to a single numeric constant? If so, evaluate it
// against the empty binding map. This is the fast-path test used to decide
// whether a gate emission can fill `params` (numeric) or must fall back to
// `param_exprs` (symbolic).
bool is_literal_constant(const ParamExpr& e) {
    switch (e.kind) {
        case ParamExpr::Kind::Literal:  return true;
        case ParamExpr::Kind::Name:     return false;
        case ParamExpr::Kind::BinaryOp: return is_literal_constant(*e.lhs) && is_literal_constant(*e.rhs);
    }
    return false;
}

// =============================================================================
// QASM3Parser — recursive descent over the token vector
// =============================================================================

class QASM3Parser {
public:
    static QuantumCircuit parse(const std::string& source) {
        auto toks = QASM3Lexer::tokenize(source);
        QASM3Parser p(std::move(toks));
        return p.run();
    }

private:
    explicit QASM3Parser(std::vector<Token> tokens) : toks_(std::move(tokens)) {}

    std::vector<Token> toks_;
    size_t pos_ = 0;

    int n_qubits_ = 0;
    int n_clbits_ = 0;
    std::unordered_map<std::string, int> qreg_offsets_;
    std::unordered_map<std::string, int> creg_offsets_;
    std::unordered_map<std::string, int> qreg_sizes_;
    std::unordered_map<std::string, int> creg_sizes_;

    QuantumCircuit qc_;

    // Custom gate definitions: name → (formal params, formal qubits, body stmts).
    struct PreCall {
        std::string base_name;
        std::vector<ParamExpr> params;
        std::vector<std::string> qubit_names;
        int n_ctrl = 0;
        bool inv = false;
        int pow_exp = 1;
        int line = 0;
    };
    struct GateDef {
        std::vector<std::string> param_names;
        std::vector<std::string> qubit_names;
        std::vector<PreCall> body;
    };
    std::unordered_map<std::string, GateDef> gate_defs_;

    // Peephole bookkeeping
    std::vector<std::vector<int>> qubit_stack_;  // per-qubit live-instruction stack
    std::vector<char> cancelled_;                // parallel to qc_.instructions

    // Active classical-condition state (consumed by next emitted instruction)
    int cond_clbit_ = -1;
    int cond_value_ = 0;

    // Active inlining context — when non-null, IDENT qubit references in
    // body statements resolve via this map rather than register offsets.
    // (Custom gates currently inline through direct substitution, so the
    // hook is reserved for future use; the read site is exercised by the
    // `inline_qubits_ != nullptr` check in parse_qubit_arg.)
    const std::unordered_map<std::string, int>* inline_qubits_ = nullptr;

    // ====================== Token utilities ======================

    const Token& peek(size_t off = 0) const noexcept {
        size_t idx = pos_ + off;
        return (idx < toks_.size()) ? toks_[idx] : toks_.back();
    }

    bool at(TT t) const noexcept { return peek().type == t; }

    bool at_kw(std::string_view kw) const noexcept {
        return peek().type == TT::KEYWORD && peek().text == kw;
    }

    bool accept(TT t) noexcept {
        if (peek().type == t) { ++pos_; return true; }
        return false;
    }

    bool accept_kw(std::string_view kw) noexcept {
        if (at_kw(kw)) { ++pos_; return true; }
        return false;
    }

    const Token& expect(TT t, std::string_view what) {
        if (peek().type != t) {
            throw std::runtime_error(
                "QASM3Parser: expected " + std::string(what) +
                " at line " + std::to_string(peek().line) +
                ", got '" + std::string(peek().text) + "'");
        }
        return toks_[pos_++];
    }

    const Token& consume() { return toks_[pos_++]; }

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error(
            "QASM3Parser: " + msg +
            " at line " + std::to_string(peek().line) +
            " (near '" + std::string(peek().text) + "')");
    }

    // ====================== Entry ======================

    QuantumCircuit run() {
        // First pass: discover register sizes, parse input-parameter
        // declarations, and parse gate definitions. We need n_qubits/n_clbits
        // before we can construct the QuantumCircuit.
        first_pass();
        if (n_qubits_ == 0) {
            throw std::runtime_error(
                "QASM3Parser: no qubit register declared");
        }
        // First pass populated qc_.parameter_names from `input` declarations.
        // Reconstructing qc_ with the now-known register sizes would discard
        // that vector, so save it across the rebuild.
        auto saved_param_names = std::move(qc_.parameter_names);
        qc_ = QuantumCircuit(n_qubits_, n_clbits_);
        qc_.parameter_names = std::move(saved_param_names);
        qubit_stack_.assign(n_qubits_, {});
        cancelled_.clear();

        // Second pass: emit instructions. `first_pass` rewinds pos_ to 0.
        while (!at(TT::END)) {
            parse_statement(/*inside_block=*/false);
        }

        // Sweep cancelled instructions.
        if (std::any_of(cancelled_.begin(), cancelled_.end(), [](char x){ return x != 0; })) {
            std::vector<Instruction> kept;
            kept.reserve(qc_.instructions.size());
            for (size_t i = 0; i < qc_.instructions.size(); ++i) {
                if (!cancelled_[i]) kept.push_back(std::move(qc_.instructions[i]));
            }
            qc_.instructions = std::move(kept);
        }

        return std::move(qc_);
    }

    // First pass: discover registers, input parameters, and gate definitions.
    // Does not emit any instructions; just populates n_qubits_, n_clbits_,
    // the offset maps, parameter_names, and gate_defs_.
    void first_pass() {
        while (!at(TT::END)) {
            // OPENQASM directive
            if (at_kw("OPENQASM")) {
                while (!at(TT::SEMICOLON) && !at(TT::END)) ++pos_;
                if (at(TT::SEMICOLON)) ++pos_;
                continue;
            }
            // include
            if (at_kw("include")) {
                while (!at(TT::SEMICOLON) && !at(TT::END)) ++pos_;
                if (at(TT::SEMICOLON)) ++pos_;
                continue;
            }
            // qubit declaration: `qubit[N] name;` or `qubit name;`
            if (at_kw("qubit") || at_kw("qreg")) {
                ++pos_;
                int size = 1;
                if (accept(TT::LBRACKET)) {
                    auto& it = expect(TT::INT, "qubit register size");
                    size = std::stoi(std::string(it.text));
                    expect(TT::RBRACKET, "']'");
                }
                auto& name_tok = expect(TT::IDENT, "register name");
                std::string name(name_tok.text);
                // QASM 2 `qreg name[N]` form: bracket after name.
                if (accept(TT::LBRACKET)) {
                    auto& it = expect(TT::INT, "qubit register size");
                    size = std::stoi(std::string(it.text));
                    expect(TT::RBRACKET, "']'");
                }
                qreg_offsets_[name] = n_qubits_;
                qreg_sizes_[name] = size;
                n_qubits_ += size;
                expect(TT::SEMICOLON, "';'");
                continue;
            }
            // bit declaration: `bit[N] name;` or `bit name;`
            if (at_kw("bit") || at_kw("creg")) {
                ++pos_;
                int size = 1;
                if (accept(TT::LBRACKET)) {
                    auto& it = expect(TT::INT, "bit register size");
                    size = std::stoi(std::string(it.text));
                    expect(TT::RBRACKET, "']'");
                }
                auto& name_tok = expect(TT::IDENT, "register name");
                std::string name(name_tok.text);
                if (accept(TT::LBRACKET)) {
                    auto& it = expect(TT::INT, "bit register size");
                    size = std::stoi(std::string(it.text));
                    expect(TT::RBRACKET, "']'");
                }
                creg_offsets_[name] = n_clbits_;
                creg_sizes_[name] = size;
                n_clbits_ += size;
                expect(TT::SEMICOLON, "';'");
                continue;
            }
            // input parameter: `input <type> name;`
            if (at_kw("input")) {
                ++pos_;
                // Skip type tokens up to the IDENT.
                while (!at(TT::IDENT) && !at(TT::SEMICOLON) && !at(TT::END)) ++pos_;
                if (at(TT::IDENT)) {
                    auto& nm = consume();
                    std::string name(nm.text);
                    if (std::find(qc_.parameter_names.begin(),
                                  qc_.parameter_names.end(), name)
                        == qc_.parameter_names.end()) {
                        qc_.parameter_names.push_back(name);
                    }
                }
                while (!at(TT::SEMICOLON) && !at(TT::END)) ++pos_;
                if (at(TT::SEMICOLON)) ++pos_;
                continue;
            }
            // gate definition: `gate name(params)? qargs { body }`
            if (at_kw("gate")) {
                parse_gate_definition();
                continue;
            }
            // Anything else — skip in first pass. Statement parsing happens
            // in second pass after registers are known.
            ++pos_;
        }
        // Rewind for second pass.
        pos_ = 0;
    }

    // ====================== Statement dispatch ======================

    // Parse one top-level statement during the emission pass.
    // `inside_block` distinguishes statements inside an `if` body.
    void parse_statement(bool inside_block) {
        // Already-parsed-in-first-pass forms we now skip.
        if (at_kw("OPENQASM") || at_kw("include")) {
            while (!at(TT::SEMICOLON) && !at(TT::END)) ++pos_;
            accept(TT::SEMICOLON);
            return;
        }
        if (at_kw("qubit") || at_kw("qreg") || at_kw("bit") || at_kw("creg")) {
            while (!at(TT::SEMICOLON) && !at(TT::END)) ++pos_;
            accept(TT::SEMICOLON);
            return;
        }
        if (at_kw("input")) {
            while (!at(TT::SEMICOLON) && !at(TT::END)) ++pos_;
            accept(TT::SEMICOLON);
            return;
        }
        if (at_kw("gate")) {
            // Skip — already captured in first pass.
            skip_balanced_block_after_gate_header();
            return;
        }
        if (at_kw("measure")) { parse_measure_prefix(); return; }
        if (at_kw("reset"))   { parse_reset(); return; }
        if (at_kw("barrier")) { parse_barrier(); return; }
        if (at_kw("if"))      { parse_if(); return; }

        // Unsupported QASM 3 constructs the plan calls out — throw with a
        // descriptive message rather than silently skipping.
        for (const char* kw : { "for", "while", "def", "delay", "stretch",
                                 "box", "cal", "defcal", "duration" }) {
            if (at_kw(kw)) {
                throw std::runtime_error(
                    std::string("QASM3Parser: timing/loop construct '") + kw +
                    "' not supported — use gate-level QASM 3 (line " +
                    std::to_string(peek().line) + ")");
            }
        }

        // `c[i] = measure q[j];` or bare `c = measure q;` : classical
        // assignment. Disambiguated from a gate call by an EQUALS appearing
        // either immediately after the IDENT (bare, single-bit register form)
        // or after a `[INT]` bracket suffix.
        if (at(TT::IDENT)) {
            if (peek(1).type == TT::EQUALS) {
                parse_classical_assign_measure();
                return;
            }
            if (peek(1).type == TT::LBRACKET) {
                size_t look = pos_ + 4;
                if (look < toks_.size() &&
                    toks_[pos_ + 2].type == TT::INT &&
                    toks_[pos_ + 3].type == TT::RBRACKET &&
                    toks_[pos_ + 4].type == TT::EQUALS) {
                    parse_classical_assign_measure();
                    return;
                }
            }
        }

        // Otherwise: a gate call (possibly modifier-prefixed).
        parse_gate_call();

        (void)inside_block;
    }

    // ====================== Gate definitions ======================

    void parse_gate_definition() {
        expect(TT::KEYWORD, "'gate'");
        auto& name_tok = expect(TT::IDENT, "gate name");
        GateDef def;
        std::string name(name_tok.text);

        // Optional parameter list
        if (accept(TT::LPAREN)) {
            if (!accept(TT::RPAREN)) {
                while (true) {
                    auto& p = expect(TT::IDENT, "parameter name");
                    def.param_names.emplace_back(p.text);
                    if (accept(TT::RPAREN)) break;
                    expect(TT::COMMA, "','");
                }
            }
        }

        // Qubit argument names
        while (!at(TT::LBRACE) && !at(TT::END)) {
            auto& q = expect(TT::IDENT, "qubit parameter");
            def.qubit_names.emplace_back(q.text);
            if (!accept(TT::COMMA)) break;
        }
        expect(TT::LBRACE, "'{'");

        // Body — each statement is a gate call referencing formal names.
        while (!at(TT::RBRACE) && !at(TT::END)) {
            def.body.push_back(parse_body_call());
        }
        expect(TT::RBRACE, "'}'");

        gate_defs_[name] = std::move(def);
    }

    // Shared tail of a `ctrl` modifier: both spec forms are accepted —
    // bare `ctrl @` adds one control, `ctrl(n) @` (R.1.18.0) adds n. Returns
    // the number of controls contributed by this modifier occurrence.
    int parse_ctrl_count() {
        int count = 1;
        if (accept(TT::LPAREN)) {
            auto& n = expect(TT::INT, "integer control count");
            count = std::stoi(std::string(n.text));
            if (count < 1) {
                fail("ctrl(" + std::to_string(count) + "): control count must be >= 1");
            }
            expect(TT::RPAREN, "')'");
        }
        expect(TT::AT, "'@'");
        return count;
    }

    // Parse a single statement inside a gate body and return a PreCall.
    // Modifier chains and parameter expressions are captured symbolically so
    // they can be evaluated later at the call site with substituted bindings.
    PreCall parse_body_call() {
        PreCall pc;
        pc.line = peek().line;

        while (true) {
            if (accept_kw("ctrl"))      { pc.n_ctrl += parse_ctrl_count(); continue; }
            if (accept_kw("inv"))       { expect(TT::AT, "'@'"); pc.inv = !pc.inv; continue; }
            if (accept_kw("pow")) {
                expect(TT::LPAREN, "'('");
                int sign = 1;
                if (accept(TT::MINUS)) sign = -1;
                else (void)accept(TT::PLUS);
                auto& n = expect(TT::INT, "integer exponent");
                pc.pow_exp *= sign * std::stoi(std::string(n.text));
                expect(TT::RPAREN, "')'");
                expect(TT::AT, "'@'");
                continue;
            }
            break;
        }

        // Gate name — KEYWORD ('U', 'CX') or IDENT
        if (at(TT::KEYWORD) || at(TT::IDENT)) {
            pc.base_name = std::string(peek().text);
            ++pos_;
        } else {
            fail("expected gate name in body");
        }

        // Optional parameter list
        if (accept(TT::LPAREN)) {
            if (!accept(TT::RPAREN)) {
                while (true) {
                    pc.params.push_back(parse_param_expr());
                    if (accept(TT::RPAREN)) break;
                    expect(TT::COMMA, "','");
                }
            }
        }

        // Qubit argument list — IDENT names referring to formal qubit args
        while (!at(TT::SEMICOLON) && !at(TT::END)) {
            auto& q = expect(TT::IDENT, "formal qubit");
            pc.qubit_names.emplace_back(q.text);
            if (!accept(TT::COMMA)) break;
        }
        expect(TT::SEMICOLON, "';'");
        return pc;
    }

    // Already past the `gate` keyword? Or before it? Used by the emission
    // pass when it encounters a gate definition (already in gate_defs_).
    void skip_balanced_block_after_gate_header() {
        // We're sitting on the `gate` keyword — skip up to the `{` then to
        // the matching `}` accounting for nesting (gate bodies don't nest,
        // but we still count safely).
        while (!at(TT::LBRACE) && !at(TT::END)) ++pos_;
        if (!accept(TT::LBRACE)) return;
        int depth = 1;
        while (depth > 0 && !at(TT::END)) {
            if (at(TT::LBRACE)) ++depth;
            else if (at(TT::RBRACE)) --depth;
            ++pos_;
        }
    }

    // ====================== Parameter expressions ======================
    // Precedence: + - (lowest) → * / → unary → atom

    ParamExpr parse_param_expr() { return parse_addsub(); }

    ParamExpr parse_addsub() {
        ParamExpr lhs = parse_muldiv();
        while (at(TT::PLUS) || at(TT::MINUS)) {
            char op = (peek().type == TT::PLUS) ? '+' : '-';
            ++pos_;
            ParamExpr rhs = parse_muldiv();
            lhs = ParamExpr::make_binary(op, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    ParamExpr parse_muldiv() {
        ParamExpr lhs = parse_unary();
        while (at(TT::STAR) || at(TT::SLASH)) {
            char op = (peek().type == TT::STAR) ? '*' : '/';
            ++pos_;
            ParamExpr rhs = parse_unary();
            lhs = ParamExpr::make_binary(op, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    ParamExpr parse_unary() {
        if (accept(TT::MINUS)) {
            ParamExpr inner = parse_unary();
            return ParamExpr::make_binary('-', ParamExpr::make_literal(0.0), std::move(inner));
        }
        if (accept(TT::PLUS)) { return parse_unary(); }
        return parse_atom();
    }

    ParamExpr parse_atom() {
        if (accept(TT::LPAREN)) {
            ParamExpr e = parse_param_expr();
            expect(TT::RPAREN, "')'");
            return e;
        }
        const Token& t = peek();
        if (t.type == TT::INT) {
            ++pos_;
            return ParamExpr::make_literal(std::stod(std::string(t.text)));
        }
        if (t.type == TT::FLOAT) {
            ++pos_;
            return ParamExpr::make_literal(std::stod(std::string(t.text)));
        }
        if (t.type == TT::IDENT || (t.type == TT::KEYWORD && t.text == "pi")) {
            ++pos_;
            if (t.text == "pi") return ParamExpr::make_literal(PI);
            if (t.text == "tau") return ParamExpr::make_literal(2.0 * PI);
            if (t.text == "euler") return ParamExpr::make_literal(2.718281828459045);
            return ParamExpr::make_name(std::string(t.text));
        }
        // KEYWORD: a few constants live in the keyword set
        if (t.type == TT::KEYWORD && (t.text == "true" || t.text == "false")) {
            ++pos_;
            return ParamExpr::make_literal(t.text == "true" ? 1.0 : 0.0);
        }
        fail("expected parameter expression");
    }

    // ====================== Qubit / classical args ======================

    // Resolve a qubit reference. Either `name[INT]` against a register, or a
    // formal name when inside a gate-body inline context.
    int parse_qubit_arg() {
        auto& nm = expect(TT::IDENT, "qubit reference");
        std::string name(nm.text);
        if (inline_qubits_ != nullptr) {
            auto it = inline_qubits_->find(name);
            if (it != inline_qubits_->end()) {
                // A formal qubit is a single bit, no brackets expected.
                return it->second;
            }
        }
        int offset;
        auto it = qreg_offsets_.find(name);
        if (it == qreg_offsets_.end()) {
            fail("unknown qubit register '" + name + "'");
        }
        offset = it->second;
        if (!accept(TT::LBRACKET)) {
            // Bare `name` referring to a single-qubit register
            int sz = qreg_sizes_[name];
            if (sz != 1) {
                fail("qubit register '" + name + "' requires an index");
            }
            return offset;
        }
        auto& idx = expect(TT::INT, "qubit index");
        int i = std::stoi(std::string(idx.text));
        expect(TT::RBRACKET, "']'");
        return offset + i;
    }

    int parse_clbit_arg() {
        auto& nm = expect(TT::IDENT, "classical bit reference");
        std::string name(nm.text);
        auto it = creg_offsets_.find(name);
        if (it == creg_offsets_.end()) {
            fail("unknown classical register '" + name + "'");
        }
        int offset = it->second;
        if (!accept(TT::LBRACKET)) {
            int sz = creg_sizes_[name];
            if (sz != 1) {
                fail("classical register '" + name + "' requires an index");
            }
            return offset;
        }
        auto& idx = expect(TT::INT, "classical bit index");
        int i = std::stoi(std::string(idx.text));
        expect(TT::RBRACKET, "']'");
        return offset + i;
    }

    // ====================== measure / reset / barrier ======================

    // `c[i] = measure q[j];`
    void parse_classical_assign_measure() {
        int c = parse_clbit_arg();
        expect(TT::EQUALS, "'='");
        if (!accept_kw("measure")) {
            fail("expected 'measure' after '='");
        }
        int q = parse_qubit_arg();
        expect(TT::SEMICOLON, "';'");
        Instruction inst;
        inst.type = Instruction::GateType::MEASURE;
        inst.qubits = { q };
        inst.clbits = { c };
        emit_instruction(std::move(inst));
    }

    // `measure q[i] -> c[j];` or `measure q[i];` (Qiskit sometimes emits this
    // when no classical bit is targeted — we then drop the measurement).
    void parse_measure_prefix() {
        expect(TT::KEYWORD, "'measure'");
        int q = parse_qubit_arg();
        int c = -1;
        if (accept(TT::ARROW)) {
            c = parse_clbit_arg();
        }
        expect(TT::SEMICOLON, "';'");
        if (c < 0) {
            // No classical target — emit a barrier-equivalent? Qiskit's
            // convention is that measure-without-target is invalid; throw.
            fail("measurement requires a classical target ('-> c[i]')");
        }
        Instruction inst;
        inst.type = Instruction::GateType::MEASURE;
        inst.qubits = { q };
        inst.clbits = { c };
        emit_instruction(std::move(inst));
    }

    void parse_reset() {
        expect(TT::KEYWORD, "'reset'");
        int q = parse_qubit_arg();
        expect(TT::SEMICOLON, "';'");
        Instruction inst;
        inst.type = Instruction::GateType::RESET;
        inst.qubits = { q };
        emit_instruction(std::move(inst));
    }

    void parse_barrier() {
        expect(TT::KEYWORD, "'barrier'");
        std::vector<int> qubits;
        while (!at(TT::SEMICOLON) && !at(TT::END)) {
            qubits.push_back(parse_qubit_arg());
            if (!accept(TT::COMMA)) break;
        }
        expect(TT::SEMICOLON, "';'");
        Instruction inst;
        inst.type = Instruction::GateType::BARRIER;
        if (qubits.empty()) {
            for (int q = 0; q < n_qubits_; ++q) qubits.push_back(q);
        }
        inst.qubits = std::move(qubits);
        emit_instruction(std::move(inst));
    }

    // ====================== if / else ======================

    void parse_if() {
        expect(TT::KEYWORD, "'if'");
        expect(TT::LPAREN, "'('");
        // Only support `c[i] == V` — single-bit conditioning.
        auto& reg = expect(TT::IDENT, "classical register");
        std::string reg_name(reg.text);
        int bit_global;
        if (accept(TT::LBRACKET)) {
            auto& idx = expect(TT::INT, "classical bit index");
            int i = std::stoi(std::string(idx.text));
            expect(TT::RBRACKET, "']'");
            auto it = creg_offsets_.find(reg_name);
            if (it == creg_offsets_.end()) fail("unknown classical register '" + reg_name + "'");
            bit_global = it->second + i;
        } else {
            // `if (c == V)` — supported only when c is single-bit.
            auto it = creg_offsets_.find(reg_name);
            if (it == creg_offsets_.end()) fail("unknown classical register '" + reg_name + "'");
            if (creg_sizes_[reg_name] != 1) {
                fail("multi-bit register condition 'if (" + reg_name +
                     " == V)' not supported — use 'if (" + reg_name + "[i] == V)'");
            }
            bit_global = it->second;
        }
        expect(TT::EQEQ, "'=='");
        auto& v = expect(TT::INT, "condition value");
        int val = std::stoi(std::string(v.text));
        expect(TT::RPAREN, "')'");

        int saved_clbit = cond_clbit_;
        int saved_value = cond_value_;
        cond_clbit_ = bit_global;
        cond_value_ = val;
        parse_then_or_block();
        cond_clbit_ = saved_clbit;
        cond_value_ = saved_value;

        if (accept_kw("else")) {
            // `else` means: condition holds when bit != val. For a single bit
            // with val ∈ {0,1}, that's bit == (1 - val).
            int else_val = (val == 0) ? 1 : 0;
            int saved2_clbit = cond_clbit_;
            int saved2_value = cond_value_;
            cond_clbit_ = bit_global;
            cond_value_ = else_val;
            parse_then_or_block();
            cond_clbit_ = saved2_clbit;
            cond_value_ = saved2_value;
        }
    }

    // Body of an `if` or `else`: either a single statement or a `{ ... }`.
    void parse_then_or_block() {
        if (accept(TT::LBRACE)) {
            while (!at(TT::RBRACE) && !at(TT::END)) {
                parse_statement(/*inside_block=*/true);
            }
            expect(TT::RBRACE, "'}'");
        } else {
            parse_statement(/*inside_block=*/true);
        }
    }

    // ====================== Gate calls (with modifier stack) ======================

    void parse_gate_call() {
        int n_ctrl = 0;
        bool inv = false;
        int pow_exp = 1;
        int line = peek().line;

        while (true) {
            if (accept_kw("ctrl"))      { n_ctrl += parse_ctrl_count(); continue; }
            if (accept_kw("inv"))       { expect(TT::AT, "'@'"); inv = !inv; continue; }
            if (accept_kw("pow")) {
                expect(TT::LPAREN, "'('");
                int sign = 1;
                if (accept(TT::MINUS)) sign = -1;
                else (void)accept(TT::PLUS);
                auto& n = expect(TT::INT, "integer exponent");
                pow_exp *= sign * std::stoi(std::string(n.text));
                expect(TT::RPAREN, "')'");
                expect(TT::AT, "'@'");
                continue;
            }
            break;
        }

        // pow(0) @ <anything> → drop entirely.
        if (pow_exp == 0) {
            while (!at(TT::SEMICOLON) && !at(TT::END)) ++pos_;
            accept(TT::SEMICOLON);
            return;
        }

        // Gate name
        std::string name;
        if (at(TT::KEYWORD) || at(TT::IDENT)) {
            name = std::string(peek().text);
            ++pos_;
        } else {
            fail("expected gate name");
        }

        // Parameters
        std::vector<ParamExpr> params;
        if (accept(TT::LPAREN)) {
            if (!accept(TT::RPAREN)) {
                while (true) {
                    params.push_back(parse_param_expr());
                    if (accept(TT::RPAREN)) break;
                    expect(TT::COMMA, "','");
                }
            }
        }

        // Qubit args
        std::vector<int> qubits;
        while (!at(TT::SEMICOLON) && !at(TT::END)) {
            qubits.push_back(parse_qubit_arg());
            if (!accept(TT::COMMA)) break;
        }
        expect(TT::SEMICOLON, "';'");

        emit_gate(name, std::move(params), std::move(qubits),
                  n_ctrl, inv, pow_exp, line);
    }

    // ====================== Gate emission ======================
    // The single chokepoint that resolves modifier stacks into concrete
    // Instructions (named gate or UNITARY).

    void emit_gate(const std::string& name,
                   std::vector<ParamExpr> params,
                   std::vector<int> qubits,
                   int n_ctrl, bool inv, int pow_exp,
                   int line)
    {
        (void)line;

        // Custom-defined gate? Inline it (recursively, with substitution).
        // Modifiers on a custom gate aren't supported in R.1.9.0 — Qiskit
        // doesn't emit that combination from to_qasm3().
        auto def_it = gate_defs_.find(name);
        if (def_it != gate_defs_.end()) {
            if (n_ctrl != 0 || inv || pow_exp != 1) {
                throw std::runtime_error(
                    "QASM3Parser: modifiers on user-defined gate '" + name +
                    "' are not supported (inline the body first)");
            }
            inline_custom_gate(def_it->second, params, qubits);
            return;
        }

        // `id` is the identity gate. Any modifier stack over it is still
        // identity (inv @ id = id, pow(n) @ id = id, ctrl @ id = identity on
        // k+1 qubits), so we drop unconditionally regardless of n_ctrl, inv,
        // or pow_exp.
        if (name == "id") return;

        // Try a fast path that maps the modifier stack to a named gate.
        if (try_emit_named(name, params, qubits, n_ctrl, inv, pow_exp)) return;

        // Slow path — matrix fallback. Requires all params to be numeric.
        emit_matrix_fallback(name, params, qubits, n_ctrl, inv, pow_exp);
    }

    // Inline a custom-gate definition. Each body statement is re-emitted with
    // parameter and qubit substitution. Recursion is bounded by the depth of
    // user-defined gates (typical Qiskit output: 1 level).
    void inline_custom_gate(const GateDef& def,
                            const std::vector<ParamExpr>& call_params,
                            const std::vector<int>& call_qubits)
    {
        if (call_params.size() != def.param_names.size()) {
            throw std::runtime_error(
                "QASM3Parser: gate parameter count mismatch (expected " +
                std::to_string(def.param_names.size()) + ", got " +
                std::to_string(call_params.size()) + ")");
        }
        if (call_qubits.size() != def.qubit_names.size()) {
            throw std::runtime_error(
                "QASM3Parser: gate qubit count mismatch (expected " +
                std::to_string(def.qubit_names.size()) + ", got " +
                std::to_string(call_qubits.size()) + ")");
        }

        std::unordered_map<std::string, ParamExpr> pmap;
        for (size_t i = 0; i < def.param_names.size(); ++i) {
            pmap.emplace(def.param_names[i], call_params[i]);  // ParamExpr copy
        }
        std::unordered_map<std::string, int> qmap;
        for (size_t i = 0; i < def.qubit_names.size(); ++i) {
            qmap[def.qubit_names[i]] = call_qubits[i];
        }

        for (const auto& bc : def.body) {
            std::vector<ParamExpr> sub_params;
            sub_params.reserve(bc.params.size());
            for (const auto& p : bc.params) {
                sub_params.push_back(substitute_in_expr(p, pmap));
            }
            std::vector<int> sub_qubits;
            sub_qubits.reserve(bc.qubit_names.size());
            for (const auto& qn : bc.qubit_names) {
                auto it = qmap.find(qn);
                if (it == qmap.end()) {
                    throw std::runtime_error(
                        "QASM3Parser: unbound qubit name '" + qn +
                        "' in body of gate (line " + std::to_string(bc.line) + ")");
                }
                sub_qubits.push_back(it->second);
            }
            emit_gate(bc.base_name, std::move(sub_params), std::move(sub_qubits),
                      bc.n_ctrl, bc.inv, bc.pow_exp, bc.line);
        }
    }

    // Try to map (name + modifier stack) to a named built-in instruction.
    // Returns true on success.
    bool try_emit_named(const std::string& name,
                        const std::vector<ParamExpr>& params,
                        const std::vector<int>& qubits,
                        int n_ctrl, bool inv, int pow_exp)
    {
        // `pow(n) @ <self-inverse>` collapses to identity (n even) or the
        // gate (n odd). Reduce here so the named lookup sees pow_exp=1.
        int eff_pow = pow_exp;

        // Helper: build params after applying inv/pow folding for known kinds.
        auto fold_angle_params = [&](double sign_inv) -> std::vector<ParamExpr> {
            // For rotations and phase: result = (eff_pow * sign_inv) * original
            // We compose a multiplicative scalar wrapper around each param.
            std::vector<ParamExpr> out;
            out.reserve(params.size());
            const double scale = sign_inv * static_cast<double>(eff_pow);
            for (const auto& p : params) {
                if (scale == 1.0) {
                    out.push_back(p);
                } else {
                    out.push_back(ParamExpr::make_binary(
                        '*',
                        ParamExpr::make_literal(scale),
                        p));
                }
            }
            return out;
        };

        const auto& table = builtin_table();
        auto it = table.find(name);

        // ============== Path 1: 0 controls ==============
        if (n_ctrl == 0) {
            if (it == table.end()) return false;
            const BuiltinSpec& spec = it->second;

            if (static_cast<int>(qubits.size()) != spec.n_qubits) return false;
            if (static_cast<int>(params.size()) != spec.n_params) return false;

            // pow folding for self-inverse 0-param gates
            if (spec.n_params == 0 && is_self_inverse(spec.type)) {
                if ((eff_pow % 2) == 0) return true;  // identity — emit nothing
                eff_pow = 1;
            }

            // inv on self-inverse 0-param: no change
            // inv on s/sdg/t/tdg/sx/sxdg: swap pairs
            std::string final_name = name;
            if (inv) {
                static const std::unordered_map<std::string_view, std::string_view> inv_pair = {
                    {"s", "sdg"}, {"sdg", "s"},
                    {"t", "tdg"}, {"tdg", "t"},
                    {"sx", "sxdg"}, {"sxdg", "sx"}
                };
                auto ip = inv_pair.find(name);
                if (ip != inv_pair.end()) {
                    final_name = std::string(ip->second);
                } else if (!is_self_inverse(spec.type) && spec.n_params == 0) {
                    // Non-self-inverse, 0-param, not in pair map (e.g., iswap) — fall to matrix
                    return false;
                }
            }

            // Parameterised rotations: fold inv (negate) and pow (scale) into the angle.
            std::vector<ParamExpr> emit_params;
            if (spec.n_params > 0) {
                // Angle folding (scale by inv*pow) is only a valid identity
                // for SINGLE-axis rotations: RX(a)^n = RX(n*a) and
                // inv RX(a) = RX(-a), same for ry/rz/p/u1. It is WRONG for
                // multi-parameter gates: Euler angles do not scale
                // (U3(t,p,l)^2 != U3(2t,2p,2l)) and inv U3(t,p,l) =
                // U3(-t,-l,-p), with phi and lambda swapped. Route those to
                // the exact matrix fallback (mat_pow / mat_dagger) instead.
                // Pre-R.1.12.2 this folded u/u2/u3 and silently imported a
                // slightly different unitary.
                if ((inv || eff_pow != 1) && spec.n_params > 1) return false;
                double sign = inv ? -1.0 : 1.0;
                emit_params = fold_angle_params(sign);
                eff_pow = 1;
            } else {
                emit_params = params;
            }

            // If eff_pow > 1 on a self-inverse-collapsed gate, eff_pow was set to 1 above.
            // For non-self-inverse with no parameter support, we can't pow-fold — fall through.
            if (eff_pow != 1 && spec.n_params == 0 && !is_self_inverse(spec.type)) return false;

            // Look up the final type (may differ from `name` due to inv-pair swap).
            auto final_it = table.find(final_name);
            if (final_it == table.end()) return false;
            return emit_concrete(final_it->second.type, emit_params, qubits);
        }

        // ============== Path 2: 1 or more controls ==============
        // Map `ctrl @ x` → cx, `ctrl @ ctrl @ x` → ccx, etc.
        // Map `ctrl @ rx(θ)` → crx(θ).
        struct CtrlMap {
            int n_ctrl;
            std::string_view base;
            std::string_view target;
        };
        static const std::vector<CtrlMap> ctrl_table = {
            {1, "x", "cx"},   {1, "y", "cy"},   {1, "z", "cz"},
            {1, "h", "ch"},   {1, "swap", "cswap"},
            {2, "x", "ccx"},  {2, "z", "ccz"},
            {1, "rx", "crx"}, {1, "ry", "cry"}, {1, "rz", "crz"},
            {1, "p", "cp"},   {1, "phase", "cp"}
        };

        // pow-fold self-inverse ctrl gates first.
        // ctrl @ x with pow(n): if n even → identity, n odd → keep. Any
        // control count over x stays self-inverse (R.1.18.0: enables the
        // wide-stack MCX fast path below to see eff_pow == 1).
        bool self_inv_base = (n_ctrl == 1 && (name == "x" || name == "y" || name == "z" ||
                                              name == "h" || name == "swap")) ||
                             (n_ctrl == 2 && (name == "x" || name == "z")) ||
                             (n_ctrl >= 3 && name == "x");
        if (self_inv_base && params.empty()) {
            if ((eff_pow % 2) == 0) return true;
            eff_pow = 1;
        }
        if (eff_pow != 1 && params.empty()) return false;

        for (const auto& cm : ctrl_table) {
            if (cm.n_ctrl != n_ctrl) continue;
            if (cm.base != name) continue;

            auto fit = table.find(cm.target);
            if (fit == table.end()) return false;
            const BuiltinSpec& spec = fit->second;
            if (static_cast<int>(qubits.size()) != spec.n_qubits) return false;
            if (static_cast<int>(params.size()) != spec.n_params) return false;

            std::vector<ParamExpr> emit_params;
            if (spec.n_params > 0) {
                double sign = inv ? -1.0 : 1.0;
                emit_params = fold_angle_params(sign);
            } else {
                if (inv && !is_self_inverse(spec.type)) return false;
                emit_params = params;
            }
            return emit_concrete(spec.type, emit_params, qubits);
        }

        // ============== Path 3: wide control stacks (R.1.18.0) ==============
        // Resolve to the first-class MCX / MCP instructions instead of the
        // matrix fallback, which would materialise a dense 2^(k+1) matrix —
        // exactly the object these instructions exist to avoid. This is also
        // what makes the to_qasm3() `ctrl(k) @` emission round-trip.
        //
        // ctrl^k @ x, k ≥ 3 → MCX. x is self-inverse, so inv is free and
        // eff_pow was folded to 1 above. (k ≤ 2 stays cx/ccx via the table:
        // the canonical named forms.)
        if (n_ctrl >= 3 && name == "x" && params.empty()) {
            if (static_cast<int>(qubits.size()) != n_ctrl + 1) return false;
            return emit_concrete(Instruction::GateType::MCX, {}, qubits);
        }
        // ctrl^k @ p(λ), k ≥ 2 → MCP over all k+1 operands (the phase is
        // symmetric). inv negates λ and pow scales it — both exact for a
        // phase gate — via the shared angle-folding helper.
        if (n_ctrl >= 2 && (name == "p" || name == "phase") && params.size() == 1) {
            if (static_cast<int>(qubits.size()) != n_ctrl + 1) return false;
            double sign = inv ? -1.0 : 1.0;
            return emit_concrete(Instruction::GateType::MCP,
                                 fold_angle_params(sign), qubits);
        }

        return false;
    }

    // Emit a concrete (typed) instruction. Resolves symbolic vs numeric.
    bool emit_concrete(Instruction::GateType type,
                       const std::vector<ParamExpr>& params,
                       const std::vector<int>& qubits)
    {
        Instruction inst;
        inst.type = type;
        inst.qubits = qubits;

        bool all_literal = true;
        for (const auto& p : params) {
            if (!is_literal_constant(p)) { all_literal = false; break; }
        }

        if (all_literal) {
            inst.params.reserve(params.size());
            for (const auto& p : params) {
                inst.params.push_back(p.eval({}));
            }
        } else {
            inst.param_exprs = params;  // deep-copied via ParamExpr copy ctor
        }

        emit_instruction(std::move(inst));
        return true;
    }

    // Matrix-fallback path. Used when no named gate can be synthesised.
    void emit_matrix_fallback(const std::string& name,
                              const std::vector<ParamExpr>& params,
                              const std::vector<int>& qubits,
                              int n_ctrl, bool inv, int pow_exp)
    {
        // Require all params to be numeric — symbolic matrix synthesis is
        // beyond R.1.9.0's scope; the caller should bind first.
        std::vector<double> num_params;
        num_params.reserve(params.size());
        for (const auto& p : params) {
            if (!is_literal_constant(p)) {
                throw std::runtime_error(
                    "QASM3Parser: symbolic parameter cannot be used with modifier "
                    "matrix fallback (gate '" + name + "'); bind parameters first");
            }
            num_params.push_back(p.eval({}));
        }

        // Build the base matrix (only 1-qubit base gates supported in fallback).
        const auto& table = builtin_table();
        auto it = table.find(name);
        if (it == table.end()) {
            throw std::runtime_error(
                "QASM3Parser: unknown gate '" + name + "' with modifier stack");
        }
        const BuiltinSpec& spec = it->second;
        // Validate parameter arity before build_1q_base dereferences `p[0]`
        // for rotation gates: catches user errors like `rx q[0]` (missing
        // angle) with a descriptive throw instead of UB.
        if (static_cast<int>(num_params.size()) != spec.n_params) {
            throw std::runtime_error(
                "QASM3Parser: gate '" + name + "' expects " +
                std::to_string(spec.n_params) + " parameter(s), got " +
                std::to_string(num_params.size()));
        }
        if (spec.n_qubits != 1) {
            throw std::runtime_error(
                "QASM3Parser: matrix fallback only handles 1-qubit base gates; "
                "encountered '" + name + "'");
        }
        Mat M = build_1q_base(name, num_params);
        int k = 1;
        if (inv) M = mat_dagger(M, k);
        if (pow_exp != 1) M = mat_pow(M, pow_exp, k);
        for (int i = 0; i < n_ctrl; ++i) {
            M = mat_add_control(M, k);
            ++k;
        }

        if (static_cast<int>(qubits.size()) != k) {
            throw std::runtime_error(
                "QASM3Parser: gate '" + name + "' expected " +
                std::to_string(k) + " qubits after modifier stack, got " +
                std::to_string(qubits.size()));
        }

        Instruction inst;
        inst.type = Instruction::GateType::UNITARY;
        inst.matrix = std::move(M);
        inst.qubits = qubits;
        inst.label = name;
        emit_instruction(std::move(inst));
    }

    // ====================== Emission + peephole ======================

    void emit_instruction(Instruction&& inst) {
        // Stamp classical condition (if any) onto this instruction.
        if (cond_clbit_ >= 0) {
            inst.condition_clbit = cond_clbit_;
            inst.condition_value = cond_value_;
        }

        // Peephole — only attempt cancellation when no condition is active.
        // Conditioned gates may legitimately repeat (different control flow).
        if (cond_clbit_ < 0 && try_cancel_with_previous(inst)) return;

        const int new_idx = static_cast<int>(qc_.instructions.size());
        qc_.instructions.push_back(std::move(inst));
        cancelled_.push_back(0);

        // Update per-qubit stacks with this index. We track all qubits the
        // instruction touches; BARRIER touches its listed qubits.
        const auto& q = qc_.instructions.back().qubits;
        for (int qi : q) {
            if (qi >= 0 && qi < static_cast<int>(qubit_stack_.size())) {
                qubit_stack_[qi].push_back(new_idx);
            }
        }
    }

    // Attempt to cancel `inst` with the most-recent live instruction on the
    // same qubits. Returns true if both are dropped.
    bool try_cancel_with_previous(const Instruction& inst) {
        // Only cancel pure-self-inverse no-parameter gates.
        if (!is_self_inverse(inst.type)) return false;
        if (!inst.params.empty() || !inst.param_exprs.empty()) return false;
        if (inst.qubits.empty()) return false;

        // For all qubits in `inst`, the top of each per-qubit stack must
        // refer to the same instruction index P, which itself must touch
        // exactly the same qubits in the same order and have the same type.
        int prev = -1;
        for (int q : inst.qubits) {
            if (q < 0 || q >= static_cast<int>(qubit_stack_.size())) return false;
            const auto& s = qubit_stack_[q];
            if (s.empty()) return false;
            if (prev < 0) prev = s.back();
            else if (s.back() != prev) return false;
        }
        if (prev < 0 || prev >= static_cast<int>(qc_.instructions.size())) return false;
        if (cancelled_[prev]) return false;

        const Instruction& prior = qc_.instructions[prev];
        if (prior.type != inst.type) return false;
        if (!prior.params.empty() || !prior.param_exprs.empty()) return false;
        if (prior.qubits != inst.qubits) return false;
        // Both instructions must be unconditioned for the cancellation to be
        // valid; conditioned gates can change semantics under different
        // classical values.
        if (prior.condition_clbit >= 0 || inst.condition_clbit >= 0) return false;

        // Cancel: mark prior dead, pop top of each affected per-qubit stack.
        cancelled_[prev] = 1;
        for (int q : inst.qubits) {
            qubit_stack_[q].pop_back();
        }
        return true;
    }
};

}  // namespace

// =============================================================================
// Bridge function — called from circuit.cpp::from_qasm3()
// =============================================================================
// Kept as a free function so circuit.cpp doesn't need to include the QASM3
// parser headers (mirrors the QASM 2 bridge pattern).

QuantumCircuit qasm3_parse_impl(const std::string& qasm) {
    return QASM3Parser::parse(qasm);
}

}  // namespace lindblad
