// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/circuit.hpp"
#include "lindblad/detail/text.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <regex>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace lindblad {

namespace {

// =============================================================================
// Qasm2Expression - one OpenQASM 2.0 `exp`, evaluated as it is read
// =============================================================================
//   exp     := real | nninteger | pi | id | exp + exp | exp - exp | exp * exp
//            | exp / exp | -exp | exp ^ exp | ( exp ) | unaryop ( exp )
//   unaryop := sin | cos | tan | exp | ln | sqrt
//
// Precedence, lowest first: + and - (left-associative), * and / (left), unary
// minus, ^ (right-associative). So a/b/c is (a/b)/c, -2^2 is -4 and 2^3^2 is
// 2^9. A number is digits with an optional fraction and exponent, read whole
// in no locale; the grammar's real requires a point, and an exponent on a
// whole number ("1e-05", which a %g exporter writes) is read as well. An id is
// a parameter of the gate being expanded, looked up in `params`; at the top
// level there are none in scope.
//
// Everything outside the grammar is refused, never guessed: text left over, an
// id naming no parameter, a division by zero, and a value that is not finite
// (ln(0), sqrt(-1), an overflow), each throwing std::runtime_error that quotes
// the expression as written.
class Qasm2Expression {
public:
    Qasm2Expression(std::string_view text, const std::unordered_map<std::string, double>& params)
        : text_(text), params_(params) {}

    double value() {
        skip_space();
        if (pos_ == text_.size()) fail("the expression is empty");
        const double v = sum();
        skip_space();
        if (pos_ != text_.size()) {
            fail("unexpected '" + std::string(text_.substr(pos_)) + "' after the expression");
        }
        return v;
    }

private:
    std::string_view text_;
    const std::unordered_map<std::string, double>& params_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("QASM2Parser: parameter expression '" + std::string(text_) +
                                 "': " + why);
    }

    // The bit pattern decides, since the build's fast-math flags let a
    // comparison with an infinity or a NaN fold away.
    double finite(double v) const {
        if (!is_finite_strict(v)) fail("evaluates to a value that is not finite");
        return v;
    }

    void skip_space() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    bool take(char c) {
        skip_space();
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }
    static bool is_digit(char c) { return c >= '0' && c <= '9'; }
    static bool is_ident_start(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }
    static bool is_ident(char c) { return is_ident_start(c) || is_digit(c); }

    double sum() {
        double v = product();
        for (;;) {
            if (take('+')) v = finite(v + product());
            else if (take('-')) v = finite(v - product());
            else return v;
        }
    }

    double product() {
        double v = unary();
        for (;;) {
            if (take('*')) {
                v = finite(v * unary());
            } else if (take('/')) {
                const double d = unary();
                if (d == 0.0) fail("division by zero");
                v = finite(v / d);
            } else {
                return v;
            }
        }
    }

    double unary() {
        if (take('-')) return -unary();
        return power();
    }

    double power() {
        const double base = primary();
        if (take('^')) return finite(std::pow(base, unary()));
        return base;
    }

    double primary() {
        skip_space();
        if (take('(')) {
            const double v = sum();
            if (!take(')')) fail("expected ')'");
            return v;
        }
        if (pos_ < text_.size() &&
            (is_digit(text_[pos_]) ||
             (text_[pos_] == '.' && pos_ + 1 < text_.size() && is_digit(text_[pos_ + 1])))) {
            return number();
        }
        if (pos_ < text_.size() && is_ident_start(text_[pos_])) {
            const std::size_t start = pos_;
            while (pos_ < text_.size() && is_ident(text_[pos_])) ++pos_;
            const std::string name(text_.substr(start, pos_ - start));
            if (name == "pi") return PI;
            if (name == "sin" || name == "cos" || name == "tan" || name == "exp" ||
                name == "ln" || name == "sqrt") {
                if (!take('(')) fail("expected '(' after '" + name + "'");
                const double x = sum();
                if (!take(')')) fail("expected ')'");
                if (name == "sin") return finite(std::sin(x));
                if (name == "cos") return finite(std::cos(x));
                if (name == "tan") return finite(std::tan(x));
                if (name == "exp") return finite(std::exp(x));
                if (name == "ln") return finite(std::log(x));
                return finite(std::sqrt(x));
            }
            const auto it = params_.find(name);
            if (it == params_.end()) fail("'" + name + "' names no parameter in scope");
            return it->second;
        }
        fail("expected a number, pi, a parameter, '(' or a function");
    }

    // Digits, an optional point and digits, an optional exponent with digits.
    double number() {
        const std::size_t start = pos_;
        while (pos_ < text_.size() && is_digit(text_[pos_])) ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && is_digit(text_[pos_])) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            std::size_t e = pos_ + 1;
            if (e < text_.size() && (text_[e] == '+' || text_[e] == '-')) ++e;
            if (e < text_.size() && is_digit(text_[e])) {
                pos_ = e;
                while (pos_ < text_.size() && is_digit(text_[pos_])) ++pos_;
            }
        }
        const std::string_view token = text_.substr(start, pos_ - start);
        try {
            return detail::parse_double(token);
        } catch (const std::out_of_range&) {
            fail("'" + std::string(token) + "' is outside the range of a double");
        }
    }
};

// "1 parameter", "2 qubits", "0 parameters".
std::string count_of(std::size_t n, const char* noun) {
    return std::to_string(n) + " " + noun + (n == 1 ? "" : "s");
}

}  // namespace

// =============================================================================
// QASM 2.0 Parser — supports standard gates + custom gate definitions
// =============================================================================

class QASM2Parser {
public:
    static QuantumCircuit parse(const std::string& qasm) {
        std::istringstream stream(qasm);
        std::string line;
        int n_qubits = 0;
        int n_clbits = 0;
        std::unordered_map<std::string, int> qreg_offsets;
        std::unordered_map<std::string, int> creg_offsets;
        std::unordered_map<std::string, int> qreg_sizes;
        std::unordered_map<std::string, int> creg_sizes;

        // Gate definition library: name -> { param_names, qubit_names, body_lines }
        std::unordered_map<std::string, GateDefinition> gate_defs;

        // First pass: find register sizes and parse gate definitions
        bool in_gate_def = false;
        std::string gate_def_accum;

        while (std::getline(stream, line)) {
            line = strip_comment(line);
            if (line.empty()) continue;

            // Accumulate multi-line gate definitions
            if (in_gate_def) {
                gate_def_accum += " " + line;
                if (line.find('}') != std::string::npos) {
                    in_gate_def = false;
                    parse_gate_definition(gate_def_accum, gate_defs);
                }
                continue;
            }

            if (starts_with_keyword(line, "gate")) {
                gate_def_accum = line;
                if (line.find('}') != std::string::npos) {
                    parse_gate_definition(gate_def_accum, gate_defs);
                } else {
                    in_gate_def = true;
                }
                continue;
            }

            if (starts_with_keyword(line, "qreg")) {
                const std::string decl = single_statement(line);
                auto bracket_pos = decl.find('[');
                auto close_pos = decl.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    std::string reg_name = trim(decl.substr(5, bracket_pos - 5));
                    int reg_size = require_nninteger(
                        trim(decl.substr(bracket_pos + 1, close_pos - bracket_pos - 1)),
                        "qreg size");
                    qreg_offsets[reg_name] = n_qubits;
                    qreg_sizes[reg_name] = reg_size;
                    n_qubits += reg_size;
                }
            } else if (starts_with_keyword(line, "creg")) {
                const std::string decl = single_statement(line);
                auto bracket_pos = decl.find('[');
                auto close_pos = decl.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    std::string reg_name = trim(decl.substr(5, bracket_pos - 5));
                    int reg_size = require_nninteger(
                        trim(decl.substr(bracket_pos + 1, close_pos - bracket_pos - 1)),
                        "creg size");
                    creg_offsets[reg_name] = n_clbits;
                    creg_sizes[reg_name] = reg_size;
                    n_clbits += reg_size;
                }
            }
        }

        if (n_qubits == 0) {
            throw std::runtime_error("No qreg found in QASM");
        }

        QuantumCircuit qc(n_qubits, n_clbits);

        // Second pass: parse gate applications
        stream.clear();
        stream.str(qasm);
        in_gate_def = false;

        // A classical condition peeled off a line is written onto every
        // instruction that line appended, once the line is done. The branches
        // below each `continue` out of the body, so the write happens at the
        // top of the next iteration and once more after the loop, rather than
        // at a tail no branch reaches.
        int cond_clbit = -1;
        int cond_value = 0;
        std::size_t body_start = 0;
        auto apply_pending_condition = [&]() {
            if (cond_clbit < 0) return;
            for (std::size_t k = body_start; k < qc.instructions.size(); ++k) {
                qc.instructions[k].set_condition(cond_clbit, cond_value);
            }
            cond_clbit = -1;
        };

        while (std::getline(stream, line)) {
            apply_pending_condition();
            line = strip_comment(line);
            if (line.empty()) continue;
            if (starts_with_keyword(line, "OPENQASM")) continue;
            if (starts_with_keyword(line, "include")) continue;
            if (starts_with_keyword(line, "qreg")) continue;
            if (starts_with_keyword(line, "creg")) continue;

            // Skip gate definition blocks in second pass
            if (in_gate_def) {
                if (line.find('}') != std::string::npos) in_gate_def = false;
                continue;
            }
            if (starts_with_keyword(line, "gate")) {
                if (line.find('}') == std::string::npos) in_gate_def = true;
                continue;
            }

            line = single_statement(line);

            // Classical condition: `if (creg == value) qop`. OpenQASM 2.0 has
            // only the register-wide form, and Instruction carries a single
            // clbit condition, so the two say the same thing exactly when the
            // register is one bit wide. That is also the only case to_qasm2()
            // ever writes. A wider register has no single-bit meaning and is
            // refused, as the QASM 3 parser refuses it; a silent narrowing
            // would be a different circuit under the caller's name.
            //
            // The guard is peeled off here and the rest of the line falls
            // through to the ordinary parse below. The condition is then
            // written onto every instruction that parse appended, so a custom
            // gate inlined into several primitives is conditioned as a whole.
            body_start = qc.instructions.size();
            if (line.size() > 3 && line.compare(0, 2, "if") == 0 &&
                (line[2] == ' ' || line[2] == '(')) {
                const auto open = line.find('(');
                const auto close = line.find(')', open == std::string::npos ? 0 : open);
                const auto eq = line.find("==");
                if (open == std::string::npos || close == std::string::npos ||
                    eq == std::string::npos || eq < open || eq > close) {
                    throw std::runtime_error(
                        "QASM2Parser: malformed condition '" + line +
                        "'; expected `if (creg == value) qop;`");
                }
                const std::string reg = trim(line.substr(open + 1, eq - open - 1));
                const std::string val = trim(line.substr(eq + 2, close - eq - 2));
                int c_off = 0, c_size = 0;
                if (!resolve_reg_whole(reg, creg_offsets, creg_sizes, c_off, c_size)) {
                    throw std::runtime_error(
                        "QASM2Parser: condition names '" + reg +
                        "', which is not a declared creg");
                }
                if (c_size != 1) {
                    throw std::runtime_error(
                        "QASM2Parser: condition on creg '" + reg + "' of width " +
                        std::to_string(c_size) +
                        "; only a one-bit register maps to a single-bit condition. "
                        "Use OpenQASM 3 for a register-wide comparison.");
                }
                // The value is an OpenQASM 2.0 integer, so `01`, `+1` and `-0`
                // are refused by spelling before 2 is refused by range. An
                // empty value is refused first: reading it as 0 would turn a
                // typo into a condition the author never wrote.
                if (val.empty()) {
                    throw std::runtime_error(
                        "QASM2Parser: condition `if (" + reg + " == )` has no value");
                }
                const int value = require_nninteger(val, "condition value");
                if (value != 0 && value != 1) {
                    throw std::runtime_error(
                        "QASM2Parser: condition value '" + val +
                        "' is not 0 or 1, the only values a one-bit register takes");
                }
                cond_clbit = c_off;
                cond_value = value;
                line = trim(line.substr(close + 1));
                if (line.empty()) {
                    throw std::runtime_error(
                        "QASM2Parser: condition `if (" + reg + " == " + val +
                        ")` guards no instruction");
                }
                // The grammar admits only a quantum operation after `if`: a
                // gate call, measure or reset. A barrier is a directive, and
                // to_qasm2() never writes one under a condition.
                if (starts_with_keyword(line, "barrier")) {
                    throw std::runtime_error(
                        "QASM2Parser: condition `if (" + reg + " == " + val +
                        ")` guards a barrier; OpenQASM 2.0 admits only a gate call, "
                        "measure or reset after `if`");
                }
            }

            // Parse gate name and arguments
            std::string gate_name;
            std::vector<double> params;
            std::vector<int> qubits;

            // Measurement: indexed form `measure q[i] -> c[j];` or the
            // standard whole-register form `measure q -> c;` (expanded to one
            // measurement per bit). Unresolvable operands THROW: silently
            // dropping a measurement corrupts the imported circuit.
            if (starts_with_keyword(line, "measure")) {
                auto arrow = line.find("->");
                if (arrow == std::string::npos)
                    throw std::runtime_error(
                        "QASM2Parser: malformed measure statement: " + line);
                const std::string lhs = line.substr(0, arrow);
                const std::string rhs = line.substr(arrow + 2);
                int q = resolve_reg_index(lhs, qreg_offsets, qreg_sizes, "qreg");
                int c = resolve_reg_index(rhs, creg_offsets, creg_sizes, "creg");
                if (q >= 0 && c >= 0) {
                    qc.measure(q, c);
                    continue;
                }
                int q_off = 0, q_size = 0, c_off = 0, c_size = 0;
                if (resolve_reg_whole(lhs, qreg_offsets, qreg_sizes, q_off, q_size) &&
                    resolve_reg_whole(rhs, creg_offsets, creg_sizes, c_off, c_size)) {
                    if (q_size != c_size)
                        throw std::runtime_error(
                            "QASM2Parser: register size mismatch in '" + line +
                            "' (qreg size " + std::to_string(q_size) +
                            ", creg size " + std::to_string(c_size) + ")");
                    for (int i = 0; i < q_size; ++i)
                        qc.measure(q_off + i, c_off + i);
                    continue;
                }
                throw std::runtime_error(
                    "QASM2Parser: could not resolve measure operands in: " + line);
            }

            // Reset: indexed `reset q[i];` or whole-register `reset q;`.
            if (starts_with_keyword(line, "reset")) {
                int q = resolve_reg_index(line, qreg_offsets, qreg_sizes, "qreg");
                if (q >= 0) {
                    qc.reset(q);
                    continue;
                }
                int q_off = 0, q_size = 0;
                if (resolve_reg_whole(line, qreg_offsets, qreg_sizes, q_off, q_size)) {
                    for (int i = 0; i < q_size; ++i) qc.reset(q_off + i);
                    continue;
                }
                throw std::runtime_error(
                    "QASM2Parser: could not resolve reset operand in: " + line);
            }

            // Barrier: honour the operand list (`barrier q[0], r;`). A bare
            // `barrier;` covers the full register, which is what to_qasm2()
            // writes for a barrier with no operands. An operand naming no
            // declared qreg is refused: widening it to the full register
            // would put a barrier on qubits the author never named.
            if (starts_with_keyword(line, "barrier")) {
                const std::string operand_str = trim(line.substr(7));
                std::vector<int> bq;
                std::istringstream ops(operand_str);
                std::string tok;
                while (std::getline(ops, tok, ',')) {
                    tok = trim(tok);
                    if (tok.empty()) continue;
                    int q = resolve_reg_index(tok, qreg_offsets, qreg_sizes, "qreg");
                    if (q >= 0) {
                        bq.push_back(q);
                        continue;
                    }
                    int q_off = 0, q_size = 0;
                    if (!resolve_reg_whole(tok, qreg_offsets, qreg_sizes, q_off, q_size)) {
                        throw std::runtime_error(
                            "QASM2Parser: barrier operand '" + tok +
                            "' is not a declared qreg");
                    }
                    for (int i = 0; i < q_size; ++i) bq.push_back(q_off + i);
                }
                if (bq.empty()) qc.barrier();
                else qc.barrier(bq);
                continue;
            }

            // Parse gate: name(params) q[i], q[j], ...
            const auto paren_open = line.find('(');
            if (paren_open != std::string::npos) {
                const auto paren_close = matching_paren(line, paren_open);
                if (paren_close == std::string::npos) {
                    throw std::runtime_error("QASM2Parser: unbalanced parentheses in '" + line + "'");
                }
                gate_name = trim(line.substr(0, paren_open));
                params = evaluate_args(line.substr(paren_open + 1, paren_close - paren_open - 1), {});
                qubits = parse_qubits_mapped(line.substr(paren_close + 1), qreg_offsets, qreg_sizes);
            } else {
                const auto space_pos = line.find(' ');
                if (space_pos == std::string::npos) {
                    throw std::runtime_error("QASM2Parser: statement '" + line +
                                             "' names no operands");
                }
                gate_name = line.substr(0, space_pos);
                qubits = parse_qubits_mapped(line.substr(space_pos + 1), qreg_offsets, qreg_sizes);
            }

            // Try built-in gates first, then custom definitions. Unknown gates
            // must surface to the caller: skipping them silently masks parser
            // bugs and gate-set mismatches, and produces round-trip mismatches
            // whose cause looks like it lies in another component.
            if (!try_apply_builtin(qc, gate_name, params, qubits, gate_defs.count(gate_name) != 0)) {
                auto it = gate_defs.find(gate_name);
                if (it != gate_defs.end()) {
                    inline_custom_gate(qc, it->second, params, qubits, gate_defs);
                } else {
                    throw std::runtime_error(
                        "QASM2Parser: unknown gate '" + gate_name +
                        "' (no built-in match and no `gate` definition in scope)");
                }
            }
        }
        apply_pending_condition();

        return qc;
    }

private:
    // =========================================================================
    // Custom gate definition storage
    // =========================================================================

    struct GateDefinition {
        std::string name;
        std::vector<std::string> param_names;   // e.g., {"a", "b"}
        std::vector<std::string> qubit_names;   // e.g., {"p", "q"}
        std::vector<std::string> body_lines;    // e.g., {"rz(a) p", "cx p,q"}
    };

    // Parse "gate name(params) qargs { body }"
    static void parse_gate_definition(
        const std::string& full_def,
        std::unordered_map<std::string, GateDefinition>& gate_defs
    ) {
        GateDefinition def;

        // Strip "gate " prefix
        std::string s = trim(full_def.substr(4));

        // Extract name
        size_t name_end = s.find_first_of("( ");
        if (name_end == std::string::npos) return;
        def.name = trim(s.substr(0, name_end));
        s = s.substr(name_end);

        // Extract parameter names (if any)
        if (!s.empty() && s[0] == '(') {
            auto close = s.find(')');
            if (close != std::string::npos) {
                std::string param_str = s.substr(1, close - 1);
                def.param_names = split_csv(param_str);
                s = s.substr(close + 1);
            }
        }

        // Extract qubit names (everything before '{')
        auto brace_open = s.find('{');
        if (brace_open == std::string::npos) return;
        std::string qarg_str = trim(s.substr(0, brace_open));
        def.qubit_names = split_csv(qarg_str);

        // Extract body (between '{' and '}')
        auto brace_close = s.find('}', brace_open);
        if (brace_close == std::string::npos) return;
        std::string body = s.substr(brace_open + 1, brace_close - brace_open - 1);

        // Split body by semicolons into individual gate calls
        std::istringstream bstream(body);
        std::string stmt;
        while (std::getline(bstream, stmt, ';')) {
            stmt = trim(stmt);
            if (!stmt.empty()) {
                def.body_lines.push_back(stmt);
            }
        }

        gate_defs[def.name] = std::move(def);
    }

    // Inline a custom gate call by substituting params and qubits.
    //
    // Everything the expansion cannot honour is refused rather than dropped: a
    // call with a different number of parameters or qubits than the
    // definition declares, a body statement naming a gate that is neither
    // built in nor defined, a body operand that is not one of the gate's qubit
    // arguments, and a gate that calls itself, directly or through another.
    // `expanding` holds the gates whose bodies are being expanded, outermost
    // first.
    static void inline_custom_gate(
        QuantumCircuit& qc,
        const GateDefinition& def,
        const std::vector<double>& actual_params,
        const std::vector<int>& actual_qubits,
        const std::unordered_map<std::string, GateDefinition>& gate_defs,
        std::vector<std::string> expanding = {}
    ) {
        if (actual_params.size() != def.param_names.size() ||
            actual_qubits.size() != def.qubit_names.size()) {
            throw std::runtime_error(
                "QASM2Parser: gate '" + def.name + "' takes " +
                count_of(def.param_names.size(), "parameter") + " and " +
                count_of(def.qubit_names.size(), "qubit") + "; it was given " +
                count_of(actual_params.size(), "parameter") + " and " +
                count_of(actual_qubits.size(), "qubit"));
        }
        if (std::find(expanding.begin(), expanding.end(), def.name) != expanding.end()) {
            std::string chain;
            for (const std::string& g : expanding) chain += g + " -> ";
            throw std::runtime_error("QASM2Parser: gate '" + def.name +
                                     "' calls itself (" + chain + def.name + ")");
        }
        expanding.push_back(def.name);

        std::unordered_map<std::string, double> param_map;
        for (std::size_t i = 0; i < def.param_names.size(); ++i) {
            param_map[def.param_names[i]] = actual_params[i];
        }
        std::unordered_map<std::string, int> qubit_map;
        for (std::size_t i = 0; i < def.qubit_names.size(); ++i) {
            qubit_map[def.qubit_names[i]] = actual_qubits[i];
        }

        for (const auto& stmt : def.body_lines) {
            std::string gate_name;
            std::vector<double> params;
            std::string qubit_part;

            const auto paren_open = stmt.find('(');
            if (paren_open != std::string::npos) {
                const auto paren_close = matching_paren(stmt, paren_open);
                if (paren_close == std::string::npos) {
                    throw std::runtime_error("QASM2Parser: gate '" + def.name +
                                             "': unbalanced parentheses in '" + stmt + "'");
                }
                gate_name = trim(stmt.substr(0, paren_open));
                params = evaluate_args(stmt.substr(paren_open + 1, paren_close - paren_open - 1),
                                       param_map);
                qubit_part = stmt.substr(paren_close + 1);
            } else {
                const auto space_pos = stmt.find(' ');
                if (space_pos == std::string::npos) {
                    throw std::runtime_error("QASM2Parser: gate '" + def.name + "' has '" + stmt +
                                             "' in its body, which names no operands");
                }
                gate_name = stmt.substr(0, space_pos);
                qubit_part = stmt.substr(space_pos + 1);
            }

            std::vector<int> qubits;
            for (const auto& qname : split_csv(qubit_part)) {
                const auto it = qubit_map.find(qname);
                if (it == qubit_map.end()) {
                    std::string formals;
                    for (std::size_t i = 0; i < def.qubit_names.size(); ++i) {
                        formals += (i > 0 ? ", " : "") + def.qubit_names[i];
                    }
                    throw std::runtime_error("QASM2Parser: gate '" + def.name + "' names '" +
                                             qname +
                                             "' in its body, which is not one of its qubit "
                                             "arguments (" + formals + ")");
                }
                qubits.push_back(it->second);
            }

            // A barrier inside a gate body spans the operands it names.
            if (gate_name == "barrier") {
                qc.barrier(qubits);
                continue;
            }
            if (try_apply_builtin(qc, gate_name, params, qubits, gate_defs.count(gate_name) != 0)) {
                continue;
            }
            const auto it2 = gate_defs.find(gate_name);
            if (it2 == gate_defs.end()) {
                throw std::runtime_error("QASM2Parser: gate '" + def.name + "' calls '" +
                                         gate_name + "', which is neither built in nor defined");
            }
            inline_custom_gate(qc, it2->second, params, qubits, gate_defs, expanding);
        }
    }

    // A gate call's arguments, each an `exp` read whole by Qasm2Expression with
    // `params` (a gate's formal parameters, or none at the top level) in scope.
    static std::vector<double> evaluate_args(
        const std::string& arg_text,
        const std::unordered_map<std::string, double>& params
    ) {
        std::vector<double> values;
        for (const std::string& arg : split_args(arg_text)) {
            values.push_back(Qasm2Expression(arg, params).value());
        }
        return values;
    }

    // The arguments between a call's parentheses, split at the commas outside
    // any nested parentheses and trimmed. "()" holds none; an empty piece
    // ("0.1,,0.3") is kept, so the expression reader refuses it rather than
    // the call silently taking one argument fewer.
    static std::vector<std::string> split_args(const std::string& s) {
        std::vector<std::string> out;
        if (trim(s).empty()) return out;
        int depth = 0;
        std::size_t start = 0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '(') ++depth;
            else if (s[i] == ')') --depth;
            else if (s[i] == ',' && depth == 0) {
                out.push_back(trim(s.substr(start, i - start)));
                start = i + 1;
            }
        }
        out.push_back(trim(s.substr(start)));
        return out;
    }

    // The ')' closing the '(' at `open`, or npos when the line never closes
    // it. An argument holds parentheses of its own (sin(pi/2)), so the first
    // ')' on the line is not necessarily the call's.
    static std::size_t matching_paren(const std::string& s, std::size_t open) {
        int depth = 0;
        for (std::size_t i = open; i < s.size(); ++i) {
            if (s[i] == '(') ++depth;
            else if (s[i] == ')' && --depth == 0) return i;
        }
        return std::string::npos;
    }

    // =========================================================================
    // Utility functions
    // =========================================================================

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    static std::vector<std::string> split_csv(const std::string& s) {
        std::vector<std::string> result;
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (!token.empty()) result.push_back(token);
        }
        return result;
    }

    // True when `c` can continue an identifier.
    static bool is_ident_char(char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    }

    // True when `line` opens with the keyword `kw` as a whole token, so a
    // register or gate whose name merely begins with a keyword ("resets",
    // "gatefoo", "qregs") is never read as that statement.
    static bool starts_with_keyword(const std::string& line, std::string_view kw) {
        return line.compare(0, kw.size(), kw) == 0 &&
               (line.size() == kw.size() || !is_ident_char(line[kw.size()]));
    }

    // The line with any `// comment` removed, trimmed. OpenQASM 2.0 has only
    // line comments. The one statement holding a string, `include`, is skipped
    // whole in both passes, so cutting at the first "//" never splits a
    // statement that is parsed.
    static std::string strip_comment(const std::string& line) {
        return trim(line.substr(0, line.find("//")));
    }

    // One statement with its terminating ';' removed. The parser reads a line
    // as one statement, so a second ';' means a second statement it would not
    // parse, and it is refused rather than dropped.
    static std::string single_statement(const std::string& line) {
        std::string stmt = line;
        if (!stmt.empty() && stmt.back() == ';') stmt.pop_back();
        if (stmt.find(';') != std::string::npos) {
            throw std::runtime_error(
                "QASM2Parser: '" + line + "' holds more than one statement; this "
                "parser reads one statement per line");
        }
        return stmt;
    }

    // OpenQASM 2.0's nninteger, the only integer its grammar has: "0", or a
    // non-zero digit followed by digits, with no sign. `text` must already be
    // trimmed. Anything else throws naming `what`, and so does a value too
    // large for an int.
    static int require_nninteger(const std::string& text, const std::string& what) {
        const bool digits_only =
            !text.empty() &&
            std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; });
        if (!digits_only || (text.size() > 1 && text[0] == '0')) {
            throw std::runtime_error(
                "QASM2Parser: " + what + " '" + text + "' is not an OpenQASM 2.0 "
                "integer (0, or digits with no leading zero, and no sign)");
        }
        int value = 0;
        const char* const end = text.data() + text.size();
        const auto [stop, ec] = std::from_chars(text.data(), end, value);
        if (ec != std::errc() || stop != end) {
            throw std::runtime_error(
                "QASM2Parser: " + what + " '" + text + "' does not fit an int");
        }
        return value;
    }

    // Global index of `name[index_text]`. Throws when `name` is not a declared
    // register of this kind, when the index is not an OpenQASM 2.0 integer, and
    // when it lies outside the register: offset + index would otherwise land on
    // a bit of the next register, which QuantumCircuit's own bounds check cannot
    // tell from a legitimate one.
    static int indexed_operand(const std::string& name, const std::string& index_text,
                               const std::unordered_map<std::string, int>& offsets,
                               const std::unordered_map<std::string, int>& sizes,
                               const char* kind) {
        const auto it = offsets.find(name);
        if (it == offsets.end()) {
            throw std::runtime_error("QASM2Parser: '" + name + "' is not a declared " +
                                     kind);
        }
        const int index = require_nninteger(trim(index_text), std::string(kind) + " index");
        const int size = sizes.at(name);
        if (index >= size) {
            throw std::runtime_error(
                "QASM2Parser: index " + std::to_string(index) + " is out of range for " +
                kind + " '" + name + "' of size " + std::to_string(size));
        }
        return it->second + index;
    }

    // The indexed operand in `s`, which may still carry its statement keyword
    // ("measure q[0] ", "reset q[1]"): the identifier directly before the first
    // '[', whitespace between them allowed, then the index up to ']'. Returns
    // -1 when `s` has no '[', so the caller can try the whole-register form.
    static int resolve_reg_index(const std::string& s,
                                 const std::unordered_map<std::string, int>& offsets,
                                 const std::unordered_map<std::string, int>& sizes,
                                 const char* kind) {
        const auto open = s.find('[');
        if (open == std::string::npos) return -1;
        const auto close = s.find(']', open);
        if (close == std::string::npos) {
            throw std::runtime_error("QASM2Parser: unterminated index in '" + trim(s) + "'");
        }
        std::size_t end = open;
        while (end > 0 && (s[end - 1] == ' ' || s[end - 1] == '\t')) --end;
        std::size_t start = end;
        while (start > 0 && is_ident_char(s[start - 1])) --start;
        return indexed_operand(s.substr(start, end - start),
                               s.substr(open + 1, close - open - 1), offsets, sizes, kind);
    }

    // Resolve a BARE register reference (no [index]) inside `s`: matches a
    // known register name as a whole token (not a substring of a longer
    // identifier, and not followed by '[') and returns its offset and size.
    static bool resolve_reg_whole(
        const std::string& s,
        const std::unordered_map<std::string, int>& offsets,
        const std::unordered_map<std::string, int>& sizes,
        int& offset_out, int& size_out
    ) {
        auto is_ident = [](char ch) {
            return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
        };
        for (const auto& [name, offset] : offsets) {
            auto pos = s.find(name);
            while (pos != std::string::npos) {
                const bool left_ok = (pos == 0) || !is_ident(s[pos - 1]);
                const size_t end = pos + name.size();
                const bool right_ok =
                    (end >= s.size()) || (!is_ident(s[end]) && s[end] != '[');
                if (left_ok && right_ok) {
                    offset_out = offset;
                    auto it = sizes.find(name);
                    size_out = (it != sizes.end()) ? it->second : 0;
                    return size_out > 0;
                }
                pos = s.find(name, pos + 1);
            }
        }
        return false;
    }

    // Parse "reg[i], reg[j], ..." into global qubit indices. Each indexed token
    // is exactly `name[index]`: the name is looked up whole and nothing may
    // follow the ']'.
    static std::vector<int> parse_qubits_mapped(
        const std::string& s,
        const std::unordered_map<std::string, int>& offsets,
        const std::unordered_map<std::string, int>& sizes
    ) {
        std::vector<int> qubits;
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            auto bracket = token.find('[');
            if (bracket == std::string::npos) continue;
            auto close = token.find(']', bracket);
            if (close == std::string::npos || close + 1 != token.size()) {
                throw std::runtime_error("QASM2Parser: malformed qubit operand '" + token +
                                         "'");
            }
            qubits.push_back(indexed_operand(trim(token.substr(0, bracket)),
                                             token.substr(bracket + 1, close - bracket - 1),
                                             offsets, sizes, "qreg"));
        }
        return qubits;
    }

    // A gate the reader applies directly: the grammar's own U and CX, and the
    // qelib1.inc gates, whose include line is accepted without being read.
    struct BuiltinGate {
        const char* name;
        std::size_t n_params;
        std::size_t n_qubits;
        void (*apply)(QuantumCircuit&, const std::vector<double>&, const std::vector<int>&);
    };

    static const std::vector<BuiltinGate>& builtin_gates() {
        using P = const std::vector<double>&;
        using Q = const std::vector<int>&;
        static const std::vector<BuiltinGate> gates = {
            {"U", 3, 1, [](QuantumCircuit& c, P p, Q q) { c.u(p[0], p[1], p[2], q[0]); }},
            {"CX", 0, 2, [](QuantumCircuit& c, P, Q q) { c.cx(q[0], q[1]); }},
            {"h", 0, 1, [](QuantumCircuit& c, P, Q q) { c.h(q[0]); }},
            {"x", 0, 1, [](QuantumCircuit& c, P, Q q) { c.x(q[0]); }},
            {"y", 0, 1, [](QuantumCircuit& c, P, Q q) { c.y(q[0]); }},
            {"z", 0, 1, [](QuantumCircuit& c, P, Q q) { c.z(q[0]); }},
            {"s", 0, 1, [](QuantumCircuit& c, P, Q q) { c.s(q[0]); }},
            {"sdg", 0, 1, [](QuantumCircuit& c, P, Q q) { c.sdg(q[0]); }},
            {"t", 0, 1, [](QuantumCircuit& c, P, Q q) { c.t(q[0]); }},
            {"tdg", 0, 1, [](QuantumCircuit& c, P, Q q) { c.tdg(q[0]); }},
            {"sx", 0, 1, [](QuantumCircuit& c, P, Q q) { c.sx(q[0]); }},
            {"rx", 1, 1, [](QuantumCircuit& c, P p, Q q) { c.rx(p[0], q[0]); }},
            {"ry", 1, 1, [](QuantumCircuit& c, P p, Q q) { c.ry(p[0], q[0]); }},
            {"rz", 1, 1, [](QuantumCircuit& c, P p, Q q) { c.rz(p[0], q[0]); }},
            {"p", 1, 1, [](QuantumCircuit& c, P p, Q q) { c.p(p[0], q[0]); }},
            {"u", 3, 1, [](QuantumCircuit& c, P p, Q q) { c.u(p[0], p[1], p[2], q[0]); }},
            {"u1", 1, 1, [](QuantumCircuit& c, P p, Q q) { c.u1(p[0], q[0]); }},
            {"u2", 2, 1, [](QuantumCircuit& c, P p, Q q) { c.u2(p[0], p[1], q[0]); }},
            {"u3", 3, 1, [](QuantumCircuit& c, P p, Q q) { c.u3(p[0], p[1], p[2], q[0]); }},
            {"cx", 0, 2, [](QuantumCircuit& c, P, Q q) { c.cx(q[0], q[1]); }},
            {"cy", 0, 2, [](QuantumCircuit& c, P, Q q) { c.cy(q[0], q[1]); }},
            {"cz", 0, 2, [](QuantumCircuit& c, P, Q q) { c.cz(q[0], q[1]); }},
            {"ch", 0, 2, [](QuantumCircuit& c, P, Q q) { c.ch(q[0], q[1]); }},
            {"swap", 0, 2, [](QuantumCircuit& c, P, Q q) { c.swap(q[0], q[1]); }},
            {"crx", 1, 2, [](QuantumCircuit& c, P p, Q q) { c.crx(p[0], q[0], q[1]); }},
            {"cry", 1, 2, [](QuantumCircuit& c, P p, Q q) { c.cry(p[0], q[0], q[1]); }},
            {"crz", 1, 2, [](QuantumCircuit& c, P p, Q q) { c.crz(p[0], q[0], q[1]); }},
            {"cp", 1, 2, [](QuantumCircuit& c, P p, Q q) { c.cp(p[0], q[0], q[1]); }},
            {"ccx", 0, 3, [](QuantumCircuit& c, P, Q q) { c.ccx(q[0], q[1], q[2]); }},
            {"cswap", 0, 3, [](QuantumCircuit& c, P, Q q) { c.cswap(q[0], q[1], q[2]); }},
            {"rxx", 1, 2, [](QuantumCircuit& c, P p, Q q) { c.rxx(p[0], q[0], q[1]); }},
            {"ryy", 1, 2, [](QuantumCircuit& c, P p, Q q) { c.ryy(p[0], q[0], q[1]); }},
            {"rzz", 1, 2, [](QuantumCircuit& c, P p, Q q) { c.rzz(p[0], q[0], q[1]); }},
        };
        return gates;
    }

    // Applies `name` when it is a built-in gate given exactly its parameters
    // and qubits, and returns whether it did. A built-in name called with
    // other counts is refused, unless the file defines a gate of that name,
    // whose definition then takes the call (`defined_in_file`).
    static bool try_apply_builtin(QuantumCircuit& qc, const std::string& name,
                                  const std::vector<double>& params,
                                  const std::vector<int>& qubits, bool defined_in_file) {
        for (const BuiltinGate& g : builtin_gates()) {
            if (name != g.name) continue;
            if (params.size() == g.n_params && qubits.size() == g.n_qubits) {
                g.apply(qc, params, qubits);
                return true;
            }
            if (defined_in_file) return false;
            throw std::runtime_error(
                "QASM2Parser: gate '" + name + "' takes " + count_of(g.n_params, "parameter") +
                " and " + count_of(g.n_qubits, "qubit") + "; it was given " +
                count_of(params.size(), "parameter") + " and " + count_of(qubits.size(), "qubit"));
        }
        return false;
    }
};

// Bridge function so circuit.cpp can call the parser without including this
// translation unit's internal class definition.
QuantumCircuit qasm2_parse_impl(const std::string& qasm) {
    return QASM2Parser::parse(qasm);
}

} // namespace lindblad
