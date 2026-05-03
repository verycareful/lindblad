#include "qpp/circuit.hpp"

#include <sstream>
#include <stdexcept>
#include <regex>
#include <unordered_map>

namespace qpp {

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

        // Gate definition library: name -> { param_names, qubit_names, body_lines }
        std::unordered_map<std::string, GateDefinition> gate_defs;

        // First pass: find register sizes and parse gate definitions
        bool in_gate_def = false;
        std::string gate_def_accum;

        while (std::getline(stream, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '/' || line.substr(0, 2) == "//") continue;

            // Accumulate multi-line gate definitions
            if (in_gate_def) {
                gate_def_accum += " " + line;
                if (line.find('}') != std::string::npos) {
                    in_gate_def = false;
                    parse_gate_definition(gate_def_accum, gate_defs);
                }
                continue;
            }

            if (line.substr(0, 4) == "gate") {
                gate_def_accum = line;
                if (line.find('}') != std::string::npos) {
                    parse_gate_definition(gate_def_accum, gate_defs);
                } else {
                    in_gate_def = true;
                }
                continue;
            }

            if (line.find("qreg") != std::string::npos) {
                auto bracket_pos = line.find('[');
                auto close_pos = line.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    std::string reg_name = trim(line.substr(5, bracket_pos - 5));
                    int reg_size = std::stoi(line.substr(bracket_pos + 1, close_pos - bracket_pos - 1));
                    qreg_offsets[reg_name] = n_qubits;
                    n_qubits += reg_size;
                }
            } else if (line.find("creg") != std::string::npos) {
                auto bracket_pos = line.find('[');
                auto close_pos = line.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    std::string reg_name = trim(line.substr(5, bracket_pos - 5));
                    int reg_size = std::stoi(line.substr(bracket_pos + 1, close_pos - bracket_pos - 1));
                    creg_offsets[reg_name] = n_clbits;
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

        while (std::getline(stream, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '/' || line.substr(0, 2) == "//") continue;
            if (line.find("OPENQASM") != std::string::npos) continue;
            if (line.find("include") != std::string::npos) continue;
            if (line.find("qreg") != std::string::npos) continue;
            if (line.find("creg") != std::string::npos) continue;

            // Skip gate definition blocks in second pass
            if (in_gate_def) {
                if (line.find('}') != std::string::npos) in_gate_def = false;
                continue;
            }
            if (line.substr(0, 4) == "gate") {
                if (line.find('}') == std::string::npos) in_gate_def = true;
                continue;
            }

            // Remove semicolon
            if (!line.empty() && line.back() == ';') {
                line.pop_back();
            }

            // Parse gate name and arguments
            std::string gate_name;
            std::vector<double> params;
            std::vector<int> qubits;

            // Check for measurement
            if (line.find("measure") != std::string::npos) {
                auto arrow = line.find("->");
                if (arrow != std::string::npos) {
                    int q = resolve_reg_index(line.substr(0, arrow), qreg_offsets);
                    int c = resolve_reg_index(line.substr(arrow + 2), creg_offsets);
                    if (q >= 0 && c >= 0) qc.measure(q, c);
                }
                continue;
            }

            if (line.find("reset") != std::string::npos) {
                int q = resolve_reg_index(line, qreg_offsets);
                if (q >= 0) qc.reset(q);
                continue;
            }

            if (line.find("barrier") != std::string::npos) {
                qc.barrier();
                continue;
            }

            // Parse gate: name(params) q[i], q[j], ...
            auto paren_open = line.find('(');
            auto paren_close = line.find(')');

            if (paren_open != std::string::npos && paren_close != std::string::npos) {
                gate_name = trim(line.substr(0, paren_open));
                std::string param_str = line.substr(paren_open + 1, paren_close - paren_open - 1);
                params = parse_params(param_str);

                std::string qubit_str = line.substr(paren_close + 1);
                qubits = parse_qubits_mapped(qubit_str, qreg_offsets);
            } else {
                // No parameters
                auto space_pos = line.find(' ');
                if (space_pos != std::string::npos) {
                    gate_name = line.substr(0, space_pos);
                    std::string qubit_str = line.substr(space_pos + 1);
                    qubits = parse_qubits_mapped(qubit_str, qreg_offsets);
                } else {
                    continue;
                }
            }

            // Try built-in gates first, then custom definitions
            if (!try_apply_builtin(qc, gate_name, params, qubits)) {
                auto it = gate_defs.find(gate_name);
                if (it != gate_defs.end()) {
                    inline_custom_gate(qc, it->second, params, qubits, gate_defs);
                }
                // Silently skip unknown gates (consistent with prior behavior)
            }
        }

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

    // Inline a custom gate call by substituting params and qubits
    static void inline_custom_gate(
        QuantumCircuit& qc,
        const GateDefinition& def,
        const std::vector<double>& actual_params,
        const std::vector<int>& actual_qubits,
        const std::unordered_map<std::string, GateDefinition>& gate_defs
    ) {
        // Build param substitution map: formal_name -> actual_value
        std::unordered_map<std::string, double> param_map;
        for (size_t i = 0; i < def.param_names.size() && i < actual_params.size(); ++i) {
            param_map[def.param_names[i]] = actual_params[i];
        }

        // Build qubit substitution map: formal_name -> actual_qubit_index
        std::unordered_map<std::string, int> qubit_map;
        for (size_t i = 0; i < def.qubit_names.size() && i < actual_qubits.size(); ++i) {
            qubit_map[def.qubit_names[i]] = actual_qubits[i];
        }

        // Process each body line
        for (const auto& stmt : def.body_lines) {
            std::string gate_name;
            std::vector<double> params;
            std::vector<int> qubits;

            auto paren_open = stmt.find('(');
            auto paren_close = stmt.find(')');

            std::string qubit_part;

            if (paren_open != std::string::npos && paren_close != std::string::npos) {
                gate_name = trim(stmt.substr(0, paren_open));
                std::string param_str = stmt.substr(paren_open + 1, paren_close - paren_open - 1);
                params = resolve_params(param_str, param_map);
                qubit_part = stmt.substr(paren_close + 1);
            } else {
                auto space_pos = stmt.find(' ');
                if (space_pos == std::string::npos) continue;
                gate_name = stmt.substr(0, space_pos);
                qubit_part = stmt.substr(space_pos + 1);
            }

            // Resolve qubit names to actual indices
            auto qubit_names = split_csv(qubit_part);
            for (const auto& qname : qubit_names) {
                auto it = qubit_map.find(qname);
                if (it != qubit_map.end()) {
                    qubits.push_back(it->second);
                }
            }

            // Apply: try built-in first, then recurse into custom defs
            if (!try_apply_builtin(qc, gate_name, params, qubits)) {
                auto it2 = gate_defs.find(gate_name);
                if (it2 != gate_defs.end()) {
                    inline_custom_gate(qc, it2->second, params, qubits, gate_defs);
                }
            }
        }
    }

    // Resolve parameter expressions, substituting formal names with actual values
    static std::vector<double> resolve_params(
        const std::string& param_str,
        const std::unordered_map<std::string, double>& param_map
    ) {
        std::vector<double> params;
        auto tokens = split_csv(param_str);
        for (const auto& tok : tokens) {
            // Check if the token is a known parameter name
            auto it = param_map.find(tok);
            if (it != param_map.end()) {
                params.push_back(it->second);
            } else {
                // Check for negated parameter: "-param"
                if (!tok.empty() && tok[0] == '-') {
                    std::string inner = trim(tok.substr(1));
                    auto it2 = param_map.find(inner);
                    if (it2 != param_map.end()) {
                        params.push_back(-it2->second);
                        continue;
                    }
                }
                // Try as a numeric / pi expression
                try {
                    params.push_back(evaluate_pi_expr(tok));
                } catch (...) {
                    // Try to evaluate as "param_name op value" expressions
                    // e.g., "a/2", "a+pi"
                    params.push_back(evaluate_param_expr(tok, param_map));
                }
            }
        }
        return params;
    }

    // Evaluate simple arithmetic expressions involving parameter names
    // Handles: "a/2", "a+pi", "a-pi/2", "2*a"
    static double evaluate_param_expr(
        const std::string& expr,
        const std::unordered_map<std::string, double>& param_map
    ) {
        // Try division: "a/N"
        auto div_pos = expr.find('/');
        if (div_pos != std::string::npos) {
            std::string lhs = trim(expr.substr(0, div_pos));
            std::string rhs = trim(expr.substr(div_pos + 1));
            double l = resolve_single(lhs, param_map);
            double r = resolve_single(rhs, param_map);
            return (r != 0.0) ? l / r : 0.0;
        }

        // Try multiplication: "N*a" or "a*N"
        auto mul_pos = expr.find('*');
        if (mul_pos != std::string::npos) {
            std::string lhs = trim(expr.substr(0, mul_pos));
            std::string rhs = trim(expr.substr(mul_pos + 1));
            return resolve_single(lhs, param_map) * resolve_single(rhs, param_map);
        }

        // Try addition/subtraction (scan from right to handle -pi correctly)
        for (int i = static_cast<int>(expr.size()) - 1; i > 0; --i) {
            if (expr[i] == '+' || expr[i] == '-') {
                std::string lhs = trim(expr.substr(0, i));
                std::string rhs = trim(expr.substr(i + 1));
                double l = resolve_single(lhs, param_map);
                double r = resolve_single(rhs, param_map);
                return (expr[i] == '+') ? l + r : l - r;
            }
        }

        return resolve_single(expr, param_map);
    }

    // Resolve a single token: either a param name, pi, or a number
    static double resolve_single(
        const std::string& tok,
        const std::unordered_map<std::string, double>& param_map
    ) {
        auto it = param_map.find(tok);
        if (it != param_map.end()) return it->second;
        try {
            return evaluate_pi_expr(tok);
        } catch (...) {
            return 0.0;
        }
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

    // Find reg[idx] in `s` using the offset map; returns global qubit/clbit index.
    static int resolve_reg_index(
        const std::string& s,
        const std::unordered_map<std::string, int>& offsets
    ) {
        for (const auto& [name, offset] : offsets) {
            std::string pat = name + "[";
            auto pos = s.find(pat);
            if (pos == std::string::npos) continue;
            auto bracket = pos + pat.size();
            auto close = s.find(']', bracket);
            if (close == std::string::npos) continue;
            return offset + std::stoi(s.substr(bracket, close - bracket));
        }
        return -1;
    }

    // Parse "reg[i], reg[j], ..." using register offset map for global indices.
    static std::vector<int> parse_qubits_mapped(
        const std::string& s,
        const std::unordered_map<std::string, int>& offsets
    ) {
        if (offsets.empty()) return parse_qubits(s);
        std::vector<int> qubits;
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            auto bracket = token.find('[');
            if (bracket == std::string::npos) continue;
            auto close = token.find(']', bracket);
            if (close == std::string::npos) continue;
            std::string reg_name = trim(token.substr(0, bracket));
            int idx = std::stoi(token.substr(bracket + 1, close - bracket - 1));
            auto it = offsets.find(reg_name);
            int offset = (it != offsets.end()) ? it->second : 0;
            qubits.push_back(offset + idx);
        }
        return qubits;
    }

    static int extract_qubit(const std::string& line, const std::string& reg_name) {
        auto pos = line.find(reg_name + "[");
        if (pos == std::string::npos) return -1;
        auto bracket = pos + reg_name.size() + 1;
        auto close = line.find(']', bracket);
        if (close == std::string::npos) return -1;
        return std::stoi(line.substr(bracket, close - bracket));
    }

    // Evaluate a single parameter token that may contain 'pi' expressions.
    // Handles: pi, -pi, pi/N, N*pi, -N*pi, N*pi/M, -pi/N etc.
    static double evaluate_pi_expr(const std::string& tok) {
        if (tok == "pi") return PI;
        if (tok == "-pi") return -PI;

        auto pi_pos = tok.find("pi");
        if (pi_pos == std::string::npos) {
            return std::stod(tok);
        }

        // Split around "pi": prefix * pi / suffix
        std::string pre = tok.substr(0, pi_pos);
        std::string post = tok.substr(pi_pos + 2);

        double val = PI;

        // Handle prefix: "", "-", "N*", "-N*"
        if (pre.empty()) {
            // just pi...
        } else if (pre == "-") {
            val = -val;
        } else {
            // Strip trailing '*' if present
            if (!pre.empty() && pre.back() == '*') pre.pop_back();
            if (pre == "-") {
                val = -val;
            } else if (!pre.empty()) {
                val *= std::stod(pre);
            }
        }

        // Handle suffix: "", "/N"
        if (!post.empty() && post[0] == '/') {
            val /= std::stod(post.substr(1));
        } else if (!post.empty() && post[0] == '*') {
            val *= std::stod(post.substr(1));
        }

        return val;
    }

    static std::vector<double> parse_params(const std::string& s) {
        std::vector<double> params;
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (!token.empty()) {
                try {
                    params.push_back(evaluate_pi_expr(token));
                } catch (...) {
                    params.push_back(0.0);
                }
            }
        }
        return params;
    }

    static std::vector<int> parse_qubits(const std::string& s) {
        std::vector<int> qubits;
        size_t pos = 0;
        while (pos < s.size()) {
            auto bracket = s.find('[', pos);
            if (bracket == std::string::npos) break;
            auto close = s.find(']', bracket);
            if (close == std::string::npos) break;
            qubits.push_back(std::stoi(s.substr(bracket + 1, close - bracket - 1)));
            pos = close + 1;
        }
        return qubits;
    }

    // Try to apply a built-in gate. Returns true if recognized.
    static bool try_apply_builtin(QuantumCircuit& qc, const std::string& name,
                                   const std::vector<double>& params,
                                   const std::vector<int>& qubits) {
        if (name == "h" && qubits.size() == 1) { qc.h(qubits[0]); return true; }
        if (name == "x" && qubits.size() == 1) { qc.x(qubits[0]); return true; }
        if (name == "y" && qubits.size() == 1) { qc.y(qubits[0]); return true; }
        if (name == "z" && qubits.size() == 1) { qc.z(qubits[0]); return true; }
        if (name == "s" && qubits.size() == 1) { qc.s(qubits[0]); return true; }
        if (name == "sdg" && qubits.size() == 1) { qc.sdg(qubits[0]); return true; }
        if (name == "t" && qubits.size() == 1) { qc.t(qubits[0]); return true; }
        if (name == "tdg" && qubits.size() == 1) { qc.tdg(qubits[0]); return true; }
        if (name == "sx" && qubits.size() == 1) { qc.sx(qubits[0]); return true; }
        if (name == "rx" && params.size() >= 1 && qubits.size() == 1) { qc.rx(params[0], qubits[0]); return true; }
        if (name == "ry" && params.size() >= 1 && qubits.size() == 1) { qc.ry(params[0], qubits[0]); return true; }
        if (name == "rz" && params.size() >= 1 && qubits.size() == 1) { qc.rz(params[0], qubits[0]); return true; }
        if (name == "p" && params.size() >= 1 && qubits.size() == 1) { qc.p(params[0], qubits[0]); return true; }
        if (name == "u" && params.size() >= 3 && qubits.size() == 1) { qc.u(params[0], params[1], params[2], qubits[0]); return true; }
        if (name == "u1" && params.size() >= 1 && qubits.size() == 1) { qc.u1(params[0], qubits[0]); return true; }
        if (name == "u2" && params.size() >= 2 && qubits.size() == 1) { qc.u2(params[0], params[1], qubits[0]); return true; }
        if (name == "u3" && params.size() >= 3 && qubits.size() == 1) { qc.u3(params[0], params[1], params[2], qubits[0]); return true; }
        if (name == "cx" && qubits.size() == 2) { qc.cx(qubits[0], qubits[1]); return true; }
        if (name == "cy" && qubits.size() == 2) { qc.cy(qubits[0], qubits[1]); return true; }
        if (name == "cz" && qubits.size() == 2) { qc.cz(qubits[0], qubits[1]); return true; }
        if (name == "ch" && qubits.size() == 2) { qc.ch(qubits[0], qubits[1]); return true; }
        if (name == "swap" && qubits.size() == 2) { qc.swap(qubits[0], qubits[1]); return true; }
        if (name == "crx" && params.size() >= 1 && qubits.size() == 2) { qc.crx(params[0], qubits[0], qubits[1]); return true; }
        if (name == "cry" && params.size() >= 1 && qubits.size() == 2) { qc.cry(params[0], qubits[0], qubits[1]); return true; }
        if (name == "crz" && params.size() >= 1 && qubits.size() == 2) { qc.crz(params[0], qubits[0], qubits[1]); return true; }
        if (name == "cp" && params.size() >= 1 && qubits.size() == 2) { qc.cp(params[0], qubits[0], qubits[1]); return true; }
        if (name == "ccx" && qubits.size() == 3) { qc.ccx(qubits[0], qubits[1], qubits[2]); return true; }
        if (name == "cswap" && qubits.size() == 3) { qc.cswap(qubits[0], qubits[1], qubits[2]); return true; }
        if (name == "rxx" && params.size() >= 1 && qubits.size() == 2) { qc.rxx(params[0], qubits[0], qubits[1]); return true; }
        if (name == "ryy" && params.size() >= 1 && qubits.size() == 2) { qc.ryy(params[0], qubits[0], qubits[1]); return true; }
        if (name == "rzz" && params.size() >= 1 && qubits.size() == 2) { qc.rzz(params[0], qubits[0], qubits[1]); return true; }
        return false;
    }
};

// Bridge function so circuit.cpp can call the parser without including this
// translation unit's internal class definition.
QuantumCircuit qasm2_parse_impl(const std::string& qasm) {
    return QASM2Parser::parse(qasm);
}

} // namespace qpp
