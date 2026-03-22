#include "qpp/circuit.hpp"

#include <sstream>
#include <stdexcept>
#include <regex>

namespace qpp {

// =============================================================================
// Basic QASM 2.0 Parser
// =============================================================================

class QASM2Parser {
public:
    static QuantumCircuit parse(const std::string& qasm) {
        std::istringstream stream(qasm);
        std::string line;
        int n_qubits = 0;
        int n_clbits = 0;

        // First pass: find register sizes
        while (std::getline(stream, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '/' || line.substr(0, 2) == "//") continue;

            if (line.find("qreg") != std::string::npos) {
                auto bracket_pos = line.find('[');
                auto close_pos = line.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    n_qubits = std::stoi(line.substr(bracket_pos + 1, close_pos - bracket_pos - 1));
                }
            } else if (line.find("creg") != std::string::npos) {
                auto bracket_pos = line.find('[');
                auto close_pos = line.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    n_clbits = std::stoi(line.substr(bracket_pos + 1, close_pos - bracket_pos - 1));
                }
            }
        }

        if (n_qubits == 0) {
            throw std::runtime_error("No qreg found in QASM");
        }

        QuantumCircuit qc(n_qubits, n_clbits);

        // Second pass: parse gates
        stream.clear();
        stream.str(qasm);

        while (std::getline(stream, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '/' || line.substr(0, 2) == "//") continue;
            if (line.find("OPENQASM") != std::string::npos) continue;
            if (line.find("include") != std::string::npos) continue;
            if (line.find("qreg") != std::string::npos) continue;
            if (line.find("creg") != std::string::npos) continue;

            // Remove semicolon
            if (!line.empty() && line.back() == ';') {
                line.pop_back();
            }

            // Parse gate name and arguments
            std::string gate_name;
            std::vector<double> params;
            std::vector<int> qubits;
            std::vector<int> clbits;

            // Check for measurement
            if (line.find("measure") != std::string::npos) {
                auto q = extract_qubit(line, "q");
                auto c = extract_qubit(line, "c");
                if (q >= 0 && c >= 0) {
                    qc.measure(q, c);
                }
                continue;
            }

            if (line.find("reset") != std::string::npos) {
                auto q = extract_qubit(line, "q");
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
                qubits = parse_qubits(qubit_str);
            } else {
                // No parameters
                auto space_pos = line.find(' ');
                if (space_pos != std::string::npos) {
                    gate_name = line.substr(0, space_pos);
                    std::string qubit_str = line.substr(space_pos + 1);
                    qubits = parse_qubits(qubit_str);
                } else {
                    continue;
                }
            }

            // Map gate name to circuit method
            apply_gate(qc, gate_name, params, qubits);
        }

        return qc;
    }

private:
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
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

    static void apply_gate(QuantumCircuit& qc, const std::string& name,
                           const std::vector<double>& params,
                           const std::vector<int>& qubits) {
        if (name == "h" && qubits.size() == 1) qc.h(qubits[0]);
        else if (name == "x" && qubits.size() == 1) qc.x(qubits[0]);
        else if (name == "y" && qubits.size() == 1) qc.y(qubits[0]);
        else if (name == "z" && qubits.size() == 1) qc.z(qubits[0]);
        else if (name == "s" && qubits.size() == 1) qc.s(qubits[0]);
        else if (name == "sdg" && qubits.size() == 1) qc.sdg(qubits[0]);
        else if (name == "t" && qubits.size() == 1) qc.t(qubits[0]);
        else if (name == "tdg" && qubits.size() == 1) qc.tdg(qubits[0]);
        else if (name == "sx" && qubits.size() == 1) qc.sx(qubits[0]);
        else if (name == "rx" && params.size() >= 1 && qubits.size() == 1) qc.rx(params[0], qubits[0]);
        else if (name == "ry" && params.size() >= 1 && qubits.size() == 1) qc.ry(params[0], qubits[0]);
        else if (name == "rz" && params.size() >= 1 && qubits.size() == 1) qc.rz(params[0], qubits[0]);
        else if (name == "p" && params.size() >= 1 && qubits.size() == 1) qc.p(params[0], qubits[0]);
        else if (name == "u" && params.size() >= 3 && qubits.size() == 1) qc.u(params[0], params[1], params[2], qubits[0]);
        else if (name == "u1" && params.size() >= 1 && qubits.size() == 1) qc.u1(params[0], qubits[0]);
        else if (name == "u2" && params.size() >= 2 && qubits.size() == 1) qc.u2(params[0], params[1], qubits[0]);
        else if (name == "u3" && params.size() >= 3 && qubits.size() == 1) qc.u3(params[0], params[1], params[2], qubits[0]);
        else if (name == "cx" && qubits.size() == 2) qc.cx(qubits[0], qubits[1]);
        else if (name == "cy" && qubits.size() == 2) qc.cy(qubits[0], qubits[1]);
        else if (name == "cz" && qubits.size() == 2) qc.cz(qubits[0], qubits[1]);
        else if (name == "ch" && qubits.size() == 2) qc.ch(qubits[0], qubits[1]);
        else if (name == "swap" && qubits.size() == 2) qc.swap(qubits[0], qubits[1]);
        else if (name == "crx" && params.size() >= 1 && qubits.size() == 2) qc.crx(params[0], qubits[0], qubits[1]);
        else if (name == "cry" && params.size() >= 1 && qubits.size() == 2) qc.cry(params[0], qubits[0], qubits[1]);
        else if (name == "crz" && params.size() >= 1 && qubits.size() == 2) qc.crz(params[0], qubits[0], qubits[1]);
        else if (name == "cp" && params.size() >= 1 && qubits.size() == 2) qc.cp(params[0], qubits[0], qubits[1]);
        else if (name == "ccx" && qubits.size() == 3) qc.ccx(qubits[0], qubits[1], qubits[2]);
        else if (name == "cswap" && qubits.size() == 3) qc.cswap(qubits[0], qubits[1], qubits[2]);
        else if (name == "rxx" && params.size() >= 1 && qubits.size() == 2) qc.rxx(params[0], qubits[0], qubits[1]);
        else if (name == "ryy" && params.size() >= 1 && qubits.size() == 2) qc.ryy(params[0], qubits[0], qubits[1]);
        else if (name == "rzz" && params.size() >= 1 && qubits.size() == 2) qc.rzz(params[0], qubits[0], qubits[1]);
    }
};

} // namespace qpp
