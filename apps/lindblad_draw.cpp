// =============================================================================
// apps/lindblad_draw.cpp : R.1.10.3 CLI circuit visualiser
// =============================================================================
// Command-line front-end to QuantumCircuit::draw(). Reads a circuit from a
// QASM2 / QASM3 file (or stdin when --stdin is set) and writes the rendered
// diagram to stdout in any of the four backends (ASCII, SVG, LaTeX, HTML).
//
// Built-in demo circuits (bell, ghz, qft, parametric, tallbox) cover the
// "I just want to see what the visualiser produces" use case without
// requiring a QASM file. Run with --list-demos for the full list.
//
// Usage:
//   lindblad_draw bell.qasm
//   lindblad_draw --mode svg bell.qasm > bell.svg
//   lindblad_draw --demo bell
//   lindblad_draw --mode latex --demo ghz
//   cat bell.qasm | lindblad_draw --stdin
//
// This is a thin C++ CLI wrapper. Performance-sensitive workflows should
// call QuantumCircuit::draw() / draw_to_file() directly from C++ rather
// than spawning this binary in a loop.

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using lindblad::DrawMode;
using lindblad::DrawOptions;
using lindblad::ParamFormat;
using lindblad::QuantumCircuit;

// Algorithm classes live in the nested lindblad::algorithms namespace; pull
// the ones used by the demo factories into scope so the call sites read
// naturally.
using lindblad::algorithms::BernsteinVazirani;
using lindblad::algorithms::DeutschJozsa;
using lindblad::algorithms::Grover;
using lindblad::algorithms::QFT;
using lindblad::algorithms::QPE;
using lindblad::algorithms::Simon;

namespace {

// =========================================================================
// Demo registry : built-in circuits the CLI can render without an input file
// =========================================================================
// Each entry pairs a short name (as typed on the command line) with a factory
// that returns the corresponding QuantumCircuit. Keeping these here rather
// than reusing tests/visualiser_fixtures.hpp avoids a dependency from the
// shipped binary onto test-only sources.

struct DemoEntry {
    const char* name;
    const char* description;
    QuantumCircuit (*build)();
};

QuantumCircuit demo_bell() {
    QuantumCircuit qc(2, 2, "bell");
    qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    return qc;
}

QuantumCircuit demo_ghz() {
    QuantumCircuit qc(3, 0, "ghz");
    qc.h(0).cx(0, 1).cx(1, 2);
    return qc;
}

QuantumCircuit demo_parametric() {
    constexpr double kPi = 3.141592653589793;
    QuantumCircuit qc(3, 0, "parametric");
    qc.rx(kPi / 2.0, 0).ry(kPi / 4.0, 1).crx(kPi / 3.0, 0, 2);
    return qc;
}

QuantumCircuit demo_tallbox() {
    constexpr double kPi = 3.141592653589793;
    QuantumCircuit qc(2, 0, "tallbox");
    qc.rxx(kPi / 4.0, 0, 1).ryy(kPi / 6.0, 0, 1).ecr(0, 1);
    return qc;
}

// Real QFT subcircuit on 4 qubits via lindblad::QFT::build_circuit.
// Demonstrates that the visualiser handles the genuine algorithm output,
// not a hand-crafted approximation. The QFT subcircuit excludes input
// preparation and measurement.
QuantumCircuit demo_qft() {
    return QFT::build_circuit(4);
}

// Inverse QFT subcircuit on 4 qubits. Useful for confirming the visualiser
// renders the swap-and-controlled-phase pattern in reverse order.
QuantumCircuit demo_iqft() {
    return QFT::build_inverse_circuit(4);
}

// =============================================================================
// Algorithm demos : exercise the existing build_circuit() factories
// =============================================================================
// Each algorithm demo constructs a minimal oracle (or unitary) and delegates
// to the algorithm class's static build_circuit method. The CLI renders the
// returned QuantumCircuit, so the visualiser is exercised on output from
// real project code rather than hand-crafted gate sequences.

// Bernstein-Vazirani with hidden string 101 (n=3). The oracle applies a
// CNOT from every input qubit whose secret bit is 1 to a shared ancilla,
// implementing f(x) = s . x mod 2 in the standard phase-kickback form.
QuantumCircuit demo_bv() {
    const std::string secret = "101";
    const int n = static_cast<int>(secret.size());
    QuantumCircuit oracle(n + 1, 0, "bv_oracle");
    for (int i = 0; i < n; ++i) {
        if (secret[i] == '1') {
            oracle.cx(i, n); // ancilla is q_n
        }
    }
    return BernsteinVazirani::build_circuit(oracle, n);
}

// Deutsch-Jozsa on n=3 with a balanced oracle (XOR of every input bit).
// The oracle satisfies f(x) = x_0 XOR x_1 XOR x_2, which is balanced.
QuantumCircuit demo_dj() {
    const int n = 3;
    QuantumCircuit oracle(n + 1, 0, "dj_oracle");
    for (int i = 0; i < n; ++i) {
        oracle.cx(i, n);
    }
    return DeutschJozsa::build_circuit(oracle, n);
}

// Grover search on 3 qubits with 1 iteration, marking the state |011> (q0=1,
// q1=1, q2=0 in little-endian). Standard X-conjugation pattern around CCZ.
QuantumCircuit demo_grover() {
    QuantumCircuit oracle(3, 0, "grover_oracle");
    oracle.x(2);             // flip q2 so CCZ phase-marks |011>
    oracle.ccz(0, 1, 2);
    oracle.x(2);
    // 1 iteration explicitly to keep the rendered output compact; auto
    // selection (-1) would use pi/4 * sqrt(N) which is also 1 for N=8.
    return Grover::build_circuit(oracle, 1);
}

// Phase estimation with 3 evaluation qubits applied to a T-gate target.
// The T gate is P(pi/4), whose eigenvalue on |1> has phase 1/8.
QuantumCircuit demo_qpe() {
    constexpr double kPi = 3.141592653589793;
    QuantumCircuit unitary(1, 0, "T_gate");
    unitary.p(kPi / 4.0, 0);
    return QPE::build_circuit(unitary, 3);
}

// Simon's algorithm with n=2 and a simple oracle. Period not solved by the
// visualiser; the demo only shows the circuit structure.
QuantumCircuit demo_simon() {
    const int n = 2;
    QuantumCircuit oracle(2 * n, 0, "simon_oracle");
    // Copy input to output qubits: CX from each q_i to q_{n+i}.
    for (int i = 0; i < n; ++i) {
        oracle.cx(i, n + i);
    }
    return Simon::build_circuit(oracle, n);
}

// Hardware-efficient VQE ansatz: alternating RY layers separated by a
// linear CNOT entangler. Three qubits, two parameter layers. Hand-built
// because VQE itself is an optimiser rather than a circuit factory.
QuantumCircuit demo_vqe() {
    constexpr double kPi = 3.141592653589793;
    QuantumCircuit qc(3, 0, "vqe_ansatz");
    qc.ry(kPi / 4.0, 0).ry(kPi / 3.0, 1).ry(kPi / 6.0, 2);
    qc.cx(0, 1).cx(1, 2);
    qc.ry(kPi / 5.0, 0).ry(kPi / 7.0, 1).ry(kPi / 8.0, 2);
    return qc;
}

// One-layer QAOA on a 3-node triangle MaxCut graph. Initial superposition
// via H, cost via RZZ on each of the three edges, mixer via RX on each
// qubit. Hand-built because QAOA::build_circuit takes SparsePauliOp inputs
// the CLI does not currently expose on the command line.
QuantumCircuit demo_qaoa() {
    constexpr double kPi = 3.141592653589793;
    QuantumCircuit qc(3, 0, "qaoa_triangle");
    qc.h(0).h(1).h(2);
    // Cost layer: RZZ(gamma) on every edge of the triangle (3 edges)
    qc.rzz(kPi / 4.0, 0, 1)
      .rzz(kPi / 4.0, 1, 2)
      .rzz(kPi / 4.0, 0, 2);
    // Mixer layer: RX(beta) on every qubit
    qc.rx(kPi / 3.0, 0).rx(kPi / 3.0, 1).rx(kPi / 3.0, 2);
    return qc;
}

const std::vector<DemoEntry>& demos() {
    static const std::vector<DemoEntry> table = {
        // Visualiser-feature showcases
        { "bell",       "two-qubit Bell pair with measurements",            &demo_bell       },
        { "ghz",        "three-qubit GHZ preparation",                      &demo_ghz        },
        { "parametric", "RX + RY + CRX showing the pi-snap parameter form", &demo_parametric },
        { "tallbox",    "RXX + RYY + ECR exercising the TallBox role",      &demo_tallbox    },

        // Real algorithm circuits via the library's build_circuit factories
        { "qft",        "four-qubit QFT subcircuit via QFT::build_circuit", &demo_qft        },
        { "iqft",       "four-qubit inverse QFT via QFT::build_inverse",    &demo_iqft       },
        { "bv",         "Bernstein-Vazirani n=3 with hidden string 101",    &demo_bv         },
        { "dj",         "Deutsch-Jozsa n=3 with balanced XOR oracle",       &demo_dj         },
        { "grover",     "Grover 3-qubit, 1 iteration, marks |011>",         &demo_grover     },
        { "qpe",        "phase estimation, 3 eval qubits, T-gate unitary",  &demo_qpe        },
        { "simon",      "Simon n=2 with a minimal period oracle",           &demo_simon      },
        { "vqe",        "hardware-efficient VQE ansatz, two RY layers",     &demo_vqe        },
        { "qaoa",       "one-layer QAOA on a triangle MaxCut graph",        &demo_qaoa       },
    };
    return table;
}

// =========================================================================
// Argument parsing
// =========================================================================
// Hand-rolled flag parser. We intentionally avoid getopt-style state so the
// binary stays portable (no POSIX-only headers) and so the help text in
// usage() stays the canonical reference. Long options only; short flags
// would invite collisions as the surface area grows.

struct CliArgs {
    DrawMode      mode         = DrawMode::ASCII;
    DrawOptions   opts;
    std::string   input_path;       // empty => use stdin or demo
    std::string   demo_name;        // empty => no demo, expect file or stdin
    std::string   output_path;      // empty => stdout
    bool          read_stdin       = false;
    bool          list_demos       = false;
    bool          show_help        = false;
    bool          qasm3            = false; // force qasm3 parser
};

void print_usage(std::ostream& out) {
    out <<
        "lindblad_draw : circuit visualiser CLI (R.1.10.3)\n"
        "\n"
        "USAGE\n"
        "  lindblad_draw [options] <circuit.qasm>\n"
        "  lindblad_draw [options] --stdin\n"
        "  lindblad_draw [options] --demo <name>\n"
        "\n"
        "BACKEND\n"
        "  --mode ascii|svg|latex|html    Output backend (default: ascii)\n"
        "\n"
        "OPTIONS\n"
        "  --output <path>                Write to file instead of stdout\n"
        "                                 (equivalent to ' > <path>' redirection,\n"
        "                                 but uses draw_to_file() internally)\n"
        "  --ascii-safe                   ASCII mode: portable -|+*X palette\n"
        "  --show-clbits                  Render the bundled c-wire\n"
        "  --no-show-params               Strip gate parameters from labels\n"
        "  --param-format pretty|raw      Numeric parameter formatting (default: pretty)\n"
        "  --fold N                       ASCII fold width (0 disables; default: 120)\n"
        "  --cell-px N                    SVG/HTML cell pitch in pixels (default: 48)\n"
        "  --legend                       LaTeX/HTML: emit a gate legend\n"
        "\n"
        "INPUT\n"
        "  --stdin                        Read QASM source from standard input\n"
        "  --demo <name>                  Use a built-in circuit (see --list-demos)\n"
        "  --qasm3                        Parse the input as QASM 3 (default: QASM 2)\n"
        "\n"
        "MISC\n"
        "  --list-demos                   Print the built-in demo catalogue and exit\n"
        "  --help, -h                     Show this message and exit\n"
        "\n"
        "NOTE\n"
        "  This CLI is a convenience front-end. Direct C++ via\n"
        "  QuantumCircuit::draw() / draw_to_file() is the recommended path\n"
        "  for any performance-sensitive or scripted workflow.\n";
}

// Pop the next argument after a flag and assert it is present. Exits the
// program with a usage error on miss; CLI errors do not throw.
const char* expect_next(int argc, char** argv, int& i, const char* flag) {
    if (i + 1 >= argc) {
        std::cerr << "lindblad_draw: missing value for " << flag << "\n";
        print_usage(std::cerr);
        std::exit(2);
    }
    return argv[++i];
}

bool parse_mode(const std::string& s, DrawMode& mode) {
    if (s == "ascii") { mode = DrawMode::ASCII; return true; }
    if (s == "svg")   { mode = DrawMode::SVG;   return true; }
    if (s == "latex") { mode = DrawMode::LATEX; return true; }
    if (s == "html")  { mode = DrawMode::HTML;  return true; }
    return false;
}

bool parse_param_format(const std::string& s, ParamFormat& fmt) {
    if (s == "pretty") { fmt = ParamFormat::Pretty; return true; }
    if (s == "raw")    { fmt = ParamFormat::Raw;    return true; }
    return false;
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            a.show_help = true;
        } else if (arg == "--mode") {
            if (!parse_mode(expect_next(argc, argv, i, "--mode"), a.mode)) {
                std::cerr << "lindblad_draw: unknown mode '" << argv[i]
                          << "' (expected ascii|svg|latex|html)\n";
                std::exit(2);
            }
        } else if (arg == "--output") {
            a.output_path = expect_next(argc, argv, i, "--output");
        } else if (arg == "--ascii-safe") {
            a.opts.ascii_safe = true;
        } else if (arg == "--show-clbits") {
            a.opts.show_clbits = true;
        } else if (arg == "--no-show-params") {
            a.opts.show_params = false;
        } else if (arg == "--param-format") {
            if (!parse_param_format(expect_next(argc, argv, i, "--param-format"),
                                    a.opts.param_format)) {
                std::cerr << "lindblad_draw: unknown param-format '" << argv[i]
                          << "' (expected pretty|raw)\n";
                std::exit(2);
            }
        } else if (arg == "--fold") {
            a.opts.fold_width = std::atoi(expect_next(argc, argv, i, "--fold"));
        } else if (arg == "--cell-px") {
            int v = std::atoi(expect_next(argc, argv, i, "--cell-px"));
            a.opts.cell_width_px  = v;
            a.opts.cell_height_px = v;
        } else if (arg == "--legend") {
            a.opts.include_legend = true;
        } else if (arg == "--stdin") {
            a.read_stdin = true;
        } else if (arg == "--demo") {
            a.demo_name = expect_next(argc, argv, i, "--demo");
        } else if (arg == "--qasm3") {
            a.qasm3 = true;
        } else if (arg == "--list-demos") {
            a.list_demos = true;
        } else if (arg.size() >= 2 && arg[0] == '-' && arg[1] == '-') {
            std::cerr << "lindblad_draw: unknown option '" << arg << "'\n";
            print_usage(std::cerr);
            std::exit(2);
        } else if (a.input_path.empty()) {
            a.input_path = arg;
        } else {
            std::cerr << "lindblad_draw: unexpected positional argument '"
                      << arg << "' (file path already set to '"
                      << a.input_path << "')\n";
            std::exit(2);
        }
    }
    return a;
}

// =========================================================================
// Circuit acquisition : demo, file, or stdin
// =========================================================================
// Each source returns a fully constructed QuantumCircuit or exits with an
// error message. We intentionally do NOT swallow QASM parse exceptions; the
// caller wants the diagnostic that lindblad emits, not a generic "could not
// parse" wrapper.

std::string slurp_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "lindblad_draw: could not open '" << path
                  << "' for reading\n";
        std::exit(1);
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

std::string slurp_stdin() {
    std::ostringstream oss;
    oss << std::cin.rdbuf();
    return oss.str();
}

const DemoEntry* find_demo(const std::string& name) {
    for (const DemoEntry& e : demos()) {
        if (name == e.name) { return &e; }
    }
    return nullptr;
}

void print_demo_list(std::ostream& out) {
    out << "Built-in demo circuits:\n";
    for (const DemoEntry& e : demos()) {
        out << "  " << e.name;
        // Pad the name column to align descriptions.
        std::size_t pad = 14 > std::strlen(e.name)
                           ? 14 - std::strlen(e.name)
                           : 1;
        for (std::size_t i = 0; i < pad; ++i) { out << ' '; }
        out << e.description << '\n';
    }
}

QuantumCircuit acquire_circuit(const CliArgs& a) {
    if (!a.demo_name.empty()) {
        const DemoEntry* e = find_demo(a.demo_name);
        if (!e) {
            std::cerr << "lindblad_draw: unknown demo '" << a.demo_name
                      << "' (use --list-demos)\n";
            std::exit(2);
        }
        return e->build();
    }
    std::string source;
    if (a.read_stdin) {
        source = slurp_stdin();
    } else if (!a.input_path.empty()) {
        source = slurp_file(a.input_path);
    } else {
        std::cerr << "lindblad_draw: no input. Pass a QASM file, --stdin, or --demo.\n";
        print_usage(std::cerr);
        std::exit(2);
    }
    return a.qasm3 ? QuantumCircuit::from_qasm3(source)
                   : QuantumCircuit::from_qasm2(source);
}

} // anonymous namespace

int main(int argc, char** argv) {
    CliArgs a = parse_args(argc, argv);

    if (a.show_help) {
        print_usage(std::cout);
        return 0;
    }
    if (a.list_demos) {
        print_demo_list(std::cout);
        return 0;
    }

    QuantumCircuit qc = acquire_circuit(a);

    if (a.output_path.empty()) {
        std::cout << qc.draw(a.mode, a.opts);
    } else {
        qc.draw_to_file(a.output_path, a.mode, a.opts);
    }
    return 0;
}
