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

// A four-qubit inverse QFT-style demo built from gates the visualiser
// already covers; not a faithful QFT, but a rich-looking circuit for
// quick rendering checks.
QuantumCircuit demo_qft() {
    constexpr double kPi = 3.141592653589793;
    QuantumCircuit qc(4, 0, "qft_like");
    qc.h(0).cp(kPi / 2.0, 1, 0).cp(kPi / 4.0, 2, 0).cp(kPi / 8.0, 3, 0)
      .h(1).cp(kPi / 2.0, 2, 1).cp(kPi / 4.0, 3, 1)
      .h(2).cp(kPi / 2.0, 3, 2)
      .h(3)
      .swap(0, 3).swap(1, 2);
    return qc;
}

const std::vector<DemoEntry>& demos() {
    static const std::vector<DemoEntry> table = {
        { "bell",       "two-qubit Bell pair with measurements",            &demo_bell },
        { "ghz",        "three-qubit GHZ preparation",                      &demo_ghz },
        { "parametric", "RX + RY + CRX showing the pi-snap parameter form", &demo_parametric },
        { "tallbox",    "RXX + RYY + ECR exercising the TallBox role",      &demo_tallbox },
        { "qft",        "four-qubit QFT-style circuit (rich rendering)",    &demo_qft },
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
