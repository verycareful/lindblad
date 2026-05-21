// =============================================================================
// tests/visualiser_regen.cpp : golden-file regeneration tool for R.1.10.1
// =============================================================================
// Standalone binary that walks every fixture x backend x variant combination
// and writes the current renderer output to the matching golden file under
// tests/golden/visualisation/<backend>/<fixture>.txt.
//
// Build target: lindblad_visualiser_regen (not wired into ctest).
// Usage      : ./build-clang/tests/lindblad_visualiser_regen
//              Writes files relative to LINDBLAD_TEST_GOLDEN_DIR (set by
//              tests/CMakeLists.txt) so the binary works from any cwd.
//
// The regen tool is the maintainer's escape hatch: it never runs during
// normal testing. When tests fail because the rendering format changed,
// regenerate, review each diff against the design spec, and commit.

#include "lindblad/circuit.hpp"

#include "visualiser_fixtures.hpp"

#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#ifndef LINDBLAD_TEST_GOLDEN_DIR
#define LINDBLAD_TEST_GOLDEN_DIR "tests/golden"
#endif

namespace fs = std::filesystem;

namespace {

struct Variant {
    std::string suffix; // appended to filename before .txt (empty = no suffix)
    lindblad::DrawOptions opts;
};

struct Fixture {
    std::string name;                                   // bell, ghz, etc.
    lindblad::QuantumCircuit (*build)();                // factory function
    std::vector<Variant> variants;                      // per-fixture variants
};

// Write `body` to <golden_dir>/visualisation/<backend>/<name><suffix>.txt.
// Creates intermediate directories on demand. Reports each file written so
// the maintainer can scan the diff before committing.
void write_golden(const std::string& backend,
                  const std::string& name,
                  const std::string& suffix,
                  const std::string& body) {
    fs::path base = fs::path(LINDBLAD_TEST_GOLDEN_DIR) / "visualisation" / backend;
    fs::create_directories(base);
    fs::path path = base / (name + suffix + ".txt");

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "ERROR: could not open " << path.string() << " for writing\n";
        return;
    }
    out << body;
    std::cout << "wrote " << path.string() << " (" << body.size() << " bytes)\n";
}

// Apply each backend renderer to `qc` with `opts` and write all four golden
// files. Variants control the per-backend filename suffix so e.g. ASCII can
// have <name>.txt for the Unicode default and <name>.ascii_safe.txt for the
// portable palette variant.
void emit_variant(const lindblad::QuantumCircuit& qc,
                  const std::string& name,
                  const Variant& v) {
    using lindblad::DrawMode;
    write_golden("ascii", name, v.suffix, qc.draw(DrawMode::ASCII, v.opts));
    write_golden("svg",   name, v.suffix, qc.draw(DrawMode::SVG,   v.opts));
    write_golden("latex", name, v.suffix, qc.draw(DrawMode::LATEX, v.opts));
    write_golden("html",  name, v.suffix, qc.draw(DrawMode::HTML,  v.opts));
}

} // anonymous namespace

int main() {
    using namespace lindblad;

    // Default DrawOptions instance used by the no-variant fixtures.
    DrawOptions defaults;

    // Variants applied to fixtures that benefit from multiple snapshots.
    DrawOptions ascii_safe;
    ascii_safe.ascii_safe = true;

    DrawOptions show_clbits;
    show_clbits.show_clbits = true;

    DrawOptions raw_params;
    raw_params.param_format = ParamFormat::Raw;

    DrawOptions no_params;
    no_params.show_params = false;

    DrawOptions with_legend;
    with_legend.include_legend = true;

    // Fixture catalogue. Adding a new fixture is a one-line entry here;
    // the regen tool picks it up automatically.
    std::vector<Fixture> fixtures = {
        { "bell",                vfx::bell_with_measures,
            { { "", defaults }, { ".ascii_safe", ascii_safe },
              { ".show_clbits", show_clbits } } },
        { "bell_unmeasured",     vfx::bell_unmeasured,
            { { "", defaults } } },
        { "ghz",                 vfx::ghz_3q,
            { { "", defaults } } },
        { "parametric",          vfx::parametric_rotations,
            { { "", defaults }, { ".raw",       raw_params },
              { ".no_params",  no_params  } } },
        { "barrier_measure",     vfx::barrier_and_measure,
            { { "", defaults }, { ".show_clbits", show_clbits } } },
        { "feedforward",         vfx::conditional_feedforward,
            { { "", defaults }, { ".show_clbits", show_clbits } } },
        { "noncontig_unitary",   vfx::non_contiguous_unitary,
            { { "", defaults } } },
        { "tallbox",             vfx::tallbox_demo,
            { { "", defaults } } },
        { "all_1q_unparam",      vfx::all_single_qubit_unparam,
            { { "", defaults } } },
        { "all_1q_param",        vfx::all_single_qubit_param,
            { { "", defaults }, { ".raw", raw_params } } },
        { "all_2q_unparam",      vfx::all_two_qubit_unparam,
            { { "", defaults } } },
        { "all_3q",              vfx::all_three_qubit,
            { { "", defaults } } },
        { "empty",               vfx::empty_2q,
            { { "", defaults } } },
        { "reset_h",             vfx::reset_and_h,
            { { "", defaults } } },
        { "bell_legend",         vfx::bell_with_measures,
            { { "", with_legend } } },
    };

    int count = 0;
    for (const Fixture& f : fixtures) {
        QuantumCircuit qc = f.build();
        for (const Variant& v : f.variants) {
            emit_variant(qc, f.name, v);
            count += 4;
        }
    }

    std::cout << "\nwrote " << count << " golden files across "
              << fixtures.size() << " fixtures\n";
    return 0;
}
