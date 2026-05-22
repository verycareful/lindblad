// =============================================================================
// tests/test_visualiser_file_io.cpp : R.1.10.3 DrawToFileTest
// =============================================================================
// Coverage for QuantumCircuit::draw_to_file(). The method is a thin wrapper
// around draw() + std::ofstream; tests cover the round-trip, every backend,
// and the error path when the destination is unwritable.
//
// CLI tests for `lindblad_draw` are intentionally omitted: argv parsing is
// mechanical and the underlying renderers are already exercised through the
// fixture / palette / catalogue suites. A manual `--help`, `--list-demos`,
// and one `--demo bell` invocation per backend covers the binary post-build.

#include "lindblad/circuit.hpp"
#include "lindblad/visualisation.hpp"

#include "visualiser_fixtures.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace lindblad;
using namespace lindblad::vfx;

namespace fs = std::filesystem;

namespace {

// Build a unique temporary file path under the system temp directory. We
// combine a high-resolution timestamp with a per-call counter so parallel
// test runs and rapid successive calls within one process never collide.
// Avoids POSIX-only ::getpid() so the test stays portable across the WSL
// build environments lindblad supports.
std::string make_tmp_path(const std::string& suffix) {
    static int counter = 0;
    ++counter;
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path dir = fs::temp_directory_path();
    std::ostringstream name;
    name << "lindblad_draw_" << stamp << "_" << counter << "." << suffix;
    return (dir / name.str()).string();
}

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

void cleanup(const std::string& path) {
    std::error_code ec;
    fs::remove(path, ec);
}

} // namespace

// =============================================================================
// Round-trip : draw_to_file output equals direct draw() output
// =============================================================================

TEST(DrawToFileTest, AsciiRoundTripMatchesDirectDraw) {
    QuantumCircuit qc = bell_with_measures();
    std::string path = make_tmp_path("txt");

    qc.draw_to_file(path, DrawMode::ASCII, {});
    const std::string from_file   = slurp(path);
    const std::string from_direct = qc.draw(DrawMode::ASCII, {});
    cleanup(path);

    EXPECT_EQ(from_file, from_direct);
}

TEST(DrawToFileTest, SvgRoundTripMatchesDirectDraw) {
    QuantumCircuit qc = ghz_3q();
    std::string path = make_tmp_path("svg");

    qc.draw_to_file(path, DrawMode::SVG, {});
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file, qc.draw(DrawMode::SVG, {}));
}

TEST(DrawToFileTest, LatexRoundTripMatchesDirectDraw) {
    QuantumCircuit qc = parametric_rotations();
    std::string path = make_tmp_path("tex");

    qc.draw_to_file(path, DrawMode::LATEX, {});
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file, qc.draw(DrawMode::LATEX, {}));
}

TEST(DrawToFileTest, HtmlRoundTripMatchesDirectDraw) {
    QuantumCircuit qc = bell_with_measures();
    std::string path = make_tmp_path("html");

    qc.draw_to_file(path, DrawMode::HTML, {});
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file, qc.draw(DrawMode::HTML, {}));
}

// =============================================================================
// Options round-trip : caller options reach the renderer
// =============================================================================

TEST(DrawToFileTest, AsciiSafeOptionReachesRenderer) {
    QuantumCircuit qc = bell_with_measures();
    std::string path = make_tmp_path("txt");

    DrawOptions opts;
    opts.ascii_safe = true;

    qc.draw_to_file(path, DrawMode::ASCII, opts);
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file, qc.draw(DrawMode::ASCII, opts));
}

TEST(DrawToFileTest, ShowClbitsOptionReachesRenderer) {
    QuantumCircuit qc = bell_with_measures();
    std::string path = make_tmp_path("txt");

    DrawOptions opts;
    opts.show_clbits = true;

    qc.draw_to_file(path, DrawMode::ASCII, opts);
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file, qc.draw(DrawMode::ASCII, opts));
}

TEST(DrawToFileTest, DefaultModeIsAscii) {
    QuantumCircuit qc = bell_with_measures();
    std::string path = make_tmp_path("txt");

    // No mode argument: defaults to DrawMode::ASCII per the header.
    qc.draw_to_file(path);
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file, qc.draw(DrawMode::ASCII, {}));
}

// =============================================================================
// Output content : file is non-empty and well-formed per backend
// =============================================================================

TEST(DrawToFileTest, SvgFileStartsWithXmlProlog) {
    QuantumCircuit qc = bell_with_measures();
    std::string path = make_tmp_path("svg");

    qc.draw_to_file(path, DrawMode::SVG, {});
    const std::string from_file = slurp(path);
    cleanup(path);

    ASSERT_FALSE(from_file.empty());
    EXPECT_EQ(from_file.find("<?xml"), 0u);
}

TEST(DrawToFileTest, HtmlFileStartsWithDoctype) {
    QuantumCircuit qc = bell_with_measures();
    std::string path = make_tmp_path("html");

    qc.draw_to_file(path, DrawMode::HTML, {});
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file.find("<!DOCTYPE html>"), 0u);
}

TEST(DrawToFileTest, LatexFileStartsWithBeginQuantikz) {
    QuantumCircuit qc = bell_unmeasured();
    std::string path = make_tmp_path("tex");

    qc.draw_to_file(path, DrawMode::LATEX, {});
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file.find("\\begin{quantikz}"), 0u);
}

// =============================================================================
// Error path : unwritable path throws std::runtime_error
// =============================================================================

TEST(DrawToFileTest, ThrowsOnUnwritableDestination) {
    QuantumCircuit qc = bell_with_measures();
    // A path under a non-existent directory: std::ofstream cannot create
    // intermediate directories, so the open fails and draw_to_file must
    // surface std::runtime_error instead of silently dropping the output.
    const std::string bad_path = "/this/directory/does/not/exist/bell.svg";

    EXPECT_THROW(qc.draw_to_file(bad_path, DrawMode::SVG, {}),
                 std::runtime_error);
}

TEST(DrawToFileTest, ErrorMessageIncludesPath) {
    QuantumCircuit qc = bell_with_measures();
    const std::string bad_path = "/no/such/directory/out.txt";
    try {
        qc.draw_to_file(bad_path);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find(bad_path), std::string::npos)
            << "error message should include the offending path; got: " << msg;
    }
}

// =============================================================================
// Overwrite semantics : draw_to_file replaces an existing file
// =============================================================================

TEST(DrawToFileTest, OverwritesExistingFile) {
    QuantumCircuit qc = bell_with_measures();
    std::string path = make_tmp_path("txt");

    // Seed the destination with garbage.
    {
        std::ofstream seed(path, std::ios::binary);
        seed << "STALE_CONTENT_FROM_PREVIOUS_RUN";
    }

    qc.draw_to_file(path, DrawMode::ASCII, {});
    const std::string from_file = slurp(path);
    cleanup(path);

    EXPECT_EQ(from_file.find("STALE_CONTENT_FROM_PREVIOUS_RUN"),
              std::string::npos)
        << "draw_to_file must truncate the destination, not append";
    EXPECT_EQ(from_file, qc.draw(DrawMode::ASCII, {}));
}
