#pragma once

// =============================================================================
// tests/golden_helpers.hpp : golden-file loading for the visualiser test suite
// =============================================================================
// Tests compare renderer output against committed golden files under
// tests/golden/visualisation/<backend>/<fixture>.txt. The path is resolved
// from a compile definition LINDBLAD_TEST_GOLDEN_DIR set by tests/CMakeLists.txt
// so the test binary can find the files regardless of the caller's cwd.
//
// First-run workflow:
//   1. Test invokes EXPECT_EQ(rendered, load_golden("ascii/bell.txt"));
//   2. The first run fails with a clear "golden file not found" message.
//   3. The maintainer runs the regen tool (lindblad_visualiser_regen) to
//      generate every golden file from the current implementation.
//   4. The maintainer reviews each generated file against the design spec
//      before committing.

#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lindblad::vfx {

// Resolve a golden file path relative to LINDBLAD_TEST_GOLDEN_DIR.
// rel_path uses forward slashes (e.g. "ascii/bell.txt"); the helper joins
// it onto the configured base directory. The base directory is set at
// build time so the same test binary works regardless of where it is run.
inline std::string golden_path(const std::string& rel_path) {
#ifndef LINDBLAD_TEST_GOLDEN_DIR
    // Fall back to a relative path when the compile definition is missing.
    // Test execution then depends on cwd; CMake should always set this so
    // the fallback only matters for ad-hoc builds.
    return std::string("tests/golden/visualisation/") + rel_path;
#else
    return std::string(LINDBLAD_TEST_GOLDEN_DIR) + "/visualisation/" + rel_path;
#endif
}

// Load a golden file as a UTF-8 string. Throws std::runtime_error with a
// descriptive message when the file is missing so first-run failures are
// easy to diagnose ("golden file not found: ascii/bell.txt").
//
// Reads in binary mode to preserve exact byte sequences; the renderers
// always emit LF line endings, so a binary read avoids CRLF translation on
// Windows hosts running the WSL toolchain.
inline std::string load_golden(const std::string& rel_path) {
    const std::string full = golden_path(rel_path);
    std::ifstream in(full, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            "golden file not found: " + full +
            " (run lindblad_visualiser_regen to generate it)");
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

} // namespace lindblad::vfx
