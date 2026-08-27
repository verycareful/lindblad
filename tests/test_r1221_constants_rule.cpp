// R.1.22.1 test wave - the constants rule, enforced rather than swept.
//
// CLAUDE.md and docs require that a numeric constant with a canonical source is
// never hand-typed: math constants come from constants.hpp, type properties from
// std::numeric_limits. Two violations survived until R.1.22.0, in
// test_r1122_fill_frontends.cpp and test_r1122_fill_transpiler.cpp, and neither
// was findable except by someone manually grepping for a value they already
// suspected. A sweep only finds what it spells, and nobody spells everything.
//
// So this walks the tree instead. It masks out comments, string literals, raw
// strings and character literals, then looks for a floating-point literal with
// enough significant digits to be a deliberate transcription and compares it
// against every constant in constants.hpp.
//
// Two deliberate design choices:
//
//   - The digit threshold, not the value alone, is what separates a
//     transcription from a coincidence. `0.5` and `1.5` are ordinary numbers
//     that happen to sit near nothing; `1.5707963267948966` is somebody typing
//     out PI_2. Eight significant digits is well past the point where a human
//     would write a number by hand for any other reason.
//
//   - String literals are masked rather than allow-listed by file. QASM source
//     embedded in a test legitimately contains `rz(1.57079632679)`, because the
//     literal is the INPUT being parsed rather than an expected value. Masking
//     handles that case wherever it appears, including in files that do not
//     exist yet.
//
// A failure prints file, line and the offending literal, so acting on it does
// not require re-deriving the search.

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

struct Known {
    const char* name;
    double value;
};

// Every constant constants.hpp exports. A transcription of any of them is the
// thing being hunted, so the table is the header's own contents rather than a
// selection from it.
const Known kKnown[] = {
    {"PI", PI},
    {"TWO_PI", TWO_PI},
    {"PI_2", PI_2},
    {"PI_4", PI_4},
    {"INV_PI", INV_PI},
    {"TWO_INV_PI", TWO_INV_PI},
    {"TWO_INV_SQRTPI", TWO_INV_SQRTPI},
    {"SQRT2", SQRT2},
    {"SQRT3", SQRT3},
    {"INV_SQRT2", INV_SQRT2},
    {"E", E},
    {"LN2", LN2},
    {"LN10", LN10},
    {"LOG2E", LOG2E},
    {"LOG10E", LOG10E},
};

// The constants suite itself must contain these values: proving the header is
// correctly rounded means comparing it against something. Nothing else in the
// tree has that excuse.
bool is_the_constants_suite(const std::string& filename) {
    return filename.rfind("test_r1201_constants", 0) == 0;
}

// Replaces every non-code byte with a space, preserving newlines so line
// numbers survive. Handles line comments, block comments, string literals with
// escapes, raw strings with arbitrary delimiters, and character literals.
std::string mask_non_code(const std::string& src) {
    std::string out(src.size(), ' ');
    const size_t n = src.size();
    size_t i = 0;
    auto keep = [&](size_t at) { out[at] = src[at]; };

    while (i < n) {
        const char c = src[i];
        if (c == '\n') { out[i] = '\n'; ++i; continue; }

        // Raw string: R"delim( ... )delim"
        if (c == 'R' && i + 1 < n && src[i + 1] == '"') {
            size_t p = i + 2;
            std::string delim;
            while (p < n && src[p] != '(') delim += src[p++];
            const std::string closer = ")" + delim + "\"";
            size_t end = src.find(closer, p);
            if (end == std::string::npos) end = n; else end += closer.size();
            for (size_t k = i; k < end && k < n; ++k)
                if (src[k] == '\n') out[k] = '\n';
            i = end;
            continue;
        }
        if (c == '"' || c == '\'') {
            const char quote = c;
            size_t p = i + 1;
            while (p < n && src[p] != quote) {
                if (src[p] == '\\') ++p;
                ++p;
            }
            for (size_t k = i; k <= p && k < n; ++k)
                if (src[k] == '\n') out[k] = '\n';
            i = (p < n) ? p + 1 : n;
            continue;
        }
        if (c == '/' && i + 1 < n && src[i + 1] == '/') {
            while (i < n && src[i] != '\n') ++i;
            continue;
        }
        if (c == '/' && i + 1 < n && src[i + 1] == '*') {
            size_t end = src.find("*/", i + 2);
            if (end == std::string::npos) end = n; else end += 2;
            for (size_t k = i; k < end && k < n; ++k)
                if (src[k] == '\n') out[k] = '\n';
            i = end;
            continue;
        }
        keep(i);
        ++i;
    }
    return out;
}

int significant_digits(const std::string& lit) {
    int digits = 0;
    bool seen_nonzero = false;
    for (char c : lit) {
        if (c == 'e' || c == 'E') break;
        if (!std::isdigit(static_cast<unsigned char>(c))) continue;
        if (c != '0') seen_nonzero = true;
        if (seen_nonzero) ++digits;
    }
    return digits;
}

struct Hit {
    std::string file;
    int line;
    std::string literal;
    std::string constant;
};

void scan_file(const std::filesystem::path& path, const std::string& rel,
               std::vector<Hit>& hits) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string masked = mask_non_code(ss.str());

    int line = 1;
    for (size_t i = 0; i < masked.size(); ++i) {
        if (masked[i] == '\n') { ++line; continue; }
        if (!std::isdigit(static_cast<unsigned char>(masked[i]))) continue;
        // Not the tail of an identifier or a longer number.
        if (i > 0) {
            const char p = masked[i - 1];
            if (std::isalnum(static_cast<unsigned char>(p)) || p == '_' ||
                p == '.' || p == 'x' || p == 'X')
                continue;
        }
        size_t j = i;
        while (j < masked.size() &&
               std::isdigit(static_cast<unsigned char>(masked[j]))) ++j;
        if (j >= masked.size() || masked[j] != '.') { i = j - 1; continue; }
        ++j;
        while (j < masked.size() &&
               std::isdigit(static_cast<unsigned char>(masked[j]))) ++j;

        const std::string lit = masked.substr(i, j - i);
        if (significant_digits(lit) >= 8) {
            const double v = std::strtod(lit.c_str(), nullptr);
            for (const Known& k : kKnown) {
                const double scale = (std::abs(k.value) > 1.0) ? std::abs(k.value) : 1.0;
                if (std::abs(v - k.value) <= 1e-6 * scale) {
                    hits.push_back({rel, line, lit, k.name});
                    break;
                }
            }
        }
        i = j - 1;
    }
}

}  // namespace

// The rule is forward-looking and tree-wide, so the scan is too. If this fails,
// the fix is to use the named constant; if a site genuinely needs the digits,
// it belongs in the constants suite where proving the digits is the point.
TEST(R1221ConstantsRule, NoSourceFileTranscribesAConstantFromConstantsHpp) {
    const std::filesystem::path root(LINDBLAD_SOURCE_ROOT);
    const char* dirs[] = {"src", "include", "tests", "apps", "benchmarks"};

    std::vector<Hit> hits;
    int files_scanned = 0;

    for (const char* d : dirs) {
        const std::filesystem::path base = root / d;
        if (!std::filesystem::exists(base)) continue;
        for (const auto& e : std::filesystem::recursive_directory_iterator(base)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext != ".cpp" && ext != ".hpp" && ext != ".h") continue;
            const std::string name = e.path().filename().string();
            if (is_the_constants_suite(name)) continue;
            ++files_scanned;
            scan_file(e.path(),
                      std::string(d) + "/" + e.path().filename().string(), hits);
        }
    }

    ASSERT_GT(files_scanned, 0)
        << "the scan found no source files, so a pass here would mean nothing. "
           "LINDBLAD_SOURCE_ROOT is " << LINDBLAD_SOURCE_ROOT;

    std::ostringstream report;
    for (const Hit& h : hits)
        report << "\n  " << h.file << ":" << h.line << "  " << h.literal
               << "  is " << h.constant << " from constants.hpp";

    EXPECT_TRUE(hits.empty())
        << hits.size() << " transcribed constant(s) across " << files_scanned
        << " files. Use the named constant instead of the digits."
        << report.str();
}
