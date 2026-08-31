// test_v11241_seam_rules.cpp - the decomposition seam, enforced rather than
// verified by hand.
//
// Two structural properties were established and then checked once, manually:
// no header under include/ names an Eigen type, and no translation unit outside
// src/eigen_backend.cpp instantiates an Eigen decomposition over the dynamic
// complex matrix type. The second was confirmed by running nm over the object
// files. Neither is expressible as a unit test on a value, and neither survives
// on its own: the first breaks the moment somebody adds a convenient include,
// and the second breaks the moment somebody writes a JacobiSVD where they need
// one. Both breakages are invisible at the call site.
//
// The second property is the one that carries weight, and the reason is worth
// stating where the rule lives. Eigen is header-only, so a decomposition
// instantiated in a translation unit becomes a weak symbol in its own COMDAT
// group, and the linker keeps exactly ONE definition per mangled name for the
// whole binary. Compile flags are not part of that name and are gone by link
// time, so the surviving copy may come from a unit built under a different
// floating-point model than the one calling it. That matters because
// -ffast-math tells the compiler no infinity or NaN can occur, which folds away
// the finiteness guard JacobiSVD writes at its own entry. A per-file
// -fno-fast-math cannot repair this: it governs which variant a unit EMITS,
// never which one survives. One emitter leaves nothing to merge.
//
// The rule is an allow-list rather than a search for a forbidden spelling,
// because the failure mode is a NEW instantiation somewhere nobody thought to
// look. Anything not named below is a violation wherever it appears, including
// in files that do not exist yet.
//
// Two entries are on that list deliberately. optimize_1q.cpp decomposes
// Matrix2cd, Matrix4d and MatrixXd, which are different C++ types from the
// backend's dynamic complex matrix and therefore mangle differently and cannot
// merge with it. They carry no floating-point-model risk from the seam, so they
// are permitted in place rather than routed through it.

#include <gtest/gtest.h>

#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The decompositions that matter. Every one is a class template whose
// instantiation emits the algorithm itself.
const char* const kDecompositions[] = {"JacobiSVD", "BDCSVD",
                                       "SelfAdjointEigenSolver"};

// The single translation unit permitted to instantiate anything.
const char* const kBackend = "src/eigen_backend.cpp";

struct Allowed {
    const char* file;  // repo-relative, forward slashes
    const char* type;  // the exact template argument
};

// Fixed-size and real-valued instantiations, which are distinct types from the
// backend's dynamic complex ones and cannot merge with them.
const Allowed kAllowed[] = {
    {"src/transpiler/optimisation/optimize_1q.cpp", "Eigen::Matrix2cd"},
    {"src/transpiler/optimisation/optimize_1q.cpp", "Eigen::Matrix4d"},
    {"src/transpiler/optimisation/optimize_1q.cpp", "Eigen::MatrixXd"},
};

struct Hit {
    std::string file;
    int line = 0;
    std::string text;
};

// Blanks out comments, string literals and character literals, keeping the byte
// count and every newline so line numbers survive. A rule about code must not
// fire on prose describing the rule, and this file and its neighbours discuss
// these type names at length.
std::string mask(const std::string& src) {
    std::string out = src;
    enum State { Code, Line, Block, Str, Chr } st = Code;
    for (std::size_t i = 0; i < src.size(); ++i) {
        const char c = src[i];
        const char n = (i + 1 < src.size()) ? src[i + 1] : '\0';
        switch (st) {
            case Code:
                if (c == '/' && n == '/') { st = Line; out[i] = out[i + 1] = ' '; ++i; }
                else if (c == '/' && n == '*') { st = Block; out[i] = out[i + 1] = ' '; ++i; }
                else if (c == '"') { st = Str; out[i] = ' '; }
                else if (c == '\'') { st = Chr; out[i] = ' '; }
                break;
            case Line:
                if (c == '\n') st = Code; else out[i] = ' ';
                break;
            case Block:
                if (c == '*' && n == '/') { st = Code; out[i] = out[i + 1] = ' '; ++i; }
                else if (c != '\n') out[i] = ' ';
                break;
            case Str:
                if (c == '\\') { out[i] = ' '; if (i + 1 < src.size()) out[++i] = ' '; }
                else if (c == '"') { st = Code; out[i] = ' '; }
                else if (c != '\n') out[i] = ' ';
                break;
            case Chr:
                if (c == '\\') { out[i] = ' '; if (i + 1 < src.size()) out[++i] = ' '; }
                else if (c == '\'') { st = Code; out[i] = ' '; }
                else if (c != '\n') out[i] = ' ';
                break;
        }
    }
    return out;
}

std::string read(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int line_of(const std::string& s, std::size_t pos) {
    int line = 1;
    for (std::size_t i = 0; i < pos && i < s.size(); ++i)
        if (s[i] == '\n') ++line;
    return line;
}

// The template argument of `Name<...>`, taken to the matching angle bracket so
// a nested template argument does not truncate it.
std::string template_argument(const std::string& s, std::size_t open) {
    int depth = 0;
    std::string arg;
    for (std::size_t i = open; i < s.size(); ++i) {
        if (s[i] == '<') { ++depth; if (depth == 1) continue; }
        else if (s[i] == '>') { if (--depth == 0) break; }
        else if (depth == 0) break;
        arg += s[i];
    }
    // Collapse whitespace so a wrapped argument compares as written.
    std::string tight;
    for (char c : arg)
        if (!std::isspace(static_cast<unsigned char>(c))) tight += c;
    return tight;
}

bool is_allowed(const std::string& file, const std::string& type) {
    for (const auto& a : kAllowed)
        if (file == a.file && type == a.type) return true;
    return false;
}

std::string relative_slashes(const std::filesystem::path& p,
                             const std::filesystem::path& root) {
    std::string s = std::filesystem::relative(p, root).string();
    for (char& c : s)
        if (c == '\\') c = '/';
    return s;
}

}  // namespace

TEST(V11241SeamRules, NoHeaderUnderIncludeNamesAnEigenType) {
    // A public header that names an Eigen type obliges every caller holding the
    // object to have Eigen available, and drags the dependency back across the
    // interface the seam exists to clear.
    const std::filesystem::path root(LINDBLAD_SOURCE_ROOT);
    const std::filesystem::path base = root / "include";
    ASSERT_TRUE(std::filesystem::exists(base))
        << "include/ not found; LINDBLAD_SOURCE_ROOT is " << LINDBLAD_SOURCE_ROOT;

    std::vector<Hit> hits;
    int scanned = 0;

    for (const auto& e : std::filesystem::recursive_directory_iterator(base)) {
        if (!e.is_regular_file()) continue;
        const std::string ext = e.path().extension().string();
        if (ext != ".hpp" && ext != ".h") continue;
        ++scanned;

        const std::string masked = mask(read(e.path()));
        for (std::size_t pos = masked.find("Eigen"); pos != std::string::npos;
             pos = masked.find("Eigen", pos + 1)) {
            hits.push_back({relative_slashes(e.path(), root),
                            line_of(masked, pos), masked.substr(pos, 48)});
        }
    }

    ASSERT_GT(scanned, 0) << "no headers scanned, so a pass here means nothing";

    std::ostringstream report;
    for (const Hit& h : hits) report << "\n  " << h.file << ":" << h.line;
    EXPECT_TRUE(hits.empty())
        << hits.size() << " header reference(s) to Eigen across " << scanned
        << " headers. Hold detail::DenseMatrix or a raw buffer at the interface "
           "and map it with a backend at the site that needs the arithmetic."
        << report.str();
}

TEST(V11241SeamRules, OnlyTheBackendInstantiatesADecomposition) {
    const std::filesystem::path root(LINDBLAD_SOURCE_ROOT);
    const char* dirs[] = {"src", "include", "tests", "apps", "benchmarks"};

    std::vector<Hit> hits;
    int scanned = 0;
    int backend_sites = 0;

    for (const char* d : dirs) {
        const std::filesystem::path base = root / d;
        if (!std::filesystem::exists(base)) continue;
        for (const auto& e : std::filesystem::recursive_directory_iterator(base)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext != ".cpp" && ext != ".hpp" && ext != ".h") continue;
            ++scanned;

            const std::string rel = relative_slashes(e.path(), root);
            const std::string masked = mask(read(e.path()));

            for (const char* name : kDecompositions) {
                const std::string needle = std::string(name) + "<";
                for (std::size_t pos = masked.find(needle);
                     pos != std::string::npos;
                     pos = masked.find(needle, pos + 1)) {
                    const std::size_t open = pos + needle.size() - 1;
                    const std::string arg = template_argument(masked, open);
                    if (rel == kBackend) { ++backend_sites; continue; }
                    if (is_allowed(rel, arg)) continue;
                    hits.push_back({rel, line_of(masked, pos),
                                    std::string(name) + "<" + arg + ">"});
                }
            }
        }
    }

    ASSERT_GT(scanned, 0) << "no source files scanned, so a pass means nothing. "
                             "LINDBLAD_SOURCE_ROOT is "
                          << LINDBLAD_SOURCE_ROOT;
    ASSERT_GT(backend_sites, 0)
        << "the backend instantiates nothing, so either " << kBackend
        << " moved or this scan is looking in the wrong place; either way an "
           "empty result below would be meaningless";

    std::ostringstream report;
    for (const Hit& h : hits) report << "\n  " << h.file << ":" << h.line << "  "
                                     << h.text;
    EXPECT_TRUE(hits.empty())
        << hits.size() << " decomposition instantiation(s) outside "
        << kBackend << ", across " << scanned
        << " files. Eigen's instantiations have vague linkage, so a second "
           "emitter lets the linker hand one translation unit's flags to "
           "another's code. Call detail::svd_thin or detail::eigh instead."
        << report.str();
}

TEST(V11241SeamRules, TheAllowListStillDescribesRealCode) {
    // An allow-list that has drifted from the tree silently stops enforcing
    // anything: an entry naming a file or a type that no longer exists would
    // excuse nothing and nobody would notice. Each entry must still match.
    const std::filesystem::path root(LINDBLAD_SOURCE_ROOT);

    for (const auto& a : kAllowed) {
        const std::filesystem::path p = root / a.file;
        ASSERT_TRUE(std::filesystem::exists(p))
            << "allow-list names " << a.file << ", which does not exist";

        const std::string masked = mask(read(p));
        bool found = false;
        for (const char* name : kDecompositions) {
            const std::string needle = std::string(name) + "<";
            for (std::size_t pos = masked.find(needle);
                 pos != std::string::npos && !found;
                 pos = masked.find(needle, pos + 1)) {
                if (template_argument(masked, pos + needle.size() - 1) == a.type)
                    found = true;
            }
        }
        EXPECT_TRUE(found) << "allow-list permits " << a.type << " in " << a.file
                           << ", but nothing there instantiates it any more. "
                              "Remove the entry so the rule keeps its teeth.";
    }
}
