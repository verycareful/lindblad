// 1.1.23.1 test wave - the tolerance vocabulary, enforced rather than assumed.
//
// Lindblad has exactly one kind of tolerance: absolute. Every check in the
// Class C framework compares a residual against an `atol` and nothing is
// scaled by the magnitude of what it is judging. That is a deliberate choice
// and 1.1.23.0 leaned on it: DEFAULT_PHYSICAL_ATOL is one number that has to
// mean one thing at every register size, which is only true while the
// comparison stays absolute. A relative tolerance introduced anywhere would
// make the same constant mean different things in different places, and it
// would do so quietly, because a relative check passes more often rather than
// failing loudly.
//
// So this walks the tree for the vocabulary instead of trusting that nobody
// adds it. It also pins the inventory of hand-typed tolerance defaults, which
// is the debt DEFAULT_PHYSICAL_ATOL was introduced to retire.
//
// One trap is encoded here because it cost a false conclusion when this was
// first checked by hand: a raw substring search for "rtol" matches kClusterTol
// and kOrderTol, both of which are ordinary ABSOLUTE tolerances whose spelling
// happens to end in "rTol". Matching therefore splits identifiers into words on
// camelCase and underscore boundaries and compares whole words, so ...erTol
// never reads as rtol.

#include <gtest/gtest.h>

#include "lindblad/validation.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

// -----------------------------------------------------------------------------
// Masking
// -----------------------------------------------------------------------------
// Comments and string literals are blanked before matching. A comment is
// exactly where the rule itself gets discussed (this file is the obvious
// example), and a string may legitimately carry any word at all, so neither is
// code and neither should be able to fail the scan.

std::string mask_non_code(const std::string& src) {
    std::string out = src;
    const std::size_t n = out.size();
    std::size_t i = 0;

    auto blank_to = [&out](std::size_t from, std::size_t to) {
        for (std::size_t k = from; k < to && k < out.size(); ++k)
            if (out[k] != '\n') out[k] = ' ';
    };

    while (i < n) {
        // Raw string: R"delim( ... )delim"
        if (out[i] == 'R' && i + 1 < n && out[i + 1] == '"') {
            const std::size_t open = out.find('(', i + 2);
            if (open == std::string::npos) break;
            const std::string delim = out.substr(i + 2, open - (i + 2));
            const std::string close = ")" + delim + "\"";
            const std::size_t end = out.find(close, open);
            if (end == std::string::npos) { blank_to(i, n); break; }
            blank_to(i, end + close.size());
            i = end + close.size();
            continue;
        }
        if (out[i] == '/' && i + 1 < n && out[i + 1] == '/') {
            std::size_t end = out.find('\n', i);
            if (end == std::string::npos) end = n;
            blank_to(i, end);
            i = end;
            continue;
        }
        if (out[i] == '/' && i + 1 < n && out[i + 1] == '*') {
            const std::size_t end = out.find("*/", i + 2);
            const std::size_t stop = (end == std::string::npos) ? n : end + 2;
            blank_to(i, stop);
            i = stop;
            continue;
        }
        if (out[i] == '"' || out[i] == '\'') {
            const char quote = out[i];
            std::size_t j = i + 1;
            while (j < n && out[j] != quote) {
                if (out[j] == '\\') ++j;
                ++j;
            }
            blank_to(i, (j < n ? j + 1 : n));
            i = (j < n ? j + 1 : n);
            continue;
        }
        ++i;
    }
    return out;
}

// -----------------------------------------------------------------------------
// Identifier splitting
// -----------------------------------------------------------------------------
// kOrderTol -> {k, order, tol};  rtol -> {rtol};  rel_tol -> {rel, tol}.
// Splitting on the case boundary is what keeps "...erTol" from reading as the
// token "rtol", which a substring search cannot distinguish.

std::vector<std::string> split_identifier(const std::string& id) {
    std::vector<std::string> words;
    std::string cur;
    for (std::size_t i = 0; i < id.size(); ++i) {
        const char c = id[i];
        if (c == '_') {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
            continue;
        }
        const bool upper = std::isupper(static_cast<unsigned char>(c)) != 0;
        const bool prev_lower =
            i > 0 && std::islower(static_cast<unsigned char>(id[i - 1])) != 0;
        const bool prev_digit =
            i > 0 && std::isdigit(static_cast<unsigned char>(id[i - 1])) != 0;
        if (upper && (prev_lower || prev_digit) && !cur.empty()) {
            words.push_back(cur);
            cur.clear();
        }
        cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

// A single word that names a relative tolerance outright.
bool is_relative_word(const std::string& w) {
    return w == "rtol" || w == "reltol" || w == "etol";
}

// Two adjacent words that together name one: rel_tol, relativeTolerance, rTol.
// A bare "r" counts only in this adjacent position, which is what separates
// rTol from kOrderTol: the latter splits to {k, order, tol} and its word before
// "tol" is "order", not "r".
bool is_relative_pair(const std::string& a, const std::string& b) {
    const bool rel = (a == "r" || a == "rel" || a == "relative");
    const bool tol = (b == "tol" || b == "tolerance" || b == "tolerances");
    return rel && tol;
}

struct Hit {
    std::string file;
    int line;
    std::string identifier;
};

void scan_file(const std::filesystem::path& path, const std::string& rel,
               std::vector<Hit>& hits) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string masked = mask_non_code(ss.str());

    int line = 1;
    std::size_t i = 0;
    while (i < masked.size()) {
        const char c = masked[i];
        if (c == '\n') { ++line; ++i; continue; }
        const bool starts =
            std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
        if (!starts) { ++i; continue; }

        std::size_t j = i;
        while (j < masked.size() &&
               (std::isalnum(static_cast<unsigned char>(masked[j])) != 0 ||
                masked[j] == '_'))
            ++j;

        const std::string id = masked.substr(i, j - i);
        const std::vector<std::string> words = split_identifier(id);
        bool bad = false;
        for (std::size_t w = 0; w < words.size() && !bad; ++w) {
            if (is_relative_word(words[w])) bad = true;
            if (w + 1 < words.size() && is_relative_pair(words[w], words[w + 1]))
                bad = true;
        }
        if (bad) hits.push_back({rel, line, id});
        i = j;
    }
}

std::vector<std::string> scan_dirs() {
    return {"src", "include", "apps", "benchmarks", "tests"};
}

// The suite that DEFINES the rule has to be able to name what it bans: the test
// names below and the match table above both spell rtol on purpose. Nothing
// else in the tree has that excuse.
bool is_this_suite(const std::string& filename) {
    return filename.rfind("test_v11231_tolerance_rules", 0) == 0;
}

}  // namespace

// =============================================================================
// No relative tolerance anywhere in Lindblad
// =============================================================================

// atol is the project's only tolerance. A relative one would make
// DEFAULT_PHYSICAL_ATOL mean a different thing at every magnitude it is applied
// to, which is precisely what one shared constant exists to prevent.
TEST(V11231ToleranceRules, NoRtolInLindblad) {
    const std::filesystem::path root(LINDBLAD_SOURCE_ROOT);

    std::vector<Hit> hits;
    int files_scanned = 0;

    for (const std::string& d : scan_dirs()) {
        const std::filesystem::path base = root / d;
        if (!std::filesystem::exists(base)) continue;
        for (const auto& e : std::filesystem::recursive_directory_iterator(base)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext != ".cpp" && ext != ".hpp" && ext != ".h") continue;
            if (is_this_suite(e.path().filename().string())) continue;
            ++files_scanned;
            scan_file(e.path(),
                      std::filesystem::relative(e.path(), root).generic_string(),
                      hits);
        }
    }

    ASSERT_GT(files_scanned, 0)
        << "the scan found no source files, so a pass here would mean nothing. "
           "LINDBLAD_SOURCE_ROOT is " << LINDBLAD_SOURCE_ROOT;

    std::ostringstream report;
    for (const Hit& h : hits)
        report << "\n  " << h.file << ":" << h.line << "  " << h.identifier;

    EXPECT_TRUE(hits.empty())
        << "Lindblad uses absolute tolerances only. A relative tolerance was "
           "introduced at:" << report.str()
        << "\nIf a relative comparison is genuinely wanted, that is a design "
           "change to make deliberately, not one to land under a name the "
           "framework does not model.";
}

// The splitter is what makes the scan above trustworthy, so it is tested
// directly rather than only through a passing sweep. kOrderTol and kClusterTol
// are real absolute tolerances in the tree and must not be flagged.
TEST(V11231ToleranceRules, TheSplitterDistinguishesRTolFromErTol) {
    const std::vector<std::string> ordinary = {"kOrderTol", "kClusterTol",
                                               "atol", "worst_tol", "tolerance"};
    for (const std::string& id : ordinary) {
        SCOPED_TRACE(id);
        const std::vector<std::string> w = split_identifier(id);
        bool bad = false;
        for (std::size_t i = 0; i < w.size(); ++i) {
            if (is_relative_word(w[i])) bad = true;
            if (i + 1 < w.size() && is_relative_pair(w[i], w[i + 1])) bad = true;
        }
        EXPECT_FALSE(bad) << "an absolute tolerance must not read as a relative one";
    }

    const std::vector<std::string> relative = {"rtol", "rTol", "kRelTol",
                                               "rel_tol", "relativeTolerance",
                                               "relative_tolerance"};
    for (const std::string& id : relative) {
        SCOPED_TRACE(id);
        const std::vector<std::string> w = split_identifier(id);
        bool bad = false;
        for (std::size_t i = 0; i < w.size(); ++i) {
            if (is_relative_word(w[i])) bad = true;
            if (i + 1 < w.size() && is_relative_pair(w[i], w[i + 1])) bad = true;
        }
        EXPECT_TRUE(bad) << "a relative tolerance must be caught";
    }
}

// Masking is the other half: this very file names rtol repeatedly in prose and
// in the table above, and must still pass its own scan.
TEST(V11231ToleranceRules, MaskingBlanksCommentsAndStrings) {
    const std::string src =
        "int a; // rtol lives in a comment\n"
        "const char* s = \"rtol\";\n"
        "int rtol_in_code;\n";
    const std::string masked = mask_non_code(src);

    EXPECT_EQ(masked.find("rtol lives"), std::string::npos)
        << "a comment must not be scanned";
    EXPECT_NE(masked.find("rtol_in_code"), std::string::npos)
        << "real code must survive masking or the scan proves nothing";
    EXPECT_EQ(std::count(masked.begin(), masked.end(), '\n'),
              std::count(src.begin(), src.end(), '\n'))
        << "masking preserves line numbering so a report points at the right line";
}

// =============================================================================
// The hand-typed tolerance inventory
// =============================================================================

// DEFAULT_PHYSICAL_ATOL exists so that a predicate over a property and the
// policy judging that property cannot disagree. Three predicates still declare
// their default by hand and one is a deliberate exception, so the inventory is
// pinned: a new hand-typed default fails here, and so does removing one without
// updating this list, which is what keeps the debt visible rather than
// forgotten.
TEST(V11231ToleranceRules, HandTypedAtolDefaultsAreExactlyTheKnownInventory) {
    struct Known {
        const char* file;
        const char* symbol;
        const char* why;
    };
    // Three carry the framework value by hand and should derive from
    // DEFAULT_PHYSICAL_ATOL. simplify is not a validity tolerance at all: it
    // drops terms below the value, so it is a pruning knob, and tightening it
    // prunes less rather than checking harder.
    const Known kInventory[] = {
        {"include/lindblad/noise.hpp", "KrausChannel::is_valid", "should derive"},
        {"include/lindblad/operators.hpp", "Operator::is_unitary", "should derive"},
        {"include/lindblad/operators.hpp", "Operator::is_hermitian", "should derive"},
        {"include/lindblad/operators.hpp", "SparsePauliOp::simplify",
         "deliberate: a pruning knob, not a validity tolerance"},
    };

    const std::filesystem::path root(LINDBLAD_SOURCE_ROOT);
    std::vector<std::string> found;

    // Headers only. A default tolerance in a public signature is part of the
    // API contract, so it has to come from the shared constant. The `atol`
    // constants inside the transpiler are synthesis thresholds deciding when an
    // angle is small enough to drop, which is not a validity tolerance and
    // carries its own value for its own reasons.
    for (const std::string& d : {std::string("include")}) {
        const std::filesystem::path base = root / d;
        if (!std::filesystem::exists(base)) continue;
        for (const auto& e : std::filesystem::recursive_directory_iterator(base)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext != ".cpp" && ext != ".hpp" && ext != ".h") continue;

            std::ifstream in(e.path(), std::ios::binary);
            std::ostringstream ss;
            ss << in.rdbuf();
            const std::string masked = mask_non_code(ss.str());

            // A parameter named atol defaulting to a numeric literal. Anything
            // else spelling 1e-12 is an algorithmic threshold that merely
            // shares the value, so the parameter name is the discriminator.
            std::size_t pos = 0;
            const std::string needle = "atol = ";
            while ((pos = masked.find(needle, pos)) != std::string::npos) {
                const std::size_t v = pos + needle.size();
                if (v < masked.size() &&
                    (std::isdigit(static_cast<unsigned char>(masked[v])) != 0 ||
                     masked[v] == '.')) {
                    const int line =
                        1 + static_cast<int>(std::count(masked.begin(),
                                                        masked.begin() + pos, '\n'));
                    found.push_back(
                        std::filesystem::relative(e.path(), root).generic_string() +
                        ":" + std::to_string(line));
                }
                pos = v;
            }
        }
    }

    std::ostringstream report;
    for (const std::string& f : found) report << "\n  " << f;

    EXPECT_EQ(found.size(), sizeof(kInventory) / sizeof(kInventory[0]))
        << "the hand-typed atol defaults are no longer the known set. Found:"
        << report.str()
        << "\nA NEW one should derive from DEFAULT_PHYSICAL_ATOL instead. "
           "If one was retired, remove it from the inventory in this test.";
}

// The constant is the single source, so the struct default has to come from it
// rather than repeat its value.
TEST(V11231ToleranceRules, ValidationOptionsDefaultsToTheSharedConstant) {
    ValidationOptions v;
    EXPECT_EQ(v.atol, DEFAULT_PHYSICAL_ATOL)
        << "a caller writing nothing gets the framework tolerance";
    EXPECT_EQ(v.policy, Validation::Throw)
        << "and the framework's default policy";

    ValidationOptions named{Validation::Warn};
    EXPECT_EQ(named.atol, DEFAULT_PHYSICAL_ATOL)
        << "naming a policy must not silently change the tolerance";
}
