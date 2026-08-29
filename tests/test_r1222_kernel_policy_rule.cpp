// R.1.22.2 - every internal kernel call passes a validation policy, enforced by
// scanning the tree rather than by remembering.
//
// The defect this guards (#86) survived two releases because the parameter that
// carries the policy is DEFAULTED. Three backends called a kernel without it,
// the default Throw reappeared underneath a caller who had explicitly opted out,
// and the compiler could point at none of those sites, because omitting a
// defaulted argument is not an error. Finding one required already suspecting it.
//
// So the question is asked mechanically instead. For each kernel entry point
// that takes a ValidationOptions, this counts the arguments at every call site
// under src/ and reports any call that stopped short of the policy.
//
// Counting arguments rather than searching for the word "Validation" is what
// makes it robust. A correct site may pass `inst.validation`, a parameter named
// `physical`, or a braced literal, and all three are the call being right; only
// the argument's ABSENCE is the defect.
//
// SCOPE, and why it stops where it does: the qubit layer. The qudit primitives
// take a ValidationOptions too, and most of their internal call sites omit it,
// but that is a different problem with a different fix. QuditGateOp carries no
// ValidationOptions field at all (#88), so those callers have no policy to
// forward and no edit inside them could produce one. They come into scope when
// #88 does.

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// A kernel entry point, and how many arguments a call must carry to have
// reached the ValidationOptions, which is always the last parameter.
struct EntryPoint {
    const char* name;
    int arity;
    // Skip a call written without a receiver. Set only where a local lambda in
    // some translation unit shares the name: clifford_sim.cpp defines its own
    // `apply_gate` lambda, and the real entry point is always reached through
    // a DensityMatrix object.
    bool member_only;
};

const EntryPoint kEntryPoints[] = {
    {"apply_unitary",           4, false},  // gates:: (sv, targets, matrix, v)
    {"apply_gate",              3, true },  // DensityMatrix (U, qubits, v)
    {"apply_channel_superop",   3, false},  // DensityMatrix (S, qubits, v)
    {"apply_kraus",             3, false},  // DensityMatrix (ops, qubits, v)
    {"apply_single_qubit_gate", 3, false},  // MPSState (U, qubit, v)
    {"apply_two_qubit_gate",    4, false},  // MPSState (U, q1, q2, v)
    {"apply_instruction",       3, false},  // StatevectorSimulator (sv, inst, v)
};

// Blank out comments, string literals, character literals and raw strings,
// preserving newlines so line numbers survive. Without this a call named in a
// comment or inside an error message would be scanned as code.
std::string mask_non_code(const std::string& src) {
    const size_t n = src.size();
    std::string out(n, ' ');
    for (size_t i = 0; i < n;) {
        if (src[i] == '\n') { out[i] = '\n'; ++i; continue; }

        // Raw string: R"delim( ... )delim"
        if (src[i] == 'R' && i + 1 < n && src[i + 1] == '"') {
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
        if (src[i] == '"' || src[i] == '\'') {
            const char quote = src[i];
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
        if (src[i] == '/' && i + 1 < n && src[i + 1] == '/') {
            while (i < n && src[i] != '\n') ++i;
            continue;
        }
        if (src[i] == '/' && i + 1 < n && src[i + 1] == '*') {
            size_t end = src.find("*/", i + 2);
            if (end == std::string::npos) end = n; else end += 2;
            for (size_t k = i; k < end && k < n; ++k)
                if (src[k] == '\n') out[k] = '\n';
            i = end;
            continue;
        }
        out[i] = src[i];
        ++i;
    }
    return out;
}

bool ident_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// What sits between the start of the line and the name. A declaration or
// definition is spelled with a return type there (`void DensityMatrix::` or
// `    void apply_gate`); a call has nothing, a receiver, or a namespace.
enum class Context { Call, MemberCall, Declaration };

Context classify(const std::string& masked, size_t name_start) {
    size_t bol = masked.rfind('\n', name_start);
    bol = (bol == std::string::npos) ? 0 : bol + 1;
    std::string prefix = masked.substr(bol, name_start - bol);

    size_t last = prefix.find_last_not_of(" \t");
    if (last == std::string::npos) return Context::Call;   // bare, this-> implied
    prefix.erase(last + 1);

    if (prefix.size() >= 1 && prefix.back() == '.') return Context::MemberCall;
    if (prefix.size() >= 2 && prefix.compare(prefix.size() - 2, 2, "->") == 0)
        return Context::MemberCall;

    if (prefix.size() >= 2 && prefix.compare(prefix.size() - 2, 2, "::") == 0) {
        // `gates::apply_unitary(` is a call; `void DensityMatrix::apply_gate(`
        // is a definition, and the return type in front is what distinguishes
        // them.
        const std::string qualifier = prefix.substr(0, prefix.size() - 2);
        return (qualifier.find(' ') == std::string::npos) ? Context::MemberCall
                                                          : Context::Declaration;
    }
    return Context::Declaration;  // a return type, so a declaration
}

// Top-level arguments of the call whose '(' is at `open`. Returns 0 for `()`.
int count_arguments(const std::string& masked, size_t open) {
    int depth = 1;
    int commas = 0;
    bool any = false;
    size_t i = open + 1;
    for (; i < masked.size() && depth > 0; ++i) {
        const char c = masked[i];
        if (c == '(' || c == '[' || c == '{') ++depth;
        else if (c == ')' || c == ']' || c == '}') --depth;
        else if (c == ',' && depth == 1) ++commas;
        if (depth == 1 && !std::isspace(static_cast<unsigned char>(c)) && c != ',')
            any = true;
    }
    return any ? commas + 1 : 0;
}

struct Hit {
    std::string file;
    int line;
    std::string name;
    int found;
    int wanted;
};

void scan_file(const std::filesystem::path& path, const std::string& rel,
               std::vector<Hit>& hits, int& calls_seen) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string masked = mask_non_code(ss.str());

    for (const EntryPoint& ep : kEntryPoints) {
        const std::string name(ep.name);
        size_t pos = 0;
        while ((pos = masked.find(name, pos)) != std::string::npos) {
            const size_t after = pos + name.size();
            const bool whole =
                (pos == 0 || !ident_char(masked[pos - 1])) &&
                (after >= masked.size() || !ident_char(masked[after]));
            size_t open = after;
            while (open < masked.size() &&
                   std::isspace(static_cast<unsigned char>(masked[open]))) ++open;
            if (!whole || open >= masked.size() || masked[open] != '(') {
                pos = after;
                continue;
            }

            const Context ctx = classify(masked, pos);
            if (ctx == Context::Declaration) { pos = after; continue; }
            if (ep.member_only && ctx != Context::MemberCall) { pos = after; continue; }

            ++calls_seen;
            const int found = count_arguments(masked, open);
            if (found < ep.arity) {
                const int line = 1 + static_cast<int>(
                    std::count(masked.begin(), masked.begin() + pos, '\n'));
                hits.push_back({rel, line, name, found, ep.arity});
            }
            pos = after;
        }
    }
}

}  // namespace

// A failure names the file, the line and the entry point, so acting on it does
// not require re-deriving the search. The fix is always the same: decide the
// policy at that site and pass it. Below a run() pre-flight that is
// {Validation::Ignore}; in code applying a matrix it was handed, it is the
// caller's own inst.validation.
TEST(R1222KernelPolicyRule, NoInternalKernelCallOmitsItsValidationPolicy) {
    const std::filesystem::path root(LINDBLAD_SOURCE_ROOT);
    const std::filesystem::path base = root / "src";

    ASSERT_TRUE(std::filesystem::exists(base))
        << "LINDBLAD_SOURCE_ROOT is " << LINDBLAD_SOURCE_ROOT;

    std::vector<Hit> hits;
    int files_scanned = 0;
    int calls_seen = 0;

    for (const auto& e : std::filesystem::recursive_directory_iterator(base)) {
        if (!e.is_regular_file()) continue;
        const std::string ext = e.path().extension().string();
        if (ext != ".cpp" && ext != ".hpp" && ext != ".h") continue;
        ++files_scanned;
        scan_file(e.path(),
                  "src/" + e.path().filename().string(), hits, calls_seen);
    }

    ASSERT_GT(files_scanned, 0) << "the scan found no source files under src/, "
                                   "so a pass here would mean nothing";
    ASSERT_GT(calls_seen, 0)
        << "the scan found no calls to any kernel entry point, which means the "
           "matching is broken rather than that the tree is clean";

    std::ostringstream report;
    for (const Hit& h : hits)
        report << "\n  " << h.file << ":" << h.line << "  " << h.name
               << " called with " << h.found << " argument(s), needs "
               << h.wanted << " to reach the ValidationOptions";

    EXPECT_TRUE(hits.empty())
        << hits.size() << " internal kernel call(s) omit the validation policy, "
           "so each takes the Throw default and overrides whatever the caller "
           "asked for:" << report.str();
}
