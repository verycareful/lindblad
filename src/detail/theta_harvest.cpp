// =============================================================================
// theta_harvest - capture real two-site blocks on their way into the SVD
// =============================================================================
// Contract, and why one representative per shape rather than every block, are
// in include/lindblad/detail/theta_harvest.hpp. This file exists in the build
// only when LINDBLAD_MPS_THETA_HARVEST is set.

#include "lindblad/detail/theta_harvest.hpp"

#ifdef LINDBLAD_MPS_THETA_HARVEST

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>

namespace lindblad {
namespace detail {

namespace {

// Directory the harvest is written to, relative to the working directory
// unless the build overrode it. A fixed location rather than an environment
// variable: the feature is already a build-time choice, and a second switch
// read at run time would make an instrumented binary behave differently from
// one run to the next.
#ifndef LINDBLAD_MPS_THETA_HARVEST_DIR
#define LINDBLAD_MPS_THETA_HARVEST_DIR "theta_harvest"
#endif

// Exact binary64 rendering. %a round-trips through strtod, which is the whole
// reason the corpus is hexfloat rather than decimal: a benchmark fed values
// that differ from the ones the simulation produced is measuring a different
// matrix.
std::string hex_of(double v) {
    char buf[64];
    const int n = std::snprintf(buf, sizeof(buf), "%a", v);
    if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(buf)) return std::string();
    return std::string(buf, static_cast<std::size_t>(n));
}

// Shape -> how many blocks of that shape were offered. Ordered so the
// histogram comes out the same way whatever order shapes were first seen in,
// which keeps two runs of the same circuit diffable.
using ShapeCounts = std::map<std::pair<int, int>, long long>;

// The counts and the destructor that writes them are ONE object, deliberately.
// Split across two function-local statics, the map is constructed second and so
// destroyed first, and the writer's destructor then walks a destroyed container.
// Owning the map removes the ordering question rather than answering it.
struct Harvest {
    ShapeCounts counts;

    // Runs as the process ends, so a run that never split writes nothing at
    // all: the object is not constructed until the first offer.
    ~Harvest() {
        std::error_code ec;
        std::filesystem::create_directories(LINDBLAD_MPS_THETA_HARVEST_DIR, ec);
        std::ofstream out(
            std::string(LINDBLAD_MPS_THETA_HARVEST_DIR) + "/shapes.txt",
            std::ios::binary);
        if (!out) return;
        out << "# rows cols count\n";
        long long total = 0;
        for (const auto& [shape, n] : counts) {
            out << shape.first << ' ' << shape.second << ' ' << n << '\n';
            total += n;
        }
        out << "# " << counts.size() << " distinct shapes, " << total
            << " splits\n";
    }
};

Harvest& harvest() {
    static Harvest h;
    return h;
}

void write_block(const std::vector<Complex128>& data, int rows, int cols) {
    std::error_code ec;
    std::filesystem::create_directories(LINDBLAD_MPS_THETA_HARVEST_DIR, ec);

    const std::string name =
        "theta_" + std::to_string(rows) + "x" + std::to_string(cols);
    std::ofstream out(
        std::string(LINDBLAD_MPS_THETA_HARVEST_DIR) + "/" + name + ".hexfloat",
        std::ios::binary);
    if (!out) return;

    out << "autonne-hexfloat 1 matrix " << name << ' ' << rows << ' ' << cols
        << " rowmajor\n";
    for (const auto& z : data) {
        out << hex_of(z.real) << ' ' << hex_of(z.imag) << '\n';
    }
}

}  // namespace

void theta_harvest_offer(const std::vector<Complex128>& data, int rows, int cols) {
    auto& n = harvest().counts[std::make_pair(rows, cols)];
    // Written on the FIRST block of a shape rather than the last, so a run that
    // is interrupted still leaves a complete file for every shape it reached.
    if (n == 0) write_block(data, rows, cols);
    ++n;
}

} // namespace detail
} // namespace lindblad

#endif // LINDBLAD_MPS_THETA_HARVEST
