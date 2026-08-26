// Strict-FP definitions of the R1211 reference residuals.
//
// This translation unit is compiled with -fno-fast-math (see tests/CMakeLists.txt).
// Everything in it is evaluated under IEEE semantics, which is what makes the
// functions here usable as oracles. The rationale, and what goes wrong without
// it, is in r1211_reference_residuals.hpp.

#include "r1211_reference_residuals.hpp"

#include <cmath>
#include <limits>

namespace r1211ref {

double reference_unitarity_deviation(const std::vector<lindblad::Complex128>& U,
                                     std::size_t rows) {
    using lindblad::Complex128;

    double worst_sq = 0.0;
    bool saw_nan = false;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < rows; ++j) {
            Complex128 acc(0.0, 0.0);
            for (std::size_t m = 0; m < rows; ++m)
                acc = acc + U[m * rows + i].conj() * U[m * rows + j];
            const Complex128 diff = acc - Complex128(i == j ? 1.0 : 0.0, 0.0);
            const double d = diff.norm_sq();
            if (std::isnan(d)) saw_nan = true;
            else if (d > worst_sq) worst_sq = d;
        }
    }
    if (saw_nan) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(worst_sq);
}

} // namespace r1211ref
