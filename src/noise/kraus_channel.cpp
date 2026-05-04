#include "lindblad/noise.hpp"

#include <cmath>
#include <numeric>

namespace lindblad {

// =============================================================================
// KrausChannel validation
// =============================================================================

bool KrausChannel::is_valid(double atol) const {
    // Check sum_k K_k† K_k = I
    size_t dim = 1ULL << n_qubits;
    std::vector<Complex128> sum(dim * dim, Complex128(0.0, 0.0));

    for (const auto& K : operators) {
        // sum += K† * K
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                Complex128 val(0.0, 0.0);
                for (size_t m = 0; m < dim; ++m) {
                    val += K[m * dim + i].conj() * K[m * dim + j];
                }
                sum[i * dim + j] += val;
            }
        }
    }

    // Check sum ≈ I
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            Complex128 diff = sum[i * dim + j] - Complex128(expected, 0.0);
            if (diff.norm_sq() > atol * atol) return false;
        }
    }
    return true;
}

double KrausChannel::trace_preserving_error() const {
    size_t dim = 1ULL << n_qubits;
    std::vector<Complex128> sum(dim * dim, Complex128(0.0, 0.0));

    for (const auto& K : operators) {
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                Complex128 val(0.0, 0.0);
                for (size_t m = 0; m < dim; ++m) {
                    val += K[m * dim + i].conj() * K[m * dim + j];
                }
                sum[i * dim + j] += val;
            }
        }
    }

    double error = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            Complex128 diff = sum[i * dim + j] - Complex128(expected, 0.0);
            error += diff.norm_sq();
        }
    }
    return std::sqrt(error);
}

} // namespace lindblad
