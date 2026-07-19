#include "lindblad/statevector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <numeric>

namespace lindblad {

// =============================================================================
// Construction / Destruction
// =============================================================================

namespace {
// Validate the qubit count BEFORE it reaches `1ULL << n` in the initializer
// list. Computing the shift with a negative or oversized n is undefined
// behaviour (shift-width overflow), so the guard must run first — a body check
// after the init list is too late. Called from the n_qubits member initializer,
// which precedes `dim` in declaration order, so a bad n throws before the shift.
inline int validated_n_qubits(int n) {
    if (n < 1 || n > 30)
        throw std::invalid_argument(
            "Statevector: n_qubits must be in [1, 30], got " + std::to_string(n));
    return n;
}
} // namespace

Statevector::Statevector(int n_qubits)
    : n_qubits(validated_n_qubits(n_qubits))
    , dim(1ULL << this->n_qubits)   // this->n_qubits is already validated
    , real_parts(nullptr)
    , imag_parts(nullptr)
{
    real_parts = aligned_alloc_doubles(dim);
    imag_parts = aligned_alloc_doubles(dim);

    initialize();
}

Statevector::~Statevector() {
    aligned_free(real_parts);
    aligned_free(imag_parts);
}

// =============================================================================
// Move semantics
// =============================================================================

Statevector::Statevector(Statevector&& other) noexcept
    : n_qubits(other.n_qubits)
    , dim(other.dim)
    , real_parts(other.real_parts)
    , imag_parts(other.imag_parts)
{
    other.real_parts = nullptr;
    other.imag_parts = nullptr;
    other.n_qubits = 0;
    other.dim = 0;
}

Statevector& Statevector::operator=(Statevector&& other) noexcept {
    if (this != &other) {
        aligned_free(real_parts);
        aligned_free(imag_parts);

        n_qubits = other.n_qubits;
        dim = other.dim;
        real_parts = other.real_parts;
        imag_parts = other.imag_parts;

        other.real_parts = nullptr;
        other.imag_parts = nullptr;
        other.n_qubits = 0;
        other.dim = 0;
    }
    return *this;
}

// =============================================================================
// Initialisation
// =============================================================================

void Statevector::initialize() {
    // Parallel zero-fill causes first-touch to distribute pages across NUMA
    // nodes on multi-socket hardware; free on UMA.
    #pragma omp parallel for schedule(static) if(dim >= (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        real_parts[i] = 0.0;
        imag_parts[i] = 0.0;
    }
    real_parts[0] = 1.0;  // |0...0⟩
}

void Statevector::initialize_basis(size_t k) {
    if (k >= dim) {
        throw std::out_of_range("Basis state index out of range");
    }
    std::memset(real_parts, 0, dim * sizeof(double));
    std::memset(imag_parts, 0, dim * sizeof(double));
    real_parts[k] = 1.0;
}

void Statevector::set_amplitudes(const double* real, const double* imag, size_t count) {
    if (count != dim) {
        throw std::invalid_argument("Amplitude count must match dimension");
    }
    std::memcpy(real_parts, real, dim * sizeof(double));
    std::memcpy(imag_parts, imag, dim * sizeof(double));
}

void Statevector::set_amplitudes(const std::vector<Complex128>& amplitudes) {
    if (amplitudes.size() != dim) {
        throw std::invalid_argument("Amplitude count must match dimension");
    }
    for (size_t i = 0; i < dim; ++i) {
        real_parts[i] = amplitudes[i].real;
        imag_parts[i] = amplitudes[i].imag;
    }
}

// =============================================================================
// Amplitude access
// =============================================================================

Complex128 Statevector::amplitude(size_t index) const {
    if (index >= dim) {
        throw std::out_of_range("Amplitude index out of range");
    }
    return {real_parts[index], imag_parts[index]};
}

std::vector<Complex128> Statevector::amplitudes() const {
    std::vector<Complex128> result(dim);
    for (size_t i = 0; i < dim; ++i) {
        result[i] = {real_parts[i], imag_parts[i]};
    }
    return result;
}

// =============================================================================
// Probabilities
// =============================================================================

double Statevector::probability(size_t index) const {
    if (index >= dim) {
        throw std::out_of_range("Probability index out of range");
    }
    return real_parts[index] * real_parts[index] +
           imag_parts[index] * imag_parts[index];
}

std::vector<double> Statevector::probabilities() const {
    std::vector<double> probs(dim);

    #pragma omp parallel for schedule(static) if(dim >= (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        probs[i] = real_parts[i] * real_parts[i] +
                   imag_parts[i] * imag_parts[i];
    }
    return probs;
}

// =============================================================================
// Norm
// =============================================================================

double Statevector::norm_sq() const {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static) if(dim >= (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        sum += real_parts[i] * real_parts[i] +
               imag_parts[i] * imag_parts[i];
    }
    return sum;
}

double Statevector::norm() const {
    return std::sqrt(norm_sq());
}

void Statevector::normalize() {
    double n = norm();
    if (n < 1e-15) {
        throw std::runtime_error("Cannot normalize zero state");
    }
    double inv_n = 1.0 / n;

    #pragma omp parallel for schedule(static) if(dim >= (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        real_parts[i] *= inv_n;
        imag_parts[i] *= inv_n;
    }
}

// =============================================================================
// Inner product
// =============================================================================

Complex128 Statevector::inner_product(const Statevector& other) const {
    if (dim != other.dim) {
        throw std::invalid_argument("Dimension mismatch for inner product");
    }

    double re = 0.0;
    double im = 0.0;

    #pragma omp parallel for reduction(+:re,im) schedule(static) if(dim >= (1<<20))
    for (size_t i = 0; i < dim; ++i) {
        // ⟨this|other⟩ = sum_i conj(this_i) * other_i
        re += real_parts[i] * other.real_parts[i] + imag_parts[i] * other.imag_parts[i];
        im += real_parts[i] * other.imag_parts[i] - imag_parts[i] * other.real_parts[i];
    }
    return {re, im};
}

// =============================================================================
// Measurement sampling
// =============================================================================

std::string Statevector::measure_once(uint64_t seed) const {
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);

    double cumulative = 0.0;
    size_t outcome = dim - 1;  // fallback

    for (size_t i = 0; i < dim; ++i) {
        cumulative += real_parts[i] * real_parts[i] +
                      imag_parts[i] * imag_parts[i];
        if (r < cumulative) {
            outcome = i;
            break;
        }
    }

    // Convert to bitstring (MSB first)
    std::string bits(n_qubits, '0');
    for (int b = 0; b < n_qubits; ++b) {
        if ((outcome >> (n_qubits - 1 - b)) & 1) {
            bits[b] = '1';
        }
    }
    return bits;
}

std::unordered_map<std::string, int> Statevector::sample_counts(
    int shots, uint64_t seed
) const {
    std::unordered_map<std::string, int> counts;
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Precompute cumulative probabilities
    std::vector<double> cum_probs(dim);
    cum_probs[0] = real_parts[0] * real_parts[0] + imag_parts[0] * imag_parts[0];
    for (size_t i = 1; i < dim; ++i) {
        cum_probs[i] = cum_probs[i - 1] +
                        real_parts[i] * real_parts[i] +
                        imag_parts[i] * imag_parts[i];
    }

    for (int s = 0; s < shots; ++s) {
        double r = dist(rng);
        // Binary search for the outcome
        auto it = std::lower_bound(cum_probs.begin(), cum_probs.end(), r);
        size_t outcome = static_cast<size_t>(std::distance(cum_probs.begin(), it));
        if (outcome >= dim) outcome = dim - 1;

        // Convert to bitstring
        std::string bits(n_qubits, '0');
        for (int b = 0; b < n_qubits; ++b) {
            if ((outcome >> (n_qubits - 1 - b)) & 1) {
                bits[b] = '1';
            }
        }
        counts[bits]++;
    }

    return counts;
}

// =============================================================================
// Clone
// =============================================================================

Statevector Statevector::clone() const {
    Statevector copy(n_qubits);
    std::memcpy(copy.real_parts, real_parts, dim * sizeof(double));
    std::memcpy(copy.imag_parts, imag_parts, dim * sizeof(double));
    return copy;
}

// =============================================================================
// String representation
// =============================================================================

std::string Statevector::to_string(int precision) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision);
    oss << "Statevector(" << n_qubits << " qubits, dim=" << dim << "):\n";

    size_t max_show = std::min(dim, static_cast<size_t>(32));

    for (size_t i = 0; i < max_show; ++i) {
        double r = real_parts[i];
        double im = imag_parts[i];
        double prob = r * r + im * im;

        if (prob > 1e-12) {
            // Format as binary string
            std::string bits(n_qubits, '0');
            for (int b = 0; b < n_qubits; ++b) {
                if ((i >> (n_qubits - 1 - b)) & 1) bits[b] = '1';
            }
            oss << "  |" << bits << "⟩: "
                << r << (im >= 0 ? "+" : "") << im << "i"
                << "  (p=" << prob << ")\n";
        }
    }

    if (dim > max_show) {
        oss << "  ... (" << dim - max_show << " more entries)\n";
    }

    return oss.str();
}

} // namespace lindblad

