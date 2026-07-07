#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <new>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace lindblad {

// =============================================================================
// Cross-platform macros for compiler intrinsics
// =============================================================================

// Population count for 64-bit integers — used in Pauli expectation values
#if defined(_MSC_VER)
#  define LINDBLAD_POPCOUNT64(x) static_cast<int>(__popcnt64(x))
#else
#  define LINDBLAD_POPCOUNT64(x) __builtin_popcountll(x)
#endif

// Note on OpenMP pragmas:
// MSVC uses OpenMP 2.0 (legacy) and silently ignores unrecognised pragma clauses
// like the aligned() clause on #pragma omp simd (OpenMP 4.0+).
// We use conditional compilation to remove the pragma on MSVC; GCC/Clang use it.

// =============================================================================
// is_finite_strict — IEEE-754 finiteness test immune to -ffast-math
// =============================================================================
// The project compiles with -ffast-math (-ffinite-math-only), under which the
// compiler may constant-fold std::isfinite / std::isnan to "always finite",
// silently disabling NaN guards (clang warns via -Wnan-infinity-disabled; GCC
// does not warn at all). This bit-pattern test (finite iff the exponent field
// is not all-ones) survives any flag set and costs one integer compare; use it
// for every guard whose job is to DETECT non-finite values.
inline bool is_finite_strict(double x) noexcept {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

// =============================================================================
// Complex128 — SIMD-friendly complex number with explicit memory layout
// =============================================================================
// Do NOT use std::complex<double> for the hot path.
// It generates suboptimal SIMD code due to strict aliasing rules.

struct alignas(16) Complex128 {
    double real = 0.0;
    double imag = 0.0;

    constexpr Complex128() noexcept = default;
    constexpr Complex128(double r, double i) noexcept : real(r), imag(i) {}
    constexpr explicit Complex128(double r) noexcept : real(r), imag(0.0) {}

    // Arithmetic
    inline constexpr Complex128 operator+(const Complex128& other) const noexcept {
        return {real + other.real, imag + other.imag};
    }

    inline constexpr Complex128 operator-(const Complex128& other) const noexcept {
        return {real - other.real, imag - other.imag};
    }

    inline constexpr Complex128 operator*(const Complex128& other) const noexcept {
        return {
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        };
    }

    inline constexpr Complex128 operator*(double scalar) const noexcept {
        return {real * scalar, imag * scalar};
    }

    inline constexpr Complex128 operator/(double scalar) const noexcept {
        return {real / scalar, imag / scalar};
    }

    inline constexpr Complex128& operator+=(const Complex128& other) noexcept {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    inline constexpr Complex128& operator-=(const Complex128& other) noexcept {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    inline constexpr Complex128& operator*=(const Complex128& other) noexcept {
        double r = real * other.real - imag * other.imag;
        double i = real * other.imag + imag * other.real;
        real = r;
        imag = i;
        return *this;
    }

    inline constexpr Complex128& operator*=(double scalar) noexcept {
        real *= scalar;
        imag *= scalar;
        return *this;
    }

    inline constexpr Complex128 operator-() const noexcept {
        return {-real, -imag};
    }

    inline constexpr bool operator==(const Complex128& other) const noexcept {
        return real == other.real && imag == other.imag;
    }

    inline constexpr bool operator!=(const Complex128& other) const noexcept {
        return !(*this == other);
    }

    // Conjugate
    inline constexpr Complex128 conj() const noexcept {
        return {real, -imag};
    }

    // Magnitude squared
    inline constexpr double norm_sq() const noexcept {
        return real * real + imag * imag;
    }

    // Magnitude
    inline double norm() const noexcept {
        return std::sqrt(norm_sq());
    }

    // Phase angle
    inline double arg() const noexcept {
        return std::atan2(imag, real);
    }

    // Polar form: r * exp(i*theta)
    static inline Complex128 polar(double r, double theta) noexcept {
        return {r * std::cos(theta), r * std::sin(theta)};
    }

    // exp(i*theta)
    static inline Complex128 exp_i(double theta) noexcept {
        return {std::cos(theta), std::sin(theta)};
    }
};

// Free function scalar * Complex128
inline constexpr Complex128 operator*(double scalar, const Complex128& c) noexcept {
    return {scalar * c.real, scalar * c.imag};
}

// =============================================================================
// Aligned memory allocation — cross-platform wrappers
// =============================================================================
// 64-byte alignment required for AVX-512

constexpr size_t CACHE_LINE_SIZE = 64;

inline double* aligned_alloc_doubles(size_t count) {
    void* ptr = nullptr;
    size_t bytes = count * sizeof(double);
    // Ensure allocation is a multiple of alignment
    if (bytes % CACHE_LINE_SIZE != 0) {
        bytes = ((bytes / CACHE_LINE_SIZE) + 1) * CACHE_LINE_SIZE;
    }
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, CACHE_LINE_SIZE);
#else
    ptr = std::aligned_alloc(CACHE_LINE_SIZE, bytes);
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<double*>(ptr);
}

inline void aligned_free(void* ptr) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// =============================================================================
// SVDMethod — SVD backend selector for the MPS layers (audit F-23)
// =============================================================================
// Jacobi is the DEFAULT on both the qubit and qudit MPS: accurate, and it
// sidesteps the Eigen BDCSVD accuracy defect found in R.1.11.2 for
// complex/degenerate inputs (docs/plans/eigen-bdcsvd-bug.md). BDC is faster for
// large bond dimension BUT IS CURRENTLY BROKEN: selecting it emits a loud
// runtime warning (warn_bdc_broken_once) because results may be silently wrong.
// TODO(R.1.13+): once the upstream BDCSVD defect is confirmed fixed, flip the
// default to BDC (it is the faster algorithm) and drop the warning.
enum class SVDMethod { Jacobi, BDC };

// =============================================================================
// Constants
// =============================================================================

constexpr double INV_SQRT2 = 0.7071067811865475;
constexpr double PI = 3.14159265358979323846;
constexpr double PI_2 = PI / 2.0;
constexpr double PI_4 = PI / 4.0;

} // namespace lindblad
