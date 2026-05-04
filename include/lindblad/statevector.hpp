#pragma once

#include "lindblad/types.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {

// =============================================================================
// Statevector — SoA (Structure of Arrays) layout for SIMD efficiency
// =============================================================================
// Two separate aligned arrays for real and imaginary parts.
// SoA layout lets AVX-512 load 8 real parts in one instruction and
// 8 imaginary parts in another, performing 8 complex multiplications
// simultaneously.

class Statevector {
public:
    int n_qubits;
    size_t dim;           // 2^n_qubits

    // Two separate aligned arrays for SoA layout
    double* real_parts;   // aligned to 64 bytes (AVX-512 requirement)
    double* imag_parts;

public:
    // Construct and initialise to |0...0⟩ state
    explicit Statevector(int n_qubits);

    // Destructor — free aligned memory
    ~Statevector();

    // Disable copy (statevectors are large)
    Statevector(const Statevector&) = delete;
    Statevector& operator=(const Statevector&) = delete;

    // Enable move
    Statevector(Statevector&& other) noexcept;
    Statevector& operator=(Statevector&& other) noexcept;

    // Initialise to |0...0⟩ state
    void initialize();

    // Set from a specific computational basis state |k⟩
    void initialize_basis(size_t k);

    // Set from external data
    void set_amplitudes(const double* real, const double* imag, size_t count);
    void set_amplitudes(const std::vector<Complex128>& amplitudes);

    // Get amplitude at index
    Complex128 amplitude(size_t index) const;

    // Get all amplitudes as Complex128 vector
    std::vector<Complex128> amplitudes() const;

    // Probability of measuring index
    double probability(size_t index) const;

    // All probabilities
    std::vector<double> probabilities() const;

    // Squared norm (should be 1.0 for valid state)
    double norm_sq() const;

    // Norm
    double norm() const;

    // Normalise the state
    void normalize();

    // Number of qubits
    int num_qubits() const { return n_qubits; }

    // Dimension
    size_t dimension() const { return dim; }

    // Inner product ⟨this|other⟩
    Complex128 inner_product(const Statevector& other) const;

    // Sample measurement outcomes
    std::unordered_map<std::string, int> sample_counts(
        int shots,
        uint64_t seed = 0
    ) const;

    // Single measurement — returns bitstring
    std::string measure_once(uint64_t seed = 0) const;

    // Create a deep copy
    Statevector clone() const;

    // String representation for debugging
    std::string to_string(int precision = 6) const;
};

} // namespace lindblad
