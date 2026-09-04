#pragma once

#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

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

    // Set from external data.
    //
    // validation = policy and tolerance for the normalization of the supplied
    // amplitudes. This is the point at which a caller hands a whole state over,
    // which is why the check lives here and not on the evolution path: a state
    // handed over is a claim, while a state the library evolved drifts by its
    // own rounding and would fail a check that nothing is wrong with.
    //
    // The default is Throw with no repair, matching every other
    // physical-validity property. Repair::Attempt renormalizes the amplitudes
    // that were given; Ignore with no repair restores the unchecked behaviour
    // at the cost of one branch. The policy is judged
    // against the CALLER'S buffer before anything is written, so a rejected
    // hand-over leaves this object exactly as it was.
    void set_amplitudes(const double* real, const double* imag, size_t count,
                        ValidationOptions validation = {});
    void set_amplitudes(const std::vector<Complex128>& amplitudes,
                        ValidationOptions validation = {});

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

    // Normalise the state. Throws when there is no norm to divide out, a zero
    // or non-finite state, rather than returning it unchanged.
    void normalize();

    // True when ⟨ψ|ψ⟩ is 1 to within atol. A predicate: it answers, it does not
    // repair and it does not throw, and a non-finite state answers false.
    bool is_normalized(double atol = DEFAULT_PHYSICAL_ATOL) const;

    // Judge this state's normalization under a validation policy.
    // Repair::Attempt renormalizes in place; without it Warn reports and
    // leaves the state as it is, Throw raises, and Ignore measures nothing, so
    // opting out costs one branch rather than a sweep of the amplitudes. A
    // state with no norm to divide out cannot be rescaled at all, so the
    // response decides that case too rather than the repair request forcing a
    // throw.
    void check_normalized(ValidationOptions validation = {});

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
