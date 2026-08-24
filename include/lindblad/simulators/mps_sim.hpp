#pragma once

#include "lindblad/types.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <unordered_map>
#include <vector>

namespace lindblad {

class Statevector;
class QuantumCircuit;

// =============================================================================
// MPSTensor — tensor for one qubit site
// =============================================================================

struct MPSTensor {
    // Shape: (bond_left, physical_dim=2, bond_right)
    int bond_left;
    int bond_right;
    // data layout: data[left * 2 * bond_right + phys * bond_right + right]
    std::vector<Complex128> data;

    MPSTensor() : bond_left(1), bond_right(1), data(2, Complex128(0.0, 0.0)) {}
    MPSTensor(int bl, int br) : bond_left(bl), bond_right(br),
        data(bl * 2 * br, Complex128(0.0, 0.0)) {}

    Complex128& operator()(int left, int phys, int right) {
        return data[left * 2 * bond_right + phys * bond_right + right];
    }
    const Complex128& operator()(int left, int phys, int right) const {
        return data[left * 2 * bond_right + phys * bond_right + right];
    }
};

// SVDMethod (Jacobi default, BDC opt-in-but-currently-broken) is declared in
// types.hpp so the qubit and qudit MPS layers share one enum.

// =============================================================================
// MPSState — Matrix Product State
// =============================================================================

class MPSState {
public:
    int n_qubits;
    int max_bond_dim;    // chi — controls accuracy vs memory tradeoff
    // Truncation budget: the maximum FRACTION OF TOTAL WEIGHT (sum of sigma^2)
    // that a bond truncation may discard. Not a magnitude threshold — a bare
    // sigma is never compared against it. See svd_truncate for why: a magnitude
    // threshold asks a question whose answer depends on the scale of the input
    // and on how the target rounded, so the same state could carry a different
    // bond dimension on a different CPU.
    double cutoff;
    // SVD backend for truncation (default Jacobi; see SVDMethod). BDC is a
    // faster opt-in that is not the default pending an upstream Eigen BDCSVD
    // accuracy fix.
    SVDMethod svd_method = SVDMethod::Jacobi;
    std::vector<MPSTensor> tensors;

public:
    // cutoff defaults to 1e-16: truncation error is then bounded at the order
    // of the reconstruction error an SVD already carries (~1.1e-16 relative),
    // so nothing the factorisation actually resolved is thrown away.
    MPSState(int n_qubits, int max_bond_dim = 64, double cutoff = 1e-16);

    // Gate application via SVD
    void apply_single_qubit_gate(
        const std::array<Complex128, 4>& U, int qubit
    );
    void apply_two_qubit_gate(
        const std::array<Complex128, 16>& U, int q1, int q2
    );

    // Truncation info
    double truncation_error() const { return total_truncation_error; }
    int current_max_bond_dim() const;

    // SVD ladder observability.
    //
    // svd_truncate runs SELECT -> VERIFY -> FALLBACK -> THROW: it distrusts the
    // SVD backend's factorisation and recomputes through the Gram route when
    // verification rejects it. Both outcomes are silent from outside, so these
    // two counters are the only way to tell a state that took the primary path
    // throughout from one that was rescued on every bond.
    //
    // svd_call_count() is the denominator: a bond split calls svd_truncate once,
    // so a bare fallback count means nothing without it. gram_fallback_count()
    // counts only the rescues that SUCCEEDED; a Gram route that also fails
    // verification throws rather than returning.
    std::size_t gram_fallback_count() const { return gram_fallbacks; }
    std::size_t svd_call_count() const { return svd_calls; }

    // Worst factorisation error the VERIFY rung accepted, as a fraction of
    // ||M||_F^2, maximised over splits. A perfect truncated SVD satisfies the
    // Frobenius identity with equality, so this reports the excess over that
    // ideal rather than the raw residual, and a clean run sits at the square
    // of machine epsilon (~1e-32). It is the ladder's own decision variable:
    // how close a run came to being rescued, and how much error the accepted
    // route let through when it was not.
    double max_verify_residual_excess() const { return max_verify_resid_excess; }

    // Measurements and expectation values
    std::vector<double> probabilities_single(int qubit) const;

    // Sequential measurement: sample a full bitstring respecting correlations.
    // Measures qubit 0, conditions on outcome, propagates boundary, repeats.
    // O(N * chi^3) per shot. Modifies internal state (projects measured qubits).
    std::string measure_sequential(std::mt19937_64& rng);

    // Convert to exact statevector (expensive, for small N only)
    Statevector to_statevector() const;

private:
    double total_truncation_error = 0.0;
    std::size_t gram_fallbacks = 0;
    std::size_t svd_calls = 0;
    double max_verify_resid_excess = 0.0;

    // SVD helper
    void svd_truncate(
        const std::vector<Complex128>& matrix,
        int rows, int cols,
        std::vector<Complex128>& U,
        std::vector<double>& S,
        std::vector<Complex128>& Vt,
        int& new_rank
    );

    // Adjacent two-qubit gate application (internal)
    void apply_two_qubit_gate_adjacent(
        const std::array<Complex128, 16>& U, int q1
    );

    // Adjacent SWAP gate (internal)
    void apply_swap_adjacent(int q);
};

// =============================================================================
// MPSSimulator
// =============================================================================

class MPSSimulator {
public:
    struct Result {
        MPSState final_state;
        std::unordered_map<std::string, int> counts;
        double simulation_time_seconds = 0.0;

        Result(int n) : final_state(n) {}
        Result(Result&&) = default;
        Result& operator=(Result&&) = default;
    };

    Result run(const QuantumCircuit& circuit, int max_bond_dim = 64,
               int shots = 1024, uint64_t seed = 0);
};

} // namespace lindblad
