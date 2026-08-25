#pragma once
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace lindblad {

// =============================================================================
// QuditDensityMatrix — density operator ρ for n qudits each of dimension d.
//
// Layout: rho[i*dim + j] = <i|ρ|j>  (row = ket index, col = bra index)
// Indexing: same little-endian mixed-radix convention as QuditStatevector.
// =============================================================================

class QuditDensityMatrix {
public:
    int n_qudits;
    int d;
    size_t dim;     // d^n_qudits
    size_t dim_sq;  // dim * dim

    std::vector<Complex128> rho;  // length dim*dim, row-major

    // Construct and initialise to |0...0><0...0|
    QuditDensityMatrix(int n_qudits, int d);

    // Construct from a pure statevector: ρ = |ψ><ψ|
    explicit QuditDensityMatrix(const QuditStatevector& sv);

    // Reset to |0...0><0...0|
    void initialize();

    // Enforce Hermiticity: ρ ← (ρ + ρ†) / 2
    void symmetrize();

    // Rescale so that Tr(ρ) = 1
    void normalize();

    double trace() const;
    double purity() const;   // Tr(ρ²)

    // -------------------------------------------------------------------------
    // Unitary evolution: ρ → U_q ρ U_q†
    // -------------------------------------------------------------------------

    void apply_1qudit(int q, const std::vector<Complex128>& U,
                      ValidationOptions validation = {});
    void apply_2qudit(int q0, int q1, const std::vector<Complex128>& U,
                      ValidationOptions validation = {});

    // -------------------------------------------------------------------------
    // Kraus channels: ρ → Σ_k K_k ρ K_k†
    // -------------------------------------------------------------------------

    void apply_kraus_1qudit(int q,
                            const std::vector<std::vector<Complex128>>& K_ops,
                            ValidationOptions validation = {});
    void apply_kraus_2qudit(int q0, int q1,
                            const std::vector<std::vector<Complex128>>& K_ops,
                            ValidationOptions validation = {});

    // -------------------------------------------------------------------------
    // Lindblad master equation step (first-order Euler):
    //   ρ ← ρ + dt * Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
    // -------------------------------------------------------------------------

    void apply_lindblad_step(int q, const std::vector<QuditLindbladOp>& ops,
                             double dt);

    // -------------------------------------------------------------------------
    // Apply all per-qudit Kraus noise from a noise model
    // Apply all noise from model: Kraus channels always; Lindblad steps if dt > 0.
    // -------------------------------------------------------------------------

    void apply_noise(const QuditNoiseModel& model, double dt = 0.0);

    // -------------------------------------------------------------------------
    // Oracle operations
    // -------------------------------------------------------------------------

    // Phase oracle: ρ_{ij} ← phase(i) * conj(phase(j)) * ρ_{ij}
    void apply_phase_oracle(
        const std::function<Complex128(const std::vector<int>&)>& phase_fn);

    // Function oracle: |x>|y> → |x>|(y + f(x)) mod d>
    // Query qudits: [0, n_query); output qudits: [n_query, n_query+n_output)
    void apply_function_oracle(
        int n_query, int n_output,
        const std::function<std::vector<int>(const std::vector<int>&)>& f);

    // -------------------------------------------------------------------------
    // Measurement
    // -------------------------------------------------------------------------

    // Sample from diagonal probabilities; collapse ρ to post-measurement state.
    // Returns per-qudit digit values in qudit-index order.
    std::vector<int> measure(uint64_t seed = 0);

    // -------------------------------------------------------------------------
    // Partial trace: keep only `keep_qudits`, trace out all others
    // Throws std::invalid_argument if keep_qudits is empty.
    // -------------------------------------------------------------------------

    QuditDensityMatrix partial_trace(const std::vector<int>& keep_qudits) const;

private:
    static size_t ipow(size_t base, int exp) noexcept;

    // Apply d×d matrix U to the ket (row) index of ρ for qudit q
    void apply_to_ket(int q, const std::vector<Complex128>& U);

    // Apply U† to the bra (column) index of ρ for qudit q
    void apply_to_bra(int q, const std::vector<Complex128>& U);

    // Apply d²×d² matrix U to ket indices for the (q0, q1) leg pair
    void apply_to_ket_2q(int q0, int q1, const std::vector<Complex128>& U);

    // Apply U† to bra indices for the (q0, q1) leg pair
    void apply_to_bra_2q(int q0, int q1, const std::vector<Complex128>& U);
};

} // namespace lindblad
