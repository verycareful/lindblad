#pragma once

#include "lindblad/types.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace lindblad {

// =============================================================================
// QuditMPS — Matrix Product State for n qudits each of dimension d.
//
// Each site tensor A_q has shape (d, chi_L, chi_R):
//   data layout: data[sigma * chi_L * chi_R + aL * chi_R + aR]
// The MPS represents the state
//   |psi> = sum_{sigma_0..sigma_{n-1}} (A_0 A_1 ... A_{n-1})_{sigma_0..sigma_{n-1}}
//                                    * |sigma_0 ... sigma_{n-1}>
// with boundary conditions chi_L[0] = chi_R[n-1] = 1.
//
// Convention: the same little-endian flat indexing as QuditStatevector,
//   flat_index = sum_q sigma_q * d^q.
//
// Bonds are truncated by SVD with `max_bond_dim` and `svd_cutoff`
// (relative to the largest singular value).
// =============================================================================

struct MPSSiteTensor {
    int d;
    int chi_L;
    int chi_R;
    // d * chi_L * chi_R entries, index = sigma * chi_L * chi_R + aL * chi_R + aR
    std::vector<Complex128> data;

    MPSSiteTensor(int d, int chi_L, int chi_R);

    Complex128& at(int sigma, int aL, int aR);
    const Complex128& at(int sigma, int aL, int aR) const;

    // Matricisation for SVD-style operations.
    //   as_left_matrix():  shape (d * chi_L, chi_R), row = sigma*chi_L + aL
    //   as_right_matrix(): shape (chi_L, d * chi_R), col = sigma*chi_R + aR
    Eigen::MatrixXcd as_left_matrix() const;
    Eigen::MatrixXcd as_right_matrix() const;

    // Inverse reshapes.
    //   from_left_matrix(M, d, chi_L):  M is (d*chi_L, chi_R)
    //   from_right_matrix(M, d, chi_R): M is (chi_L, d*chi_R)
    static MPSSiteTensor from_left_matrix(const Eigen::MatrixXcd& M, int d, int chi_L);
    static MPSSiteTensor from_right_matrix(const Eigen::MatrixXcd& M, int d, int chi_R);
};

class QuditMPS {
public:
    int n_qudits;
    int d;
    int max_bond_dim;
    double svd_cutoff;
    std::vector<MPSSiteTensor> tensors;

    // Construct in state |0...0> with bond dim 1.
    QuditMPS(int n_qudits, int d, int max_bond_dim = 64, double svd_cutoff = 1e-12);

    // Construct from a dense statevector via sequential left-to-right SVDs.
    explicit QuditMPS(const QuditStatevector& sv,
                      int max_bond_dim = 64, double svd_cutoff = 1e-12);

    // Full contraction back to a dense statevector. Use for small systems only.
    QuditStatevector to_statevector() const;

    // <psi|psi>; ideally 1.0 after normalisation.
    double norm_sq() const;

    // Rescale tensors[0] so the state has unit norm.
    void normalize();

    // --- Gate / oracle / measurement API ---------------------------------------

    // d x d unitary on qudit q.  Row-major: U[row*d + col].
    void apply_1qudit(int q, const std::vector<Complex128>& U);

    // d^2 x d^2 unitary on adjacent qudits (q, q+1).
    // Row index r = out_q*d + out_{q+1}; col index c = in_q*d + in_{q+1}.
    void apply_2qudit_adjacent(int q, const std::vector<Complex128>& U);

    // d^2 x d^2 unitary on arbitrary qudits (q0, q1), q0 != q1.
    // Non-adjacent pairs are handled with a SWAP chain.
    void apply_2qudit(int q0, int q1, const std::vector<Complex128>& U);

    // Per-basis-state phase: amplitude[idx] *= phase_fn(digits(idx)).
    // Fallback path via dense statevector (always exact, may be slow).
    void apply_phase_oracle(
        const std::function<Complex128(const std::vector<int>&)>& phase_fn);

    // Function oracle |x>|y> -> |x>|(y + f(x)) mod d>.
    // f takes the flat index of the query register (sum_i x_i * d^i) and returns
    // the flat index of the value to add to the output register.
    // Fallback path via dense statevector.
    void apply_function_oracle(int n_query, int n_output,
                               const std::function<int(int)>& f);

    // Sample one outcome from |amplitude[i]|^2 (via dense statevector).
    std::vector<int> measure(uint64_t seed = 0);

    // --- Canonicalisation ------------------------------------------------------

    // Left-canonical sweep: each A_q satisfies sum_{sigma,aL} conj(A)*A = I.
    // Singular values are absorbed into the right neighbour.
    void left_canonicalize();

    // Right-canonical sweep: each A_q satisfies sum_{sigma,aR} A*conj(A) = I.
    // Singular values are absorbed into the left neighbour.
    void right_canonicalize();

private:
    static size_t ipow(size_t base, int exp) noexcept;

    // Build the (d*chi_L) x (d*chi_R) "two-site tensor"
    //   Theta[sigma_q * chi_L + aL, sigma_{q+1} * chi_R + aR]
    //     = sum_{am} A_q[sigma_q, aL, am] * A_{q+1}[sigma_{q+1}, am, aR]
    Eigen::MatrixXcd contract_two_sites(int q) const;

    // SVD-split Theta back into tensors[q] (left-isometry) and tensors[q+1]
    // (absorbing the singular values), truncating to max_bond_dim and svd_cutoff.
    void split_two_sites(int q, const Eigen::MatrixXcd& Theta);

    // SWAP the physical indices of sites (q, q+1) — used to chain non-adjacent
    // gates into a sequence of adjacent operations.
    void apply_swap(int q);

    // Right environments for left-to-right contractions (unused convenience hook).
    std::vector<Eigen::MatrixXcd> build_right_envs() const;
};

} // namespace lindblad
