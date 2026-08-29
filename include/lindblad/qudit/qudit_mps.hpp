#pragma once

#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "lindblad/detail/svd_truncate.hpp"
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
// Bonds are truncated by SVD with `max_bond_dim` and `svd_cutoff`, the maximum
// fraction of total weight (sum of sigma^2) a split may discard. Same rule and
// same meaning as MPSState::cutoff in the qubit layer.
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
    // Fraction of total weight (sum of sigma^2) truncation may discard. Not a
    // magnitude threshold: a bare singular value is never compared against it.
    double svd_cutoff;
    // SVD backend: default accurate Jacobi. BDC is a faster opt-in that is
    // CURRENTLY BROKEN (Eigen BDCSVD bug) and emits a loud
    // runtime warning when selected. Shared enum lives in types.hpp.
    SVDMethod svd_method = SVDMethod::Jacobi;
    std::vector<MPSSiteTensor> tensors;

    // Construct in state |0...0> with bond dim 1.
    QuditMPS(int n_qudits, int d, int max_bond_dim = 64, double svd_cutoff = 1e-16);

    // Construct from a dense statevector via sequential left-to-right SVDs.
    explicit QuditMPS(const QuditStatevector& sv,
                      int max_bond_dim = 64, double svd_cutoff = 1e-16);

    // Full contraction back to a dense statevector. Use for small systems only.
    QuditStatevector to_statevector() const;

    // <psi|psi>; ideally 1.0 after normalisation.
    double norm_sq() const;

    // Rescale tensors[0] so the state has unit norm.
    void normalize();

    // True when the norm is 1 to within atol. A predicate: it answers, it does not
    // repair and it does not throw, and a non-finite state answers false.
    bool is_normalized(double atol = DEFAULT_PHYSICAL_ATOL) const;

    // Judge this state's normalization under a validation policy. Fix
    // renormalizes in place; Warn reports it and leaves the state as it is;
    // Throw raises; Ignore measures nothing, so opting out costs one branch
    // rather than a full pass. Fix on a state with nothing to divide out
    // throws rather than returning it unrepaired.
    void check_normalized(ValidationOptions validation = {});

    // --- Gate / oracle / measurement API ---------------------------------------

    // d x d unitary on qudit q.  Row-major: U[row*d + col].
    void apply_1qudit(int q, const std::vector<Complex128>& U,
                      ValidationOptions validation = {});

    // d^2 x d^2 unitary on adjacent qudits (q, q+1).
    // Row index r = out_q*d + out_{q+1}; col index c = in_q*d + in_{q+1}.
    void apply_2qudit_adjacent(int q, const std::vector<Complex128>& U,
                               ValidationOptions validation = {});

    // d^2 x d^2 unitary on arbitrary qudits (q0, q1), q0 != q1.
    // Non-adjacent pairs are handled with a SWAP chain.
    void apply_2qudit(int q0, int q1, const std::vector<Complex128>& U,
                      ValidationOptions validation = {});

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

    // --- Truncation and SVD-ladder observability --------------------------------
    //
    // Every bond split runs SELECT -> VERIFY -> FALLBACK -> THROW: the
    // factorisation the SVD backend returns is measured against the block it
    // came from, and recomputed through an independent route when it does not
    // reconstruct. Both outcomes yield equally valid tensors, so a rescued
    // state is indistinguishable from a clean one without these counters.

    // Total weight (sum of sigma^2) discarded across every split so far.
    double truncation_error() const { return total_truncation_error; }

    // svd_call_count() is the denominator: a bond split calls the truncation
    // once, so a bare fallback count means nothing without it.
    // gram_fallback_count() counts only the rescues that SUCCEEDED; a Gram
    // route that also fails verification throws rather than returning.
    std::size_t svd_call_count() const { return svd_calls; }
    std::size_t gram_fallback_count() const { return gram_fallbacks; }

    // Worst factorisation error the VERIFY rung accepted, as a fraction of
    // ||M||_F^2, maximised over splits. A perfect truncated SVD satisfies the
    // Frobenius identity with equality, so this reports the excess over that
    // ideal rather than the raw residual, and a clean run sits at the square of
    // machine epsilon.
    double max_verify_residual_excess() const { return max_verify_resid_excess; }

private:
    static size_t ipow(size_t base, int exp) noexcept;

    double total_truncation_error = 0.0;
    std::size_t svd_calls = 0;
    std::size_t gram_fallbacks = 0;
    double max_verify_resid_excess = 0.0;

    // The single truncation path for every bond split in this class. Runs the
    // shared verified factorisation over `M` and folds the outcome into the
    // counters above, so no site can accumulate them differently or skip them.
    // ctx names the call site in any exception message.
    detail::SvdTruncation truncate_block(const Eigen::MatrixXcd& M,
                                         const char* ctx);

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
