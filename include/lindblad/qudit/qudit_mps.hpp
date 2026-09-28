// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "lindblad/detail/svd_truncate.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"

#include "lindblad/detail/dense_matrix.hpp"
#include "lindblad/detail/fidelity_ledger.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <utility>
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
//
// Canonical form works as in the qubit layer (MPSState in
// simulators/mps_sim.hpp): the chain keeps an open span [lo, hi] with every
// site left of it left-orthonormal (as_left_matrix() an isometry) and every
// site right of it right-orthonormal (as_right_matrix() a co-isometry), moves
// its orthogonality centre by exact QR and LQ steps, and lets canonical_form
// decide which bond splits move the centre onto their block first. The rank
// bound the Auto rule reads is min(d chi_L, d chi_R).
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
    detail::DenseMatrix as_left_matrix() const;
    detail::DenseMatrix as_right_matrix() const;

    // Inverse reshapes.
    //   from_left_matrix(M, d, chi_L):  M is (d*chi_L, chi_R)
    //   from_right_matrix(M, d, chi_R): M is (chi_L, d*chi_R)
    static MPSSiteTensor from_left_matrix(const detail::DenseMatrix& M, int d,
                                          int chi_L);
    static MPSSiteTensor from_right_matrix(const detail::DenseMatrix& M, int d,
                                           int chi_R);
};

class QuditMPS {
public:
    int n_qudits;
    int d;
    int max_bond_dim;
    // Fraction of total weight (sum of sigma^2) truncation may discard. Not a
    // magnitude threshold: a bare singular value is never compared against it.
    double svd_cutoff;
    // Factorisation every bond split asks for first, and whether a rejected
    // one may descend the rescue ladder. Same meaning as MPSState::svd_method
    // and MPSState::svd_rescue; the enum is documented in types.hpp.
    SVDMethod svd_method = SVDMethod::BDC;
    bool svd_rescue = true;
    // Which bond splits first move the orthogonality centre onto their block.
    // Same meaning as MPSState::canonical_form; the enum is documented in
    // types.hpp.
    CanonicalForm canonical_form = CanonicalForm::Always;
    // Whether a gate no policy measured is measured for this chain's own
    // records anyway. Same meaning as MPSState::unchecked_gates; the enum is
    // documented in types.hpp.
    UncheckedGates unchecked_gates = UncheckedGates::Track;

    // Construct in state |0...0> with bond dim 1, centred on site 0.
    QuditMPS(int n_qudits, int d, int max_bond_dim = 64,
             double svd_cutoff = MPS_DEFAULT_CUTOFF);

    // Construct from a dense statevector via sequential left-to-right SVDs,
    // which leave the centre on the last site. Its splits run with the default
    // factorisation settings and count towards the new object's figures.
    explicit QuditMPS(const QuditStatevector& sv,
                      int max_bond_dim = 64,
                      double svd_cutoff = MPS_DEFAULT_CUTOFF);

    // The site tensors, left to right, one per qudit. Read-only, so reading
    // them cannot disturb the open span the chain's operations rely on.
    const std::vector<MPSSiteTensor>& tensors() const noexcept { return tensors_; }

    // Replace the chain with `sites`, one tensor per qudit, qudit 0 first.
    // Validated before anything is replaced: the count must equal n_qudits,
    // every site must have physical dimension d, every bond must be at least
    // 1 with the two outer ones exactly 1, neighbouring bonds must agree, each
    // data array must hold d * chi_L * chi_R entries, and every entry must be
    // finite. Throws std::invalid_argument naming the first violation, leaving
    // the state as it was. As MPSState::set_tensors: the open span becomes the
    // whole chain, the fidelity figures reset to exact, and the profile
    // counters and truncation_error() are untouched.
    void set_tensors(std::vector<MPSSiteTensor> sites);

    // The open span {lo, hi}; {c, c} is mixed canonical form centred on c.
    std::pair<int, int> open_span() const noexcept { return {span_lo, span_hi}; }

    // Move the orthogonality centre to `site`, leaving open_span() equal to
    // {site, site}. A gauge change made of QR and LQ steps, as
    // MPSState::canonicalize.
    void canonicalize(int site);

    // Full contraction back to a dense statevector. Use for small systems only.
    QuditStatevector to_statevector() const;

    // <psi|psi>; ideally 1.0 after normalisation. Contracts only the open
    // span, since the sites outside it contract to the identity, so at a
    // single-site centre it is that site's squared Frobenius norm.
    double norm_sq() const;

    // Rescale the first site of the open span so the state has unit norm. A
    // site outside the span would lose its orthonormality if it were scaled.
    void normalize();

    // True when the norm is 1 to within atol. A predicate: it answers, it does not
    // repair and it does not throw, and a non-finite state answers false.
    bool is_normalized(double atol = DEFAULT_PHYSICAL_ATOL) const;

    // Judge this state's normalization under a validation policy.
    // Repair::Attempt renormalizes in place; without it Warn reports and
    // leaves the state as it is, Throw raises, and Ignore measures nothing,
    // so opting out costs one branch rather than a full pass. A state with
    // nothing to divide out cannot be rescaled at all, so the response
    // decides that case too rather than the repair request forcing a throw.
    void check_normalized(ValidationOptions validation = {});

    // --- Gate / oracle / measurement API ---------------------------------------

    // d x d unitary on qudit q.  Row-major: U[row*d + col]. Leaves open_span()
    // as it is when U is unitary: a unitary on the physical index preserves
    // both orthonormalities.
    //
    // Every gate below applies U exactly as given, whatever its unitarity.
    // When U is not unitary to DEFAULT_PHYSICAL_ATOL (known from the policy's
    // own measurement, or from one taken for the chain's records as
    // unchecked_gates says), apply_1qudit on a site outside the open span
    // widens the span over that site, and any gate empties the fidelity
    // figures, as on MPSState. Nothing is reported.
    void apply_1qudit(int q, const std::vector<Complex128>& U,
                      ValidationOptions validation = {});

    // d^2 x d^2 unitary on adjacent qudits (q, q+1). Project LSB-first
    // convention (docs/Architecture.md "Conventions"): qudit q is the LEAST
    // significant digit of the index, so row r = out_{q+1}*d + out_q and
    // column c = in_{q+1}*d + in_q.
    void apply_2qudit_adjacent(int q, const std::vector<Complex128>& U,
                               ValidationOptions validation = {});

    // d^2 x d^2 unitary on arbitrary qudits (q0, q1), q0 != q1.
    // Non-adjacent pairs are handled with a SWAP chain, each split of which
    // leaves the centre where the next one starts.
    void apply_2qudit(int q0, int q1, const std::vector<Complex128>& U,
                      ValidationOptions validation = {});

    // Per-basis-state phase: amplitude[idx] *= phase_fn(digits(idx)).
    // Fallback path via dense statevector (always exact, may be slow). The
    // chain is rebuilt with this object's own settings, the rebuild's splits
    // count like any other, and the centre ends on the last site.
    void apply_phase_oracle(
        const std::function<Complex128(const std::vector<int>&)>& phase_fn);

    // Function oracle |x>|y> -> |x>|(y + f(x)) mod d>.
    // f takes the flat index of the query register (sum_i x_i * d^i) and returns
    // the flat index of the value to add to the output register.
    // Fallback path via dense statevector, rebuilt as apply_phase_oracle is.
    void apply_function_oracle(int n_query, int n_output,
                               const std::function<int(int)>& f);

    // Sample one outcome from |amplitude[i]|^2 without collapsing the state.
    // The centre moves to qudit 0, where every other site is
    // right-orthonormal, and the digits are drawn left to right carrying a
    // vector on the bond, O(n·d·chi²). A gauge change only: the state and
    // the fidelity figures are unchanged. seed == 0 draws a seed from
    // std::random_device.
    //
    // Each digit is drawn from its marginals divided by their sum, the chain's
    // own normalised distribution, and never lands on a digit of weight zero.
    // A chain with no norm, zero or non-finite, is refused with
    // std::runtime_error before the first draw.
    std::vector<int> measure(uint64_t seed = 0);

    // Measure qudit `q` in the computational basis and collapse the chain onto
    // the outcome, returned as a digit in [0, d). The centre moves to the
    // qudit first, so its marginals are local and the collapse renormalises
    // that site alone; afterwards the state has unit norm and open_span() is
    // {q, q}. Draws one uniform from `rng`. The fidelity figures become empty.
    // The draw is from the chain's own normalised distribution, as measure()
    // draws, and a chain with no norm is refused before it, leaving `rng` as
    // it was.
    int measure_qudit(int q, std::mt19937_64& rng);

    // --- Canonicalisation ------------------------------------------------------

    // Left-canonical sweep by truncated SVD: each A_q for q < n-1 satisfies
    // sum_{sigma,aL} conj(A)*A = I, and singular values are absorbed into the
    // right neighbour, so the centre ends on the last site. The centre moves
    // to site 0 first, so every split in the sweep is taken in canonical gauge
    // and truncates on the state's Schmidt coefficients. Each split counts like
    // any other.
    void left_canonicalize();

    // Right-canonical sweep, the mirror image: each A_q for q > 0 satisfies
    // sum_{sigma,aR} A*conj(A) = I, singular values are absorbed into the left
    // neighbour, the centre moves to the last site first and ends on site 0.
    void right_canonicalize();

    // --- Truncation and SVD-ladder observability --------------------------------
    //
    // Every bond split runs SELECT -> VERIFY -> FALLBACK -> THROW: the
    // factorisation the SVD backend returns is measured against the block it
    // came from, and recomputed through an independent route when it does not
    // reconstruct. Both outcomes yield equally valid tensors, so a rescued
    // state is indistinguishable from a clean one without these counters.

    // Total weight (sum of sigma^2) discarded across every split so far, each
    // term the absolute weight its split threw away. It ACCUMULATES, so two
    // runs are comparable only when they perform the same splits. A split in
    // canonical gauge discards that weight from the state, so with no collapse
    // and no normalisation the total equals how far norm_sq() has fallen, to
    // rounding; the meaning in full is with MPSState::truncation_error().
    double truncation_error() const { return total_truncation_error; }

    // How close the chain is to the state an untruncated evolution would hold:
    // an estimate that is not a bound, and a rigorous lower bound. Same
    // figures, same rules (1 on a new chain and after set_tensors(), empty
    // after any collapse or any gate that is not unitary) as
    // MPSState::fidelity_estimate() and MPSState::fidelity_lower_bound().
    std::optional<double> fidelity_estimate() const noexcept {
        return fidelity.estimate();
    }
    std::optional<double> fidelity_lower_bound() const noexcept {
        return fidelity.lower_bound();
    }

    // svd_call_count() is the denominator: a bond split calls the truncation
    // once, so a bare rescue count means nothing without it.
    // jacobi_rescue_count() and gram_fallback_count() count only the rescues
    // that SUCCEEDED, one or the other per rescued split; a split on which
    // every rung fails throws rather than returning. floor_rejected_weight()
    // is the Gram route's validity-floor cost, booked outside
    // truncation_error(); the meaning of each is with the qubit MPSState's
    // accessors of the same names.
    std::size_t svd_call_count() const { return svd_calls; }
    std::size_t jacobi_rescue_count() const { return jacobi_rescues; }
    std::size_t gram_fallback_count() const { return gram_fallbacks; }
    double floor_rejected_weight() const { return floor_rejected; }

    // Time spent in the bond-split factorisation path, in nanoseconds, over
    // the same splits svd_call_count() counts and bracketing the whole ladder
    // the way the qubit MPSState::svd_time_ns() does, so the two layers'
    // figures mean the same thing.
    std::uint64_t svd_time_ns() const { return svd_nanos; }

    // Worst factorisation error the VERIFY rung accepted, as a fraction of
    // ||M||_F^2, maximised over splits. A perfect truncated SVD satisfies the
    // Frobenius identity with equality, so this reports the excess over that
    // ideal rather than the raw residual, and a clean run sits at the square of
    // machine epsilon.
    double max_verify_residual_excess() const { return max_verify_resid_excess; }

private:
    static size_t ipow(size_t base, int exp) noexcept;

    std::vector<MPSSiteTensor> tensors_;
    // The open span; see the class comment.
    int span_lo = 0;
    int span_hi = 0;
    detail::FidelityLedger fidelity;

    double total_truncation_error = 0.0;
    std::size_t svd_calls = 0;
    std::size_t jacobi_rescues = 0;
    std::size_t gram_fallbacks = 0;
    double floor_rejected = 0.0;
    std::uint64_t svd_nanos = 0;
    double max_verify_resid_excess = 0.0;

    // Which site of a split receives the singular values; as in MPSState.
    enum class Absorb { Left, Right };

    // The single truncation path for every bond split in this class. Runs the
    // shared verified factorisation over `M`, folds the outcome into the
    // counters above once it has returned, and records the split's discarded
    // fraction into `ledger`, so no site can accumulate them differently or
    // skip them. ctx names the call site in any exception message.
    detail::SvdTruncation truncate_block(const detail::DenseMatrix& M,
                                         const char* ctx,
                                         detail::FidelityLedger& ledger);

    // Replace the chain with the sequential-SVD factorisation of `sv`, with
    // this object's settings, counting every split. Built beside the chain,
    // so a throw part way through leaves it and its figures as they were.
    void rebuild_from(const QuditStatevector& sv);

    // Build the (d*chi_L) x (d*chi_R) "two-site tensor"
    //   Theta[sigma_q * chi_L + aL, sigma_{q+1} * chi_R + aR]
    //     = sum_{am} A_q[sigma_q, aL, am] * A_{q+1}[sigma_{q+1}, am, aR]
    detail::DenseMatrix contract_two_sites(int q) const;

    // SVD-split Theta back into tensors[q] and tensors[q+1], truncating to
    // max_bond_dim and svd_cutoff. The singular values go to the side
    // `absorb` names and the other keeps its isometry.
    void split_two_sites(int q, const detail::DenseMatrix& Theta, Absorb absorb);

    // The adjacent two-qudit gate after validation: moves the centre when
    // split_needs_centre says so, contracts, applies U, splits.
    void gate_adjacent(int q, const std::vector<Complex128>& U, Absorb absorb);

    // apply_2qudit after validation: distinct qudits in either order. A
    // reversed pair exchanges the two digit roles of U, which permutes its
    // rows and columns alike and so keeps what validation judged, then
    // applies it to the ordered pair. Nothing here judges U again, so the
    // caller's policy decides once and its repaired matrix is what lands.
    void gate_pair(int q0, int q1, const std::vector<Complex128>& U);

    // SWAP the physical indices of sites (q, q+1), used to chain non-adjacent
    // gates into a sequence of adjacent operations.
    void apply_swap(int q, Absorb absorb);

    // As MPSState: the CanonicalForm rule for the block at (q, q+1), one step
    // of the centre each way, and narrowing the open span into [a, b].
    bool split_needs_centre(int q) const;
    void shift_right(int q);
    void shift_left(int q);
    void focus(int a, int b);

    // Squared norms of the d physical slices of `site`: the raw marginals when
    // the site is the centre.
    std::vector<double> centre_marginals(int site) const;
};

} // namespace lindblad
