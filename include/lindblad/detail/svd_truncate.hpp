#pragma once

#include "lindblad/types.hpp"

// MatrixOrder is defined with the backend that consumes it. Neither header
// names an Eigen type, so this interface does not oblige a caller to have Eigen
// available to hold a factorisation.
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/detail/dense_matrix.hpp"

// =============================================================================
// detail::svd_truncate_verified - a truncated SVD that trusts nothing
// =============================================================================
// Every MPS bond split factorises a block and keeps the leading directions.
// The factorisation comes from a third-party SVD, and this routine treats that
// output as a claim to be checked rather than an answer to be used.
//
// The reason is specific and measured. On degenerate rank-deficient inputs
// (the two-site thetas of a 13-qubit period-finding circuit, spectrum
// {1,1,1,1,0,...}) Eigen 3.4.0 returns factorisations whose shape morphs with
// tiny input and codegen perturbations: NaN inside null-space singular vectors,
// non-finite entries interleaved into S displacing real sigmas, an all-non-
// finite S, and worst of all a KEPT singular vector that is wrong and perfectly
// finite. That last mode carries no marker at all, so it cannot be pattern-
// matched away; the only way to find it is to measure the factorisation against
// the matrix it came from.
//
// The ladder is SELECT -> VERIFY -> FALLBACK -> THROW, and each rung is
// documented at its implementation. A caller sees only a valid factorisation or
// an exception: which rung produced it is reported through the return value so
// a caller that wants to count rescues can, and ignored otherwise.
//
// DEFINED in src/svd_truncate.cpp, under the project-wide flags. The two pieces
// this ladder cannot afford to have optimised are quarantined where they live
// rather than here: the factorisation itself in src/eigen_backend.cpp, whose
// entry guard -ffast-math would otherwise delete, and the reconstruction
// residual in src/svd_verify.cpp, which subtracts two nearly identical matrices
// to decide accept or reject. Selection, budgeting and rescue routing are
// ordinary arithmetic and are compiled as such.

namespace lindblad {
namespace detail {

// -----------------------------------------------------------------------------
// SvdTruncation - the kept slice, plus what it cost to get it
// -----------------------------------------------------------------------------
// U and V are the KEPT columns only, in descending sigma order. V is returned
// as V, not V†, because that is the form both callers absorb into a neighbouring
// tensor.

struct SvdTruncation {
    DenseMatrix U;  // rows x rank, column-major
    RealVector S;   // rank
    DenseMatrix V;  // cols x rank, column-major
    int rank = 0;

    // Weight (Σ sigma²) this split CHOSE to throw away: directions the weight
    // budget or the bond cap rejected. A caller accumulates it into its own
    // truncation-error total.
    double discarded_weight = 0.0;

    // Weight rejected by the Gram route's validity floor, reported separately
    // because it is not truncation. Forming G = M†M squares the condition
    // number, so a singular value that is exactly zero in the input comes back
    // at ~sqrt(eps) and carries ~eps of weight that was never in the matrix.
    // Folding that into `discarded_weight` would report a bond which discarded
    // nothing as having lost something.
    //
    // Always zero on the primary route, which uses no floor. A nonzero value
    // here therefore says two things at once: this split was rescued, and this
    // much of its spectrum was too ill-conditioned for the rescue to trust.
    // It is surfaced rather than dropped so a caller can see that the rescue
    // route paid for its robustness, which is the whole cost of taking it.
    double floor_rejected_weight = 0.0;

    // How far the accepted factorisation sat above a perfect one, as a fraction
    // of ‖M‖_F². A truncated SVD satisfies the Frobenius identity with
    // EQUALITY, so this reports the excess over that ideal rather than a raw
    // residual, and means the same thing on a bond that truncated heavily and
    // one that truncated nothing. A clean run sits at the square of machine
    // epsilon.
    double residual_excess = 0.0;

    // True when the primary factorisation failed verification and the Gram
    // route produced the returned slice. Both outcomes are equally valid
    // tensors, so this is the only way to tell a state that took the primary
    // path throughout from one that was rescued on every bond.
    bool used_gram_fallback = false;
};

// -----------------------------------------------------------------------------
// svd_truncate_verified
// -----------------------------------------------------------------------------
// data   = the block, rows x cols, contiguous in `order`
// cutoff = the FRACTION OF TOTAL WEIGHT (Σ sigma²) this split may discard. Not
//          a magnitude threshold: a bare sigma is never compared against it.
//          A magnitude rule asks a question whose answer depends on the scale
//          of the input and on how the target rounded its way there, so the
//          same state carries a different bond dimension on a different CPU.
//          A weight fraction is scale-free and bounds the physical error
//          directly. The budget is a CEILING, not a quota: on a bimodal
//          spectrum there is nothing between the noise and the budget, so
//          nothing extra is discarded.
// ctx    = caller name, used in the exception messages so a failure names the
//          layer it came from.
//
// Throws std::runtime_error when both the backend factorisation and the Gram
// fallback fail verification, rather than returning a corrupt tensor.
SvdTruncation svd_truncate_verified(const Complex128* data, int rows, int cols,
                                    MatrixOrder order, int max_bond_dim,
                                    double cutoff, SVDMethod method,
                                    const char* ctx);

} // namespace detail
} // namespace lindblad
