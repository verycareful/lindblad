#pragma once

#include "lindblad/types.hpp"

#include <Eigen/Dense>

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
// DEFINED in src/svd_truncate.cpp, which is compiled -fno-fast-math. That flag
// is load-bearing rather than cautious. Under -ffast-math the same Eigen input
// yields damage that reaches the KEPT singular channels as finite-but-wrong
// data, which no truncation-side check can see; under strict FP the damage is
// confined to discardable null-space columns, which SELECT then handles. The
// check and the flag guard different halves of the same defect, so this routine
// needs both and lives in its own translation unit to keep the second one from
// depending on who included it.

namespace lindblad {
namespace detail {

// -----------------------------------------------------------------------------
// MatrixOrder - how the caller's block is laid out in memory
// -----------------------------------------------------------------------------
// Both orders are accepted so neither caller has to transpose or copy a block
// before handing it over: the qubit layer builds row-major blocks, the qudit
// layer holds column-major Eigen matrices, and the routine maps whichever it is
// given in place.

enum class MatrixOrder { RowMajor, ColMajor };

// -----------------------------------------------------------------------------
// SvdTruncation - the kept slice, plus what it cost to get it
// -----------------------------------------------------------------------------
// U and V are the KEPT columns only, in descending sigma order. V is returned
// as V, not V†, because that is the form both callers absorb into a neighbouring
// tensor.

struct SvdTruncation {
    Eigen::MatrixXcd U;  // rows x rank
    Eigen::VectorXd S;   // rank
    Eigen::MatrixXcd V;  // cols x rank
    int rank = 0;

    // Weight (Σ sigma²) this split threw away, summed over every finite sigma
    // not kept. A caller accumulates it into its own truncation-error total.
    double discarded_weight = 0.0;

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
