#pragma once

#include "lindblad/types.hpp"

#include <cstddef>

// =============================================================================
// detail::project_to_unitary - the repair behind Repair::Attempt for unitarity
// =============================================================================
// The nearest unitary to a square matrix M, in the Frobenius sense, is the
// unitary polar factor: with M = W Σ V† its thin SVD, that factor is W V†.
// Replacing Σ with the identity discards exactly the non-unitary part and keeps
// every direction, so a matrix that drifted from unitarity by accumulated
// rounding is pulled back to the unitary closest to what the caller meant.
//
// This is the one repair in the physical-validity family that verifies itself.
// Its whole postcondition is that the output is unitary, and
// unitarity_deviation already measures precisely that, so the caller can check
// the result against the same tolerance that rejected the input rather than
// trusting the routine.
//
// A matrix far from unitary is repaired just as willingly as one that drifted:
// the projection is defined for any M, and Repair::Attempt is a caller asking
// for the nearest unitary, not for a diagnosis. The measured residual before repair is
// what tells a caller how far it was, and Warn reports exactly that without
// repairing.

namespace lindblad {
namespace detail {

// Replaces the rows x rows matrix U in place with its unitary polar factor.
//
// Returns false when the factorisation fails or produces a non-finite factor,
// in which case U is left UNCHANGED rather than partly written: a caller that
// asked for a repair and did not get one must still hold what it handed over.
bool project_to_unitary(Complex128* U, std::size_t rows);

}  // namespace detail
}  // namespace lindblad
