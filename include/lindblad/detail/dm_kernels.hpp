// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

// =============================================================================
// Density-matrix kernels - the passes whose team the model sizes
// =============================================================================
// ρ is row-major dim × dim, so a row is contiguous. Each kernel below is one
// pass over ρ and starts its own parallel region of dm_team(...) threads (one
// serial pass when that is 1). They are declared here so that the team model
// (dm_team.cpp) can time the very code a run executes on a scratch matrix of
// each size; density_matrix_sim.cpp defines them and DensityMatrix's gate and
// reduction methods call them.
//
// sub_off and bg come from dm_build_tables: sub_off[s] is the offset of
// sub-index s of the k target qubits (MSB-first: sub-index bit qi addresses
// qubits[k-1-qi]), bg the background indices, those with every target bit
// zero.

#include "lindblad/types.hpp"

#include <cstddef>
#include <vector>

namespace lindblad {
namespace detail {

void dm_build_tables(int n_qubits, std::size_t dim, const std::vector<int>& qubits,
                     std::vector<std::size_t>& sub_off, std::vector<std::size_t>& bg);

// ρ → S·vec(ρ block) for every (row background, column background) pair; S is
// the (sub_dim² × sub_dim²) superoperator in the internal sub-index convention.
void dm_superop_apply_inplace(Complex128* data, const std::vector<Complex128>& S,
                              const std::vector<std::size_t>& sub_off,
                              const std::vector<std::size_t>& bg, std::size_t sub_dim,
                              std::size_t dim);

// ρ → U·ρ on the row index.
void dm_ket_apply_inplace(Complex128* data, const std::vector<Complex128>& U,
                          const std::vector<std::size_t>& sub_off,
                          const std::vector<std::size_t>& bg, std::size_t sub_dim,
                          std::size_t dim);

// ρ → ρ·U† on the column index.
void dm_bra_apply_inplace(Complex128* data, const std::vector<Complex128>& U,
                          const std::vector<std::size_t>& sub_off,
                          const std::vector<std::size_t>& bg, std::size_t sub_dim,
                          std::size_t dim);

// Σ over rows r of sign(r) · ρ[r, r ⊕ flip_mask], sign = (-1)^popcount(r &
// sign_mask): the trace of ρ against one Pauli string, before the string's
// coefficient. Summed in the fixed segments of reduce_shares_team, so the bits
// are the same on any team.
struct DmPauliTrace {
    double re = 0.0;
    double im = 0.0;
};
DmPauliTrace dm_pauli_term_trace(const Complex128* data, std::size_t dim, std::size_t flip_mask,
                                 std::size_t sign_mask);

}  // namespace detail
}  // namespace lindblad
