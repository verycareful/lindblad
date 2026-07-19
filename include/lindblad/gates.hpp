#pragma once

#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <vector>

namespace lindblad {
namespace gates {

// =============================================================================
// Validation contract
// =============================================================================
// Every apply_* primitive fails loud on a malformed operand before touching
// memory: an out-of-range qubit throws std::out_of_range, a bad operand
// structure (non-distinct qubits where required, wrong matrix/permutation size)
// throws std::invalid_argument. Messages match the circuit layer
// (QuantumCircuit::validate_qubit). The checks are a handful of integer
// comparisons at the O(2^n) kernel entry, so they are free next to the sweep.
// Because they throw, none of these functions is noexcept. See
// include/lindblad/detail/validate.hpp.

// =============================================================================
// Single-qubit gates
// =============================================================================

// Pauli gates
void apply_x(Statevector& sv, int q);
void apply_y(Statevector& sv, int q);
void apply_z(Statevector& sv, int q);

// Hadamard
void apply_h(Statevector& sv, int q);

// Phase gates
void apply_s(Statevector& sv, int q);
void apply_sdg(Statevector& sv, int q);
void apply_t(Statevector& sv, int q);
void apply_tdg(Statevector& sv, int q);

// Sqrt X gates
void apply_sx(Statevector& sv, int q);
void apply_sxdg(Statevector& sv, int q);

// Parameterised rotation gates
void apply_rx(Statevector& sv, int q, double theta);
void apply_ry(Statevector& sv, int q, double theta);
void apply_rz(Statevector& sv, int q, double theta);

// Phase gate P(lambda) = diag(1, exp(i*lambda))
void apply_p(Statevector& sv, int q, double lambda);

// General single-qubit unitary U(theta, phi, lambda)
void apply_u(Statevector& sv, int q,
             double theta, double phi, double lambda);
void apply_u1(Statevector& sv, int q, double lambda);
void apply_u2(Statevector& sv, int q, double phi, double lambda);
void apply_u3(Statevector& sv, int q,
              double theta, double phi, double lambda);

// =============================================================================
// Two-qubit gates
// =============================================================================

// Controlled NOT (CNOT)
void apply_cx(Statevector& sv, int ctrl, int tgt);

// Controlled Y
void apply_cy(Statevector& sv, int ctrl, int tgt);

// Controlled Z
void apply_cz(Statevector& sv, int ctrl, int tgt);

// Controlled H
void apply_ch(Statevector& sv, int ctrl, int tgt);

// SWAP
void apply_swap(Statevector& sv, int q1, int q2);

// iSWAP
void apply_iswap(Statevector& sv, int q1, int q2);

// Controlled rotation gates
void apply_crx(Statevector& sv, int ctrl, int tgt, double theta);
void apply_cry(Statevector& sv, int ctrl, int tgt, double theta);
void apply_crz(Statevector& sv, int ctrl, int tgt, double theta);

// Controlled phase
void apply_cp(Statevector& sv, int ctrl, int tgt, double lambda);

// Controlled U
void apply_cu(Statevector& sv, int ctrl, int tgt,
              double theta, double phi, double lambda, double gamma);

// Echoed cross-resonance
void apply_ecr(Statevector& sv, int q1, int q2);

// Ising interaction gates
void apply_rzx(Statevector& sv, int q1, int q2, double theta);
void apply_rxx(Statevector& sv, int q1, int q2, double theta);
void apply_ryy(Statevector& sv, int q1, int q2, double theta);
void apply_rzz(Statevector& sv, int q1, int q2, double theta);

// =============================================================================
// Three-qubit gates
// =============================================================================

// Toffoli (CCX)
void apply_ccx(Statevector& sv, int c1, int c2, int tgt);

// CCZ
void apply_ccz(Statevector& sv, int c1, int c2, int tgt);

// Fredkin (CSWAP)
void apply_cswap(Statevector& sv, int ctrl, int q1, int q2);

// Margolus (RCCX)
void apply_rccx(Statevector& sv, int c1, int c2, int tgt);

// =============================================================================
// Arbitrary N-qubit unitary
// =============================================================================

void apply_unitary(
    Statevector& sv,
    const std::vector<int>& targets,
    const std::vector<Complex128>& matrix  // row-major, 2^k × 2^k
);

// =============================================================================
// Multi-controlled and permutation operations (structured; no dense matrix)
// =============================================================================

// Multi-controlled X: flip `target` on every amplitude whose control qubits
// are all |1>. Any number of controls (0 == plain X). O(dim), disjoint pairs.
void apply_mcx(Statevector& sv, const std::vector<int>& controls,
               int target);

// Multi-controlled phase: multiply by exp(i*lambda) on every amplitude whose
// listed qubits are all |1> (symmetric controls). O(dim).
void apply_mcp(Statevector& sv, const std::vector<int>& qubits,
               double lambda);

// Basis permutation on target qubits: |x> -> |perm[x]> within the 2^k target
// subspace (LSB = qubits[0]). perm must be a bijection of [0, 2^k). O(dim).
void apply_permutation(Statevector& sv, const std::vector<int>& qubits,
                       const std::vector<int>& perm);

} // namespace gates
} // namespace lindblad
