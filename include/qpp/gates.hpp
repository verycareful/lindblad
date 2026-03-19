#pragma once

#include "qpp/statevector.hpp"
#include "qpp/types.hpp"

#include <vector>

namespace qpp {
namespace gates {

// =============================================================================
// Single-qubit gates
// =============================================================================

// Pauli gates
void apply_x(Statevector& sv, int q) noexcept;
void apply_y(Statevector& sv, int q) noexcept;
void apply_z(Statevector& sv, int q) noexcept;

// Hadamard
void apply_h(Statevector& sv, int q) noexcept;

// Phase gates
void apply_s(Statevector& sv, int q) noexcept;
void apply_sdg(Statevector& sv, int q) noexcept;
void apply_t(Statevector& sv, int q) noexcept;
void apply_tdg(Statevector& sv, int q) noexcept;

// Sqrt X gates
void apply_sx(Statevector& sv, int q) noexcept;
void apply_sxdg(Statevector& sv, int q) noexcept;

// Parameterised rotation gates
void apply_rx(Statevector& sv, int q, double theta) noexcept;
void apply_ry(Statevector& sv, int q, double theta) noexcept;
void apply_rz(Statevector& sv, int q, double theta) noexcept;

// Phase gate P(lambda) = diag(1, exp(i*lambda))
void apply_p(Statevector& sv, int q, double lambda) noexcept;

// General single-qubit unitary U(theta, phi, lambda)
void apply_u(Statevector& sv, int q,
             double theta, double phi, double lambda) noexcept;
void apply_u1(Statevector& sv, int q, double lambda) noexcept;
void apply_u2(Statevector& sv, int q, double phi, double lambda) noexcept;
void apply_u3(Statevector& sv, int q,
              double theta, double phi, double lambda) noexcept;

// =============================================================================
// Two-qubit gates
// =============================================================================

// Controlled NOT (CNOT)
void apply_cx(Statevector& sv, int ctrl, int tgt) noexcept;

// Controlled Y
void apply_cy(Statevector& sv, int ctrl, int tgt) noexcept;

// Controlled Z
void apply_cz(Statevector& sv, int ctrl, int tgt) noexcept;

// Controlled H
void apply_ch(Statevector& sv, int ctrl, int tgt) noexcept;

// SWAP
void apply_swap(Statevector& sv, int q1, int q2) noexcept;

// iSWAP
void apply_iswap(Statevector& sv, int q1, int q2) noexcept;

// Controlled rotation gates
void apply_crx(Statevector& sv, int ctrl, int tgt, double theta) noexcept;
void apply_cry(Statevector& sv, int ctrl, int tgt, double theta) noexcept;
void apply_crz(Statevector& sv, int ctrl, int tgt, double theta) noexcept;

// Controlled phase
void apply_cp(Statevector& sv, int ctrl, int tgt, double lambda) noexcept;

// Controlled U
void apply_cu(Statevector& sv, int ctrl, int tgt,
              double theta, double phi, double lambda, double gamma) noexcept;

// Echoed cross-resonance
void apply_ecr(Statevector& sv, int q1, int q2) noexcept;

// Ising interaction gates
void apply_rzx(Statevector& sv, int q1, int q2, double theta) noexcept;
void apply_rxx(Statevector& sv, int q1, int q2, double theta) noexcept;
void apply_ryy(Statevector& sv, int q1, int q2, double theta) noexcept;
void apply_rzz(Statevector& sv, int q1, int q2, double theta) noexcept;

// =============================================================================
// Three-qubit gates
// =============================================================================

// Toffoli (CCX)
void apply_ccx(Statevector& sv, int c1, int c2, int tgt) noexcept;

// CCZ
void apply_ccz(Statevector& sv, int c1, int c2, int tgt) noexcept;

// Fredkin (CSWAP)
void apply_cswap(Statevector& sv, int ctrl, int q1, int q2) noexcept;

// Margolus (RCCX)
void apply_rccx(Statevector& sv, int c1, int c2, int tgt) noexcept;

// =============================================================================
// Arbitrary N-qubit unitary
// =============================================================================

void apply_unitary(
    Statevector& sv,
    const std::vector<int>& targets,
    const std::vector<Complex128>& matrix  // row-major, 2^k × 2^k
);

} // namespace gates
} // namespace qpp
