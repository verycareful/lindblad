// high_level_decompose.hpp — exact gate-level lowering of MCX / MCP / PERMUTATION
//
// INTERNAL header (src/, not include/): shared between the HighLevelDecompose
// transpiler pass and the QASM exporters in circuit.cpp. The public surface is
// the pass itself (include/lindblad/transpiler.hpp) and the exporter options.
//
// Every function returns an EXACT realization of the high-level instruction as
// a sequence of standard instructions; no approximation is introduced anywhere
// in this module (the only non-integer arithmetic is the recursive halving of
// phase angles, λ/2^j, which is exact binary scaling).
//
// Output gate alphabet:
//   lower_mcx / lower_mcp        → { X, H, P, CP, CX, CCX }
//   lower_permutation            → { X, CX, SWAP, MCX }   (one level: MCX kept
//                                  high-level so QASM 3 can emit it compactly
//                                  via `ctrl @`; callers needing base gates
//                                  flatten with lower_fully)
//   lower_fully                  → { X, H, P, CP, CX, CCX, SWAP }
//   lower_ccx                    → { H, P, CX }            (the exact 6-CX
//                                  T-ladder; used by the pass to floor its
//                                  output at 2-qubit gates under a constrained
//                                  coupling map, where routing cannot place
//                                  any 3-qubit gate on triangle-free targets)
//
// All emitted instructions inherit the source instruction's classical
// condition (condition_clbit / condition_value): a conditioned high-level op
// lowers to a conditioned gate block, preserving feedforward semantics.
//
// Constructions (all ancilla-free, self-contained on the instruction's own
// qubits; complexity notes in the .cpp):
//   MCP  — Lemma-7.5-style λ/2 recursion (Barenco et al. 1995):
//          MCP(λ; q1..qm) = CP(λ/2)(q_{m-1}, q_m)
//                         · MCX(q1..q_{m-2} → q_{m-1})
//                         · CP(-λ/2)(q_{m-1}, q_m)
//                         · MCX(q1..q_{m-2} → q_{m-1})
//                         · MCP(λ/2; q1..q_{m-2}, q_m)
//   MCX  — k ≤ 2 direct (X / CX / CCX); k ≥ 3 via H(t) · MCP(π; C ∪ {t}) · H(t)
//          with the inner MCXs lowered by borrowed-wire halving (one dirty
//          borrowed qubit is always available inside the recursion).
//   PERM — qubit-relabel maps (perm[x] a pure bit permutation) → ≤ k-1 SWAPs;
//          general basis maps → cycle decomposition into transpositions, each
//          transposition = CX fan-in + pattern-controlled MCX + CX fan-out.

#pragma once

#include "lindblad/circuit.hpp"

#include <optional>
#include <vector>

namespace lindblad::hld {

// True iff `inst` is one of the three high-level ops this module lowers.
bool is_high_level(const Instruction& inst);

// MCX: controls may be empty (plain X). Throws std::invalid_argument on
// duplicate qubits or control == target (mirrors QuantumCircuit::mcx).
std::vector<Instruction> lower_mcx(const std::vector<int>& controls, int target);

// MCP: symmetric phase λ on the all-ones subspace of `qubits` (≥ 1 qubit).
std::vector<Instruction> lower_mcp(double lambda, const std::vector<int>& qubits);

// CCX: the standard exact 15-gate / 6-CX T-ladder (T = P(π/4)), equal to the
// Toffoli WITHOUT global-phase slack. Alphabet { H, P, CX }: routable on any
// connected coupling map. Throws std::invalid_argument on duplicate operands.
std::vector<Instruction> lower_ccx(int a, int b, int target);

// PERMUTATION: `perm` is the basis-index map (size 2^k, LSB = qubits[0]).
// Validates that perm is a bijection over [0, 2^k) and throws
// std::invalid_argument otherwise (from_json can construct unvalidated
// instructions, so the check cannot be skipped here). Output may contain MCX.
std::vector<Instruction> lower_permutation(const std::vector<int>& perm,
                                           const std::vector<int>& qubits);

// If `perm` is induced by a relabeling of the k qubit wires, return that wire
// permutation σ (bit j of the input moves to bit σ[j] of the image), else
// std::nullopt. Used by lower_permutation for the cheap SWAP-network path and
// exposed for the exporters' cost decisions.
std::optional<std::vector<int>> as_qubit_relabel(const std::vector<int>& perm);

// Dispatch on inst.type and return the FULLY lowered base-gate sequence
// ({ X, H, P, CP, CX, CCX, SWAP }), conditions propagated. Non-high-level
// instructions return a single-element copy of `inst` unchanged.
std::vector<Instruction> lower_fully(const Instruction& inst);

} // namespace lindblad::hld
