// two_qubit_decompose.hpp — exact gate-level lowering of a 2-qubit UNITARY
//
// INTERNAL header (src/, not include/): shared between the ConsolidateBlocks
// transpiler pass, which owns the KAK machinery, and the QASM exporters in
// circuit.cpp, which need the same numerics. The seam exists so there is ONE
// implementation of the decomposition. A second copy could drift from this one
// without either looking wrong on its own, and agreement between two
// implementations proves only that they agree.
//
// Output gate alphabet: { U, RXX, RYY, RZZ }. The QASM 2 exporter emits all
// four and its parser reads all four, so the lowering round-trips within that
// dialect without adding gate support.
//
// EXACTNESS, and why the global phase comes back separately rather than folded
// in: the emitted sequence reproduces the operand only up to a global phase,
// because the single-qubit corrections are emitted as U(theta, phi, lambda),
// which spans SU(2) rather than U(2). The two export formats then diverge.
// QASM 3 has a first-class `gphase` and can restore the phase exactly. QASM 2
// has no representation for it at all, so the caller must decide whether to
// refuse or to record the loss. Handing the phase back lets each caller make
// that decision instead of having it made for them here.
//
// Returns nullopt when the decomposition does not verify against the operand,
// which callers must treat as "cannot lower" rather than falling back to
// anything approximate.

#pragma once

#include "lindblad/circuit.hpp"

#include <optional>
#include <vector>

namespace lindblad {
namespace tqd {

struct Lowered {
    std::vector<Instruction> instructions;

    // Radians. The operand equals exp(i * global_phase) times the operator of
    // `instructions`, so a format that can express a global phase restores the
    // operand exactly by emitting it, and a format that cannot loses precisely
    // this much and nothing else.
    double global_phase = 0.0;
};

// Lowers a 2-qubit UNITARY instruction onto its own operands, carrying the
// source instruction's classical condition and validation policy onto every
// emitted instruction. nullopt when `inst` is not a 2-qubit operand this
// module can represent, or when the decomposition fails verification.
//
// KNOWN GAP, pinned by R1221LoweringSeam.EveryEmittedInstructionCarriesConditionAndPolicy:
// a CONDITIONAL operand never gets this far. The 4x4 extraction it relies on
// declines any instruction carrying a classical condition, so the condition it
// promises to carry is currently unreachable and the caller reports the refusal
// as a numerical one. Neither exporter writes conditions into its text either,
// so accepting them here without fixing that would emit gates whose condition
// had silently vanished.
std::optional<Lowered> lower_2q_unitary(const Instruction& inst);

}  // namespace tqd
}  // namespace lindblad
