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
// emitted instruction. A conditional operand lowers like any other: every gate
// in the sequence carries the same condition, so the block fires or does not
// fire as a whole.
//
// The operand must BE a UNITARY carrying sixteen entries. The matrix is read
// from `Instruction::matrix` rather than reconstructed, so a gate whose 4x4 is
// implied by its type rather than stored has nothing here to read.
//
// nullopt on anything else, and on a decomposition that does not verify against
// the operand. Callers must treat that as "cannot lower" rather than falling
// back to anything approximate.
//
// Conditioning each emitted instruction is equivalent to conditioning the
// sequence as a block: the emitted alphabet is quantum gates only, none of
// which writes a classical bit, so the condition cannot change between the
// first instruction and the last.
std::optional<Lowered> lower_2q_unitary(const Instruction& inst);

}  // namespace tqd
}  // namespace lindblad
