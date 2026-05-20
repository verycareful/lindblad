#pragma once

// =============================================================================
// src/visualisation/gate_builders.hpp : Tier 3 hand-written gate builders
// =============================================================================
// Builders for gates whose visual structure does not fit the Tier 1 (single
// box) or Tier 2 (slot/role composite) catalogues. Four cases live here:
//
//   BARRIER : one BarrierPart per touched qubit (or every qubit when the
//             instruction's qubit list is empty). No strut.
//   MEASURE : MeasurePart on inst.qubits[0]; when show_clbits is true and the
//             c-wire row exists, a strut drops from the qubit row to the
//             c-wire row at the bottom of the document.
//   RESET   : ResetPart on inst.qubits[0]; no strut.
//   UNITARY : a single tall BoxPart spanning rows [min(qubits)..max(qubits)],
//             labelled with inst.label. The box itself spans the rows, so no
//             strut is drawn. Non-contiguous qubit sets inflate the visual
//             row span beyond the actual operand count; this is the documented
//             behaviour in docs/api/visualisation.md.
//
// Each builder is a thin function that consumes an Instruction plus the
// caller's DrawOptions and returns a complete Glyph (the column index is set
// by the layout pass, not by the builder). Definitions live in
// gate_builders.cpp (added in a later task).

#include "document.hpp"
#include "lindblad/circuit.hpp"

namespace lindblad::viz {

// Build the glyph for an Instruction::GateType::BARRIER.
// inst = the barrier instruction (may have empty qubits => full-width)
// opts = caller options (currently unused; reserved for future styling)
Glyph build_barrier_glyph(const Instruction& inst, const DrawOptions& opts);

// Build the glyph for an Instruction::GateType::MEASURE.
// inst = the measure instruction; inst.qubits[0] is the measured qubit,
//        inst.clbits[0] is the destination classical bit
// opts = caller options; opts.show_clbits drives strut emission
Glyph build_measure_glyph(const Instruction& inst, const DrawOptions& opts);

// Build the glyph for an Instruction::GateType::RESET.
// inst = the reset instruction; inst.qubits[0] is the reset target
// opts = caller options (currently unused)
Glyph build_reset_glyph(const Instruction& inst, const DrawOptions& opts);

// Build the glyph for an Instruction::GateType::UNITARY.
// inst = the unitary instruction; inst.label labels the tall box and
//        [min(qubits)..max(qubits)] sets the row span
// opts = caller options (currently unused; reserved for future styling)
Glyph build_unitary_glyph(const Instruction& inst, const DrawOptions& opts);

} // namespace lindblad::viz
