#pragma once

// =============================================================================
// src/visualisation/composite_catalogue.hpp : Tier 2 multi-bullet gate catalogue
// =============================================================================
// Declares the CompositePart / CompositeGate structs and the
// composite_catalogue() accessor. Covers the controlled and multi-bullet
// gates that decompose into a fixed pattern of slot/role pairs across
// inst.qubits[]: CX CY CZ CH SWAP ISWAP CRX CRY CRZ CP CU CCX CCZ CSWAP RCCX
// plus the two-qubit interaction gates RXX RYY RZZ RZX ECR that render as a
// single tall labelled box.
//
// build_composite_glyph(inst, rule) walks rule.parts and emits one GlyphPart
// per entry, placed at inst.qubits[part.qubit_slot]. When rule.draw_strut is
// true the strut spans [min..max] of the placed rows. TallBox is the special
// case that emits a single BoxPart with rowspan = N at the topmost slot's
// row and draws no strut (the box itself spans the rows).
//
// Definitions live in composite_catalogue.cpp (added in a later task).

#include "lindblad/circuit.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad::viz {

// CompositePart : one slot/role entry inside a composite rule.
// qubit_slot = index into inst.qubits (0 = first listed qubit)
// role       = which visual primitive to emit at the resolved row
struct CompositePart {
    int qubit_slot = 0;
    enum class Role {
        CtrlBullet,  // filled control disc
        AntiCtrl,    // open anti-control ring
        XorTarget,   // CNOT-style target circle with cross
        SwapX,       // SWAP arm marker
        Box,         // single-wire labelled box
        TallBox      // tall labelled box that spans every slot in the rule
    } role = Role::CtrlBullet;
};

// CompositeGate : the full per-gate render rule.
// parts            = ordered slot/role list driving the builder
// draw_strut       = emit a vertical connector across [min..max] of placed rows
// box_label        = label used by Box and TallBox roles (e.g. "Y", "RX", "iSwap")
// box_show_params  = append numeric params to the box label when true
struct CompositeGate {
    std::vector<CompositePart> parts;
    bool        draw_strut      = true;
    std::string box_label       = "";
    bool        box_show_params = false;
};

// Returns the static catalogue (built once on first call, immutable
// thereafter). Lookup miss => the gate is not a composite and falls through
// to the Tier 1 box-gate catalogue in the dispatcher.
const std::unordered_map<Instruction::GateType, CompositeGate>& composite_catalogue();

} // namespace lindblad::viz
