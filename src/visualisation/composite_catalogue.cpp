// =============================================================================
// src/visualisation/composite_catalogue.cpp : Tier 2 multi-bullet gate catalogue
// =============================================================================
// Definitions for the Tier 2 catalogue declared in composite_catalogue.hpp.
// Covers controlled and multi-bullet gates whose visual decomposes into a
// fixed pattern of slot/role pairs across inst.qubits[]:
//
//   CX  CY  CZ  CH  SWAP  ISWAP  CRX  CRY  CRZ  CP  CU
//   CCX CCZ CSWAP RCCX
//   RXX RYY RZZ RZX ECR    (TallBox: a single labelled box spanning two wires)
//
// Layout notes:
//   - CZ and CCZ use CtrlBullet on every involved qubit because the phase
//     action is symmetric (no "dot target" primitive in the GlyphPart variant).
//   - The two-qubit interaction gates (RXX / RYY / RZZ / RZX / ECR) set
//     draw_strut = false because the TallBox itself spans the rows visually;
//     a separate strut would draw a redundant line through the box.
//   - box_show_params drives whether numeric / symbolic params get appended
//     to the box label by the builder (e.g. CRX prints "RX(pi/2)", but CY
//     prints just "Y").

#include "composite_catalogue.hpp"

namespace lindblad::viz {

// Build the static catalogue once on first call. The ordering follows the
// enum declaration in include/lindblad/circuit.hpp so a maintainer encounters
// gates here in the same order they appear elsewhere in the codebase.
const std::unordered_map<Instruction::GateType, CompositeGate>& composite_catalogue() {
    using GT = Instruction::GateType;
    using R  = CompositePart::Role;
    static const std::unordered_map<GT, CompositeGate> table = {
        // Controlled Paulis. CX is the canonical CNOT; CY and CZ use the
        // symmetric / boxed variants per the spec example block.
        { GT::CX,    { { {0, R::CtrlBullet}, {1, R::XorTarget } }, true,  "",     false } },
        { GT::CY,    { { {0, R::CtrlBullet}, {1, R::Box       } }, true,  "Y",    false } },
        { GT::CZ,    { { {0, R::CtrlBullet}, {1, R::CtrlBullet} }, true,  "",     false } },

        // Controlled Hadamard.
        { GT::CH,    { { {0, R::CtrlBullet}, {1, R::Box       } }, true,  "H",    false } },

        // SWAP family. SWAP draws an "x" on each arm. iSWAP renders as a pair
        // of labelled boxes connected by a strut (visual: two-wire phase gate).
        { GT::SWAP,  { { {0, R::SwapX     }, {1, R::SwapX     } }, true,  "",     false } },
        { GT::ISWAP, { { {0, R::Box       }, {1, R::Box       } }, true,  "iSwap", false } },

        // Controlled rotations and controlled phase / U.
        { GT::CRX,   { { {0, R::CtrlBullet}, {1, R::Box       } }, true,  "RX",   true  } },
        { GT::CRY,   { { {0, R::CtrlBullet}, {1, R::Box       } }, true,  "RY",   true  } },
        { GT::CRZ,   { { {0, R::CtrlBullet}, {1, R::Box       } }, true,  "RZ",   true  } },
        { GT::CP,    { { {0, R::CtrlBullet}, {1, R::Box       } }, true,  "P",    true  } },
        { GT::CU,    { { {0, R::CtrlBullet}, {1, R::Box       } }, true,  "U",    true  } },

        // Three-qubit controlled / multi-bullet gates.
        { GT::CCX,   { { {0, R::CtrlBullet}, {1, R::CtrlBullet}, {2, R::XorTarget } }, true, "", false } },
        { GT::CCZ,   { { {0, R::CtrlBullet}, {1, R::CtrlBullet}, {2, R::CtrlBullet} }, true, "", false } },
        { GT::CSWAP, { { {0, R::CtrlBullet}, {1, R::SwapX     }, {2, R::SwapX     } }, true, "", false } },
        { GT::RCCX,  { { {0, R::CtrlBullet}, {1, R::CtrlBullet}, {2, R::XorTarget } }, true, "", false } },

        // Two-qubit interaction gates: a single tall labelled box spanning
        // both wires. draw_strut = false because the box itself spans the
        // rows (a strut would draw a duplicate line through the box).
        { GT::RXX,   { { {0, R::TallBox}, {1, R::TallBox} }, false, "RXX", true  } },
        { GT::RYY,   { { {0, R::TallBox}, {1, R::TallBox} }, false, "RYY", true  } },
        { GT::RZZ,   { { {0, R::TallBox}, {1, R::TallBox} }, false, "RZZ", true  } },
        { GT::RZX,   { { {0, R::TallBox}, {1, R::TallBox} }, false, "RZX", true  } },
        { GT::ECR,   { { {0, R::TallBox}, {1, R::TallBox} }, false, "ECR", false } },
    };
    return table;
}

} // namespace lindblad::viz
