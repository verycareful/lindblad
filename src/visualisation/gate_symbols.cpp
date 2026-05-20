// =============================================================================
// src/visualisation/gate_symbols.cpp : Tier 1 single-qubit box gate catalogue
// =============================================================================
// Definitions for the Tier 1 catalogue declared in gate_symbols.hpp. One entry
// per single-qubit "box" gate listed in the design spec section 6.1, plus the
// PARAM_* symbolic variants. The catalogue is the seam a maintainer edits when
// adding or restyling a box gate: layout, struts, and every backend renderer
// remain untouched.
//
// Colour palette (matches spec section 6.1 example block):
//   Hadamard family   (H)              : blue   fill #e8eef9, stroke #2a4a8f
//   Pauli family      (X, Y, Z)        : red    fill #fce8e8, stroke #8f2a2a
//   Phase / clock     (S, SDG, T, TDG) : yellow fill #fff4d8, stroke #8f6e2a
//   sqrt(X) family    (SX, SXDG)       : purple fill #f1e8fc, stroke #6e2a8f
//   Rotation / param  (RX, RY, RZ, P,  : green  fill #e0f0e0, stroke #2a6e2a
//                      U,  U1, U2, U3,
//                      PARAM_*)
//
// latex_macro is filled in only when the default "\gate{label}" assembly would
// look wrong (dagger gates, axis-subscripted rotations, multi-symbol labels).
// An empty latex_macro signals the renderer to fall back to "\gate{label}".

#include "gate_symbols.hpp"

namespace lindblad::viz {

// Build the static catalogue once on first call. Construction order mirrors
// the enum declaration in include/lindblad/circuit.hpp so the table reads top
// to bottom in the same order a maintainer encounters the gates elsewhere.
const std::unordered_map<Instruction::GateType, GateSymbol>& symbol_catalogue() {
    using GT = Instruction::GateType;
    static const std::unordered_map<GT, GateSymbol> table = {
        // Hadamard
        { GT::H,    { "H",    false, "#e8eef9", "#2a4a8f", ""                       } },

        // Pauli family
        { GT::X,    { "X",    false, "#fce8e8", "#8f2a2a", ""                       } },
        { GT::Y,    { "Y",    false, "#fce8e8", "#8f2a2a", ""                       } },
        { GT::Z,    { "Z",    false, "#fce8e8", "#8f2a2a", ""                       } },

        // Phase / clock family (S, T, and their daggers)
        { GT::S,    { "S",    false, "#fff4d8", "#8f6e2a", ""                       } },
        { GT::SDG,  { "S†", false, "#fff4d8", "#8f6e2a", "\\gate{S^{\\dagger}}" } },
        { GT::T,    { "T",    false, "#fff4d8", "#8f6e2a", ""                       } },
        { GT::TDG,  { "T†", false, "#fff4d8", "#8f6e2a", "\\gate{T^{\\dagger}}" } },

        // sqrt(X) family
        { GT::SX,   { "SX",   false, "#f1e8fc", "#6e2a8f", "\\gate{\\sqrt{X}}"      } },
        { GT::SXDG, { "SX†", false, "#f1e8fc", "#6e2a8f", "\\gate{\\sqrt{X}^{\\dagger}}" } },

        // Numeric rotations and phase / U family. show_params = true so the
        // label assembler appends "(theta)" / "(theta, phi, lambda)" when the
        // caller has not disabled show_params via DrawOptions.
        { GT::RX,   { "RX",   true,  "#e0f0e0", "#2a6e2a", "\\gate{R_X}"            } },
        { GT::RY,   { "RY",   true,  "#e0f0e0", "#2a6e2a", "\\gate{R_Y}"            } },
        { GT::RZ,   { "RZ",   true,  "#e0f0e0", "#2a6e2a", "\\gate{R_Z}"            } },
        { GT::P,    { "P",    true,  "#e0f0e0", "#2a6e2a", ""                       } },
        { GT::U,    { "U",    true,  "#e0f0e0", "#2a6e2a", ""                       } },
        { GT::U1,   { "U1",   true,  "#e0f0e0", "#2a6e2a", "\\gate{U_1}"            } },
        { GT::U2,   { "U2",   true,  "#e0f0e0", "#2a6e2a", "\\gate{U_2}"            } },
        { GT::U3,   { "U3",   true,  "#e0f0e0", "#2a6e2a", "\\gate{U_3}"            } },

        // Symbolic (QASM 3 named-parameter) variants. Same colour and label
        // family as their numeric siblings; the formatter renders any embedded
        // ParamExpr in place of the numeric value.
        { GT::PARAM_RX, { "RX", true, "#e0f0e0", "#2a6e2a", "\\gate{R_X}" } },
        { GT::PARAM_RY, { "RY", true, "#e0f0e0", "#2a6e2a", "\\gate{R_Y}" } },
        { GT::PARAM_RZ, { "RZ", true, "#e0f0e0", "#2a6e2a", "\\gate{R_Z}" } },
        { GT::PARAM_P,  { "P",  true, "#e0f0e0", "#2a6e2a", ""             } },
        { GT::PARAM_U,  { "U",  true, "#e0f0e0", "#2a6e2a", ""             } },
    };
    return table;
}

} // namespace lindblad::viz
