// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/detail/preflight.hpp"
#include "lindblad/detail/report.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace lindblad {
namespace detail {

namespace {

using GT = Instruction::GateType;

FailurePoint point_of(const Instruction& inst, std::size_t index) {
    FailurePoint p;
    p.instruction = static_cast<int>(index);
    p.gate = inst.gate_name();
    p.qubits = inst.qubits;
    return p;
}

bool is_unbound(const Instruction& inst) {
    return inst.is_parameterised() || !inst.param_exprs.empty();
}

// What a gate type reads from its instruction: every kernel indexes qubits[0..]
// and params[0..] by type without checking, so a shorter instruction is read
// out of bounds and a longer one has operands silently ignored.
//   qubits = the exact operand count; AT_LEAST_ONE for the variable-width
//            types, ANY for a BARRIER
//   params = the fewest parameters the type reads
struct Shape {
    static constexpr int AT_LEAST_ONE = -1;
    static constexpr int ANY = -2;
    int qubits;
    int params;
};

Shape shape_of(GT type) {
    switch (type) {
        case GT::H: case GT::X: case GT::Y: case GT::Z: case GT::S: case GT::SDG:
        case GT::T: case GT::TDG: case GT::SX: case GT::SXDG:
            return {1, 0};
        case GT::RX: case GT::RY: case GT::RZ: case GT::P: case GT::U1:
            return {1, 1};
        case GT::U2:
            return {1, 2};
        case GT::U: case GT::U3:
            return {1, 3};
        case GT::CX: case GT::CY: case GT::CZ: case GT::CH: case GT::SWAP:
        case GT::ISWAP: case GT::ECR:
            return {2, 0};
        case GT::CRX: case GT::CRY: case GT::CRZ: case GT::CP: case GT::RZX:
        case GT::RXX: case GT::RYY: case GT::RZZ:
            return {2, 1};
        case GT::CU:
            return {2, 4};
        case GT::CCX: case GT::CCZ: case GT::CSWAP: case GT::RCCX:
            return {3, 0};
        case GT::MEASURE: case GT::RESET:
        case GT::UNITARY: case GT::MCX: case GT::PERMUTATION:
            return {Shape::AT_LEAST_ONE, 0};
        case GT::MCP:
            return {Shape::AT_LEAST_ONE, 1};
        case GT::BARRIER:
            return {Shape::ANY, 0};
        case GT::PARAM_RX: case GT::PARAM_RY: case GT::PARAM_RZ: case GT::PARAM_P:
        case GT::PARAM_U:
            return {Shape::ANY, 0};  // refused as unbound before this is read
    }
    return {Shape::ANY, 0};
}

// The widest a UNITARY or a PERMUTATION can be and still be indexed: a
// matrix's (2^k)^2 entries fit a size_t up to k = 31, and a map's images are
// ints, which address [0, 2^31).
constexpr std::size_t MAX_STRUCTURED_WIDTH = 31;

std::string plural(std::size_t n, const char* noun) {
    return std::to_string(n) + " " + noun + (n == 1 ? "" : "s");
}

}  // namespace

void preflight_instructions(
    const QuantumCircuit& circuit, const char* entry_point,
    const std::function<std::string(const Instruction&, int)>& backend_check) {
    for (std::size_t i = 0; i < circuit.instructions.size(); ++i) {
        const Instruction& inst = circuit.instructions[i];

        if (is_unbound(inst)) {
            raise<InvalidArgument>(entry_point,
                "the gate has an unbound parameter; call assign_parameters() first",
                point_of(inst, i));
        }

        const Shape shape = shape_of(inst.type);
        const std::size_t k = inst.qubits.size();
        if (shape.qubits >= 0 && k != static_cast<std::size_t>(shape.qubits)) {
            raise<InvalidArgument>(entry_point,
                "the gate acts on " + plural(static_cast<std::size_t>(shape.qubits), "qubit") +
                    "; it names " + std::to_string(k),
                point_of(inst, i));
        }
        if (shape.qubits == Shape::AT_LEAST_ONE && k == 0) {
            raise<InvalidArgument>(entry_point, "the gate names no qubits", point_of(inst, i));
        }
        if (inst.type != GT::BARRIER) {
            for (std::size_t a = 0; a < k; ++a) {
                for (std::size_t b = a + 1; b < k; ++b) {
                    if (inst.qubits[a] == inst.qubits[b]) {
                        raise<InvalidArgument>(entry_point,
                            "qubit " + std::to_string(inst.qubits[a]) +
                                " is named twice; a gate's operands must be distinct",
                            point_of(inst, i));
                    }
                }
            }
        }
        if (inst.params.size() < static_cast<std::size_t>(shape.params)) {
            raise<InvalidArgument>(entry_point,
                "the gate reads " + plural(static_cast<std::size_t>(shape.params), "parameter") +
                    "; it carries " + std::to_string(inst.params.size()),
                point_of(inst, i));
        }

        for (std::size_t j = 0; j < inst.params.size(); ++j) {
            if (!is_finite_strict(inst.params[j])) {
                raise<InvalidArgument>(entry_point,
                    "parameter " + std::to_string(j) + " is " +
                        std::to_string(inst.params[j]) + "; every parameter must be finite",
                    point_of(inst, i));
            }
        }

        if (inst.type == GT::UNITARY) {
            if (k > MAX_STRUCTURED_WIDTH) {
                raise<InvalidArgument>(entry_point,
                    "the UNITARY acts on " + std::to_string(k) +
                        " qubits, wider than any matrix can be indexed",
                    point_of(inst, i));
            }
            const std::size_t rows = std::size_t{1} << k;
            if (inst.matrix.size() != rows * rows) {
                raise<InvalidArgument>(entry_point,
                    "matrix size mismatch: the UNITARY acts on " + plural(k, "qubit") +
                        ", so its matrix must have " +
                        std::to_string(rows * rows) + " entries; it has " +
                        std::to_string(inst.matrix.size()),
                    point_of(inst, i));
            }
        }

        if (inst.type == GT::PERMUTATION) {
            if (k > MAX_STRUCTURED_WIDTH) {
                raise<InvalidArgument>(entry_point,
                    "the PERMUTATION acts on " + std::to_string(k) +
                        " qubits, wider than any map can be indexed",
                    point_of(inst, i));
            }
            const std::size_t dim = std::size_t{1} << k;
            if (inst.permutation.size() != dim) {
                raise<InvalidArgument>(entry_point,
                    "the PERMUTATION acts on " + plural(k, "qubit") + ", so its map must have " +
                        std::to_string(dim) + " entries; it has " +
                        std::to_string(inst.permutation.size()),
                    point_of(inst, i));
            }
            std::vector<char> seen(dim, 0);
            for (const int image : inst.permutation) {
                if (image < 0 || static_cast<std::size_t>(image) >= dim) {
                    raise<InvalidArgument>(entry_point,
                        "the PERMUTATION's map sends a state to " + std::to_string(image) +
                            ", outside [0, " + std::to_string(dim) + ")",
                        point_of(inst, i));
                }
                if (seen[static_cast<std::size_t>(image)]) {
                    raise<InvalidArgument>(entry_point,
                        "the PERMUTATION's map sends two states to " + std::to_string(image) +
                            ", so it is not a bijection",
                        point_of(inst, i));
                }
                seen[static_cast<std::size_t>(image)] = 1;
            }
        }

        if (backend_check) {
            const std::string reason = backend_check(inst, circuit.n_qubits);
            if (!reason.empty()) raise<InvalidArgument>(entry_point, reason, point_of(inst, i));
        }
    }
}

}  // namespace detail
}  // namespace lindblad
