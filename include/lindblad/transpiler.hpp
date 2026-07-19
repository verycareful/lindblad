#pragma once

#include "lindblad/dag.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/types.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lindblad {

// =============================================================================
// CouplingMap
// =============================================================================

class CouplingMap {
public:
    int n_physical_qubits;
    std::vector<std::pair<int,int>> edges;  // directed edges

    CouplingMap() : n_physical_qubits(0) {}
    explicit CouplingMap(int n) : n_physical_qubits(n) {}

    bool is_connected(int q1, int q2) const;
    std::vector<int> shortest_path(int q1, int q2) const;
    std::vector<std::vector<int>> distance_matrix() const;
    bool is_connected_graph() const;

    static CouplingMap linear(int n);
    static CouplingMap grid(int rows, int cols);
    static CouplingMap heavy_hex(int n);
    static CouplingMap all_to_all(int n);
};

// =============================================================================
// TranspilationContext
// =============================================================================

struct TranspilationContext {
    CouplingMap coupling_map;
    std::vector<std::string> basis_gates;
    std::vector<int> initial_layout;  // logical -> physical qubit mapping
    int optimization_level = 1;
};

// =============================================================================
// TranspilationPass interface
// =============================================================================

class TranspilationPass {
public:
    virtual DAGCircuit run(
        const DAGCircuit& dag,
        const TranspilationContext& ctx
    ) const = 0;
    virtual std::string name() const = 0;
    virtual ~TranspilationPass() = default;
};

// High-level decomposition (R.1.18.0) — pre-routing stage-0 pass.
//
// Lowers the three high-level instructions to routable, translatable gates:
// MCX / MCP to { X, H, P, CP, CX, CCX } (ancilla-free, exact) and PERMUTATION
// to a SWAP network when the basis map is a pure wire relabeling, otherwise to
// an exact transposition synthesis. Classical conditions are propagated onto
// every emitted gate. Instructions other than the three pass through
// unchanged; a circuit containing none of them is returned as-is.
//
// Coupling-map floor (R.1.18.2): when ctx.coupling_map is constrained
// (n_physical_qubits > 0) the output additionally contains NO gate wider than
// two qubits — every CCX, whether produced by the lowering or written by the
// user, is flattened into the exact 6-CX T-ladder ({ H, P, CX }). Routing
// executes a 3-qubit gate only when all three wire pairs are adjacent, which
// triangle-free targets (path, grid, heavy-hex) never provide; the 2-qubit
// floor routes on any connected map. Unconstrained targets keep CCX, so a
// ccx-bearing basis stays native.
//
// Composition rule (preset pipelines): composed only when the transpile
// target actually requires the lowering — a constrained coupling map (routing
// handles at most 3-qubit gates) or a non-empty basis_gates list (the
// equivalence library cannot reach the three ops). An unconstrained,
// basis-free transpile() keeps them native: the backends execute MCX / MCP /
// PERMUTATION directly, and unconditional lowering would only pessimize.
// Compose the pass manually for unconditional lowering.
class HighLevelDecompose : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "HighLevelDecompose"; }
};

// Layout passes
//
// Shared output invariant (R.1.15.0): when a coupling map is present, both
// passes return a DAG with n_qubits == coupling_map.n_physical_qubits — every
// physical slot holds a (possibly idle) logical wire, so circuits smaller
// than the device stay routable (idle-wire SWAPs are legal). TrivialLayout is
// the identity embedding; SabreLayout additionally remaps wires to the
// SABRE-chosen layout (Li et al. 2019). Circuits with more qubits than the
// device throw std::invalid_argument. With no coupling map (n_physical == 0),
// both passes return the DAG unchanged.
class TrivialLayout : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "TrivialLayout"; }
};

class SabreLayout : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "SabreLayout"; }
};

// Routing passes
class SabreSwap : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "SabreSwap"; }
};

class StochasticSwap : public TranspilationPass {
    int trials = 20;
    int seed = 42;
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "StochasticSwap"; }
};

// Basis translation — decompose gates into ctx.basis_gates (default: cx+u3).
// The output is VERIFIED (R.1.15.0): every returned gate is in the target
// basis, or std::invalid_argument is thrown naming the offending gate.
// Gates without a decomposition path (MCX/MCP/PERMUTATION when the stage-0
// HighLevelDecompose pass was not run first, UNITARY, unbound PARAM_* gates)
// previously passed through silently. Classical conditions are propagated
// onto every emitted gate.
class BasisTranslator : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "BasisTranslator"; }
};

// Optimization passes
class Optimize1qGates : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "Optimize1qGates"; }
};

class CXCancellation : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "CXCancellation"; }
};

class ConsolidateBlocks : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "ConsolidateBlocks"; }
};

// Commutative gate cancellation — merge/cancel rotation gates through commuting intermediates
class CommutativeCancellation : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "CommutativeCancellation"; }
};

// Remove diagonal gates (RZ, P, T, S, Z, U1) whose only successor is MEASURE
class RemoveDiagonalGatesBeforeMeasure : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "RemoveDiagonalGatesBeforeMeasure"; }
};

// Remove RESET on qubits already in |0⟩ (at circuit start or after a prior RESET)
class RemoveResetInZeroState : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "RemoveResetInZeroState"; }
};

// Scheduling passes — assign time slots to instructions
class ASAPSchedule : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "ASAPSchedule"; }
};

class ALAPSchedule : public TranspilationPass {
public:
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const override;
    std::string name() const override { return "ALAPSchedule"; }
};

// =============================================================================
// PassManager
// =============================================================================

class PassManager {
public:
    std::vector<std::unique_ptr<TranspilationPass>> passes;

    void append(std::unique_ptr<TranspilationPass> pass);
    DAGCircuit run(const DAGCircuit& dag, const TranspilationContext& ctx) const;
};

// Preset pass managers — per-STAGE composition (non-cumulative, R.1.15.0):
//   layout+routing   levels 0-1: TrivialLayout → SabreSwap
//                    levels 2-3: SabreLayout → SabreSwap  (one routing pass)
//   optimisation     level-dependent chain (see preset_pass_manager.cpp)
//   translation      BasisTranslator, composed iff basis_gates is non-empty;
//                    always the FINAL stage, so the basis contract holds on
//                    the returned circuit.
// optimization_level outside [0, 3] throws std::invalid_argument (level -1
// previously returned an empty manager = silent identity transpile).
// ctx.initial_layout is honoured at levels 0-1 only; leave it empty at
// levels >= 2 (SabreLayout chooses the layout).
PassManager preset_pass_manager(
    int optimization_level,
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates
);

// Convenience: transpile a circuit directly.
//   coupling_map — routing constraint. When present, the output circuit has
//     n_qubits == n_physical_qubits (device width; smaller circuits are
//     identity-embedded, larger ones throw std::invalid_argument).
//     CouplingMap() (n = 0) means unconstrained; an edgeless CouplingMap(n)
//     is literal (unroutable for any 2q gate: SABRE throws).
//   basis_gates — target basis. Non-empty: the output contains ONLY these
//     gates or std::invalid_argument is thrown (see BasisTranslator).
//     Empty: no basis translation. (Before R.1.15.0 this parameter was
//     silently ignored.)
//   optimization_level — 0..3; out of range throws std::invalid_argument.
QuantumCircuit transpile(
    const QuantumCircuit& circuit,
    const CouplingMap& coupling_map = CouplingMap(),
    const std::vector<std::string>& basis_gates = {},
    int optimization_level = 1
);

} // namespace lindblad
