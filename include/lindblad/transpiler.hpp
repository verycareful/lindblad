#pragma once

#include "lindblad/dag.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/types.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lindblad {

// Forward declaration
struct BackendProperties;

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

// Layout passes
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

// Basis translation
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

// Preset pass managers
PassManager preset_pass_manager(
    int optimization_level,
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates
);

// Convenience: transpile a circuit directly
QuantumCircuit transpile(
    const QuantumCircuit& circuit,
    const CouplingMap& coupling_map = CouplingMap(),
    const std::vector<std::string>& basis_gates = {},
    int optimization_level = 1
);

} // namespace lindblad
