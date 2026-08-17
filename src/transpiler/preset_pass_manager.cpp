// preset_pass_manager.cpp — PassManager + preset pipeline composition + transpile()
//
// Pipeline composition lives here rather than in optimize_1q.cpp: it is not a
// 1-qubit-optimisation concern (SRP).
//
// Preset levels compose per STAGE, non-cumulatively:
//
//   stage 0  high-level decomposition
//                               HighLevelDecompose, composed at EVERY level
//                               iff the target needs it: constrained coupling
//                               map (routing handles at most 3q gates, and the
//                               pass floors its output at 2q there —
//                               triangle-free maps route nothing wider) OR
//                               non-empty basis_gates (the
//                               equivalence library cannot reach
//                               MCX/MCP/PERMUTATION). With neither, the ops
//                               stay native for the backends (lowering would
//                               only pessimize) — compose the pass manually
//                               for unconditional lowering.
//   stage 1  layout + routing   L0/L1: TrivialLayout → SabreSwap
//                               L2/L3: SabreLayout → SabreSwap
//   stage 2  optimisation       L0: none
//                               L1: RemoveResetInZeroState → Optimize1qGates
//                                   → CXCancellation
//                                   → RemoveDiagonalGatesBeforeMeasure
//                               L2: L1 chain with CommutativeCancellation
//                                   inserted before the diagonal removal
//                               L3: L2 chain, then ConsolidateBlocks (KAK)
//                                   plus a second cleanup sweep
//   stage 3  basis translation  BasisTranslator, composed iff basis_gates is
//                               non-empty; ALWAYS the final stage (see below)
//
// Per-stage composition means exactly one layout choice and exactly one routing
// pass at every level. Composing cumulatively instead would run a level-0
// TrivialLayout + SabreSwap block ahead of SabreLayout, leaving it to re-route
// an already-SWAP-laden circuit whose SWAPs it cannot undo.
//
// Why translation is LAST: Optimize1qGates emits U/RY/RZ and
// ConsolidateBlocks emits RXX/RYY/RZZ unconditionally — the optimisation
// passes are basis-oblivious, so any earlier slot lets the optimiser
// re-introduce non-basis gates after translation. Last is the only position
// where the "output contains only basis gates" guarantee holds
// unconditionally (BasisTranslator verifies it and throws otherwise).
// Routing legality is preserved because every BasisTranslator decomposition
// maps a gate to 1q gates + CX on the SAME qubit pair(s), which routing
// already made adjacent (3q gates survive routing only when every wire pair
// is adjacent). Basis-aware resynthesis for better post-translation gate
// counts is future work.
//
// initial_layout note: TranspilationContext::initial_layout is honoured by
// SabreSwap at levels 0-1 (TrivialLayout keeps logical indices). At levels
// >= 2 SabreLayout chooses the layout and returns a physically-indexed DAG
// on which the subsequent SabreSwap must run with an identity layout — leave
// initial_layout empty at those levels.

#include "lindblad/transpiler.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace lindblad {

// =============================================================================
// PassManager — ordered pass execution
// =============================================================================

void PassManager::append(std::unique_ptr<TranspilationPass> pass) {
    passes.push_back(std::move(pass));
}

DAGCircuit PassManager::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    DAGCircuit current = dag;
    for (const auto& pass : passes) {
        current = pass->run(current, ctx);
    }
    return current;
}

// =============================================================================
// preset_pass_manager — per-stage pipeline composition
// =============================================================================

PassManager preset_pass_manager(
    int optimization_level,
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates
) {
    // The coupling map is consumed here: together with basis_gates it decides
    // whether the stage-0 HighLevelDecompose pass is composed. basis_gates
    // additionally decides the translation stage. Every composed pass still
    // reads both from the TranspilationContext handed to run(); transpile()
    // passes the same values into this function and the context, so
    // composition and execution always agree.
    if (optimization_level < 0 || optimization_level > 3) {
        // Fail loud rather than resolve out-of-range levels silently: a
        // negative level must not yield an EMPTY manager (identity transpile),
        // and level >= 4 must not quietly behave as level 3.
        throw std::invalid_argument(
            "lindblad::preset_pass_manager: optimization_level " +
            std::to_string(optimization_level) +
            " is out of range (valid levels: 0..3)");
    }

    PassManager pm;

    // ---- Stage 0: high-level decomposition (composition rule in header) ----
    if (coupling_map.n_physical_qubits > 0 || !basis_gates.empty()) {
        pm.append(std::make_unique<HighLevelDecompose>());
    }

    // ---- Stage 1: layout + routing (one layout choice, one routing pass) ---
    if (optimization_level <= 1) {
        pm.append(std::make_unique<TrivialLayout>());
    } else {
        pm.append(std::make_unique<SabreLayout>());
    }
    pm.append(std::make_unique<SabreSwap>());

    // ---- Stage 2: optimisation ---------------------------------------------
    if (optimization_level >= 1) {
        pm.append(std::make_unique<RemoveResetInZeroState>());
        pm.append(std::make_unique<Optimize1qGates>());
        pm.append(std::make_unique<CXCancellation>());
        if (optimization_level >= 2) {
            pm.append(std::make_unique<CommutativeCancellation>());
        }
        pm.append(std::make_unique<RemoveDiagonalGatesBeforeMeasure>());
    }
    if (optimization_level >= 3) {
        // Block consolidation (KAK), then a cleanup sweep over its output.
        pm.append(std::make_unique<ConsolidateBlocks>());
        pm.append(std::make_unique<Optimize1qGates>());
        pm.append(std::make_unique<CXCancellation>());
        pm.append(std::make_unique<CommutativeCancellation>());
        pm.append(std::make_unique<RemoveDiagonalGatesBeforeMeasure>());
    }

    // ---- Stage 3: basis translation (always the FINAL stage) ---------------
    if (!basis_gates.empty()) {
        pm.append(std::make_unique<BasisTranslator>());
    }

    return pm;
}

// =============================================================================
// transpile — convenience entry point
// =============================================================================

QuantumCircuit transpile(
    const QuantumCircuit& circuit,
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates,
    int optimization_level
) {
    auto dag = DAGCircuit::from_circuit(circuit);
    TranspilationContext ctx;
    ctx.coupling_map = coupling_map;
    ctx.basis_gates = basis_gates;
    ctx.optimization_level = optimization_level;

    auto pm = preset_pass_manager(optimization_level, coupling_map, basis_gates);
    auto result_dag = pm.run(dag, ctx);
    return result_dag.to_circuit();
}

} // namespace lindblad
