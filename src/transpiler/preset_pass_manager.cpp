// preset_pass_manager.cpp — PassManager + preset pipeline composition + transpile()
//
// Extracted from optimize_1q.cpp in R.1.15.0 (SRP: pipeline composition is
// not a 1-qubit-optimisation concern).
//
// Preset levels compose per STAGE (non-cumulative) since R.1.15.0:
//
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
// Before R.1.15.0 the levels composed cumulatively: level >= 2 first ran the
// level-0 TrivialLayout + SabreSwap block, so the initial routing pass always
// executed on the unexpanded DAG (frozen-slot defect, layout_expansion.hpp)
// and SabreLayout then had to re-route an already-SWAP-laden circuit whose
// pass-1 SWAPs it could not undo. Now: one layout choice, one routing pass.
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
    const CouplingMap& /*coupling_map*/,
    const std::vector<std::string>& basis_gates
) {
    // The coupling map is intentionally not consumed here: pass COMPOSITION
    // does not depend on it (every composed pass reads it from the
    // TranspilationContext handed to run()). basis_gates IS consumed since
    // R.1.15.0: it decides whether the translation stage is composed.
    // transpile() passes the same vector into both this function and the
    // context, so composition and execution always agree.
    if (optimization_level < 0 || optimization_level > 3) {
        // Fail loud: level -1 previously returned an EMPTY manager (silent
        // identity transpile) and level >= 4 silently behaved as level 3.
        throw std::invalid_argument(
            "lindblad::preset_pass_manager: optimization_level " +
            std::to_string(optimization_level) +
            " is out of range (valid levels: 0..3)");
    }

    PassManager pm;

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
