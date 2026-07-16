// R.1.15.1 routing regression suite for the R.1.15.0 transpiler correctness
// wave (frozen-slot expansion #47, non-cumulative stage presets #48,
// basis_gates wiring #49).
//
// Contracts pinned here:
//   - circuits SMALLER than the coupling map route on line/grid/heavy-hex
//     (legality, termination, run-to-run determinism), output width equals
//     the device width, and the routed circuit equals the identity-embedded
//     original up to the wire permutation produced by the inserted SWAPs
//     (verified exactly at the unitary level for small devices);
//   - a NON-symmetric basis state survives route + measure end-to-end with
//     unchanged clbit keys (CLAUDE.md convention rule: symmetric tests mask
//     every convention bug);
//   - preset composition is per-stage: exactly one routing pass at every
//     level, SabreLayout first at level >= 2, BasisTranslator last iff
//     basis_gates is non-empty; levels outside 0..3 throw;
//   - unroutable input throws (never hangs), from the LAYOUT stage at
//     level >= 2 and from routing at level <= 1;
//   - initial_layout is validated (range/duplicates/size) and a partial
//     layout is completed deterministically;
//   - with a non-empty basis the output contains ONLY basis gates or the
//     translator throws naming the gate (MCX / UNITARY / unbound PARAM_*,
//     cx+u3-unreachable bases); u/u3 alias emission follows the basis;
//     classical conditions survive decomposition, structurally and
//     behaviourally.
//
// Deliberate choice: no hardcoded golden SWAP counts. The pipeline is
// deterministic, so determinism is asserted by running twice and requiring
// identical instruction streams; correctness is asserted by unitary/
// measurement semantics. Golden counts would pin the current heuristic
// scores, which are not part of the public contract.
//
// Test-only release content (R.1.15.1).

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/transpiler.hpp"

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

QuantumCircuit run_pass(const TranspilationPass& pass, const QuantumCircuit& qc,
                        TranspilationContext ctx = {}) {
    auto dag = DAGCircuit::from_circuit(qc);
    return pass.run(dag, ctx).to_circuit();
}

// Every multi-qubit gate acts on coupling-adjacent wire pairs (barriers and
// measures carry no routing constraint).
void expect_hardware_legal(const QuantumCircuit& qc, const CouplingMap& cm) {
    for (const auto& inst : qc.instructions) {
        if (inst.type == GT::BARRIER || inst.type == GT::MEASURE ||
            inst.type == GT::RESET) {
            continue;
        }
        const auto& q = inst.qubits;
        for (size_t a = 0; a < q.size(); ++a) {
            for (size_t b = a + 1; b < q.size(); ++b) {
                EXPECT_TRUE(cm.is_connected(q[a], q[b]) ||
                            cm.is_connected(q[b], q[a]))
                    << inst.gate_name() << " on non-adjacent physical pair "
                    << q[a] << "," << q[b];
            }
        }
    }
}

// Exact instruction-stream equality: the pipeline is deterministic, so two
// runs on identical input must agree bit-for-bit (types, operands, params,
// conditions). Catches any nondeterminism sneaking into layout or routing.
void expect_identical_streams(const QuantumCircuit& a, const QuantumCircuit& b) {
    ASSERT_EQ(a.n_qubits, b.n_qubits);
    ASSERT_EQ(a.instructions.size(), b.instructions.size());
    for (size_t i = 0; i < a.instructions.size(); ++i) {
        const auto& x = a.instructions[i];
        const auto& y = b.instructions[i];
        EXPECT_EQ(x.type, y.type) << "type @ " << i;
        EXPECT_EQ(x.qubits, y.qubits) << "qubits @ " << i;
        EXPECT_EQ(x.clbits, y.clbits) << "clbits @ " << i;
        EXPECT_EQ(x.params, y.params) << "params @ " << i;
        EXPECT_EQ(x.condition_clbit, y.condition_clbit) << "cond clbit @ " << i;
        EXPECT_EQ(x.condition_value, y.condition_value) << "cond value @ " << i;
    }
}

// Column-major full unitary of a measure-free circuit via basis-column
// statevector simulation: U[i + dim*j] = <i|U|j>. Small devices only.
std::vector<std::complex<double>> unitary_of(const QuantumCircuit& qc) {
    const int dim = 1 << qc.n_qubits;
    std::vector<std::complex<double>> U(static_cast<size_t>(dim) * dim);
    StatevectorSimulator sim;
    for (int col = 0; col < dim; ++col) {
        Statevector basis(qc.n_qubits);
        basis.initialize_basis(col);
        for (const auto& inst : qc.instructions) sim.apply_instruction(basis, inst);
        for (int row = 0; row < dim; ++row) {
            U[static_cast<size_t>(row) + static_cast<size_t>(dim) * col] =
                {basis.real_parts[row], basis.imag_parts[row]};
        }
    }
    return U;
}

// LEVELS 0-1 ONLY (TrivialLayout = identity embedding): routing then acts by
// left-multiplication with its inserted SWAPs, so
// P = U_routed * U_original^dagger is exactly a (phase x) permutation matrix:
// one modulus-1 entry per column and per row, zeros elsewhere. This is the
// exact statement of "routing only inserts SWAPs; it never changes physics".
// (At level >= 2 SabreLayout CONJUGATES by its wire relabeling, so the
// product is no longer a bare permutation -- those levels are checked at the
// state level instead, see expect_state_relabeling.)
void expect_permutation_times_original(const QuantumCircuit& routed,
                                       const QuantumCircuit& original,
                                       int n_physical, double tol = 1e-7) {
    ASSERT_EQ(routed.n_qubits, n_physical);
    QuantumCircuit padded = original;      // identity embedding: same gates,
    padded.n_qubits = n_physical;          // idle wires added at the top
    const int dim = 1 << n_physical;
    const auto Ur = unitary_of(routed);
    const auto Uo = unitary_of(padded);

    std::vector<int> row_hits(static_cast<size_t>(dim), 0);
    for (int j = 0; j < dim; ++j) {        // P[:,j] = Ur * conj(Uo[j,:])
        int ones = 0;
        for (int i = 0; i < dim; ++i) {
            std::complex<double> p(0, 0);
            for (int k = 0; k < dim; ++k) {
                p += Ur[static_cast<size_t>(i) + static_cast<size_t>(dim) * k] *
                     std::conj(Uo[static_cast<size_t>(j) +
                                  static_cast<size_t>(dim) * k]);
            }
            const double m = std::abs(p);
            if (m > tol) {
                EXPECT_NEAR(m, 1.0, tol)
                    << "non-unimodular entry at (" << i << "," << j << ")";
                ++ones;
                ++row_hits[static_cast<size_t>(i)];
            }
        }
        EXPECT_EQ(ones, 1) << "column " << j << " is not a permutation column";
    }
    for (int i = 0; i < dim; ++i) {
        EXPECT_EQ(row_hits[static_cast<size_t>(i)], 1)
            << "row " << i << " is not a permutation row";
    }
}

// Final state of a measure-free circuit from |0...0>.
std::vector<std::complex<double>> state_of(const QuantumCircuit& qc) {
    Statevector sv(qc.n_qubits);
    sv.initialize_basis(0);
    StatevectorSimulator sim;
    for (const auto& inst : qc.instructions) sim.apply_instruction(sv, inst);
    const int dim = 1 << qc.n_qubits;
    std::vector<std::complex<double>> psi(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) psi[i] = {sv.real_parts[i], sv.imag_parts[i]};
    return psi;
}

// ALL LEVELS: layout relabeling + inserted SWAPs act on |0...0> as a pure
// basis-index relabeling of the identity-embedded original's state (the
// relabeling maps |0...0> to itself), so the SORTED probability spectra must
// agree exactly. Phase-blind, but combined with the level-0/1 unitary check,
// the clbit-keyed end-to-end tests, and determinism this pins the contract.
void expect_state_relabeling(const QuantumCircuit& routed,
                             const QuantumCircuit& original,
                             int n_physical, double tol = 1e-9) {
    ASSERT_EQ(routed.n_qubits, n_physical);
    QuantumCircuit padded = original;
    padded.n_qubits = n_physical;
    auto pr = state_of(routed);
    auto po = state_of(padded);
    ASSERT_EQ(pr.size(), po.size());
    std::vector<double> probs_r(pr.size()), probs_o(po.size());
    for (size_t i = 0; i < pr.size(); ++i) probs_r[i] = std::norm(pr[i]);
    for (size_t i = 0; i < po.size(); ++i) probs_o[i] = std::norm(po[i]);
    std::sort(probs_r.begin(), probs_r.end());
    std::sort(probs_o.begin(), probs_o.end());
    for (size_t i = 0; i < probs_r.size(); ++i) {
        EXPECT_NEAR(probs_r[i], probs_o[i], tol) << "sorted prob @ " << i;
    }
}

int count_gate(const QuantumCircuit& qc, GT t) {
    int n = 0;
    for (const auto& inst : qc.instructions) n += (inst.type == t) ? 1 : 0;
    return n;
}

bool all_in_basis(const QuantumCircuit& qc,
                  const std::vector<std::string>& basis) {
    for (const auto& inst : qc.instructions) {
        if (inst.type == GT::MEASURE || inst.type == GT::BARRIER ||
            inst.type == GT::RESET) {
            continue;
        }
        bool ok = false;
        for (const auto& b : basis) ok = ok || (inst.gate_name() == b);
        if (!ok) return false;
    }
    return true;
}

// Unitary equivalence up to global phase (same width only).
void expect_equiv(const QuantumCircuit& a, const QuantumCircuit& b,
                  double tol = 1e-7) {
    ASSERT_EQ(a.n_qubits, b.n_qubits);
    const auto ma = unitary_of(a);
    const auto mb = unitary_of(b);
    ASSERT_EQ(ma.size(), mb.size());
    std::complex<double> phase(1, 0);
    for (size_t i = 0; i < ma.size(); ++i) {
        if (std::abs(mb[i]) > 1e-9 && std::abs(ma[i]) > 1e-9) {
            phase = ma[i] / mb[i];
            break;
        }
    }
    for (size_t i = 0; i < ma.size(); ++i) {
        EXPECT_NEAR(std::abs(ma[i] - phase * mb[i]), 0.0, tol) << "entry " << i;
    }
}

}  // namespace

// =============================================================================
// R1151Expansion — frozen-slot fix (#47): smaller-than-map circuits route
// =============================================================================

TEST(R1151Expansion, SmallerCircuitRoutesOnLineAllLevels) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 2).cx(1, 2).cx(0, 1);
    auto cm = CouplingMap::linear(5);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        QuantumCircuit out;
        ASSERT_NO_THROW(out = transpile(qc, cm, {}, level));
        EXPECT_EQ(out.n_qubits, 5) << "output width must equal device width";
        expect_hardware_legal(out, cm);
        if (level <= 1) {
            expect_permutation_times_original(out, qc, 5);  // exact, unitary
        }
        expect_state_relabeling(out, qc, 5);  // all levels incl. SabreLayout
    }
}

TEST(R1151Expansion, SmallerCircuitRoutesOnGridAllLevels) {
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 3).cx(1, 2).cx(0, 2).cx(3, 1);
    auto cm = CouplingMap::grid(2, 3);  // 6 physical slots
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        QuantumCircuit out;
        ASSERT_NO_THROW(out = transpile(qc, cm, {}, level));
        EXPECT_EQ(out.n_qubits, 6);
        expect_hardware_legal(out, cm);
        if (level <= 1) {
            expect_permutation_times_original(out, qc, 6);
        }
        expect_state_relabeling(out, qc, 6);
    }
}

// Superposition end-to-end at every level: a GHZ prepared over non-adjacent
// pairs, routed onto the grid, must sample only the two GHZ keys. Exercises
// SabreLayout's wire relabeling + routing + measure consistency behaviourally.
TEST(R1151Expansion, GhzSuperpositionSurvivesAllLevels) {
    QuantumCircuit qc(3, 3);
    qc.h(0).cx(0, 2).cx(2, 1);
    qc.measure_all();
    auto cm = CouplingMap::grid(2, 3);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, cm, {}, level);
        EXPECT_EQ(out.n_qubits, 6);
        StatevectorSimulator sim;
        auto res = sim.run(out, 128, 13);
        ASSERT_TRUE(res.success);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            EXPECT_TRUE(bits == "000" || bits == "111")
                << "non-GHZ key '" << bits << "' after routing (level "
                << level << ")";
        }
    }
}

// The original defect scenario: a small circuit on a REAL branchy heavy-hex
// (27 slots). Before R.1.15.0 this thrashed until the SWAP budget threw a
// misleading "disconnected components" error. Structural assertions only
// (2^27 amplitudes are not simulable in a unit test).
TEST(R1151Expansion, SmallerCircuitRoutesOnHeavyHex27AllLevels) {
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 3).cx(1, 2).cx(0, 2).cx(2, 3);
    auto cm = CouplingMap::heavy_hex(27);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        QuantumCircuit out;
        ASSERT_NO_THROW(out = transpile(qc, cm, {}, level))
            << "frozen-slot regression: n < n_physical must be routable";
        EXPECT_EQ(out.n_qubits, 27);
        expect_hardware_legal(out, cm);
    }
}

// End-to-end convention check with a NON-symmetric value (CLAUDE.md rule):
// prepare |K=5> = q2=1,q0=1 on 4 wires, route onto a 6-slot line, sample.
// Clbit keys are permutation-free (measures follow their logical wires), so
// every sampled bitstring must still read "0101" (q0 rightmost).
TEST(R1151Expansion, NonSymmetricBasisStateSurvivesRouteAndMeasure) {
    QuantumCircuit qc(4, 4);
    qc.x(0).x(2);
    // Diagonal 2q gates on NON-adjacent pairs: they force real routing work
    // but cannot change a computational basis state (phase only), so the
    // sampled key is exactly the prepared K for every level. (A CX pair here
    // would flip bits of the basis state and mask a convention bug.)
    qc.cz(0, 3).cz(1, 3).cz(0, 2);
    qc.measure_all();
    auto cm = CouplingMap::linear(6);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, cm, {}, level);
        EXPECT_EQ(out.n_qubits, 6);
        StatevectorSimulator sim;
        auto res = sim.run(out, 128, 7);
        ASSERT_TRUE(res.success);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            EXPECT_EQ(bits, "0101")
                << "clbit keys must be invariant under routing (level "
                << level << ")";
        }
    }
}

TEST(R1151Expansion, WiderThanDeviceThrowsEverywhere) {
    QuantumCircuit qc(5);
    qc.h(0).cx(0, 4);
    auto cm = CouplingMap::linear(3);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        EXPECT_THROW(transpile(qc, cm, {}, level), std::invalid_argument);
    }
    TranspilationContext ctx;
    ctx.coupling_map = cm;
    EXPECT_THROW(run_pass(TrivialLayout(), qc, ctx), std::invalid_argument);
    EXPECT_THROW(run_pass(SabreLayout(), qc, ctx), std::invalid_argument);
    EXPECT_THROW(run_pass(SabreSwap(), qc, ctx), std::invalid_argument);
}

TEST(R1151Expansion, UnconstrainedAndEqualWidthUnchanged) {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2).rz(0.4, 2);

    // No map: width untouched, semantics preserved at every level.
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("unconstrained level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap(), {}, level);
        EXPECT_EQ(out.n_qubits, 3);
        expect_equiv(out, qc);
    }

    // Map exactly the circuit's size: no expansion, still legal.
    auto cm = CouplingMap::linear(3);
    auto out = transpile(qc, cm, {}, 1);
    EXPECT_EQ(out.n_qubits, 3);
    expect_hardware_legal(out, cm);
}

// =============================================================================
// R1151Determinism — identical input -> identical instruction stream
// =============================================================================

TEST(R1151Determinism, TranspileDeterministicAcrossLevelsAndMaps) {
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 3).cx(1, 2).rz(0.3, 1).cx(0, 2).cx(2, 3);
    struct Topo { const char* name; CouplingMap cm; };
    const Topo topos[] = {
        {"linear5", CouplingMap::linear(5)},
        {"grid2x3", CouplingMap::grid(2, 3)},
        {"heavyhex27", CouplingMap::heavy_hex(27)},
    };
    for (const auto& t : topos) {
        for (int level = 0; level <= 3; ++level) {
            SCOPED_TRACE(std::string(t.name) + " level " + std::to_string(level));
            auto a = transpile(qc, t.cm, {}, level);
            auto b = transpile(qc, t.cm, {}, level);
            expect_identical_streams(a, b);
        }
    }
}

// =============================================================================
// R1151Presets — stage composition (#48) and level validation
// =============================================================================

TEST(R1151Presets, StageTableCompositionPinned) {
    auto cm = CouplingMap::linear(4);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto pm = preset_pass_manager(level, cm, {});
        ASSERT_GE(pm.passes.size(), 2u);

        // Layout slot: Trivial at 0-1, SABRE at 2-3; routing follows.
        const std::string expected_layout =
            (level <= 1) ? "TrivialLayout" : "SabreLayout";
        EXPECT_EQ(pm.passes[0]->name(), expected_layout);
        EXPECT_EQ(pm.passes[1]->name(), "SabreSwap");

        // Exactly ONE routing pass per level (the cumulative composition ran
        // TrivialLayout+SabreSwap in front of SabreLayout at level >= 2).
        int routing_passes = 0, layout_passes = 0, translators = 0;
        for (const auto& p : pm.passes) {
            const auto n = p->name();
            routing_passes += (n == "SabreSwap") ? 1 : 0;
            layout_passes += (n == "TrivialLayout" || n == "SabreLayout") ? 1 : 0;
            translators += (n == "BasisTranslator") ? 1 : 0;
        }
        EXPECT_EQ(routing_passes, 1) << "one routing pass per level";
        EXPECT_EQ(layout_passes, 1) << "one layout choice per level";
        EXPECT_EQ(translators, 0) << "no translator with empty basis_gates";

        // With a basis, BasisTranslator exists and is the FINAL stage.
        auto pmb = preset_pass_manager(level, cm, {"cx", "u3"});
        ASSERT_FALSE(pmb.passes.empty());
        EXPECT_EQ(pmb.passes.back()->name(), "BasisTranslator");
        int tb = 0;
        for (const auto& p : pmb.passes) tb += (p->name() == "BasisTranslator");
        EXPECT_EQ(tb, 1);
    }
}

TEST(R1151Presets, LevelOutOfRangeThrows) {
    auto cm = CouplingMap::linear(2);
    EXPECT_THROW(preset_pass_manager(-1, cm, {}), std::invalid_argument);
    EXPECT_THROW(preset_pass_manager(4, cm, {}), std::invalid_argument);
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    EXPECT_THROW(transpile(qc, CouplingMap(), {}, -1), std::invalid_argument);
    EXPECT_THROW(transpile(qc, CouplingMap(), {}, 4), std::invalid_argument);
}

// Unroutable input must THROW, never hang, at every level. At level >= 2 the
// throw comes from the LAYOUT stage (SabreLayout's new SWAP budget) -- the
// message prefix proves SabreLayout now runs first.
TEST(R1151Presets, DisconnectedMapThrowsFromTheRightStage) {
    CouplingMap islands(4);
    islands.edges = {{0, 1}, {1, 0}, {2, 3}, {3, 2}};
    QuantumCircuit qc(4);
    qc.cx(0, 2);  // spans the islands

    for (int level = 0; level <= 1; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        EXPECT_THROW(transpile(qc, islands, {}, level), std::runtime_error);
    }
    for (int level = 2; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        try {
            transpile(qc, islands, {}, level);
            FAIL() << "unroutable circuit must throw";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("SABRE layout"),
                      std::string::npos)
                << "level >= 2 must fail in the layout stage, got: "
                << e.what();
        }
    }
}

// =============================================================================
// R1151InitialLayout — validation + deterministic completion
// =============================================================================

TEST(R1151InitialLayout, ValidLayoutHonoured) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    TranspilationContext ctx;
    ctx.coupling_map = CouplingMap::linear(2);
    ctx.initial_layout = {1, 0};
    auto out = run_pass(SabreSwap(), qc, ctx);
    EXPECT_EQ(count_gate(out, GT::CX), 1);
    EXPECT_EQ(count_gate(out, GT::SWAP), 0) << "adjacent under the layout";
    expect_hardware_legal(out, ctx.coupling_map);
}

// A partial layout on an EXPANDED DAG (idle wires from TrivialLayout) is
// completed with the unused slots in ascending order -- deterministically.
TEST(R1151InitialLayout, PartialLayoutCompletedDeterministically) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    TranspilationContext ctx;
    ctx.coupling_map = CouplingMap::linear(4);
    ctx.initial_layout = {2, 1};  // logical 0 -> phys 2, logical 1 -> phys 1

    auto pm = preset_pass_manager(0, ctx.coupling_map, {});
    auto dag = DAGCircuit::from_circuit(qc);
    auto a = pm.run(dag, ctx).to_circuit();
    auto b = pm.run(dag, ctx).to_circuit();
    expect_identical_streams(a, b);

    // Placement is adjacent: the cx lands on physical {2,1}, zero SWAPs.
    EXPECT_EQ(count_gate(a, GT::SWAP), 0);
    ASSERT_EQ(count_gate(a, GT::CX), 1);
    for (const auto& inst : a.instructions) {
        if (inst.type == GT::CX) {
            EXPECT_EQ(inst.qubits, (std::vector<int>{2, 1}));
        }
    }
    expect_hardware_legal(a, ctx.coupling_map);
}

TEST(R1151InitialLayout, MalformedLayoutThrows) {
    QuantumCircuit qc(2);
    qc.cx(0, 1);
    auto cm = CouplingMap::linear(4);

    TranspilationContext out_of_range;
    out_of_range.coupling_map = cm;
    out_of_range.initial_layout = {5, 0};
    EXPECT_THROW(run_pass(SabreSwap(), qc, out_of_range), std::invalid_argument);

    TranspilationContext negative;
    negative.coupling_map = cm;
    negative.initial_layout = {-1, 0};
    EXPECT_THROW(run_pass(SabreSwap(), qc, negative), std::invalid_argument);

    TranspilationContext duplicate;
    duplicate.coupling_map = cm;
    duplicate.initial_layout = {1, 1};
    EXPECT_THROW(run_pass(SabreSwap(), qc, duplicate), std::invalid_argument);

    TranspilationContext oversize;
    oversize.coupling_map = cm;
    oversize.initial_layout = {0, 1, 2};  // 3 entries, 2 qubit wires
    EXPECT_THROW(run_pass(SabreSwap(), qc, oversize), std::invalid_argument);
}

// =============================================================================
// R1151Basis — basis_gates wiring (#49)
// =============================================================================

TEST(R1151Basis, TranspileHonoursBasisAtEveryLevelWithRouting) {
    QuantumCircuit qc(3, 3);
    qc.h(0).t(1).cx(0, 2).swap(0, 1).cx(1, 2);
    qc.measure_all();
    const std::vector<std::string> basis = {"cx", "u3"};
    auto cm = CouplingMap::linear(4);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, cm, basis, level);
        EXPECT_TRUE(all_in_basis(out, basis))
            << "output must contain only basis gates";
        expect_hardware_legal(out, cm);  // translation stays on routed pairs
        EXPECT_EQ(count_gate(out, GT::MEASURE), 3) << "measures preserved";
    }
}

TEST(R1151Basis, BasisOutputPreservesSemanticsUnconstrained) {
    QuantumCircuit qc(2);
    qc.h(0).t(0).cx(0, 1).rz(0.7, 1).ry(0.3, 0);
    const std::vector<std::string> basis = {"cx", "u3"};
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap(), basis, level);
        EXPECT_TRUE(all_in_basis(out, basis));
        expect_equiv(out, qc);
    }
}

// The basis names "u" (not "u3"): emitted 1q gates follow the caller's name.
TEST(R1151Basis, UAliasEmissionFollowsBasis) {
    QuantumCircuit qc(2);
    qc.h(0).t(1).cx(0, 1);
    auto out = transpile(qc, CouplingMap(), {"u", "cx"}, 1);
    EXPECT_TRUE(all_in_basis(out, {"u", "cx"}));
    EXPECT_EQ(count_gate(out, GT::U3), 0) << "u3 must not appear when the "
                                             "basis names u";
    expect_equiv(out, qc);
}

TEST(R1151Basis, NativeGatesPassThrough) {
    QuantumCircuit qc(2, 1);
    qc.h(0).t(1).barrier().reset(1).cx(0, 1).measure(0, 0);
    auto out = transpile(qc, CouplingMap(), {"h", "t", "cx"}, 0);
    EXPECT_EQ(count_gate(out, GT::H), 1);
    EXPECT_EQ(count_gate(out, GT::T), 1);
    EXPECT_EQ(count_gate(out, GT::CX), 1);
    EXPECT_EQ(count_gate(out, GT::BARRIER), 1);
    EXPECT_EQ(count_gate(out, GT::RESET), 1);
    EXPECT_EQ(count_gate(out, GT::MEASURE), 1);
}

TEST(R1151Basis, UntranslatableGatesThrowLoudlyAndNameTheGate) {
    const std::vector<std::string> basis = {"cx", "u3"};

    {   // MCX has no lowering yet: must throw, naming the gate.
        QuantumCircuit qc(4);
        qc.mcx({0, 1, 2}, 3);
        try {
            transpile(qc, CouplingMap(), basis, 1);
            FAIL() << "mcx under a non-empty basis must throw";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("mcx"), std::string::npos)
                << e.what();
        }
        // Empty basis == no translation stage: the same circuit transpiles.
        EXPECT_NO_THROW(transpile(qc, CouplingMap(), {}, 1));
    }

    {   // 1q UNITARY: no synthesis path in the translator.
        QuantumCircuit qc(1);
        qc.unitary({Complex128(0, 0), Complex128(1, 0),
                    Complex128(1, 0), Complex128(0, 0)}, {0});
        EXPECT_THROW(transpile(qc, CouplingMap(), basis, 0),
                     std::invalid_argument);
    }

    {   // Unbound symbolic parameter: numeric decomposition impossible.
        QuantumCircuit qc(1);
        qc.rx("theta", 0);
        EXPECT_THROW(transpile(qc, CouplingMap(), basis, 0),
                     std::invalid_argument);
    }

    {   // Basis that can reach neither the source gate nor cx+u3.
        QuantumCircuit qc(1);
        qc.h(0);
        EXPECT_THROW(transpile(qc, CouplingMap(), {"rz", "sx"}, 0),
                     std::invalid_argument);
    }
}

// Structural half: every instruction a conditioned gate decomposes into
// carries the condition; unconditioned gates stay unconditioned.
TEST(R1151Basis, ConditionPropagatedOntoDecomposition) {
    QuantumCircuit qc(2, 1);
    qc.h(0);
    qc.measure(0, 0);
    qc.p_if(0.7, 1, 0, 1);  // P is not in the basis: it decomposes to u3
    auto out = transpile(qc, CouplingMap(), {"cx", "u3"}, 0);

    int conditioned = 0, unconditioned_gates = 0;
    for (const auto& inst : out.instructions) {
        if (inst.type == GT::MEASURE) continue;
        if (inst.condition_clbit >= 0) {
            ++conditioned;
            EXPECT_EQ(inst.condition_clbit, 0);
            EXPECT_EQ(inst.condition_value, 1);
        } else {
            ++unconditioned_gates;
        }
    }
    EXPECT_GE(conditioned, 1) << "the decomposed p_if lost its condition";
    EXPECT_GE(unconditioned_gates, 1) << "the h must stay unconditioned";
}

// Behavioural half: feedforward survives the full pipeline (routing +
// optimisation + translation). x(0) forces clbit 0 to 1, the conditional X
// then fires deterministically: every sampled key is "11".
TEST(R1151Basis, FeedforwardSurvivesFullPipelineEndToEnd) {
    QuantumCircuit qc(2, 2);
    qc.x(0);
    qc.measure(0, 0);
    qc.add_if(0, 1, GT::X, {1});
    qc.measure(1, 1);
    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap::linear(3), {"cx", "u3"}, level);
        EXPECT_TRUE(all_in_basis(out, {"cx", "u3"}));
        StatevectorSimulator sim;
        auto res = sim.run(out, 64, 11);
        ASSERT_TRUE(res.success);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            EXPECT_EQ(bits, "11")
                << "conditional X was lost in the pipeline (level " << level
                << ")";
        }
    }
}